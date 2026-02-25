#!/usr/bin/env python3
"""
transcode.py - Production-grade video transcoding engine for Jellyfin/iOS direct play.

Target: Ubuntu 22.04, Python 3.10.6, RTX 3060, system ffmpeg/ffprobe.
Output: HEVC (H.265) via NVENC, AAC-LC stereo, MP4 container (hvc1 tag).
"""

import argparse
import json
import logging
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".wmv", ".flv",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".webm",
    # .vob intentionally excluded -- DVD VOB files are handled via
    # discover_dvd_titles() which concatenates VTS segments properly
}

# Preferred audio language tags -> normalized ISO 639-2
PREFERRED_LANG_MAP = {
    "cze": "ces", "ces": "ces", "cs": "ces",
    "eng": "eng", "en": "eng",
    "jpn": "jpn", "ja": "jpn",
    "zho": "zho", "chi": "zho", "zh": "zho", "cmn": "zho", "yue": "zho",
}

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
HDR_PRIMARIES = {"bt2020"}
HDR_SPACES = {"bt2020nc", "bt2020c", "bt2020_ncl", "bt2020_cl"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("transcode")


def setup_logging(log_path: Optional[str]) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# SQLite state management
# ---------------------------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT UNIQUE NOT NULL,
    out_dir TEXT NOT NULL,
    out_file TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Thread-safe DB access
#
# Problem: sqlite3 connections are not safe to share across threads when
# multiple threads issue DML concurrently. Even with check_same_thread=False,
# a second thread can collide with an implicit transaction opened by the first.
#
# Solution: Use a module-level path + lock. Every DB helper opens its own
# short-lived connection, executes, commits, and closes. WAL journal mode
# allows concurrent readers alongside a single writer, so this is both
# correct and fast enough for our write-seldom workload.
# ---------------------------------------------------------------------------

_db_path: str = ""
_db_lock: threading.Lock = threading.Lock()


def _db_open() -> sqlite3.Connection:
    """Open a fresh, short-lived connection to the state DB."""
    conn = sqlite3.connect(_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_connect(db_path: str) -> None:
    """Initialize the DB (create schema). Returns None; path stored globally."""
    global _db_path
    _db_path = db_path
    with _db_lock:
        conn = _db_open()
        conn.executescript(DB_SCHEMA)
        conn.commit()
        conn.close()


def db_upsert_job(src: str, out_dir: str, out_file: str) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        conn = _db_open()
        try:
            conn.execute(
                """
                INSERT INTO jobs (src, out_dir, out_file, status, attempts, last_error, updated_at)
                VALUES (?, ?, ?, 'pending', 0, NULL, ?)
                ON CONFLICT(src) DO UPDATE SET
                    out_dir=excluded.out_dir,
                    out_file=excluded.out_file,
                    updated_at=excluded.updated_at
                WHERE status NOT IN ('done', 'skipped')
                """,
                (src, out_dir, out_file, now),
            )
            conn.commit()
        finally:
            conn.close()


def db_update_status(
    src: str,
    status: str,
    last_error: Optional[str] = None,
    increment_attempts: bool = False,
) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        conn = _db_open()
        try:
            if increment_attempts:
                conn.execute(
                    "UPDATE jobs SET status=?, attempts=attempts+1, last_error=?, updated_at=? WHERE src=?",
                    (status, last_error, now, src),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status=?, last_error=?, updated_at=? WHERE src=?",
                    (status, last_error, now, src),
                )
            conn.commit()
        finally:
            conn.close()


def db_get_runnable(max_retry: int) -> list[sqlite3.Row]:
    with _db_lock:
        conn = _db_open()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='pending' OR (status='failed' AND attempts <= ?)",
                (max_retry,),
            ).fetchall()
            return rows
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Session timing helpers
# ---------------------------------------------------------------------------

def db_meta_get(key: str) -> Optional[str]:
    """Read a value from the meta table. Returns None if not found."""
    with _db_lock:
        conn = _db_open()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()


def db_meta_set(key: str, value: str) -> None:
    """Write a value to the meta table (upsert)."""
    with _db_lock:
        conn = _db_open()
        try:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value)
            )
            conn.commit()
        finally:
            conn.close()


def session_start() -> float:
    """
    Mark the start of a new session. Returns the accumulated total elapsed
    seconds from previous sessions (for display alongside session elapsed).
    """
    now = time.time()
    db_meta_set("session_start_wall", str(now))
    total_str = db_meta_get("total_elapsed_seconds")
    return float(total_str) if total_str else 0.0


def session_stop() -> None:
    """
    Called when the engine is about to exit. Accumulates session elapsed
    into total_elapsed_seconds in the DB.
    """
    start_str = db_meta_get("session_start_wall")
    if not start_str:
        return
    session_secs = time.time() - float(start_str)
    total_str = db_meta_get("total_elapsed_seconds")
    total = float(total_str) if total_str else 0.0
    db_meta_set("total_elapsed_seconds", str(total + session_secs))
    db_meta_set("session_start_wall", "")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_dvd_titles(input_dir: Path) -> list["DVDTitle"]:
    """
    Walk input_dir looking for DVD structures.
    A DVD structure is a directory that contains a VIDEO_TS subdirectory
    (standard rip) OR directly contains VTS_XX_YY.VOB files (flat rip).

    For each DVD we find all title sets (VTS_01, VTS_02, ...) and within
    each title set collect the feature VOBs in order:
        VTS_XX_1.VOB, VTS_XX_2.VOB, ... (skip VTS_XX_0.VOB = menu)

    The largest title set (by total byte size) is assumed to be the feature.
    Returns a list of DVDTitle objects, one per DVD directory found.
    """
    titles = []
    seen_dirs = set()

    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)

        # Detect VIDEO_TS subdirectory style
        vob_dir = None
        if "VIDEO_TS" in [d.upper() for d in dirs]:
            # Find the actual VIDEO_TS dir (case-insensitive)
            for d in dirs:
                if d.upper() == "VIDEO_TS":
                    vob_dir = root_path / d
                    break
        # Detect flat VOB style (VOBs directly in this folder)
        elif any(f.upper().startswith("VTS_") and f.upper().endswith(".VOB")
                 for f in files):
            vob_dir = root_path

        if vob_dir is None or vob_dir in seen_dirs:
            continue
        seen_dirs.add(vob_dir)

        # Collect all feature VOBs grouped by title set number
        # VTS_01_1.VOB, VTS_01_2.VOB ... = title set 01, segments 1,2,...
        # VTS_01_0.VOB = menu for title set 01 -- skip
        title_sets: dict[int, list[Path]] = {}
        for f in sorted(vob_dir.iterdir()):
            m = re.match(r"VTS_(\d+)_(\d+)\.VOB$", f.name.upper())
            if not m:
                continue
            ts_num = int(m.group(1))
            seg_num = int(m.group(2))
            if seg_num == 0:
                continue  # menu VOB, skip
            title_sets.setdefault(ts_num, []).append(f)

        if not title_sets:
            continue

        # Pick the title set with the most total bytes = main feature
        def ts_size(vobs):
            return sum(v.stat().st_size for v in vobs if v.exists())

        best_ts = max(title_sets.keys(), key=lambda k: ts_size(title_sets[k]))
        feature_vobs = sorted(title_sets[best_ts])

        if not feature_vobs:
            continue

        # The DVD name is the movie folder (parent of VIDEO_TS, or the flat dir)
        if vob_dir.name.upper() == "VIDEO_TS":
            dvd_name = vob_dir.parent.name
            dvd_root = vob_dir.parent
        else:
            dvd_name = vob_dir.name
            dvd_root = vob_dir

        titles.append(DVDTitle(
            dvd_root=dvd_root,
            dvd_name=dvd_name,
            vobs=feature_vobs,
        ))

    return titles


class DVDTitle:
    """
    Represents a single DVD title (feature film) as a concat of VOB segments.
    Acts as a drop-in for Path in most of the pipeline by implementing
    the same interface (.name, str(), etc.).
    """
    def __init__(self, dvd_root: Path, dvd_name: str, vobs: list[Path]) -> None:
        self.dvd_root  = dvd_root
        self.dvd_name  = dvd_name
        self.vobs      = vobs           # sorted feature VOB segments
        # Concat URI understood by ffmpeg: concat:file1|file2|...
        self.concat_uri = "concat:" + "|".join(str(v) for v in vobs)
        # Use dvd_root as the "path" for DB keying and output naming
        self._path = dvd_root

    # Make it behave like a Path for the parts of the code that need it
    @property
    def name(self) -> str:
        return self.dvd_name

    @property
    def stem(self) -> str:
        return self.dvd_name

    @property
    def parent(self) -> Path:
        return self._path.parent

    @property
    def suffix(self) -> str:
        return ".vob"

    def __str__(self) -> str:
        # When passed as src to ffmpeg/ffprobe, use the concat URI
        return self.concat_uri

    def __fspath__(self) -> str:
        return self.concat_uri

    def __repr__(self) -> str:
        return f"DVDTitle({self.dvd_name!r}, vobs={len(self.vobs)})"

    def with_suffix(self, suffix: str) -> Path:
        # Used by transcode_job for .part file naming -- return a real Path
        return self._path.parent / (self.dvd_name + suffix)


def discover_videos(input_dir: Path) -> list:
    """
    Returns a mixed list of Path (regular video files) and DVDTitle objects.
    DVD directories are detected first; their containing folders are excluded
    from the regular file walk to prevent double-processing.
    """
    dvd_titles = discover_dvd_titles(input_dir)

    # Collect only the VOB-containing directories so we skip their .vob files
    # in the regular walk. Do NOT add dvd_root.parent -- that would exclude
    # sibling movie folders at the same level as the DVD folder.
    dvd_vob_dirs = set()
    for t in dvd_titles:
        dvd_vob_dirs.add(t.dvd_root)
        for v in t.vobs:
            dvd_vob_dirs.add(v.parent)  # VIDEO_TS subdir if present

    videos = []
    for root, _dirs, files in os.walk(input_dir):
        root_path = Path(root)
        # Skip directories that are part of a DVD VOB structure
        if any(root_path == d or root_path.is_relative_to(d)
               for d in dvd_vob_dirs):
            continue
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(root_path / fname)

    return sorted(videos) + dvd_titles


def compute_output_names(
    src,
    input_dir: Path,
    output_dir: Path,
    tv_mode: bool = False,
    assigned: set = None,
) -> tuple[Path, Path, str]:
    """
    Returns (out_dir, out_file_final, stem).
    Handles both regular Path objects and DVDTitle objects.

    Movies mode (default):
        Show/Season 01/S01E01.mkv -> output/Show/Show.mp4
        Movie Name/Movie Name.mkv -> output/Movie Name/Movie Name.mp4

    TV mode (--tv-mode):
        Show/Season 01/S01E01.mkv -> output/Show/Season 01/S01E01.mp4
        preserves full relative path, keeps original filename stem

    assigned: optional set of Path objects already assigned in this scan pass.
        Used to avoid collisions when multiple source files map to the same
        output name (e.g. multiple files in the same movie folder). Without
        this, the on-disk check would miss in-progress encodes that haven't
        created their output file yet.
    """
    if assigned is None:
        assigned = set()

    def _unique(out_dir: Path, base: str) -> tuple[Path, Path, str]:
        """Find a candidate name not on disk and not already assigned."""
        candidate = base
        counter = 0
        while (out_dir / f"{candidate}.mp4").exists() or               (out_dir / f"{candidate}.mp4") in assigned:
            counter += 1
            candidate = f"{base}_{counter}"
        result = out_dir / f"{candidate}.mp4"
        assigned.add(result)
        return out_dir, result, candidate

    if isinstance(src, DVDTitle):
        name = src.dvd_name
        out_dir = output_dir / name
        return _unique(out_dir, name)

    if tv_mode:
        try:
            rel = src.relative_to(input_dir)
        except ValueError:
            rel = Path(src.name)
        out_file = output_dir / rel.with_suffix(".mp4")
        out_dir = out_file.parent
        # TV mode: filenames are unique by definition (full path preserved)
        assigned.add(out_file)
        return out_dir, out_file, src.stem

    # Movies mode: collapse to top-level folder name
    try:
        rel = src.parent.relative_to(input_dir)
        if str(rel) == ".":
            name = src.stem
        else:
            name = rel.parts[0]
    except ValueError:
        name = src.stem

    out_dir = output_dir / name
    return _unique(out_dir, name)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def ffprobe_json(ffprobe_bin: str, src: str, probesize: str, analyzeduration: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-v", "error",          # capture actual error messages (not "quiet")
        "-probesize", probesize,
        "-analyzeduration", analyzeduration,
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        src,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    stderr_msg = result.stderr.strip()

    if result.returncode != 0:
        hint = stderr_msg[:2000] if stderr_msg else "(no stderr — file may be empty, missing, or unreadable)"
        raise RuntimeError(f"ffprobe failed: {hint}")

    stdout = result.stdout.strip()
    if not stdout:
        hint = stderr_msg[:500] if stderr_msg else "empty output"
        raise RuntimeError(f"ffprobe returned no JSON output: {hint}")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe JSON parse error: {e} | stderr: {stderr_msg[:500]}")


def get_video_stream(probe: dict) -> Optional[dict]:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return None


def get_audio_streams(probe: dict) -> list[dict]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]


# Subtitle codecs that MP4 can carry as mov_text (text-based only).
# Bitmap formats (pgssub, dvdsub, hdmv_pgs_subtitle, dvb_subtitle) are
# dropped -- MP4 cannot reliably carry image-based subtitle streams.
TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text",
    "webvtt", "text", "microdvd",
}

# ASS/SSA codecs that should also be extracted to sidecar .ass files
# so Jellyfin can use the fully styled version.
ASS_SUBTITLE_CODECS = {"ass", "ssa"}


def get_subtitle_streams(probe: dict) -> list[dict]:
    """Return all subtitle streams that are text-based and MP4-compatible."""
    subs = []
    for s in probe.get("streams", []):
        if s.get("codec_type") != "subtitle":
            continue
        codec = (s.get("codec_name") or "").lower()
        if codec in TEXT_SUBTITLE_CODECS:
            subs.append(s)
    return subs


def extract_ass_sidecars(
    ffmpeg_bin: str,
    src: str,
    out_dir: Path,
    out_stem: str,
    subtitle_streams: list[dict],
    probesize: str,
    analyzeduration: str,
) -> int:
    """
    Extract ASS/SSA embedded subtitle streams from src to .ass sidecar files.
    Output filenames: <out_stem>.<lang>.ass (or <out_stem>.<index>.ass if no lang).
    Returns count of extracted files.
    """
    ass_streams = [
        s for s in subtitle_streams
        if (s.get("codec_name") or "").lower() in ASS_SUBTITLE_CODECS
    ]
    if not ass_streams:
        return 0

    count = 0
    for s in ass_streams:
        tags = s.get("tags") or {}
        lang = (tags.get("language") or tags.get("LANGUAGE") or "").lower().strip()
        suffix = lang if lang else str(s["index"])
        out_path = out_dir / f"{out_stem}.{suffix}.ass"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel", "error",
            "-probesize", probesize,
            "-analyzeduration", analyzeduration,
            "-i", src,
            "-map", f"0:{s['index']}",
            "-c:s", "copy",
            str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                count += 1
            else:
                logger.warning(
                    f"Failed to extract ASS sub stream {s['index']} "
                    f"from {src}: {result.stderr[:500]}"
                )
        except Exception as e:
            logger.warning(f"ASS extraction exception for stream {s['index']}: {e}")

    return count


def is_hdr(vs: dict) -> bool:
    transfer = (vs.get("color_transfer") or "").lower()
    primaries = (vs.get("color_primaries") or "").lower()
    space = (vs.get("color_space") or "").lower()
    pix_fmt = (vs.get("pix_fmt") or "").lower()

    if transfer in HDR_TRANSFERS:
        return True
    if primaries in HDR_PRIMARIES:
        return True
    if space in HDR_SPACES:
        return True
    # 10-bit with bt2020 hint
    is_10bit = "p010" in pix_fmt or "yuv420p10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt
    if is_10bit and ("bt2020" in primaries or "bt2020" in space or "bt2020" in transfer):
        return True
    return False


def is_10bit_source(vs: dict) -> bool:
    pix_fmt = (vs.get("pix_fmt") or "").lower()
    return (
        "p010" in pix_fmt or "yuv420p10" in pix_fmt
        or "10le" in pix_fmt or "10be" in pix_fmt
        or "yuv422p10" in pix_fmt or "yuv444p10" in pix_fmt
    )


def select_audio_tracks(audio_streams: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Returns (selected_streams, normalized_iso_langs).
    If at least one preferred language found, keep only preferred tracks.
    Otherwise keep all.
    """
    preferred = []
    preferred_langs = []
    for s in audio_streams:
        tags = s.get("tags") or {}
        lang = (tags.get("language") or tags.get("LANGUAGE") or "").lower().strip()
        iso = PREFERRED_LANG_MAP.get(lang)
        if iso:
            preferred.append(s)
            preferred_langs.append(iso)

    if preferred:
        return preferred, preferred_langs

    # No preferred: keep all, unknown lang
    langs = []
    for s in audio_streams:
        tags = s.get("tags") or {}
        lang = (tags.get("language") or tags.get("LANGUAGE") or "").lower().strip()
        iso = PREFERRED_LANG_MAP.get(lang) or (lang if lang else "und")
        langs.append(iso)
    return audio_streams, langs


# ---------------------------------------------------------------------------
# Filter availability detection
# ---------------------------------------------------------------------------

def detect_filter(ffmpeg_bin: str, filter_name: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-filters"],
            capture_output=True, text=True, timeout=30,
        )
        return filter_name in result.stdout
    except Exception:
        return False


# Codecs supported by cuvid hardware decode.
# Used both for detection and per-file codec gating.
CUVID_CODECS = {
    "hevc":  "hevc_cuvid",
    "h264":  "h264_cuvid",
    "vp9":   "vp9_cuvid",
    "av1":   "av1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "vc1":   "vc1_cuvid",
}

def detect_hwdec_cuda(ffmpeg_bin: str) -> set:
    """
    Detect which cuvid decoders are available.
    Returns a set of codec names (e.g. {"hevc", "h264"}) that have cuvid support.
    Empty set means no hardware decode available.
    jellyfin-ffmpeg includes cuvid decoders by default when compiled with CUDA support.
    """
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-decoders"],
            capture_output=True, text=True, timeout=30,
        )
        available = set()
        for codec, cuvid in CUVID_CODECS.items():
            if cuvid in result.stdout:
                available.add(codec)
        return available
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Video filter chain builder
# ---------------------------------------------------------------------------

def build_video_filter(
    vs: dict,
    max_height: int,
    hdr: bool,
    has_scale_cuda: bool,
    has_zscale: bool,
    force_cpu: bool,
    no_hwaccel: bool,
    has_tonemap_cuda: bool = False,
    has_hwdec_cuda: set = None,
) -> tuple[str, list[str], str]:
    """
    Returns (vf_string, extra_input_args, pipeline_label).
    extra_input_args: additional ffmpeg args before -i (e.g., hwaccel options).

    has_hwdec_cuda: set of codec names with cuvid support (e.g. {'hevc', 'h264'}).
    When non-empty and GPU pipeline active, hardware decode is used per codec.
    The decoded frame arrives as a CUDA surface, skipping format= + hwupload_cuda.
    """
    height = vs.get("height") or 0
    need_scale = height > max_height and height > 0

    src_10bit = is_10bit_source(vs)

    # Build pipeline label for logging
    gpu_pipeline = has_scale_cuda and not force_cpu and not no_hwaccel
    # Gate hwdec on the actual source codec — only use cuvid if the specific
    # decoder is available. has_hwdec_cuda is a set of supported codec names.
    src_codec = (vs.get("codec_name") or "").lower()
    hwdec = gpu_pipeline and bool(has_hwdec_cuda) and (src_codec in (has_hwdec_cuda or set()))
    hwdec_tag = "+hwdec" if hwdec else ""
    if hdr and has_tonemap_cuda and has_scale_cuda and not force_cpu and not no_hwaccel:
        label = f"HDR->SDR(GPU-CUDA-tonemap_cuda{hwdec_tag}+NVENC)"
    elif hdr:
        label = "HDR->SDR(CPU-zscale+tonemap+NVENC)"
    elif force_cpu or no_hwaccel:
        label = "SDR-CPU(scale+NVENC)"
    elif has_scale_cuda and need_scale:
        label = f"SDR-GPU(scale_cuda{hwdec_tag}+NVENC)"
    else:
        label = "SDR-direct(NVENC)"

    # HDR pipeline
    if hdr:
        if has_tonemap_cuda and has_scale_cuda and not force_cpu and not no_hwaccel:
            # GPU CUDA tonemapping via jellyfin-ffmpeg tonemap_cuda filter.
            # tonemap_cuda is a true CUDA filter that operates on GPU surfaces.
            # tonemapx (different filter) is CPU-only despite the name.
            #
            # Two variants depending on has_hwdec_cuda:
            #
            # CPU decode (default):
            #   format=p010le → hwupload_cuda → scale_cuda → tonemap_cuda → hwdownload
            #
            # CUDA hwdec (has_hwdec_cuda=True):
            #   -hwaccel cuda -hwaccel_output_format cuda on input
            #   Frame arrives as CUDA surface — skip format= + hwupload_cuda
            #   scale_cuda → tonemap_cuda → hwdownload
            #   Eliminates CPU HEVC decode — major load reduction for 4K HDR.
            if need_scale:
                scale_step = f"scale_cuda=w=-2:h={max_height}:interp_algo=lanczos:format=p010le,"
            else:
                scale_step = "scale_cuda=w=0:h=0:passthrough=1:format=p010le,"

            if hwdec:
                # Frame already on GPU as CUDA surface — no format/hwupload needed
                vf = (
                    f"{scale_step}"
                    f"tonemap_cuda=tonemap=bt2390:desat=0:format=yuv420p,"
                    f"hwdownload,"
                    f"format=yuv420p"
                )
                extra_input_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            else:
                # CPU decode: normalize to p010le, upload to GPU, process, download
                vf = (
                    f"format=p010le,"
                    f"hwupload_cuda,"
                    f"{scale_step}"
                    f"tonemap_cuda=tonemap=bt2390:desat=0:format=yuv420p,"
                    f"hwdownload,"
                    f"format=yuv420p"
                )
                extra_input_args = []
            return vf, extra_input_args, label

        # CPU fallback: zscale + hable tonemap
        filters = []
        if has_zscale:
            if need_scale:
                filters.append(f"zscale=w=-2:h={max_height}:filter=lanczos:rangein=limited:range=limited:primariesin=bt2020:primaries=bt709:transferin=smpte2084:transfer=bt709:matrixin=bt2020nc:matrix=bt709")
            else:
                filters.append("zscale=primariesin=bt2020:primaries=bt709:transferin=smpte2084:transfer=bt709:matrixin=bt2020nc:matrix=bt709:rangein=limited:range=limited")
            filters.append("tonemap=tonemap=hable:desat=0")
        else:
            if need_scale:
                filters.append(f"scale=-2:{max_height}:flags=lanczos")
            filters.append("tonemap=tonemap=hable:desat=0")
        filters.append("format=yuv420p")
        vf = ",".join(filters)
        return vf, [], label

    # SDR paths
    if force_cpu or no_hwaccel:
        filters = []
        if need_scale:
            filters.append(f"scale=-2:{max_height}:flags=lanczos")
        filters.append("format=yuv420p")
        vf = ",".join(filters)
        return vf, [], label

    # GPU scale path
    # Default: CPU decode + hwupload + scale_cuda + hwdownload.
    # With hwdec: -hwaccel cuda puts decoded frames directly on GPU as CUDA
    # surfaces, skipping format= + hwupload_cuda entirely. Reduces CPU load
    # significantly for high-bitrate SDR sources (VC-1, AVC, HEVC).
    # nvidia-patch removes the 3-session NVENC limit so hwdec is safe to use.
    if has_scale_cuda and need_scale:
        if hwdec:
            # Frame arrives as CUDA surface — skip format/hwupload.
            # CUDA surface format depends on bit depth:
            #   8-bit source  → nv12  CUDA surface → hwdownload to nv12  → yuv420p
            #   10-bit source → p010le CUDA surface → hwdownload to p010le → yuv420p
            # Using the wrong download format causes:
            #   "Invalid output format nv12/p010le for hwframe download"
            if src_10bit:
                dl_fmt = "p010le"
            else:
                dl_fmt = "nv12"
            vf = (f"scale_cuda=w=-2:h={max_height}:interp_algo=lanczos,"
                  f"hwdownload,format={dl_fmt},format=yuv420p")
            return vf, ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], label
        if src_10bit:
            # Normalize to p010le on CPU before upload so hwupload_cuda always
            # receives a known-good format regardless of source codec/container.
            # scale_cuda handles p010le natively on the GPU.
            vf = (f"format=p010le,hwupload_cuda,"
                  f"scale_cuda=w=-2:h={max_height}:interp_algo=lanczos,"
                  f"hwdownload,format=p010le,format=yuv420p")
        else:
            # Normalize to nv12 on CPU first — nv12 is the native CUDA surface
            # format so hwupload_cuda accepts it from any source codec (XviD,
            # MPEG2, VC-1, AVC, etc.) without rejection.
            vf = (f"format=nv12,hwupload_cuda,"
                  f"scale_cuda=w=-2:h={max_height}:interp_algo=lanczos,"
                  f"hwdownload,format=nv12,format=yuv420p")
        # No extra input args: CPU decodes, GPU only scales and encodes.
        return vf, [], label

    # No GPU scale or not needed.
    # If hwdec is active the frame is on GPU — must hwdownload before format=
    # If no scale needed and hwdec active: hwdownload → format=yuv420p
    if hwdec and not need_scale:
        dl_fmt = "p010le" if src_10bit else "nv12"
        vf = f"hwdownload,format={dl_fmt},format=yuv420p"
        return vf, ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], label
    if need_scale:
        vf = f"scale=-2:{max_height}:flags=lanczos,format=yuv420p"
    else:
        vf = "format=yuv420p"
    return vf, [], label


# ---------------------------------------------------------------------------
# ffmpeg command builder
# ---------------------------------------------------------------------------

def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    src: str,
    out_part: str,
    vs: dict,
    selected_audio: list[dict],
    audio_langs: list[str],
    vf: str,
    extra_input_args: list[str],
    args: argparse.Namespace,
    is_dvd: bool = False,
    selected_subs: list[dict] = None,
) -> list[str]:
    probesize = getattr(args, "probesize", "200M")
    analyzeduration = getattr(args, "analyzeduration", "200M")

    cmd = [ffmpeg_bin]
    cmd += ["-y"]
    cmd += ["-loglevel", "warning"]
    cmd += ["-progress", "pipe:1"]

    # Robustness flags
    cmd += ["-fflags", "+genpts+discardcorrupt"]
    cmd += ["-err_detect", "ignore_err"]
    cmd += ["-ignore_unknown"]
    cmd += ["-max_error_rate", "1.0"]
    cmd += ["-avoid_negative_ts", "make_zero"]

    # Probing
    cmd += ["-probesize", probesize]
    cmd += ["-analyzeduration", analyzeduration]

    # hwaccel input args (must come before -i)
    cmd += extra_input_args

    cmd += ["-i", src]

    # Stream mapping: video + selected audio + text subtitles
    cmd += ["-map", "0:v:0"]
    for s in selected_audio:
        cmd += ["-map", f"0:{s['index']}"]
    if selected_subs:
        for s in selected_subs:
            cmd += ["-map", f"0:{s['index']}"]
    else:
        cmd += ["-sn"]  # no subtitle streams to map

    cmd += ["-dn"]
    cmd += ["-map_chapters", "-1"]
    cmd += ["-map_metadata", "0"]

    # Video codec
    cmd += ["-c:v", "hevc_nvenc"]
    rc = getattr(args, "rc", "vbr_hq")
    # vbr_hq / cbr_hq are deprecated in jellyfin-ffmpeg 7.x.
    # Map them to the modern equivalents: -rc vbr/-rc cbr + -tune hq + -multipass fullres.
    if rc == "vbr_hq":
        cmd += ["-rc", "vbr", "-tune", "hq", "-multipass", "fullres"]
    elif rc == "cbr_hq":
        cmd += ["-rc", "cbr", "-tune", "hq", "-multipass", "fullres"]
    else:
        cmd += ["-rc", rc]
    cmd += ["-cq:v", str(getattr(args, "cq", 28))]
    cmd += ["-b:v", "0"]
    cmd += ["-preset", getattr(args, "preset", "p4")]
    cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-tag:v", "hvc1"]
    # Write moov atom at start so partial files are recoverable,
    # and avoid "Unable to re-open output file for shifting data" on
    # network/FUSE filesystems that don't support seeking after close.
    cmd += ["-movflags", "+faststart"]

    # Video filter
    if vf:
        cmd += ["-vf", vf]

    # Audio codec per track
    audio_filter = getattr(args, "audio_filter", "dynaudnorm=g=11")
    audio_bitrate = getattr(args, "audio_bitrate", "96k")

    for n, (s, lang) in enumerate(zip(selected_audio, audio_langs)):
        cmd += [f"-c:a:{n}", "aac"]
        cmd += [f"-b:a:{n}", audio_bitrate]
        cmd += [f"-ac:{n}", "2"]
        cmd += [f"-ar:{n}", "48000"]   # normalize to 48 kHz (standard for video)

        # Build per-track filter chain.
        # pan= referencing surround channels (FC, BL, BR, SL, SR) fails on
        # stereo/mono sources because those channels don't exist.
        # Strip the pan= prefix for tracks with <= 2 channels.
        src_channels = s.get("channels") or 0
        if src_channels <= 2 and audio_filter.lstrip().startswith("pan="):
            # Remove "pan=stereo|..." prefix, keep remainder of chain
            pan_end = audio_filter.find(",", audio_filter.find("|"))
            if pan_end != -1:
                effective_filter = audio_filter[pan_end + 1:]
            else:
                effective_filter = "acompressor=threshold=-18dB:ratio=2:attack=20:release=200:makeup=1,alimiter=limit=-1.5dB:level=true"
        else:
            effective_filter = audio_filter

        # AAC encoder requires fltp internally -- always append aformat last.
        full_audio_filter = effective_filter + ",aformat=sample_fmts=fltp:channel_layouts=stereo"
        cmd += [f"-filter:a:{n}", full_audio_filter]
        cmd += [f"-metadata:s:a:{n}", f"language={lang}"]

    # Subtitle codec: convert all text subs to mov_text (only format MP4 supports)
    if selected_subs:
        for n, s in enumerate(selected_subs):
            cmd += [f"-c:s:{n}", "mov_text"]
            tags = s.get("tags") or {}
            lang = (tags.get("language") or tags.get("LANGUAGE") or "und").lower()
            cmd += [f"-metadata:s:s:{n}", f"language={lang}"]

    # Container flags
    cmd += ["-movflags", "+faststart"]

    cmd += [out_part]
    return cmd


# ---------------------------------------------------------------------------
# Fallback detection
# ---------------------------------------------------------------------------

FALLBACK_PATTERNS = [
    "auto_scaler_0",
    "Function not implemented",
    "Invalid output format for hwframe download",
    "Error reinitializing filters",
    "Impossible to convert",
    "Unable to parse option value",   # bad filter option (e.g. tonemapx syntax error)
    "Error applying option",          # filter option rejected
    # NVENC session limit hit - fall back to CPU scale, still encode with NVENC
    "OpenEncodeSessionEx failed",
    "No capable devices found",
    "incompatible client key",
]


def stderr_needs_fallback(stderr: str) -> bool:
    for pat in FALLBACK_PATTERNS:
        if pat in stderr:
            return True
    return False


def is_nvenc_session_error(stderr: str) -> bool:
    """True if the error is specifically an NVENC session limit rejection."""
    return (
        "OpenEncodeSessionEx failed" in stderr
        or "incompatible client key" in stderr
        or "No capable devices found" in stderr
    )


# Global semaphore: limits concurrent hevc_nvenc encoder sessions.
# RTX 3060 (and most consumer GPUs) allow max 3 simultaneous NVENC sessions
# without the nvidia-patch. Set to 3 by default; overridden in main() from
# --jobs if the user explicitly sets fewer workers.
_nvenc_semaphore: threading.Semaphore = threading.Semaphore(3)


# ---------------------------------------------------------------------------
# Subtitle copy
# ---------------------------------------------------------------------------

# Subtitle sidecars controlled by the "Copy All Subs" toggle.
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".sub", ".idx", ".sup", ".pgs", ".txt"}

# Non-video sidecar files always copied regardless of toggle.
# .ass/.ssa are here (not in SUBTITLE_EXTENSIONS) because they are also
# embedded as mov_text fallback -- we always want the original alongside.
SIDECAR_EXTENSIONS = {
    ".nfo",           # Jellyfin/Kodi metadata
    ".jpg", ".jpeg",  # poster, fanart, thumb images
    ".png", ".webp",  # poster, fanart, thumb images
    ".tbn",           # Kodi/Emby thumbnail
    ".ass", ".ssa",   # always copy ASS/SSA originals (embedded as mov_text fallback)
}


def copy_sidecar_files(src_video: Path, out_dir: Path, copy_all_subs: bool) -> int:
    """
    Copy subtitle and metadata sidecar files to the output folder.
    If src_video is a directory (DVD root), search within it directly.

    Subtitles (.srt, .ass, etc.):
      - copy_all_subs=True:  copy all subtitle files in the folder
      - copy_all_subs=False: copy only subtitles whose stem starts with the
                              video filename stem (e.g. Movie.en.srt)

    Sidecar files (.nfo, .jpg, .png, etc.):
      Always copied regardless of copy_all_subs -- these are folder-level
      metadata that belong to the output regardless.

    Returns total count of copied files.
    """
    if src_video.is_dir():
        src_dir = src_video
        stem = None
    else:
        src_dir = src_video.parent
        stem = src_video.stem

    count = 0
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        dest = out_dir / f.name
        try:
            if ext in SUBTITLE_EXTENSIONS:
                if copy_all_subs or (stem and f.stem.startswith(stem)):
                    shutil.copy2(f, dest)
                    count += 1
            elif ext in SIDECAR_EXTENSIONS:
                shutil.copy2(f, dest)
                count += 1
        except Exception as e:
            logger.warning(f"Failed to copy sidecar {f}: {e}")
    return count


# ---------------------------------------------------------------------------
# Single job transcoding
# ---------------------------------------------------------------------------

def transcode_job(
    src_path: Path,
    out_dir: Path,
    out_file: Path,
    args: argparse.Namespace,
    has_scale_cuda: bool,
    has_zscale: bool,
    worker_fps: dict,
    worker_id: int,
    ui_state: "UIStateWriter" = None,
    has_tonemap_cuda: bool = False,
    has_hwdec_cuda: set = None,
) -> bool:
    """
    Transcode a single file. Returns True on success.
    Updates DB state.
    """
    ffmpeg_bin = getattr(args, "ffmpeg", "ffmpeg")
    ffprobe_bin = getattr(args, "ffprobe", "ffprobe")
    probesize = getattr(args, "probesize", "200M")
    analyzeduration = getattr(args, "analyzeduration", "200M")
    max_height = getattr(args, "max_height", 720)
    no_hwaccel = getattr(args, "no_hwaccel", False)
    dry_run = getattr(args, "dry_run", False)
    copy_all_subs = getattr(args, "copy_all_subs", False)
    # DB key is always str(src_path) -- for DVDs this is the concat URI
    src = str(src_path)
    # For ffprobe, use the first VOB directly (concat URI confuses stream detection)
    # For ffmpeg, use the concat URI so all segments are processed as one stream
    is_dvd = isinstance(src_path, DVDTitle)
    probe_src = str(src_path.vobs[0]) if is_dvd else src

    db_update_status(src, "running", increment_attempts=True)

    try:
        probe = ffprobe_json(ffprobe_bin, probe_src, probesize, analyzeduration)
    except Exception as e:
        err = str(e)
        logger.error(f"[W{worker_id}] ffprobe failed for {src}: {err}")
        db_update_status(src, "failed", last_error=err[:4000])
        return False

    vs = get_video_stream(probe)
    if vs is None:
        err = "No video stream found"
        logger.warning(f"[W{worker_id}] {err}: {src}")
        db_update_status(src, "failed", last_error=err)
        return False

    audio_streams = get_audio_streams(probe)
    selected_audio, audio_langs = select_audio_tracks(audio_streams)
    selected_subs = get_subtitle_streams(probe)

    hdr = is_hdr(vs)
    height = vs.get("height") or 0
    need_scale = height > max_height and height > 0
    transfer = vs.get("color_transfer", "unknown")
    primaries = vs.get("color_primaries", "unknown")
    space = vs.get("color_space", "unknown")
    pix_fmt = vs.get("pix_fmt", "unknown")

    vf, extra_input_args, pipeline_label = build_video_filter(
        vs, max_height, hdr, has_scale_cuda, has_zscale,
        force_cpu=False, no_hwaccel=no_hwaccel, has_tonemap_cuda=has_tonemap_cuda,
        has_hwdec_cuda=has_hwdec_cuda,
    )

    # Calculate total frame count for progress reporting.
    # Priority:
    #   1. nb_frames from video stream (exact, present in AVI/MP4)
    #   2. duration from video stream × r_frame_rate (MKV often has this)
    #   3. duration from container format × r_frame_rate (MKV fallback —
    #      MKV stores duration at format level, not stream level)
    #   4. avg_frame_rate if r_frame_rate is a timebase (90000/1 etc.)
    total_frames = 0
    try:
        # Resolve duration: stream first, then container format
        dur_str = vs.get("duration")
        if not dur_str or dur_str == "N/A":
            dur_str = probe.get("format", {}).get("duration")

        # Resolve fps: r_frame_rate preferred, avg_frame_rate as fallback
        fps_str = vs.get("r_frame_rate") or vs.get("avg_frame_rate")

        if vs.get("nb_frames") and vs["nb_frames"] != "N/A":
            total_frames = int(vs["nb_frames"])
        elif dur_str and fps_str and dur_str != "N/A":
            dur = float(dur_str)
            num, den = fps_str.split("/")
            fps_src = float(num) / float(den) if float(den) else 0
            # Sanity check: real frame rates are under 120fps.
            # Timebases like 90000/1 would give absurd counts — skip them.
            if 1 <= fps_src <= 120:
                total_frames = int(dur * fps_src)
    except Exception:
        total_frames = 0

    logger.info(
        f"[W{worker_id}] START src={src_path.name} | dst={out_file} | "
        f"height={height} need_scale={need_scale} HDR={hdr} | "
        f"transfer={transfer} primaries={primaries} space={space} pix_fmt={pix_fmt} | "
        f"pipeline={pipeline_label} | vf={vf!r} | "
        f"audio_count={len(selected_audio)} langs={audio_langs} | "
        f"subs={len(selected_subs)} | total_frames={total_frames}"
    )

    if ui_state:
        ui_state.worker_start(worker_id, src_path.name, pipeline_label, total_frames=total_frames)
        ui_state.flush()

    out_dir.mkdir(parents=True, exist_ok=True)
    # out_file is always a real Path; for DVDs src_path.with_suffix returns
    # a real Path too (see DVDTitle.with_suffix), so this is always safe
    out_part = out_dir / (out_file.stem + ".part.mp4")

    if dry_run:
        logger.info(f"[W{worker_id}] DRY-RUN: would transcode {src_path.name}")
        db_update_status(src, "done")
        return True

    # Seconds of no frame progress before worker is considered frozen.
    WATCHDOG_TIMEOUT = 300

    def run_ffmpeg(cmd: list[str], worker_id: int, worker_fps: dict) -> tuple[int, str]:
        """
        Run ffmpeg, parse FPS/frame from -progress pipe:1, return (returncode, stderr).
        stderr is fully drained before returning so pattern matching is reliable.
        The NVENC semaphore is acquired before launching to prevent session exhaustion.
        A watchdog thread kills ffmpeg if no frame progress is seen for WATCHDOG_TIMEOUT seconds.
        """
        stderr_buf = []
        _frame_count = [0]
        _last_progress_time = [time.monotonic()]
        _watchdog_stop = threading.Event()

        # Acquire NVENC session slot before starting ffmpeg.
        # This blocks if 3 workers are already encoding, serializing the 4th
        # until one finishes -- no rejected sessions, no wasted retries.
        _nvenc_semaphore.acquire()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            def read_stderr():
                for line in proc.stderr:
                    stderr_buf.append(line)

            def read_stdout():
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith("fps="):
                        try:
                            fps_val = float(line.split("=", 1)[1])
                            worker_fps[worker_id] = fps_val
                            if ui_state:
                                pct = (_frame_count[0] / total_frames) if total_frames > 0 else 0.0
                                ui_state.worker_update(worker_id, fps_val, _frame_count[0], progress=min(pct, 1.0))
                                ui_state.flush()
                        except ValueError:
                            pass
                    elif line.startswith("frame="):
                        try:
                            new_count = int(line.split("=", 1)[1])
                            if new_count > _frame_count[0]:
                                _frame_count[0] = new_count
                                _last_progress_time[0] = time.monotonic()
                        except ValueError:
                            pass

            def watchdog():
                while not _watchdog_stop.is_set():
                    time.sleep(10)
                    if _watchdog_stop.is_set():
                        break
                    elapsed = time.monotonic() - _last_progress_time[0]
                    if elapsed > WATCHDOG_TIMEOUT:
                        logger.warning(
                            f"[W{worker_id}] WATCHDOG: no frame progress for "
                            f"{int(elapsed)}s on {src_path.name}, killing ffmpeg"
                        )
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        break

            t_err = threading.Thread(target=read_stderr, daemon=True)
            t_out = threading.Thread(target=read_stdout, daemon=True)
            t_wd  = threading.Thread(target=watchdog,    daemon=True)
            t_err.start()
            t_out.start()
            t_wd.start()
            proc.wait()
            _watchdog_stop.set()
            # Join stderr BEFORE releasing semaphore so the full output is
            # captured before the next worker starts and fills the pipe.
            t_err.join(timeout=10)
            t_out.join(timeout=10)
            t_wd.join(timeout=15)
        except Exception as exc:
            return -1, str(exc)
        finally:
            _nvenc_semaphore.release()

        worker_fps.pop(worker_id, None)
        stderr_text = "".join(stderr_buf)
        return proc.returncode, stderr_text

    # First attempt
    cmd = build_ffmpeg_cmd(
        ffmpeg_bin, src, str(out_part), vs, selected_audio, audio_langs,
        vf, extra_input_args, args, is_dvd=is_dvd, selected_subs=selected_subs,
    )
    returncode, stderr = run_ffmpeg(cmd, worker_id, worker_fps)

    # Fallback attempt if needed
    fallback_used = False
    if returncode != 0 and stderr_needs_fallback(stderr):
        logger.warning(
            f"[W{worker_id}] Pipeline error detected "
            f"(nvenc_session={is_nvenc_session_error(stderr)}), "
            f"retrying with CPU-safe pipeline. "
            f"First attempt stderr: {stderr[-2000:].strip()}"
        )
        fallback_used = True
        vf_cpu, _, label_cpu = build_video_filter(
            vs, max_height, hdr, has_scale_cuda=False, has_zscale=has_zscale,
            force_cpu=True, no_hwaccel=no_hwaccel, has_tonemap_cuda=False,
        )
        cmd2 = build_ffmpeg_cmd(
            ffmpeg_bin, src, str(out_part), vs, selected_audio, audio_langs,
            vf_cpu, [], args, is_dvd=is_dvd, selected_subs=selected_subs,
        )
        returncode, stderr = run_ffmpeg(cmd2, worker_id, worker_fps)

    if returncode != 0:
        stderr_tail = stderr[-8000:]
        err_msg = f"ffmpeg failed (rc={returncode}, fallback={fallback_used})\n{stderr_tail}"
        logger.error(f"[W{worker_id}] FAIL {src_path.name}: {err_msg}")
        db_update_status(src, "failed", last_error=err_msg[:8000])
        if out_part.exists():
            out_part.unlink()
        return False

    # Atomic rename
    try:
        out_part.rename(out_file)
    except Exception as e:
        logger.error(f"[W{worker_id}] Failed to rename {out_part} -> {out_file}: {e}")
        db_update_status(src, "failed", last_error=str(e))
        return False

    # Copy subtitles and sidecar files -- for DVDs look in dvd_root
    sub_src = src_path._path if isinstance(src_path, DVDTitle) else src_path
    sub_count = copy_sidecar_files(sub_src, out_dir, copy_all_subs)

    # Extract embedded ASS/SSA subtitle streams to .ass sidecar files
    # so Jellyfin can use the fully styled version alongside the mov_text fallback.
    # Skip for DVDs -- VOB subtitle streams are bitmap-based, not ASS.
    ass_count = 0
    if not is_dvd and selected_subs:
        ass_count = extract_ass_sidecars(
            ffmpeg_bin, probe_src, out_dir, out_file.stem,
            selected_subs, probesize, analyzeduration,
        )

    logger.info(
        f"[W{worker_id}] DONE {out_file} "
        f"(sidecars copied: {sub_count}, ass extracted: {ass_count})"
    )
    db_update_status(src, "done")
    return True


# ---------------------------------------------------------------------------
# UI State writer (JSON IPC for transcode_ui.py dashboard)
# ---------------------------------------------------------------------------

class UIStateWriter:
    """
    Writes a JSON state snapshot to a file on each update so that
    transcode_ui.py can read it and render a live dashboard.
    Thread-safe. No-op if ui_state_path is None.
    """

    def __init__(self, path: Optional[str]) -> None:
        self._path = path
        self._lock = threading.Lock()
        # Per-worker info: {str(worker_id): {filename, fps, frames, elapsed, pipeline, progress}}
        self._workers: dict = {}
        # Completed jobs pending flush (cleared after each write)
        self._pending_completed: list = []
        self._total         = 0
        self._total_workers = 0
        self._done    = 0
        self._failed  = 0
        self._skipped = 0
        self._running = 0

    def set_total(self, n: int) -> None:
        with self._lock:
            self._total = n

    def set_total_workers(self, n: int) -> None:
        with self._lock:
            self._total_workers = n

    def worker_start(self, worker_id: int, filename: str, pipeline: str, total_frames: int = 0) -> None:
        with self._lock:
            self._workers[str(worker_id)] = {
                "filename": filename,
                "fps": 0.0,
                "frames": 0,
                "total_frames": total_frames,
                "elapsed": 0.0,
                "pipeline": pipeline,
                "progress": 0.0,
                "_start": time.monotonic(),
            }

    def worker_update(self, worker_id: int, fps: float, frames: int, progress: float = 0.0) -> None:
        with self._lock:
            key = str(worker_id)
            if key in self._workers:
                w = self._workers[key]
                w["fps"]     = fps
                w["frames"]  = frames
                w["elapsed"] = time.monotonic() - w.get("_start", time.monotonic())
                w["progress"] = progress

    def worker_done(self, worker_id: int, status: str, duration: float, filename: str) -> None:
        with self._lock:
            key = str(worker_id)
            self._workers.pop(key, None)
            self._pending_completed.append({
                "status":   status,
                "filename": filename,
                "duration": round(duration, 1),
            })
            if status == "done":
                self._done += 1
            elif status == "failed":
                self._failed += 1
            elif status == "skipped":
                self._skipped += 1

    def set_running(self, n: int) -> None:
        with self._lock:
            self._running = n

    def flush(self, engine_done: bool = False, engine_error: str = "") -> None:
        if not self._path:
            return
        with self._lock:
            # Build a serializable copy of workers (exclude internal _start key)
            workers_out = {}
            for wid, winfo in self._workers.items():
                workers_out[wid] = {k: v for k, v in winfo.items() if not k.startswith("_")}

            data = {
                "total":         self._total,
                "done":          self._done,
                "failed":        self._failed,
                "skipped":       self._skipped,
                "running":       self._running,
                "total_workers": self._total_workers,
                "workers":       workers_out,
                "new_completed": list(self._pending_completed),
                "engine_done":   engine_done,
                "engine_error":  engine_error,
            }
            self._pending_completed = []

        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple terminal progress bar. Suppressed when a UI state file is active."""

    def __init__(self, total: int, silent: bool = False) -> None:
        self.total = total
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.running = 0
        self.silent = silent
        self._lock = threading.Lock()

    def update(self, done_delta=0, failed_delta=0, skipped_delta=0, running_delta=0) -> None:
        with self._lock:
            self.done += done_delta
            self.failed += failed_delta
            self.skipped += skipped_delta
            self.running += running_delta

    def render(self, worker_fps: dict) -> str:
        with self._lock:
            processed = self.done + self.failed + self.skipped
            bar_width = 30
            frac = processed / self.total if self.total else 0
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            fps_parts = [f"W{wid}:{fps:.1f}fps" for wid, fps in sorted(worker_fps.items())]
            fps_str = " ".join(fps_parts) if fps_parts else ""
            return (
                f"\r[{bar}] {processed}/{self.total} "
                f"done={self.done} fail={self.failed} skip={self.skipped} run={self.running} "
                f"{fps_str}   "
            )

    def print(self, worker_fps: dict) -> None:
        if self.silent:
            return
        sys.stdout.write(self.render(worker_fps))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def worker_thread(
    job_queue: queue.Queue,
    args: argparse.Namespace,
    has_scale_cuda: bool,
    has_zscale: bool,
    progress: ProgressBar,
    worker_fps: dict,
    worker_id: int,
    stop_event: threading.Event,
    ui_state: UIStateWriter,
    has_tonemap_cuda: bool = False,
    has_hwdec_cuda: set = None,
) -> None:
    while not stop_event.is_set():
        try:
            item = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        src_str, out_dir_str, out_file_str = item
        # Reconstruct DVDTitle from concat URI stored in DB, or use plain Path
        if src_str.startswith("concat:"):
            vob_paths = [Path(p) for p in src_str[len("concat:"):].split("|")]
            dvd_root = vob_paths[0].parent
            if dvd_root.name.upper() == "VIDEO_TS":
                dvd_root = dvd_root.parent
            src_path = DVDTitle(dvd_root=dvd_root, dvd_name=dvd_root.name, vobs=vob_paths)
        else:
            src_path = Path(src_str)
        out_dir = Path(out_dir_str)
        out_file = Path(out_file_str)

        progress.update(running_delta=1)
        ui_state.set_running(progress.running)
        t_start = time.monotonic()

        ok = transcode_job(
            src_path, out_dir, out_file, args,
            has_scale_cuda, has_zscale, worker_fps, worker_id,
            ui_state=ui_state, has_tonemap_cuda=has_tonemap_cuda,
            has_hwdec_cuda=has_hwdec_cuda,
        )

        duration = time.monotonic() - t_start
        status   = "done" if ok else "failed"
        ui_state.worker_done(worker_id, status, duration, src_path.name)

        progress.update(running_delta=-1)
        ui_state.set_running(progress.running)
        if ok:
            progress.update(done_delta=1)
        else:
            progress.update(failed_delta=1)

        ui_state.flush()
        job_queue.task_done()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcode videos for Jellyfin/iOS direct play using HEVC NVENC."
    )
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--state-db", default="transcode_state.db", help="SQLite DB path")
    parser.add_argument("--log", default=None, help="Log file path")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    parser.add_argument("--jobs", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--threads", type=int, default=0, help="ffmpeg thread count (0=auto)")
    parser.add_argument("--retry", type=int, default=1, help="Max retries for failed jobs")
    parser.add_argument("--retry-backoff", type=float, default=5.0, help="Seconds between retries")
    parser.add_argument("--max-height", type=int, default=720, help="Max output height")
    parser.add_argument("--cq", type=int, default=28, help="NVENC CQ value")
    parser.add_argument("--preset", default="p4", help="NVENC preset (p1-p7)")
    parser.add_argument("--rc", default="vbr_hq", help="NVENC rate control mode")
    parser.add_argument("--audio-bitrate", default="96k", help="Audio bitrate per track")
    parser.add_argument("--audio-filter", default="acompressor=threshold=-18dB:ratio=2:attack=20:release=200:makeup=1,alimiter=limit=-1.5dB:level=true", help="Audio filter chain")
    parser.add_argument("--copy-all-subs", action="store_true", help="Copy all subtitle files in source folder")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output exists")
    parser.add_argument("--no-hwaccel", action="store_true", help="Disable GPU hwaccel for scaling")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without transcoding")
    parser.add_argument("--probesize", default="200M", help="ffprobe probesize")
    parser.add_argument("--analyzeduration", default="200M", help="ffprobe analyzeduration")
    parser.add_argument("--tv-mode", action="store_true",
        help="TV show mode: mirror full relative path structure instead of collapsing to show name")
    parser.add_argument("--ui-state", default=None,
        help="Path to JSON file for live UI state (used by transcode_ui.py)")

    args = parser.parse_args()

    setup_logging(args.log)

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Transcode engine starting | input={input_dir} output={output_dir} "
        f"jobs={args.jobs} max_height={args.max_height} cq={args.cq} "
        f"preset={args.preset} rc={args.rc} audio_bitrate={args.audio_bitrate} "
        f"skip_existing={args.skip_existing} dry_run={args.dry_run}"
    )

    # Detect filter availability
    has_scale_cuda    = detect_filter(args.ffmpeg, "scale_cuda")    and not args.no_hwaccel
    has_zscale        = detect_filter(args.ffmpeg, "zscale")
    has_tonemap_cuda  = detect_filter(args.ffmpeg, "tonemap_cuda")  and not args.no_hwaccel
    # hwdec: detect CUDA hardware decoder availability by probing the decoder list.
    # Only enabled when GPU pipeline is active (no_hwaccel=False).
    # nvidia-patch removes the 3-session NVENC limit so hwdec no longer
    # competes with NVENC sessions — all workers can decode on GPU freely.
    _hwdec_set        = detect_hwdec_cuda(args.ffmpeg) if not args.no_hwaccel else set()
    has_hwdec_cuda    = _hwdec_set   # set of codec names with cuvid support
    logger.info(
        f"Filter availability: scale_cuda={has_scale_cuda} "
        f"zscale={has_zscale} tonemap_cuda={has_tonemap_cuda} "
        f"hwdec_cuda={sorted(has_hwdec_cuda)}"
    )

    # Connect to DB
    db_connect(args.state_db)

    # Start session timer — returns accumulated total from previous sessions
    _session_total_before = session_start()
    logger.info(f"Session started. Previous total elapsed: {int(_session_total_before)}s")

    # Discover and register jobs
    videos = discover_videos(input_dir)
    logger.info(f"Found {len(videos)} video files.")

    tv_mode = getattr(args, "tv_mode", False)
    _assigned_outputs: set = set()  # track assigned paths within this scan to avoid collisions
    for src_path in videos:
        out_dir, out_file, stem = compute_output_names(
            src_path, input_dir, output_dir,
            tv_mode=tv_mode, assigned=_assigned_outputs,
        )
        db_upsert_job(str(src_path), str(out_dir), str(out_file))

        if args.skip_existing and out_file.exists():
            db_update_status(str(src_path), "skipped", last_error="output exists")

    # Get runnable jobs
    runnable = db_get_runnable(args.retry)
    logger.info(f"Runnable jobs: {len(runnable)}")

    if not runnable:
        logger.info("No jobs to run.")
        sys.exit(0)

    # Setup UI state writer (None = disabled, prints text progress bar instead)
    ui_state_path = getattr(args, "ui_state", None)
    ui_state = UIStateWriter(ui_state_path)
    ui_state.set_total(len(runnable))
    ui_state.set_total_workers(args.jobs)
    ui_state.flush()

    # NVENC session semaphore — set to args.jobs directly.
    # nvidia-patch has been applied on the host, removing the artificial
    # 3-session consumer limit. The semaphore now simply mirrors --jobs
    # so all workers can encode simultaneously.
    global _nvenc_semaphore
    _nvenc_semaphore = threading.Semaphore(args.jobs)
    logger.info(f"NVENC session semaphore set to {args.jobs} (uncapped, nvidia-patch active)")

    # Setup progress and workers
    silent_bar = ui_state_path is not None
    progress = ProgressBar(total=len(runnable), silent=silent_bar)
    worker_fps: dict[int, float] = {}

    job_queue: queue.Queue = queue.Queue()
    for row in runnable:
        job_queue.put((row["src"], row["out_dir"], row["out_file"]))

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received stop signal, draining queue...")
        stop_event.set()
        session_stop()  # persist elapsed before workers drain
        # Empty the queue so workers see empty and stop taking new jobs
        while not job_queue.empty():
            try:
                job_queue.get_nowait()
                job_queue.task_done()
            except queue.Empty:
                break

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = []
    for wid in range(args.jobs):
        t = threading.Thread(
            target=worker_thread,
            args=(job_queue, args, has_scale_cuda, has_zscale,
                  progress, worker_fps, wid, stop_event, ui_state,
                  has_tonemap_cuda, has_hwdec_cuda),
            daemon=True,
            name=f"worker-{wid}",
        )
        t.start()
        threads.append(t)

    # Progress rendering loop
    try:
        while any(t.is_alive() for t in threads):
            progress.print(worker_fps)
            time.sleep(0.5)
            # Check if queue is done
            if job_queue.empty() and progress.running == 0:
                time.sleep(0.5)
                break
    except KeyboardInterrupt:
        stop_event.set()

    # Final join
    job_queue.join()
    for t in threads:
        t.join(timeout=10)

    progress.print(worker_fps)
    print()  # newline after progress bar

    logger.info(
        f"Engine finished | done={progress.done} failed={progress.failed} "
        f"skipped={progress.skipped}"
    )

    # Persist session elapsed to DB for next-start total display
    session_stop()

    ui_state.flush(engine_done=True)


if __name__ == "__main__":
    main()
