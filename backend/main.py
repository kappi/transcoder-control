"""
Transcode Web Backend - FastAPI application
Manages transcode.py as a persistent service, streams live state via WebSocket,
exposes REST endpoints for configuration and job management.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).parent.parent
FRONTEND_DIR   = BASE_DIR / "frontend"
TRANSCODE_PY   = Path(os.environ.get("TRANSCODE_PY", BASE_DIR / "transcode.py"))
CONFIG_FILE    = Path(os.environ.get("CONFIG_FILE", "/config/settings.json"))
UI_STATE_FILE  = Path(os.environ.get("UI_STATE_FILE", "/tmp/transcode_ui_state.json"))
LOG_FILE       = Path(os.environ.get("LOG_FILE", "/logs/transcode.log"))
STATE_DB       = Path(os.environ.get("STATE_DB", "/config/transcode.db"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("transcode-web")

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "input_dir":       "/input",
    "output_dir":      "/output",
    "jobs":            3,
    "threads":         4,
    "cq":              28,
    "preset":          "p4",
    "rc":              "vbr_hq",
    "audio_bitrate":   "96k",
    "audio_filter":    "acompressor=threshold=-18dB:ratio=2:attack=20:release=200:makeup=1,alimiter=limit=-1.5dB:level=true",
    "max_height":      720,
    "retry":           1,
    "retry_backoff":   5.0,
    "probesize":       "200M",
    "analyzeduration": "200M",
    "skip_existing":   True,
    "copy_all_subs":   False,
    "no_hwaccel":      False,
    "dry_run":         False,
    "tv_mode":         False,
}

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            cfg = {**DEFAULT_CONFIG, **saved}
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Nvidia-smi hardware metrics
# ---------------------------------------------------------------------------

_hw_cache: dict = {}
_hw_cache_time: float = 0.0
_HW_TTL = 1.5  # seconds


async def get_hw_metrics() -> dict:
    global _hw_cache, _hw_cache_time
    now = time.monotonic()
    if now - _hw_cache_time < _HW_TTL:
        return _hw_cache

    result = {"gpu_available": False, "cpu_util": 0.0, "ram_used": 0, "ram_total": 0}

    # GPU via nvidia-smi
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=4)
        line = stdout.decode().strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            result.update({
                "gpu_available": True,
                "gpu_name":      parts[0],
                "gpu_util":      int(float(parts[1])),
                "gpu_mem_used":  int(float(parts[2])),
                "gpu_mem_total": int(float(parts[3])),
                "gpu_power":     round(float(parts[4]), 1),
                "gpu_temp":      int(float(parts[5])),
            })
    except Exception:
        pass

    # CPU + RAM via /proc
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = list(map(int, line.split()[1:]))
        total = sum(vals)
        idle  = vals[3] + (vals[4] if len(vals) > 4 else 0)

        prev = _hw_cache.get("_cpu_raw")
        if prev:
            dt = total - prev[0]
            di = idle  - prev[1]
            result["cpu_util"] = round(100.0 * (1.0 - di / dt) if dt > 0 else 0.0, 1)
        result["_cpu_raw"] = [total, idle]
        result["cpu_count"] = os.cpu_count() or 1

        with open("/proc/meminfo") as f:
            memlines = f.readlines()
        mem = {}
        for ml in memlines:
            p = ml.split()
            if len(p) >= 2:
                mem[p[0].rstrip(":")] = int(p[1])
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        result["ram_used"]  = (total_kb - avail_kb) // 1024
        result["ram_total"] = total_kb // 1024
    except Exception:
        pass

    _hw_cache = result
    _hw_cache_time = now
    return result


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

class TranscodeService:
    """
    Manages transcode.py as a long-running subprocess.
    Reads the UI state JSON file to provide live job status.
    """

    def __init__(self) -> None:
        self.proc:          Optional[asyncio.subprocess.Process] = None
        self.status:        str  = "stopped"   # stopped | starting | running | stopping
        self.started_at:    Optional[float] = None
        self.last_state:    dict = {}
        self.log_tail:      list[str] = []     # last N log lines (ring buffer)
        self._log_max:      int  = 500
        self._lock:         asyncio.Lock = asyncio.Lock()
        self._log_task:     Optional[asyncio.Task] = None
        self._watch_task:   Optional[asyncio.Task] = None

    async def start(self, cfg: dict) -> None:
        async with self._lock:
            if self.proc and self.proc.returncode is None:
                raise RuntimeError("Service is already running")

            # Build transcode.py command
            cmd = [sys.executable, str(TRANSCODE_PY)]
            cmd += ["--input",            cfg["input_dir"]]
            cmd += ["--output",           cfg["output_dir"]]
            cmd += ["--state-db",         str(STATE_DB)]
            cmd += ["--log",              str(LOG_FILE)]
            cmd += ["--jobs",             str(cfg["jobs"])]
            cmd += ["--threads",          str(cfg["threads"])]
            cmd += ["--cq",               str(cfg["cq"])]
            cmd += ["--preset",           cfg["preset"]]
            cmd += ["--rc",               cfg["rc"]]
            cmd += ["--audio-bitrate",    cfg["audio_bitrate"]]
            cmd += ["--audio-filter",     cfg["audio_filter"]]
            cmd += ["--max-height",       str(cfg["max_height"])]
            cmd += ["--retry",            str(cfg["retry"])]
            cmd += ["--retry-backoff",    str(cfg["retry_backoff"])]
            cmd += ["--probesize",        cfg["probesize"]]
            cmd += ["--analyzeduration",  cfg["analyzeduration"]]
            cmd += ["--ui-state",         str(UI_STATE_FILE)]

            # Use jellyfin-ffmpeg if available (set via FFMPEG_BIN/FFPROBE_BIN env)
            ffmpeg_bin  = os.environ.get("FFMPEG_BIN",  "ffmpeg")
            ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe")
            cmd += ["--ffmpeg",  ffmpeg_bin]
            cmd += ["--ffprobe", ffprobe_bin]

            if cfg.get("skip_existing"):   cmd += ["--skip-existing"]
            if cfg.get("copy_all_subs"):   cmd += ["--copy-all-subs"]
            if cfg.get("no_hwaccel"):      cmd += ["--no-hwaccel"]
            if cfg.get("dry_run"):         cmd += ["--dry-run"]
            if cfg.get("tv_mode"):         cmd += ["--tv-mode"]

            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_DB.parent.mkdir(parents=True, exist_ok=True)

            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.status     = "running"
            self.started_at = time.monotonic()
            self.log_tail   = []

            logger.info(f"transcode.py started (pid={self.proc.pid})")

            # Tasks to read stdout and watch for exit
            self._log_task   = asyncio.create_task(self._read_logs())
            self._watch_task = asyncio.create_task(self._watch_exit())

    async def stop(self) -> None:
        async with self._lock:
            if not self.proc or self.proc.returncode is not None:
                self.status = "stopped"
                return
            self.status = "stopping"
            self.proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
            self.status = "stopped"
            logger.info("transcode.py stopped")

    async def _read_logs(self) -> None:
        try:
            async for raw in self.proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                self.log_tail.append(line)
                if len(self.log_tail) > self._log_max:
                    self.log_tail = self.log_tail[-self._log_max:]
        except Exception:
            pass

    async def _watch_exit(self) -> None:
        try:
            await self.proc.wait()
        except Exception:
            pass
        if self.status != "stopping":
            self.status = "stopped"
            logger.info(f"transcode.py exited (rc={self.proc.returncode})")

    def read_ui_state(self) -> dict:
        try:
            with open(UI_STATE_FILE) as f:
                data = json.load(f)
            self.last_state = data
            return data
        except Exception:
            return self.last_state

    def get_uptime(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


# Global service instance
service = TranscodeService()

# ---------------------------------------------------------------------------
# DB helpers (read-only for the web UI)
# ---------------------------------------------------------------------------

def db_get_jobs(limit: int = 200) -> list[dict]:
    if not STATE_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(STATE_DB), timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"DB read failed: {e}")
        return []


def db_reset_failed() -> int:
    if not STATE_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(STATE_DB), timeout=5)
        cur = conn.execute(
            "UPDATE jobs SET status='pending', attempts=0, last_error=NULL WHERE status='failed'"
        )
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return affected
    except Exception as e:
        logger.warning(f"DB reset failed: {e}")
        return 0


def db_clear_done() -> int:
    if not STATE_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(STATE_DB), timeout=5)
        cur = conn.execute("DELETE FROM jobs WHERE status IN ('done', 'skipped')")
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return affected
    except Exception as e:
        logger.warning(f"DB clear failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Transcode Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigModel(BaseModel):
    input_dir:       str
    output_dir:      str
    jobs:            int
    threads:         int
    cq:              int
    preset:          str
    rc:              str
    audio_bitrate:   str
    audio_filter:    str
    max_height:      int
    retry:           int
    retry_backoff:   float
    probesize:       str
    analyzeduration: str
    skip_existing:   bool
    copy_all_subs:   bool
    no_hwaccel:      bool
    dry_run:         bool
    tv_mode:         bool


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.get("/api/config")
async def get_config():
    return load_config()


@app.post("/api/config")
async def post_config(cfg: ConfigModel):
    data = cfg.model_dump()
    save_config(data)
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    hw = await get_hw_metrics()
    ui = service.read_ui_state()
    return {
        "service_status": service.status,
        "uptime":         round(service.get_uptime()),
        "pid":            service.proc.pid if service.is_running() else None,
        "hw":             hw,
        "ui":             ui,
    }


@app.post("/api/service/start")
async def start_service(background_tasks: BackgroundTasks):
    if service.is_running():
        raise HTTPException(400, "Service is already running")
    cfg = load_config()

    # Validate paths exist
    input_path = Path(cfg["input_dir"])
    if not input_path.exists():
        raise HTTPException(400, f"Input directory does not exist: {cfg['input_dir']}")

    async def _start():
        try:
            await service.start(cfg)
        except Exception as e:
            logger.error(f"Failed to start service: {e}")

    background_tasks.add_task(_start)
    return {"ok": True, "message": "Service starting..."}


@app.post("/api/service/stop")
async def stop_service():
    if not service.is_running():
        raise HTTPException(400, "Service is not running")
    await service.stop()
    return {"ok": True}


@app.get("/api/jobs")
async def get_jobs(limit: int = 200):
    return db_get_jobs(limit)


@app.post("/api/jobs/reset-failed")
async def reset_failed():
    n = db_reset_failed()
    return {"ok": True, "affected": n}


@app.post("/api/jobs/clear-done")
async def clear_done():
    n = db_clear_done()
    return {"ok": True, "affected": n}


def db_reset_all() -> int:
    if not STATE_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(STATE_DB), timeout=5)
        cur = conn.execute(
            "UPDATE jobs SET status='pending', attempts=0, last_error=NULL"
        )
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return affected
    except Exception as e:
        logger.warning(f"DB reset all failed: {e}")
        return 0


def db_reset_done() -> int:
    if not STATE_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(STATE_DB), timeout=5)
        cur = conn.execute(
            "UPDATE jobs SET status='pending', attempts=0, last_error=NULL WHERE status='done'"
        )
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return affected
    except Exception as e:
        logger.warning(f"DB reset done failed: {e}")
        return 0


@app.post("/api/jobs/reset-all")
async def reset_all():
    n = db_reset_all()
    return {"ok": True, "affected": n}


@app.post("/api/jobs/wipe-db")
async def wipe_db():
    """Delete the entire SQLite database file. Engine must not be running."""
    if service.is_running():
        raise HTTPException(400, "Stop the service before wiping the database")
    try:
        if STATE_DB.exists():
            STATE_DB.unlink()
            logger.info("Database wiped")
            return {"ok": True, "message": "Database deleted. Will be recreated on next start."}
        return {"ok": True, "message": "Database did not exist."}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete database: {e}")


@app.post("/api/jobs/reset-done")
async def reset_done():
    n = db_reset_done()
    return {"ok": True, "affected": n}


@app.get("/api/logs")
async def get_logs(lines: int = 200):
    return {"lines": service.log_tail[-lines:]}


@app.get("/api/scan")
async def scan_input():
    """Return a count of video files found in the input directory."""
    cfg = load_config()
    input_dir = Path(cfg["input_dir"])
    if not input_dir.exists():
        return {"found": 0, "error": "Input directory does not exist"}
    VIDEO_EXT = {".mkv",".mp4",".m4v",".avi",".mov",".wmv",
                 ".flv",".ts",".m2ts",".mts",".vob",".mpg",".mpeg",".webm"}
    count = 0
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXT:
                count += 1
    return {"found": count, "input_dir": str(input_dir)}


# ---------------------------------------------------------------------------
# WebSocket - live dashboard feed
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c != ws]

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            try:
                # Compose live snapshot
                hw = await get_hw_metrics()
                ui = service.read_ui_state()
                msg = {
                    "type":           "state",
                    "service_status": service.status,
                    "uptime":         round(service.get_uptime()),
                    "hw":             hw,
                    "ui":             ui,
                    "log_tail":       service.log_tail[-50:],
                    "ts":             time.time(),
                }
                await ws.send_json(msg)
            except Exception as e:
                logger.debug(f"WS send error: {e}")
                break
            await asyncio.sleep(0.75)
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
