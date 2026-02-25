#!/usr/bin/env python3
"""
transcode_ui.py - Real-time terminal dashboard for transcode.py

Launches transcode.py as a subprocess and renders a live curses TUI showing:
  - Per-worker job progress (filename, FPS, frames, elapsed time)
  - Completed jobs table (status, duration, outcome)
  - GPU metrics: utilization, power draw, VRAM used/total, temperature
  - CPU utilization and RAM usage
  - Overall progress bar and ETA

Usage:
    python transcode_ui.py --input /media/src --output /media/out [all transcode.py flags]

The UI reads a JSON state file written by transcode.py (--ui-state flag injected automatically).
All original transcode.py flags are forwarded transparently.

Requirements: Python 3.10+, stdlib only.
GPU metrics via nvidia-smi (must be on PATH). CPU/RAM via /proc (Linux).
"""

import argparse
import collections
import curses
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants / layout
# ---------------------------------------------------------------------------

VERSION = "1.0.0"
UI_STATE_FILE = "/tmp/transcode_ui_state.json"
REFRESH_HZ = 4          # screen redraws per second
METRICS_HZ = 2          # hardware metrics polls per second
HISTORY_LEN = 60        # seconds of sparkline history

# Color pair IDs
C_NORMAL   = 0
C_HEADER   = 1
C_ACCENT   = 2
C_SUCCESS  = 3
C_FAIL     = 4
C_WARN     = 5
C_DIM      = 6
C_RUNNING  = 7
C_BAR_FILL = 8
C_BAR_EMPTY= 9
C_BORDER   = 10
C_TITLE    = 11

# Box-drawing chars (UTF-8)
H = "\u2500"   # horizontal
V = "\u2502"   # vertical
TL = "\u250c"  # top-left corner
TR = "\u2510"
BL = "\u2514"
BR = "\u2518"
LT = "\u251c"  # left tee
RT = "\u2524"
TT = "\u252c"  # top tee
BT = "\u2534"
CROSS = "\u253c"
BLOCK_FULL  = "\u2588"
BLOCK_LIGHT = "\u2591"
ARROW = "\u25b6"

SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


# ---------------------------------------------------------------------------
# Hardware metrics collection
# ---------------------------------------------------------------------------

class HardwareMetrics:
    """Thread-safe container for GPU + CPU + RAM readings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # GPU
        self.gpu_util: int = 0          # %
        self.gpu_mem_used: int = 0      # MiB
        self.gpu_mem_total: int = 0     # MiB
        self.gpu_power: float = 0.0     # Watts
        self.gpu_temp: int = 0          # Celsius
        self.gpu_name: str = "GPU"
        self.gpu_available: bool = False
        # CPU
        self.cpu_util: float = 0.0      # %
        self.cpu_count: int = 1
        # RAM
        self.ram_used: int = 0          # MiB
        self.ram_total: int = 0         # MiB
        # History for sparklines (deque of 0-100 floats)
        self.gpu_util_hist:  collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self.cpu_util_hist:  collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self.gpu_power_hist: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        # Uptime
        self.start_time: float = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "gpu_util": self.gpu_util,
                "gpu_mem_used": self.gpu_mem_used,
                "gpu_mem_total": self.gpu_mem_total,
                "gpu_power": self.gpu_power,
                "gpu_temp": self.gpu_temp,
                "gpu_name": self.gpu_name,
                "gpu_available": self.gpu_available,
                "cpu_util": self.cpu_util,
                "cpu_count": self.cpu_count,
                "ram_used": self.ram_used,
                "ram_total": self.ram_total,
                "gpu_util_hist": list(self.gpu_util_hist),
                "cpu_util_hist": list(self.cpu_util_hist),
                "gpu_power_hist": list(self.gpu_power_hist),
            }

    def update_gpu(self, util, mem_used, mem_total, power, temp, name) -> None:
        with self._lock:
            self.gpu_util = util
            self.gpu_mem_used = mem_used
            self.gpu_mem_total = mem_total
            self.gpu_power = power
            self.gpu_temp = temp
            self.gpu_name = name
            self.gpu_available = True
            self.gpu_util_hist.append(float(util))
            self.gpu_power_hist.append(float(power))

    def update_cpu(self, util, count, ram_used, ram_total) -> None:
        with self._lock:
            self.cpu_util = util
            self.cpu_count = count
            self.ram_used = ram_used
            self.ram_total = ram_total
            self.cpu_util_hist.append(float(util))


def poll_nvidia_smi(metrics: HardwareMetrics) -> None:
    """Parse nvidia-smi output and update metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            return
        name     = parts[0]
        util     = int(float(parts[1]))
        mem_used = int(float(parts[2]))
        mem_tot  = int(float(parts[3]))
        power    = float(parts[4])
        temp     = int(float(parts[5]))
        metrics.update_gpu(util, mem_used, mem_tot, power, temp, name)
    except Exception:
        pass


# Linux /proc based CPU/RAM reading
_prev_cpu_times: list[int] = []

def poll_cpu_ram(metrics: HardwareMetrics) -> None:
    global _prev_cpu_times
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = list(map(int, line.split()[1:]))
        total = sum(vals)
        idle  = vals[3] + (vals[4] if len(vals) > 4 else 0)
        if _prev_cpu_times:
            prev_total, prev_idle = _prev_cpu_times
            dtotal = total - prev_total
            didle  = idle  - prev_idle
            util = 100.0 * (1.0 - didle / dtotal) if dtotal > 0 else 0.0
        else:
            util = 0.0
        _prev_cpu_times = [total, idle]

        count = os.cpu_count() or 1

        with open("/proc/meminfo") as f:
            memlines = f.readlines()
        mem_kv = {}
        for ml in memlines:
            parts = ml.split()
            if len(parts) >= 2:
                mem_kv[parts[0].rstrip(":")] = int(parts[1])

        total_kb    = mem_kv.get("MemTotal", 0)
        avail_kb    = mem_kv.get("MemAvailable", mem_kv.get("MemFree", 0))
        used_kb     = total_kb - avail_kb
        ram_used    = used_kb  // 1024
        ram_total   = total_kb // 1024

        metrics.update_cpu(util, count, ram_used, ram_total)
    except Exception:
        pass


def metrics_thread(metrics: HardwareMetrics, stop: threading.Event) -> None:
    interval = 1.0 / METRICS_HZ
    while not stop.is_set():
        poll_nvidia_smi(metrics)
        poll_cpu_ram(metrics)
        time.sleep(interval)


# ---------------------------------------------------------------------------
# UI State reader (JSON written by transcode.py)
# ---------------------------------------------------------------------------

class UIState:
    """Thread-safe store for transcode engine state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total: int = 0
        self.done: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.running: int = 0
        # Per-worker: {worker_id: {filename, fps, frames, elapsed, pipeline}}
        self.workers: dict = {}
        # Completed jobs ring buffer (newest first)
        self.completed: collections.deque = collections.deque(maxlen=200)
        self.started_at: float = time.monotonic()
        self.total_workers: int = 0   # configured --jobs value
        self.engine_done: bool = False
        self.engine_error: str = ""

    def update_from_file(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            with self._lock:
                self.total         = data.get("total", self.total)
                self.done          = data.get("done", self.done)
                self.failed        = data.get("failed", self.failed)
                self.skipped       = data.get("skipped", self.skipped)
                self.running       = data.get("running", self.running)
                self.total_workers = data.get("total_workers", self.total_workers)
                self.workers       = data.get("workers", self.workers)
                new_completed = data.get("new_completed", [])
                for item in new_completed:
                    self.completed.appendleft(item)
                self.engine_done  = data.get("engine_done", False)
                self.engine_error = data.get("engine_error", "")
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "done": self.done,
                "failed": self.failed,
                "skipped": self.skipped,
                "running": self.running,
                "total_workers": self.total_workers,
                "workers": dict(self.workers),
                "completed": list(self.completed),
                "engine_done": self.engine_done,
                "engine_error": self.engine_error,
                "started_at": self.started_at,
            }


def state_reader_thread(state: UIState, path: str, stop: threading.Event) -> None:
    while not stop.is_set():
        state.update_from_file(path)
        time.sleep(0.25)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()

    # Color 8 (bright black / dark gray) only exists on 256-color terminals.
    # Fall back to COLOR_WHITE on 8-color terminals (xterm, plain linux console).
    dim_fg = 8 if curses.COLORS >= 256 else curses.COLOR_WHITE

    curses.init_pair(C_HEADER,    curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(C_ACCENT,    curses.COLOR_CYAN,   -1)
    curses.init_pair(C_SUCCESS,   curses.COLOR_GREEN,  -1)
    curses.init_pair(C_FAIL,      curses.COLOR_RED,    -1)
    curses.init_pair(C_WARN,      curses.COLOR_YELLOW, -1)
    curses.init_pair(C_DIM,       dim_fg,              -1)
    curses.init_pair(C_RUNNING,   curses.COLOR_CYAN,   -1)
    curses.init_pair(C_BAR_FILL,  curses.COLOR_CYAN,   -1)
    curses.init_pair(C_BAR_EMPTY, dim_fg,              -1)
    curses.init_pair(C_BORDER,    curses.COLOR_CYAN,   -1)
    curses.init_pair(C_TITLE,     curses.COLOR_WHITE,  -1)


def safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr that silently clips at window boundaries."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    if x < 0:
        text = text[-x:]
        x = 0
    available = w - x - 1
    if available <= 0:
        return
    try:
        win.addstr(y, x, text[:available], attr)
    except curses.error:
        pass


def draw_box(win, y: int, x: int, h: int, w: int, title: str = "", attr: int = 0) -> None:
    """Draw a box with optional title."""
    if h < 2 or w < 2:
        return
    battr = attr | curses.color_pair(C_BORDER)
    tattr = curses.color_pair(C_TITLE) | curses.A_BOLD

    # Top edge
    safe_addstr(win, y,     x,     TL + H * (w - 2) + TR, battr)
    # Bottom edge
    safe_addstr(win, y+h-1, x,     BL + H * (w - 2) + BR, battr)
    # Sides
    for row in range(1, h - 1):
        safe_addstr(win, y + row, x,         V, battr)
        safe_addstr(win, y + row, x + w - 1, V, battr)

    if title:
        tx = x + 2
        label = f" {title} "
        safe_addstr(win, y, tx, label, tattr)


def draw_hbar(win, y: int, x: int, width: int, frac: float,
              color_fill: int = C_BAR_FILL, color_empty: int = C_BAR_EMPTY) -> None:
    """Draw a horizontal percentage bar."""
    filled = max(0, min(width, int(width * frac)))
    empty  = width - filled
    safe_addstr(win, y, x,          BLOCK_FULL  * filled, curses.color_pair(color_fill)  | curses.A_BOLD)
    safe_addstr(win, y, x + filled, BLOCK_LIGHT * empty,  curses.color_pair(color_empty))


def sparkline(history: list, width: int) -> str:
    """Build a sparkline string from a list of 0-100 floats."""
    if not history:
        return " " * width
    # Sample evenly
    n = len(history)
    if n >= width:
        samples = [history[int(i * n / width)] for i in range(width)]
    else:
        samples = history + [0.0] * (width - n)
    max_val = max(samples) if max(samples) > 0 else 100.0
    result = ""
    for v in samples:
        idx = int((v / max_val) * (len(SPARK_CHARS) - 1))
        result += SPARK_CHARS[idx]
    return result


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_mib(mib: int) -> str:
    if mib >= 1024:
        return f"{mib/1024:.1f}GiB"
    return f"{mib}MiB"


def truncate(s: str, w: int) -> str:
    if len(s) <= w:
        return s
    return s[:w-1] + "\u2026"  # ellipsis


def pct_color(pct: float) -> int:
    if pct >= 90:
        return curses.color_pair(C_FAIL) | curses.A_BOLD
    if pct >= 70:
        return curses.color_pair(C_WARN)
    return curses.color_pair(C_SUCCESS)


# ---------------------------------------------------------------------------
# Screen sections
# ---------------------------------------------------------------------------

def draw_header(win, snap: dict, hw: dict, elapsed: float) -> int:
    """Draw title bar. Returns next Y."""
    h, w = win.getmaxyx()
    title = f"  {ARROW} TRANSCODE DASHBOARD  v{VERSION}  "
    right = f" elapsed: {fmt_duration(elapsed)} "
    safe_addstr(win, 0, 0, " " * w, curses.color_pair(C_HEADER) | curses.A_BOLD)
    safe_addstr(win, 0, 0, title,   curses.color_pair(C_HEADER) | curses.A_BOLD)
    safe_addstr(win, 0, w - len(right) - 1, right, curses.color_pair(C_HEADER) | curses.A_BOLD)
    return 1


def draw_overall_progress(win, y: int, snap: dict) -> int:
    """Draw summary stats + overall bar. Returns next Y."""
    h, w = win.getmaxyx()
    total   = snap["total"]
    done    = snap["done"]
    failed  = snap["failed"]
    skipped = snap["skipped"]
    running = snap["running"]
    processed = done + failed + skipped
    frac = processed / total if total else 0.0

    # ETA
    elapsed = time.monotonic() - snap["started_at"]
    if frac > 0.01 and elapsed > 5:
        eta = elapsed / frac - elapsed
        eta_str = fmt_duration(eta)
    else:
        eta_str = "---"

    draw_box(win, y, 0, 4, w, "OVERALL PROGRESS")
    row = y + 1
    bar_w = w - 24
    bar_x = 1

    # Progress bar
    draw_hbar(win, row, bar_x + 1, bar_w, frac)
    pct_lbl = f" {frac*100:5.1f}%"
    safe_addstr(win, row, bar_x + bar_w + 1, pct_lbl, curses.color_pair(C_ACCENT) | curses.A_BOLD)

    # Stats row
    stats = (
        f"  {ARROW} {processed}/{total} files"
        f"   done={done}"
        f"   fail={failed}"
        f"   skip={skipped}"
        f"   active={running}"
        f"   ETA {eta_str}"
    )
    safe_addstr(win, row + 1, 1, truncate(stats, w - 2), curses.color_pair(C_DIM))

    return y + 4


def draw_workers(win, y: int, snap: dict, max_rows: int) -> int:
    """Draw active worker slots. Returns next Y."""
    h, w = win.getmaxyx()
    workers = snap["workers"]
    # Use the configured job count (total_workers) so the box is always tall enough
    # for all slots, even if some workers haven't started their first file yet.
    # Fall back to len(workers) if total_workers is not in snap.
    num_slots = max(snap.get("total_workers", len(workers)), len(workers), 1)
    # box height: 1 border-top + 1 header row + 1 row per slot + 1 border-bottom
    box_h = num_slots + 3
    # Never shrink below what's needed for workers - ignore max_rows cap here
    # (caller already reserved the right amount of space)

    draw_box(win, y, 0, box_h, w, "ACTIVE WORKERS")

    # Column widths
    wid_col   = 4
    file_col  = max(20, w - 4 - 8 - 8 - 8 - 12 - 5)
    fps_col   = 7
    frames_col= 8
    time_col  = 8
    pipe_col  = 12

    hdr = (
        f" {'WKR':<{wid_col}}"
        f"{'FILENAME':<{file_col}}"
        f"{'FPS':>{fps_col}}"
        f"{'FRAMES':>{frames_col}}"
        f"{'ELAPSED':>{time_col}}"
        f"{'PIPELINE':>{pipe_col}} "
    )
    safe_addstr(win, y + 1, 0, truncate(hdr, w), curses.color_pair(C_DIM))

    row = y + 2
    for wid_str in [str(i) for i in range(num_slots)]:
        if row >= y + box_h - 1:
            break
        winfo = workers.get(wid_str, {})
        if not winfo:
            # Worker slot is idle - show placeholder and continue
            safe_addstr(win, row, 1,
                " W{:<3} [waiting for job...]".format(wid_str),
                curses.color_pair(C_DIM))
            row += 1
            continue
        fname    = winfo.get("filename", "---")
        fps      = winfo.get("fps", 0.0)
        frames   = winfo.get("frames", 0)
        elapsed  = winfo.get("elapsed", 0.0)
        pipeline = winfo.get("pipeline", "")

        # Per-file progress bar (if we have frame info)
        progress = winfo.get("progress", 0.0)  # 0.0-1.0

        fps_color   = curses.color_pair(C_ACCENT) | curses.A_BOLD if fps > 0 else curses.color_pair(C_DIM)
        fname_short = truncate(fname, file_col)

        safe_addstr(win, row, 1,
            f" W{wid_str:<{wid_col-1}}",
            curses.color_pair(C_RUNNING) | curses.A_BOLD)
        safe_addstr(win, row, 1 + wid_col + 1,
            f"{fname_short:<{file_col}}",
            curses.color_pair(C_TITLE))
        safe_addstr(win, row, 1 + wid_col + 1 + file_col,
            f"{fps:>{fps_col}.1f}",
            fps_color)
        safe_addstr(win, row, 1 + wid_col + 1 + file_col + fps_col,
            f"{frames:>{frames_col}}",
            curses.color_pair(C_DIM))
        safe_addstr(win, row, 1 + wid_col + 1 + file_col + fps_col + frames_col,
            f"{fmt_duration(elapsed):>{time_col}}",
            curses.color_pair(C_DIM))
        safe_addstr(win, row, 1 + wid_col + 1 + file_col + fps_col + frames_col + time_col,
            f"{truncate(pipeline, pipe_col):>{pipe_col}}",
            curses.color_pair(C_DIM))

        row += 1

    # (idle slots are already handled inline above)

    return y + box_h


def draw_completed(win, y: int, snap: dict, max_rows: int) -> int:
    """Draw completed jobs ring. Returns next Y."""
    h, w = win.getmaxyx()
    completed = snap["completed"]
    box_h = min(len(completed) + 3, max_rows, h - y - 1)
    box_h = max(box_h, 4)

    draw_box(win, y, 0, box_h, w, "COMPLETED JOBS")

    # Header
    status_col = 7
    dur_col    = 9
    name_col   = max(20, w - status_col - dur_col - 4)
    hdr = (
        f" {'STATUS':<{status_col}}"
        f"{'DURATION':>{dur_col}}"
        f"  {'FILENAME':<{name_col}}"
    )
    safe_addstr(win, y + 1, 0, truncate(hdr, w), curses.color_pair(C_DIM))

    row = y + 2
    for item in completed:
        if row >= y + box_h - 1:
            break
        status   = item.get("status", "?")
        filename = item.get("filename", "")
        duration = item.get("duration", 0.0)

        if status == "done":
            sattr = curses.color_pair(C_SUCCESS) | curses.A_BOLD
            icon  = "\u2714"  # checkmark
        elif status == "failed":
            sattr = curses.color_pair(C_FAIL) | curses.A_BOLD
            icon  = "\u2718"  # cross
        elif status == "skipped":
            sattr = curses.color_pair(C_WARN)
            icon  = "\u25b7"  # skip
        else:
            sattr = curses.color_pair(C_DIM)
            icon  = "?"

        fname_short = truncate(filename, name_col)
        safe_addstr(win, row, 1, f" {icon} {status:<{status_col-2}}", sattr)
        safe_addstr(win, row, 1 + status_col,
            f"{fmt_duration(duration):>{dur_col}}  ", curses.color_pair(C_DIM))
        safe_addstr(win, row, 1 + status_col + dur_col + 2,
            fname_short, curses.color_pair(C_TITLE))
        row += 1

    return y + box_h


def draw_hardware(win, y: int, hw: dict) -> int:
    """Draw GPU + CPU + RAM panel side by side. Returns next Y."""
    h, w = win.getmaxyx()
    panel_h = 7
    if y + panel_h >= h:
        return y

    half = w // 2
    # GPU box (left)
    draw_box(win, y, 0, panel_h, half, f"GPU  {hw.get('gpu_name','')[:20]}")
    # CPU/RAM box (right)
    draw_box(win, y, half, panel_h, w - half, "CPU / RAM")

    # --- GPU panel ---
    gpu_avail = hw.get("gpu_available", False)
    if gpu_avail:
        gpu_util  = hw.get("gpu_util", 0)
        gpu_mem_u = hw.get("gpu_mem_used", 0)
        gpu_mem_t = hw.get("gpu_mem_total", 1)
        gpu_power = hw.get("gpu_power", 0.0)
        gpu_temp  = hw.get("gpu_temp", 0)
        gpu_hist  = hw.get("gpu_util_hist", [])
        pow_hist  = hw.get("gpu_power_hist", [])

        bar_w = half - 16
        mem_frac = gpu_mem_u / gpu_mem_t if gpu_mem_t else 0

        # GPU Util
        safe_addstr(win, y+1, 2, f"UTIL  {gpu_util:3d}%  ", pct_color(gpu_util))
        draw_hbar(win, y+1, 12, bar_w, gpu_util / 100)

        # VRAM
        vram_lbl = f"VRAM  {fmt_mib(gpu_mem_u)}/{fmt_mib(gpu_mem_t)}"
        safe_addstr(win, y+2, 2, vram_lbl, curses.color_pair(C_ACCENT))
        draw_hbar(win, y+2, 12, bar_w, mem_frac)

        # Power + Temp
        safe_addstr(win, y+3, 2,
            f"POWR  {gpu_power:5.0f}W    TEMP {gpu_temp}C",
            curses.color_pair(C_DIM))

        # Sparkline: GPU util history
        spark_w = half - 4
        spark = sparkline(gpu_hist, spark_w)
        safe_addstr(win, y+4, 2, "util ", curses.color_pair(C_DIM))
        safe_addstr(win, y+4, 7, spark, curses.color_pair(C_ACCENT))

        # Sparkline: power history (normalized to 300W max for RTX 3060)
        pow_normalized = [min(100, (v / 170) * 100) for v in pow_hist]
        pspark = sparkline(pow_normalized, spark_w)
        safe_addstr(win, y+5, 2, "powr ", curses.color_pair(C_DIM))
        safe_addstr(win, y+5, 7, pspark, curses.color_pair(C_WARN))
    else:
        safe_addstr(win, y+2, 2, "nvidia-smi not available", curses.color_pair(C_DIM))
        safe_addstr(win, y+3, 2, "GPU metrics disabled", curses.color_pair(C_DIM))

    # --- CPU/RAM panel ---
    cpu_util  = hw.get("cpu_util", 0.0)
    cpu_count = hw.get("cpu_count", 1)
    ram_used  = hw.get("ram_used", 0)
    ram_total = hw.get("ram_total", 1)
    cpu_hist  = hw.get("cpu_util_hist", [])
    ram_frac  = ram_used / ram_total if ram_total else 0
    bar_w2    = w - half - 16

    # CPU
    safe_addstr(win, y+1, half+2, f"CPU   {cpu_util:5.1f}%  ", pct_color(cpu_util))
    draw_hbar(win, y+1, half+12, bar_w2, cpu_util / 100)

    # RAM
    ram_lbl = f"RAM   {fmt_mib(ram_used)}/{fmt_mib(ram_total)}"
    safe_addstr(win, y+2, half+2, ram_lbl, curses.color_pair(C_ACCENT))
    draw_hbar(win, y+2, half+12, bar_w2, ram_frac)

    # CPU core count
    safe_addstr(win, y+3, half+2, f"CORES {cpu_count}", curses.color_pair(C_DIM))

    # Sparkline
    spark_w2 = w - half - 4
    cspark = sparkline(cpu_hist, spark_w2)
    safe_addstr(win, y+4, half+2, "util ", curses.color_pair(C_DIM))
    safe_addstr(win, y+4, half+7, cspark, curses.color_pair(C_SUCCESS))

    return y + panel_h


def draw_footer(win, snap: dict) -> None:
    """Draw status line at very bottom."""
    h, w = win.getmaxyx()
    y = h - 1
    if snap.get("engine_done"):
        msg = f"  Engine finished. Done={snap['done']} Fail={snap['failed']} Skip={snap['skipped']}  [Q to quit]"
        safe_addstr(win, y, 0, " " * (w-1), curses.color_pair(C_SUCCESS) | curses.A_BOLD)
        safe_addstr(win, y, 0, truncate(msg, w-1), curses.color_pair(C_SUCCESS) | curses.A_BOLD)
    elif snap.get("engine_error"):
        msg = f"  ENGINE ERROR: {snap['engine_error']}"
        safe_addstr(win, y, 0, " " * (w-1), curses.color_pair(C_FAIL) | curses.A_BOLD)
        safe_addstr(win, y, 0, truncate(msg, w-1), curses.color_pair(C_FAIL) | curses.A_BOLD)
    else:
        msg = "  [Q] Quit  [R] Reset view  [arrows] Scroll log  "
        safe_addstr(win, y, 0, " " * (w-1), curses.color_pair(C_HEADER))
        safe_addstr(win, y, 0, truncate(msg, w-1), curses.color_pair(C_HEADER))


# ---------------------------------------------------------------------------
# Main curses rendering loop
# ---------------------------------------------------------------------------

def render_loop(
    stdscr,
    state: UIState,
    metrics: HardwareMetrics,
    stop: threading.Event,
    proc: Optional[subprocess.Popen],
) -> None:
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(1000 / REFRESH_HZ))

    while not stop.is_set():
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
            break

        snap = state.snapshot()
        hw   = metrics.snapshot()
        elapsed = time.monotonic() - snap["started_at"]

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        y = draw_header(stdscr, snap, hw, elapsed)
        y = draw_overall_progress(stdscr, y, snap)

        # Layout: workers get exactly what they need (num_slots + 3 rows),
        # completed list gets whatever remains. HW panel=7, footer=1.
        hw_rows     = 7
        footer_rows = 1
        num_slots   = max(snap.get("total_workers", len(snap["workers"])),
                          len(snap["workers"]), 1)
        # 1 top border + 1 col header + num_slots data rows + 1 bottom border
        worker_rows = num_slots + 3
        # Overall progress=4, header=1 already consumed in y
        available   = h - y - hw_rows - footer_rows
        worker_rows = min(worker_rows, available - 2)  # always leave 2 for completed box
        comp_rows   = available - worker_rows

        y = draw_workers(stdscr, y, snap, max_rows=worker_rows)
        y = draw_completed(stdscr, y, snap, max_rows=comp_rows)
        y = draw_hardware(stdscr, y, hw)
        draw_footer(stdscr, snap)

        stdscr.noutrefresh()
        curses.doupdate()

        # Auto-exit after engine done
        if snap.get("engine_done") and proc and proc.poll() is not None:
            time.sleep(3)
            # Wait for user Q
            pass


# ---------------------------------------------------------------------------
# Transcode subprocess launcher with state relay
# ---------------------------------------------------------------------------

def launch_transcode(args_forward: list[str], state_file: str) -> subprocess.Popen:
    """Launch transcode.py with forwarded args + injected --ui-state."""
    cmd = [sys.executable, Path(__file__).parent / "transcode.py"] + args_forward
    cmd += ["--ui-state", state_file]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ---------------------------------------------------------------------------
# Demo / simulation mode (when transcode.py is not present)
# ---------------------------------------------------------------------------

def simulation_writer(state_file: str, stop: threading.Event) -> None:
    """
    Write simulated state to state_file so the UI can be demonstrated
    without running the actual transcoder.
    """
    import random
    import math

    filenames = [
        "The.Grand.Budapest.Hotel.2014.mkv",
        "Blade.Runner.2049.UHD.mkv",
        "Oppenheimer.2023.IMAX.mkv",
        "Dune.Part.Two.2024.mkv",
        "Parasite.2019.Korean.mkv",
        "Everything.Everywhere.2022.mkv",
        "The.Batman.2022.HEVC.mkv",
        "Tenet.2020.4K.mkv",
        "Arrival.2016.mkv",
        "Annihilation.2018.mkv",
        "Hereditary.2018.mkv",
        "Midsommar.2019.mkv",
        "The.Lighthouse.2019.mkv",
        "Portrait.of.a.Lady.2019.mkv",
        "Mad.Max.Fury.Road.2015.mkv",
    ]
    pipelines = ["SDR-GPU(scale_cuda+NVENC)", "HDR->SDR(CPU-zscale+NVENC)", "SDR-direct(NVENC)"]

    total    = len(filenames)
    done     = 0
    failed   = 0
    skipped  = 0
    completed_list = []
    workers  = {}
    pending  = list(filenames)
    in_flight = {}  # wid -> {file, start, fps, frames}

    t0 = time.monotonic()
    num_workers = 3

    while not stop.is_set():
        elapsed = time.monotonic() - t0

        # Assign pending files to idle workers
        for wid in range(num_workers):
            wid_s = str(wid)
            if wid_s not in in_flight and pending:
                fname = pending.pop(0)
                in_flight[wid_s] = {
                    "filename": fname,
                    "start": time.monotonic(),
                    "fps": round(random.uniform(15, 45), 1),
                    "frames": 0,
                    "pipeline": random.choice(pipelines),
                    "duration": random.uniform(5400, 8200),  # total frames
                }

        # Update in-flight
        for wid_s, info in list(in_flight.items()):
            job_elapsed = time.monotonic() - info["start"]
            fps  = info["fps"] + random.uniform(-2, 2)
            info["frames"] = int(fps * job_elapsed)
            info["fps"]    = round(fps, 1)
            workers[wid_s] = {
                "filename": info["filename"],
                "fps": info["fps"],
                "frames": info["frames"],
                "elapsed": job_elapsed,
                "pipeline": info["pipeline"],
                "progress": min(1.0, info["frames"] / info["duration"]),
            }
            # Simulate completion
            if job_elapsed > random.uniform(8, 15):
                status = "done" if random.random() > 0.1 else "failed"
                completed_list.insert(0, {
                    "status": status,
                    "filename": info["filename"],
                    "duration": job_elapsed,
                })
                if status == "done":
                    done += 1
                else:
                    failed += 1
                del in_flight[wid_s]
                del workers[wid_s]

        running = len(in_flight)
        state_data = {
            "total":         total,
            "done":          done,
            "failed":        failed,
            "skipped":       skipped,
            "running":       running,
            "workers":       workers,
            "new_completed": completed_list[:5],
            "engine_done":   (not pending and not in_flight),
            "engine_error":  "",
        }
        # Persist accumulated
        completed_list = completed_list[:200]

        try:
            with open(state_file, "w") as f:
                json.dump(state_data, f)
        except Exception:
            pass

        if not pending and not in_flight:
            time.sleep(2)
            stop.set()

        time.sleep(0.25)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Transcoding TUI dashboard. All unknown flags are forwarded to transcode.py.",
        add_help=True,
    )
    parser.add_argument("--demo", action="store_true",
        help="Run in demo/simulation mode without launching transcode.py")
    parser.add_argument("--state-file", default=UI_STATE_FILE,
        help=f"Path for IPC state JSON (default: {UI_STATE_FILE})")
    parser.add_argument("--no-gpu-metrics", action="store_true",
        help="Disable nvidia-smi polling")

    known, forward = parser.parse_known_args()
    return known, forward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ui_args, forward_args = parse_args()
    state_file = ui_args.state_file

    # Clean up stale state file
    try:
        os.unlink(state_file)
    except FileNotFoundError:
        pass

    state   = UIState()
    metrics = HardwareMetrics()
    stop    = threading.Event()
    proc    = None

    # Start hardware metrics poller
    if not ui_args.no_gpu_metrics:
        mt = threading.Thread(
            target=metrics_thread, args=(metrics, stop), daemon=True
        )
        mt.start()

    # Start state reader
    srt = threading.Thread(
        target=state_reader_thread, args=(state, state_file, stop), daemon=True
    )
    srt.start()

    if ui_args.demo:
        # Simulation mode
        sim_t = threading.Thread(
            target=simulation_writer, args=(state_file, stop), daemon=True
        )
        sim_t.start()
    else:
        # Check transcode.py exists
        transcode_py = Path(__file__).parent / "transcode.py"
        if not transcode_py.exists():
            print(f"ERROR: transcode.py not found at {transcode_py}")
            print("Run with --demo to see the UI without transcoding.")
            sys.exit(1)

        if not forward_args:
            print("ERROR: No arguments forwarded to transcode.py (at minimum --input and --output).")
            print("Example: python transcode_ui.py --input /src --output /dst")
            print("         python transcode_ui.py --demo   (simulation mode)")
            sys.exit(1)

        proc = launch_transcode(forward_args, state_file)

        # Monitor subprocess for premature exit
        def watch_proc():
            proc.wait()
            if proc.returncode != 0:
                state._lock and None  # force update
            stop.set()

        wt = threading.Thread(target=watch_proc, daemon=True)
        wt.start()

    def handle_signal(signum, frame):
        stop.set()
        if proc and proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run curses
    try:
        curses.wrapper(render_loop, state, metrics, stop, proc)
    except Exception as e:
        stop.set()
        if proc and proc.poll() is None:
            proc.terminate()
        print(f"\nUI error: {e}")
        raise

    # Cleanup
    stop.set()
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    try:
        os.unlink(state_file)
    except FileNotFoundError:
        pass

    print(f"\nSession ended. Done={state.done} Failed={state.failed} Skipped={state.skipped}")


if __name__ == "__main__":
    main()

