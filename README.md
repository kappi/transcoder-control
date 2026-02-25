# 🎬 Transcoder // Control

A self-hosted, GPU-accelerated video transcoding pipeline for pre-converting a movie/TV library to HEVC MP4 for Jellyfin direct play. Built for an NVIDIA GPU with a web dashboard for monitoring and control.

---

## Features

- **NVIDIA NVENC** hardware encoding (`hevc_nvenc`) — fast, efficient H.265 output
- **GPU HDR→SDR tonemapping** via `tonemap_cuda` (jellyfin-ffmpeg) — no CPU bottleneck on HDR content
- **GPU scaling** via `scale_cuda` with CPU fallback
- **Parallel workers** — configurable job count, semaphore-limited NVENC sessions
- **nvidia-patch** support — detects unlimited NVENC sessions automatically
- **Embedded subtitle handling** — text subs embedded as `mov_text`; ASS/SSA also extracted as sidecar `.ass` files
- **DVD concat support** — handles `VIDEO_TS/` and flat VOB structures automatically
- **TV show mode** — mirrors Season/Episode folder structure in output
- **Sidecar file copying** — `.nfo`, posters, subtitles copied alongside transcoded files
- **Web dashboard** — real-time worker status, GPU/CPU metrics, job queue, log viewer
- **Dark/light theme** — toggle in the UI, preference saved locally
- **Mobile-friendly** — responsive layout with bottom navigation on small screens
- **Automatic CPU fallback** — pipeline errors trigger retry with software pipeline

---

## Stack

| Component | Technology |
|-----------|-----------|
| Transcoder engine | Python 3.10, jellyfin-ffmpeg 7.x |
| Backend API | FastAPI + uvicorn |
| Frontend | Single-file HTML/JS/CSS |
| Database | SQLite (WAL mode) |
| Container | Docker + NVIDIA Container Toolkit |
| GPU | NVIDIA RTX (tested on RTX 3060) |

---

## Requirements

- Docker + Docker Compose
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 525 on host
- [nvidia-patch](https://github.com/keylase/nvidia-patch) (optional, removes 3-session NVENC limit)

---

## Quick Start

**1. Clone the repo:**
```bash
git clone https://github.com/kappi/transcoder-control.git
cd transcoder-control
```

**2. Edit `docker-compose.yml`** — set your input and output volume paths:
```yaml
volumes:
  - /your/movies:/input:ro
  - /your/output:/output
```

**3. Build and start:**
```bash
docker compose up -d --build
```

**4. Open the dashboard:**
```
http://localhost:8080
```

**5. Configure and start transcoding:**
- Go to **Settings**, set input/output directories, click **Scan** to verify
- Click **▶ Start** — workers will begin processing

---

## Configuration

All settings are available in the web UI under **Settings**. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| Workers | 4 | Parallel transcode jobs |
| Max Height | 720 | Downscale target (px) |
| CQ Value | 32 | Quality (lower = better, larger file) |
| NVENC Preset | p4 | Speed/quality tradeoff (p1=fastest, p7=best) |
| Rate Control | vbr | NVENC rate control mode |
| Audio Bitrate | 96k | Per-track AAC bitrate |
| TV Show Mode | off | Preserve Season/Episode folder structure |
| Skip Existing | on | Skip files already transcoded |
| No HW Accel | off | Force CPU-only pipeline |

---

## Pipeline

### HDR content (GPU)
```
CPU decode → format=p010le → hwupload_cuda → scale_cuda → tonemap_cuda → hwdownload → hevc_nvenc
```
Requires `tonemap_cuda` filter (included in jellyfin-ffmpeg). Falls back to CPU zscale+hable if unavailable.

### SDR content (GPU scale)
```
CPU decode → format=nv12 → hwupload_cuda → scale_cuda → hwdownload → hevc_nvenc
```

### Fallback (CPU)
```
CPU decode → scale → zscale (HDR) → tonemap → hevc_nvenc
```
Triggered automatically on pipeline errors.

---

## Output Structure

### Movies
```
/output/
  Movie Title (2024)/
    Movie Title (2024).mp4       ← video + embedded mov_text subs
    Movie Title (2024).eng.ass   ← extracted ASS sidecar (full styling)
    Movie Title (2024).ces.ass
    poster.jpg
    movie.nfo
```

### TV Shows (TV Mode enabled)
```
/output/
  Show Name/
    Season 01/
      Show Name - S01E01.mp4
      Show Name - S01E01.eng.ass
```

---

## Subtitle Handling

| Type | Handling |
|------|----------|
| SRT, WebVTT, mov_text | Embedded in MP4 as `mov_text` |
| ASS / SSA | Embedded as `mov_text` + extracted as `.ass` sidecar |
| PGS, DVDSUB (bitmap) | Dropped — not supported in MP4 |

Jellyfin picks up `.ass` sidecar files automatically for full subtitle styling.

---

## Access

This tool is designed for **local network use**. For remote access, [Tailscale](https://tailscale.com/) is recommended — zero config, no open ports.

---

## Docker Compose Example

```yaml
services:
  transcoder-web:
    build: .
    container_name: transcoder-web
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /your/movies:/input:ro
      - /your/output:/output
      - ./config:/config
      - ./logs:/logs
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Development

Update frontend (volume-mounted, live reload):
```bash
# Just edit frontend/index.html and refresh browser
```

Update transcode engine (hot-swap without rebuild):
```bash
docker cp transcode.py transcoder-web:/app/transcode.py
# Takes effect on next Start
```

Full rebuild:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## License

MIT
