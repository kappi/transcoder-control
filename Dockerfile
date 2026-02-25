# ── Single-stage build ────────────────────────────────────────────────────────
#
# Base: Ubuntu 22.04 with CUDA 12.x runtime (provides nvidia-smi and NVENC).
# IMPORTANT: Requires the NVIDIA Container Toolkit on the host.
#
# ffmpeg: jellyfin-ffmpeg7 (replaces system ffmpeg).
#   Installed to /usr/lib/jellyfin-ffmpeg/ffmpeg
#   Includes: tonemapx (CUDA HDR tonemapping), scale_cuda, zscale, hevc_nvenc.
#   GPU HDR->SDR tonemapping uses CUDA (tonemapx filter) -- no Vulkan needed.
#
FROM nvcr.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04

LABEL maintainer="transcoder-web"
LABEL description="HEVC transcoding dashboard with NVIDIA NVENC + GPU HDR tonemapping"

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ca-certificates \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── jellyfin-ffmpeg ───────────────────────────────────────────────────────────
# Replaces system ffmpeg. Provides tonemapx (CUDA HDR->SDR), scale_cuda,
# zscale, hevc_nvenc, and all standard filters.
# Binary locations:
#   /usr/lib/jellyfin-ffmpeg/ffmpeg
#   /usr/lib/jellyfin-ffmpeg/ffprobe
ARG JFFMPEG_VERSION=7.1.3-3
ARG JFFMPEG_DISTRO=jammy
RUN wget -q \
    "https://github.com/jellyfin/jellyfin-ffmpeg/releases/download/v${JFFMPEG_VERSION}/jellyfin-ffmpeg7_${JFFMPEG_VERSION}-${JFFMPEG_DISTRO}_amd64.deb" \
    -O /tmp/jellyfin-ffmpeg.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/jellyfin-ffmpeg.deb \
    && rm /tmp/jellyfin-ffmpeg.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 / pip3 resolve to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python3 -m pip install --upgrade pip --quiet

# ── Python dependencies ───────────────────────────────────────────────────────
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── App files ─────────────────────────────────────────────────────────────────
WORKDIR /app

COPY backend/        /app/backend/
COPY frontend/       /app/frontend/
COPY transcode.py    /app/transcode.py
COPY transcode_ui.py     /app/transcode_ui.py
COPY transcode_monitor.py /app/transcode_monitor.py

# ── Persistent directories ────────────────────────────────────────────────────
RUN mkdir -p /input /output /config /logs

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSCODE_PY=/app/transcode.py
ENV CONFIG_FILE=/config/settings.json
ENV UI_STATE_FILE=/tmp/transcode_ui_state.json
ENV LOG_FILE=/logs/transcode.log
ENV STATE_DB=/config/transcode.db
ENV FFMPEG_BIN=/usr/lib/jellyfin-ffmpeg/ffmpeg
ENV FFPROBE_BIN=/usr/lib/jellyfin-ffmpeg/ffprobe

EXPOSE 8080

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/api/status || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python3", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "warning"]
