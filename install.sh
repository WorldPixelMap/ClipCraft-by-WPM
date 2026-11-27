#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Local ClipCraft by WPM — One-Command Setup (macOS / Linux) ==="

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] FFmpeg not found. Install it first:"
  echo "    macOS: brew install ffmpeg node"
  echo "    Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg nodejs python3-venv"
  exit 1
fi

PY_BIN=""
for cand in python3.12 python3.13 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY_BIN="$cand"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo "[ERROR] Python 3.11+ not found. Please install Python 3.12 first."
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import fastapi, uvicorn, yt_dlp, faster_whisper, cv2, numpy, scipy, pysubs2; from app.face_tracker import ensure_yunet_model; ensure_yunet_model(); print('Setup complete! Run: .venv/bin/python run.py')"
