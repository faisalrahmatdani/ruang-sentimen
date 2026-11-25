#!/usr/bin/env bash
# Pre-fetch both models (cached after the first run), then serve the app on :7860.
cd "$(dirname "$0")/.."
python -c "
import os
from huggingface_hub import snapshot_download
for k in ('MODEL_SENTIMENT', 'MODEL_EMOTION'):
    snapshot_download(os.environ[k])
" || echo "Gagal mengunduh model lebih dulu; app akan mengunduh saat dipakai."
pkill -f "uvicorn app:app" || true
nohup setsid python -m uvicorn app:app --host 0.0.0.0 --port 7860 > /tmp/ruang-sentimen.log 2>&1 &
echo "Ruang Sentimen berjalan di port 7860 (log: /tmp/ruang-sentimen.log)"
