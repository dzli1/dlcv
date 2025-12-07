#!/usr/bin/env bash
set -euo pipefail

cd /home/${USER}

git clone https://github.com/YOUR_GITHUB_USERNAME/dlcv.git repo || true
cd repo

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: pull processed dataset from GCS
if [[ -n "${GCS_DATA_BUCKET:-}" ]]; then
  gsutil -m rsync -r "${GCS_DATA_BUCKET}" data/processed
fi

python train.py --arch convnext_tiny --epochs 30 --batch-size 96 --image-size 224

if [[ -n "${GCS_CHECKPOINT_BUCKET:-}" ]]; then
  gsutil cp checkpoints/*.pt "${GCS_CHECKPOINT_BUCKET}"
fi
