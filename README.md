## Style & Artist Classification — Phase 1

This repo now focuses on the multi-task classifier (movement + artist) before branching into the NLP side projects. All tooling is local-first but designed so you can push to Google Cloud for full-scale training.

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Preparation

We rely on the public WikiArt dump from Hugging Face (`huggan/wikiart`). Run the preparation script to filter 1400–1970 Western movements, enforce resolution/medium constraints, deduplicate via perceptual hashes, stratify splits with artist disjointness, and generate plots:

```bash
python dataset_prep.py                 # full run
python dataset_prep.py --max-samples 2 # smoke test
```

Artifacts:

- `data/processed/wikiart_metadata.parquet` — canonical metadata table (movement/artist IDs, splits, stats).
- `data/figures/*.png` — class distributions, year/resolution histograms.
- `data/processed/images/...` — filtered images grouped by movement (only when not using `--skip-export`).

You can tweak filters (movement count thresholds, artist buckets, mediums) via `config.py`.

### Training

`train.py` builds a shared-backbone, dual-head network. Architectures supported: `resnet50`, `convnext_tiny`, `efficientnet_b3`, `vit_base_patch16_224`. Example:

```bash
python train.py \
  --arch convnext_tiny \
  --epochs 30 \
  --batch-size 64 \
  --image-size 224
```

Outputs:

- Best checkpoint under `checkpoints/`.
- Confusion matrices + embeddings for test split.
- Run history appended to `training_history.json`.

### Architecture + Hyperparameter Plan

1. **Baseline**: ResNet-50 to establish reference accuracy (movement + artist heads).
2. **CNN sweep**: EfficientNet-B3 and ConvNeXt-T, tune learning rates (1e-4–5e-4), batch sizes (32–128), and label smoothing (0–0.1).
3. **Transformer sweep**: ViT-B/16 with ImageNet-21k pretraining, try 384px crops and layer-wise LR decay via `--lr` + manual config edits.
4. **Regularization**: Dropout 0.2–0.5, stochastic depth (ConvNeXt), and mixup/cutmix (toggle inside `data_loader.build_transforms` if needed).
5. **Optimizers**: Compare AdamW (default) vs. SGD + momentum for the best-performing backbone.

Track each run with consistent seeds and log per-class F1/confusion matrices for regression analysis (Baroque vs. Rococo confusions, etc.).

### Google Cloud Readiness

- Containerize this repo with `requirements.txt`.
- Upload `data/processed` and `checkpoints` to a GCS bucket if needed (`config.GCP_BUCKET` hook ready).
- Example script stub lives at `scripts/train_on_gcp.sh`; edit with your project/zone and run `bash scripts/train_on_gcp.sh`.

### Next Steps

After we lock the classifier, we can reuse the metadata + embeddings for the NLP descriptor generator and generative side project outlined in `PLAN.md`.

