#!/bin/bash

# Full pipeline execution script for GCP
# This script runs the entire art classification pipeline from data download to evaluation

set -e  # Exit on error

echo "=========================================="
echo "Art Classification Pipeline - Full Run"
echo "=========================================="
echo ""

# Step 1: Download dataset
echo "[1/5] Downloading WikiArt dataset from Hugging Face..."
python scripts/download_dataset.py

# Step 2: Prepare data splits
echo ""
echo "[2/5] Preparing train/val/test splits..."
python scripts/prepare_data.py

# Step 3: Visualize dataset
echo ""
echo "[3/5] Creating dataset visualizations..."
python scripts/visualize_dataset.py

# Step 4: Train models
echo ""
echo "[4/5] Training 2 multi-task models (ResNet + ViT)..."
python train_models.py --batch_size 64 --num_epochs 20 --num_workers 8

# Step 5: Evaluate models
echo ""
echo "[5/5] Evaluating models on test set..."
python evaluate_models.py --split test --batch_size 64 --num_workers 8

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Logs: ./logs/"
echo "  - Reports: ./reports/"
echo "  - Visualizations: ./data/figures/"
echo ""
