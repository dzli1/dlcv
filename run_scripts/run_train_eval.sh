#!/bin/bash

# Train and evaluate only (assumes data is already prepared)
# Use this if you've already run the preprocessing steps

set -e  # Exit on error

echo "=========================================="
echo "Training and Evaluation"
echo "=========================================="
echo ""

# Train models
echo "[1/2] Training 2 multi-task models (ResNet + ViT)..."
python train_models.py --batch_size 64 --num_epochs 20 --num_workers 8

# Evaluate models
echo ""
echo "[2/2] Evaluating models on test set..."
python evaluate_models.py --split test --batch_size 64 --num_workers 8

echo ""
echo "=========================================="
echo "Training and Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Logs: ./logs/"
echo "  - Reports: ./reports/"
echo ""
