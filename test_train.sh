#!/bin/bash

# Quick test: Train ViT for 1 epoch only
# Use this to verify everything works before running full training

set -e  # Exit on error

echo "=========================================="
echo "Quick Training Test - ViT (1 epoch)"
echo "=========================================="
echo ""

echo "Training ViT multi-task model for 1 epoch..."
python train_models.py --models vit --batch_size 64 --num_epochs 1 --num_workers 8

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "If this succeeded, you can run the full training:"
echo "  ./run_train_eval.sh"
echo ""
