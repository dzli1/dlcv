#!/bin/bash

# Quick test: Train Gram model for 1 epoch only
# Verifies Gram matrix computation works before full training

set -e  # Exit on error

echo "=========================================="
echo "Testing Gram-Based Style Classifier"
echo "=========================================="
echo ""

echo "Training ResNet Gram Style model for 1 epoch..."
python -c "
import torch
from models import ResNetGramStyleClassifier
from utils import get_dataloaders
import json
import os

# Load metadata
metadata_dir = './data/metadata'
with open(os.path.join(metadata_dir, 'label_mappings.json'), 'r') as f:
    mappings = json.load(f)

num_styles = mappings['num_styles']
print(f'Number of styles: {num_styles}')

# Create model
print('Creating ResNetGramStyleClassifier...')
model = ResNetGramStyleClassifier(num_styles, pretrained=True)
print('✓ Model created successfully')

# Test forward pass
print('Testing forward pass with dummy input...')
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)
print(f'✓ Forward pass successful. Output shape: {output.shape}')

# Test forward_with_aux
print('Testing auxiliary forward pass...')
main_out, gram_out, fused_out = model.forward_with_aux(dummy_input)
print(f'✓ Auxiliary forward pass successful')
print(f'  Main output: {main_out.shape}')
print(f'  Gram output: {gram_out.shape}')
print(f'  Fused output: {fused_out.shape}')

print('')
print('All tests passed! Gram model is working correctly.')
"

echo ""
echo "Now training for 1 real epoch..."
python train_models.py --models resnet --batch_size 64 --num_epochs 1 --num_workers 8 2>&1 | grep -A 50 "resnet_gram_style"

echo ""
echo "=========================================="
echo "Gram Model Test Complete!"
echo "=========================================="
echo ""
echo "If training succeeded, run full training with:"
echo "  python train_models.py --batch_size 64 --num_epochs 20"
echo ""
