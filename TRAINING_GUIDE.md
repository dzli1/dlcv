# Training Guide

## Step 1: Prepare Dataset (First Time Only)

If you haven't prepared the dataset yet, you need to download and process the WikiArt dataset:

```bash
python dataset_prep.py
```

This will:
- Download the WikiArt dataset from Hugging Face
- Filter paintings by year (1400-1970), resolution, medium, and genre
- Create train/val/test splits
- Generate label mappings for movements and artists
- Save processed data to `data/processed/`

**Note**: This may take a while depending on your internet connection and dataset size.

## Step 2: Train the Model

### Basic Training (Default Settings)

```bash
python train.py
```

This uses the default settings:
- Architecture: `convnext_tiny`
- Epochs: 25
- Batch size: 64
- Learning rate: 2e-4
- Image size: 224

### Custom Training

You can customize various parameters:

```bash
python train.py \
    --arch convnext_tiny \
    --epochs 25 \
    --batch-size 64 \
    --lr 2e-4 \
    --dropout 0.3 \
    --label-smoothing 0.05
```

### Available Architectures

You can choose from these architectures:
- `resnet50` - Classic ResNet
- `convnext_tiny` - ConvNeXt (default, recommended)
- `efficientnet_b3` - EfficientNet
- `vit_base_patch16_224` - Vision Transformer

Example:
```bash
python train.py --arch resnet50
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `convnext_tiny` | Model architecture |
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--val-batch-size` | 64 | Validation batch size |
| `--image-size` | 224 | Input image size |
| `--lr` | 2e-4 | Learning rate |
| `--weight-decay` | 0.05 | Weight decay for optimizer |
| `--dropout` | 0.3 | Dropout rate |
| `--label-smoothing` | 0.05 | Label smoothing factor |
| `--grad-accum` | 1 | Gradient accumulation steps |
| `--no-pretrained` | False | Disable pretrained weights |
| `--checkpoint-name` | None | Custom checkpoint name |

### Examples

**Quick test run (fewer epochs):**
```bash
python train.py --epochs 5 --batch-size 32
```

**Train ResNet50:**
```bash
python train.py --arch resnet50 --epochs 30
```

**Train with smaller batch size (if you run out of memory):**
```bash
python train.py --batch-size 32 --val-batch-size 32
```

**Train without pretrained weights:**
```bash
python train.py --no-pretrained
```

## Step 3: Monitor Training

During training, you'll see output like:
```
Epoch 1/25 | Train Loss 2.3456 | Val Loss 1.9876 | Val Movement Acc 0.234 | Val Artist Acc 0.156
```

The script will:
- Save the best model to `checkpoints/{arch}_best.pt`
- Save confusion matrices to `reports/confusion_matrices/`
- Save test embeddings to `checkpoints/{arch}_test_embeddings.npz`
- Save training history to `training_history.json`

## Step 4: Run Visualizations (After Training)

After training, you can visualize model predictions:

```bash
python run_visualizations.py \
    --checkpoint checkpoints/convnext_tiny_best.pt \
    --mode both \
    --task movement
```

Options:
- `--checkpoint`: Path to checkpoint file
- `--mode`: `gradcam`, `embeddings`, or `both`
- `--task`: `movement` or `artist`
- `--image-path`: Path to specific image (optional)
- `--num-samples`: Number of samples to visualize (default: 5)

## Troubleshooting

### Out of Memory Errors

If you get CUDA out of memory errors:
1. Reduce batch size: `--batch-size 32` or `--batch-size 16`
2. Use gradient accumulation: `--grad-accum 2` (effectively doubles batch size)
3. Reduce image size: `--image-size 192`

### Missing Dataset Files

If you get errors about missing parquet files:
1. Make sure you ran `python dataset_prep.py` first
2. Check that `data/processed/wikiart_splits.parquet` exists
3. Check that `data/processed/label_mappings.json` exists

### Slow Training

- Reduce `NUM_WORKERS` in `config.py` if you have CPU issues
- Use a smaller architecture like `resnet50`
- Reduce image size

## Quick Start Summary

```bash
# 1. Prepare dataset (first time only)
python dataset_prep.py

# 2. Train model
python train.py

# 3. Visualize results
python run_visualizations.py --checkpoint checkpoints/convnext_tiny_best.pt
```

