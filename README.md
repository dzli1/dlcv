# Deep Learning Art Classification

A comprehensive deep learning project for classifying Western art by **artistic style** and **artist** using state-of-the-art computer vision models.

## Project Overview

This project trains and evaluates multiple model architectures on the WikiArt dataset to classify paintings by:
- **Style** (e.g., Impressionism, Baroque, Cubism, etc.)
- **Artist** (e.g., Vincent van Gogh, Pablo Picasso, etc.)

### Model Architectures
- **ResNet50**: Convolutional neural network with residual connections
  - Style-only classifier
  - Artist-only classifier
  - Multi-task classifier (style + artist)

- **Vision Transformer (ViT)**: Attention-based architecture
  - Style-only classifier
  - Artist-only classifier
  - Multi-task classifier (style + artist)

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for GCP) or Apple Silicon (MPS) for local testing
- 50GB+ free disk space for dataset

### Installation

1. **Clone the repository** (if on GCP, upload files):
```bash
cd dlcv
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage Workflow

Follow these steps in order to run the complete pipeline:

### Step 1: Download Dataset

Download the WikiArt dataset from Hugging Face and filter to top 10-12 styles and 50-100 artists:

```bash
python scripts/download_dataset.py
```

This will:
- Download WikiArt from Hugging Face
- Filter to most common styles (with at least 500 images)
- Filter to most common artists (with at least 50 images)
- Save filtered images to `data/raw/`
- Create metadata files in `data/metadata/`

**Expected output:**
- `data/raw/` - Filtered images
- `data/metadata/dataset_metadata.json` - Image metadata
- `data/metadata/label_mappings.json` - Style/artist label mappings

### Step 2: Prepare Data Splits

Split the dataset into train/val/test sets (80/10/10):

```bash
python scripts/prepare_data.py
```

This will:
- Perform stratified split by style
- Organize images into `data/processed/{train,val,test}/`
- Create CSV files with labels
- Generate dataset summary

**Expected output:**
- `data/processed/train/` - Training images
- `data/processed/val/` - Validation images
- `data/processed/test/` - Test images
- `data/processed/{train,val,test}_labels.csv` - Label files

### Step 3: Visualize Dataset

Explore the dataset distribution and sample images:

```bash
python scripts/visualize_dataset.py
```

This will create:
- Style and artist distribution plots
- Sample images from each style
- Artist-style heatmap
- Dataset statistics report

**Expected output:**
- `data/figures/style_distribution.png`
- `data/figures/artist_distribution_top30.png`
- `data/figures/sample_images_by_style.png`
- `data/figures/artist_style_heatmap.png`
- `data/figures/dataset_statistics.txt`

### Step 4: Train Models

Train multi-task models (both ResNet and ViT):

```bash
# Train both models (ResNet + ViT multi-task)
python train_models.py --batch_size 64 --num_epochs 20 --num_workers 8

# Or train specific architecture
python train_models.py --models resnet  # ResNet only
python train_models.py --models vit     # ViT only
```

**Training options:**
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Number of training epochs (default: 30, recommend 20 for speed)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_workers`: Data loading workers (default: 8)
- `--models`: Which models to train (all/resnet/vit)
- `--no_amp`: Disable mixed precision training

**Expected output:**
- `checkpoints/{model_name}/best_model.pth` - Best model checkpoint
- `checkpoints/{model_name}/latest_model.pth` - Latest checkpoint
- `logs/{model_name}/training_history.json` - Training metrics
- `logs/{model_name}/training_curves.png` - Training/validation curves

**Models trained (2 multi-task models):**
1. `resnet_multitask` - ResNet50 for both style and artist
2. `vit_multitask` - ViT for both style and artist

Note: Multi-task models are more efficient than training separate models for each task.

### Step 5: Evaluate Models

Evaluate all trained models and generate comprehensive reports:

```bash
# Evaluate on test set
python evaluate_models.py --split test --batch_size 64 --num_workers 8

# Evaluate on validation set
python evaluate_models.py --split val
```

This will generate:
- Confusion matrices for each model
- Embedding visualizations (t-SNE)
- Detailed metrics (accuracy, F1 scores)
- Model comparison report

**Expected output:**
- `reports/{model_name}/test_metrics.json` - Detailed metrics
- `reports/confusion_matrices/{model_name}/test_*_confusion_matrix.png` - Confusion matrices
- `reports/embeddings/{model_name}/test_embeddings_*.png` - t-SNE visualizations
- `reports/model_comparison.json` - Cross-model comparison
- `reports/comparison_summary.txt` - Summary report

## Directory Structure

```
dlcv/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── PLAN.md                       # Project planning document
│
├── scripts/                      # Data preparation scripts
│   ├── download_dataset.py       # Download WikiArt from Hugging Face
│   ├── prepare_data.py           # Split and organize data
│   └── visualize_dataset.py      # Dataset visualizations
│
├── models/                       # Model architectures
│   ├── __init__.py
│   ├── resnet_models.py          # ResNet variants
│   └── vit_models.py             # ViT variants
│
├── utils/                        # Utilities
│   ├── __init__.py
│   ├── dataset.py                # Dataset and DataLoader
│   └── metrics.py                # Evaluation metrics
│
├── train_models.py               # Main training script
├── evaluate_models.py            # Evaluation script
│
├── data/                         # Data directory
│   ├── raw/                      # Downloaded images
│   ├── processed/                # Train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── metadata/                 # Dataset metadata
│   └── figures/                  # Dataset visualizations
│
├── checkpoints/                  # Model checkpoints
│   ├── resnet_style/
│   ├── resnet_artist/
│   ├── resnet_multitask/
│   ├── vit_style/
│   ├── vit_artist/
│   └── vit_multitask/
│
├── logs/                         # Training logs
│   └── {model_name}/
│       ├── training_history.json
│       └── training_curves.png
│
└── reports/                      # Evaluation reports
    ├── {model_name}/
    │   └── test_metrics.json
    ├── confusion_matrices/
    │   └── {model_name}/
    ├── embeddings/
    │   └── {model_name}/
    ├── model_comparison.json
    └── comparison_summary.txt
```

## Google Cloud Platform (GCP) Setup

### 1. Create GCP Instance

Recommended configuration:
- **Machine type**: n1-highmem-8 (8 vCPUs, 52 GB memory)
- **GPU**: 1x NVIDIA T4 or V100
- **Boot disk**: 100 GB SSD
- **OS**: Ubuntu 20.04 LTS with CUDA pre-installed

### 2. Upload Code to GCP

```bash
# From local machine
gcloud compute scp --recurse /path/to/dlcv [INSTANCE_NAME]:~/ --zone=[ZONE]

# Or use Cloud Storage
gsutil -m cp -r /path/to/dlcv gs://[BUCKET_NAME]/
```

### 3. SSH into Instance

```bash
gcloud compute ssh [INSTANCE_NAME] --zone=[ZONE]
```

### 4. Install Dependencies

```bash
cd dlcv
pip install -r requirements.txt
```

### 5. Run Pipeline

```bash
# Download and prepare data
python scripts/download_dataset.py
python scripts/prepare_data.py
python scripts/visualize_dataset.py

# Train all models
python train_models.py --batch_size 64 --num_epochs 30

# Evaluate
python evaluate_models.py --split test
```

### 6. Download Results

```bash
# From local machine
gcloud compute scp --recurse [INSTANCE_NAME]:~/dlcv/checkpoints ./checkpoints --zone=[ZONE]
gcloud compute scp --recurse [INSTANCE_NAME]:~/dlcv/reports ./reports --zone=[ZONE]
gcloud compute scp --recurse [INSTANCE_NAME]:~/dlcv/logs ./logs --zone=[ZONE]
```

## Quick Start Summary

```bash
# Complete pipeline in order:
python scripts/download_dataset.py                              # Step 1: Download data
python scripts/prepare_data.py                                  # Step 2: Prepare splits
python scripts/visualize_dataset.py                             # Step 3: Visualize data
python train_models.py --batch_size 64 --num_epochs 20          # Step 4: Train 2 models
python evaluate_models.py --split test                          # Step 5: Evaluate models
```

**Total time:**
- Single GPU (L4/T4/V100): ~4-8 hours
- Dual GPU (2x L4/T4/V100): ~3-5 hours (use `train_parallel.py`)

## Parallel Training (Optional - For 2+ GPUs)

If you have multiple GPUs, train both models simultaneously to cut time in half:

```bash
# Automatically detects GPUs and trains in parallel
python train_parallel.py --batch_size 64 --num_epochs 20

# ResNet trains on GPU 0, ViT on GPU 1 simultaneously
# Time: ~1-2.5 hours for training (half of sequential)
```

**GPU Memory Requirements:**
- ResNet Multi-task: ~10 GB VRAM
- ViT Multi-task: ~12 GB VRAM
- **Cannot train both on single L4 (24GB)** - too tight, will crash
- **Sequential training recommended for single GPU**

## Key Features

### Data Pipeline
- Automatic download from Hugging Face
- Smart filtering by style/artist frequency
- Stratified train/val/test split
- Comprehensive data augmentation

### Model Training
- Transfer learning from ImageNet pretrained models
- Mixed precision training (AMP) for faster training
- Learning rate scheduling
- Best model checkpointing
- Training history tracking

### Evaluation
- Multi-metric evaluation (accuracy, F1 scores)
- Confusion matrices
- t-SNE embedding visualizations
- Cross-model comparison

### Visualizations
- Dataset distribution analysis
- Training/validation curves
- Confusion matrices per model
- Embedding space visualizations
- Artist-style relationship heatmaps

## Expected Results

After training, you should see:
- **Style classification**: 60-80% accuracy (depending on number of styles)
- **Artist classification**: 40-70% accuracy (depending on number of artists)
- **Multi-task models**: Slight performance trade-off vs single-task, but more efficient

ViT models typically perform slightly better than ResNet on art classification due to better handling of artistic style features.

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Reduce number of workers: `--num_workers 4`

### Slow Training
- Ensure GPU is being used (check with `nvidia-smi`)
- Enable mixed precision (remove `--no_amp`)
- Increase batch size if memory allows

### Dataset Download Issues
- Check Hugging Face availability
- Ensure sufficient disk space
- Try alternative dataset names in `download_dataset.py`

## License

This project is for educational purposes.

## Acknowledgments

- WikiArt dataset from Hugging Face
- PyTorch and torchvision for deep learning framework
- Pre-trained models from ImageNet
