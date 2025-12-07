# Implementation Plan: Painting Style Matcher

## Overview

This project implements a deep learning system for classifying paintings by time period and style. The implementation follows a "get baseline fast, then iterate" approach.

## Project Structure

```
dlcv/
├── config.py              # Configuration parameters
├── data_loader.py          # CSV-based Dataset and DataLoaders
├── model.py               # ResNet18 models (single-task & multi-task)
├── train.py               # Training script with logging & metrics
├── visualize.py           # Grad-CAM and embedding visualization utilities
├── run_visualizations.py   # Script to run interpretability analysis
├── create_csv_from_folders.py  # Helper to convert folder structure to CSV
└── IMPLEMENTATION_PLAN.md # This file
```

## Implementation Phases

### Phase 1: Baseline (✅ Complete)

**Goal**: Get a working baseline quickly for sanity checking.

**Components**:
1. **CSV-based Dataset** (`data_loader.py`)
   - Reads images and labels from CSV files
   - Columns: `image_path`, `time_period`, (optional: `style`)
   - Supports both CSV and ImageFolder (backward compatibility)

2. **Data Augmentations**
   - RandomResizedCrop(224)
   - RandomHorizontalFlip
   - ColorJitter (brightness, contrast, saturation, hue)
   - Normalize (ImageNet stats)

3. **Baseline Model** (`model.py`)
   - Pretrained ResNet18
   - Single head for time-period classification
   - Frozen base (only train final layer)

4. **Training** (`train.py`)
   - 5-10 epochs for sanity check
   - Logging: TensorBoard or CSV
   - Metrics tracked:
     - Total accuracy
     - Per-class accuracy
     - Confusion matrices
     - Classification reports

**Usage**:
```bash
# Prepare CSV files (if using folder structure)
python create_csv_from_folders.py --data_dir ./data --split train
python create_csv_from_folders.py --data_dir ./data --split val

# Train baseline
python train.py
```

### Phase 2: Multi-Task Model (✅ Complete)

**Goal**: Upgrade to multi-task learning (time period + style).

**Components**:
1. **Multi-Task Architecture** (`model.py`)
   - Shared MLP: `ResNet features → 512-dim → Dropout`
   - Two heads:
     - `fc_time`: Time period classification
     - `fc_style`: Style classification

2. **Weighted Loss**
   - `L = L_time + λ * L_style`
   - Tune `λ` (STYLE_LOSS_WEIGHT in config.py)

**Usage**:
```python
# In config.py, set:
USE_MULTI_TASK = True
STYLE_LOSS_WEIGHT = 1.0  # Tune this

# CSV must include 'style' column
python train.py
```

### Phase 3: Interpretability (✅ Complete)

**Goal**: Understand model decisions and visualize embeddings.

**Components**:
1. **Grad-CAM** (`visualize.py`)
   - Visualizes which parts of image the model focuses on
   - Works for both correct predictions and misclassifications

2. **Embedding Visualization**
   - Extract features from penultimate layer
   - t-SNE and UMAP dimensionality reduction
   - Plot with class labels

**Usage**:
```bash
# After training, run visualizations
python run_visualizations.py \
    --checkpoint best_base_model.pth \
    --num_classes 10 \
    --mode both
```

## Configuration

Key parameters in `config.py`:

```python
# Data
DATA_DIR = './data'
CSV_TRAIN = './data/train.csv'
CSV_VAL = './data/val.csv'

# Training
BATCH_SIZE = 32
NUM_EPOCHS_BASE = 10  # Baseline epochs
BASE_LR = 1e-3

# Multi-task
USE_MULTI_TASK = False
STYLE_LOSS_WEIGHT = 1.0

# Logging
USE_TENSORBOARD = True
LOG_DIR = './logs'
RESULTS_DIR = './results'
```

## Data Format

### CSV Format

**Required columns**:
- `image_path`: Relative path from DATA_DIR (e.g., `train/renaissance/painting1.jpg`)
- `time_period`: Integer class label (e.g., 0, 1, 2, ...)

**Optional columns** (for multi-task):
- `style`: Integer style class label

**Example**:
```csv
image_path,time_period,style
train/renaissance/painting1.jpg,0,5
train/baroque/painting2.jpg,1,3
val/impressionism/painting3.jpg,2,7
```

### Folder Structure (Alternative)

If using ImageFolder format:
```
data/
├── train/
│   ├── 0/  # Time period 0
│   ├── 1/  # Time period 1
│   └── ...
└── val/
    ├── 0/
    ├── 1/
    └── ...
```

Convert to CSV:
```bash
python create_csv_from_folders.py --data_dir ./data --split train
python create_csv_from_folders.py --data_dir ./data --split val
```

## Training Workflow

### 1. Baseline Training

```bash
# Single-task, frozen base
python train.py
```

**Outputs**:
- `best_base_model.pth`: Best model checkpoint
- `training_history.json`: Training metrics
- `results/confusion_matrix_best.png`: Confusion matrix
- `results/classification_report_best.json`: Per-class metrics
- TensorBoard logs in `logs/` (if enabled)

### 2. Fine-Tuning (Optional)

After baseline, unfreeze base and fine-tune:
```python
# In train.py, add fine-tuning phase
model = setup_model(NUM_CLASSES, DEVICE, freeze_base=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH_BASE))
optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
# Continue training...
```

### 3. Multi-Task Training

```python
# In config.py
USE_MULTI_TASK = True
STYLE_LOSS_WEIGHT = 1.0  # Tune this

# Ensure CSV has 'style' column
python train.py
```

## Visualization & Analysis

### Grad-CAM

Visualize model attention on specific images:
```python
from visualize import visualize_gradcam
from model import setup_model

model = setup_model(num_classes=10, device='cpu', freeze_base=False)
model.load_state_dict(torch.load('best_base_model.pth'))

visualize_gradcam(
    model, 
    'path/to/image.jpg',
    model.layer4,  # Target layer
    class_names=['Renaissance', 'Baroque', ...],
    save_path='gradcam.png'
)
```

### Embedding Visualization

```python
from visualize import extract_embeddings, visualize_embeddings

# Extract embeddings
embeddings, labels, _ = extract_embeddings(model, val_dataloader, device='cpu')

# Visualize with t-SNE
visualize_embeddings(embeddings, labels, class_names, method='tsne', 
                    save_path='embeddings_tsne.png')

# Visualize with UMAP
visualize_embeddings(embeddings, labels, class_names, method='umap',
                    save_path='embeddings_umap.png')
```

## Next Steps / Iterations

1. **Hyperparameter Tuning**
   - Learning rate schedules
   - Batch size optimization
   - Data augmentation strategies
   - Loss weight (λ) for multi-task

2. **Model Architecture**
   - Try different backbones (ResNet50, EfficientNet, Vision Transformer)
   - Experiment with attention mechanisms
   - Ensemble methods

3. **Data**
   - Collect more training data
   - Balance classes
   - Data cleaning and quality checks

4. **Advanced Interpretability**
   - Error analysis notebooks
   - Per-class Grad-CAM analysis
   - Feature importance analysis

5. **Stretch Goals**
   - LLM integration for captions
   - Object/person detection
   - Style attribute prediction

## Dependencies

```bash
pip install torch torchvision
pip install pandas pillow
pip install matplotlib seaborn
pip install scikit-learn
pip install tensorboard  # For TensorBoard logging
pip install umap-learn  # For UMAP visualization (optional)
pip install opencv-python  # For Grad-CAM
```

## Notes

- The code supports both MPS (Apple Silicon) and CPU/GPU
- Default batch size is 32, adjust based on available memory
- For sanity checks, use a small subset of data first
- Monitor per-class accuracy to identify problematic classes
- Use confusion matrices to understand common misclassifications

