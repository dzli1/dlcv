# Art Style Classification with Deep Learning Ensemble

This project implements a deep learning pipeline for classifying art styles using multiple state-of-the-art architectures (ResNet, EfficientNet, VGG16, and Vision Transformer) and combines them into an ensemble model for improved accuracy.

## Project Overview

The goal of this project is to classify artwork images into 20 different art style categories using transfer learning and ensemble methods. The project trains four different deep learning models independently and then combines their predictions to achieve better performance than any single model.

### Key Features

- **Multi-Architecture Training**: Trains ResNet50, EfficientNet-B0, VGG16-BN, and Vision Transformer (ViT-B/16)
- **Two-Phase Training**: Base training with frozen backbone, followed by fine-tuning
- **Ensemble Methods**: Combines predictions using average, weighted average, and majority voting
- **Comprehensive Evaluation**: Generates confusion matrices, metrics, and detailed reports
- **GPU Support**: Automatic CUDA detection and utilization

## Project Structure

### Core Configuration Files

#### `config.py`
Central configuration file containing:
- Device selection (CUDA/MPS/CPU) with automatic detection
- Data directory paths
- Training hyperparameters (batch size, learning rates, epochs)
- ImageNet normalization parameters

**Key Parameters:**
- `BATCH_SIZE = 32`
- `NUM_EPOCHS_BASE = 20` (frozen backbone training)
- `NUM_EPOCHS_TUNE = 10` (fine-tuning)
- `BASE_LR = 1e-3` (classification head learning rate)
- `FINE_TUNE_LR = 1e-5` (fine-tuning learning rate)

#### `data_loader.py`
Handles data loading and preprocessing:
- **`get_dataloaders()`**: Creates train/val/test splits with stratification
- **`get_data_transforms()`**: Defines data augmentation and normalization
- **`TransformDataset`**: Wrapper for applying different transforms to subsets
- Performs 70/15/15 stratified split across 20 art style classes
- Returns DataLoaders and dataset information

#### `model.py`
Model factory function:
- **`setup_model()`**: Creates and configures models for different architectures
- Supports: `resnet50`, `efficientnet_b0`, `vgg16_bn`, `vit_b_16`
- Handles freezing/unfreezing backbone layers
- Replaces classification heads with custom layers (512 hidden units)

### Training Scripts

#### `train_resnet.py`
Trains ResNet50 model:
- **Phase 1**: Trains only the classification head (frozen backbone)
- **Phase 2**: Fine-tunes entire model (unfrozen)
- Saves checkpoints: `best_resnet50_base.pth`, `final_best_tuned_model.pth`
- Saves history: `training_history_resnet.json`

#### `train_efficientnet.py`
Trains EfficientNet-B0 model:
- Same two-phase training approach
- Saves checkpoints: `best_efficientnet_b0.pth`, `best_efficientnet_b0_tuned.pth`
- Saves history: `training_history_efficientnet.json`

#### `train_vgg16.py`
Trains VGG16-BN model:
- Same two-phase training approach
- Saves checkpoints: `best_vgg16_bn.pth`, `best_vgg16_bn_tuned.pth`
- Saves history: `training_history_vgg16.json`

#### `train_vit.py`
Trains Vision Transformer (ViT-B/16) model:
- Same two-phase training approach
- Uses transformer architecture for global attention
- Saves checkpoints: `best_vit_b_16_base.pth`, `best_vit_b_16_tuned.pth`
- Saves history: `training_history_vit.json`

### Evaluation Scripts

#### `test_individual_models.py`
Tests individual models on the test set:
- Evaluates ResNet, EfficientNet, and VGG16 separately
- Reports accuracy, F1-macro, and F1-weighted scores
- Prints summary comparison of all models

**Usage:**
```bash
python test_individual_models.py
```

#### `generate_confusion_matrices.py`
Generates confusion matrices for all trained models:
- Evaluates each model on test set
- Creates normalized and raw confusion matrices
- Saves metrics JSON files
- Outputs saved to `reports/confusion_matrices/{model_name}/`

**Usage:**
```bash
python generate_confusion_matrices.py
```

**Output Files:**
- `test_confusion_matrix_normalized.png` - Normalized percentages
- `test_confusion_matrix_raw.png` - Raw prediction counts
- `test_metrics.json` - Detailed metrics

#### `evaluate_ensemble.py`
Evaluates ensemble of all models:
- **Step 1**: Evaluates each model individually on test set
- **Step 2**: Combines predictions using three methods:
  - **Average**: Equal weights for all models
  - **Weighted Average**: Weights based on individual model accuracy
  - **Majority Vote**: Voting-based combination
- Reports best ensemble method
- Saves results to `reports/ensemble_results.json` and `reports/ensemble_summary.txt`

**Usage:**
```bash
python evaluate_ensemble.py
```

### Utility Modules

#### `utils/metrics.py`
Evaluation metrics and visualization:
- **`calculate_metrics()`**: Computes accuracy, F1 scores (macro/weighted)
- **`plot_confusion_matrix()`**: Generates confusion matrix visualizations
- **`plot_training_curves()`**: Creates training/validation curves
- **`save_metrics_json()`**: Saves metrics to JSON format

#### `utils/dataset.py`
Additional dataset utilities (if needed for other pipelines)

### Model Architectures

#### `models/resnet_models.py`
ResNet-based model implementations (for multi-task variants)

#### `models/vit_models.py`
Vision Transformer model implementations (for multi-task variants)

## Dataset

The project uses an art style classification dataset with:
- **20 art style classes**: Abstract_Expressionism, Art_Nouveau_Modern, Baroque, Color_Field_Painting, Cubism, Early_Renaissance, Expressionism, High_Renaissance, Impressionism, Mannerism_Late_Renaissance, Minimalism, Naive_Art_Primitivism, Northern_Renaissance, Pop_Art, Post_Impressionism, Realism, Rococo, Romanticism, Symbolism, Ukiyo_e
- **Data location**: `./artset/` (organized by class folders)
- **Total images**: ~18,000 images
- **Split**: 70% train, 15% validation, 15% test (stratified)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure CUDA is available** (for GPU training):
   - The script automatically detects and uses CUDA if available
   - For CPU-only, it will fall back automatically

## Usage

### Training Models

Train each model individually (can be run in parallel in separate terminals):

```bash
# Train ResNet50
python train_resnet.py

# Train EfficientNet-B0
python train_efficientnet.py

# Train VGG16-BN
python train_vgg16.py

# Train ViT-B/16
python train_vit.py
```

Each training script:
1. Loads data with stratified train/val/test splits
2. **Phase 1**: Trains classification head with frozen backbone (20 epochs)
3. **Phase 2**: Fine-tunes entire model (10 epochs)
4. Saves best checkpoints and training history

### Evaluation

After training completes:

1. **Test individual models:**
```bash
python test_individual_models.py
```

2. **Generate confusion matrices:**
```bash
python generate_confusion_matrices.py
```

3. **Evaluate ensemble:**
```bash
python evaluate_ensemble.py
```

## Results

### Individual Model Performance (Test Set)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| **ViT-B/16** | **61.70%** | 0.6129 | 0.6129 |
| **ResNet50** | **59.44%** | 0.5928 | 0.5928 |
| **VGG16-BN** | **56.74%** | 0.5638 | 0.5638 |
| **EfficientNet-B0** | **55.19%** | 0.5482 | 0.5482 |

**Key Findings:**
- Vision Transformer (ViT) performs best as a single model
- All models achieve >55% accuracy on 20-class classification
- ResNet50 is the second-best performing model

### Ensemble Performance

| Ensemble Method | Accuracy | F1 (Macro) | Improvement Over Best Single Model |
|----------------|----------|------------|-----------------------------------|
| **Weighted Average** | **63.74%** | 0.6332 | **+2.04%** over ViT |
| **Average (Equal Weights)** | **63.52%** | 0.6311 | **+1.82%** over ViT |
| **Majority Vote** | **63.00%** | 0.6269 | **+1.30%** over ViT |

**Key Findings:**
- Ensemble improves accuracy by **2.04%** over the best single model
- Weighted average (by individual accuracy) performs best
- All ensemble methods outperform individual models
- The ensemble achieves **63.74% accuracy** on the test set

### Output Files

All results are saved in the `reports/` directory:

```
reports/
├── confusion_matrices/
│   ├── resnet50/
│   │   ├── test_confusion_matrix_normalized.png
│   │   ├── test_confusion_matrix_raw.png
│   │   └── test_metrics.json
│   ├── efficientnet_b0/
│   ├── vgg16_bn/
│   └── vit_b_16/
├── ensemble_results.json
└── ensemble_summary.txt
```

## Technical Details

### Training Strategy

**Two-Phase Training Approach:**

1. **Phase 1 - Base Training:**
   - Freezes pretrained backbone (ImageNet weights)
   - Trains only the new classification head
   - Learning rate: 1e-3
   - Epochs: 20
   - Purpose: Learn to map pretrained features to art style classes

2. **Phase 2 - Fine-Tuning:**
   - Unfreezes entire model
   - Fine-tunes all layers with smaller learning rate
   - Learning rate: 1e-5
   - Epochs: 10
   - Purpose: Adapt pretrained features to art classification task

### Model Architectures

- **ResNet50**: Deep residual network with 50 layers, hierarchical feature extraction
- **EfficientNet-B0**: Efficient architecture balancing depth, width, and resolution
- **VGG16-BN**: Deep convolutional network with batch normalization
- **ViT-B/16**: Vision Transformer using self-attention for global context

### Ensemble Methods

1. **Average (Equal Weights)**: Simple average of all model probabilities
2. **Weighted Average**: Weighted by individual model accuracy
3. **Majority Vote**: Voting-based combination with weighted votes

## Project Philosophy

This project demonstrates:
- **Transfer Learning**: Using pretrained ImageNet models for art classification
- **Ensemble Learning**: Combining multiple models for improved performance
- **Architecture Diversity**: Using both CNN (ResNet, EfficientNet, VGG) and Transformer (ViT) architectures
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and ensemble analysis

The ensemble approach leverages the complementary strengths of different architectures:
- CNNs excel at local feature extraction (textures, brushstrokes)
- Transformers excel at global context and long-range dependencies (composition)
- Combining them provides robust classification across diverse art styles

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch (with CUDA support recommended)
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Notes

- Models can be trained in parallel on separate GPUs/terminals
- Training time varies by model: ~1-3 hours per model on RTX 4090
- All models use the same data splits for fair comparison
- Checkpoints are saved after each phase for resumability
