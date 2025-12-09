"""
Generate confusion matrices for all trained models (ResNet, EfficientNet, VGG16).
Evaluates models on test set and creates visualization outputs.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

from config import DEVICE, BATCH_SIZE
from data_loader import get_dataloaders
from model import setup_model
from utils.metrics import plot_confusion_matrix, calculate_metrics, save_metrics_json


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset and return predictions and true labels.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the dataset
        device: torch device
        
    Returns:
        all_preds: List of predicted labels
        all_labels: List of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def generate_confusion_matrix_for_model(arch, checkpoint_path, output_dir='reports/confusion_matrices'):
    """
    Generate confusion matrix for a specific model.
    
    Args:
        arch: Architecture name ('resnet50', 'efficientnet_b0', 'vgg16_bn')
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {arch.upper()}")
    print(f"{'='*60}\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {arch}.")
        return
    
    # Load data
    print("Loading data...")
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    # Setup model
    print(f"Loading model: {arch}...")
    model = setup_model(num_classes, DEVICE, arch=arch, freeze_base=False)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Evaluate model
    print("\nEvaluating on test set...")
    predictions, true_labels = evaluate_model(model, test_loader, DEVICE)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(true_labels, predictions, class_names=class_names)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {arch.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"{'='*60}\n")
    
    # Create output directory
    model_output_dir = os.path.join(output_dir, arch)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Generate confusion matrix (normalized)
    cm_path_normalized = os.path.join(model_output_dir, 'test_confusion_matrix_normalized.png')
    print(f"Generating normalized confusion matrix...")
    plot_confusion_matrix(
        true_labels, predictions, class_names,
        cm_path_normalized,
        title=f'{arch.upper()} - Test Set Confusion Matrix (Normalized)',
        normalize=True
    )
    print(f"Saved to: {cm_path_normalized}")
    
    # Generate confusion matrix (raw counts)
    cm_path_raw = os.path.join(model_output_dir, 'test_confusion_matrix_raw.png')
    print(f"Generating raw confusion matrix...")
    plot_confusion_matrix(
        true_labels, predictions, class_names,
        cm_path_raw,
        title=f'{arch.upper()} - Test Set Confusion Matrix (Raw Counts)',
        normalize=False
    )
    print(f"Saved to: {cm_path_raw}")
    
    # Save metrics
    metrics_path = os.path.join(model_output_dir, 'test_metrics.json')
    save_metrics_json(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n✓ Completed evaluation for {arch.upper()}\n")


def main():
    """Main function to generate confusion matrices for all models."""
    
    # Define models and their checkpoint paths
    models_config = [
        {
            'arch': 'resnet50',
            'checkpoint': 'final_best_tuned_model.pth',
            'display_name': 'ResNet50'
        },
        {
            'arch': 'efficientnet_b0',
            'checkpoint': 'best_efficientnet_b0_tuned.pth',
            'display_name': 'EfficientNet-B0'
        },
        {
            'arch': 'vgg16_bn',
            'checkpoint': 'best_vgg16_bn_tuned.pth',
            'display_name': 'VGG16-BN'
        },
        {
            'arch': 'vit_b_16',
            'checkpoint': 'best_vit_b_16_tuned.pth',
            'display_name': 'ViT-B/16'
        }
    ]
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX GENERATION")
    print("="*60)
    print("\nThis script will evaluate all trained models and generate confusion matrices.")
    print("Make sure all models have finished training before running this script.\n")
    
    # Process each model
    results = {}
    for model_config in models_config:
        arch = model_config['arch']
        checkpoint = model_config['checkpoint']
        
        try:
            generate_confusion_matrix_for_model(arch, checkpoint)
            results[arch] = 'success'
        except Exception as e:
            print(f"Error processing {arch}: {e}")
            results[arch] = f'error: {str(e)}'
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_config in models_config:
        arch = model_config['arch']
        status = results.get(arch, 'not processed')
        print(f"{model_config['display_name']:20s}: {status}")
    print("="*60)
    print("\nAll confusion matrices saved to: reports/confusion_matrices/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

