# Generate confusion matrices for all trained models

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
    # Evaluate model and return predictions and true labels
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
    # Generate confusion matrix for a specific model
    print(f"\nEvaluating {arch.upper()}\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {arch}.")
        return
    
    # Load data
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    # Setup model
    model = setup_model(num_classes, DEVICE, arch=arch, freeze_base=False)
    
    # Load checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Evaluate model
    predictions, true_labels = evaluate_model(model, test_loader, DEVICE)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predictions, class_names=class_names)
    
    # Print summary
    print(f"\nResults for {arch.upper()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
    
    # Create output directory
    model_output_dir = os.path.join(output_dir, arch)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Generate confusion matrix (normalized)
    cm_path_normalized = os.path.join(model_output_dir, 'test_confusion_matrix_normalized.png')
    plot_confusion_matrix(
        true_labels, predictions, class_names,
        cm_path_normalized,
        title=f'{arch.upper()} - Test Set Confusion Matrix (Normalized)',
        normalize=True
    )
    print(f"Saved to: {cm_path_normalized}")
    
    # Generate confusion matrix (raw counts)
    cm_path_raw = os.path.join(model_output_dir, 'test_confusion_matrix_raw.png')
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
    
    print(f"\nCompleted evaluation for {arch.upper()}\n")


def main():
    # Main function to generate confusion matrices for all models
    
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
    
    print("\nCONFUSION MATRIX GENERATION")
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
    print("\nSUMMARY")
    for model_config in models_config:
        arch = model_config['arch']
        status = results.get(arch, 'not processed')
        print(f"{model_config['display_name']:20s}: {status}")
    print("\nAll confusion matrices saved to: reports/confusion_matrices/\n")


if __name__ == '__main__':
    main()

