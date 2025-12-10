# Test individual models on test set

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import DEVICE
from data_loader import get_dataloaders
from model import setup_model
from utils.metrics import calculate_metrics


def evaluate_model_on_test(model, test_loader, device):
    # Evaluate model on test set and return predictions and true labels
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def test_model(arch, checkpoint_path, num_classes, class_names, test_loader, device):
    # Test a single model on test set and return results
    print(f"\nTesting {arch.upper()}\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        # Load model
        print(f"Loading model: {arch}...")
        model = setup_model(num_classes, device, arch=arch, freeze_base=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Model loaded successfully")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        predictions, true_labels = evaluate_model_on_test(model, test_loader, device)
        
        # Calculate metrics
        print(f"\nCalculating metrics...")
        metrics = calculate_metrics(true_labels, predictions, class_names=class_names)
        
        # Print results
        print(f"\nRESULTS for {arch.upper()}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'arch': arch,
            'checkpoint': checkpoint_path,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted']
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    # Main function to test individual models
    
    print("\nINDIVIDUAL MODEL TESTING ON TEST SET")
    print("\nThis script will evaluate ResNet, EfficientNet, and VGG16 on the test set.")
    print("Make sure the models have finished training.\n")
    
    # Load test data
    print("Loading test data...")
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    print(f"Test set size: {len(image_datasets['test'])} samples")
    print(f"Number of classes: {num_classes}\n")
    
    # Define models to test
    models_to_test = [
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
        }
    ]
    
    # Test each model
    results = {}
    for model_config in models_to_test:
        result = test_model(
            model_config['arch'],
            model_config['checkpoint'],
            num_classes,
            class_names,
            test_loader,
            DEVICE
        )
        if result:
            results[model_config['arch']] = {
                **result,
                'display_name': model_config['display_name']
            }
    
    # Print summary
    if results:
        print("\nSUMMARY")
        print(f"\n{'Model':<20s} {'Accuracy':<12s} {'F1 (Macro)':<12s} {'F1 (Weighted)':<15s}")
        print("-" * 60)
        
        for arch, result in results.items():
            print(f"{result['display_name']:<20s} "
                  f"{result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)  "
                  f"{result['f1_macro']:.4f}        "
                  f"{result['f1_weighted']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n✓ BEST MODEL: {best_model[1]['display_name']}")
        print(f"  Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
        print(f"  F1 (Macro): {best_model[1]['f1_macro']:.4f}\n")
    else:
        print("\n❌ No models were successfully tested. Check that checkpoints exist.\n")


if __name__ == '__main__':
    main()

