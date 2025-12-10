

import os
import torch
import numpy as np
from tqdm import tqdm

from config import DEVICE
from data_loader import get_dataloaders
from model import setup_model
from utils.metrics import plot_confusion_matrix, calculate_metrics, save_metrics_json


def evaluate_model(model, dataloader, device):
    # Evaluate model and return predictions and probabilities
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Evaluating model"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def load_model(arch, checkpoint_path, num_classes, device):
    # Load a trained model from checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping {arch}.")
        return None
    
    try:
        model = setup_model(num_classes, device, arch=arch, freeze_base=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded {arch}")
        return model
    except Exception as e:
        print(f"Error loading {arch}: {e}")
        return None


def ensemble_predictions_weighted(predictions_list, probabilities_list, weights):
    # Combine predictions using weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    ensemble_probs = np.zeros_like(probabilities_list[0])
    for probs, weight in zip(probabilities_list, weights):
        ensemble_probs += probs * weight
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, ensemble_probs


def main():
    # Generate confusion matrix for ensemble
    
    print("\nENSEMBLE CONFUSION MATRIX GENERATION")
    
    # Model configurations
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
    
    # Load data
    print("\nLoading test data")
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    # Get true labels
    print("Extracting true labels")
    true_labels = []
    for _, labels in tqdm(test_loader, desc="Loading labels"):
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    # Load all models
    print("\nLoading models")
    models = []
    loaded_configs = []
    
    for config in models_config:
        model = load_model(
            config['arch'],
            config['checkpoint'],
            num_classes,
            DEVICE
        )
        if model is not None:
            models.append(model)
            loaded_configs.append(config)
    
    if len(models) == 0:
        print("Error: No models loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models")
    
    # Evaluate each model
    print("\nEvaluating individual models")
    all_predictions = []
    all_probabilities = []
    individual_accuracies = []
    
    for model, config in zip(models, loaded_configs):
        print(f"  Evaluating {config['display_name']}")
        preds, probs = evaluate_model(model, test_loader, DEVICE)
        all_predictions.append(preds)
        all_probabilities.append(probs)
        
        # Calculate accuracy for weighting
        accuracy = np.mean(preds == true_labels)
        individual_accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.4f}")
    
    # Calculate weights based on individual accuracies
    weights = individual_accuracies / np.sum(individual_accuracies)
    print(f"\nEnsemble weights:")
    for config, weight in zip(loaded_configs, weights):
        print(f"  {config['display_name']:20s}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Generate ensemble predictions (weighted average - best method)
    print("\nGenerating ensemble predictions (weighted average)...")
    ensemble_preds, ensemble_probs = ensemble_predictions_weighted(
        all_predictions, all_probabilities, weights
    )
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, ensemble_preds, class_names=class_names)
    
    print("\nENSEMBLE RESULTS")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
    
    # Create output directory
    output_dir = 'reports/confusion_matrices/ensemble'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate normalized confusion matrix
    cm_path_normalized = os.path.join(output_dir, 'test_confusion_matrix_normalized.png')
    print(f"Generating normalized confusion matrix...")
    plot_confusion_matrix(
        true_labels, ensemble_preds, class_names,
        cm_path_normalized,
        title='Ensemble (Weighted Average) - Test Set Confusion Matrix (Normalized)',
        normalize=True
    )
    print(f"Saved to: {cm_path_normalized}")
    
    # Generate raw confusion matrix
    cm_path_raw = os.path.join(output_dir, 'test_confusion_matrix_raw.png')
    print(f"Generating raw confusion matrix...")
    plot_confusion_matrix(
        true_labels, ensemble_preds, class_names,
        cm_path_raw,
        title='Ensemble (Weighted Average) - Test Set Confusion Matrix (Raw Counts)',
        normalize=False
    )
    print(f"Saved to: {cm_path_raw}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    ensemble_metrics = {
        'method': 'weighted_average',
        'weights': dict(zip([c['arch'] for c in loaded_configs], weights.tolist())),
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'classification_report': metrics.get('classification_report', {})
    }
    save_metrics_json(ensemble_metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    # Clean up
    for model in models:
        del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nEnsemble confusion matrix generated successfully!")
    print(f"  Location: {output_dir}/")


if __name__ == '__main__':
    main()

