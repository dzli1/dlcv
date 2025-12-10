# Evaluate ensemble of all trained models

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from collections import Counter

from config import DEVICE, BATCH_SIZE
from data_loader import get_dataloaders
from model import setup_model
from utils.metrics import calculate_metrics, save_metrics_json


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
        print(f"✓ Loaded {arch} from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading {arch}: {e}")
        return None


def ensemble_predictions(predictions_list, probabilities_list, method='weighted_average', weights=None):
    # Combine predictions from multiple models using specified method
    if weights is None:
        weights = [1.0] * len(probabilities_list)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    if method == 'weighted_average' or method == 'average':
        # Weighted average of probabilities
        ensemble_probs = np.zeros_like(probabilities_list[0])
        for probs, weight in zip(probabilities_list, weights):
            ensemble_probs += probs * weight
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
    elif method == 'majority_vote':
        # Majority voting
        ensemble_preds = []
        for i in range(len(predictions_list[0])):
            votes = [preds[i] for preds in predictions_list]
            # Count votes (weighted)
            vote_counts = Counter()
            for vote, weight in zip(votes, weights):
                vote_counts[vote] += weight
            ensemble_preds.append(vote_counts.most_common(1)[0][0])
        ensemble_preds = np.array(ensemble_preds)
        ensemble_probs = None  # Not applicable for voting
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_preds, ensemble_probs


def evaluate_ensemble(models_config, test_loader, num_classes, class_names, device):
    # Evaluate ensemble of models on test set and return results
    print("\nENSEMBLE EVALUATION")
    
    # Load all models
    print("\nLoading models...")
    models = []
    loaded_configs = []
    
    for config in models_config:
        model = load_model(
            config['arch'],
            config['checkpoint'],
            num_classes,
            device
        )
        if model is not None:
            models.append(model)
            loaded_configs.append(config)
    
    if len(models) == 0:
        print("Error: No models loaded successfully!")
        return None
    
    print(f"\nSuccessfully loaded {len(models)} models:")
    for config in loaded_configs:
        print(f"  - {config['display_name']}")
    
    # Get true labels from test set
    print("\nExtracting true labels from test set...")
    true_labels = []
    for _, labels in tqdm(test_loader, desc="Loading labels"):
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    # Evaluate each model individually
    print("\nINDIVIDUAL MODEL EVALUATION")
    
    all_predictions = []
    all_probabilities = []
    individual_results = {}
    
    for i, (model, config) in enumerate(zip(models, loaded_configs)):
        print(f"\nEvaluating {config['display_name']}...")
        preds, probs = evaluate_model(model, test_loader, device)
        all_predictions.append(preds)
        all_probabilities.append(probs)
        
        # Calculate individual metrics
        metrics = calculate_metrics(true_labels, preds, class_names=class_names)
        individual_results[config['arch']] = {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'display_name': config['display_name']
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Ensemble predictions
    print("\nENSEMBLE PREDICTIONS")
    
    ensemble_results = {}
    
    # Method 1: Simple Average (equal weights)
    print("\n1. Ensemble Method: Average (equal weights)")
    ensemble_preds_avg, ensemble_probs_avg = ensemble_predictions(
        all_predictions, all_probabilities, method='average'
    )
    metrics_avg = calculate_metrics(true_labels, ensemble_preds_avg, class_names=class_names)
    ensemble_results['average'] = {
        'method': 'average',
        'accuracy': metrics_avg['accuracy'],
        'f1_macro': metrics_avg['f1_macro'],
        'f1_weighted': metrics_avg['f1_weighted']
    }
    print(f"  Accuracy: {metrics_avg['accuracy']:.4f}")
    print(f"  F1 (Macro): {metrics_avg['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {metrics_avg['f1_weighted']:.4f}")
    
    # Method 2: Weighted Average (by individual accuracy)
    print("\n2. Ensemble Method: Weighted Average (by accuracy)")
    accuracies = [individual_results[config['arch']]['accuracy'] for config in loaded_configs]
    weights = accuracies / np.sum(accuracies)  # Normalize
    print(f"  Weights: {dict(zip([c['display_name'] for c in loaded_configs], weights))}")
    
    ensemble_preds_weighted, ensemble_probs_weighted = ensemble_predictions(
        all_predictions, all_probabilities, method='weighted_average', weights=weights
    )
    metrics_weighted = calculate_metrics(true_labels, ensemble_preds_weighted, class_names=class_names)
    ensemble_results['weighted_average'] = {
        'method': 'weighted_average',
        'weights': dict(zip([c['arch'] for c in loaded_configs], weights.tolist())),
        'accuracy': metrics_weighted['accuracy'],
        'f1_macro': metrics_weighted['f1_macro'],
        'f1_weighted': metrics_weighted['f1_weighted']
    }
    print(f"  Accuracy: {metrics_weighted['accuracy']:.4f}")
    print(f"  F1 (Macro): {metrics_weighted['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {metrics_weighted['f1_weighted']:.4f}")
    
    # Method 3: Majority Vote
    print("\n3. Ensemble Method: Majority Vote")
    ensemble_preds_vote, _ = ensemble_predictions(
        all_predictions, all_probabilities, method='majority_vote', weights=weights
    )
    metrics_vote = calculate_metrics(true_labels, ensemble_preds_vote, class_names=class_names)
    ensemble_results['majority_vote'] = {
        'method': 'majority_vote',
        'accuracy': metrics_vote['accuracy'],
        'f1_macro': metrics_vote['f1_macro'],
        'f1_weighted': metrics_vote['f1_weighted']
    }
    print(f"  Accuracy: {metrics_vote['accuracy']:.4f}")
    print(f"  F1 (Macro): {metrics_vote['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {metrics_vote['f1_weighted']:.4f}")
    
    # Summary
    print("\nSUMMARY")
    
    print("\nIndividual Model Results:")
    for arch, results in individual_results.items():
        print(f"  {results['display_name']:20s}: Acc={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
    
    print("\nEnsemble Results:")
    print(f"  Average (Equal Weights)    : Acc={ensemble_results['average']['accuracy']:.4f}, F1={ensemble_results['average']['f1_macro']:.4f}")
    print(f"  Weighted Average           : Acc={ensemble_results['weighted_average']['accuracy']:.4f}, F1={ensemble_results['weighted_average']['f1_macro']:.4f}")
    print(f"  Majority Vote              : Acc={ensemble_results['majority_vote']['accuracy']:.4f}, F1={ensemble_results['majority_vote']['f1_macro']:.4f}")
    
    # Find best ensemble method
    best_method = max(ensemble_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best Ensemble Method: {best_method[0]} (Accuracy: {best_method[1]['accuracy']:.4f})")
    
    # Prepare results dictionary
    results = {
        'individual_models': individual_results,
        'ensemble_methods': ensemble_results,
        'best_method': best_method[0],
        'best_accuracy': best_method[1]['accuracy']
    }
    
    # Clean up
    for model in models:
        del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def main():
    # Main function to evaluate ensemble
    
    # Define models and their checkpoint paths
    # All four models: ResNet, EfficientNet, VGG16, and ViT
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
    
    print("\nENSEMBLE MODEL EVALUATION")
    print("\nThis script will:")
    print("  1. Load all trained models")
    print("  2. Evaluate each model individually on test set")
    print("  3. Combine predictions using multiple ensemble methods")
    print("  4. Report accuracy and F1 scores for all methods")
    print("\nMake sure all models have finished training before running this script.\n")
    
    # Load data
    print("Loading test data...")
    dataloaders, image_datasets, num_classes, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    print(f"Test set size: {len(image_datasets['test'])} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}\n")
    
    # Evaluate ensemble
    results = evaluate_ensemble(
        models_config,
        test_loader,
        num_classes,
        class_names,
        DEVICE
    )
    
    if results is None:
        print("\nError: Could not evaluate ensemble. Check that model checkpoints exist.")
        return
    
    # Save results
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'ensemble_results.json')
    save_metrics_json(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Create summary text file
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ENSEMBLE EVALUATION SUMMARY\n\n")
        
        f.write("INDIVIDUAL MODEL RESULTS:\n")
        f.write("-" * 60 + "\n")
        for arch, model_results in results['individual_models'].items():
            f.write(f"{model_results['display_name']:20s}: ")
            f.write(f"Accuracy={model_results['accuracy']:.4f}, ")
            f.write(f"F1_Macro={model_results['f1_macro']:.4f}, ")
            f.write(f"F1_Weighted={model_results['f1_weighted']:.4f}\n")
        
        f.write("\nENSEMBLE RESULTS:\n")
        f.write("-" * 60 + "\n")
        for method_name, method_results in results['ensemble_methods'].items():
            f.write(f"{method_name:20s}: ")
            f.write(f"Accuracy={method_results['accuracy']:.4f}, ")
            f.write(f"F1_Macro={method_results['f1_macro']:.4f}, ")
            f.write(f"F1_Weighted={method_results['f1_weighted']:.4f}\n")
        
        f.write(f"\nBEST ENSEMBLE METHOD: {results['best_method']}\n")
        f.write(f"BEST ACCURACY: {results['best_accuracy']:.4f}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\nENSEMBLE EVALUATION COMPLETE!\n")


if __name__ == '__main__':
    main()

