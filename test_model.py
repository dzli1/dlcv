import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json

from config import DEVICE, CHECKPOINT_PATH_TUNE
from model import setup_model
from data_loader import get_dataloaders


def load_models(num_classes, device):
    """
    Load ensemble models if their checkpoints exist.
    Uses tuned checkpoints if present; falls back to base.
    """
    candidates = [
        ('resnet50', CHECKPOINT_PATH_TUNE, ['final_best_tuned_model.pth', CHECKPOINT_PATH_TUNE]),
        ('efficientnet_b0', 'best_efficientnet_b0_tuned.pth', ['best_efficientnet_b0_tuned.pth', 'best_efficientnet_b0.pth']),
        ('vgg16_bn', 'best_vgg16_bn_tuned.pth', ['best_vgg16_bn_tuned.pth', 'best_vgg16_bn.pth']),
    ]

    models = {}
    for arch, label, paths in candidates:
        ckpt = next((p for p in paths if os.path.exists(p)), None)
        if ckpt is None:
            print(f"Skipping {arch}: no checkpoint found ({paths})")
            continue
        print(f"Loading {arch} from {ckpt}...")
        model = setup_model(num_classes, device, arch=arch, freeze_base=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[arch] = model

    if not models:
        raise RuntimeError("No models loaded. Please train and ensure checkpoints exist.")

    return models


def evaluate_test_set():
    # Load data (includes test set)
    dataloaders, image_datasets, NUM_CLASSES, class_names = get_dataloaders()

    # Load ensemble models
    models = load_models(NUM_CLASSES, DEVICE)
    model_names = list(models.keys())
    print(f"\nEnsemble members: {model_names}")

    # Evaluate on test set
    test_loader = dataloaders['test']
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0

    print("\nEvaluating ensemble on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Sum logits from all models
            logits_sum = None
            for m in models.values():
                outputs = m(inputs)
                logits_sum = outputs if logits_sum is None else logits_sum + outputs

            # Average logits
            logits_avg = logits_sum / len(models)

            # Loss & predictions
            loss = criterion(logits_avg, labels)
            _, preds = torch.max(logits_avg, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(image_datasets['test'])
    test_acc = running_corrects.float() / len(image_datasets['test'])

    print("\n" + "="*50)
    print("ENSEMBLE TEST SET RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Total Test Samples: {len(image_datasets['test'])}")
    print(f"Models used: {model_names}")
    print("="*50)

    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc.item()),
        'test_samples': len(image_datasets['test']),
        'models': model_names,
    }

    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to test_results.json")


if __name__ == '__main__':
    evaluate_test_set()

