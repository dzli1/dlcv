"""
Training script for artist classification using folder-based dataset.
Works with output from create_set.py (ImageFolder structure).
Trains ResNet, ViT, and Gram-based ResNet models.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models import (
    ResNetArtistClassifier,
    ResNetGramStyleClassifier,  # We'll adapt this for artist classification
    ViTArtistClassifier
)
from utils.metrics import plot_training_curves, save_metrics_json


class Trainer:
    """Unified trainer for artist classification models."""

    def __init__(self, model, model_name, device, output_dir):
        """
        Args:
            model: PyTorch model
            model_name: Name identifier (e.g., 'resnet_artist')
            device: torch device
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir

        # Create output directories
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints', model_name)
        self.logs_dir = os.path.join(output_dir, 'logs', model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, dataloader, criterion, optimizer, scaler=None):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f'Training {self.model_name}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total += images.size(0)

            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def validate(self, dataloader, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f'Validating {self.model_name}'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                total += images.size(0)

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs, learning_rate, use_amp=True):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Task: Artist Classification")
        print(f"{'='*60}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Mixed precision training
        scaler = GradScaler() if use_amp else None

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, scaler)
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, optimizer)
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth', epoch, optimizer)

        # Save training history
        self.save_history()
        self.plot_history()

        print(f"\n{'='*60}")
        print(f"Training complete for {self.model_name}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")

    def save_checkpoint(self, filename, epoch, optimizer):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.logs_dir, 'training_history.json')
        save_metrics_json(self.history, history_path)

    def plot_history(self):
        """Plot training curves."""
        curves_path = os.path.join(self.logs_dir, 'training_curves.png')
        plot_training_curves(self.history, curves_path, title=f'{self.model_name} Training History')


def get_transforms(split='train', image_size=224):
    """
    Get appropriate transforms for train/val splits.

    Args:
        split: 'train' or 'val'
        image_size: Target image size (default: 224)

    Returns:
        torchvision.transforms composition
    """
    # ImageNet normalization (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        # Data augmentation for training
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33))
        ])
    else:
        # No augmentation for validation
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(data_dir, batch_size, num_workers, val_split=0.2, image_size=224):
    """
    Create train and validation DataLoaders from ImageFolder structure.

    Args:
        data_dir: Directory containing artist subfolders
        batch_size: Batch size
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        image_size: Target image size

    Returns:
        train_loader, val_loader, class_names, num_classes
    """
    # Create full dataset with training transforms temporarily
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms('train', image_size))

    # Get class information
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply correct transforms to each split
    train_dataset.dataset.transform = get_transforms('train', image_size)
    val_dataset.dataset.transform = get_transforms('val', image_size)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, class_names, num_classes


class ResNetGramArtistClassifier(nn.Module):
    """
    Adapter to use ResNetGramStyleClassifier for artist classification.
    The Gram matrix features work for any classification task, not just style.
    """
    def __init__(self, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Import here to avoid circular dependency
        from models import ResNetGramStyleClassifier
        # Use the Gram model but for artist classification
        self.model = ResNetGramStyleClassifier(
            num_styles=num_artists,  # Use num_artists instead
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    def forward(self, x):
        return self.model(x)

    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()


def main():
    parser = argparse.ArgumentParser(description='Train artist classification models from folder structure')
    parser.add_argument('--data_dir', type=str, default='./artset', help='Path to artist folders')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--models', type=str, default='all', help='Which models to train (all, resnet, vit, gram)')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, class_names, num_artists = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    print(f"\nDataset Info:")
    print(f"  Total Artists: {num_artists}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"\nArtist classes: {class_names[:10]}..." if len(class_names) > 10 else f"\nArtist classes: {class_names}")

    # Save class names for later use
    metadata = {
        'num_artists': num_artists,
        'class_names': class_names,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'created_at': datetime.now().isoformat()
    }

    metadata_path = os.path.join(args.output_dir, 'artist_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    # Define models to train
    models_to_train = []

    if args.models in ['all', 'resnet']:
        models_to_train.append(
            (ResNetArtistClassifier(num_artists, pretrained=True), 'resnet_artist')
        )

    if args.models in ['all', 'gram']:
        models_to_train.append(
            (ResNetGramArtistClassifier(num_artists, pretrained=True), 'resnet_gram_artist')
        )

    if args.models in ['all', 'vit']:
        models_to_train.append(
            (ViTArtistClassifier(num_artists, pretrained=True), 'vit_artist')
        )

    # Train each model
    for model, model_name in models_to_train:
        trainer = Trainer(model, model_name, device, args.output_dir)
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            use_amp=(not args.no_amp and device.type == 'cuda')
        )

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
