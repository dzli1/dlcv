"""
Main training script for all model variants.
Trains ResNet and ViT models for style-only, artist-only, and multi-task classification.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models import (
    ResNetStyleClassifier, ResNetArtistClassifier, ResNetMultiTaskClassifier,
    ResNetGramStyleClassifier,
    ViTStyleClassifier, ViTArtistClassifier, ViTMultiTaskClassifier
)
from utils import get_dataloaders
from utils.metrics import plot_training_curves, save_metrics_json


class Trainer:
    """Unified trainer for all model variants."""

    def __init__(self, model, model_name, task_type, device, output_dir):
        """
        Args:
            model: PyTorch model
            model_name: Name identifier (e.g., 'resnet_style')
            task_type: 'style', 'artist', or 'multitask'
            device: torch device
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.task_type = task_type
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

        # For multi-task
        if task_type == 'multitask':
            self.history.update({
                'train_style_acc': [],
                'train_artist_acc': [],
                'val_style_acc': [],
                'val_artist_acc': []
            })

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, dataloader, criterion, optimizer, scaler=None):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # For multi-task
        if self.task_type == 'multitask':
            style_correct = 0
            artist_correct = 0

        pbar = tqdm(dataloader, desc=f'Training {self.model_name}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            style_labels = batch['style_label'].to(self.device)
            artist_labels = batch['artist_label'].to(self.device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    if self.task_type == 'style':
                        outputs = self.model(images)
                        loss = criterion(outputs, style_labels)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == style_labels).sum().item()
                    elif self.task_type == 'artist':
                        outputs = self.model(images)
                        loss = criterion(outputs, artist_labels)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == artist_labels).sum().item()
                    else:  # multitask
                        style_out, artist_out = self.model(images)
                        loss = criterion(style_out, style_labels) + criterion(artist_out, artist_labels)
                        _, style_pred = torch.max(style_out, 1)
                        _, artist_pred = torch.max(artist_out, 1)
                        style_correct += (style_pred == style_labels).sum().item()
                        artist_correct += (artist_pred == artist_labels).sum().item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if self.task_type == 'style':
                    outputs = self.model(images)
                    loss = criterion(outputs, style_labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == style_labels).sum().item()
                elif self.task_type == 'artist':
                    outputs = self.model(images)
                    loss = criterion(outputs, artist_labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == artist_labels).sum().item()
                else:  # multitask
                    style_out, artist_out = self.model(images)
                    loss = criterion(style_out, style_labels) + criterion(artist_out, artist_labels)
                    _, style_pred = torch.max(style_out, 1)
                    _, artist_pred = torch.max(artist_out, 1)
                    style_correct += (style_pred == style_labels).sum().item()
                    artist_correct += (artist_pred == artist_labels).sum().item()

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total += images.size(0)

            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        epoch_loss = running_loss / len(dataloader)

        if self.task_type == 'multitask':
            style_acc = 100 * style_correct / total
            artist_acc = 100 * artist_correct / total
            return epoch_loss, style_acc, artist_acc
        else:
            accuracy = 100 * correct / total
            return epoch_loss, accuracy

    def validate(self, dataloader, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # For multi-task
        if self.task_type == 'multitask':
            style_correct = 0
            artist_correct = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Validating {self.model_name}'):
                images = batch['image'].to(self.device)
                style_labels = batch['style_label'].to(self.device)
                artist_labels = batch['artist_label'].to(self.device)

                if self.task_type == 'style':
                    outputs = self.model(images)
                    loss = criterion(outputs, style_labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == style_labels).sum().item()
                elif self.task_type == 'artist':
                    outputs = self.model(images)
                    loss = criterion(outputs, artist_labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == artist_labels).sum().item()
                else:  # multitask
                    style_out, artist_out = self.model(images)
                    loss = criterion(style_out, style_labels) + criterion(artist_out, artist_labels)
                    _, style_pred = torch.max(style_out, 1)
                    _, artist_pred = torch.max(artist_out, 1)
                    style_correct += (style_pred == style_labels).sum().item()
                    artist_correct += (artist_pred == artist_labels).sum().item()

                running_loss += loss.item()
                total += images.size(0)

        epoch_loss = running_loss / len(dataloader)

        if self.task_type == 'multitask':
            style_acc = 100 * style_correct / total
            artist_acc = 100 * artist_correct / total
            return epoch_loss, style_acc, artist_acc
        else:
            accuracy = 100 * correct / total
            return epoch_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs, learning_rate, use_amp=True):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Task: {self.task_type}")
        print(f"{'='*60}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Mixed precision training
        scaler = GradScaler() if use_amp else None

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Train
            train_results = self.train_epoch(train_loader, criterion, optimizer, scaler)
            val_results = self.validate(val_loader, criterion)

            # Update history
            if self.task_type == 'multitask':
                train_loss, train_style_acc, train_artist_acc = train_results
                val_loss, val_style_acc, val_artist_acc = val_results

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_style_acc'].append(train_style_acc)
                self.history['train_artist_acc'].append(train_artist_acc)
                self.history['val_style_acc'].append(val_style_acc)
                self.history['val_artist_acc'].append(val_artist_acc)

                # Average accuracy for saving best model
                train_acc = (train_style_acc + train_artist_acc) / 2
                val_acc = (val_style_acc + val_artist_acc) / 2
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)

                print(f"Train Loss: {train_loss:.4f} | Style Acc: {train_style_acc:.2f}% | Artist Acc: {train_artist_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Style Acc: {val_style_acc:.2f}% | Artist Acc: {val_artist_acc:.2f}%")
            else:
                train_loss, train_acc = train_results
                val_loss, val_acc = val_results

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)

                print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check (commented out)
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     print(f"Early stopping counter: {patience_counter}/{patience}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, optimizer)
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth', epoch, optimizer)

            # Early stopping (commented out)
            # if patience_counter >= patience:
            #     print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            #     print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
            #     break

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


def main():
    parser = argparse.ArgumentParser(description='Train art classification models')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--models', type=str, default='all', help='Which models to train (all, resnet, vit)')
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

    # Load metadata to get number of classes
    metadata_dir = os.path.join(args.data_dir.replace('processed', 'metadata'))
    with open(os.path.join(metadata_dir, 'label_mappings.json'), 'r') as f:
        mappings = json.load(f)

    num_styles = mappings['num_styles']
    num_artists = mappings['num_artists']

    print(f"\nDataset Info:")
    print(f"  Styles: {num_styles}")
    print(f"  Artists: {num_artists}")

    # Get dataloaders
    print(f"\nLoading data...")
    dataloaders, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Define models to train
    models_to_train = []

    if args.models in ['all', 'resnet']:
        models_to_train.extend([
            (ResNetStyleClassifier(num_styles, pretrained=True), 'resnet_style', 'style'),
            (ResNetArtistClassifier(num_artists, pretrained=True), 'resnet_artist', 'artist'),
            (ResNetMultiTaskClassifier(num_styles, num_artists, pretrained=True), 'resnet_multitask', 'multitask'),
            (ResNetGramStyleClassifier(num_styles, pretrained=True), 'resnet_gram_style', 'style'),
        ])

    if args.models in ['all', 'vit']:
        models_to_train.extend([
            (ViTStyleClassifier(num_styles, pretrained=True), 'vit_style', 'style'),
            (ViTArtistClassifier(num_artists, pretrained=True), 'vit_artist', 'artist'),
            (ViTMultiTaskClassifier(num_styles, num_artists, pretrained=True), 'vit_multitask', 'multitask'),
        ])

    # Train each model
    for model, model_name, task_type in models_to_train:
        trainer = Trainer(model, model_name, task_type, device, args.output_dir)
        trainer.train(
            dataloaders['train'],
            dataloaders['val'],
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
