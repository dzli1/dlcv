"""
Resume training from existing checkpoints and continue for additional epochs.
Works with CSV-based dataset (processed format with style and artist labels).
Saves all outputs to a new folder to preserve original training runs.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime

from models import (
    ResNetStyleClassifier,
    ResNetArtistClassifier,
    ResNetMultiTaskClassifier,
    ResNetGramStyleClassifier,
    ViTStyleClassifier,
    ViTArtistClassifier,
    ViTMultiTaskClassifier
)
from utils import get_dataloaders
from utils.metrics import plot_training_curves, save_metrics_json


class ResumeTrainer:
    """Trainer that resumes from checkpoint and continues training."""

    def __init__(self, model, model_name, task_type, checkpoint_path, device, output_dir):
        """
        Args:
            model: PyTorch model
            model_name: Name identifier
            task_type: 'style', 'artist', or 'multitask'
            checkpoint_path: Path to checkpoint to resume from
            device: torch device
            output_dir: New output directory
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.task_type = task_type
        self.device = device
        self.output_dir = output_dir

        # Create new output directories
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints', model_name)
        self.logs_dir = os.path.join(output_dir, 'logs', model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load previous training history
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        })

        # For multi-task, ensure extra keys exist
        if task_type == 'multitask':
            if 'train_style_acc' not in self.history:
                self.history.update({
                    'train_style_acc': [],
                    'train_artist_acc': [],
                    'val_style_acc': [],
                    'val_artist_acc': []
                })

        # Best model tracking
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resuming from epoch {self.start_epoch}")
        print(f"Previous best val acc: {self.best_val_acc:.2f}%")
        print(f"Previous history length: {len(self.history['train_loss'])} epochs")

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

    def train(self, train_loader, val_loader, additional_epochs, learning_rate, use_amp=True):
        """Continue training for additional epochs."""
        print(f"\n{'='*60}")
        print(f"Resuming Training: {self.model_name}")
        print(f"Task: {self.task_type}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"Additional epochs: {additional_epochs}")
        print(f"Total epochs after: {self.start_epoch + additional_epochs}")
        print(f"{'='*60}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        scaler = GradScaler() if use_amp else None

        for epoch in range(additional_epochs):
            current_epoch = self.start_epoch + epoch
            print(f"\nEpoch {current_epoch + 1}")
            print("-" * 40)

            # Train and validate
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

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', current_epoch, optimizer)
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth', current_epoch, optimizer)

        # Save final training history
        self.save_history()
        self.plot_history()

        print(f"\n{'='*60}")
        print(f"Training complete for {self.model_name}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Total epochs trained: {self.start_epoch + additional_epochs}")
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
        plot_training_curves(self.history, curves_path, title=f'{self.model_name} Training History (Resumed)')


def main():
    parser = argparse.ArgumentParser(description='Resume training from existing checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory with checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data (with CSVs)')
    parser.add_argument('--output_dir', type=str, default='./resumed_training', help='New output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--additional_epochs', type=int, default=10, help='Additional epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--models', type=str, default='all', help='Which models to resume (all, resnet, vit, specific name)')
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

    # Create new output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nNew output directory: {output_dir}")

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
    print(f"\nLoading data from {args.data_dir}...")
    dataloaders, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Define all possible models
    model_configs = {
        'resnet_style': (ResNetStyleClassifier(num_styles, pretrained=True), 'style'),
        'resnet_artist': (ResNetArtistClassifier(num_artists, pretrained=True), 'artist'),
        'resnet_multitask': (ResNetMultiTaskClassifier(num_styles, num_artists, pretrained=True), 'multitask'),
        'resnet_gram_style': (ResNetGramStyleClassifier(num_styles, pretrained=True), 'style'),
        'vit_style': (ViTStyleClassifier(num_styles, pretrained=True), 'style'),
        'vit_artist': (ViTArtistClassifier(num_artists, pretrained=True), 'artist'),
        'vit_multitask': (ViTMultiTaskClassifier(num_styles, num_artists, pretrained=True), 'multitask'),
    }

    # Determine which models to resume
    models_to_resume = []

    if args.models == 'all':
        # Try all models
        for model_name, (model, task_type) in model_configs.items():
            checkpoint_path = os.path.join(args.checkpoint_dir, model_name, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                models_to_resume.append((model, model_name, task_type, checkpoint_path))
            else:
                print(f"Skipping {model_name}: checkpoint not found")
    elif args.models == 'resnet':
        # All ResNet models
        for model_name in ['resnet_style', 'resnet_artist', 'resnet_multitask', 'resnet_gram_style']:
            checkpoint_path = os.path.join(args.checkpoint_dir, model_name, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                model, task_type = model_configs[model_name]
                models_to_resume.append((model, model_name, task_type, checkpoint_path))
            else:
                print(f"Skipping {model_name}: checkpoint not found")
    elif args.models == 'vit':
        # All ViT models
        for model_name in ['vit_style', 'vit_artist', 'vit_multitask']:
            checkpoint_path = os.path.join(args.checkpoint_dir, model_name, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                model, task_type = model_configs[model_name]
                models_to_resume.append((model, model_name, task_type, checkpoint_path))
            else:
                print(f"Skipping {model_name}: checkpoint not found")
    else:
        # Specific model name
        if args.models in model_configs:
            checkpoint_path = os.path.join(args.checkpoint_dir, args.models, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                model, task_type = model_configs[args.models]
                models_to_resume.append((model, args.models, task_type, checkpoint_path))
            else:
                print(f"Error: Checkpoint not found: {checkpoint_path}")
        else:
            print(f"Error: Unknown model name: {args.models}")
            print(f"Available models: {', '.join(model_configs.keys())}")
            return

    if not models_to_resume:
        print("\nError: No valid checkpoints found!")
        print(f"Checked directory: {args.checkpoint_dir}")
        return

    print(f"\nFound {len(models_to_resume)} model(s) to resume\n")

    # Resume training for each model
    for model, model_name, task_type, checkpoint_path in models_to_resume:
        trainer = ResumeTrainer(
            model=model,
            model_name=model_name,
            task_type=task_type,
            checkpoint_path=checkpoint_path,
            device=device,
            output_dir=output_dir
        )

        trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            additional_epochs=args.additional_epochs,
            learning_rate=args.learning_rate,
            use_amp=(not args.no_amp and device.type == 'cuda')
        )

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save resume info
    resume_info = {
        'original_checkpoint_dir': args.checkpoint_dir,
        'data_dir': args.data_dir,
        'resumed_at': timestamp,
        'additional_epochs': args.additional_epochs,
        'learning_rate': args.learning_rate,
        'models_resumed': [name for _, name, _, _ in models_to_resume],
        'num_styles': num_styles,
        'num_artists': num_artists
    }

    info_path = os.path.join(output_dir, 'resume_info.json')
    with open(info_path, 'w') as f:
        json.dump(resume_info, f, indent=2)

    print("\n" + "="*60)
    print("ALL MODELS RESUMED AND TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\nNew checkpoints and logs saved to: {output_dir}")
    print(f"Resume info: {info_path}")


if __name__ == '__main__':
    main()
