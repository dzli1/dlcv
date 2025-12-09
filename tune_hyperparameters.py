# """
# Hyperparameter tuning for ResNet Style classifier.
# Tests different combinations of learning rate, batch size, weight decay, and dropout.
# """

# import os
# import json
# import argparse
# import itertools
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# import numpy as np
# from datetime import datetime

# from models import ResNetStyleClassifier
# from utils import get_dataloaders
# from utils.metrics import save_metrics_json


# class HyperparameterTuner:
#     """Hyperparameter tuning with grid search."""

#     def __init__(self, num_styles, device, output_dir):
#         self.num_styles = num_styles
#         self.device = device
#         self.output_dir = output_dir
#         self.results = []

#         # Create output directory
#         self.tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning')
#         os.makedirs(self.tuning_dir, exist_ok=True)

#     def train_with_config(self, config, train_loader, val_loader, num_epochs=15):
#         """Train model with specific hyperparameter configuration."""
#         print(f"\n{'='*60}")
#         print(f"Testing configuration:")
#         print(f"  Learning Rate: {config['lr']}")
#         print(f"  Batch Size: {config['batch_size']}")
#         print(f"  Weight Decay: {config['weight_decay']}")
#         print(f"  Dropout: {config['dropout']}")
#         print(f"{'='*60}\n")

#         # Create model with custom dropout
#         model = ResNetStyleClassifier(
#             num_styles=self.num_styles,
#             pretrained=True,
#             freeze_backbone=False
#         )

#         # Modify dropout if needed
#         if config['dropout'] != 0.5:
#             # Find and replace dropout layer
#             for name, module in model.backbone.fc.named_children():
#                 if isinstance(module, nn.Dropout):
#                     model.backbone.fc[int(name)] = nn.Dropout(config['dropout'])

#         model = model.to(self.device)

#         # Setup training
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW(
#             model.parameters(),
#             lr=config['lr'],
#             weight_decay=config['weight_decay']
#         )
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=2, verbose=False
#         )
#         scaler = GradScaler() if self.device.type == 'cuda' else None

#         # Training history
#         history = {
#             'train_loss': [],
#             'val_loss': [],
#             'train_acc': [],
#             'val_acc': []
#         }

#         best_val_acc = 0.0
#         best_val_loss = float('inf')

#         # Training loop
#         for epoch in range(num_epochs):
#             # Train
#             model.train()
#             train_loss = 0.0
#             train_correct = 0
#             train_total = 0

#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
#             for batch in pbar:
#                 images = batch['image'].to(self.device)
#                 labels = batch['style_label'].to(self.device)

#                 optimizer.zero_grad()

#                 if scaler is not None:
#                     with autocast():
#                         outputs = model(images)
#                         loss = criterion(outputs, labels)
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()

#                 train_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 train_correct += (predicted == labels).sum().item()
#                 train_total += labels.size(0)

#                 pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

#             # Validation
#             model.eval()
#             val_loss = 0.0
#             val_correct = 0
#             val_total = 0

#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False):
#                     images = batch['image'].to(self.device)
#                     labels = batch['style_label'].to(self.device)

#                     outputs = model(images)
#                     loss = criterion(outputs, labels)

#                     val_loss += loss.item()
#                     _, predicted = torch.max(outputs, 1)
#                     val_correct += (predicted == labels).sum().item()
#                     val_total += labels.size(0)

#             # Calculate metrics
#             epoch_train_loss = train_loss / len(train_loader)
#             epoch_val_loss = val_loss / len(val_loader)
#             epoch_train_acc = 100 * train_correct / train_total
#             epoch_val_acc = 100 * val_correct / val_total

#             history['train_loss'].append(epoch_train_loss)
#             history['val_loss'].append(epoch_val_loss)
#             history['train_acc'].append(epoch_train_acc)
#             history['val_acc'].append(epoch_val_acc)

#             # Update best
#             if epoch_val_acc > best_val_acc:
#                 best_val_acc = epoch_val_acc
#             if epoch_val_loss < best_val_loss:
#                 best_val_loss = epoch_val_loss

#             # Learning rate scheduling
#             scheduler.step(epoch_val_loss)

#             print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.2f}%, "
#                   f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.2f}%")

#         # Return results
#         result = {
#             'config': config,
#             'best_val_acc': best_val_acc,
#             'best_val_loss': best_val_loss,
#             'final_val_acc': history['val_acc'][-1],
#             'final_train_acc': history['train_acc'][-1],
#             'history': history
#         }

#         # Clean up
#         del model
#         del optimizer
#         del scheduler
#         torch.cuda.empty_cache() if torch.cuda.is_available() else None

#         return result

#     def grid_search(self, param_grid, train_loader, val_loader, num_epochs=15):
#         """Perform grid search over hyperparameters."""
#         print(f"\n{'='*60}")
#         print("Starting Hyperparameter Grid Search")
#         print(f"{'='*60}\n")

#         # Generate all combinations
#         keys = param_grid.keys()
#         values = param_grid.values()
#         combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#         print(f"Total configurations to test: {len(combinations)}\n")

#         # Test each configuration
#         for idx, config in enumerate(combinations, 1):
#             print(f"\n[{idx}/{len(combinations)}] Testing configuration...")

#             result = self.train_with_config(config, train_loader, val_loader, num_epochs)
#             self.results.append(result)

#             print(f"✓ Best Val Accuracy: {result['best_val_acc']:.2f}%")

#         # Sort by best validation accuracy
#         self.results.sort(key=lambda x: x['best_val_acc'], reverse=True)

#         # Save results
#         self.save_results()

#     def save_results(self):
#         """Save tuning results."""
#         # Save full results
#         results_file = os.path.join(self.tuning_dir, 'tuning_results.json')
#         save_metrics_json(self.results, results_file)

#         # Create summary report
#         summary_lines = []
#         summary_lines.append("="*80)
#         summary_lines.append("HYPERPARAMETER TUNING RESULTS")
#         summary_lines.append("="*80)
#         summary_lines.append("")
#         summary_lines.append(f"Total configurations tested: {len(self.results)}")
#         summary_lines.append("")

#         summary_lines.append("TOP 5 CONFIGURATIONS:")
#         summary_lines.append("-" * 80)
#         for idx, result in enumerate(self.results[:5], 1):
#             config = result['config']
#             summary_lines.append(f"\n{idx}. Val Accuracy: {result['best_val_acc']:.2f}%")
#             summary_lines.append(f"   Learning Rate:  {config['lr']}")
#             summary_lines.append(f"   Batch Size:     {config['batch_size']}")
#             summary_lines.append(f"   Weight Decay:   {config['weight_decay']}")
#             summary_lines.append(f"   Dropout:        {config['dropout']}")
#             summary_lines.append(f"   Final Val Acc:  {result['final_val_acc']:.2f}%")

#         summary_lines.append("\n" + "="*80)
#         summary_lines.append("BEST CONFIGURATION:")
#         summary_lines.append("="*80)
#         best = self.results[0]
#         summary_lines.append(f"Learning Rate:  {best['config']['lr']}")
#         summary_lines.append(f"Batch Size:     {best['config']['batch_size']}")
#         summary_lines.append(f"Weight Decay:   {best['config']['weight_decay']}")
#         summary_lines.append(f"Dropout:        {best['config']['dropout']}")
#         summary_lines.append(f"Best Val Acc:   {best['best_val_acc']:.2f}%")
#         summary_lines.append("")

#         summary_file = os.path.join(self.tuning_dir, 'tuning_summary.txt')
#         with open(summary_file, 'w') as f:
#             f.write('\n'.join(summary_lines))

#         print("\n" + '\n'.join(summary_lines))
#         print(f"\nResults saved to {self.tuning_dir}")


# def main():
#     parser = argparse.ArgumentParser(description='Hyperparameter tuning for ResNet Style classifier')
#     parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data')
#     parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
#     parser.add_argument('--num_epochs', type=int, default=15, help='Epochs per configuration')
#     parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers')
#     parser.add_argument('--quick', action='store_true', help='Quick test with fewer configurations')
#     args = parser.parse_args()

#     # Set device
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
#     elif torch.backends.mps.is_available():
#         device = torch.device('mps')
#         print("Using MPS (Apple Silicon)")
#     else:
#         device = torch.device('cpu')
#         print("Using CPU")

#     # Load metadata
#     metadata_dir = os.path.join(args.data_dir.replace('processed', 'metadata'))
#     with open(os.path.join(metadata_dir, 'label_mappings.json'), 'r') as f:
#         mappings = json.load(f)

#     num_styles = mappings['num_styles']
#     print(f"\nNumber of styles: {num_styles}")

#     # Define hyperparameter grid
#     if args.quick:
#         # Quick test with fewer options
#         param_grid = {
#             'lr': [1e-4, 5e-4],
#             'batch_size': [32, 64],
#             'weight_decay': [1e-4, 1e-3],
#             'dropout': [0.3, 0.5]
#         }
#         print("\nQuick mode: Testing 16 configurations")
#     else:
#         # Full grid search
#         param_grid = {
#             'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
#             'batch_size': [32, 64, 128],
#             'weight_decay': [0, 1e-5, 1e-4, 1e-3],
#             'dropout': [0.2, 0.3, 0.5, 0.7]
#         }
#         print("\nFull mode: Testing 240 configurations")

#     print("\nParameter grid:")
#     for key, values in param_grid.items():
#         print(f"  {key}: {values}")

#     # Create tuner
#     tuner = HyperparameterTuner(num_styles, device, args.output_dir)

#     # Note: We'll create dataloaders on the fly with different batch sizes
#     # For now, use a base batch size to get the val loader
#     print("\nLoading validation data...")
#     dataloaders, _ = get_dataloaders(
#         data_dir=args.data_dir,
#         batch_size=64,  # We'll recreate with correct batch size
#         num_workers=args.num_workers
#     )

#     # Store original dataset
#     from utils.dataset import ArtDataset, get_transforms
#     train_dataset = ArtDataset(
#         os.path.join(args.data_dir, 'train'),
#         os.path.join(args.data_dir, 'train_labels.csv'),
#         get_transforms('train')
#     )
#     val_dataset = ArtDataset(
#         os.path.join(args.data_dir, 'val'),
#         os.path.join(args.data_dir, 'val_labels.csv'),
#         get_transforms('val')
#     )

#     # Override grid_search to create dataloaders with correct batch size
#     original_grid_search = tuner.grid_search

#     def grid_search_wrapper(param_grid, _, __, num_epochs):
#         keys = param_grid.keys()
#         values = param_grid.values()
#         combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#         print(f"Total configurations to test: {len(combinations)}\n")

#         for idx, config in enumerate(combinations, 1):
#             print(f"\n[{idx}/{len(combinations)}] Testing configuration...")

#             # Create dataloaders with config batch size
#             from torch.utils.data import DataLoader
#             train_loader = DataLoader(
#                 train_dataset,
#                 batch_size=config['batch_size'],
#                 shuffle=True,
#                 num_workers=args.num_workers,
#                 pin_memory=True
#             )
#             val_loader = DataLoader(
#                 val_dataset,
#                 batch_size=config['batch_size'],
#                 shuffle=False,
#                 num_workers=args.num_workers,
#                 pin_memory=True
#             )

#             result = tuner.train_with_config(config, train_loader, val_loader, num_epochs)
#             tuner.results.append(result)

#             print(f"✓ Best Val Accuracy: {result['best_val_acc']:.2f}%")

#         tuner.results.sort(key=lambda x: x['best_val_acc'], reverse=True)
#         tuner.save_results()

#     tuner.grid_search = lambda pg, tl, vl, ne: grid_search_wrapper(pg, None, None, ne)

#     # Run grid search
#     tuner.grid_search(param_grid, None, None, args.num_epochs)

#     print("\n" + "="*60)
#     print("HYPERPARAMETER TUNING COMPLETE!")
#     print("="*60)
#     print(f"\nResults saved to: {tuner.tuning_dir}")
#     print(f"View summary: cat {tuner.tuning_dir}/tuning_summary.txt")


# if __name__ == '__main__':
#     main()
