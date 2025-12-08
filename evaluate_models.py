"""
Comprehensive evaluation script for all trained models.
Generates confusion matrices, embeddings visualizations, and detailed metrics.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from models import (
    ResNetStyleClassifier, ResNetArtistClassifier, ResNetMultiTaskClassifier,
    ViTStyleClassifier, ViTArtistClassifier, ViTMultiTaskClassifier
)
from utils import get_dataloaders
from utils.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_json


class ModelEvaluator:
    """Evaluator for trained models."""

    def __init__(self, model, model_name, task_type, device, mappings, output_dir):
        """
        Args:
            model: PyTorch model
            model_name: Name identifier
            task_type: 'style', 'artist', or 'multitask'
            device: torch device
            mappings: Label mappings dictionary
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.task_type = task_type
        self.device = device
        self.mappings = mappings
        self.output_dir = output_dir

        # Create output directories
        self.eval_dir = os.path.join(output_dir, 'reports', model_name)
        self.cm_dir = os.path.join(output_dir, 'reports', 'confusion_matrices', model_name)
        self.emb_dir = os.path.join(output_dir, 'reports', 'embeddings', model_name)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Best Val Accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    def evaluate(self, dataloader, split_name='test'):
        """Evaluate model on a dataset."""
        print(f"\nEvaluating {self.model_name} on {split_name} set...")

        self.model.eval()

        all_style_labels = []
        all_artist_labels = []
        all_style_preds = []
        all_artist_preds = []
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Evaluating'):
                images = batch['image'].to(self.device)
                style_labels = batch['style_label']
                artist_labels = batch['artist_label']

                if self.task_type == 'style':
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_style_preds.extend(predicted.cpu().numpy())
                    all_style_labels.extend(style_labels.numpy())

                elif self.task_type == 'artist':
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_artist_preds.extend(predicted.cpu().numpy())
                    all_artist_labels.extend(artist_labels.numpy())

                else:  # multitask
                    style_out, artist_out = self.model(images)
                    _, style_pred = torch.max(style_out, 1)
                    _, artist_pred = torch.max(artist_out, 1)
                    all_style_preds.extend(style_pred.cpu().numpy())
                    all_artist_preds.extend(artist_pred.cpu().numpy())
                    all_style_labels.extend(style_labels.numpy())
                    all_artist_labels.extend(artist_labels.numpy())

                    # Extract embeddings for visualization
                    if hasattr(self.model, 'get_embeddings'):
                        embeddings = self.model.get_embeddings(images)
                        all_embeddings.append(embeddings.cpu().numpy())

        # Calculate and save metrics
        results = {}

        if self.task_type in ['style', 'multitask']:
            style_names = [self.mappings['idx_to_style'][str(i)] for i in range(self.mappings['num_styles'])]
            style_metrics = calculate_metrics(all_style_labels, all_style_preds, style_names)
            results['style_metrics'] = style_metrics

            print(f"\nStyle Classification Metrics:")
            print(f"  Accuracy: {style_metrics['accuracy']:.4f}")
            print(f"  F1 (Macro): {style_metrics['f1_macro']:.4f}")
            print(f"  F1 (Weighted): {style_metrics['f1_weighted']:.4f}")

            # Plot confusion matrix
            cm_path = os.path.join(self.cm_dir, f'{split_name}_style_confusion_matrix.png')
            plot_confusion_matrix(
                all_style_labels, all_style_preds, style_names,
                cm_path, title=f'{self.model_name} - Style Classification ({split_name})'
            )

        if self.task_type in ['artist', 'multitask']:
            artist_names = [self.mappings['idx_to_artist'][str(i)] for i in range(self.mappings['num_artists'])]
            artist_metrics = calculate_metrics(all_artist_labels, all_artist_preds, artist_names)
            results['artist_metrics'] = artist_metrics

            print(f"\nArtist Classification Metrics:")
            print(f"  Accuracy: {artist_metrics['accuracy']:.4f}")
            print(f"  F1 (Macro): {artist_metrics['f1_macro']:.4f}")
            print(f"  F1 (Weighted): {artist_metrics['f1_weighted']:.4f}")

            # Plot confusion matrix
            cm_path = os.path.join(self.cm_dir, f'{split_name}_artist_confusion_matrix.png')
            plot_confusion_matrix(
                all_artist_labels, all_artist_preds, artist_names,
                cm_path, title=f'{self.model_name} - Artist Classification ({split_name})'
            )

        # Visualize embeddings (for multitask models)
        if all_embeddings and self.task_type == 'multitask':
            embeddings = np.vstack(all_embeddings)
            self.visualize_embeddings(embeddings, all_style_labels, all_artist_labels, split_name)

        # Save metrics
        metrics_path = os.path.join(self.eval_dir, f'{split_name}_metrics.json')
        save_metrics_json(results, metrics_path)

        return results

    def visualize_embeddings(self, embeddings, style_labels, artist_labels, split_name):
        """Visualize embeddings using t-SNE."""
        print("\nGenerating t-SNE embeddings visualization...")

        # Sample if too many points
        max_points = 5000
        if len(embeddings) > max_points:
            indices = np.random.choice(len(embeddings), max_points, replace=False)
            embeddings = embeddings[indices]
            style_labels = np.array(style_labels)[indices]
            artist_labels = np.array(artist_labels)[indices]

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot by style
        fig, ax = plt.subplots(figsize=(12, 10))
        style_names = [self.mappings['idx_to_style'][str(i)] for i in range(self.mappings['num_styles'])]
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=style_labels, cmap='tab20', alpha=0.6, s=10
        )
        legend = ax.legend(
            handles=scatter.legend_elements()[0],
            labels=style_names,
            title='Style',
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=8
        )
        ax.set_title(f'{self.model_name} - Embeddings Colored by Style ({split_name})', fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.emb_dir, f'{split_name}_embeddings_style.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Plot by artist (top 20 artists for clarity)
        fig, ax = plt.subplots(figsize=(12, 10))
        top_20_artists = sorted(set(artist_labels), key=lambda x: list(artist_labels).count(x), reverse=True)[:20]
        mask = np.isin(artist_labels, top_20_artists)
        filtered_embeddings = embeddings_2d[mask]
        filtered_labels = np.array(artist_labels)[mask]

        scatter = ax.scatter(
            filtered_embeddings[:, 0], filtered_embeddings[:, 1],
            c=filtered_labels, cmap='tab20', alpha=0.6, s=10
        )
        artist_names = [self.mappings['idx_to_artist'][str(i)] for i in top_20_artists]
        legend = ax.legend(
            handles=scatter.legend_elements()[0],
            labels=artist_names,
            title='Artist (Top 20)',
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=8
        )
        ax.set_title(f'{self.model_name} - Embeddings Colored by Artist ({split_name})', fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.emb_dir, f'{split_name}_embeddings_artist_top20.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved embeddings visualizations to {self.emb_dir}")


def create_comparison_report(output_dir):
    """Create a comparison report across all models."""
    print("\nCreating model comparison report...")

    reports_dir = os.path.join(output_dir, 'reports')
    model_dirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d)) and d != 'confusion_matrices' and d != 'embeddings']

    comparison = {}

    for model_name in model_dirs:
        metrics_file = os.path.join(reports_dir, model_name, 'test_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            comparison[model_name] = metrics

    # Save comparison
    comparison_file = os.path.join(reports_dir, 'model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Create summary table
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("MODEL COMPARISON SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append("")

    for model_name, metrics in sorted(comparison.items()):
        summary_lines.append(f"\n{model_name.upper()}")
        summary_lines.append("-" * 80)

        if 'style_metrics' in metrics:
            summary_lines.append(f"  Style Classification:")
            summary_lines.append(f"    Accuracy: {metrics['style_metrics']['accuracy']:.4f}")
            summary_lines.append(f"    F1 (Macro): {metrics['style_metrics']['f1_macro']:.4f}")
            summary_lines.append(f"    F1 (Weighted): {metrics['style_metrics']['f1_weighted']:.4f}")

        if 'artist_metrics' in metrics:
            summary_lines.append(f"  Artist Classification:")
            summary_lines.append(f"    Accuracy: {metrics['artist_metrics']['accuracy']:.4f}")
            summary_lines.append(f"    F1 (Macro): {metrics['artist_metrics']['f1_macro']:.4f}")
            summary_lines.append(f"    F1 (Weighted): {metrics['artist_metrics']['f1_weighted']:.4f}")

    summary_file = os.path.join(reports_dir, 'comparison_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print(f"\nComparison report saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Which split to evaluate')
    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load metadata
    metadata_dir = os.path.join(args.data_dir.replace('processed', 'metadata'))
    with open(os.path.join(metadata_dir, 'label_mappings.json'), 'r') as f:
        mappings = json.load(f)

    num_styles = mappings['num_styles']
    num_artists = mappings['num_artists']

    # Get dataloaders
    print(f"\nLoading data...")
    dataloaders, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Define models to evaluate (only multi-task models)
    models_to_eval = [
        (ResNetMultiTaskClassifier(num_styles, num_artists), 'resnet_multitask', 'multitask'),
        (ViTMultiTaskClassifier(num_styles, num_artists), 'vit_multitask', 'multitask'),
    ]

    # Evaluate each model
    for model, model_name, task_type in models_to_eval:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', model_name, 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            print(f"\nSkipping {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")

        evaluator = ModelEvaluator(model, model_name, task_type, device, mappings, args.output_dir)
        evaluator.load_checkpoint(checkpoint_path)
        evaluator.evaluate(dataloaders[args.split], split_name=args.split)

        # Clean up
        del model
        del evaluator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create comparison report
    create_comparison_report(args.output_dir)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
