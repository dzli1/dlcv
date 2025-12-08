"""
Visualize dataset statistics and sample images.
Creates comprehensive visualizations for understanding the data distribution.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from collections import Counter
import random

# Configuration
DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
random.seed(42)


def load_data():
    """Load dataset information."""
    # Load mappings
    mappings_file = os.path.join(METADATA_DIR, "label_mappings.json")
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    # Load split data
    splits_data = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(PROCESSED_DIR, f"{split}_labels.csv")
        splits_data[split] = pd.read_csv(csv_path)

    return mappings, splits_data


def plot_class_distribution(splits_data, mappings):
    """Plot distribution of styles and artists across splits."""
    print("Creating class distribution plots...")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Style distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Style Distribution Across Splits', fontsize=16, fontweight='bold')

    for idx, (split_name, data) in enumerate(splits_data.items()):
        style_counts = data['style'].value_counts()
        axes[idx].barh(range(len(style_counts)), style_counts.values)
        axes[idx].set_yticks(range(len(style_counts)))
        axes[idx].set_yticklabels(style_counts.index, fontsize=9)
        axes[idx].set_xlabel('Number of Images')
        axes[idx].set_title(f'{split_name.capitalize()} ({len(data)} images)')
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'style_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Artist distribution (top 30)
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.suptitle('Top 30 Artists Distribution Across Splits', fontsize=16, fontweight='bold')

    for idx, (split_name, data) in enumerate(splits_data.items()):
        artist_counts = data['artist'].value_counts().head(30)
        axes[idx].barh(range(len(artist_counts)), artist_counts.values)
        axes[idx].set_yticks(range(len(artist_counts)))
        axes[idx].set_yticklabels(artist_counts.index, fontsize=8)
        axes[idx].set_xlabel('Number of Images')
        axes[idx].set_title(f'{split_name.capitalize()} (Top 30 Artists)')
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'artist_distribution_top30.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Combined style distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    combined_data = pd.concat([
        splits_data['train'][['style']].assign(split='train'),
        splits_data['val'][['style']].assign(split='val'),
        splits_data['test'][['style']].assign(split='test')
    ])

    style_order = splits_data['train']['style'].value_counts().index
    sns.countplot(data=combined_data, y='style', hue='split', order=style_order, ax=ax)
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Style')
    ax.set_title('Style Distribution Across All Splits', fontsize=14, fontweight='bold')
    ax.legend(title='Split')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'style_distribution_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved class distribution plots to {FIGURES_DIR}")


def plot_split_statistics(splits_data):
    """Plot overall split statistics."""
    print("Creating split statistics plots...")

    # Split sizes
    split_sizes = {name: len(data) for name, data in splits_data.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    axes[0].bar(split_sizes.keys(), split_sizes.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Dataset Split Sizes', fontweight='bold')
    for i, (split, size) in enumerate(split_sizes.items()):
        axes[0].text(i, size + 50, str(size), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(split_sizes.values(), labels=split_sizes.keys(), autopct='%1.1f%%',
                colors=['#1f77b4', '#ff7f0e', '#2ca02c'], startangle=90)
    axes[1].set_title('Dataset Split Proportions', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'split_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved split statistics to {FIGURES_DIR}")


def plot_sample_images(splits_data):
    """Plot sample images from each style."""
    print("Creating sample image grid...")

    # Get unique styles
    styles = splits_data['train']['style'].unique()

    # Create grid: 3-4 columns, enough rows for all styles
    n_cols = 4
    n_rows = (len(styles) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten() if len(styles) > 1 else [axes]

    for idx, style in enumerate(styles):
        # Get random image from this style
        style_images = splits_data['train'][splits_data['train']['style'] == style]
        sample = style_images.sample(1).iloc[0]

        # Load and display image
        img_path = os.path.join(PROCESSED_DIR, 'train', sample['filename'])
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{style}\n(Artist: {sample['artist']})", fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\n{style}", ha='center', va='center')
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(styles), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sample Images from Each Style', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sample_images_by_style.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved sample images to {FIGURES_DIR}")


def plot_artist_style_heatmap(splits_data):
    """Create heatmap showing artist-style relationships."""
    print("Creating artist-style heatmap...")

    train_data = splits_data['train']

    # Create cross-tabulation (limit to top 30 artists for readability)
    top_artists = train_data['artist'].value_counts().head(30).index
    filtered_data = train_data[train_data['artist'].isin(top_artists)]

    crosstab = pd.crosstab(filtered_data['artist'], filtered_data['style'])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(crosstab, annot=False, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Images'})
    ax.set_xlabel('Style', fontweight='bold')
    ax.set_ylabel('Artist (Top 30)', fontweight='bold')
    ax.set_title('Artist-Style Distribution (Training Set)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'artist_style_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved artist-style heatmap to {FIGURES_DIR}")


def create_statistics_report(mappings, splits_data):
    """Create a text report with detailed statistics."""
    print("Creating statistics report...")

    report_lines = []
    report_lines.append("="*70)
    report_lines.append("WIKIART DATASET STATISTICS REPORT")
    report_lines.append("="*70)
    report_lines.append("")

    # Overall statistics
    total_images = sum(len(data) for data in splits_data.values())
    report_lines.append(f"Total Images: {total_images}")
    report_lines.append(f"Number of Styles: {mappings['num_styles']}")
    report_lines.append(f"Number of Artists: {mappings['num_artists']}")
    report_lines.append("")

    # Split statistics
    report_lines.append("SPLIT STATISTICS")
    report_lines.append("-" * 70)
    for split_name, data in splits_data.items():
        report_lines.append(f"\n{split_name.upper()} SET:")
        report_lines.append(f"  Total images: {len(data)}")
        report_lines.append(f"  Percentage: {len(data)/total_images*100:.2f}%")
        report_lines.append(f"  Unique styles: {data['style'].nunique()}")
        report_lines.append(f"  Unique artists: {data['artist'].nunique()}")

    # Style statistics
    report_lines.append("\n" + "="*70)
    report_lines.append("STYLE STATISTICS (Training Set)")
    report_lines.append("-" * 70)
    style_counts = splits_data['train']['style'].value_counts()
    for style, count in style_counts.items():
        report_lines.append(f"  {style:30s}: {count:5d} images ({count/len(splits_data['train'])*100:.2f}%)")

    # Artist statistics (top 20)
    report_lines.append("\n" + "="*70)
    report_lines.append("TOP 20 ARTISTS (Training Set)")
    report_lines.append("-" * 70)
    artist_counts = splits_data['train']['artist'].value_counts().head(20)
    for rank, (artist, count) in enumerate(artist_counts.items(), 1):
        report_lines.append(f"  {rank:2d}. {artist:35s}: {count:5d} images")

    # Save report
    report_path = os.path.join(FIGURES_DIR, 'dataset_statistics.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"  Saved statistics report to {report_path}")

    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    """Main execution function."""
    print("="*60)
    print("Dataset Visualization")
    print("="*60)
    print()

    # Load data
    mappings, splits_data = load_data()

    # Create visualizations
    plot_class_distribution(splits_data, mappings)
    plot_split_statistics(splits_data)
    plot_sample_images(splits_data)
    plot_artist_style_heatmap(splits_data)

    # Create report
    create_statistics_report(mappings, splits_data)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nAll visualizations saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
