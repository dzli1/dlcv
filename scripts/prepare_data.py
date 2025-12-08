"""
Prepare data for training: split into train/val/test sets and create PyTorch datasets.
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42


def load_metadata():
    """Load metadata from download step."""
    metadata_file = os.path.join(METADATA_DIR, "dataset_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    mappings_file = os.path.join(METADATA_DIR, "label_mappings.json")
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    return metadata, mappings


def stratified_split(metadata, mappings):
    """
    Split dataset into train/val/test with stratification by style.
    This ensures balanced representation of each style in all splits.
    """
    print("Performing stratified split by style...")

    random.seed(RANDOM_SEED)

    # Group by style
    style_groups = {}
    for item in metadata:
        style = item['style']
        if style not in style_groups:
            style_groups[style] = []
        style_groups[style].append(item)

    train_data = []
    val_data = []
    test_data = []

    # Split each style group
    for style, items in style_groups.items():
        random.shuffle(items)

        n_total = len(items)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    # Shuffle the splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_data)} images ({len(train_data)/len(metadata)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} images ({len(val_data)/len(metadata)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} images ({len(test_data)/len(metadata)*100:.1f}%)")

    return train_data, val_data, test_data


def organize_images(train_data, val_data, test_data, mappings):
    """
    Organize images into train/val/test directories.
    Also create CSV files with labels for easier loading.
    """
    print("\nOrganizing images into train/val/test directories...")

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Copy images
        for item in tqdm(split_data, desc=f"Copying {split_name}"):
            src = os.path.join(RAW_DIR, item['filename'])
            dst = os.path.join(split_dir, item['filename'])

            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Create labels CSV
        csv_path = os.path.join(PROCESSED_DIR, f"{split_name}_labels.csv")
        with open(csv_path, 'w') as f:
            f.write("filename,style,artist,style_idx,artist_idx\n")
            for item in split_data:
                style_idx = mappings['style_to_idx'][item['style']]
                artist_idx = mappings['artist_to_idx'][item['artist']]
                f.write(f"{item['filename']},{item['style']},{item['artist']},{style_idx},{artist_idx}\n")

        print(f"  {split_name}: {len(split_data)} images -> {split_dir}")
        print(f"  Labels saved to {csv_path}")


def print_statistics(train_data, val_data, test_data):
    """Print detailed statistics about the splits."""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    splits = {
        'Train': train_data,
        'Val': val_data,
        'Test': test_data
    }

    for split_name, split_data in splits.items():
        print(f"\n{split_name} Split:")

        # Style distribution
        style_counts = Counter(item['style'] for item in split_data)
        print(f"  Total images: {len(split_data)}")
        print(f"  Styles represented: {len(style_counts)}")

        # Artist distribution
        artist_counts = Counter(item['artist'] for item in split_data)
        print(f"  Artists represented: {len(artist_counts)}")

        # Top 5 styles
        print(f"  Top 5 styles:")
        for style, count in style_counts.most_common(5):
            print(f"    {style}: {count} images")


def create_dataset_summary():
    """Create a summary JSON with all dataset information."""
    metadata_file = os.path.join(METADATA_DIR, "dataset_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    mappings_file = os.path.join(METADATA_DIR, "label_mappings.json")
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    # Count images per split
    splits = {}
    for split_name in ['train', 'val', 'test']:
        csv_path = os.path.join(PROCESSED_DIR, f"{split_name}_labels.csv")
        with open(csv_path, 'r') as f:
            splits[split_name] = len(f.readlines()) - 1  # Subtract header

    summary = {
        'total_images': len(metadata),
        'num_styles': mappings['num_styles'],
        'num_artists': mappings['num_artists'],
        'splits': splits,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'random_seed': RANDOM_SEED
    }

    summary_file = os.path.join(METADATA_DIR, "dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset summary saved to {summary_file}")


def main():
    """Main execution function."""
    print("="*60)
    print("Data Preparation and Splitting")
    print("="*60)

    # Load metadata
    metadata, mappings = load_metadata()
    print(f"Loaded {len(metadata)} images")

    # Perform stratified split
    train_data, val_data, test_data = stratified_split(metadata, mappings)

    # Organize images
    organize_images(train_data, val_data, test_data, mappings)

    # Print statistics
    print_statistics(train_data, val_data, test_data)

    # Create summary
    create_dataset_summary()

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Visualize dataset: python scripts/visualize_dataset.py")
    print("  2. Train models: python train_models.py")


if __name__ == "__main__":
    main()
