import os
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm

DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42


def load_metadata():
    metadata_file = os.path.join(METADATA_DIR, "dataset_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    mappings_file = os.path.join(METADATA_DIR, "label_mappings.json")
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    return metadata, mappings


def stratified_split(metadata, mappings):

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

    return train_data, val_data, test_data


def organize_images(train_data, val_data, test_data, mappings):

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



def print_statistics(train_data, val_data, test_data):
    pass

def create_dataset_summary():
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



def main():

    # Load metadata
    metadata, mappings = load_metadata()

    # Perform stratified split
    train_data, val_data, test_data = stratified_split(metadata, mappings)

    # Organize images
    organize_images(train_data, val_data, test_data, mappings)

    # Create summary
    create_dataset_summary()


if __name__ == "__main__":
    main()
