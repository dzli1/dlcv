"""
Download WikiArt dataset from Hugging Face and prepare for training.
Filters to top 10-12 styles and top 50-100 artists based on image counts.
"""

import os
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil

# Configuration
DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# Filtering parameters
TARGET_STYLES = 12  # Top N styles
TARGET_ARTISTS = 100  # Top N artists
MIN_IMAGES_PER_STYLE = 500  # Minimum images to consider a style
MIN_IMAGES_PER_ARTIST = 50  # Minimum images to consider an artist


def download_wikiart():
    """Download WikiArt dataset from Hugging Face."""
    print("Downloading WikiArt dataset from Hugging Face...")
    print("This may take a while depending on your connection...")

    # Try different WikiArt datasets
    dataset_names = ["huggan/wikiart", "Artificio/WikiArt"]

    dataset = None
    for name in dataset_names:
        try:
            print(f"Trying to load: {name}")
            dataset = load_dataset(name, split="train", trust_remote_code=True)
            print(f"Successfully loaded {name}")
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue

    if dataset is None:
        raise RuntimeError("Failed to load any WikiArt dataset. Please check Hugging Face availability.")

    return dataset


def analyze_dataset(dataset):
    """Analyze dataset to find top styles and artists."""
    print("\nAnalyzing dataset distribution...")

    # Count styles and artists
    style_counts = Counter()
    artist_counts = Counter()

    for item in tqdm(dataset, desc="Counting"):
        style = item.get('style', item.get('genre', 'unknown'))
        artist = item.get('artist', 'unknown')

        style_counts[style] += 1
        artist_counts[artist] += 1

    # Filter and select top styles
    valid_styles = {style: count for style, count in style_counts.items()
                    if count >= MIN_IMAGES_PER_STYLE}
    top_styles = sorted(valid_styles.items(), key=lambda x: x[1], reverse=True)[:TARGET_STYLES]

    # Filter and select top artists
    valid_artists = {artist: count for artist, count in artist_counts.items()
                     if count >= MIN_IMAGES_PER_ARTIST}
    top_artists = sorted(valid_artists.items(), key=lambda x: x[1], reverse=True)[:TARGET_ARTISTS]

    print(f"\nSelected {len(top_styles)} styles:")
    for style, count in top_styles:
        print(f"  {style}: {count} images")

    print(f"\nSelected {len(top_artists)} artists:")
    for i, (artist, count) in enumerate(top_artists[:10], 1):
        print(f"  {i}. {artist}: {count} images")
    print(f"  ... and {len(top_artists) - 10} more artists")

    return dict(top_styles), dict(top_artists)


def save_filtered_dataset(dataset, selected_styles, selected_artists):
    """Save filtered images and create metadata."""
    print("\nSaving filtered dataset...")

    # Create directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Prepare metadata
    metadata = []
    saved_count = 0

    for idx, item in enumerate(tqdm(dataset, desc="Processing images")):
        # Extract fields (handle different dataset formats)
        style = item.get('style', item.get('genre', 'unknown'))
        artist = item.get('artist', 'unknown')
        image = item.get('image', None)

        # Skip if not in selected styles or artists
        if style not in selected_styles or artist not in selected_artists:
            continue

        if image is None:
            continue

        try:
            # Save image
            filename = f"img_{saved_count:06d}.jpg"
            filepath = os.path.join(RAW_DIR, filename)

            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image.save(filepath, 'JPEG', quality=95)

            # Add metadata
            metadata.append({
                'filename': filename,
                'style': style,
                'artist': artist,
                'original_idx': idx
            })

            saved_count += 1

        except Exception as e:
            print(f"\nError saving image {idx}: {e}")
            continue

    # Save metadata
    metadata_file = os.path.join(METADATA_DIR, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {saved_count} images to {RAW_DIR}")
    print(f"Metadata saved to {metadata_file}")

    return metadata


def create_label_mappings(metadata):
    """Create and save label mappings for styles and artists."""
    print("\nCreating label mappings...")

    # Get unique styles and artists
    styles = sorted(set(item['style'] for item in metadata))
    artists = sorted(set(item['artist'] for item in metadata))

    # Create mappings
    style_to_idx = {style: idx for idx, style in enumerate(styles)}
    artist_to_idx = {artist: idx for idx, artist in enumerate(artists)}

    idx_to_style = {idx: style for style, idx in style_to_idx.items()}
    idx_to_artist = {idx: artist for artist, idx in artist_to_idx.items()}

    # Save mappings
    mappings = {
        'style_to_idx': style_to_idx,
        'artist_to_idx': artist_to_idx,
        'idx_to_style': idx_to_style,
        'idx_to_artist': idx_to_artist,
        'num_styles': len(styles),
        'num_artists': len(artists)
    }

    mappings_file = os.path.join(METADATA_DIR, "label_mappings.json")
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=2)

    print(f"Label mappings saved to {mappings_file}")
    print(f"  Styles: {len(styles)}")
    print(f"  Artists: {len(artists)}")

    return mappings


def main():
    """Main execution function."""
    print("="*60)
    print("WikiArt Dataset Download and Preparation")
    print("="*60)

    # Download dataset
    dataset = download_wikiart()
    print(f"\nTotal images in dataset: {len(dataset)}")

    # Analyze and filter
    selected_styles, selected_artists = analyze_dataset(dataset)

    # Save filtered dataset
    metadata = save_filtered_dataset(dataset, selected_styles, selected_artists)

    # Create label mappings
    mappings = create_label_mappings(metadata)

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"Images saved: {len(metadata)}")
    print(f"Styles: {mappings['num_styles']}")
    print(f"Artists: {mappings['num_artists']}")
    print(f"\nNext step: Run data preprocessing and splitting")
    print("  python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
