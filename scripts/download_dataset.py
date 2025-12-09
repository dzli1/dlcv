
import os
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil

DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

TARGET_STYLES = 12  
TARGET_ARTISTS = 100  
MIN_IMAGES_PER_STYLE = 500  
MIN_IMAGES_PER_ARTIST = 50  


def download_wikiart():

    dataset_names = ["huggan/wikiart", "Artificio/WikiArt"]

    dataset = None
    for name in dataset_names:
        try:
            dataset = load_dataset(name, split="train", trust_remote_code=True)
            break
        except Exception as e:
            continue

    if dataset is None:
        raise RuntimeError("Failed to load any WikiArt dataset. Please check Hugging Face availability.")

    return dataset


def analyze_dataset(dataset):

    style_counts = Counter()
    artist_counts = Counter()

    for item in tqdm(dataset, desc="Counting"):
        style = item.get('style', item.get('genre', 'unknown'))
        artist = item.get('artist', 'unknown')

        style_counts[style] += 1
        artist_counts[artist] += 1

    valid_styles = {style: count for style, count in style_counts.items()
                    if count >= MIN_IMAGES_PER_STYLE}
    top_styles = sorted(valid_styles.items(), key=lambda x: x[1], reverse=True)[:TARGET_STYLES]

    valid_artists = {artist: count for artist, count in artist_counts.items()
                     if count >= MIN_IMAGES_PER_ARTIST}
    top_artists = sorted(valid_artists.items(), key=lambda x: x[1], reverse=True)[:TARGET_ARTISTS]

    return dict(top_styles), dict(top_artists)


def save_filtered_dataset(dataset, selected_styles, selected_artists):

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    metadata = []
    saved_count = 0

    for idx, item in enumerate(tqdm(dataset, desc="Processing images")):
        style = item.get('style', item.get('genre', 'unknown'))
        artist = item.get('artist', 'unknown')
        image = item.get('image', None)

        if style not in selected_styles or artist not in selected_artists:
            continue

        if image is None:
            continue

        try:
            filename = f"img_{saved_count:06d}.jpg"
            filepath = os.path.join(RAW_DIR, filename)

            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image.save(filepath, 'JPEG', quality=95)

            metadata.append({
                'filename': filename,
                'style': style,
                'artist': artist,
                'original_idx': idx
            })

            saved_count += 1

        except Exception as e:
            continue

    metadata_file = os.path.join(METADATA_DIR, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def create_label_mappings(metadata):

    styles = sorted(set(item['style'] for item in metadata))
    artists = sorted(set(item['artist'] for item in metadata))

    style_to_idx = {style: idx for idx, style in enumerate(styles)}
    artist_to_idx = {artist: idx for idx, artist in enumerate(artists)}

    idx_to_style = {idx: style for style, idx in style_to_idx.items()}
    idx_to_artist = {idx: artist for artist, idx in artist_to_idx.items()}

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


    return mappings


def main():

    dataset = download_wikiart()

    selected_styles, selected_artists = analyze_dataset(dataset)

    metadata = save_filtered_dataset(dataset, selected_styles, selected_artists)

    mappings = create_label_mappings(metadata)



if __name__ == "__main__":
    main()
