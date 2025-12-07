import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from config import (
    ALLOW_ARTIST_OTHER_BUCKET,
    ARTIST_OTHER_LABEL,
    FIGURES_DIR,
    HF_CACHE_DIR,
    HF_DATASET_CONFIG,
    HF_DATASET_NAME,
    HF_DATASET_SPLIT,
    LABEL_MAPPING_FILE,
    MAX_YEAR,
    METADATA_PARQUET,
    MIN_ARTIST_SAMPLES,
    MIN_MOVEMENT_SAMPLES,
    MIN_RESOLUTION,
    MIN_YEAR,
    PRIMARY_GENRES,
    PRIMARY_MEDIUMS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SPLIT_METADATA_PARQUET,
    TARGET_NUM_ARTISTS,
    TARGET_NUM_MOVEMENTS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

try:
    import imagehash
except ImportError:  # pragma: no cover - optional dependency
    imagehash = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger("dataset_prep")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def extract_year(raw_date: Optional[str]) -> Optional[int]:
    if not raw_date:
        return None
    matches = re.findall(r"(1[4-9]\d{2}|20\d{2})", raw_date)
    if not matches:
        return None
    year = int(matches[0])
    return year


def compute_hash(image: Image.Image) -> Optional[str]:
    if imagehash is None:
        return None
    return str(imagehash.phash(image))


def load_wikiart_dataset(max_samples: Optional[int] = None, streaming: bool = False):
    LOGGER.info("Loading Hugging Face dataset %s (%s)", HF_DATASET_NAME, HF_DATASET_CONFIG)
    if streaming:
        LOGGER.info("Using streaming mode to avoid downloading full dataset")
        dataset = load_dataset(
            HF_DATASET_NAME,
            HF_DATASET_CONFIG,
            cache_dir=str(HF_CACHE_DIR),
            split=HF_DATASET_SPLIT,
            streaming=True
        )
        # For streaming, we'll limit in the loop instead
        return dataset
    else:
        dataset = load_dataset(
            HF_DATASET_NAME,
            HF_DATASET_CONFIG,
            cache_dir=str(HF_CACHE_DIR),
            split=HF_DATASET_SPLIT
        )
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset


def extract_metadata(dataset, export_images: bool, image_root: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
    records: List[Dict] = []
    unresolved_mediums = Counter()
    unresolved_genres = Counter()
    image_root.mkdir(parents=True, exist_ok=True)

    # Handle streaming vs regular datasets
    if hasattr(dataset, 'take'):  # Streaming dataset
        iterator = dataset.take(max_samples) if max_samples else dataset
    else:
        iterator = dataset
    
    for idx, row in enumerate(tqdm(iterator, desc="Extracting metadata", total=max_samples if max_samples else None)):
        if max_samples and idx >= max_samples:
            break
        
        # Handle both dict and row objects
        if hasattr(row, 'keys'):
            row_dict = row
        else:
            row_dict = dict(row) if hasattr(row, '__dict__') else row
        
        image: Image.Image = row_dict["image"]
        width, height = image.size
        shortest_edge = min(width, height)
        movement = str(row_dict.get("style") or row_dict.get("movement") or "").strip()
        artist = str(row_dict.get("artist") or "").strip()
        genre = str(row_dict.get("genre") or "").strip().lower()
        medium = str(row_dict.get("technique") or row_dict.get("materials") or "").strip().lower()
        year = extract_year(row_dict.get("date") or row_dict.get("year"))
        title = str(row_dict.get("title") or f"untitled_{row_dict.get('contentId', '')}").strip()

        if not movement or not artist:
            continue

        normalized_movement = movement.lower()
        normalized_artist = artist.strip()
        normalized_genre = genre or "unknown"
        normalized_medium = medium or "unknown"

        record_id = row_dict.get("contentId") or row_dict.get("painting_id") or slugify(f"{title}-{artist}-{year or 'na'}")
        phash = compute_hash(image)

        image_rel_path = None
        if export_images:
            movement_dir = image_root / slugify(normalized_movement)
            movement_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{slugify(artist)}-{record_id}.jpg"
            image_path = movement_dir / filename
            image.save(image_path, format="JPEG", quality=95)
            try:
                image_rel_path = image_path.relative_to(PROCESSED_DIR)
            except ValueError:
                image_rel_path = image_path
        else:
            image_path = row.get("image_path")
            image_rel_path = image_path

        record = {
            "record_id": record_id,
            "image_path": str(image_rel_path) if image_rel_path else None,
            "movement": normalized_movement,
            "artist": normalized_artist,
            "genre": normalized_genre,
            "medium": normalized_medium,
            "title": title,
            "year": year,
            "width": width,
            "height": height,
            "shortest_edge": shortest_edge,
            "phash": phash,
        }
        if record["image_path"] is None:
            LOGGER.warning("Skipping record %s because image_path is missing (use export).", record_id)
            continue
        records.append(record)

        if normalized_medium == "unknown":
            unresolved_mediums[medium] += 1
        if normalized_genre == "unknown":
            unresolved_genres[genre] += 1

    LOGGER.info("Extracted %d candidate records", len(records))
    return pd.DataFrame(records)


def filter_by_year(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    filtered = df[
        (df["year"].between(MIN_YEAR, MAX_YEAR, inclusive="both"))
        | df["year"].isna()
    ]
    LOGGER.info("Year filter dropped %d rows", before - len(filtered))
    return filtered


def filter_by_resolution(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    filtered = df[df["shortest_edge"] >= MIN_RESOLUTION]
    LOGGER.info("Resolution filter dropped %d rows", before - len(filtered))
    return filtered


def select_top_categories(series: pd.Series, min_count: int, target: int) -> List[str]:
    counts = series.value_counts()
    filtered = counts[counts >= min_count]
    if len(filtered) < target:
        LOGGER.warning(
            "Found only %d categories with >= %d samples; using top %d categories instead",
            len(filtered),
            min_count,
            target,
        )
        filtered = counts.head(target)
    else:
        filtered = filtered.head(target)
    return filtered.index.tolist()


def filter_movements(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    top_movements = select_top_categories(df["movement"], MIN_MOVEMENT_SAMPLES, TARGET_NUM_MOVEMENTS)
    filtered = df[df["movement"].isin(top_movements)]
    LOGGER.info("Selected %d movements", len(top_movements))
    return filtered, top_movements


def filter_artists(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    top_artists = select_top_categories(df["artist"], MIN_ARTIST_SAMPLES, TARGET_NUM_ARTISTS)
    df = df.copy()
    if ALLOW_ARTIST_OTHER_BUCKET:
        df["artist"] = df["artist"].apply(lambda x: x if x in top_artists else ARTIST_OTHER_LABEL)
        if ARTIST_OTHER_LABEL not in top_artists:
            top_artists.append(ARTIST_OTHER_LABEL)
    else:
        df = df[df["artist"].isin(top_artists)]
    LOGGER.info("Selected %d artists (other bucket: %s)", len(top_artists), ALLOW_ARTIST_OTHER_BUCKET)
    return df, top_artists


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "phash" not in df or df["phash"].isna().all():
        LOGGER.warning("Perceptual hash unavailable; skipping duplicate removal")
        return df
    before = len(df)
    df = df.drop_duplicates(subset=["phash"])
    LOGGER.info("Removed %d duplicate records via perceptual hash", before - len(df))
    return df


def apply_medium_genre_filters(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    before = len(df)
    if not strict:
        # Relaxed filtering for small samples - just skip obviously bad data
        LOGGER.info("Using relaxed medium/genre filtering")
        return df
    
    def medium_ok(value: str) -> bool:
        if not isinstance(value, str) or value == "unknown":
            return True
        return any(token in value for token in PRIMARY_MEDIUMS)

    df = df[
        df["medium"].apply(medium_ok)
        & (df["genre"].isin(PRIMARY_GENRES) | df["genre"] == "unknown")
    ]
    LOGGER.info("Medium/genre filter dropped %d rows", before - len(df))
    return df


def build_label_mappings(movements: List[str], artists: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    movement_to_idx = {movement: idx for idx, movement in enumerate(sorted(movements))}
    artist_to_idx = {artist: idx for idx, artist in enumerate(sorted(artists))}
    return movement_to_idx, artist_to_idx


def assign_labels(df: pd.DataFrame, movement_map: Dict[str, int], artist_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["movement_id"] = df["movement"].map(movement_map)
    df["artist_id"] = df["artist"].map(artist_map)
    df = df.dropna(subset=["movement_id", "artist_id"])
    df["movement_id"] = df["movement_id"].astype(int)
    df["artist_id"] = df["artist_id"].astype(int)
    return df


def stratified_group_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df["artist"]))
    df.loc[df.index[train_idx], "split"] = "train"

    temp_df = df.loc[df.index[temp_idx]]
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=val_ratio_adjusted, random_state=RANDOM_SEED + 1)
    val_idx_temp, test_idx_temp = next(gss_val.split(temp_df, groups=temp_df["artist"]))
    df.loc[temp_df.index[val_idx_temp], "split"] = "val"
    df.loc[temp_df.index[test_idx_temp], "split"] = "test"
    return df


def save_artifacts(df: pd.DataFrame, movement_map: Dict[str, int], artist_map: Dict[str, int]) -> None:
    df.to_parquet(METADATA_PARQUET, index=False)
    df.to_parquet(SPLIT_METADATA_PARQUET, index=False)
    stats = {
        "movements": movement_map,
        "artists": artist_map,
        "samples_per_split": df["split"].value_counts().to_dict(),
    }
    with open(LABEL_MAPPING_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    LOGGER.info("Saved metadata to %s", METADATA_PARQUET)


def visualize_distribution(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plots = [
        ("movement_distribution.png", "Movement Distribution", "movement"),
        ("artist_distribution.png", "Top 25 Artists", "artist"),
    ]

    for filename, title, column in plots:
        plt.figure(figsize=(12, 6))
        if column == "artist":
            df[column].value_counts().head(25).plot(kind="bar")
        else:
            df[column].value_counts().plot(kind="bar")
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_path = FIGURES_DIR / filename
        plt.savefig(output_path, dpi=200)
        plt.close()
        LOGGER.info("Saved visualization %s", output_path)

    plt.figure(figsize=(8, 5))
    year_data = pd.to_numeric(df["year"], errors='coerce').dropna()
    if len(year_data) > 0:
        year_data.plot(kind="hist", bins=30)
    plt.title("Year Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "year_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["shortest_edge"].plot(kind="hist", bins=30)
    plt.title("Resolution (shortest edge) Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "resolution_distribution.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiArt dataset splits and metadata.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset for debugging.")
    parser.add_argument("--skip-export", action="store_true", help="Do not export images (assumes image_path already populated).")
    parser.add_argument("--image-root", type=Path, default=PROCESSED_DIR / "images", help="Directory to export filtered images.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode to avoid downloading full dataset (requires --max-samples).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.streaming and not args.max_samples:
        LOGGER.warning("Streaming mode requires --max-samples. Setting to 5000.")
        args.max_samples = 5000
    
    # Use relaxed filtering for small samples
    use_strict_filter = args.max_samples is None or args.max_samples >= 10000
    
    dataset = load_wikiart_dataset(args.max_samples, streaming=args.streaming)
    df = extract_metadata(dataset, export_images=not args.skip_export, image_root=args.image_root, max_samples=args.max_samples)
    df = filter_by_year(df)
    df = filter_by_resolution(df)
    df = apply_medium_genre_filters(df, strict=use_strict_filter)
    df = remove_duplicates(df)
    df, movements = filter_movements(df)
    df, artists = filter_artists(df)
    movement_map, artist_map = build_label_mappings(movements, artists)
    df = assign_labels(df, movement_map, artist_map)
    df = stratified_group_split(df)
    save_artifacts(df, movement_map, artist_map)
    visualize_distribution(df)
    LOGGER.info("Finished preparing dataset. Samples per split:\n%s", df["split"].value_counts())


if __name__ == "__main__":
    main()
