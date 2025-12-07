import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE_CNN,
    LABEL_MAPPING_FILE,
    NUM_WORKERS,
    PIN_MEMORY,
    PROCESSED_DIR,
    SPLIT_METADATA_PARQUET,
    VAL_BATCH_SIZE,
)


def load_label_mappings() -> Tuple[Dict[int, str], Dict[int, str]]:
    with open(LABEL_MAPPING_FILE) as f:
        mappings = json.load(f)
    movements = mappings["movements"]
    artists = mappings["artists"]
    inv_movements = {int(idx): movement for movement, idx in movements.items()}
    inv_artists = {int(idx): artist for artist, idx in artists.items()}
    return inv_movements, inv_artists


class WikiArtDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, split: str, transform=None, image_root: Path = PROCESSED_DIR):
        self.df = metadata[metadata["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        movement = torch.tensor(row["movement_id"], dtype=torch.long)
        artist = torch.tensor(row["artist_id"], dtype=torch.long)
        return image, movement, artist, row["record_id"]


def build_transforms(image_size: int = IMAGE_SIZE_CNN):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def _weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    movement_counts = df["movement_id"].value_counts().to_dict()
    artist_counts = df["artist_id"].value_counts().to_dict()

    weights = []
    for _, row in df.iterrows():
        movement_weight = 1.0 / movement_counts[row["movement_id"]]
        artist_weight = 1.0 / artist_counts[row["artist_id"]]
        weights.append(0.5 * movement_weight + 0.5 * artist_weight)

    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def get_dataloaders(
    metadata_path: Path = SPLIT_METADATA_PARQUET,
    image_size: int = IMAGE_SIZE_CNN,
    batch_size: int = BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    use_weighted_sampler: bool = True,
):
    metadata = pd.read_parquet(metadata_path)
    train_transform, eval_transform = build_transforms(image_size)

    datasets = {
        split: WikiArtDataset(metadata, split=split, transform=train_transform if split == "train" else eval_transform)
        for split in ["train", "val", "test"]
    }

    sampler = _weighted_sampler(datasets["train"].df) if use_weighted_sampler else None

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
    }

    movement_map, artist_map = load_label_mappings()
    dataset_info = {
        "movement_map": movement_map,
        "artist_map": artist_map,
        "num_movements": len(movement_map),
        "num_artists": len(artist_map),
    }
    return dataloaders, datasets, dataset_info
