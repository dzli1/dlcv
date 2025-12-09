"""
Dataset and DataLoader utilities for art classification.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ArtDataset(Dataset):
    """Custom dataset for art images with style and artist labels."""

    def __init__(self, data_dir, labels_csv, transform=None):
        """
        Args:
            data_dir: Directory containing images
            labels_csv: Path to CSV file with labels
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]

        # Load image
        img_path = os.path.join(self.data_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        style_label = int(row['style_idx'])
        artist_label = int(row['artist_idx'])

        return {
            'image': image,
            'style_label': style_label,
            'artist_label': artist_label,
            'filename': row['filename']
        }


def get_transforms(split='train', image_size=224):
    """
    Get appropriate transforms for train/val/test splits.

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size (default: 224 for ResNet/ViT)

    Returns:
        torchvision.transforms composition
    """
    # ImageNet normalization (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        # Data augmentation for training (stronger to prevent overfitting)
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Gentler
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),
            transforms.RandomRotation(degrees=3),  # Very small
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))  # Much lower
        ])
    else:
        # No augmentation for val/test
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(data_dir='./data/processed', batch_size=32, num_workers=4, image_size=224):
    """
    Create DataLoaders for train, val, and test sets.

    Args:
        data_dir: Base directory containing processed data
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        image_size: Target image size

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders
        dict: Dictionary containing dataset metadata
    """
    dataloaders = {}
    datasets = {}

    for split in ['train', 'val', 'test']:
        # Paths
        split_dir = os.path.join(data_dir, split)
        labels_csv = os.path.join(data_dir, f'{split}_labels.csv')

        # Create dataset
        transform = get_transforms(split, image_size)
        dataset = ArtDataset(split_dir, labels_csv, transform)
        datasets[split] = dataset

        # Create dataloader
        shuffle = (split == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,  # Faster transfer to GPU
            persistent_workers=(num_workers > 0)  # Keep workers alive
        )
        dataloaders[split] = dataloader

    # Get metadata
    metadata = {
        'num_samples': {split: len(datasets[split]) for split in ['train', 'val', 'test']},
        'batch_size': batch_size,
        'image_size': image_size
    }

    return dataloaders, metadata
