import os
import random
import torch
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import DATA_DIR, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_data_transforms():
    return {
        'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        'test': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    }

class TransformDataset:
    """Wrapper to apply different transforms to a subset of a dataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def get_dataloaders(train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
    """
    Load data from artset folder and split into train/val/test sets with stratification.
    Ensures each art category contributes proportionally to each split.
    
    Args:
        train_split: Proportion of data to use for training (default: 0.7)
        val_split: Proportion of data to use for validation (default: 0.15)
        test_split: Proportion of data to use for testing (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        dataloaders: Dictionary with 'train', 'val', 'test' DataLoaders
        image_datasets: Dictionary with 'train', 'val', 'test' datasets
        NUM_CLASSES: Number of classes
        class_names: List of class names
    """
    # Validate splits sum to 1.0
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    data_transforms = get_data_transforms()
    
    # Load full dataset from artset (each class folder is a class)
    # Use no transform initially, we'll apply transforms after splitting
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    
    # Get class names
    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    
    # Group indices by class for stratified splitting
    # Use samples attribute instead of iterating (much faster - doesn't load images)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for class_idx in class_indices:
        random.shuffle(class_indices[class_idx])
    
    # Split each class proportionally
    train_indices = []
    val_indices = []
    test_indices = []
    
    print(f"\nPerforming stratified split across {NUM_CLASSES} classes:")
    print(f"  Train: {train_split*100:.1f}%, Val: {val_split*100:.1f}%, Test: {test_split*100:.1f}%\n")
    
    for class_idx, class_name in enumerate(class_names):
        indices = class_indices[class_idx]
        n_total = len(indices)
        
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val  # Remaining goes to test
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
        
        print(f"  {class_name:30s}: Train={n_train:4d}, Val={n_val:4d}, Test={n_test:4d} (Total={n_total:4d})")
    
    # Shuffle the final splits
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    print(f"\nTotal split sizes:")
    print(f"  Train: {len(train_indices)} images ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"  Val:   {len(val_indices)} images ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} images ({len(test_indices)/len(full_dataset)*100:.1f}%)\n")
    
    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    # Apply appropriate transforms to each subset
    train_dataset = TransformDataset(train_subset, data_transforms['train'])
    val_dataset = TransformDataset(val_subset, data_transforms['val'])
    test_dataset = TransformDataset(test_subset, data_transforms['test'])
    
    # Create datasets dict for compatibility
    image_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    
    return dataloaders, image_datasets, NUM_CLASSES, class_names