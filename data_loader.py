import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DATA_DIR, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_data_transforms():
    return {
        'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    }

def get_dataloaders():
    data_transforms = get_data_transforms()
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}

    # Parameters kept on one line as requested
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    NUM_CLASSES = len(class_names)
    
    return dataloaders, image_datasets, NUM_CLASSES, class_names