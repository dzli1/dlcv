import torch.nn as nn
from torchvision import models

def setup_model(num_classes, device, freeze_base=True, arch='resnet50'):
    """
    Setup model with support for multiple architectures.
    
    Args:
        num_classes: Number of output classes
        device: torch device
        freeze_base: Whether to freeze backbone weights
        arch: Architecture name ('resnet50', 'efficientnet_b0', 'vgg16_bn', 'vit_b_16')
    
    Returns:
        Model ready for training
    """
    if arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        for param in model.parameters():
            param.requires_grad = not freeze_base
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        for param in model.parameters():
            param.requires_grad = not freeze_base
        
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        
        for param in model.parameters():
            param.requires_grad = not freeze_base
        
        # VGG16 classifier is a Sequential with 7 layers
        # Replace the last layer (index 6) with our custom classifier
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    elif arch == 'vit_b_16':
        # Vision Transformer B/16
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze backbone if requested
        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace classification head
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )
        
        # Unfreeze head parameters
        for param in model.heads.head.parameters():
            param.requires_grad = True
        
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from 'resnet50', 'efficientnet_b0', 'vgg16_bn', 'vit_b_16'")
    
    return model.to(device)