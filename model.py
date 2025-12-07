import torch.nn as nn
from torchvision import models

def setup_model(num_classes, device, freeze_base=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = not freeze_base

    num_ftrs = model.fc.in_features
    # Defining the sequence on one line to avoid vertical splitting
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    
    return model.to(device)