"""
ResNet-based models for art classification.
Includes single-task (style-only, artist-only) and multi-task variants.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetStyleClassifier(nn.Module):
    """ResNet50 for style classification only."""

    def __init__(self, num_styles, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_styles)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNetArtistClassifier(nn.Module):
    """ResNet50 for artist classification only."""

    def __init__(self, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_artists = num_artists

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_artists)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNetMultiTaskClassifier(nn.Module):
    """ResNet50 for multi-task classification (style + artist)."""

    def __init__(self, num_styles, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles
        self.num_artists = num_artists

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Task-specific heads
        self.style_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_styles)
        )

        self.artist_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_artists)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        shared_features = self.shared_fc(features)

        # Task-specific predictions
        style_logits = self.style_head(shared_features)
        artist_logits = self.artist_head(shared_features)

        return style_logits, artist_logits

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_embeddings(self, x):
        """Extract shared embeddings for visualization."""
        features = self.backbone(x)
        return self.shared_fc(features)
