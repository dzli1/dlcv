"""
Vision Transformer (ViT) models for art classification.
Includes single-task (style-only, artist-only) and multi-task variants.
"""

import torch
import torch.nn as nn
from torchvision import models


class ViTStyleClassifier(nn.Module):
    """Vision Transformer for style classification only."""

    def __init__(self, num_styles, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles

        # Load pretrained ViT-B/16
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_styles)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ViTArtistClassifier(nn.Module):
    """Vision Transformer for artist classification only."""

    def __init__(self, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_artists = num_artists

        # Load pretrained ViT-B/16
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_artists)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ViTMultiTaskClassifier(nn.Module):
    """Vision Transformer for multi-task classification (style + artist)."""

    def __init__(self, num_styles, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles
        self.num_artists = num_artists

        # Load pretrained ViT-B/16
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove classification head
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )

        # Task-specific heads
        self.style_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(384, num_styles)
        )

        self.artist_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(384, num_artists)
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
