"""
ResNet-based models for art classification.
Includes single-task (style-only, artist-only) and multi-task variants.
"""

import torch
import torch.nn as nn
from torchvision import models
from .gram_utils import GramFeatureExtractor


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
            nn.Dropout(0.6),
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
            nn.Dropout(0.6),
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
            nn.Dropout(0.6)
        )

        # Task-specific heads
        self.style_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_styles)
        )

        self.artist_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
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


class ResNetGramStyleClassifier(nn.Module):
    """
    ResNet50 with Gram-matrix based style-aware auxiliary head.

    This architecture explicitly captures style statistics through Gram matrices
    computed from intermediate feature maps, fusing them with standard deep features
    for improved style classification.
    """

    def __init__(self, num_styles, pretrained=True, freeze_backbone=False,
                 gram_layer='layer3', gram_channels=512, fusion_weight=0.5):
        """
        Args:
            num_styles: Number of style classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone during training
            gram_layer: Which ResNet layer to extract Gram features from ('layer3' or 'layer4')
            gram_channels: Channel reduction for Gram computation (512 recommended)
            fusion_weight: Weight for combining main and Gram predictions (0.5 = equal weight)
        """
        super().__init__()
        self.num_styles = num_styles
        self.fusion_weight = fusion_weight

        # Load pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Extract layers up to the Gram extraction point
        if gram_layer == 'layer3':
            # Layer3 output: (B, 1024, H, W)
            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3
            )
            gram_in_channels = 1024
            # Continue with layer4 for main path
            self.backbone_continuation = nn.Sequential(
                backbone.layer4,
                backbone.avgpool
            )
            main_features = 2048
        elif gram_layer == 'layer4':
            # Layer4 output: (B, 2048, H, W)
            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            gram_in_channels = 2048
            self.backbone_continuation = backbone.avgpool
            main_features = 2048
        else:
            raise ValueError(f"gram_layer must be 'layer3' or 'layer4', got {gram_layer}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.backbone_continuation.parameters():
                param.requires_grad = False

        # Gram feature extractor with channel reduction
        self.gram_extractor = GramFeatureExtractor(
            in_channels=gram_in_channels,
            out_channels=gram_channels,
            use_reduction=True
        )
        gram_dim = self.gram_extractor.get_output_dim()

        # Main classification head (from final features)
        self.main_head = nn.Sequential(
            nn.Linear(main_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_styles)
        )

        # Gram-based auxiliary head (from style statistics)
        # Keep this lightweight since Gram features are large
        self.gram_head = nn.Sequential(
            nn.Linear(gram_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_styles)
        )

    def forward(self, x):
        """
        Forward pass with dual-head prediction.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            logits: Fused style predictions (B, num_styles)
        """
        # Extract intermediate features for Gram computation
        intermediate_features = self.feature_extractor(x)

        # Compute Gram-based style features
        gram_features = self.gram_extractor(intermediate_features)

        # Continue through backbone to get final features
        final_features = self.backbone_continuation(intermediate_features)
        final_features = torch.flatten(final_features, 1)

        # Two prediction heads
        main_logits = self.main_head(final_features)
        gram_logits = self.gram_head(gram_features)

        # Fuse predictions (weighted combination)
        fused_logits = (1 - self.fusion_weight) * main_logits + self.fusion_weight * gram_logits

        return fused_logits

    def forward_with_aux(self, x):
        """
        Forward pass returning both main and Gram predictions separately.
        Useful for training with auxiliary loss.

        Returns:
            main_logits: Predictions from main head
            gram_logits: Predictions from Gram head
            fused_logits: Weighted combination
        """
        intermediate_features = self.feature_extractor(x)
        gram_features = self.gram_extractor(intermediate_features)
        final_features = self.backbone_continuation(intermediate_features)
        final_features = torch.flatten(final_features, 1)

        main_logits = self.main_head(final_features)
        gram_logits = self.gram_head(gram_features)
        fused_logits = (1 - self.fusion_weight) * main_logits + self.fusion_weight * gram_logits

        return main_logits, gram_logits, fused_logits

    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.backbone_continuation.parameters():
            param.requires_grad = True
