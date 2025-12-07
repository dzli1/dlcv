from typing import Tuple

import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def _build_resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_resnet50(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_timm_model(arch: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if timm is None:
        raise ImportError("timm is required for architecture {}".format(arch))
    model = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
    feature_dim = model.num_features
    return model, feature_dim


def build_backbone(arch: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    if arch == "resnet18":
        return _build_resnet18(pretrained)
    if arch == "resnet50":
        return _build_resnet50(pretrained)
    return _build_timm_model(arch, pretrained)


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        arch: str,
        num_movements: int,
        num_artists: int,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone, feature_dim = build_backbone(arch, pretrained)
        hidden_dim = max(512, feature_dim // 2)
        self.movement_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_movements),
        )
        self.artist_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_artists),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.mean(dim=[2, 3])
        movement_logits = self.movement_head(features)
        artist_logits = self.artist_head(features)
        return {
            "features": features,
            "movement_logits": movement_logits,
            "artist_logits": artist_logits,
        }
