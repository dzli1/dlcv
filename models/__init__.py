"""
Model architectures for art classification.
"""

from .resnet_models import (
    ResNetStyleClassifier,
    ResNetArtistClassifier,
    ResNetMultiTaskClassifier,
    ResNetGramStyleClassifier
)
from .vit_models import ViTStyleClassifier, ViTArtistClassifier, ViTMultiTaskClassifier

__all__ = [
    'ResNetStyleClassifier',
    'ResNetArtistClassifier',
    'ResNetMultiTaskClassifier',
    'ResNetGramStyleClassifier',
    'ViTStyleClassifier',
    'ViTArtistClassifier',
    'ViTMultiTaskClassifier',
]
