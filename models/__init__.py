"""
Model architectures for art classification.
"""

from .resnet_models import ResNetStyleClassifier, ResNetArtistClassifier, ResNetMultiTaskClassifier
from .vit_models import ViTStyleClassifier, ViTArtistClassifier, ViTMultiTaskClassifier

__all__ = [
    'ResNetStyleClassifier',
    'ResNetArtistClassifier',
    'ResNetMultiTaskClassifier',
    'ViTStyleClassifier',
    'ViTArtistClassifier',
    'ViTMultiTaskClassifier',
]
