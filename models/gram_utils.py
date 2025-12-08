"""
Gram matrix utilities for style-aware classification.
"""

import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    """
    Compute Gram matrix from feature maps.

    Gram matrix captures style statistics (texture correlations) by computing
    the inner product between feature maps, commonly used in neural style transfer.
    """

    def __init__(self, normalize=True):
        """
        Args:
            normalize: Whether to normalize by spatial dimensions (H*W)
        """
        super().__init__()
        self.normalize = normalize

    def forward(self, features):
        """
        Compute Gram matrix.

        Args:
            features: Tensor of shape (B, C, H, W)

        Returns:
            gram: Tensor of shape (B, C, C)
        """
        B, C, H, W = features.size()

        # Reshape: B × C × H × W → B × C × (H*W)
        features_flat = features.view(B, C, H * W)

        # Gram matrix: (B × C × HW) @ (B × HW × C) = B × C × C
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))

        # Normalize by spatial size
        if self.normalize:
            gram = gram / (H * W)

        return gram


class GramFeatureExtractor(nn.Module):
    """
    Extract Gram-based style features from intermediate layer.
    Includes optional channel reduction for efficiency.
    """

    def __init__(self, in_channels, out_channels=512, use_reduction=True):
        """
        Args:
            in_channels: Number of input channels from ResNet layer
            out_channels: Reduced channel count (if use_reduction=True)
            use_reduction: Whether to reduce channels before computing Gram
        """
        super().__init__()
        self.use_reduction = use_reduction

        if use_reduction:
            # 1×1 conv for channel reduction
            self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.gram_channels = out_channels
        else:
            self.channel_reduction = nn.Identity()
            self.gram_channels = in_channels

        # Gram matrix computation
        self.gram = GramMatrix(normalize=True)

        # Output dimension is flattened upper triangle of C×C matrix
        # For simplicity, we'll flatten the entire matrix
        self.gram_dim = self.gram_channels * self.gram_channels

    def forward(self, features):
        """
        Extract Gram-based features.

        Args:
            features: Tensor of shape (B, C_in, H, W)

        Returns:
            gram_features: Flattened Gram matrix of shape (B, C_out^2)
        """
        # Optional channel reduction
        features = self.channel_reduction(features)

        # Compute Gram matrix: B × C × C
        gram = self.gram(features)

        # Flatten: B × C × C → B × (C^2)
        B = gram.size(0)
        gram_features = gram.view(B, -1)

        return gram_features

    def get_output_dim(self):
        """Return the dimensionality of Gram features."""
        return self.gram_dim


class CompactGramFeatureExtractor(nn.Module):
    """
    More compact version: uses only upper triangle of Gram matrix.
    Reduces feature dimension from C^2 to C*(C+1)/2.
    """

    def __init__(self, in_channels, out_channels=512, use_reduction=True):
        super().__init__()
        self.use_reduction = use_reduction

        if use_reduction:
            self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.gram_channels = out_channels
        else:
            self.channel_reduction = nn.Identity()
            self.gram_channels = in_channels

        self.gram = GramMatrix(normalize=True)

        # Upper triangle dimension: C*(C+1)/2
        self.gram_dim = (self.gram_channels * (self.gram_channels + 1)) // 2

    def forward(self, features):
        """Extract compact Gram features (upper triangle only)."""
        features = self.channel_reduction(features)
        gram = self.gram(features)

        # Extract upper triangle (including diagonal)
        B, C, _ = gram.size()

        # Create mask for upper triangle
        mask = torch.triu(torch.ones(C, C, device=gram.device, dtype=torch.bool))

        # Extract and flatten upper triangle
        gram_features = []
        for b in range(B):
            upper_tri = gram[b][mask]
            gram_features.append(upper_tri)

        gram_features = torch.stack(gram_features, dim=0)

        return gram_features

    def get_output_dim(self):
        """Return the dimensionality of compact Gram features."""
        return self.gram_dim
