
import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, features):
        B, C, H, W = features.size()

        features_flat = features.view(B, C, H * W)

        # Gram matrix
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))

        if self.normalize:
            gram = gram / (H * W)

        return gram


class GramFeatureExtractor(nn.Module):

    def __init__(self, in_channels, out_channels=512, use_reduction=True):
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
        features = self.channel_reduction(features)

        gram = self.gram(features)
        B = gram.size(0)
        gram_features = gram.view(B, -1)

        return gram_features

    def get_output_dim(self):
        return self.gram_dim


class CompactGramFeatureExtractor(nn.Module):

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

        self.gram_dim = (self.gram_channels * (self.gram_channels + 1)) // 2

    def forward(self, features):
        features = self.channel_reduction(features)
        gram = self.gram(features)

        B, C, _ = gram.size()

        mask = torch.triu(torch.ones(C, C, device=gram.device, dtype=torch.bool))

        gram_features = []
        for b in range(B):
            upper_tri = gram[b][mask]
            gram_features.append(upper_tri)

        gram_features = torch.stack(gram_features, dim=0)

        return gram_features

    def get_output_dim(self):
        return self.gram_dim
