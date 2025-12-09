
import torch
import torch.nn as nn
from torchvision import models
from .gram_utils import GramFeatureExtractor


class ResNetStyleClassifier(nn.Module):

    def __init__(self, num_styles, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNetArtistClassifier(nn.Module):

    def __init__(self, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_artists = num_artists

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNetMultiTaskClassifier(nn.Module):

    def __init__(self, num_styles, num_artists, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_styles = num_styles
        self.num_artists = num_artists

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )

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
        features = self.backbone(x)
        shared_features = self.shared_fc(features)

        style_logits = self.style_head(shared_features)
        artist_logits = self.artist_head(shared_features)

        return style_logits, artist_logits

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_embeddings(self, x):
        features = self.backbone(x)
        return self.shared_fc(features)


class ResNetGramStyleClassifier(nn.Module):

    def __init__(self, num_styles, pretrained=True, freeze_backbone=False,
                 gram_layer='layer3', gram_channels=512, fusion_weight=0.5):
        super().__init__()
        self.num_styles = num_styles
        self.fusion_weight = fusion_weight

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if gram_layer == 'layer3':
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
            self.backbone_continuation = nn.Sequential(
                backbone.layer4,
                backbone.avgpool
            )
            main_features = 2048
        elif gram_layer == 'layer4':
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

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.backbone_continuation.parameters():
                param.requires_grad = False

        self.gram_extractor = GramFeatureExtractor(
            in_channels=gram_in_channels,
            out_channels=gram_channels,
            use_reduction=True
        )
        gram_dim = self.gram_extractor.get_output_dim()

        self.main_head = nn.Sequential(
            nn.Linear(main_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_styles)
        )
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
        intermediate_features = self.feature_extractor(x)

        gram_features = self.gram_extractor(intermediate_features)

        final_features = self.backbone_continuation(intermediate_features)
        final_features = torch.flatten(final_features, 1)

        # Two prediction heads
        main_logits = self.main_head(final_features)
        gram_logits = self.gram_head(gram_features)

        # Fuse predictions (weighted combination)
        fused_logits = (1 - self.fusion_weight) * main_logits + self.fusion_weight * gram_logits

        return fused_logits

    def forward_with_aux(self, x):
        intermediate_features = self.feature_extractor(x)
        gram_features = self.gram_extractor(intermediate_features)
        final_features = self.backbone_continuation(intermediate_features)
        final_features = torch.flatten(final_features, 1)

        main_logits = self.main_head(final_features)
        gram_logits = self.gram_head(gram_features)
        fused_logits = (1 - self.fusion_weight) * main_logits + self.fusion_weight * gram_logits

        return main_logits, gram_logits, fused_logits

    def unfreeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.backbone_continuation.parameters():
            param.requires_grad = True
