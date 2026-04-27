import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class M1_MonocularCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.5):
        super(M1_MonocularCNN, self).__init__()

        # ── Backbone: ResNet-50 ───────────────────────────────────────

        if pretrained:
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # ── Global Average Pooling ────────────────────────────────────

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Dropout ───────────────────────────────────────────────────

        self.dropout = nn.Dropout(p=dropout)

        # ── Classification Head ───────────────────────────────────────

        self.classifier = nn.Linear(2048, num_classes)

        # ── Weight Initialisation ─────────────────────────────────────

        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x_left):
        # Feature extraction
        features = self.backbone(x_left)

        # Global average pooling
        pooled = self.gap(features)

        # Flatten
        pooled = pooled.flatten(1)

        # Dropout regularisation
        pooled = self.dropout(pooled)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_feature_map(self, x_left):
        return self.backbone(x_left)


def get_baseline_cnn(config=None):
    if config is None:
        config = {}

    return M1_MonocularCNN(
        num_classes=config.get('num_classes', 3),
        pretrained=config.get('pretrained', True),
        dropout=config.get('dropout', 0.5),
    )