import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class M2_LateConcatCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True,
                 fc_hidden_dim=1024, dropout=0.5):
        super(M2_LateConcatCNN, self).__init__()

        # ── Two Independent ResNet-50 Streams ────────────────────────
       
        if pretrained:
            left_resnet  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            right_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            left_resnet  = models.resnet50(weights=None)
            right_resnet = models.resnet50(weights=None)

        # Remove GAP and FC from both backbone
        # Output: [B, 2048, 15, 15] per stream
        self.stream_L = nn.Sequential(*list(left_resnet.children())[:-2])
        self.stream_R = nn.Sequential(*list(right_resnet.children())[:-2])

        # ── Global Average Pooling ────────────────────────────────────

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classification Head ───────────────────────────────────────

        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, fc_hidden_dim),   # 4096 → 1024
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden_dim, num_classes)  # 1024 → 3
        )

        # ── Weight Initialisation ─────────────────────────────────────
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_left, x_right):
        """
        Forward pass for M2 late-concatenation model.

        Args:
            x_left:  torch.Tensor [B, 3, 480, 480]  Left eye image
            x_right: torch.Tensor [B, 3, 480, 480]  Right eye image
                     (horizontal flip of left in synthetic pairing)

        Returns:
            logits: torch.Tensor [B, 3]
        """
        # ── Independent Feature Extraction ───────────────────────────

        # Left stream
        feat_L = self.stream_L(x_left)
        # Right stream
        feat_R = self.stream_R(x_right)

        # ── Global Average Pooling ────────────────────────────────────

        f_L = self.gap(feat_L).flatten(1)
        f_R = self.gap(feat_R).flatten(1)

        # ── Late Concatenation ────────────────────────────────────────
        
        joint = torch.cat([f_L, f_R], dim=1)

        # ── Classification ────────────────────────────────────────────

        logits = self.classifier(joint)

        return logits

    def get_feature_maps(self, x_left, x_right):
        return self.stream_L(x_left), self.stream_R(x_right)


def get_late_concat_model(config=None):
    if config is None:
        config = {}

    return M2_LateConcatCNN(
        num_classes=config.get('num_classes', 3),
        pretrained=config.get('pretrained', True),
        fc_hidden_dim=config.get('fc_hidden_dim', 1024),
        dropout=config.get('dropout', 0.5),
    )