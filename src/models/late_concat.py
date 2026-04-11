import torch
import torch.nn as nn
import torchvision.models as models


class LateConcatFusionCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, fc_hidden_dim=1024, dropout=0.5):
        super().__init__()

        self.left_backbone = models.resnet50(pretrained=pretrained)
        self.right_backbone = models.resnet50(pretrained=pretrained)

        feat_dim = self.left_backbone.fc.in_features
        self.left_backbone = nn.Sequential(*list(self.left_backbone.children())[:-1])
        self.right_backbone = nn.Sequential(*list(self.right_backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, left_x, right_x):
        left_feat = self.left_backbone(left_x)
        right_feat = self.right_backbone(right_x)
        fused = torch.cat([left_feat, right_feat], dim=1)
        return self.classifier(fused)


def get_late_concat_model(config=None):
    if config is None:
        config = {}

    return LateConcatFusionCNN(
        num_classes=config.get("num_classes", 3),
        pretrained=config.get("pretrained", True),
        fc_hidden_dim=config.get("fc_hidden_dim", 1024),
        dropout=config.get("dropout", 0.5),
    )
