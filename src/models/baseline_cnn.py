import torch
import torch.nn as nn
import torchvision.models as models


class BaselineMonocularCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, fc_hidden_dim=512, dropout=0.5):
        super(BaselineMonocularCNN, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

        self.num_classes = num_classes
        self.num_features = num_features

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        return self.backbone(x)


def get_baseline_cnn(config=None):
    if config is None:
        config = {}

    return BaselineMonocularCNN(
        num_classes=config.get("num_classes", 3),
        pretrained=config.get("pretrained", True),
        fc_hidden_dim=config.get("fc_hidden_dim", 512),
        dropout=config.get("dropout", 0.5),
    )
