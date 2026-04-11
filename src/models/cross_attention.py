import torch
import torch.nn as nn
import torchvision.models as models


class CrossAttentionFusionCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, attn_dim=256, num_heads=8,
                 fc_hidden_dim=512, dropout=0.5):
        super().__init__()

        left_resnet = models.resnet50(pretrained=pretrained)
        right_resnet = models.resnet50(pretrained=pretrained)

        self.left_encoder = nn.Sequential(*list(left_resnet.children())[:-2])
        self.right_encoder = nn.Sequential(*list(right_resnet.children())[:-2])

        self.left_proj = nn.Conv2d(2048, attn_dim, kernel_size=1)
        self.right_proj = nn.Conv2d(2048, attn_dim, kernel_size=1)

        self.cross_attn_l2r = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_r2l = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_l = nn.LayerNorm(attn_dim)
        self.norm_r = nn.LayerNorm(attn_dim)

        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def _flatten_hw(self, feat_map):
        b, c, h, w = feat_map.shape
        return feat_map.view(b, c, h * w).transpose(1, 2)

    def forward(self, left_x, right_x):
        left_feat = self.left_encoder(left_x)
        right_feat = self.right_encoder(right_x)

        left_tokens = self._flatten_hw(self.left_proj(left_feat))
        right_tokens = self._flatten_hw(self.right_proj(right_feat))

        left_ctx, _ = self.cross_attn_l2r(left_tokens, right_tokens, right_tokens)
        left_tokens = self.norm_l(left_tokens + left_ctx)

        right_ctx, _ = self.cross_attn_r2l(right_tokens, left_tokens, left_tokens)
        right_tokens = self.norm_r(right_tokens + right_ctx)

        left_vec = left_tokens.mean(dim=1)
        right_vec = right_tokens.mean(dim=1)

        fused = torch.cat([left_vec, right_vec], dim=1)
        return self.classifier(fused)


def get_cross_attention_model(config=None):
    if config is None:
        config = {}

    return CrossAttentionFusionCNN(
        num_classes=config.get("num_classes", 3),
        pretrained=config.get("pretrained", True),
        attn_dim=config.get("attn_dim", 256),
        num_heads=config.get("num_heads", 8),
        fc_hidden_dim=config.get("fc_hidden_dim", 512),
        dropout=config.get("dropout", 0.5),
    )
