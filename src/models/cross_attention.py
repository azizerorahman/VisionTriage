import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super(BidirectionalCrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        # L>R attention
        self.attn_L2R = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,    # Input shape: [B, seq_len, embed_dim]
        )

        # R>L attention
        self.attn_R2L = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalisation after attention
        self.norm_L = nn.LayerNorm(embed_dim)
        self.norm_R = nn.LayerNorm(embed_dim)

    def forward(self, F_L, F_R):
        # ── Left attends to Right ────────────────────────────────

        F_L_attended, _ = self.attn_L2R(
            query=F_L,
            key=F_R,
            value=F_R
        )   # [B, 225, 256]

        # ── Right attends to Left ────────────────────────────────

        F_R_attended, _ = self.attn_R2L(
            query=F_R,
            key=F_L,
            value=F_L
        )   # [B, 225, 256]

        # ── Layer Normalisation ───────────────────────────────────────

        F_L_attended = self.norm_L(F_L_attended)
        F_R_attended = self.norm_R(F_R_attended)

        return F_L_attended, F_R_attended


class M3_CrossAttentionCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True,
                 attn_dim=256, num_heads=8,
                 fc_hidden_dim=512, dropout=0.5):

        super(M3_CrossAttentionCNN, self).__init__()

        self.attn_dim = attn_dim
        self.num_heads = num_heads

        # ── Two Independent ResNet-50 Backbones ─────────────
        if pretrained:
            left_resnet  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            right_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            left_resnet  = models.resnet50(weights=None)
            right_resnet = models.resnet50(weights=None)

        self.left_encoder  = nn.Sequential(*list(left_resnet.children())[:-2])
        self.right_encoder = nn.Sequential(*list(right_resnet.children())[:-2])

        # ── SHARED Linear Projection ────────────────────────

        self.proj = nn.Linear(2048, attn_dim)

        # ── Bidirectional Cross-Attention ────────────────────
 
        self.cross_attn = BidirectionalCrossAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=0.0,    # No attention dropout
        )

        # ── Classification Head ──────────────────────────────
 
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(attn_dim * 2, num_classes)

        # ── Weight Initialisation ─────────────────────────────────────
        
        nn.init.kaiming_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x_left, x_right):
        B = x_left.shape[0]

        # ── Feature Extraction ───────────────────────────────

        feat_L = self.left_encoder(x_left)
        feat_R = self.right_encoder(x_right)

        # ── Reshape + Shared Linear Projection ───────────────
        F_L = feat_L.flatten(2).permute(0, 2, 1)
        F_R = feat_R.flatten(2).permute(0, 2, 1)

        # SHARED projection to both streams
        F_L = self.proj(F_L)
        F_R = self.proj(F_R)

        # ── Bidirectional Cross-Attention ────────────────────

        F_L_att, F_R_att = self.cross_attn(F_L, F_R)

        # ── Pooling and Classification ───────────────────────

        f_L = F_L_att.mean(dim=1)
        f_R = F_R_att.mean(dim=1)

        joint = torch.cat([f_L, f_R], dim=1)

        # Dropout and classification
        joint = self.dropout(joint)
        logits = self.classifier(joint)

        return logits

    def get_feature_maps(self, x_left, x_right):
        return self.left_encoder(x_left), self.right_encoder(x_right)

    def get_attended_features(self, x_left, x_right):
        B = x_left.shape[0]

        feat_L = self.left_encoder(x_left)
        feat_R = self.right_encoder(x_right)

        F_L = feat_L.flatten(2).permute(0, 2, 1)
        F_R = feat_R.flatten(2).permute(0, 2, 1)

        F_L = self.proj(F_L)
        F_R = self.proj(F_R)

        F_L_att, F_R_att = self.cross_attn(F_L, F_R)

        att_L = F_L_att.permute(0, 2, 1).reshape(B, self.attn_dim, 15, 15)
        att_R = F_R_att.permute(0, 2, 1).reshape(B, self.attn_dim, 15, 15)

        return att_L, att_R


def get_cross_attention_model(config=None):
    if config is None:
        config = {}

    return M3_CrossAttentionCNN(
        num_classes=config.get('num_classes', 3),
        pretrained=config.get('pretrained', True),
        attn_dim=config.get('attn_dim', 256),
        num_heads=config.get('num_heads', 8),
        fc_hidden_dim=config.get('fc_hidden_dim', 512),
        dropout=config.get('dropout', 0.5),
    )