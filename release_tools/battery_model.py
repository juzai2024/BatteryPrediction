"""Model definitions used by the battery-capacity inference demo."""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer


class BatteryVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer encoder returning the class-token embedding."""

    def forward_features(self, images):
        embeddings = self.patch_embed(images)
        class_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        embeddings = torch.cat((class_tokens, embeddings), dim=1)
        embeddings = self.pos_drop(embeddings + self.pos_embed)
        for block in self.blocks:
            embeddings = block(embeddings)
        embeddings = self.norm(embeddings)
        return embeddings[:, 0]


def vit_battery_patch16(**kwargs):
    """Build the ViT-L/16 encoder used by the released checkpoint."""

    return BatteryVisionTransformer(
        num_classes=0,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


class CapacityHead(nn.Module):
    """MLP regression head used to predict battery capacity in Ah."""

    def __init__(
        self,
        input_dim=1024,
        output_dim=1,
        hidden_dim=512,
        proj_dim=128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.reg_head = nn.Linear(hidden_dim, output_dim)
        self.proj_head = nn.Linear(hidden_dim, proj_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, features):
        hidden = self.drop(self.relu(self.fc1(features)))
        capacity = self.reg_head(hidden)
        projection = F.normalize(self.proj_head(hidden), dim=-1)
        return capacity, projection
