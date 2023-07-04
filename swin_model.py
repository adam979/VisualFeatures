import torch
import torchvision.models as models
from torchvision.transforms import InterpolationMode
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
from einops import rearrange, repeat


class WindowAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = rearrange(x, "b h w d -> b d h w")
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = rearrange(q, "b d h w -> b h w d")
        k = rearrange(k, "b d h w -> b w h d")
        v = rearrange(v, "b d h w -> b h w d")

        attn = self.softmax(torch.einsum("bhwhd,bwhd->bhwh", q, k))
        attn = attn.masked_fill(attn.isnan(), 0.0)

        out = torch.einsum("bhwhd,bhwd->bhwhd", attn, v)
        out = rearrange(out, "b h w d -> b d h w")
        out = self.proj(out)
        out = out * (1.0 - self.proj_drop)

        return out


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        self.attn = WindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.mlp = Mlp(
            in_features=2 * embed_dim,
            hidden_features=int(4 * embed_dim),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x)
        x = x + self.mlp(x)
        x = self.norm_layer(x)
        return x


class SwinTransformerStage(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, window_size):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                SwinTransformerBlock(
                    embed_dim=embed_dim, num_heads=num_heads, window_size=window_size
                )
                for _ in range(depths)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=1000, embed_dim=96):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.ModuleList(
            [
                SwinTransformerStage(
                    embed_dim=embed_dim,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=[7, 7, 7, 7],
                )
                for _ in range(4)
            ]
        )

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        for stage in self.stages:
            x = stage(x)
        x = rearrange(x, "b c h w -> b (h w) c")  # Flatten spatial dimensions
        x = self.head(x)
        return x


def get_visual_features(model, x):
    x = model.embed(x)
    for stage in model.stages:
        x = stage(x)
    return x


def extract_visual_features(images, model):
    # Preprocess images if needed
    transform = model.transforms
    images = transform(images)

    # Extract visual features
    visual_features = get_visual_features(model, images)

    return visual_features
