import torch
import torch.nn as nn
from torchvision.models import swin_transformer


class SwinV2(nn.Module):
    def __init__(self, weights=None):
        super(SwinV2, self).__init__()

        self.backbone = swin_transformer.SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            mlp_ratio=4.0,
            stochastic_depth_prob=0.2,
            num_classes=1000,
        )

        if weights is not None:
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/swin_b-68c6b09e.pth"
                )
            )

        # Remove the classification head
        self.backbone.head = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


def extract_visual_features(images, model):
    features = model(images)
    return features
