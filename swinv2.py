import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import swin_transformer


class LocalFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(LocalFeatureExtractor, self).__init__()

        self.conv = nn.Conv2d(in_channels, 10, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


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

        # Add additional layers to extract local features
        self.local_features = LocalFeatureExtractor(96)

    def forward(self, x):
        global_features = self.backbone(x)
        local_features = self.local_features(global_features)
        return global_features, local_features


def extract_visual_features(images, model):
    # Convert grayscale images to 3 channels (assuming input is grayscale with 1 channel)
    if images.size(1) == 1:
        images_rgb = images.repeat(1, 3, 1, 1)
    else:
        images_rgb = images

    print("Shape of images_rgb:", images_rgb.size())

    # Extract features using the modified SwinV2 model
    global_features, local_features = model(images_rgb)
    print("Shape of global_features (after Swin Transformer):", global_features.size())
    print("Shape of local_features (before convolution):", local_features.size())

    # Apply adaptive average pooling to global_features to resize it to a fixed spatial size
    global_features = F.adaptive_avg_pool2d(global_features, (7, 7))

    # Apply the convolution operation to the local features tensor
    local_features = model.local_features(global_features)

    print(
        "Shape of global_features (after adaptive avg pooling):", global_features.size()
    )
    print("Shape of local_features (after convolution):", local_features.size())

    return global_features, local_features
