import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """
    Lightweight decoder for segmentation.

    Takes transformer feature map and upsamples
    to original resolution.
    """

    def __init__(self, in_channels=768, mid_channels=256, out_channels=1):
        super().__init__()

        # reduce channel dimension
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)

        self.conv_out = nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1)

    def forward(self, x, output_size=None):
        """
        x: (B, D, H, W)  -> transformer features
        output_size: (H, W) of original image (optional)
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.conv_out(x)

        # upsample to match input resolution
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        return x


class RoadSegmentationModel(nn.Module):
    """
    Full model = Prithvi backbone + decoder
    """

    def __init__(self, backbone, decoder):
        super().__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        input_size = x.shape[-2:]  # original H, W

        features = self.backbone(x)

        out = self.decoder(features, output_size=input_size)

        return out