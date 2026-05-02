import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for segmentation baseline.
    Not too strong, not too weak — fair comparison.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # decoder (no skip connections, intentionally simple)
        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(x)

        x = self.final(x)

        return x