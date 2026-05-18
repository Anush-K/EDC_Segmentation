import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Conv → BN → ReLU → Conv → BN → ReLU

    BatchNorm is placed between conv and activation, which is the standard
    UNet convention. It stabilises training, reduces sensitivity to weight
    initialisation, and typically improves generalisation on small datasets.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # bias=False is correct when followed by BatchNorm:
        # BN's beta parameter absorbs any constant offset, so a conv bias
        # would be redundant and waste parameters.

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Lightweight 3-level UNet.

    Encoder:  3 (or 4) → 64 → 128 → 256  (bridge)
    Decoder:  256 → 128 → 64 → 1 (sigmoid)

    in_channels=3  for Approaches 1 & 3 (RGB only)
    in_channels=4  for Approach  2      (RGB + heatmap channel)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ---------- encoder ----------
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # ---------- bridge ----------
        self.bridge = DoubleConv(128, 256)

        # ---------- decoder ----------
        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)   # 128 up + 128 skip = 256 in

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)    # 64 up  + 64  skip = 128 in

        # ---------- output ----------
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        # bridge
        b = self.bridge(self.pool2(d2))

        # decoder with skip connections
        u2 = self.up2(b)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return torch.sigmoid(self.out(u1))