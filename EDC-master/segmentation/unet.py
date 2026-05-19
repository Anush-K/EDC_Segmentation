# segmentation/unet.py
# Architecture unchanged from the corrected version.
# 4-level encoder (64->128->256->512), BatchNorm after every Conv,
# Dropout at bottleneck for MC-Dropout uncertainty at test time.
# in_channels=4 (RGB + heatmap channel) is the default for the proposed model.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU (+ optional Dropout)"""

    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    4-level UNet with BatchNorm and bottleneck Dropout.

    in_channels : 3 (image only) or 4 (image + heatmap)
    dropout_p   : Dropout probability at bottleneck.
                  Set 0.3 during training; keep at eval() for deterministic
                  inference, or keep train() for MC-Dropout uncertainty maps.
    """

    def __init__(self, in_channels=4, dropout_p=0.3):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bridge = DoubleConv(256, 512, dropout_p=dropout_p)

        # Decoder
        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)   # 256 up + 256 skip = 512

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)   # 128 up + 128 skip = 256

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)    # 64  up + 64  skip = 128

        self.out = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        d1     = self.down1(x)
        d2     = self.down2(self.pool1(d1))
        d3     = self.down3(self.pool2(d2))
        bridge = self.bridge(self.pool3(d3))

        u3 = self.conv3(torch.cat([self.up3(bridge), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3),     d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2),     d1], dim=1))

        return torch.sigmoid(self.out(u1))

    def mc_dropout_predict(self, x, n_passes=20):
        """
        Stochastic inference for uncertainty estimation.
        Keep model in train() mode so Dropout stays active.
        Returns: mean prediction (N,1,H,W) and std uncertainty (N,1,H,W).
        """
        self.train()
        preds = torch.stack([self.forward(x) for _ in range(n_passes)], dim=0)
        return preds.mean(0), preds.std(0)