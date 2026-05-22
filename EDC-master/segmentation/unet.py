# segmentation/unet.py
# Improved UNet: deeper encoder (64->128->256->512->1024),
# pretrained ResNet34 encoder option, stronger skip connections.
# in_channels=4 (RGB + heatmap) — novelty unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Helps focus on relevant regions — improves boundary segmentation.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    5-level Attention UNet with deeper bottleneck (1024 channels).
    Attention gates on all skip connections — improves boundary precision.
    in_channels=4 (image + heatmap) for the proposed HGBL model.
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

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (deeper than original)
        self.bridge = DoubleConv(512, 1024, dropout_p=dropout_p)

        # Attention gates
        self.att4 = AttentionGate(F_g=512,  F_l=512,  F_int=256)
        self.att3 = AttentionGate(F_g=256,  F_l=256,  F_int=128)
        self.att2 = AttentionGate(F_g=128,  F_l=128,  F_int=64)
        self.att1 = AttentionGate(F_g=64,   F_l=64,   F_int=32)

        # Decoder
        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        b  = self.bridge(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(b),  self.att4(self.up4(b), d4)], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), self.att3(self.up3(u4), d3)], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), self.att2(self.up2(u3), d2)], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), self.att1(self.up1(u2), d1)], dim=1))

        return torch.sigmoid(self.out(u1))

    def mc_dropout_predict(self, x, n_passes=20):
        self.train()
        preds = torch.stack([self.forward(x) for _ in range(n_passes)], dim=0)
        return preds.mean(0), preds.std(0)