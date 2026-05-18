# segmentation/unet.py
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# FIX: added BatchNorm after every Conv2d — essential for stable training
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
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


# ---------------------------------------------------------------------------
# UNet
# FIX: expanded from 3 to 4 encoder levels (64→128→256→512) for adequate
#      receptive field on 256×256 LGG images.
#      Dropout added at bottleneck for MC-Dropout uncertainty at test time.
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    Standard 4-level UNet with BatchNorm.

    Args:
        in_channels : 3 (RGB only) or 4 (RGB + heatmap)
        dropout_p   : dropout probability at the bottleneck
                      (use 0.3–0.5 if you want MC-Dropout uncertainty maps)
    """

    def __init__(self, in_channels=3, dropout_p=0.3):
        super().__init__()

        # ---- Encoder ----
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # ---- Bottleneck (with dropout for uncertainty) ----
        self.bridge = DoubleConv(256, 512, dropout_p=dropout_p)

        # ---- Decoder ----
        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)   # 256 up + 256 skip = 512 in

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)   # 128 up + 128 skip = 256 in

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)    # 64 up  + 64  skip = 128 in

        # ---- Output ----
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        # Bottleneck
        bridge = self.bridge(self.pool3(d3))

        # Decoder
        u3 = self.up3(bridge)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return torch.sigmoid(self.out(u1))

    # ------------------------------------------------------------------
    # MC-Dropout inference helper
    # ------------------------------------------------------------------
    def mc_dropout_predict(self, x, n_passes=20):
        """
        Run n_passes stochastic forward passes with dropout enabled.
        Returns mean prediction and per-pixel uncertainty (std).

        Usage:
            model.train()   # keeps dropout active
            mean, uncertainty = model.mc_dropout_predict(x, n_passes=20)
            model.eval()
        """
        self.train()   # activates Dropout
        preds = torch.stack([self.forward(x) for _ in range(n_passes)], dim=0)
        mean        = preds.mean(0)
        uncertainty = preds.std(0)
        return mean, uncertainty