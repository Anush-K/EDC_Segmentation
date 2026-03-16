import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self,in_ch,out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):

    def __init__(self,in_channels):

        super().__init__()

        self.down1 = DoubleConv(in_channels,64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(128,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv2 = DoubleConv(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv1 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        bridge = self.bridge(self.pool2(d2))

        u2 = self.up2(bridge)
        u2 = torch.cat([u2,d2],dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1,d1],dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.out(u1))