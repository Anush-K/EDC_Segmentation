"""Model factory (file name hgbl_models.py, so it never clashes with the EDC-master/models/ package).
  --arch yours     : YOUR UNet, loaded directly from your existing file (default
                     .../segmentation/unet.py) - nothing is copied or renamed.
                     Give another location with --unet_path. It is called as
                     UNet(in_channels=C, dropout_p=0.3) and returns probabilities (sigmoid).
  --arch attunet   : 5-level Attention U-Net (Oktay et al.) defined below, sigmoid output.
  --arch smp_unet  : U-Net with ImageNet ResNet-34 encoder (pip install segmentation-models-pytorch).
"""
import os
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
UNET_CANDIDATES = [
    os.path.join(HERE, "..", "segmentation", "unet.py"),          # EDC-master/segmentation/unet.py
    os.path.join(HERE, "..", "..", "segmentation", "unet.py"),
    "/home/cs24d0008/EDC_Segmentation/EDC-master/segmentation/unet.py",
    "/home/cs24d0008/EDC_Segmentation/segmentation/unet.py",
]


def load_user_unet(path=None):
    """Import UNet from your unet.py under the private name 'user_unet' (no name clash)."""
    cands = [path] if path else UNET_CANDIDATES
    for p in cands:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("user_unet", os.path.abspath(p))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            print("[hgbl_models] using UNet from", os.path.abspath(p))
            return mod.UNet
    raise FileNotFoundError("unet.py not found; give --unet_path /full/path/to/segmentation/unet.py "
                            "(tried: %s)" % cands)

class Conv2(nn.Module):
    def __init__(self, i, o, p=0.0):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                               nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                               nn.Dropout2d(p) if p > 0 else nn.Identity())
    def forward(self, x):
        return self.b(x)

class AttGate(nn.Module):
    def __init__(self, fg, fl, fi):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(fg, fi, 1), nn.BatchNorm2d(fi))
        self.wx = nn.Sequential(nn.Conv2d(fl, fi, 1), nn.BatchNorm2d(fi))
        self.psi = nn.Sequential(nn.Conv2d(fi, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=4, base=32, dropout_p=0.3):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([Conv2(in_ch, c[0])] + [Conv2(c[i], c[i + 1], dropout_p if i == 3 else 0) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, 2) for i in range(4)])
        self.att = nn.ModuleList([AttGate(c[i], c[i], max(c[i] // 2, 8)) for i in range(4)])
        self.dec = nn.ModuleList([Conv2(c[i] * 2, c[i]) for i in range(4)])
        self.out = nn.Conv2d(c[0], 1, 1)
    def forward(self, x):
        s = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); s.append(x)
        x = s[-1]
        for i in reversed(range(4)):
            g = self.up[i](x); x = self.dec[i](torch.cat([g, self.att[i](g, s[i])], 1))
        return torch.sigmoid(self.out(x))

class _Sig(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return torch.sigmoid(self.m(x))

def build_model(arch, in_ch, dropout_p=0.3, unet_path=None):
    if arch == "yours":
        UNet = load_user_unet(unet_path)
        return UNet(in_channels=in_ch, dropout_p=dropout_p)
    if arch == "attunet":
        return AttentionUNet(in_ch, dropout_p=dropout_p)
    if arch == "smp_unet":
        import segmentation_models_pytorch as smp
        return _Sig(smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=1))
    raise ValueError(arch)