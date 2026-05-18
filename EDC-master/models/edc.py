# models/edc.py
from models.resnet import resnet50, wide_resnet50_2
from models.resnet_decoder import resnet50_decoder, wide_resnet50_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import math


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid  = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid  = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean     = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
        torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels,
        bias=False, padding=kernel_size // 2,
    )
    gaussian_filter.weight.data     = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


class Dict2Obj(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


# ---------------------------------------------------------------------------
# Anti-collapse variance regularisation
# ---------------------------------------------------------------------------
def variance_reg_loss(feat, eps=1e-4):
    """
    VICReg-style variance term: penalises when std of each feature dimension
    across the batch drops below 1.  Prevents representation collapse.

    Args:
        feat : (B, C, H, W)  or  (B, C)
    Returns:
        scalar loss
    """
    if feat.dim() == 4:
        # Flatten spatial dims → (B, C)
        feat = feat.mean(dim=[2, 3])
    # feat: (B, C)
    std  = torch.sqrt(feat.var(dim=0) + eps)          # (C,)
    loss = torch.mean(F.relu(1.0 - std))              # push std ≥ 1
    return loss


# ---------------------------------------------------------------------------
# R50_R50  —  corrected forward logic
# ---------------------------------------------------------------------------
class R50_R50(nn.Module):
    """
    ResNet-50 encoder  +  ResNet-50 decoder for EDC anomaly detection.

    Key fixes vs original:
      • stop_grad=False by default  — encoder receives gradients from ALL
        three skip-level cosine losses, not just the bottleneck path.
        This is the primary fix for representation collapse.
      • Variance regularisation added to the training loss — prevents
        the encoder from outputting constant feature maps.
      • bn_pretrain=True by default — keeps encoder BN in eval mode so
        batch statistics from ImageNet pretraining are preserved.
        This stabilises early training on a small medical dataset.
    """

    def __init__(
        self,
        img_size=256,
        train_encoder=True,
        stop_grad=False,        # FIX: was True — caused collapse
        reshape=True,
        bn_pretrain=True,       # FIX: was False — keep ImageNet BN stats
        anomap_layer=[1, 2, 3],
        var_reg_weight=0.04,    # weight for variance regularisation loss
    ):
        super().__init__()
        self.edc_encoder    = resnet50(pretrained=True)
        self.edc_decoder    = resnet50_decoder(pretrained=False, inplanes=[2048])
        self.train_encoder  = train_encoder
        self.stop_grad      = stop_grad
        self.reshape        = reshape
        self.bn_pretrain    = bn_pretrain
        self.anomap_layer   = anomap_layer
        self.var_reg_weight = var_reg_weight

    def forward(self, x):
        # ---- Encoder mode control ----------------------------------------
        # Keep BN in eval to preserve ImageNet running stats on small dataset
        if self.bn_pretrain and self.edc_encoder.training:
            # Only freeze BN; conv weights still receive gradients
            for m in self.edc_encoder.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        # ---- Forward pass ------------------------------------------------
        e1, e2, e3, e4 = self.edc_encoder(x)

        if not self.train_encoder:
            e4 = e4.detach()

        d1, d2, d3 = self.edc_decoder(e4)

        # FIX: stop_grad=False means encoder gets gradients from cosine losses
        # Only detach when train_encoder is explicitly False
        if not self.train_encoder:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()
        elif self.stop_grad:
            # Legacy fallback — not recommended, causes collapse
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()
        # else: gradients flow freely through e1, e2, e3 → FIX

        # ---- Cosine similarity losses ------------------------------------
        if self.reshape:
            l1 = 1. - torch.cosine_similarity(
                d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(
                d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(
                d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        recon_loss = l1 + l2 + l3

        # ---- Variance regularisation -------------------------------------
        # FIX: prevents encoder features from collapsing to constants
        if self.train_encoder and not self.stop_grad:
            vr1 = variance_reg_loss(e1)
            vr2 = variance_reg_loss(e2)
            vr3 = variance_reg_loss(e3)
            var_loss = (vr1 + vr2 + vr3) / 3.0
        else:
            var_loss = torch.zeros(1, device=x.device).squeeze()

        loss = recon_loss + self.var_reg_weight * var_loss

        # ---- Pixel-wise anomaly maps (no grad needed) --------------------
        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        p2_up = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2_up, p3_up][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        # ---- Feature std monitoring (diagnostic) -------------------------
        with torch.no_grad():
            e1_std = F.normalize(
                e1.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(
                e2.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(
                e3.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss':     loss,
            'recon_loss': recon_loss,
            'var_loss': var_loss,
            'p_all':    p_all,
            'p1':       p1,
            'p2':       p2_up,
            'p3':       p3_up,
            'e1_std':   e1_std,
            'e2_std':   e2_std,
            'e3_std':   e3_std,
        }


# ---------------------------------------------------------------------------
# WR50_WR50  — same fixes applied
# ---------------------------------------------------------------------------
class WR50_WR50(nn.Module):
    def __init__(
        self,
        img_size=256,
        train_encoder=True,
        stop_grad=False,        # FIX
        reshape=True,
        bn_pretrain=True,       # FIX
        anomap_layer=[1, 2, 3],
        var_reg_weight=0.04,
    ):
        super().__init__()
        self.edc_encoder    = wide_resnet50_2(pretrained=True)
        self.edc_decoder    = wide_resnet50_decoder(pretrained=False, inplanes=[2048])
        self.train_encoder  = train_encoder
        self.stop_grad      = stop_grad
        self.reshape        = reshape
        self.bn_pretrain    = bn_pretrain
        self.anomap_layer   = anomap_layer
        self.var_reg_weight = var_reg_weight

    def forward(self, x):
        if self.bn_pretrain and self.edc_encoder.training:
            for m in self.edc_encoder.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]
        e1, e2, e3, e4 = self.edc_encoder(x)

        if not self.train_encoder:
            e4 = e4.detach()

        d1, d2, d3 = self.edc_decoder(e4)

        if not self.train_encoder:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()
        elif self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        if self.reshape:
            l1 = 1. - torch.cosine_similarity(
                d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(
                d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(
                d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        recon_loss = l1 + l2 + l3

        if self.train_encoder and not self.stop_grad:
            vr1 = variance_reg_loss(e1)
            vr2 = variance_reg_loss(e2)
            vr3 = variance_reg_loss(e3)
            var_loss = (vr1 + vr2 + vr3) / 3.0
        else:
            var_loss = torch.zeros(1, device=x.device).squeeze()

        loss = recon_loss + self.var_reg_weight * var_loss

        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        p2_up = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2_up, p3_up][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(
                e1.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(
                e2.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(
                e3.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss':       loss,
            'recon_loss': recon_loss,
            'var_loss':   var_loss,
            'p_all':      p_all,
            'p1':         p1,
            'p2':         p2_up,
            'p3':         p3_up,
            'e1_std':     e1_std,
            'e2_std':     e2_std,
            'e3_std':     e3_std,
        }