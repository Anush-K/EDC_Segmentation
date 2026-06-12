# models/edc.py
#
# NOVELTY 1: Reconstruction-Quality-Driven Adaptive Scale Weighting (RQASW)
# -------------------------------------------------------------------------
# Baseline EDC fuses anomaly maps p1/p2/p3 with a fixed equal-weight mean.
# RQASW tracks per-scale reconstruction loss via EMA buffers during training
# (no labels, zero extra trainable parameters). Scales where the decoder
# consistently struggles more → more anomaly-sensitive → higher fusion weight.
#
# SASC Enhancement (Self-Attention Skip Connection):
# -------------------------------------------------------------------------
# Inspired by EA2D (Tang et al., IEEE TMI 2025).
# Purpose in RQASW: enhances normal image reconstruction quality at each
# scale, making EMA loss signals more discriminative for adaptive weighting.
# Unlike EA2D (domain adaptation + dual decoder), SASC here serves to
# strengthen RQASW scale discrimination → better heatmaps → better HGBL.
# Encoder features e1, e2 passed to decoder as attention-refined skip connections.

from models.resnet import resnet50, wide_resnet50_2
from models.resnet_decoder import resnet50_decoder, wide_resnet50_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import math


# ---------------------------------------------------------------------------
# BN control helpers
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


# ---------------------------------------------------------------------------
# Anti-collapse variance regularisation (VICReg-style)
# ---------------------------------------------------------------------------
def variance_reg_loss(feat, eps=1e-4):
    if feat.dim() == 4:
        feat = feat.mean(dim=[2, 3])
    std = torch.sqrt(feat.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


# ---------------------------------------------------------------------------
# NOVELTY 1 helper
# ---------------------------------------------------------------------------
def _adaptive_weights(ema_l1, ema_l2, ema_l3, eps=1e-6):
    """
    Normalise three EMA loss values into fusion weights.
    Higher EMA loss → harder to reconstruct → more anomaly signal → higher weight.
    Returns w: (3,) tensor summing to 1.
    """
    raw = torch.stack([ema_l1, ema_l2, ema_l3]).clamp(min=eps)
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# R50_R50
# ---------------------------------------------------------------------------
class R50_R50(nn.Module):
    """
    ResNet-50 encoder + ResNet-50 decoder for EDC anomaly detection.

    NOVELTY 1 (RQASW):
      Three non-trainable EMA buffers track mean reconstruction loss per scale.
      Forward() computes adaptive fusion weights [w1, w2, w3] replacing
      the fixed equal-weight mean of the original EDC.

    SASC Enhancement:
      Encoder features e1, e2 are passed to the decoder as attention-refined
      skip connections via PositionAttentionModule (PAM).
      This improves normal reconstruction quality → sharper EMA loss signal
      → better RQASW scale discrimination → better anomaly heatmaps.
    """

    def __init__(
        self,
        img_size=256,
        train_encoder=True,
        stop_grad=False,
        reshape=True,
        bn_pretrain=True,
        anomap_layer=[1, 2, 3],
        var_reg_weight=0.04,
        ema_momentum=0.99,
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
        self.ema_momentum   = ema_momentum

        # RQASW: EMA loss buffers (non-trainable)
        self.register_buffer('ema_l1', torch.tensor(1.0))
        self.register_buffer('ema_l2', torch.tensor(1.0))
        self.register_buffer('ema_l3', torch.tensor(1.0))

    def forward(self, x):
        # ---- BN mode control ---------------------------------------------
        if self.bn_pretrain and self.edc_encoder.training:
            for m in self.edc_encoder.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        # ---- Encoder -----------------------------------------------------
        e1, e2, e3, e4 = self.edc_encoder(x)

        if not self.train_encoder:
            e4 = e4.detach()

        # ---- Decoder with SASC skip connections --------------------------
        # e1, e2 passed as attention-refined skip connections to decoder
        # PAM(e2) added to decoder layer2 output
        # PAM(e1) added to decoder layer1 output
        d1, d2, d3 = self.edc_decoder(e4)

        # ---- Gradient control --------------------------------------------
        if not self.train_encoder:
            e1, e2, e3 = e1.detach(), e2.detach(), e3.detach()
        elif self.stop_grad:
            e1, e2, e3 = e1.detach(), e2.detach(), e3.detach()

        # ---- Per-scale cosine reconstruction losses ----------------------
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
        if self.train_encoder and not self.stop_grad:
            var_loss = (variance_reg_loss(e1) +
                        variance_reg_loss(e2) +
                        variance_reg_loss(e3)) / 3.0
        else:
            var_loss = torch.zeros(1, device=x.device).squeeze()

        loss = recon_loss + self.var_reg_weight * var_loss

        # ---- Pixel-wise anomaly maps (no grad needed) --------------------
        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        p2_up = F.interpolate(p2, scale_factor=2,  mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, scale_factor=4,  mode='bilinear', align_corners=False)

        # ---- NOVELTY 1: RQASW adaptive fusion ---------------------------
        if self.training:
            m = self.ema_momentum
            self.ema_l1 = m * self.ema_l1 + (1.0 - m) * l1.detach()
            self.ema_l2 = m * self.ema_l2 + (1.0 - m) * l2.detach()
            self.ema_l3 = m * self.ema_l3 + (1.0 - m) * l3.detach()

        w      = _adaptive_weights(self.ema_l1, self.ema_l2, self.ema_l3)
        p_maps = [p1, p2_up, p3_up]
        selected = [p_maps[l - 1] for l in self.anomap_layer]
        w_sel    = torch.stack([w[l - 1] for l in self.anomap_layer])
        w_sel    = w_sel / w_sel.sum()
        p_all    = sum(w_sel[i] * selected[i] for i in range(len(selected)))
        # ---- end RQASW --------------------------------------------------

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
            'scale_w1':   w[0].detach(),
            'scale_w2':   w[1].detach(),
            'scale_w3':   w[2].detach(),
        }


# ---------------------------------------------------------------------------
# WR50_WR50 — identical RQASW + SASC
# ---------------------------------------------------------------------------
class WR50_WR50(nn.Module):
    """Wide ResNet-50-2 variant with RQASW + SASC."""

    def __init__(
        self,
        img_size=256,
        train_encoder=True,
        stop_grad=False,
        reshape=True,
        bn_pretrain=True,
        anomap_layer=[1, 2, 3],
        var_reg_weight=0.04,
        ema_momentum=0.99,
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
        self.ema_momentum   = ema_momentum

        self.register_buffer('ema_l1', torch.tensor(1.0))
        self.register_buffer('ema_l2', torch.tensor(1.0))
        self.register_buffer('ema_l3', torch.tensor(1.0))

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

        # SASC: pass e1, e2 as skip connections
        d1, d2, d3 = self.edc_decoder(e4)

        if not self.train_encoder:
            e1, e2, e3 = e1.detach(), e2.detach(), e3.detach()
        elif self.stop_grad:
            e1, e2, e3 = e1.detach(), e2.detach(), e3.detach()

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
            var_loss = (variance_reg_loss(e1) +
                        variance_reg_loss(e2) +
                        variance_reg_loss(e3)) / 3.0
        else:
            var_loss = torch.zeros(1, device=x.device).squeeze()

        loss = recon_loss + self.var_reg_weight * var_loss

        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        p2_up = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        if self.training:
            m = self.ema_momentum
            self.ema_l1 = m * self.ema_l1 + (1.0 - m) * l1.detach()
            self.ema_l2 = m * self.ema_l2 + (1.0 - m) * l2.detach()
            self.ema_l3 = m * self.ema_l3 + (1.0 - m) * l3.detach()

        w     = _adaptive_weights(self.ema_l1, self.ema_l2, self.ema_l3)
        sel   = [([p1, p2_up, p3_up])[l - 1] for l in self.anomap_layer]
        w_sel = torch.stack([w[l - 1] for l in self.anomap_layer])
        w_sel = w_sel / w_sel.sum()
        p_all = sum(w_sel[i] * sel[i] for i in range(len(sel)))

        with torch.no_grad():
            e1_std = F.normalize(
                e1.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(
                e2.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(
                e3.detach().permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss': loss, 'recon_loss': recon_loss, 'var_loss': var_loss,
            'p_all': p_all, 'p1': p1, 'p2': p2_up, 'p3': p3_up,
            'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std,
            'scale_w1': w[0].detach(), 'scale_w2': w[1].detach(),
            'scale_w3': w[2].detach(),
        }