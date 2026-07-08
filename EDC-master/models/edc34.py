# models/edc34.py
#
# BACKBONE ABLATION VARIANT: ResNet-34 encoder + ResNet-34 decoder
# -------------------------------------------------------------------------
# Standalone file — does NOT modify models/edc.py.
# Same RQASW (Novelty 1) fusion logic as R50_R50, only the backbone differs.
#
# IMPORTANT CHANNEL NOTE:
# ResNet34 uses BasicBlock (expansion=1), so layer4 output is 512 channels,
# not 2048 like ResNet50/ResNeXt50/WideResNet50 (Bottleneck, expansion=4).
# The decoder must be built with inplanes=[512] and BasicBlock-style
# resnet34_decoder (already defined in models/resnet_decoder.py) —
# NOT resnet50_decoder, which would silently mismatch shapes.

from models.resnet import resnet34
from models.resnet_decoder import resnet34_decoder
from models.edc import (
    disable_running_stats,
    enable_running_stats,
    variance_reg_loss,
    _adaptive_weights,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class R34_R34(nn.Module):
    """
    ResNet-34 encoder + ResNet-34 decoder for EDC anomaly detection.
    Backbone ablation variant of R50_R50 — identical RQASW fusion logic.
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
        use_rqasw=True,
    ):
        super().__init__()
        self.edc_encoder    = resnet34(pretrained=True)
        # ResNet34 layer4 output = 512 channels (BasicBlock, expansion=1)
        self.edc_decoder    = resnet34_decoder(pretrained=False, inplanes=[512])
        self.train_encoder  = train_encoder
        self.stop_grad      = stop_grad
        self.reshape        = reshape
        self.bn_pretrain    = bn_pretrain
        self.anomap_layer   = anomap_layer
        self.var_reg_weight = var_reg_weight
        self.ema_momentum   = ema_momentum
        self.use_rqasw      = use_rqasw

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

        if self.use_rqasw:
            w = _adaptive_weights(self.ema_l1, self.ema_l2, self.ema_l3)
        else:
            w = torch.ones(3, device=x.device) / 3.0

        p_maps   = [p1, p2_up, p3_up]
        selected = [p_maps[l - 1] for l in self.anomap_layer]
        w_sel    = torch.stack([w[l - 1] for l in self.anomap_layer])
        w_sel    = w_sel / w_sel.sum()
        p_all    = sum(w_sel[i] * selected[i] for i in range(len(selected)))

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