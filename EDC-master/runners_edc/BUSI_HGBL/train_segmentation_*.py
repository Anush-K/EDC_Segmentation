"""Losses, identical in form to train_segmentation_*.py (probabilities, not logits):
   baseline = BCE + Tversky(0.3, 0.7) + Focal(alpha 0.8, gamma 2)
   HGBL     = Omega-weighted BCE + Tversky + Focal,
   Omega    = (1 + lam_b * B) * (1 + lam_h * S * sg(pred)),  B = mean of 3 boundary bands.
"""
import torch
import torch.nn.functional as F

EPS = 1e-8

def tversky_loss(pred, target, alpha=0.3, beta=0.7):
    tp = (pred * target).sum(); fp = (pred * (1 - target)).sum(); fn = ((1 - pred) * target).sum()
    return 1. - (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    return (alpha * (1 - pt) ** gamma * bce).mean()

def boundary_map(mask):
    B = torch.zeros_like(mask)
    for ks in (3, 5, 7):
        p = ks // 2
        dil = F.max_pool2d(mask, ks, stride=1, padding=p)
        ero = -F.max_pool2d(-mask, ks, stride=1, padding=p)
        B = B + (dil - ero).clamp(0., 1.)
    return (B / 3.0).clamp(0., 1.)

def baseline_loss(pred, mask):
    return F.binary_cross_entropy(pred, mask) + tversky_loss(pred, mask) + focal_loss(pred, mask)

def hgbl_loss(pred, mask, heat, lam_b=5.0, lam_h=1.0, heat_term="S_pred"):
    """heat_term: S_pred (paper, S * sg(pred)) | S (heatmap alone) | S_gt (S * M)."""
    with torch.no_grad():
        B = boundary_map(mask)
        if heat_term == "S":
            G = heat
        elif heat_term == "S_gt":
            G = heat * mask
        else:
            G = heat * pred.detach()
        W = (1. + lam_b * B) * (1. + lam_h * G)
    wbce = (F.binary_cross_entropy(pred, mask, reduction="none") * W).mean()
    return wbce + tversky_loss(pred, mask) + focal_loss(pred, mask)