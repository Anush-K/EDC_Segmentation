"""Per-image metrics on binary masks (numpy bool HxW)."""
import numpy as np
from scipy import ndimage

def dice_iou(p, g):
    inter = np.logical_and(p, g).sum(); ps, gs = p.sum(), g.sum()
    if ps + gs == 0:
        return 1.0, 1.0
    return 2 * inter / (ps + gs), inter / (ps + gs - inter)

def _surface(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)

def hd95(p, g):
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float(np.hypot(*p.shape))
    sp, sg_ = _surface(p), _surface(g)
    dg = ndimage.distance_transform_edt(~sg_); dp = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dg[sp], dp[sg_]])
    return float(np.percentile(d, 95))

def boundary_f1(p, g, tol=2):
    sp, sg_ = _surface(p), _surface(g)
    if not sp.any() and not sg_.any():
        return 1.0
    if not sp.any() or not sg_.any():
        return 0.0
    dg = ndimage.distance_transform_edt(~sg_); dp = ndimage.distance_transform_edt(~sp)
    prec = (dg[sp] <= tol).mean(); rec = (dp[sg_] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def all_metrics(p, g):
    d, i = dice_iou(p, g)
    return {"dice": float(d), "iou": float(i), "hd95": hd95(p, g), "bf1": float(boundary_f1(p, g))}
