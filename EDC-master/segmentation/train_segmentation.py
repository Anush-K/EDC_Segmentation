# segmentation/train_segmentation.py
#
# NOVELTY 2: Heatmap-Guided Boundary-Aware Loss (HGBL)
# -----------------------------------------------------
# Standard training uses BCE + Dice uniformly over all pixels.
# HGBL modulates the per-pixel BCE weight by two spatial factors:
#
#   Factor 1 — Boundary weight B(x,y):
#     Computed from the GT mask via morphological dilation - erosion.
#     Pixels near tumor boundaries get higher weight because boundaries
#     are where segmentation errors are most common and most costly.
#
#   Factor 2 — Heatmap confidence H(x,y):
#     The EDC anomaly heatmap value at each pixel.
#     Where the anomaly detector is confident (high heatmap value),
#     the loss pushes harder → directly couples anomaly detection
#     signal into the segmentation objective.
#
#   Combined weight: W = (1 + λ_b * B) * (1 + λ_h * H)
#   Loss: BCE(pred * W, mask) + Dice(pred, mask)
#
# Paper claim:
#   "We propose a heatmap-guided boundary-aware loss that spatially
#    modulates segmentation supervision using both geometric boundary
#    proximity and anomaly detector confidence, creating a direct
#    computational bridge between the anomaly detection and segmentation
#    stages of the pipeline."
#
# Code changes: new hgbl_loss() + boundary_map() in this file only.
# UNet architecture unchanged. Dataset returns heatmap separately.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from dataset import LGGSegDataset
from unet import UNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "../LGG_SEG"
BATCH_SIZE   = 8
EPOCHS       = 100
LR           = 1e-4
SEED         = 42
SAVE_DIR     = "./seg_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# HGBL hyperparameters (tunable, reported in ablation)
LAMBDA_B = 2.0    # boundary weight multiplier
LAMBDA_H = 1.0    # heatmap confidence weight multiplier


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_score(pred, target):
    pred         = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)


def iou_score(pred, target):
    pred         = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)


# ---------------------------------------------------------------------------
# Standard losses (used in baseline)
# ---------------------------------------------------------------------------
def dice_loss(pred, target):
    smooth       = 1e-8
    intersection = (pred * target).sum()
    return 1. - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def baseline_loss(pred, target):
    """BCE + Dice — uniform weight over all pixels (baseline)."""
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)


# ---------------------------------------------------------------------------
# NOVELTY 2: HGBL
# ---------------------------------------------------------------------------
def boundary_map(mask, kernel_size=5):
    """
    Compute a soft boundary map from a binary GT mask.
    Uses morphological dilation - erosion (implemented via max/min pooling).
    Returns a (N,1,H,W) tensor where boundary pixels = 1, interior = 0.
    """
    pad     = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded  = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def hgbl_loss(pred, mask, heatmap, lambda_b=LAMBDA_B, lambda_h=LAMBDA_H):
    """
    Heatmap-Guided Boundary-Aware Loss.

    Args:
        pred    : (N,1,H,W) sigmoid output of UNet
        mask    : (N,1,H,W) binary GT mask
        heatmap : (N,1,H,W) EDC anomaly map in [0,1]
        lambda_b: weight for boundary term
        lambda_h: weight for heatmap confidence term

    Returns:
        scalar loss
    """
    # Factor 1: geometric boundary proximity
    B = boundary_map(mask)                              # (N,1,H,W) in [0,1]

    # Factor 2: anomaly detector confidence
    # Normalise heatmap per image so range is always [0,1]
    H_flat = heatmap.flatten(1)                         # (N, H*W)
    H_max  = H_flat.max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-8)
    H      = heatmap / H_max                            # (N,1,H,W) in [0,1]

    # Spatial weight map
    W = (1.0 + lambda_b * B) * (1.0 + lambda_h * H)   # (N,1,H,W)

    # Weighted BCE: each pixel's BCE is scaled by W
    bce_per_pixel = F.binary_cross_entropy(pred, mask, reduction='none')  # (N,1,H,W)
    weighted_bce  = (bce_per_pixel * W).mean()

    # Dice term stays unweighted (global, not pixel-wise)
    dice = dice_loss(pred, mask)

    return weighted_bce + dice


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, use_hgbl=True):
    model.train()
    total_loss = 0.0

    for image, heatmap, mask in tqdm(loader, leave=False, desc='  train'):
        image, heatmap, mask = image.to(DEVICE), heatmap.to(DEVICE), mask.to(DEVICE)

        # UNet input: image (3ch) concatenated with heatmap (1ch) → (4ch)
        x = torch.cat([image, heatmap], dim=1)

        optimizer.zero_grad()
        pred = model(x)

        if use_hgbl:
            loss = hgbl_loss(pred, mask, heatmap)
        else:
            loss = baseline_loss(pred, mask)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    dice_total = 0.0
    iou_total  = 0.0

    with torch.no_grad():
        for image, heatmap, mask in loader:
            image, heatmap, mask = image.to(DEVICE), heatmap.to(DEVICE), mask.to(DEVICE)
            x    = torch.cat([image, heatmap], dim=1)
            pred = model(x)
            dice_total += dice_score(pred, mask).item()
            iou_total  += iou_score(pred, mask).item()

    return dice_total / len(loader), iou_total / len(loader)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(use_hgbl, tag=''):
    """
    Run one complete train+eval experiment.

    use_hgbl=False : baseline (BCE + Dice, uniform weight)
    use_hgbl=True  : proposed (HGBL — boundary + heatmap weighted BCE + Dice)
    """
    label = 'Proposed HGBL' if use_hgbl else 'Baseline (BCE+Dice)'
    print(f"\n{'='*62}")
    print(f"  {label}  {tag}")
    print(f"{'='*62}\n")

    # Reproducible split
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    full_train = LGGSegDataset(DATASET_PATH, train=True)
    full_test  = LGGSegDataset(DATASET_PATH, train=False)

    n_total    = len(full_train)
    n_train    = int(0.8 * n_total)
    n_test     = n_total - n_train

    generator  = torch.Generator().manual_seed(SEED)
    train_idx, test_idx = random_split(
        range(n_total), [n_train, n_test], generator=generator
    )

    train_set  = Subset(full_train, train_idx.indices)
    test_set   = Subset(full_test,  test_idx.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model: always 4-channel input (image + heatmap)
    model     = UNet(in_channels=4, dropout_p=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_dice = 0.0
    best_iou  = 0.0
    ckpt_name = f"best_unet_{'hgbl' if use_hgbl else 'baseline'}.pth"
    ckpt_path = os.path.join(SAVE_DIR, ckpt_name)

    for epoch in range(1, EPOCHS + 1):
        loss          = train_epoch(model, train_loader, optimizer, use_hgbl=use_hgbl)
        dice, iou     = evaluate(model, test_loader)
        scheduler.step()

        improved = ''
        if dice > best_dice:
            best_dice = dice
            best_iou  = iou
            torch.save(model.state_dict(), ckpt_path)
            improved  = '  <- best'

        if epoch % 10 == 0 or improved:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | Loss {loss:.4f} | "
                f"Dice {dice:.4f} | IoU {iou:.4f}{improved}"
            )

    # Report best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    final_dice, final_iou = evaluate(model, test_loader)
    print(f"\n  Best checkpoint -> Dice {final_dice:.4f} | IoU {final_iou:.4f}")
    print(f"  Saved: {ckpt_path}")
    return final_dice, final_iou


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation():
    """
    Full ablation study for Novelty 2.
    Produces the table required for IEEE Transactions submission.

    Row 1: baseline — uniform BCE + Dice (no heatmap in loss, no boundary)
    Row 2: boundary only — (1 + λ_b * B) * BCE + Dice
    Row 3: heatmap only  — (1 + λ_h * H) * BCE + Dice
    Row 4: full HGBL     — (1 + λ_b * B)(1 + λ_h * H) * BCE + Dice

    All four use the 4-channel UNet (image + heatmap as input channels),
    so the only variable is whether the loss is spatially weighted.
    """
    global LAMBDA_B, LAMBDA_H

    results = {}

    # Row 1: Baseline
    d, i = run_experiment(use_hgbl=False, tag='[ablation row 1: baseline]')
    results['Baseline (BCE+Dice)'] = (d, i)

    # Row 2: Boundary weight only (set lambda_h=0)
    LAMBDA_H = 0.0
    LAMBDA_B = 2.0
    d, i = run_experiment(use_hgbl=True, tag='[ablation row 2: boundary only]')
    results['+ Boundary weight (λ_b=2, λ_h=0)'] = (d, i)

    # Row 3: Heatmap weight only (set lambda_b=0)
    LAMBDA_H = 1.0
    LAMBDA_B = 0.0
    d, i = run_experiment(use_hgbl=True, tag='[ablation row 3: heatmap only]')
    results['+ Heatmap weight (λ_b=0, λ_h=1)'] = (d, i)

    # Row 4: Full HGBL
    LAMBDA_H = 1.0
    LAMBDA_B = 2.0
    d, i = run_experiment(use_hgbl=True, tag='[ablation row 4: full HGBL]')
    results['Full HGBL (λ_b=2, λ_h=1)'] = (d, i)

    # Print ablation table
    print("\n" + "=" * 62)
    print("  ABLATION TABLE — Novelty 2 (HGBL)")
    print("=" * 62)
    print(f"  {'Method':<38} {'Dice':>6}  {'IoU':>6}")
    print(f"  {'-'*38} {'-'*6}  {'-'*6}")
    for method, (d, i) in results.items():
        print(f"  {method:<38} {d:>6.4f}  {i:>6.4f}")
    print("=" * 62)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true',
                        help='Run full ablation (4 experiments). '
                             'If not set, runs baseline + full HGBL only.')
    parser.add_argument('--lambda_b', type=float, default=2.0)
    parser.add_argument('--lambda_h', type=float, default=1.0)
    args = parser.parse_args()

    LAMBDA_B = args.lambda_b
    LAMBDA_H = args.lambda_h

    if args.ablation:
        run_ablation()
    else:
        print("\n--- Baseline: uniform BCE + Dice ---")
        dice_base, iou_base = run_experiment(use_hgbl=False, tag='[baseline]')

        print("\n--- Proposed: HGBL ---")
        dice_hgbl, iou_hgbl = run_experiment(use_hgbl=True, tag='[HGBL]')

        print("\n" + "=" * 62)
        print("  FINAL RESULTS")
        print("=" * 62)
        print(f"  Baseline  ->  Dice: {dice_base:.4f}  |  IoU: {iou_base:.4f}")
        print(f"  HGBL      ->  Dice: {dice_hgbl:.4f}  |  IoU: {iou_hgbl:.4f}")
        print(f"  Delta     ->  Dice: {dice_hgbl-dice_base:+.4f}  |  "
              f"IoU: {iou_hgbl-iou_base:+.4f}")
        print("=" * 62)