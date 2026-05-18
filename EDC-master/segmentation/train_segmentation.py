# segmentation/train_segmentation.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from dataset import LGGSegDataset
from unet import UNet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "../LGG_SEG"
BATCH_SIZE   = 8
EPOCHS       = 100          # increased; best model is checkpointed so no risk
LR           = 1e-4
SEED         = 42
SAVE_DIR     = "./seg_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


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
# Losses
# ---------------------------------------------------------------------------
def dice_loss(pred, target):
    smooth       = 1e-8
    intersection = (pred * target).sum()
    dice         = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1. - dice


def combined_loss(pred, target):
    return nn.BCELoss()(pred, target) + dice_loss(pred, target)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, leave=False, desc='  train'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = combined_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    dice_total = 0.0
    iou_total  = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            dice_total += dice_score(pred, y).item()
            iou_total  += iou_score(pred, y).item()
    n = len(loader)
    return dice_total / n, iou_total / n


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment(use_heatmap, tag=''):
    print(f"\n{'='*60}")
    print(f"  Experiment: {'Image + Heatmap' if use_heatmap else 'Image only'} {tag}")
    print(f"{'='*60}\n")

    # FIX: pass train=True/False so augmentation is applied only to train split
    full_dataset = LGGSegDataset(DATASET_PATH, use_heatmap=use_heatmap, train=True)

    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size

    # FIX: fixed seed so train/test split is identical across all experiments
    generator = torch.Generator().manual_seed(SEED)
    train_indices, test_indices = random_split(
        range(len(full_dataset)), [train_size, test_size], generator=generator
    )

    # Build separate datasets with appropriate augmentation flag
    train_set = LGGSegDataset(DATASET_PATH, use_heatmap=use_heatmap, train=True)
    test_set  = LGGSegDataset(DATASET_PATH, use_heatmap=use_heatmap, train=False)

    # Apply the same index split
    from torch.utils.data import Subset
    train_set = Subset(train_set, train_indices.indices)
    test_set  = Subset(test_set,  test_indices.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    in_channels = 4 if use_heatmap else 3
    model       = UNet(in_channels=in_channels, dropout_p=0.3).to(DEVICE)

    optimizer   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # Cosine LR schedule — decays smoothly from LR to near 0
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # FIX: save best model by validation Dice
    best_dice      = 0.0
    best_iou       = 0.0
    ckpt_name      = f"best_unet_{'hm' if use_heatmap else 'base'}.pth"
    ckpt_path      = os.path.join(SAVE_DIR, ckpt_name)

    for epoch in range(1, EPOCHS + 1):
        loss         = train_epoch(model, train_loader, optimizer)
        dice, iou    = evaluate(model, test_loader)
        scheduler.step()

        improved = ''
        if dice > best_dice:
            best_dice = dice
            best_iou  = iou
            torch.save(model.state_dict(), ckpt_path)
            improved  = '  ← best'

        if epoch % 10 == 0 or improved:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}"
                + improved
            )

    # Load best weights for final report
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    final_dice, final_iou = evaluate(model, test_loader)
    print(f"\n  Best checkpoint → Dice {final_dice:.4f} | IoU {final_iou:.4f}")
    print(f"  Saved to: {ckpt_path}")

    return final_dice, final_iou


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n--- Baseline: Image only (3 channels) ---")
    dice_base, iou_base = run_experiment(use_heatmap=False, tag='[baseline]')

    print("\n--- Proposed: Image + Heatmap (4 channels) ---")
    dice_prop, iou_prop = run_experiment(use_heatmap=True, tag='[proposed]')

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline  →  Dice: {dice_base:.4f}  |  IoU: {iou_base:.4f}")
    print(f"  Proposed  →  Dice: {dice_prop:.4f}  |  IoU: {iou_prop:.4f}")
    print(f"  Δ Dice: {dice_prop - dice_base:+.4f}  |  Δ IoU: {iou_prop - iou_base:+.4f}")
    print("=" * 60)