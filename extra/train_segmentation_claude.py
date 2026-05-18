# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import cv2

# from dataset import LGGSegDataset
# from unet import UNet


# # ---------------------------
# # Reproducibility
# # ---------------------------

# SEED = 42

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(SEED)


# # ---------------------------
# # Config
# # ---------------------------

# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# DATASET_PATH = "../LGG_SEG"
# BATCH_SIZE   = 8
# EPOCHS       = 30
# LR           = 1e-4
# TRAIN_RATIO  = 0.8


# # ---------------------------
# # Shared index split
# # ---------------------------

# def make_split_indices(total_size, train_ratio=0.8, seed=42):
#     indices = list(range(total_size))
#     rng = random.Random(seed)
#     rng.shuffle(indices)
#     split = int(train_ratio * total_size)
#     return indices[:split], indices[split:]


# # ---------------------------
# # Metrics  —  true per-sample average
# # ---------------------------

# def compute_metrics(pred, target):
#     """
#     pred, target : (B, 1, H, W) — pred is raw sigmoid output in [0,1]
#     Returns (mean_dice, mean_iou) as Python floats, averaged per sample.
#     """
#     pred_bin = (pred > 0.5).float()
#     dims = (1, 2, 3)

#     intersection = (pred_bin * target).sum(dim=dims)
#     denom        = pred_bin.sum(dim=dims) + target.sum(dim=dims)
#     union        = denom - intersection

#     dice = (2.0 * intersection + 1e-8) / (denom + 1e-8)
#     iou  = (intersection + 1e-8) / (union + 1e-8)

#     return dice.mean().item(), iou.mean().item()


# # ---------------------------
# # Loss
# # ---------------------------

# def dice_loss(pred, target):
#     smooth = 1e-8
#     intersection = (pred * target).sum()
#     return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# bce_fn = nn.BCELoss()

# def combined_loss(pred, target):
#     return bce_fn(pred, target) + dice_loss(pred, target)


# # ---------------------------
# # Training loop
# # ---------------------------

# def train_epoch(model, loader, optimizer):
#     model.train()
#     total_loss = 0.0

#     for x, y in tqdm(loader, leave=False):
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         optimizer.zero_grad()
#         pred = model(x)
#         loss = combined_loss(pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(loader)


# # ---------------------------
# # Evaluation loop
# # ---------------------------

# def evaluate(model, loader):
#     model.eval()
#     all_dice, all_iou = [], []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = model(x)
#             d, i = compute_metrics(pred, y)
#             bs = x.shape[0]
#             all_dice.append((d, bs))
#             all_iou.append((i, bs))

#     total     = sum(bs for _, bs in all_dice)
#     mean_dice = sum(d * bs for d, bs in all_dice) / total
#     mean_iou  = sum(i * bs for i, bs in all_iou)  / total
#     return mean_dice, mean_iou


# # ---------------------------
# # Experiment runner
# # ---------------------------

# def run_experiment(
#     label,
#     use_heatmap=False,
#     use_pseudo=False,
#     pseudo_mode='soft',
#     pseudo_threshold=0.5,
#     save_dir=None,
#     mode=None,
# ):
#     print(f"\n{'='*55}")
#     print(f"  {label}")
#     print(f"{'='*55}")

#     # Both dataset objects share skip_empty=True so file lists are identical
#     # and our single shared index split is valid across both.
#     train_dataset = LGGSegDataset(
#         DATASET_PATH,
#         use_heatmap=use_heatmap,
#         use_pseudo=use_pseudo,
#         eval_mode=False,
#         pseudo_mode=pseudo_mode,
#         pseudo_threshold=pseudo_threshold,
#         skip_empty=True,
#     )
#     test_dataset = LGGSegDataset(
#         DATASET_PATH,
#         use_heatmap=use_heatmap,
#         use_pseudo=use_pseudo,
#         eval_mode=True,           # always real GT at test time
#         pseudo_mode=pseudo_mode,
#         pseudo_threshold=pseudo_threshold,
#         skip_empty=True,
#     )

#     # Single shared split
#     train_idx, test_idx = make_split_indices(len(train_dataset), TRAIN_RATIO, seed=SEED)

#     train_set = Subset(train_dataset, train_idx)
#     test_set  = Subset(test_dataset,  test_idx)

#     print(f"  Train: {len(train_set)}  |  Test: {len(test_set)}")

#     train_loader = DataLoader(
#         train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
#     )

#     in_channels = 4 if use_heatmap else 3
#     model = UNet(in_channels).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     best_dice, best_iou = 0.0, 0.0

#     for epoch in range(EPOCHS):
#         loss      = train_epoch(model, train_loader, optimizer)
#         dice, iou = evaluate(model, test_loader)

#         if dice > best_dice:
#             best_dice = dice
#             best_iou  = iou

#         print(
#             f"  Epoch {epoch+1:3d}/{EPOCHS} | "
#             f"Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}"
#         )

#     print(f"\n  >> Best Dice: {best_dice:.4f}  Best IoU: {best_iou:.4f}")

#     if save_dir is not None and mode is not None:
#         save_grid_predictions(model, test_loader, DEVICE, save_dir, mode)

#     return best_dice, best_iou


# # ---------------------------
# # Visualisation
# # ---------------------------

# def save_grid_predictions(model, loader, device, save_dir, mode):
#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)
#     results = []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             pred     = model(x)
#             pred_bin = (pred > 0.5).float()

#             dims         = (1, 2, 3)
#             intersection = (pred_bin * y).sum(dim=dims)
#             denom        = pred_bin.sum(dim=dims) + y.sum(dim=dims)
#             union        = denom - intersection
#             batch_dice   = (2.0 * intersection) / (denom + 1e-8)
#             batch_iou    = intersection / (union + 1e-8)

#             for i in range(x.shape[0]):
#                 results.append({
#                     "dice":  batch_dice[i].item(),
#                     "iou":   batch_iou[i].item(),
#                     "image": x[i].cpu(),
#                     "pred":  pred_bin[i].cpu(),
#                     "gt":    y[i].cpu(),
#                 })

#     results.sort(key=lambda r: r["dice"], reverse=True)

#     mid = len(results) // 2
#     rng = random.Random(SEED)
#     selected = [
#         rng.choice(results[:3]),
#         rng.choice(results[max(0, mid-1): mid+2]),
#         rng.choice(results[-3:]),
#     ]

#     def add_label(img, text):
#         out = img.copy()
#         cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (255, 255, 255), 2)
#         return out

#     for idx, s in enumerate(selected):
#         img_np  = s["image"].numpy()
#         pred_np = s["pred"].numpy()[0]
#         gt_np   = s["gt"].numpy()[0]

#         img_rgb  = (np.transpose(img_np[:3], (1, 2, 0)) * 255).astype(np.uint8)
#         pred_img = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#         gt_img   = cv2.cvtColor((gt_np   * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         panels = [add_label(img_rgb, "Image")]

#         if mode == "baseline":
#             panels += [add_label(pred_img, "Prediction"),
#                        add_label(gt_img,   "Ground Truth")]

#         elif mode == "heatmap_guided":
#             hm_gray = cv2.cvtColor(
#                 (img_np[3] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
#             )
#             panels += [add_label(hm_gray,  "Heatmap"),
#                        add_label(pred_img, "Prediction"),
#                        add_label(gt_img,   "Ground Truth")]

#         elif mode in ("pseudo_soft", "pseudo_binary"):
#             # At test time y is always real GT (eval_mode=True) — label it correctly
#             panels += [add_label(pred_img, "Prediction"),
#                        add_label(gt_img,   "Real GT")]

#         grid = np.concatenate(panels, axis=1)
#         cv2.putText(grid,
#                     f"Dice: {s['dice']:.4f} | IoU: {s['iou']:.4f}",
#                     (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         save_path = os.path.join(save_dir, f"sample_{idx}.png")
#         cv2.imwrite(save_path, grid)
#         print(f"  Saved {save_path}")


# # ---------------------------
# # Entry point
# # ---------------------------

# if __name__ == "__main__":

#     # Approach 1: Baseline
#     dice1, iou1 = run_experiment(
#         label       = "Approach 1: Image → Real Mask (Baseline)",
#         use_heatmap = False,
#         use_pseudo  = False,
#         save_dir    = "outputs/baseline",
#         mode        = "baseline",
#     )

#     # Approach 2: Heatmap-guided
#     dice2, iou2 = run_experiment(
#         label       = "Approach 2: Image + Heatmap → Real Mask",
#         use_heatmap = True,
#         use_pseudo  = False,
#         save_dir    = "outputs/heatmap_guided",
#         mode        = "heatmap_guided",
#     )

#     # Approach 3a: Pseudo soft labels
#     dice3s, iou3s = run_experiment(
#         label       = "Approach 3 (soft): Image → Heatmap as soft GT",
#         use_heatmap = False,
#         use_pseudo  = True,
#         pseudo_mode = 'soft',
#         save_dir    = "outputs/pseudo_soft",
#         mode        = "pseudo_soft",
#     )

#     # Approach 3b: Pseudo binary labels
#     dice3b, iou3b = run_experiment(
#         label            = "Approach 3 (binary t=0.5): Image → Binarised Heatmap GT",
#         use_heatmap      = False,
#         use_pseudo       = True,
#         pseudo_mode      = 'binary',
#         pseudo_threshold = 0.5,
#         save_dir         = "outputs/pseudo_binary",
#         mode             = "pseudo_binary",
#     )

#     # Summary
#     print("\n" + "="*55)
#     print("  FINAL RESULTS (best Dice over all epochs)")
#     print("="*55)
#     print(f"  Approach 1  (Baseline)           Dice: {dice1:.4f}  IoU: {iou1:.4f}")
#     print(f"  Approach 2  (Heatmap-guided)     Dice: {dice2:.4f}  IoU: {iou2:.4f}")
#     print(f"  Approach 3a (Pseudo soft)         Dice: {dice3s:.4f}  IoU: {iou3s:.4f}")
#     print(f"  Approach 3b (Pseudo binary 0.5)  Dice: {dice3b:.4f}  IoU: {iou3b:.4f}")

"""
train_segmentation.py
---------------------
Approach 1 : Image → Real mask          (fully supervised baseline)
Approach 2 : Image + Heatmap → Real mask (heatmap-guided supervised)
Approach 3 : Image → Heatmap pseudo GT  (annotation-free)

Key facts from diagnose_heatmap.py:
  - Best heatmap-mask Dice (any threshold): 0.2197 at t=0.60
  - GT pixel coverage: 2.95%  |  Heatmap at t=0.60: 11%
  - Heatmaps are spatially imprecise (3-4x over-coverage)
  - Soft labels recommended over binary

Approach 3 is evaluated TWO ways:
  1. vs pseudo GT  : did the model learn the heatmap signal?
  2. vs real GT    : how far is annotation-free from supervised?

The "raw heatmap Dice" (0.2197) is reported as a no-learning baseline.
Any Approach 3 result above this is a genuine contribution.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import cv2

from dataset import LGGSegDataset
from unet import UNet


# ---------------------------
# Reproducibility
# ---------------------------

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ---------------------------
# Config
# ---------------------------

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH      = "../LGG_SEG"
BATCH_SIZE        = 8
EPOCHS            = 50
LR                = 1e-4
TRAIN_RATIO       = 0.8

# From diagnose_heatmap.py: best pixel-level Dice vs real mask is at t=0.60
# This is also used for pseudo eval mode (testing against heatmap GT)
PSEUDO_THRESHOLD  = 0.60

# Raw heatmap Dice from diagnose_heatmap.py — reported in paper as no-learning baseline
RAW_HEATMAP_DICE  = 0.2197
RAW_HEATMAP_IOU   = 0.1351


# ---------------------------
# Index split
# ---------------------------

def make_split_indices(total_size, train_ratio=0.8, seed=42):
    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(train_ratio * total_size)
    return indices[:split], indices[split:]


# ---------------------------
# Post-processing: keep largest connected component
# ---------------------------

def keep_largest_component(pred_np):
    """
    pred_np : (H, W) binary numpy array, values 0 or 1.
    Returns : (H, W) binary array with only the largest connected component kept.

    Rationale: heatmaps over-cover (11% vs 2.95% GT coverage). The UNet
    may produce diffuse multi-blob predictions. Keeping only the largest
    component sharpens the prediction toward the real tumour region and
    reduces false positive area, improving Dice against real masks.
    """
    pred_uint8 = (pred_np * 255).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred_uint8, connectivity=8
    )
    if n_labels <= 1:
        return pred_np  # nothing or only background

    # stats[0] is background — skip it, find largest foreground component
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.float32)


def apply_largest_component_batch(pred_bin_tensor):
    """
    pred_bin_tensor : (B, 1, H, W) binary tensor on CPU.
    Returns         : (B, 1, H, W) binary tensor after largest-component filtering.
    """
    out = []
    arr = pred_bin_tensor.numpy()
    for i in range(arr.shape[0]):
        filtered = keep_largest_component(arr[i, 0])
        out.append(filtered[np.newaxis])
    return torch.tensor(np.stack(out, axis=0))


# ---------------------------
# Metrics
# ---------------------------

def compute_metrics_batch(pred_bin, target):
    """
    pred_bin : (B, 1, H, W) binary float tensor
    target   : (B, 1, H, W) binary float tensor
    Returns  : (mean_dice, mean_iou) as Python floats
    """
    dims = (1, 2, 3)
    intersection = (pred_bin * target).sum(dim=dims)
    denom        = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    union        = denom - intersection
    dice = ((2.0 * intersection + 1e-8) / (denom + 1e-8))
    iou  = ((intersection + 1e-8) / (union + 1e-8))
    return dice.mean().item(), iou.mean().item()


def weighted_mean(score_bs_list):
    total = sum(bs for _, bs in score_bs_list)
    return sum(s * bs for s, bs in score_bs_list) / total


# ---------------------------
# Loss
# ---------------------------

def dice_loss(pred, target):
    smooth = 1e-8
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)

bce_fn = nn.BCELoss()

def combined_loss(pred, target):
    return bce_fn(pred, target) + dice_loss(pred, target)


# ---------------------------
# Training loop
# ---------------------------

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------
# Evaluation
# ---------------------------

def evaluate(model, loader, use_postproc=False):
    """
    Returns (mean_dice, mean_iou).
    If use_postproc=True, applies largest-component filtering before scoring.
    """
    model.eval()
    dice_acc, iou_acc = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred     = model(x)
            pred_bin = (pred > 0.5).float()

            if use_postproc:
                pred_bin = apply_largest_component_batch(
                    pred_bin.cpu()
                ).to(DEVICE)

            d, i = compute_metrics_batch(pred_bin, y)
            bs = x.shape[0]
            dice_acc.append((d, bs))
            iou_acc.append((i, bs))

    return weighted_mean(dice_acc), weighted_mean(iou_acc)


# ---------------------------
# Dataset / loader builder
# ---------------------------

def build_loaders(use_heatmap, use_pseudo, pseudo_mode, eval_mode_test):
    train_ds = LGGSegDataset(
        DATASET_PATH,
        use_heatmap=use_heatmap,
        use_pseudo=use_pseudo,
        eval_mode='none',
        pseudo_mode=pseudo_mode,
        pseudo_threshold=PSEUDO_THRESHOLD,
        skip_empty=True,
    )
    test_ds = LGGSegDataset(
        DATASET_PATH,
        use_heatmap=use_heatmap,
        use_pseudo=use_pseudo,
        eval_mode=eval_mode_test,
        pseudo_mode=pseudo_mode,
        pseudo_threshold=PSEUDO_THRESHOLD,
        skip_empty=True,
    )

    train_idx, test_idx = make_split_indices(len(train_ds), TRAIN_RATIO, SEED)
    train_set = Subset(train_ds, train_idx)
    test_set  = Subset(test_ds,  test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader, len(train_set), len(test_set)


# ---------------------------
# Experiment runner
# ---------------------------

def run_experiment(label, use_heatmap, use_pseudo, pseudo_mode='soft',
                   eval_mode_test='real', save_dir=None, vis_mode=None):

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    train_loader, test_loader, n_train, n_test = build_loaders(
        use_heatmap, use_pseudo, pseudo_mode, eval_mode_test
    )
    print(f"  Train: {n_train}  |  Test: {n_test}  |  Eval GT: {eval_mode_test}")

    model     = UNet(4 if use_heatmap else 3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice, best_iou = 0.0, 0.0

    for epoch in range(EPOCHS):
        loss      = train_epoch(model, train_loader, optimizer)
        dice, iou = evaluate(model, test_loader, use_postproc=False)

        if dice > best_dice:
            best_dice, best_iou = dice, iou

        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}")

    print(f"\n  >> Best Dice: {best_dice:.4f}  Best IoU: {best_iou:.4f}")

    if save_dir and vis_mode:
        save_grid_predictions(model, test_loader, save_dir, vis_mode)

    return model, best_dice, best_iou


# ---------------------------
# Visualisation
# ---------------------------

def save_grid_predictions(model, loader, save_dir, vis_mode, use_postproc=False):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    results = []

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            pred    = model(x)
            pb      = (pred > 0.5).float()
            if use_postproc:
                pb = apply_largest_component_batch(pb.cpu()).to(DEVICE)

            dims  = (1, 2, 3)
            inter = (pb * y).sum(dim=dims)
            denom = pb.sum(dim=dims) + y.sum(dim=dims)
            union = denom - inter
            d_b   = (2.0 * inter) / (denom + 1e-8)
            i_b   = inter / (union + 1e-8)

            for i in range(x.shape[0]):
                results.append({
                    "dice":  d_b[i].item(), "iou": i_b[i].item(),
                    "image": x[i].cpu().numpy(),
                    "pred":  pb[i].cpu().numpy()[0],
                    "gt":    y[i].cpu().numpy()[0],
                })

    results.sort(key=lambda r: r["dice"], reverse=True)
    mid = len(results) // 2
    rng = random.Random(SEED)
    selected = [
        rng.choice(results[:3]),
        rng.choice(results[max(0, mid-1): mid+2]),
        rng.choice(results[-3:]),
    ]

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        return out

    for idx, s in enumerate(selected):
        img_np  = s["image"]
        pred_np = s["pred"]
        gt_np   = s["gt"]

        img_rgb  = (np.transpose(img_np[:3], (1, 2, 0)) * 255).astype(np.uint8)
        pred_img = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        gt_img   = cv2.cvtColor((gt_np   * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        panels = [add_label(img_rgb, "Image")]

        if vis_mode == "baseline":
            panels += [add_label(pred_img, "Prediction"),
                       add_label(gt_img,   "Ground Truth")]
        elif vis_mode == "heatmap_guided":
            hm_bgr = cv2.cvtColor(
                (img_np[3] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            panels += [add_label(hm_bgr,   "Heatmap"),
                       add_label(pred_img, "Prediction"),
                       add_label(gt_img,   "Ground Truth")]
        elif vis_mode == "pseudo_vs_pseudo":
            panels += [add_label(pred_img, "Prediction"),
                       add_label(gt_img,   "Pseudo GT (heatmap)")]
        elif vis_mode == "pseudo_vs_real":
            panels += [add_label(pred_img, "Prediction"),
                       add_label(gt_img,   "Real GT")]

        grid = np.concatenate(panels, axis=1)
        cv2.putText(grid, f"Dice: {s['dice']:.4f} | IoU: {s['iou']:.4f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(save_dir, f"sample_{idx}.png"), grid)
        print(f"  Saved sample_{idx}.png")


# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":

    # Approach 1: Baseline
    _, dice1, iou1 = run_experiment(
        label="Approach 1: Image → Real Mask (Baseline)",
        use_heatmap=False, use_pseudo=False,
        eval_mode_test='real',
        save_dir="outputs/baseline", vis_mode="baseline",
    )

    # Approach 2: Heatmap-guided
    _, dice2, iou2 = run_experiment(
        label="Approach 2: Image + Heatmap → Real Mask",
        use_heatmap=True, use_pseudo=False,
        eval_mode_test='real',
        save_dir="outputs/heatmap_guided", vis_mode="heatmap_guided",
    )

    # Approach 3 — train once, evaluate twice
    # 3a: vs pseudo GT (proves the model learned the heatmap signal)
    model3, dice3_pp, iou3_pp = run_experiment(
        label="Approach 3 (soft labels): vs Pseudo GT",
        use_heatmap=False, use_pseudo=True, pseudo_mode='soft',
        eval_mode_test='pseudo',
        save_dir="outputs/pseudo_vs_pseudo", vis_mode="pseudo_vs_pseudo",
    )

    # 3b: same trained model, now evaluated vs real GT
    # Re-build test loader with eval_mode='real', same split
    _, test_loader_real, _, _ = build_loaders(
        use_heatmap=False, use_pseudo=True,
        pseudo_mode='soft', eval_mode_test='real',
    )
    dice3_pr, iou3_pr = evaluate(model3, test_loader_real, use_postproc=False)
    # Also evaluate with largest-component post-processing
    dice3_pr_pp, iou3_pr_pp = evaluate(model3, test_loader_real, use_postproc=True)
    save_grid_predictions(model3, test_loader_real,
                          "outputs/pseudo_vs_real", "pseudo_vs_real")

    # Final table
    print("\n" + "="*65)
    print("  FINAL RESULTS  (for paper table)")
    print("="*65)
    print(f"  Raw heatmap (no model, t=0.60)          "
          f"Dice: {RAW_HEATMAP_DICE:.4f}  IoU: {RAW_HEATMAP_IOU:.4f}  ← no-learning baseline")
    print()
    print(f"  Approach 1  (Baseline)                  "
          f"Dice: {dice1:.4f}  IoU: {iou1:.4f}")
    print(f"  Approach 2  (Heatmap-guided)             "
          f"Dice: {dice2:.4f}  IoU: {iou2:.4f}")
    print()
    print(f"  Approach 3  vs Pseudo GT                 "
          f"Dice: {dice3_pp:.4f}  IoU: {iou3_pp:.4f}  ← model learned heatmap")
    print(f"  Approach 3  vs Real GT (no postproc)     "
          f"Dice: {dice3_pr:.4f}  IoU: {iou3_pr:.4f}  ← annotation-free quality")
    print(f"  Approach 3  vs Real GT (+largest comp.)  "
          f"Dice: {dice3_pr_pp:.4f}  IoU: {iou3_pr_pp:.4f}  ← with post-processing")
    print()
    gap = dice1 - dice3_pr
    lift = dice3_pr - RAW_HEATMAP_DICE
    print(f"  Model lift over raw heatmap:  +{lift:.4f} Dice")
    print(f"  Gap vs full supervision:       {gap:.4f} Dice")
    print("="*65)