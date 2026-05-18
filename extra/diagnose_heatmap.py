"""
diagnose_heatmap.py
--------------------
Run this BEFORE tuning the threshold.
It measures how well the EDC heatmaps align with the real masks
across a range of thresholds, and prints the optimal one.

Usage (from inside EDC-master/segmentation/):
    python diagnose_heatmap.py

It reads directly from LGG_SEG so no extra setup is needed.
"""

import os
import cv2
import numpy as np


DATASET_PATH  = "../LGG_SEG"
IMAGE_DIR     = os.path.join(DATASET_PATH, "images")
HEATMAP_DIR   = os.path.join(DATASET_PATH, "heatmaps")
MASK_DIR      = os.path.join(DATASET_PATH, "masks")

THRESHOLDS    = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def dice_np(pred_bin, gt_bin):
    intersection = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0   # both empty — perfect match
    return (2.0 * intersection + 1e-8) / (denom + 1e-8)


def iou_np(pred_bin, gt_bin):
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection + 1e-8) / (union + 1e-8)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    files = sorted(os.listdir(IMAGE_DIR))

    # Filter to non-empty masks only (same as skip_empty=True in dataset)
    valid_files = []
    for name in files:
        mask = cv2.imread(os.path.join(MASK_DIR, name), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            valid_files.append(name)

    print(f"Non-empty samples found: {len(valid_files)}\n")

    # Per-threshold accumulators
    results = {t: {"dice": [], "iou": [], "hm_coverage": [], "gt_coverage": []} 
               for t in THRESHOLDS}

    # Also track raw heatmap stats (independent of threshold)
    hm_means, hm_maxs = [], []

    for name in valid_files:
        # Load and resize mask
        mask = cv2.imread(os.path.join(MASK_DIR, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        gt_bin = (mask > 0).astype(np.float32)

        # Load and resize heatmap
        hm_name = name.replace(".tif", ".png")
        hm = cv2.imread(os.path.join(HEATMAP_DIR, hm_name), cv2.IMREAD_GRAYSCALE)
        if hm is None:
            print(f"  WARNING: missing heatmap for {name}, skipping.")
            continue
        hm = cv2.resize(hm, (256, 256))
        hm_norm = hm.astype(np.float32) / 255.0

        hm_means.append(hm_norm.mean())
        hm_maxs.append(hm_norm.max())

        gt_coverage = gt_bin.mean()   # fraction of pixels that are tumour in real mask

        for t in THRESHOLDS:
            pred_bin = (hm_norm > t).astype(np.float32)
            hm_coverage = pred_bin.mean()

            d = dice_np(pred_bin, gt_bin)
            i = iou_np(pred_bin, gt_bin)

            results[t]["dice"].append(d)
            results[t]["iou"].append(i)
            results[t]["hm_coverage"].append(hm_coverage)
            results[t]["gt_coverage"].append(gt_coverage)

    # ------------------------------------------------------------------
    # Print heatmap stats
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  HEATMAP RAW STATISTICS  (across all non-empty samples)")
    print("=" * 60)
    print(f"  Mean pixel value (normalised) : {np.mean(hm_means):.4f}")
    print(f"  Mean max  value  (normalised) : {np.mean(hm_maxs):.4f}")
    print()
    print("  Interpretation:")
    if np.mean(hm_means) < 0.15:
        print("  → Heatmaps are SPARSE / weak activations.")
        print("    Use a LOW threshold (0.1–0.2).")
    elif np.mean(hm_means) > 0.35:
        print("  → Heatmaps are DIFFUSE / broad activations.")
        print("    Use a HIGH threshold (0.4–0.6).")
    else:
        print("  → Heatmaps have MODERATE activations.")
        print("    Threshold 0.2–0.4 range is the right search space.")

    # ------------------------------------------------------------------
    # Print per-threshold results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  THRESHOLD vs DICE/IoU vs COVERAGE")
    print("=" * 60)
    print(f"  {'Threshold':>10} | {'Dice':>8} | {'IoU':>8} | "
          f"{'HM Cover%':>10} | {'GT Cover%':>10}")
    print("  " + "-" * 56)

    best_dice_t  = None
    best_dice_val = -1.0

    for t in THRESHOLDS:
        mean_dice = np.mean(results[t]["dice"])
        mean_iou  = np.mean(results[t]["iou"])
        mean_hm   = np.mean(results[t]["hm_coverage"]) * 100
        mean_gt   = np.mean(results[t]["gt_coverage"]) * 100

        marker = ""
        if mean_dice > best_dice_val:
            best_dice_val = mean_dice
            best_dice_t   = t
            marker = "  ← best"

        print(f"  {t:>10.2f} | {mean_dice:>8.4f} | {mean_iou:>8.4f} | "
              f"{mean_hm:>9.2f}% | {mean_gt:>9.2f}%{marker}")

    print()
    print(f"  RECOMMENDED THRESHOLD: {best_dice_t}")
    print(f"  Heatmap-mask Dice at this threshold: {best_dice_val:.4f}")
    print()

    # ------------------------------------------------------------------
    # Interpretation of the best threshold Dice
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  WHAT THIS MEANS FOR YOUR PAPER")
    print("=" * 60)
    if best_dice_val >= 0.60:
        print("  Heatmap quality is GOOD (Dice >= 0.60).")
        print("  Approach 3 has a strong ceiling. Expect competitive results.")
    elif best_dice_val >= 0.40:
        print("  Heatmap quality is MODERATE (Dice 0.40–0.60).")
        print("  Approach 3 will trail Approach 1 but should be reportable.")
        print("  Consider framing as 'annotation-free approximation'.")
    else:
        print("  Heatmap quality is WEAK (Dice < 0.40).")
        print("  Binary pseudo labels will hurt. Use SOFT labels instead.")
        print("  Soft labels tolerate imprecise heatmaps much better.")


if __name__ == "__main__":
    main()