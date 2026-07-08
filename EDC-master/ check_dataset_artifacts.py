"""
Dataset Artifact / Shortcut Checker
-----------------------------------
Checks whether a dataset has the same problem the EDC paper (Fig. 5) found in
Chest X-ray and HyperKvasir: medically-irrelevant visual differences between
normal and abnormal images (calibration marks, corner color, different capture
devices) that a model could exploit as a shortcut instead of learning real
pathology.

Usage:
    python check_dataset_artifacts.py --normal_dir /path/to/normal \
                                       --abnormal_dir /path/to/abnormal \
                                       --dataset_name BUSI \
                                       --out_dir ./artifact_check_output

Requires: numpy, pillow, scikit-learn, scipy, matplotlib
    pip install numpy pillow scikit-learn scipy matplotlib --break-system-packages
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS])


def load_resized(path, size=(256, 256)):
    img = Image.open(path).convert('RGB').resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


def corner_patches(img, patch_frac=0.12):
    """Extract 4 corner patches (top-left, top-right, bottom-left, bottom-right)."""
    h, w, _ = img.shape
    ph, pw = int(h * patch_frac), int(w * patch_frac)
    corners = {
        'top_left': img[:ph, :pw],
        'top_right': img[:ph, -pw:],
        'bottom_left': img[-ph:, :pw],
        'bottom_right': img[-ph:, -pw:],
    }
    return corners


def corner_mean_color(img, patch_frac=0.12):
    """Return a flat vector of mean RGB per corner (12 values total)."""
    corners = corner_patches(img, patch_frac)
    vals = []
    for _, patch in corners.items():
        vals.extend(patch.reshape(-1, 3).mean(axis=0).tolist())
    return np.array(vals)  # length 12: 4 corners x 3 channels


def border_only_features(img, patch_frac=0.12):
    """Features from ONLY the border ring, deliberately excluding the center
    where real pathology would live. Used for the shortcut-classifier test."""
    corners = corner_patches(img, patch_frac)
    feats = []
    for _, patch in corners.items():
        feats.append(patch.reshape(-1, 3).mean(axis=0))
        feats.append(patch.reshape(-1, 3).std(axis=0))
    return np.concatenate(feats)  # 4 corners x (mean+std) x 3 channels = 24 dims


def run_visual_grid(normal_paths, abnormal_paths, out_dir, dataset_name, n=6):
    n = min(n, len(normal_paths), len(abnormal_paths))
    sample_normal = random.sample(normal_paths, n)
    sample_abnormal = random.sample(abnormal_paths, n)

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for i, p in enumerate(sample_normal):
        axes[0, i].imshow(Image.open(p).convert('RGB'))
        axes[0, i].axis('off')
    for i, p in enumerate(sample_abnormal):
        axes[1, i].imshow(Image.open(p).convert('RGB'))
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Normal', fontsize=11)
    axes[1, 0].set_ylabel('Abnormal', fontsize=11)
    fig.suptitle(f'{dataset_name}: random normal vs. abnormal samples '
                 '(inspect corners/borders/labels by eye)', fontsize=11)
    plt.tight_layout()
    out_path = Path(out_dir) / f'{dataset_name}_visual_grid.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[1/3] Visual grid saved -> {out_path}")


def run_corner_stats(normal_paths, abnormal_paths, out_dir, dataset_name, n=200):
    n_normal = min(n, len(normal_paths))
    n_abnormal = min(n, len(abnormal_paths))
    sample_normal = random.sample(normal_paths, n_normal)
    sample_abnormal = random.sample(abnormal_paths, n_abnormal)

    normal_vecs = np.array([corner_mean_color(load_resized(p)) for p in sample_normal])
    abnormal_vecs = np.array([corner_mean_color(load_resized(p)) for p in sample_abnormal])

    corner_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    channel_names = ['R', 'G', 'B']

    print(f"\n[2/3] Corner color statistics ({dataset_name}), "
          f"n_normal={n_normal}, n_abnormal={n_abnormal}")
    print(f"{'corner':<14}{'channel':<8}{'normal_mean':<14}{'abnormal_mean':<16}"
          f"{'t-stat':<10}{'p-value':<10}{'flag'}")

    flagged = []
    idx = 0
    for corner in corner_names:
        for ch in channel_names:
            n_vals = normal_vecs[:, idx]
            a_vals = abnormal_vecs[:, idx]
            t_stat, p_val = stats.ttest_ind(n_vals, a_vals, equal_var=False)
            flag = "<-- CHECK" if p_val < 0.001 and abs(n_vals.mean() - a_vals.mean()) > 0.02 else ""
            if flag:
                flagged.append((corner, ch, p_val))
            print(f"{corner:<14}{ch:<8}{n_vals.mean():<14.4f}{a_vals.mean():<16.4f}"
                  f"{t_stat:<10.2f}{p_val:<10.2e}{flag}")
            idx += 1

    if flagged:
        print(f"\n  {len(flagged)} corner/channel combination(s) show a statistically "
              f"significant AND visually meaningful difference (p<0.001, |diff|>0.02).")
        print("  This matches the pattern EDC's Fig. 5 flagged in HyperKvasir — worth a manual look.")
    else:
        print("\n  No corner/channel combination shows both a significant and "
              "meaningful difference. No obvious corner-color shortcut detected.")

    return flagged


def run_shortcut_classifier(normal_paths, abnormal_paths, out_dir, dataset_name, n=200, seed=0):
    n_normal = min(n, len(normal_paths))
    n_abnormal = min(n, len(abnormal_paths))
    random.seed(seed)
    sample_normal = random.sample(normal_paths, n_normal)
    sample_abnormal = random.sample(abnormal_paths, n_abnormal)

    X, y = [], []
    for p in sample_normal:
        X.append(border_only_features(load_resized(p)))
        y.append(0)
    for p in sample_abnormal:
        X.append(border_only_features(load_resized(p)))
        y.append(1)
    X = np.array(X)
    y = np.array(y)

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    mean_auc = scores.mean()

    print(f"\n[3/3] Shortcut classifier test ({dataset_name})")
    print(f"  Trained on BORDER-ONLY features (center/lesion region excluded).")
    print(f"  5-fold cross-validated AUC: {mean_auc:.4f} (per-fold: {np.round(scores, 4)})")
    if mean_auc > 0.70:
        print(f"  --> HIGH. A classifier using only border pixels (no lesion content) "
              f"achieves {mean_auc:.2f} AUC. This strongly suggests a shortcut artifact, "
              f"similar to the Chest X-ray / HyperKvasir issue in the EDC paper.")
    elif mean_auc > 0.60:
        print(f"  --> MODERATE. Some border signal exists ({mean_auc:.2f} AUC). Worth a "
              f"manual look, though this could also be a mild, benign confound.")
    else:
        print(f"  --> LOW ({mean_auc:.2f} AUC, close to chance). No evidence that border-only "
              f"features can separate the classes. No shortcut artifact detected here.")

    return mean_auc


def main():
    parser = argparse.ArgumentParser(description="Check a dataset for EDC-Fig.5-style artifacts.")
    parser.add_argument('--normal_dir', required=True)
    parser.add_argument('--abnormal_dir', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--out_dir', default='./artifact_check_output')
    parser.add_argument('--n_samples', type=int, default=200,
                         help='Number of images to sample per class for stats/classifier tests')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    normal_paths = list_images(args.normal_dir)
    abnormal_paths = list_images(args.abnormal_dir)
    print(f"Found {len(normal_paths)} normal images, {len(abnormal_paths)} abnormal images.")

    if len(normal_paths) == 0 or len(abnormal_paths) == 0:
        raise SystemExit("No images found in one of the given folders. Check the paths.")

    run_visual_grid(normal_paths, abnormal_paths, args.out_dir, args.dataset_name)
    run_corner_stats(normal_paths, abnormal_paths, args.out_dir, args.dataset_name, n=args.n_samples)
    run_shortcut_classifier(normal_paths, abnormal_paths, args.out_dir, args.dataset_name, n=args.n_samples)

    print(f"\nDone. Visual grid saved in {args.out_dir}/. "
          f"Review the printed statistics above for both datasets you care about.")


if __name__ == '__main__':
    main()