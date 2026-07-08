"""
Dataset Exclusion Figure Builder
--------------------------------
Builds a Fig.-5-style comparison figure (normal vs. abnormal, artifact
circled) using REAL images from your own dataset folders. Matches the
layout/annotation style of EDC's own Fig. 5.

Usage:
    python build_artifact_figure.py \
        --normal_dir /path/to/normal \
        --abnormal_dir /path/to/abnormal \
        --dataset_name Kvasir \
        --corner bottom_left \
        --out_dir ./artifact_figures

--corner should match whichever corner your check_dataset_artifacts.py run
flagged most strongly (e.g. bottom_left for Kvasir based on the printed stats).
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from PIL import Image

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

CORNER_POS = {
    # (x_frac, y_frac) as fraction of image width/height, for placing the
    # dashed circle roughly over that corner region
    'top_left':     (0.12, 0.12),
    'top_right':    (0.88, 0.12),
    'bottom_left':  (0.12, 0.88),
    'bottom_right': (0.88, 0.88),
}


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS])


def annotate_corner(ax, img_w, img_h, corner, color='red'):
    """Draw a dashed ellipse + arrow pointing at the given corner region,
    matching the visual style of EDC's Fig. 5."""
    cx_frac, cy_frac = CORNER_POS[corner]
    cx, cy = cx_frac * img_w, cy_frac * img_h
    ew, eh = img_w * 0.28, img_h * 0.28

    ellipse = Ellipse((cx, cy), ew, eh, fill=False, edgecolor=color,
                       linestyle='--', linewidth=1.8)
    ax.add_patch(ellipse)

    # Arrow from just outside the ellipse toward its edge, echoing Fig. 5's style
    arrow_start = (cx - ew * 0.9, cy - eh * 0.9) if 'top' in corner or 'left' in corner \
        else (cx + ew * 0.9, cy - eh * 0.9)
    arrow = FancyArrowPatch(arrow_start, (cx - ew * 0.3, cy - eh * 0.3),
                             arrowstyle='->', mutation_scale=15,
                             color=color, linewidth=1.5)
    ax.add_patch(arrow)


def build_figure(normal_dir, abnormal_dir, dataset_name, corner, out_dir, seed=0):
    random.seed(seed)
    normal_paths = list_images(normal_dir)
    abnormal_paths = list_images(abnormal_dir)

    if not normal_paths or not abnormal_paths:
        raise SystemExit("No images found in one of the given folders. Check the paths.")

    normal_img_path = random.choice(normal_paths)
    abnormal_img_path = random.choice(abnormal_paths)

    normal_img = Image.open(normal_img_path).convert('RGB')
    abnormal_img = Image.open(abnormal_img_path).convert('RGB')

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    for ax, img, title in zip(axes, [normal_img, abnormal_img], ['Normal Image', 'Anomalous Image']):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        annotate_corner(ax, img.width, img.height, corner)

    fig.suptitle(
        f"Dataset Exclusion Check — {dataset_name}\n"
        f"(flagged region: {corner.replace('_', ' ')} corner, "
        f"see check_dataset_artifacts.py output for statistics)",
        fontsize=11
    )
    plt.tight_layout()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / f"{dataset_name}_artifact_figure.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved -> {save_path}")
    print(f"  Normal image used:   {normal_img_path}")
    print(f"  Abnormal image used: {abnormal_img_path}")
    print(f"  (Re-run with a different --seed to sample different examples "
          f"if these two aren't visually clear.)")


def main():
    parser = argparse.ArgumentParser(description="Build a Fig.-5-style artifact comparison figure.")
    parser.add_argument('--normal_dir', required=True)
    parser.add_argument('--abnormal_dir', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--corner', required=True,
                         choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'],
                         help='Which corner to circle, based on your artifact-checker output.')
    parser.add_argument('--out_dir', default='./artifact_figures')
    parser.add_argument('--seed', type=int, default=0,
                         help='Change this to sample a different normal/abnormal pair.')
    args = parser.parse_args()

    build_figure(args.normal_dir, args.abnormal_dir, args.dataset_name,
                 args.corner, args.out_dir, args.seed)


if __name__ == '__main__':
    main()