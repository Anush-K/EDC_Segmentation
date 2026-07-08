"""
build_qualitative_figure2.py

Builds the Stage-2 segmentation qualitative figure for BUSI, LGG,
COVID-19, and Kvasir -- Image / Heatmap (the 4th input channel fed to
the model) / Ground Truth / Predicted Mask, at 9 examples per dataset
spanning the full Dice-quantile range (non-cherry-picked selection,
matching your Section 4.7 methodology text).

For each dataset, this script:
  1. Rebuilds the same test split used during training (same SEED=42,
     same 80/20 random_split), so selected examples are genuinely
     held-out.
  2. Loads the saved HGBL checkpoint for that dataset.
  3. Runs inference on every test image, computes per-image Dice.
  4. Selects 9 examples evenly spaced across the sorted Dice
     distribution (0th, 12.5th, 25th, ... 100th percentile), so the
     figure shows a full spread from worst to best performance rather
     than just three points.
  5. Assembles one composite: for each dataset, 4 rows (Image /
     Heatmap / Ground Truth / Predicted Mask) x 9 columns
     (evenly-spaced Dice percentile examples).

CONFIG BELOW USES PATHS CONFIRMED ON THE SERVER (BUSI_SEG / LGG_SEG /
COVID19_SEG / KVASIR_SEG live one level above EDC-master, and the
seg_checkpoints_* folders live directly inside EDC-master).

Usage:
    cd segmentation/
    python build_qualitative_figure2.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt

from unet import UNet

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42  # must match SEED used in train_segmentation_<name>.py

N_COLUMNS = 9  # number of examples shown per dataset

DATASETS = {
    "BUSI": {
        "module": "dataset_busi",
        "class":  "BUSISegDataset",
        "path":   "../../BUSI_SEG",
        "ckpt_hgbl": "../seg_checkpoints_busi/best_unet_busi_hgbl.pth",
    },
    "LGG": {
        "module": "dataset_lgg",
        "class":  "LGGSegDataset",
        "path":   "../../LGG_SEG",
        "ckpt_hgbl": "../seg_checkpoints_lgg/best_unet_lgg_hgbl.pth",
    },
    "COVID-19": {
        "module": "dataset_covid19",
        "class":  "COVID19SegDataset",
        "path":   "../../COVID19_SEG",
        "ckpt_hgbl": "../seg_checkpoints_covid19/best_unet_covid19_hgbl.pth",
    },
    "Kvasir": {
        "module": "dataset_kvasir",
        "class":  "KvasirSegDataset",
        "path":   "../../KVASIR_SEG",
        "ckpt_hgbl": "../seg_checkpoints_kvasir/best_unet_kvasir_hgbl.pth",
    },
}

OUTPUT_PATH = "Fig_segmentation_qualitative_9col.png"

# ============================================================
# END CONFIG
# ============================================================


def dice_score_np(pred, target, eps=1e-8):
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    return (2. * inter + eps) / (pred.sum() + target.sum() + eps)


def load_dataset_class(module_name, class_name):
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def get_test_subset(dataset_cls, path):
    full = dataset_cls(path)
    n  = len(full)
    nt = max(1, int(0.8 * n))
    nv = n - nt
    gen = torch.Generator().manual_seed(SEED)
    _, vi = random_split(range(n), [nt, nv], generator=gen)
    return Subset(full, vi.indices)


def load_model(ckpt_path):
    model = UNet(in_channels=4, dropout_p=0.3).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference_all(model, subset):
    results = []
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for image, heatmap, mask in loader:
            image_d, heatmap_d = image.to(DEVICE), heatmap.to(DEVICE)
            pred = model(torch.cat([image_d, heatmap_d], dim=1))
            results.append({
                "image":   image[0].permute(1, 2, 0).cpu().numpy(),
                "heatmap": heatmap[0, 0].cpu().numpy(),
                "mask":    mask[0, 0].cpu().numpy(),
                "pred":    pred[0, 0].cpu().numpy(),
            })
    return results


def select_n_quantile_examples(results, n=N_COLUMNS):
    """
    Select n examples evenly spaced across the sorted Dice distribution
    (0th, ..., 100th percentile), without duplicate indices where
    possible. Non-cherry-picked: positions are fixed by percentile,
    not by visual inspection.
    """
    dices = np.array([dice_score_np(r["pred"], r["mask"]) for r in results])
    order = np.argsort(dices)  # ascending
    n_avail = len(order)

    if n_avail <= n:
        # Not enough examples to fill n unique slots -- use all, in order
        chosen_positions = np.arange(n_avail)
    else:
        # Evenly spaced positions from 0 (worst) to n_avail-1 (best)
        chosen_positions = np.linspace(0, n_avail - 1, n).round().astype(int)
        chosen_positions = np.unique(chosen_positions)
        # If rounding collapsed some positions, top up from remaining
        # indices spaced as evenly as possible until we have n
        if len(chosen_positions) < n:
            remaining = [i for i in range(n_avail) if i not in chosen_positions]
            extra_needed = n - len(chosen_positions)
            step = max(1, len(remaining) // max(1, extra_needed))
            fill = remaining[::step][:extra_needed]
            chosen_positions = np.sort(np.concatenate([chosen_positions, fill]))

    chosen_idx = [int(order[p]) for p in chosen_positions]
    return chosen_idx, dices


def to_uint8_rgb(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def to_uint8_mask(mask):
    return (np.clip(mask, 0, 1) * 255).astype(np.uint8)


def build_dataset_panel(name, cfg):
    print(f"Loading {name} ...")
    dataset_cls = load_dataset_class(cfg["module"], cfg["class"])
    test_subset = get_test_subset(dataset_cls, cfg["path"])

    model = load_model(cfg["ckpt_hgbl"])

    print(f"  Running inference on {len(test_subset)} test images ...")
    results = run_inference_all(model, test_subset)

    chosen_idx, dices = select_n_quantile_examples(results, n=N_COLUMNS)
    print(f"  Selected {len(chosen_idx)} examples spanning Dice "
          f"{dices[chosen_idx[0]]:.3f} -> {dices[chosen_idx[-1]]:.3f}: "
          + ", ".join(f"{dices[i]:.3f}" for i in chosen_idx))

    panel = []
    for idx in chosen_idx:
        r = results[idx]
        panel.append({
            "image":   to_uint8_rgb(r["image"]),
            "heatmap": to_uint8_mask(r["heatmap"]),
            "gt":      to_uint8_mask(r["mask"]),
            "pred":    to_uint8_mask(r["pred"]),
            "dice":    dice_score_np(r["pred"], r["mask"]),
        })
    return panel


def main():
    all_panels = {}
    for name, cfg in DATASETS.items():
        all_panels[name] = build_dataset_panel(name, cfg)

    dataset_names = list(DATASETS.keys())
    row_labels = ["Image", "Heatmap (input)", "Ground Truth", "Predicted Mask"]

    n_datasets = len(dataset_names)
    n_rows = n_datasets * 4
    n_cols = N_COLUMNS

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.0),
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )

    for d_idx, dname in enumerate(dataset_names):
        panel = all_panels[dname]
        r0 = d_idx * 4
        n_examples = len(panel)

        for col_idx in range(n_cols):
            if col_idx < n_examples:
                ex = panel[col_idx]

                axes[r0 + 0, col_idx].imshow(ex["image"])
                axes[r0 + 1, col_idx].imshow(ex["heatmap"], cmap="jet")
                axes[r0 + 2, col_idx].imshow(ex["gt"], cmap="gray")
                axes[r0 + 3, col_idx].imshow(ex["pred"], cmap="gray")

                axes[r0 + 3, col_idx].set_title(
                    f"Dice={ex['dice']:.3f}", fontsize=9, y=-0.15
                )
            for rr in range(4):
                axes[r0 + rr, col_idx].axis("off")

        for rr, rlabel in enumerate(row_labels):
            axes[r0 + rr, 0].text(
                -0.5, 0.5, rlabel, transform=axes[r0 + rr, 0].transAxes,
                fontsize=9, va="center", ha="right",
            )

        axes[r0, 0].text(
            -1.15, 0.5, dname, transform=axes[r0, 0].transAxes,
            fontsize=13, fontweight="bold", rotation=90, va="center", ha="center",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
