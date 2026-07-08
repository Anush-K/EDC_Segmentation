"""
build_qualitative_figure1.py

Builds the Stage-1 classification qualitative figure for BUSI, LGG,
COVID-19, and Kvasir -- same raw-image + colored-heatmap-overlay
layout as your original Fig. 6 (build_qualitative_figure.py), just
applied to the four segmentation-validation datasets instead of
APTOS/Br35H/OCT2017/ISIC.

This is Figure 1 of two separate figures for Section 4.7:
  - Figure 1 (this script): Stage 1 classification heatmaps
  - Figure 2 (build_qualitative_figure2.py): Stage 2 segmentation
    comparison (Image / GT / Baseline / HGBL)

CONFIG BELOW USES PLACEHOLDER FILENAMES ("EXAMPLE...") — you'll need
to fill in real filenames the same way we did for the first figure.
Run the ls commands noted in each dataset's comment block, then
replace the placeholders with real, confirmed filenames.

Usage:
    python build_qualitative_figure2.py
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "/home/cs24d0008/EDC_Segmentation"
CODE_ROOT = "/home/cs24d0008/EDC_Segmentation/EDC-master"

# Consolidated heatmap folder — same one used for the first figure,
# confirmed to already contain edc_busi/edc_lgg/edc_covid19/edc_kvasir
# subfolders based on your earlier `ls Heatmap/` listing.
HEATMAP_ROOT = os.path.join(DATA_ROOT, "Heatmap")

# ------------------------------------------------------------
# Before running, confirm real paths/filenames with:
#   ls /home/cs24d0008/EDC_Segmentation/BUSI/test/ABNORMAL | head -10
#   ls /home/cs24d0008/EDC_Segmentation/BUSI/test/NORMAL | head -3
#   ls /home/cs24d0008/EDC_Segmentation/Heatmap/edc_busi/ | head -5
# (repeat for LGG, COVID19, Kvasir — folder names may differ slightly,
#  e.g. "COVID19" vs "COVID-19", check your actual directory listing)
# ------------------------------------------------------------
DATASETS = {
    "BUSI": {
        # Confirmed location: EDC-master/datasets/BUSI
        "raw_dir":     os.path.join(CODE_ROOT, "datasets/BUSI/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_busi", "heatmap"),
        "normal_raw":  os.path.join(CODE_ROOT, "datasets/BUSI/test/NORMAL/normal (102).png"),
        "abnormal": [
            ("", "malignant (100).png"),
            ("", "malignant (101).png"),
            ("", "malignant (102).png"),
            ("", "malignant (103).png"),
            ("", "malignant (104).png"),
            ("", "malignant (105).png"),
            ("", "malignant (106).png"),
            ("", "malignant (107).png"),
            ("", "malignant (108).png"),
        ],
    },
    "LGG": {
        # Confirmed real folder name: LGG_new (not LGG), directly under
        # EDC_Segmentation/, same pattern as APTOS/Br35H/OCT2017/ISIC.
        "raw_dir":     os.path.join(DATA_ROOT, "LGG_new/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_lgg", "heatmap"),
        "normal_raw":  os.path.join(DATA_ROOT, "LGG_new/test/NORMAL/TCGA_CS_4941_19960909_23.tif"),
        "abnormal": [
            ("", "TCGA_CS_4941_19960909_11.tif"),
            ("", "TCGA_CS_4941_19960909_13.tif"),
            ("", "TCGA_CS_4941_19960909_16.tif"),
            ("", "TCGA_CS_4941_19960909_17.tif"),
            ("", "TCGA_CS_4942_19970222_11.tif"),
            ("", "TCGA_CS_4942_19970222_13.tif"),
            ("", "TCGA_CS_4943_20000902_14.tif"),
            ("", "TCGA_CS_4943_20000902_16.tif"),
            ("", "TCGA_CS_4943_20000902_18.tif"),
        ],
    },
    "COVID-19": {
        # Confirmed location: EDC-master/datasets/COVID19
        "raw_dir":     os.path.join(CODE_ROOT, "datasets/COVID19/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_covid19", "heatmap"),
        "normal_raw":  os.path.join(CODE_ROOT, "datasets/COVID19/test/NORMAL/Normal-10005.png"),
        "abnormal": [
            ("", "COVID-100.png"),
            ("", "COVID-1015.png"),
            ("", "COVID-1018.png"),
            ("", "COVID-1022.png"),
            ("", "COVID-1025.png"),
            ("", "COVID-1028.png"),
            ("", "COVID-1041.png"),
            ("", "COVID-1052.png"),
            ("", "COVID-1053.png"),
        ],
    },
    "Kvasir": {
        # Confirmed location: EDC-master/datasets/KVASIR (NOT directly
        # under EDC_Segmentation/Kvasir like the other datasets) —
        # internal test/ABNORMAL, test/NORMAL structure to be confirmed.
        "raw_dir":     os.path.join(CODE_ROOT, "datasets/KVASIR/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_kvasir", "heatmap"),
        "normal_raw":  os.path.join(CODE_ROOT, "datasets/KVASIR/test/NORMAL/normal-cecum__01af3454-037f-4708-b73c-6ec4423b6a61.jpg"),
        "abnormal": [
            ("", "dyed-lifted-polyps__00cf9508-6ad1-4db9-840a-519c1d515c30.jpg"),
            ("", "dyed-lifted-polyps__031a6e89-d3b7-48c8-8e2f-db991030f959.jpg"),
            ("", "dyed-lifted-polyps__08c397e0-1463-47df-abef-b25831fc4f9c.jpg"),
            ("", "dyed-lifted-polyps__0a542bb0-0172-4bc4-890d-327fa45b85b3.jpg"),
            ("", "dyed-lifted-polyps__0a693b92-89c9-48ea-9ad8-dc1d2f84aaca.jpg"),
            ("", "dyed-lifted-polyps__0c2ecef0-333b-44ce-97cc-b7780bfc5848.jpg"),
            ("", "dyed-lifted-polyps__0ffa8ce1-8645-44c3-b2bb-bc14ae01d6a6.jpg"),
            ("", "dyed-lifted-polyps__10d78f91-9f3c-42c4-bca3-a8f8c91fccd7.jpg"),
            ("", "dyed-lifted-polyps__135530e9-3e61-4124-8f67-b15741942531.jpg"),
        ],
    },
}

OUTPUT_PATH = os.path.join(CODE_ROOT, "Fig_classification_seg_datasets.png")
THUMB_SIZE  = (160, 160)

# ============================================================
# END CONFIG
# ============================================================


def load_and_resize(path, size=THUMB_SIZE):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def heatmap_path_for(raw_filename, heatmap_dir):
    """Points to the COLORED overlay file, matching Figure A's style."""
    stem = os.path.splitext(raw_filename)[0]
    return os.path.join(heatmap_dir, stem + "_overlay.png")


def build_dataset_rows(cfg):
    raw_row = [load_and_resize(cfg["normal_raw"])]
    hm_row  = [None]
    labels  = ["Normal"]

    for subtype, fname in cfg["abnormal"]:
        raw_path = os.path.join(cfg["raw_dir"], fname)
        hm_path  = heatmap_path_for(fname, cfg["heatmap_dir"])
        raw_row.append(load_and_resize(raw_path))
        hm_row.append(load_and_resize(hm_path))
        labels.append(subtype)

    return raw_row, hm_row, labels


def main():
    all_rows = []
    max_cols = 0

    for name, cfg in DATASETS.items():
        print(f"Loading {name} ...")
        raw_row, hm_row, labels = build_dataset_rows(cfg)
        all_rows.append((name, raw_row, hm_row, labels))
        max_cols = max(max_cols, len(raw_row))

    n_datasets = len(all_rows)
    n_rows     = n_datasets * 2
    n_cols     = max_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.6, n_rows * 1.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )

    for d_idx, (name, raw_row, hm_row, labels) in enumerate(all_rows):
        raw_ax_row = axes[d_idx * 2]
        hm_ax_row  = axes[d_idx * 2 + 1]

        for c in range(n_cols):
            raw_ax_row[c].axis("off")
            hm_ax_row[c].axis("off")

            if c < len(raw_row) and raw_row[c] is not None:
                raw_ax_row[c].imshow(raw_row[c])
                if labels[c]:
                    raw_ax_row[c].set_title(labels[c], fontsize=9)

            if c < len(hm_row) and hm_row[c] is not None:
                hm_ax_row[c].imshow(hm_row[c])

        raw_ax_row[0].text(
            -0.35, 0.5, name, transform=raw_ax_row[0].transAxes,
            fontsize=11, fontweight="bold", rotation=90,
            va="center", ha="center",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved composite figure -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()