"""
build_qualitative_figure.py

Builds the final composite qualitative anomaly-localisation figure
(matching your reference Fig. 6 layout) directly from files already
on your server:
  - Raw abnormal images  : datasets/<DATASET>/test/ABNORMAL/<filename>
  - Heatmap maps         : saved_models/<run_name>/heatmap/<stem>_map.png

You only need to edit the CONFIG section below with the actual
filenames you want to feature per dataset/subtype. Everything else
(loading, resizing, grid layout, labels) is fully automatic.

Usage (run on your server, inside EDC-master):
    python build_qualitative_figure.py
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — EDIT THIS SECTION WITH YOUR ACTUAL FILE NAMES
# ============================================================
# Datasets (APTOS, Br35H, OCT2017, ISIC2018) live directly under
# EDC_Segmentation/, NOT inside EDC-master/datasets/.
DATA_ROOT = "/home/cs24d0008/EDC_Segmentation"

# Code, saved_models, and heatmap outputs live inside EDC-master/.
CODE_ROOT = "/home/cs24d0008/EDC_Segmentation/EDC-master"

# ------------------------------------------------------------
# HEATMAP_ROOT: your consolidated heatmap folder, uploaded under
# EDC_Segmentation/Heatmap. It contains per-dataset subfolders
# (edc_aptos, edc_br35h, edc_oct, edc_isic2018, etc.), confirmed
# via `ls`. Each dataset below points at its own subfolder.
# ------------------------------------------------------------
HEATMAP_ROOT = os.path.join(DATA_ROOT, "Heatmap")

# For each dataset: raw image folder, heatmap folder, and the list
# of (subtype_label, filename) pairs to feature. filename must exist
# in BOTH the raw folder and match "<stem>_map.png" in the heatmap
# folder (stem = filename without extension).
#
# normal_raw is the single reference normal image shown in column 1.
DATASETS = {
    "OCT2017": {
        # OCT2017 uses per-class subfolders (CNV/DME/DRUSEN/NORMAL),
        # NOT a merged ABNORMAL folder like the other three datasets.
        "raw_dir":     None,
        "class_subdirs": True,
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_oct", "heatmap"),
        "normal_raw":  os.path.join(DATA_ROOT, "OCT2017/test/NORMAL/NORMAL-100980-1.jpeg"),
        "abnormal": [
            ("CNV",    "CNV-1032178-1.jpeg"),
            ("CNV",    "CNV-1034361-1.jpeg"),
            ("CNV",    "CNV-1034361-2.jpeg"),
            ("DME",    "DME-1029576-1.jpeg"),
            ("DME",    "DME-1029576-2.jpeg"),
            ("DME",    "DME-1029576-3.jpeg"),
            ("DRUSEN", "DRUSEN-1021298-1.jpeg"),
            ("DRUSEN", "DRUSEN-11129-1.jpeg"),
            ("DRUSEN", "DRUSEN-11129-2.jpeg"),
        ],
    },
    "APTOS": {
        # Confirmed nested path: APTOS/APTOS/test/... (not APTOS/test/...)
        "raw_dir":     os.path.join(DATA_ROOT, "APTOS/APTOS/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_aptos", "heatmap"),
        "normal_raw":  os.path.join(DATA_ROOT, "APTOS/APTOS/test/NORMAL/005b95c28852.png"),
        "abnormal": [
            ("", "000c1434d8d7.png"),
            ("", "001639a390f0.png"),
            ("", "0024cdab0c1e.png"),
            ("", "0083ee8054ee.png"),
            ("", "00a8624548a9.png"),
            ("", "00b74780d31d.png"),
            ("", "00cb6555d108.png"),
            ("", "00e4ddff966a.png"),
            ("", "0104b032c141.png"),
        ],
    },
    "Br35H": {
        "raw_dir":     os.path.join(DATA_ROOT, "Br35H/test/ABNORMAL"),
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_br35h", "heatmap"),
        "normal_raw":  os.path.join(DATA_ROOT, "Br35H/test/NORMAL/no1000.jpg"),
        "abnormal": [
            ("", "y0.jpg"),
            ("", "y1000.jpg"),
            ("", "y1001.jpg"),
            ("", "y1002.jpg"),
            ("", "y1003.jpg"),
            ("", "y1004.jpg"),
            ("", "y1005.jpg"),
            ("", "y1006.jpg"),
            ("", "y1007.jpg"),
        ],
    },
    "ISIC": {
        # Confirmed: ISIC2018/test/ABNORMAL and ISIC2018/test/NORMAL,
        # same structure as APTOS/Br35H (NOT flat test/ as first assumed).
        # Filenames (ISIC_XXXXXXX.jpg) don't encode diagnosis subtype,
        # so subtype labels are left blank here — fill in MEL/BCC/AKIEC/
        # etc. yourself if you have the ISIC2018 ground-truth CSV mapping
        # filename -> diagnosis; otherwise these just show as "Abnormal".
        "raw_dir":     os.path.join(DATA_ROOT, "ISIC2018/test/ABNORMAL"),
        # Confirmed via ls: files sit DIRECTLY inside edc_isic2018/,
        # with NO nested heatmap/ subfolder (unlike OCT2017/APTOS/Br35H,
        # which do have that extra nesting level).
        "heatmap_dir": os.path.join(HEATMAP_ROOT, "edc_isic2018"),
        "normal_raw":  os.path.join(DATA_ROOT, "ISIC2018/test/NORMAL/ISIC_0034321.jpg"),
        "abnormal": [
            ("Abnormal", "ISIC_0034323.jpg"),
            ("Abnormal", "ISIC_0034326.jpg"),
            ("Abnormal", "ISIC_0034329.jpg"),
            ("Abnormal", "ISIC_0034332.jpg"),
            ("Abnormal", "ISIC_0034333.jpg"),
            ("Abnormal", "ISIC_0034334.jpg"),
            ("Abnormal", "ISIC_0034338.jpg"),
            ("Abnormal", "ISIC_0034343.jpg"),
            ("Abnormal", "ISIC_0034344.jpg"),
        ],
    },
}

OUTPUT_PATH = os.path.join(CODE_ROOT, "Fig6_qualitative_composite.png")
THUMB_SIZE  = (160, 160)   # resize every tile to this size for a uniform grid

# ============================================================
# END CONFIG
# ============================================================


def load_and_resize(path, size=THUMB_SIZE):
    """Load an image and resize it, returning None (with a warning) if missing."""
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def heatmap_path_for(raw_filename, heatmap_dir):
    """
    Points to the COLORED overlay file (heatmap blended onto the original
    image with a JET colormap), not the grayscale _map.png. This matches
    EDC's own reference figure style, where the second row per dataset
    shows a colored heatmap overlaid on the anatomical structure.
    Both files are already generated by methods/edc1.py's
    save_anomaly_map() for every training run — no new heatmap generation
    needed, just pointing at the other file that already exists.
    """
    stem = os.path.splitext(raw_filename)[0]
    return os.path.join(heatmap_dir, stem + "_overlay.png")


def build_dataset_rows(cfg):
    """Returns (raw_row_images, heatmap_row_images, subtype_labels) including the normal column first."""
    raw_row = [load_and_resize(cfg["normal_raw"])]
    hm_row  = [None]  # no heatmap for the normal reference column
    labels  = ["Normal"]

    use_class_subdirs = cfg.get("class_subdirs", False)

    for subtype, fname in cfg["abnormal"]:
        if use_class_subdirs:
            # OCT2017-style: OCT2017/test/<subtype>/<filename>
            data_root_for_class = os.path.dirname(os.path.dirname(cfg["normal_raw"]))
            raw_path = os.path.join(data_root_for_class, subtype, fname)
        else:
            raw_path = os.path.join(cfg["raw_dir"], fname)

        hm_path = heatmap_path_for(fname, cfg["heatmap_dir"])
        raw_row.append(load_and_resize(raw_path))
        hm_row.append(load_and_resize(hm_path))
        labels.append(subtype)

    return raw_row, hm_row, labels


def main():
    all_rows = []       # list of (dataset_name, raw_row, hm_row, labels)
    max_cols = 0

    for name, cfg in DATASETS.items():
        print(f"Loading {name} ...")
        raw_row, hm_row, labels = build_dataset_rows(cfg)
        all_rows.append((name, raw_row, hm_row, labels))
        max_cols = max(max_cols, len(raw_row))

    n_datasets = len(all_rows)
    n_rows     = n_datasets * 2   # raw + heatmap row per dataset
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
                # FIX: every dataset gets its own title row now, instead of
                # only the first dataset (OCT2017) setting titles that then
                # visually applied to every other dataset's columns too.
                if labels[c]:
                    raw_ax_row[c].set_title(labels[c], fontsize=9)

            if c < len(hm_row) and hm_row[c] is not None:
                hm_ax_row[c].imshow(hm_row[c])

        # dataset label on the left edge, vertically centred across its two rows
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