# segmentation/generate_seg_dataset_lgg.py
"""
Assembles the LGG_SEG dataset used by the segmentation stage.

Reads:
  - ABNORMAL test images  → LGG_new/test/ABNORMAL/
  - EDC anomaly heatmaps  → saved_models/edc_lgg/heatmap/*_map.png
  - Ground-truth masks    → LGG_new/masks/*_mask.tif

Writes:
  - LGG_SEG/images/   <stem>.tif
  - LGG_SEG/heatmaps/ <stem>.png
  - LGG_SEG/masks/    <stem>.tif
"""

import os
import sys
import shutil

# Allow running from segmentation/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.config_lgg import BASE_DIR, CODE_DIR, DATASET_DIR
    EDC_HEATMAP_DIR  = os.path.join(CODE_DIR,    "saved_models", "edc_lgg", "heatmap")
    ABNORMAL_IMG_DIR = os.path.join(DATASET_DIR, "test", "ABNORMAL")
    MASK_DIR         = os.path.join(DATASET_DIR, "masks")
    OUTPUT_DIR       = os.path.join(BASE_DIR,    "LGG_SEG")
except ImportError:
    # Fallback to relative paths when config is unavailable
    BASE_DIR         = "/home/cs24d0008/EDC_Segmentation"
    CODE_DIR         = os.path.join(BASE_DIR, "EDC-master")
    EDC_HEATMAP_DIR  = os.path.join(CODE_DIR, "saved_models", "edc_lgg", "heatmap")
    ABNORMAL_IMG_DIR = os.path.join(BASE_DIR, "LGG_new", "test", "ABNORMAL")
    MASK_DIR         = os.path.join(BASE_DIR, "LGG_new", "masks")
    OUTPUT_DIR       = os.path.join(BASE_DIR, "LGG_SEG")

IMG_OUT  = os.path.join(OUTPUT_DIR, "images")
HM_OUT   = os.path.join(OUTPUT_DIR, "heatmaps")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

for d in (IMG_OUT, HM_OUT, MASK_OUT):
    os.makedirs(d, exist_ok=True)

# Print resolved paths
print("===== Resolved Paths =====")
print(f"  ABNORMAL images : {ABNORMAL_IMG_DIR}")
print(f"  Heatmaps        : {EDC_HEATMAP_DIR}")
print(f"  Masks           : {MASK_DIR}")
print(f"  Output          : {OUTPUT_DIR}")
print("==========================\n")


def main():
    if not os.path.exists(ABNORMAL_IMG_DIR):
        print(f"ERROR: ABNORMAL folder not found: {ABNORMAL_IMG_DIR}")
        return

    count_ok     = 0
    miss_heatmap = []
    miss_mask    = []

    all_files = sorted([
        f for f in os.listdir(ABNORMAL_IMG_DIR)
        if f.lower().endswith(('.tif', '.png', '.jpg')) and not f.startswith('.')
    ])

    if not all_files:
        print(f"ERROR: No image files found in {ABNORMAL_IMG_DIR}")
        return

    for file in all_files:
        base = os.path.splitext(file)[0]

        image_path   = os.path.join(ABNORMAL_IMG_DIR, file)
        heatmap_path = os.path.join(EDC_HEATMAP_DIR,  base + "_map.png")
        mask_path    = os.path.join(MASK_DIR,          base + "_mask.tif")

        if not os.path.exists(heatmap_path):
            miss_heatmap.append(base)
            continue

        if not os.path.exists(mask_path):
            miss_mask.append(base)
            continue

        shutil.copy(image_path,   os.path.join(IMG_OUT,  file))
        shutil.copy(heatmap_path, os.path.join(HM_OUT,   base + ".png"))
        shutil.copy(mask_path,    os.path.join(MASK_OUT, base + ".tif"))

        count_ok += 1

    print("\n===== generate_seg_dataset_lgg summary =====")
    print(f"  ABNORMAL images found : {len(all_files)}")
    print(f"  Triplets assembled    : {count_ok}")

    if miss_heatmap:
        print(f"\n  Missing heatmaps ({len(miss_heatmap)}):")
        for m in miss_heatmap[:10]:
            print(f"    {m}")
        if len(miss_heatmap) > 10:
            print(f"    ... and {len(miss_heatmap) - 10} more")

    if miss_mask:
        print(f"\n  Missing masks ({len(miss_mask)}):")
        for m in miss_mask[:10]:
            print(f"    {m}")
        if len(miss_mask) > 10:
            print(f"    ... and {len(miss_mask) - 10} more")

    print(f"\nSegmentation dataset written to: {OUTPUT_DIR}")
    print(f"  images/   → {count_ok} files")
    print(f"  heatmaps/ → {count_ok} files")
    print(f"  masks/    → {count_ok} files")


if __name__ == "__main__":
    main()