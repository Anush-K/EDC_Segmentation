# segmentation/generate_seg_dataset_covid19.py
"""
Assembles the COVID19_SEG dataset used by the segmentation stage.

Reads:
  - ABNORMAL test images  → EDC-master/datasets/COVID19/test/ABNORMAL/
  - EDC anomaly heatmaps  → EDC-master/saved_models/edc_covid19/heatmap/*_map.png
  - Ground-truth masks    → EDC-master/datasets/COVID19/masks/*_mask.png

Writes:
  - EDC_Segmentation/COVID19_SEG/images/   <stem>.png
  - EDC_Segmentation/COVID19_SEG/heatmaps/ <stem>.png
  - EDC_Segmentation/COVID19_SEG/masks/    <stem>_mask.png
"""

import os
import sys
import shutil

# Allow running from segmentation/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.config_covid19 import BASE_DIR, CODE_DIR, DATASET_DIR
    EDC_HEATMAP_DIR  = os.path.join(CODE_DIR,    "saved_models", "edc_covid19", "heatmap")
    ABNORMAL_IMG_DIR = os.path.join(DATASET_DIR, "test", "ABNORMAL")
    MASK_DIR         = os.path.join(DATASET_DIR, "masks")
    OUTPUT_DIR       = os.path.join(BASE_DIR,    "COVID19_SEG")
except ImportError:
    # Fallback: adjust these if config is unavailable
    BASE_DIR         = "/home/cs24d0008/EDC_Segmentation"
    CODE_DIR         = os.path.join(BASE_DIR, "EDC-master")
    EDC_HEATMAP_DIR  = os.path.join(CODE_DIR, "saved_models", "edc_covid19", "heatmap")
    ABNORMAL_IMG_DIR = os.path.join(CODE_DIR, "datasets", "COVID19", "test", "ABNORMAL")
    MASK_DIR         = os.path.join(CODE_DIR, "datasets", "COVID19", "masks")
    OUTPUT_DIR       = os.path.join(BASE_DIR, "COVID19_SEG")

IMG_OUT  = os.path.join(OUTPUT_DIR, "images")
HM_OUT   = os.path.join(OUTPUT_DIR, "heatmaps")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

for d in (IMG_OUT, HM_OUT, MASK_OUT):
    os.makedirs(d, exist_ok=True)

# Print resolved paths so you can verify before running
print("===== Resolved Paths =====")
print(f"  ABNORMAL images : {ABNORMAL_IMG_DIR}")
print(f"  Heatmaps        : {EDC_HEATMAP_DIR}")
print(f"  Masks           : {MASK_DIR}")
print(f"  Output          : {OUTPUT_DIR}")
print("==========================\n")


def main():
    # Verify ABNORMAL dir exists before proceeding
    if not os.path.exists(ABNORMAL_IMG_DIR):
        print(f"ERROR: ABNORMAL folder not found: {ABNORMAL_IMG_DIR}")
        print("Please check your COVID19 dataset path in configs/config_covid19.py")
        return

    count_ok     = 0
    miss_heatmap = []
    miss_mask    = []

    all_files = sorted([
        f for f in os.listdir(ABNORMAL_IMG_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
    ])

    if not all_files:
        print(f"ERROR: No image files found in {ABNORMAL_IMG_DIR}")
        return

    for file in all_files:
        base = os.path.splitext(file)[0]   # e.g. COVID-1

        image_path   = os.path.join(ABNORMAL_IMG_DIR, file)
        heatmap_path = os.path.join(EDC_HEATMAP_DIR,  base + "_map.png")
        mask_path    = os.path.join(MASK_DIR,          base + "_mask.png")

        # Check heatmap
        if not os.path.exists(heatmap_path):
            miss_heatmap.append(base)
            continue

        # Check mask
        if not os.path.exists(mask_path):
            miss_mask.append(base)
            continue

        # Copy files — image and heatmap use plain stem; mask keeps _mask suffix
        shutil.copy(image_path,   os.path.join(IMG_OUT,  file))
        shutil.copy(heatmap_path, os.path.join(HM_OUT,   base + ".png"))
        shutil.copy(mask_path,    os.path.join(MASK_OUT, base + "_mask.png"))

        count_ok += 1

    # ---- Summary --------------------------------------------------------
    print("\n===== generate_seg_dataset_covid19 summary =====")
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
