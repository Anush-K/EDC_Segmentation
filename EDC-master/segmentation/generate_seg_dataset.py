# segmentation/generate_seg_dataset.py
"""
Assembles the LGG_SEG dataset used by the segmentation stage.

Reads:
  - ABNORMAL test images  → LGG/test/ABNORMAL/
  - EDC anomaly heatmaps  → saved_models/edc_lgg/heatmap/*_map.png
  - Ground-truth masks    → LGG/masks/*_mask.tif

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
    from configs.config_lgg import BASE_DIR, CODE_DIR
    EDC_HEATMAP_DIR  = os.path.join(CODE_DIR,  "saved_models", "edc_lgg", "heatmap")
    ABNORMAL_IMG_DIR = os.path.join(BASE_DIR,  "LGG", "test", "ABNORMAL")
    MASK_DIR         = os.path.join(BASE_DIR,  "LGG", "masks")
    OUTPUT_DIR       = os.path.join(BASE_DIR,  "LGG_SEG")
except ImportError:
    # Fallback to relative paths when config is unavailable
    EDC_HEATMAP_DIR  = "../EDC-master/saved_models/edc_lgg/heatmap"
    ABNORMAL_IMG_DIR = "../LGG/test/ABNORMAL"
    MASK_DIR         = "../LGG/masks"
    OUTPUT_DIR       = "../LGG_SEG"

IMG_OUT  = os.path.join(OUTPUT_DIR, "images")
HM_OUT   = os.path.join(OUTPUT_DIR, "heatmaps")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

for d in (IMG_OUT, HM_OUT, MASK_OUT):
    os.makedirs(d, exist_ok=True)


def main():
    count_ok      = 0
    miss_heatmap  = []
    miss_mask     = []

    all_files = sorted([
        f for f in os.listdir(ABNORMAL_IMG_DIR)
        if f.lower().endswith('.tif') and not f.startswith('.')
    ])

    if not all_files:
        print(f"ERROR: No .tif files found in {ABNORMAL_IMG_DIR}")
        return

    for file in all_files:
        base = os.path.splitext(file)[0]   # e.g. TCGA_CS_4942_19970222_11

        image_path   = os.path.join(ABNORMAL_IMG_DIR, file)
        heatmap_path = os.path.join(EDC_HEATMAP_DIR,  base + "_map.png")
        mask_path    = os.path.join(MASK_DIR,          base + "_mask.tif")

        # Check heatmap
        if not os.path.exists(heatmap_path):
            miss_heatmap.append(base)
            continue

        # Check mask
        if not os.path.exists(mask_path):
            miss_mask.append(base)
            continue

        # Copy files — destination filenames use plain stem (no _mask suffix)
        shutil.copy(image_path,   os.path.join(IMG_OUT,  file))
        shutil.copy(heatmap_path, os.path.join(HM_OUT,   base + ".png"))
        shutil.copy(mask_path,    os.path.join(MASK_OUT, base + ".tif"))

        count_ok += 1

    # ---- Summary --------------------------------------------------------
    print("\n===== generate_seg_dataset summary =====")
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