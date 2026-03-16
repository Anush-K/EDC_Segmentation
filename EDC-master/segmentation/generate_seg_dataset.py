import os
import shutil

# -------- paths --------

EDC_HEATMAP_DIR = "saved_models/edc_lgg/heatmap"
ABNORMAL_IMG_DIR = "../LGG/test/ABNORMAL"
MASK_DIR = "../LGG/masks"

OUTPUT_DIR = "../LGG_SEG"

IMG_OUT = os.path.join(OUTPUT_DIR, "images")
HM_OUT = os.path.join(OUTPUT_DIR, "heatmaps")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(HM_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# -------- process --------

count = 0

for file in os.listdir(ABNORMAL_IMG_DIR):

    if not file.endswith(".tif"):
        continue

    base = file.replace(".tif", "")

    image_path = os.path.join(ABNORMAL_IMG_DIR, file)
    heatmap_path = os.path.join(EDC_HEATMAP_DIR, base + "_map.png")
    mask_path = os.path.join(MASK_DIR, base + "_mask.tif")

    if not os.path.exists(heatmap_path):
        print("Missing heatmap:", base)
        continue

    if not os.path.exists(mask_path):
        print("Missing mask:", base)
        continue

    shutil.copy(image_path, os.path.join(IMG_OUT, file))
    shutil.copy(heatmap_path, os.path.join(HM_OUT, base + ".png"))
    shutil.copy(mask_path, os.path.join(MASK_OUT, base + ".tif"))

    count += 1

print("Segmentation dataset created.")
print("Total samples:", count)