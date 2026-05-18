import os
import shutil

# 🔥 CORRECT BASE PATH
BASE_DIR = "/home/cs24d0008/EDC_Segmentation/EDC-master"

# -------- PATHS --------
EDC_HEATMAP_DIR = os.path.join(BASE_DIR, "saved_models/edc_busi/heatmap")

ABNORMAL_IMG_DIR = os.path.join(BASE_DIR, "datasets/BUSI/test/ABNORMAL")
MASK_DIR = os.path.join(BASE_DIR, "datasets/BUSI/masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "datasets/BUSI_SEG")

IMG_OUT = os.path.join(OUTPUT_DIR, "images")
HM_OUT = os.path.join(OUTPUT_DIR, "heatmaps")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

# -------- CREATE --------
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(HM_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# -------- CHECK --------
if not os.path.exists(ABNORMAL_IMG_DIR):
    raise Exception(f"❌ ABNORMAL folder not found: {ABNORMAL_IMG_DIR}")

if not os.path.exists(EDC_HEATMAP_DIR):
    raise Exception(f"❌ Heatmap folder not found: {EDC_HEATMAP_DIR}")

# -------- PROCESS --------
count = 0

for file in os.listdir(ABNORMAL_IMG_DIR):

    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    base = os.path.splitext(file)[0]

    image_path = os.path.join(ABNORMAL_IMG_DIR, file)
    heatmap_path = os.path.join(EDC_HEATMAP_DIR, base + "_map.png")
    mask_path = os.path.join(MASK_DIR, base + "_mask.png")

    if not os.path.exists(heatmap_path):
        print("Missing heatmap:", base)
        continue

    if not os.path.exists(mask_path):
        print("Missing mask:", base)
        continue

    shutil.copy(image_path, os.path.join(IMG_OUT, file))
    shutil.copy(heatmap_path, os.path.join(HM_OUT, base + ".png"))
    shutil.copy(mask_path, os.path.join(MASK_OUT, base + ".png"))

    count += 1

print("\n✅ Segmentation dataset created (BUSI)")
print("Total samples:", count)