import os
import cv2
import random

# ✅ YOUR ACTUAL PATHS
source_dir = "/home/cs24d0008/EDC_Segmentation/EDC-master/datasets/BUSI_Orig"
target_dir = "/home/cs24d0008/EDC_Segmentation/EDC-master/datasets/BUSI"

normal_dir = os.path.join(source_dir, "normal")
malignant_dir = os.path.join(source_dir, "malignant")

# safety check
if not os.path.exists(normal_dir):
    raise Exception(f"❌ normal folder not found: {normal_dir}")

if not os.path.exists(malignant_dir):
    raise Exception(f"❌ malignant folder not found: {malignant_dir}")

# create output folders
os.makedirs(os.path.join(target_dir, "train", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test", "ABNORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

normal_paths = []
abnormal_pairs = []

# ---------------- NORMAL ----------------
for file in os.listdir(normal_dir):

    if "_mask" in file:
        continue

    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        normal_paths.append(os.path.join(normal_dir, file))

# ---------------- MALIGNANT ----------------
for file in os.listdir(malignant_dir):

    if "_mask" in file:
        continue

    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(malignant_dir, file)
    base = os.path.splitext(file)[0]
    mask_path = os.path.join(malignant_dir, base + "_mask.png")

    if os.path.exists(mask_path):
        abnormal_pairs.append((img_path, mask_path))

print("Total normal:", len(normal_paths))
print("Total abnormal:", len(abnormal_pairs))

# ---------------- SPLIT ----------------
random.shuffle(normal_paths)
split = int(len(normal_paths) * 0.8)

train_normals = normal_paths[:split]
test_normals = normal_paths[split:]

# ---------------- FUNCTIONS ----------------
def process_img(src, dst):
    img = cv2.imread(src)
    if img is None:
        return
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(dst, img)

def process_mask(src, dst):
    mask = cv2.imread(src, 0)
    if mask is None:
        return
    mask = cv2.resize(mask, (256, 256))
    cv2.imwrite(dst, mask)

# ---------------- SAVE NORMAL ----------------
for path in train_normals:
    process_img(path, os.path.join(target_dir, "train", "NORMAL", os.path.basename(path)))

for path in test_normals:
    process_img(path, os.path.join(target_dir, "test", "NORMAL", os.path.basename(path)))

# ---------------- SAVE ABNORMAL ----------------
for img_path, mask_path in abnormal_pairs:

    name = os.path.basename(img_path)
    base = os.path.splitext(name)[0]

    process_img(img_path, os.path.join(target_dir, "test", "ABNORMAL", name))
    process_mask(mask_path, os.path.join(target_dir, "masks", base + "_mask.png"))

print("\n✅ DONE: BUSI dataset prepared successfully!")