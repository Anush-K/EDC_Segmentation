import os
import cv2
import random
import numpy as np
from shutil import copyfile

# --------------------------------------------------
# SOURCE DATASET PATH
# --------------------------------------------------
source_dir = "/home/cs24d0008/EDC_Segmentation/LGG/LGG"

# --------------------------------------------------
# TARGET DATASET PATH
# --------------------------------------------------
target_dir = "/home/cs24d0008/EDC_Segmentation/LGG_new"

# --------------------------------------------------
# Create required folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

normal_paths   = []
abnormal_paths = []

# --------------------------------------------------
# Source subfolders
# --------------------------------------------------
train_normal_dir   = os.path.join(source_dir, "train", "NORMAL")
test_normal_dir    = os.path.join(source_dir, "test",  "NORMAL")
test_abnormal_dir  = os.path.join(source_dir, "test",  "ABNORMAL")
masks_dir          = os.path.join(source_dir, "masks")

# --------------------------------------------------
# Verify folders exist
# --------------------------------------------------
for d in [train_normal_dir, test_normal_dir, test_abnormal_dir, masks_dir]:
    if not os.path.exists(d):
        raise Exception(f"❌ Folder not found: {d}")

# --------------------------------------------------
# Load NORMAL images (train + test combined for re-split)
# --------------------------------------------------
for file in sorted(os.listdir(train_normal_dir)):
    if file.lower().endswith(('.tif', '.png', '.jpg')):
        normal_paths.append(os.path.join(train_normal_dir, file))

for file in sorted(os.listdir(test_normal_dir)):
    if file.lower().endswith(('.tif', '.png', '.jpg')):
        normal_paths.append(os.path.join(test_normal_dir, file))

# --------------------------------------------------
# Load ABNORMAL images + masks
# --------------------------------------------------
for file in sorted(os.listdir(test_abnormal_dir)):
    if not file.lower().endswith(('.tif', '.png', '.jpg')):
        continue
    img_path  = os.path.join(test_abnormal_dir, file)
    base      = os.path.splitext(file)[0]
    mask_path = os.path.join(masks_dir, base + "_mask.tif")
    if os.path.exists(mask_path):
        abnormal_paths.append((img_path, mask_path))
    else:
        abnormal_paths.append((img_path, None))

# --------------------------------------------------
# Print raw dataset statistics
# --------------------------------------------------
print("\n========== RAW DATASET INFO ==========")
print("Total normal images   :", len(normal_paths))
print("Total abnormal images :", len(abnormal_paths))
print("Total masks           :", sum(1 for _, m in abnormal_paths if m))

# --------------------------------------------------
# Split normal 80% train / 20% test
# --------------------------------------------------
random.shuffle(normal_paths)
split         = int(len(normal_paths) * 0.8)
train_normals = normal_paths[:split]
test_normals  = normal_paths[split:]

# --------------------------------------------------
# ✅ BALANCE: cap ABNORMAL to match NORMAL test count → ratio 1:1
# --------------------------------------------------
n_test_normal = len(test_normals)

random.shuffle(abnormal_paths)
test_abnormals = abnormal_paths[:n_test_normal]

print(f"\n[BALANCE] ABNORMAL capped: "
      f"{len(abnormal_paths)} → {len(test_abnormals)} "
      f"(= NORMAL test count {n_test_normal})")

# --------------------------------------------------
# Create train/test folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "train", "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "ABNORMAL"), exist_ok=True)

# --------------------------------------------------
# Resize and save helper
# --------------------------------------------------
def process_and_save(src, dst, is_mask=False):

    if is_mask:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(src)

    if img is None:
        print(f"❌ Could not read: {src}")
        return

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img = cv2.resize(img, (256, 256), interpolation=interpolation)
    cv2.imwrite(dst, img)

# --------------------------------------------------
# Save training NORMAL images
# --------------------------------------------------
for path in train_normals:
    name = os.path.basename(path)
    dst  = os.path.join(target_dir, "train", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save testing NORMAL images
# --------------------------------------------------
for path in test_normals:
    name = os.path.basename(path)
    dst  = os.path.join(target_dir, "test", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save ABNORMAL images + masks (balanced only)
# --------------------------------------------------
for img_path, mask_path in test_abnormals:

    name = os.path.basename(img_path)
    base = os.path.splitext(name)[0]

    dst      = os.path.join(target_dir, "test",  "ABNORMAL", name)
    mask_dst = os.path.join(target_dir, "masks", base + "_mask.tif")

    process_and_save(img_path, dst)

    if mask_path is not None:
        copyfile(mask_path, mask_dst)

# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n========== FINAL DATASET ==========")
print("Train NORMAL  :", len(train_normals))
print("Test  NORMAL  :", len(test_normals))
print("Test  ABNORMAL:", len(test_abnormals))
print("Masks         :", sum(1 for _, m in test_abnormals if m))
print(f"\nClass ratio (test) : "
      f"NORMAL {len(test_normals)} : ABNORMAL {len(test_abnormals)} "
      f"= 1 : {len(test_abnormals)/max(len(test_normals),1):.2f}")
print("\n✅ Dataset preparation complete.")