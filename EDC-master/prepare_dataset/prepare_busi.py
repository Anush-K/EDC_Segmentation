import os
import cv2
import random

# --------------------------------------------------
# SOURCE DATASET PATH
# --------------------------------------------------
source_dir = "/home/cs24d0008/EDC_Segmentation/BUSI_Orig"

# --------------------------------------------------
# TARGET DATASET PATH
# --------------------------------------------------
target_dir = "/home/cs24d0008/EDC_Segmentation/EDC-master/datasets/BUSI"

# --------------------------------------------------
# Create required folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

# --------------------------------------------------
# Dataset folders
# --------------------------------------------------
normal_dir    = os.path.join(source_dir, "normal")
malignant_dir = os.path.join(source_dir, "malignant")

# --------------------------------------------------
# Verify folders exist
# --------------------------------------------------
if not os.path.exists(normal_dir):
    raise Exception(f"❌ normal folder not found: {normal_dir}")

if not os.path.exists(malignant_dir):
    raise Exception(f"❌ malignant folder not found: {malignant_dir}")

normal_paths   = []
abnormal_paths = []

# --------------------------------------------------
# Load NORMAL images
# --------------------------------------------------
for file in os.listdir(normal_dir):

    if "_mask" in file:
        continue

    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        normal_paths.append(os.path.join(normal_dir, file))

# --------------------------------------------------
# Load ABNORMAL images + masks
# --------------------------------------------------
for file in os.listdir(malignant_dir):

    if "_mask" in file:
        continue

    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path  = os.path.join(malignant_dir, file)
    base      = os.path.splitext(file)[0]
    mask_path = os.path.join(malignant_dir, base + "_mask.png")

    if os.path.exists(mask_path):
        abnormal_paths.append((img_path, mask_path))

# --------------------------------------------------
# Print raw dataset statistics
# --------------------------------------------------
print("\n========== RAW DATASET INFO ==========")
print("Total normal images   :", len(normal_paths))
print("Total abnormal images :", len(abnormal_paths))
print("Total masks           :", len(abnormal_paths))

# --------------------------------------------------
# Split normal 80% train / 20% test
# --------------------------------------------------
random.shuffle(normal_paths)

split         = int(len(normal_paths) * 0.8)
train_normals = normal_paths[:split]
test_normals  = normal_paths[split:]

# --------------------------------------------------
# ✅ BALANCE: cap ABNORMAL to exactly match NORMAL test count
# e.g. normal test = 27  →  abnormal test = 27  →  ratio 1:1
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
# Save ABNORMAL images + masks (balanced subset only)
# --------------------------------------------------
for img_path, mask_path in test_abnormals:

    name     = os.path.basename(img_path)
    base     = os.path.splitext(name)[0]
    dst      = os.path.join(target_dir, "test",  "ABNORMAL", name)
    mask_dst = os.path.join(target_dir, "masks", base + "_mask.png")

    process_and_save(img_path,  dst)
    process_and_save(mask_path, mask_dst, is_mask=True)

# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n========== FINAL DATASET ==========")
print("Train NORMAL  :", len(train_normals))
print("Test  NORMAL  :", len(test_normals))
print("Test  ABNORMAL:", len(test_abnormals))
print("Masks         :", len(test_abnormals))
print(f"\nClass ratio (test) : "
      f"NORMAL {len(test_normals)} : ABNORMAL {len(test_abnormals)} "
      f"= 1 : {len(test_abnormals)/max(len(test_normals),1):.2f}")
print("\n✅ Dataset preparation complete.")