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
# ✅ FIX: use BOTH normal + benign as NORMAL class
# benign tumors look more like normal tissue than malignant
# This gives EDC ~400 normal training images instead of 106
# --------------------------------------------------
normal_dir   = os.path.join(source_dir, "normal")
benign_dir   = os.path.join(source_dir, "benign")
malignant_dir = os.path.join(source_dir, "malignant")

# --------------------------------------------------
# Verify folders exist
# --------------------------------------------------
for d, name in [(normal_dir, "normal"), (malignant_dir, "malignant")]:
    if not os.path.exists(d):
        raise Exception(f"❌ {name} folder not found: {d}")

normal_paths   = []
abnormal_paths = []

# --------------------------------------------------
# Load NORMAL images (normal folder)
# --------------------------------------------------
for file in os.listdir(normal_dir):
    if "_mask" in file:
        continue
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        normal_paths.append(os.path.join(normal_dir, file))

print(f"Normal images          : {len(normal_paths)}")

# --------------------------------------------------
# ✅ Load BENIGN images as additional NORMAL training
# benign has masks but we treat them as normal for EDC
# --------------------------------------------------
benign_count = 0
if os.path.exists(benign_dir):
    for file in os.listdir(benign_dir):
        if "_mask" in file:
            continue
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            normal_paths.append(os.path.join(benign_dir, file))
            benign_count += 1
    print(f"Benign images (→NORMAL): {benign_count}")
else:
    print(f"⚠️  benign folder not found at {benign_dir} — using normal only")

print(f"Total NORMAL           : {len(normal_paths)}")

# --------------------------------------------------
# Load ABNORMAL images (malignant only)
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

print(f"Total ABNORMAL         : {len(abnormal_paths)}")

# --------------------------------------------------
# Print raw dataset statistics
# --------------------------------------------------
print("\n========== RAW DATASET INFO ==========")
print("Total normal images   :", len(normal_paths))
print("Total abnormal images :", len(abnormal_paths))

# --------------------------------------------------
# Split normal 80% train / 20% test
# --------------------------------------------------
random.shuffle(normal_paths)
split         = int(len(normal_paths) * 0.8)
train_normals = normal_paths[:split]
test_normals  = normal_paths[split:]

# ✅ Use ALL abnormal for test (unbalanced — more reliable AUC)
test_abnormals = abnormal_paths

print(f"\n[UNBALANCED] Using ALL {len(test_abnormals)} ABNORMAL for test")
print(f"  Train NORMAL  : {len(train_normals)}")
print(f"  Test  NORMAL  : {len(test_normals)}")
print(f"  Test  ABNORMAL: {len(test_abnormals)}")

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
# Save training NORMAL images (normal + benign)
# --------------------------------------------------
for path in train_normals:
    name = os.path.basename(path)
    # prefix benign images to avoid filename collision
    if benign_dir in path:
        name = "benign__" + name
    dst  = os.path.join(target_dir, "train", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save testing NORMAL images
# --------------------------------------------------
for path in test_normals:
    name = os.path.basename(path)
    if benign_dir in path:
        name = "benign__" + name
    dst  = os.path.join(target_dir, "test", "NORMAL", name)
    process_and_save(path, dst)

# --------------------------------------------------
# Save ALL ABNORMAL images + masks
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