import os
import cv2
import random

# --------------------------------------------------
# SOURCE DATASET PATH
# --------------------------------------------------
source_dir = "/home/cs24d0008/EDC_Segmentation/COVID-19/COVID-19_Radiography_Dataset"

# --------------------------------------------------
# TARGET DATASET PATH
# --------------------------------------------------
target_dir = "/home/cs24d0008/EDC_Segmentation/EDC-master/datasets/COVID19"

# --------------------------------------------------
# Create required folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

# --------------------------------------------------
# Dataset folder names (exactly as on disk)
# NORMAL   → Normal/images/          (10,192 items)
# ABNORMAL → COVID/images/           ( 3,616 items)
#            Lung_Opacity/images/     ( 6,012 items)
#            Viral Pneumonia/images/  ( 1,345 items)
# --------------------------------------------------
NORMAL_CLASS     = "Normal"
ABNORMAL_CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Viral Pneumonia",
]

# --------------------------------------------------
# Verify all class folders exist
# --------------------------------------------------
for cls in [NORMAL_CLASS] + ABNORMAL_CLASSES:
    img_dir = os.path.join(source_dir, cls, "images")
    if not os.path.exists(img_dir):
        raise Exception(f"❌ Folder not found: {img_dir}")

normal_paths   = []
abnormal_paths = []

# --------------------------------------------------
# Load NORMAL images
# --------------------------------------------------
normal_img_dir = os.path.join(source_dir, NORMAL_CLASS, "images")

for file in sorted(os.listdir(normal_img_dir)):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    normal_paths.append(os.path.join(normal_img_dir, file))

# --------------------------------------------------
# Load ABNORMAL images + masks
# --------------------------------------------------
for cls in ABNORMAL_CLASSES:

    img_dir  = os.path.join(source_dir, cls, "images")
    mask_dir = os.path.join(source_dir, cls, "masks")

    for file in sorted(os.listdir(img_dir)):

        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path  = os.path.join(img_dir,  file)
        mask_path = os.path.join(mask_dir, file)

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
train_normals = normal_paths[:split]       # 8,153
test_normals  = normal_paths[split:]       # 2,039

# --------------------------------------------------
# ✅ BALANCE: cap ABNORMAL to exactly match NORMAL test count
# normal test = 2,039  →  abnormal test = 2,039  →  ratio 1:1
# --------------------------------------------------
n_test_normal = len(test_normals)          # 2,039

random.shuffle(abnormal_paths)
test_abnormals = abnormal_paths[:n_test_normal]   # take first 2,039

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
# Save ABNORMAL images + masks (balanced 2,039 only)
# --------------------------------------------------
for img_path, mask_path in test_abnormals:

    name = os.path.basename(img_path)
    base = os.path.splitext(name)[0]

    dst      = os.path.join(target_dir, "test",  "ABNORMAL", name)
    mask_dst = os.path.join(target_dir, "masks", base + "_mask.png")

    process_and_save(img_path, dst)

    if mask_path is not None:
        process_and_save(mask_path, mask_dst, is_mask=True)

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