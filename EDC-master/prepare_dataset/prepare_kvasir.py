import os
import cv2
import random
import shutil

# --------------------------------------------------
# SOURCE DATASET PATHS
# --------------------------------------------------
KVASIR_SEG_DIR   = "/home/cs24d0008/EDC_Segmentation/kvasir-seg/Kvasir-SEG"
KVASIR_V2_DIR    = "/home/cs24d0008/EDC_Segmentation/kvasir-dataset-v2/kvasir-dataset-v2"

# --------------------------------------------------
# TARGET DATASET PATH
# --------------------------------------------------
target_dir = "/home/cs24d0008/EDC_Segmentation/EDC-master/datasets/KVASIR"

# --------------------------------------------------
# Kvasir-SEG subfolders
# --------------------------------------------------
SEG_IMG_DIR  = os.path.join(KVASIR_SEG_DIR, "images")
SEG_MASK_DIR = os.path.join(KVASIR_SEG_DIR, "masks")

# --------------------------------------------------
# kvasir-dataset-v2: NORMAL class folders
# --------------------------------------------------
NORMAL_SOURCES = [
    os.path.join(KVASIR_V2_DIR, "normal-cecum"),
    os.path.join(KVASIR_V2_DIR, "normal-pylorus"),
    os.path.join(KVASIR_V2_DIR, "normal-z-line"),
]

# --------------------------------------------------
# kvasir-dataset-v2: ABNORMAL class folders (no masks)
# --------------------------------------------------
ABNORMAL_V2_SOURCES = [
    os.path.join(KVASIR_V2_DIR, "dyed-lifted-polyps"),
    os.path.join(KVASIR_V2_DIR, "dyed-resection-margins"),
    os.path.join(KVASIR_V2_DIR, "esophagitis"),
    os.path.join(KVASIR_V2_DIR, "polyps"),
    os.path.join(KVASIR_V2_DIR, "ulcerative-colitis"),
]

# --------------------------------------------------
# Verify key folders exist
# --------------------------------------------------
for d in [SEG_IMG_DIR, SEG_MASK_DIR] + NORMAL_SOURCES + ABNORMAL_V2_SOURCES:
    if not os.path.exists(d):
        raise Exception(f"❌ Folder not found: {d}")

# --------------------------------------------------
# Create output folders
# --------------------------------------------------
os.makedirs(os.path.join(target_dir, "train", "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "NORMAL"),   exist_ok=True)
os.makedirs(os.path.join(target_dir, "test",  "ABNORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "masks"),              exist_ok=True)

random.seed(1)

# --------------------------------------------------
# Resize and save helper
# --------------------------------------------------
def process_and_save(src, dst, is_mask=False):

    if is_mask:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(src)

    if img is None:
        print(f"  ❌ Could not read: {src}")
        return

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img = cv2.resize(img, (256, 256), interpolation=interpolation)
    cv2.imwrite(dst, img)

# --------------------------------------------------
# Collect NORMAL images from all 3 normal folders
# --------------------------------------------------
normal_paths = []

for folder in NORMAL_SOURCES:

    folder_name = os.path.basename(folder)
    count       = 0

    for file in sorted(os.listdir(folder)):

        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            normal_paths.append((
                os.path.join(folder, file),
                folder_name + "__" + file
            ))
            count += 1

    print(f"  Normal  [{folder_name}] : {count} images")

# --------------------------------------------------
# Collect ABNORMAL from Kvasir-SEG (has masks)
# --------------------------------------------------
abnormal_seg_pairs = []

for file in sorted(os.listdir(SEG_IMG_DIR)):

    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path  = os.path.join(SEG_IMG_DIR,  file)
    mask_path = os.path.join(SEG_MASK_DIR, file)

    if os.path.exists(mask_path):
        abnormal_seg_pairs.append((img_path, mask_path, file))

print(f"\n  Abnormal [kvasir-seg] : {len(abnormal_seg_pairs)} image+mask pairs")

# --------------------------------------------------
# Collect ABNORMAL from kvasir-dataset-v2 (no masks)
# --------------------------------------------------
abnormal_v2_paths = []

for folder in ABNORMAL_V2_SOURCES:

    folder_name = os.path.basename(folder)
    count       = 0

    for file in sorted(os.listdir(folder)):

        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            abnormal_v2_paths.append((
                os.path.join(folder, file),
                folder_name + "__" + file
            ))
            count += 1

    print(f"  Abnormal [{folder_name}] : {count} images")

# --------------------------------------------------
# Summary counts
# --------------------------------------------------
total_normal   = len(normal_paths)
total_abnormal = len(abnormal_seg_pairs) + len(abnormal_v2_paths)

print(f"\n========== DATASET INFO ==========")
print(f"Total normal images   : {total_normal}")
print(f"Total abnormal images : {total_abnormal}")
print(f"  - With masks        : {len(abnormal_seg_pairs)}  (kvasir-seg)")
print(f"  - Without masks     : {len(abnormal_v2_paths)}  (kvasir-dataset-v2)")

# --------------------------------------------------
# Split NORMAL 80/20
# --------------------------------------------------
random.shuffle(normal_paths)

split         = int(len(normal_paths) * 0.8)
train_normals = normal_paths[:split]
test_normals  = normal_paths[split:]

# --------------------------------------------------
# ✅ BALANCE: cap ABNORMAL test to match NORMAL test count
# --------------------------------------------------
n_test_normal = len(test_normals)

all_abnormal_combined = [
    ('seg', img, mask, orig) for img, mask, orig in abnormal_seg_pairs
] + [
    ('v2', img, None, save) for img, save in abnormal_v2_paths
]

random.shuffle(all_abnormal_combined)
test_abnormals = all_abnormal_combined[:n_test_normal]

print(f"\n[BALANCE] ABNORMAL capped: "
      f"{total_abnormal} → {len(test_abnormals)} "
      f"(= NORMAL test count {n_test_normal})")

# --------------------------------------------------
# Save training NORMAL images
# --------------------------------------------------
print("\nSaving train NORMAL...")

for img_path, save_name in train_normals:
    dst = os.path.join(target_dir, "train", "NORMAL", save_name)
    process_and_save(img_path, dst)

# --------------------------------------------------
# Save testing NORMAL images
# --------------------------------------------------
print("Saving test NORMAL...")

for img_path, save_name in test_normals:
    dst = os.path.join(target_dir, "test", "NORMAL", save_name)
    process_and_save(img_path, dst)

# --------------------------------------------------
# Save ABNORMAL images (balanced)
# --------------------------------------------------
print("Saving test ABNORMAL (balanced)...")

seg_saved = 0
v2_saved  = 0

for entry in test_abnormals:

    kind = entry[0]

    if kind == 'seg':
        _, img_path, mask_path, orig_name = entry
        base      = os.path.splitext(orig_name)[0]
        save_name = "kvasir-seg__" + orig_name
        dst       = os.path.join(target_dir, "test",  "ABNORMAL", save_name)
        mask_dst  = os.path.join(target_dir, "masks",
                                 "kvasir-seg__" + base + "_mask.png")
        process_and_save(img_path,  dst)
        process_and_save(mask_path, mask_dst, is_mask=True)
        seg_saved += 1

    else:
        _, img_path, _, save_name = entry
        dst = os.path.join(target_dir, "test", "ABNORMAL", save_name)
        process_and_save(img_path, dst)
        v2_saved += 1

# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n========== FINAL DATASET ==========")
print(f"Train NORMAL  : {len(train_normals)}")
print(f"Test  NORMAL  : {len(test_normals)}")
print(f"Test  ABNORMAL: {len(test_abnormals)}")
print(f"  - With masks : {seg_saved}")
print(f"  - No masks   : {v2_saved}")
print(f"Masks saved    : {seg_saved}")
print(f"\nClass ratio (test) : "
      f"NORMAL {len(test_normals)} : ABNORMAL {len(test_abnormals)} "
      f"= 1 : {len(test_abnormals)/max(len(test_normals),1):.2f}")
print(f"\nDataset written to: {target_dir}")
print("\n✅ Dataset preparation complete.")