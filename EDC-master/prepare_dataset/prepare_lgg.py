import os
import cv2
import random
import numpy as np
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, required=True)
parser.add_argument('--save-folder', type=str, required=True)
args = parser.parse_args()

source_dir = args.data_folder
target_dir = args.save_folder
os.makedirs(os.path.join(target_dir, "masks"), exist_ok=True)

random.seed(1)

normal_paths = []
abnormal_paths = []

for root, dirs, files in os.walk(source_dir):

    for file in files:

        if file.endswith("_mask.tif"):

            mask_path = os.path.join(root, file)

            # derive image filename
            img_file = file.replace("_mask.tif", ".tif")
            img_path = os.path.join(root, img_file)

            if not os.path.exists(img_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            if np.sum(mask) == 0:
                normal_paths.append(img_path)
            else:
                abnormal_paths.append((img_path, mask_path))

print("Total normal:", len(normal_paths))
abnormal_count = len([m for m, _ in abnormal_paths])
print("Total abnormal:", abnormal_count)
mask_count = len([m for _, m in abnormal_paths])
print("Total masks:", mask_count)

# split normals 80/20
random.shuffle(normal_paths)

split = int(len(normal_paths) * 0.8)

train_normals = normal_paths[:split]
test_normals = normal_paths[split:]

# create folders
os.makedirs(os.path.join(target_dir, "train", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test", "ABNORMAL"), exist_ok=True)

# resize function
def process_and_save(src, dst):

    img = cv2.imread(src)

    if img is None:
        return

    img = cv2.resize(img, (256,256))

    cv2.imwrite(dst, img)

# save train normals
for path in train_normals:

    name = os.path.basename(path)

    dst = os.path.join(target_dir, "train", "NORMAL", name)

    process_and_save(path, dst)

# save test normals
for path in test_normals:

    name = os.path.basename(path)

    dst = os.path.join(target_dir, "test", "NORMAL", name)

    process_and_save(path, dst)

# save abnormal
for img_path, mask_path in abnormal_paths:

    name = os.path.basename(img_path)

    dst = os.path.join(target_dir, "test", "ABNORMAL", name)
    mask_dst = os.path.join(target_dir, "masks", os.path.basename(mask_path))

    process_and_save(img_path, dst)
    copyfile(mask_path, mask_dst)

    process_and_save(img_path, dst)

print("Dataset preparation complete.")