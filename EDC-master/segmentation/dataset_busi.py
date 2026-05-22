# segmentation/dataset_busi.py
#
# BUSI version of dataset.py (LGG).
# __getitem__ returns heatmap as a SEPARATE tensor alongside the image,
# rather than concatenating it as a 4th channel.
#
# BUSI-specific notes:
#   - Images   : <basename>.png         e.g. malignant (1).png
#   - Masks    : <basename>_mask.png    e.g. malignant (1)_mask.png
#   - Heatmaps : <basename>.png         e.g. malignant (1).png

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_seg_transform(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5,
            ),
            A.ElasticTransform(
                alpha=30, sigma=5, alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101, p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.3,
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ], additional_targets={'mask': 'mask'})
    return None


class BUSISegDataset(Dataset):
    """
    Returns (image_tensor, heatmap_tensor, mask_tensor).

    image_tensor   : (3, 256, 256) float32 in [0, 1]
    heatmap_tensor : (1, 256, 256) float32 in [0, 1]
    mask_tensor    : (1, 256, 256) float32 binary

    Expected folder layout under `root`:
        images/    — e.g. malignant (1).png
        heatmaps/  — e.g. malignant (1).png
        masks/     — e.g. malignant (1)_mask.png
    """

    def __init__(self, root, train=True):
        self.image_dir   = os.path.join(root, 'images')
        self.heatmap_dir = os.path.join(root, 'heatmaps')
        self.mask_dir    = os.path.join(root, 'masks')
        self.train       = train
        self.aug         = get_seg_transform(train=train)

        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if not f.startswith('.')
            and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]   # e.g. "malignant (1)"

        # ---- Image (RGB) -------------------------------------------------
        img_path = os.path.join(self.image_dir, name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        # ---- Mask (binary) -----------------------------------------------
        # ✅ FIX: mask is named <basename>_mask.png
        mask_name = base + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(
                f"Mask not found: {mask_path}\n"
                f"  Expected name: '{mask_name}'\n"
                f"  Check masks/ folder contents."
            )
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)

        # ---- Augmentation (image + mask jointly) -------------------------
        if self.aug is not None:
            aug   = self.aug(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        # ---- Heatmap (loaded AFTER augmentation) -------------------------
        # ✅ Heatmap named <basename>.png (no _mask suffix)
        hm_name  = base + '.png'
        hm_path  = os.path.join(self.heatmap_dir, hm_name)
        heatmap  = cv2.imread(hm_path, 0)
        if heatmap is None:
            raise FileNotFoundError(
                f"Heatmap not found: {hm_path}\n"
                f"  Expected name: '{hm_name}'\n"
                f"  Check heatmaps/ folder contents."
            )
        heatmap = cv2.resize(heatmap, (256, 256))

        # ---- To float tensors --------------------------------------------
        image_t   = torch.tensor(
            image.astype(np.float32) / 255.0).permute(2, 0, 1)   # (3,256,256)
        heatmap_t = torch.tensor(
            heatmap.astype(np.float32) / 255.0).unsqueeze(0)      # (1,256,256)
        mask_t    = torch.tensor(
            mask.astype(np.float32)).unsqueeze(0)                  # (1,256,256)

        return image_t, heatmap_t, mask_t