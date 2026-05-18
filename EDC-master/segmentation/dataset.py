# segmentation/dataset.py
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


# ---------------------------------------------------------------------------
# Augmentation pipelines
# FIX: added proper augmentation for training; removed the magic 0.3 heatmap
#      scalar (the UNet's first conv layer learns the correct weighting itself)
# ---------------------------------------------------------------------------
def get_seg_transform(train=True):
    """
    Returns an albumentations pipeline.
    When use_heatmap=True the caller stacks the heatmap as an extra channel
    AFTER augmentation (so we augment consistently on the image/mask pair only).
    """
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
    else:
        return None     # no augmentation at test time


# ---------------------------------------------------------------------------
# LGGSegDataset
# ---------------------------------------------------------------------------
class LGGSegDataset(Dataset):
    """
    Loads (image, [heatmap,] mask) triples from:
        root/images/   *.tif
        root/heatmaps/ *.png
        root/masks/    *.tif

    Args:
        root       : path to LGG_SEG folder
        use_heatmap: if True, concatenate heatmap as channel 4 of the image
        train      : if True, apply data augmentation
    """

    def __init__(self, root, use_heatmap=False, train=True):
        self.image_dir   = os.path.join(root, 'images')
        self.heatmap_dir = os.path.join(root, 'heatmaps')
        self.mask_dir    = os.path.join(root, 'masks')

        self.files       = sorted([
            f for f in os.listdir(self.image_dir)
            if not f.startswith('.')
        ])
        self.use_heatmap = use_heatmap
        self.train       = train
        self.aug         = get_seg_transform(train=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # ---- image (HWC uint8) ----------------------------------------
        image = cv2.imread(os.path.join(self.image_dir, name))
        if image is None:
            raise FileNotFoundError(f"Image not found: {os.path.join(self.image_dir, name)}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        # ---- mask (HW uint8) ------------------------------------------
        mask_name = name   # filenames are already aligned (no _mask suffix)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {os.path.join(self.mask_dir, mask_name)}")
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)   # binary {0,1}

        # ---- augmentation on image + mask together --------------------
        if self.aug is not None:
            augmented = self.aug(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        # ---- to float tensors -----------------------------------------
        image_f = image.astype(np.float32) / 255.0   # (H,W,3)
        image_t = torch.tensor(image_f.transpose(2, 0, 1))  # (3,H,W)

        mask_t  = torch.tensor(mask.astype(np.float32)).unsqueeze(0)  # (1,H,W)

        # ---- optional heatmap channel ---------------------------------
        if self.use_heatmap:
            hm_name = name.replace('.tif', '.png')
            hm_path = os.path.join(self.heatmap_dir, hm_name)
            heatmap = cv2.imread(hm_path, 0)
            if heatmap is None:
                raise FileNotFoundError(f"Heatmap not found: {hm_path}")
            heatmap = cv2.resize(heatmap, (256, 256))

            # FIX: removed the arbitrary 0.3 multiplier — let the first conv
            #      learn the appropriate scale from the data
            heatmap_f = heatmap.astype(np.float32) / 255.0  # (H,W)
            heatmap_t = torch.tensor(heatmap_f).unsqueeze(0)  # (1,H,W)

            image_t = torch.cat([image_t, heatmap_t], dim=0)  # (4,H,W)

        return image_t, mask_t