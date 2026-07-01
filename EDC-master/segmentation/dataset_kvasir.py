# segmentation/dataset_kvasir.py
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
                alpha=30, sigma=5,
                border_mode=cv2.BORDER_REFLECT_101, p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.3,
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ], additional_targets={'mask': 'mask'})
    return None


class KvasirSegDataset(Dataset):
    """
    Returns (image_tensor, heatmap_tensor, mask_tensor).
    image_tensor   : (3, 256, 256) float32
    heatmap_tensor : (1, 256, 256) float32
    mask_tensor    : (1, 256, 256) float32 binary
    """

    def __init__(self, root, train=True):
        self.image_dir   = os.path.join(root, 'images')
        self.heatmap_dir = os.path.join(root, 'heatmaps')
        self.mask_dir    = os.path.join(root, 'masks')

        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if not f.startswith('.')
        ])
        self.train = train
        self.aug   = get_seg_transform(train=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]

        # Image
        image = cv2.imread(os.path.join(self.image_dir, name))
        if image is None:
            raise FileNotFoundError(f"Image not found: {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        # Mask — Kvasir-SEG masks have same name as image
        mask_path = os.path.join(self.mask_dir, name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base + '_mask.png')
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found for: {name}")
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)

        # Augmentation
        if self.aug is not None:
            aug   = self.aug(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        # Heatmap (after augmentation)
        heatmap = cv2.imread(
            os.path.join(self.heatmap_dir, base + '.png'), 0)
        if heatmap is None:
            raise FileNotFoundError(f"Heatmap not found: {base}.png")
        heatmap = cv2.resize(heatmap, (256, 256))

        image_t   = torch.tensor(
            image.astype(np.float32) / 255.0).permute(2, 0, 1)
        heatmap_t = torch.tensor(
            heatmap.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t    = torch.tensor(
            mask.astype(np.float32)).unsqueeze(0)

        return image_t, heatmap_t, mask_t