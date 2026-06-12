# datasets/dataset_busi.py
# ✅ KEY FIX: Strong augmentation for train set
# EDC trains on normal images only — with only 106 train images,
# augmentation effectively multiplies the training data.

import os
import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


# ===========================================================================
# ✅ Strong augmentation for EDC training on small BUSI normal set
# ===========================================================================
def get_edc_train_transform(img_size=256):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2,
            rotate_limit=30, p=0.7
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3, p=0.5
        ),
        A.ElasticTransform(
            alpha=30, sigma=5,
            alpha_affine=5, p=0.3
        ),
        A.GridDistortion(p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.Resize(img_size, img_size),
    ])


# ===========================================================================
# AD_Dataset — used by edc_busi.py for anomaly detection training
# ===========================================================================
class AD_Dataset(Dataset):

    def __init__(
        self,
        name='busi',
        train=True,
        data_dir='./BUSI',
        img_size=256,
        crop_size=256,
    ):
        self.train    = train
        self.img_size = img_size

        if train:
            self.root = os.path.join(data_dir, 'train', 'NORMAL')
            # ✅ Strong augmentation during training
            self.aug = get_edc_train_transform(img_size)
        else:
            self.normal_root   = os.path.join(data_dir, 'test', 'NORMAL')
            self.abnormal_root = os.path.join(data_dir, 'test', 'ABNORMAL')
            self.aug = None

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.img_paths = []
        self.targets   = []

        if train:
            for file in sorted(os.listdir(self.root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.img_paths.append(os.path.join(self.root, file))
                    self.targets.append(0)
        else:
            for file in sorted(os.listdir(self.normal_root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.img_paths.append(os.path.join(self.normal_root, file))
                    self.targets.append(0)
            for file in sorted(os.listdir(self.abnormal_root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.img_paths.append(os.path.join(self.abnormal_root, file))
                    self.targets.append(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label    = self.targets[idx]
        filename = os.path.basename(img_path)

        # ✅ Apply augmentation during training
        if self.train and self.aug is not None:
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.aug(image=image)
            image = augmented['image']
            image = torch.tensor(
                image.astype(np.float32) / 255.0
            ).permute(2, 0, 1)
        else:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)

        dummy_mask = torch.zeros(
            (1, image.shape[1], image.shape[2]),
            dtype=torch.float32
        )

        return (idx, image, dummy_mask, label, filename)

    def get_dset(self):
        return self


# ===========================================================================
# Segmentation augmentation
# ===========================================================================
def get_seg_transform(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.ElasticTransform(
                alpha=30, sigma=5, alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101, p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15, p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ], additional_targets={'mask': 'mask'})
    return None


# ===========================================================================
# BUSISegDataset — used by train_segmentation_busi.py
# ===========================================================================
class BUSISegDataset(Dataset):
    """
    Returns (image_tensor, heatmap_tensor, mask_tensor).
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
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]

        image = cv2.imread(os.path.join(self.image_dir, name))
        if image is None:
            raise FileNotFoundError(f"Image not found: {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        mask_name = base + '_mask.png'
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_name}")
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)

        if self.aug is not None:
            aug   = self.aug(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        hm_name = base + '.png'
        heatmap = cv2.imread(os.path.join(self.heatmap_dir, hm_name), 0)
        if heatmap is None:
            raise FileNotFoundError(f"Heatmap not found: {hm_name}")
        heatmap = cv2.resize(heatmap, (256, 256))

        image_t   = torch.tensor(
            image.astype(np.float32) / 255.0).permute(2, 0, 1)
        heatmap_t = torch.tensor(
            heatmap.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t    = torch.tensor(
            mask.astype(np.float32)).unsqueeze(0)

        return image_t, heatmap_t, mask_t