# datasets/dataset.py
import os
import random
import numpy as np
import cv2
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .transforms import PixelShuffle, CutMix, MeanDropout
from .data_utils import get_onehot

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet']  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    return pil_loader(path)


def divide255(image, **kwargs):
    return (image / 255.0).astype('float32')


# ---------------------------------------------------------------------------
# Transforms
# FIX: added proper augmentation for training; eval keeps only resize+crop
# ---------------------------------------------------------------------------
def get_transform(img_size, crop_size, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(crop_size, crop_size),
            # --- augmentations (safe for anomaly detection on MRI) ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(crop_size, crop_size),
        ])


# ---------------------------------------------------------------------------
# BasicDataset
# ---------------------------------------------------------------------------
class BasicDataset(Dataset):
    """
    Returns (idx, normalised_tensor, raw_crop_uint8, target, img_path).
    """

    def __init__(
        self,
        img_paths,
        targets=None,
        transform=None,
        train=True,
        imagenet_norm=True,
        *args, **kwargs,
    ):
        super().__init__()
        self.img_paths    = img_paths
        self.targets      = targets
        self.transform    = transform
        self.train        = train
        self.totensor     = A.Compose([
            A.Normalize() if imagenet_norm else A.Lambda(image=divide255),
            ToTensorV2(),
        ])

    def __getitem__(self, idx):
        target = None if self.targets is None else self.targets[idx]

        img      = default_loader(self.img_paths[idx])
        img      = np.array(img)
        img_path = self.img_paths[idx]

        img_t = self.transform(image=img)['image']      # augmented / cropped
        img_n = self.totensor(image=img_t)['image']     # normalised tensor

        return idx, img_n, img_t, target, img_path

    def __len__(self):
        return len(self.img_paths)


# ---------------------------------------------------------------------------
# AD_Dataset  — wraps directory structure into BasicDataset
# ---------------------------------------------------------------------------
class AD_Dataset:
    """
    Loads LGG-style dataset:
      train/NORMAL/
      test/NORMAL/
      test/ABNORMAL/
    """

    def __init__(
        self,
        name='lgg_mri',
        img_size=256,
        crop_size=256,
        train=True,
        data_dir='../LGG',
        transform=None,
        train_samples_limit=10000,
        imagenet_norm=True,
    ):
        self.name                = name
        self.train               = train
        self.data_dir            = data_dir
        self.train_samples_limit = train_samples_limit
        self.imagenet_norm       = imagenet_norm
        self.transform = transform if transform is not None \
                         else get_transform(img_size, crop_size, train)

    def get_data(self):
        if self.train:
            train_path = os.path.join(self.data_dir, 'train', 'NORMAL')
            norm_files = [
                f for f in os.listdir(train_path)
                if not f.startswith('.')
            ]
            if len(norm_files) > self.train_samples_limit:
                norm_files = random.choices(norm_files, k=self.train_samples_limit)
            img_paths = [os.path.join(train_path, f) for f in norm_files]
            targets   = list(np.zeros(len(img_paths)))
        else:
            img_paths, targets = [], []
            test_root = os.path.join(self.data_dir, 'test')
            for sub_dir in sorted(os.listdir(test_root)):
                if sub_dir.startswith('.'):
                    continue
                sub_path = os.path.join(test_root, sub_dir)
                if not os.path.isdir(sub_path):
                    continue
                files = [f for f in os.listdir(sub_path) if not f.startswith('.')]
                paths = [os.path.join(sub_path, f) for f in files]
                img_paths.extend(paths)
                label = 0 if sub_dir == 'NORMAL' else 1
                targets.extend([label] * len(paths))

        return img_paths, targets

    def get_dset(self):
        img_paths, targets = self.get_data()
        return BasicDataset(
            img_paths, targets,
            transform=self.transform,
            imagenet_norm=self.imagenet_norm,
        )