import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ✅ FIX: paper states APTOS images are "preprocessed to remove the
# redundant background" before resizing — this step was entirely missing.
# Fundus photos have large black borders around the circular retina;
# without removing them, a large fraction of every image is dead
# background diluting the anomaly signal. Standard approach (used widely
# in diabetic-retinopathy preprocessing pipelines): threshold to find the
# non-black retina region, crop to its bounding box.
def crop_fundus_background(pil_image, tol=7):
    img = np.array(pil_image.convert('L'))
    mask = img > tol
    if mask.sum() == 0:
        # Fully black image (shouldn't happen, but don't crop to nothing)
        return pil_image
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return pil_image.crop((int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1))


class AD_Dataset(Dataset):

    def __init__(
        self,
        name='fundus',
        train=True,
        data_dir='./APTOS',
        img_size=256,
        crop_size=256,
        imagenet_norm=True,
    ):
        self.train = train

        if train:
            self.root = os.path.join(data_dir, 'train', 'NORMAL')
        else:
            self.normal_root   = os.path.join(data_dir, 'test', 'NORMAL')
            self.abnormal_root = os.path.join(data_dir, 'test', 'ABNORMAL')

        # Original clean transform — no augmentation
        # Background crop happens in __getitem__ (PIL crop) BEFORE this
        # Resize, since the crop must operate on the original image size.
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
        if imagenet_norm:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        self.transform = transforms.Compose(transform_list)

        # ✅ FIX: separate transform with NO normalization, used only to
        # produce a viewable image for heatmap overlays. Previously this
        # slot was filled with an all-zero "dummy_mask" (same bug found
        # and fixed in Br35H) — every saved heatmap overlay would have
        # been a black square. Doesn't affect AUC/training at all.
        self.vis_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.img_paths = []
        self.targets   = []

        if train:
            for file in sorted(os.listdir(self.root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(self.root, file))
                    self.targets.append(0)
        else:
            for file in sorted(os.listdir(self.normal_root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(self.normal_root, file))
                    self.targets.append(0)
            for file in sorted(os.listdir(self.abnormal_root)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(self.abnormal_root, file))
                    self.targets.append(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label    = self.targets[idx]
        filename = os.path.basename(img_path)
        pil_image = Image.open(img_path).convert('RGB')
        # ✅ FIX: remove redundant black background BEFORE resizing
        pil_image = crop_fundus_background(pil_image)
        image     = self.transform(pil_image)
        # ✅ FIX: real image for visualization, replacing all-zero dummy_mask
        image_vis = self.vis_transform(pil_image)
        return (idx, image, image_vis, label, filename)

    def get_dset(self):
        return self