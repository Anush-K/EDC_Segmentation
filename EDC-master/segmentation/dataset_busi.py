import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BUSISegDataset(Dataset):

    def __init__(self, root, use_heatmap=False, use_pseudo=False):

        self.image_dir = os.path.join(root, "images")
        self.heatmap_dir = os.path.join(root, "heatmaps")
        self.mask_dir = os.path.join(root, "masks")

        # safety check
        if not os.path.exists(self.image_dir):
            raise Exception(f"❌ Image folder not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise Exception(f"❌ Mask folder not found: {self.mask_dir}")

        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.use_heatmap = use_heatmap
        self.use_pseudo = use_pseudo

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        name = self.files[idx]
        base = os.path.splitext(name)[0]

        # ---------------- IMAGE ----------------
        img_path = os.path.join(self.image_dir, name)
        image = cv2.imread(img_path)

        if image is None:
            raise Exception(f"❌ Failed to read image: {img_path}")

        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        # ---------------- MASK ----------------
        if self.use_pseudo:
            heatmap_path = os.path.join(self.heatmap_dir, base + ".png")

            if not os.path.exists(heatmap_path):
                raise Exception(f"❌ Missing heatmap: {heatmap_path}")

            heatmap = cv2.imread(heatmap_path, 0)
            heatmap = cv2.resize(heatmap, (256, 256))
            heatmap = heatmap.astype(np.float32) / 255.0

            mask = (heatmap > 0.6).astype(np.float32)

        else:
            mask_path = os.path.join(self.mask_dir, base + ".png")

            if not os.path.exists(mask_path):
                raise Exception(f"❌ Missing mask: {mask_path}")

            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 0).astype(np.float32)

        # ---------------- HEATMAP INPUT ----------------
        if self.use_heatmap:

            heatmap_path = os.path.join(self.heatmap_dir, base + ".png")

            if not os.path.exists(heatmap_path):
                raise Exception(f"❌ Missing heatmap: {heatmap_path}")

            heatmap = cv2.imread(heatmap_path, 0)
            heatmap = cv2.resize(heatmap, (256, 256))
            heatmap = heatmap.astype(np.float32) / 255.0

            heatmap = np.expand_dims(heatmap * 0.3, 0)

            image = np.concatenate([image, heatmap], axis=0)

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)