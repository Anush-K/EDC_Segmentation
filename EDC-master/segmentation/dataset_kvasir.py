import os
import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


# ===========================================================================
# AD_Dataset — used by edc_kvasir.py for anomaly detection training
# ===========================================================================
class AD_Dataset(Dataset):

    def __init__(
        self,
        name='kvasir',
        train=True,
        data_dir='./KVASIR',
        img_size=256,
        crop_size=256,
    ):

        self.train = train

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------
        if train:

            self.root = os.path.join(
                data_dir,
                'train',
                'NORMAL'
            )

        # --------------------------------------------------
        # TEST
        # --------------------------------------------------
        else:

            self.normal_root = os.path.join(
                data_dir,
                'test',
                'NORMAL'
            )

            self.abnormal_root = os.path.join(
                data_dir,
                'test',
                'ABNORMAL'
            )

        # --------------------------------------------------
        # TRANSFORMS
        # --------------------------------------------------
        self.transform = transforms.Compose([

            transforms.Resize(
                (img_size, img_size)
            ),

            transforms.ToTensor(),

        ])

        self.img_paths = []
        self.targets   = []

        # --------------------------------------------------
        # TRAIN NORMAL IMAGES
        # --------------------------------------------------
        if train:

            for file in sorted(
                os.listdir(self.root)
            ):

                if file.lower().endswith(
                    ('.png', '.jpg', '.jpeg')
                ):

                    self.img_paths.append(
                        os.path.join(
                            self.root,
                            file
                        )
                    )

                    self.targets.append(0)

        # --------------------------------------------------
        # TEST NORMAL
        # --------------------------------------------------
        else:

            for file in sorted(
                os.listdir(self.normal_root)
            ):

                if file.lower().endswith(
                    ('.png', '.jpg', '.jpeg')
                ):

                    self.img_paths.append(
                        os.path.join(
                            self.normal_root,
                            file
                        )
                    )

                    self.targets.append(0)

            # --------------------------------------------------
            # TEST ABNORMAL
            # (includes Kvasir-SEG polyps + kvasir-v2 abnormal classes)
            # --------------------------------------------------
            for file in sorted(
                os.listdir(self.abnormal_root)
            ):

                if file.lower().endswith(
                    ('.png', '.jpg', '.jpeg')
                ):

                    self.img_paths.append(
                        os.path.join(
                            self.abnormal_root,
                            file
                        )
                    )

                    self.targets.append(1)

    def __len__(self):

        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]

        label    = self.targets[idx]

        filename = os.path.basename(img_path)

        # --------------------------------------------------
        # LOAD IMAGE
        # --------------------------------------------------
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        # --------------------------------------------------
        # DUMMY MASK FOR EDC COMPATIBILITY
        # Shape: (1, H, W)
        # --------------------------------------------------
        dummy_mask = torch.zeros(
            (
                1,
                image.shape[1],
                image.shape[2]
            ),
            dtype=torch.float32
        )

        # --------------------------------------------------
        # EDC FORMAT
        # idx, image, gt_mask, label, filename
        # --------------------------------------------------
        return (
            idx,
            image,
            dummy_mask,
            label,
            filename
        )

    def get_dset(self):

        return self


# ===========================================================================
# KVASIRSegDataset — used by train_segmentation_kvasir.py
# Returns (image_tensor, heatmap_tensor, mask_tensor)
#
# Expected folder layout under `root`:
#   images/    — kvasir-seg__<name>.jpg  etc.
#   heatmaps/  — <basename>.png  (EDC anomaly maps)
#   masks/     — <basename>_mask.png  (GT polyp masks)
# ===========================================================================
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


class KVASIRSegDataset(Dataset):
    """
    Returns (image_tensor, heatmap_tensor, mask_tensor).

    image_tensor   : (3, 256, 256) float32 in [0, 1]  — RGB image
    heatmap_tensor : (1, 256, 256) float32 in [0, 1]  — EDC anomaly map
    mask_tensor    : (1, 256, 256) float32 binary      — GT polyp mask

    train_segmentation_kvasir.py concatenates image + heatmap → (4, 256, 256)
    for the UNet, and passes heatmap separately to hgbl_loss().

    Expected folder layout under `root`:
        images/    — e.g. kvasir-seg__cjfd3wz0000n6026nrb5ikxss.jpg
        heatmaps/  — e.g. kvasir-seg__cjfd3wz0000n6026nrb5ikxss.png
        masks/     — e.g. kvasir-seg__cjfd3wz0000n6026nrb5ikxss_mask.png
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
        base = os.path.splitext(name)[0]   # e.g. "kvasir-seg__cjfd3wz..."

        # ---- Image (RGB) -------------------------------------------------
        img_path = os.path.join(self.image_dir, name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        # ---- Mask (binary) -----------------------------------------------
        # Mask named: <basename>_mask.png
        mask_name = base + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(
                f"Mask not found: {mask_path}\n"
                f"  Expected: '{mask_name}'"
            )
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)

        # ---- Augmentation (image + mask jointly) -------------------------
        if self.aug is not None:
            aug   = self.aug(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        # ---- Heatmap (loaded AFTER augmentation) -------------------------
        # Heatmap named: <basename>.png
        hm_name = base + '.png'
        hm_path = os.path.join(self.heatmap_dir, hm_name)
        heatmap = cv2.imread(hm_path, 0)
        if heatmap is None:
            raise FileNotFoundError(
                f"Heatmap not found: {hm_path}\n"
                f"  Expected: '{hm_name}'"
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
