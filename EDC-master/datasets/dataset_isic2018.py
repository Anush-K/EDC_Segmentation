import os
import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class AD_Dataset(Dataset):

    def __init__(
        self,
        name='skin',
        train=True,
        data_dir='./ISIC2018',
        img_size=256,
        crop_size=224,
        imagenet_norm=True,
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
        # imagenet_norm=True  → normalize with ImageNet stats
        # crop_size=224       → center crop after resize
        # --------------------------------------------------
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
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
        # Shape: (1, H, W) — matches cropped spatial size
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