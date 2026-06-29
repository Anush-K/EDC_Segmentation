import os
import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class AD_Dataset(Dataset):

    def __init__(
        self,
        name='oct2017',
        train=True,
        data_dir='./OCT2017',
        img_size=256,
        crop_size=256,
        train_samples_limit=10000,
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

            # FIX: OCT2017's raw official structure has 3 separate
            # disease-class folders (CNV, DME, DRUSEN), not a single
            # pre-merged ABNORMAL folder. Confirmed against
            # config_oct2017.py's own documented layout -- this dataset
            # is never run through a prepare_*.py merge script.
            self.abnormal_roots = [
                os.path.join(data_dir, 'test', 'CNV'),
                os.path.join(data_dir, 'test', 'DME'),
                os.path.join(data_dir, 'test', 'DRUSEN'),
            ]

        # --------------------------------------------------
        # TRANSFORMS
        # FIX: official edc_oct.py never overrides imagenet_norm, so it
        # uses the shared dataset.py default (imagenet_norm=True).
        # Confirmed via official repo grep -- OCT2017 IS normalized,
        # same as every other dataset. The prior "raw pixel distribution"
        # comment was unverified and incorrect.
        # No center crop — resize directly to img_size
        # --------------------------------------------------
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

        self.img_paths = []
        self.targets   = []

        # --------------------------------------------------
        # TRAIN NORMAL IMAGES
        # OCT2017 train/NORMAL is very large (~26k images)
        # train_samples_limit caps it to avoid memory issues
        # --------------------------------------------------
        if train:

            all_files = sorted([
                f for f in os.listdir(self.root)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            # Apply limit if set
            if train_samples_limit is not None and train_samples_limit > 0:
                all_files = all_files[:train_samples_limit]

            for file in all_files:

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
            # TEST ABNORMAL (CNV + DME + DRUSEN combined)
            # --------------------------------------------------
            for abnormal_root in self.abnormal_roots:

                for file in sorted(
                    os.listdir(abnormal_root)
                ):

                    if file.lower().endswith(
                        ('.png', '.jpg', '.jpeg')
                    ):

                        self.img_paths.append(
                            os.path.join(
                                abnormal_root,
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
        # OCT scans are grayscale — convert to RGB for
        # ResNet encoder compatibility
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