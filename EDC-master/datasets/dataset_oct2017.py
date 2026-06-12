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
        # No imagenet norm — OCT2017 uses raw pixel distribution
        # No center crop — resize directly to img_size
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