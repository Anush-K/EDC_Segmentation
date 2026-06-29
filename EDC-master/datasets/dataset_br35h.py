import os
import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class AD_Dataset(Dataset):

    def __init__(
        self,
        name='brain',
        train=True,
        data_dir='./BR35H',
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
        # Run A: clean paper-replication — only the two genuine bugs
        # (ImageNet normalization, bn_pretrain) stay fixed. No per-image
        # intensity normalization, no augmentation — those are separate
        # hypotheses to test AFTER confirming whether the runner-level
        # fixes (lr/lr_encoder swap, stop_grad, clip, BN momentum) alone
        # close the gap.
        # --------------------------------------------------
        self.transform = transforms.Compose([

            transforms.Resize(
                (img_size, img_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

        ])

        # --------------------------------------------------
        # ✅ FIX (kept): separate transform with NO normalization, used
        # only to produce a viewable image for heatmap overlays. This
        # doesn't affect training/AUC at all — it only fixes the
        # previously-broken (all-black) heatmap visualizations.
        # --------------------------------------------------
        self.vis_transform = transforms.Compose([
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
        pil_image = Image.open(img_path).convert('RGB')

        image = self.transform(pil_image)

        # --------------------------------------------------
        # ✅ FIX (kept): real (resized, un-normalized) image for
        # visualization, replacing the old all-zero dummy_mask.
        # --------------------------------------------------
        image_vis = self.vis_transform(pil_image)

        # --------------------------------------------------
        # EDC FORMAT
        # idx, image, image_vis (for overlay), label, filename
        # --------------------------------------------------
        return (
            idx,
            image,
            image_vis,
            label,
            filename
        )

    def get_dset(self):

        return self