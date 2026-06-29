import os
import csv
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
        self.data_dir = data_dir

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
        # BUILD FILE LIST FROM THE OFFICIAL GROUND-TRUTH CSV
        # instead of trusting pre-sorted NORMAL/ABNORMAL folders.
        #
        # Official ISIC2018 Task 3 GT CSV columns:
        #   image, MEL, NV, BCC, AKIEC, BKL, DF, VASC   (one-hot)
        #
        # Label rule (matches the EDC paper, Sec. IV-A):
        #   NV == 1.0  -> NORMAL   (0)
        #   otherwise  -> ABNORMAL (1)
        #
        # TRAIN uses the official Training set, NORMAL-only.
        # TEST  uses the official Validation set, NORMAL+ABNORMAL.
        # --------------------------------------------------
        if train:
            gt_dir = os.path.join(
                data_dir, 'original', 'ISIC2018_Task3_Training_GroundTruth'
            )
            img_dir = os.path.join(
                data_dir, 'original', 'ISIC2018_Task3_Training_Input'
            )
        else:
            gt_dir = os.path.join(
                data_dir, 'original', 'ISIC2018_Task3_Validation_GroundTruth'
            )
            img_dir = os.path.join(
                data_dir, 'original', 'ISIC2018_Task3_Validation_Input'
            )

        self._build_from_official_csv(gt_dir, img_dir, normal_only=train)

    # --------------------------------------------------
    # locate the single .csv inside a GT folder
    # --------------------------------------------------
    @staticmethod
    def _find_csv(folder):

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"[dataset_isic2018.py] Ground-truth folder not found: "
                f"{folder}\n"
                f"This loader reads labels directly from the official "
                f"ISIC2018 Task 3 GroundTruth CSV. Make sure 'original/' "
                f"(with the four official ISIC2018_Task3_* subfolders) "
                f"still exists under your data_dir. If you deleted it, "
                f"re-download Training/Validation Data + GroundTruth for "
                f"Task 3 from "
                f"https://challenge.isic-archive.com/data/#2018"
            )

        for f in os.listdir(folder):
            if f.lower().endswith('.csv'):
                return os.path.join(folder, f)

        raise FileNotFoundError(
            f"[dataset_isic2018.py] No .csv file found in {folder}"
        )

    # --------------------------------------------------
    # find the actual image file for an image_id
    # --------------------------------------------------
    @staticmethod
    def _find_image(img_dir, image_id):

        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):

            candidate = os.path.join(img_dir, image_id + ext)

            if os.path.exists(candidate):
                return candidate

        return None

    # --------------------------------------------------
    # parse the GT csv and populate self.img_paths / self.targets
    # --------------------------------------------------
    def _build_from_official_csv(self, gt_dir, img_dir, normal_only):

        csv_path = self._find_csv(gt_dir)

        expected_cols = {
            'image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'
        }

        n_missing = 0

        with open(csv_path, newline='') as f:

            reader = csv.DictReader(f)

            if not expected_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"[dataset_isic2018.py] Unexpected CSV columns in "
                    f"{csv_path}: {reader.fieldnames}. "
                    f"Expected at least: {expected_cols}"
                )

            for row in reader:

                image_id  = row['image'].strip()
                is_normal = float(row['NV']) == 1.0

                # TRAIN: keep only NORMAL (matches official 6705-only set)
                if normal_only and not is_normal:
                    continue

                img_path = self._find_image(img_dir, image_id)

                if img_path is None:
                    n_missing += 1
                    continue

                self.img_paths.append(img_path)
                self.targets.append(0 if is_normal else 1)

        if n_missing:
            print(
                f"[dataset_isic2018.py] WARNING: {n_missing} image_ids "
                f"listed in {csv_path} had no matching file in {img_dir}."
            )

        # --------------------------------------------------
        # sanity check against the official EDC paper split counts
        # --------------------------------------------------
        n_normal   = sum(1 for t in self.targets if t == 0)
        n_abnormal = sum(1 for t in self.targets if t == 1)

        if self.train:
            expected_normal = 6705
            status = "OK" if n_normal == expected_normal else "MISMATCH"
            print(
                f"[dataset_isic2018.py] TRAIN loaded: {n_normal} normal "
                f"(expected {expected_normal}) [{status}]"
            )
        else:
            expected_normal, expected_abnormal = 97, 96
            status = (
                "OK" if (n_normal, n_abnormal) == (expected_normal, expected_abnormal)
                else "MISMATCH"
            )
            print(
                f"[dataset_isic2018.py] TEST loaded: {n_normal} normal / "
                f"{n_abnormal} abnormal "
                f"(expected {expected_normal}/{expected_abnormal}) [{status}]"
            )

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
