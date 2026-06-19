import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AD_Dataset(Dataset):

    def __init__(
        self,
        name='fundus',
        train=True,
        data_dir='./APTOS',
        img_size=256,
        crop_size=256,
    ):
        self.train = train

        if train:
            self.root = os.path.join(data_dir, 'train', 'NORMAL')
        else:
            self.normal_root   = os.path.join(data_dir, 'test', 'NORMAL')
            self.abnormal_root = os.path.join(data_dir, 'test', 'ABNORMAL')

        # Original clean transform — no augmentation
        self.transform = transforms.Compose([
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
        image    = Image.open(img_path).convert('RGB')
        image    = self.transform(image)
        dummy_mask = torch.zeros(
            (1, image.shape[1], image.shape[2]),
            dtype=torch.float32
        )
        return (idx, image, dummy_mask, label, filename)

    def get_dset(self):
        return self