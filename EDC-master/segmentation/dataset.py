import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LGGSegDataset(Dataset):

    def __init__(self, root, use_heatmap=False):

        self.image_dir = os.path.join(root, "images")
        self.heatmap_dir = os.path.join(root, "heatmaps")
        self.mask_dir = os.path.join(root, "masks")

        self.files = sorted(os.listdir(self.image_dir))
        self.use_heatmap = use_heatmap

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        name = self.files[idx]

        image = cv2.imread(os.path.join(self.image_dir, name))
        image = cv2.resize(image, (256,256))

        image = image.astype(np.float32) / 255.0
        image = image.transpose(2,0,1)

        # mask_name = name.replace(".tif","_mask.tif")
        mask_name = name #heatmaps/mask would be of form TCGA_CS_4942_19970222_11.tif and not *_mask.tif because of this line
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name),0)
        mask = cv2.resize(mask,(256,256))
        mask = (mask > 0).astype(np.float32)

        if self.use_heatmap:

            heatmap = cv2.imread(
                os.path.join(self.heatmap_dir,name.replace(".tif",".png")),
                0
            )

            heatmap = cv2.resize(heatmap,(256,256))
            heatmap = heatmap.astype(np.float32)/255.0
            heatmap = heatmap * 0.5
            heatmap = np.expand_dims(heatmap,0)

            image = np.concatenate([image,heatmap],axis=0)

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)