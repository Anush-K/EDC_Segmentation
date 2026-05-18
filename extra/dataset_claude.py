# import os
# import cv2
# import torch
# import numpy as np
# from torch.utils.data import Dataset


# class LGGSegDataset(Dataset):
#     """
#     Dataset for LGG brain tumour segmentation.

#     Arguments
#     ---------
#     root          : path to LGG_SEG folder (contains images/, masks/, heatmaps/)
#     use_heatmap   : if True, concatenate heatmap as 4th input channel (Approach 2)
#     use_pseudo    : if True, use heatmap as training GT instead of real mask (Approach 3)
#     eval_mode     : if True, always load the real mask as GT (used at test time for all approaches)
#     pseudo_mode   : 'soft'   → raw normalised heatmap values in [0,1] as GT
#                     'binary' → threshold heatmap to 0/1 using pseudo_threshold
#     pseudo_threshold : threshold value for binary pseudo mode (default 0.3)
#     skip_empty    : if True, exclude samples whose real mask is all-zero (no tumour).
#                     Recommended True: these samples carry no segmentation signal and
#                     push the model toward predicting nothing.
#     """

#     def __init__(
#         self,
#         root,
#         use_heatmap=False,
#         use_pseudo=False,
#         eval_mode=False,
#         pseudo_mode='soft',          # 'soft' or 'binary'
#         pseudo_threshold=0.3,
#         skip_empty=True,             # NEW: filter out all-zero masks
#     ):
#         self.image_dir   = os.path.join(root, "images")
#         self.heatmap_dir = os.path.join(root, "heatmaps")
#         self.mask_dir    = os.path.join(root, "masks")

#         self.use_heatmap      = use_heatmap
#         self.use_pseudo       = use_pseudo
#         self.eval_mode        = eval_mode
#         self.pseudo_mode      = pseudo_mode
#         self.pseudo_threshold = pseudo_threshold

#         all_files = sorted(os.listdir(self.image_dir))

#         if skip_empty:
#             self.files = self._filter_non_empty(all_files)
#             n_removed = len(all_files) - len(self.files)
#             if n_removed > 0:
#                 print(f"[LGGSegDataset] skip_empty=True: removed {n_removed} "
#                       f"empty-mask samples, {len(self.files)} remain.")
#         else:
#             self.files = all_files

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     def _filter_non_empty(self, all_files):
#         """
#         Keep only files whose corresponding mask has at least one positive pixel.
#         This is checked once at dataset construction (fast, just reads headers).
#         """
#         kept = []
#         for name in all_files:
#             mask_path = os.path.join(self.mask_dir, name)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if mask is not None and mask.max() > 0:
#                 kept.append(name)
#         return kept

#     def _load_heatmap(self, name):
#         hm_name = name.replace(".tif", ".png")
#         hm = cv2.imread(os.path.join(self.heatmap_dir, hm_name), cv2.IMREAD_GRAYSCALE)
#         if hm is None:
#             raise FileNotFoundError(
#                 f"Heatmap not found: {os.path.join(self.heatmap_dir, hm_name)}"
#             )
#         hm = cv2.resize(hm, (256, 256))
#         return hm.astype(np.float32) / 255.0   # [0, 1]

#     # ------------------------------------------------------------------
#     # Dataset interface
#     # ------------------------------------------------------------------

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]

#         # ---- image ----
#         image = cv2.imread(os.path.join(self.image_dir, name))
#         image = cv2.resize(image, (256, 256))
#         image = image.astype(np.float32) / 255.0
#         image = image.transpose(2, 0, 1)        # HWC → CHW

#         # ---- ground truth ----
#         if self.use_pseudo and not self.eval_mode:
#             # Training for Approach 3: heatmap is the supervision signal
#             hm = self._load_heatmap(name)

#             if self.pseudo_mode == 'soft':
#                 # Continuous values in [0,1] — gentler, better for imprecise heatmaps
#                 mask = hm

#             elif self.pseudo_mode == 'binary':
#                 # Hard 0/1 labels — works well when heatmap aligns closely with real mask
#                 mask = (hm > self.pseudo_threshold).astype(np.float32)

#             else:
#                 raise ValueError(f"pseudo_mode must be 'soft' or 'binary', got '{self.pseudo_mode}'")

#         else:
#             # Real mask — used for Approaches 1 & 2 always, and for Approach 3 at eval time
#             mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
#             mask = cv2.resize(mask, (256, 256))
#             mask = (mask > 0).astype(np.float32)

#         # ---- optional heatmap input channel (Approach 2) ----
#         if self.use_heatmap:
#             hm = self._load_heatmap(name)
#             hm = np.expand_dims(hm, 0)          # (1, H, W)
#             image = np.concatenate([image, hm], axis=0)   # (4, H, W)

#         return torch.tensor(image), torch.tensor(mask).unsqueeze(0)

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class LGGSegDataset(Dataset):
    """
    Dataset for LGG brain tumour segmentation.

    eval_mode : 'none'   → training (use pseudo GT if use_pseudo=True)
                'real'   → test against real mask
                'pseudo' → test against thresholded heatmap (Approach 3 internal eval)
    pseudo_mode       : 'soft' | 'binary'
    pseudo_threshold  : threshold for binary pseudo GT and pseudo eval mode
    skip_empty        : drop samples with all-zero real mask
    """

    def __init__(
        self,
        root,
        use_heatmap=False,
        use_pseudo=False,
        eval_mode='none',
        pseudo_mode='soft',
        pseudo_threshold=0.60,      # set from diagnose_heatmap.py: best threshold is 0.60
        skip_empty=True,
    ):
        self.image_dir        = os.path.join(root, "images")
        self.heatmap_dir      = os.path.join(root, "heatmaps")
        self.mask_dir         = os.path.join(root, "masks")

        self.use_heatmap      = use_heatmap
        self.use_pseudo       = use_pseudo
        self.eval_mode        = eval_mode
        self.pseudo_mode      = pseudo_mode
        self.pseudo_threshold = pseudo_threshold

        assert eval_mode in ('none', 'real', 'pseudo'), \
            f"eval_mode must be 'none', 'real', or 'pseudo'. Got '{eval_mode}'"

        all_files = sorted(os.listdir(self.image_dir))
        if skip_empty:
            self.files = self._filter_non_empty(all_files)
            removed = len(all_files) - len(self.files)
            if removed:
                print(f"[LGGSegDataset] Removed {removed} empty-mask samples. "
                      f"{len(self.files)} remain.")
        else:
            self.files = all_files

    def _filter_non_empty(self, all_files):
        kept = []
        for name in all_files:
            mask = cv2.imread(
                os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE
            )
            if mask is not None and mask.max() > 0:
                kept.append(name)
        return kept

    def _load_heatmap(self, name):
        hm_name = name.replace(".tif", ".png")
        hm = cv2.imread(
            os.path.join(self.heatmap_dir, hm_name), cv2.IMREAD_GRAYSCALE
        )
        if hm is None:
            raise FileNotFoundError(
                f"Heatmap not found: {os.path.join(self.heatmap_dir, hm_name)}"
            )
        hm = cv2.resize(hm, (256, 256))
        return hm.astype(np.float32) / 255.0

    def _load_real_mask(self, name):
        mask = cv2.imread(
            os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(mask, (256, 256))
        return (mask > 0).astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # ---- input image ----
        image = cv2.imread(os.path.join(self.image_dir, name))
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        # ---- ground truth ----
        if self.eval_mode == 'none':
            if self.use_pseudo:
                hm = self._load_heatmap(name)
                if self.pseudo_mode == 'soft':
                    mask = hm                                           # [0,1] soft labels
                else:
                    mask = (hm > self.pseudo_threshold).astype(np.float32)  # binary
            else:
                mask = self._load_real_mask(name)

        elif self.eval_mode == 'real':
            mask = self._load_real_mask(name)

        elif self.eval_mode == 'pseudo':
            # Binary thresholded heatmap as GT — used to measure
            # whether the model successfully learned from pseudo supervision
            hm = self._load_heatmap(name)
            mask = (hm > self.pseudo_threshold).astype(np.float32)

        # ---- optional 4th channel: heatmap input (Approach 2) ----
        if self.use_heatmap:
            hm = self._load_heatmap(name)
            hm = np.expand_dims(hm, 0)
            image = np.concatenate([image, hm], axis=0)

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)