"""Stage 2 dataset. Fixes the two bugs of dataset_*_seg.py:
 (1) the heatmap is now transformed TOGETHER with the image and the mask;
 (2) validation/test images are never augmented (separate dataset objects).
Augmentation is written with numpy/cv2 only (no albumentations version problems)."""
import os, csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

SIZE = 256

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def load_heat(heat_dir, stem):
    p = os.path.join(heat_dir, stem + ".npy")
    if os.path.exists(p):
        h = np.load(p).astype(np.float32)
    else:
        q = os.path.join(heat_dir, stem + ".png")
        h = cv2.imread(q, 0)
        if h is None:
            raise FileNotFoundError("heatmap not found: %s(.npy/.png)" % os.path.join(heat_dir, stem))
        h = h.astype(np.float32) / 255.0
    if h.ndim == 3:
        h = h[..., 0]
    h = cv2.resize(h, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return h / max(float(h.max()), 1e-8)          # per-image max normalisation (as in the paper)

# ---------------- joint augmentation ----------------
def _affine(arrs, rng):
    ang = rng.uniform(-15, 15); sc = 1 + rng.uniform(-0.1, 0.1)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * SIZE
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, sc); M[:, 2] += (tx, ty)
    out = []
    for a, interp in arrs:
        out.append(cv2.warpAffine(a, M, (SIZE, SIZE), flags=interp, borderMode=cv2.BORDER_REFLECT_101))
    return out

def _elastic(arrs, rng, alpha=30, sigma=5):
    dx = cv2.GaussianBlur((rng.rand(SIZE, SIZE) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(SIZE, SIZE) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    mx, my = (x + dx).astype(np.float32), (y + dy).astype(np.float32)
    return [cv2.remap(a, mx, my, interp, borderMode=cv2.BORDER_REFLECT_101) for a, interp in arrs]

def augment(img, heat, mask, rng):
    """img uint8 HxWx3, heat float HxW, mask uint8 HxW. Geometric: all three. Photometric: img only."""
    if rng.rand() < 0.5: img, heat, mask = img[:, ::-1], heat[:, ::-1], mask[:, ::-1]
    if rng.rand() < 0.5: img, heat, mask = img[::-1], heat[::-1], mask[::-1]
    if rng.rand() < 0.5:
        k = rng.randint(1, 4); img, heat, mask = np.rot90(img, k), np.rot90(heat, k), np.rot90(mask, k)
    img, heat, mask = [np.ascontiguousarray(a) for a in (img, heat, mask)]
    trip = [(img, cv2.INTER_LINEAR), (heat, cv2.INTER_LINEAR), (mask, cv2.INTER_NEAREST)]
    if rng.rand() < 0.5: img, heat, mask = _affine(trip, rng); trip = [(img, cv2.INTER_LINEAR), (heat, cv2.INTER_LINEAR), (mask, cv2.INTER_NEAREST)]
    if rng.rand() < 0.3: img, heat, mask = _elastic(trip, rng)
    img = img.astype(np.float32)
    if rng.rand() < 0.3:                                   # brightness / contrast, image only
        img = img * (1 + rng.uniform(-0.15, 0.15)) + 255 * rng.uniform(-0.15, 0.15)
    if rng.rand() < 0.2:                                   # Gaussian noise, image only
        img = img + rng.normal(0, np.sqrt(rng.uniform(5, 20)), img.shape)
    return np.clip(img, 0, 255), heat, mask

class SegDataset(Dataset):
    """Returns image (3,H,W), heat (1,H,W), mask (1,H,W), index.
    heat_mode: real | zero | shuffle   (shuffle = heatmap of another image of the same split,
    fixed permutation given by perm_seed, used for both the input and the loss)."""
    def __init__(self, csv_path, heat_dir, train, heat_mode="real", perm_seed=0):
        self.rows = read_csv(csv_path); self.heat_dir = heat_dir; self.train = train
        self.heat_mode = heat_mode
        n = len(self.rows); perm = np.random.RandomState(perm_seed).permutation(n)
        if n > 1:                                           # derangement: nobody keeps its own map
            for i in range(n):
                if perm[i] == i:
                    j = (i + 1) % n; perm[i], perm[j] = perm[j], perm[i]
        self.perm = perm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["image"])
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        m = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(r["mask"])
        m = (cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        if self.heat_mode == "zero":
            h = np.zeros((SIZE, SIZE), np.float32)
        else:
            stem = self.rows[self.perm[i]]["stem"] if self.heat_mode == "shuffle" else r["stem"]
            h = load_heat(self.heat_dir, stem)
        if self.train:
            img, h, m = augment(img, h, m, np.random)
        img = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        return (img, torch.from_numpy(np.asarray(h, np.float32))[None],
                torch.from_numpy(np.asarray(m, np.float32))[None], i)