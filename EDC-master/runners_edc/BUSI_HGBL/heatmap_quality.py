"""STEP 5 - How well does the Stage 1 heatmap mark the target? (Table: heatmap quality)
Pixel AUROC of S against the mask, and pointing game (max of S inside the mask).
  python heatmap_quality.py --dataset lgg --splits splits --heat_dir ../LGG_SEG/heatmaps_npy
"""
import os, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from seg_data import SegDataset
from common import save_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--splits", default="splits")
ap.add_argument("--heat_dir", required=True); ap.add_argument("--part", default="test")
ap.add_argument("--out", default="results")
a = ap.parse_args()
ds = SegDataset(os.path.join(a.splits, a.dataset, "stage2_%s.csv" % a.part), a.heat_dir, train=False)
aucs, hits = [], []
for i in range(len(ds)):
    _, h, m, _ = ds[i]; h = h[0].numpy(); m = m[0].numpy() > 0.5
    if m.any() and (~m).any():
        aucs.append(roc_auc_score(m.ravel(), h.ravel()))
    if m.any():
        hits.append(bool(m.ravel()[h.argmax()]))
res = dict(dataset=a.dataset, part=a.part, n=len(ds), pixel_auroc=float(np.mean(aucs)),
           pixel_auroc_sd=float(np.std(aucs)), pointing_game=float(np.mean(hits)))
print(res); save_json(res, os.path.join(a.out, a.dataset, "heatmap_quality_%s.json" % a.part))