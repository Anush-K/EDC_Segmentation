"""
Improve the Stage 1 heatmap with LABEL-FREE post-processing, chosen on the VALIDATION split.

Options tested by --scan (same fixed step for every image):
  blur  : Gaussian smoothing (sigma in pixels at 256x256) - removes speckle-like noise
  mask  : keep only the bright tissue / organ region of the image (largest bright component,
          holes filled, eroded by `erode` px) - removes heat on background, borders, text, rim
Quality on the validation tumour/lesion images: pixel AUROC of the heatmap against the mask, and
pointing game (hottest pixel inside the mask). The test split is NOT used for the choice.

Usage:
  python heatmap_postprocess.py --dataset busi --seg_root /home/cs24d0008/EDC_Segmentation/BUSI_SEG_HGBL --scan
  python heatmap_postprocess.py --dataset busi --seg_root ... --blur 4 --mask 1 --erode 5 --write
      -> writes <seg_root>/heatmaps_pp/  (use it with  train_seg.py --heat_dir <seg_root>/heatmaps_pp)
"""
import os, csv, argparse, itertools
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

SIZE = 256


def tissue_mask(img, thr=20, erode=5):
    g = img.max(axis=2) if img.ndim == 3 else img
    m = (g > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones(g.shape, np.float32)
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m = m | (1 - ff)
    if erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    return m.astype(np.float32)


def process(img, heat, blur, mask, erode):
    h = heat.copy()
    if blur > 0:
        h = cv2.GaussianBlur(h, (0, 0), blur)
    if mask:
        h = h * tissue_mask(img, erode=erode)
    lo, hi = h.min(), h.max()
    return (h - lo) / (hi - lo) if hi > lo else h


def load(row, heat_dir):
    img = cv2.resize(cv2.imread(row["image"]), (SIZE, SIZE))
    p = os.path.join(heat_dir, row["stem"] + ".png")
    h = cv2.resize(cv2.imread(p, 0).astype(np.float32) / 255.0, (SIZE, SIZE))
    m = cv2.resize(cv2.imread(row["mask"], 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
    return img, h, m


def quality(data, blur, mask, erode):
    aucs, hits = [], []
    for img, h, m in data:
        q = process(img, h, blur, mask, erode)
        if m.any() and (~m).any():
            aucs.append(roc_auc_score(m.ravel(), q.ravel()))
        if m.any():
            hits.append(bool(m.ravel()[q.argmax()]))
    return float(np.mean(aucs)), float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--blur", type=float, default=0)
    ap.add_argument("--mask", type=int, default=0)
    ap.add_argument("--erode", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    heat_dir = os.path.join(a.seg_root, "heatmaps")
    rd = lambda p: list(csv.DictReader(open(os.path.join(a.seg_root, "splits", a.dataset, "stage2_%s.csv" % p))))
    if a.scan:
        val = [load(r, heat_dir) for r in rd("val")]
        print("VALIDATION split (%d images)  -  choose the best row here" % len(val))
        print("  %-28s %s" % ("setting", "pixel AUROC   pointing game"))
        for blur, (mask, erode) in itertools.product((0, 2, 4, 8), ((0, 0), (1, 0), (1, 5), (1, 15))):
            auc, pg = quality(val, blur, mask, erode)
            name = "blur %g, %s" % (blur, "no mask" if not mask else "mask erode %d" % erode)
            print("  %-28s %.3f         %.1f%%" % (name, auc, 100 * pg))
    if a.write:
        out = os.path.join(a.seg_root, "heatmaps_pp"); os.makedirs(out, exist_ok=True); n = 0
        for p in ("train", "val", "test"):
            for r in rd(p):
                img, h, _ = load(r, heat_dir)
                cv2.imwrite(os.path.join(out, r["stem"] + ".png"), (255 * process(img, h, a.blur, a.mask, a.erode)).astype(np.uint8)); n += 1
        print("wrote %d heatmaps (blur %g, mask %d, erode %d) to %s" % (n, a.blur, a.mask, a.erode, out))


if __name__ == "__main__":
    main()