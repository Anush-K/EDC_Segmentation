import os, glob, argparse
import numpy as np
import cv2

def brain_mask(img, thr=20, erode=10):
    g = img.max(axis=2)
    m = (g > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones(g.shape, np.uint8)
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m = m | (1 - ff)
    if erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    return m

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--erode", type=int, default=10)
a = ap.parse_args()
for name in ("LGG_HGBL", "LGG_HGBL_VAL", "LGG_HGBL_EXPORT"):
    src = os.path.join(a.data_root, name); dst = src + "_SS"; n = 0
    for sub in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src, sub, "*"))):
            img = cv2.imread(f)
            if img is None:
                continue
            cv2.imwrite(os.path.join(dst, sub, os.path.basename(f)), img * brain_mask(img, erode=a.erode)[..., None]); n += 1
    print("%s -> %s : %d images" % (name, dst, n))
