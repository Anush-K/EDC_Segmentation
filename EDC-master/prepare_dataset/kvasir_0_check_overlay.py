"""
KVASIR - STEP 0: find the on-screen overlay (shortcut) before preparing the data.

Three checks, for normal vs polyp images:
  1. GREEN BOX: share of images with the green position box in the lower-left corner,
     and its region (relative coordinates 0-1).
  2. BORDER-ONLY CLASSIFIER: a logistic regression that sees ONLY the outer border of each
     image (the centre is hidden). AUC near 0.5 = no shortcut; AUC well above 0.5 = the classes
     can be told apart from the border alone (overlay / frame differences).
  3. DIFFERENCE MAP: where the average normal and average polyp image differ most
     (overlay_difference.png); bright areas = where a shortcut sits.
Run it once as it is (find the problem) and once WITH --crop_black (Kvasir v2 frames cropped like
Kvasir-SEG) to check that the border AUC drops.

Usage:
  python kvasir_0_check_overlay.py --normal_dirs <normal-cecum> <normal-pylorus> <normal-z-line> \
         --polyp_dir <Kvasir-SEG/images> [--blackout X0 Y0 X1 Y1]
"""
import os, glob, argparse
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

G = 64            # analysis grid
BORDER = 0.15     # border width (share of the image) seen by the border-only classifier


def list_imgs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def crop_black(im, thr=25, frac=0.5):
    """Remove the dark frame around the endoscopic view (Kvasir v2 frames are 720x576 with a
    black frame and a black side panel; Kvasir-SEG images were already cropped like this).
    Keeps the rows/columns in which more than `frac` of the pixels are brighter than `thr`."""
    g = im if im.ndim == 2 else im.max(axis=2)
    rows = np.where((g > thr).mean(1) > frac)[0]; cols = np.where((g > thr).mean(0) > frac)[0]
    if len(rows) < 10 or len(cols) < 10:
        return im
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def read(f, blackout, crop=False):
    im = cv2.imread(f)
    if im is None:
        return None
    if crop:
        im = crop_black(im)
    im = cv2.resize(im, (G, G), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if blackout:
        x0, y0, x1, y1 = [int(round(v * G)) for v in blackout]
        im[y0:y1, x0:x1] = 0
    return im


def green_frac(im):
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    return ((g > 0.45) & (g - r > 0.2) & (g - b > 0.2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dirs", nargs="+", required=True)
    ap.add_argument("--polyp_dir", required=True)
    ap.add_argument("--blackout", type=float, nargs=4, default=None)
    ap.add_argument("--crop_black", action="store_true", help="remove the dark frame first (as Kvasir-SEG)")
    ap.add_argument("--out", default="overlay_difference.png")
    a = ap.parse_args()
    nf = sum([list_imgs(d) for d in a.normal_dirs], []); pf = list_imgs(a.polyp_dir)
    X, y, greens = [], [], {0: [], 1: []}
    for lab, files in ((0, nf), (1, pf)):
        for f in files:
            im = read(f, a.blackout, a.crop_black)
            if im is None:
                continue
            X.append(im); y.append(lab); greens[lab].append(green_frac(im))
    X = np.stack(X); y = np.array(y)
    print("images: %d normal, %d polyp%s" % ((y == 0).sum(), (y == 1).sum(),
          ("   (blackout %s applied)" % a.blackout if a.blackout else "") + ("   (dark frame cropped)" if a.crop_black else "")))

    # 1. green box (lower-left quarter)
    for lab, name in ((0, "normal"), (1, "polyp ")):
        gm = np.stack(greens[lab]); ll = gm[:, G // 2:, :G // 2].sum((1, 2))
        print("%s with green box in lower-left: %.1f%%" % (name, 100 * (ll >= 3).mean()))
    gsum = np.stack(greens[0] + greens[1]).sum(0)
    if gsum[G // 2:, :G // 2].max() > 0:
        occ = np.zeros_like(gsum, bool); q = gsum[G // 2:, :G // 2]
        occ[G // 2:, :G // 2] = q > 0.1 * q.max()
        ys, xs = np.where(occ)
        print("green box region (x0 y0 x1 y1): %.3f %.3f %.3f %.3f" % (
            xs.min() / G, ys.min() / G, (xs.max() + 1) / G, (ys.max() + 1) / G))

    # 2. border-only classifier
    m = np.zeros((G, G), bool); b = int(round(BORDER * G))
    m[:b] = m[-b:] = True; m[:, :b] = m[:, -b:] = True
    feats = X[:, m, :].reshape(len(X), -1)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), feats, y, cv=cv, method="predict_proba")[:, 1]
    print("BORDER-ONLY classifier AUC (5-fold): %.3f   (0.5 = no shortcut)" % roc_auc_score(y, p))
    # per-region AUC: which border part carries the shortcut
    regs = {"top": (slice(0, b), slice(0, G)), "bottom": (slice(G - b, G), slice(0, G)),
            "left": (slice(0, G), slice(0, b)), "right": (slice(0, G), slice(G - b, G)),
            "lower-left corner": (slice(G // 2, G), slice(0, G // 2))}
    for k, (sy, sx) in regs.items():
        f2 = X[:, sy, sx, :].reshape(len(X), -1)
        p2 = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), f2, y, cv=cv, method="predict_proba")[:, 1]
        print("   region %-18s AUC %.3f" % (k, roc_auc_score(y, p2)))

    # 3. difference map
    d = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0)).mean(-1)
    vis = (255 * d / (d.max() + 1e-8)).astype(np.uint8)
    vis = cv2.applyColorMap(cv2.resize(vis, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(a.out, vis)
    print("difference map (bright = classes differ) written to", a.out)


if __name__ == "__main__":
    main()