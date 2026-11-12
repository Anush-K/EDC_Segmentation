"""STEP 1 - Check the Stage 1 heatmaps that Stage 2 reads, and convert them to
scalar .npy files.

Why: the current dataset code reads <stem>.png with cv2.imread(path, 0).
If the PNG is a JET-coloured map, the grayscale value is NOT the anomaly score:
in JET both the lowest score (dark blue) and the highest score (dark red) become
dark grey, and mid scores (cyan/yellow) become bright. The network would then get
a wrong heatmap. This script detects that case and inverts the JET colour map back
to a score in [0,1]. If the PNG is an overlay on the image (heatmap blended with
the scan), inversion is NOT reliable: then re-export raw maps from Stage 1.

Usage:
  python check_heatmaps.py --seg_root ../LGG_SEG            # report only
  python check_heatmaps.py --seg_root ../LGG_SEG --convert  # writes heatmaps_npy/
"""
import os, argparse, glob
import numpy as np
import cv2

def jet_lut():
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    lut = cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 3).astype(np.float32)  # BGR
    return lut

LUT = jet_lut()

def invert_jet(bgr):
    """Nearest JET colour -> index/255. Returns (score map, mean colour error)."""
    px = bgr.reshape(-1, 3).astype(np.float32)
    out = np.empty(len(px), np.float32); err = np.empty(len(px), np.float32)
    for i in range(0, len(px), 65536):           # chunked to keep memory small
        d = ((px[i:i + 65536, None, :] - LUT[None]) ** 2).sum(-1)
        k = d.argmin(1)
        out[i:i + 65536] = k / 255.0
        err[i:i + 65536] = np.sqrt(d[np.arange(len(k)), k])
    return out.reshape(bgr.shape[:2]), float(err.mean())

def classify(img):
    if img.ndim == 2 or img.shape[2] == 1:
        return "gray"
    b, g, r = [img[..., c].astype(np.int32) for c in range(3)]
    if np.abs(b - g).max() <= 2 and np.abs(g - r).max() <= 2:
        return "gray"
    return "color"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", required=True, help="folder with images/ heatmaps/ masks/")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--max_err", type=float, default=12.0,
                    help="mean RGB distance to the JET curve above which the PNG is treated as an overlay")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.seg_root, "heatmaps", "*.png")))
    if not files:
        raise SystemExit("no heatmaps/*.png under " + a.seg_root)
    kinds, errs = {}, []
    for f in files[:50]:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        k = classify(im); kinds[k] = kinds.get(k, 0) + 1
        if k == "color":
            errs.append(invert_jet(im[..., :3])[1])
    print("first 50 heatmaps:", kinds)
    mode = "gray"
    if kinds.get("color", 0):
        e = float(np.mean(errs))
        print("mean distance to the JET colour curve: %.1f (0 = pure JET map)" % e)
        if e > a.max_err:
            print("RESULT: colour heatmaps that are NOT pure JET (probably overlaid on the image).\n"
                  "        Inversion is unreliable. Re-export the raw p_all maps from Stage 1 as .npy.")
            return
        mode = "jet"
        print("RESULT: JET-coloured heatmaps. Your current code (imread(...,0)) used WRONG values.\n"
              "        Run with --convert and use --heat_dir heatmaps_npy in train_seg.py.")
    else:
        print("RESULT: grayscale heatmaps. Values are usable; --convert just copies them to .npy.")
    if a.convert:
        out = os.path.join(a.seg_root, "heatmaps_npy"); os.makedirs(out, exist_ok=True)
        for f in files:
            im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if mode == "jet" and im.ndim == 3:
                s, _ = invert_jet(im[..., :3])
            else:
                s = (im if im.ndim == 2 else im[..., 0]).astype(np.float32) / 255.0
            np.save(os.path.join(out, os.path.splitext(os.path.basename(f))[0] + ".npy"), s.astype(np.float32))
        print("wrote %d files to %s" % (len(files), out))

if __name__ == "__main__":
    main()