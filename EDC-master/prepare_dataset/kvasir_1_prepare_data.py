import os, csv, glob, shutil, argparse
import numpy as np
import cv2

SIZE = 256


def list_imgs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def split3(items, rng, frac=(0.7, 0.1)):
    items = sorted(items); rng.shuffle(items)
    n = len(items); a = int(round(frac[0] * n)); b = int(round(frac[1] * n))
    return items[:a], items[a:a + b], items[a + b:]


def crop_black(im, thr=25, frac=0.5):
    g = im if im.ndim == 2 else im.max(axis=2)
    rows = np.where((g > thr).mean(1) > frac)[0]; cols = np.where((g > thr).mean(0) > frac)[0]
    if len(rows) < 10 or len(cols) < 10:
        return im
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def load(path, is_mask=False, crop=False):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read " + path)
    if crop:
        im = crop_black(im)
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    if is_mask:
        im = ((im > 127) * 255).astype(np.uint8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dirs", nargs="+", required=True)
    ap.add_argument("--polyp_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--crop_black", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    normals = []
    for d in a.normal_dirs:
        cls = os.path.basename(os.path.normpath(d))
        normals += [(cls + "__" + os.path.splitext(os.path.basename(f))[0], f) for f in list_imgs(d)]
    polyps = []
    for f in list_imgs(os.path.join(a.polyp_dir, "images")):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = glob.glob(os.path.join(a.polyp_dir, "masks", stem + ".*"))
        if not m:
            print("WARNING no mask:", f); continue
        polyps.append((stem, f, m[0]))
    print("found: %d normal, %d polyp images" % (len(normals), len(polyps)), flush=True)
    ntr, nva, nte = split3([n[0] for n in normals], rng)
    ptr, pva, pte = split3([p[0] for p in polyps], rng)
    nsplit = {s: "train" for s in ntr}; nsplit.update({s: "val" for s in nva}); nsplit.update({s: "test" for s in nte})
    psplit = {s: "train" for s in ptr}; psplit.update({s: "val" for s in pva}); psplit.update({s: "test" for s in pte})
    D = os.path.join(a.data_root, "KVASIR_HGBL"); V = os.path.join(a.data_root, "KVASIR_HGBL_VAL")
    X = os.path.join(a.data_root, "KVASIR_HGBL_EXPORT")
    for root in (D, V, X):
        if os.path.exists(root):
            shutil.rmtree(root)
        for sub in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL"):
            os.makedirs(os.path.join(root, sub))
    for sub in ("images", "masks", "splits/kvasir"):
        os.makedirs(os.path.join(a.seg_root, sub), exist_ok=True)
    for stem, f in normals:
        im = load(f, crop=a.crop_black); k = nsplit[stem]; name = stem + ".png"
        if k == "train":
            cv2.imwrite(os.path.join(D, "train/NORMAL", name), im); cv2.imwrite(os.path.join(V, "train/NORMAL", name), im)
        elif k == "val":
            cv2.imwrite(os.path.join(V, "test/NORMAL", name), im)
        else:
            cv2.imwrite(os.path.join(D, "test/NORMAL", name), im)
    rows = {"train": [], "val": [], "test": []}
    for stem, f, mf in polyps:
        im = load(f); mk = load(mf, True); k = psplit[stem]; name = stem + ".png"
        si, sm = os.path.join(a.seg_root, "images", name), os.path.join(a.seg_root, "masks", name)
        cv2.imwrite(si, im); cv2.imwrite(sm, mk); cv2.imwrite(os.path.join(X, "test/ABNORMAL", name), im)
        if k == "val":
            cv2.imwrite(os.path.join(V, "test/ABNORMAL", name), im)
        elif k == "test":
            cv2.imwrite(os.path.join(D, "test/ABNORMAL", name), im)
        rows[k].append((os.path.abspath(si), os.path.abspath(sm), stem, stem))
    for k in rows:
        with open(os.path.join(a.seg_root, "splits", "kvasir", "stage2_%s.csv" % k), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "mask", "stem", "patient"]); w.writerows(sorted(rows[k]))
    c = lambda d: len(os.listdir(d))
    rep = ["KVASIR (image-level split, seed %d, dark frame cropped: %s)" % (a.seed, a.crop_black),
           "Stage 1 train : %d normal" % c(os.path.join(D, "train/NORMAL")),
           "Stage 1 VAL   : %d normal + %d polyp" % (c(os.path.join(V, "test/NORMAL")), c(os.path.join(V, "test/ABNORMAL"))),
           "Stage 1 TEST  : %d normal + %d polyp" % (c(os.path.join(D, "test/NORMAL")), c(os.path.join(D, "test/ABNORMAL"))),
           "Stage 2 train/val/test: %d / %d / %d" % tuple(len(rows[k]) for k in ("train", "val", "test"))]
    open(os.path.join(a.seg_root, "kvasir_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
