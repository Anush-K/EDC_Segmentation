"""
KVASIR - STEP 1: build every Kvasir folder and list used by the HGBL paper.

Input
  normal : Kvasir v2 classes normal-cecum, normal-pylorus, normal-z-line  (3 x 1,000 images)
  polyp  : Kvasir-SEG images/ + masks/                                      (1,000 images)
Split: image level (Kvasir has no patient ids), 70 / 10 / 20 %, seed 42, separately for the
normal and the polyp pool:
    Stage 1 train : normal train
    Stage 1 val   : normal val  + polyp val     (model selection + threshold)
    Stage 1 test  : normal test + polyp test
    Stage 2       : polyp train / val / test
Overlay: --blackout x0 y0 x1 y1 (relative coordinates, from kvasir_0_check_overlay.py) paints
that region black in EVERY image (normal and polyp, all splits), so the green position box can
not be used as a shortcut. Mask pixels inside the region are set to 0 (the polyp part there is
not visible any more); the report gives how many masks were touched, and images that lose
more than --max_removed (default 50%) of their polyp are left out of Stage 2.
All images and masks are resized to 256 x 256 (INTER_AREA / INTER_NEAREST) and saved as PNG.

Output
  <data_root>/KVASIR_HGBL/train/NORMAL, test/NORMAL, test/ABNORMAL     data_dir of edc_kvasir_hgbl.py
  <data_root>/KVASIR_HGBL_VAL/train/NORMAL, test/NORMAL, test/ABNORMAL val_dir
  <data_root>/KVASIR_HGBL_EXPORT/test/ABNORMAL = all polyp images      export_dir (Stage 2 heatmaps)
  <seg_root>/images/<stem>.png, <seg_root>/masks/<stem>.png
  <seg_root>/splits/kvasir/stage2_{train,val,test}.csv                 columns: image,mask,stem,patient
  <seg_root>/kvasir_report.txt
"""
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


def load(path, blackout, is_mask=False):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read " + path)
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    if is_mask:
        im = ((im > 127) * 255).astype(np.uint8)
    if blackout:
        x0, y0, x1, y1 = [int(round(v * SIZE)) for v in blackout]
        im[y0:y1, x0:x1] = 0
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dirs", nargs="+", required=True)
    ap.add_argument("--polyp_dir", required=True, help="Kvasir-SEG folder with images/ and masks/")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--blackout", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--max_removed", type=float, default=0.5,
                    help="polyp images whose mask loses more than this share under the blackout are left out of Stage 2")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    normals = []
    for d in a.normal_dirs:      # prefix with the class name: file names repeat across classes
        cls = os.path.basename(os.path.normpath(d))
        normals += [(cls + "__" + os.path.splitext(os.path.basename(f))[0], f) for f in list_imgs(d)]
    polyps = []
    for f in list_imgs(os.path.join(a.polyp_dir, "images")):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = [p for p in glob.glob(os.path.join(a.polyp_dir, "masks", stem + ".*"))]
        if not m:
            print("WARNING no mask:", f); continue
        polyps.append((stem, f, m[0]))
    print("found: %d normal, %d polyp images" % (len(normals), len(polyps)))

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
        im = load(f, a.blackout); k = nsplit[stem]; name = stem + ".png"
        if k == "train":
            cv2.imwrite(os.path.join(D, "train/NORMAL", name), im); cv2.imwrite(os.path.join(V, "train/NORMAL", name), im)
        elif k == "val":
            cv2.imwrite(os.path.join(V, "test/NORMAL", name), im)
        else:
            cv2.imwrite(os.path.join(D, "test/NORMAL", name), im)

    rows = {"train": [], "val": [], "test": []}; touched = 0; lost = []; dropped = []
    for stem, f, mf in polyps:
        im = load(f, a.blackout); mk_full = load(mf, None, True); mk = load(mf, a.blackout, True)
        share = 0.0
        if a.blackout and mk_full.sum() > 0 and (mk_full.sum() - mk.sum()) > 0:
            share = 1 - mk.sum() / mk_full.sum(); touched += 1; lost.append(share)
        k = psplit[stem]; name = stem + ".png"
        si, sm = os.path.join(a.seg_root, "images", name), os.path.join(a.seg_root, "masks", name)
        cv2.imwrite(si, im); cv2.imwrite(sm, mk); cv2.imwrite(os.path.join(X, "test/ABNORMAL", name), im)
        if k == "val":
            cv2.imwrite(os.path.join(V, "test/ABNORMAL", name), im)
        elif k == "test":
            cv2.imwrite(os.path.join(D, "test/ABNORMAL", name), im)
        if share > a.max_removed:
            dropped.append(stem); continue          # stays in Stage 1, not used in Stage 2
        rows[k].append((os.path.abspath(si), os.path.abspath(sm), stem, stem))
    sp = os.path.join(a.seg_root, "splits", "kvasir")
    for k in rows:
        with open(os.path.join(sp, "stage2_%s.csv" % k), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "mask", "stem", "patient"]); w.writerows(sorted(rows[k]))

    c = lambda d: len(os.listdir(d))
    rep = ["KVASIR (image-level split, seed %d)" % a.seed,
           "found: %d normal, %d polyp" % (len(normals), len(polyps)),
           "Stage 1 train : %d normal" % c(os.path.join(D, "train/NORMAL")),
           "Stage 1 VAL   : %d normal + %d polyp" % (c(os.path.join(V, "test/NORMAL")), c(os.path.join(V, "test/ABNORMAL"))),
           "Stage 1 TEST  : %d normal + %d polyp" % (c(os.path.join(D, "test/NORMAL")), c(os.path.join(D, "test/ABNORMAL"))),
           "Stage 2 train/val/test: %d / %d / %d" % tuple(len(rows[k]) for k in ("train", "val", "test")),
           "blackout region: %s" % (a.blackout if a.blackout else "none (overlay kept)")]
    if a.blackout:
        rep.append("masks touched by the blackout: %d of %d (mean share of the polyp removed: %.1f%%)" % (
            touched, len(polyps), 100 * float(np.mean(lost)) if lost else 0.0))
        rep.append("left out of Stage 2 (more than %d%% of the polyp hidden): %d %s" % (
            100 * a.max_removed, len(dropped), ", ".join(dropped[:10])))
    open(os.path.join(a.seg_root, "kvasir_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()