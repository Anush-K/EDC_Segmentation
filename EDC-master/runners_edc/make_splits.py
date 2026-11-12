"""STEP 2 - Build leakage-free split lists for BOTH stages.

Stage 2 (segmentation): abnormal images with masks, split 70/10/20 into train/val/test.
  * LGG   : split by PATIENT (TCGA_xx_xxxx); normal slices of a patient follow the patient.
  * BUSI  : split by image, STRATIFIED by class (benign / malignant read from the file name).
  * others: split by image.
Stage 1 (anomaly detection): trains on normal images only.
  * Normal images: either your own fixed split (--normal_train/--normal_val/--normal_test,
    e.g. BUSI 85 / 21 / 27) or a 70/10/20 split of --normal_dirs.
  * Abnormal images in Stage 1 val = Stage 2 val images (needed to choose the threshold).
  * Abnormal images in Stage 1 test = --stage1_test_abnormal all (every abnormal image, as in
    your BUSI table: 27 + 647 = 674) or test (only the Stage 2 test images).

Example BUSI (your split: 85 / 21 / 27 normal, 647 abnormal):
  python make_splits.py --dataset busi --seg_root ../BUSI_SEG \
     --normal_train ../EDC-master/datasets/BUSI/train/NORMAL \
     --normal_val   ../EDC-master/datasets/BUSI/val/NORMAL \
     --normal_test  ../EDC-master/datasets/BUSI/test/NORMAL \
     --stage1_test_abnormal all
Example LGG:
  python make_splits.py --dataset lgg --seg_root ../LGG_SEG \
     --normal_dirs ../EDC-master/datasets/LGG/train/NORMAL ../EDC-master/datasets/LGG/test/NORMAL \
     --materialize ../EDC-master/datasets/LGG_v2
<seg_root> must contain images/ and masks/ (and heatmaps/ for Stage 2).
Outputs in splits/<ds>/: stage2_{train,val,test}.csv, stage1_{train,val,test}.csv, counts.txt,
and report.txt (images with more than one mask file, exact duplicate images).
"""
import os, re, argparse, glob, csv, hashlib
from collections import defaultdict
import numpy as np
import cv2
from common import lgg_patient

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_RE = re.compile(r"_mask(_\d+)?$")

def list_images(d):
    fs = []
    for f in glob.glob(os.path.join(d, "*")):
        st = os.path.splitext(os.path.basename(f))[0]
        if f.lower().endswith(IMG_EXT) and not os.path.basename(f).startswith(".") and not MASK_RE.search(st):
            fs.append(os.path.abspath(f))
    return sorted(fs)

def find_masks(mask_dir, img):
    stem, ext = os.path.splitext(os.path.basename(img))
    first = None
    for c in [os.path.basename(img), stem + "_mask.png", stem + ".png", stem + ".tif", stem + "_mask" + ext]:
        p = os.path.join(mask_dir, c)
        if os.path.exists(p):
            first = p; break
    extra = sorted(glob.glob(os.path.join(mask_dir, glob.escape(stem) + "_mask_[0-9]*.png")))
    return first, extra

def busi_class(stem):
    s = stem.lower()
    return "benign" if s.startswith("benign") else ("malignant" if s.startswith("malignant") else "other")

def split_groups(groups, seed, frac=(0.7, 0.1, 0.2)):
    g = sorted(set(groups)); rng = np.random.RandomState(seed); rng.shuffle(g)
    n = len(g); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
    return set(g[:a]), set(g[a:b]), set(g[b:])

def write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

def pix_hash(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None if im is None else hashlib.md5(im.tobytes() + str(im.shape).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["busi", "lgg", "covid19", "kvasir"])
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--normal_dirs", nargs="*", default=[], help="normal pool to split 70/10/20")
    ap.add_argument("--normal_train"); ap.add_argument("--normal_val"); ap.add_argument("--normal_test")
    ap.add_argument("--stage1_test_abnormal", choices=["all", "test"], default="test")
    ap.add_argument("--merge_masks", action="store_true",
                    help="OR-merge extra mask files (<stem>_mask_1.png ...) into masks_merged/ and use them")
    ap.add_argument("--out", default="splits"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--materialize", default=None)
    a = ap.parse_args()
    fixed = a.normal_train is not None
    if fixed and not (a.normal_val and a.normal_test):
        raise SystemExit("give --normal_train, --normal_val and --normal_test together")
    if not fixed and not a.normal_dirs:
        raise SystemExit("give --normal_dirs or the three --normal_* folders")
    if a.dataset == "lgg" and fixed:
        raise SystemExit("LGG needs the patient-level split: use --normal_dirs")

    pid = lgg_patient if a.dataset == "lgg" else (lambda p: os.path.basename(p))
    od = os.path.join(a.out, a.dataset); os.makedirs(od, exist_ok=True)
    rep = []
    mdir = os.path.join(a.seg_root, "masks"); merged_dir = os.path.join(a.seg_root, "masks_merged")
    rows_ab, multi = [], []
    for p in list_images(os.path.join(a.seg_root, "images")):
        m, extra = find_masks(mdir, p)
        if m is None:
            print("WARNING no mask for", p); continue
        if extra:
            multi.append(os.path.basename(p))
            if a.merge_masks:
                mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 0
                for e in extra:
                    mk |= cv2.imread(e, cv2.IMREAD_GRAYSCALE) > 0
                os.makedirs(merged_dir, exist_ok=True)
                m = os.path.join(merged_dir, os.path.splitext(os.path.basename(p))[0] + "_mask.png")
                cv2.imwrite(m, (mk * 255).astype(np.uint8))
        st = os.path.splitext(os.path.basename(p))[0]
        rows_ab.append((p, os.path.abspath(m), st, pid(p)))
    rep.append("images with more than one mask file: %d (%s)" % (
        len(multi), "merged with OR" if a.merge_masks else "ONLY the first mask is used; add --merge_masks to merge"))
    rep += ["   " + x for x in multi]

    if fixed:
        s1n = {"train": list_images(a.normal_train), "val": list_images(a.normal_val), "test": list_images(a.normal_test)}
    else:
        normals = sorted(set(sum([list_images(d) for d in a.normal_dirs], [])))

    if a.dataset == "lgg":
        tr, va, te = split_groups([r[3] for r in rows_ab] + [pid(p) for p in normals], a.seed)
        side = lambda pt: "train" if pt in tr else ("val" if pt in va else "test")
        s2 = {k: [r for r in rows_ab if side(r[3]) == k] for k in ("train", "val", "test")}
        s1n = {k: [p for p in normals if side(pid(p)) == k] for k in ("train", "val", "test")}
    else:
        if a.dataset == "busi":
            tr, va, te = set(), set(), set()
            for c in ("benign", "malignant", "other"):
                g = [r[3] for r in rows_ab if busi_class(r[2]) == c]
                if g:
                    x, y, z = split_groups(g, a.seed); tr |= x; va |= y; te |= z
        else:
            tr, va, te = split_groups([r[3] for r in rows_ab], a.seed)
        s2 = {"train": [r for r in rows_ab if r[3] in tr], "val": [r for r in rows_ab if r[3] in va],
              "test": [r for r in rows_ab if r[3] in te]}
        if not fixed:
            x, y, z = split_groups(normals, a.seed + 1)
            s1n = {"train": [p for p in normals if p in x], "val": [p for p in normals if p in y],
                   "test": [p for p in normals if p in z]}

    s1a = {"train": [], "val": s2["val"], "test": rows_ab if a.stage1_test_abnormal == "all" else s2["test"]}
    for k in ("train", "val", "test"):
        write(os.path.join(od, "stage2_%s.csv" % k), s2[k], ["image", "mask", "stem", "patient"])
        write(os.path.join(od, "stage1_%s.csv" % k), [(p, 0, pid(p)) for p in s1n[k]] + [(r[0], 1, r[3]) for r in s1a[k]],
              ["image", "label", "patient"])

    # ---- leakage checks ----
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(r[0] for r in s2[x]) & set(r[0] for r in s2[y]), "Stage 2 overlap"
        assert not set(s1n[x]) & set(s1n[y]), "Stage 1 normal overlap"
    if a.dataset == "lgg":
        P = {k: set(pid(p) for p in s1n[k]) | set(r[3] for r in s2[k]) for k in s2}
        assert not (P["train"] & P["test"]) and not (P["train"] & P["val"]) and not (P["val"] & P["test"])
    # exact duplicate images (same pixels) in different splits -> report
    where = defaultdict(set)
    for k in ("train", "val", "test"):
        for r in s2[k]: where[pix_hash(r[0])].add("S2-" + k + ":" + os.path.basename(r[0]))
        for p in s1n[k]: where[pix_hash(p)].add("N-" + k + ":" + os.path.basename(p))
    dups = [sorted(v) for h, v in where.items() if h and len(v) > 1]
    cross = [d for d in dups if len(set(x.split(":")[0] for x in d)) > 1]
    rep.append("exact duplicate groups: %d, of which in DIFFERENT splits/classes: %d" % (len(dups), len(cross)))
    rep += ["   " + ", ".join(d) for d in cross]
    if cross:
        rep.append("   -> move or remove these images and run again (they leak between splits)")

    txt = ["dataset %s (Stage 2 split unit: %s, seed %d)" % (a.dataset, "patient" if a.dataset == "lgg" else "image", a.seed),
           "Stage 1  normal train/val/test: %d / %d / %d   abnormal val/test: %d / %d   (test total %d)" % (
               len(s1n["train"]), len(s1n["val"]), len(s1n["test"]), len(s1a["val"]), len(s1a["test"]),
               len(s1n["test"]) + len(s1a["test"])),
           "Stage 2  train/val/test: %d / %d / %d" % tuple(len(s2[k]) for k in ("train", "val", "test"))]
    if a.dataset == "busi":
        for c in ("benign", "malignant"):
            txt.append("Stage 2  %-9s %d / %d / %d" % ((c,) + tuple(sum(busi_class(r[2]) == c for r in s2[k]) for k in ("train", "val", "test"))))
    if a.dataset == "lgg":
        txt.append("patients: %d / %d / %d" % (len(tr), len(va), len(te)))
    if a.stage1_test_abnormal == "all":
        txt.append("NOTE: the %d Stage 2 val images are in the Stage 1 val AND test sets (threshold chosen with them)." % len(s2["val"]))
    open(os.path.join(od, "counts.txt"), "w").write("\n".join(txt) + "\n")
    open(os.path.join(od, "report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(txt)); print("\n".join(rep[:1] + [r for r in rep if r.startswith("exact")]))

    if a.materialize:
        def link(items, d):
            os.makedirs(d, exist_ok=True)
            for p in items:
                t = os.path.join(d, os.path.basename(p))
                if not os.path.exists(t): os.symlink(p, t)
        for k in ("train", "val", "test"):
            link(s1n[k], os.path.join(a.materialize, k, "NORMAL"))
            if k != "train": link([r[0] for r in s1a[k]], os.path.join(a.materialize, k, "ABNORMAL"))
        link([r[0] for r in rows_ab], os.path.join(a.materialize, "heatmap_export", "ABNORMAL"))
        print("Stage 1 folders written to", a.materialize)

if __name__ == "__main__":
    main()