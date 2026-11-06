"""
BUSI - STEP 1: build every BUSI folder and list used by the HGBL paper.

Input: the ORIGINAL BUSI dataset (Dataset_BUSI_with_GT)
    <busi_root>/normal/     normal (1).png, normal (1)_mask.png, ...          133 images
    <busi_root>/benign/     benign (1).png, benign (1)_mask.png, [_mask_1.png] 437 images
    <busi_root>/malignant/  malignant (1).png, malignant (1)_mask.png, ...    210 images

Normal images (Stage 1):  85 train / 21 val / 27 test
    * --normal_split_from DIR reuses YOUR existing split (matched by file name):
      DIR/train/NORMAL, DIR/val/NORMAL, DIR/test/NORMAL are used as they are; if DIR has no
      val/NORMAL (e.g. old BUSI: train 106 / test 27), the 27 test images are kept and 21
      validation images are taken from the 106 training images (-> 85 / 21 / 27);
    * otherwise a random 85 / 21 / 27 split with --seed.
Abnormal images: benign + malignant = 647
    * Stage 1 test : all 647 abnormal + 27 normal = 674 (your table)
    * Stage 1 val  : 21 normal + the 65 Stage 2 validation images (to choose the threshold)
    * Stage 2      : 647 split 70/10/20, stratified by class
                     -> 453 / 65 / 129 (benign 306/44/87, malignant 147/21/42)
Masks: all mask files of an image (_mask, _mask_1, ...) are merged with a pixel-wise OR
       (use --no_merge_masks to keep only <stem>_mask.png).
All images and masks are resized to 256 x 256 (images INTER_AREA, masks INTER_NEAREST),
as in your previous prepare_busi.py.

Output
  <data_root>/BUSI_HGBL/train/NORMAL        85   } Stage 1 training + TEST evaluation
  <data_root>/BUSI_HGBL/test/NORMAL         27   }   (data_dir for busi_2_stage1_rqasw.py)
  <data_root>/BUSI_HGBL/test/ABNORMAL      647   }
  <data_root>/BUSI_HGBL/masks/  <stem>_mask.png  (for the Stage 1 dataset class)
  <data_root>/BUSI_HGBL_VAL/train/NORMAL    85   } Stage 1 VALIDATION evaluation
  <data_root>/BUSI_HGBL_VAL/test/NORMAL     21   }   (val_dir for busi_2_stage1_rqasw.py)
  <data_root>/BUSI_HGBL_VAL/test/ABNORMAL   65   }
  <data_root>/BUSI_HGBL_VAL/masks/
  <seg_root>/images/ (647)  <seg_root>/masks/ (647, same file name as the image)
  <seg_root>/splits/busi/stage2_{train,val,test}.csv   columns: image,mask,stem,class
  <seg_root>/splits/busi/stage1_normal_{train,val,test}.txt
  <seg_root>/busi_report.txt   counts, merged masks, duplicate images

Usage:
  python busi_1_prepare_data.py --busi_root ../BUSI_Orig \
      --data_root ../EDC-master/datasets --seg_root ../BUSI_SEG_HGBL \
      [--normal_split_from ../EDC-master/datasets/BUSI]
"""
import os, re, csv, glob, shutil, hashlib, argparse
from collections import defaultdict
import numpy as np
import cv2

SIZE = 256
MASK_RE = re.compile(r"^(?P<stem>.+?)_mask(?:_\d+)?$")


def read_class(folder):
    """Return {stem: (image_path, [mask paths])} for one BUSI class folder."""
    imgs, masks = {}, defaultdict(list)
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        m = MASK_RE.match(stem)
        if m:
            masks[m.group("stem")].append(f)
        else:
            imgs[stem] = f
    return {s: (p, sorted(masks.get(s, []))) for s, p in imgs.items()}


def save_img(src, dst):
    im = cv2.imread(src)
    if im is None:
        raise IOError("cannot read " + src)
    cv2.imwrite(dst, cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA))


def merged_mask(paths, merge):
    use = paths if merge else [p for p in paths if p.endswith("_mask.png")][:1] or paths[:1]
    mk = None
    for p in use:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0
        mk = m if mk is None else (mk | m)
    m = (mk * 255).astype(np.uint8)
    return cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def split3(items, n_tr, n_va, rng):
    items = sorted(items); rng.shuffle(items)
    return items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]


def frac_split(items, rng, frac=(0.7, 0.1)):
    n = len(items); a = int(round(frac[0] * n)); b = int(round(frac[1] * n))
    return split3(items, a, b, rng)


def mkdirs(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi_root", required=True)
    ap.add_argument("--data_root", required=True, help="e.g. ../EDC-master/datasets")
    ap.add_argument("--seg_root", required=True, help="e.g. ../BUSI_SEG_HGBL")
    ap.add_argument("--normal_split_from", default=None)
    ap.add_argument("--n_normal", type=int, nargs=3, default=[85, 21, 27])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_merge_masks", action="store_true")
    ap.add_argument("--exclude_stage2", nargs="*", default=[],
                    help="stems removed from Stage 2 only (they stay in the Stage 1 test set), "
                         "e.g. the duplicate pair 'benign (433)' 'malignant (145)'")
    a = ap.parse_args()
    merge = not a.no_merge_masks
    rng = np.random.RandomState(a.seed)

    normal = read_class(os.path.join(a.busi_root, "normal"))
    benign = read_class(os.path.join(a.busi_root, "benign"))
    malig = read_class(os.path.join(a.busi_root, "malignant"))
    print("found: normal %d, benign %d, malignant %d" % (len(normal), len(benign), len(malig)))
    rep = ["found: normal %d, benign %d, malignant %d" % (len(normal), len(benign), len(malig))]

    # ---------- normal split 85 / 21 / 27 ----------
    if a.normal_split_from:
        nsplit = {}
        has_val = os.path.isdir(os.path.join(a.normal_split_from, "val", "NORMAL"))
        for k in (("train", "val", "test") if has_val else ("train", "test")):
            d = os.path.join(a.normal_split_from, k, "NORMAL")
            names = [os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            missing = [n for n in names if n not in normal]
            if missing:
                raise SystemExit("%d files of %s are not in BUSI normal/: %s" % (len(missing), d, missing[:3]))
            nsplit[k] = sorted(names)
        if not has_val:
            # old folder has only train/ and test/: keep its test images, and take the
            # validation images out of its training images
            tr = sorted(nsplit["train"]); rng.shuffle(tr)
            nv = a.n_normal[1]
            nsplit["val"], nsplit["train"] = sorted(tr[:nv]), sorted(tr[nv:])
            rep.append("old split has no val/NORMAL: test kept, %d validation images taken from its train" % nv)
        rep.append("normal split reused from %s" % a.normal_split_from)
    else:
        tr, va, te = split3(list(normal), a.n_normal[0], a.n_normal[1], rng)
        nsplit = {"train": tr, "val": va, "test": te}
        rep.append("normal split: random, seed %d" % a.seed)
    rep.append("normal train/val/test: %d / %d / %d" % tuple(len(nsplit[k]) for k in ("train", "val", "test")))

    # ---------- abnormal: stratified 70/10/20 for Stage 2 ----------
    abn = {}
    for cname, d in (("benign", benign), ("malignant", malig)):
        for s, (p, ms) in d.items():
            if not ms:
                rep.append("WARNING no mask, skipped: %s" % s); continue
            abn[s] = (cname, p, ms)
    s2 = {"train": [], "val": [], "test": []}
    excl = set(a.exclude_stage2)
    bad = excl - set(abn)
    if bad:
        raise SystemExit("--exclude_stage2: not found: %s" % sorted(bad))
    for cname in ("benign", "malignant"):
        stems = [s for s in abn if abn[s][0] == cname and s not in excl]
        tr, va, te = frac_split(stems, rng)
        s2["train"] += tr; s2["val"] += va; s2["test"] += te
        rep.append("Stage 2 %-9s train/val/test: %d / %d / %d" % (cname, len(tr), len(va), len(te)))
    rep.append("Stage 2 total     train/val/test: %d / %d / %d" % tuple(len(s2[k]) for k in ("train", "val", "test")))
    if excl:
        rep.append("excluded from Stage 2 (kept in Stage 1 test): %s" % ", ".join(sorted(excl)))
    multi = [s for s in abn if len(abn[s][2]) > 1]
    rep.append("images with more than one mask file: %d (%s)" % (len(multi), "merged with OR" if merge else "first mask only"))
    rep += ["   " + s for s in sorted(multi)]

    # ---------- write folders ----------
    D = os.path.join(a.data_root, "BUSI_HGBL"); V = os.path.join(a.data_root, "BUSI_HGBL_VAL")
    for root in (D, V):
        if os.path.exists(root):
            shutil.rmtree(root)
    mkdirs(*[os.path.join(r, x) for r in (D, V) for x in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL", "masks")])
    mkdirs(os.path.join(a.seg_root, "images"), os.path.join(a.seg_root, "masks"), os.path.join(a.seg_root, "splits", "busi"))

    def put_normal(stem, dst_dir):
        p = normal[stem][0]
        save_img(p, os.path.join(dst_dir, os.path.basename(p)))
    for s in nsplit["train"]:
        put_normal(s, os.path.join(D, "train/NORMAL")); put_normal(s, os.path.join(V, "train/NORMAL"))
    for s in nsplit["test"]:
        put_normal(s, os.path.join(D, "test/NORMAL"))
    for s in nsplit["val"]:
        put_normal(s, os.path.join(V, "test/NORMAL"))

    val_set = set(s2["val"]); rows = {k: [] for k in s2}
    split_of = {s: k for k in s2 for s in s2[k]}
    for s, (cname, p, ms) in sorted(abn.items()):
        name = os.path.basename(p); m = merged_mask(ms, merge)
        img_seg = os.path.join(a.seg_root, "images", name); msk_seg = os.path.join(a.seg_root, "masks", name)
        save_img(p, img_seg); cv2.imwrite(msk_seg, m)
        shutil.copy(img_seg, os.path.join(D, "test/ABNORMAL", name))
        cv2.imwrite(os.path.join(D, "masks", s + "_mask.png"), m)
        if s in val_set:
            shutil.copy(img_seg, os.path.join(V, "test/ABNORMAL", name))
            cv2.imwrite(os.path.join(V, "masks", s + "_mask.png"), m)
        if s in split_of:
            rows[split_of[s]].append((os.path.abspath(img_seg), os.path.abspath(msk_seg), s, cname))
    for k in rows:
        with open(os.path.join(a.seg_root, "splits", "busi", "stage2_%s.csv" % k), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image", "mask", "stem", "class"]); w.writerows(rows[k])
        open(os.path.join(a.seg_root, "splits", "busi", "stage1_normal_%s.txt" % k), "w").write("\n".join(nsplit[k]) + "\n")

    # ---------- duplicate check (report only; the dataset is not changed) ----------
    where = defaultdict(list)
    for k in nsplit:
        for s in nsplit[k]:
            where[hashlib.md5(cv2.imread(normal[s][0], 0).tobytes()).hexdigest()].append("normal-%s:%s" % (k, s))
    for s, (cname, p, ms) in abn.items():
        where[hashlib.md5(cv2.imread(p, 0).tobytes()).hexdigest()].append("%s-S2%s:%s" % (cname, split_of.get(s, "excluded"), s))
    dups = [v for v in where.values() if len(v) > 1]
    bad = [v for v in dups if len(set(x.split(":")[0] for x in v)) > 1
           and not all("S2excluded" in x for x in v)]
    rep.append("exact duplicate image groups: %d; in different classes or splits: %d" % (len(dups), len(bad)))
    rep += ["   " + ", ".join(v) for v in bad]
    if bad:
        rep.append("   -> mention these in the paper or move one copy so both copies are in the same split")

    rep.append("Stage 1 TEST : %d normal + %d abnormal = %d" % (len(nsplit["test"]), len(abn), len(nsplit["test"]) + len(abn)))
    rep.append("Stage 1 VAL  : %d normal + %d abnormal = %d" % (len(nsplit["val"]), len(val_set), len(nsplit["val"]) + len(val_set)))
    open(os.path.join(a.seg_root, "busi_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()