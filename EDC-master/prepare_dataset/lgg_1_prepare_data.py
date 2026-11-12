"""
LGG - STEP 1: build every LGG folder and list used by the HGBL paper (all 1,373 tumour slices,
split BY PATIENT, the same patient split for Stage 1 and Stage 2).

Input: the ORIGINAL Kaggle dataset (lgg-mri-segmentation/kaggle_3m)
    <kaggle_root>/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif       (3 channels)
    <kaggle_root>/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif  (empty if no tumour)
A slice is ABNORMAL if its mask has at least one tumour pixel, otherwise NORMAL
(expected: 1,373 abnormal + 2,556 normal = 3,929 slices, 110 patients).

Split: patients 70 / 10 / 20 % (train / val / test), seed 42. Every slice of a patient goes to
the patient's split, for both stages, so no patient is in two splits.
    Stage 1 train : NORMAL slices of train patients
    Stage 1 val   : NORMAL + ABNORMAL slices of val patients   (model selection + threshold)
    Stage 1 test  : NORMAL + ABNORMAL slices of test patients
    Stage 2       : ABNORMAL slices of train / val / test patients
Images and masks are resized to 256 x 256 (INTER_AREA / INTER_NEAREST), as in prepare_lgg.py.

Output
  <data_root>/LGG_HGBL/train/NORMAL, test/NORMAL, test/ABNORMAL     data_dir  of edc_lgg_hgbl.py
  <data_root>/LGG_HGBL_VAL/train/NORMAL, test/NORMAL, test/ABNORMAL val_dir   of edc_lgg_hgbl.py
  <data_root>/LGG_HGBL_EXPORT/test/ABNORMAL  = all 1,373 tumour slices (heatmaps for Stage 2)
  <seg_root>/images/<stem>.png, <seg_root>/masks/<stem>.png         Stage 2 images / masks
  <seg_root>/splits/lgg/stage2_{train,val,test}.csv                 columns: image,mask,stem,patient
  <seg_root>/splits/lgg/patients_{train,val,test}.txt
  <seg_root>/lgg_report.txt
Usage:
  python lgg_1_prepare_data.py \
      --kaggle_root /home/cs24d0008/EDC_Segmentation/EDC-master/lgg-mri-segmentation/kaggle_3m \
      --data_root /home/cs24d0008/EDC_Segmentation/EDC-master/datasets \
      --seg_root /home/cs24d0008/EDC_Segmentation/LGG_SEG_HGBL
"""
import os, csv, glob, shutil, argparse
import numpy as np
import cv2

SIZE = 256


def patient_id(stem):
    """TCGA_CS_4941_19960909_12 -> TCGA_CS_4941"""
    return "_".join(stem.split("_")[:3])


def save_img(src, dst):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read " + src)
    cv2.imwrite(dst, cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, nargs=2, default=[0.7, 0.1], help="train and val share of patients")
    a = ap.parse_args()

    # ---------- read all slices ----------
    slices = []                      # (stem, image_path, mask_path, patient, is_abnormal)
    for d in sorted(glob.glob(os.path.join(a.kaggle_root, "TCGA_*"))):
        if not os.path.isdir(d):
            continue
        for img in sorted(glob.glob(os.path.join(d, "*.tif"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            if stem.endswith("_mask"):
                continue
            m = os.path.join(d, stem + "_mask.tif")
            if not os.path.exists(m):
                print("WARNING no mask:", img); continue
            mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            slices.append((stem, img, m, patient_id(stem), bool((mk > 0).any())))
    n_ab = sum(s[4] for s in slices); n_no = len(slices) - n_ab
    pats = sorted(set(s[3] for s in slices))
    print("found: %d slices (%d abnormal, %d normal), %d patients" % (len(slices), n_ab, n_no, len(pats)))

    # ---------- patient split ----------
    rng = np.random.RandomState(a.seed); p = list(pats); rng.shuffle(p)
    ntr = int(round(a.frac[0] * len(p))); nva = int(round(a.frac[1] * len(p)))
    split_of = {}
    for i, pt in enumerate(p):
        split_of[pt] = "train" if i < ntr else ("val" if i < ntr + nva else "test")

    # ---------- folders ----------
    D = os.path.join(a.data_root, "LGG_HGBL"); V = os.path.join(a.data_root, "LGG_HGBL_VAL")
    X = os.path.join(a.data_root, "LGG_HGBL_EXPORT")
    for root in (D, V, X):
        if os.path.exists(root):
            shutil.rmtree(root)
    for root in (D, V):
        for sub in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL"):
            os.makedirs(os.path.join(root, sub))
    for sub in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL"):   # NORMAL folders stay empty
        os.makedirs(os.path.join(X, sub))
    for sub in ("images", "masks", "splits/lgg"):
        os.makedirs(os.path.join(a.seg_root, sub), exist_ok=True)

    rows = {"train": [], "val": [], "test": []}
    cnt = {(k, c): 0 for k in rows for c in (0, 1)}
    for stem, img, m, pt, ab in slices:
        k = split_of[pt]; cnt[(k, int(ab))] += 1
        name = stem + ".tif"
        if not ab:
            if k == "train":
                save_img(img, os.path.join(D, "train/NORMAL", name))
                shutil.copy(os.path.join(D, "train/NORMAL", name), os.path.join(V, "train/NORMAL", name))
            elif k == "val":
                save_img(img, os.path.join(V, "test/NORMAL", name))
            else:
                save_img(img, os.path.join(D, "test/NORMAL", name))
            continue
        # abnormal slice: Stage 2 image + mask, Stage 1 val/test, heatmap export
        seg_img = os.path.join(a.seg_root, "images", stem + ".png")
        seg_msk = os.path.join(a.seg_root, "masks", stem + ".png")
        save_img(img, seg_img)
        mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(seg_msk, ((cv2.resize(mk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0) * 255).astype(np.uint8))
        im256 = cv2.imread(seg_img)
        cv2.imwrite(os.path.join(X, "test/ABNORMAL", name), im256)
        if k == "val":
            cv2.imwrite(os.path.join(V, "test/ABNORMAL", name), im256)
        elif k == "test":
            cv2.imwrite(os.path.join(D, "test/ABNORMAL", name), im256)
        rows[k].append((os.path.abspath(seg_img), os.path.abspath(seg_msk), stem, pt))

    sp = os.path.join(a.seg_root, "splits", "lgg")
    for k in rows:
        with open(os.path.join(sp, "stage2_%s.csv" % k), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image", "mask", "stem", "patient"]); w.writerows(sorted(rows[k]))
        open(os.path.join(sp, "patients_%s.txt" % k), "w").write(
            "\n".join(sorted(pt for pt in pats if split_of[pt] == k)) + "\n")

    # ---------- leakage check ----------
    P = {k: set(r[3] for r in rows[k]) | set(pt for pt in pats if split_of[pt] == k) for k in rows}
    assert not (P["train"] & P["val"]) and not (P["train"] & P["test"]) and not (P["val"] & P["test"])

    rep = ["LGG (patient-level split, seed %d)" % a.seed,
           "slices: %d (abnormal %d, normal %d), patients: %d" % (len(slices), n_ab, n_no, len(pats)),
           "patients train/val/test: %d / %d / %d" % tuple(sum(split_of[x] == k for x in pats) for k in ("train", "val", "test")),
           "Stage 1 train : %d normal" % cnt[("train", 0)],
           "Stage 1 VAL   : %d normal + %d abnormal = %d" % (cnt[("val", 0)], cnt[("val", 1)], cnt[("val", 0)] + cnt[("val", 1)]),
           "Stage 1 TEST  : %d normal + %d abnormal = %d" % (cnt[("test", 0)], cnt[("test", 1)], cnt[("test", 0)] + cnt[("test", 1)]),
           "Stage 2 train/val/test: %d / %d / %d (tumour slices)" % tuple(len(rows[k]) for k in ("train", "val", "test")),
           "heatmap export: %d tumour slices" % n_ab,
           "leakage check: no patient in two splits - OK"]
    open(os.path.join(a.seg_root, "lgg_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()