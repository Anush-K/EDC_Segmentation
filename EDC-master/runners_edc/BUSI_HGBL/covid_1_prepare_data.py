"""COVID-19 step 1 (HGBL paper): COVID-19 Radiography Database, image-level split 70/10/20 (seed 42).
Normal/images -> Stage 1 normal pool; COVID/images + COVID/masks (lung masks) -> abnormal pool and Stage 2.
Writes COVID19_HGBL, COVID19_HGBL_VAL, COVID19_HGBL_EXPORT (Stage 1) and <seg_root>/{images,masks,splits/covid19}."""
import os, csv, glob, shutil, argparse
import numpy as np
import cv2

SIZE = 256


def list_imgs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def split3(items, rng, frac=(0.7, 0.1)):
    items = sorted(items); rng.shuffle(items)
    n = len(items); a = int(round(frac[0] * n)); b = int(round(frac[1] * n))
    return items[:a], items[a:a + b], items[a + b:]


def load(path, is_mask=False):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read " + path)
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    return ((im > 127) * 255).astype(np.uint8) if is_mask else im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covid_root", required=True, help="COVID-19_Radiography_Dataset")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    normals = [(os.path.splitext(os.path.basename(f))[0], f) for f in list_imgs(os.path.join(a.covid_root, "Normal", "images"))]
    covid = []
    for f in list_imgs(os.path.join(a.covid_root, "COVID", "images")):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = os.path.join(a.covid_root, "COVID", "masks", os.path.basename(f))
        if not os.path.exists(m):
            print("WARNING no mask:", f); continue
        covid.append((stem, f, m))
    print("found: %d normal, %d COVID images (with masks)" % (len(normals), len(covid)), flush=True)
    ntr, nva, nte = split3([n[0] for n in normals], rng)
    ctr, cva, cte = split3([c[0] for c in covid], rng)
    nsplit = {s: "train" for s in ntr}; nsplit.update({s: "val" for s in nva}); nsplit.update({s: "test" for s in nte})
    csplit = {s: "train" for s in ctr}; csplit.update({s: "val" for s in cva}); csplit.update({s: "test" for s in cte})
    D = os.path.join(a.data_root, "COVID19_HGBL"); V = os.path.join(a.data_root, "COVID19_HGBL_VAL")
    X = os.path.join(a.data_root, "COVID19_HGBL_EXPORT")
    for root in (D, V, X):
        if os.path.exists(root):
            shutil.rmtree(root)
        for sub in ("train/NORMAL", "test/NORMAL", "test/ABNORMAL"):
            os.makedirs(os.path.join(root, sub))
    for sub in ("images", "masks", "splits/covid19"):
        os.makedirs(os.path.join(a.seg_root, sub), exist_ok=True)
    for stem, f in normals:
        im = load(f); k = nsplit[stem]; name = stem + ".png"
        if k == "train":
            cv2.imwrite(os.path.join(D, "train/NORMAL", name), im); cv2.imwrite(os.path.join(V, "train/NORMAL", name), im)
        elif k == "val":
            cv2.imwrite(os.path.join(V, "test/NORMAL", name), im)
        else:
            cv2.imwrite(os.path.join(D, "test/NORMAL", name), im)
    rows = {"train": [], "val": [], "test": []}
    for stem, f, mf in covid:
        im = load(f); mk = load(mf, True); k = csplit[stem]; name = stem + ".png"
        si, sm = os.path.join(a.seg_root, "images", name), os.path.join(a.seg_root, "masks", name)
        cv2.imwrite(si, im); cv2.imwrite(sm, mk); cv2.imwrite(os.path.join(X, "test/ABNORMAL", name), im)
        if k == "val":
            cv2.imwrite(os.path.join(V, "test/ABNORMAL", name), im)
        elif k == "test":
            cv2.imwrite(os.path.join(D, "test/ABNORMAL", name), im)
        rows[k].append((os.path.abspath(si), os.path.abspath(sm), stem, stem))
    for k in rows:
        with open(os.path.join(a.seg_root, "splits", "covid19", "stage2_%s.csv" % k), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "mask", "stem", "patient"]); w.writerows(sorted(rows[k]))
    c = lambda d: len(os.listdir(d))
    rep = ["COVID-19 Radiography (image-level split, seed %d)" % a.seed,
           "Stage 1 train : %d normal" % c(os.path.join(D, "train/NORMAL")),
           "Stage 1 VAL   : %d normal + %d COVID" % (c(os.path.join(V, "test/NORMAL")), c(os.path.join(V, "test/ABNORMAL"))),
           "Stage 1 TEST  : %d normal + %d COVID" % (c(os.path.join(D, "test/NORMAL")), c(os.path.join(D, "test/ABNORMAL"))),
           "Stage 2 train/val/test: %d / %d / %d" % tuple(len(rows[k]) for k in ("train", "val", "test"))]
    open(os.path.join(a.seg_root, "covid19_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()