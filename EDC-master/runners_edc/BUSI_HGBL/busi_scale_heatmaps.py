"""
BUSI - Idea B: find a better Stage 1 heatmap WITHOUT re-training.

Loads the 5 Stage 1 checkpoints (saved_models/edc_busi_hgbl/seed_<s>/model_best.pth, chosen on
validation AUC), computes the three scale maps p1, p2, p3 and the RQASW map p_all for every
Stage 2 image, averages each over the 5 seeds (each map min-max normalised per image first),
and compares candidate heatmaps on the VALIDATION split only:
    rqasw (current) | p1 | p2 | p3 | mean (EDC equal weights) | p1+p2 | p2+p3 | max
Pixel AUROC against the lesion mask and pointing game are printed. Then
    --write NAME   writes <seg_root>/heatmaps_NAME/<stem>.png (grayscale, same format as now).

Put this file in runners_edc/BUSI_HGBL/ (it finds EDC-master itself).
Usage:
  cd /home/cs24d0008/EDC_Segmentation/EDC-master
  python runners_edc/BUSI_HGBL/busi_scale_heatmaps.py --seg_root /home/cs24d0008/EDC_Segmentation/BUSI_SEG_HGBL
  python runners_edc/BUSI_HGBL/busi_scale_heatmaps.py --seg_root ... --write p1
"""
import os, sys, csv, argparse


def _find_edc_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if all(os.path.isdir(os.path.join(d, x)) for x in ("methods", "models", "helper_modules")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("EDC-master not found")


ROOT = _find_edc_root(); sys.path.insert(0, ROOT)
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models.edc import R50_R50

SIZE = 256


def mm(a):
    lo = a.amin(dim=(-2, -1), keepdim=True); hi = a.amax(dim=(-2, -1), keepdim=True)
    return (a - lo) / (hi - lo).clamp(min=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", required=True)
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "saved_models", "edc_busi_hgbl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dataset", default="busi")
    ap.add_argument("--var_reg_weight", type=float, default=1.0)
    ap.add_argument("--bn_pretrain", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--write", default=None, help="candidate name to write, e.g. p1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = {p: list(csv.DictReader(open(os.path.join(a.seg_root, "splits", a.dataset, "stage2_%s.csv" % p))))
            for p in ("train", "val", "test")}
    allrows = rows["train"] + rows["val"] + rows["test"]
    names = ["p1", "p2", "p3", "p_all"]
    acc = {k: np.zeros((len(allrows), SIZE, SIZE), np.float32) for k in names}

    for s in a.seeds:
        ck = os.path.join(a.ckpt_dir, "seed_%d" % s, "model_best.pth")
        model = R50_R50(img_size=SIZE, train_encoder=True, stop_grad=False, reshape=True,
                        bn_pretrain=bool(a.bn_pretrain), var_reg_weight=a.var_reg_weight,
                        ema_momentum=0.999, use_rqasw=True)
        st = torch.load(ck, map_location="cpu"); model.load_state_dict(st["model"] if "model" in st else st)
        model = model.to(a.device).eval()
        print("seed %d: %s" % (s, ck), flush=True)
        with torch.no_grad():
            for i in range(0, len(allrows), a.bs):
                batch = allrows[i:i + a.bs]
                x = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(r["image"]), (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
                              for r in batch]).astype(np.float32) / 255.0
                out = model(torch.from_numpy(x).permute(0, 3, 1, 2).to(a.device))
                for k in names:
                    m = F.interpolate(out[k].float(), size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
                    acc[k][i:i + len(batch)] += mm(m).cpu().numpy() / len(a.seeds)

    p1, p2, p3, pa = (acc[k] for k in names)
    cand = {"rqasw (current)": pa, "p1": p1, "p2": p2, "p3": p3, "mean (EDC)": (p1 + p2 + p3) / 3,
            "p1+p2": (p1 + p2) / 2, "p2+p3": (p2 + p3) / 2, "max": np.maximum(np.maximum(p1, p2), p3)}
    nval0 = len(rows["train"]); nval = len(rows["val"])
    masks = [cv2.resize(cv2.imread(r["mask"], 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0 for r in rows["val"]]
    print("\nVALIDATION split (%d images) - choose the best row here" % nval)
    print("  %-18s pixel AUROC   pointing game" % "candidate")
    for k, maps in cand.items():
        aucs, hits = [], []
        for j, m in enumerate(masks):
            h = maps[nval0 + j]
            if m.any() and (~m).any():
                aucs.append(roc_auc_score(m.ravel(), h.ravel()))
            if m.any():
                hits.append(bool(m.ravel()[h.argmax()]))
        print("  %-18s %.3f         %.1f%%" % (k, np.mean(aucs), 100 * np.mean(hits)), flush=True)

    if a.write:
        key = {"rqasw": "rqasw (current)", "mean": "mean (EDC)"}.get(a.write, a.write)
        maps = cand[key]; out = os.path.join(a.seg_root, "heatmaps_" + a.write.replace("+", "_"))
        os.makedirs(out, exist_ok=True)
        for r, h in zip(allrows, maps):
            lo, hi = h.min(), h.max(); h = (h - lo) / (hi - lo) if hi > lo else h
            cv2.imwrite(os.path.join(out, r["stem"] + ".png"), (255 * h).astype(np.uint8))
        print("wrote %d heatmaps to %s" % (len(allrows), out))


if __name__ == "__main__":
    main()