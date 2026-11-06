"""STEP 4 - Train ONE Stage 2 configuration with ONE seed.

Fixes compared with train_segmentation_*.py:
  * heatmap augmented together with image and mask; no augmentation on val/test
  * checkpoint chosen on the VALIDATION split; the test split is used once, at the end
  * OneCycleLR stepped once per BATCH (old code stepped once per epoch, so the
    learning rate never left the warm-up phase)
  * Dice/IoU computed PER IMAGE (old code: per batch of 8, then averaged)
  * HD95 and boundary F1 added; per-image results saved for the Wilcoxon test
  * seeded DataLoader workers (reproducible runs)

Runs (rows of the ablation table in the paper):
  R1  3-ch RGB          baseline loss            (heatmap not used at all)
  R2  4-ch RGB+S        baseline loss            <- BASELINE of Table 9
  R3  4-ch RGB+S        HGBL, lam_h = 0          (boundary factor only)
  R4  4-ch RGB+S        HGBL, lam_b = 0          (heatmap factor only)
  R5  4-ch RGB+S        HGBL full                <- HGBL of Table 9
  R6  4-ch blank S      HGBL                     (control: no heatmap content)
  R7  4-ch shuffled S   HGBL                     (control: wrong heatmap)
Example:
  python train_seg.py --dataset lgg --run R5 --seed 1 --splits splits \
      --heat_dir ../LGG_SEG/heatmaps_npy --arch yours
"""
import os, time, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import seed_everything, worker_init_fn, save_json
from seg_data import SegDataset
from losses import baseline_loss, hgbl_loss
from metrics import all_metrics
from hgbl_models import build_model

RUNS = {
    "R1": dict(in_ch=3, loss="base", heat_mode="real"),
    "R2": dict(in_ch=4, loss="base", heat_mode="real"),
    "R3": dict(in_ch=4, loss="hgbl", heat_mode="real", lam_h=0.0),
    "R4": dict(in_ch=4, loss="hgbl", heat_mode="real", lam_b=0.0),
    "R5": dict(in_ch=4, loss="hgbl", heat_mode="real"),
    "R6": dict(in_ch=4, loss="hgbl", heat_mode="zero"),
    "R7": dict(in_ch=4, loss="hgbl", heat_mode="shuffle"),
}

def forward(model, img, heat, in_ch):
    return model(torch.cat([img, heat], 1) if in_ch == 4 else img)

@torch.no_grad()
def predict(model, loader, in_ch, dev):
    model.eval(); out = {}
    for img, heat, mask, idx in loader:
        p = forward(model, img.to(dev), heat.to(dev), in_ch)
        pb = (p > 0.5).cpu().numpy()[:, 0]; g = mask.numpy()[:, 0] > 0.5
        for k, i in enumerate(idx.tolist()):
            out[i] = (pb[k], g[k])
    return out

def mean_dice(pred):
    d = []
    for p, g in pred.values():
        s = p.sum() + g.sum()
        d.append(1.0 if s == 0 else 2 * np.logical_and(p, g).sum() / s)
    return float(np.mean(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--heat_dir", required=True, help="heatmaps_npy (recommended) or heatmaps")
    ap.add_argument("--arch", default="yours", choices=["yours", "attunet", "smp_unet"])
    ap.add_argument("--unet_path", default=None, help="your unet.py (default: EDC-master/segmentation/unet.py)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam_b", type=float, default=5.0)
    ap.add_argument("--lam_h", type=float, default=1.0)
    ap.add_argument("--heat_term", default="S_pred", choices=["S_pred", "S", "S_gt", "gated", "gated_gt"])
    ap.add_argument("--heat_dropout", type=float, default=0.0,
                    help="training only: probability of zeroing the heatmap INPUT channel of an image "
                         "(the loss still uses the heatmap); makes the network robust to weak heatmaps")
    ap.add_argument("--select_on", default="val", choices=["val"], help="checkpoint selection split (val only)")
    ap.add_argument("--no_test", action="store_true", help="lambda grid: report validation only")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="extra name, e.g. lb5_lh1 for the lambda grid")
    a = ap.parse_args()

    cfg = dict(RUNS[a.run]); lam_b = cfg.get("lam_b", a.lam_b); lam_h = cfg.get("lam_h", a.lam_h)
    seed_everything(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, a.dataset, a.arch, a.run + (("_" + a.tag) if a.tag else ""), "seed%d" % a.seed)
    os.makedirs(od, exist_ok=True)
    sp = os.path.join(a.splits, a.dataset)
    mk = lambda part, train: SegDataset(os.path.join(sp, "stage2_%s.csv" % part), a.heat_dir, train,
                                        cfg["heat_mode"], perm_seed=1000 + a.seed)
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(mk("train", True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    worker_init_fn=worker_init_fn, generator=g, persistent_workers=a.workers > 0)
    vl = DataLoader(mk("val", False), a.bs, shuffle=False, num_workers=a.workers)
    model = build_model(a.arch, cfg["in_ch"], unet_path=a.unet_path).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=len(tl),
                                                epochs=a.epochs, pct_start=0.1)
    best, best_ep, log = -1.0, -1, []
    ckpt = os.path.join(od, "best.pth"); t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for img, heat, mask, _ in tl:
            img, heat, mask = img.to(dev), heat.to(dev), mask.to(dev)
            heat_in = heat
            if a.heat_dropout > 0:
                keep = (torch.rand(heat.shape[0], 1, 1, 1, device=heat.device) >= a.heat_dropout).float()
                heat_in = heat * keep
            pred = forward(model, img, heat_in, cfg["in_ch"])
            if cfg["loss"] == "base":
                loss = baseline_loss(pred, mask)
            else:
                loss = hgbl_loss(pred, mask, heat, lam_b, lam_h, a.heat_term)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        vd = mean_dice(predict(model, vl, cfg["in_ch"], dev))
        log.append((ep, tot / len(tl), vd))
        if vd > best:
            best, best_ep = vd, ep; torch.save(model.state_dict(), ckpt)
        if ep % 10 == 0 or ep == 1:
            print("ep %3d loss %.4f val_dice %.4f (best %.4f @%d) %.0fs" % (ep, tot / len(tl), vd, best, best_ep, time.time() - t0), flush=True)
    with open(os.path.join(od, "log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_dice"]); w.writerows(log)
    summ = dict(dataset=a.dataset, run=a.run, seed=a.seed, arch=a.arch, lam_b=lam_b, lam_h=lam_h,
                heat_term=a.heat_term, heat_dropout=a.heat_dropout, heat_mode=cfg["heat_mode"], in_ch=cfg["in_ch"], loss=cfg["loss"],
                best_val_dice=best, best_epoch=best_ep, params=n_par, epochs=a.epochs)
    if not a.no_test:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        te = mk("test", False)
        pr = predict(model, DataLoader(te, a.bs, shuffle=False, num_workers=a.workers), cfg["in_ch"], dev)
        rows = []
        for i in sorted(pr):
            m = all_metrics(*pr[i]); m["image"] = te.rows[i]["stem"]; m["empty_pred"] = int(not pr[i][0].any())
            rows.append(m)
        with open(os.path.join(od, "per_image.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, ["image", "dice", "iou", "hd95", "bf1", "empty_pred"]); w.writeheader(); w.writerows(rows)
        for k in ("dice", "iou", "hd95", "bf1"):
            summ["test_" + k] = float(np.mean([r[k] for r in rows]))
        summ["test_empty_pred"] = int(sum(r["empty_pred"] for r in rows)); summ["n_test"] = len(rows)
        print("TEST dice %.4f iou %.4f hd95 %.2f bf1 %.4f" % tuple(summ["test_" + k] for k in ("dice", "iou", "hd95", "bf1")))
    save_json(summ, os.path.join(od, "summary.json"))
    if os.path.exists(ckpt) and os.environ.get("KEEP_CKPT", "1") == "0":
        os.remove(ckpt)

if __name__ == "__main__":
    main()