"""
Fig. 7 (qualitative results) from the FINAL Stage 2 models.

For each dataset it loads the selected HGBL model (BUSI R5, LGG R3, Kvasir R5, COVID-19 R5; seed 1),
predicts every test image, sorts the test images by Dice and takes 9 images at evenly spaced
ranks (lowest, ..., highest Dice) - a fixed rule, no hand-picking. Rows per dataset: image,
Stage 1 heatmap (viridis colour map), ground truth, prediction; the Dice is written below.
Output: Fig7_qualitative.pdf (vector) and Fig7_qualitative.png (300 dpi).

Put this file in runners_edc/BUSI_HGBL/ (next to seg_data.py and hgbl_models.py) and run there:
  python make_fig7.py
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seg_data import SegDataset
from hgbl_models import build_model

B = "/home/cs24d0008/EDC_Segmentation"
RES = B + "/HGBL_results"
DATASETS = [  # name, label, seg_root, selected run
    ("busi", "BUSI", B + "/BUSI_SEG_HGBL", "R5"),
    ("lgg", "LGG", B + "/LGG_SEG_HGBL", "R3"),
    ("kvasir", "Kvasir", B + "/KVASIR_SEG_HGBL", "R5"),
    ("covid19", "COVID-19", B + "/COVID19_SEG_HGBL", "R5"),
]
N = 9


@torch.no_grad()
def predict(ds, root, run, seed, unet_path, dev):
    d = f"{RES}/{ds}/yours/{run}/seed{seed}"
    cfg = json.load(open(d + "/summary.json"))
    model = build_model("yours", cfg["in_ch"], unet_path=unet_path)
    model.load_state_dict(torch.load(d + "/best.pth", map_location="cpu"))
    model = model.to(dev).eval()
    data = SegDataset(f"{root}/splits/{ds}/stage2_test.csv", root + "/heatmaps", train=False)
    out = []
    for i in range(len(data)):
        img, heat, mask, _ = data[i]
        x = torch.cat([img, heat], 0) if cfg["in_ch"] == 4 else img
        p = model(x[None].to(dev))[0, 0].cpu().numpy() > 0.5
        g = mask[0].numpy() > 0.5
        s = p.sum() + g.sum()
        dice = 1.0 if s == 0 else 2 * np.logical_and(p, g).sum() / s
        out.append((dice, img.permute(1, 2, 0).numpy(), heat[0].numpy(), g, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unet_path", default=B + "/EDC-master/segmentation/unet.py")
    ap.add_argument("--out", default="Fig7_qualitative")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = ["Image", "Heatmap", "Ground truth", "Prediction"]
    fig, axes = plt.subplots(4 * len(DATASETS), N, figsize=(N * 0.95, 4 * len(DATASETS) * 0.98))
    for b, (ds, label, root, run) in enumerate(DATASETS):
        res = sorted(predict(ds, root, run, a.seed, a.unet_path, dev), key=lambda t: t[0])
        idx = np.round(np.linspace(0, len(res) - 1, N)).astype(int)     # evenly spaced Dice ranks
        print(f"{label}: {len(res)} test images, mean Dice {np.mean([r[0] for r in res]):.4f}", flush=True)
        for c, k in enumerate(idx):
            dice, img, heat, g, p = res[k]
            for r, im in enumerate((img, heat, g, p)):
                ax = axes[4 * b + r, c]
                ax.imshow(im, cmap=None if r == 0 else ("viridis" if r == 1 else "gray"),
                          vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c == 0:
                    ax.set_ylabel(rows[r], fontsize=6.5, rotation=0, ha="right", va="center")
            axes[4 * b + 3, c].set_xlabel(f"Dice {dice:.3f}", fontsize=6)
        axes[4 * b, 0].annotate(label, xy=(-1.05, 0.5), xycoords="axes fraction", rotation=90,
                                fontsize=9, fontweight="bold", ha="center", va="top")
    plt.subplots_adjust(left=0.13, right=0.995, top=0.995, bottom=0.02, wspace=0.04, hspace=0.12)
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=300)
    print("written", a.out + ".pdf and .png (300 dpi)")


if __name__ == "__main__":
    main()