"""STEP 6 - Collect all runs: mean +/- SD over seeds, Wilcoxon test, LaTeX rows for the paper.
  python aggregate.py --results results --arch yours
"""
import os, glob, json, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--arch", default="yours")
ap.add_argument("--splits", default="splits", help="for BUSI per-class results (class column)")
ap.add_argument("--busi_classes", default=None, help="classes.csv from prepare_busi_v2.py: prints benign/malignant Dice")
a = ap.parse_args()
DS = ["busi", "lgg", "kvasir", "covid19"]; NAME = {"busi": "BUSI", "lgg": "LGG", "kvasir": "Kvasir", "covid19": "COVID-19"}

def runs(ds, run):
    return sorted(glob.glob(os.path.join(a.results, ds, a.arch, run, "seed*")))

def seed_means(ds, run, k):
    v = []
    for d in runs(ds, run):
        p = os.path.join(d, "summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            if "test_" + k in s: v.append(s["test_" + k])
    return np.array(v)

def seeds_of(ds, run):
    return set(os.path.basename(d) for d in runs(ds, run) if os.path.exists(os.path.join(d, "per_image.csv")))

def per_image(ds, run, k="dice", seeds=None):
    """Per-image metric averaged over seeds. If seeds is given, only those seeds are used,
    so two runs are always compared on the SAME seeds."""
    fs = [os.path.join(d, "per_image.csv") for d in runs(ds, run)
          if seeds is None or os.path.basename(d) in seeds]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs: return None
    df = pd.concat([pd.read_csv(f)[["image", k]].set_index("image")[k].rename(i) for i, f in enumerate(fs)], axis=1)
    return df.mean(1)                                   # per image, mean over seeds

def matched(ds, x, y, k="dice"):
    """(u, v, common seeds): both runs averaged over the seeds they have in common."""
    c = seeds_of(ds, x) & seeds_of(ds, y)
    if not c: return None, None, c
    return per_image(ds, x, k, c), per_image(ds, y, k, c), c

def fmt(v, scale=100):
    if len(v) == 0: return "--"
    return "%.2f $\\pm$ %.2f" % (scale * v.mean(), scale * v.std(ddof=1)) if len(v) > 1 else "%.2f" % (scale * v[0])

print("\n=== Table 9 (baseline R2 vs HGBL R5) ===")
for ds in DS:
    cells = []
    for k, sc in (("dice", 100), ("iou", 100), ("hd95", 1), ("bf1", 100)):
        for r in ("R2", "R5"):
            cells.append(fmt(seed_means(ds, r, k), sc))
    a5, a2, _ = matched(ds, "R5", "R2"); pv = "--"
    if a2 is not None and a5 is not None:
        j = a2.index.intersection(a5.index); pv = "%.3g" % wilcoxon(a5[j], a2[j]).pvalue
    print("%s & %s & %s \\\\" % (NAME[ds], " & ".join(cells), pv))

if a.busi_classes:
    cm = pd.read_csv(a.busi_classes).set_index("stem")["class"]
    print("\n=== BUSI by class (Dice %, mean over seeds per image, then over images) ===")
    for r in ("R2", "R5"):
        v = per_image("busi", r)
        if v is not None:
            print(r, "  ".join("%s %.2f (n=%d)" % (c, 100 * v[cm.reindex(v.index) == c].mean(), int((cm.reindex(v.index) == c).sum()))
                              for c in sorted(cm.unique())))

print("\n=== Per-class Dice (BUSI: benign / malignant), mean over seeds ===")
for ds in DS:
    p = os.path.join(a.splits, ds, "stage2_test.csv")
    if not os.path.exists(p): continue
    t = pd.read_csv(p)
    if "class" not in t.columns: continue
    cls = dict(zip(t["stem"].astype(str), t["class"]))
    for r in ("R2", "R5"):
        u = per_image(ds, r)
        if u is None: continue
        c = pd.Series([cls.get(str(i), "?") for i in u.index], index=u.index)
        print("%-8s %s  " % (ds, r) + "  ".join("%s %.2f (n=%d)" % (k, 100 * u[c == k].mean(), (c == k).sum()) for k in sorted(set(c))))

print("\n=== Ablation table, Dice ===")
for r, lab in [("R1", "(i)"), ("R2", "(ii)"), ("R3", "(iii)"), ("R4", "(iv)"), ("R5", "(v)"), ("R6", "(vi)"), ("R7", "(vii)")]:
    sd = sorted(seeds_of("busi", r))
    print(lab, " & ".join(fmt(seed_means(ds, r, "dice")) for ds in DS), "\\\\",
          ("   [BUSI seeds: %s]" % ",".join(sd)) if sd else "")

print("\n=== Paired tests on per-image Dice, on the SEEDS BOTH RUNS HAVE ===")
for ds in DS:
    for x, y in [("R5", "R2"), ("R5", "R3"), ("R5", "R6"), ("R5", "R7"), ("R3", "R2"), ("R7", "R6"), ("R3", "R6")]:
        u, v, c = matched(ds, x, y)
        if u is None or v is None: continue
        j = u.index.intersection(v.index)
        print("%-8s %s vs %s: mean diff %+.2f  p = %.3g   (common seeds: %s)" % (
            ds, x, y, 100 * (u[j] - v[j]).mean(), wilcoxon(u[j], v[j]).pvalue, ",".join(sorted(c))))

print("\n=== Ablation, all metrics (per dataset): Dice / IoU / HD95 / BF1, mean +/- SD over seeds ===")
for ds in DS:
    rows = []
    for r, lab in [("R1", "(i)"), ("R2", "(ii)"), ("R3", "(iii)"), ("R4", "(iv)"), ("R5", "(v)"), ("R6", "(vi)"), ("R7", "(vii)")]:
        if not seeds_of(ds, r): continue
        rows.append("%-6s %-4s Dice %s | IoU %s | HD95 %s | BF1 %s   [%d seeds]" % (
            lab, r, fmt(seed_means(ds, r, "dice")), fmt(seed_means(ds, r, "iou")),
            fmt(seed_means(ds, r, "hd95"), 1), fmt(seed_means(ds, r, "bf1")), len(seeds_of(ds, r))))
    if rows:
        print(NAME[ds]); print("\n".join(rows))

print("\n=== Paired tests on boundary F1 and HD95 (same seeds) ===")
for ds in DS:
    for x, y in [("R5", "R2"), ("R5", "R3"), ("R5", "R6"), ("R5", "R7"), ("R3", "R6")]:
        for k, sc in (("bf1", 100), ("hd95", 1)):
            u, v, c = matched(ds, x, y, k)
            if u is None or v is None: continue
            j = u.index.intersection(v.index)
            print("%-8s %s vs %s  %-4s: mean diff %+.2f  p = %.3g   (seeds: %s)" % (
                ds, x, y, k, sc * (u[j] - v[j]).mean(), wilcoxon(u[j], v[j]).pvalue, ",".join(sorted(c))))

print("\n=== Heatmap quality (gains on common seeds) ===")
for ds in DS:
    p = os.path.join(a.results, ds, "heatmap_quality_test.json")
    if os.path.exists(p):
        q = json.load(open(p))
        u, v, _ = matched(ds, "R5", "R2"); g = 100 * (u - v).mean() if u is not None else float("nan")
        u, v, _ = matched(ds, "R5", "R7"); l = 100 * (u - v).mean() if u is not None else float("nan")
        print("%s & %.3f & %.3f & %+.2f & %+.2f \\\\" % (NAME[ds], q["pixel_auroc"], q["pointing_game"], g, l))

print("\n=== Lambda grid (validation Dice, mean over datasets and seeds) ===")
for lb in ("1", "3", "5", "10"):
    row = []
    for lh in ("0", "0.5", "1", "2"):
        v = []
        for ds in DS:
            for d in glob.glob(os.path.join(a.results, ds, a.arch, "R5_lb%s_lh%s" % (lb, lh), "seed*", "summary.json")):
                v.append(json.load(open(d))["best_val_dice"])
        row.append("%.2f" % (100 * np.mean(v)) if v else "--")
    print("lambda_b = %s & %s \\\\" % (lb, " & ".join(row)))