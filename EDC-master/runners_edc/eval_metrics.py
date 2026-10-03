"""Compute AUC, F1, ACC, SEN, SPE from image-level anomaly scores.

Protocol (same as the EDC paper): the anomalous class is positive, AUC is threshold-free, and the
operating threshold for F1/ACC/SEN/SPE is the one that maximizes F1.

Usage
-----
    python eval_metrics.py scores.csv            # columns: label,score   (label 1 = abnormal, 0 = normal)
    python eval_metrics.py run1.csv run2.csv ... # several runs: prints each run and the mean over runs

`score` is S_RQASW = M(p_RQASW) for each test image (max of the fused map for OCT2017).
Save one CSV per run (last-iteration model) for RQASW, and do the same for EDC to cross-check the quoted row.
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def metrics(labels, scores):
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    auc = roc_auc_score(y, s)
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    t = thr[int(np.argmax(f1s))]
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sen = tp / (tp + fn); spe = tn / (tn + fp); pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * sen / max(pre + sen, 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(AUC=100 * auc, F1=100 * f1, ACC=100 * acc, SEN=100 * sen, SPE=100 * spe)


def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1 if not _numeric_first_row(path) else 0)
    return a[:, 0], a[:, 1]


def _numeric_first_row(path):
    with open(path) as f:
        first = f.readline().strip().split(",")
    try:
        float(first[0]); return True
    except ValueError:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # self-test on synthetic scores (250 normal / 750 abnormal, like the OCT2017 test set)
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(250), np.ones(750)]
        s = np.r_[rng.normal(0.0, 1, 250), rng.normal(2.5, 1, 750)]
        print({k: round(v, 2) for k, v in metrics(y, s).items()})
        sys.exit(0)
    rows = []
    for p in sys.argv[1:]:
        y, s = load(p)
        m = metrics(y, s); rows.append(m)
        print(p, {k: round(v, 2) for k, v in m.items()})
    if len(rows) > 1:
        print("mean", {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]})