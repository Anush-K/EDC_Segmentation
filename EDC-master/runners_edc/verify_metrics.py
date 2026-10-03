"""
verify_metrics.py - independent check of AUC, F1, ACC, SEN, SPE.

Usage:
    python verify_metrics.py isic_scores.npz

The .npz file must contain two arrays:
    labels : 0 = normal, 1 = abnormal   (one per test image)
    scores : image-level anomaly score  (higher = more abnormal)

The threshold is the one that maximizes F1 (same protocol as EDC).
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    auc = roc_auc_score(labels, scores)

    best = None
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())

        prec = tp / (tp + fp) if tp + fp else 0.0
        sen = tp / (tp + fn) if tp + fn else 0.0
        spe = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * sen / (prec + sen) if prec + sen else 0.0
        acc = (tp + tn) / len(labels)

        if best is None or f1 > best["F1"]:
            best = dict(F1=f1, ACC=acc, SEN=sen, SPE=spe, thr=t,
                        TP=tp, TN=tn, FP=fp, FN=fn)
    return auc, best


if __name__ == "__main__":
    path = sys.argv[1]
    if path.endswith(".csv"):
        # paperproto_*_seedN.csv files: header "label,score"
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        labels, scores = arr[:, 0].astype(int), arr[:, 1]
    else:
        data = np.load(path)
        labels, scores = data["labels"], data["scores"]

    print(f"Test images : {len(labels)}  "
          f"(normal={int((labels == 0).sum())}, abnormal={int((labels == 1).sum())})")

    auc, b = evaluate(labels, scores)
    print(f"AUC : {100 * auc:.2f}")
    print(f"F1  : {100 * b['F1']:.2f}")
    print(f"ACC : {100 * b['ACC']:.2f}")
    print(f"SEN : {100 * b['SEN']:.2f}")
    print(f"SPE : {100 * b['SPE']:.2f}")
    print(f"Threshold = {b['thr']:.6f}  |  TP={b['TP']} TN={b['TN']} FP={b['FP']} FN={b['FN']}")