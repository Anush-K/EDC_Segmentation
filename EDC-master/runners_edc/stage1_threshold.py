"""STEP 3b - Stage 1 metrics with the threshold chosen on VALIDATION (no test-set tuning).
Give two CSV files with one row per image: a label column (1 = abnormal) and a score column
(higher = more abnormal), one for the validation split and one for the test split.
  python stage1_threshold.py --val val_scores.csv --test test_scores.csv --label_col label --score_col score
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--label_col", default="label"); ap.add_argument("--score_col", default="score")
a = ap.parse_args()
v, t = pd.read_csv(a.val), pd.read_csv(a.test)
P, R, T = precision_recall_curve(v[a.label_col], v[a.score_col])
f1 = 2 * P * R / (P + R + 1e-8); tau = T[int(np.argmax(f1[:-1]))]
y, s = t[a.label_col].values.astype(int), t[a.score_col].values
yp = (s >= tau).astype(int)
tp = int(((yp == 1) & (y == 1)).sum()); tn = int(((yp == 0) & (y == 0)).sum())
fp = int(((yp == 1) & (y == 0)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
print("threshold (from validation) = %.6f" % tau)
print("TEST  AUC %.2f  F1 %.2f  ACC %.2f  Recall %.2f  Specificity %.2f  Precision %.2f   (TP %d TN %d FP %d FN %d)" % (
    100 * roc_auc_score(y, s), 100 * 2 * prec * rec / max(prec + rec, 1e-8), 100 * (tp + tn) / len(y),
    100 * rec, 100 * spec, 100 * prec, tp, tn, fp, fn))