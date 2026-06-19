import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

paths = [
    "./saved_models/edc_aptos/results_test_edc_aptos.csv",
    "./saved_models/edc_aptos_v3/results_test_edc_aptos.csv",
    "./saved_models/edc_aptos_seed2_long/results_test_edc_aptos.csv",
    "./saved_models/edc_aptos_stable/results_test_edc_aptos.csv",
]

for path in paths:
    try:
        df = pd.read_csv(path)
        y_true  = df["GT"].values
        y_score = df["Score"].values

        auc = roc_auc_score(y_true, y_score)

        fpr_list, tpr_list = [], []
        thresholds = np.unique(y_score)
        best_j, best_thresh = -1, 0.5
        for t in thresholds:
            pred = (y_score >= t).astype(int)
            tp = ((pred==1)&(y_true==1)).sum()
            fn = ((pred==0)&(y_true==1)).sum()
            fp = ((pred==1)&(y_true==0)).sum()
            tn = ((pred==0)&(y_true==0)).sum()
            tpr = tp/(tp+fn+1e-8)
            fpr = fp/(fp+tn+1e-8)
            j = tpr - fpr
            if j > best_j:
                best_j, best_thresh = j, t

        y_pred = (y_score >= best_thresh).astype(int)
        tn = ((y_pred==0)&(y_true==0)).sum()
        n  = (y_true==0).sum()

        print(f"\n=== {path} ===")
        print(f"AUC:         {auc:.4f}")
        print(f"Threshold:   {best_thresh:.4f}")
        print(f"Precision:   {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:      {recall_score(y_true, y_pred):.4f}")
        print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1:          {f1_score(y_true, y_pred):.4f}")
        print(f"Specificity: {tn/n:.4f}")
    except Exception as e:
        print(f"\n=== {path} ===\nSkipped: {e}")
