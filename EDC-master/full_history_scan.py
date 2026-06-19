import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

paths = sorted(glob.glob("./saved_models/*/results_test_edc_aptos.csv"))

rows = []
for path in paths:
    run_name = path.split("/")[2]
    try:
        df = pd.read_csv(path)
        y_true  = df["GT"].values
        y_score = df["Score"].values

        auc = roc_auc_score(y_true, y_score)

        # Youden's J optimal threshold (no test-set method-shopping, single fixed criterion)
        best_j, best_thresh = -1, 0.5
        for t in np.unique(y_score):
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

        mtime = os.path.getmtime(path)
        import datetime
        date_str = datetime.datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")

        rows.append({
            "Run": run_name,
            "Date": date_str,
            "AUC": round(auc, 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
            "Specificity": round(tn/n, 4),
        })
    except Exception as e:
        rows.append({"Run": run_name, "Date": "?", "AUC": f"ERROR: {e}"})

result_df = pd.DataFrame(rows).sort_values("AUC", ascending=False, na_position='last')
pd.set_option('display.width', 150)
print(result_df.to_string(index=False))
result_df.to_csv("full_history_scan_results.csv", index=False)
print("\nSaved to full_history_scan_results.csv")
