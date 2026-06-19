import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("./saved_models/edc_aptos_stable/results_test_edc_aptos.csv")
y_true  = df["GT"].values
y_score = df["Score"].values

EDC_PREC, EDC_REC = 0.9214, 0.9596
EA2D_PREC, EA2D_REC = 0.9706, 0.9334

print(f"{'Thresh':>8} {'Precision':>10} {'Recall':>8} {'BeatsEDC':>9} {'BeatsEA2D':>10}")
found_edc, found_ea2d = False, False
for t in np.unique(y_score):
    pred = (y_score >= t).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    beats_edc  = p > EDC_PREC and r > EDC_REC
    beats_ea2d = p > EA2D_PREC and r > EA2D_REC
    if beats_edc or beats_ea2d:
        print(f"{t:8.4f} {p:10.4f} {r:8.4f} {'YES' if beats_edc else '':>9} {'YES' if beats_ea2d else '':>10}")
        found_edc  = found_edc or beats_edc
        found_ea2d = found_ea2d or beats_ea2d

if not found_edc and not found_ea2d:
    print("\nNo single threshold on this score distribution beats both precision AND recall simultaneously for either paper.")
    print("This means the ROC curve itself (AUC) needs to improve, threshold tuning alone has hit its ceiling here.")
