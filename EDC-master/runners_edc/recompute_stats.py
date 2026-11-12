# recompute_stats.py -- CPU only, no GPU needed
import numpy as np
from scipy import stats

# Paste the last-iteration values from your logs, seeds 0,1,2,3,4 in order.
# Leave a list empty [] if you don't have that metric yet.
runs = {
    "APTOS": {
        "edc":   {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
        "rqasw": {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
    },
    "ISIC2018": {
        "edc":   {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
        "rqasw": {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
    },
    "Br35H": {
        "edc":   {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
        "rqasw": {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
    },
    "OCT2017": {
        "edc":   {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
        "rqasw": {"AUC": [], "F1": [], "ACC": [], "SEN": [], "SPE": []},
    },
}
METRICS = ["AUC", "F1", "ACC", "SEN", "SPE"]

for ds, m in runs.items():
    print(f"===== {ds} =====")
    for meth in ("edc", "rqasw"):
        cells_mean, cells_sd = [], []
        for k in METRICS:
            v = np.array(m[meth][k], dtype=float)
            if len(v) < 2:
                cells_mean.append("--"); cells_sd.append("")
                continue
            cells_mean.append(f"{v.mean():.2f}")
            cells_sd.append(f"\\sd{{{v.std(ddof=1):.2f}}}")
        print(f"{meth:6s} & " + " & ".join(cells_mean) + r" \\")
        print("       & " + " & ".join(cells_sd) + r" \\")
    a = np.array(m["edc"]["AUC"], dtype=float)
    b = np.array(m["rqasw"]["AUC"], dtype=float)
    if len(a) >= 2 and len(b) >= 2:
        t, p = stats.ttest_ind(b, a, equal_var=False)   # two-sided Welch
        print(f"Table I row: t = {t:.2f}, p = {p:.3f}")
    print()