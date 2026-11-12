from scipy import stats
import numpy as np

n = 5  # seeds

data = {
    "OCT2017": {
        "test": "welch",
        "edc_mean": 98.26, "edc_std": 0.20,
        "rqasw_mean": 98.63, "rqasw_std": 0.25,
    },
    "APTOS": {
        "test": "one-sample",
        "edc_mean": 95.41,
        "rqasw_mean": 96.85, "rqasw_std": 0.32,
    },
    "Br35H": {
        "test": "one-sample",
        "edc_mean": 99.85,
        "rqasw_mean": 99.96, "rqasw_std": 0.06,
    },
    "ISIC2018": {
        "test": "welch",
        "edc_mean": 87.75, "edc_std": 2.41,
        "rqasw_mean": 89.97, "rqasw_std": 1.48,
    },
}

print(f"{'Dataset':<10}{'Test':<12}{'t':>8}{'p':>10}")
print("-" * 40)

for name, d in data.items():
    if d["test"] == "welch":
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=d["edc_mean"], std1=d["edc_std"], nobs1=n,
            mean2=d["rqasw_mean"], std2=d["rqasw_std"], nobs2=n,
            equal_var=False
        )
    else:
        se = d["rqasw_std"] / np.sqrt(n)
        t_stat = (d["rqasw_mean"] - d["edc_mean"]) / se
        df = n - 1
        p_val = 2 * stats.t.sf(abs(t_stat), df)

    print(f"{name:<10}{d['test']:<12}{t_stat:>8.2f}{p_val:>10.3f}")
