import shutil
f = "runners_edc/edc_oct2017.py"
src = open(f).read()
if "PAPER PROTOCOL" in src:
    raise SystemExit("already patched")
anchor = "    y_true = all_y_true[0]\n"
assert src.count(anchor) == 1, "anchor line not found exactly once - paste me lines 158-192 of the file"
patch = '''    # ---- paper protocol: per-seed metrics, mean +- std; no seed selection, no ensemble ----
    import time
    from sklearn.metrics import precision_recall_curve
    def _m(y, s):
        y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
        auc = roc_auc_score(y, s)
        p, r, t = precision_recall_curve(y, s)
        f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        pr = (s >= t[int(np.argmax(f))]).astype(int)
        tp = ((pr == 1) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum()
        fp = ((pr == 1) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
        sen = tp / (tp + fn); spe = tn / (tn + fp); pre = tp / max(tp + fp, 1)
        return [100 * auc, 100 * 2 * pre * sen / max(pre + sen, 1e-12),
                100 * (tp + tn) / len(y), 100 * sen, 100 * spe]
    _rows = np.array([_m(all_y_true[i], all_y_scores[i]) for i in range(len(seeds))])
    _tag = getattr(args, 'save_name', 'run') + time.strftime('_%H%M%S')
    for i, sd in enumerate(seeds):
        np.savetxt("oct_%s_seed%d.csv" % (_tag, sd), np.c_[all_y_true[i], all_y_scores[i]],
                   delimiter=",", header="label,score", comments="")
    _mu, _sd = _rows.mean(0), _rows.std(0, ddof=1)
    logger.info("PAPER PROTOCOL over %d seeds (mean +- std), use_best_checkpoint=%s" % (len(seeds), args.use_best_checkpoint))
    for _n, _a, _b in zip(["AUC", "F1", "ACC", "SEN", "SPE"], _mu, _sd):
        logger.info("  %s: %.2f +- %.2f" % (_n, _a, _b))

'''
shutil.copy(f, f + ".bak")
open(f, "w").write(src.replace(anchor, patch + anchor))
print("patched; backup saved as", f + ".bak")
