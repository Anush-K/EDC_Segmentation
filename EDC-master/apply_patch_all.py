import re, shutil, sys

PATCH = '''    # ---- paper protocol: per-seed metrics, mean +- std; no seed selection, no ensemble ----
    import os, time
    from sklearn.metrics import precision_recall_curve
    def _pp_m(y, s):
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
    _pp_rows = np.array([_pp_m(all_y_true[i], all_y_scores[i]) for i in range(len(seeds))])
    _pp_tag = str(getattr(args, 'save_name', os.path.basename(__file__))) + time.strftime('_%H%M%S')
    for _i, _sd in enumerate(seeds):
        np.savetxt("paperproto_%s_seed%d.csv" % (_pp_tag, _sd), np.c_[all_y_true[_i], all_y_scores[_i]],
                   delimiter=",", header="label,score", comments="")
    logger.info("PAPER PROTOCOL over %d seeds (mean +- std); use_rqasw=%s; per-seed rows are AUC F1 ACC SEN SPE"
                % (len(seeds), getattr(args, 'use_rqasw', 'n/a')))
    for _i, _sd in enumerate(seeds):
        logger.info("  seed %d: %s" % (_sd, " ".join("%.2f" % v for v in _pp_rows[_i])))
    for _n, _a, _b in zip(["AUC", "F1", "ACC", "SEN", "SPE"], _pp_rows.mean(0), _pp_rows.std(0, ddof=1)):
        logger.info("  %s: %.2f +- %.2f" % (_n, _a, _b))
    if 'all_y1_scores' in locals():
        for _k, _lst in enumerate([all_y1_scores, all_y2_scores, all_y3_scores], 1):
            _v = np.array([100 * float(np.ravel(x)[0]) for x in _lst])
            logger.info("  single-scale AUC p%d: %.2f +- %.2f" % (_k, _v.mean(), _v.std(ddof=1)))

'''
ANCHOR = re.compile(r'^    y_true\s*=\s*all_y_true\[0\]\s*$', re.M)

for f in sys.argv[1:]:
    src = open(f).read()
    if "PAPER PROTOCOL" in src:
        print(f, ": already patched, skipped"); continue
    m = ANCHOR.findall(src)
    if len(m) != 1:
        print(f, ": anchor found %d times (need exactly 1) - NOT patched" % len(m)); continue
    shutil.copy(f, f + ".bak2")
    pos = ANCHOR.search(src).start()
    open(f, "w").write(src[:pos] + PATCH + src[pos:])
    print(f, ": patched (backup", f + ".bak2)")
