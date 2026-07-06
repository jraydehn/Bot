"""
S3 — Aggregate WF results across window sizes: per-config weekly/yearly
AUC+IC, incremental deltas vs the W=15 baseline with block-bootstrap
p-values (same protocol as the hourly rebuild's s5_analyze.py).
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)
WINDOWS = [15, 30, 45, 60]

frames = {W: pd.read_parquet(HERE / f"wf_preds_W{W}.parquet") for W in WINDOWS}


def weekly_table(ev):
    rows = []
    for wk, g in ev.groupby(ev.index.to_period("W-WED")):
        if g.label.nunique() < 2 or len(g) < 50:
            continue
        rows.append({"week": str(wk), "wk_end": g.index[-1], "n": len(g),
                     "auc": roc_auc_score(g.label, g.p),
                     "ic": spearmanr(g.p, g.label).statistic})
    return pd.DataFrame(rows)


summ, weekly = [], {}
for W in WINDOWS:
    ev = frames[W].dropna()
    w = weekly_table(ev)
    weekly[W] = w
    row = {"W": W, "n": len(ev), "n_weeks": len(w),
           "auc_overall": roc_auc_score(ev.label, ev.p),
           "auc_weekly_mean": w.auc.mean(),
           "pct_weeks_gt_05": (w.auc > 0.5).mean(),
           "ic_weekly_mean": w.ic.mean(),
           "ic_tstat": w.ic.mean() / (w.ic.std() / np.sqrt(len(w)))}
    for yr, g in ev.groupby(ev.index.year):
        if g.label.nunique() > 1:
            row[f"auc_{yr}"] = roc_auc_score(g.label, g.p)
    summ.append(row)
summ = pd.DataFrame(summ).set_index("W")
print("=== CONFIG SUMMARY ===")
print(summ.round(4).to_string())


def block_boot_p(deltas, block=8, n_boot=4000):
    d = np.asarray(deltas)
    n = len(d)
    if n < block + 2:
        return np.nan
    means = np.empty(n_boot)
    nblk = int(np.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n, nblk)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[i] = d[idx[:n]].mean()
    return float((means <= 0).mean())


print("\n=== INCREMENTAL vs W=15 (matched weeks; is the LARGER window "
      "significantly BETTER (p_boot low & positive) or WORSE? ===")
base_w = weekly[15].set_index("week")
inc = []
for W in WINDOWS:
    if W == 15:
        continue
    w = weekly[W].set_index("week")
    common = base_w.index.intersection(w.index)
    d_auc = (w.loc[common, "auc"] - base_w.loc[common, "auc"]).values
    d_ic = (w.loc[common, "ic"] - base_w.loc[common, "ic"]).values
    p_worse = block_boot_p(-d_auc)  # P(mean delta >= 0) i.e. tests if W is worse
    inc.append({"W": W, "n_weeks": len(common),
                "d_auc_weekly_mean": d_auc.mean(), "d_ic_weekly_mean": d_ic.mean(),
                "p_boot_W_better": block_boot_p(d_auc),
                "p_boot_W_worse": p_worse,
                "pct_weeks_W_better": (d_auc > 0).mean()})
inc = pd.DataFrame(inc).set_index("W")
print(inc.round(4).to_string())

print("\n=== per-feature weekly IC by config (top |t| per window size) ===")
ds = pd.read_parquet(HERE / "window_sweep_dataset.parquet").sort_index()
ds = ds[ds.index >= (ds.index[0] + pd.Timedelta(days=90))]
wk = ds.index.to_period("W-WED")
for W in WINDOWS:
    feats = [f"rv_{W}", f"upmin_frac_{W}", f"maxdd_{W}", f"volskew_last_{W}",
             f"ret_first_{W}", f"ret_last_{W}", f"rv_{W}_z10d"]
    rows = []
    for f in feats:
        sub = pd.DataFrame({"x": ds[f], "y": ds["label"], "wk": wk}).dropna()
        if len(sub) < 500:
            continue
        ics = sub.groupby("wk").apply(
            lambda g: spearmanr(g.x, g.y).statistic if len(g) > 30 and g.y.nunique() > 1 else np.nan).dropna()
        rows.append({"feature": f, "ic_mean": ics.mean(),
                     "ic_t": ics.mean() / (ics.std() / np.sqrt(len(ics)))})
    fic = pd.DataFrame(rows).sort_values("ic_t", key=abs, ascending=False)
    print(f"\n--- W={W} ---")
    print(fic.round(4).to_string(index=False))
