"""
S3 -- Aggregate ETH WF results: per-config weekly/yearly AUC+IC,
incremental deltas vs AB baseline with block-bootstrap p-values.
Same acceptance bar as the BTC rebuild: d_auc>=+0.004, all-year
d_ic>0, boot p<0.05.
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

configs = [p.stem.replace("wf_preds_", "") for p in HERE.glob("wf_preds_*.parquet")]
frames = {c: pd.read_parquet(HERE / f"wf_preds_{c}.parquet") for c in configs}


def weekly_table(ev):
    rows = []
    for wk, g in ev.groupby(ev.index.to_period("W-WED")):
        if g.label.nunique() < 2 or len(g) < 20:
            continue
        rows.append({"week": str(wk), "wk_end": g.index[-1], "n": len(g),
                     "auc": roc_auc_score(g.label, g.p), "ic": spearmanr(g.p, g.label).statistic})
    return pd.DataFrame(rows)


summ, weekly = [], {}
for c in configs:
    ev = frames[c].dropna()
    w = weekly_table(ev)
    weekly[c] = w
    row = {"config": c, "n": len(ev), "n_weeks": len(w),
           "auc_overall": roc_auc_score(ev.label, ev.p), "auc_weekly_mean": w.auc.mean(),
           "pct_weeks_gt_05": (w.auc > 0.5).mean(), "ic_weekly_mean": w.ic.mean(),
           "ic_tstat": w.ic.mean() / (w.ic.std() / np.sqrt(len(w)))}
    for yr, g in ev.groupby(ev.index.year):
        if g.label.nunique() > 1:
            row[f"auc_{yr}"] = roc_auc_score(g.label, g.p)
    summ.append(row)
summ = pd.DataFrame(summ).set_index("config")
print("=== CONFIG SUMMARY ===")
print(summ.round(4).to_string())


def block_boot_p(deltas, block=8, n_boot=4000):
    d = np.asarray(deltas); n = len(d)
    if n < block + 2:
        return np.nan
    means = np.empty(n_boot); nblk = int(np.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n, nblk)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[i] = d[idx[:n]].mean()
    return float((means <= 0).mean())


print("\n=== INCREMENTAL vs AB (matched weeks) ===")
base_w = weekly["AB"].set_index("week")
inc = []
for c in configs:
    if c in ("AB", "A"):
        continue
    w = weekly[c].set_index("week")
    common = base_w.index.intersection(w.index)
    d_auc = (w.loc[common, "auc"] - base_w.loc[common, "auc"]).values
    d_ic = (w.loc[common, "ic"] - base_w.loc[common, "ic"]).values
    yrs = pd.to_datetime(w.loc[common, "wk_end"]).dt.year
    per_yr_ic = {f"d_ic_{yr}": d_ic[(yrs == yr).values].mean() for yr in sorted(yrs.unique())}
    inc.append({"config": c, "n_weeks": len(common), "d_auc_weekly_mean": d_auc.mean(),
                "d_ic_weekly_mean": d_ic.mean(), "p_boot_auc": block_boot_p(d_auc),
                "p_boot_ic": block_boot_p(d_ic), "pct_weeks_improved": (d_auc > 0).mean(), **per_yr_ic})
inc = pd.DataFrame(inc).set_index("config")
print(inc.round(4).to_string())
