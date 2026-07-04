#!/usr/bin/env python3
"""S5 — Aggregate WF results: per-config weekly/yearly AUC + IC, incremental
deltas vs AB with block-bootstrap p-values, and per-feature weekly rank-IC
(per-year) for all candidate features.

Outputs: summary_configs.csv, incremental_vs_AB.csv, per_feature_ic.csv
"""
import json, warnings
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
                     "auc": roc_auc_score(g.label, g.p),
                     "ic": spearmanr(g.p, g.label).statistic})
    return pd.DataFrame(rows)

summ, weekly = [], {}
for c in configs:
    ev = frames[c].dropna()
    w = weekly_table(ev)
    weekly[c] = w
    row = {"config": c, "n": len(ev), "n_weeks": len(w),
           "auc_overall": roc_auc_score(ev.label, ev.p),
           "auc_weekly_mean": w.auc.mean(),
           "pct_weeks_gt_05": (w.auc > 0.5).mean(),
           "ic_weekly_mean": w.ic.mean(),
           "ic_tstat": w.ic.mean() / (w.ic.std() / np.sqrt(len(w)))}
    for yr, g in ev.groupby(ev.index.year):
        if g.label.nunique() > 1:
            row[f"auc_{yr}"] = roc_auc_score(g.label, g.p)
            wy = w[pd.to_datetime(w.wk_end).dt.year == yr]
            row[f"ic_{yr}"] = wy.ic.mean()
    summ.append(row)
summ = pd.DataFrame(summ).set_index("config")
summ.to_csv(HERE / "summary_configs.csv")
print("=== CONFIG SUMMARY ===")
print(summ.round(4).to_string())

# ── incremental vs AB (matched weeks), block bootstrap ────────────────────
def block_boot_p(deltas, block=8, n_boot=4000):
    """P(mean delta <= 0) under circular block bootstrap."""
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

inc = []
base_w = weekly["AB"].set_index("week")
for c in configs:
    if c == "AB":
        continue
    w = weekly[c].set_index("week")
    common = base_w.index.intersection(w.index)
    d_auc = (w.loc[common, "auc"] - base_w.loc[common, "auc"]).values
    d_ic = (w.loc[common, "ic"] - base_w.loc[common, "ic"]).values
    yrs = pd.to_datetime(w.loc[common, "wk_end"]).dt.year
    per_yr_ic = {f"d_ic_{yr}": d_ic[(yrs == yr).values].mean() for yr in sorted(yrs.unique())}
    inc.append({"config": c, "n_weeks": len(common),
                "d_auc_weekly_mean": d_auc.mean(),
                "d_ic_weekly_mean": d_ic.mean(),
                "p_boot_auc": block_boot_p(d_auc),
                "p_boot_ic": block_boot_p(d_ic),
                "pct_weeks_improved": (d_auc > 0).mean(), **per_yr_ic})
inc = pd.DataFrame(inc).set_index("config")
inc.to_csv(HERE / "incremental_vs_AB.csv")
print("\n=== INCREMENTAL vs AB (matched weeks; earns place if d_auc>=+0.004, "
      "all-year d_ic>0, boot p<0.05) ===")
print(inc.round(4).to_string())

# ── per-feature weekly IC by year ─────────────────────────────────────────
G = json.load(open(HERE / "feature_groups.json"))
df = pd.read_parquet(HERE / "extended_dataset.parquet").sort_index()
df = df[df.index >= "2021-01-06"]
cand = [f for grp in ("B", "C", "Ft", "Fcg", "R", "M") for f in G[grp]]
rows = []
wk = df.index.to_period("W-WED")
for f in cand:
    sub = pd.DataFrame({"x": df[f], "y": df["label"], "wk": wk,
                        "yr": df.index.year}).dropna()
    if len(sub) < 500:
        continue
    ics = sub.groupby("wk").apply(
        lambda g: spearmanr(g.x, g.y).statistic if len(g) > 20 and g.y.nunique() > 1 else np.nan)
    ics = ics.dropna()
    yr_of_wk = ics.index.to_timestamp(how="end").year
    row = {"feature": f, "n_weeks": len(ics), "ic_mean": ics.mean(),
           "ic_t": ics.mean() / (ics.std() / np.sqrt(len(ics)))}
    for yr in sorted(set(yr_of_wk)):
        row[f"ic_{yr}"] = ics[yr_of_wk == yr].mean()
    rows.append(row)
fic = pd.DataFrame(rows).sort_values("ic_t", key=abs, ascending=False)
fic.to_csv(HERE / "per_feature_ic.csv", index=False)
print("\n=== PER-FEATURE WEEKLY IC (top 20 by |t|) ===")
print(fic.head(20).round(4).to_string(index=False))
print("\nS5 DONE")
