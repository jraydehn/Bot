#!/usr/bin/env python3
"""S4 — Walk-forward benchmark on the extended dataset (one config per run).

Protocol (identical to reform_results/pup_v2_reform_20260702/p1_longhist_benchmark.py):
expanding window, weekly refits, embargo 1 bar (train ts <= week_start - 2h),
LGBM 300/0.03/depth4, exp recency weights, 5% chrono val tail w/ early stop.

Extended: WF starts 2021-01-06 (1y burn-in). ABFcg runs only weeks >=
2026-01-12 (CoinGlass depth), training on full history with NaN flow pre-2026.

usage: s4_wf.py CONFIG   where CONFIG in A, AB, ABC, ABR, ABFt, ABM, ABFcg, FINAL
Saves wf_preds_<CONFIG>.parquet (index ts, cols p, label).
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
G = json.load(open(HERE / "feature_groups.json"))

CONFIGS = {
    "A": G["A"],
    "AB": G["A"] + G["B"],
    "ABC": G["A"] + G["B"] + G["C"],
    "ABR": G["A"] + G["B"] + G["R"],
    "ABFt": G["A"] + G["B"] + G["Ft"],
    "ABM": G["A"] + G["B"] + G["M"],
    "ABFcg": G["A"] + G["B"] + G["Fcg"],
    # C subgroups: alts (rets/basket/dominance) vs session interactions
    "ABCa": G["A"] + G["B"] + [f for f in G["C"] if "_x_" not in f],
    "ABCs": G["A"] + G["B"] + [f for f in G["C"] if "_x_" in f],
}
fj = HERE / "final_features.json"
if fj.exists():
    CONFIGS["FINAL"] = json.load(open(fj))

name = sys.argv[1]
feats = CONFIGS[name]
assert feats, f"empty feature list for {name}"

df = pd.read_parquet(HERE / "extended_dataset.parquet").sort_index()
X = df[feats].values.astype(float)
y = df["label"].values.astype(int)
ts = df.index

WF_START = pd.Timestamp("2021-01-06", tz="UTC")
if name == "ABFcg":
    WF_START = pd.Timestamp("2026-01-12", tz="UTC")
week_starts = pd.date_range(WF_START, ts[-1], freq="7D")

def fit(tr_idx, va_idx):
    t = np.arange(len(tr_idx), dtype=float)
    w = np.exp(1.5 * t / max(t[-1], 1))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                           num_leaves=15, min_child_samples=60, reg_lambda=5.0,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbose=-1, n_jobs=3)
    m.fit(X[tr_idx], y[tr_idx], sample_weight=w,
          eval_set=[(X[va_idx], y[va_idx])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return m

t0 = time.time()
p = np.full(len(df), np.nan)
for i, ws in enumerate(week_starts):
    te = np.where((ts >= ws) & (ts < ws + pd.Timedelta(days=7)))[0]
    if len(te) == 0:
        continue
    tr = np.where(ts <= ws - pd.Timedelta(hours=2))[0]
    nv = max(int(len(tr) * 0.05), 200)
    m = fit(tr[:-nv], tr[-nv:])
    p[te] = m.predict_proba(X[te])[:, 1]
    if i % 25 == 0:
        print(f"{name} week {i}/{len(week_starts)} ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame({"p": p, "label": y}, index=ts)
out.to_parquet(HERE / f"wf_preds_{name}.parquet")
ev = out.dropna()
from sklearn.metrics import roc_auc_score
print(f"{name}: overall AUC={roc_auc_score(ev.label, ev.p):.4f} n={len(ev)} "
      f"({time.time()-t0:.0f}s)")
