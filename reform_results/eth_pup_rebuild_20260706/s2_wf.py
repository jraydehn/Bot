"""
S2 -- Walk-forward benchmark per feature-group config, identical protocol
to the BTC rebuild's s4_wf.py: expanding window, weekly refits, 2h
embargo, LGBM 300/0.03/depth4, exp recency weights, 5% chrono val tail.

usage: s2_wf.py CONFIG   where CONFIG in A, AB, ABC, ABCs, ABM, ABR
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import json

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
G = json.load(open(HERE / "feature_groups.json"))

CONFIGS = {
    "A": G["A"],
    "AB": G["A"] + G["B"],
    "ABC": G["A"] + G["B"] + G["C"],
    "ABCs": G["A"] + G["B"] + G["Cs"],
    "ABM": G["A"] + G["B"] + G["M"],
    "ABR": G["A"] + G["B"] + G["R"],
    "AC": G["A"] + G["C"],
}

name = sys.argv[1]
feats = CONFIGS[name]
assert feats, f"empty feature list for {name}"

df = pd.read_parquet(HERE / "eth_dataset.parquet").sort_index()
df = df.dropna(subset=feats + ["label"])
X = df[feats].values.astype(float)
y = df["label"].values.astype(int)
ts = df.index
print(f"{name}: {len(df)} usable rows ({ts.min()} -> {ts.max()})")

WF_START = pd.Timestamp("2021-01-06", tz="UTC")
if name == "ABM":
    WF_START = max(WF_START, ts.min() + pd.Timedelta(days=90))
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
    if len(tr) < 500:
        continue
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
