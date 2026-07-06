"""
S2 — Walk-forward benchmark per window-size config (15/30/45/60 min),
mirroring the hourly p_up_v3 rebuild's protocol (s4_wf.py): expanding
window, weekly refits, embargo before each refit, LGBM 300/0.03/depth4,
exponential recency weights, 5% chronological val tail for early stopping.

Embargo is scaled down from the hourly build's 2h to 30min, since 15-min
bars decide 4x more often -- a 2h embargo would swallow 8 decision points
per refit boundary for no added safety (the largest feature window is 60
min, so 30min embargo already exceeds the label horizon with margin).

usage: s2_wf.py W   where W in 15, 30, 45, 60
Saves wf_preds_W<W>.parquet (index ts, cols p, label).
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

W = int(sys.argv[1])
FEATS = [f"rv_{W}", f"upmin_frac_{W}", f"maxdd_{W}", f"volskew_last_{W}",
         f"ret_first_{W}", f"ret_last_{W}", f"rv_{W}_z10d"]

df = pd.read_parquet(HERE / "window_sweep_dataset.parquet").sort_index()
X = df[FEATS].values.astype(float)
y = df["label"].values.astype(int)
ts = df.index

WF_START = ts[0] + pd.Timedelta(days=90)  # 90d burn-in for the 960-bar rv_z10d norm
week_starts = pd.date_range(WF_START, ts[-1], freq="7D")
EMBARGO = pd.Timedelta(minutes=30)


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
    tr = np.where(ts <= ws - EMBARGO)[0]
    if len(tr) < 500:
        continue
    nv = max(int(len(tr) * 0.05), 200)
    m = fit(tr[:-nv], tr[-nv:])
    p[te] = m.predict_proba(X[te])[:, 1]
    if i % 20 == 0:
        print(f"W={W} week {i}/{len(week_starts)} ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame({"p": p, "label": y}, index=ts)
out.to_parquet(HERE / f"wf_preds_W{W}.parquet")
ev = out.dropna()
print(f"W={W}: overall AUC={roc_auc_score(ev.label, ev.p):.4f} n={len(ev)} "
      f"({time.time()-t0:.0f}s)")
