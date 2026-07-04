#!/usr/bin/env python3
"""S6 — Train + save the final honest p_up rebuild artifact (v3 candidate).

Features: A (16 price/tech, lag-corrected fixed spec) + B (ETH/SOL lead-lag)
+ M (intra-hour 1m microstructure) + Cs (session-return interactions) = 35.
Model: LGBM (same hyperparams as the honest WF benchmark), trained on the full
extended dataset (2020-01 -> 2026-07-04) with a 5% chronological validation
tail for early stopping + exp recency weights.

Honest metrics come from the walk-forward run (wf_preds_FINAL.parquet), NOT
from this fit. Artifact: btc_p_up_v3_20260704.pkl
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
FEATURES = json.load(open(HERE / "final_features.json"))
df = pd.read_parquet(HERE / "extended_dataset.parquet").sort_index()
X = df[FEATURES].values.astype(float)
y = df["label"].values.astype(int)
n = len(df)
nv = max(int(n * 0.05), 200)
t = np.arange(n - nv, dtype=float)
w = np.exp(1.5 * t / t[-1])
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                       num_leaves=15, min_child_samples=60, reg_lambda=5.0,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       verbose=-1, n_jobs=4)
m.fit(X[:n - nv], y[:n - nv], sample_weight=w,
      eval_set=[(X[n - nv:], y[n - nv:])],
      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
auc_tail = roc_auc_score(y[n - nv:], m.predict_proba(X[n - nv:])[:, 1])

# honest WF metrics
ev = pd.read_parquet(HERE / "wf_preds_FINAL.parquet").dropna()
wf_auc = roc_auc_score(ev.label, ev.p)
wk = ev.groupby(ev.index.tz_convert(None).to_period("W-WED")).apply(
    lambda g: roc_auc_score(g.label, g.p) if g.label.nunique() > 1 and len(g) > 20 else np.nan).dropna()
ics = ev.groupby(ev.index.tz_convert(None).to_period("W-WED")).apply(
    lambda g: spearmanr(g.p, g.label).statistic if g.label.nunique() > 1 and len(g) > 20 else np.nan).dropna()
per_year = {int(yr): float(roc_auc_score(g.label, g.p))
            for yr, g in ev.groupby(ev.index.year) if g.label.nunique() > 1}

meta = {
    "clf": m, "features": FEATURES,
    "trained": "2026-07-04 honest rebuild (extended 2020-2026 history)",
    "spec": "A(16 lag-corrected price/tech) + B(ETH/SOL leadlag) + "
            "M(7 intra-hour 1m micro) + Cs(4 session-x-return interactions); "
            "all features from completed bars <= decision time T+1h; "
            "4h shifted +3h; rolling z only; label next-1h close direction",
    "honest_wf": {"auc_overall": float(wf_auc), "auc_weekly_mean": float(wk.mean()),
                  "ic_weekly_mean": float(ics.mean()),
                  "ic_tstat": float(ics.mean() / (ics.std() / np.sqrt(len(ics)))),
                  "auc_by_year": per_year, "n_oos": int(len(ev)), "n_weeks": int(len(wk))},
    "val_tail_auc": float(auc_tail),
    "output_range_p05_p95": [float(ev.p.quantile(.05)), float(ev.p.quantile(.95))],
    "note": "honest output is NARROW (~[0.40,0.63]); 15m drift K and fire zones "
            "must be retuned (see s7_pnl_replay.py)",
}
out = HERE / "btc_p_up_v3_20260704.pkl"
with open(out, "wb") as f:
    pickle.dump(meta, f)
print(f"saved {out.name}: val_tail AUC={auc_tail:.4f}  WF AUC={wf_auc:.4f} "
      f"weekly {wk.mean():.4f}  IC t={meta['honest_wf']['ic_tstat']:.1f}")
gain = m.booster_.feature_importance("gain"); tot = gain.sum() or 1
print("top importances:")
for nm, g in sorted(zip(FEATURES, gain), key=lambda x: -x[1])[:12]:
    print(f"  {nm:<22} {g / tot * 100:5.1f}%")
