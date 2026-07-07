"""
S4 -- Train + save the final SOL p_up model. Features: A(16) ONLY -- unlike
BOTH BTC (A+B+M+Cs) and ETH (A+C), NONE of B/C/Cs/R/M cleared significance
for SOL (all p_boot>=0.30, ABM confidently worse p=0.99). SOL's honest signal
comes entirely from its own technicals; no cross-asset, alt-coin, regime, or
microstructure group adds anything. Same LGBM config, full-history fit with
5% chrono val tail; honest metrics come from the walk-forward run.
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
G = json.load(open(HERE / "feature_groups.json"))
FEATURES = G["A"]
json.dump(FEATURES, open(HERE / "final_features.json", "w"), indent=1)

df = pd.read_parquet(HERE / "sol_dataset.parquet").sort_index()
df = df.dropna(subset=FEATURES + ["label"])
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

ev = pd.read_parquet(HERE / "wf_preds_A.parquet").dropna()
wf_auc = roc_auc_score(ev.label, ev.p)
wk = ev.groupby(ev.index.tz_convert(None).to_period("W-WED")).apply(
    lambda g: roc_auc_score(g.label, g.p) if g.label.nunique() > 1 and len(g) > 20 else np.nan).dropna()
ics = ev.groupby(ev.index.tz_convert(None).to_period("W-WED")).apply(
    lambda g: spearmanr(g.p, g.label).statistic if g.label.nunique() > 1 and len(g) > 20 else np.nan).dropna()
per_year = {int(yr): float(roc_auc_score(g.label, g.p)) for yr, g in ev.groupby(ev.index.year) if g.label.nunique() > 1}

meta = {
    "clf": m, "features": FEATURES, "asset": "SOL",
    "trained": "2026-07-06 honest SOL p_up rebuild (2020-2026 history)",
    "spec": "A(16 lag-corrected SOL price/tech) ONLY. B (BTC/ETH cross-asset), "
            "C (alt-coin+dominance+session), R (regime/Donchian), M (intra-hour 1m "
            "microstructure) ALL tested and REJECTED for SOL (p_boot>=0.30, ABM "
            "confidently worse p=0.99) -- different from BOTH BTC's A+B+M+Cs and "
            "ETH's A+C winning shapes. All features from completed bars; label = "
            "next-1h close direction. Beats existing leaky sol_p_up_v2_new.pkl's "
            "honest AUC~0.503 (this model: 0.5304 overall, all-years-positive).",
    "honest_wf": {"auc_overall": float(wf_auc), "auc_weekly_mean": float(wk.mean()),
                  "ic_weekly_mean": float(ics.mean()),
                  "ic_tstat": float(ics.mean() / (ics.std() / np.sqrt(len(ics)))),
                  "auc_by_year": per_year, "n_oos": int(len(ev)), "n_weeks": int(len(wk))},
    "val_tail_auc": float(auc_tail),
    "output_range_p05_p95": [float(ev.p.quantile(.05)), float(ev.p.quantile(.95))],
}
with open(HERE / "sol_p_up_v1_20260706.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"saved sol_p_up_v1_20260706.pkl: val_tail AUC={auc_tail:.4f}  WF AUC={wf_auc:.4f} "
      f"weekly {wk.mean():.4f}  IC t={meta['honest_wf']['ic_tstat']:.1f}")
print(f"per-year AUC: {per_year}")
print(f"output range p05-p95: {meta['output_range_p05_p95']}")
gain = m.booster_.feature_importance("gain"); tot = gain.sum() or 1
print("top importances:")
for nm, g in sorted(zip(FEATURES, gain), key=lambda x: -x[1])[:12]:
    print(f"  {nm:<22} {g / tot * 100:5.1f}%")
