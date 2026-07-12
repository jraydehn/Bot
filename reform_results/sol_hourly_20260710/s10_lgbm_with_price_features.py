"""
S10 -- s9 showed the current sol_lgbm.pkl (signal-only features, no
offset_pct/p_market/tau -- deliberately excluded as "pricing leakage") is
useless-to-harmful in the tradeable p_market 0.35-0.65 zone: Brier=0.39,
badly overconfident, loses money at every margin.

Closing question: is that a fixable methodology problem, or is SOL's
hourly market just close to efficient in that zone regardless of model
form? Train a proper LGBM WITH offset_pct/p_market/tau_minutes included
as features (so it can actually learn strike-distance effects, not just
re-derive "was this a bullish scan session"), using a ticker-grouped
time-based split (train/test never share a contract_ticker, to prevent
the within-hour leakage s9 found). Evaluate the SAME way: Brier in the
uncertain zone, vs p_market alone.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LGBM = False

FEATURES = [
    "composite_p_up", "composite_trend", "composite_rev",
    "ema_stack_bias", "ema_stretch_score", "stoch_k",
    "vwap_stretch_score", "vwap_distance_pct",
    "vol_score", "vpin_score", "obi_score", "confirmation_score", "no_score",
    "funding_bias", "vol_eff", "chg_30m", "chg_10m", "chg_5m",
    "bp_5m", "body_15m", "dir_15m", "adx_1h", "rvol_1h", "squeeze_1h",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
    "offset_pct", "p_market", "tau_minutes",  # <-- the excluded "pricing" features, now INCLUDED
]

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "resolved_yes"] + FEATURES).sort_values("logged_at").reset_index(drop=True)
for c in FEATURES + ["resolved_yes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FEATURES + ["resolved_yes"])
print(f"rows: {len(df)}  tickers: {df['contract_ticker'].nunique()}  "
      f"range: {df['logged_at'].min()} -> {df['logged_at'].max()}")

# ticker-grouped time split: order tickers by their first appearance, split 70/10/20
tk_order = df.groupby("contract_ticker")["logged_at"].min().sort_values()
n_tk = len(tk_order)
tk_train = set(tk_order.index[:int(n_tk * 0.70)])
tk_val = set(tk_order.index[int(n_tk * 0.70):int(n_tk * 0.80)])
tk_test = set(tk_order.index[int(n_tk * 0.80):])

tr = df[df["contract_ticker"].isin(tk_train)]
va = df[df["contract_ticker"].isin(tk_val)]
te = df[df["contract_ticker"].isin(tk_test)]
print(f"train tk={len(tk_train)} n={len(tr)} | val tk={len(tk_val)} n={len(va)} | test tk={len(tk_test)} n={len(te)}")

X_tr, y_tr = tr[FEATURES].values, tr["resolved_yes"].values.astype(int)
X_va, y_va = va[FEATURES].values, va["resolved_yes"].values.astype(int)
X_te, y_te = te[FEATURES].values, te["resolved_yes"].values.astype(int)

if USE_LGBM:
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, max_depth=3, num_leaves=10,
                              min_child_samples=20, reg_lambda=8.0, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
else:
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.02, max_depth=3,
                                          min_samples_leaf=20, max_leaf_nodes=10, l2_regularization=8.0,
                                          early_stopping=False, random_state=42)
clf.fit(X_tr, y_tr)
p_te = clf.predict_proba(X_te)[:, 1]
p_va = clf.predict_proba(X_va)[:, 1]
print(f"\nRow-level AUC: val={roc_auc_score(y_va, p_va):.4f}  test={roc_auc_score(y_te, p_te):.4f}")

# ticker-clustered eval (avoid the s9 pseudo-replication trap)
te2 = te.copy()
te2["p_new"] = p_te
tk_te = te2.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
auc_tk = roc_auc_score((tk_te["y"] >= 0.5).astype(int), tk_te["p"])
brier_new = float(np.mean((tk_te["p"] - tk_te["y"]) ** 2))
brier_pm = float(np.mean((tk_te["pm"] - tk_te["y"]) ** 2))
print(f"Ticker-clustered AUC (test, held-out tickers): {auc_tk:.4f}")
print(f"Ticker-clustered Brier -- new model: {brier_new:.4f}   p_market alone: {brier_pm:.4f}")

# isolate to uncertain zone on test set
unc = te2[te2["p_market"].between(0.35, 0.65)]
tk_unc = unc.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
if len(tk_unc) >= 15:
    brier_new_u = float(np.mean((tk_unc["p"] - tk_unc["y"]) ** 2))
    brier_pm_u = float(np.mean((tk_unc["pm"] - tk_unc["y"]) ** 2))
    corr_new = np.corrcoef(tk_unc["p"] - 0.5, tk_unc["y"])[0, 1]
    print(f"\nUncertain zone (test, p_market 0.35-0.65): n={len(unc)} tickers={len(tk_unc)}")
    print(f"  Brier -- new model: {brier_new_u:.4f}   p_market alone: {brier_pm_u:.4f}")
    print(f"  corr(new_model-0.5, outcome): {corr_new:+.4f}")
else:
    print(f"\nUncertain zone in test set too thin: tickers={len(tk_unc)}")

print(f"\nFeature importance:")
try:
    imp = pd.Series(clf.feature_importances_ / (clf.feature_importances_.sum() or 1), index=FEATURES).sort_values(ascending=False)
    for f, v in imp.items():
        print(f"  {f:<20s} {v:.3f}")
except AttributeError:
    pass

print("\nDONE_S10")
