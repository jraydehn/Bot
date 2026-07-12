"""
S14 -- adapt SOL 15m's LGBM design to hourly, properly this time. s10 tested
an hourly LGBM but used p_market DIRECTLY as a feature -- which is why it
just reconstructed the market (feature importance: p_market 31% + offset_pct
22% = >50%) and added zero edge in the uncertain zone.

SOL 15m's actual live model does something subtly but importantly different:
it uses z_score (the raw log-normal geometric distance, computed from vol,
NOT the market's already-priced probability) plus raw technical signals --
never composite_p_up/trend/rev, never p_market directly. That forces the
model to do real work estimating probability from geometry + technicals,
rather than being handed the market's own answer to lightly adjust.

This adapts that exact philosophy to the hourly archive: compute z_score
(log(strike/spot)/sigma_tau using vol_eff, matching 15m's formula), and
use only raw technicals + orthogonal microstructure (including obi_score,
independently validated on 07-10 as carrying real information the
composite score doesn't). Explicitly excludes p_market, composite_p_up/
trend/rev, p_gbdt, p_up_v2. Wrapped in CalibratedClassifierCV, same as the
live 15m model. Ticker-grouped time split (kills the s9 pseudo-replication
trap). Evaluated the same way as everything else this investigation:
Brier in the p_market-uncertain zone specifically, since that's the only
population where beating the market matters.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
    BASE_CLF = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4,
                                            num_leaves=15, min_child_samples=30, reg_lambda=5.0,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    BASE_CLF = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=4,
                                                        min_samples_leaf=30, l2_regularization=5.0, random_state=42)

# 15m-style philosophy: geometric distance (z_score) + raw technicals + orthogonal
# microstructure. NO p_market, NO composite_p_up/trend/rev, NO p_gbdt/p_up_v2.
FEATURES = [
    "z_score", "offset_pct",
    "stoch_k", "bp_5m", "body_15m", "dir_15m",
    "vwap_distance_pct", "vwap_stretch_score", "ema_stack_bias", "ema_stretch_score",
    "chg_5m", "chg_10m", "chg_30m",
    "adx_1h", "rvol_1h", "squeeze_1h", "vol_score",
    "vpin_score", "obi_score", "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct", "funding_bias",
]

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "resolved_yes", "p_market", "spot", "strike", "tau_minutes", "vol_eff"])
df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
df["vol_eff"] = pd.to_numeric(df["vol_eff"], errors="coerce")
df = df[(df["vol_eff"] > 0) & (df["tau_minutes"] > 0)]

sigma_tau = df["vol_eff"] * np.sqrt(df["tau_minutes"])
df["z_score"] = np.log(df["strike"] / df["spot"]) / sigma_tau

for c in FEATURES + ["resolved_yes"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[c for c in FEATURES if c != "z_score"] + ["resolved_yes"])
df = df.sort_values("logged_at").reset_index(drop=True)
print(f"rows: {len(df)}  tickers: {df['contract_ticker'].nunique()}  "
      f"range: {df['logged_at'].min()} -> {df['logged_at'].max()}")

# ticker-grouped time split (same as s10) -- prevents same-hour-strike-ladder leakage
tk_order = df.groupby("contract_ticker")["logged_at"].min().sort_values()
n_tk = len(tk_order)
tk_train = set(tk_order.index[:int(n_tk * 0.70)])
tk_test = set(tk_order.index[int(n_tk * 0.80):])
tk_val = set(tk_order.index[int(n_tk * 0.70):int(n_tk * 0.80)])

tr = df[df["contract_ticker"].isin(tk_train)]
va = df[df["contract_ticker"].isin(tk_val)]
te = df[df["contract_ticker"].isin(tk_test)]
print(f"train tk={len(tk_train)} n={len(tr)} | val tk={len(tk_val)} n={len(va)} | test tk={len(tk_test)} n={len(te)}")

X_tr, y_tr = tr[FEATURES].values, tr["resolved_yes"].values.astype(int)
X_va, y_va = va[FEATURES].values, va["resolved_yes"].values.astype(int)
X_te, y_te = te[FEATURES].values, te["resolved_yes"].values.astype(int)

base = BASE_CLF()
clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
clf.fit(X_tr, y_tr)

p_va = clf.predict_proba(X_va)[:, 1]
p_te = clf.predict_proba(X_te)[:, 1]
print(f"\nRow-level AUC: val={roc_auc_score(y_va, p_va):.4f}  test={roc_auc_score(y_te, p_te):.4f}")

te2 = te.copy()
te2["p_new"] = p_te
tk_te = te2.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
auc_tk = roc_auc_score((tk_te["y"] >= 0.5).astype(int), tk_te["p"])
brier_new = float(np.mean((tk_te["p"] - tk_te["y"]) ** 2))
brier_pm = float(np.mean((tk_te["pm"] - tk_te["y"]) ** 2))
print(f"Ticker-clustered AUC (test, held-out tickers): {auc_tk:.4f}")
print(f"Ticker-clustered Brier -- 15m-style model: {brier_new:.4f}   p_market alone: {brier_pm:.4f}")

unc = te2[te2["p_market"].between(0.35, 0.65)]
tk_unc = unc.groupby("contract_ticker").agg(p=("p_new", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
print(f"\nUncertain zone (test, p_market 0.35-0.65): n={len(unc)} tickers={len(tk_unc)}")
if len(tk_unc) >= 15:
    brier_new_u = float(np.mean((tk_unc["p"] - tk_unc["y"]) ** 2))
    brier_pm_u = float(np.mean((tk_unc["pm"] - tk_unc["y"]) ** 2))
    corr_new = np.corrcoef(tk_unc["p"] - 0.5, tk_unc["y"])[0, 1]
    corr_pm = np.corrcoef(tk_unc["pm"] - 0.5, tk_unc["y"])[0, 1]
    print(f"  Brier -- 15m-style model: {brier_new_u:.4f}   p_market alone: {brier_pm_u:.4f}")
    print(f"  corr(model-0.5, outcome): {corr_new:+.4f}   corr(p_market-0.5, outcome): {corr_pm:+.4f}")

# $ sim: bet whenever model disagrees with market by margin, uncertain zone
print(f"\n$ PnL proxy in uncertain zone, various margins:")
for margin in [0.03, 0.05, 0.08]:
    edge_yes = unc["p_new"] - unc["p_market"]
    edge_no = (1 - unc["p_new"]) - (1 - unc["p_market"])
    take_yes = edge_yes > margin
    take_no = edge_no > margin
    bets = []
    if take_yes.sum() > 0:
        sub = unc[take_yes]
        bets.append(pd.DataFrame({"win": sub["resolved_yes"], "cost": sub["p_market"], "tk": sub["contract_ticker"]}))
    if take_no.sum() > 0:
        sub = unc[take_no]
        bets.append(pd.DataFrame({"win": 1 - sub["resolved_yes"], "cost": 1 - sub["p_market"], "tk": sub["contract_ticker"]}))
    if not bets:
        print(f"  margin={margin}: no bets")
        continue
    allbets = pd.concat(bets)
    tk = allbets.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    print(f"  margin={margin}: n={len(tk):4d}  WR={tk['win'].mean():.1%}  BE={tk['cost'].mean():.1%}  "
          f"total=${pnl.sum():.2f}  $/bet=${pnl.sum()/len(tk):.2f}")

print(f"\nFeature importance (via base LGBM refit on full train, for interpretability):")
try:
    base2 = BASE_CLF()
    base2.fit(X_tr, y_tr)
    imp = pd.Series(base2.feature_importances_ / (base2.feature_importances_.sum() or 1), index=FEATURES).sort_values(ascending=False)
    for f, v in imp.items():
        print(f"  {f:<20s} {v:.3f}")
except Exception as e:
    print(f"  (skipped: {e})")

print("\nDONE_S14")
