"""
Retrain BTC LGBM shadow model with expanded feature set.

New additions vs prior model (28 features):
  - From archive:  offset_pct, pm_drift_5m, tau_minutes
  - Backfilled:    rsi_14, cci_20, macd_hist, di_plus, di_minus, stoch_k_4h, mfi_14

Run: python3 retrain_btc_lgbm.py
Overwrites reform_results/btc_lgbm.pkl after showing results.
"""

import glob, math, pickle, shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

RESULTS_DIR = Path("results")
REFORM_DIR  = Path("reform_results")
ARCHIVE_CSV = RESULTS_DIR / "btc_scan_archive.csv"
MODEL_PATH  = REFORM_DIR  / "btc_lgbm.pkl"
BACKUP_PATH = REFORM_DIR  / "btc_lgbm_pre_platt_recal_20260615.pkl"  # already exists

# ---------------------------------------------------------------------------
# 1. Load Binance 1h candles and compute new indicators
# ---------------------------------------------------------------------------
print("Loading 1h candles …")
parquet = sorted(glob.glob("data/binanceus_BTCUSDT_1h_1970-01-01_*.parquet"))[-1]
c1h = pd.read_parquet(parquet).sort_index()
c1h.index = pd.to_datetime(c1h.index, utc=True)
hi, lo, cl, vol = c1h["high"], c1h["low"], c1h["close"], c1h["volume"]

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

def _macd_hist(c, f=12, s=26, sig=9):
    macd = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    return macd - macd.ewm(span=sig, adjust=False).mean()

def _di(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    dm_plus  = (h - h.shift()).clip(lower=0)
    dm_minus = (l.shift() - l).clip(lower=0)
    dm_plus[dm_plus <= (l.shift() - l).clip(lower=0)]  = 0
    dm_minus[dm_minus <= (h - h.shift()).clip(lower=0)] = 0
    di_plus  = 100 * dm_plus.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-10)
    di_minus = 100 * dm_minus.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-10)
    return di_plus, di_minus

def _stoch(h, l, c, k=14, d=3):
    llo = l.rolling(k).min()
    hhi = h.rolling(k).max()
    raw = 100 * (c - llo) / (hhi - llo + 1e-10)
    return raw.rolling(d).mean()

def _mfi(h, l, c, v, n=14):
    tp  = (h + l + c) / 3
    mf  = tp * v
    pos = mf.where(tp > tp.shift(), 0)
    neg = mf.where(tp < tp.shift(), 0)
    mfr = pos.rolling(n).sum() / (neg.rolling(n).sum() + 1e-10)
    return 100 - 100 / (1 + mfr)

print("Computing 1h indicators …")
ind1h = pd.DataFrame(index=c1h.index)
ind1h["rsi_14_1h"]   = _rsi(cl, 14)
ind1h["cci_20_1h"]   = _cci(hi, lo, cl, 20)
ind1h["macd_hist_1h"]= _macd_hist(cl)
ind1h["di_plus_1h"], ind1h["di_minus_1h"] = _di(hi, lo, cl, 14)
ind1h["mfi_14_1h"]   = _mfi(hi, lo, cl, vol, 14)

print("Computing 4h stoch_k …")
c4h = c1h.resample("4h", origin="start_day").agg(
    {"high": "max", "low": "min", "close": "last"}
).dropna()
stoch4h = _stoch(c4h["high"], c4h["low"], c4h["close"], 14, 3)
stoch4h.name = "stoch_k_4h"
stoch4h_1h = stoch4h.reindex(c1h.index, method="ffill")  # forward-fill to 1h bars
ind1h["stoch_k_4h"] = stoch4h_1h

ind1h.index.name = "bar_ts"
print(f"Indicator matrix: {len(ind1h):,} rows × {len(ind1h.columns)} cols")

# ---------------------------------------------------------------------------
# 2. Load archive and join indicators
# ---------------------------------------------------------------------------
print("\nLoading scan archive …")
arc = pd.read_csv(ARCHIVE_CSV, low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, format="mixed")
arc = arc[arc["resolved_yes"].notna() & arc["logged_at"].notna()].copy()
arc["resolved_yes"] = arc["resolved_yes"].astype(float)
arc = arc[arc["resolved_yes"].isin([0.0, 1.0])].copy()
arc["resolved_yes"] = arc["resolved_yes"].astype(int)
arc["bar_ts"] = arc["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
arc = arc.sort_values("logged_at").reset_index(drop=True)

arc = arc.merge(ind1h.reset_index(), on="bar_ts", how="left")
print(f"Archive after join: {len(arc):,} rows")

# Coverage check on new cols
new_cols = list(ind1h.columns)
print("\nNew indicator coverage:")
for c in new_cols:
    cov = pd.to_numeric(arc[c], errors="coerce").notna().mean()
    print(f"  {c:<20} {cov:.1%}")

# ---------------------------------------------------------------------------
# 3. Build feature matrix
# ---------------------------------------------------------------------------
EXISTING = [
    "composite_p_up", "composite_trend", "composite_rev",
    "ema_stack_bias", "ema_stretch_score",
    "vwap_stretch_score", "vwap_distance_pct",
    "stoch_k", "chg_30m", "chg_10m", "chg_5m",
    "bp_5m", "body_15m", "dir_15m",
    "vol_score", "vpin_score", "obi_score",
    "confirmation_score", "no_score",
    "funding_bias", "vol_eff",
    "adx_1h", "rvol_1h", "squeeze_1h",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
]
FROM_ARCHIVE = ["offset_pct", "pm_drift_5m", "tau_minutes"]
BACKFILLED   = new_cols  # rsi_14_1h, cci_20_1h, macd_hist_1h, di_plus_1h, di_minus_1h, stoch_k_4h, mfi_14_1h

ALL_FEATURES = EXISTING + FROM_ARCHIVE + BACKFILLED
print(f"\nTotal features: {len(ALL_FEATURES)} ({len(EXISTING)} existing + {len(FROM_ARCHIVE)} archive + {len(BACKFILLED)} backfilled)")

for col in ALL_FEATURES:
    arc[col] = pd.to_numeric(arc.get(col, float("nan")), errors="coerce")

X_all = arc[ALL_FEATURES].fillna(0).values.astype(float)
y_all = arc["resolved_yes"].values

# ---------------------------------------------------------------------------
# 4. Time-ordered 70/15/15 split
# ---------------------------------------------------------------------------
n = len(arc)
i_val  = int(n * 0.70)
i_test = int(n * 0.85)

X_tr, y_tr = X_all[:i_val],        y_all[:i_val]
X_va, y_va = X_all[i_val:i_test],  y_all[i_val:i_test]
X_te, y_te = X_all[i_test:],       y_all[i_test:]

print(f"\nSplit: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")
print(f"  WR train={y_tr.mean():.3f}  val={y_va.mean():.3f}  test={y_te.mean():.3f}")

# ---------------------------------------------------------------------------
# 5. Train LightGBM
# ---------------------------------------------------------------------------
print("\nTraining LightGBM …")
clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)
clf.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    feature_name=ALL_FEATURES,
    callbacks=[],
)

auc_tr = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:,1])
auc_va = roc_auc_score(y_va, clf.predict_proba(X_va)[:,1])
auc_te = roc_auc_score(y_te, clf.predict_proba(X_te)[:,1])
print(f"\nAUC — train={auc_tr:.4f}  val={auc_va:.4f}  test={auc_te:.4f}")

# ---------------------------------------------------------------------------
# 6. Platt calibration on val set
# ---------------------------------------------------------------------------
p_raw_va = clf.predict_proba(X_va)[:,1]
logits_va = np.log(np.clip(p_raw_va, 1e-6, 1-1e-6) / (1 - np.clip(p_raw_va, 1e-6, 1-1e-6)))
platt = LogisticRegression(random_state=42)
platt.fit(logits_va.reshape(-1,1), y_va)
print(f"Platt — coef={platt.coef_[0][0]:.4f}  intercept={platt.intercept_[0]:.4f}")

# Evaluate calibrated model on test
p_raw_te  = clf.predict_proba(X_te)[:,1]
logits_te = np.log(np.clip(p_raw_te, 1e-6, 1-1e-6) / (1 - np.clip(p_raw_te, 1e-6, 1-1e-6)))
p_cal_te  = platt.predict_proba(logits_te.reshape(-1,1))[:,1]
brier_raw = brier_score_loss(y_te, p_raw_te)
brier_cal = brier_score_loss(y_te, p_cal_te)
print(f"Test Brier — raw={brier_raw:.4f}  calibrated={brier_cal:.4f}")
print(f"Test p_cal mean={p_cal_te.mean():.4f}  actual WR={y_te.mean():.4f}")

print("\nCalibration bins (test):")
for lo, hi in [(0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]:
    m = (p_cal_te >= lo) & (p_cal_te < hi)
    if m.sum() > 50:
        print(f"  p_cal=[{lo:.1f},{hi:.1f}): n={m.sum():,}  pred={p_cal_te[m].mean():.3f}  actual={y_te[m].mean():.3f}")

# ---------------------------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------------------------
print("\nTop-20 feature importances (gain):")
fi = sorted(zip(ALL_FEATURES, clf.feature_importances_), key=lambda x: -x[1])
for name, imp in fi[:20]:
    tag = " [NEW]" if name in FROM_ARCHIVE + BACKFILLED else ""
    print(f"  {name:<28} {imp:>8.1f}{tag}")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
pkg = {
    "clf":      clf,
    "platt":    platt,
    "features": ALL_FEATURES,
    "auc_tr":   auc_tr,
    "auc_va":   auc_va,
    "auc_te":   auc_te,
}
# Backup old model if not already backed up
if not BACKUP_PATH.exists():
    shutil.copy(MODEL_PATH, BACKUP_PATH)
    print(f"\nBackup: {BACKUP_PATH.name}")

MODEL_PATH.write_bytes(pickle.dumps(pkg))
print(f"Saved: {MODEL_PATH}")
print("\nDone.")
