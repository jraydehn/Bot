#!/usr/bin/env python3
"""
train_btc_lgbm_15m.py — Train a LightGBM meta-model on the BTC 15m scan archive.

Input:  results/btc_scan_archive_15m.csv  (all evaluated contracts, not just trades)
Target: resolved_yes (did YES resolve?)
Output: reform_results/btc_lgbm_15m.pkl

Feature set: 42 signals including 15m wick/ATR/range/consec, 1h RSI/MACD,
vol ratios, VWAP, EMA, CoinGlass fear/greed, Coinalyze liq/OI.

Run: python3 train_btc_lgbm_15m.py  (wait for 2,000+ resolved rows in archive)
"""

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
OUT_DIR     = Path("reform_results")
OUT_DIR.mkdir(exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72

# ── Feature set ──────────────────────────────────────────────────────────────

# Signal-only features — no pricing leakage (no offset_pct, p_market, tau_minutes).
# Source: paper_trades_{asset}15m.csv — all evaluated contracts (trade + pass rows).
FEATURES = [
    # 15m / 5m microstructure
    "body_15m", "dir_15m",
    "bp_5m", "bp_1h",
    "stoch_k_5m", "stoch_k_15m", "stoch_k_1h",
    "chg_1m", "chg_5m", "chg_15m", "chg_1h",
    "vwap_dist",
    "vol_ratio",
    "ema_bias",
    # 1h context
    "consec_dir_1h", "dir_1h",
    "donchian_breakout_1h",
    "engulfing_1h",
    "stoch_cross_1h",
    "realized_vol_annual",
    # Composite signal
    "composite_p_up",
    # Coinalyze positioning
    "liq_score", "liq_bias", "oi_chg_pct",
]
FEATURES = list(dict.fromkeys(FEATURES))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(asset: str) -> pd.DataFrame:
    # Prefer scan_archive_15m (all contracts); fall back to paper_trades (executed+pass).
    scan_path  = RESULTS_DIR / f"{asset.lower()}_scan_archive_15m.csv"
    trade_path = RESULTS_DIR / f"paper_trades_{asset.lower()}15m.csv"
    path = scan_path if scan_path.exists() else trade_path
    if not path.exists():
        raise RuntimeError(f"Not found: {scan_path} or {trade_path}")
    df = pd.read_csv(path, low_memory=False)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    # Dedup by contract_ticker keeping first occurrence (scan archive logs every scan cycle)
    if "contract_ticker" in df.columns:
        df = df.drop_duplicates(subset=["contract_ticker"], keep="first").copy()
    for c in FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = float("nan")
    ts_col = "logged_at" if "logged_at" in df.columns else "decision_time"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.rename(columns={ts_col: "logged_at"})
    df = df.sort_values("logged_at").reset_index(drop=True)
    return df


def compute_pnl(df: pd.DataFrame) -> pd.Series:
    """Dollar P&L — uses bet_amount if present (paper trades), else returns zeros."""
    if "bet_amount" not in df.columns or "side" not in df.columns:
        return pd.Series(0.0, index=df.index)
    ba = pd.to_numeric(df["bet_amount"], errors="coerce").fillna(0)
    pm = pd.to_numeric(df["p_market"], errors="coerce")
    side = df["side"].str.strip().str.lower()
    res = pd.to_numeric(df["resolved_yes"], errors="coerce")
    return np.where(
        side == "yes",
        np.where(res == 1, ba * (1 / pm - 1), -ba),
        np.where(res == 0, ba * (1 / (1 - pm) - 1), -ba),
    )


# ── Train ─────────────────────────────────────────────────────────────────────

def train(asset: str = "BTC"):
    output_path = OUT_DIR / f"{asset.lower()}_lgbm_15m.pkl"
    print(SEP)
    print(f"  {asset} 15m LightGBM — training on all evaluated contracts (signal-only features)")
    print(SEP)

    df = load_data(asset)
    print(f"Loaded {len(df)} resolved rows  "
          f"({df['logged_at'].iloc[0].date()} → {df['logged_at'].iloc[-1].date()})")

    # Chronological 60 / 20 / 20 split
    n = len(df)
    i_val  = int(n * 0.60)
    i_test = int(n * 0.80)
    df_tr  = df.iloc[:i_val].copy()
    df_val = df.iloc[i_val:i_test].copy()
    df_te  = df.iloc[i_test:].copy()
    print(f"Split — train: {len(df_tr)}  val: {len(df_val)}  test: {len(df_te)}")
    print()

    feats = [f for f in FEATURES if f in df.columns]
    X_tr  = df_tr[feats].values.astype(float)
    y_tr  = df_tr["resolved_yes"].values.astype(int)
    X_val = df_val[feats].values.astype(float)
    y_val = df_val["resolved_yes"].values.astype(int)
    X_te  = df_te[feats].values.astype(float)
    y_te  = df_te["resolved_yes"].values.astype(int)

    # LightGBM via sklearn HistGradientBoostingClassifier (handles NaN natively)
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        _using_lgbm = True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
        )
        _using_lgbm = False

    clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_val)[:, 1]
    p_te  = clf.predict_proba(X_te)[:, 1]

    auc_val = roc_auc_score(y_val, p_val)
    auc_te  = roc_auc_score(y_te, p_te)
    print(f"Val AUC:  {auc_val:.4f}   Test AUC: {auc_te:.4f}")
    print()

    # Platt calibration (fit on val, evaluate on test)
    logits_val = np.log(np.clip(p_val, 1e-6, 1-1e-6) / np.clip(1-p_val, 1e-6, 1-1e-6))
    logits_te  = np.log(np.clip(p_te,  1e-6, 1-1e-6) / np.clip(1-p_te,  1e-6, 1-1e-6))
    platt = LogisticRegression(C=1e4)
    platt.fit(logits_val.reshape(-1, 1), y_val)
    p_cal_te = platt.predict_proba(logits_te.reshape(-1, 1))[:, 1]
    auc_cal_te = roc_auc_score(y_te, p_cal_te)
    print(f"Calibrated test AUC: {auc_cal_te:.4f}")

    # Calibration check by bucket
    print(SEP2)
    print("Calibration (test set, calibrated):")
    buckets = np.arange(0, 1.01, 0.1)
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (p_cal_te >= lo) & (p_cal_te < hi)
        if mask.sum() < 3:
            continue
        actual = y_te[mask].mean()
        pred   = p_cal_te[mask].mean()
        print(f"  pred [{lo:.1f},{hi:.1f}): n={mask.sum():3d}  pred={pred:.2f}  actual={actual:.2f}  Δ={actual-pred:+.2f}")
    print()

    # Feature importance
    if _using_lgbm:
        imp = clf.feature_importances_
        total = imp.sum() or 1
        ranked = sorted(zip(feats, imp), key=lambda x: -x[1])
        print("Feature importances (gain):")
        for feat, score in ranked[:15]:
            print(f"  {feat:<28} {score/total:.1%}")
        print()

    # P&L comparison on test set
    print(SEP2)
    print("P&L comparison on test set:")
    df_te = df_te.copy()
    df_te["p_gbdt_cal"] = p_cal_te
    df_te["pnl"] = compute_pnl(df_te)

    pm_te = df_te["p_market"].values
    # For scan archive: model predicts p(YES resolves); GBDT edge = p_cal - p_market
    gbdt_edge = p_cal_te - pm_te

    mask_agree    = gbdt_edge >= 0
    mask_disagree = gbdt_edge < -0.03

    pnl_all      = df_te["pnl"].sum()
    pnl_agree    = df_te.loc[mask_agree,   "pnl"].sum()
    pnl_disagree = df_te.loc[mask_disagree,"pnl"].sum()
    pnl_neutral  = df_te.loc[~mask_agree & ~mask_disagree, "pnl"].sum()

    print(f"  All test rows:          n={len(df_te):3d}  PnL=${pnl_all:+.0f}")
    print(f"  GBDT bullish (≥0):      n={mask_agree.sum():3d}  PnL=${pnl_agree:+.0f}")
    print(f"  GBDT bearish (<-0.03):  n={mask_disagree.sum():3d}  PnL=${pnl_disagree:+.0f}")
    print(f"  GBDT neutral (-0.03,0): n={(~mask_agree & ~mask_disagree).sum():3d}  PnL=${pnl_neutral:+.0f}")
    print()

    # Calibration: YES resolution rate vs GBDT prediction
    print("YES resolution rate by GBDT bucket:")
    for lo, hi in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1)):
        mask = (p_cal_te >= lo) & (p_cal_te < hi)
        if mask.sum() < 5:
            continue
        actual = y_te[mask].mean()
        print(f"  [{lo:.1f},{hi:.1f}): n={mask.sum():3d}  YES_rate={actual:.2f}  pred={p_cal_te[mask].mean():.2f}")
    print()

    # Save
    pipe = {
        "features":   feats,
        "clf":        clf,
        "platt":      platt,
        "auc_val":    auc_val,
        "auc_te":     auc_te,
        "auc_te_cal": auc_cal_te,
        "n_train":    len(df_tr),
        "n_val":      len(df_val),
        "n_test":     len(df_te),
    }
    with open(output_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved → {output_path}")
    print(SEP)


if __name__ == "__main__":
    import sys
    assets = sys.argv[1:] or ["BTC", "ETH", "SOL"]
    for a in assets:
        try:
            train(a.upper())
        except RuntimeError as e:
            print(f"  Skipping {a}: {e}")
