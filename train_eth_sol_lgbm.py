#!/usr/bin/env python3
"""
train_eth_sol_lgbm.py — Train LGBM shadow models for ETH and SOL.

Same architecture as train_btc_lgbm.py (LightGBM + Platt scaling).
Outputs:
  reform_results/eth_lgbm.pkl
  reform_results/sol_lgbm.pkl

Run: python3 train_eth_sol_lgbm.py
"""

import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
    _USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _USE_LGBM = False

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

VAL_FRAC  = 0.10
TEST_FRAC = 0.20

# Signal-only features — no pricing leakage (no offset_pct, p_market, tau_minutes, side_enc).
# Source: {asset}_scan_archive.csv — all scanned contracts, not just executed trades.
FEATURES = [
    "composite_p_up", "composite_trend", "composite_rev",
    "ema_stack_bias", "ema_stretch_score", "stoch_k",
    "vwap_stretch_score", "vwap_distance_pct",
    "vol_score", "vpin_score", "obi_score", "confirmation_score", "no_score",
    "funding_bias", "vol_eff",
    "chg_30m", "chg_10m", "chg_5m",
    "bp_5m", "body_15m", "dir_15m",
    "adx_1h", "rvol_1h", "squeeze_1h",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
]
FEATURES = list(dict.fromkeys(FEATURES))

SEP  = "=" * 72
SEP2 = "-" * 72


def load_archive(asset: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{asset.lower()}_scan_archive.csv"
    if not path.exists():
        raise RuntimeError(f"Scan archive not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["resolved_yes"].notna()].copy()
    df = df.drop_duplicates(subset=["contract_ticker", "logged_at"]).copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True)
    df = df.sort_values("logged_at").reset_index(drop=True)
    for col in FEATURES + ["resolved_yes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stats(sub, label=""):
    n = len(sub)
    if n == 0:
        return
    p_yes = sub["resolved_yes"].mean()
    pm    = sub["p_market"].mean() if "p_market" in sub.columns else float("nan")
    pgbdt = sub["p_gbdt"].mean() if "p_gbdt" in sub.columns else float("nan")
    flag = " ★" if abs(p_yes - pm) > 0.05 and n >= 15 else ""
    print(f"  {label:<42}  n={n:>4}  p_yes_actual={p_yes:.1%}  "
          f"pm={pm:.1%}  p_gbdt={pgbdt:.1%}{flag}")


def train(df: pd.DataFrame, asset: str) -> tuple:
    n = len(df)
    n_test  = max(int(n * TEST_FRAC), 20)
    n_val   = max(int(n * VAL_FRAC), 15)
    n_train = n - n_val - n_test

    tr = df.iloc[:n_train]
    va = df.iloc[n_train:n_train + n_val]
    te = df.iloc[n_train + n_val:]

    print(f"\n  Time split:")
    print(f"    Train: n={len(tr):>4}  {tr['logged_at'].min().date()} → {tr['logged_at'].max().date()}")
    print(f"    Val:   n={len(va):>4}  {va['logged_at'].min().date()} → {va['logged_at'].max().date()}")
    print(f"    Test:  n={len(te):>4}  {te['logged_at'].min().date()} → {te['logged_at'].max().date()}")

    feats = [f for f in FEATURES if f in df.columns]
    print(f"\n  Features used: {len(feats)}")
    for f in feats:
        nn = df[f].notna().sum()
        print(f"    {f:<30}  {nn:>4}/{n} ({nn/n:.0%})")

    X_tr = tr[feats].values
    y_tr = tr["resolved_yes"].values.astype(int)
    X_va = va[feats].values
    y_va = va["resolved_yes"].values.astype(int)
    X_te = te[feats].values
    y_te = te["resolved_yes"].values.astype(int)

    t_vals = np.array([t.timestamp() for t in tr["logged_at"]])
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min if t_max > t_min else 1.0
    sample_weights = np.exp(1.5 * (t_vals - t_min) / t_range)

    if _USE_LGBM:
        print(f"\n  Training LightGBM ({asset})...")
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=10,
            min_child_samples=20,
            reg_lambda=8.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        print(f"\n  Training HistGBT ({asset})...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.02, max_depth=3,
            min_samples_leaf=20, max_leaf_nodes=10,
            l2_regularization=8.0, early_stopping=False, random_state=42,
        )
    clf.fit(X_tr, y_tr, sample_weight=sample_weights)

    p_va_raw = clf.predict_proba(X_va)[:, 1]
    p_tr_raw = clf.predict_proba(X_tr)[:, 1]
    auc_tr   = roc_auc_score(y_tr, p_tr_raw)
    auc_va   = roc_auc_score(y_va, p_va_raw)
    print(f"  Train AUC: {auc_tr:.4f}  Val AUC: {auc_va:.4f}  (gap: {auc_tr-auc_va:+.4f})")

    logit_va = np.log(np.clip(p_va_raw, 1e-6, 1-1e-6) / (1 - np.clip(p_va_raw, 1e-6, 1-1e-6)))
    platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt.fit(logit_va.reshape(-1, 1), y_va)
    p_va_cal = np.clip(platt.predict_proba(logit_va.reshape(-1, 1))[:, 1], 1e-4, 1-1e-4)
    print(f"  Val AUC post-Platt: {roc_auc_score(y_va, p_va_cal):.4f}  "
          f"log_loss: {log_loss(y_va, p_va_cal, labels=[0,1]):.4f}")

    p_te_raw = clf.predict_proba(X_te)[:, 1]
    logit_te = np.log(np.clip(p_te_raw, 1e-6, 1-1e-6) / (1 - np.clip(p_te_raw, 1e-6, 1-1e-6)))
    p_te_cal = np.clip(platt.predict_proba(logit_te.reshape(-1, 1))[:, 1], 1e-4, 1-1e-4)
    auc_te   = roc_auc_score(y_te, p_te_cal)
    print(f"  Test AUC (calibrated): {auc_te:.4f}")

    # Feature importance
    print(f"\n  Feature importance:")
    try:
        imp = pd.Series(clf.feature_importances_ / (clf.feature_importances_.sum() or 1),
                        index=feats).sort_values(ascending=False)
        for fname, val in imp.items():
            print(f"    {fname:<30}  {val:.3f}  {'█' * int(val * 40)}")
    except AttributeError:
        pass

    # Calibration analysis on test set
    te_df = te.copy()
    te_df["p_gbdt"] = p_te_cal
    print(f"\n  Test set calibration:")
    _stats(te_df, "  All")
    lo = te_df["p_gbdt"].quantile(0.33)
    hi = te_df["p_gbdt"].quantile(0.67)
    te_df["tier"] = pd.cut(te_df["p_gbdt"], bins=[-np.inf, lo, hi, np.inf],
                            labels=["low", "mid", "high"])
    for tier in ["low", "mid", "high"]:
        _stats(te_df[te_df["tier"] == tier], f"  p_gbdt tier={tier}")

    if "p_market" in te_df.columns:
        pm = te_df["p_market"].fillna(0.5)
        yes_edge = te_df[te_df["p_gbdt"] > pm]
        no_edge  = te_df[te_df["p_gbdt"] <= pm]
        print(f"\n  Edge direction vs actual:")
        _stats(yes_edge, "  GBDT sees YES edge (p_gbdt > pm)")
        _stats(no_edge,  "  GBDT sees NO edge  (p_gbdt <= pm)")

    pipe = {"clf": clf, "platt": platt, "features": feats,
            "auc_tr": auc_tr, "auc_va": auc_va, "auc_te": auc_te}
    return pipe


def main():
    for asset, out_name in [("ETH", "eth_lgbm.pkl"), ("SOL", "sol_lgbm.pkl")]:
        print(f"\n{SEP}")
        print(f"  {asset} LGBM — Training on scan archive (all scanned contracts, unbiased)")
        print(SEP)

        try:
            df = load_archive(asset)
        except RuntimeError as e:
            print(f"  Skipping {asset}: {e}")
            continue

        print(f"  Loaded {len(df)} resolved contracts  "
              f"({df['logged_at'].min().date()} → {df['logged_at'].max().date()})")
        print(f"  P(YES resolves): {df['resolved_yes'].mean():.1%}  "
              f"mean p_market: {df['p_market'].mean():.1%}" if "p_market" in df.columns else "")

        pipe = train(df, asset)
        out_path = OUT_DIR / out_name
        with open(out_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"\n  Saved → {out_path}")

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()
