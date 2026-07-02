#!/usr/bin/env python3
"""
train_btc_lgbm.py — Train a gradient-boosted tree model on the BTC paper trade archive.

Uses sklearn HistGradientBoostingClassifier (same algorithm as LightGBM / XGBoost family).
Input features: all signals already logged to paper_trades.csv.
Target: would_win (did this specific trade resolve as a win?).

This is a "meta-model" layer: the existing pipeline computes signals, we learn
which signal combinations actually predict outcomes in our specific trade universe.

Outputs:
  reform_results/btc_lgbm.pkl  — calibrated pipeline for shadow mode
  Prints: AUC, calibration, feature importance, P&L comparison on test set

Shadow integration: load btc_lgbm.pkl in paper_trade_runner.py to log p_gbdt
alongside p_yes_model without changing trade logic. Compare for 2-4 weeks before
deciding whether to blend or replace.

Run: python3 train_btc_lgbm.py
"""

import glob
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
try:
    import lightgbm as lgb
    _USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _USE_LGBM = False

warnings.filterwarnings("ignore")

RESULTS_DIR  = Path(__file__).parent / "results"
ARCHIVE_PATH = RESULTS_DIR / "btc_scan_archive.csv"
OUT_DIR      = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH   = OUT_DIR / "btc_lgbm.pkl"

# Time-based split: last 20% of resolved trades = test, prior 10% = val
VAL_FRAC  = 0.10
TEST_FRAC = 0.20

# Features available in paper_trades.csv — HistGBT handles NaN natively.
# Target is resolved_yes (P(YES resolves)), not would_win, so p_gbdt is
# directly interpretable as p_yes for both YES and NO side edge calculations:
#   YES edge = p_gbdt - pm
#   NO edge  = pm - p_gbdt
FEATURES = [
    # Composite model signals
    "composite_p_up",       # calibrated directional probability
    "composite_trend",      # trend vote count
    "composite_rev",        # reversion vote count

    # EMA / VWAP
    "ema_stack_bias",       # EMA stack alignment (-1/0/+1)
    "ema_stretch_score",    # EMA stretch score
    "vwap_stretch_score",   # VWAP distance bucket (-2..+2)
    "vwap_distance_pct",    # raw VWAP distance %

    # Momentum / microstructure
    "stoch_k",              # 14-period stochastic %K
    "chg_30m",              # 30m price change %
    "chg_10m",              # 10m price change %
    "chg_5m",               # 5m price change %
    "bp_5m",                # 5m buying pressure
    "body_15m",             # 15m candle body ratio
    "dir_15m",              # 15m bar direction (+1/-1)

    # Flow / sentiment
    "vol_score",            # volume regime (+1/-1/0)
    "vpin_score",           # VPIN (informed flow)
    "obi_score",            # order book imbalance score
    "confirmation_score",   # composite confirmation score
    "no_score",             # NO-side confirmation score
    "funding_bias",         # funding rate bias

    # Vol / regime
    "vol_eff",              # effective vol used in p_model
    "adx_1h",               # 1h ADX (trend strength)
    "rvol_1h",              # 1h relative volume
    "squeeze_1h",           # 1h BB/KC squeeze flag

    # Coinalyze liquidation + positioning (backfilled from Apr 15 onward)
    "liq_score",            # composite -2..+2: +1/2 = squeeze, -1/-2 = cascade
    "liq_bias",             # (short_liqs - long_liqs) / total_liqs
    "ls_long_pct",          # % of perp positions long
    "oi_chg_pct",           # OI % change per 1h bar

    # OU mean reversion (24h rolling AR(1) on log-prices; live from 2026-06-03)
    # Sim IC: Spearman +0.153 on 57k rows. Reversal at |tau_drift|>0.003 — LGBM handles non-linearity.
    "ou_z_score",           # (log(spot) - ou_mean) / ou_sigma; +ve = extended up, -ve = extended down
    "ou_halflife_min",      # reversion half-life in minutes; < tau = reversion within contract window
    "ou_tau_drift",         # E[log_return over tau] under OU; key τ-aware direction feature

    # Semi-Markov vol-regime (shadow from 2026-06-03; no gate effect on these features)
    "hmm_vol_state",        # hard Viterbi rank: 0=R0 low-vol, 1=R1 high-vol
    "hmm_r1_prob",          # soft posterior P(R1|data) 0-1
    "hmm_time_in_state",    # sojourn depth in bars: early=1-3, mid=4-15, deep=16+
]
FEATURES = list(dict.fromkeys(FEATURES))

SEP  = "=" * 72
SEP2 = "-" * 72


# ── Data loading ──────────────────────────────────────────────────────────────

def load_archive() -> pd.DataFrame:
    # Load btc_scan_archive.csv — all scanned contracts, unbiased (not just executed trades).
    if not ARCHIVE_PATH.exists():
        raise RuntimeError(f"Expected {ARCHIVE_PATH} — not found")
    df = pd.read_csv(ARCHIVE_PATH, low_memory=False)

    # Keep only resolved BTC rows
    df = df[df["resolved_yes"].notna()].copy()
    df = df[df["contract_ticker"].str.startswith("KXBTC", na=False)].copy()
    df = df.drop_duplicates(subset=["contract_ticker", "logged_at"]).copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True)
    df = df.sort_values("logged_at").reset_index(drop=True)

    num_cols = [
        "offset_pct", "p_market", "tau_minutes",
        "composite_p_up", "composite_trend", "composite_rev",
        "ema_stack_bias", "ema_stretch_score", "stoch_k",
        "vwap_stretch_score", "vwap_distance_pct", "vol_score",
        "vpin_score", "obi_score", "confirmation_score", "no_score",
        "funding_bias", "chg_30m", "chg_10m", "chg_5m",
        "bp_5m", "body_15m", "dir_15m", "vol_eff",
        "adx_1h", "rvol_1h", "squeeze_1h",
        "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
        "resolved_yes",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _stats(sub: pd.DataFrame, label: str = ""):
    n = len(sub)
    if n == 0:
        return
    wr_yes = sub["resolved_yes"].mean()   # P(YES resolves)
    pm     = sub["p_market"].mean()
    pgbdt  = sub["p_gbdt"].mean() if "p_gbdt" in sub.columns else float("nan")
    flag = " ★" if abs(wr_yes - pm) > 0.05 and n >= 15 else ""
    print(f"  {label:<42}  n={n:>4}  p_yes_actual={wr_yes:.1%}  "
          f"pm={pm:.1%}  p_gbdt={pgbdt:.1%}{flag}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    n = len(df)
    n_test = max(int(n * TEST_FRAC), 30)
    n_val  = max(int(n * VAL_FRAC), 20)
    n_train = n - n_val - n_test

    tr = df.iloc[:n_train]
    va = df.iloc[n_train:n_train + n_val]
    te = df.iloc[n_train + n_val:]

    print(f"\n  Time split:")
    print(f"    Train: n={len(tr):>4}  {tr['logged_at'].min().date()} → {tr['logged_at'].max().date()}")
    print(f"    Val:   n={len(va):>4}  {va['logged_at'].min().date()} → {va['logged_at'].max().date()}")
    print(f"    Test:  n={len(te):>4}  {te['logged_at'].min().date()} → {te['logged_at'].max().date()}")

    # Only use features that exist in the dataframe
    feats = [f for f in FEATURES if f in df.columns]
    print(f"\n  Features used: {len(feats)}")
    for f in feats:
        n_notnull = df[f].notna().sum()
        print(f"    {f:<30}  {n_notnull:>4}/{n} non-null ({n_notnull/n:.0%})")

    # Target: resolved_yes (P(YES resolves)) — consistent p_yes for both sides.
    # YES edge = p_gbdt - pm;  NO edge = pm - p_gbdt  (same formula, no inversion needed)
    X_tr = tr[feats].values
    y_tr = tr["resolved_yes"].values.astype(int)
    X_va = va[feats].values
    y_va = va["resolved_yes"].values.astype(int)
    X_te = te[feats].values
    y_te = te["resolved_yes"].values.astype(int)

    # Recency weighting: most-recent training trades weighted 3× more than oldest.
    t_vals = np.array([t.timestamp() for t in tr["logged_at"]])
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min if t_max > t_min else 1.0
    sample_weights = np.exp(1.5 * (t_vals - t_min) / t_range)

    if _USE_LGBM:
        print(f"\n  Training LightGBM...")
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=12,
            min_child_samples=25,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        print(f"\n  Training HistGradientBoostingClassifier (LightGBM not available)...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.02, max_depth=3,
            min_samples_leaf=25, max_leaf_nodes=12,
            l2_regularization=5.0, early_stopping=False, random_state=42,
        )
    clf.fit(X_tr, y_tr, sample_weight=sample_weights)

    p_va_raw = clf.predict_proba(X_va)[:, 1]
    p_tr_raw = clf.predict_proba(X_tr)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr_raw)
    auc_va = roc_auc_score(y_va, p_va_raw)
    print(f"  Train AUC: {auc_tr:.4f}  Val AUC: {auc_va:.4f}  "
          f"(gap: {auc_tr - auc_va:+.4f})")

    # Platt scaling (logistic regression on logit of raw scores) — more robust than
    # isotonic for small val windows because it has only 2 params and doesn't memorise
    # the val distribution.
    logit_va = np.log(np.clip(p_va_raw, 1e-6, 1 - 1e-6) /
                      (1 - np.clip(p_va_raw, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt.fit(logit_va.reshape(-1, 1), y_va)
    p_va_cal = np.clip(platt.predict_proba(logit_va.reshape(-1, 1))[:, 1], 1e-4, 1 - 1e-4)
    auc_va_cal = roc_auc_score(y_va, p_va_cal)
    ll_va_cal  = log_loss(y_va, p_va_cal, labels=[0, 1])
    print(f"  Val AUC post-Platt: {auc_va_cal:.4f}  log_loss: {ll_va_cal:.4f}")

    # Test performance
    p_te_raw   = clf.predict_proba(X_te)[:, 1]
    logit_te   = np.log(np.clip(p_te_raw, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(p_te_raw, 1e-6, 1 - 1e-6)))
    p_te_cal   = np.clip(platt.predict_proba(logit_te.reshape(-1, 1))[:, 1], 1e-4, 1 - 1e-4)
    auc_te     = roc_auc_score(y_te, p_te_cal)
    print(f"  Test AUC (calibrated): {auc_te:.4f}")

    pipe = {
        "clf":       clf,
        "platt":     platt,
        "features":  feats,
        "auc_tr":    auc_tr,
        "auc_va":    auc_va,
        "auc_te":    auc_te,
    }
    return pipe, te, p_te_cal


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(pipe: dict):
    print(f"\n  Feature importance:")
    clf   = pipe["clf"]
    feats = pipe["features"]
    try:
        imp_raw = clf.feature_importances_
        total = imp_raw.sum() or 1.0
        imp = pd.Series(imp_raw / total, index=feats).sort_values(ascending=False)
        for fname, val in imp.items():
            bar = "█" * int(val * 40)
            print(f"    {fname:<30}  {val:.3f}  {bar}")
    except (AttributeError, ValueError):
        print("  (feature_importances_ not available)")


# ── Calibration ───────────────────────────────────────────────────────────────

def print_calibration(y_true: np.ndarray, p_pred: np.ndarray, label: str = "GBDT"):
    print(f"\n  Calibration check ({label}) — predicted vs actual WR:")
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=8, strategy="quantile")
    print(f"  {'Pred bucket':<20}  {'Actual WR':>10}  {'Δ':>8}")
    for mp, fp in zip(mean_pred, frac_pos):
        flag = " ★" if abs(fp - mp) > 0.08 else ""
        print(f"  pred≈{mp:.2f}              {fp:.2f}      {fp-mp:>+.2f}{flag}")


# ── P&L comparison on test set ────────────────────────────────────────────────

def pnl_comparison(te: pd.DataFrame, p_gbdt: np.ndarray):
    print(f"\n{SEP2}")
    print("  GBDT calibration analysis on test set (all scanned contracts)")
    print(SEP2)

    te = te.copy()
    te["p_gbdt"] = p_gbdt

    print(f"\n  All test contracts:")
    _stats(te, "  All")

    # Calibration by p_gbdt bucket (tertiles)
    print(f"\n  By p_gbdt bucket:")
    lo = te["p_gbdt"].quantile(0.33)
    hi = te["p_gbdt"].quantile(0.67)
    te["gbdt_tier"] = pd.cut(te["p_gbdt"],
                              bins=[-np.inf, lo, hi, np.inf],
                              labels=["low", "mid", "high"])
    for q in ["low", "mid", "high"]:
        sub = te[te["gbdt_tier"] == q]
        _stats(sub, f"  p_gbdt tier={q} (≤{lo:.2f}/≤{hi:.2f}/>)")

    # Edge direction: p_gbdt vs p_market
    # p_gbdt > p_market → YES has edge; p_gbdt < p_market → NO has edge
    print(f"\n  GBDT edge direction vs actual P(YES):")
    pm = te["p_market"].fillna(0.5)
    yes_edge = te[te["p_gbdt"] > pm]
    no_edge  = te[te["p_gbdt"] <= pm]
    _stats(yes_edge, "  GBDT sees YES edge (p_gbdt > pm)")
    _stats(no_edge,  "  GBDT sees NO edge (p_gbdt <= pm)")

    # High-conviction GBDT
    if len(te) > 40:
        thresh_hi = te["p_gbdt"].quantile(0.80)
        thresh_lo = te["p_gbdt"].quantile(0.20)
        print(f"\n  Extreme predictions:")
        _stats(te[te["p_gbdt"] >= thresh_hi], f"  High conviction (p_gbdt ≥ {thresh_hi:.2f})")
        _stats(te[te["p_gbdt"] < thresh_lo],  f"  Low conviction  (p_gbdt < {thresh_lo:.2f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  BTC LGBM — Training on scan archive (all scanned contracts, unbiased)")
    print(SEP)

    print("\nLoading BTC scan archive...", end=" ", flush=True)
    df = load_archive()
    print(f"{len(df)} resolved contracts")
    print(f"  Date range: {df['logged_at'].min().date()} → {df['logged_at'].max().date()}")
    print(f"  P(YES resolves): {df['resolved_yes'].mean():.1%}  mean p_market: {df['p_market'].mean():.1%}")

    print(f"\n{SEP2}")
    print("  Training")
    print(SEP2)
    pipe, te, p_te_cal = train(df)

    print(f"\n{SEP2}")
    print("  Feature importance")
    print(SEP2)
    print_feature_importance(pipe)

    print(f"\n{SEP2}")
    print("  Calibration")
    print(SEP2)
    print_calibration(te["resolved_yes"].values.astype(int), p_te_cal, "GBDT test set")

    pnl_comparison(te, p_te_cal)

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n  Saved → {MODEL_PATH}")

    print(f"\n  Next step: restart paper_trade_runner.py — it will load btc_lgbm.pkl")
    print(f"  and log p_gbdt alongside p_yes_model for comparison.")
    print(f"  No trade logic changes until GBDT shows better live PnL.\n")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
