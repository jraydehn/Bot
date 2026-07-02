#!/usr/bin/env python3
"""
calibrate_k_pup_v2_hourly.py — Calibrate k for hourly BTC p_up_v2 drift reform.

Tests Model A (baseline) plus four alternative vol drift factors:
  A: z_drift = Φ⁻¹(p_up_v2) × k × √(τ/60)                         [no vol]
  D: z_drift = … × vol_trend       sigma_24h / sigma_72h             [vol acceleration]
  E: z_drift = … × term_spread     sigma_6h / sigma_168h             [short/long realized spread]
  F: z_drift = … × momentum_conf   abs(net_6h_ret) / sigma_6h        [momentum clarity]
  G: z_drift = … × bb_expansion    bb_width_24h / bb_width_72h_mean  [BB expansion ratio]

Each factor tested in both forward (×) and inverse (/) directions.
All factors clipped to [0.3, 3.0].

Uses hourly-appropriate tau values [30, 45, 60, 75, 90] min.
Settlement = close[t+1] (next 1h bar close).
"""

import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
MODEL_PATH = ROOT / "reform_results" / "btc_p_up_v2.pkl"
CACHE_PATH = Path("/tmp/calib_feat_dataset.pkl")

sys.path.insert(0, str(ROOT))
from train_btc_p_up_v2 import build_dataset

STRIKE_N         = [-7, -5, -3, -2, -1, 0, 1, 2, 3]
TAU_MINUTES_LIST = [30, 45, 60, 75, 90]
P_MARKET_BAND    = (0.20, 0.80)
FACTOR_CLIP      = (0.3, 3.0)

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


def build_factor_series(df1h: pd.DataFrame) -> pd.DataFrame:
    """Compute all four vol factors from 1h OHLCV. Returns DataFrame aligned to df1h.index."""
    c = df1h["close"].astype(float)
    h = df1h["high"].astype(float)
    l = df1h["low"].astype(float)
    log_ret = np.log(c / c.shift(1))

    # Realized vol windows
    sigma_6h   = log_ret.rolling(6).std()
    sigma_24h  = log_ret.rolling(24).std()
    sigma_72h  = log_ret.rolling(72).std()
    sigma_168h = log_ret.rolling(168).std()

    # D: vol acceleration — short-run vs medium-run realized vol
    vol_trend = (sigma_24h / sigma_72h.replace(0, np.nan)).clip(*FACTOR_CLIP)

    # E: term spread — very-short vs long realized vol (proxy for realized/implied term structure)
    term_spread = (sigma_6h / sigma_168h.replace(0, np.nan)).clip(*FACTOR_CLIP)

    # F: momentum clarity — |net 6h return| / sigma_6h (z-score of recent trend strength)
    net_6h = (c / c.shift(6) - 1).abs()
    momentum_conf = (net_6h / (sigma_6h * math.sqrt(6)).replace(0, np.nan)).clip(*FACTOR_CLIP)

    # G: Bollinger Band expansion — current BB width vs 72h mean BB width
    bb_mid    = c.rolling(20).mean()
    bb_std    = c.rolling(20).std()
    bb_width  = (2 * bb_std / bb_mid.replace(0, np.nan))          # normalized width
    bb_expansion = (bb_width / bb_width.rolling(72).mean().replace(0, np.nan)).clip(*FACTOR_CLIP)

    return pd.DataFrame({
        "D_vol_trend":     vol_trend,
        "E_term_spread":   term_spread,
        "F_momentum_conf": momentum_conf,
        "G_bb_expansion":  bb_expansion,
    }, index=df1h.index)


def sigma_tau_fn(vol_per_hour: float, tau_min: float) -> float:
    if math.isnan(vol_per_hour) or vol_per_hour <= 0:
        return float("nan")
    return vol_per_hour * math.sqrt(tau_min / 60.0)


def log_loss_model(k: float, records: list, factor_idx: int, inverse: bool) -> float:
    """
    factor_idx: index into the per-record factor tuple (0=D,1=E,2=F,3=G); -1 = no vol (A)
    inverse: divide by factor instead of multiply
    """
    eps = 1e-7
    ll  = 0.0
    for z, p_up, factors, y, tau_min in records:
        tau_scale = math.sqrt(tau_min / 60.0)
        ppf_val   = norm.ppf(float(np.clip(p_up, 0.01, 0.99)))
        if factor_idx < 0:
            z_drift = ppf_val * k * tau_scale
        else:
            fv = factors[factor_idx]
            if inverse:
                z_drift = ppf_val * k * tau_scale / fv if fv > 0 else ppf_val * k * tau_scale
            else:
                z_drift = ppf_val * k * tau_scale * fv
        z_adj = z - z_drift
        p = float(np.clip(1.0 - norm.cdf(z_adj), eps, 1 - eps))
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -ll / len(records)


def calibrate_one(records, label, factor_idx, inverse):
    res = minimize_scalar(log_loss_model, bounds=(-0.5, 4.0), method="bounded",
                          args=(records, factor_idx, inverse))
    ll_base = log_loss_model(0.0, records, factor_idx, inverse)
    return res.x, res.fun, ll_base


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return
    print("Loading p_up_v2 model...")
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    clf = pipe["clf"]

    if CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}...")
        df = pd.read_pickle(CACHE_PATH)
    else:
        print("Building feature dataset (takes a few minutes)...")
        df = build_dataset()
        df.to_pickle(CACHE_PATH)

    print(f"Dataset: {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")

    print("Running p_up_v2 inference...")
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    X       = df[FEATURES].values.astype(float)
    p_up_v2 = clf.predict_proba(X)[:, 1]
    df["p_up_v2"] = p_up_v2

    data_dir = ROOT / "data"
    f1h = sorted(data_dir.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h_full = pd.read_parquet(f1h)
    df1h_full.index = pd.to_datetime(df1h_full.index, utc=True)
    df1h_full = df1h_full.sort_index()

    print("Computing factor series from 1h data...")
    log_ret_1h  = np.log(df1h_full["close"] / df1h_full["close"].shift(1))
    sigma_24h   = log_ret_1h.rolling(24).std()
    factors_df  = build_factor_series(df1h_full)
    settlement  = df1h_full["close"].shift(-1)

    sig_at    = sigma_24h.reindex(df.index)
    set_at    = settlement.reindex(df.index)
    fac_at    = factors_df.reindex(df.index)
    fac_cols  = list(factors_df.columns)

    print("\nBuilding calibration records...")
    records = []
    skipped = {"no_settle": 0, "no_vol": 0, "no_fac": 0, "band": 0}

    for ts, row in df.iterrows():
        p_up = float(row["p_up_v2"])
        if math.isnan(p_up):
            continue
        spot     = float(row["close"])
        settle   = float(set_at[ts]) if ts in set_at.index and not math.isnan(set_at[ts]) else float("nan")
        sig      = float(sig_at[ts]) if ts in sig_at.index and not math.isnan(sig_at[ts]) else float("nan")
        fac_row  = fac_at.loc[ts] if ts in fac_at.index else None

        if math.isnan(settle):  skipped["no_settle"] += 1; continue
        if math.isnan(sig) or sig <= 0: skipped["no_vol"] += 1; continue
        if fac_row is None or fac_row.isna().any(): skipped["no_fac"] += 1; continue

        factors = tuple(float(fac_row[c]) for c in fac_cols)
        spot_r  = round(spot / 100) * 100

        for tau_min in TAU_MINUTES_LIST:
            st = sigma_tau_fn(sig, tau_min)
            if math.isnan(st) or st <= 0:
                continue
            for n in STRIKE_N:
                strike = spot_r + n * 100 + 99.99
                if strike <= 0:
                    continue
                z_strike  = math.log(strike / spot) / st
                p_lognorm = float(1.0 - norm.cdf(z_strike))
                if not (P_MARKET_BAND[0] <= p_lognorm <= P_MARKET_BAND[1]):
                    skipped["band"] += 1
                    continue
                records.append((z_strike, p_up, factors, int(settle > strike), tau_min))

    print(f"  Records: {len(records):,}  YES-rate: {np.mean([r[3] for r in records]):.1%}")
    for k, v in skipped.items():
        print(f"  Skipped ({k}): {v:,}")
    if not records:
        print("ERROR: no records"); return

    # Factor stats
    print("\nFactor stats (mean / std / p5 / p95):")
    for i, col in enumerate(fac_cols):
        vals = np.array([r[2][i] for r in records])
        print(f"  {col:<22} mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"p5={np.percentile(vals,5):.3f}  p95={np.percentile(vals,95):.3f}")

    # Calibrate all models
    models = [
        ("A  no vol",                   -1, False),
        ("D+ vol_trend    ×",            0, False),
        ("D- vol_trend    /",            0, True),
        ("E+ term_spread  ×",            1, False),
        ("E- term_spread  /",            1, True),
        ("F+ momentum_conf×",            2, False),
        ("F- momentum_conf/",            2, True),
        ("G+ bb_expansion ×",            3, False),
        ("G- bb_expansion /",            3, True),
    ]

    print("\n" + "=" * 70)
    print("RESULTS  (sorted by log-loss)")
    print("=" * 70)
    print(f"  {'Model':<24}  {'k':>6}  {'ll':>8}  {'Δ vs A':>8}")

    results = []
    k_a = ll_a = None
    for label, fidx, inv in models:
        k_opt, ll_opt, _ = calibrate_one(records, label, fidx, inv)
        results.append((label, k_opt, ll_opt))
        if label.startswith("A"):
            k_a, ll_a = k_opt, ll_opt

    results.sort(key=lambda x: x[2])
    for label, k_opt, ll_opt in results:
        delta = ll_opt - ll_a
        marker = "  ★ WINNER" if ll_opt == min(r[2] for r in results) else ""
        print(f"  {label:<24}  k={k_opt:5.3f}  ll={ll_opt:.5f}  Δ={delta:+.5f}{marker}")

    winner = results[0]
    print(f"\nWinner: {winner[0]}  k={winner[1]:.4f}  ll={winner[2]:.5f}")
    if winner[2] < ll_a:
        delta = winner[2] - ll_a
        print(f"Beats no-vol by {abs(delta):.5f} ({abs(delta)/ll_a*100:.3f}%)")
    else:
        print("Model A (no vol) wins — no vol factor improves calibration.")

    print("\nz_drift magnitudes for winning model at p_up=0.75 / 0.25:")
    best_k = winner[1]
    for tau_ex in [30, 45, 60, 75, 90]:
        scale = math.sqrt(tau_ex / 60.0)
        hi = norm.ppf(0.75) * best_k * scale
        lo = norm.ppf(0.25) * best_k * scale
        print(f"  τ={tau_ex:2d}m  z_drift@p75={hi:+.3f}  z_drift@p25={lo:+.3f}")


if __name__ == "__main__":
    main()
