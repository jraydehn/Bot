#!/usr/bin/env python3
"""
calibrate_k_pup_v2.py — Calibrate the p_up_v2 drift factor k.

Simulates Kalshi-style BTC daily contracts on 2+ years of historical
Binance 1h data to find the empirically optimal k such that:

    z_drift       = Φ⁻¹(p_up_v2) × k × √(τ/60)
    p_model_yes   = 1 − Φ(z_strike − z_drift)

minimises log-loss against actual bar-close outcomes.

The √(τ/60) factor scales drift with time-remaining so that at τ=60 min
the full directional signal is applied, decaying to ~0 near expiry — matching
how drift scales in Brownian motion and fixing the horizon mismatch where
p_up_v2 (trained on next-1h-bar direction) was applied to 15–45 min contracts.

Contract structure mirrors live Kalshi BTCD observations:
  - Strikes on $100 grid: round(spot/100)×100 + n×100 + 99.99
  - n ∈ {-7,-5,-3,-2,-1,0,+1,+2,+3}  (empirical -$700/+$300 asymmetric range)
  - tau_minutes ∈ {15, 25, 35, 45}    (empirical mid-life range, median ~33 min)
  - Spot  = close[t]   (last observable close before bar t+1 opens)
  - Settlement = close[t+1]            (bar t+1 close = Kalshi settlement proxy)
  - Filter: log-normal p_yes ∈ [0.20, 0.80]  (only zone where drift matters)

Outputs k_yes and k_no with full log-loss grid and optimal values.
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODEL_PATH = ROOT / "reform_results" / "btc_p_up_v2.pkl"
CACHE_PATH = Path("/tmp/calib_feat_dataset.pkl")

sys.path.insert(0, str(ROOT))
from train_btc_p_up_v2 import build_dataset

# ── constants (matching live Kalshi observations) ─────────────────────────────
STRIKE_N         = [-7, -5, -3, -2, -1, 0, 1, 2, 3]   # × $100 from rounded spot
TAU_MINUTES_LIST = [15, 25, 35, 45]                     # mid-life tau samples
P_MARKET_BAND    = (0.20, 0.80)                         # "interesting" zone

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── sigma_tau helpers ─────────────────────────────────────────────────────────

def build_vol_series(df1h: pd.DataFrame) -> pd.Series:
    """24h rolling hourly log-return std (per-hour vol)."""
    log_ret = np.log(df1h["close"] / df1h["close"].shift(1))
    return log_ret.rolling(24).std()


def sigma_tau(vol_per_hour: float, tau_min: float) -> float:
    """Total vol to expiry: scale hourly vol to tau_minutes."""
    if math.isnan(vol_per_hour) or vol_per_hour <= 0:
        return float("nan")
    return vol_per_hour * math.sqrt(tau_min / 60.0)


# ── log-loss ─────────────────────────────────────────────────────────────────

def log_loss(k: float, records: list, side: str = "yes") -> float:
    eps = 1e-7
    ll  = 0.0
    for z, p_up, y, tau_min in records:
        tau_scale = math.sqrt(tau_min / 60.0)
        z_drift   = norm.ppf(float(np.clip(p_up, 0.01, 0.99))) * k * tau_scale
        z_adj     = z - z_drift
        if side == "yes":
            p = float(np.clip(1.0 - norm.cdf(z_adj), eps, 1 - eps))
        else:
            p = float(np.clip(norm.cdf(z_adj), eps, 1 - eps))
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -ll / len(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return
    print("Loading p_up_v2 model...")
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    clf = pipe["clf"]

    # ── build or load feature dataset ─────────────────────────────────────────
    if CACHE_PATH.exists():
        print(f"Loading cached feature dataset from {CACHE_PATH}...")
        df = pd.read_pickle(CACHE_PATH)
    else:
        print("Building feature dataset (this takes a few minutes)...")
        df = build_dataset()
        df.to_pickle(CACHE_PATH)
        print(f"Cached to {CACHE_PATH}")

    print(f"Dataset: {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")

    # ── run p_up_v2 inference ─────────────────────────────────────────────────
    print("Running p_up_v2 inference...")
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    X       = df[FEATURES].values.astype(float)
    p_up_v2 = clf.predict_proba(X)[:, 1]
    df["p_up_v2"] = p_up_v2

    print(f"\np_up_v2 distribution across {len(df):,} bars:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct:2d}: {np.percentile(p_up_v2, pct):.3f}")
    print(f"  mean={p_up_v2.mean():.3f}  std={p_up_v2.std():.3f}")

    # ── load 1h data for vol + settlement prices ──────────────────────────────
    data_dir = ROOT / "data"
    f1h = sorted(data_dir.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h_full = pd.read_parquet(f1h)
    df1h_full.index = pd.to_datetime(df1h_full.index, utc=True)
    df1h_full = df1h_full.sort_index()

    vol_ser = build_vol_series(df1h_full)

    # close[t+1] = settlement price for contracts observed at close[t]
    settlement_ser = df1h_full["close"].shift(-1).reindex(df.index)
    vol_at         = vol_ser.reindex(df.index)

    # ── build calibration records ─────────────────────────────────────────────
    print("\nBuilding calibration records...")
    records_yes = []
    records_no  = []

    skipped_no_settle = 0
    skipped_no_vol    = 0
    skipped_band      = 0

    for ts, row in df.iterrows():
        p_up = float(row["p_up_v2"])
        if math.isnan(p_up):
            continue

        spot       = float(row["close"])          # close[t] = observable spot
        settlement = float(settlement_ser[ts]) if ts in settlement_ser.index and not math.isnan(settlement_ser[ts]) else float("nan")
        vol_h      = float(vol_at[ts])             if ts in vol_at.index      else float("nan")

        if math.isnan(settlement):
            skipped_no_settle += 1
            continue
        if math.isnan(vol_h) or vol_h <= 0:
            skipped_no_vol += 1
            continue

        spot_rounded = round(spot / 100) * 100

        for tau_min in TAU_MINUTES_LIST:
            sig = sigma_tau(vol_h, tau_min)
            if math.isnan(sig) or sig <= 0:
                continue

            for n in STRIKE_N:
                strike = spot_rounded + n * 100 + 99.99

                if strike <= 0:
                    continue

                z_strike   = math.log(strike / spot) / sig
                p_lognorm  = float(1.0 - norm.cdf(z_strike))   # pure structural YES prob

                # Only keep the "interesting" zone
                lo, hi = P_MARKET_BAND
                if not (lo <= p_lognorm <= hi):
                    skipped_band += 1
                    continue

                y_yes = int(settlement > strike)
                records_yes.append((z_strike, p_up, y_yes, tau_min))
                records_no.append((z_strike, p_up, 1 - y_yes, tau_min))

    print(f"  Total calibration records: {len(records_yes):,}")
    print(f"  Skipped (no settlement): {skipped_no_settle:,}")
    print(f"  Skipped (no vol):        {skipped_no_vol:,}")
    print(f"  Skipped (outside band):  {skipped_band:,}")
    if not records_yes:
        print("ERROR: no records — check data paths")
        return

    yes_rate = np.mean([r[2] for r in records_yes])
    print(f"  YES rate in records: {yes_rate:.1%}")

    # ── p_up_v2 distribution in calibration set ───────────────────────────────
    pups = np.array([r[1] for r in records_yes])
    print(f"\n  p_up_v2 in calibration set (should have real variance):")
    print(f"  mean={pups.mean():.3f}  std={pups.std():.3f}  "
          f"p5={np.percentile(pups,5):.3f}  p95={np.percentile(pups,95):.3f}")

    # ── grid search k ─────────────────────────────────────────────────────────
    k_grid = [-0.30, -0.15, 0.0, 0.07, 0.15, 0.30, 0.50, 0.70, 1.00, 1.40, 2.00]

    print("\n" + "=" * 60)
    print("YES MODEL  (p_model_yes = 1 − Φ(z − Φ⁻¹(p_up)×k))")
    print("=" * 60)
    ll_base_yes = log_loss(0.0, records_yes, "yes")
    print(f"  k=0.00  (baseline)  log-loss={ll_base_yes:.5f}")
    for k in k_grid:
        if k == 0.0:
            continue
        ll = log_loss(k, records_yes, "yes")
        marker = " ◄ current k_yes (DRIFT_MULTIPLIER)" if abs(k - 0.70) < 0.01 else ""
        marker = marker or (" ◄ adaptive k_yes (max)" if abs(k - 1.40) < 0.01 else "")
        print(f"  k={k:+.2f}  log-loss={ll:.5f}  Δ={ll - ll_base_yes:+.5f}{marker}")

    res_yes = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                              args=(records_yes, "yes"))
    print(f"\n  ★ Optimal k_yes = {res_yes.x:.4f}  "
          f"log-loss={res_yes.fun:.5f}  "
          f"vs k=0 Δ={res_yes.fun - ll_base_yes:+.5f}")

    print("\n" + "=" * 60)
    print("NO  MODEL  (p_model_no  = Φ(z − Φ⁻¹(p_up)×k))")
    print("=" * 60)
    ll_base_no = log_loss(0.0, records_no, "no")
    print(f"  k=0.00  (baseline)  log-loss={ll_base_no:.5f}")
    for k in k_grid:
        if k == 0.0:
            continue
        ll = log_loss(k, records_no, "no")
        marker = " ◄ current K_DRIFT_NO_BTC" if abs(k - 0.15) < 0.01 else ""
        print(f"  k={k:+.2f}  log-loss={ll:.5f}  Δ={ll - ll_base_no:+.5f}{marker}")

    res_no = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                             args=(records_no, "no"))
    print(f"\n  ★ Optimal k_no  = {res_no.x:.4f}  "
          f"log-loss={res_no.fun:.5f}  "
          f"vs k=0 Δ={res_no.fun - ll_base_no:+.5f}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Optimal k_yes  = {res_yes.x:.4f}   (current DRIFT_MULTIPLIER BTC = 0.70)")
    print(f"  Optimal k_no   = {res_no.x:.4f}   (current K_DRIFT_NO_BTC        = 0.15)")
    print()
    print("  Formula: z_drift = Φ⁻¹(p_up_v2) × k × √(τ/60)")
    print("  Effective z_drift magnitudes at optimal k_yes:")
    for tau_ex in [15, 25, 35, 45, 60]:
        scale  = math.sqrt(tau_ex / 60.0)
        ex_hi  = norm.ppf(0.75) * res_yes.x * scale
        ex_lo  = norm.ppf(0.25) * res_yes.x * scale
        print(f"    τ={tau_ex:2d}m  scale={scale:.3f}  "
              f"z_drift @ p_up=0.75 → {ex_hi:+.3f}  "
              f"z_drift @ p_up=0.25 → {ex_lo:+.3f}")
    print()

    # ── stability check: split by era ─────────────────────────────────────────
    print("ERA STABILITY  (does optimal k change across time?)")
    print("-" * 60)
    # Split records roughly in thirds by order (proxy for time)
    n = len(records_yes)
    thirds = [
        ("Early   (2024–early 2025)", records_yes[:n//3],          records_no[:n//3]),
        ("Mid     (mid 2025)",         records_yes[n//3:2*n//3],    records_no[n//3:2*n//3]),
        ("Recent  (late 2025–2026)",   records_yes[2*n//3:],        records_no[2*n//3:]),
    ]
    for label, ry, rn in thirds:
        if len(ry) < 100:
            continue
        ry_opt = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                 args=(ry, "yes"))
        rn_opt = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                 args=(rn, "no"))
        print(f"  {label}  k_yes={ry_opt.x:.3f}  k_no={rn_opt.x:.3f}  n={len(ry):,}")

    # ── strike-territory split: YES-territory vs NO-territory ─────────────────
    # YES-territory: z_strike < 0  (strike below spot, structure favours YES)
    # NO-territory:  z_strike > 0  (strike above spot, structure favours NO)
    # Allows k_yes and k_no to diverge based on where each side naturally trades.
    print()
    print("STRIKE-TERRITORY SPLIT  (YES-territory vs NO-territory k)")
    print("-" * 60)

    yes_terr_yes = [(z, p, y, t) for z, p, y, t in records_yes if z < 0]
    yes_terr_no  = [(z, p, y, t) for z, p, y, t in records_no  if z < 0]
    no_terr_yes  = [(z, p, y, t) for z, p, y, t in records_yes if z > 0]
    no_terr_no   = [(z, p, y, t) for z, p, y, t in records_no  if z > 0]

    print(f"  YES-territory (z<0): n={len(yes_terr_yes):,}  "
          f"YES-rate={np.mean([r[2] for r in yes_terr_yes]):.1%}")
    print(f"  NO-territory  (z>0): n={len(no_terr_yes):,}  "
          f"YES-rate={np.mean([r[2] for r in no_terr_yes]):.1%}")

    # k_yes calibrated only on YES-territory (where YES contracts live)
    k_yes_terr = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                 args=(yes_terr_yes, "yes"))
    # k_no calibrated only on NO-territory (where NO contracts live)
    k_no_terr  = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                 args=(no_terr_no, "no"))

    print(f"\n  ★ k_yes (YES-territory) = {k_yes_terr.x:.4f}  "
          f"log-loss={k_yes_terr.fun:.5f}")
    print(f"  ★ k_no  (NO-territory)  = {k_no_terr.x:.4f}  "
          f"log-loss={k_no_terr.fun:.5f}")
    print()
    print(f"  Divergence: k_yes={k_yes_terr.x:.3f}  k_no={k_no_terr.x:.3f}  "
          f"diff={k_yes_terr.x - k_no_terr.x:+.3f}")

    # Era stability for territory-split k
    print()
    print("  Era stability for territory-split k:")
    n = len(yes_terr_yes)
    thirds_terr = [
        ("Early  ", yes_terr_yes[:n//3], no_terr_no[:n//3]),
        ("Mid    ", yes_terr_yes[n//3:2*n//3], no_terr_no[n//3:2*n//3]),
        ("Recent ", yes_terr_yes[2*n//3:], no_terr_no[2*n//3:]),
    ]
    for label, ry_t, rn_t in thirds_terr:
        if len(ry_t) < 50 or len(rn_t) < 50:
            continue
        ry_opt_t = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                   args=(ry_t, "yes"))
        rn_opt_t = minimize_scalar(log_loss, bounds=(-0.5, 3.0), method="bounded",
                                   args=(rn_t, "no"))
        print(f"    {label}  k_yes={ry_opt_t.x:.3f} (n={len(ry_t):,})  "
              f"k_no={rn_opt_t.x:.3f} (n={len(rn_t):,})")


if __name__ == "__main__":
    main()
