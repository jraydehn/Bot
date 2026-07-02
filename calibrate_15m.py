#!/usr/bin/env python3
"""
calibrate_15m.py — Build per-asset 15m p_up calibration tables.

Methodology
-----------
Uses the same composite_scores_30m() scores (1h trend ffilled + 15m/5m reversion),
but measures the 15-minute forward price outcome instead of 30 minutes.

At each 30m bar T:
  - Scores: compute_scores_30m() aligned to 30m grid (same as 30m calibration)
  - Outcome: is close at T+15min > close at T?  (from 1m data)

This gives the correct p_up for Kalshi contracts with τ ≤ 15min remaining.
The lookup_p_up_blended() function then blends 30m→15m tables for τ in [0,30).

Run:
    python3 calibrate_15m.py              # all assets
    python3 calibrate_15m.py --asset ETH
    python3 calibrate_15m.py --asset BTC
    python3 calibrate_15m.py --asset SOL
"""

import argparse
import glob
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA = BASE / "data"

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores_30m, BASELINE_UP, SMOOTHING_N,
    ASSET_BASELINES,
)

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
SEP  = "=" * 80
SEP2 = "-" * 80

CAL_PATHS_15M = {
    "BTC": BASE / "composite_calibration_btc_15m.json",
    "ETH": BASE / "composite_calibration_eth_15m.json",
    "SOL": BASE / "composite_calibration_sol_15m.json",
}


def load_asset(asset: str):
    sym = f"{asset}USDT"
    f_1h = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_1m = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]

    ohlcv_1h = pd.read_parquet(f_1h)
    ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
    ohlcv_1h = ohlcv_1h.sort_index()

    ohlcv_1m = pd.read_parquet(f_1m)
    ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
    ohlcv_1m = ohlcv_1m.sort_index()

    df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    close_1m  = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)

    print(f"  1h: {len(ohlcv_1h):,}  15m: {len(df_15m):,}  1m: {len(ohlcv_1m):,}")
    print(f"  Range: {ohlcv_1h.index[0].date()} → {ohlcv_1h.index[-1].date()}")
    return ohlcv_1h, df_15m, close_1m, volume_1m


def build_15m_outcomes(close_1m: pd.Series, ts_30m: pd.DatetimeIndex) -> pd.Series:
    """
    For each 30m bar T, find the 1m close at T+15min.
    Returns a boolean Series (True = price higher 15min later) aligned to ts_30m.
    """
    outcomes = {}
    for T in ts_30m:
        T15 = T + pd.Timedelta(minutes=15)
        # find nearest 1m bar at or after T+15
        future = close_1m.index[close_1m.index >= T15]
        if len(future) == 0:
            continue
        c_now  = close_1m.get(T)
        if c_now is None or pd.isna(c_now):
            # try nearest bar at or after T
            at_or_after = close_1m.index[close_1m.index >= T]
            if len(at_or_after) == 0:
                continue
            c_now = close_1m.iloc[close_1m.index.get_loc(at_or_after[0])]
        c_t15 = close_1m.iloc[close_1m.index.get_loc(future[0])]
        outcomes[T] = int(c_t15 > c_now)

    return pd.Series(outcomes, dtype=int)


def run_calibration(asset: str):
    print(f"\n{'█'*80}")
    print(f"  15M CALIBRATION — {asset}")
    print(f"{'█'*80}")

    baseline = ASSET_BASELINES.get(asset.upper(), BASELINE_UP)
    print(f"\nLoading {asset} data...")
    ohlcv_1h, df_15m, close_1m, volume_1m = load_asset(asset)

    # 30m timestamp grid
    close_30m = close_1m.resample("30min", origin="start_day").last().dropna()
    ts_30m = close_30m.index

    print("Computing 30m composite scores...")
    trend_ser, rev_ser = compute_scores_30m(
        ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float),
        ohlcv_1h["low"].astype(float),   ohlcv_1h["volume"].astype(float),
        df_15m["close"].astype(float),   df_15m["high"].astype(float),
        df_15m["low"].astype(float),
        close_1m, volume_1m, ts_30m,
    )

    print("Computing 15m forward outcomes...")
    next_up_15m = build_15m_outcomes(close_1m, ts_30m)

    df = pd.DataFrame({
        "trend":   trend_ser,
        "rev":     rev_ser,
        "next_up": next_up_15m,
    }).dropna()

    # test set only
    df = df[df.index >= TEST_START]
    n_test   = len(df)
    obs_base = df["next_up"].mean()

    print(f"\nTest 30m bars: {n_test:,}  |  Observed 15m up%: {obs_base:.1%}  "
          f"(asset baseline: {baseline:.1%})")

    # ── Score distributions ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TREND SCORE — 15m forward outcome  ({asset})")
    print(f"  Baseline up% = {obs_base:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for ts_val in sorted(df["trend"].unique()):
        sub  = df[df["trend"] == ts_val]
        n    = len(sub)
        up   = sub["next_up"].mean()
        edge = up - obs_base
        if n >= 20:
            z  = (up - obs_base) / math.sqrt(obs_base * (1 - obs_base) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
            pv_str = f"{pv:.3f}"
            sig = ("★★★" if pv < 0.01 and abs(edge) > 0.05 else
                   "★★ " if pv < 0.05 and abs(edge) > 0.03 else
                   "★  " if pv < 0.10 and abs(edge) > 0.01 else "")
        else:
            pv_str, sig = "  —  ", ""
        print(f"  {ts_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}  {sig}")

    print(f"\n{SEP}")
    print(f"  REVERSION SCORE — 15m forward outcome  ({asset})")
    print(f"  Baseline up% = {obs_base:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for rv_val in sorted(df["rev"].unique()):
        sub  = df[df["rev"] == rv_val]
        n    = len(sub)
        up   = sub["next_up"].mean()
        edge = up - obs_base
        if n >= 20:
            z  = (up - obs_base) / math.sqrt(obs_base * (1 - obs_base) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
            pv_str = f"{pv:.3f}"
            sig = ("★★★" if pv < 0.01 and abs(edge) > 0.05 else
                   "★★ " if pv < 0.05 and abs(edge) > 0.03 else
                   "★  " if pv < 0.10 and abs(edge) > 0.01 else "")
        else:
            pv_str, sig = "  —  ", ""
        print(f"  {rv_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}  {sig}")

    # ── Calibration grid ─────────────────────────────────────────────────────
    df["tb"] = df["trend"].clip(-3, 3)
    df["rb"] = df["rev"].clip(-11, 11)

    print(f"\n{SEP}")
    print(f"  CALIBRATION GRID — 15m  ({asset}   trend × rev → 15m up%)")
    print(f"  Format: up% [n]   |  baseline {obs_base:.1%}")
    print(SEP2)

    trend_bins = sorted(df["tb"].unique())
    rev_bins   = sorted(df["rb"].unique())

    hdr = f"  {'Rev →':>8}  " + "".join(f"  {rb:>+4d}  " for rb in rev_bins)
    print(hdr)
    print(f"  {'Trend ↓':>8}  " + "-" * (len(rev_bins) * 8))

    calibration = {}
    for tb in trend_bins:
        row_str = f"  {tb:>+8d}  "
        for rb in rev_bins:
            cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
            n = len(cell)
            if n >= 10:
                up    = cell["next_up"].mean()
                row_str += f"  {up:.0%}[{n:3d}]"
                w     = min(1.0, n / SMOOTHING_N)
                p_cal = w * up + (1 - w) * baseline
                calibration[(int(tb), int(rb))] = round(float(p_cal), 4)
            else:
                row_str += f"  — [{n:3d}]"
        print(row_str)

    # ── Compare 15m vs 30m signal strength ──────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SIGNAL COMPARISON: 15m vs 30m outcome  ({asset})")
    print(f"  (Same composite scores, different forward window)")
    print(SEP2)
    buckets = [
        ("Strong YES (rev≥+4, trend≥0)",   (df["rev"] >= 4)  & (df["trend"] >= 0)),
        ("Moderate YES (rev +2/+3)",         df["rev"].between(2, 3)),
        ("Neutral (rev -1 to +1)",           df["rev"].between(-1, 1)),
        ("Moderate NO (rev -2/-3)",          df["rev"].between(-3, -2)),
        ("Strong NO (rev≤-4, trend≤0)",     (df["rev"] <= -4) & (df["trend"] <= 0)),
    ]
    print(f"  {'Bucket':<40}  {'n':>5}  {'15m up%':>8}  {'edge':>7}")
    print(SEP2)
    for label, mask in buckets:
        sub = df[mask]
        if len(sub) < 20:
            continue
        up   = sub["next_up"].mean()
        edge = up - obs_base
        print(f"  {label:<40}  {len(sub):>5,}  {up:>8.1%}  {edge:>+7.1%}")

    # ── Save ─────────────────────────────────────────────────────────────────
    path = CAL_PATHS_15M[asset.upper()]
    raw  = {f"{k[0]},{k[1]}": v for k, v in calibration.items()}
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  15m calibration saved → {path}  ({len(calibration)} cells)")

    return calibration, obs_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="all",
                        help="BTC, ETH, SOL, or all (default: all)")
    args = parser.parse_args()

    assets = ["BTC", "ETH", "SOL"] if args.asset.lower() == "all" else [args.asset.upper()]
    for asset in assets:
        run_calibration(asset)


if __name__ == "__main__":
    main()
