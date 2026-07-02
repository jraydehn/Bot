#!/usr/bin/env python3
"""
calibrate_90m.py — Build per-asset 90m and 120m p_up calibration tables.

Methodology
-----------
Uses 4h-scale composite scores (4h trend + 4h/1h reversion via _reversion_votes_4h),
measuring the 90-minute and 120-minute forward price outcomes from each 30m bar.

At each 30m bar T:
  - Scores: compute_scores_90m() aligned to 30m grid (4h trend + 4h reversion)
  - Outcome 90m:  is close at T+90min > close at T?
  - Outcome 120m: is close at T+120min > close at T?

These tables extend lookup_p_up_blended() for contracts with τ ≥ 60min:
  tau [60, 90):   blend 90m ↔ 1h
  tau [90, 120):  blend 120m ↔ 90m
  tau >= 120:     pure 120m

Run:
    python3 calibrate_90m.py              # all assets, both 90m and 120m
    python3 calibrate_90m.py --asset BTC
    python3 calibrate_90m.py --asset ETH
    python3 calibrate_90m.py --asset SOL
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
    compute_scores_90m, BASELINE_UP, SMOOTHING_N,
    ASSET_BASELINES,
)

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
SEP  = "=" * 80
SEP2 = "-" * 80

CAL_PATHS_90M = {
    "BTC": BASE / "composite_calibration_btc_90m.json",
    "ETH": BASE / "composite_calibration_eth_90m.json",
    "SOL": BASE / "composite_calibration_sol_90m.json",
}
CAL_PATHS_120M = {
    "BTC": BASE / "composite_calibration_btc_120m.json",
    "ETH": BASE / "composite_calibration_eth_120m.json",
    "SOL": BASE / "composite_calibration_sol_120m.json",
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

    close_1m  = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)

    print(f"  1h: {len(ohlcv_1h):,}  1m: {len(ohlcv_1m):,}")
    print(f"  Range: {ohlcv_1h.index[0].date()} → {ohlcv_1h.index[-1].date()}")
    return ohlcv_1h, close_1m, volume_1m


def build_outcomes(close_1m: pd.Series, ts_30m: pd.DatetimeIndex,
                   horizon_minutes: int) -> pd.Series:
    """
    For each 30m bar T, find the 1m close at T+horizon_minutes.
    Returns a boolean int Series (1 = price higher, 0 = lower/equal) aligned to ts_30m.
    """
    outcomes = {}
    for T in ts_30m:
        T_fut = T + pd.Timedelta(minutes=horizon_minutes)
        future = close_1m.index[close_1m.index >= T_fut]
        if len(future) == 0:
            continue
        c_now = close_1m.get(T)
        if c_now is None or pd.isna(c_now):
            at_or_after = close_1m.index[close_1m.index >= T]
            if len(at_or_after) == 0:
                continue
            c_now = close_1m.iloc[close_1m.index.get_loc(at_or_after[0])]
        c_fut = close_1m.iloc[close_1m.index.get_loc(future[0])]
        outcomes[T] = int(c_fut > c_now)
    return pd.Series(outcomes, dtype=int)


def build_calibration(df: pd.DataFrame, baseline: float, horizon_label: str,
                      asset: str) -> dict:
    """Build the (tb, rb) → p_cal calibration dict from a scored + outcome DataFrame."""
    df = df.copy()
    df["tb"] = df["trend"].clip(-3, 3)
    df["rb"] = df["rev"].clip(-11, 11)

    trend_bins = sorted(df["tb"].unique())
    rev_bins   = sorted(df["rb"].unique())

    print(f"\n{SEP}")
    print(f"  CALIBRATION GRID — {horizon_label}  ({asset}   trend × rev → {horizon_label} up%)")
    print(f"  Format: up% [n]   |  baseline {baseline:.1%}")
    print(SEP2)

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

    return calibration


def print_score_distributions(df: pd.DataFrame, obs_base: float, label: str, asset: str):
    print(f"\n{SEP}")
    print(f"  TREND SCORE — {label} forward outcome  ({asset})")
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
    print(f"  REVERSION SCORE — {label} forward outcome  ({asset})")
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


def run_calibration(asset: str):
    print(f"\n{'█'*80}")
    print(f"  90m / 120m CALIBRATION — {asset}")
    print(f"{'█'*80}")

    baseline = ASSET_BASELINES.get(asset.upper(), BASELINE_UP)
    print(f"\nLoading {asset} data...")
    ohlcv_1h, close_1m, volume_1m = load_asset(asset)

    # 30m timestamp grid
    close_30m = close_1m.resample("30min", origin="start_day").last().dropna()
    ts_30m = close_30m.index

    print("Computing 4h-scale composite scores...")
    trend_ser, rev_ser = compute_scores_90m(
        ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float),
        ohlcv_1h["low"].astype(float),   ohlcv_1h["volume"].astype(float),
        close_1m, volume_1m, ts_30m,
    )

    print("Computing 90m forward outcomes...")
    next_up_90m  = build_outcomes(close_1m, ts_30m, 90)
    print("Computing 120m forward outcomes...")
    next_up_120m = build_outcomes(close_1m, ts_30m, 120)

    df_base = pd.DataFrame({
        "trend": trend_ser,
        "rev":   rev_ser,
    })

    df_90m = df_base.copy()
    df_90m["next_up"] = next_up_90m
    df_90m = df_90m.dropna()
    df_90m = df_90m[df_90m.index >= TEST_START]

    df_120m = df_base.copy()
    df_120m["next_up"] = next_up_120m
    df_120m = df_120m.dropna()
    df_120m = df_120m[df_120m.index >= TEST_START]

    obs_90m  = df_90m["next_up"].mean()
    obs_120m = df_120m["next_up"].mean()

    print(f"\nTest 30m bars — 90m: {len(df_90m):,}  |  observed up%: {obs_90m:.1%}")
    print(f"Test 30m bars — 120m: {len(df_120m):,} |  observed up%: {obs_120m:.1%}")
    print(f"Asset baseline: {baseline:.1%}")

    # Score distributions
    print_score_distributions(df_90m, obs_90m, "90m", asset)
    print_score_distributions(df_120m, obs_120m, "120m", asset)

    # Build and save calibration tables
    cal_90m  = build_calibration(df_90m,  obs_90m,  "90m",  asset)
    cal_120m = build_calibration(df_120m, obs_120m, "120m", asset)

    path_90m = CAL_PATHS_90M[asset.upper()]
    raw_90m  = {f"{k[0]},{k[1]}": v for k, v in cal_90m.items()}
    with open(path_90m, "w") as f:
        json.dump(raw_90m, f, indent=2)
    print(f"\n  90m calibration saved → {path_90m}  ({len(cal_90m)} cells)")

    path_120m = CAL_PATHS_120M[asset.upper()]
    raw_120m  = {f"{k[0]},{k[1]}": v for k, v in cal_120m.items()}
    with open(path_120m, "w") as f:
        json.dump(raw_120m, f, indent=2)
    print(f"  120m calibration saved → {path_120m}  ({len(cal_120m)} cells)")

    # ── Signal comparison across horizons ────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SIGNAL COMPARISON: 90m vs 120m outcome  ({asset})")
    print(f"  (Same 4h composite scores, different forward windows)")
    print(SEP2)
    shared_idx = df_90m.index.intersection(df_120m.index)
    df90s  = df_90m.loc[shared_idx]
    df120s = df_120m.loc[shared_idx]
    buckets = [
        ("Strong YES (rev≥+4, trend≥0)",  (df90s["rev"] >= 4)  & (df90s["trend"] >= 0)),
        ("Moderate YES (rev +2/+3)",        df90s["rev"].between(2, 3)),
        ("Neutral (rev -1 to +1)",          df90s["rev"].between(-1, 1)),
        ("Moderate NO (rev -2/-3)",         df90s["rev"].between(-3, -2)),
        ("Strong NO (rev≤-4, trend≤0)",    (df90s["rev"] <= -4) & (df90s["trend"] <= 0)),
    ]
    print(f"  {'Bucket':<40}  {'n':>5}  {'90m up%':>8}  {'120m up%':>9}  {'edge 90m':>9}  {'edge 120m':>10}")
    print(SEP2)
    for label, mask in buckets:
        sub90  = df90s[mask]
        sub120 = df120s[mask]
        if len(sub90) < 20:
            continue
        up90  = sub90["next_up"].mean()
        up120 = sub120["next_up"].mean() if len(sub120) >= 5 else float("nan")
        e90   = up90  - obs_90m
        e120  = up120 - obs_120m if not math.isnan(up120) else float("nan")
        up120_str = f"{up120:.1%}" if not math.isnan(up120) else "   —  "
        e120_str  = f"{e120:+.1%}" if not math.isnan(e120)  else "    —  "
        print(f"  {label:<40}  {len(sub90):>5,}  {up90:>8.1%}  {up120_str:>9}  {e90:>+9.1%}  {e120_str:>10}")

    return cal_90m, cal_120m


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
