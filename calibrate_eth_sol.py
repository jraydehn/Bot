#!/usr/bin/env python3
"""
calibrate_eth_sol.py — Run the composite indicator calibration for ETH and SOL.

Uses the same compute_scores() scoring logic as BTC (composite_scorer.py) but
measures the actual hourly up% per (trend_bin, rev_bin) cell for each asset.

Outputs:
  - Trend score distribution
  - Reversion score distribution
  - Calibration grid (trend_bin × rev_bin → up%)
  - Composite signal bucket summary
  - Trend × Reversion interaction table
  - Comparison to BTC baseline

Run:
    python3 calibrate_eth_sol.py
    python3 calibrate_eth_sol.py --asset ETH
    python3 calibrate_eth_sol.py --asset SOL
"""

import argparse
import glob
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
from composite_scorer import compute_scores, BASELINE_UP, SMOOTHING_N, save_calibration

BTC_BASELINE = BASELINE_UP   # 50.4%

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

SEP  = "=" * 80
SEP2 = "-" * 80


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

    df_4h = ohlcv_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    close_1m  = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)

    print(f"  1h: {len(ohlcv_1h):,}  4h: {len(df_4h):,}  15m: {len(df_15m):,}  "
          f"1m: {len(ohlcv_1m):,}")
    print(f"  Range: {ohlcv_1h.index[0].date()} → {ohlcv_1h.index[-1].date()}")

    return ohlcv_1h, df_4h, df_15m, close_1m, volume_1m


def run_calibration(asset: str):
    print(f"\n{'█'*80}")
    print(f"  COMPOSITE CALIBRATION — {asset}")
    print(f"{'█'*80}")

    print(f"\nLoading {asset} data...")
    ohlcv_1h, df_4h, df_15m, close_1m, volume_1m = load_asset(asset)
    ts_1h = ohlcv_1h.index

    print("Computing composite scores (this takes ~30s)...")
    trend_ser, rev_ser = compute_scores(
        ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float),
        ohlcv_1h["low"].astype(float),   ohlcv_1h["volume"].astype(float),
        df_4h["close"].astype(float),    df_4h["high"].astype(float),
        df_4h["low"].astype(float),      df_4h["volume"].astype(float),
        df_15m["close"].astype(float),   df_15m["high"].astype(float),
        df_15m["low"].astype(float),
        close_1m, volume_1m, ts_1h,
    )

    # Next-hour outcome
    next_ret = np.log(ohlcv_1h["close"] / ohlcv_1h["close"].shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(int)

    # Test set only
    test_mask = ts_1h >= TEST_START
    idx = np.where(test_mask)[0][:-1]

    df = pd.DataFrame({
        "trend":   trend_ser.values[idx],
        "rev":     rev_ser.values[idx],
        "next_up": next_up.values[idx],
    })

    n_test   = len(df)
    baseline = df["next_up"].mean()

    print(f"\nTest hours: {n_test:,}  |  Baseline up%: {baseline:.1%}  "
          f"(BTC baseline: {BTC_BASELINE:.1%})")

    # ── 1. Trend Score distribution ───────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TREND SCORE distribution  ({asset}  4h continuation)")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for ts_val in sorted(df["trend"].unique()):
        sub  = df[df["trend"] == ts_val]
        n    = len(sub)
        up   = sub["next_up"].mean()
        edge = up - baseline
        if n >= 20:
            z  = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
        else:
            pv = float("nan")
        pv_str = f"{pv:.3f}" if not math.isnan(pv) else "  —  "
        sig = "★★★" if (not math.isnan(pv) and pv < 0.01 and abs(edge) > 0.05) else \
              "★★ " if (not math.isnan(pv) and pv < 0.05 and abs(edge) > 0.03) else \
              "★  " if (not math.isnan(pv) and pv < 0.10 and abs(edge) > 0.01) else ""
        print(f"  {ts_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}  {sig}")

    # ── 2. Reversion Score distribution ──────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  REVERSION SCORE distribution  ({asset}  1h/15m mean reversion)")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for rv_val in sorted(df["rev"].unique()):
        sub  = df[df["rev"] == rv_val]
        n    = len(sub)
        up   = sub["next_up"].mean()
        edge = up - baseline
        if n >= 20:
            z  = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
        else:
            pv = float("nan")
        pv_str = f"{pv:.3f}" if not math.isnan(pv) else "  —  "
        sig = "★★★" if (not math.isnan(pv) and pv < 0.01 and abs(edge) > 0.05) else \
              "★★ " if (not math.isnan(pv) and pv < 0.05 and abs(edge) > 0.03) else \
              "★  " if (not math.isnan(pv) and pv < 0.10 and abs(edge) > 0.01) else ""
        print(f"  {rv_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}  {sig}")

    # ── 3. Calibration grid ───────────────────────────────────────────────────
    df["tb"] = df["trend"].clip(-3, 3)
    df["rb"] = df["rev"].clip(-5, 5)

    print(f"\n{SEP}")
    print(f"  CALIBRATION GRID  ({asset}  trend_bucket × reversion_bucket)")
    print(f"  Format: up% [n]   |  baseline {baseline:.1%}")
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
            n    = len(cell)
            if n >= 10:
                up    = cell["next_up"].mean()
                row_str += f"  {up:.0%}[{n:3d}]"
                w     = min(1.0, n / SMOOTHING_N)
                p_cal = w * up + (1 - w) * baseline
                calibration[(int(tb), int(rb))] = round(float(p_cal), 4)
            else:
                row_str += f"  — [{n:3d}]"
        print(row_str)

    # ── 4. Composite signal bucket summary ────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  COMPOSITE SIGNAL BUCKETS  ({asset})")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)

    conditions = [
        ("Strong YES (rev≥+4, trend≥0)",          (df["rev"] >= 4)  & (df["trend"] >= 0)),
        ("Moderate YES (rev +2/+3)",               df["rev"].between(2, 3)),
        ("Trend+Rev agree bullish (t≥1,r≥2)",     (df["trend"] >= 1) & (df["rev"] >= 2)),
        ("Rev bullish, trend bearish (t<0,r≥2)",  (df["trend"] < 0)  & (df["rev"] >= 2)),
        ("Neutral (rev -1 to +1)",                 df["rev"].between(-1, 1)),
        ("Trend+Rev agree bearish (t≤-1,r≤-2)",  (df["trend"] <= -1) & (df["rev"] <= -2)),
        ("Rev bearish, trend bullish (t>0,r≤-2)", (df["trend"] > 0)  & (df["rev"] <= -2)),
        ("Moderate NO (rev -2/-3)",                df["rev"].between(-3, -2)),
        ("Strong NO (rev≤-4, trend≤0)",           (df["rev"] <= -4)  & (df["trend"] <= 0)),
    ]

    print(f"  {'Condition':<45}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for label, mask in conditions:
        sub = df[mask]
        n   = len(sub)
        if n < 20:
            print(f"  {label:<45}   {n:>6,}   (too few)")
            continue
        up   = sub["next_up"].mean()
        edge = up - baseline
        z    = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
        pv   = 2 * (1 - norm.cdf(abs(z)))
        sig  = "★★★" if pv < 0.01 and abs(edge) > 0.05 else \
               "★★ " if pv < 0.05 and abs(edge) > 0.03 else \
               "★  " if pv < 0.10 and abs(edge) > 0.01 else ""
        print(f"  {label:<45}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv:.3f}  {sig}")

    # ── 5. Trend × Reversion interaction ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TREND × REVERSION INTERACTION  ({asset})")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)

    for rev_cond, rev_label in [
        (df["rev"] >= 4,         "Rev ≥ +4 (strong bullish)"),
        (df["rev"].between(2,3), "Rev +2/+3 (moderate bullish)"),
        (df["rev"].between(-3,-2),"Rev -2/-3 (moderate bearish)"),
        (df["rev"] <= -4,        "Rev ≤ -4 (strong bearish)"),
    ]:
        print(f"\n  {rev_label}")
        for trend_cond, trend_label in [
            (df["trend"] >= 2,          "  trend ≥ +2 (strong bull)"),
            (df["trend"].between(0,1),  "  trend 0/+1 (mild bull)"),
            (df["trend"].between(-1,0), "  trend -1/0 (mild bear)"),
            (df["trend"] <= -2,         "  trend ≤ -2 (strong bear)"),
        ]:
            sub = df[rev_cond & trend_cond]
            n   = len(sub)
            if n < 10:
                print(f"  {trend_label:<35}   n={n:4d}   (too few)")
                continue
            up   = sub["next_up"].mean()
            edge = up - baseline
            z    = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv   = 2 * (1 - norm.cdf(abs(z)))
            sig  = "★★★" if pv < 0.01 and abs(edge) > 0.05 else \
                   "★★ " if pv < 0.05 and abs(edge) > 0.03 else \
                   "★  " if pv < 0.10 and abs(edge) > 0.01 else ""
            print(f"  {trend_label:<35}   n={n:4d}   up={up:.1%}   edge={edge:+.1%}   p={pv:.3f}  {sig}")

    # ── 6. Key comparison to BTC ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SUMMARY vs BTC")
    print(SEP2)
    print(f"  Baseline up%:  {asset}={baseline:.1%}  BTC={BTC_BASELINE:.1%}")
    strong_yes = df[(df["rev"] >= 4) & (df["trend"] >= 0)]
    strong_no  = df[(df["rev"] <= -4) & (df["trend"] <= 0)]
    neutral    = df[df["rev"].between(-1, 1)]
    if len(strong_yes) >= 20:
        print(f"  Strong YES signal:  {asset} up%={strong_yes['next_up'].mean():.1%}  (n={len(strong_yes)})")
    if len(strong_no) >= 20:
        print(f"  Strong NO signal:   {asset} up%={strong_no['next_up'].mean():.1%}  (n={len(strong_no)})")
    if len(neutral) >= 20:
        print(f"  Neutral signal:     {asset} up%={neutral['next_up'].mean():.1%}  (n={len(neutral)})")

    print(f"\n  Calibration cells populated: {len(calibration)}")
    save_calibration(calibration, asset)

    return calibration, baseline, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="both",
                        help="ETH, SOL, or both (default: both)")
    args = parser.parse_args()

    assets = []
    if args.asset.upper() == "BOTH":
        assets = ["ETH", "SOL"]
    else:
        assets = [args.asset.upper()]

    results = {}
    for asset in assets:
        cal, baseline, df = run_calibration(asset)
        results[asset] = {"calibration": cal, "baseline": baseline, "df": df}

    if len(results) == 2:
        print(f"\n{'█'*80}")
        print("  CROSS-ASSET COMPARISON")
        print(f"{'█'*80}")
        for asset, r in results.items():
            b = r["baseline"]
            df = r["df"]
            strong_yes = df[(df["rev"] >= 4) & (df["trend"] >= 0)]
            strong_no  = df[(df["rev"] <= -4) & (df["trend"] <= 0)]
            sy_up = f"{strong_yes['next_up'].mean():.1%}" if len(strong_yes) >= 20 else "n/a"
            sn_up = f"{strong_no['next_up'].mean():.1%}"  if len(strong_no)  >= 20 else "n/a"
            print(f"  {asset}:  baseline={b:.1%}  strong_YES_up={sy_up}  strong_NO_up={sn_up}  "
                  f"cal_cells={len(r['calibration'])}")
        print(f"  BTC:  baseline={BTC_BASELINE:.1%}  (reference)")


if __name__ == "__main__":
    main()
