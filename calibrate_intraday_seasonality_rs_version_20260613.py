"""
calibrate_intraday_seasonality.py

Computes hour-of-day (UTC) realized volatility multipliers from historical
1-minute BTC OHLCV data using the Rogers-Satchell estimator.

Method:
  1. Compute RS variance per 1-minute bar
  2. Sum RS variance within each calendar hour → per-hour realized variance
  3. Group by UTC hour (0–23) → mean per-hour variance
  4. Normalize so that the weighted mean multiplier = 1.0
  5. Save to intraday_vol_seasonality_btc.json

The output is a dict {hour_str: multiplier} where multiplier > 1 means that
hour is historically more volatile than average. Apply by scaling vol_multi:
    vol_eff_adj = vol_multi * seasonality_multiplier[current_utc_hour]

Usage:
    python3 calibrate_intraday_seasonality.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "binanceus_BTCUSDT_1m_1970-01-01_2026-06-10.parquet"
OUT_PATH  = Path(__file__).parent / "intraday_vol_seasonality_btc.json"

# Use data from 2024 onwards — avoids pre-ETF era regime differences
TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")


def rogers_satchell_var(df: pd.DataFrame) -> pd.Series:
    """
    Rogers-Satchell variance per bar (drift-robust OHLC estimator).
    RS = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)
    Guaranteed >= 0 for valid OHLC (H>=O,C and L<=O,C). Clip for safety.
    """
    h = np.log(df["high"]  / df["open"])
    l = np.log(df["low"]   / df["open"])
    c = np.log(df["close"] / df["open"])
    rs = h * (h - c) + l * (l - c)
    return rs.clip(lower=0)


def main():
    print(f"Loading {DATA_PATH.name} ...")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()

    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        if ts_col:
            df = df.set_index(pd.to_datetime(df[ts_col], utc=True))
        else:
            raise ValueError("Cannot find timestamp column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df[df.index >= TRAIN_START].copy()
    print(f"  Rows after {TRAIN_START.date()}: {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # Rogers-Satchell variance per 1-minute bar
    print("Computing Rogers-Satchell variance per bar ...")
    rs_var = rogers_satchell_var(df)
    rs_var.index = df.index

    # Sum RS variance within each calendar hour = realized variance for that hour
    hourly_var = rs_var.resample("1h").sum()
    hourly_vol = np.sqrt(hourly_var)   # realized vol per hour (in log-return units)

    # Drop incomplete hours (< 55 bars) — session open/close artifacts
    bar_count   = rs_var.resample("1h").count()
    hourly_vol  = hourly_vol[bar_count >= 55]

    print(f"  Valid hours: {len(hourly_vol):,}")

    # Group by UTC hour of day
    hourly_df = pd.DataFrame({"vol": hourly_vol, "hour": hourly_vol.index.hour})

    by_hour = hourly_df.groupby("hour")["vol"].agg(["mean", "std", "count"])
    by_hour.columns = ["mean_vol", "std_vol", "n"]

    # Global mean (weighted equally across hours for normalization)
    global_mean = by_hour["mean_vol"].mean()

    by_hour["multiplier"] = by_hour["mean_vol"] / global_mean

    print("\nHour-of-day volatility multipliers (UTC):")
    print(f"  Global mean hourly vol: {global_mean:.6f}")
    print(f"  {'Hour':>4}  {'Mean vol':>10}  {'Multiplier':>10}  {'n':>6}")
    print(f"  {'-'*36}")
    for hour, row in by_hour.iterrows():
        bar = "█" * int(row["multiplier"] * 20)
        print(f"  {hour:4d}  {row['mean_vol']:10.6f}  {row['multiplier']:10.4f}  "
              f"{int(row['n']):6d}  {bar}")

    # Save
    out = {str(h): round(float(row["multiplier"]), 4)
           for h, row in by_hour.iterrows()}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")
    print(f"  Min multiplier: {min(out.values()):.4f}  Max: {max(out.values()):.4f}")

    # Summary stats
    high_vol_hours = [h for h, m in out.items() if float(m) > 1.15]
    low_vol_hours  = [h for h, m in out.items() if float(m) < 0.85]
    print(f"  High-vol hours (>1.15×): {high_vol_hours}")
    print(f"  Low-vol  hours (<0.85×): {low_vol_hours}")


if __name__ == "__main__":
    main()
