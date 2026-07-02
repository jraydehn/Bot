"""
calibrate_intraday_seasonality.py

Computes hour-of-day (UTC) realized volatility multipliers from historical
1-minute BTC/ETH/SOL OHLCV data using close-to-close squared log-returns.

Why C2C and not Rogers-Satchell:
  Binance US data has 65-80% flat bars (H=L=O=C) per 1-minute interval due
  to low exchange liquidity. Range estimators like RS assume a continuously
  observed path — on sparse data the observed H/L are drawn from far fewer
  ticks than the true extremes, producing systematic downward bias (~0.19×).
  Close-to-close only requires two prints per bar (the closes), which are
  reliably populated regardless of tick sparsity.

Method:
  1. Compute squared log-return per 1-minute bar: (log(close/close[-1]))²
  2. Sum within each calendar hour → per-hour realized variance
  3. Drop incomplete hours (< 55 bars)
  4. Group by UTC hour (0-23) → mean per-hour variance → vol = sqrt(mean_var)
  5. Normalize so unweighted mean multiplier = 1.0
  6. Save to intraday_vol_seasonality_{asset}.json

Usage:
    python3 calibrate_intraday_seasonality.py           # all three assets
    python3 calibrate_intraday_seasonality.py BTC       # single asset
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent

ASSET_PARQUET = {
    "BTC": "binanceus_BTCUSDT_1m_1970-01-01_2026-06-10.parquet",
    "ETH": "binanceus_ETHUSDT_1m_1970-01-01_2026-06-10.parquet",
    "SOL": "binanceus_SOLUSDT_1m_1970-01-01_2026-06-10.parquet",
}

# Use post-2024 data only — avoids pre-ETF era structural differences
TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")


def calibrate_asset(asset: str) -> dict:
    parquet_name = ASSET_PARQUET.get(asset.upper())
    candidates = list(DATA_DIR.glob(f"binanceus_{asset.upper()}USDT_1m_*.parquet"))
    if parquet_name and (DATA_DIR / parquet_name).exists():
        p = DATA_DIR / parquet_name
    elif candidates:
        p = candidates[0]
    else:
        print(f"  {asset}: no parquet found in {DATA_DIR} — skipping")
        return {}

    print(f"\n{'='*60}")
    print(f"  {asset}: loading {p.name} ...")
    df = pd.read_parquet(p)
    df.columns = df.columns.str.lower()

    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        if ts_col:
            df = df.set_index(pd.to_datetime(df[ts_col], utc=True))
        else:
            raise ValueError(f"{asset}: cannot find timestamp column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df[df.index >= TRAIN_START].copy()
    print(f"  Rows after {TRAIN_START.date()}: {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    flat_rate = (df["high"] == df["low"]).mean()
    print(f"  Flat bar rate: {flat_rate:.1%}  (reason for using C2C not RS)")

    # Close-to-close squared log-return per bar
    log_ret_sq = np.log(df["close"] / df["close"].shift(1)) ** 2
    log_ret_sq.iloc[0] = np.nan          # first bar has no previous close
    log_ret_sq.index = df.index

    # Sum squared returns within each calendar hour = realized variance for that hour
    hourly_var  = log_ret_sq.resample("1h").sum()
    bar_count   = log_ret_sq.resample("1h").count()

    # Drop incomplete hours
    hourly_var  = hourly_var[bar_count >= 55]
    hourly_vol  = np.sqrt(hourly_var)

    print(f"  Valid hours: {len(hourly_vol):,}")

    hourly_df = pd.DataFrame({"vol": hourly_vol, "hour": hourly_vol.index.hour})
    by_hour   = hourly_df.groupby("hour")["vol"].agg(["mean", "std", "count"])
    by_hour.columns = ["mean_vol", "std_vol", "n"]

    global_mean        = by_hour["mean_vol"].mean()
    by_hour["multiplier"] = by_hour["mean_vol"] / global_mean

    print(f"\n  Hour-of-day vol multipliers (UTC), global mean vol = {global_mean:.6f}:")
    print(f"  {'Hour':>4}  {'Mean vol':>10}  {'Multiplier':>10}  {'n':>6}")
    print(f"  {'-'*38}")
    for hour, row in by_hour.iterrows():
        bar = "█" * int(row["multiplier"] * 20)
        print(f"  {hour:4d}  {row['mean_vol']:10.6f}  {row['multiplier']:10.4f}  "
              f"{int(row['n']):6d}  {bar}")

    out = {str(h): round(float(row["multiplier"]), 4)
           for h, row in by_hour.iterrows()}

    high_vol = [h for h, m in out.items() if float(m) > 1.15]
    low_vol  = [h for h, m in out.items() if float(m) < 0.85]
    print(f"\n  High-vol hours (>1.15×): {high_vol}")
    print(f"  Low-vol  hours (<0.85×): {low_vol}")
    print(f"  Range: {min(out.values()):.4f}× – {max(out.values()):.4f}×")

    out_path = OUT_DIR / f"intraday_vol_seasonality_{asset.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path.name}")

    return out


def main():
    assets = sys.argv[1:] if len(sys.argv) > 1 else list(ASSET_PARQUET.keys())
    print(f"Calibrating intraday vol seasonality for: {assets}")
    print(f"Estimator: close-to-close squared log-returns (C2C)")
    print(f"Period: {TRAIN_START.date()} onward")

    for asset in assets:
        calibrate_asset(asset.upper())

    print("\nDone.")


if __name__ == "__main__":
    main()
