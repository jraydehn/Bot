"""
Incremental OHLCV updater.

Reads the most recent 1m parquet, fetches only the missing candles since
the last timestamp, merges, and saves new parquet files for 1m, 1h, and 4h.

Usage:
    python3 update_data.py
    python3 update_data.py --source bybit
"""

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_data import fetch_ohlcv, resample_ohlcv

DATA_DIR = Path(__file__).parent / "data"


def find_latest_parquet(interval: str, symbol: str) -> Path:
    matches = sorted(
        DATA_DIR.glob(f"*{symbol}_{interval}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    # Exclude checkpoint files
    matches = [m for m in matches if ".ckpt." not in m.name]
    if not matches:
        raise FileNotFoundError(f"No existing {interval} parquet found for {symbol}. Run fetch_data.py first.")
    return matches[-1]


def main(source: str = "binanceus", asset: str = "BTC"):
    from live_signal import ASSET_CONFIG
    cfg    = ASSET_CONFIG.get(asset.upper(), ASSET_CONFIG["BTC"])
    symbol = cfg["binance_symbol"]

    now_utc  = datetime.now(timezone.utc)
    today    = now_utc.strftime("%Y-%m-%d")
    tomorrow = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

    # --- Update 1m ---
    print(f"\n[1/3] Updating 1m data ({asset})...")
    base_path = find_latest_parquet("1m", symbol)
    print(f"  Base file: {base_path.name}")
    df_existing = pd.read_parquet(base_path)
    if df_existing.index.tz is None:
        df_existing.index = df_existing.index.tz_localize("UTC")

    last_ts = df_existing.index[-1]
    # Derive start date from existing file to preserve original filename prefix
    start = str(df_existing.index[0].date())
    resume_from = (last_ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
    print(f"  Last candle: {last_ts}  — fetching from {resume_from} to {today}")

    new_1m = fetch_ohlcv(symbol=symbol, interval="1m",
                         start=resume_from, end=tomorrow,
                         source=source, verbose=True)

    if new_1m is None or new_1m.empty:
        print(f"  No new candles — data already current.")
        df_1m = df_existing
    else:
        df_1m = pd.concat([df_existing, new_1m])
        df_1m = df_1m[~df_1m.index.duplicated(keep="last")].sort_index()

    out_1m = DATA_DIR / f"{source}_{symbol}_1m_{start}_{today}.parquet"
    df_1m.to_parquet(out_1m)
    print(f"  Saved → {out_1m.name}  ({len(df_1m):,} candles)")

    # --- Resample to 1h ---
    print(f"\n[2/3] Resampling to 1h ({asset})...")
    df_1h = resample_ohlcv(df_1m, "1h")
    out_1h = DATA_DIR / f"{source}_{symbol}_1h_{start}_{today}.parquet"
    df_1h.to_parquet(out_1h)
    print(f"  Saved → {out_1h.name}  ({len(df_1h):,} candles, last: {df_1h.index[-1]})")

    # --- Resample to 4h ---
    print(f"\n[3/3] Resampling to 4h ({asset})...")
    df_4h = resample_ohlcv(df_1m, "4h")
    out_4h = DATA_DIR / f"{source}_{symbol}_4h_{start}_{today}.parquet"
    df_4h.to_parquet(out_4h)
    print(f"  Saved → {out_4h.name}  ({len(df_4h):,} candles, last: {df_4h.index[-1]})")

    print(f"\nDone. {asset} data updated through {today}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="binanceus",
                        choices=["binanceus", "bybit", "binance"])
    parser.add_argument("--asset", default="BTC",
                        help="Asset to update: BTC, ETH, SOL (default: BTC)")
    args = parser.parse_args()
    main(source=args.source, asset=args.asset)
