"""
backfill_cvd.py — Backfill cvd_4h to all paper_trades and scan_archive CSVs.

CVD definition (per 1h bar):
  delta      = 2 * taker_buy_quote_vol - total_quote_vol
  cvd_4h     = rolling sum of the last 4 bars' deltas (= last 4h of net buying pressure)
  Positive   = net buying; negative = net selling.

Source: Binance.us /api/v3/klines (free, no subscription needed).

Timestamp alignment (same as backfill_keltner.py):
  Hourly CSVs (paper_trades / scan_archive): bar_ts = logged_at.floor('1H') - 1H
  15m CSVs: bar_ts = close_time.floor('1H') - 1H
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"

ASSET_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

TARGETS = [
    # (csv_path,                              ts_col,       asset)
    (RES_DIR / "paper_trades.csv",            "logged_at",  "BTC"),
    (RES_DIR / "paper_trades_eth.csv",        "logged_at",  "ETH"),
    (RES_DIR / "paper_trades_sol.csv",        "logged_at",  "SOL"),
    (RES_DIR / "btc_scan_archive.csv",        "logged_at",  "BTC"),
    (RES_DIR / "eth_scan_archive.csv",        "logged_at",  "ETH"),
    (RES_DIR / "sol_scan_archive.csv",        "logged_at",  "SOL"),
    (RES_DIR / "paper_trades_btc15m.csv",     "close_time", "BTC"),
    (RES_DIR / "paper_trades_eth15m.csv",     "close_time", "ETH"),
    (RES_DIR / "paper_trades_sol15m.csv",     "close_time", "SOL"),
]


# ── Fetch full CVD series ──────────────────────────────────────────────────────

def fetch_cvd_series(asset: str, start: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch all 1h klines from `start` to now for `asset` from Binance.us.
    Returns a DataFrame with columns [bar_open, cvd_4h] indexed by bar_open (UTC).
    """
    symbol  = ASSET_SYMBOLS[asset]
    url     = "https://api.binance.us/api/v3/klines"
    rows    = []
    cur_ms  = int(start.timestamp() * 1000)
    now_ms  = int(pd.Timestamp.utcnow().timestamp() * 1000)

    while cur_ms < now_ms:
        r = requests.get(url, params={
            "symbol": symbol, "interval": "1h",
            "startTime": cur_ms, "limit": 1000,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur_ms = int(batch[-1][0]) + 3_600_000  # next bar
        if len(batch) < 1000:
            break
        time.sleep(0.1)

    if not rows:
        raise RuntimeError(f"No kline data returned for {asset}")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"]       = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["taker_buy_quote"] = df["taker_buy_quote"].astype(float)
    df["quote_vol"]       = df["quote_vol"].astype(float)
    df["delta"]           = 2.0 * df["taker_buy_quote"] - df["quote_vol"]

    df = df.sort_values("open_time").drop_duplicates("open_time")
    df["cvd_4h"] = df["delta"].rolling(4, min_periods=4).sum().round(2)
    df = df.rename(columns={"open_time": "bar_open"})
    return df[["bar_open", "cvd_4h"]].dropna(subset=["cvd_4h"])


# ── Backfill one CSV ───────────────────────────────────────────────────────────

def backfill_csv(csv_path: Path, ts_col: str, cvd: pd.DataFrame) -> int:
    """Add/overwrite cvd_4h in csv_path. Returns number of rows filled."""
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)

    # Drop existing cvd_4h so re-runs don't create _x/_y conflicts
    df = df.drop(columns=["cvd_4h"], errors="ignore")

    # Parse timestamp safely — format='mixed' handles both naive and tz-aware strings
    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", utc=True, errors="coerce")

    # Compute completed bar open time
    df["_bar_ts"] = df[ts_col].dt.floor("1h") - pd.Timedelta(hours=1)

    # Sort for merge_asof, preserving original order
    df["_orig_order"] = range(len(df))
    nat_mask  = df["_bar_ts"].isna()
    df_valid  = df[~nat_mask].sort_values("_bar_ts").copy()
    df_nat    = df[nat_mask].copy()

    merged_valid = pd.merge_asof(
        df_valid,
        cvd.sort_values("bar_open"),
        left_on="_bar_ts",
        right_on="bar_open",
        direction="backward",
        tolerance=pd.Timedelta("1h"),  # exact-bar match only
    )

    merged = pd.concat([merged_valid, df_nat], ignore_index=True)
    merged = merged.sort_values("_orig_order").reset_index(drop=True)
    merged = merged.drop(columns=["_bar_ts", "_orig_order", "bar_open"], errors="ignore")

    assert len(merged) == original_len, \
        f"ROW COUNT MISMATCH: {original_len} → {len(merged)} — aborting"

    merged.to_csv(csv_path, index=False)
    return int(merged["cvd_4h"].notna().sum())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Fetch from April 1 — covers the earliest paper_trades_eth.csv (Apr 15)
    # with a 2-week buffer for the 4-bar rolling window to stabilise.
    fetch_from = pd.Timestamp("2026-04-01", tz="UTC")

    cvd_cache: dict[str, pd.DataFrame] = {}
    for asset in ("BTC", "ETH", "SOL"):
        print(f"Fetching {asset} 1h CVD series from {fetch_from.date()} ...", end=" ", flush=True)
        cvd_cache[asset] = fetch_cvd_series(asset, fetch_from)
        print(f"{len(cvd_cache[asset])} bars  "
              f"(latest: {cvd_cache[asset]['bar_open'].max().date()})")

    for csv_path, ts_col, asset in TARGETS:
        if not csv_path.exists():
            print(f"  SKIP (not found): {csv_path.name}")
            continue
        n_rows = len(pd.read_csv(csv_path, usecols=[ts_col]))
        print(f"  {csv_path.name} ({n_rows:,} rows) ...", end=" ", flush=True)
        filled = backfill_csv(csv_path, ts_col, cvd_cache[asset])
        print(f"filled {filled:,} / {n_rows:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
