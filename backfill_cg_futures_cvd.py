"""
backfill_cg_futures_cvd.py — Backfill cg_futures_delta_4h, cg_futures_ratio_4h,
cg_futures_cvd_12h to all paper_trades and scan_archive CSVs.

Source: CoinGlass /api/futures/aggregated-taker-buy-sell-volume/history
        (Binance + OKX + Bybit perpetual futures, 4h bars)

Fields:
  cg_futures_delta_4h  = buy_usd - sell_usd  for the last completed 4h bar
  cg_futures_ratio_4h  = buy_usd / sell_usd  for the last completed 4h bar; >1 = net buying
  cg_futures_cvd_12h   = rolling 3-bar sum of deltas (12h cumulative futures CVD)

Timestamp alignment (same as backfill_keltner.py / backfill_cvd.py):
  Hourly CSVs: bar_ts = logged_at.floor('4H')  (align to 4h bar open)
  15m CSVs:    bar_ts = close_time.floor('4H')
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"

_BASE    = "https://open-api-v4.coinglass.com"
_API_KEY = os.environ.get("COINGLASS_API_KEY", "8f0a30c29a5e424ba2641f649051786b")
_HEADERS = {"CG-API-KEY": _API_KEY}
_EXCHANGES = "Binance,OKX,Bybit"

ASSET_MAP = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"}

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


def fetch_futures_taker_series(asset: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch up to `limit` 4h bars of aggregated futures taker volume from CoinGlass.
    Returns DataFrame with columns [bar_open, cg_futures_delta_4h, cg_futures_ratio_4h,
    cg_futures_cvd_12h] indexed by bar_open (UTC 4h bar open time).
    """
    r = requests.get(
        f"{_BASE}/api/futures/aggregated-taker-buy-sell-volume/history",
        headers=_HEADERS,
        params={"symbol": asset, "interval": "4h", "limit": limit,
                "exchange_list": _EXCHANGES},
        timeout=15,
    )
    body = r.json()
    if body.get("code") != "0":
        raise RuntimeError(f"CoinGlass {asset} futures taker: {body.get('msg','unknown error')}")

    data = body.get("data") or []
    if not data:
        raise RuntimeError(f"No data returned for {asset}")

    df = pd.DataFrame(data)
    df["bar_open"]   = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["buy"]        = df["aggregated_buy_volume_usd"].astype(float)
    df["sell"]       = df["aggregated_sell_volume_usd"].astype(float)
    df["delta"]      = df["buy"] - df["sell"]
    df["ratio"]      = (df["buy"] / df["sell"].replace(0, float("nan"))).fillna(1.0)

    df = df.sort_values("bar_open").drop_duplicates("bar_open").reset_index(drop=True)
    df["cvd_12h"]    = df["delta"].rolling(3, min_periods=3).sum()

    return df[["bar_open", "delta", "ratio", "cvd_12h"]].rename(columns={
        "delta":   "cg_futures_delta_4h",
        "ratio":   "cg_futures_ratio_4h",
        "cvd_12h": "cg_futures_cvd_12h",
    }).dropna(subset=["cg_futures_cvd_12h"])


def backfill_csv(csv_path: Path, ts_col: str, series: pd.DataFrame) -> int:
    """Add/overwrite the three cg_futures_* columns. Returns rows filled."""
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)

    df = df.drop(columns=["cg_futures_delta_4h", "cg_futures_ratio_4h",
                           "cg_futures_cvd_12h"], errors="ignore")

    # Parse timestamp safely — format='mixed' handles both naive and tz-aware strings
    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", utc=True, errors="coerce")

    # Align to 4h bar open (floor to 4h)
    df["_bar_ts"] = df[ts_col].dt.floor("4h")

    df["_orig_order"] = range(len(df))
    nat_mask  = df["_bar_ts"].isna()
    df_valid  = df[~nat_mask].sort_values("_bar_ts").copy()
    df_nat    = df[nat_mask].copy()

    merged_valid = pd.merge_asof(
        df_valid,
        series.sort_values("bar_open"),
        left_on="_bar_ts",
        right_on="bar_open",
        direction="backward",
        tolerance=pd.Timedelta("4h"),
    )

    merged = pd.concat([merged_valid, df_nat], ignore_index=True)
    merged = merged.sort_values("_orig_order").reset_index(drop=True)
    merged = merged.drop(columns=["_bar_ts", "_orig_order", "bar_open"], errors="ignore")

    assert len(merged) == original_len, \
        f"ROW COUNT MISMATCH: {original_len} → {len(merged)} in {csv_path.name}"

    merged.to_csv(csv_path, index=False)
    return int(merged["cg_futures_delta_4h"].notna().sum())


def main():
    series_cache: dict[str, pd.DataFrame] = {}
    for asset in ("BTC", "ETH", "SOL"):
        print(f"Fetching {asset} futures taker series ...", end=" ", flush=True)
        series_cache[asset] = fetch_futures_taker_series(asset)
        s = series_cache[asset]
        print(f"{len(s)} bars  "
              f"({s['bar_open'].min().date()} → {s['bar_open'].max().date()})")
        time.sleep(0.2)

    for csv_path, ts_col, asset in TARGETS:
        if not csv_path.exists():
            print(f"  SKIP (not found): {csv_path.name}")
            continue
        n_rows = len(pd.read_csv(csv_path, usecols=[ts_col]))
        print(f"  {csv_path.name} ({n_rows:,} rows) ...", end=" ", flush=True)
        filled = backfill_csv(csv_path, ts_col, series_cache[asset])
        print(f"filled {filled:,} / {n_rows:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
