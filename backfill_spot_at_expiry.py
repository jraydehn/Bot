"""
backfill_spot_at_expiry.py — Fill missing spot_at_expiry / price_move_pct / miss_pct
for all resolved rows across BTC/ETH/SOL 15m paper trade CSVs.

Binance has full 1m historical data so every close_time is recoverable.
Run once; safe to re-run (skips rows that already have spot_at_expiry).
"""

import time
from pathlib import Path

import pandas as pd
import requests

RESULTS = Path(__file__).parent / "results"

ASSET_CONFIG = {
    "BTC": {"csv": RESULTS / "paper_trades_btc15m.csv", "symbol": "BTCUSDT"},
    "ETH": {"csv": RESULTS / "paper_trades_eth15m.csv", "symbol": "ETHUSDT"},
    "SOL": {"csv": RESULTS / "paper_trades_sol15m.csv", "symbol": "SOLUSDT"},
}


def fetch_price_at(close_dt: pd.Timestamp, symbol: str) -> "float | None":
    end_ms = int(close_dt.timestamp() * 1000)
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "endTime": end_ms, "limit": 2},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[-1][4])
    except Exception as e:
        print(f"    [warn] Binance fetch failed: {e}")
    return None


def backfill_asset(asset: str) -> None:
    cfg = ASSET_CONFIG[asset]
    path = cfg["csv"]
    symbol = cfg["symbol"]

    if not path.exists():
        print(f"[{asset}] CSV not found: {path}")
        return

    df = pd.read_csv(path, low_memory=False)
    mask = df["resolved_yes"].notna() & (df["resolved_yes"].astype(str) != "")
    targets = df[mask]
    print(f"[{asset}] {len(targets)} rows to backfill  ({len(df)} total)")

    filled = 0
    for idx, row in targets.iterrows():
        ct_str = str(row.get("close_time", ""))
        spot_scan = float(row.get("spot", 0) or 0)
        floor_s   = float(row.get("floor_strike", 0) or 0)

        if not ct_str or spot_scan <= 0:
            continue

        try:
            close_dt = pd.Timestamp(ct_str).tz_convert("UTC")
        except Exception:
            print(f"  [skip] bad close_time: {ct_str}")
            continue

        spot_exp = fetch_price_at(close_dt, symbol)
        if spot_exp is None:
            print(f"  [miss] {ct_str} — no price returned")
            continue

        df.at[idx, "spot_at_expiry"] = round(spot_exp, 4)
        if spot_scan > 0:
            df.at[idx, "price_move_pct"] = round((spot_exp - spot_scan) / spot_scan * 100, 4)
        if floor_s > 0:
            df.at[idx, "miss_pct"] = round((spot_exp - floor_s) / floor_s * 100, 4)

        filled += 1
        if filled % 10 == 0:
            print(f"  ... {filled}/{len(targets)} filled")

        time.sleep(0.08)  # ~12 req/s, well under Binance 1200/min limit

    df.to_csv(path, index=False)
    print(f"[{asset}] Done — filled {filled}/{len(targets)} rows")


if __name__ == "__main__":
    for asset in ["BTC", "ETH", "SOL"]:
        backfill_asset(asset)
        print()
