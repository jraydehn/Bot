"""
backfill_expiry_prices.py

Backfills spot_at_expiry, price_move_pct, and miss_pct for all already-resolved
rows across every live paper-trade and scan-archive CSV.

Fetches the Binance 1m close price at each contract's expiry timestamp.
Skips rows where spot_at_expiry is already populated.

Run once:
    python3 backfill_expiry_prices.py [--dry-run]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from live_signal import fetch_spot_at_time

RESULTS = Path(__file__).parent / "results"

# (filename, asset, close_ts_col, strike_col)
TARGETS = [
    # 1h paper trades
    ("paper_trades.csv",     "BTC", "close_ts",   "strike"),
    ("paper_trades_eth.csv", "ETH", "close_ts",   "strike"),
    ("paper_trades_sol.csv", "SOL", "close_ts",   "strike"),
    # 15m paper trades (active)
    ("paper_trades_btc15m.csv",  "BTC", "close_time", "floor_strike"),
    ("paper_trades_eth15m.csv",  "ETH", "close_time", "floor_strike"),
    ("paper_trades_sol15m.csv",  "SOL", "close_time", "floor_strike"),
    # 15m paper trades (legacy)
    ("paper_trades_btc15.csv",   "BTC", "close_time", "floor_strike"),
    # Scan archives
    ("btc_scan_archive.csv",     "BTC", "close_ts",   "strike"),
    ("eth_scan_archive.csv",     "ETH", "close_ts",   "strike"),
    ("sol_scan_archive.csv",     "SOL", "close_ts",   "strike"),
    ("btc_scan_archive_15m.csv", "BTC", "close_ts",   "strike"),
    ("eth_scan_archive_15m.csv", "ETH", "close_ts",   "strike"),
    ("sol_scan_archive_15m.csv", "SOL", "close_ts",   "strike"),
]

NEW_COLS = ["spot_at_expiry", "price_move_pct", "miss_pct"]


def process_file(path: Path, asset: str, ts_col: str, strike_col: str,
                 price_cache: dict, dry_run: bool) -> int:
    if not path.exists():
        print(f"  [skip] {path.name} — not found")
        return 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print(f"  [skip] {path.name} — empty")
        return 0

    # Add missing columns
    added_cols = [c for c in NEW_COLS if c not in cols]
    all_cols = cols + added_cols

    filled = 0
    for row in rows:
        if not (row.get("resolved_yes") or "").strip():
            continue
        if (row.get("spot_at_expiry") or "").strip():
            continue

        close_ts = (row.get(ts_col) or "").strip()
        spot_str  = (row.get("spot") or "").strip()
        strike_str = (row.get(strike_col) or "").strip()

        if not close_ts or not spot_str:
            continue

        if dry_run:
            filled += 1
            continue

        cache_key = (close_ts, asset)
        if cache_key not in price_cache:
            price_cache[cache_key] = fetch_spot_at_time(close_ts, asset)
            time.sleep(0.08)  # ~12 req/s — well under Binance 1200/min limit

        spot_exp = price_cache[cache_key]
        if spot_exp is None:
            continue

        try:
            spot_scan = float(spot_str)
            if spot_scan > 0:
                row["spot_at_expiry"] = round(spot_exp, 2)
                row["price_move_pct"] = round((spot_exp - spot_scan) / spot_scan * 100, 4)
        except (ValueError, TypeError):
            pass

        try:
            strike = float(strike_str)
            if strike > 0:
                row["miss_pct"] = round((spot_exp - strike) / strike * 100, 4)
        except (ValueError, TypeError):
            pass

        filled += 1

    if filled == 0:
        print(f"  {path.name}: nothing to fill")
        return 0

    if dry_run:
        print(f"  [dry-run] {path.name}: would fill {filled} rows ({asset})")
        return filled

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"  {path.name}: filled {filled} rows ({asset})")
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    price_cache: dict = {}
    total = 0

    for filename, asset, ts_col, strike_col in TARGETS:
        path = RESULTS / filename
        total += process_file(path, asset, ts_col, strike_col,
                              price_cache, args.dry_run)

    print(f"\nTotal rows filled: {total}")


if __name__ == "__main__":
    main()
