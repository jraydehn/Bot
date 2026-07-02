"""
backfill_composite.py — Recompute composite_trend, composite_rev, composite_p_up
for resolved paper trade rows that are missing those values.

Uses historical parquet files to replay compute_current_scores() at each
logged_at timestamp with no lookahead bias.

Usage:
    python3 backfill_composite.py                  # all assets
    python3 backfill_composite.py --asset BTC      # single asset
    python3 backfill_composite.py --dry-run        # preview without writing
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_point import _find_parquet
from composite_scorer import compute_current_scores, lookup_p_up
from paper_trade_runner import get_csv_path

DATA_DIR = Path(__file__).parent / "data"
ASSET_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

CSV_COLUMNS = [
    "logged_at", "decision_time", "contract_ticker", "close_ts",
    "spot", "strike", "offset_pct", "p_market", "p_market_source",
    "p_yes_model", "z_score", "vol_60m", "vol_60m_model", "vol_implied_kalshi", "vol_ratio", "spread", "vol_eff",
    "structure_bias", "confirmation_bias", "confirmation_score", "no_score",
    "obi_score", "obi_raw", "obi_exchanges",
    "vpin_score", "vpin_raw",
    "funding_bias", "avg_funding_rate",
    "vol_score", "vwap_score", "vwap_signal", "vwap_total", "vwap_stretch_score", "vwap_distance_pct", "bearish_rejection", "bullish_rejection", "ema_stretch_score",
    "stoch_bias", "stoch_k", "stoch_d", "stoch_crossover_active",
    "ema_stack_bias",
    "ema_alignment", "z_shift", "direction_strength", "raw_edge", "net_edge",
    "decision", "side", "neutral_gate", "pure_edge_gate",
    "contracts_scanned", "tau_minutes", "gate_blocked",
    "kelly_fraction", "bet_fraction", "bet_amount", "bankroll",
    "composite_trend", "composite_rev", "composite_p_up",
    "resolved_yes", "would_win", "would_pnl",
]


def load_ohlcv(asset: str) -> tuple:
    """Load 1h and 1m parquets for the given asset. Returns (df_1h, df_1m)."""
    symbol = ASSET_SYMBOLS[asset]
    path_1h = _find_parquet("1h", symbol)
    path_1m = _find_parquet("1m", symbol)
    print(f"  [load] 1h: {path_1h.name}")
    print(f"  [load] 1m: {path_1m.name}")
    df_1h = pd.read_parquet(path_1h)
    df_1m = pd.read_parquet(path_1m)
    for df in (df_1h, df_1m):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    return df_1h, df_1m


def recompute_scores(ts: pd.Timestamp, df_1h: pd.DataFrame, df_1m: pd.DataFrame, asset: str):
    """
    Replay composite scorer at timestamp ts with no lookahead.
    Returns (trend_score, rev_score, p_up) or raises on failure.
    """
    # Slice strictly up to ts (inclusive of ts bar since it was the live bar at trade time)
    h1_slice = df_1h[df_1h.index <= ts]
    m1_slice = df_1m[df_1m.index <= ts].iloc[-1700:]

    if len(h1_slice) < 50:
        raise ValueError(f"Too few 1h bars ({len(h1_slice)}) before {ts}")
    if len(m1_slice) < 200:
        raise ValueError(f"Too few 1m bars ({len(m1_slice)}) before {ts}")

    # Derive 4h and 15m the same way paper_trade_runner does
    df_4h = h1_slice.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    df_15m = m1_slice.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    trend, rev = compute_current_scores(
        h1_slice, df_4h, df_15m,
        m1_slice["close"].astype(float),
        m1_slice["volume"].astype(float),
    )
    p_up = lookup_p_up(trend, rev, asset=asset)
    return trend, rev, p_up


def parse_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def backfill_asset(asset: str, dry_run: bool = False) -> int:
    csv_path = get_csv_path(asset)
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return 0

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Find rows needing backfill: resolved trades with missing composite columns
    needs_backfill = [
        r for r in rows
        if r.get("decision", "").strip() == "trade"
        and (r.get("resolved_yes") or "").strip()
        and not (r.get("composite_trend") or "").strip()
    ]

    if not needs_backfill:
        print(f"  [skip] No rows need backfill for {asset}")
        return 0

    print(f"  [{asset}] {len(needs_backfill)} rows to backfill")
    df_1h, df_1m = load_ohlcv(asset)

    filled = 0
    failed = 0
    for row in needs_backfill:
        try:
            ts = parse_ts(row["logged_at"])
        except Exception as e:
            print(f"    [warn] Cannot parse logged_at={row['logged_at']!r}: {e}")
            failed += 1
            continue

        try:
            trend, rev, p_up = recompute_scores(ts, df_1h, df_1m, asset)
        except Exception as e:
            print(f"    [warn] {ts}: {e}")
            failed += 1
            continue

        row["composite_trend"] = str(trend)
        row["composite_rev"]   = str(rev)
        row["composite_p_up"]  = str(round(p_up, 6))
        filled += 1
        print(f"    {ts}  trend={trend:+d}  rev={rev:+d}  p_up={p_up:.3f}")

    print(f"  [{asset}] filled={filled}  failed={failed}")

    if not dry_run and filled > 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [{asset}] Saved {csv_path.name}")

    return filled


def main():
    parser = argparse.ArgumentParser(description="Backfill composite scores for resolved trades")
    parser.add_argument("--asset", type=str, default=None, help="BTC, ETH, or SOL (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing CSV")
    args = parser.parse_args()

    assets = [args.asset.upper()] if args.asset else ["BTC", "ETH", "SOL"]
    total = 0
    for asset in assets:
        print(f"\n=== {asset} ===")
        total += backfill_asset(asset, dry_run=args.dry_run)

    print(f"\nTotal rows filled: {total}")
    if args.dry_run:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
