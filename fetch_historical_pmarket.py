"""
Fetch real Kalshi p_market_yes_open for every other historical decision point.

For each sampled hourly timestamp in the backtest window:
  1. Compute spot price and 0.5% OTM strike from OHLCV data.
  2. Find the matching KXBTCD contract on Kalshi for that strike/hour.
  3. Fetch the opening candlestick to get the real YES market price.
  4. Save to results/historical_pmarket.csv for use in backtesting.

Usage:
    export KALSHI_KEY_ID=your-key-id
    export KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
    python3 fetch_historical_pmarket.py
    python3 fetch_historical_pmarket.py --offset 0.005 --step 2 --out results/historical_pmarket.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from evaluate_point import load_data

BASE_URL       = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER  = "KXBTCD"
SLEEP_BETWEEN  = 0.25   # seconds between API calls
CANDLE_WINDOW  = 300    # 5-minute window to find opening candlestick
DEFAULT_OFFSET = 0.005
DEFAULT_STEP   = 2      # sample every Nth decision point (2 = half)
DEFAULT_OUT    = "results/historical_pmarket.csv"
MIN_WARMUP_4H  = 89     # must match backtest warmup


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_auth() -> Optional[KalshiAuth]:
    key_id   = os.environ.get("KALSHI_KEY_ID", "").strip()
    key_path = os.environ.get("KALSHI_KEY_PATH", "").strip()
    if not key_id or not key_path:
        print("  No credentials — set KALSHI_KEY_ID and KALSHI_KEY_PATH.")
        return None
    pem = Path(key_path).expanduser().read_text()
    return KalshiAuth(key_id, pem)


def kalshi_get(path: str, params: dict, auth: KalshiAuth) -> dict:
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    headers.update(auth.create_auth_headers("GET", url))
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if not resp.ok:
        return {}
    return resp.json()


# ---------------------------------------------------------------------------
# Find the Kalshi contract matching a given hour + strike
# ---------------------------------------------------------------------------

def find_contract(auth: KalshiAuth, target_ts: pd.Timestamp, strike: float) -> Optional[dict]:
    """
    Search settled KXBTCD markets for a contract whose open_time matches
    target_ts (within ±5 minutes) and whose floor_strike is within $200 of strike.
    Returns the raw market dict or None.
    """
    # Round strike to nearest $100 to match Kalshi contract increments
    strike_lo = strike - 200
    strike_hi = strike + 200

    params = {"series_ticker": SERIES_TICKER, "status": "settled", "limit": 200}
    data = kalshi_get("/markets", params, auth)
    markets = data.get("markets") or []

    best = None
    best_dist = float("inf")

    for m in markets:
        open_raw = m.get("open_time")
        if not open_raw:
            continue
        try:
            open_dt = pd.Timestamp(open_raw, tz="UTC")
        except Exception:
            continue

        # Must be within 5 minutes of target
        diff_min = abs((open_dt - target_ts).total_seconds()) / 60
        if diff_min > 5:
            continue

        # Strike must be in range
        fs = m.get("floor_strike")
        if fs is None:
            continue
        try:
            fs = float(fs)
        except (ValueError, TypeError):
            continue

        if not (strike_lo <= fs <= strike_hi):
            continue

        dist = abs(fs - strike)
        if dist < best_dist:
            best_dist = dist
            best = m

    return best


# ---------------------------------------------------------------------------
# Fetch opening candlestick YES price for a contract
# ---------------------------------------------------------------------------

def fetch_opening_price(auth: KalshiAuth, ticker: str, open_time: str) -> Optional[float]:
    """
    Fetch the 1-minute candlestick at contract open and return the YES mid-price.
    """
    try:
        open_dt = pd.Timestamp(open_time, tz="UTC").to_pydatetime()
    except Exception:
        return None

    start_ts = int(open_dt.timestamp())
    end_ts   = start_ts + CANDLE_WINDOW

    path   = f"/series/{SERIES_TICKER}/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1}
    data   = kalshi_get(path, params, auth)
    candles = data.get("candlesticks") or []

    for c in candles:
        bid_d = c.get("yes_bid") or {}
        ask_d = c.get("yes_ask") or {}
        try:
            bid = float(bid_d.get("open_dollars", 0) or 0)
            ask = float(ask_d.get("open_dollars", 0) or 0)
        except (ValueError, TypeError):
            continue
        if ask > 0:
            return (bid + ask) / 2.0

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch real Kalshi p_market for sampled backtest decision points."
    )
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET,
                        help=f"Strike offset above spot (default {DEFAULT_OFFSET})")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help=f"Sample every Nth decision point (default {DEFAULT_STEP} = every other)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output CSV path (default {DEFAULT_OUT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max decision points to process (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("  HISTORICAL p_market FETCH")
    print("=" * 60)

    auth = load_auth()
    if auth is None:
        return

    print("\nLoading OHLCV data...")
    df_1m, df_1h, df_4h = load_data()

    # Build decision timestamps — same logic as backtest.py
    decision_times = df_1h.index.copy()
    min_history_ts = df_4h.index[MIN_WARMUP_4H]
    decision_times = decision_times[decision_times > min_history_ts]

    # Sample every Nth point
    decision_times = decision_times[::args.step]
    if args.limit:
        decision_times = decision_times[:args.limit]

    total = len(decision_times)
    print(f"\n  Decision points (sampled 1/{args.step}): {total:,}")
    print(f"  Strike offset : {args.offset:.3%} above spot")
    print(f"  Est. time     : ~{total * SLEEP_BETWEEN / 60:.0f} minutes\n")

    records  = []
    found    = 0
    missing  = 0

    for i, ts in enumerate(decision_times, 1):
        if i % 50 == 0:
            print(f"  [{i:>5}/{total}]  found={found}  missing={missing}")

        # Spot price at this timestamp
        hist_1m = df_1m.loc[df_1m.index <= ts]
        if hist_1m.empty:
            missing += 1
            continue
        spot   = float(hist_1m["close"].iloc[-1])
        strike = spot * (1 + abs(args.offset))

        # Refresh auth every 25 min
        elapsed = time.time()
        auth2 = KalshiAuth(
            os.environ.get("KALSHI_KEY_ID", ""),
            Path(os.environ.get("KALSHI_KEY_PATH", "")).expanduser().read_text()
        ) if elapsed % (25 * 60) < SLEEP_BETWEEN else auth

        # Find matching Kalshi contract
        contract = find_contract(auth, ts, strike)
        time.sleep(SLEEP_BETWEEN)

        if contract is None:
            missing += 1
            records.append({
                "decision_time": ts,
                "spot":          spot,
                "strike":        strike,
                "ticker":        None,
                "p_market_yes":  None,
            })
            continue

        ticker    = contract.get("ticker", "")
        open_time = contract.get("open_time", "")

        # Fetch opening candlestick
        p_market = fetch_opening_price(auth, ticker, open_time)
        time.sleep(SLEEP_BETWEEN)

        if p_market is not None:
            found += 1
        else:
            missing += 1

        records.append({
            "decision_time": ts,
            "spot":          spot,
            "strike":        strike,
            "ticker":        ticker,
            "p_market_yes":  p_market,
        })

    print(f"\n  Done: {found} found, {missing} missing out of {total}")

    df = pd.DataFrame(records)
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    if df["p_market_yes"].notna().any():
        col = df["p_market_yes"].dropna()
        print(f"\n  p_market_yes stats:")
        print(f"    mean={col.mean():.3f}  median={col.median():.3f}")
        print(f"    min={col.min():.3f}   max={col.max():.3f}  n={len(col):,}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
