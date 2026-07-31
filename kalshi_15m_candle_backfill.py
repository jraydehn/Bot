"""Backfill 1-min Kalshi candlesticks for settled 15m markets. 2026-07-31.

Purpose: reconstruct months of 15m microstructure history (per-contract pm
trajectory at 1-min resolution + full cross-strike ladder) that our scan
archives only started logging ~07-22. Quote history is point-in-time fact —
the legitimate backfill class (like the 4h stoch/rsi backfill; NOT the HMM
class).

Output: results/kalshi_15m_candles_{asset}.csv, one row per (market, minute):
  ticker, close_time, result, floor_strike, end_ts, bid_close, ask_close,
  price_close, volume_fp, oi_fp
Resume-safe: already-fetched tickers are skipped; flushes continuously.

Usage: python3 kalshi_15m_candle_backfill.py BTC [days_back=60]
"""
import sys
import time
import csv as _csv
from pathlib import Path

import pandas as pd

from live_signal import load_auth, kalshi_get

BASE = Path(__file__).parent
SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M"}
SLEEP = 0.13  # ~7.5 req/s, under Kalshi rate limits

COLS = ["ticker", "close_time", "result", "floor_strike", "end_ts",
        "bid_close", "ask_close", "price_close", "volume_fp", "oi_fp"]


def list_settled(series: str, auth, min_ts: int, max_ts: int) -> list:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 200,
                  "min_close_ts": min_ts, "max_close_ts": max_ts}
        if cursor:
            params["cursor"] = cursor
        r = kalshi_get("/markets", params, auth)
        mk = r.get("markets", [])
        out.extend(mk)
        cursor = r.get("cursor")
        if not cursor or not mk:
            return out
        time.sleep(SLEEP)


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    series = SERIES[asset]
    out_path = BASE / "results" / f"kalshi_15m_candles_{asset.lower()}.csv"
    auth = load_auth()

    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["ticker"], low_memory=False)
                   ["ticker"].unique())
    else:
        out_path.write_text(",".join(COLS) + "\n")
    print(f"[{asset}] {len(done)} tickers already fetched")

    now = int(time.time())
    markets = []
    # list in 5-day windows to keep pagination bounded
    for w_end in range(now, now - days * 86400, -5 * 86400):
        w_start = max(w_end - 5 * 86400, now - days * 86400)
        mk = list_settled(series, auth, w_start, w_end)
        markets.extend(mk)
        print(f"  listed {len(mk):4d} settled markets "
              f"{pd.Timestamp(w_start, unit='s')} .. {pd.Timestamp(w_end, unit='s')}")
        time.sleep(SLEEP)
    seen = set()
    markets = [m for m in markets
               if m["ticker"] not in done and not
               (m["ticker"] in seen or seen.add(m["ticker"]))]
    print(f"[{asset}] {len(markets)} markets to fetch")

    f = open(out_path, "a", newline="")
    w = _csv.writer(f)
    n_ok = n_err = 0
    for i, m in enumerate(markets):
        try:
            ct = int(pd.Timestamp(m["close_time"]).timestamp())
            c = kalshi_get(f"/series/{series}/markets/{m['ticker']}/candlesticks",
                           {"start_ts": ct - 1260, "end_ts": ct,
                            "period_interval": 1}, auth)
            for cd in c.get("candlesticks", []):
                w.writerow([m["ticker"], m["close_time"], m.get("result", ""),
                            m.get("floor_strike", ""), cd.get("end_period_ts", ""),
                            (cd.get("yes_bid") or {}).get("close_dollars", ""),
                            (cd.get("yes_ask") or {}).get("close_dollars", ""),
                            (cd.get("price") or {}).get("close_dollars", ""),
                            cd.get("volume_fp", ""), cd.get("open_interest_fp", "")])
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  [err] {m['ticker']}: {e}")
        if i % 200 == 0:
            f.flush()
            print(f"  {i}/{len(markets)} fetched (ok={n_ok} err={n_err})")
        time.sleep(SLEEP)
    f.close()
    print(f"[{asset}] DONE: ok={n_ok} err={n_err} → {out_path}")


if __name__ == "__main__":
    main()
