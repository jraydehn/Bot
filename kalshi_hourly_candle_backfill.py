"""Backfill 1-min Kalshi candlesticks for settled HOURLY markets. 2026-08-04.

Clone of kalshi_15m_candle_backfill for the hourly series (KXBTCD/KXETHD/
KXSOLD) with a 65-minute candle window per market. Purpose: per-minute
bid/ask ground truth for the MAKER-EXECUTION study — simulating whether
resting limit orders would have filled (price trading through the limit)
and at what effective cost, for the composite-signal book and the thin-edge
signal class generally.

Output: results/kalshi_1h_candles_{asset}.csv
Usage: python3 kalshi_hourly_candle_backfill.py SOL [days_back=45]
"""
import sys
import time
import csv as _csv
from pathlib import Path

import pandas as pd

from live_signal import load_auth, kalshi_get

BASE = Path(__file__).parent
SERIES = {"BTC": "KXBTCD", "ETH": "KXETHD", "SOL": "KXSOLD"}
SLEEP = 0.13

COLS = ["ticker", "close_time", "result", "floor_strike", "end_ts",
        "bid_close", "ask_close", "price_close", "price_low", "price_high",
        "volume_fp", "oi_fp"]


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
    asset = (sys.argv[1] if len(sys.argv) > 1 else "SOL").upper()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    series = SERIES[asset]
    out_path = BASE / "results" / f"kalshi_1h_candles_{asset.lower()}.csv"
    auth = load_auth()

    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["ticker"], low_memory=False)
                   ["ticker"].unique())
    else:
        out_path.write_text(",".join(COLS) + "\n")
    print(f"[{asset}/{series}] {len(done)} tickers already fetched")

    now = int(time.time())
    markets = []
    for w_end in range(now, now - days * 86400, -5 * 86400):
        w_start = max(w_end - 5 * 86400, now - days * 86400)
        mk = list_settled(series, auth, w_start, w_end)
        markets.extend(mk)
        print(f"  listed {len(mk):4d} settled {pd.Timestamp(w_start, unit='s')} "
              f".. {pd.Timestamp(w_end, unit='s')}")
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
                           {"start_ts": ct - 3900, "end_ts": ct,
                            "period_interval": 1}, auth)
            for cd in c.get("candlesticks", []):
                pr = cd.get("price") or {}
                w.writerow([m["ticker"], m["close_time"], m.get("result", ""),
                            m.get("floor_strike", ""), cd.get("end_period_ts", ""),
                            (cd.get("yes_bid") or {}).get("close_dollars", ""),
                            (cd.get("yes_ask") or {}).get("close_dollars", ""),
                            pr.get("close_dollars", ""), pr.get("low_dollars", ""),
                            pr.get("high_dollars", ""),
                            cd.get("volume_fp", ""), cd.get("open_interest_fp", "")])
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  [err] {m['ticker']}: {e}")
        if i % 300 == 0:
            f.flush()
            print(f"  {i}/{len(markets)} (ok={n_ok} err={n_err})")
        time.sleep(SLEEP)
    f.close()
    print(f"[{asset}] DONE: ok={n_ok} err={n_err} → {out_path}")


if __name__ == "__main__":
    main()
