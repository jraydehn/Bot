#!/usr/bin/env python3
"""S2 — Point-in-time historical flow/positioning from CoinGlass v4, 1h bars.

Tier depth probe (2026-07-04): earliest allowed start_time = 1767631657000
(2026-01-05). Coinalyze free tier is SHALLOWER (OI history only from
~2026-05-01), so CoinGlass is the primary flow source; Coinalyze skipped.

Endpoints (BTC, interval=1h):
  futures/aggregated-taker-buy-sell-volume/history  (exchange_list=Binance,OKX,Bybit)
  spot/aggregated-taker-buy-sell-volume/history     (exchange_list=Binance,OKX,Coinbase)
  futures/open-interest/aggregated-history
  futures/funding-rate/oi-weight-history
  futures/liquidation/aggregated-history            (exchange_list=Binance,OKX,Bybit)

Timestamps are bar OPEN times (ms). Bar t completes at t+1h — the same
convention as the backbone (row T decision time = T+1h), so bar T merges onto
backbone row T with no extra shift. The last (possibly partial) bar of each
fetch is dropped.

Output: cg_flow_sol_1h.parquet
"""
import time
from pathlib import Path
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
KEY = "8f0a30c29a5e424ba2641f649051786b"
BASE = "https://open-api-v4.coinglass.com/api"
H = {"CG-API-KEY": KEY}
START_MS = 1767631660000
NOW_MS = int(time.time() * 1000)

def fetch(path, params, rename):
    cur, out = START_MS, []
    while cur < NOW_MS:
        p = dict(params, start_time=cur, end_time=NOW_MS, limit=2000)
        for attempt in range(5):
            try:
                r = requests.get(f"{BASE}/{path}", headers=H, params=p, timeout=20)
                j = r.json()
                if j.get("code") != "0":
                    raise RuntimeError(j)
                break
            except Exception as e:
                print(f"retry {attempt} {path} @{cur}: {e}", flush=True)
                time.sleep(3 + 3 * attempt)
        else:
            raise RuntimeError(f"failed {path}")
        data = j["data"]
        if not data:
            break
        out.extend(data)
        nxt = data[-1]["time"] + 3_600_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(1.2)
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    df.index.name = "open_time"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.rename(columns=rename)[list(rename.values())].astype(float)
    return df.iloc[:-1]  # drop possibly-partial last bar

parts = []
parts.append(fetch("futures/aggregated-taker-buy-sell-volume/history",
    {"symbol": "SOL", "interval": "1h", "exchange_list": "Binance,OKX,Bybit"},
    {"aggregated_buy_volume_usd": "fut_buy_usd", "aggregated_sell_volume_usd": "fut_sell_usd"}))
print("fut taker done", flush=True)
parts.append(fetch("spot/aggregated-taker-buy-sell-volume/history",
    {"symbol": "SOL", "interval": "1h", "exchange_list": "Binance,OKX,Coinbase"},
    {"aggregated_buy_volume_usd": "spot_buy_usd", "aggregated_sell_volume_usd": "spot_sell_usd"}))
print("spot taker done", flush=True)
parts.append(fetch("futures/open-interest/aggregated-history",
    {"symbol": "SOL", "interval": "1h"},
    {"close": "oi_close", "high": "oi_high", "low": "oi_low"}))
print("OI done", flush=True)
parts.append(fetch("futures/funding-rate/oi-weight-history",
    {"symbol": "SOL", "interval": "1h"},
    {"close": "funding_close"}))
print("funding done", flush=True)
parts.append(fetch("futures/liquidation/aggregated-history",
    {"symbol": "SOL", "interval": "1h", "exchange_list": "Binance,OKX,Bybit"},
    {"aggregated_long_liquidation_usd": "liq_long_usd",
     "aggregated_short_liquidation_usd": "liq_short_usd"}))
print("liq done", flush=True)

df = pd.concat(parts, axis=1).sort_index()
df.to_parquet(OUT / "cg_flow_sol_1h.parquet")
print(f"saved cg_flow_sol_1h.parquet: {len(df):,} rows {df.index[0]} -> {df.index[-1]}")
print(df.isna().mean().round(4).to_string())
print("S2 DONE")
