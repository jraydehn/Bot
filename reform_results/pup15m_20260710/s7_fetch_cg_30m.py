"""
S7 -- fetch CoinGlass v4 flow/positioning for BTC at 30m interval (the finest
the current plan allows; 15m/5m are 403). Depth cap ~90 days rolling
(probed 2026-07-10: first available bar 2026-04-11). Same endpoints and
conventions as the validated hourly fetch (s2_fetch_flow.py, pup_v2_rebuild):
timestamps are bar OPEN (ms); bar t completes at t+30m; last partial bar
of each fetch dropped. Output: cg_flow_btc_30m.parquet
"""
import time
from pathlib import Path
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
KEY = "8f0a30c29a5e424ba2641f649051786b"
BASE = "https://open-api-v4.coinglass.com/api"
H = {"CG-API-KEY": KEY}
NOW_MS = int(time.time() * 1000)
START_MS = NOW_MS - 92 * 24 * 3600 * 1000

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
        nxt = data[-1]["time"] + 1_800_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(1.2)
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.rename(columns=rename)[list(rename.values())]
    return df.iloc[:-1]  # drop last (possibly partial) bar

parts = []
parts.append(fetch("futures/aggregated-taker-buy-sell-volume/history",
                   {"symbol": "BTC", "interval": "30m", "exchange_list": "Binance,OKX,Bybit"},
                   {"aggregated_buy_volume_usd": "fut_buy_usd",
                    "aggregated_sell_volume_usd": "fut_sell_usd"}))
print("futures taker:", len(parts[-1]), flush=True)
parts.append(fetch("spot/aggregated-taker-buy-sell-volume/history",
                   {"symbol": "BTC", "interval": "30m", "exchange_list": "Binance,OKX,Coinbase"},
                   {"aggregated_buy_volume_usd": "spot_buy_usd",
                    "aggregated_sell_volume_usd": "spot_sell_usd"}))
print("spot taker:", len(parts[-1]), flush=True)
parts.append(fetch("futures/open-interest/aggregated-history",
                   {"symbol": "BTC", "interval": "30m"},
                   {"close": "oi_close"}))
print("oi:", len(parts[-1]), flush=True)
parts.append(fetch("futures/liquidation/aggregated-history",
                   {"symbol": "BTC", "interval": "30m", "exchange_list": "Binance,OKX,Bybit"},
                   {"aggregated_long_liquidation_usd": "liq_long_usd",
                    "aggregated_short_liquidation_usd": "liq_short_usd"}))
print("liq:", len(parts[-1]), flush=True)

df = pd.concat(parts, axis=1)
print("merged:", df.shape, df.index.min(), "->", df.index.max())
print("nulls per col:", df.isna().sum().to_dict())
df.to_parquet(OUT / "cg_flow_btc_30m.parquet")
print("DONE_S7")
