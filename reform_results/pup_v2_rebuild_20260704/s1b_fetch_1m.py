#!/usr/bin/env python3
"""S1b — BTC 1m klines 2020-01 -> now from BinanceUS (for intra-hour
microstructure features). Saved yearly then merged to hist_BTCUSDT_1m.parquet."""
import time
from pathlib import Path
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
BASE = "https://api.binance.us/api/v3/klines"

def fetch_range(start_ms, end_ms):
    rows, cur = [], start_ms
    sess = requests.Session()
    while cur < end_ms:
        for attempt in range(6):
            try:
                r = sess.get(BASE, params={"symbol": "BTCUSDT", "interval": "1m",
                                           "startTime": cur, "limit": 1000}, timeout=15)
                if r.status_code == 429:
                    time.sleep(15); continue
                r.raise_for_status()
                batch = r.json(); break
            except Exception as e:
                print(f"retry {attempt} @{cur}: {e}", flush=True)
                time.sleep(3 + 3 * attempt)
        else:
            raise RuntimeError(f"failed @{cur}")
        if not batch:
            cur += 1000 * 60_000
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + 60_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    return rows

frames = []
for yr in range(2020, 2027):
    s = int(pd.Timestamp(f"{yr}-01-01", tz="UTC").timestamp() * 1000)
    e = min(int(pd.Timestamp(f"{yr+1}-01-01", tz="UTC").timestamp() * 1000),
            int(time.time() * 1000))
    t0 = time.time()
    rows = fetch_range(s, e)
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume",
                                     "close_time","qvol","ntrades","tbb","tbq","ig"])
    df = df[["open_time","open","high","low","close","volume","ntrades","tbb"]]
    for c in ("open","high","low","close","volume","tbb"):
        df[c] = df[c].astype(float)
    df["ntrades"] = df["ntrades"].astype(int)
    df.rename(columns={"tbb": "taker_buy_vol"}, inplace=True)
    df.index = pd.to_datetime(df.pop("open_time"), unit="ms", utc=True)
    df.index.name = "open_time"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(OUT / f"_1m_{yr}.parquet")
    frames.append(df)
    print(f"{yr}: {len(df):,} bars ({time.time()-t0:.0f}s)", flush=True)

full = pd.concat(frames)
full = full[~full.index.duplicated(keep="first")].sort_index()
full.to_parquet(OUT / "hist_BTCUSDT_1m.parquet")
print(f"TOTAL {len(full):,} bars {full.index[0]} -> {full.index[-1]}")
print("S1B DONE")
