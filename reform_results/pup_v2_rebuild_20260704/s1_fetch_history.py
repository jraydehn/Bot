#!/usr/bin/env python3
"""S1 — Extend BinanceUS OHLCV history back to 2020 (BTC/ETH/SOL 1h, BTC 15m,
alt-basket 1h: XRP/DOGE/ADA) and splice with the existing 2024+ parquets.

Same source (BinanceUS klines API) serves 2020+ directly, so no cross-source
splice is required; we still validate the overlap window (2024-01..2024-03)
against the existing repo parquets (require corr>0.999 & report max rel diff).
4h frames are built by resampling the merged 1h (UTC 00-aligned, open-time
indexed) and validated against the existing repo 4h parquet.

Outputs (this dir):
  hist_BTCUSDT_1h.parquet, hist_ETHUSDT_1h.parquet, hist_SOLUSDT_1h.parquet
  hist_BTCUSDT_4h.parquet
  hist_BTCUSDT_15m.parquet
  hist_XRPUSDT_1h.parquet, hist_DOGEUSDT_1h.parquet, hist_ADAUSDT_1h.parquet
  s1_splice_report.txt
"""
import time, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
BASE = "https://api.binance.us/api/v3/klines"

INTERVAL_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}

def fetch(symbol, interval, start_ms, end_ms):
    rows, cur = [], start_ms
    step = INTERVAL_MS[interval]
    sess = requests.Session()
    while cur < end_ms:
        for attempt in range(5):
            try:
                r = sess.get(BASE, params={"symbol": symbol, "interval": interval,
                                           "startTime": cur, "limit": 1000}, timeout=15)
                if r.status_code == 429:
                    time.sleep(10); continue
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                print(f"  retry {attempt} {symbol} {interval} @{cur}: {e}", flush=True)
                time.sleep(3 + 3 * attempt)
        else:
            raise RuntimeError(f"fetch failed {symbol} {interval} @{cur}")
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
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
    return df

def latest_repo_pq(sym, tf):
    pats = [f"binanceus_{sym}_{tf}_1970*.parquet", f"binanceus_{sym}_{tf}_2024*.parquet"]
    files = []
    for p in pats:
        files += glob.glob(str(PROJ / "data" / p))
    best, best_end = None, None
    for f in sorted(files):
        d = pd.read_parquet(f)
        if len(d) < 1000:
            continue
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        d = d[d.index >= pd.Timestamp("2000-01-01", tz="UTC")]  # drop epoch-0 junk
        if best_end is None or d.index[-1] > best_end:
            best, best_end = d, d.index[-1]
    return best

rep = open(OUT / "s1_splice_report.txt", "w")
def log(msg):
    print(msg, flush=True); rep.write(msg + "\n"); rep.flush()

START = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
NOW = int(time.time() * 1000)

jobs = [("BTCUSDT","1h"), ("ETHUSDT","1h"), ("SOLUSDT","1h"),
        ("XRPUSDT","1h"), ("DOGEUSDT","1h"), ("ADAUSDT","1h"),
        ("BTCUSDT","15m")]

for sym, tf in jobs:
    t0 = time.time()
    fresh = fetch(sym, tf, START, NOW)
    log(f"{sym} {tf}: fetched {len(fresh):,} bars {fresh.index[0]} -> {fresh.index[-1]} ({time.time()-t0:.0f}s)")
    repo = latest_repo_pq(sym, tf)
    if repo is not None:
        ov = fresh["close"].reindex(repo.index).dropna()
        ov2 = repo["close"].reindex(ov.index)
        corr = np.corrcoef(ov.values, ov2.values)[0, 1] if len(ov) > 100 else np.nan
        mrd = (ov - ov2).abs().div(ov2).max() if len(ov) else np.nan
        log(f"  overlap vs repo parquet: n={len(ov):,} corr={corr:.6f} max_rel_diff={mrd:.2e}")
        merged = pd.concat([fresh, repo.reindex(columns=fresh.columns)])
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    else:
        merged = fresh
        log("  no repo parquet to splice (fresh only)")
    fp = OUT / f"hist_{sym}_{tf}.parquet"
    merged.to_parquet(fp)
    log(f"  saved {fp.name}: {len(merged):,} rows {merged.index[0]} -> {merged.index[-1]}")

# 4h resample of merged BTC 1h, validated vs repo 4h
b1 = pd.read_parquet(OUT / "hist_BTCUSDT_1h.parquet")
b4 = pd.DataFrame({
    "open": b1["open"].resample("4h").first(),
    "high": b1["high"].resample("4h").max(),
    "low": b1["low"].resample("4h").min(),
    "close": b1["close"].resample("4h").last(),
    "volume": b1["volume"].resample("4h").sum(),
}).dropna(subset=["close"])
repo4 = latest_repo_pq("BTCUSDT", "4h")
ov = b4["close"].reindex(repo4.index).dropna(); ov2 = repo4["close"].reindex(ov.index)
log(f"BTC 4h resample vs repo 4h: n={len(ov):,} corr={np.corrcoef(ov,ov2)[0,1]:.6f} "
    f"max_rel_diff={(ov-ov2).abs().div(ov2).max():.2e}")
b4.to_parquet(OUT / "hist_BTCUSDT_4h.parquet")
log(f"saved hist_BTCUSDT_4h.parquet: {len(b4):,} rows {b4.index[0]} -> {b4.index[-1]}")
rep.close()
print("S1 DONE")
