"""
S6 -- reconcile s2 (IC=0.22 on next-hour direction, 2025-01-01 to now) vs
s5 (near-zero correlation with strike-hit outcome, 05-21 to 07-11 archive
window only). Recompute IC on NEXT-HOUR DIRECTION restricted to the SAME
recent window s3/s4/s5 used, split by month, to check for regime decay vs
a target-mismatch (direction != strike-touch) explanation.
"""
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from composite_scorer import compute_scores


def load_asset(asset):
    sym = f"{asset}USDT"
    f_1h = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_1m = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    ohlcv_1h = pd.read_parquet(f_1h); ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True); ohlcv_1h = ohlcv_1h.sort_index()
    ohlcv_1m = pd.read_parquet(f_1m); ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True); ohlcv_1m = ohlcv_1m.sort_index()
    df_15m = ohlcv_1m.resample("15min", origin="start_day").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    df_4h = ohlcv_1h.resample("4h", origin="start_day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return ohlcv_1h, df_4h, df_15m, ohlcv_1m["close"].astype(float), ohlcv_1m["volume"].astype(float)


ohlcv_1h, df_4h, df_15m, close_1m, volume_1m = load_asset("SOL")
ts_1h = ohlcv_1h.index
trend_ser, rev_ser = compute_scores(
    ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float), ohlcv_1h["low"].astype(float), ohlcv_1h["volume"].astype(float),
    df_4h["close"].astype(float), df_4h["high"].astype(float), df_4h["low"].astype(float), df_4h["volume"].astype(float),
    df_15m["close"].astype(float), df_15m["high"].astype(float), df_15m["low"].astype(float),
    close_1m, volume_1m, ts_1h,
)
next_ret = np.log(ohlcv_1h["close"] / ohlcv_1h["close"].shift(1)).shift(-1)
df = pd.DataFrame({"ts": ts_1h, "trend": trend_ser.values, "rev": rev_ser.values, "next_ret": next_ret.values}).dropna()

df["month"] = df["ts"].dt.to_period("M").astype(str)
print("=== IC(rev+trend, next_hour_return) by month, full available history ===")
for m, sub in df.groupby("month"):
    if len(sub) < 100:
        continue
    combo = sub["rev"] + sub["trend"]
    ic = combo.corr(sub["next_ret"])
    print(f"  {m}: n={len(sub):5d}  IC={ic:+.4f}")

recent = df[df["ts"] >= pd.Timestamp("2026-05-21", tz="UTC")]
combo_r = recent["rev"] + recent["trend"]
print(f"\nRecent window (05-21 -> now, matches s3/s4/s5 archive): n={len(recent)}  IC={combo_r.corr(recent['next_ret']):+.4f}")

print("\nDONE_S6")
