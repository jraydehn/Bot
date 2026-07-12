"""
S2 -- how strong is SOL's composite trend/rev signal, vs BTC's, on the SAME
long-history methodology? DRIFT_MULTIPLIER = SOL:0.20 vs BTC:1.40 (7x gap).
Is that gap justified by signal strength, or is SOL being over-dampened?

Also: composite signal buckets (extreme rev/trend combos) -- BTC's edge
often concentrates in the tails even when the average signal is weak, so
check tails separately from the average Brier/IC read.
"""
import glob
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
DATA = BASE / "data"
sys.path.insert(0, str(BASE))
from composite_scorer import compute_scores, BASELINE_UP

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def load_asset(asset):
    sym = f"{asset}USDT"
    f_1h = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_1m = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    ohlcv_1h = pd.read_parquet(f_1h); ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True); ohlcv_1h = ohlcv_1h.sort_index()
    ohlcv_1m = pd.read_parquet(f_1m); ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True); ohlcv_1m = ohlcv_1m.sort_index()
    df_15m = ohlcv_1m.resample("15min", origin="start_day").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    df_4h = ohlcv_1h.resample("4h", origin="start_day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return ohlcv_1h, df_4h, df_15m, ohlcv_1m["close"].astype(float), ohlcv_1m["volume"].astype(float)


def analyze(asset):
    print(f"\n{'='*70}\n  {asset}\n{'='*70}")
    ohlcv_1h, df_4h, df_15m, close_1m, volume_1m = load_asset(asset)
    ts_1h = ohlcv_1h.index
    trend_ser, rev_ser = compute_scores(
        ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float), ohlcv_1h["low"].astype(float), ohlcv_1h["volume"].astype(float),
        df_4h["close"].astype(float), df_4h["high"].astype(float), df_4h["low"].astype(float), df_4h["volume"].astype(float),
        df_15m["close"].astype(float), df_15m["high"].astype(float), df_15m["low"].astype(float),
        close_1m, volume_1m, ts_1h,
    )
    next_ret = np.log(ohlcv_1h["close"] / ohlcv_1h["close"].shift(1)).shift(-1)
    next_up = (next_ret > 0).astype(int)
    test_mask = ts_1h >= TEST_START
    idx = np.where(test_mask)[0][:-1]
    df = pd.DataFrame({"trend": trend_ser.values[idx], "rev": rev_ser.values[idx], "next_up": next_up.values[idx], "next_ret": next_ret.values[idx]})
    baseline = df["next_up"].mean()

    # IC: correlation of rev score with next-hour return (rev is the dominant driver in score_to_p_model's z_drift)
    ic_rev = df["rev"].corr(df["next_ret"])
    ic_trend = df["trend"].corr(df["next_ret"])
    combo = df["rev"] + df["trend"]
    ic_combo = combo.corr(df["next_ret"])
    print(f"  n={len(df)}  baseline_up={baseline:.1%}")
    print(f"  IC(rev, next_ret)   = {ic_rev:+.4f}")
    print(f"  IC(trend, next_ret) = {ic_trend:+.4f}")
    print(f"  IC(rev+trend, next_ret) = {ic_combo:+.4f}")

    # tails: strong rev signal buckets
    for lbl, mask in [
        ("Strong YES (rev>=4,trend>=0)", (df["rev"] >= 4) & (df["trend"] >= 0)),
        ("Strong NO  (rev<=-4,trend<=0)", (df["rev"] <= -4) & (df["trend"] <= 0)),
        ("Neutral (rev in [-1,1])", df["rev"].between(-1, 1)),
    ]:
        sub = df[mask]
        n = len(sub)
        if n < 20:
            print(f"  {lbl:<32s} n={n:5d}  (thin)")
            continue
        up = sub["next_up"].mean()
        edge = up - baseline
        z = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
        pv = 2 * (1 - norm.cdf(abs(z)))
        print(f"  {lbl:<32s} n={n:5d}  up={up:.1%}  edge={edge:+.1%}  p={pv:.4f}")

    return dict(asset=asset, ic_rev=ic_rev, ic_trend=ic_trend, ic_combo=ic_combo, n=len(df))


r_sol = analyze("SOL")
r_btc = analyze("BTC")

print(f"\n{'='*70}\n  SIGNAL STRENGTH COMPARISON (long-history, 2025-01-01 -> now)\n{'='*70}")
print(f"  {'':>10s} {'IC(rev)':>10s} {'IC(trend)':>10s} {'IC(combo)':>10s}")
print(f"  {'SOL':>10s} {r_sol['ic_rev']:>10.4f} {r_sol['ic_trend']:>10.4f} {r_sol['ic_combo']:>10.4f}")
print(f"  {'BTC':>10s} {r_btc['ic_rev']:>10.4f} {r_btc['ic_trend']:>10.4f} {r_btc['ic_combo']:>10.4f}")
ratio = abs(r_sol['ic_combo']) / abs(r_btc['ic_combo']) if r_btc['ic_combo'] != 0 else float('nan')
print(f"\n  SOL/BTC combo-IC ratio: {ratio:.3f}  (current DRIFT_MULTIPLIER ratio SOL/BTC = {0.20/1.40:.3f})")
print("\nDONE_S2")
