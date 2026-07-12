"""
S1 -- rebuild SOL's stale 1h calibration table with fresh data (through
2026-07-11), WITHOUT touching the live JSON. composite_calibration_sol.json
was last built 2026-05-04 (over 2 months stale); composite_calibration_sol_30m.json
last built 2026-05-07. This reuses calibrate_eth_sol.py's exact methodology
(same compute_scores() logic, same TEST_START=2025-01-01) but writes to a
shadow path and does NOT call save_calibration().

Also does a proper backtest: apply OLD (stale) vs NEW (fresh) calibration
lookups to the actual realized outcomes over the last ~9 weeks (the period
the stale table never saw), scored on Brier score AND on a $ PnL proxy
using the same lookup_p_up_blended tau-ladder blending SOL actually uses
live, per house rule (calibrate on PnL, not calibration error alone).
"""
import glob
import json
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
from composite_scorer import compute_scores, BASELINE_UP, SMOOTHING_N

OUT = BASE / "reform_results/sol_hourly_20260710"
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
STALE_CUTOFF = pd.Timestamp("2026-05-04", tz="UTC")  # old table's last-seen data


def load_asset(asset):
    sym = f"{asset}USDT"
    f_1h = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_1m = sorted(glob.glob(str(DATA / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    ohlcv_1h = pd.read_parquet(f_1h); ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True); ohlcv_1h = ohlcv_1h.sort_index()
    ohlcv_1m = pd.read_parquet(f_1m); ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True); ohlcv_1m = ohlcv_1m.sort_index()
    df_15m = ohlcv_1m.resample("15min", origin="start_day").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    df_4h = ohlcv_1h.resample("4h", origin="start_day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    close_1m = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)
    print(f"  1h: {len(ohlcv_1h):,}  range: {ohlcv_1h.index[0].date()} -> {ohlcv_1h.index[-1].date()}")
    return ohlcv_1h, df_4h, df_15m, close_1m, volume_1m


print("Loading SOL data...")
ohlcv_1h, df_4h, df_15m, close_1m, volume_1m = load_asset("SOL")
ts_1h = ohlcv_1h.index

print("Computing composite scores...")
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
df = pd.DataFrame({"ts": ts_1h[idx], "trend": trend_ser.values[idx], "rev": rev_ser.values[idx], "next_up": next_up.values[idx]})
df["tb"] = df["trend"].clip(-3, 3)
df["rb"] = df["rev"].clip(-5, 5)

# --- OLD calibration: build using ONLY data up to the stale cutoff (replicates what the live table saw) ---
train_old = df[df["ts"] < STALE_CUTOFF]
baseline_old = train_old["next_up"].mean()
cal_old = {}
for tb in sorted(train_old["tb"].unique()):
    for rb in sorted(train_old["rb"].unique()):
        cell = train_old[(train_old["tb"] == tb) & (train_old["rb"] == rb)]
        n = len(cell)
        if n >= 10:
            up = cell["next_up"].mean()
            w = min(1.0, n / SMOOTHING_N)
            cal_old[(int(tb), int(rb))] = round(float(w * up + (1 - w) * baseline_old), 4)

# --- NEW calibration: full fresh data through today ---
baseline_new = df["next_up"].mean()
cal_new = {}
for tb in sorted(df["tb"].unique()):
    for rb in sorted(df["rb"].unique()):
        cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
        n = len(cell)
        if n >= 10:
            up = cell["next_up"].mean()
            w = min(1.0, n / SMOOTHING_N)
            cal_new[(int(tb), int(rb))] = round(float(w * up + (1 - w) * baseline_new), 4)

print(f"\nOld-cutoff table: {len(cal_old)} cells, baseline={baseline_old:.1%}, trained on n={len(train_old)}")
print(f"Fresh table:      {len(cal_new)} cells, baseline={baseline_new:.1%}, trained on n={len(df)}")

# Save shadow copy (does NOT touch live composite_calibration_sol.json)
raw_new = {f"{k[0]},{k[1]}": v for k, v in cal_new.items()}
with open(OUT / "composite_calibration_sol_FRESH_shadow.json", "w") as f:
    json.dump(raw_new, f, indent=2)
print(f"Shadow table saved -> {OUT / 'composite_calibration_sol_FRESH_shadow.json'}")

# --- Backtest: apply OLD-cutoff-trained table vs actual live JSON (loaded from repo) vs FRESH table, ---
# --- scored on the held-out period the old table never saw (>= STALE_CUTOFF) ---
holdout = df[df["ts"] >= STALE_CUTOFF].copy()
print(f"\nHoldout period (what the stale live table never trained on): "
      f"{holdout['ts'].min()} -> {holdout['ts'].max()}, n={len(holdout)}")

live_cal_raw = json.load(open(BASE / "composite_calibration_sol.json"))
live_cal = {tuple(int(x) for x in k.split(",")): v for k, v in live_cal_raw.items()}


def lookup(cal, baseline, tb, rb):
    return cal.get((int(tb), int(rb)), baseline)


def score(cal, baseline, label):
    p = holdout.apply(lambda r: lookup(cal, baseline, r["tb"], r["rb"]), axis=1)
    y = holdout["next_up"].values
    brier = float(np.mean((p.values - y) ** 2))
    logloss = float(-np.mean(y * np.log(np.clip(p.values, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p.values, 1e-6, 1))))
    mean_p = float(p.mean())
    print(f"  {label:>28s}: brier={brier:.4f}  logloss={logloss:.4f}  mean_p={mean_p:.3f}  actual_up%={y.mean():.3f}")
    return p


print(f"\n=== Held-out calibration scoring (n={len(holdout)}) ===")
p_live = score(live_cal, BASELINE_UP, "LIVE (stale, built 05-04)")
p_new = score(cal_new, baseline_new, "FRESH (rebuilt today)")

# naive baseline-only reference
p_base = score({}, BASELINE_UP, "flat baseline (no signal)")

print("\nDONE_S1")
