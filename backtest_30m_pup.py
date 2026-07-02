"""
backtest_30m_pup.py — Proper simulation of tau-blended 30m p_up vs 1h-only.

Computes actual 30m composite scores (trend from 1h bars, reversion from 15m/5m)
for every 30m timestamp in history, then joins to paper_trades.csv by trade time.

Usage: python3 backtest_30m_pup.py
"""
import sys, glob, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

from composite_scorer import (
    compute_scores, compute_scores_30m,
    lookup_p_up, lookup_p_up_blended,
    DRIFT_MULTIPLIER,
)

MIN_NET_EDGE = 0.01
K_DRIFT      = DRIFT_MULTIPLIER["BTC"]  # 1.40

# ── 1. Load OHLCV ─────────────────────────────────────────────────────────────
print("Loading OHLCV data...")
f1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))[-1]
ohlcv_1m = pd.read_parquet(f1m)
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()
print(f"  1m: {len(ohlcv_1m):,} rows  {ohlcv_1m.index[0].date()} → {ohlcv_1m.index[-1].date()}")

ohlcv_1h = ohlcv_1m.resample("1h", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

ohlcv_4h = ohlcv_1h.resample("4h", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

ohlcv_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

ohlcv_30m = ohlcv_1m.resample("30min", origin="start_day").agg(
    {"close":"last"}
).dropna(subset=["close"])

ts_1h  = ohlcv_1h.index
ts_30m = ohlcv_30m.index
print(f"  1h:{len(ohlcv_1h):,}  4h:{len(ohlcv_4h):,}  15m:{len(ohlcv_15m):,}  30m:{len(ts_30m):,}")

# ── 2. Compute 1h scores ───────────────────────────────────────────────────────
print("\nComputing 1h scores (4h trend + 1h/15m reversion)...")
t1h, r1h = compute_scores(
    ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float),
    ohlcv_1h["low"].astype(float),   ohlcv_1h["volume"].astype(float),
    ohlcv_4h["close"].astype(float), ohlcv_4h["high"].astype(float),
    ohlcv_4h["low"].astype(float),   ohlcv_4h["volume"].astype(float),
    ohlcv_15m["close"].astype(float), ohlcv_15m["high"].astype(float),
    ohlcv_15m["low"].astype(float),
    ohlcv_1m["close"].astype(float), ohlcv_1m["volume"].astype(float),
    ts_1h,
)
scores_1h = pd.DataFrame({"trend_1h": t1h, "rev_1h": r1h})
print(f"  Done — {len(scores_1h):,} rows")

# ── 3. Compute 30m scores ──────────────────────────────────────────────────────
print("\nComputing 30m scores (1h trend + 15m/5m reversion) — may take 2-3 min...")
t30, r30 = compute_scores_30m(
    ohlcv_1h["close"].astype(float), ohlcv_1h["high"].astype(float),
    ohlcv_1h["low"].astype(float),   ohlcv_1h["volume"].astype(float),
    ohlcv_15m["close"].astype(float), ohlcv_15m["high"].astype(float),
    ohlcv_15m["low"].astype(float),
    ohlcv_1m["close"].astype(float), ohlcv_1m["volume"].astype(float),
    ts_30m,
)
scores_30m = pd.DataFrame({"trend_30m": t30, "rev_30m": r30})
print(f"  Done — {len(scores_30m):,} rows")

# ── 4. Load paper trades ───────────────────────────────────────────────────────
print("\nLoading paper trades...")
pt = pd.read_csv(BASE / "results/paper_trades.csv", low_memory=False)
pt["logged_at"] = pd.to_datetime(pt["logged_at"], utc=True)

for col in ["composite_trend","composite_rev","tau_minutes","p_market",
            "z_score","would_pnl","resolved_yes"]:
    pt[col] = pd.to_numeric(pt[col], errors="coerce")

btc = pt[
    (pt["decision"] == "trade") &
    (pt["composite_trend"].notna()) &
    (pt["z_score"].notna()) &
    (pt["resolved_yes"].notna()) &
    (pt["tau_minutes"].notna()) &
    (pt["tau_minutes"] > 0)
].copy()
print(f"  Resolved BTC trades with composite scores: {len(btc)}")

# ── 5. Join scores by timestamp ────────────────────────────────────────────────
btc["ts_30m_key"] = btc["logged_at"].dt.floor("30min")

# 1h scores: use the logged values (composite_trend/rev) as ground truth
btc["trend_1h"] = btc["composite_trend"].astype(int)
btc["rev_1h"]   = btc["composite_rev"].astype(int)

# 30m scores: join from computed series
scores_30m.index = scores_30m.index.tz_localize("UTC") if scores_30m.index.tz is None else scores_30m.index
btc = btc.join(scores_30m, on="ts_30m_key", how="left")

n_missing = btc["trend_30m"].isna().sum()
print(f"  30m scores matched: {len(btc)-n_missing}/{len(btc)}  (missing={n_missing} near data boundary)")
btc = btc[btc["trend_30m"].notna()].copy()
btc["trend_30m"] = btc["trend_30m"].astype(int)
btc["rev_30m"]   = btc["rev_30m"].astype(int)

# ── 6. Compute p_up and p_model under each scenario ───────────────────────────
print("\nComputing p_up and p_model...")

p_up_1h    = np.array([lookup_p_up(int(t), int(r), "BTC")
                        for t, r in zip(btc["trend_1h"], btc["rev_1h"])])
p_up_blend = np.array([
    lookup_p_up_blended(int(t1), int(r1), int(t30), int(r30), float(tau), "BTC")
    for t1, r1, t30, r30, tau in zip(
        btc["trend_1h"], btc["rev_1h"],
        btc["trend_30m"], btc["rev_30m"],
        btc["tau_minutes"],
    )
])

z = btc["z_score"].values
btc["p_up_1h"]       = p_up_1h
btc["p_up_blend"]    = p_up_blend
btc["p_model_1h"]    = np.clip(1 - norm.cdf(z - norm.ppf(p_up_1h)    * K_DRIFT), 0.01, 0.99)
btc["p_model_blend"] = np.clip(1 - norm.cdf(z - norm.ppf(p_up_blend) * K_DRIFT), 0.01, 0.99)
btc["edge_1h"]       = btc["p_model_1h"]    - btc["p_market"]
btc["edge_blend"]    = btc["p_model_blend"] - btc["p_market"]
btc["taken_1h"]      = btc["edge_1h"]    >= MIN_NET_EDGE
btc["taken_blend"]   = btc["edge_blend"] >= MIN_NET_EDGE
btc["won"]           = btc["resolved_yes"].notna()  # all rows here are resolved

# ── 7. Results ─────────────────────────────────────────────────────────────────
SEP  = "=" * 62
SEP2 = "-" * 62

baseline_pnl = btc[btc["taken_1h"]]["would_pnl"].sum()
blend_pnl    = btc[btc["taken_blend"]]["would_pnl"].sum()
dropped      = btc[btc["taken_1h"]  & ~btc["taken_blend"]]
added        = btc[~btc["taken_1h"] & btc["taken_blend"]]

print(f"\n{SEP}")
print(f"  RESULTS: 1h-only p_up  vs  tau-blended 30m p_up")
print(f"  (Using actual 30m scores: trend=1h-bar, rev=15m/5m indicators)")
print(SEP)
print(f"  Baseline (1h-only):  n={btc['taken_1h'].sum():4d}  PnL=${baseline_pnl:+.2f}")
print(f"  Blended:             n={btc['taken_blend'].sum():4d}  PnL=${blend_pnl:+.2f}")
print(f"  Delta:               n={btc['taken_blend'].sum()-btc['taken_1h'].sum():+4d}  PnL=${blend_pnl-baseline_pnl:+.2f}")

print(f"\n  Trades DROPPED by blending (1h takes → blend rejects):  n={len(dropped)}")
if len(dropped):
    print(f"    Would-PnL if taken: ${dropped['would_pnl'].sum():+.2f}  (negative = correctly avoided)")

print(f"\n  Trades ADDED by blending (1h skips → blend takes):      n={len(added)}")
if len(added):
    print(f"    Would-PnL if taken: ${added['would_pnl'].sum():+.2f}  (positive = correctly added)")

print(f"\n{SEP2}")
print(f"  By tau bucket:")
print(f"  {'tau':>8}  {'1h n':>5}  {'1h PnL':>9}  {'blend n':>7}  {'blend PnL':>10}  {'delta':>9}")
print(SEP2)
bins   = [0, 15, 30, 45, 60, 999]
labels = ["<15", "15-30", "30-45", "45-60", ">60"]
btc["tau_bin"] = pd.cut(btc["tau_minutes"], bins=bins, labels=labels)
for tb, grp in btc.groupby("tau_bin", observed=True):
    n1   = grp["taken_1h"].sum()
    nb   = grp["taken_blend"].sum()
    p1   = grp[grp["taken_1h"]]["would_pnl"].sum()
    pb   = grp[grp["taken_blend"]]["would_pnl"].sum()
    print(f"  {tb:>8}  {n1:>5}  ${p1:>+8.2f}  {nb:>7}  ${pb:>+9.2f}  ${pb-p1:>+8.2f}")

print(SEP2)
print(f"\n  p_up delta (blend − 1h):")
delta = btc["p_up_blend"] - btc["p_up_1h"]
print(f"    mean={delta.mean():+.4f}  std={delta.std():.4f}  max={delta.max():+.4f}  min={delta.min():+.4f}")
print(f"    |delta| > 0.02: {(delta.abs()>0.02).sum()} trades  |delta| > 0.05: {(delta.abs()>0.05).sum()} trades")

print(f"\n{SEP}")
print(f"  Net P&L improvement: ${blend_pnl - baseline_pnl:+.2f}")
