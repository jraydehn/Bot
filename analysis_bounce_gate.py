"""
analysis_bounce_gate.py — Does stochastic crossup from oversold predict YES bounces
in the high-vol-ratio regime?

Research question:
  In hours where vol_ratio (σ_model / σ_kalshi) > 1.20 — the regime where NO bets
  consistently lose — does a stochastic bullish crossover or oversold position
  reliably predict that BTC will bounce, making YES bets viable?

Method:
  1. Compute vol_ratio and 15m stochastic for every hour in the test set (Jan 2025–Apr 2026).
  2. Bucket hours by (vol_ratio regime) × (stochastic signal).
  3. For each bucket, report YES win rate at each OTM offset (+0.001 to +0.005).
  4. If YES win rate exceeds breakeven in the crossup bucket, the gate is viable.

Breakeven YES win rates by p_market (approximate):
  p_market=0.25 → need 25% win  (payout = 3×)
  p_market=0.30 → need 30% win  (payout = 2.33×)
  p_market=0.35 → need 35% win  (payout = 1.86×)
  p_market=0.40 → need 40% win  (payout = 1.5×)

No gates applied here — raw win rates only.
"""

import sys, math, glob, warnings
from pathlib import Path
from scipy.stats import norm as sp_norm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

# ── CONSTANTS (same as backtest_unbiased.py) ──────────────────────────────────
KALSHI_VOL_WINDOW = 1440
KALSHI_VOL_LAG    = 120
MODEL_VOL_WINDOW  = 60
WARMUP_BARS       = KALSHI_VOL_WINDOW + KALSHI_VOL_LAG + 60
TAU               = 60

OFFSETS_YES = [0.001, 0.002, 0.003, 0.005]   # small OTM — bounce needs to reach strike
TEST_START  = pd.Timestamp("2025-01-01", tz="UTC")

STOCH_K_PERIOD   = 14
STOCH_D_PERIOD   = 3
STOCH_OVERSOLD   = 20
STOCH_OVERBOUGHT = 80

SEP = "=" * 72

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
print(f"Loading 1m: {files_1m[-1]}")
ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
print(f"Loading 1h: {files_1h[-1]}")
ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h = ohlcv_1h["close"].values.astype(float)
ts_1h    = ohlcv_1h.index
n1h      = len(ts_1h)

# ── VOL SERIES ────────────────────────────────────────────────────────────────
print("\nComputing vol series...")
close_1m = ohlcv_1m["close"].values.astype(float)
log_ret  = pd.Series(
    np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0),
    index=ohlcv_1m.index,
)
sigma_model_1m  = log_ret.rolling(MODEL_VOL_WINDOW).std()
sigma_kalshi_1m = log_ret.rolling(KALSHI_VOL_WINDOW).std().shift(KALSHI_VOL_LAG)
ohlcv_1m_idx    = ohlcv_1m.index

# ── STOCHASTIC ON 15m BARS (vectorized) ───────────────────────────────────────
print("Computing 15m stochastic (vectorized)...")
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg({
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}).dropna(subset=["close"])

lowest_low   = df_15m["low"].rolling(STOCH_K_PERIOD).min()
highest_high = df_15m["high"].rolling(STOCH_K_PERIOD).max()
hl_range     = (highest_high - lowest_low).replace(0, float("nan"))
stoch_k_15m  = ((df_15m["close"] - lowest_low) / hl_range) * 100
stoch_d_15m  = stoch_k_15m.rolling(STOCH_D_PERIOD).mean()

# Crossover detection on 15m series
k_curr = stoch_k_15m
k_prev = stoch_k_15m.shift(1)
d_curr = stoch_d_15m
d_prev = stoch_d_15m.shift(1)

bullish_xover_15m = (
    (k_prev < d_prev) &    # K was below D
    (k_curr > d_curr) &    # K just crossed above D
    (k_prev < STOCH_OVERSOLD)  # originated from oversold
)
# Allow crossup that fired on the previous 15m bar too (still actionable)
bullish_xover_15m_lag1 = bullish_xover_15m.shift(1).fillna(False)
bullish_xover_active_15m = bullish_xover_15m | bullish_xover_15m_lag1

in_oversold_15m  = k_curr < STOCH_OVERSOLD
in_overbought_15m = k_curr > STOCH_OVERBOUGHT

# Forward-fill 15m stochastic values to 1m index (use last known 15m bar)
stoch_k_1m             = stoch_k_15m.reindex(ohlcv_1m.index, method="ffill")
stoch_d_1m             = stoch_d_15m.reindex(ohlcv_1m.index, method="ffill")
bullish_xover_active_1m = bullish_xover_active_15m.reindex(ohlcv_1m.index, method="ffill").fillna(False)
in_oversold_1m          = in_oversold_15m.reindex(ohlcv_1m.index, method="ffill").fillna(False)
in_overbought_1m        = in_overbought_15m.reindex(ohlcv_1m.index, method="ffill").fillna(False)

print("Pre-computation done.\n")


# ── MAIN ANALYSIS LOOP ────────────────────────────────────────────────────────
print(SEP)
print("ANALYSIS — YES win rates by (vol_ratio regime) × (stochastic signal)")
print(f"Test set: Jan 2025 – Apr 2026  |  OTM YES offsets: {OFFSETS_YES}")
print(SEP)

rows = []

for i_h in range(50, n1h - 1):
    ts_now = ts_1h[i_h]
    if ts_now < TEST_START:
        continue

    spot       = float(close_1h[i_h])
    next_close = float(close_1h[i_h + 1])

    pos1m = int(ohlcv_1m_idx.searchsorted(ts_now, side="right")) - 1
    if pos1m < WARMUP_BARS:
        continue

    sig_m = float(sigma_model_1m.iat[pos1m])
    sig_k = float(sigma_kalshi_1m.iat[pos1m])
    if np.isnan(sig_m) or sig_m <= 0 or np.isnan(sig_k) or sig_k <= 0:
        continue

    vr        = sig_m / sig_k
    stoch_k_v = float(stoch_k_1m.iat[pos1m])
    stoch_d_v = float(stoch_d_1m.iat[pos1m])
    xover     = bool(bullish_xover_active_1m.iat[pos1m])
    oversold  = bool(in_oversold_1m.iat[pos1m])

    # Classify stochastic signal
    if xover:
        stoch_signal = "crossup"        # K crossed above D from oversold — strongest
    elif oversold:
        stoch_signal = "oversold"       # in oversold zone, no crossup yet
    elif bool(in_overbought_1m.iat[pos1m]):
        stoch_signal = "overbought"
    else:
        stoch_signal = "neutral"

    for offset in OFFSETS_YES:
        K           = spot * (1.0 + offset)
        actual_yes  = int(next_close > K)
        rows.append({
            "vol_ratio":    round(vr, 3),
            "stoch_k":      round(stoch_k_v, 1),
            "stoch_d":      round(stoch_d_v, 1),
            "stoch_signal": stoch_signal,
            "offset":       offset,
            "actual_yes":   actual_yes,
            "spot":         spot,
            "K":            K,
            "next_close":   next_close,
        })

df = pd.DataFrame(rows)

# ── RESULTS ───────────────────────────────────────────────────────────────────

VR_BINS = [
    ("low   (<0.70)",    df["vol_ratio"] < 0.70),
    ("mid  (0.70–1.20)", (df["vol_ratio"] >= 0.70) & (df["vol_ratio"] < 1.20)),
    ("high (>1.20)",     df["vol_ratio"] >= 1.20),
    ("very (>1.50)",     df["vol_ratio"] >= 1.50),
]

SIGNALS = ["crossup", "oversold", "neutral", "overbought"]

for label, vr_mask in VR_BINS:
    sub_vr = df[vr_mask]
    n_hrs  = sub_vr.groupby(["vol_ratio", "offset"]).ngroups  # approx
    n_obs  = len(sub_vr[sub_vr["offset"] == OFFSETS_YES[0]])
    print(f"\nvol_ratio = {label}  ({n_obs:,} hours)")
    print(f"  {'signal':>10}  {'offset':>8}  {'n':>6}  {'YES_win%':>9}  {'breakeven':>10}  {'verdict'}")
    print("  " + "-" * 65)
    for sig in SIGNALS:
        sig_mask = sub_vr["stoch_signal"] == sig
        sub_s    = sub_vr[sig_mask]
        if len(sub_s) == 0:
            continue
        for off in OFFSETS_YES:
            sub_o   = sub_s[sub_s["offset"] == off]
            if len(sub_o) < 5:
                continue
            win_pct = sub_o["actual_yes"].mean() * 100
            # Approximate breakeven: p_market ≈ p_yes_lognormal using median sig_k at these bars
            # Rough estimate: breakeven = p_market (what Kalshi would charge)
            # Use 30% as rough benchmark for +0.1–0.5% OTM
            be = {0.001: 40, 0.002: 33, 0.003: 28, 0.005: 21}[off]
            verdict = "EDGE" if win_pct >= be + 5 else "MARGINAL" if win_pct >= be else "LOSING"
            print(f"  {sig:>10}  {off:+8.3f}  {len(sub_o):>6,}  {win_pct:>8.1f}%  {be:>9}%+   {verdict}")


# ── CROSSUP DEEP DIVE ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEEP DIVE — crossup signal in high-vol-ratio regime (vol_ratio > 1.20)")
print(SEP)

high_xover = df[(df["vol_ratio"] >= 1.20) & (df["stoch_signal"] == "crossup")]
n_hours_xover = len(high_xover[high_xover["offset"] == OFFSETS_YES[0]])
print(f"Hours with high vol_ratio + crossup: {n_hours_xover}")

if n_hours_xover > 0:
    print(f"\n  {'offset':>8}  {'n':>6}  {'YES_win%':>9}  {'avg_move_pct':>13}")
    print("  " + "-" * 45)
    for off in OFFSETS_YES:
        sub = high_xover[high_xover["offset"] == off]
        if len(sub) == 0:
            continue
        win_pct   = sub["actual_yes"].mean() * 100
        avg_move  = ((sub["next_close"] - sub["spot"]) / sub["spot"] * 100).mean()
        print(f"  {off:+8.3f}  {len(sub):>6,}  {win_pct:>8.1f}%  {avg_move:>+12.3f}%")

    # Monthly breakdown for crossup YES trades at best offset
    best_off = OFFSETS_YES[1]  # +0.002
    sub_mo   = high_xover[high_xover["offset"] == best_off].copy()
    sub_mo["month"] = pd.to_datetime(
        sub_mo["spot"].index if hasattr(sub_mo["spot"], "index") else range(len(sub_mo))
    ).strftime("%Y-%m") if False else None

    print(f"\n  Vol_ratio distribution at crossup hours:")
    for lo, hi in [(1.2, 1.5), (1.5, 2.0), (2.0, 99)]:
        s = high_xover[(high_xover["vol_ratio"] >= lo) & (high_xover["vol_ratio"] < hi) & (high_xover["offset"] == OFFSETS_YES[0])]
        if len(s):
            print(f"    {lo:.1f}–{hi:.1f}x: {len(s):,} hours, YES win {s['actual_yes'].mean()*100:.1f}%")

# ── CROSSUP vs RAW OVERSOLD COMPARISON ────────────────────────────────────────
print(f"\n{SEP}")
print("COMPARISON — crossup vs raw oversold (high vol_ratio only)")
print("(crossup = momentum actively turning; oversold = still falling/stuck)")
print(SEP)

high_vr = df[df["vol_ratio"] >= 1.20]
print(f"\n  {'signal':>12}  {'offset':>8}  {'n_hours':>8}  {'YES_win%':>9}")
print("  " + "-" * 45)
for sig in ["crossup", "oversold"]:
    for off in OFFSETS_YES:
        sub = high_vr[(high_vr["stoch_signal"] == sig) & (high_vr["offset"] == off)]
        if len(sub) < 3:
            print(f"  {sig:>12}  {off:+8.3f}  {'<5':>8}  {'—':>9}")
            continue
        win_pct = sub["actual_yes"].mean() * 100
        print(f"  {sig:>12}  {off:+8.3f}  {len(sub):>8,}  {win_pct:>8.1f}%")
