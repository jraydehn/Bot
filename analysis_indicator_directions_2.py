"""
analysis_indicator_directions_2.py — Extended indicator directional correlation test.

Round 2: Indicators not covered in round 1, different timeframes, and custom designs.

Proven in round 1 (mean reversion):
  Stochastic zone (15m) ★★★, VWAP position ★★★, EMA Stretch ★★★, RSI ★★★, BB Position ★★

New indicators tested here:
  1.  Donchian Channel        — price position within N-bar high/low range (1h: 20/50/100)
  2.  Vol Score (revisit)     — vol_ratio, ATR percentile, vol regime (expanding/contracting)
  3.  RSI multi-timeframe     — 4h RSI, daily RSI, and 1h vs 4h alignment
  4.  Stochastic on 1h        — same signal but computed on hourly bars (vs 15m in round 1)
  5.  Williams %R (1h, 14p)   — inverse stochastic, faster signal
  6.  CCI (1h, 20p)           — commodity channel index, deviation from mean price
  7.  Keltner Channel (1h)    — like BB but uses ATR; position within/outside channel
  8.  ATR Regime              — current ATR vs trailing 20-bar ATR (vol expansion/contraction)
  9.  Session time of day     — UTC hour effect on directional bias
  10. Day of week             — Monday vs Friday vs weekend effects
  11. Candle close position   — where price closed within the hourly bar range
  12. Consecutive bars        — how many bars in a row up or down (exhaustion signal)
  13. Inside/outside bar      — price compression (inside) or expansion (outside)
  14. Multi-TF trend agree    — 15m + 1h + 4h EMA direction all agree (confluence)
  15. Distance from swing     — % below recent 20-bar high / above recent 20-bar low
  16. Pivot points (daily)    — price vs classic daily pivot (P, R1, S1)
  17. Volatility of vol       — second-order: how much is vol itself changing?
  18. Price momentum vs vol   — move size relative to recent vol (z-score of 1h move)

Test set: Jan 2025 – Apr 2026
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

SEP  = "=" * 76
SEP2 = "-" * 76
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# ── Load data ──────────────────────────────────────────────────────────────────
print(SEP)
print("Loading data...")
print(SEP)

files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))

ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h  = pd.Series(ohlcv_1h["close"].values.astype(float),  index=ohlcv_1h.index)
high_1h   = pd.Series(ohlcv_1h["high"].values.astype(float),   index=ohlcv_1h.index)
low_1h    = pd.Series(ohlcv_1h["low"].values.astype(float),    index=ohlcv_1h.index)
open_1h   = pd.Series(ohlcv_1h["open"].values.astype(float),   index=ohlcv_1h.index)
volume_1h = pd.Series(ohlcv_1h["volume"].values.astype(float), index=ohlcv_1h.index)
ts_1h     = ohlcv_1h.index
n1h       = len(close_1h)

next_ret = np.log(close_1h / close_1h.shift(1)).shift(-1)
next_up  = (next_ret > 0).astype(int)

# ── Build 4h bars ──────────────────────────────────────────────────────────────
ohlcv_4h = ohlcv_1h.resample("4h", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])
close_4h  = ohlcv_4h["close"]
high_4h   = ohlcv_4h["high"]
low_4h    = ohlcv_4h["low"]

# ── Build daily bars ───────────────────────────────────────────────────────────
ohlcv_1d = ohlcv_1h.resample("1D", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])
close_1d = ohlcv_1d["close"]
high_1d  = ohlcv_1d["high"]
low_1d   = ohlcv_1d["low"]

# ── 15m bars ───────────────────────────────────────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"high":"max","low":"min","close":"last"}
).dropna(subset=["close"])

print(f"  1h bars: {n1h:,}  |  4h bars: {len(ohlcv_4h):,}  |  daily bars: {len(ohlcv_1d):,}")


# ════════════════════════════════════════════════════════════════════════════════
# COMPUTE INDICATORS
# ════════════════════════════════════════════════════════════════════════════════
print("\nComputing indicators...")

# ── 1. Donchian Channel (1h, three periods) ─────────────────────────────────
for period in [20, 50, 100]:
    dc_high = high_1h.rolling(period).max()
    dc_low  = low_1h.rolling(period).min()
    dc_rng  = (dc_high - dc_low).replace(0, float("nan"))
    pct_b   = (close_1h - dc_low) / dc_rng   # 0 = at channel low, 1 = at high
    locals()[f"dc_{period}_pct"]  = pct_b
    locals()[f"dc_{period}_high"] = dc_high
    locals()[f"dc_{period}_low"]  = dc_low

# Donchian on 4h (20-bar = ~3.3 days of context)
dc_4h_high = high_4h.rolling(20).max()
dc_4h_low  = low_4h.rolling(20).min()
dc_4h_rng  = (dc_4h_high - dc_4h_low).replace(0, float("nan"))
dc_4h_pct  = ((close_4h - dc_4h_low) / dc_4h_rng).reindex(ts_1h, method="ffill")

def dc_to_signal(pct, label_prefix):
    sig = pd.Series("mid", index=pct.index, dtype=object)
    sig[pct < 0.10] = "near_low"
    sig[pct > 0.90] = "near_high"
    sig[pct < 0.20] = "lower_zone"
    sig[pct > 0.80] = "upper_zone"
    return sig

dc20_sig  = dc_to_signal(dc_20_pct,  "DC20")
dc50_sig  = dc_to_signal(dc_50_pct,  "DC50")
dc100_sig = dc_to_signal(dc_100_pct, "DC100")
dc4h_sig  = dc_to_signal(dc_4h_pct,  "DC4h")

# ── 2. Vol Score (revisit): ATR percentile and vol regime ──────────────────
# ATR on 1h (14 period)
tr_1h  = pd.concat([
    high_1h - low_1h,
    (high_1h - close_1h.shift(1)).abs(),
    (low_1h  - close_1h.shift(1)).abs()
], axis=1).max(axis=1)
atr_1h = tr_1h.ewm(com=13, adjust=False).mean()

# ATR percentile (where is current ATR vs trailing 200h?)
atr_pct_rank = atr_1h.rolling(200).rank(pct=True)

atr_regime = pd.Series("mid_vol", index=ts_1h, dtype=object)
atr_regime[atr_pct_rank < 0.20] = "low_vol"     # compression
atr_regime[atr_pct_rank > 0.80] = "high_vol"    # expansion / spike
atr_regime[atr_pct_rank > 0.95] = "vol_spike"   # extreme spike

# Vol regime direction: is ATR expanding or contracting (momentum of vol)
atr_slope = (atr_1h - atr_1h.shift(6)) / atr_1h.shift(6)   # 6h change in ATR
vol_expanding   = atr_slope > 0.10   # ATR growing >10% over 6h
vol_contracting = atr_slope < -0.10  # ATR shrinking >10%

vol_direction = pd.Series("stable", index=ts_1h, dtype=object)
vol_direction[vol_expanding]   = "expanding"
vol_direction[vol_contracting] = "contracting"

# Vol-normalized move size: z-score of last 1h return vs rolling vol
log_ret_1h = np.log(close_1h / close_1h.shift(1))
vol_z = log_ret_1h / atr_1h.replace(0, float("nan"))   # how many ATRs did we move?
vol_z_sig = pd.Series("normal", index=ts_1h, dtype=object)
vol_z_sig[vol_z >  1.5] = "large_up"     # large up move → reversal?
vol_z_sig[vol_z < -1.5] = "large_down"   # large down move → reversal?
vol_z_sig[vol_z.abs() < 0.3] = "tiny"    # tiny move → continuation?

# ── 3. RSI on 4h and daily ─────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

rsi_4h_raw  = compute_rsi(close_4h, 14)
rsi_4h_1h   = rsi_4h_raw.reindex(ts_1h, method="ffill")

rsi_1d_raw  = compute_rsi(close_1d, 14)
rsi_1d_1h   = rsi_1d_raw.reindex(ts_1h, method="ffill")

rsi_1h_raw  = compute_rsi(close_1h, 14)

def rsi_to_signal(rsi):
    sig = pd.Series("neutral", index=rsi.index, dtype=object)
    sig[rsi < 30] = "oversold"
    sig[rsi > 70] = "overbought"
    sig[rsi < 20] = "extreme_oversold"
    sig[rsi > 80] = "extreme_overbought"
    return sig

rsi_4h_sig = rsi_to_signal(rsi_4h_1h)
rsi_1d_sig = rsi_to_signal(rsi_1d_1h)

# Multi-TF RSI agreement: 1h and 4h both in same extreme zone
rsi_1h_os = rsi_1h_raw < 30
rsi_1h_ob = rsi_1h_raw > 70
rsi_4h_os = rsi_4h_1h  < 30
rsi_4h_ob = rsi_4h_1h  > 70

rsi_mtf = pd.Series("disagree", index=ts_1h, dtype=object)
rsi_mtf[(rsi_1h_os) & (rsi_4h_os)] = "both_oversold"
rsi_mtf[(rsi_1h_ob) & (rsi_4h_ob)] = "both_overbought"
rsi_mtf[(rsi_1h_os) & (~rsi_4h_os) & (~rsi_4h_ob)] = "1h_oversold_only"
rsi_mtf[(rsi_1h_ob) & (~rsi_4h_ob) & (~rsi_4h_os)] = "1h_overbought_only"
rsi_mtf[(~rsi_1h_os) & (~rsi_1h_ob) & (rsi_4h_os)]  = "4h_oversold_only"
rsi_mtf[(~rsi_1h_os) & (~rsi_1h_ob) & (rsi_4h_ob)]  = "4h_overbought_only"

# ── 4. Stochastic on 1h bars (vs 15m in round 1) ──────────────────────────
ll_1h_14 = low_1h.rolling(14).min()
hh_1h_14 = high_1h.rolling(14).max()
hl_r_1h  = (hh_1h_14 - ll_1h_14).replace(0, float("nan"))
stk_1h   = ((close_1h - ll_1h_14) / hl_r_1h) * 100
std_1h   = stk_1h.rolling(3).mean()

stoch_1h_sig = pd.Series("neutral", index=ts_1h, dtype=object)
stoch_1h_sig[stk_1h < 20] = "oversold"
stoch_1h_sig[stk_1h > 80] = "overbought"
stoch_1h_sig[stk_1h < 10] = "extreme_oversold"
stoch_1h_sig[stk_1h > 90] = "extreme_overbought"

# ── 5. Williams %R (1h, 14 period) ─────────────────────────────────────────
wpr_14 = -100 * (hh_1h_14 - close_1h) / hl_r_1h
# Williams %R: -100 = at low (oversold), 0 = at high (overbought)
wpr_sig = pd.Series("neutral", index=ts_1h, dtype=object)
wpr_sig[wpr_14 < -80] = "oversold"     # near 14-bar low → expect bounce up
wpr_sig[wpr_14 > -20] = "overbought"   # near 14-bar high → expect fade

# ── 6. CCI (1h, 20 period) — deviation from typical price mean ─────────────
typical_1h = (high_1h + low_1h + close_1h) / 3
cci_mean   = typical_1h.rolling(20).mean()
cci_mad    = typical_1h.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
cci_1h     = (typical_1h - cci_mean) / (0.015 * cci_mad.replace(0, float("nan")))

cci_sig = pd.Series("neutral", index=ts_1h, dtype=object)
cci_sig[cci_1h < -100] = "oversold"     # extreme below → bounce
cci_sig[cci_1h >  100] = "overbought"   # extreme above → fade
cci_sig[cci_1h < -200] = "extreme_oversold"
cci_sig[cci_1h >  200] = "extreme_overbought"

# ── 7. Keltner Channel (1h, 20 EMA ± 2×ATR) ──────────────────────────────
ema20_1h  = close_1h.ewm(span=20, adjust=False).mean()
kc_upper  = ema20_1h + 2 * atr_1h
kc_lower  = ema20_1h - 2 * atr_1h
kc_width  = kc_upper - kc_lower

# %K position within Keltner: 0 = at lower, 1 = at upper
kc_pct    = (close_1h - kc_lower) / kc_width.replace(0, float("nan"))

kc_sig = pd.Series("mid", index=ts_1h, dtype=object)
kc_sig[kc_pct < 0.15]            = "lower_zone"   # near lower KC → bounce
kc_sig[kc_pct > 0.85]            = "upper_zone"   # near upper KC → fade
kc_sig[close_1h < kc_lower]      = "below_KC"     # below channel → very oversold
kc_sig[close_1h > kc_upper]      = "above_KC"     # above channel → very overbought

# ── 8. ATR regime + vol of vol ─────────────────────────────────────────────
# Vol of vol: standard deviation of ATR over last 20 bars (how unstable is vol?)
vov = atr_1h.rolling(20).std() / atr_1h.rolling(20).mean().replace(0, float("nan"))
vov_sig = pd.Series("stable_vol", index=ts_1h, dtype=object)
vov_sig[vov > vov.rolling(100).quantile(0.80)] = "unstable_vol"
vov_sig[vov < vov.rolling(100).quantile(0.20)] = "very_stable_vol"

# ── 9. Session time of day (UTC hour) ─────────────────────────────────────
hour_utc = pd.Series(ts_1h.hour, index=ts_1h)

# Traditional session buckets
def hour_to_session(h):
    if 0  <= h < 4:  return "asia_early"     # 00-04 UTC — quiet Asia
    if 4  <= h < 8:  return "asia_late"      # 04-08 UTC — Asia close / EU open
    if 8  <= h < 12: return "eu_morning"     # 08-12 UTC — London morning
    if 12 <= h < 16: return "us_open"        # 12-16 UTC — NY open + EU overlap
    if 16 <= h < 20: return "us_afternoon"   # 16-20 UTC — NY afternoon
    return "overnight"                        # 20-24 UTC — NY close / overnight

session = hour_utc.map(hour_to_session)

# ── 10. Day of week ────────────────────────────────────────────────────────
dow = pd.Series(ts_1h.dayofweek, index=ts_1h)  # 0=Mon, 6=Sun
dow_name = dow.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})

# ── 11. Candle close position within bar (1h) ─────────────────────────────
bar_range  = (high_1h - low_1h).replace(0, float("nan"))
close_pos  = (close_1h - low_1h) / bar_range   # 0 = closed at low, 1 = at high

close_pos_sig = pd.Series("mid", index=ts_1h, dtype=object)
close_pos_sig[close_pos > 0.80] = "strong_close_up"    # closed near top of bar
close_pos_sig[close_pos < 0.20] = "strong_close_down"  # closed near bottom
close_pos_sig[close_pos.between(0.40, 0.60)] = "doji"  # indecision

# ── 12. Consecutive bars (exhaustion) ─────────────────────────────────────
direction_1h = (close_1h > close_1h.shift(1)).astype(int)  # 1=up, 0=down

def count_consecutive(series):
    """Count consecutive same-direction bars ending at each point."""
    arr = series.values
    counts = np.zeros(len(arr))
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            counts[i] = counts[i-1] + 1
        else:
            counts[i] = 1
    return pd.Series(counts, index=series.index)

consec = count_consecutive(direction_1h)
consec_sig = pd.Series("1-2_bars", index=ts_1h, dtype=object)
consec_sig[direction_1h == 1] = "streak_up"
consec_sig[direction_1h == 0] = "streak_down"
# Override with streak length
streak_up_long   = (direction_1h == 1) & (consec >= 4)
streak_down_long = (direction_1h == 0) & (consec >= 4)
consec_sig[streak_up_long]   = "long_streak_up_4+"
consec_sig[streak_down_long] = "long_streak_down_4+"
streak_up_med   = (direction_1h == 1) & (consec == 3)
streak_down_med = (direction_1h == 0) & (consec == 3)
consec_sig[streak_up_med]   = "streak_up_3"
consec_sig[streak_down_med] = "streak_down_3"

# Simpler: just count how many of last N bars were up
for n_bars in [3, 5, 8]:
    pct_up = direction_1h.rolling(n_bars).mean()
    sig = pd.Series("balanced", index=ts_1h, dtype=object)
    sig[pct_up >= 0.8] = "mostly_up"
    sig[pct_up <= 0.2] = "mostly_down"
    locals()[f"pct_up_{n_bars}b_sig"] = sig
    locals()[f"pct_up_{n_bars}b_raw"] = pct_up

# ── 13. Inside / outside bar ───────────────────────────────────────────────
inside_bar  = (high_1h < high_1h.shift(1)) & (low_1h > low_1h.shift(1))
outside_bar = (high_1h > high_1h.shift(1)) & (low_1h < low_1h.shift(1))

bar_type = pd.Series("normal", index=ts_1h, dtype=object)
bar_type[inside_bar]  = "inside"    # compression → continuation or breakout
bar_type[outside_bar] = "outside"   # expansion → exhaustion or new trend

# After inside bar: what happens next?  (signal is on the CURRENT bar = inside, predicting NEXT)
after_inside  = inside_bar.shift(1).fillna(False)
after_outside = outside_bar.shift(1).fillna(False)
post_bar_sig  = pd.Series("after_normal", index=ts_1h, dtype=object)
post_bar_sig[after_inside]  = "after_inside"
post_bar_sig[after_outside] = "after_outside"

# ── 14. Multi-TF EMA trend confluence ──────────────────────────────────────
# 15m, 1h, 4h all have EMA20 above EMA50 → strong bullish confluence
ema20_4h = close_4h.ewm(span=20, adjust=False).mean()
ema50_4h = close_4h.ewm(span=50, adjust=False).mean()
ema_bull_4h = (ema20_4h > ema50_4h).reindex(ts_1h, method="ffill")
ema_bear_4h = (ema20_4h < ema50_4h).reindex(ts_1h, method="ffill")

ema20_1h_s = close_1h.ewm(span=20, adjust=False).mean()
ema50_1h_s = close_1h.ewm(span=50, adjust=False).mean()
ema_bull_1h = ema20_1h_s > ema50_1h_s
ema_bear_1h = ema20_1h_s < ema50_1h_s

ema20_15m = df_15m["close"].ewm(span=20, adjust=False).mean()
ema50_15m = df_15m["close"].ewm(span=50, adjust=False).mean()
ema_bull_15m = (ema20_15m > ema50_15m).resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
ema_bear_15m = (ema20_15m < ema50_15m).resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

mtf_ema = pd.Series("mixed", index=ts_1h, dtype=object)
mtf_ema[ema_bull_15m & ema_bull_1h & ema_bull_4h] = "all_bullish"
mtf_ema[ema_bear_15m & ema_bear_1h & ema_bear_4h] = "all_bearish"
mtf_ema[ema_bull_1h & ema_bull_4h & ~ema_bull_15m] = "1h4h_bull_15m_bear"
mtf_ema[ema_bear_1h & ema_bear_4h & ~ema_bear_15m] = "1h4h_bear_15m_bull"

# ── 15. Distance from 20-bar swing high/low ────────────────────────────────
swing_high_20 = high_1h.rolling(20).max()
swing_low_20  = low_1h.rolling(20).min()
dist_from_high = (close_1h - swing_high_20) / swing_high_20   # always ≤0
dist_from_low  = (close_1h - swing_low_20)  / swing_low_20    # always ≥0

# Where in the swing range is price?
swing_rng = swing_high_20 - swing_low_20
swing_pct = (close_1h - swing_low_20) / swing_rng.replace(0, float("nan"))

swing_sig = pd.Series("mid_swing", index=ts_1h, dtype=object)
swing_sig[swing_pct < 0.10] = "at_swing_low"
swing_sig[swing_pct > 0.90] = "at_swing_high"
swing_sig[swing_pct < 0.25] = "lower_swing"
swing_sig[swing_pct > 0.75] = "upper_swing"

# ── 16. Daily Pivot Points ─────────────────────────────────────────────────
# Classic pivot: P = (H + L + C) / 3
# R1 = 2P - L,  S1 = 2P - H
# Compute from PREVIOUS day's OHLC, forward fill to current day
prev_high  = ohlcv_1d["high"].shift(1)
prev_low   = ohlcv_1d["low"].shift(1)
prev_close = ohlcv_1d["close"].shift(1)

pivot     = (prev_high + prev_low + prev_close) / 3
pivot_r1  = 2 * pivot - prev_low
pivot_s1  = 2 * pivot - prev_high
pivot_r2  = pivot + (prev_high - prev_low)
pivot_s2  = pivot - (prev_high - prev_low)

pivot_1h   = pivot.reindex(ts_1h, method="ffill")
pivot_r1_1h= pivot_r1.reindex(ts_1h, method="ffill")
pivot_s1_1h= pivot_s1.reindex(ts_1h, method="ffill")

pivot_zone = pd.Series("between_s1_r1", index=ts_1h, dtype=object)
pivot_zone[close_1h > pivot_r1_1h] = "above_R1"
pivot_zone[close_1h < pivot_s1_1h] = "below_S1"
pivot_zone[(close_1h >= pivot_1h) & (close_1h <= pivot_r1_1h)] = "pivot_to_R1"
pivot_zone[(close_1h >= pivot_s1_1h) & (close_1h < pivot_1h)]  = "S1_to_pivot"

# ── 17. Vol of Vol (second-order volatility) ───────────────────────────────
# Already computed vov above — also test ATR trend
atr_trend = pd.Series("stable", index=ts_1h, dtype=object)
atr_trend[atr_slope >  0.20] = "vol_spiking"     # sharp vol expansion
atr_trend[atr_slope < -0.20] = "vol_collapsing"  # sharp vol contraction

# ── 18. Vol-adjusted move z-score (custom) ─────────────────────────────────
# Z-score: how extreme was the last 1h move relative to recent realized vol?
# Negative = large down move (potential bounce) — mean reversion signal
roll_vol = log_ret_1h.rolling(24).std()
move_z   = log_ret_1h / roll_vol.replace(0, float("nan"))

move_z_sig = pd.Series("normal", index=ts_1h, dtype=object)
move_z_sig[move_z >  2.0] = "large_up_move"    # >2σ up last hour
move_z_sig[move_z < -2.0] = "large_down_move"  # >2σ down last hour
move_z_sig[move_z >  1.5] = "big_up"
move_z_sig[move_z < -1.5] = "big_down"
move_z_sig[move_z.abs() < 0.25] = "flat"

print("  All indicators computed.\n")


# ════════════════════════════════════════════════════════════════════════════════
# BUILD MASTER DATASET
# ════════════════════════════════════════════════════════════════════════════════
test_mask = ts_1h >= TEST_START
idx_test  = np.where(test_mask)[0][:-1]

master = pd.DataFrame({
    "ts":          ts_1h[idx_test],
    "next_ret":    next_ret.values[idx_test],
    "next_up":     next_up.values[idx_test],
    # Donchian
    "dc20_pct":    dc_20_pct.values[idx_test],
    "dc50_pct":    dc_50_pct.values[idx_test],
    "dc100_pct":   dc_100_pct.values[idx_test],
    "dc4h_pct":    dc_4h_pct.values[idx_test],
    "dc20_sig":    dc20_sig.values[idx_test],
    "dc50_sig":    dc50_sig.values[idx_test],
    "dc100_sig":   dc100_sig.values[idx_test],
    "dc4h_sig":    dc4h_sig.values[idx_test],
    # Vol score
    "atr_regime":  atr_regime.values[idx_test],
    "vol_dir":     vol_direction.values[idx_test],
    "vol_z_sig":   vol_z_sig.values[idx_test],
    "vol_z_raw":   vol_z.values[idx_test],
    "atr_pct":     atr_pct_rank.values[idx_test],
    # RSI multi-TF
    "rsi_4h_sig":  rsi_4h_sig.values[idx_test],
    "rsi_1d_sig":  rsi_1d_sig.values[idx_test],
    "rsi_mtf":     rsi_mtf.values[idx_test],
    "rsi_4h_raw":  rsi_4h_1h.values[idx_test],
    # Stoch 1h
    "stoch_1h_sig":stoch_1h_sig.values[idx_test],
    "stk_1h_raw":  stk_1h.values[idx_test],
    # Williams %R
    "wpr_sig":     wpr_sig.values[idx_test],
    "wpr_raw":     wpr_14.values[idx_test],
    # CCI
    "cci_sig":     cci_sig.values[idx_test],
    "cci_raw":     cci_1h.values[idx_test],
    # Keltner
    "kc_sig":      kc_sig.values[idx_test],
    "kc_pct":      kc_pct.values[idx_test],
    # Vol of vol / ATR trend
    "vov_sig":     vov_sig.values[idx_test],
    "atr_trend":   atr_trend.values[idx_test],
    # Session / calendar
    "session":     session.values[idx_test],
    "dow":         dow_name.values[idx_test],
    # Candle patterns
    "close_pos_sig":close_pos_sig.values[idx_test],
    "close_pos_raw":close_pos.values[idx_test],
    "bar_type":    bar_type.values[idx_test],
    "post_bar_sig":post_bar_sig.values[idx_test],
    # Consecutive bars
    "pct3b_sig":   pct_up_3b_sig.values[idx_test],
    "pct5b_sig":   pct_up_5b_sig.values[idx_test],
    "pct8b_sig":   pct_up_8b_sig.values[idx_test],
    "pct3b_raw":   pct_up_3b_raw.values[idx_test],
    "pct5b_raw":   pct_up_5b_raw.values[idx_test],
    # Multi-TF EMA
    "mtf_ema":     mtf_ema.values[idx_test],
    # Swing position
    "swing_sig":   swing_sig.values[idx_test],
    "swing_pct":   swing_pct.values[idx_test],
    # Pivot points
    "pivot_zone":  pivot_zone.values[idx_test],
    # Move z-score
    "move_z_sig":  move_z_sig.values[idx_test],
    "move_z_raw":  move_z.values[idx_test],
}).dropna(subset=["next_ret"])

N_TOTAL = len(master)
UP_BASE = master["next_up"].mean()

print(f"Test hours: {N_TOTAL:,}")
print(f"Baseline up%: {UP_BASE:.1%}")
print(f"Mean 1h return: {master['next_ret'].mean():.4%}\n")


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def analyze_categorical(df, col, label, ordered_states=None, min_n=30,
                        bullish_states=None, bearish_states=None):
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Baseline: up% = {UP_BASE:.1%}")
    print(SEP2)
    print(f"  {'State':>24}  {'n':>6}  {'up%':>7}  {'vs_base':>8}  {'mean_ret%':>10}  {'p-val':>8}  signal")
    print("  " + "-" * 76)

    states = ordered_states if ordered_states else sorted(df[col].dropna().unique())
    scorecard_entries = []
    for state in states:
        sub = df[df[col] == state].dropna(subset=["next_ret"])
        if len(sub) < min_n:
            continue
        n    = len(sub)
        up_p = sub["next_up"].mean()
        mr   = sub["next_ret"].mean() * 100
        diff = up_p - UP_BASE
        se   = (UP_BASE * (1 - UP_BASE) / n) ** 0.5
        z    = diff / se if se > 0 else 0
        pval = 2 * (1 - norm.cdf(abs(z)))

        if bullish_states and state in bullish_states:
            sig = ("★ STRONG" if diff > 0.05 and pval < 0.05 else
                   "good"     if diff > 0.02 and pval < 0.10 else
                   "weak"     if diff > 0    else "FAIL")
        elif bearish_states and state in bearish_states:
            sig = ("★ STRONG" if diff < -0.05 and pval < 0.05 else
                   "good"     if diff < -0.02 and pval < 0.10 else
                   "weak"     if diff < 0     else "FAIL")
        else:
            sig = ""

        pval_s = f"{pval:.3f}" if pval < 0.999 else ">0.99"
        print(f"  {str(state):>24}  {n:>6,}  {up_p:>6.1%}  {diff:>+7.1%}  {mr:>+9.3f}%  {pval_s:>8}  {sig}")
        scorecard_entries.append((state, n, up_p, diff, mr, pval, sig))
    return scorecard_entries


def analyze_continuous(df, col, label, min_n=50):
    sub = df[[col, "next_ret", "next_up"]].dropna()
    if len(sub) < min_n:
        print(f"\n  {label}: insufficient data ({len(sub)})")
        return
    r_p, p_p = pearsonr(sub[col],  sub["next_ret"])
    r_s, p_s = spearmanr(sub[col], sub["next_ret"])
    sig = "★★★" if abs(r_s) > 0.05 and p_s < 0.01 else ("★★" if abs(r_s) > 0.03 and p_s < 0.05 else ("★" if p_s < 0.10 else "—"))
    print(f"\n  {label}")
    print(f"    Pearson r={r_p:+.4f} p={p_p:.4f} | Spearman r={r_s:+.4f} p={p_s:.4f}  {sig}")


# ════════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

# 1. Donchian Channel
for col, label, period in [
    ("dc20_sig",  "1a. DONCHIAN CHANNEL 20-bar (1h) — ~20h range", 20),
    ("dc50_sig",  "1b. DONCHIAN CHANNEL 50-bar (1h) — ~2 days",    50),
    ("dc100_sig", "1c. DONCHIAN CHANNEL 100-bar (1h) — ~4 days",   100),
    ("dc4h_sig",  "1d. DONCHIAN CHANNEL 20-bar (4h) — ~3 days",     20),
]:
    analyze_categorical(master, col, label,
        ordered_states=["near_low","lower_zone","mid","upper_zone","near_high"],
        bullish_states=["near_low","lower_zone"],
        bearish_states=["near_high","upper_zone"])

for col, label in [("dc20_pct","DC20 %pos"),("dc50_pct","DC50 %pos"),
                   ("dc100_pct","DC100 %pos"),("dc4h_pct","DC4h %pos")]:
    analyze_continuous(master, col, label)

# 2. Vol Score
analyze_categorical(master, "atr_regime",
    "2a. ATR REGIME — vol level (1h ATR vs 200-bar percentile)",
    ordered_states=["low_vol","mid_vol","high_vol","vol_spike"])

analyze_categorical(master, "vol_dir",
    "2b. VOL DIRECTION — is ATR expanding or contracting?",
    ordered_states=["contracting","stable","expanding"],
    bullish_states=["contracting"], bearish_states=["expanding"])

analyze_categorical(master, "vol_z_sig",
    "2c. VOL-NORMALIZED MOVE — last hour move in units of ATR",
    ordered_states=["large_down","big_down","normal","flat","big_up","large_up"],
    bullish_states=["large_down","big_down"],
    bearish_states=["large_up","big_up"])

analyze_continuous(master, "move_z_raw", "Vol-normalized move z-score (continuous)")
analyze_continuous(master, "atr_pct",    "ATR percentile rank (continuous)")

# 3. RSI multi-timeframe
analyze_categorical(master, "rsi_4h_sig",
    "3a. RSI 4h (14-period, forward-filled to 1h)",
    ordered_states=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bullish_states=["oversold","extreme_oversold"],
    bearish_states=["overbought","extreme_overbought"])

analyze_categorical(master, "rsi_1d_sig",
    "3b. RSI Daily (14-period)",
    ordered_states=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bullish_states=["oversold","extreme_oversold"],
    bearish_states=["overbought","extreme_overbought"])

analyze_categorical(master, "rsi_mtf",
    "3c. RSI Multi-TF agreement (1h + 4h both extreme?)",
    ordered_states=["both_oversold","1h_oversold_only","4h_oversold_only",
                    "disagree","4h_overbought_only","1h_overbought_only","both_overbought"],
    bullish_states=["both_oversold","1h_oversold_only","4h_oversold_only"],
    bearish_states=["both_overbought","1h_overbought_only","4h_overbought_only"])

analyze_continuous(master, "rsi_4h_raw", "RSI 4h raw value")

# 4. Stochastic 1h
analyze_categorical(master, "stoch_1h_sig",
    "4. STOCHASTIC on 1h bars (14/3) — compare to 15m in round 1",
    ordered_states=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bullish_states=["oversold","extreme_oversold"],
    bearish_states=["overbought","extreme_overbought"])

analyze_continuous(master, "stk_1h_raw", "Stoch K (1h bars) raw value")

# 5. Williams %R
analyze_categorical(master, "wpr_sig",
    "5. WILLIAMS %R (1h, 14-period)",
    ordered_states=["oversold","neutral","overbought"],
    bullish_states=["oversold"], bearish_states=["overbought"])

analyze_continuous(master, "wpr_raw", "Williams %R raw value")

# 6. CCI
analyze_categorical(master, "cci_sig",
    "6. CCI (1h, 20-period)",
    ordered_states=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bullish_states=["oversold","extreme_oversold"],
    bearish_states=["overbought","extreme_overbought"])

analyze_continuous(master, "cci_raw", "CCI raw value")

# 7. Keltner Channel
analyze_categorical(master, "kc_sig",
    "7. KELTNER CHANNEL (1h, 20 EMA ± 2×ATR)",
    ordered_states=["below_KC","lower_zone","mid","upper_zone","above_KC"],
    bullish_states=["below_KC","lower_zone"],
    bearish_states=["above_KC","upper_zone"])

analyze_continuous(master, "kc_pct", "Keltner %position")

# 8. Vol of Vol / ATR trend
analyze_categorical(master, "atr_trend",
    "8a. ATR TREND — is volatility spiking or collapsing?",
    ordered_states=["vol_collapsing","stable","vol_spiking"])

analyze_categorical(master, "vov_sig",
    "8b. VOL OF VOL — stability of volatility itself",
    ordered_states=["very_stable_vol","stable_vol","unstable_vol"])

# 9. Session time
analyze_categorical(master, "session",
    "9. SESSION TIME (UTC hour buckets)",
    ordered_states=["asia_early","asia_late","eu_morning","us_open","us_afternoon","overnight"])

# 10. Day of week
analyze_categorical(master, "dow",
    "10. DAY OF WEEK",
    ordered_states=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

# 11. Candle close position
analyze_categorical(master, "close_pos_sig",
    "11. CANDLE CLOSE POSITION within 1h bar",
    ordered_states=["strong_close_down","mid","doji","strong_close_up"],
    bullish_states=["strong_close_up"], bearish_states=["strong_close_down"])

analyze_continuous(master, "close_pos_raw", "Close position %  (0=at low, 1=at high)")

# 12. Consecutive bars / recent direction dominance
analyze_categorical(master, "pct3b_sig",
    "12a. % UP bars in last 3 hours (mean reversion vs continuation?)",
    ordered_states=["mostly_down","balanced","mostly_up"],
    bullish_states=["mostly_down"], bearish_states=["mostly_up"])

analyze_categorical(master, "pct5b_sig",
    "12b. % UP bars in last 5 hours",
    ordered_states=["mostly_down","balanced","mostly_up"],
    bullish_states=["mostly_down"], bearish_states=["mostly_up"])

analyze_categorical(master, "pct8b_sig",
    "12c. % UP bars in last 8 hours",
    ordered_states=["mostly_down","balanced","mostly_up"],
    bullish_states=["mostly_down"], bearish_states=["mostly_up"])

analyze_continuous(master, "pct3b_raw", "% up bars last 3h raw")
analyze_continuous(master, "pct5b_raw", "% up bars last 5h raw")

# 13. Inside/outside bar
analyze_categorical(master, "bar_type",
    "13a. BAR TYPE — inside / outside / normal",
    ordered_states=["inside","normal","outside"])

analyze_categorical(master, "post_bar_sig",
    "13b. POST BAR — next bar after inside/outside",
    ordered_states=["after_inside","after_normal","after_outside"])

# 14. Multi-TF EMA confluence
analyze_categorical(master, "mtf_ema",
    "14. MULTI-TF EMA CONFLUENCE (15m + 1h + 4h)",
    ordered_states=["all_bullish","1h4h_bull_15m_bear","mixed","1h4h_bear_15m_bull","all_bearish"],
    bullish_states=["all_bullish"], bearish_states=["all_bearish"])

# 15. Swing position
analyze_categorical(master, "swing_sig",
    "15. SWING POSITION — where in 20-bar high/low range?",
    ordered_states=["at_swing_low","lower_swing","mid_swing","upper_swing","at_swing_high"],
    bullish_states=["at_swing_low","lower_swing"],
    bearish_states=["at_swing_high","upper_swing"])

analyze_continuous(master, "swing_pct", "Swing position % (0=at low, 1=at high)")

# 16. Pivot points
analyze_categorical(master, "pivot_zone",
    "16. DAILY PIVOT POINTS — price vs P, R1, S1",
    ordered_states=["below_S1","S1_to_pivot","pivot_to_R1","above_R1"],
    bullish_states=["below_S1"], bearish_states=["above_R1"])

# 17. Move z-score
analyze_categorical(master, "move_z_sig",
    "17. VOL-ADJUSTED MOVE Z-SCORE — last hour move in σ units",
    ordered_states=["large_down_move","big_down","flat","normal","big_up","large_up_move"],
    bullish_states=["large_down_move","big_down"],
    bearish_states=["large_up_move","big_up"])


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY SCORECARD — All round-2 indicators ranked by directional reliability")
print(f"  Baseline up% = {UP_BASE:.1%}  |  n≥100 and p<0.10 to qualify")
print(SEP)

scorecard = []

def score_it(df, col, bull_states, bear_states, label, min_n=100, max_p=0.10):
    best_edge = 0.0
    best_dir  = "—"
    best_st   = "—"
    best_n    = 0
    for state in bull_states + bear_states:
        sub = df[df[col] == state].dropna(subset=["next_ret"])
        if len(sub) < min_n: continue
        up_p = sub["next_up"].mean()
        diff = (up_p - UP_BASE) if state in bull_states else (UP_BASE - up_p)
        se   = (UP_BASE*(1-UP_BASE)/len(sub))**0.5
        pval = 2*(1-norm.cdf(abs(diff/se))) if se > 0 else 1.0
        if diff > best_edge and pval < max_p:
            best_edge = diff
            best_dir  = "bullish" if state in bull_states else "bearish"
            best_st   = str(state)
            best_n    = len(sub)
    scorecard.append((label, best_edge, best_dir, best_st, best_n))

# Donchian
for col, lbl in [("dc20_sig","Donchian 20-bar (1h)"),("dc50_sig","Donchian 50-bar (1h)"),
                  ("dc100_sig","Donchian 100-bar (1h)"),("dc4h_sig","Donchian 20-bar (4h)")]:
    score_it(master, col, ["near_low","lower_zone"], ["near_high","upper_zone"], lbl)

# Vol
score_it(master, "atr_regime", [], ["vol_spike","high_vol"],            "ATR Regime (vol level)")
score_it(master, "vol_dir",    ["contracting"], ["expanding"],           "Vol Direction (ATR slope)")
score_it(master, "vol_z_sig",  ["large_down","big_down"],["large_up","big_up"], "Vol-adjusted move (z-score)")

# RSI multi-TF
score_it(master, "rsi_4h_sig", ["oversold","extreme_oversold"],["overbought","extreme_overbought"], "RSI 4h")
score_it(master, "rsi_1d_sig", ["oversold","extreme_oversold"],["overbought","extreme_overbought"], "RSI Daily")
score_it(master, "rsi_mtf",    ["both_oversold","1h_oversold_only"],["both_overbought","1h_overbought_only"], "RSI Multi-TF (1h+4h)")

# Oscillators
score_it(master, "stoch_1h_sig", ["oversold","extreme_oversold"],["overbought","extreme_overbought"], "Stochastic 1h bars")
score_it(master, "wpr_sig",      ["oversold"],                   ["overbought"],                      "Williams %R (1h)")
score_it(master, "cci_sig",      ["oversold","extreme_oversold"],["overbought","extreme_overbought"], "CCI (1h, 20p)")
score_it(master, "kc_sig",       ["below_KC","lower_zone"],      ["above_KC","upper_zone"],           "Keltner Channel (1h)")

# Vol of vol
score_it(master, "atr_trend",  [], ["vol_spiking"],    "ATR Trend (vol spiking)")
score_it(master, "vov_sig",    [], [],                  "Vol of Vol")

# Calendar
score_it(master, "session", [], [],   "Session time of day")
score_it(master, "dow",     [], [],   "Day of week")

# Price action
score_it(master, "close_pos_sig", ["strong_close_up"], ["strong_close_down"], "Candle close position")
score_it(master, "bar_type",      [],                  [],                    "Bar type (inside/outside)")
score_it(master, "post_bar_sig",  [],                  [],                    "Post inside/outside bar")

# Consecutive
score_it(master, "pct3b_sig",  ["mostly_down"], ["mostly_up"], "% up bars last 3h")
score_it(master, "pct5b_sig",  ["mostly_down"], ["mostly_up"], "% up bars last 5h")
score_it(master, "pct8b_sig",  ["mostly_down"], ["mostly_up"], "% up bars last 8h")

# Structure
score_it(master, "mtf_ema",    ["all_bullish"], ["all_bearish"],              "Multi-TF EMA confluence")
score_it(master, "swing_sig",  ["at_swing_low","lower_swing"],["at_swing_high","upper_swing"], "Swing position")
score_it(master, "pivot_zone", ["below_S1"],    ["above_R1"],                 "Daily pivot points")
score_it(master, "move_z_sig", ["large_down_move","big_down"],["large_up_move","big_up"],     "Move z-score (1h)")

scorecard.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Indicator':>38}  {'edge':>7}  {'direction':>10}  {'best_state':>22}  {'n':>6}")
print("  " + "-" * 95)
for label, edge, direction, state, n in scorecard:
    stars = ("★★★" if edge > 0.05 else "★★ " if edge > 0.03 else "★  " if edge > 0.01 else "   ")
    flag  = "" if edge == 0 else ""
    print(f"  {label:>38}  {edge:>+6.1%}  {direction:>10}  {state:>22}  {n:>6,}  {stars}")

print(f"""
Round 1 champion recap (for comparison):
  Stochastic zone (15m)  +8.4% ★★★
  VWAP Position          +8.3% ★★★
  EMA Stretch (5m)       +6.7% ★★★
  RSI (1h)               +5.0% ★★★
  BB Position (1h)       +4.1% ★★

Any round-2 indicator beating +4.1% is a genuine addition.
""")
