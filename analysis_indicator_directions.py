"""
analysis_indicator_directions.py — Systematic indicator directional correlation test.

Goal: For each indicator, determine whether its signal state (bullish/bearish/neutral)
reliably predicts the DIRECTION of the next 1h BTC price move.

This is pure directional accuracy testing — not PnL simulation, not option pricing.
The question is simply: when indicator X says "bullish", does price actually go up?

Metrics per indicator state:
  - n_hours        : sample size
  - mean_return    : average 1h log return (positive = bullish confirmation)
  - dir_accuracy   : % of hours where price moved in the predicted direction
  - hit_up         : % of hours price moved up (vs signal says bullish)
  - hit_down       : % of hours price moved down (vs signal says bearish)
  - IC             : information coefficient — Pearson correlation of signal
                     strength (if continuous) with actual 1h return

Indicators tested:
  1.  EMA Alignment      — 1h, EMA-20 vs EMA-50
  2.  EMA Stack          — 15m, EMA-9/21/50 stack
  3.  EMA Slope          — 1h EMA-20 slope (rate of change, 3-bar)
  4.  EMA Stretch        — 5m EMA-20 deviation (mean reversion)
  5.  RSI                — 1h, 14-period (oversold/neutral/overbought)
  6.  MACD               — 1h, 12/26/9 (signal cross + histogram)
  7.  Stochastic K/D     — 15m, 14/3 (crossover + zone)
  8.  ADX + Direction    — 15m, 14-period (strength + DI direction)
  9.  Bollinger Position — 1h, 20/2 (where in bands is price?)
  10. Rate of Change     — 1h, multiple lookback periods (1h, 4h, 12h, 24h)
  11. Volume Momentum    — 1h, volume vs 20-bar MA (confirm or diverge)
  12. VWAP Position      — daily session VWAP vs current price

Test set: Jan 2025 – Apr 2026 (walk-forward, calibrated on 2024)
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

SEP  = "=" * 76
SEP2 = "-" * 76
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# ── Load data ─────────────────────────────────────────────────────────────────
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
volume_1h = pd.Series(ohlcv_1h["volume"].values.astype(float), index=ohlcv_1h.index)
ts_1h     = ohlcv_1h.index
n1h       = len(close_1h)

# Next-hour log return — what we're predicting
log_ret_1h = np.log(close_1h / close_1h.shift(1))
next_ret   = log_ret_1h.shift(-1)       # return in the NEXT hour (the prediction target)
next_up    = (next_ret > 0).astype(int) # 1 if price went up, 0 if down

print(f"  1h bars: {n1h:,}")
print(f"  Test hours: {(ts_1h >= TEST_START).sum():,}")
print(f"  Mean 1h return: {log_ret_1h.mean():.4%}  Std: {log_ret_1h.std():.4%}")


# ════════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL INDICATORS
# ════════════════════════════════════════════════════════════════════════════════
print("\nComputing all indicators...")

# ── 1. EMA Alignment (1h EMA-20 vs EMA-50) ───────────────────────────────────
ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
ema50_1h = close_1h.ewm(span=50, adjust=False).mean()
CONFIRM  = 3

ema_align = pd.Series("neutral", index=ts_1h, dtype=object)
for i in range(CONFIRM, n1h):
    e20 = ema20_1h.values[i-CONFIRM:i]
    e50 = ema50_1h.values[i-CONFIRM:i]
    cl  = close_1h.values[i-CONFIRM:i]
    if all(e20 > e50) and all(cl > e20):
        ema_align.iat[i] = "bullish"
    elif all(e20 < e50) or all(cl < e50):
        ema_align.iat[i] = "bearish"

# Continuous signal: spread (EMA20 - EMA50) / price — positive = bullish
ema_spread = (ema20_1h - ema50_1h) / close_1h

# ── 2. EMA Stack (15m EMA-9/21/50) ───────────────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

sk9_15m  = df_15m["close"].ewm(span=9,  adjust=False).mean()
sk21_15m = df_15m["close"].ewm(span=21, adjust=False).mean()
sk50_15m = df_15m["close"].ewm(span=50, adjust=False).mean()
cl_15m   = df_15m["close"]

bull_stack_15m = (sk9_15m > sk21_15m) & (sk21_15m > sk50_15m) & (cl_15m > sk9_15m)
bear_stack_15m = (sk9_15m < sk21_15m) & (sk21_15m < sk50_15m) & (cl_15m < sk9_15m)

ema_stack_raw = pd.Series(0, index=df_15m.index)
ema_stack_raw[bull_stack_15m] = 1
ema_stack_raw[bear_stack_15m] = -1

# Resample to 1h (last 15m bar in hour)
ema_stack_1h = ema_stack_raw.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

# ── 3. EMA Slope (1h EMA-20, rate of change over 3 bars) ─────────────────────
ema20_slope = (ema20_1h - ema20_1h.shift(3)) / ema20_1h.shift(3)

ema_slope_signal = pd.Series("neutral", index=ts_1h, dtype=object)
slope_thresh = 0.001  # 0.1% change in EMA over 3 hours
ema_slope_signal[ema20_slope >  slope_thresh] = "bullish"
ema_slope_signal[ema20_slope < -slope_thresh] = "bearish"

# ── 4. EMA Stretch (5m EMA-20 deviation, mean reversion) ─────────────────────
close_1m_s = pd.Series(ohlcv_1m["close"].values.astype(float), index=ohlcv_1m.index)
df_5m_close = close_1m_s.resample("5min", origin="start_day").last().dropna()
ema20_5m    = df_5m_close.ewm(span=20, adjust=False).mean()
stretch_5m  = (df_5m_close - ema20_5m) / ema20_5m

stretch_1h  = stretch_5m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
STRETCH_THRESH = 0.001  # ±0.1%
ema_stretch_signal = pd.Series("neutral", index=ts_1h, dtype=object)
ema_stretch_signal[stretch_1h >  STRETCH_THRESH] = "overbought"  # expect down (-1)
ema_stretch_signal[stretch_1h < -STRETCH_THRESH] = "oversold"    # expect up  (+1)

# ── 5. RSI (1h, 14-period) ───────────────────────────────────────────────────
delta_1h = close_1h.diff()
gain_1h  = delta_1h.clip(lower=0).ewm(com=13, adjust=False).mean()
loss_1h  = (-delta_1h.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi_1h   = 100 - (100 / (1 + gain_1h / loss_1h.replace(0, 1e-10)))

rsi_signal = pd.Series("neutral", index=ts_1h, dtype=object)
rsi_signal[rsi_1h < 30] = "oversold"     # expect up
rsi_signal[rsi_1h > 70] = "overbought"   # expect down
# Fine-grained RSI zones
rsi_zone = pd.cut(rsi_1h,
    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
    labels=["<20","20-30","30-40","40-50","50-60","60-70","70-80",">80"]
)

# ── 6. MACD (1h, 12/26/9) ────────────────────────────────────────────────────
ema12_1h   = close_1h.ewm(span=12, adjust=False).mean()
ema26_1h   = close_1h.ewm(span=26, adjust=False).mean()
macd_line  = ema12_1h - ema26_1h
macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
macd_hist  = macd_line - macd_sig

# Signal states
macd_signal = pd.Series("neutral", index=ts_1h, dtype=object)
# MACD above signal line AND histogram positive = bullish momentum
macd_signal[(macd_line > macd_sig) & (macd_hist > 0)] = "bullish"
macd_signal[(macd_line < macd_sig) & (macd_hist < 0)] = "bearish"

# MACD crossover (freshly crossed)
macd_crossed_up   = (macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))
macd_crossed_down = (macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))

macd_cross_signal = pd.Series("none", index=ts_1h, dtype=object)
macd_cross_signal[macd_crossed_up]   = "crossed_up"
macd_cross_signal[macd_crossed_down] = "crossed_down"
# Hold for 3 bars after crossover
for shift in [1, 2]:
    macd_cross_signal[macd_crossed_up.shift(shift).fillna(False)   & (macd_cross_signal == "none")] = "crossed_up_lag"
    macd_cross_signal[macd_crossed_down.shift(shift).fillna(False) & (macd_cross_signal == "none")] = "crossed_down_lag"

# ── 7. Stochastic K/D (15m, 14/3) ────────────────────────────────────────────
ll_14 = df_15m["low"].rolling(14).min()
hh_14 = df_15m["high"].rolling(14).max()
hl_r  = (hh_14 - ll_14).replace(0, float("nan"))
stk   = ((df_15m["close"] - ll_14) / hl_r) * 100
std   = stk.rolling(3).mean()

# Crossover
stk_prev = stk.shift(1)
std_prev = std.shift(1)
xup_15m  = (stk_prev < std_prev) & (stk > std) & (stk_prev < 20)
xdn_15m  = (stk_prev > std_prev) & (stk < std) & (stk_prev > 80)

stk_1h   = stk.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
std_1h   = std.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
xup_1h   = xup_15m.resample("1h", origin="start_day").max().reindex(ts_1h, method="ffill").fillna(False)
xdn_1h   = xdn_15m.resample("1h", origin="start_day").max().reindex(ts_1h, method="ffill").fillna(False)

stoch_signal = pd.Series("neutral", index=ts_1h, dtype=object)
stoch_signal[stk_1h < 20]  = "oversold"
stoch_signal[stk_1h > 80]  = "overbought"
stoch_signal[xup_1h]       = "crossup"      # override: crossup from oversold
stoch_signal[xdn_1h]       = "crossdown"    # override: crossdown from overbought

# ── 8. ADX + DI Direction (15m, 14-period) ───────────────────────────────────
_h   = df_15m["high"]
_l   = df_15m["low"]
_cp  = df_15m["close"].shift(1)
tr_  = pd.concat([_h-_l, (_h-_cp).abs(), (_l-_cp).abs()], axis=1).max(axis=1)
dmp_ = (_h-_h.shift(1)).clip(lower=0).where((_h-_h.shift(1))>(_l.shift(1)-_l), 0)
dmm_ = (_l.shift(1)-_l).clip(lower=0).where((_l.shift(1)-_l)>(_h-_h.shift(1)), 0)
atr_ = tr_.ewm(com=13, adjust=False).mean()
dip_ = 100 * dmp_.ewm(com=13, adjust=False).mean() / atr_.replace(0, 1e-10)
dim_ = 100 * dmm_.ewm(com=13, adjust=False).mean() / atr_.replace(0, 1e-10)
dx_  = 100 * (dip_-dim_).abs() / (dip_+dim_).replace(0, 1e-10)
adx_ = dx_.ewm(com=13, adjust=False).mean()

# DI direction: +DI > -DI = bullish trend direction
di_bull_15m = dip_ > dim_

adx_1h_v  = adx_.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
dibull_1h  = di_bull_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

# Signal: combine ADX strength with DI direction
adx_signal = pd.Series("ranging", index=ts_1h, dtype=object)
adx_signal[(adx_1h_v >= 20) & dibull_1h]  = "trending_up"
adx_signal[(adx_1h_v >= 20) & ~dibull_1h] = "trending_down"
adx_signal[adx_1h_v > 35]                 = "strong_trend"   # overwrite with "strong" regardless of direction
# Separate strong trend direction
adx_strong_bull = (adx_1h_v > 35) & dibull_1h
adx_strong_bear = (adx_1h_v > 35) & ~dibull_1h

adx_dir_signal = pd.Series("ranging", index=ts_1h, dtype=object)
adx_dir_signal[(adx_1h_v >= 20) & (adx_1h_v <= 35) & dibull_1h]  = "moderate_up"
adx_dir_signal[(adx_1h_v >= 20) & (adx_1h_v <= 35) & ~dibull_1h] = "moderate_down"
adx_dir_signal[adx_strong_bull]                                    = "strong_up"
adx_dir_signal[adx_strong_bear]                                    = "strong_down"

# ── 9. Bollinger Band Position (1h, 20/2) ────────────────────────────────────
bb_mid_1h  = close_1h.rolling(20).mean()
bb_std_1h  = close_1h.rolling(20).std()
bb_upper   = bb_mid_1h + 2 * bb_std_1h
bb_lower   = bb_mid_1h - 2 * bb_std_1h
bb_width_1h= (4 * bb_std_1h) / bb_mid_1h

# %B = (close - lower) / (upper - lower)  0=at lower band, 1=at upper band
bb_pct_b   = (close_1h - bb_lower) / (bb_upper - bb_lower).replace(0, float("nan"))
bb_pct_rank= bb_width_1h.rolling(500).rank(pct=True)  # width percentile (squeeze)

bb_signal  = pd.Series("mid", index=ts_1h, dtype=object)
bb_signal[bb_pct_b < 0.1]  = "lower_band"    # near lower band → potential bounce
bb_signal[bb_pct_b > 0.9]  = "upper_band"    # near upper band → potential fade
bb_signal[bb_pct_b < 0.2]  = "lower_zone"    # broader lower zone
bb_signal[bb_pct_b > 0.8]  = "upper_zone"    # broader upper zone
bb_signal[bb_pct_rank < 0.10] = "squeeze"    # top override: extreme squeeze

# ── 10. Rate of Change (multiple periods) ────────────────────────────────────
roc_1h   = close_1h.pct_change(1)    # 1h momentum
roc_4h   = close_1h.pct_change(4)    # 4h momentum
roc_12h  = close_1h.pct_change(12)   # 12h momentum
roc_24h  = close_1h.pct_change(24)   # 24h momentum

def roc_to_signal(roc, thresh_pct):
    sig = pd.Series("neutral", index=ts_1h, dtype=object)
    sig[roc >  thresh_pct] = "bullish"
    sig[roc < -thresh_pct] = "bearish"
    return sig

roc_1h_sig  = roc_to_signal(roc_1h,  0.003)   # ±0.3% threshold
roc_4h_sig  = roc_to_signal(roc_4h,  0.010)   # ±1.0%
roc_12h_sig = roc_to_signal(roc_12h, 0.020)   # ±2.0%
roc_24h_sig = roc_to_signal(roc_24h, 0.030)   # ±3.0%

# ── 11. Volume Momentum (1h, volume vs 20-bar SMA) ───────────────────────────
vol_ma_1h    = volume_1h.rolling(20).mean()
vol_ratio_1h = volume_1h / vol_ma_1h.replace(0, float("nan"))
price_dir_1h = (close_1h > close_1h.shift(1)).astype(int) * 2 - 1  # +1 up, -1 down

# Volume confirmed: high volume + direction = conviction
# Low volume: noise (fade signal)
vol_signal = pd.Series("low_vol", index=ts_1h, dtype=object)
vol_signal[(vol_ratio_1h > 1.5) & (price_dir_1h > 0)]  = "high_vol_up"
vol_signal[(vol_ratio_1h > 1.5) & (price_dir_1h < 0)]  = "high_vol_down"
vol_signal[(vol_ratio_1h >= 0.8) & (vol_ratio_1h <= 1.5)] = "avg_vol"

# ── 12. VWAP Position (daily session, resets at 00:00 UTC) ───────────────────
close_1m_v  = pd.Series(ohlcv_1m["close"].values.astype(float),  index=ohlcv_1m.index)
volume_1m_v = pd.Series(ohlcv_1m["volume"].values.astype(float), index=ohlcv_1m.index)
date_1m     = ohlcv_1m.index.normalize()

tp_1m = close_1m_v  # simplified: use close as typical price
cum_tpv = (tp_1m * volume_1m_v).groupby(date_1m).cumsum()
cum_vol = volume_1m_v.groupby(date_1m).cumsum()
vwap_1m = cum_tpv / cum_vol.replace(0, float("nan"))

# Resample to 1h
vwap_1h     = vwap_1m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
vwap_dev    = (close_1h - vwap_1h) / vwap_1h   # % deviation from VWAP

vwap_signal = pd.Series("near_vwap", index=ts_1h, dtype=object)
vwap_signal[vwap_dev >  0.005]  = "above_vwap"    # >0.5% above
vwap_signal[vwap_dev < -0.005]  = "below_vwap"    # >0.5% below
vwap_signal[vwap_dev >  0.015]  = "far_above_vwap" # >1.5% above (overextended)
vwap_signal[vwap_dev < -0.015]  = "far_below_vwap" # >1.5% below

print("  All indicators computed.\n")


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

# Build master dataframe — one row per test hour
test_mask = ts_1h >= TEST_START
idx_test  = np.where(test_mask)[0][:-1]  # exclude last hour (no next_ret)

master = pd.DataFrame({
    "ts":              ts_1h[idx_test],
    "next_ret":        next_ret.values[idx_test],
    "next_up":         next_up.values[idx_test],
    "rsi":             rsi_1h.values[idx_test],
    "ema_align":       ema_align.values[idx_test],
    "ema_spread":      ema_spread.values[idx_test],
    "ema_stack":       ema_stack_1h.values[idx_test],
    "ema_slope_sig":   ema_slope_signal.values[idx_test],
    "ema_slope_raw":   ema20_slope.values[idx_test],
    "ema_stretch_sig": ema_stretch_signal.values[idx_test],
    "stretch_raw":     stretch_1h.values[idx_test],
    "rsi_signal":      rsi_signal.values[idx_test],
    "rsi_zone":        rsi_zone.values[idx_test],
    "macd_signal":     macd_signal.values[idx_test],
    "macd_hist":       macd_hist.values[idx_test],
    "macd_cross":      macd_cross_signal.values[idx_test],
    "stoch_signal":    stoch_signal.values[idx_test],
    "stk":             stk_1h.values[idx_test],
    "adx":             adx_1h_v.values[idx_test],
    "adx_dir_signal":  adx_dir_signal.values[idx_test],
    "di_bull":         dibull_1h.values[idx_test],
    "bb_signal":       bb_signal.values[idx_test],
    "bb_pct_b":        bb_pct_b.values[idx_test],
    "bb_squeeze_rank": bb_pct_rank.values[idx_test],
    "roc_1h":          roc_1h.values[idx_test],
    "roc_4h":          roc_4h.values[idx_test],
    "roc_12h":         roc_12h.values[idx_test],
    "roc_24h":         roc_24h.values[idx_test],
    "roc_1h_sig":      roc_1h_sig.values[idx_test],
    "roc_4h_sig":      roc_4h_sig.values[idx_test],
    "roc_12h_sig":     roc_12h_sig.values[idx_test],
    "roc_24h_sig":     roc_24h_sig.values[idx_test],
    "vol_signal":      vol_signal.values[idx_test],
    "vol_ratio":       vol_ratio_1h.values[idx_test],
    "vwap_signal":     vwap_signal.values[idx_test],
    "vwap_dev":        vwap_dev.values[idx_test],
}).dropna(subset=["next_ret"])

N_TOTAL = len(master)
UP_BASE = master["next_up"].mean()

print(f"Test hours: {N_TOTAL:,}")
print(f"Baseline up%: {UP_BASE:.1%}  (price went up this fraction of hours)")
print(f"Mean 1h return: {master['next_ret'].mean():.4%}")


def analyze_categorical(df, col, label, ordered_states=None, min_n=30,
                         bullish_states=None, bearish_states=None):
    """
    For each state of a categorical signal column, compute:
      - n, up%, mean_return, vs baseline
    Returns a sorted summary. If bullish_states/bearish_states given,
    computes directional accuracy.
    """
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Baseline: up% = {UP_BASE:.1%}  mean_ret = {df['next_ret'].mean():.4%}")
    print(SEP2)
    print(f"  {'State':>20}  {'n':>6}  {'up%':>7}  {'vs_base':>8}  {'mean_ret%':>10}  {'p-val':>8}  signal")
    print("  " + "-" * 72)

    states = ordered_states if ordered_states else sorted(df[col].unique())
    results = []
    for state in states:
        sub = df[df[col] == state]
        if len(sub) < min_n:
            continue
        n    = len(sub)
        up_p = sub["next_up"].mean()
        mr   = sub["next_ret"].mean() * 100
        diff = up_p - UP_BASE

        # Bootstrap p-value (permutation test simplified: normal approx)
        se   = (UP_BASE * (1 - UP_BASE) / n) ** 0.5
        z    = diff / se if se > 0 else 0
        # Two-tailed p
        from scipy.stats import norm
        pval = 2 * (1 - norm.cdf(abs(z)))

        # Signal classification
        if bullish_states and state in bullish_states:
            pred = "bullish"
            correct_p = up_p
            sig = ("★ STRONG" if diff > 0.05 and pval < 0.05 else
                   "good"     if diff > 0.02 and pval < 0.10 else
                   "weak"     if diff > 0    else "FAIL")
        elif bearish_states and state in bearish_states:
            pred = "bearish"
            correct_p = 1 - up_p
            sig = ("★ STRONG" if diff < -0.05 and pval < 0.05 else
                   "good"     if diff < -0.02 and pval < 0.10 else
                   "weak"     if diff < 0     else "FAIL")
        else:
            pred = "—"
            correct_p = float("nan")
            sig = ""

        pval_s = f"{pval:.3f}" if pval < 0.999 else ">0.99"
        print(f"  {str(state):>20}  {n:>6,}  {up_p:>6.1%}  {diff:>+7.1%}  {mr:>+9.3f}%  {pval_s:>8}  {sig}")
        results.append({"state": state, "n": n, "up_pct": up_p, "vs_base": diff,
                        "mean_ret": mr, "pval": pval, "signal": sig})
    return results


def analyze_continuous(df, col, label, min_n=30):
    """
    For a continuous signal, compute Pearson and Spearman correlation
    with next_ret and next_up. Also show quintile breakdown.
    """
    print(f"\n{SEP}")
    print(f"  {label} — Continuous correlation with next 1h return")
    print(SEP2)

    sub = df[[col, "next_ret", "next_up"]].dropna()
    if len(sub) < 50:
        print(f"  Insufficient data ({len(sub)} rows)")
        return

    r_pearson, p_pearson   = pearsonr(sub[col], sub["next_ret"])
    r_spearman, p_spearman = spearmanr(sub[col], sub["next_ret"])

    print(f"  Pearson r  (linear): {r_pearson:+.4f}  p={p_pearson:.4f}")
    print(f"  Spearman r (rank)  : {r_spearman:+.4f}  p={p_spearman:.4f}")

    # Quintile breakdown
    try:
        sub["quintile"] = pd.qcut(sub[col], q=5, labels=["Q1\n(lowest)","Q2","Q3","Q4","Q5\n(highest)"])
        print(f"\n  {'Quintile':>14}  {'n':>6}  {'up%':>7}  {'vs_base':>8}  {'mean_ret%':>10}")
        print("  " + "-" * 52)
        for q, grp in sub.groupby("quintile", observed=True):
            if len(grp) < min_n: continue
            up_p = grp["next_up"].mean()
            mr   = grp["next_ret"].mean() * 100
            diff = up_p - UP_BASE
            print(f"  {str(q).replace(chr(10),' '):>14}  {len(grp):>6,}  {up_p:>6.1%}  {diff:>+7.1%}  {mr:>+9.3f}%")
    except Exception as e:
        print(f"  (quintile breakdown failed: {e})")


# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 1: EMA ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "ema_align", "1. EMA ALIGNMENT (1h EMA-20 vs EMA-50, 3-bar confirm)",
    ordered_states=["bullish","neutral","bearish"],
    bullish_states=["bullish"], bearish_states=["bearish"])

analyze_continuous(master, "ema_spread", "   EMA Spread (EMA20-EMA50)/price — continuous version")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 2: EMA STACK
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "ema_stack", "2. EMA STACK (15m EMA-9/21/50 + price vs EMA9)",
    ordered_states=[1, 0, -1],
    bullish_states=[1], bearish_states=[-1])

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 3: EMA SLOPE
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "ema_slope_sig", "3. EMA SLOPE (1h EMA-20 slope over 3 bars, thresh ±0.1%)",
    ordered_states=["bullish","neutral","bearish"],
    bullish_states=["bullish"], bearish_states=["bearish"])

analyze_continuous(master, "ema_slope_raw", "   EMA Slope raw value — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 4: EMA STRETCH
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "ema_stretch_sig", "4. EMA STRETCH (5m EMA-20 deviation, ±0.1%, MEAN REVERSION)",
    ordered_states=["overbought","neutral","oversold"],
    bullish_states=["oversold"], bearish_states=["overbought"])

analyze_continuous(master, "stretch_raw", "   EMA Stretch raw % deviation — continuous (negative = bullish)")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 5: RSI
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "rsi_signal", "5a. RSI (1h, 14-period) — oversold/neutral/overbought",
    ordered_states=["oversold","neutral","overbought"],
    bullish_states=["oversold"], bearish_states=["overbought"])

analyze_categorical(master, "rsi_zone", "5b. RSI Fine-grained zones",
    ordered_states=["<20","20-30","30-40","40-50","50-60","60-70","70-80",">80"])

analyze_continuous(master, "rsi", "   RSI raw value — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 6: MACD
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "macd_signal", "6a. MACD (1h, 12/26/9) — histogram + signal line state",
    ordered_states=["bullish","neutral","bearish"],
    bullish_states=["bullish"], bearish_states=["bearish"])

analyze_categorical(master, "macd_cross", "6b. MACD Crossover (fresh cross event)",
    ordered_states=["crossed_up","crossed_up_lag","none","crossed_down_lag","crossed_down"],
    bullish_states=["crossed_up","crossed_up_lag"],
    bearish_states=["crossed_down","crossed_down_lag"])

analyze_continuous(master, "macd_hist", "   MACD Histogram — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 7: STOCHASTIC
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "stoch_signal", "7. STOCHASTIC (15m, 14/3) — zones + crossover",
    ordered_states=["crossup","oversold","neutral","overbought","crossdown"],
    bullish_states=["crossup","oversold"], bearish_states=["crossdown","overbought"])

analyze_continuous(master, "stk", "   Stochastic K raw value — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 8: ADX + DIRECTION
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "adx_dir_signal", "8. ADX + DI DIRECTION (15m, 14-period)",
    ordered_states=["strong_up","moderate_up","ranging","moderate_down","strong_down"],
    bullish_states=["strong_up","moderate_up"],
    bearish_states=["strong_down","moderate_down"])

analyze_continuous(master, "adx", "   ADX raw strength — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 9: BOLLINGER BAND POSITION
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "bb_signal", "9. BOLLINGER BAND POSITION (1h, 20/2)",
    ordered_states=["lower_band","lower_zone","mid","upper_zone","upper_band","squeeze"],
    bullish_states=["lower_band","lower_zone"], bearish_states=["upper_band","upper_zone"])

analyze_continuous(master, "bb_pct_b", "   BB %B — continuous (0=lower, 0.5=mid, 1=upper)")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 10: RATE OF CHANGE
# ════════════════════════════════════════════════════════════════════════════════
for col, label in [
    ("roc_1h_sig",  "10a. ROC 1h  (±0.3% threshold)"),
    ("roc_4h_sig",  "10b. ROC 4h  (±1.0% threshold)"),
    ("roc_12h_sig", "10c. ROC 12h (±2.0% threshold)"),
    ("roc_24h_sig", "10d. ROC 24h (±3.0% threshold)"),
]:
    analyze_categorical(master, col, label,
        ordered_states=["bullish","neutral","bearish"],
        bullish_states=["bullish"], bearish_states=["bearish"])

for col, label in [
    ("roc_1h",  "   ROC 1h  raw — continuous"),
    ("roc_4h",  "   ROC 4h  raw — continuous"),
    ("roc_12h", "   ROC 12h raw — continuous"),
    ("roc_24h", "   ROC 24h raw — continuous"),
]:
    analyze_continuous(master, col, label)

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 11: VOLUME MOMENTUM
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "vol_signal", "11. VOLUME MOMENTUM (1h, volume vs 20-bar SMA)",
    ordered_states=["high_vol_up","avg_vol","low_vol","high_vol_down"],
    bullish_states=["high_vol_up"], bearish_states=["high_vol_down"])

analyze_continuous(master, "vol_ratio", "   Volume ratio (current/MA) — continuous")

# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR 12: VWAP POSITION
# ════════════════════════════════════════════════════════════════════════════════
analyze_categorical(master, "vwap_signal", "12. VWAP POSITION (daily session, resets 00:00 UTC)",
    ordered_states=["far_above_vwap","above_vwap","near_vwap","below_vwap","far_below_vwap"],
    bullish_states=["below_vwap","far_below_vwap"],
    bearish_states=["above_vwap","far_above_vwap"])

analyze_continuous(master, "vwap_dev", "   VWAP deviation % — continuous (negative = below VWAP = bullish?)")


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY SCORECARD — Ranked by directional reliability")
print("  Metric: |up% - baseline| for the strongest bullish/bearish state")
print("  Only states with n≥100 and p<0.10 counted as reliable")
print(SEP)

scorecard = []

def score_indicator(df, col, bull_states, bear_states, label, min_n=100, max_p=0.10):
    from scipy.stats import norm
    best = 0.0
    direction = "—"
    state_label = "—"
    n_best = 0
    for state in bull_states + bear_states:
        sub = df[df[col] == state]
        if len(sub) < min_n: continue
        up_p = sub["next_up"].mean()
        diff = up_p - UP_BASE
        if state in bear_states: diff = -diff  # flip for bearish
        se   = (UP_BASE * (1 - UP_BASE) / len(sub)) ** 0.5
        z    = diff / se if se > 0 else 0
        from scipy.stats import norm
        pval = 2 * (1 - norm.cdf(abs(z)))
        if diff > best and pval < max_p:
            best = diff
            direction = "bullish" if state in bull_states else "bearish"
            state_label = str(state)
            n_best = len(sub)
    scorecard.append((label, best, direction, state_label, n_best))

score_indicator(master, "ema_align",      ["bullish"], ["bearish"],          "EMA Alignment (1h)")
score_indicator(master, "ema_stack",      [1],         [-1],                 "EMA Stack (15m)")
score_indicator(master, "ema_slope_sig",  ["bullish"], ["bearish"],          "EMA Slope (1h EMA20 slope)")
score_indicator(master, "ema_stretch_sig",["oversold"],["overbought"],       "EMA Stretch (5m, mean rev)")
score_indicator(master, "rsi_signal",     ["oversold"],["overbought"],       "RSI (1h, 14p)")
score_indicator(master, "macd_signal",    ["bullish"], ["bearish"],          "MACD (1h, 12/26/9)")
score_indicator(master, "macd_cross",     ["crossed_up","crossed_up_lag"],
                                          ["crossed_down","crossed_down_lag"],"MACD Crossover (fresh)")
score_indicator(master, "stoch_signal",   ["crossup","oversold"],
                                          ["crossdown","overbought"],         "Stochastic (15m, 14/3)")
score_indicator(master, "adx_dir_signal", ["strong_up","moderate_up"],
                                          ["strong_down","moderate_down"],    "ADX + DI Direction (15m)")
score_indicator(master, "bb_signal",      ["lower_band","lower_zone"],
                                          ["upper_band","upper_zone"],        "BB Position (1h)")
score_indicator(master, "roc_1h_sig",     ["bullish"], ["bearish"],          "ROC 1h  (±0.3%)")
score_indicator(master, "roc_4h_sig",     ["bullish"], ["bearish"],          "ROC 4h  (±1.0%)")
score_indicator(master, "roc_12h_sig",    ["bullish"], ["bearish"],          "ROC 12h (±2.0%)")
score_indicator(master, "roc_24h_sig",    ["bullish"], ["bearish"],          "ROC 24h (±3.0%)")
score_indicator(master, "vol_signal",     ["high_vol_up"], ["high_vol_down"],"Volume Momentum (1h)")
score_indicator(master, "vwap_signal",    ["below_vwap","far_below_vwap"],
                                          ["above_vwap","far_above_vwap"],   "VWAP Position (daily)")

scorecard.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Indicator':>35}  {'edge':>7}  {'direction':>10}  {'best_state':>20}  {'n':>6}")
print("  " + "-" * 88)
for label, edge, direction, state, n in scorecard:
    reliability = ("★★★" if edge > 0.05 else
                   "★★ " if edge > 0.03 else
                   "★  " if edge > 0.01 else
                   "   ")
    print(f"  {label:>35}  {edge:>+6.1%}  {direction:>10}  {state:>20}  {n:>6,}  {reliability}")

print(f"""
Notes:
  edge = how much the best signal state lifts directional accuracy above baseline
  Only states with n≥100 and p<0.10 shown (statistically meaningful)
  ★★★ = strong (>5% lift), ★★ = moderate (3-5%), ★ = weak (1-3%)
  Baseline up% = {UP_BASE:.1%}
""")
