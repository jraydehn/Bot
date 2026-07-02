"""
analysis_indicator_directions_3.py — Timeframe sweep for previously untested combos.

Round 3 focus:
  - 15m Donchian (20, 50, 100 bars = ~5h, 12.5h, 25h range)
  - Indicators that FAILED on one timeframe, retested on others:
      MACD        : failed on 1h  → test 15m, 4h
      ADX+DI      : failed on 15m → test 1h, 4h
      Volume      : failed on 1h  → test 15m (higher resolution)
      RSI         : proven on 1h  → test 15m (faster), already have 4h/daily
      Stochastic  : proven on 15m, 1h → test 4h (larger swing context)
      BB position : proven on 1h  → test 4h (macro band context)
      Keltner     : proven on 1h  → test 15m (short-term overextension)
      EMA align   : failed on 1h  → test 4h (is it trend-following at macro scale?)
      Pivot pts   : failed on daily → test 4h pivots (intra-day levels)
      Williams %R : proven on 1h  → test 15m, 4h

Baseline up% ≈ 50.4%  (Jan 2025 – Apr 2026 test set)
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

SEP  = "=" * 76
SEP2 = "-" * 76
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
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

next_ret  = np.log(close_1h / close_1h.shift(1)).shift(-1)
next_up   = (next_ret > 0).astype(int)

# ── Resample bars ──────────────────────────────────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

ohlcv_4h = ohlcv_1h.resample("4h", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])
close_4h  = ohlcv_4h["close"]
high_4h   = ohlcv_4h["high"]
low_4h    = ohlcv_4h["low"]
volume_4h = ohlcv_4h["volume"]

print(f"  1h: {n1h:,}  15m: {len(df_15m):,}  4h: {len(ohlcv_4h):,}")
print("Computing indicators...\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def resample_to_1h(series, method="ffill"):
    return series.resample("1h", origin="start_day").last().reindex(ts_1h, method=method)

def dc_signal(pct):
    sig = pd.Series("mid", index=pct.index, dtype=object)
    sig[pct < 0.20] = "lower_zone"
    sig[pct > 0.80] = "upper_zone"
    sig[pct < 0.10] = "near_low"
    sig[pct > 0.90] = "near_high"
    return sig

def pct_b(close, low_n, high_n):
    rng = (high_n - low_n).replace(0, float("nan"))
    return (close - low_n) / rng

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def stoch_series(h, l, c, k=14, d=3):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    sk = (c - ll) / (hh - ll).replace(0, float("nan")) * 100
    sd = sk.rolling(d).mean()
    return sk, sd

def adx_series(h, l, c, p=14):
    cp  = c.shift(1)
    tr  = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    dmp = (h-h.shift(1)).clip(lower=0).where((h-h.shift(1))>(l.shift(1)-l), 0)
    dmm = (l.shift(1)-l).clip(lower=0).where((l.shift(1)-l)>(h-h.shift(1)), 0)
    atr = tr.ewm(com=p-1, adjust=False).mean()
    dip = 100 * dmp.ewm(com=p-1, adjust=False).mean() / atr.replace(0, 1e-10)
    dim = 100 * dmm.ewm(com=p-1, adjust=False).mean() / atr.replace(0, 1e-10)
    dx  = 100 * (dip-dim).abs() / (dip+dim).replace(0, 1e-10)
    adx = dx.ewm(com=p-1, adjust=False).mean()
    return adx, dip, dim

def adx_sig(adx, dip, dim):
    bull = dip > dim
    sig  = pd.Series("ranging", index=adx.index, dtype=object)
    sig[(adx >= 20) & (adx <= 35) &  bull] = "moderate_up"
    sig[(adx >= 20) & (adx <= 35) & ~bull] = "moderate_down"
    sig[(adx >  35) &  bull]               = "strong_up"
    sig[(adx >  35) & ~bull]               = "strong_down"
    return sig

def keltner_sig(close, high, low, span=20, mult=2):
    ema  = close.ewm(span=span, adjust=False).mean()
    cp   = close.shift(1)
    tr   = pd.concat([high-low, (high-cp).abs(), (low-cp).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(com=span-1, adjust=False).mean()
    up   = ema + mult * atr
    dn   = ema - mult * atr
    w    = (up - dn).replace(0, float("nan"))
    pct  = (close - dn) / w
    sig  = pd.Series("mid", index=close.index, dtype=object)
    sig[pct < 0.15]       = "lower_zone"
    sig[pct > 0.85]       = "upper_zone"
    sig[close < dn]       = "below_KC"
    sig[close > up]       = "above_KC"
    return sig, pct

def wpr_series(h, l, c, p=14):
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, float("nan"))

def wpr_sig(wpr):
    sig = pd.Series("neutral", index=wpr.index, dtype=object)
    sig[wpr < -80] = "oversold"
    sig[wpr > -20] = "overbought"
    return sig


# ════════════════════════════════════════════════════════════════════════════════
# 15m INDICATORS
# ════════════════════════════════════════════════════════════════════════════════

h15 = df_15m["high"]
l15 = df_15m["low"]
c15 = df_15m["close"]
v15 = df_15m["volume"]

# ── 15m Donchian: 20, 50, 100 bars = ~5h, 12.5h, 25h ─────────────────────────
for bars in [20, 50, 100]:
    dc_h = h15.rolling(bars).max()
    dc_l = l15.rolling(bars).min()
    p    = pct_b(c15, dc_l, dc_h)
    s    = dc_signal(p)
    locals()[f"dc15_{bars}_pct"] = resample_to_1h(p)
    locals()[f"dc15_{bars}_sig"] = resample_to_1h(s)

# ── 15m MACD (12/26/9 on 15m closes) ─────────────────────────────────────────
ema12_15m = c15.ewm(span=12, adjust=False).mean()
ema26_15m = c15.ewm(span=26, adjust=False).mean()
macd_15m  = ema12_15m - ema26_15m
msig_15m  = macd_15m.ewm(span=9, adjust=False).mean()
mhist_15m = macd_15m - msig_15m

macd15_sig = pd.Series("neutral", index=df_15m.index, dtype=object)
macd15_sig[(macd_15m > msig_15m) & (mhist_15m > 0)] = "bullish"
macd15_sig[(macd_15m < msig_15m) & (mhist_15m < 0)] = "bearish"

macd15_cross_up   = (macd_15m > msig_15m) & (macd_15m.shift(1) <= msig_15m.shift(1))
macd15_cross_down = (macd_15m < msig_15m) & (macd_15m.shift(1) >= msig_15m.shift(1))

macd15_cross = pd.Series("none", index=df_15m.index, dtype=object)
macd15_cross[macd15_cross_up]   = "crossed_up"
macd15_cross[macd15_cross_down] = "crossed_down"
for sh in [1, 2, 3]:
    macd15_cross[macd15_cross_up.shift(sh).fillna(False)   & (macd15_cross == "none")] = "up_lag"
    macd15_cross[macd15_cross_down.shift(sh).fillna(False) & (macd15_cross == "none")] = "down_lag"

macd15_sig_1h   = resample_to_1h(macd15_sig)
macd15_cross_1h = resample_to_1h(macd15_cross)
macd15_hist_1h  = resample_to_1h(mhist_15m)

# ── 15m ADX + DI ──────────────────────────────────────────────────────────────
adx15, dip15, dim15 = adx_series(h15, l15, c15)
adxsig15 = adx_sig(adx15, dip15, dim15)
adx15_sig_1h = resample_to_1h(adxsig15)
adx15_raw_1h = resample_to_1h(adx15)

# ── 15m Volume momentum ───────────────────────────────────────────────────────
vol_ma_15m   = v15.rolling(20).mean()
vol_ratio_15m = v15 / vol_ma_15m.replace(0, float("nan"))
pdir_15m      = (c15 > c15.shift(1)).astype(int) * 2 - 1

volsig_15m = pd.Series("avg_vol", index=df_15m.index, dtype=object)
volsig_15m[(vol_ratio_15m > 1.5) & (pdir_15m > 0)] = "high_vol_up"
volsig_15m[(vol_ratio_15m > 1.5) & (pdir_15m < 0)] = "high_vol_down"
volsig_15m[vol_ratio_15m < 0.5]                     = "low_vol"

volsig_15m_1h     = resample_to_1h(volsig_15m)
volratio_15m_1h   = resample_to_1h(vol_ratio_15m)

# ── 15m RSI (14) ──────────────────────────────────────────────────────────────
rsi_15m = rsi_series(c15, 14)
rsisig_15m = pd.Series("neutral", index=df_15m.index, dtype=object)
rsisig_15m[rsi_15m < 30] = "oversold"
rsisig_15m[rsi_15m > 70] = "overbought"
rsisig_15m[rsi_15m < 20] = "extreme_oversold"
rsisig_15m[rsi_15m > 80] = "extreme_overbought"
rsi15_sig_1h = resample_to_1h(rsisig_15m)
rsi15_raw_1h = resample_to_1h(rsi_15m)

# ── 15m Williams %R (14) ──────────────────────────────────────────────────────
wpr_15m     = wpr_series(h15, l15, c15, 14)
wprsig_15m  = wpr_sig(wpr_15m)
wpr15_sig_1h = resample_to_1h(wprsig_15m)
wpr15_raw_1h = resample_to_1h(wpr_15m)

# ── 15m Keltner Channel ───────────────────────────────────────────────────────
kcsig_15m, kcpct_15m = keltner_sig(c15, h15, l15)
kc15_sig_1h  = resample_to_1h(kcsig_15m)
kc15_pct_1h  = resample_to_1h(kcpct_15m)

# ── 15m BB position ───────────────────────────────────────────────────────────
bb_mid_15m = c15.rolling(20).mean()
bb_std_15m = c15.rolling(20).std()
bb_up_15m  = bb_mid_15m + 2 * bb_std_15m
bb_dn_15m  = bb_mid_15m - 2 * bb_std_15m
bb_pctb_15m = pct_b(c15, bb_dn_15m, bb_up_15m)

bbsig_15m = pd.Series("mid", index=df_15m.index, dtype=object)
bbsig_15m[bb_pctb_15m < 0.20] = "lower_zone"
bbsig_15m[bb_pctb_15m > 0.80] = "upper_zone"
bbsig_15m[bb_pctb_15m < 0.10] = "near_low"
bbsig_15m[bb_pctb_15m > 0.90] = "near_high"

bb15_sig_1h = resample_to_1h(bbsig_15m)
bb15_pct_1h = resample_to_1h(bb_pctb_15m)


# ════════════════════════════════════════════════════════════════════════════════
# 4h INDICATORS
# ════════════════════════════════════════════════════════════════════════════════

h4 = high_4h
l4 = low_4h
c4 = close_4h
v4 = volume_4h

# ── 4h MACD (12/26/9) ─────────────────────────────────────────────────────────
ema12_4h = c4.ewm(span=12, adjust=False).mean()
ema26_4h = c4.ewm(span=26, adjust=False).mean()
macd_4h  = ema12_4h - ema26_4h
msig_4h  = macd_4h.ewm(span=9, adjust=False).mean()
mhist_4h = macd_4h - msig_4h

macd4h_sig_raw = pd.Series("neutral", index=ohlcv_4h.index, dtype=object)
macd4h_sig_raw[(macd_4h > msig_4h) & (mhist_4h > 0)] = "bullish"
macd4h_sig_raw[(macd_4h < msig_4h) & (mhist_4h < 0)] = "bearish"

macd4h_cross_raw = pd.Series("none", index=ohlcv_4h.index, dtype=object)
xup_4h   = (macd_4h > msig_4h) & (macd_4h.shift(1) <= msig_4h.shift(1))
xdown_4h = (macd_4h < msig_4h) & (macd_4h.shift(1) >= msig_4h.shift(1))
macd4h_cross_raw[xup_4h]   = "crossed_up"
macd4h_cross_raw[xdown_4h] = "crossed_down"
for sh in [1, 2]:
    macd4h_cross_raw[xup_4h.shift(sh).fillna(False)   & (macd4h_cross_raw == "none")] = "up_lag"
    macd4h_cross_raw[xdown_4h.shift(sh).fillna(False) & (macd4h_cross_raw == "none")] = "down_lag"

macd4h_sig_1h   = macd4h_sig_raw.reindex(ts_1h, method="ffill")
macd4h_cross_1h = macd4h_cross_raw.reindex(ts_1h, method="ffill")
macd4h_hist_1h  = mhist_4h.reindex(ts_1h, method="ffill")

# ── 4h ADX + DI ───────────────────────────────────────────────────────────────
adx4h, dip4h, dim4h = adx_series(h4, l4, c4)
adxsig4h_raw = adx_sig(adx4h, dip4h, dim4h)
adx4h_sig_1h = adxsig4h_raw.reindex(ts_1h, method="ffill")
adx4h_raw_1h = adx4h.reindex(ts_1h, method="ffill")

# ── 4h Stochastic (14/3) ──────────────────────────────────────────────────────
stk4h, std4h = stoch_series(h4, l4, c4)
stochsig4h = pd.Series("neutral", index=ohlcv_4h.index, dtype=object)
stochsig4h[stk4h < 20] = "oversold"
stochsig4h[stk4h > 80] = "overbought"
stochsig4h[stk4h < 10] = "extreme_oversold"
stochsig4h[stk4h > 90] = "extreme_overbought"
stoch4h_sig_1h = stochsig4h.reindex(ts_1h, method="ffill")
stk4h_raw_1h   = stk4h.reindex(ts_1h, method="ffill")

# ── 4h BB Position ────────────────────────────────────────────────────────────
bb_mid_4h  = c4.rolling(20).mean()
bb_std_4h  = c4.rolling(20).std()
bb_up_4h   = bb_mid_4h + 2 * bb_std_4h
bb_dn_4h   = bb_mid_4h - 2 * bb_std_4h
bb_pctb_4h = pct_b(c4, bb_dn_4h, bb_up_4h)

bbsig4h_raw = pd.Series("mid", index=ohlcv_4h.index, dtype=object)
bbsig4h_raw[bb_pctb_4h < 0.20] = "lower_zone"
bbsig4h_raw[bb_pctb_4h > 0.80] = "upper_zone"
bbsig4h_raw[bb_pctb_4h < 0.10] = "near_low"
bbsig4h_raw[bb_pctb_4h > 0.90] = "near_high"

bb4h_sig_1h = bbsig4h_raw.reindex(ts_1h, method="ffill")
bb4h_pct_1h = bb_pctb_4h.reindex(ts_1h, method="ffill")

# ── 4h Keltner Channel ────────────────────────────────────────────────────────
kcsig4h_raw, kcpct4h_raw = keltner_sig(c4, h4, l4)
kc4h_sig_1h = kcsig4h_raw.reindex(ts_1h, method="ffill")
kc4h_pct_1h = kcpct4h_raw.reindex(ts_1h, method="ffill")

# ── 4h Williams %R (14) ───────────────────────────────────────────────────────
wpr_4h_raw  = wpr_series(h4, l4, c4, 14)
wprsig4h    = wpr_sig(wpr_4h_raw)
wpr4h_sig_1h = wprsig4h.reindex(ts_1h, method="ffill")
wpr4h_raw_1h = wpr_4h_raw.reindex(ts_1h, method="ffill")

# ── 4h EMA Alignment (EMA20 vs EMA50) ────────────────────────────────────────
ema20_4h = c4.ewm(span=20, adjust=False).mean()
ema50_4h = c4.ewm(span=50, adjust=False).mean()

emaalign4h_raw = pd.Series("neutral", index=ohlcv_4h.index, dtype=object)
for i in range(3, len(c4)):
    e20 = ema20_4h.values[i-3:i]
    e50 = ema50_4h.values[i-3:i]
    cl  = c4.values[i-3:i]
    if all(e20 > e50) and all(cl > e20):
        emaalign4h_raw.iat[i] = "bullish"
    elif all(e20 < e50) or all(cl < e50):
        emaalign4h_raw.iat[i] = "bearish"

emaalign4h_1h = emaalign4h_raw.reindex(ts_1h, method="ffill")

# ── 4h Volume momentum ────────────────────────────────────────────────────────
vol_ma_4h    = v4.rolling(20).mean()
vol_ratio_4h = v4 / vol_ma_4h.replace(0, float("nan"))
pdir_4h      = (c4 > c4.shift(1)).astype(int) * 2 - 1

volsig_4h = pd.Series("avg_vol", index=ohlcv_4h.index, dtype=object)
volsig_4h[(vol_ratio_4h > 1.5) & (pdir_4h > 0)] = "high_vol_up"
volsig_4h[(vol_ratio_4h > 1.5) & (pdir_4h < 0)] = "high_vol_down"
volsig_4h[vol_ratio_4h < 0.5]                    = "low_vol"

volsig_4h_1h   = volsig_4h.reindex(ts_1h, method="ffill")
volratio_4h_1h = vol_ratio_4h.reindex(ts_1h, method="ffill")

# ── 4h Pivot Points (rolling: prev 4h bar OHLC → next bar pivot) ─────────────
piv4h     = (h4.shift(1) + l4.shift(1) + c4.shift(1)) / 3
piv4h_r1  = 2 * piv4h - l4.shift(1)
piv4h_s1  = 2 * piv4h - h4.shift(1)

piv4h_1h  = piv4h.reindex(ts_1h, method="ffill")
pivr1_4h  = piv4h_r1.reindex(ts_1h, method="ffill")
pivs1_4h  = piv4h_s1.reindex(ts_1h, method="ffill")

piv4h_zone = pd.Series("between", index=ts_1h, dtype=object)
piv4h_zone[close_1h > pivr1_4h] = "above_R1"
piv4h_zone[close_1h < pivs1_4h] = "below_S1"
piv4h_zone[(close_1h >= piv4h_1h) & (close_1h <= pivr1_4h)] = "pivot_to_R1"
piv4h_zone[(close_1h >= pivs1_4h) & (close_1h < piv4h_1h)]  = "S1_to_pivot"


# ════════════════════════════════════════════════════════════════════════════════
# BUILD MASTER DATASET
# ════════════════════════════════════════════════════════════════════════════════
test_mask = ts_1h >= TEST_START
idx_test  = np.where(test_mask)[0][:-1]

master = pd.DataFrame({
    "ts":           ts_1h[idx_test],
    "next_ret":     next_ret.values[idx_test],
    "next_up":      next_up.values[idx_test],
    # 15m Donchian
    "dc15_20_sig":  dc15_20_sig.values[idx_test],
    "dc15_50_sig":  dc15_50_sig.values[idx_test],
    "dc15_100_sig": dc15_100_sig.values[idx_test],
    "dc15_20_pct":  dc15_20_pct.values[idx_test],
    "dc15_50_pct":  dc15_50_pct.values[idx_test],
    "dc15_100_pct": dc15_100_pct.values[idx_test],
    # 15m MACD
    "macd15_sig":   macd15_sig_1h.values[idx_test],
    "macd15_cross": macd15_cross_1h.values[idx_test],
    "macd15_hist":  macd15_hist_1h.values[idx_test],
    # 15m ADX
    "adx15_sig":    adx15_sig_1h.values[idx_test],
    "adx15_raw":    adx15_raw_1h.values[idx_test],
    # 15m Volume
    "vol15_sig":    volsig_15m_1h.values[idx_test],
    "vol15_ratio":  volratio_15m_1h.values[idx_test],
    # 15m RSI
    "rsi15_sig":    rsi15_sig_1h.values[idx_test],
    "rsi15_raw":    rsi15_raw_1h.values[idx_test],
    # 15m Williams %R
    "wpr15_sig":    wpr15_sig_1h.values[idx_test],
    "wpr15_raw":    wpr15_raw_1h.values[idx_test],
    # 15m Keltner
    "kc15_sig":     kc15_sig_1h.values[idx_test],
    "kc15_pct":     kc15_pct_1h.values[idx_test],
    # 15m BB
    "bb15_sig":     bb15_sig_1h.values[idx_test],
    "bb15_pct":     bb15_pct_1h.values[idx_test],
    # 4h MACD
    "macd4h_sig":   macd4h_sig_1h.values[idx_test],
    "macd4h_cross": macd4h_cross_1h.values[idx_test],
    "macd4h_hist":  macd4h_hist_1h.values[idx_test],
    # 4h ADX
    "adx4h_sig":    adx4h_sig_1h.values[idx_test],
    "adx4h_raw":    adx4h_raw_1h.values[idx_test],
    # 4h Stochastic
    "stoch4h_sig":  stoch4h_sig_1h.values[idx_test],
    "stk4h_raw":    stk4h_raw_1h.values[idx_test],
    # 4h BB
    "bb4h_sig":     bb4h_sig_1h.values[idx_test],
    "bb4h_pct":     bb4h_pct_1h.values[idx_test],
    # 4h Keltner
    "kc4h_sig":     kc4h_sig_1h.values[idx_test],
    "kc4h_pct":     kc4h_pct_1h.values[idx_test],
    # 4h Williams %R
    "wpr4h_sig":    wpr4h_sig_1h.values[idx_test],
    "wpr4h_raw":    wpr4h_raw_1h.values[idx_test],
    # 4h EMA alignment
    "emaalign4h":   emaalign4h_1h.values[idx_test],
    # 4h Volume
    "vol4h_sig":    volsig_4h_1h.values[idx_test],
    "vol4h_ratio":  volratio_4h_1h.values[idx_test],
    # 4h Pivot
    "piv4h_zone":   piv4h_zone.values[idx_test],
}).dropna(subset=["next_ret"])

N_TOTAL = len(master)
UP_BASE = master["next_up"].mean()
print(f"Test hours: {N_TOTAL:,}  |  Baseline up%: {UP_BASE:.1%}\n")


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════
def analyze(df, col, label, ordered=None, bull=None, bear=None, min_n=30):
    print(f"\n{SEP}\n  {label}\n  Baseline up% = {UP_BASE:.1%}\n{SEP2}")
    print(f"  {'State':>24}  {'n':>6}  {'up%':>7}  {'vs_base':>8}  {'mean_ret%':>10}  {'p-val':>8}  sig")
    print("  " + "-" * 76)
    states = ordered if ordered else sorted(df[col].dropna().unique())
    for state in states:
        sub = df[df[col] == state].dropna(subset=["next_ret"])
        if len(sub) < min_n: continue
        up_p = sub["next_up"].mean()
        mr   = sub["next_ret"].mean() * 100
        diff = up_p - UP_BASE
        se   = (UP_BASE*(1-UP_BASE)/len(sub))**0.5
        pval = 2*(1-norm.cdf(abs(diff/se))) if se>0 else 1.
        if bull and state in bull:
            sig = ("★ STRONG" if diff>0.05 and pval<0.05 else "good" if diff>0.02 and pval<0.10
                   else "weak" if diff>0 else "FAIL")
        elif bear and state in bear:
            sig = ("★ STRONG" if diff<-0.05 and pval<0.05 else "good" if diff<-0.02 and pval<0.10
                   else "weak" if diff<0 else "FAIL")
        else: sig = ""
        pval_s = f"{pval:.3f}" if pval < 0.999 else ">0.99"
        print(f"  {str(state):>24}  {len(sub):>6,}  {up_p:>6.1%}  {diff:>+7.1%}  {mr:>+9.3f}%  {pval_s:>8}  {sig}")

def cont(df, col, label):
    sub = df[[col,"next_ret","next_up"]].dropna()
    if len(sub)<50: return
    rp,pp = pearsonr(sub[col], sub["next_ret"])
    rs,ps = spearmanr(sub[col], sub["next_ret"])
    stars = "★★★" if abs(rs)>0.05 and ps<0.01 else "★★" if abs(rs)>0.03 and ps<0.05 else "★" if ps<0.10 else "—"
    print(f"  {label}: Pearson r={rp:+.4f} p={pp:.4f} | Spearman r={rs:+.4f} p={ps:.4f}  {stars}")


# ── 15m Donchian ──────────────────────────────────────────────────────────────
for col, lbl in [
    ("dc15_20_sig",  "1a. DONCHIAN 15m 20-bar (~5h range)"),
    ("dc15_50_sig",  "1b. DONCHIAN 15m 50-bar (~12.5h range)"),
    ("dc15_100_sig", "1c. DONCHIAN 15m 100-bar (~25h range)"),
]:
    analyze(master, col, lbl,
        ordered=["near_low","lower_zone","mid","upper_zone","near_high"],
        bull=["near_low","lower_zone"], bear=["near_high","upper_zone"])

print()
for col, lbl in [("dc15_20_pct","DC15 20-bar %pos"),("dc15_50_pct","DC15 50-bar %pos"),
                 ("dc15_100_pct","DC15 100-bar %pos")]:
    cont(master, col, lbl)

# ── 15m MACD ──────────────────────────────────────────────────────────────────
analyze(master, "macd15_sig",   "2a. MACD 15m (12/26/9) — histogram state",
    ordered=["bullish","neutral","bearish"], bull=["bullish"], bear=["bearish"])
analyze(master, "macd15_cross", "2b. MACD 15m — crossover events",
    ordered=["crossed_up","up_lag","none","down_lag","crossed_down"],
    bull=["crossed_up","up_lag"], bear=["crossed_down","down_lag"])
cont(master, "macd15_hist", "MACD 15m histogram (continuous)")

# ── 15m ADX ───────────────────────────────────────────────────────────────────
analyze(master, "adx15_sig", "3. ADX+DI 15m (14-period) — retested from round 1",
    ordered=["strong_up","moderate_up","ranging","moderate_down","strong_down"],
    bull=["strong_up","moderate_up"], bear=["strong_down","moderate_down"])
cont(master, "adx15_raw", "ADX 15m raw strength")

# ── 15m Volume ────────────────────────────────────────────────────────────────
analyze(master, "vol15_sig", "4. VOLUME MOMENTUM 15m (vs 20-bar MA)",
    ordered=["high_vol_up","avg_vol","low_vol","high_vol_down"],
    bull=["high_vol_up"], bear=["high_vol_down"])
cont(master, "vol15_ratio", "Volume ratio 15m (continuous)")

# ── 15m RSI ───────────────────────────────────────────────────────────────────
analyze(master, "rsi15_sig", "5. RSI 15m (14-period)",
    ordered=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bull=["oversold","extreme_oversold"], bear=["overbought","extreme_overbought"])
cont(master, "rsi15_raw", "RSI 15m raw value")

# ── 15m Williams %R ───────────────────────────────────────────────────────────
analyze(master, "wpr15_sig", "6. WILLIAMS %R 15m (14-period)",
    ordered=["oversold","neutral","overbought"],
    bull=["oversold"], bear=["overbought"])
cont(master, "wpr15_raw", "Williams %R 15m raw")

# ── 15m Keltner ───────────────────────────────────────────────────────────────
analyze(master, "kc15_sig", "7. KELTNER CHANNEL 15m (20 EMA ± 2×ATR)",
    ordered=["below_KC","lower_zone","mid","upper_zone","above_KC"],
    bull=["below_KC","lower_zone"], bear=["above_KC","upper_zone"])
cont(master, "kc15_pct", "Keltner 15m %position")

# ── 15m BB ────────────────────────────────────────────────────────────────────
analyze(master, "bb15_sig", "8. BB POSITION 15m (20-bar, 2σ)",
    ordered=["near_low","lower_zone","mid","upper_zone","near_high"],
    bull=["near_low","lower_zone"], bear=["near_high","upper_zone"])
cont(master, "bb15_pct", "BB 15m %B (continuous)")

# ── 4h MACD ───────────────────────────────────────────────────────────────────
analyze(master, "macd4h_sig",   "9a. MACD 4h (12/26/9) — histogram state",
    ordered=["bullish","neutral","bearish"], bull=["bullish"], bear=["bearish"])
analyze(master, "macd4h_cross", "9b. MACD 4h — crossover events",
    ordered=["crossed_up","up_lag","none","down_lag","crossed_down"],
    bull=["crossed_up","up_lag"], bear=["crossed_down","down_lag"])
cont(master, "macd4h_hist", "MACD 4h histogram (continuous)")

# ── 4h ADX ────────────────────────────────────────────────────────────────────
analyze(master, "adx4h_sig", "10. ADX+DI 4h (14-period)",
    ordered=["strong_up","moderate_up","ranging","moderate_down","strong_down"],
    bull=["strong_up","moderate_up"], bear=["strong_down","moderate_down"])
cont(master, "adx4h_raw", "ADX 4h raw strength")

# ── 4h Stochastic ─────────────────────────────────────────────────────────────
analyze(master, "stoch4h_sig", "11. STOCHASTIC 4h (14/3)",
    ordered=["extreme_oversold","oversold","neutral","overbought","extreme_overbought"],
    bull=["oversold","extreme_oversold"], bear=["overbought","extreme_overbought"])
cont(master, "stk4h_raw", "Stochastic K 4h raw")

# ── 4h BB ─────────────────────────────────────────────────────────────────────
analyze(master, "bb4h_sig", "12. BB POSITION 4h (20-bar, 2σ)",
    ordered=["near_low","lower_zone","mid","upper_zone","near_high"],
    bull=["near_low","lower_zone"], bear=["near_high","upper_zone"])
cont(master, "bb4h_pct", "BB 4h %B (continuous)")

# ── 4h Keltner ────────────────────────────────────────────────────────────────
analyze(master, "kc4h_sig", "13. KELTNER CHANNEL 4h (20 EMA ± 2×ATR)",
    ordered=["below_KC","lower_zone","mid","upper_zone","above_KC"],
    bull=["below_KC","lower_zone"], bear=["above_KC","upper_zone"])
cont(master, "kc4h_pct", "Keltner 4h %position")

# ── 4h Williams %R ────────────────────────────────────────────────────────────
analyze(master, "wpr4h_sig", "14. WILLIAMS %R 4h (14-period)",
    ordered=["oversold","neutral","overbought"],
    bull=["oversold"], bear=["overbought"])
cont(master, "wpr4h_raw", "Williams %R 4h raw")

# ── 4h EMA Alignment ──────────────────────────────────────────────────────────
analyze(master, "emaalign4h", "15. EMA ALIGNMENT 4h (EMA20 vs EMA50, 3-bar confirm)",
    ordered=["bullish","neutral","bearish"],
    bull=["bullish"], bear=["bearish"])

# ── 4h Volume ─────────────────────────────────────────────────────────────────
analyze(master, "vol4h_sig", "16. VOLUME MOMENTUM 4h (vs 20-bar MA)",
    ordered=["high_vol_up","avg_vol","low_vol","high_vol_down"],
    bull=["high_vol_up"], bear=["high_vol_down"])
cont(master, "vol4h_ratio", "Volume ratio 4h (continuous)")

# ── 4h Pivot ──────────────────────────────────────────────────────────────────
analyze(master, "piv4h_zone", "17. 4H PIVOT POINTS (prev 4h bar → P, R1, S1)",
    ordered=["below_S1","S1_to_pivot","pivot_to_R1","above_R1"],
    bull=["below_S1"], bear=["above_R1"])


# ════════════════════════════════════════════════════════════════════════════════
# SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SCORECARD — Round 3 ranked by directional edge")
print(f"  Baseline up% = {UP_BASE:.1%}  |  n≥100, p<0.10 to qualify")
print(SEP)

scorecard = []

def score(df, col, bull_s, bear_s, label, min_n=100, max_p=0.10):
    best_edge, best_dir, best_st, best_n = 0., "—", "—", 0
    for state in bull_s + bear_s:
        sub = df[df[col]==state].dropna(subset=["next_ret"])
        if len(sub)<min_n: continue
        up_p = sub["next_up"].mean()
        diff = (up_p-UP_BASE) if state in bull_s else (UP_BASE-up_p)
        se   = (UP_BASE*(1-UP_BASE)/len(sub))**0.5
        pval = 2*(1-norm.cdf(abs(diff/se))) if se>0 else 1.
        if diff>best_edge and pval<max_p:
            best_edge,best_dir,best_st,best_n = diff, ("bullish" if state in bull_s else "bearish"), str(state), len(sub)
    scorecard.append((label, best_edge, best_dir, best_st, best_n))

score(master,"dc15_20_sig", ["near_low","lower_zone"],["near_high","upper_zone"],"Donchian 15m 20-bar (~5h)")
score(master,"dc15_50_sig", ["near_low","lower_zone"],["near_high","upper_zone"],"Donchian 15m 50-bar (~12.5h)")
score(master,"dc15_100_sig",["near_low","lower_zone"],["near_high","upper_zone"],"Donchian 15m 100-bar (~25h)")
score(master,"macd15_sig",  ["bullish"],["bearish"],                             "MACD 15m")
score(master,"macd15_cross",["crossed_up","up_lag"],["crossed_down","down_lag"], "MACD 15m crossover")
score(master,"adx15_sig",   ["strong_up","moderate_up"],["strong_down","moderate_down"],"ADX+DI 15m")
score(master,"vol15_sig",   ["high_vol_up"],["high_vol_down"],                   "Volume 15m")
score(master,"rsi15_sig",   ["oversold","extreme_oversold"],["overbought","extreme_overbought"],"RSI 15m")
score(master,"wpr15_sig",   ["oversold"],["overbought"],                         "Williams %R 15m")
score(master,"kc15_sig",    ["below_KC","lower_zone"],["above_KC","upper_zone"], "Keltner 15m")
score(master,"bb15_sig",    ["near_low","lower_zone"],["near_high","upper_zone"],"BB Position 15m")
score(master,"macd4h_sig",  ["bullish"],["bearish"],                             "MACD 4h")
score(master,"macd4h_cross",["crossed_up","up_lag"],["crossed_down","down_lag"], "MACD 4h crossover")
score(master,"adx4h_sig",   ["strong_up","moderate_up"],["strong_down","moderate_down"],"ADX+DI 4h")
score(master,"stoch4h_sig", ["oversold","extreme_oversold"],["overbought","extreme_overbought"],"Stochastic 4h")
score(master,"bb4h_sig",    ["near_low","lower_zone"],["near_high","upper_zone"],"BB Position 4h")
score(master,"kc4h_sig",    ["below_KC","lower_zone"],["above_KC","upper_zone"], "Keltner 4h")
score(master,"wpr4h_sig",   ["oversold"],["overbought"],                         "Williams %R 4h")
score(master,"emaalign4h",  ["bullish"],["bearish"],                             "EMA Alignment 4h")
score(master,"vol4h_sig",   ["high_vol_up"],["high_vol_down"],                   "Volume 4h")
score(master,"piv4h_zone",  ["below_S1"],["above_R1"],                           "Pivot Points 4h")

scorecard.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Indicator':>36}  {'edge':>7}  {'dir':>9}  {'best_state':>22}  {'n':>6}")
print("  " + "-" * 90)
for lbl, edge, direction, state, n in scorecard:
    stars = "★★★" if edge>0.05 else "★★ " if edge>0.03 else "★  " if edge>0.01 else "   "
    print(f"  {lbl:>36}  {edge:>+6.1%}  {direction:>9}  {state:>22}  {n:>6,}  {stars}")

print(f"""
Round 1+2 champions for reference:
  RSI Multi-TF (1h+4h disagree)  +13.1% ★★★
  Stochastic 15m zone             +8.4%  ★★★
  Stochastic 1h zone              +8.3%  ★★★
  VWAP Position                   +8.3%  ★★★
  Keltner 1h                      +7.6%  ★★★
  EMA Stretch 5m                  +6.7%  ★★★
  Williams %R 1h                  +5.7%  ★★★
  Move z-score 1h                 +5.7%  ★★★
  % up bars last 3h               +5.7%  ★★★
  Donchian 1h 20-bar              +5.1%  ★★★
  RSI 1h                          +5.0%  ★★★
""")
