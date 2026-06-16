"""
composite_scorer.py — Two-layer directional scoring for BTC next-hour price movement.

Architecture
============
  Layer 1 — TREND SCORE (4h indicators, continuation signals)
    Indicators: Volume 4h, MACD 4h crossover, Stochastic 4h, BB 4h,
                Keltner 4h, Williams %R 4h
    Range:  -6 to +6  (positive = 4h trend is bullish)
    Logic:  At 4h scale these indicators are TREND-FOLLOWING.
            Overbought/near-high → continuation up.
            Oversold/near-low    → continuation down.

  Layer 2 — REVERSION SCORE (1h/15m indicators, mean-reversion signals)
    Indicators: RSI multi-TF, Stochastic 15m, Stochastic 1h, VWAP,
                Donchian 15m 20-bar, Keltner 15m, Williams %R 1h,
                Move z-score
    Range:  -15 to +15  (positive = oversold → expect bounce up)
    Logic:  At 1h/15m scale these indicators are MEAN-REVERTING.
            Oversold/near-low → expect bounce up.
            Overbought/near-high → expect fade down.

Calibration
===========
  Baseline up% = 50.4% (Jan 2025 – Apr 2026, 11,108 test hours)

  Each (trend_bin, reversion_bin) cell reports:
    - n          : sample size
    - up%        : observed fraction of hours BTC went up
    - edge       : up% - 50.4% baseline
    - p_up       : calibrated probability (smoothed toward baseline for small n)

  When run as a script: outputs the full calibration tables.
  When imported as a module: exports compute_scores() and lookup_p_up().

Production usage
================
  from composite_scorer import compute_scores, lookup_p_up
  trend, rev = compute_scores(close_1h, high_1h, low_1h, volume_1h,
                               close_4h, high_4h, low_4h, volume_4h,
                               close_15m, high_15m, low_15m,
                               close_1m, volume_1m, ts_1h)
  p_up = lookup_p_up(trend, rev)   # calibrated probability, e.g. 0.587
"""

import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

BASELINE_UP = 0.504   # BTC 1h upward drift rate (test set Jan 2025–Apr 2026)
SMOOTHING_N = 30      # minimum sample for full weight; below this blend toward baseline

# Per-asset baselines (measured on test set Jan 2025–Apr 2026)
ASSET_BASELINES = {"BTC": 0.504, "ETH": 0.509, "SOL": 0.500}

# Per-asset drift multiplier for YES model (k_drift_yes) applied in score_to_p_model().
# Fit per-asset against 15mo data to minimize mean calibration bias (restored April-15 values).
#   BTC: k=1.40 (drift under-applied at k=1.0; restored 2026-06-16)
#   ETH: k=0.80
#   SOL: k=0.20
DRIFT_MULTIPLIER = {"BTC": 1.40, "ETH": 0.80, "SOL": 0.20}


K_DRIFT_NO_BTC = 0.30   # BTC NO drift (restored April-15 value; was zeroed 2026-05-20)
K_DRIFT_NO_ETH = 0.20   # ETH NO drift — simulation sweep (2026-05-08) showed K=0.20 closest to breakeven at pm [0.25,0.45)
K_DRIFT_NO     = {"BTC": K_DRIFT_NO_BTC, "ETH": K_DRIFT_NO_ETH}


# ════════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _stoch_k(h, l, c, k=14):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, float("nan")) * 100


def _atr(h, l, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()


def _keltner_pct(h, l, c, span=20, mult=2):
    ema = c.ewm(span=span, adjust=False).mean()
    atr = _atr(h, l, c, span)
    up  = ema + mult * atr
    dn  = ema - mult * atr
    w   = (up - dn).replace(0, float("nan"))
    return (c - dn) / w, dn, up


def _wpr(h, l, c, p=14):
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, float("nan"))


def _macd_cross(c, f=12, s=26, sig=9):
    ema_f  = c.ewm(span=f, adjust=False).mean()
    ema_s  = c.ewm(span=s, adjust=False).mean()
    macd   = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    xup    = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    xdn    = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    state  = pd.Series("none", index=c.index, dtype=object)
    state[xup]  = "crossed_up"
    state[xdn]  = "crossed_down"
    for sh in [1, 2]:
        state[xup.shift(sh).fillna(False)  & (state == "none")] = "up_lag"
        state[xdn.shift(sh).fillna(False)  & (state == "none")] = "down_lag"
    return state


def _vol_signal_4h(close, volume):
    vol_ma  = volume.rolling(20).mean()
    ratio   = volume / vol_ma.replace(0, float("nan"))
    pdir    = (close > close.shift(1)).astype(int) * 2 - 1
    sig     = pd.Series("avg", index=close.index, dtype=object)
    sig[(ratio > 1.5) & (pdir > 0)] = "high_vol_up"
    sig[(ratio > 1.5) & (pdir < 0)] = "high_vol_down"
    sig[ratio < 0.5]                 = "low_vol"
    return sig


def _bb_pct(h, l, c, n=20):
    mid  = c.rolling(n).mean()
    std  = c.rolling(n).std()
    up   = mid + 2 * std
    dn   = mid - 2 * std
    rng  = (up - dn).replace(0, float("nan"))
    return (c - dn) / rng


def _dc_pct(h, l, c, n=20):
    dc_h = h.rolling(n).max()
    dc_l = l.rolling(n).min()
    rng  = (dc_h - dc_l).replace(0, float("nan"))
    return (c - dc_l) / rng


def _vwap_1h(close_1m, volume_1m):
    """Daily-reset VWAP resampled to 1h."""
    date_1m = close_1m.index.normalize()
    tpv     = close_1m * volume_1m
    cum_tpv = tpv.groupby(date_1m).cumsum()
    cum_vol = volume_1m.groupby(date_1m).cumsum()
    vwap_1m = cum_tpv / cum_vol.replace(0, float("nan"))
    return vwap_1m.resample("1h", origin="start_day").last()


# ════════════════════════════════════════════════════════════════════════════════
# TREND SCORE  (4h scale — continuation)
# Positive = 4h trend is bullish; negative = bearish.
# ════════════════════════════════════════════════════════════════════════════════

def _trend_votes(close_4h, high_4h, low_4h, volume_4h):
    """
    Returns a pd.Series of trend_score (int, -6 to +6) on the 4h index.

    Voting rules (each = ±1):
      Stoch 4h          : overbought/extreme_overbought = +1; oversold = -1
      Volume 4h         : high_vol_up = +1; high_vol_down = -1
      MACD 4h crossover : crossed_up/up_lag = +1; crossed_down/down_lag = -1
      BB 4h             : near_high/upper_zone = +1; near_low/lower_zone = -1
      Keltner 4h        : above_KC/upper_zone = +1; below_KC/lower_zone = -1
      Williams %R 4h    : overbought = +1; oversold = -1
    """
    score = pd.Series(0, index=close_4h.index)

    # 1. Stochastic 4h
    stk4 = _stoch_k(high_4h, low_4h, close_4h, 14)
    score += (stk4 > 80).astype(int)
    score -= (stk4 < 20).astype(int)

    # 2. Volume 4h
    vsig = _vol_signal_4h(close_4h, volume_4h)
    score += (vsig == "high_vol_up").astype(int)
    score -= (vsig == "high_vol_down").astype(int)

    # 3. MACD 4h crossover
    macd_st = _macd_cross(close_4h)
    score += macd_st.isin(["crossed_up", "up_lag"]).astype(int)
    score -= macd_st.isin(["crossed_down", "down_lag"]).astype(int)

    # 4. BB 4h
    bb4 = _bb_pct(high_4h, low_4h, close_4h, 20)
    score += (bb4 > 0.80).astype(int)
    score -= (bb4 < 0.20).astype(int)

    # 5. Keltner 4h
    kc4_pct, kc4_dn, kc4_up = _keltner_pct(high_4h, low_4h, close_4h, 20, 2)
    score += ((kc4_pct > 0.85) | (close_4h > kc4_up)).astype(int)
    score -= ((kc4_pct < 0.15) | (close_4h < kc4_dn)).astype(int)

    # 6. Williams %R 4h
    wpr4 = _wpr(high_4h, low_4h, close_4h, 14)
    score += (wpr4 > -20).astype(int)    # overbought (near 14-bar high)
    score -= (wpr4 < -80).astype(int)    # oversold   (near 14-bar low)

    return score.clip(-6, 6)


def _trend_votes_1h(close_1h, high_1h, low_1h, volume_1h):
    """
    Returns a pd.Series of trend_score (int, -6 to +6) on the 1h index.

    Same 6-vote structure as _trend_votes but uses 1h bars instead of 4h.
    Used for the 30m composite system where the trend layer is computed on 1h.
    """
    score = pd.Series(0, index=close_1h.index)

    # 1. Stochastic 1h
    stk1h = _stoch_k(high_1h, low_1h, close_1h, 14)
    score += (stk1h > 80).astype(int)
    score -= (stk1h < 20).astype(int)

    # 2. Volume 1h
    vsig = _vol_signal_4h(close_1h, volume_1h)
    score += (vsig == "high_vol_up").astype(int)
    score -= (vsig == "high_vol_down").astype(int)

    # 3. MACD 1h crossover
    macd_st = _macd_cross(close_1h)
    score += macd_st.isin(["crossed_up", "up_lag"]).astype(int)
    score -= macd_st.isin(["crossed_down", "down_lag"]).astype(int)

    # 4. BB 1h
    bb1h = _bb_pct(high_1h, low_1h, close_1h, 20)
    score += (bb1h > 0.80).astype(int)
    score -= (bb1h < 0.20).astype(int)

    # 5. Keltner 1h
    kc1h_pct, kc1h_dn, kc1h_up = _keltner_pct(high_1h, low_1h, close_1h, 20, 2)
    score += ((kc1h_pct > 0.85) | (close_1h > kc1h_up)).astype(int)
    score -= ((kc1h_pct < 0.15) | (close_1h < kc1h_dn)).astype(int)

    # 6. Williams %R 1h
    wpr1h = _wpr(high_1h, low_1h, close_1h, 14)
    score += (wpr1h > -20).astype(int)    # overbought (near 14-bar high)
    score -= (wpr1h < -80).astype(int)    # oversold   (near 14-bar low)

    return score.clip(-6, 6)


# ════════════════════════════════════════════════════════════════════════════════
# REVERSION SCORE  (1h/15m scale — mean reversion)
# Positive = oversold → expect bounce up; negative = overbought → expect fade.
# ════════════════════════════════════════════════════════════════════════════════

def _reversion_votes(close_1h, high_1h, low_1h,
                     close_15m, high_15m, low_15m,
                     close_1m, volume_1m, ts_1h):
    """
    Returns a pd.Series of reversion_score (int) aligned to ts_1h.

    Voting rules:
      RSI multi-TF (1h vs 4h):  1h_oversold_only = +2; 1h_overbought_only = -2
                                  both_oversold     = +1; both_overbought    = -1
      Stoch 15m zone:            extreme_oversold = +2; oversold = +1
                                  overbought = -1; extreme_overbought = -2
      Stoch 1h zone:             same weights as 15m
      VWAP:                      far_below = +2; below = +1; above = -1; far_above = -2
      Donchian 15m 20-bar:       near_low = +2; lower_zone = +1
                                  upper_zone = -1; near_high = -2
      Keltner 15m:               below_KC = +2; lower_zone = +1
                                  upper_zone = -1; above_KC = -2
      Williams %R 1h:            oversold = +1; overbought = -1
      Move z-score 1h:           large_down (< -2σ) = +2; big_down (-1.5 to -2σ) = +1
                                  big_up (1.5 to 2σ) = -1; large_up (> +2σ)  = -2
    """
    score = pd.Series(0.0, index=ts_1h)

    # ── RSI Multi-TF ──────────────────────────────────────────────────────────
    rsi_1h  = _rsi(close_1h, 14)
    ohlcv_4h = close_1h.resample("4h", origin="start_day").last().dropna()
    rsi_4h_raw = _rsi(ohlcv_4h, 14)
    rsi_4h  = rsi_4h_raw.reindex(ts_1h, method="ffill")

    rsi_1h_os = rsi_1h < 30
    rsi_1h_ob = rsi_1h > 70
    rsi_4h_os = rsi_4h < 30
    rsi_4h_ob = rsi_4h > 70

    # 1h oversold but 4h neutral → strongest mean-reversion setup
    score += 2 * (rsi_1h_os & ~rsi_4h_os & ~rsi_4h_ob).astype(int)
    score -= 2 * (rsi_1h_ob & ~rsi_4h_ob & ~rsi_4h_os).astype(int)
    # Both same zone → weaker signal
    score += 1 * (rsi_1h_os & rsi_4h_os).astype(int)
    score -= 1 * (rsi_1h_ob & rsi_4h_ob).astype(int)

    # ── Stochastic 15m zone ───────────────────────────────────────────────────
    stk15 = _stoch_k(high_15m, low_15m, close_15m, 14)
    stk15_1h = stk15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    score += 2 * (stk15_1h < 10).astype(int)
    score += 1 * ((stk15_1h >= 10) & (stk15_1h < 20)).astype(int)
    score -= 1 * ((stk15_1h > 80) & (stk15_1h <= 90)).astype(int)
    score -= 2 * (stk15_1h > 90).astype(int)

    # ── Stochastic 1h zone ────────────────────────────────────────────────────
    stk1h = _stoch_k(high_1h, low_1h, close_1h, 14)

    score += 2 * (stk1h < 10).astype(int)
    score += 1 * ((stk1h >= 10) & (stk1h < 20)).astype(int)
    score -= 1 * ((stk1h > 80) & (stk1h <= 90)).astype(int)
    score -= 2 * (stk1h > 90).astype(int)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_h = _vwap_1h(close_1m, volume_1m).reindex(ts_1h, method="ffill")
    vwap_dev = (close_1h - vwap_h) / vwap_h.replace(0, float("nan"))

    score += 2 * (vwap_dev < -0.015).astype(int)
    score += 1 * ((vwap_dev >= -0.015) & (vwap_dev < -0.005)).astype(int)
    score -= 1 * ((vwap_dev > 0.005) & (vwap_dev <= 0.015)).astype(int)
    score -= 2 * (vwap_dev > 0.015).astype(int)

    # ── Donchian 15m 20-bar ───────────────────────────────────────────────────
    dc15 = _dc_pct(high_15m, low_15m, close_15m, 20)
    dc15_1h = dc15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    score += 2 * (dc15_1h < 0.10).astype(int)
    score += 1 * ((dc15_1h >= 0.10) & (dc15_1h < 0.20)).astype(int)
    score -= 1 * ((dc15_1h > 0.80) & (dc15_1h <= 0.90)).astype(int)
    score -= 2 * (dc15_1h > 0.90).astype(int)

    # ── Keltner 15m ───────────────────────────────────────────────────────────
    kc15_pct, kc15_dn, kc15_up = _keltner_pct(high_15m, low_15m, close_15m, 20, 2)
    close_15m_1h = close_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    kc15_pct_1h  = kc15_pct.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    kc15_dn_1h   = kc15_dn.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    kc15_up_1h   = kc15_up.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    score += 2 * (close_15m_1h < kc15_dn_1h).astype(int)
    score += 1 * ((kc15_pct_1h >= 0.0) & (kc15_pct_1h < 0.15) & (close_15m_1h >= kc15_dn_1h)).astype(int)
    score -= 1 * ((kc15_pct_1h > 0.85) & (kc15_pct_1h <= 1.0) & (close_15m_1h <= kc15_up_1h)).astype(int)
    score -= 2 * (close_15m_1h > kc15_up_1h).astype(int)

    # ── Williams %R 1h ────────────────────────────────────────────────────────
    wpr1h = _wpr(high_1h, low_1h, close_1h, 14)
    score += 1 * (wpr1h < -80).astype(int)
    score -= 1 * (wpr1h > -20).astype(int)

    # ── Move z-score (last 1h move vs 24h rolling vol) ───────────────────────
    log_ret = np.log(close_1h / close_1h.shift(1))
    roll_vol = log_ret.rolling(24).std()
    move_z = log_ret / roll_vol.replace(0, float("nan"))

    score += 2 * (move_z < -2.0).astype(int)
    score += 1 * ((move_z >= -2.0) & (move_z < -1.5)).astype(int)
    score -= 1 * ((move_z > 1.5) & (move_z <= 2.0)).astype(int)
    score -= 2 * (move_z > 2.0).astype(int)

    return score.clip(-15, 15)


def _reversion_votes_15m(close_15m, high_15m, low_15m,
                          close_1h, high_1h, low_1h,
                          close_1m, volume_1m, ts_30m):
    """
    Returns a pd.Series of reversion_score (int) aligned to ts_30m.

    Timeframes shifted one level shorter than _reversion_votes:
      RSI:         15m vs 1h  (was 1h vs 4h)
      Stoch zones: 5m and 15m (was 15m and 1h)
      VWAP:        daily-reset VWAP ffilled to 30m bars vs 30m close
      Donchian:    5m 20-bar  (was 15m 20-bar)
      Keltner:     5m         (was 15m)
      Williams %R: 15m        (was 1h)
      Move z-score: 15m 48-bar rolling (was 1h 24-bar)
    """
    score = pd.Series(0.0, index=ts_30m)

    # ── Build 5m bars from 1m ─────────────────────────────────────────────────
    close_5m  = close_1m.resample("5min", origin="start_day").last().dropna()
    high_5m   = close_1m.resample("5min", origin="start_day").max().dropna()
    low_5m    = close_1m.resample("5min", origin="start_day").min().dropna()

    # ── RSI Multi-TF  (15m vs 1h) ────────────────────────────────────────────
    rsi_15m     = _rsi(close_15m, 14)
    rsi_1h_raw  = _rsi(close_1h, 14)

    rsi_15m_30m = rsi_15m.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    rsi_1h_30m  = rsi_1h_raw.reindex(ts_30m, method="ffill")

    rsi_15m_os = rsi_15m_30m < 30
    rsi_15m_ob = rsi_15m_30m > 70
    rsi_1h_os  = rsi_1h_30m  < 30
    rsi_1h_ob  = rsi_1h_30m  > 70

    score += 2 * (rsi_15m_os & ~rsi_1h_os & ~rsi_1h_ob).astype(int)
    score -= 2 * (rsi_15m_ob & ~rsi_1h_ob & ~rsi_1h_os).astype(int)
    score += 1 * (rsi_15m_os & rsi_1h_os).astype(int)
    score -= 1 * (rsi_15m_ob & rsi_1h_ob).astype(int)

    # ── Stochastic 5m zone ────────────────────────────────────────────────────
    stk5 = _stoch_k(high_5m, low_5m, close_5m, 14)
    stk5_30m = stk5.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")

    score += 2 * (stk5_30m < 10).astype(int)
    score += 1 * ((stk5_30m >= 10) & (stk5_30m < 20)).astype(int)
    score -= 1 * ((stk5_30m > 80) & (stk5_30m <= 90)).astype(int)
    score -= 2 * (stk5_30m > 90).astype(int)

    # ── Stochastic 15m zone ───────────────────────────────────────────────────
    stk15 = _stoch_k(high_15m, low_15m, close_15m, 14)
    stk15_30m = stk15.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")

    score += 2 * (stk15_30m < 10).astype(int)
    score += 1 * ((stk15_30m >= 10) & (stk15_30m < 20)).astype(int)
    score -= 1 * ((stk15_30m > 80) & (stk15_30m <= 90)).astype(int)
    score -= 2 * (stk15_30m > 90).astype(int)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_h   = _vwap_1h(close_1m, volume_1m).reindex(ts_30m, method="ffill")
    close_30m_ser = close_1m.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    vwap_dev = (close_30m_ser - vwap_h) / vwap_h.replace(0, float("nan"))

    score += 2 * (vwap_dev < -0.015).astype(int)
    score += 1 * ((vwap_dev >= -0.015) & (vwap_dev < -0.005)).astype(int)
    score -= 1 * ((vwap_dev > 0.005) & (vwap_dev <= 0.015)).astype(int)
    score -= 2 * (vwap_dev > 0.015).astype(int)

    # ── Donchian 5m 20-bar ────────────────────────────────────────────────────
    dc5 = _dc_pct(high_5m, low_5m, close_5m, 20)
    dc5_30m = dc5.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")

    score += 2 * (dc5_30m < 0.10).astype(int)
    score += 1 * ((dc5_30m >= 0.10) & (dc5_30m < 0.20)).astype(int)
    score -= 1 * ((dc5_30m > 0.80) & (dc5_30m <= 0.90)).astype(int)
    score -= 2 * (dc5_30m > 0.90).astype(int)

    # ── Keltner 5m ────────────────────────────────────────────────────────────
    kc5_pct, kc5_dn, kc5_up = _keltner_pct(high_5m, low_5m, close_5m, 20, 2)
    close_5m_30m  = close_5m.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    kc5_pct_30m   = kc5_pct.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    kc5_dn_30m    = kc5_dn.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    kc5_up_30m    = kc5_up.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")

    score += 2 * (close_5m_30m < kc5_dn_30m).astype(int)
    score += 1 * ((kc5_pct_30m >= 0.0) & (kc5_pct_30m < 0.15) & (close_5m_30m >= kc5_dn_30m)).astype(int)
    score -= 1 * ((kc5_pct_30m > 0.85) & (kc5_pct_30m <= 1.0) & (close_5m_30m <= kc5_up_30m)).astype(int)
    score -= 2 * (close_5m_30m > kc5_up_30m).astype(int)

    # ── Williams %R 15m ───────────────────────────────────────────────────────
    wpr15 = _wpr(high_15m, low_15m, close_15m, 14)
    wpr15_30m = wpr15.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")
    score += 1 * (wpr15_30m < -80).astype(int)
    score -= 1 * (wpr15_30m > -20).astype(int)

    # ── Move z-score (last 15m move vs 48-bar rolling vol) ───────────────────
    log_ret_15m = np.log(close_15m / close_15m.shift(1))
    roll_vol_15m = log_ret_15m.rolling(48).std()
    move_z_15m = log_ret_15m / roll_vol_15m.replace(0, float("nan"))
    move_z_30m = move_z_15m.resample("30min", origin="start_day").last().reindex(ts_30m, method="ffill")

    score += 2 * (move_z_30m < -2.0).astype(int)
    score += 1 * ((move_z_30m >= -2.0) & (move_z_30m < -1.5)).astype(int)
    score -= 1 * ((move_z_30m > 1.5) & (move_z_30m <= 2.0)).astype(int)
    score -= 2 * (move_z_30m > 2.0).astype(int)

    return score.clip(-15, 15)


def _reversion_votes_4h(close_4h, high_4h, low_4h,
                         close_1h, high_1h, low_1h,
                         close_1m, volume_1m, ts_grid):
    """
    Returns a pd.Series of reversion_score (int) aligned to ts_grid.

    Timeframes shifted one level longer than _reversion_votes (1h/15m → 4h/1h):
      RSI:           4h vs 1D  (was 1h vs 4h)
      Stoch zones:   1h + 4h   (was 15m + 1h)
      VWAP:          daily-reset (same)
      Donchian:      1h 20-bar  (was 15m 20-bar)
      Keltner:       1h         (was 15m)
      Williams %R:   4h         (was 1h)
      Move z-score:  4h 24-bar  (was 1h 24-bar)
    """
    score = pd.Series(0.0, index=ts_grid)

    # ── RSI Multi-TF  (4h vs 1D) ─────────────────────────────────────────────
    rsi_4h      = _rsi(close_4h, 14)
    close_1d    = close_1h.resample("1D", origin="start_day").last().dropna()
    rsi_1d_raw  = _rsi(close_1d, 14)
    rsi_4h_grid = rsi_4h.reindex(ts_grid, method="ffill")
    rsi_1d_grid = rsi_1d_raw.reindex(ts_grid, method="ffill")

    rsi_4h_os = rsi_4h_grid < 30
    rsi_4h_ob = rsi_4h_grid > 70
    rsi_1d_os = rsi_1d_grid < 30
    rsi_1d_ob = rsi_1d_grid > 70

    score += 2 * (rsi_4h_os & ~rsi_1d_os & ~rsi_1d_ob).astype(int)
    score -= 2 * (rsi_4h_ob & ~rsi_1d_ob & ~rsi_1d_os).astype(int)
    score += 1 * (rsi_4h_os & rsi_1d_os).astype(int)
    score -= 1 * (rsi_4h_ob & rsi_1d_ob).astype(int)

    # ── Stochastic 1h zone ────────────────────────────────────────────────────
    stk1h      = _stoch_k(high_1h, low_1h, close_1h, 14)
    stk1h_grid = stk1h.reindex(ts_grid, method="ffill")

    score += 2 * (stk1h_grid < 10).astype(int)
    score += 1 * ((stk1h_grid >= 10) & (stk1h_grid < 20)).astype(int)
    score -= 1 * ((stk1h_grid > 80) & (stk1h_grid <= 90)).astype(int)
    score -= 2 * (stk1h_grid > 90).astype(int)

    # ── Stochastic 4h zone ────────────────────────────────────────────────────
    stk4h      = _stoch_k(high_4h, low_4h, close_4h, 14)
    stk4h_grid = stk4h.reindex(ts_grid, method="ffill")

    score += 2 * (stk4h_grid < 10).astype(int)
    score += 1 * ((stk4h_grid >= 10) & (stk4h_grid < 20)).astype(int)
    score -= 1 * ((stk4h_grid > 80) & (stk4h_grid <= 90)).astype(int)
    score -= 2 * (stk4h_grid > 90).astype(int)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_h        = _vwap_1h(close_1m, volume_1m).reindex(ts_grid, method="ffill")
    close_4h_grid = close_4h.reindex(ts_grid, method="ffill")
    vwap_dev      = (close_4h_grid - vwap_h) / vwap_h.replace(0, float("nan"))

    score += 2 * (vwap_dev < -0.015).astype(int)
    score += 1 * ((vwap_dev >= -0.015) & (vwap_dev < -0.005)).astype(int)
    score -= 1 * ((vwap_dev > 0.005) & (vwap_dev <= 0.015)).astype(int)
    score -= 2 * (vwap_dev > 0.015).astype(int)

    # ── Donchian 1h 20-bar ────────────────────────────────────────────────────
    dc1h      = _dc_pct(high_1h, low_1h, close_1h, 20)
    dc1h_grid = dc1h.reindex(ts_grid, method="ffill")

    score += 2 * (dc1h_grid < 0.10).astype(int)
    score += 1 * ((dc1h_grid >= 0.10) & (dc1h_grid < 0.20)).astype(int)
    score -= 1 * ((dc1h_grid > 0.80) & (dc1h_grid <= 0.90)).astype(int)
    score -= 2 * (dc1h_grid > 0.90).astype(int)

    # ── Keltner 1h ────────────────────────────────────────────────────────────
    kc1h_pct, kc1h_dn, kc1h_up = _keltner_pct(high_1h, low_1h, close_1h, 20, 2)
    close_1h_grid = close_1h.reindex(ts_grid, method="ffill")
    kc1h_pct_grid = kc1h_pct.reindex(ts_grid, method="ffill")
    kc1h_dn_grid  = kc1h_dn.reindex(ts_grid, method="ffill")
    kc1h_up_grid  = kc1h_up.reindex(ts_grid, method="ffill")

    score += 2 * (close_1h_grid < kc1h_dn_grid).astype(int)
    score += 1 * ((kc1h_pct_grid >= 0.0) & (kc1h_pct_grid < 0.15) & (close_1h_grid >= kc1h_dn_grid)).astype(int)
    score -= 1 * ((kc1h_pct_grid > 0.85) & (kc1h_pct_grid <= 1.0) & (close_1h_grid <= kc1h_up_grid)).astype(int)
    score -= 2 * (close_1h_grid > kc1h_up_grid).astype(int)

    # ── Williams %R 4h ────────────────────────────────────────────────────────
    wpr4h      = _wpr(high_4h, low_4h, close_4h, 14)
    wpr4h_grid = wpr4h.reindex(ts_grid, method="ffill")
    score += 1 * (wpr4h_grid < -80).astype(int)
    score -= 1 * (wpr4h_grid > -20).astype(int)

    # ── Move z-score (last 4h move vs 24-bar rolling vol) ────────────────────
    log_ret_4h  = np.log(close_4h / close_4h.shift(1))
    roll_vol_4h = log_ret_4h.rolling(24).std()
    move_z_4h   = log_ret_4h / roll_vol_4h.replace(0, float("nan"))
    move_z_grid = move_z_4h.reindex(ts_grid, method="ffill")

    score += 2 * (move_z_grid < -2.0).astype(int)
    score += 1 * ((move_z_grid >= -2.0) & (move_z_grid < -1.5)).astype(int)
    score -= 1 * ((move_z_grid > 1.5) & (move_z_grid <= 2.0)).astype(int)
    score -= 2 * (move_z_grid > 2.0).astype(int)

    return score.clip(-15, 15)


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (for live trading integration)
# ════════════════════════════════════════════════════════════════════════════════

def compute_scores(close_1h, high_1h, low_1h, volume_1h,
                   close_4h, high_4h, low_4h, volume_4h,
                   close_15m, high_15m, low_15m,
                   close_1m, volume_1m, ts_1h):
    """
    Compute (trend_score, reversion_score) for every hour in ts_1h.

    Returns:
        trend_score     : pd.Series int aligned to ts_1h, range -6 to +6
        reversion_score : pd.Series int aligned to ts_1h, range -15 to +15
    """
    trend_4h = _trend_votes(close_4h, high_4h, low_4h, volume_4h)
    trend_1h = trend_4h.reindex(ts_1h, method="ffill").fillna(0).astype(int)

    reversion = _reversion_votes(close_1h, high_1h, low_1h,
                                  close_15m, high_15m, low_15m,
                                  close_1m, volume_1m, ts_1h).fillna(0).astype(int)

    return trend_1h, reversion


def compute_scores_30m(close_1h, high_1h, low_1h, volume_1h,
                        close_15m, high_15m, low_15m,
                        close_1m, volume_1m, ts_30m):
    """
    Compute (trend_score, reversion_score) for every 30m bar in ts_30m.

    Trend layer: 1h indicators (ffilled to 30m).
    Reversion layer: 15m/5m indicators aligned to 30m.

    Returns:
        trend_30m     : pd.Series int aligned to ts_30m, range -6 to +6
        reversion_30m : pd.Series int aligned to ts_30m, range -15 to +15
    """
    trend_1h_ser = _trend_votes_1h(close_1h, high_1h, low_1h, volume_1h)
    trend_30m = trend_1h_ser.reindex(ts_30m, method="ffill").fillna(0).astype(int)
    reversion_30m = _reversion_votes_15m(
        close_15m, high_15m, low_15m,
        close_1h, high_1h, low_1h,
        close_1m, volume_1m, ts_30m,
    ).fillna(0).astype(int)
    return trend_30m, reversion_30m


def compute_scores_90m(close_1h, high_1h, low_1h, volume_1h,
                        close_1m, volume_1m, ts_grid):
    """
    Compute (trend_score, reversion_score) using 4h-scale indicators aligned to ts_grid.

    Trend layer:     4h indicators (_trend_votes, ffilled to ts_grid).
    Reversion layer: 4h-scale reversion (_reversion_votes_4h, RSI 4h/1D,
                     Stoch 1h+4h, Donchian 1h, Keltner 1h, WPR 4h, Move-z 4h).

    Used for calibrating 90m and 120m forward outcomes. ts_grid is typically
    the 30m bar index (same as 30m calibration for consistent alignment).

    Returns:
        trend_4h  : pd.Series int aligned to ts_grid, range -6 to +6
        rev_4h    : pd.Series int aligned to ts_grid, range -15 to +15
    """
    close_4h  = close_1h.resample("4h", origin="start_day").last().dropna()
    high_4h   = high_1h.resample("4h", origin="start_day").max().dropna()
    low_4h    = low_1h.resample("4h", origin="start_day").min().dropna()
    volume_4h = volume_1h.resample("4h", origin="start_day").sum().dropna()

    trend_4h_ser = _trend_votes(close_4h, high_4h, low_4h, volume_4h)
    trend_grid   = trend_4h_ser.reindex(ts_grid, method="ffill").fillna(0).astype(int)
    rev_grid     = _reversion_votes_4h(
        close_4h, high_4h, low_4h,
        close_1h, high_1h, low_1h,
        close_1m, volume_1m, ts_grid,
    ).fillna(0).astype(int)
    return trend_grid, rev_grid


# ── Calibration persistence ──────────────────────────────────────────────────
# Per-asset calibration JSONs loaded at import time.
# Written by: python3 composite_scorer.py   (BTC)
#             python3 calibrate_eth_sol.py  (ETH, SOL)
# If a file doesn't exist, lookup_p_up() falls back to a linear estimate.

_CAL_PATHS = {
    "BTC": Path(__file__).parent / "composite_calibration.json",
    "ETH": Path(__file__).parent / "composite_calibration_eth.json",
    "SOL": Path(__file__).parent / "composite_calibration_sol.json",
}

# Keep BTC alias for backward compatibility
_CAL_PATH = _CAL_PATHS["BTC"]

CALIBRATION: dict = {}                        # BTC (backward compat)
_CALIBRATIONS: dict = {"BTC": {}, "ETH": {}, "SOL": {}}


def _load_calibration_for(asset: str):
    import json
    path = _CAL_PATHS.get(asset.upper())
    if path is None or not path.exists():
        return
    try:
        with open(path) as f:
            raw = json.load(f)
        cal = {}
        for k, v in raw.items():
            tb, rb = map(int, k.split(","))
            cal[(tb, rb)] = float(v)
        _CALIBRATIONS[asset.upper()] = cal
        if asset.upper() == "BTC":
            CALIBRATION.update(cal)   # keep global alias in sync
    except Exception:
        pass


for _a in ("BTC", "ETH", "SOL"):
    _load_calibration_for(_a)


def save_calibration(cal: dict, asset: str):
    """Save a calibration dict to the per-asset JSON file."""
    import json
    path = _CAL_PATHS.get(asset.upper())
    if path is None:
        raise ValueError(f"Unknown asset: {asset}")
    raw = {f"{k[0]},{k[1]}": v for k, v in cal.items()}
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    _CALIBRATIONS[asset.upper()] = cal
    if asset.upper() == "BTC":
        CALIBRATION.update(cal)
    print(f"  Calibration saved → {path}  ({len(cal)} cells)")


_CAL_PATHS_30M = {
    "BTC": Path(__file__).parent / "composite_calibration_btc_30m.json",
    "ETH": Path(__file__).parent / "composite_calibration_eth_30m.json",
    "SOL": Path(__file__).parent / "composite_calibration_sol_30m.json",
}
_CAL_30M_BY_ASSET: dict = {"BTC": {}, "ETH": {}, "SOL": {}}

# BTC alias preserved for any direct _CAL_30M references
_CAL_PATH_BTC_30M = _CAL_PATHS_30M["BTC"]
_CAL_30M          = _CAL_30M_BY_ASSET["BTC"]


def _load_calibration_30m_for(asset: str):
    import json
    path = _CAL_PATHS_30M.get(asset.upper())
    if path is None or not path.exists():
        return
    try:
        with open(path) as f:
            raw = json.load(f)
        cal = {}
        for k, v in raw.items():
            tb, rb = map(int, k.split(","))
            cal[(tb, rb)] = float(v)
        _CAL_30M_BY_ASSET[asset.upper()].update(cal)
    except Exception:
        pass


for _a30 in ("BTC", "ETH", "SOL"):
    _load_calibration_30m_for(_a30)


def save_calibration_30m(cal: dict, asset: str = "BTC"):
    """Save 30m calibration dict to the per-asset JSON file."""
    import json
    asset = asset.upper()
    path  = _CAL_PATHS_30M.get(asset)
    if path is None:
        raise ValueError(f"No 30m calibration path for asset {asset}")
    raw = {f"{k[0]},{k[1]}": v for k, v in cal.items()}
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    _CAL_30M_BY_ASSET[asset].update(cal)
    print(f"  30m calibration saved → {path}  ({len(cal)} cells)")


_CAL_PATHS_15M = {
    "BTC": Path(__file__).parent / "composite_calibration_btc_15m.json",
    "ETH": Path(__file__).parent / "composite_calibration_eth_15m.json",
    "SOL": Path(__file__).parent / "composite_calibration_sol_15m.json",
}
_CAL_15M_BY_ASSET: dict = {"BTC": {}, "ETH": {}, "SOL": {}}


def _load_calibration_15m_for(asset: str):
    import json
    path = _CAL_PATHS_15M.get(asset.upper())
    if path is None or not path.exists():
        return
    try:
        with open(path) as f:
            raw = json.load(f)
        cal = {}
        for k, v in raw.items():
            tb, rb = map(int, k.split(","))
            cal[(tb, rb)] = float(v)
        _CAL_15M_BY_ASSET[asset.upper()].update(cal)
    except Exception:
        pass


for _a15 in ("BTC", "ETH", "SOL"):
    _load_calibration_15m_for(_a15)


def _lookup_p_up_15m_for(trend_score: int, rev_score: int, asset: str = "BTC") -> float:
    """Look up 15m calibrated p(up) for any asset. Falls back to linear if cell absent."""
    asset    = asset.upper()
    baseline = ASSET_BASELINES.get(asset, BASELINE_UP)
    cal      = _CAL_15M_BY_ASSET.get(asset, {})
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score,  -11, 11))
    if (tb, rb) in cal:
        return cal[(tb, rb)]
    raw = baseline + 0.006 * rb + 0.003 * tb
    return float(np.clip(raw, 0.25, 0.80))


_CAL_PATHS_90M = {
    "BTC": Path(__file__).parent / "composite_calibration_btc_90m.json",
    "ETH": Path(__file__).parent / "composite_calibration_eth_90m.json",
    "SOL": Path(__file__).parent / "composite_calibration_sol_90m.json",
}
_CAL_90M_BY_ASSET: dict = {"BTC": {}, "ETH": {}, "SOL": {}}


def _load_calibration_90m_for(asset: str):
    import json
    path = _CAL_PATHS_90M.get(asset.upper())
    if path is None or not path.exists():
        return
    try:
        with open(path) as f:
            raw = json.load(f)
        cal = {}
        for k, v in raw.items():
            tb, rb = map(int, k.split(","))
            cal[(tb, rb)] = float(v)
        _CAL_90M_BY_ASSET[asset.upper()].update(cal)
    except Exception:
        pass


for _a90 in ("BTC", "ETH", "SOL"):
    _load_calibration_90m_for(_a90)


def _lookup_p_up_90m_for(trend_score: int, rev_score: int, asset: str = "BTC") -> float:
    """Look up 90m calibrated p(up) for any asset. Falls back to linear if cell absent."""
    asset    = asset.upper()
    baseline = ASSET_BASELINES.get(asset, BASELINE_UP)
    cal      = _CAL_90M_BY_ASSET.get(asset, {})
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score,  -11, 11))
    if (tb, rb) in cal:
        return cal[(tb, rb)]
    raw = baseline + 0.006 * rb + 0.003 * tb
    return float(np.clip(raw, 0.25, 0.80))


_CAL_PATHS_120M = {
    "BTC": Path(__file__).parent / "composite_calibration_btc_120m.json",
    "ETH": Path(__file__).parent / "composite_calibration_eth_120m.json",
    "SOL": Path(__file__).parent / "composite_calibration_sol_120m.json",
}
_CAL_120M_BY_ASSET: dict = {"BTC": {}, "ETH": {}, "SOL": {}}


def _load_calibration_120m_for(asset: str):
    import json
    path = _CAL_PATHS_120M.get(asset.upper())
    if path is None or not path.exists():
        return
    try:
        with open(path) as f:
            raw = json.load(f)
        cal = {}
        for k, v in raw.items():
            tb, rb = map(int, k.split(","))
            cal[(tb, rb)] = float(v)
        _CAL_120M_BY_ASSET[asset.upper()].update(cal)
    except Exception:
        pass


for _a120 in ("BTC", "ETH", "SOL"):
    _load_calibration_120m_for(_a120)


def _lookup_p_up_120m_for(trend_score: int, rev_score: int, asset: str = "BTC") -> float:
    """Look up 120m calibrated p(up) for any asset. Falls back to linear if cell absent."""
    asset    = asset.upper()
    baseline = ASSET_BASELINES.get(asset, BASELINE_UP)
    cal      = _CAL_120M_BY_ASSET.get(asset, {})
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score,  -11, 11))
    if (tb, rb) in cal:
        return cal[(tb, rb)]
    raw = baseline + 0.006 * rb + 0.003 * tb
    return float(np.clip(raw, 0.25, 0.80))


# Keep backward-compat alias for BTC-only callers
def _save_calibration():
    save_calibration(CALIBRATION, "BTC")


def lookup_p_up(trend_score: int, reversion_score: int, asset: str = "BTC") -> float:
    """
    Look up calibrated p(up) for a given (trend, reversion) score pair.

    Uses the per-asset empirical calibration table loaded at import time.
    Falls back to a linear estimate if the cell is not in the table.

    Returns:
        float in (0, 1) — probability the asset closes up in the next hour.
    """
    asset = asset.upper()
    baseline = ASSET_BASELINES.get(asset, BASELINE_UP)
    cal = _CALIBRATIONS.get(asset, {})
    # Per-asset rev clip to match actual JSON coverage. Previously a global
    # clip(±8) silently dead-coded BTC's rev∈[±9, ±11] cells (23 trained cells
    # never read) and forced ETH/SOL's |rev|∈[6, 8] rows to read cells that
    # don't exist (then fall back). Now the clip matches each asset's JSON
    # range exactly. (2026-04-30 — see audit thread.)
    _REV_CLIP = {"BTC": 11, "ETH": 5, "SOL": 5}.get(asset, 8)
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(reversion_score, -_REV_CLIP, _REV_CLIP))
    key = (tb, rb)
    if key in cal:
        return cal[key]
    # Fallback: linear estimate from score
    raw = baseline + 0.006 * rb + 0.003 * tb
    return float(np.clip(raw, 0.25, 0.80))


def _lookup_p_up_30m_for(trend_score: int, rev_score: int, asset: str = "BTC") -> float:
    """
    Look up 30m calibrated p(up) for any asset from its 30m calibration table.
    Falls back to a linear estimate if the cell is absent.
    """
    asset    = asset.upper()
    baseline = ASSET_BASELINES.get(asset, BASELINE_UP)
    cal      = _CAL_30M_BY_ASSET.get(asset, {})
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score,  -11, 11))
    if (tb, rb) in cal:
        return cal[(tb, rb)]
    raw = baseline + 0.006 * rb + 0.003 * tb
    return float(np.clip(raw, 0.25, 0.80))


# BTC-only alias kept for any legacy callers
def _lookup_p_up_30m(trend_score: int, rev_score: int) -> float:
    return _lookup_p_up_30m_for(trend_score, rev_score, asset="BTC")


def lookup_p_up_blended(
    trend_1h: int, rev_1h: int,
    trend_30m: int, rev_30m: int,
    tau_minutes: float,
    asset: str = "BTC",
    trend_4h: int = 0, rev_4h: int = 0,
) -> float:
    """
    Blend p_up tables across tau, using progressively coarser calibration
    as time-to-expiry increases.

    Full ladder (when all tables present):
      tau >= 120:       pure 120m  (4h-scale scores)
      tau [90, 120):    blend 120m → 90m
      tau [60, 90):     blend 90m  → 1h
      tau [30, 60):     blend 1h   → 30m
      tau [0, 30):      blend 30m  → 15m

    Falls back to the coarser table for any tier whose JSON is absent.
    trend_4h / rev_4h default to 0 so existing callers without 4h scores
    continue to work (they just get baseline for the upper tiers until
    updated to pass the computed 4h scores).
    """
    p_up_1h = lookup_p_up(trend_1h, rev_1h, asset=asset)

    _has_90m  = bool(_CAL_90M_BY_ASSET.get(asset.upper()))
    _has_120m = bool(_CAL_120M_BY_ASSET.get(asset.upper()))

    if _has_90m or _has_120m:
        p_up_90m  = _lookup_p_up_90m_for(trend_4h, rev_4h, asset=asset)  if _has_90m  else p_up_1h
        p_up_120m = _lookup_p_up_120m_for(trend_4h, rev_4h, asset=asset) if _has_120m else p_up_90m

        # tau >= 120 → pure 120m
        if tau_minutes >= 120.0:
            p_top = p_up_120m
        else:
            # blend 120m → 90m over [90, 120)
            w90 = float(np.clip((120.0 - tau_minutes) / 30.0, 0.0, 1.0))
            blended_120_90 = (1.0 - w90) * p_up_120m + w90 * p_up_90m

            if tau_minutes >= 90.0:
                p_top = blended_120_90
            else:
                # blend 90m → 1h over [60, 90)
                w1h   = float(np.clip((90.0 - tau_minutes) / 30.0, 0.0, 1.0))
                p_top = (1.0 - w1h) * blended_120_90 + w1h * p_up_1h

        if tau_minutes >= 60.0:
            return p_top
        # below 60m: use p_top as the "1h anchor" for the 30m/15m blend
        p_up_1h = p_top

    # ── 30m tier ─────────────────────────────────────────────────────────────
    if not _CAL_30M_BY_ASSET.get(asset.upper()):
        return p_up_1h
    p_up_30m       = _lookup_p_up_30m_for(trend_30m, rev_30m, asset=asset)
    weight_30m     = float(np.clip((60.0 - tau_minutes) / 30.0, 0.0, 1.0))
    blended_1h_30m = (1.0 - weight_30m) * p_up_1h + weight_30m * p_up_30m

    # ── 15m tier ─────────────────────────────────────────────────────────────
    if not _CAL_15M_BY_ASSET.get(asset.upper()):
        return blended_1h_30m
    p_up_15m   = _lookup_p_up_15m_for(trend_30m, rev_30m, asset=asset)
    weight_15m = float(np.clip((30.0 - tau_minutes) / 15.0, 0.0, 1.0))
    return (1.0 - weight_15m) * blended_1h_30m + weight_15m * p_up_15m


# ════════════════════════════════════════════════════════════════════════════════
# CALIBRATION — run when this file is executed directly
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# LIVE-USE HELPERS  (called from paper_trade_runner.py each minute)
# ════════════════════════════════════════════════════════════════════════════════

def compute_current_scores(df_1h, df_4h, df_15m, close_1m, volume_1m):
    """
    Compute (trend_score, reversion_score) for the most recent 1h bar.

    Args:
        df_1h     : DataFrame [open,high,low,close,volume] on 1h bars, ≥200 bars
        df_4h     : same columns on 4h bars, ≥50 bars
        df_15m    : same columns on 15m bars, ≥400 bars
        close_1m  : Series of 1m closes (≥1500 bars for VWAP)
        volume_1m : Series of 1m volumes, same index as close_1m

    Returns:
        (trend_score, reversion_score) as ints
    """
    ts_1h = df_1h.index
    trend_all, rev_all = compute_scores(
        df_1h["close"].astype(float), df_1h["high"].astype(float),
        df_1h["low"].astype(float),   df_1h["volume"].astype(float),
        df_4h["close"].astype(float), df_4h["high"].astype(float),
        df_4h["low"].astype(float),   df_4h["volume"].astype(float),
        df_15m["close"].astype(float), df_15m["high"].astype(float),
        df_15m["low"].astype(float),
        close_1m.astype(float), volume_1m.astype(float),
        ts_1h,
    )
    return int(trend_all.iloc[-1]), int(rev_all.iloc[-1])


def compute_current_scores_30m(df_1h, df_15m, close_1m, volume_1m):
    """
    Compute (trend_score, reversion_score) on the 30m timeframe for the most recent bar.

    Args:
        df_1h     : DataFrame [open,high,low,close,volume] on 1h bars, ≥200 bars
        df_15m    : DataFrame on 15m bars, ≥400 bars
        close_1m  : Series of 1m closes (≥1500 bars for VWAP)
        volume_1m : Series of 1m volumes, same index as close_1m

    Returns:
        (trend_score_30m, reversion_score_30m) as ints
    """
    close_30m = close_1m.resample("30min", origin="start_day").last().dropna()
    ts_30m = close_30m.index
    trend_30m, rev_30m = compute_scores_30m(
        df_1h["close"].astype(float), df_1h["high"].astype(float),
        df_1h["low"].astype(float),   df_1h["volume"].astype(float),
        df_15m["close"].astype(float), df_15m["high"].astype(float),
        df_15m["low"].astype(float),
        close_1m.astype(float), volume_1m.astype(float),
        ts_30m,
    )
    return int(trend_30m.iloc[-1]), int(rev_30m.iloc[-1])


def compute_current_scores_90m(df_1h, close_1m, volume_1m):
    """
    Compute (trend_score, reversion_score) on the 4h timeframe for the most recent bar.

    Uses the 30m grid as the alignment anchor (same as 30m calibration) so the
    returned scores match the 90m/120m calibration table lookup keys.

    Args:
        df_1h     : DataFrame [open,high,low,close,volume] on 1h bars, ≥200 bars
        close_1m  : Series of 1m closes (≥1500 bars for VWAP)
        volume_1m : Series of 1m volumes, same index as close_1m

    Returns:
        (trend_4h_score, reversion_4h_score) as ints
    """
    close_30m = close_1m.resample("30min", origin="start_day").last().dropna()
    ts_grid   = close_30m.index
    trend_grid, rev_grid = compute_scores_90m(
        df_1h["close"].astype(float), df_1h["high"].astype(float),
        df_1h["low"].astype(float),   df_1h["volume"].astype(float),
        close_1m.astype(float), volume_1m.astype(float),
        ts_grid,
    )
    return int(trend_grid.iloc[-1]), int(rev_grid.iloc[-1])


def score_to_p_model(trend_score: int, reversion_score: int,
                     spot: float, strike: float, sigma_tau: float,
                     asset: str = "BTC", p_up_override: float = None,
                     z_drift_override: float = None) -> float:
    """
    Convert composite scores into a calibrated p_model for a specific contract.

    Applies the empirically-calibrated p_up as a drift term in the log-normal model:
        z_strike = log(K / S) / sigma_tau           — pure log-normal distance to strike
        z_drift  = Φ⁻¹(p_up) * DRIFT_MULTIPLIER     — composite signal in z-units,
                                                      per-asset scaled
        z_adj    = z_strike - z_drift
        p_model  = 1 - Φ(z_adj)

    For BTC, z_drift_override replaces the p_up/k_drift path entirely when provided.
    This allows empirical rolling z-drift (computed from resolved trade outcomes) to
    serve as the drift term instead of the lookup-table-based composite p_up.

    Args:
        trend_score     : int, -6 to +6
        reversion_score : int, -15 to +15
        spot            : current BTC price
        strike          : contract strike price
        sigma_tau       : total vol to expiry = vol_per_min * sqrt(tau_minutes)
        p_up_override   : if provided, use this p_up instead of calling lookup_p_up
        z_drift_override: if provided and asset==BTC, use directly as z_drift

    Returns:
        float in [0.01, 0.99]
    """
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    if z_drift_override is not None:
        z_drift = z_drift_override
    else:
        p_up    = p_up_override if p_up_override is not None else lookup_p_up(trend_score, reversion_score, asset=asset)
        asset_u = asset.upper()
        if asset_u == "ETH":
            k_drift = 1.0 * math.exp(-0.3 * max(0, trend_score))
        else:
            k_drift = DRIFT_MULTIPLIER.get(asset_u, 1.0)
        z_drift = norm.ppf(p_up) * k_drift
    z_adj = z_strike - z_drift
    return float(np.clip(1 - norm.cdf(z_adj), 0.01, 0.99))


def score_to_p_no_model(
    trend_score: int,
    reversion_score: int,
    spot: float,
    strike: float,
    sigma_tau: float,
    asset: str = "BTC",
    p_up_override: float = None,
    z_drift_override: float = None,
) -> float:
    """
    Independent NO probability model (BTC and ETH).

    Uses K_DRIFT_NO[asset] instead of the YES drift multiplier.
    Lower drift means less directional bias — the model says NO when the
    raw log-normal says NO, with only mild drift correction.
      BTC: K=0.30  ETH: K=0.20 (simulation-validated 2026-05-08)

    p_no = Φ(z_strike − norm.ppf(p_up) × k_drift_no)

    Since k_drift_yes ≠ k_drift_no, p_yes + p_no ≠ 1 (models are independent).
    Pass (1 − p_no) to evaluate_trade() so the edge formula gives correct NO edge.

    For BTC, z_drift_override bypasses the p_up/k_drift_no path entirely.

    Args:
        p_up_override   : if provided, use this p_up instead of calling lookup_p_up
        z_drift_override: if provided and asset==BTC, use directly as z_drift

    Returns:
        float in [0.01, 0.99]
    """
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    if z_drift_override is not None:
        z_drift = z_drift_override
    else:
        k_no    = K_DRIFT_NO.get(asset, K_DRIFT_NO_BTC)
        p_up    = p_up_override if p_up_override is not None else lookup_p_up(trend_score, reversion_score, asset=asset)
        z_drift = norm.ppf(p_up) * k_no
    z_adj = z_strike - z_drift
    return float(np.clip(norm.cdf(z_adj), 0.01, 0.99))


def composite_to_confirmation(trend_score: int, reversion_score: int):
    """
    Map composite (trend, reversion) scores to the confirmation API used by
    evaluate_trade() in decision.py.

    Returns:
        confirmation_score : int  — YES signal strength; higher = more bullish
        no_score           : int  — NO signal strength; higher = more bearish
        ema_alignment      : str  — always "neutral" (composite p_model handles
                                    direction; bypasses Gate EMA-Dir)

    Mapping rationale (from calibration grid):
        YES signal requires trend ≥ 0 to be trusted; bearish trend cancels reversion.
        NO  signal requires trend ≤ -1; bullish trend cancels bearish reversion.
    """
    # --- YES (confirmation_score) ---
    if trend_score >= 2 and reversion_score >= 4:
        cscore = 5
    elif trend_score >= 1 and reversion_score >= 3:
        cscore = 4
    elif trend_score >= 0 and reversion_score >= 2:
        cscore = 3
    elif trend_score >= 0 and reversion_score >= 1:
        cscore = 2
    elif trend_score < 0 and reversion_score >= 2:
        cscore = -1   # reversion says bullish but trend fights it — weak
    elif trend_score < 0 and reversion_score >= 0:
        cscore = -2   # trend strongly against YES
    else:
        cscore = 0

    # --- NO (no_score) — higher = more bearish confirmation for NO bets ---
    # Gate NS in decision.py requires no_score >= 1 for BTC NO bets.
    # Threshold kept permissive (either leg bearish = 1) to match the original
    # confirm.no_score semantics (1 of 4 indicators bearish was sufficient).
    if trend_score <= -2 and reversion_score <= -4:
        nscore = 4
    elif trend_score <= -1 and reversion_score <= -3:
        nscore = 3
    elif trend_score <= -1 and reversion_score <= -2:
        nscore = 2
    elif trend_score <= -1 or reversion_score <= -1:
        nscore = 1   # either leg bearish → meets Gate NS minimum
    elif trend_score > 0 and reversion_score <= -2:
        nscore = -1  # trend fights NO signal — suppress
    else:
        nscore = 0

    # Always neutral: composite p_model already encodes trend direction.
    # Setting "neutral" bypasses Gate EMA-Dir (which only blocks bullish+YES),
    # letting the edge gate (Gate 3) do the quality filtering.
    return cscore, nscore, "neutral"


if __name__ == "__main__":
    import sys

    BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
    TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

    _mode_arg = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "1h"
    _MODE_30M = "30m" in _mode_arg
    _CAL_ASSET = (
        "ETH" if _mode_arg in ("eth_30m", "eth") else
        "SOL" if _mode_arg in ("sol_30m", "sol") else
        "BTC"
    )
    _TICKER_PREFIX = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}[_CAL_ASSET]

    print(f"Loading data ({_CAL_ASSET})...")
    files_1m = sorted(glob.glob(str(BASE / f"data/binanceus_{_TICKER_PREFIX}_1m_2024-01-01_*.parquet")))
    files_1h = sorted(glob.glob(str(BASE / f"data/binanceus_{_TICKER_PREFIX}_1h_2024-01-01_*.parquet")))

    ohlcv_1m = pd.read_parquet(files_1m[-1])
    ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
    ohlcv_1m = ohlcv_1m.sort_index()

    ohlcv_1h = pd.read_parquet(files_1h[-1])
    ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
    ohlcv_1h = ohlcv_1h.sort_index()

    close_1h  = ohlcv_1h["close"].astype(float)
    high_1h   = ohlcv_1h["high"].astype(float)
    low_1h    = ohlcv_1h["low"].astype(float)
    volume_1h = ohlcv_1h["volume"].astype(float)
    ts_1h     = ohlcv_1h.index

    df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    ohlcv_4h = ohlcv_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    close_1m_s  = ohlcv_1m["close"].astype(float)
    volume_1m_s = ohlcv_1m["volume"].astype(float)

    print(f"  1h: {len(ohlcv_1h):,}  15m: {len(df_15m):,}  4h: {len(ohlcv_4h):,}")

    if _MODE_30M:
        print("Computing 30m composite scores...")
        close_30m = ohlcv_1m["close"].resample("30min", origin="start_day").last().dropna()
        ts_30m = close_30m.index
        trend_ser_30m, rev_ser_30m = compute_scores_30m(
            close_1h, high_1h, low_1h, volume_1h,
            df_15m["close"].astype(float), df_15m["high"].astype(float),
            df_15m["low"].astype(float),
            close_1m_s, volume_1m_s, ts_30m,
        )
        next_ret_30m = np.log(close_30m / close_30m.shift(1)).shift(-1)
        next_up_30m  = (next_ret_30m > 0).astype(int)

        test_mask_30m = ts_30m >= TEST_START
        idx_30m = np.where(test_mask_30m)[0][:-1]  # exclude final (no next return)

        df = pd.DataFrame({
            "trend":   trend_ser_30m.values[idx_30m],
            "rev":     rev_ser_30m.values[idx_30m],
            "next_up": next_up_30m.values[idx_30m],
        })

        n_test = len(df)
        baseline = df["next_up"].mean()
        print(f"\nTest 30m bars: {n_test:,}  |  Baseline up%: {baseline:.1%}")

        # Bucket trend: -3 to +3 (clipped); rev: -11 to +11
        df["tb"] = df["trend"].clip(-3, 3)
        df["rb"] = df["rev"].clip(-11, 11)

        trend_bins = sorted(df["tb"].unique())
        rev_bins   = sorted(df["rb"].unique())

        SEP  = "=" * 80
        SEP2 = "-" * 80
        print(f"\n{SEP}")
        print("  30M CALIBRATION GRID  (trend_bucket × reversion_bucket)")
        print(f"  Format: up% [n]   |  baseline {baseline:.1%}")
        print(SEP2)

        cal_30m: dict = {}
        for tb in trend_bins:
            for rb in rev_bins:
                cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
                n    = len(cell)
                if n >= 10:
                    up   = cell["next_up"].mean()
                    w    = min(1.0, n / SMOOTHING_N)
                    p_cal = w * up + (1 - w) * baseline
                    cal_30m[(int(tb), int(rb))] = round(float(p_cal), 4)

        print(f"  {len(cal_30m)} cells populated  |  SMOOTHING_N={SMOOTHING_N}")
        save_calibration_30m(cal_30m, asset=_CAL_ASSET)
        print(f"\nDone (30m mode, asset={_CAL_ASSET}).")
        sys.exit(0)

    print("Computing composite scores...")

    trend_ser, rev_ser = compute_scores(
        close_1h, high_1h, low_1h, volume_1h,
        ohlcv_4h["close"].astype(float), ohlcv_4h["high"].astype(float),
        ohlcv_4h["low"].astype(float),   ohlcv_4h["volume"].astype(float),
        df_15m["close"].astype(float), df_15m["high"].astype(float),
        df_15m["low"].astype(float),
        close_1m_s, volume_1m_s, ts_1h,
    )

    # Next-hour outcome
    next_ret = np.log(close_1h / close_1h.shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(int)

    # Build test dataframe
    test_mask = ts_1h >= TEST_START
    idx = np.where(test_mask)[0][:-1]  # exclude final (no next return)

    df = pd.DataFrame({
        "trend":   trend_ser.values[idx],
        "rev":     rev_ser.values[idx],
        "next_up": next_up.values[idx],
    })

    n_test = len(df)
    baseline = df["next_up"].mean()
    print(f"\nTest hours: {n_test:,}  |  Baseline up%: {baseline:.1%}")

    SEP  = "=" * 80
    SEP2 = "-" * 80

    # ── 1. Trend Score distribution ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TREND SCORE distribution  (4h continuation)")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for ts_val in sorted(df["trend"].unique()):
        sub = df[df["trend"] == ts_val]
        n   = len(sub)
        up  = sub["next_up"].mean()
        edge = up - baseline
        if n >= 20:
            z = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
        else:
            pv = float("nan")
        pv_str = f"{pv:.3f}" if not math.isnan(pv) else "  —  "
        print(f"  {ts_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}")

    # ── 2. Reversion Score distribution ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  REVERSION SCORE distribution  (1h/15m mean reversion)")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)
    print(f"  {'Score':>8}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for rv_val in sorted(df["rev"].unique()):
        sub = df[df["rev"] == rv_val]
        n   = len(sub)
        up  = sub["next_up"].mean()
        edge = up - baseline
        if n >= 20:
            z = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv = 2 * (1 - norm.cdf(abs(z)))
        else:
            pv = float("nan")
        pv_str = f"{pv:.3f}" if not math.isnan(pv) else "  —  "
        sig = "★★★" if (not math.isnan(pv) and pv < 0.01 and abs(edge) > 0.05) else \
              "★★ " if (not math.isnan(pv) and pv < 0.05 and abs(edge) > 0.03) else \
              "★  " if (not math.isnan(pv) and pv < 0.10 and abs(edge) > 0.01) else ""
        print(f"  {rv_val:>+8d}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv_str:>7}  {sig}")

    # ── 3. Bucketed calibration grid ─────────────────────────────────────────────
    # Bucket trend: -3 to +3 (clipped)
    # Bucket rev:   ≤-8 (pooled into -8), -7 to +7 individual, ≥+8 (pooled into +8)
    # Pooling extremes at ±8 gives sufficient n at tails for stable monotonic estimates.
    df["tb"] = df["trend"].clip(-3, 3)
    df["rb"] = df["rev"].clip(-8, 8)

    print(f"\n{SEP}")
    print("  CALIBRATION GRID  (trend_bucket × reversion_bucket)")
    print(f"  Format: up% [n]   |  baseline {baseline:.1%}")
    print(SEP2)

    trend_bins = sorted(df["tb"].unique())
    rev_bins   = sorted(df["rb"].unique())

    # Header
    hdr = f"  {'Rev →':>8}  " + "".join(f"  {rb:>+4d}  " for rb in rev_bins)
    print(hdr)
    print(f"  {'Trend ↓':>8}  " + "-" * (len(rev_bins) * 8))

    rows = []
    for tb in trend_bins:
        row_str = f"  {tb:>+8d}  "
        for rb in rev_bins:
            cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
            n    = len(cell)
            if n >= 10:
                up = cell["next_up"].mean()
                row_str += f"  {up:.0%}[{n:3d}]"
                # Populate calibration table (smoothed)
                w = min(1.0, n / SMOOTHING_N)
                p_cal = w * up + (1 - w) * baseline
                CALIBRATION[(int(tb), int(rb))] = round(float(p_cal), 4)
            else:
                row_str += f"  — [{n:3d}]"
        rows.append((tb, row_str))

    for _, row_str in rows:
        print(row_str)

    # ── 4. Coarser summary: strong bullish vs strong bearish composites ───────────
    print(f"\n{SEP}")
    print("  COMPOSITE SIGNAL BUCKETS  (coarse directional summary)")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)

    conditions = [
        ("Strong YES (rev≥+4, trend≥0)",   (df["rev"] >= 4) & (df["trend"] >= 0)),
        ("Moderate YES (rev +2/+3)",        (df["rev"].between(2, 3))),
        ("Trend+Rev agree bullish (t≥1,r≥2)", (df["trend"] >= 1) & (df["rev"] >= 2)),
        ("Rev bullish, trend bearish (t<0,r≥2)", (df["trend"] < 0) & (df["rev"] >= 2)),
        ("Neutral (rev -1 to +1)",          (df["rev"].between(-1, 1))),
        ("Trend+Rev agree bearish (t≤-1,r≤-2)", (df["trend"] <= -1) & (df["rev"] <= -2)),
        ("Rev bearish, trend bullish (t>0,r≤-2)", (df["trend"] > 0) & (df["rev"] <= -2)),
        ("Moderate NO (rev -2/-3)",         (df["rev"].between(-3, -2))),
        ("Strong NO (rev≤-4, trend≤0)",     (df["rev"] <= -4) & (df["trend"] <= 0)),
    ]

    print(f"  {'Condition':<45}   {'n':>6}   {'up%':>6}   {'edge':>7}   {'p-val':>7}")
    print(SEP2)
    for label, mask in conditions:
        sub = df[mask]
        n   = len(sub)
        if n < 20:
            print(f"  {label:<45}   {n:>6,}   (too few)")
            continue
        up   = sub["next_up"].mean()
        edge = up - baseline
        z    = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
        pv   = 2 * (1 - norm.cdf(abs(z)))
        sig  = "★★★" if pv < 0.01 and abs(edge) > 0.05 else \
               "★★ " if pv < 0.05 and abs(edge) > 0.03 else \
               "★  " if pv < 0.10 and abs(edge) > 0.01 else ""
        print(f"  {label:<45}   {n:>6,}   {up:>6.1%}   {edge:>+7.1%}   {pv:.3f}  {sig}")

    # ── 5. Interaction: does trend direction amplify reversion signal? ────────────
    print(f"\n{SEP}")
    print("  TREND × REVERSION INTERACTION")
    print(f"  Does 4h trend direction amplify the 1h mean-reversion signal?")
    print(f"  Baseline up% = {baseline:.1%}")
    print(SEP2)

    for rev_cond, rev_label in [
        (df["rev"] >= 4,   "Rev ≥ +4 (strong bullish)"),
        (df["rev"].between(2, 3), "Rev +2/+3 (moderate bullish)"),
        (df["rev"].between(-3, -2), "Rev -2/-3 (moderate bearish)"),
        (df["rev"] <= -4,  "Rev ≤ -4 (strong bearish)"),
    ]:
        print(f"\n  {rev_label}:")
        for trend_cond, trend_label in [
            (df["trend"] >= 2,    "  trend ≥+2 (4h strongly bullish)"),
            (df["trend"].between(0, 1), "  trend 0/+1 (4h neutral/slight bull)"),
            (df["trend"].between(-1, -1), "  trend = -1 (4h slight bear)"),
            (df["trend"] <= -2,   "  trend ≤-2 (4h strongly bearish)"),
        ]:
            sub = df[rev_cond & trend_cond]
            n   = len(sub)
            if n < 15:
                print(f"    {trend_label:<40}  n={n}  (too few)")
                continue
            up   = sub["next_up"].mean()
            edge = up - baseline
            z    = (up - baseline) / math.sqrt(baseline * (1 - baseline) / n)
            pv   = 2 * (1 - norm.cdf(abs(z)))
            sig  = "★★★" if pv < 0.01 and abs(edge) > 0.05 else \
                   "★★ " if pv < 0.05 and abs(edge) > 0.03 else \
                   "★  " if pv < 0.10 and abs(edge) > 0.01 else ""
            print(f"    {trend_label:<40}  n={n:4d}  up={up:.1%}  edge={edge:+.1%}  p={pv:.3f}  {sig}")

    print(f"\n{SEP}")
    print("  CALIBRATION LOOKUP TABLE  (populated in CALIBRATION dict)")
    print(f"  {len(CALIBRATION)} cells populated  |  fallback: linear estimate")
    print(SEP)

    save_calibration(CALIBRATION, _CAL_ASSET)
    print(f"\nDone (1h mode, asset={_CAL_ASSET}).")
