#!/usr/bin/env python3
"""
analyze_1h_signal_correlation.py

Comprehensive signal → 1h forward direction correlation sweep for BTC, ETH, SOL.

Signal groups:
  Price action   : bp, chg, ROC (3/6/12/24h), z-score, VWAP distance
  Candle struct  : body, wicks, engulfing, harami, morning/evening star,
                   3 soldiers/crows, tweezer, pin bar, inside/outside bar, doji, hammer
  Momentum       : stoch K/D/cross/OB/OS, RSI, MACD, CCI, Williams %R, consecutive dir
  Trend          : EMA bias/stack/gap, Ichimoku (tenkan/kijun cross, cloud position)
  Volume/flow    : OBV, OBV-EMA divergence, CMF, MFI, Force Index, cumulative delta,
                   vol ratio, volume divergence from price
  Volatility     : ATR ratio, BB width/pct/squeeze, range ratio
  Donchian       : position, width, breakout (20-bar)
  SMC / structure: swing high/low, BOS (break of structure), Fair Value Gap,
                   market structure (HH/HL vs LH/LL), ChoCH proxy
  Pivot points   : daily pivot, R1/R2/S1/S2 distance, price vs pivot
  ADX            : ADX level, DI diff, ADX trend (+1/-1/0)
  Agreement      : multi-TF EMA/BP/dir/stoch agree (1h+4h, 1h+4h+1d)
  External       : Coinalyze liq_bias, liq_score, ls_long/short, oi_chg_pct,
                   funding_rate, cumulative_funding (3h, 8h, 24h)

Forward horizons : fwd_1h, fwd_4h, fwd_8h, fwd_24h
Signal TFs       : 1h (same as bet), 4h (one level up), 1d (two levels up)

Run:
  python3 analyze_1h_signal_correlation.py
  python3 analyze_1h_signal_correlation.py --asset BTC
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
SEP  = "=" * 82
SEP2 = "-" * 82

ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
COINALYZE_SYMBOLS = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}
COINALYZE_KEY  = "d5841821-3f45-4e5f-9ee7-d2779d2fb01b"
COINALYZE_BASE = "https://api.coinalyze.net/v1"

FWD_HORIZONS = {"fwd_1h": 1, "fwd_4h": 4, "fwd_8h": 8, "fwd_24h": 24}
MIN_N  = 200
TOP_N  = 35


# ── parquet helpers ────────────────────────────────────────────────────────────
def latest_parquet(sym):
    files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 1m parquet for {sym}")
    return files[-1]

def load_1m(sym):
    df = pd.read_parquet(latest_parquet(sym))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open","high","low","close","volume"]].sort_index()


# ── primitives ────────────────────────────────────────────────────────────────

def _bp(c, l, h):
    r = h - l
    return np.where(r > 0, (c - l) / r, 0.5)

def _body(o, c, h, l):
    r = h - l
    return np.where(r > 0, (c - o).abs() / r, 0.0)

def _upper_wick(o, c, h, l):
    r = h - l
    return np.where(r > 0, (h - np.maximum(o, c)) / r, 0.0)

def _lower_wick(o, c, h, l):
    r = h - l
    return np.where(r > 0, (np.minimum(o, c) - l) / r, 0.0)

def _engulfing(o, c):
    body = c - o
    prev = body.shift()
    bull = (body > 0) & (body.abs() > prev.abs()) & (prev < 0)
    bear = (body < 0) & (body.abs() > prev.abs()) & (prev > 0)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _harami(o, c):
    body = c - o
    prev = body.shift()
    prev_o = o.shift(); prev_c = c.shift()
    prev_h = np.maximum(prev_o, prev_c); prev_l = np.minimum(prev_o, prev_c)
    cur_h  = np.maximum(o, c);          cur_l  = np.minimum(o, c)
    inside = (cur_h <= prev_h) & (cur_l >= prev_l)
    bull   = inside & (prev < 0) & (body > 0)
    bear   = inside & (prev > 0) & (body < 0)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _three_soldiers_crows(o, c, period=3):
    body = c - o
    bull = (body > 0) & (body.shift() > 0) & (body.shift(2) > 0)
    bear = (body < 0) & (body.shift() < 0) & (body.shift(2) < 0)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _tweezer(h, l, close):
    tol = close * 0.001
    top    = np.abs(h - h.shift()) < tol.abs()
    bottom = np.abs(l - l.shift()) < tol.abs()
    bear = top  & (close < close.shift())
    bull = bottom & (close > close.shift())
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=h.index).astype(float)

def _inside_bar(h, l):
    inside = (h < h.shift()) & (l > l.shift())
    return inside.astype(float)

def _outside_bar(o, c, h, l):
    outside = (h > h.shift()) & (l < l.shift())
    body = c - o
    bull = outside & (body > 0)
    bear = outside & (body < 0)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _pin_bar(o, c, h, l, body_thresh=0.25, wick_thresh=0.6):
    r = h - l
    body_frac = np.where(r > 0, (c - o).abs() / r, 0.0)
    lw_frac   = np.where(r > 0, (np.minimum(o,c) - l) / r, 0.0)
    uw_frac   = np.where(r > 0, (h - np.maximum(o,c)) / r, 0.0)
    bull = (body_frac <= body_thresh) & (lw_frac >= wick_thresh)
    bear = (body_frac <= body_thresh) & (uw_frac >= wick_thresh)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _morning_evening_star(o, c):
    body   = c - o
    body2  = body.shift()
    body3  = body.shift(2)
    small2 = body2.abs() < body3.abs() * 0.4
    morning = (body3 < 0) & small2 & (body > 0) & (c > o.shift(2))
    evening = (body3 > 0) & small2 & (body < 0) & (c < o.shift(2))
    return pd.Series(np.where(morning, 1, np.where(evening, -1, 0)), index=o.index).astype(float)

def _hammer(o, c, h, l, body_thresh=0.35, wick_thresh=2.0):
    r    = h - l
    body = (c - o).abs()
    lw   = np.minimum(o,c) - l
    uw   = h - np.maximum(o,c)
    bf   = np.where(r > 0, body / r, 0.0)
    bull = (bf <= body_thresh) & (np.where(body > 0, lw / body, 0) >= wick_thresh) & (uw < body)
    bear = (bf <= body_thresh) & (np.where(body > 0, uw / body, 0) >= wick_thresh) & (lw < body)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=o.index).astype(float)

def _doji(o, c, h, l, thresh=0.05):
    r = h - l
    return (np.where(r > 0, (c - o).abs() / r, 0.0) <= thresh).astype(float)

def _stoch_k(h, l, c, k=14):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    return pd.Series(np.where((hi-lo) > 0, 100*(c-lo)/(hi-lo), 50.0), index=c.index)

def _rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0); ls = (-d).clip(lower=0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = ls.ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def _macd(c, f=12, s=26, sg=9):
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sig = m.ewm(span=sg, adjust=False).mean()
    return m, sig, m - sig

def _cci(c, h, l, p=20):
    tp  = (h + l + c) / 3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def _williams_r(h, l, c, p=14):
    hh = h.rolling(p).max(); ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)

def _atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, min_periods=p).mean()

def _adx(h, l, c, p=14):
    tr = _atr(h, l, c, p)
    up = h.diff(); dn = -l.diff()
    dmp = np.where((up > dn) & (up > 0), up, 0.0)
    dmn = np.where((dn > up) & (dn > 0), dn, 0.0)
    smp = pd.Series(dmp, index=c.index).ewm(com=p-1, min_periods=p).mean()
    smn = pd.Series(dmn, index=c.index).ewm(com=p-1, min_periods=p).mean()
    tr_ = tr.replace(0, np.nan)
    dip = 100 * smp / tr_; din = 100 * smn / tr_
    dx  = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    adx = dx.ewm(com=p-1, min_periods=p).mean()
    return adx, dip, din

def _bb(c, n=20, ns=2):
    sma = c.rolling(n).mean(); std = c.rolling(n).std()
    ub = sma + ns*std; lb = sma - ns*std
    return (ub - lb) / sma.replace(0, np.nan), (c - lb) / (ub - lb).replace(0, np.nan)

def _donchian(h, l, c, n=20):
    dh = h.rolling(n).max(); dl = l.rolling(n).min()
    dr = dh - dl
    pos = (c - dl) / dr.replace(0, np.nan)
    width = dr / c.replace(0, np.nan)
    brk = pd.Series(np.where(c >= dh.shift(), 1, np.where(c <= dl.shift(), -1, 0)), index=c.index)
    return pos, width, brk.astype(float)

def _obv(c, v):
    return (np.sign(c.diff()) * v).fillna(0).cumsum()

def _cmf(c, h, l, v, p=20):
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = (mfm * v).fillna(0)
    return mfv.rolling(p).sum() / v.rolling(p).sum().replace(0, np.nan)

def _mfi(c, h, l, v, p=14):
    tp  = (h + l + c) / 3
    mf  = tp * v
    pos = mf.where(tp > tp.shift(), 0)
    neg = mf.where(tp < tp.shift(), 0)
    mfr = pos.rolling(p).sum() / neg.rolling(p).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)

def _force_index(c, v, p=13):
    return (c.diff() * v).ewm(span=p, adjust=False).mean()

def _cum_delta(o, c, v, p=10):
    return (np.sign(c - o) * v).rolling(p).sum()

def _z_score(c, n=20):
    return (c - c.rolling(n).mean()) / c.rolling(n).std().replace(0, np.nan)

def _vwap_rolling(c, v, n=24):
    return (c - (c*v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)) / c.replace(0, np.nan) * 100

def _roc(c, p):
    return c.pct_change(p) * 100

def _consecutive_dir(c):
    dirs = np.sign(c.diff())
    streak = pd.Series(0.0, index=c.index)
    for i in range(1, len(dirs)):
        d = dirs.iloc[i]
        s = streak.iloc[i-1]
        if d == 0:
            streak.iloc[i] = 0
        elif s == 0 or np.sign(d) == np.sign(s):
            streak.iloc[i] = s + d
        else:
            streak.iloc[i] = d
    return streak.clip(-6, 6)

def _ichimoku(h, l, c):
    tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    tk_cross = pd.Series(np.where(
        (tenkan > kijun) & (tenkan.shift() <= kijun.shift()),  1,
        np.where((tenkan < kijun) & (tenkan.shift() >= kijun.shift()), -1, 0)
    ), index=c.index).astype(float)
    # Price vs cloud (no look-ahead: use span shifted forward, only available 26 bars back)
    above_cloud = ((c > span_a) & (c > span_b)).astype(float)
    below_cloud = ((c < span_a) & (c < span_b)).astype(float)
    cloud_pos   = above_cloud - below_cloud
    bull_cloud  = (span_a > span_b).astype(float)  # bullish cloud color
    tk_above_kijun = np.sign(tenkan - kijun).astype(float)
    price_vs_tenkan = np.sign(c - tenkan).astype(float)
    price_vs_kijun  = np.sign(c - kijun).astype(float)
    return tk_cross, cloud_pos, bull_cloud, tk_above_kijun, price_vs_tenkan, price_vs_kijun

# ── SMC helpers ────────────────────────────────────────────────────────────────

def _swing_levels(h, l, lookback=10):
    """Recent swing high and swing low using rolling window (no look-ahead)."""
    swing_high = h.rolling(lookback).max()
    swing_low  = l.rolling(lookback).min()
    return swing_high, swing_low

def _bos(c, h, l, lookback=20):
    """Break of structure: close breaks above recent swing high (+1) or below low (-1)."""
    sh = h.rolling(lookback).max().shift(1)
    sl = l.rolling(lookback).min().shift(1)
    bull = (c > sh).astype(float)
    bear = (c < sl).astype(float)
    return bull - bear

def _choch_proxy(c, h, l, lookback=20):
    """Change of character: BOS in opposite direction to recent trend."""
    bos = _bos(c, h, l, lookback)
    trend = np.sign(c - c.shift(lookback))
    choch = ((bos == 1) & (trend < 0)) | ((bos == -1) & (trend > 0))
    return choch.astype(float)

def _fvg(h, l):
    """Fair Value Gap: bullish if current low > 2-bars-ago high; bearish if current high < 2-bars-ago low."""
    bull = (l > h.shift(2)).astype(float)
    bear = (h < l.shift(2)).astype(float)
    return bull - bear

def _market_structure(c, h, l, lookback=10):
    """Higher High + Higher Low = bullish (+1); Lower Low + Lower High = bearish (-1)."""
    roll_h = h.rolling(lookback).max()
    roll_l = l.rolling(lookback).min()
    hh = (roll_h > roll_h.shift(lookback)).astype(float)
    hl = (roll_l > roll_l.shift(lookback)).astype(float)
    lh = (roll_h < roll_h.shift(lookback)).astype(float)
    ll = (roll_l < roll_l.shift(lookback)).astype(float)
    bull = hh.astype(bool) & hl.astype(bool)
    bear = lh.astype(bool) & ll.astype(bool)
    return pd.Series(np.where(bull, 1.0, np.where(bear, -1.0, 0.0)), index=c.index)

def _premium_discount(c, h, l, lookback=20):
    """Premium (above midpoint of recent range) / Discount (below)."""
    range_high = h.rolling(lookback).max()
    range_low  = l.rolling(lookback).min()
    midpoint   = (range_high + range_low) / 2
    return np.sign(c - midpoint).astype(float)

def _obv_divergence(c, v, p=10):
    """Price and OBV moving in opposite directions (divergence signal)."""
    obv = _obv(c, v)
    price_roc = c.pct_change(p)
    obv_roc   = obv.diff(p) / (obv.abs().rolling(p).mean() + 1e-9)
    same_dir  = np.sign(price_roc) == np.sign(obv_roc)
    return pd.Series(np.where(same_dir, np.sign(price_roc), -np.sign(price_roc)),
                     index=c.index).astype(float)

def _pivot_distance(c, df_1d_shifted):
    """Distance from current close to yesterday's pivot point (%)."""
    prev_h = df_1d_shifted["high"]
    prev_l = df_1d_shifted["low"]
    prev_c = df_1d_shifted["close"]
    pp = (prev_h + prev_l + prev_c) / 3
    r1 = 2 * pp - prev_l
    s1 = 2 * pp - prev_h
    pp_dist = (c - pp) / pp.replace(0, np.nan) * 100
    r1_dist = (c - r1) / r1.replace(0, np.nan) * 100
    s1_dist = (c - s1) / s1.replace(0, np.nan) * 100
    above_pp = np.sign(c - pp).astype(float)
    return pp_dist, r1_dist, s1_dist, above_pp


# ── signal computation ────────────────────────────────────────────────────────

def compute_signals_for_tf(df: pd.DataFrame, label: str,
                            df_1d_shifted: pd.DataFrame = None) -> pd.DataFrame:
    o=df["open"]; h=df["high"]; l=df["low"]; c=df["close"]; v=df["volume"]
    sig = pd.DataFrame(index=df.index)

    # ── Price action ───────────────────────────────────────────────────────────
    sig[f"bp_{label}"]          = _bp(c, l, h)
    sig[f"body_{label}"]        = _body(o, c, h, l)
    sig[f"dir_{label}"]         = np.sign(c - o).astype(float)
    sig[f"chg_{label}"]         = c.pct_change() * 100
    sig[f"z_score_{label}"]     = _z_score(c, 20)
    sig[f"vwap_dist_{label}"]   = _vwap_rolling(c, v, 24)
    sig[f"roc_3_{label}"]       = _roc(c, 3)
    sig[f"roc_6_{label}"]       = _roc(c, 6)
    sig[f"roc_12_{label}"]      = _roc(c, 12)
    if label == "1h":
        sig[f"roc_24_{label}"]  = _roc(c, 24)

    # ── Candle structure ───────────────────────────────────────────────────────
    uw = _upper_wick(o, c, h, l)
    lw = _lower_wick(o, c, h, l)
    sig[f"upper_wick_{label}"]        = uw
    sig[f"lower_wick_{label}"]        = lw
    sig[f"wick_imbalance_{label}"]    = lw - uw
    sig[f"engulfing_{label}"]         = _engulfing(o, c)
    sig[f"harami_{label}"]            = _harami(o, c)
    sig[f"three_in_row_{label}"]      = _three_soldiers_crows(o, c)
    sig[f"tweezer_{label}"]           = _tweezer(h, l, c)
    sig[f"inside_bar_{label}"]        = _inside_bar(h, l)
    sig[f"outside_bar_{label}"]       = _outside_bar(o, c, h, l)
    sig[f"pin_bar_{label}"]           = _pin_bar(o, c, h, l)
    sig[f"morning_evening_star_{label}"] = _morning_evening_star(o, c)
    sig[f"hammer_{label}"]            = _hammer(o, c, h, l)
    sig[f"doji_{label}"]              = _doji(o, c, h, l)
    sig[f"high_in_range_{label}"]     = _bp(h, l, h)    # where high sits in day range
    sig[f"close_in_range_{label}"]    = _bp(c, l, h)    # same as bp

    # ── Momentum ───────────────────────────────────────────────────────────────
    sk = _stoch_k(h, l, c, 14)
    sd = sk.rolling(3).mean()
    cross_up = (sk > sd) & (sk.shift() <= sd.shift())
    cross_dn = (sk < sd) & (sk.shift() >= sd.shift())
    sig[f"stoch_k_{label}"]     = sk
    sig[f"stoch_d_{label}"]     = sd
    sig[f"stoch_cross_{label}"] = np.where(cross_up, 1, np.where(cross_dn, -1, 0)).astype(float)
    sig[f"stoch_ob_{label}"]    = (sk >= 80).astype(float)
    sig[f"stoch_os_{label}"]    = (sk <= 20).astype(float)

    rsi_v = _rsi(c, 14)
    sig[f"rsi_{label}"]    = rsi_v
    sig[f"rsi_ob_{label}"] = (rsi_v >= 70).astype(float)
    sig[f"rsi_os_{label}"] = (rsi_v <= 30).astype(float)

    ml, ms, mh = _macd(c)
    sig[f"macd_line_{label}"]  = ml
    sig[f"macd_hist_{label}"]  = mh
    cx_up = (mh > 0) & (mh.shift() <= 0)
    cx_dn = (mh < 0) & (mh.shift() >= 0)
    sig[f"macd_cross_{label}"] = np.where(cx_up, 1, np.where(cx_dn, -1, 0)).astype(float)

    sig[f"cci_{label}"]         = _cci(c, h, l, 20)
    sig[f"cci_ob_{label}"]      = (_cci(c, h, l, 20) >= 100).astype(float)
    sig[f"cci_os_{label}"]      = (_cci(c, h, l, 20) <= -100).astype(float)
    sig[f"williams_r_{label}"]  = _williams_r(h, l, c, 14)
    sig[f"consec_dir_{label}"]  = _consecutive_dir(c)

    # ── Trend (EMA) ────────────────────────────────────────────────────────────
    e5   = c.ewm(span=5,  adjust=False).mean()
    e20  = c.ewm(span=20, adjust=False).mean()
    e50  = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    sig[f"ema_bias_{label}"]    = np.sign(e5 - e20).astype(float)
    sig[f"ema_stack_{label}"]   = np.where(
        (e5>e20)&(e20>e50),  1.0, np.where((e5<e20)&(e20<e50), -1.0, 0.0))
    sig[f"ema_gap_{label}"]     = (e5 - e20) / c.replace(0, np.nan) * 100
    sig[f"price_vs_e200_{label}"] = np.sign(c - e200).astype(float)
    e200_cross = pd.Series(np.where(
        (c > e200) & (c.shift() <= e200.shift()),  1,
        np.where((c < e200) & (c.shift() >= e200.shift()), -1, 0)
    ), index=c.index).astype(float)
    sig[f"e200_cross_{label}"]  = e200_cross

    # ── Ichimoku ───────────────────────────────────────────────────────────────
    tk_cross, cloud_pos, bull_cloud, tk_vs_kijun, pvt, pvk = _ichimoku(h, l, c)
    sig[f"ichi_tk_cross_{label}"]  = tk_cross
    sig[f"ichi_cloud_{label}"]     = cloud_pos
    sig[f"ichi_cloud_color_{label}"] = bull_cloud
    sig[f"ichi_tk_kijun_{label}"]  = tk_vs_kijun
    sig[f"ichi_price_tenkan_{label}"] = pvt
    sig[f"ichi_price_kijun_{label}"]  = pvk

    # ── ADX / trend strength ───────────────────────────────────────────────────
    adx_v, di_p, di_n = _adx(h, l, c, 14)
    sig[f"adx_{label}"]         = adx_v
    sig[f"di_diff_{label}"]     = di_p - di_n
    sig[f"adx_trend_{label}"]   = np.where(
        (adx_v>25)&(di_p>di_n),  1.0, np.where((adx_v>25)&(di_n>di_p), -1.0, 0.0))
    sig[f"adx_strong_{label}"]  = (adx_v > 40).astype(float)

    # ── Volume & Flow ──────────────────────────────────────────────────────────
    obv_v  = _obv(c, v)
    obv_ema = obv_v.ewm(span=20, adjust=False).mean()
    sig[f"obv_{label}"]            = obv_v
    sig[f"obv_slope_{label}"]      = obv_v.diff(3)
    sig[f"obv_vs_ema_{label}"]     = np.sign(obv_v - obv_ema).astype(float)
    sig[f"obv_divergence_{label}"] = _obv_divergence(c, v, 5)
    sig[f"cmf_{label}"]            = _cmf(c, h, l, v, 20)
    sig[f"mfi_{label}"]            = _mfi(c, h, l, v, 14)
    sig[f"mfi_ob_{label}"]         = (_mfi(c, h, l, v, 14) >= 80).astype(float)
    sig[f"mfi_os_{label}"]         = (_mfi(c, h, l, v, 14) <= 20).astype(float)
    sig[f"force_index_{label}"]    = np.sign(_force_index(c, v, 13)).astype(float)
    sig[f"cum_delta_{label}"]      = _cum_delta(o, c, v, 10)
    sig[f"cum_delta_sign_{label}"] = np.sign(_cum_delta(o, c, v, 10)).astype(float)
    vm = v.rolling(20).mean()
    sig[f"vol_ratio_{label}"]      = v / vm.replace(0, np.nan)
    sig[f"high_vol_{label}"]       = (v > vm * 2).astype(float)

    # ── Volatility ─────────────────────────────────────────────────────────────
    br = h - l; avg_br = br.rolling(20).mean()
    sig[f"range_ratio_{label}"]  = br / avg_br.replace(0, np.nan)
    sig[f"atr_ratio_{label}"]    = _atr(h, l, c, 14) / c.replace(0, np.nan) * 100
    bw, bp_bb = _bb(c, 20, 2)
    sig[f"bb_width_{label}"]     = bw
    sig[f"bb_pct_{label}"]       = bp_bb
    sig[f"bb_squeeze_{label}"]   = (bw < bw.rolling(50).quantile(0.20)).astype(float)

    # ── Donchian ───────────────────────────────────────────────────────────────
    dc_pos, dc_wid, dc_brk = _donchian(h, l, c, 20)
    sig[f"donchian_pos_{label}"]   = dc_pos
    sig[f"donchian_width_{label}"] = dc_wid
    sig[f"donchian_brk_{label}"]   = dc_brk

    # ── SMC / Structure ────────────────────────────────────────────────────────
    sig[f"bos_{label}"]            = _bos(c, h, l, 20)
    sig[f"bos_strict_{label}"]     = _bos(c, h, l, 10)
    sig[f"choch_{label}"]          = _choch_proxy(c, h, l, 20)
    sig[f"fvg_{label}"]            = _fvg(h, l)
    sig[f"mkt_struct_{label}"]     = _market_structure(c, h, l, 10)
    sig[f"premium_disc_{label}"]   = _premium_discount(c, h, l, 20)

    # ── Pivot points (only at 1h, using daily data) ────────────────────────────
    if label == "1h" and df_1d_shifted is not None:
        pp_d, r1_d, s1_d, above_pp = _pivot_distance(c, df_1d_shifted)
        sig["pp_dist_1h"]    = pp_d
        sig["r1_dist_1h"]    = r1_d
        sig["s1_dist_1h"]    = s1_d
        sig["above_pivot_1h"] = above_pp

    return sig


def build_signals(df_1m: pd.DataFrame) -> pd.DataFrame:
    def resamp(tf):
        return df_1m.resample(tf).agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"),     close=("close","last"),
            volume=("volume","sum"),
        ).dropna()

    df_1h = resamp("1h")
    df_4h = resamp("4h")
    df_1d = resamp("1D")

    # Yesterday's daily data for pivot points (shift 1 daily, then ffill to 1h)
    df_1d_prev = df_1d.shift(1).reindex(df_1h.index, method="ffill")

    sig_1h = compute_signals_for_tf(df_1h, "1h", df_1d_shifted=df_1d_prev)
    sig_4h = compute_signals_for_tf(df_4h, "4h")
    sig_1d = compute_signals_for_tf(df_1d, "1d")

    # Forward returns on 1h bars
    fwd = pd.DataFrame(index=df_1h.index)
    c1h = df_1h["close"]
    for lbl, n in FWD_HORIZONS.items():
        fwd[lbl] = c1h.shift(-n) / c1h - 1.0
    fwd["fwd_dir_1h"] = np.sign(fwd["fwd_1h"])

    # Use last COMPLETED bar: shift 1 at native resolution, ffill to 1h index
    sig_1h_lag = sig_1h.shift(1)
    sig_4h_lag = sig_4h.shift(1).reindex(df_1h.index, method="ffill")
    sig_1d_lag = sig_1d.shift(1).reindex(df_1h.index, method="ffill")

    result = pd.concat([sig_1h_lag, sig_4h_lag, sig_1d_lag, fwd], axis=1)

    # ── Combined / agreement signals ───────────────────────────────────────────
    def agree(a, b, pos=1, neg=-1):
        return np.where((a==pos)&(b==pos), 1.0, np.where((a==neg)&(b==neg), -1.0, 0.0))

    result["ema_stack_agree_1h4h"]  = agree(result["ema_stack_1h"],  result["ema_stack_4h"])
    result["ema_stack_agree_all"]   = np.where(
        (result["ema_stack_1h"]==1)&(result["ema_stack_4h"]==1)&(result["ema_stack_1d"]==1), 1.0,
        np.where((result["ema_stack_1h"]==-1)&(result["ema_stack_4h"]==-1)&(result["ema_stack_1d"]==-1), -1.0, 0.0))
    result["bp_agree_1h4h"]         = agree(np.sign(result["bp_1h"]-0.5), np.sign(result["bp_4h"]-0.5))
    result["bp_agree_all"]          = np.where(
        (result["bp_1h"]>0.5)&(result["bp_4h"]>0.5)&(result["bp_1d"]>0.5), 1.0,
        np.where((result["bp_1h"]<0.5)&(result["bp_4h"]<0.5)&(result["bp_1d"]<0.5), -1.0, 0.0))
    result["dir_agree_1h4h"]        = agree(result["dir_1h"],  result["dir_4h"])
    result["bos_agree_1h4h"]        = agree(result["bos_1h"],  result["bos_4h"],  pos=1.0, neg=-1.0)
    result["ichi_agree_1h4h"]       = agree(result["ichi_cloud_1h"], result["ichi_cloud_4h"])
    result["adx_trend_agree_1h4h"]  = agree(result["adx_trend_1h"],  result["adx_trend_4h"])
    result["obv_ema_agree_1h4h"]    = agree(result["obv_vs_ema_1h"], result["obv_vs_ema_4h"])
    result["cmf_agree_1h4h"]        = agree(np.sign(result["cmf_1h"]), np.sign(result["cmf_4h"]))
    result["stoch_cross_agree_1h4h"]= agree(result["stoch_cross_1h"], result["stoch_cross_4h"])
    result["mkt_struct_agree_1h4h"] = agree(result["mkt_struct_1h"], result["mkt_struct_4h"])
    # Trend + volume flow confluence
    result["ema_obv_confluence"]    = agree(result["ema_stack_4h"], result["obv_vs_ema_1h"])
    result["ema_cmf_confluence"]    = agree(result["ema_stack_4h"], np.sign(result["cmf_1h"]))
    # ADX confirms ema direction
    result["adx_ema_1h"]            = agree(result["adx_trend_1h"], result["ema_stack_1h"])
    result["adx_ema_4h"]            = agree(result["adx_trend_4h"], result["ema_stack_4h"])
    # BOS + SMC confluence
    result["bos_ema_agree_4h"]      = agree(result["bos_4h"],  result["ema_stack_4h"],  pos=1.0, neg=-1.0)
    result["fvg_ema_agree_1h"]      = agree(result["fvg_1h"],  result["ema_stack_1h"],  pos=1.0, neg=-1.0)
    result["ichi_adx_1h"]           = agree(result["ichi_cloud_1h"], result["adx_trend_1h"])

    return result


# ── Coinalyze (expanded: + funding rate) ─────────────────────────────────────

def fetch_coinalyze(symbol: str) -> pd.DataFrame:
    now_unix  = int(time.time())
    from_unix = now_unix - 90 * 24 * 3600
    params    = {"symbols": symbol, "interval": "1hour",
                 "from": from_unix, "to": now_unix, "api_key": COINALYZE_KEY}

    def _get(endpoint, cols):
        try:
            r = requests.get(f"{COINALYZE_BASE}/{endpoint}", params=params, timeout=15)
            r.raise_for_status()
            rows = r.json()[0]["history"]
            df   = pd.DataFrame(rows)
            df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df = df.set_index("t").rename(columns=cols)
            return df[list(cols.values())]  # keep only the renamed target columns
        except Exception as e:
            print(f"  [{endpoint}] {e}")
            return pd.DataFrame()

    time.sleep(0.1)
    liq_df = _get("liquidation-history",      {"l": "long_liq",  "s": "short_liq"})
    time.sleep(0.3)
    ls_df  = _get("long-short-ratio-history", {"l": "ls_long",   "s": "ls_short"})
    time.sleep(0.3)
    oi_df  = _get("open-interest-history",    {"o": "oi"})
    time.sleep(0.3)
    fr_df  = _get("funding-rate-history",     {"r": "funding_rate"})

    frames = [f for f in [liq_df, ls_df, oi_df, fr_df] if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")
    df = df.astype(float)

    # Liq signals
    if "long_liq" in df.columns and "short_liq" in df.columns:
        tl = df["long_liq"] + df["short_liq"]
        df["liq_bias"] = np.where(tl > 0.001, (df["short_liq"] - df["long_liq"]) / tl, 0.0)
        score = pd.Series(0, index=df.index)
        score += (df["liq_bias"] >= _LIQ_BIAS_STRONG).astype(int)
        score -= (df["liq_bias"] <= -_LIQ_BIAS_STRONG).astype(int)
        if "ls_short" in df.columns:
            score += (df["ls_short"] >= _LS_CROWD_THRESH).astype(int)
            score -= (df["ls_long"]  >= _LS_CROWD_THRESH).astype(int)
        df["liq_score"] = score.clip(-2, 2)

    # OI signals
    if "oi" in df.columns:
        df["oi_chg_pct"] = df["oi"].pct_change() * 100

    # Funding signals
    if "funding_rate" in df.columns:
        df["funding_rate_bps"] = df["funding_rate"] * 10000  # in basis points
        df["funding_pos"]      = (df["funding_rate"] > 0).astype(float) - (df["funding_rate"] < 0).astype(float)
        df["funding_cum_3h"]   = df["funding_rate"].rolling(3).sum()
        df["funding_cum_8h"]   = df["funding_rate"].rolling(8).sum()
        df["funding_cum_24h"]  = df["funding_rate"].rolling(24).sum()
        df["funding_extreme_pos"] = (df["funding_rate_bps"] >  8).astype(float)
        df["funding_extreme_neg"] = (df["funding_rate_bps"] < -8).astype(float)
        df["funding_extreme"]     = df["funding_extreme_pos"] - df["funding_extreme_neg"]

    keep = [c for c in ["liq_bias","liq_score","ls_long","ls_short","oi_chg_pct",
                         "funding_rate","funding_rate_bps","funding_pos",
                         "funding_cum_3h","funding_cum_8h","funding_cum_24h","funding_extreme"]
            if c in df.columns]
    return df[keep]


# ── correlation engine ────────────────────────────────────────────────────────

def correlate_signals(data, signal_cols, target_cols, min_n=MIN_N):
    rows = []
    for sig in signal_cols:
        if sig not in data.columns:
            continue
        for tgt in target_cols:
            if tgt not in data.columns:
                continue
            sub = data[[sig, tgt]].dropna()
            n   = len(sub)
            if n < min_n:
                continue
            s = sub[sig].values.astype(float)
            t = sub[tgt].values.astype(float)
            if np.std(s) < 1e-9 or np.std(t) < 1e-9:
                continue
            pr, pp = stats.pearsonr(s, t)
            sr, _  = stats.spearmanr(s, t)
            rows.append({"signal": sig, "target": tgt,
                         "pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "n": n})
    return pd.DataFrame(rows)


def decile_analysis(data, signal, target, n=10):
    sub = data[[signal, target]].dropna()
    if len(sub) < n * 10:
        return ""
    try:
        sub = sub.copy()
        sub["d"] = pd.qcut(sub[signal], n, labels=False, duplicates="drop")
        grp = sub.groupby("d")[target].agg(["mean","count"])
        lines = [f"    {'D':<4} {'N':>6} {'Mean fwd%':>11}"]
        for d, row in grp.iterrows():
            bar = ("+" if row["mean"] > 0 else "-") * min(int(abs(row["mean"]) * 4000), 30)
            lines.append(f"    {d+1:<4} {int(row['count']):>6} {row['mean']*100:>+10.3f}%  {bar}")
        return "\n".join(lines)
    except Exception:
        return ""


# ── category helper ───────────────────────────────────────────────────────────

def _cat(name):
    if any(x in name for x in ("liq","ls_","oi_chg","funding")):      return "External"
    if any(x in name for x in ("agree","confluence","ema_obv","ema_cmf",
                                "adx_ema","bos_ema","fvg_ema","ichi_adx")): return "Combined"
    if "1d" in name:                                                   return "Daily"
    if "4h" in name:                                                   return "4h"
    if any(x in name for x in ("bos","choch","fvg","mkt_struct","premium")): return "SMC"
    if any(x in name for x in ("donchian",)):                          return "Donchian"
    if any(x in name for x in ("bb_","range_ratio","atr_ratio","vol_ratio",
                                "high_vol")):                          return "Volatility"
    if any(x in name for x in ("ema_","e200","price_vs_e200","adx","di_diff",
                                "ichi")):                              return "Trend"
    if any(x in name for x in ("stoch","rsi","macd","consec","cci","williams","roc_")): return "Momentum"
    if any(x in name for x in ("obv","cmf","mfi","force","cum_delta","vol")): return "Volume/Flow"
    if any(x in name for x in ("hammer","doji","engulf","harami","three_in",
                                "tweezer","inside_bar","outside_bar","pin_bar",
                                "morning_ev","wick","body","dir_","high_in",
                                "close_in")): return "Candle"
    if any(x in name for x in ("pp_dist","r1_dist","s1_dist","above_pivot")): return "Pivots"
    if any(x in name for x in ("bp_","chg_","z_score","vwap","roc_24")): return "Price Action"
    return "1h"


# ── reporting ─────────────────────────────────────────────────────────────────

def report_asset(asset, data, corr_df):
    primary = "fwd_1h"
    sub = corr_df[corr_df["target"] == primary].copy()
    sub["abs_r"] = sub["pearson_r"].abs()
    sub = sub.sort_values("abs_r", ascending=False)

    print(f"\n{SEP}")
    print(f"  {asset} — {len(data):,} 1h bars | signals: {len(sub)}")
    print(SEP)

    def _r(sig, tgt):
        m = corr_df[(corr_df["signal"] == sig) & (corr_df["target"] == tgt)]
        return f"{m['pearson_r'].iloc[0]:+.3f}" if len(m) else "  n/a"

    def _sr(sig):
        m = sub[sub["signal"] == sig]
        return f"{m['spearman_r'].iloc[0]:+.3f}" if len(m) else "  n/a"

    print(f"\n  {'Signal':<44} {'Cat':<12} {'r_1h':>6} {'r_4h':>6} {'r_8h':>6} "
          f"{'r_24h':>6} {'spr':>6} {'p':>7} {'N':>6}")
    print(f"  {'-'*44} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")

    for i, (_, row) in enumerate(sub.iterrows()):
        if i >= TOP_N:
            break
        sg = row["signal"]
        print(f"  {sg:<44} {_cat(sg):<12} {row['pearson_r']:+6.3f} {_r(sg,'fwd_4h'):>6} "
              f"{_r(sg,'fwd_8h'):>6} {_r(sg,'fwd_24h'):>6} {_sr(sg):>6} "
              f"{row['pearson_p']:>7.1e} {int(row['n']):>6}")

    # Category summary
    print(f"\n  Category summary — mean |r| for fwd_1h:")
    print(f"  {'-'*58}")
    cats = sub.copy()
    cats["cat"] = cats["signal"].map(_cat)
    cs = cats.groupby("cat")["abs_r"].agg(["mean","max","count"]).sort_values("mean", ascending=False)
    for cat, row in cs.iterrows():
        bar = "█" * int(row["mean"] * 300)
        print(f"  {cat:<14}  mean={row['mean']:.4f}  max={row['max']:.4f}  "
              f"n={int(row['count']):>3}  {bar}")

    # Decile analysis: top 6
    print(f"\n  Top-6 decile analysis vs fwd_1h:")
    for sg in sub.head(6)["signal"].tolist():
        txt = decile_analysis(data, sg, primary)
        if txt:
            rv = sub[sub["signal"] == sg]["pearson_r"].iloc[0]
            print(f"\n  {sg}  (r={rv:+.4f})")
            print(txt)

    # Binary breakdown
    print(f"\n  Binary signal breakdown (mean fwd_1h when signal = +1 vs −1):")
    bin_sigs = [s for s in sub["signal"] if any(x in s for x in
        ("dir_","ema_bias","ema_stack","stoch_cross","macd_cross","engulf","bos_",
         "choch","fvg","mkt_struct","agree","liq_score","funding_extreme","adx_trend",
         "ichi_cloud","ichi_tk_cross","e200_cross","above_pivot","harami","three_in",
         "tweezer","outside_bar","pin_bar","morning_ev","obv_vs_ema","force_index",
         "cmf","cum_delta_sign","premium_disc","obv_divergence"))]
    print(f"  {'Signal':<44} {'Up%':>9} {'Dn%':>9} {'Spread':>8} {'N+':>6} {'N-':>6}")
    print(f"  {'-'*44} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*6}")
    shown = 0
    for sg in bin_sigs[:35]:
        if sg not in data.columns:
            continue
        sub2 = data[[sg, primary]].dropna()
        pos  = sub2[sub2[sg] > 0][primary]
        neg  = sub2[sub2[sg] < 0][primary]
        if len(pos) < 30 or len(neg) < 30:
            continue
        spread = pos.mean() - neg.mean()
        print(f"  {sg:<44} {pos.mean()*100:>+8.3f}% {neg.mean()*100:>+8.3f}% "
              f"{spread*100:>+7.3f}% {len(pos):>6} {len(neg):>6}")
        shown += 1
        if shown >= 25:
            break

    # Funding rate breakdown
    if "funding_rate_bps" in data.columns:
        print(f"\n  Funding rate breakdown:")
        sub3 = data[["funding_rate_bps", primary]].dropna()
        if len(sub3) >= 100:
            try:
                sub3 = sub3.copy()
                sub3["bucket"] = pd.cut(sub3["funding_rate_bps"],
                    bins=[-100, -8, -4, -1, 1, 4, 8, 100], labels=["<<-8","-8:-4","-4:-1","-1:+1","+1:+4","+4:+8",">>+8"])
                grp = sub3.groupby("bucket")[primary].agg(["mean","count"])
                print(f"  {'Funding(bps)':<12} {'N':>6} {'Mean fwd_1h%':>14}")
                for bkt, r in grp.iterrows():
                    if r["count"] < 10: continue
                    bar = ("+" if r["mean"] > 0 else "-") * min(int(abs(r["mean"]) * 4000), 25)
                    print(f"  {str(bkt):<12} {int(r['count']):>6} {r['mean']*100:>+13.3f}%  {bar}")
            except Exception as e:
                print(f"  (funding breakdown error: {e})")


# ── main ──────────────────────────────────────────────────────────────────────

def run_asset(asset, sym):
    print(f"\n{SEP}")
    print(f"  Loading {asset} ({sym}) …")
    df_1m = load_1m(sym)
    print(f"  {len(df_1m):,} 1m bars  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")

    print(f"  Computing signals …", end=" ", flush=True)
    data = build_signals(df_1m)
    print(f"{len(data):,} 1h bars, {len(data.columns)} columns")

    print(f"  Fetching Coinalyze (1h, 90-day) …", end=" ", flush=True)
    cz_sym = COINALYZE_SYMBOLS.get(asset)
    if cz_sym:
        cz = fetch_coinalyze(cz_sym)
        if not cz.empty:
            cz_lagged = cz.shift(1).reindex(data.index, method="ffill")
            data = data.join(cz_lagged, how="left")
            print(f"{len(cz)} bars, cols: {list(cz.columns)}")
        else:
            print("unavailable")

    target_cols = list(FWD_HORIZONS.keys()) + ["fwd_dir_1h"]
    signal_cols = [c for c in data.columns if c not in target_cols]

    print(f"  Running correlations ({len(signal_cols)} signals × {len(FWD_HORIZONS)} horizons) …",
          end=" ", flush=True)
    corr_df = correlate_signals(data, signal_cols, list(FWD_HORIZONS.keys()))
    print(f"{len(corr_df)} valid pairs")

    report_asset(asset, data, corr_df)
    return data, corr_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ALL", choices=list(ASSETS.keys()) + ["ALL"])
    args = parser.parse_args()

    print(SEP)
    print("  Comprehensive 1h signal correlation sweep")
    print("  Signals: price action / candle / momentum / trend / volume+flow /")
    print("           volatility / Donchian / SMC / pivots / Ichimoku / ADX /")
    print("           agreement / Coinalyze (liq + LS + OI + funding)")
    print(f"  TFs: 1h + 4h + 1d  |  Horizons: 1h/4h/8h/24h  |  Min N: {MIN_N}")
    print(SEP)

    targets = {args.asset: ASSETS[args.asset]} if args.asset != "ALL" else ASSETS
    results = {}
    for asset, sym in targets.items():
        data, corr = run_asset(asset, sym)
        results[asset] = (data, corr)

    if len(results) == 3:
        print(f"\n{SEP}")
        print("  CROSS-ASSET: signals in top-20 |r_1h| for ALL three assets")
        print(SEP)
        top_sets = []
        for asset, (_, corr) in results.items():
            sub = corr[corr["target"] == "fwd_1h"].copy()
            sub["abs_r"] = sub["pearson_r"].abs()
            top_sets.append(set(sub.nlargest(20, "abs_r")["signal"]))
        universal = top_sets[0] & top_sets[1] & top_sets[2]
        all_corrs = {asset: corr for asset, (_, corr) in results.items()}
        def _rv(asset, sig):
            m = all_corrs[asset]
            m2 = m[(m["signal"]==sig)&(m["target"]=="fwd_1h")]
            return f"{m2['pearson_r'].iloc[0]:+.3f}" if len(m2) else " n/a "
        if universal:
            print(f"  {len(universal)} universal signals (top-20 for all three assets):")
            sigs_r = []
            for sig in universal:
                avg_r = np.mean([abs(float(_rv(a, sig).strip())) for a in results])
                sigs_r.append((avg_r, sig))
            for avg_r, sig in sorted(sigs_r, reverse=True):
                print(f"  {sig:<46} BTC:{_rv('BTC',sig)} ETH:{_rv('ETH',sig)} SOL:{_rv('SOL',sig)}  "
                      f"(avg|r|={avg_r:.3f}  cat={_cat(sig)})")
        else:
            print("  No universal top-20. Showing signals in top-20 for ≥2 assets:")
            from collections import Counter
            all_top = []
            for asset, (_, corr) in results.items():
                sub = corr[corr["target"]=="fwd_1h"].copy()
                sub["abs_r"] = sub["pearson_r"].abs()
                all_top.extend(sub.nlargest(20, "abs_r")["signal"].tolist())
            cnt = Counter(all_top)
            shown = 0
            for sig, cnt_v in sorted(cnt.items(), key=lambda x: -x[1]):
                if cnt_v < 2 or shown > 30: continue
                print(f"  {sig:<46} (in {cnt_v}/3) BTC:{_rv('BTC',sig)} "
                      f"ETH:{_rv('ETH',sig)} SOL:{_rv('SOL',sig)}  cat={_cat(sig)}")
                shown += 1

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
