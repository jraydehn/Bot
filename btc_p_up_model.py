"""
btc_p_up_model.py — BTC directional p_up v2.

Replaces the composite 2D lookup table with a trained LightGBM classifier.
Combines price-based features (MACD, stoch, momentum, Bollinger, RSI) with
live signals from the confirm object (stoch_bias, vpin_score, confirmation_bias).

Model path: reform_results/btc_p_up_v2.pkl
Falls back to None when the model file is absent so the caller can use
the existing lookup_p_up_blended path unchanged.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "reform_results" / "btc_p_up_v2.pkl"
_CACHE: "dict | None | str" = "unloaded"

# Re-enabled 2026-05-20: used as rolling regime indicator (4h window >= 0.52 blocks NO bets).
# NOT applied as z_drift multiplier (k=0 in composite_scorer). Disable if model breaks.
_DISABLED = False

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]

NAN = float("nan")


def _load() -> "dict | None":
    global _CACHE
    if _CACHE != "unloaded":
        return _CACHE
    if not _MODEL_PATH.exists():
        _CACHE = None
        return None
    try:
        with open(_MODEL_PATH, "rb") as f:
            _CACHE = pickle.load(f)
        return _CACHE
    except Exception:
        _CACHE = None
        return None


# ── indicator helpers (vectorised pandas) ────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, p: int = 14) -> float:
    if len(s) < p + 1:
        return NAN
    d = s.diff().dropna().iloc[-(p * 3):]
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    rs = g / l.replace(0, 1e-10)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def _stoch_k(h: pd.Series, lo: pd.Series, c: pd.Series, k: int = 14) -> float:
    if len(c) < k:
        return NAN
    ll = lo.rolling(k).min().iloc[-1]
    hh = h.rolling(k).max().iloc[-1]
    rng = hh - ll
    if rng == 0 or math.isnan(rng):
        return NAN
    return float((c.iloc[-1] - ll) / rng * 100)

def _atr(h: pd.Series, lo: pd.Series, c: pd.Series, p: int = 14) -> float:
    if len(c) < 2:
        return NAN
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return float(tr.ewm(com=p - 1, adjust=False).mean().iloc[-1])

def _macd_hist(c: pd.Series, f: int = 12, s: int = 26, sig: int = 9) -> float:
    if len(c) < s + sig:
        return NAN
    macd = _ema(c, f) - _ema(c, s)
    signal = macd.ewm(span=sig, adjust=False).mean()
    return float((macd - signal).iloc[-1])

def _bb_pct(c: pd.Series, n: int = 20) -> float:
    if len(c) < n:
        return NAN
    mid = c.rolling(n).mean().iloc[-1]
    std = c.rolling(n).std().iloc[-1]
    if std == 0 or math.isnan(std):
        return NAN
    lo = mid - 2 * std
    hi = mid + 2 * std
    return float((c.iloc[-1] - lo) / (hi - lo)) if (hi - lo) > 0 else NAN

def _ema50_dist(c: pd.Series) -> float:
    if len(c) < 50:
        return NAN
    e50 = _ema(c, 50).iloc[-1]
    if e50 == 0:
        return NAN
    return float((c.iloc[-1] - e50) / e50 * 100)

def _daily_vwap_dist(df_1h: pd.DataFrame) -> float:
    """Approximate daily VWAP from 1h bars — distance of latest close from VWAP."""
    try:
        today = df_1h.index[-1].date()
        day = df_1h[df_1h.index.date == today]
        if day.empty:
            day = df_1h.iloc[-24:]
        tp = (day["high"] + day["low"] + day["close"]) / 3
        vwap = (tp * day["volume"]).cumsum() / day["volume"].cumsum()
        dist = (day["close"].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100
        return float(dist)
    except Exception:
        return NAN

def _chg_4h_atr(df_4h: pd.DataFrame) -> float:
    if len(df_4h) < 18:
        return NAN
    atr_val = _atr(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    if atr_val == 0 or math.isnan(atr_val):
        return NAN
    return float((df_4h["close"].iloc[-1] - df_4h["close"].iloc[-5]) / atr_val)


# ── public API ────────────────────────────────────────────────────────────

def compute_p_up(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    confirm,
    composite_trend: float,
    composite_rev: float,
    composite_p_up_1h: float,
    pm_drift_5m: float = NAN,
) -> "float | None":
    """
    Compute new directional p_up using the trained v2 model.

    Args:
        df_1h            : 1h OHLCV bars (≥60 bars)
        df_4h            : 4h OHLCV bars (≥20 bars)
        confirm          : ConfirmationResult object from paper_trade_runner
        composite_trend  : current 1h composite trend score
        composite_rev    : current 1h composite reversion score
        composite_p_up_1h: 1h-only lookup p_up (from lookup_p_up)
        pm_drift_5m      : 5m market price drift (NaN if unavailable)

    Returns:
        float in [0.02, 0.98] or None if model not loaded.
    """
    if _DISABLED:
        return None
    pipe = _load()
    if pipe is None:
        return None

    # Drop in-progress bar — last 1h bar may be incomplete at inference time;
    # training used only completed bars so this aligns inference with training.
    if len(df_1h) > 1:
        df_1h = df_1h.iloc[:-1]

    clf = pipe["clf"]
    c1h = df_1h["close"]
    c4h = df_4h["close"]

    # price-based features
    sk4h  = _stoch_k(df_4h["high"], df_4h["low"], c4h, 14)
    e50d  = _ema50_dist(c1h)
    r4h   = _rsi(c4h, 14)
    r1h   = _rsi(c1h, 14)
    mh1h  = _macd_hist(c1h)
    sk1h  = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else NAN
    vwap  = _daily_vwap_dist(df_1h)
    c4at  = _chg_4h_atr(df_4h)
    bbp   = _bb_pct(c1h)

    # new: EMA/VWAP stretch signals from confirm object
    ema_stack  = float(confirm.ema_stack_bias)    if confirm.ema_stack_bias    is not None else NAN
    ema_ex     = float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else NAN
    vwap_vs    = float(confirm.stretch_score)     if confirm.stretch_score     is not None else NAN

    # live signals
    cb  = float(confirm.confirmation_bias) if confirm.confirmation_bias is not None else NAN
    sb  = float(confirm.stoch_bias)        if confirm.stoch_bias        is not None else NAN
    vp  = float(confirm.vpin_score)        if confirm.vpin_score        is not None else NAN

    # relative volume from 1h bars
    rvol = float(df_1h["volume"].iloc[-1] / df_1h["volume"].iloc[-25:-1].mean()) \
           if len(df_1h) >= 26 else NAN

    vec = np.array([[
        sk4h, e50d, r4h, r1h, mh1h,
        sk1h, vwap, c4at, bbp,
        float(composite_trend), float(composite_rev), float(composite_p_up_1h),
        ema_stack, ema_ex, vwap_vs,
        cb, sb, vp,
        pm_drift_5m, rvol,
    ]], dtype=float)

    p = float(clf.predict_proba(vec)[0, 1])
    return float(np.clip(p, 0.02, 0.98))
