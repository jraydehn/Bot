"""
asset_p_up_model.py — Directional p_up v2 for ETH and SOL (shadow mode).

Identical feature set and inference path as btc_p_up_model.py.
Returns p(next 1h close > current close) in [0.02, 0.98], or None if
the model file is absent or the asset is not supported.

Model paths:
  reform_results/eth_p_up_v2.pkl
  reform_results/sol_p_up_v2.pkl

Shadow only — does not gate any trades.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent

_PATHS = {
    "ETH": _ROOT / "reform_results" / "eth_p_up_v2.pkl",
    "SOL": _ROOT / "reform_results" / "sol_p_up_v2.pkl",
}

_CACHE: dict = {}   # asset -> pipe dict or None

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]

NAN = float("nan")


def _load(asset: str):
    if asset not in _PATHS:
        return None
    if asset in _CACHE:
        return _CACHE[asset]
    path = _PATHS[asset]
    if not path.exists():
        _CACHE[asset] = None
        return None
    try:
        with open(path, "rb") as f:
            _CACHE[asset] = pickle.load(f)
        return _CACHE[asset]
    except Exception:
        _CACHE[asset] = None
        return None


# ── indicator helpers (vectorised) ────────────────────────────────────────────

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
    try:
        today = df_1h.index[-1].date()
        day   = df_1h[df_1h.index.date == today]
        if day.empty:
            day = df_1h.iloc[-24:]
        tp   = (day["high"] + day["low"] + day["close"]) / 3
        vwap = (tp * day["volume"]).cumsum() / day["volume"].cumsum()
        return float((day["close"].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100)
    except Exception:
        return NAN

def _chg_4h_atr(df_4h: pd.DataFrame) -> float:
    if len(df_4h) < 18:
        return NAN
    atr_val = _atr(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    if atr_val == 0 or math.isnan(atr_val):
        return NAN
    return float((df_4h["close"].iloc[-1] - df_4h["close"].iloc[-5]) / atr_val)


# ── public API ────────────────────────────────────────────────────────────────

def compute_p_up(
    asset: str,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    confirm,
    composite_trend: float,
    composite_rev: float,
    composite_p_up_1h: float,
    pm_drift_5m: float = NAN,
) -> "float | None":
    """
    Shadow directional p_up for ETH/SOL.
    Returns float in [0.02, 0.98] or None if model not loaded.
    """
    pipe = _load(asset)
    if pipe is None:
        return None

    clf = pipe["clf"]
    c1h = df_1h["close"]
    c4h = df_4h["close"]

    sk4h  = _stoch_k(df_4h["high"], df_4h["low"], c4h, 14)
    e50d  = _ema50_dist(c1h)
    r4h   = _rsi(c4h, 14)
    r1h   = _rsi(c1h, 14)
    mh1h  = _macd_hist(c1h)
    sk1h  = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else NAN
    vwap  = _daily_vwap_dist(df_1h)
    c4at  = _chg_4h_atr(df_4h)
    bbp   = _bb_pct(c1h)

    ema_stack = float(confirm.ema_stack_bias)    if confirm.ema_stack_bias    is not None else NAN
    ema_ex    = float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else NAN
    vwap_vs   = float(confirm.stretch_score)     if confirm.stretch_score     is not None else NAN

    cb  = float(confirm.confirmation_bias) if confirm.confirmation_bias is not None else NAN
    sb  = float(confirm.stoch_bias)        if confirm.stoch_bias        is not None else NAN
    vp  = float(confirm.vpin_score)        if confirm.vpin_score        is not None else NAN

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
