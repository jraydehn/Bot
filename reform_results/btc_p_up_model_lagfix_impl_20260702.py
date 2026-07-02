"""
btc_p_up_model.py — BTC directional p_up v2.

Replaces the composite 2D lookup table with a trained LightGBM classifier.
Combines price-based features (MACD, stoch, momentum, Bollinger, RSI) with
live signals from the confirm object (stoch_bias, vpin_score, confirmation_bias).

Model path: reform_results/btc_p_up_v2.pkl
Falls back to None when the model file is absent so the caller can use
the existing lookup_p_up_blended path unchanged.

[2026-07-02 FEATURE FIX] Vector-level audit showed the live path fed the model
inputs that diverged from the training pipeline:
  - vwap_distance_pct was passed in PERCENT while training used FRACTION (x100 bug)
  - 4h features (stoch_k_4h, rsi_4h, chg_4h_atr) used the in-progress partial 4h
    bar, while training merged the *containing* 4h bar on open time (which leaks
    1-3h of future data for 75% of rows — irreproducible live by definition)
  - stoch_k / ema_stack / ema_stretch came from confirm's fresh mid-hour 5m/15m
    values instead of the training-timing 15m bar (open <= last completed 1h bar)
  - vwap_stretch_score came from confirm's 1m sigma-band score instead of the
    training 1h-session day-std cut
  - composite_trend/rev/p_up came from the runner's (+/-5 expanded) scheme instead
    of the training-formula (+/-6 4h votes / +/-8 1h votes) values
  - rvol denominator excluded the current bar (training rolling(24) includes it)
compute_p_up now builds a training-consistent, leak-free vector internally
(_build_vector_fixed). 4h features use the LAST COMPLETED 4h bar. On any
exception it falls back to the exact legacy construction (_build_vector_legacy).
Models trained on the lag-corrected dataset (see reform_results/
btc_p_up_v2_lagfix_*.pkl) match this vector spec exactly.
Backup of pre-fix module: btc_p_up_model_pre_feature_fix_20260702.py
"""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "reform_results" / "btc_p_up_v2.pkl"
_CAL_PATH = Path(__file__).parent / "composite_calibration.json"
_CACHE: "dict | None | str" = "unloaded"
_CAL_CACHE: "dict | None | str" = "unloaded"

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


def _load_cal() -> "dict | None":
    """Cached composite_calibration.json (same table the training pipeline uses)."""
    global _CAL_CACHE
    if _CAL_CACHE != "unloaded":
        return _CAL_CACHE
    try:
        with open(_CAL_PATH) as f:
            _CAL_CACHE = json.load(f)
    except Exception:
        _CAL_CACHE = None
    return _CAL_CACHE


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

def _rsi_train(s: pd.Series, p: int = 14, tail: int = 600) -> float:
    """RSI matching the training pipeline (full-series ewm; tail is converged)."""
    if len(s) < p + 1:
        return NAN
    d = s.iloc[-tail:].diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return float((100 - 100 / (1 + g / l.replace(0, 1e-10))).iloc[-1])

def _rsi_train_series(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

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


# ── [2026-07-02] training-consistent fixed-vector construction ────────────

def _completed_4h(df_4h: pd.DataFrame, t_wall: pd.Timestamp) -> pd.DataFrame:
    """4h bars fully closed as of t_wall (bar_open + 4h <= t_wall). Leak-free."""
    return df_4h[df_4h.index + pd.Timedelta(hours=4) <= t_wall]

def _session_vwap_fixed(df_1h: pd.DataFrame):
    """Training-formula session VWAP on completed 1h bars of the current UTC day.
    Returns (dist_fraction, stretch_score) where the day-std cut uses only bars
    seen so far (expanding — no full-day lookahead)."""
    today = df_1h.index[-1].date()
    day = df_1h[df_1h.index.date == today]
    if day.empty:
        day = df_1h.iloc[-24:]
    tp = (day["high"] + day["low"] + day["close"]) / 3
    cum_vol = day["volume"].cumsum()
    if float(cum_vol.iloc[-1]) <= 0:
        return NAN, NAN
    vwap = (tp * day["volume"]).cumsum() / cum_vol
    v = float(vwap.iloc[-1])
    dist = (float(day["close"].iloc[-1]) - v) / v          # FRACTION (training scale)
    if len(day) < 3:
        return dist, NAN
    day_std = float(tp.std())
    if day_std != day_std or day_std <= 0:
        return dist, NAN
    z = dist / (day_std / v)
    if z <= -2:
        stretch = 2.0
    elif z <= -1:
        stretch = 1.0
    elif z < 1:
        stretch = 0.0
    elif z < 2:
        stretch = -1.0
    else:
        stretch = -2.0
    return dist, stretch

def _features_15m_fixed(df_15m: pd.DataFrame, t_last_1h: pd.Timestamp):
    """Training-timing 15m features: last completed 15m bar with open <= last
    completed 1h bar open (mirrors the training merge_asof). Returns
    (stoch_k, ema_stack_bias, ema_stretch_score); NaN entries on failure."""
    w = df_15m[df_15m.index <= t_last_1h]
    if len(w) < 20:
        return NAN, NAN, NAN
    w = w.iloc[-120:]
    c = w["close"]
    ll = w["low"].rolling(14).min().iloc[-1]
    hh = w["high"].rolling(14).max().iloc[-1]
    rng = hh - ll
    sk = float((c.iloc[-1] - ll) / rng * 100) if (rng == rng and rng != 0) else NAN
    e9 = float(_ema(c, 9).iloc[-1])
    e21 = float(_ema(c, 21).iloc[-1])
    e50 = float(_ema(c, 50).iloc[-1])
    cl = float(c.iloc[-1])
    if e9 > e21 > e50 and cl > e9:
        stack = 1.0
    elif e9 < e21 < e50 and cl < e9:
        stack = -1.0
    else:
        stack = 0.0
    e20 = float(_ema(c, 20).iloc[-1])
    st = (cl - e20) / e20 if e20 else NAN
    if st != st:
        stretch = NAN
    elif st > 0.001:
        stretch = -1.0
    elif st < -0.001:
        stretch = 1.0
    else:
        stretch = 0.0
    return sk, stack, stretch

def _composite_fixed(df_1h: pd.DataFrame, df4c: pd.DataFrame):
    """Training-formula composite_trend (4h votes, clip +/-6, COMPLETED 4h bars
    only), composite_rev (1h votes, clip +/-8) and calibration-table p_up."""
    # ---- trend from completed 4h bars ----
    d4 = df4c.iloc[-300:]
    c4 = d4["close"]
    rsi4 = float(_rsi_train_series(c4).iloc[-1])
    macd4 = _ema(c4, 12) - _ema(c4, 26)
    sig4 = macd4.ewm(span=9, adjust=False).mean()
    mid4 = c4.rolling(20).mean().iloc[-1]
    std4 = c4.rolling(20).std().iloc[-1]
    bbp4 = (float(c4.iloc[-1]) - (mid4 - 2 * std4)) / (4 * std4) if (std4 and std4 == std4) else NAN
    ll4 = d4["low"].rolling(14).min().iloc[-1]
    hh4 = d4["high"].rolling(14).max().iloc[-1]
    sk4 = (float(c4.iloc[-1]) - ll4) / (hh4 - ll4) * 100 if (hh4 - ll4) else NAN
    wr4 = -100 * (hh4 - float(c4.iloc[-1])) / (hh4 - ll4) if (hh4 - ll4) else NAN
    vma4 = d4["volume"].rolling(20).mean().iloc[-1]
    vrat4 = float(d4["volume"].iloc[-1]) / vma4 if vma4 else NAN
    up4 = float(c4.iloc[-1]) > float(c4.iloc[-2])
    trend = 0.0
    trend += (1.0 if rsi4 > 55 else 0.0) - (1.0 if rsi4 < 45 else 0.0)
    trend += 1.0 if float(macd4.iloc[-1]) > float(sig4.iloc[-1]) else -1.0
    trend += (1.0 if bbp4 > 0.80 else 0.0) - (1.0 if bbp4 < 0.20 else 0.0)
    trend += (1.0 if sk4 > 80 else 0.0) - (1.0 if sk4 < 20 else 0.0)
    trend += (1.0 if wr4 > -20 else 0.0) - (1.0 if wr4 < -80 else 0.0)
    if vrat4 == vrat4 and vrat4 > 1.5:
        trend += 1.0 if up4 else -1.0
    trend = float(np.clip(trend, -6, 6))
    # ---- rev from completed 1h bars ----
    d1 = df_1h.iloc[-200:]
    c1 = d1["close"]
    rsi1 = float(_rsi_train_series(c1).iloc[-1])
    ll1 = d1["low"].rolling(14).min().iloc[-1]
    hh1 = d1["high"].rolling(14).max().iloc[-1]
    sk1 = (float(c1.iloc[-1]) - ll1) / (hh1 - ll1) * 100 if (hh1 - ll1) else NAN
    vd_frac, _ = _session_vwap_fixed(d1)
    vd = vd_frac * 100 if vd_frac == vd_frac else NAN   # training rev votes use PERCENT
    log_ret = np.log(c1 / c1.shift(1))
    z_std = float(log_ret.rolling(24).std().iloc[-1])
    z = float(log_ret.iloc[-1]) / z_std if (z_std and z_std == z_std) else NAN
    rev = 0.0
    rev += 2.0 * (rsi1 < 30) + 1.0 * (rsi1 < 40)
    rev -= 2.0 * (rsi1 > 70) + 1.0 * (rsi1 > 60)
    if sk1 == sk1:
        rev += 2.0 * (sk1 < 10) + 1.0 * (sk1 < 20)
        rev -= 2.0 * (sk1 > 90) + 1.0 * (sk1 > 80)
    if vd == vd:
        rev += 2.0 * (vd < -1.5) + 1.0 * (vd < -0.5)
        rev -= 2.0 * (vd > 1.5) + 1.0 * (vd > 0.5)
    if z == z:
        rev += 2.0 * (z < -2.0) + 1.0 * (z < -1.5)
        rev -= 2.0 * (z > 2.0) + 1.0 * (z > 1.5)
    rev = float(np.clip(rev, -8, 8))
    # ---- calibration lookup (training formula) ----
    p_up = 0.504
    cal = _load_cal()
    if cal:
        e = cal.get(f"{int(round(trend))}_{int(round(rev))}")
        if e and e.get("n", 0) >= 5:
            p_up = float(e["p_yes"])
    return trend, rev, p_up


def _build_vector_fixed(df_1h, df_4h, confirm, pm_drift_5m, df_15m):
    """Training-consistent, leak-free feature vector. df_1h must already have
    the in-progress bar dropped. Raises on any problem (caller falls back)."""
    if len(df_1h) < 60 or len(df_4h) < 20:
        raise ValueError("insufficient history")
    c1h = df_1h["close"]
    t_last = df_1h.index[-1]                       # last completed 1h bar open
    t_wall = t_last + pd.Timedelta(hours=1)        # its close (decision-hour start)

    df4c = _completed_4h(df_4h, t_wall)            # completed 4h bars only
    if len(df4c) < 20:
        raise ValueError("insufficient completed 4h bars")
    c4h = df4c["close"]

    sk4h = _stoch_k(df4c["high"], df4c["low"], c4h, 14)
    r4h = _rsi_train(c4h, 14, tail=300)
    # chg_4h_atr training-consistent: close - close.shift(5) (legacy _chg_4h_atr
    # used iloc[-5] = 4 bars back — off-by-one vs the training pipeline)
    c4at = NAN
    if len(df4c) >= 18:
        _atr4 = _atr(df4c["high"], df4c["low"], c4h, 14)
        if _atr4 == _atr4 and _atr4 != 0:
            c4at = float((c4h.iloc[-1] - c4h.iloc[-6]) / _atr4)

    e50d = _ema50_dist(c1h)
    r1h = _rsi_train(c1h, 14, tail=600)
    mh1h = _macd_hist(c1h)
    bbp = _bb_pct(c1h)

    vwap_frac, vwap_stretch = _session_vwap_fixed(df_1h)   # FRACTION + 1h-session cut

    # 15m features at training timing; confirm fallback keeps prior behavior
    sk1h = ema_stack = ema_ex = NAN
    if df_15m is not None and len(df_15m) >= 20:
        sk1h, ema_stack, ema_ex = _features_15m_fixed(df_15m, t_last)
    if sk1h != sk1h:
        sk1h = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else NAN
    if ema_stack != ema_stack:
        ema_stack = float(confirm.ema_stack_bias) if confirm.ema_stack_bias is not None else NAN
    if ema_ex != ema_ex:
        ema_ex = float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else NAN
    # NOTE: vwap_stretch stays NaN when <3 bars into the UTC day — the training
    # value is NaN there too; do NOT substitute confirm's 1m sigma-band score.

    trend, rev, comp_p_up = _composite_fixed(df_1h, df4c)

    # live signals (all-NaN in training -> model-inert; kept for compatibility)
    cb = float(confirm.confirmation_bias) if confirm.confirmation_bias is not None else NAN
    sb = float(confirm.stoch_bias) if confirm.stoch_bias is not None else NAN
    vp = float(confirm.vpin_score) if confirm.vpin_score is not None else NAN

    # rvol: training rolling(24).mean INCLUDES the current bar
    rvol = float(df_1h["volume"].iloc[-1] / df_1h["volume"].iloc[-24:].mean()) \
           if len(df_1h) >= 24 else NAN

    return np.array([[
        sk4h, e50d, r4h, r1h, mh1h,
        sk1h, vwap_frac, c4at, bbp,
        trend, rev, comp_p_up,
        ema_stack, ema_ex, vwap_stretch,
        cb, sb, vp,
        pm_drift_5m, rvol,
    ]], dtype=float)


def _build_vector_legacy(df_1h, df_4h, confirm, composite_trend, composite_rev,
                         composite_p_up_1h, pm_drift_5m):
    """Pre-2026-07-02 vector construction, unchanged (fallback path)."""
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

    return np.array([[
        sk4h, e50d, r4h, r1h, mh1h,
        sk1h, vwap, c4at, bbp,
        float(composite_trend), float(composite_rev), float(composite_p_up_1h),
        ema_stack, ema_ex, vwap_vs,
        cb, sb, vp,
        pm_drift_5m, rvol,
    ]], dtype=float)


# ── public API ────────────────────────────────────────────────────────────

def compute_p_up(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    confirm,
    composite_trend: float,
    composite_rev: float,
    composite_p_up_1h: float,
    pm_drift_5m: float = NAN,
    df_15m: "pd.DataFrame | None" = None,
) -> "float | None":
    """
    Compute new directional p_up using the trained v2 model.

    Args:
        df_1h            : 1h OHLCV bars (≥60 bars)
        df_4h            : 4h OHLCV bars (≥20 bars)
        confirm          : ConfirmationResult object from paper_trade_runner
        composite_trend  : current 1h composite trend score (legacy fallback only)
        composite_rev    : current 1h composite reversion score (legacy fallback only)
        composite_p_up_1h: 1h-only lookup p_up (legacy fallback only)
        pm_drift_5m      : 5m market price drift (NaN if unavailable)
        df_15m           : optional 15m OHLCV bars (resampled from live 1m).
                           When provided, stoch_k/ema_stack/ema_stretch are
                           computed at training timing instead of from confirm.

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

    # [2026-07-02] training-consistent vector; fall back to legacy on ANY error
    vec = None
    try:
        vec = _build_vector_fixed(df_1h, df_4h, confirm, pm_drift_5m, df_15m)
        if not np.isfinite(vec).any():
            vec = None
    except Exception:
        vec = None
    if vec is None:
        vec = _build_vector_legacy(df_1h, df_4h, confirm, composite_trend,
                                   composite_rev, composite_p_up_1h, pm_drift_5m)

    p = float(clf.predict_proba(vec)[0, 1])
    return float(np.clip(p, 0.02, 0.98))
