"""
direction_layer.py — Directional signal layer for side selection.

Computes a combined direction score from validated trend (4h continuation) and
mean-reversion (1h/15m) signals, with corrections applied based on historical
validation against 19,947 hours of BTCUSDT data.

Key corrections vs original composite_scorer:
  - Volume 4h: INVERTED (high_vol_down → +1, high_vol_up → -1)
    Selling climaxes precede bounces; buying climaxes precede fades.
  - RSI multi-TF both-oversold: changed from +1 → -1
    Both 1h+4h oversold = strong downtrend continuation, not bounce setup.
  - RSI multi-TF both-overbought: changed from -1 → +1
    Both 1h+4h overbought = strong uptrend continuation.

Direction label → side selection:
  "bullish" (score > BULL_THRESHOLD) → prefer YES bets
  "bearish" (score < BEAR_THRESHOLD) → prefer NO bets
  "neutral"                          → prefer range-bound bets:
      offset_pct > 0  (strike above spot) → OTM NO  (wins if price stays below)
      offset_pct < 0  (strike below spot) → ITM YES (wins if price stays above)
      offset_pct ≈ 0  (ATM)              → skip

Usage:
    from direction_layer import compute_direction_score
    trend, rev, score, label = compute_direction_score(df_1h, df_4h, df_15m, df_1m)
"""

from typing import Optional

import numpy as np
import pandas as pd

# Thresholds for direction label assignment.
# Historical validation: trend score ≥3 had +4.2% edge; ≥4 had +8.8% edge.
# Combined trend+rev scores are larger; ±5 gives a balanced neutral band.
BULL_THRESHOLD = 5
BEAR_THRESHOLD = -5


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _stoch_k(h: pd.Series, l: pd.Series, c: pd.Series, k: int = 14) -> pd.Series:
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()


def _keltner_pct(h: pd.Series, l: pd.Series, c: pd.Series,
                 span: int = 20, mult: float = 2.0):
    ema = c.ewm(span=span, adjust=False).mean()
    atr = _atr(h, l, c, span)
    up = ema + mult * atr
    dn = ema - mult * atr
    pct = (c - dn) / (up - dn).replace(0, np.nan)
    return pct, dn, up


def _bb_pct(c: pd.Series, n: int = 20) -> pd.Series:
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    up = mid + 2 * std
    dn = mid - 2 * std
    return (c - dn) / (up - dn).replace(0, np.nan)


def _wpr(h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def _macd_state(c: pd.Series, f: int = 12, s: int = 26, sig: int = 9) -> int:
    """Returns +1 for bullish cross/lag, -1 for bearish, 0 for none."""
    ema_f = c.ewm(span=f, adjust=False).mean()
    ema_s = c.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    xup = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    xdn = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    state = pd.Series(0, index=c.index)
    state[xup] = 1
    state[xdn] = -1
    for sh in [1, 2]:
        state[(xup.shift(sh).fillna(False)) & (state == 0)] = 1
        state[(xdn.shift(sh).fillna(False)) & (state == 0)] = -1
    return int(state.iloc[-1])


def _dc_pct(h: pd.Series, l: pd.Series, n: int = 20) -> pd.Series:
    dc_h = h.rolling(n).max()
    dc_l = l.rolling(n).min()
    return (h - dc_l) / (dc_h - dc_l).replace(0, np.nan)


# When price moves sharply in one direction, stochastic extremes signal continuation
# (not mean reversion). These thresholds define "sharp" per asset based on realized vol.
_SHARP_MOVE_THRESHOLDS = {
    "BTC": 0.008,   # 0.8% in 30 minutes
    "ETH": 0.015,   # 1.5% in 30 minutes
    "SOL": 0.020,   # 2.0% in 30 minutes
}


def compute_direction_score(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    asset: str = "BTC",
    sharp_move_pct: float = 0.0,
) -> tuple:
    """
    Compute combined directional score from trend (4h) and mean-reversion (1h/15m).

    Args:
        df_1h          : 1h OHLCV bars, at least 50 bars.
        df_4h          : 4h OHLCV bars (resampled from 1h), at least 20 bars.
        df_15m         : 15m OHLCV bars, at least 30 bars.
        df_1m          : 1m bars for VWAP computation.
        sharp_move_pct : Recent 30m price change as a fraction (e.g. -0.015 = -1.5%).
                         When a sharp move is detected, stochastic extreme votes are
                         flipped from mean-reversion to continuation signals.

    Returns:
        (trend_score, rev_score, combined_score, label, details)
          trend_score   : int, -7 to +7 (4h continuation votes)
          rev_score     : int, typically -15 to +15 (1h/15m mean-reversion votes)
          combined_score: int = trend_score + rev_score
          label         : "bullish" / "bearish" / "neutral"
          details       : dict of individual signal values for diagnostics
    """
    for df in (df_1h, df_4h, df_15m, df_1m):
        df.columns = df.columns.str.lower()

    c1h = df_1h["close"]
    h1h = df_1h["high"]
    l1h = df_1h["low"]
    v1h = df_1h["volume"]

    c4h = df_4h["close"]
    h4h = df_4h["high"]
    l4h = df_4h["low"]
    v4h = df_4h["volume"]

    c15 = df_15m["close"]
    h15 = df_15m["high"]
    l15 = df_15m["low"]

    ts_1h = c1h.index
    details = {}
    trend_score = 0
    rev_score = 0

    # Determine if a sharp directional move is active (flips stoch interpretation)
    _sharp_thresh = _SHARP_MOVE_THRESHOLDS.get(asset.upper(), 0.008)
    _sharp_down = sharp_move_pct < -_sharp_thresh
    _sharp_up   = sharp_move_pct > _sharp_thresh
    details["sharp_move_pct"] = round(sharp_move_pct * 100, 3)
    details["stoch_flipped"]  = _sharp_down or _sharp_up

    # ════════════════════════════════════════════════════════════════════════
    # TREND SCORE — 4h continuation signals (each ±1, max ±7)
    # ════════════════════════════════════════════════════════════════════════

    # 1. Stochastic 4h (+6.9% / -10.2% edge validated)
    try:
        stk4 = _stoch_k(h4h, l4h, c4h, 14)
        v = int((stk4.iloc[-1] > 80)) - int((stk4.iloc[-1] < 20))
        trend_score += v
        details["stoch_4h"] = round(float(stk4.iloc[-1]), 1)
        details["stoch_4h_vote"] = v
    except Exception:
        details["stoch_4h"] = None

    # 2. RSI 4h (+7.0% / -9.1% edge validated)
    try:
        rsi4 = _rsi(c4h, 14)
        v = int(rsi4.iloc[-1] > 70) - int(rsi4.iloc[-1] < 30)
        trend_score += v
        details["rsi_4h"] = round(float(rsi4.iloc[-1]), 1)
        details["rsi_4h_vote"] = v
    except Exception:
        details["rsi_4h"] = None

    # 3. BB %B 4h (+6.8% above / -5.3% below validated)
    try:
        bb4 = _bb_pct(c4h, 20)
        v = int(bb4.iloc[-1] > 0.80) - int(bb4.iloc[-1] < 0.20)
        trend_score += v
        details["bb_4h"] = round(float(bb4.iloc[-1]), 3)
        details["bb_4h_vote"] = v
    except Exception:
        details["bb_4h"] = None

    # 4. Keltner 4h (+7.4% above / -7.3% below validated)
    try:
        kc4, kc4_dn, kc4_up = _keltner_pct(h4h, l4h, c4h, 20, 2)
        v = int((kc4.iloc[-1] > 0.85) or (c4h.iloc[-1] > kc4_up.iloc[-1])) \
          - int((kc4.iloc[-1] < 0.15) or (c4h.iloc[-1] < kc4_dn.iloc[-1]))
        trend_score += v
        details["kc_4h"] = round(float(kc4.iloc[-1]), 3)
        details["kc_4h_vote"] = v
    except Exception:
        details["kc_4h"] = None

    # 5. Williams %R 4h (+12.5% near 0 / -10.1% near -100 validated)
    try:
        wpr4 = _wpr(h4h, l4h, c4h, 14)
        v = int(wpr4.iloc[-1] > -20) - int(wpr4.iloc[-1] < -80)
        trend_score += v
        details["wpr_4h"] = round(float(wpr4.iloc[-1]), 1)
        details["wpr_4h_vote"] = v
    except Exception:
        details["wpr_4h"] = None

    # 6. MACD 4h crossover (+3.9% / -2.8% edge validated)
    try:
        v = _macd_state(c4h)
        trend_score += v
        details["macd_4h_vote"] = v
    except Exception:
        details["macd_4h_vote"] = None

    # 7. Volume 4h directional — INVERTED from original composite
    # Validated: high_vol_down → +3.7% edge; high_vol_up → -2.7% edge
    # Selling climaxes precede bounces; buying climaxes precede fades.
    try:
        vol_ma4 = v4h.rolling(20).mean()
        vol_ratio4 = float(v4h.iloc[-1] / vol_ma4.iloc[-1]) if vol_ma4.iloc[-1] > 0 else 1.0
        pdir4 = 1 if c4h.iloc[-1] > c4h.iloc[-2] else -1
        if vol_ratio4 > 1.5:
            v = -pdir4  # INVERTED: high vol up → -1; high vol down → +1
        else:
            v = 0
        trend_score += v
        details["vol_4h_ratio"] = round(vol_ratio4, 3)
        details["vol_4h_vote"] = v
    except Exception:
        details["vol_4h_vote"] = None

    # ════════════════════════════════════════════════════════════════════════
    # REVERSION SCORE — 1h/15m mean-reversion signals
    # ════════════════════════════════════════════════════════════════════════

    # 1. RSI multi-TF interaction (strongest validated signal: +11.1% / -13.1%)
    # CORRECTED: both oversold = -1 (was +1); both overbought = +1 (was -1)
    try:
        rsi1h = _rsi(c1h, 14)
        rsi4h_from_1h = _rsi(c1h.resample("4h", origin="start_day").last().dropna(), 14)
        rsi4h_at_1h = rsi4h_from_1h.reindex(ts_1h, method="ffill")

        r1h = float(rsi1h.iloc[-1])
        r4h = float(rsi4h_at_1h.iloc[-1]) if not rsi4h_at_1h.empty else 50.0

        os1h = r1h < 30
        ob1h = r1h > 70
        os4h = r4h < 30
        ob4h = r4h > 70

        # Both-OS/OB votes are asset-specific (validated per asset):
        #   BTC: both OS → -1 (downtrend continuation, -5.2% edge)
        #        both OB → +1 (uptrend continuation, +6.6% edge)
        #   ETH: both OS → -1 (downtrend continuation, -4.8% edge)
        #        both OB →  0 (neutral, only +1.5% — not significant)
        #   SOL: both OS →  0 (neutral, -0.8% — not significant)
        #        both OB →  0 (neutral, -0.2% — not significant)
        _asset = asset.upper()
        if os1h and not os4h and not ob4h:
            v = +2   # 1h oversold only: strong bounce setup across all assets
        elif ob1h and not ob4h and not os4h:
            v = -2   # 1h overbought only: strong fade setup across all assets
        elif os1h and os4h:
            v = -1 if _asset in ("BTC", "ETH") else 0
        elif ob1h and ob4h:
            v = +1 if _asset == "BTC" else 0
        else:
            v = 0
        rev_score += v
        details["rsi_1h"] = round(r1h, 1)
        details["rsi_4h_proxy"] = round(r4h, 1)
        details["rsi_multitf_vote"] = v
    except Exception:
        details["rsi_multitf_vote"] = None

    # 2. Stochastic 15m (+9.7% / -9.0% validated)
    # During sharp directional moves, stoch extremes signal continuation, not reversion.
    # A sharp drop pushes stoch oversold — flipping +2 → -2 marks bearish continuation.
    # A sharp rally pushes stoch overbought — flipping -2 → +2 marks bullish continuation.
    try:
        stk15 = _stoch_k(h15, l15, c15, 14)
        stk15_1h = stk15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        val = float(stk15_1h.iloc[-1])
        if val < 10:
            v = +2
        elif val < 20:
            v = +1
        elif val > 90:
            v = -2
        elif val > 80:
            v = -1
        else:
            v = 0
        if _sharp_down and v > 0:    # oversold during sharp drop → continuation down
            v = -v
        elif _sharp_up and v < 0:    # overbought during sharp rally → continuation up
            v = -v
        rev_score += v
        details["stoch_15m"] = round(val, 1)
        details["stoch_15m_vote"] = v
    except Exception:
        details["stoch_15m_vote"] = None

    # 3. Stochastic 1h (+7.9% / -7.7% validated)
    try:
        stk1h = _stoch_k(h1h, l1h, c1h, 14)
        val = float(stk1h.iloc[-1])
        if val < 10:
            v = +2
        elif val < 20:
            v = +1
        elif val > 90:
            v = -2
        elif val > 80:
            v = -1
        else:
            v = 0
        if _sharp_down and v > 0:    # oversold during sharp drop → continuation down
            v = -v
        elif _sharp_up and v < 0:    # overbought during sharp rally → continuation up
            v = -v
        rev_score += v
        details["stoch_1h"] = round(val, 1)
        details["stoch_1h_vote"] = v
    except Exception:
        details["stoch_1h_vote"] = None

    # 4. VWAP deviation (+8.4% below / -7.2% above validated)
    try:
        date_1m = df_1m.index.normalize()
        tpv = df_1m["close"] * df_1m["volume"]
        cum_tpv = tpv.groupby(date_1m).cumsum()
        cum_vol = df_1m["volume"].groupby(date_1m).cumsum()
        vwap_1m = cum_tpv / cum_vol.replace(0, np.nan)
        vwap_1h = vwap_1m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        vwap_dev = float((c1h.iloc[-1] - vwap_1h.iloc[-1]) / vwap_1h.iloc[-1]) \
                   if vwap_1h.iloc[-1] > 0 else 0.0
        if vwap_dev < -0.015:
            v = +2
        elif vwap_dev < -0.005:
            v = +1
        elif vwap_dev > 0.015:
            v = -2
        elif vwap_dev > 0.005:
            v = -1
        else:
            v = 0
        rev_score += v
        details["vwap_dev_pct"] = round(vwap_dev * 100, 4)
        details["vwap_vote"] = v
    except Exception:
        details["vwap_vote"] = None

    # 5. Donchian 15m (+3.8% / -5.4% validated)
    try:
        dc15 = _dc_pct(h15, l15, 20)
        dc15_1h = dc15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        val = float(dc15_1h.iloc[-1])
        if val < 0.10:
            v = +2
        elif val < 0.20:
            v = +1
        elif val > 0.90:
            v = -2
        elif val > 0.80:
            v = -1
        else:
            v = 0
        rev_score += v
        details["dc_15m"] = round(val, 3)
        details["dc_15m_vote"] = v
    except Exception:
        details["dc_15m_vote"] = None

    # 6. Keltner 15m (+5.1–7.5% / -7.0% validated)
    try:
        kc15, kc15_dn, kc15_up = _keltner_pct(h15, l15, c15, 20, 2)
        kc15_pct_1h = kc15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        c15_1h = c15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        kc15_dn_1h = kc15_dn.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        kc15_up_1h = kc15_up.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
        kp = float(kc15_pct_1h.iloc[-1])
        cl = float(c15_1h.iloc[-1])
        dn = float(kc15_dn_1h.iloc[-1])
        up = float(kc15_up_1h.iloc[-1])
        if cl < dn:
            v = +2
        elif 0.0 <= kp < 0.15:
            v = +1
        elif cl > up:
            v = -2
        elif 0.85 < kp <= 1.0:
            v = -1
        else:
            v = 0
        rev_score += v
        details["kc_15m"] = round(kp, 3)
        details["kc_15m_vote"] = v
    except Exception:
        details["kc_15m_vote"] = None

    # 7. Williams %R 1h (+6.2% oversold / -4.9% overbought validated)
    try:
        wpr1h = _wpr(h1h, l1h, c1h, 14)
        val = float(wpr1h.iloc[-1])
        v = int(val < -80) - int(val > -20)
        rev_score += v
        details["wpr_1h"] = round(val, 1)
        details["wpr_1h_vote"] = v
    except Exception:
        details["wpr_1h_vote"] = None

    # 8. Move z-score (+9.2% at z<-1.5 / -6.5% at z>1.5 validated)
    try:
        log_ret = np.log(c1h / c1h.shift(1))
        roll_vol = log_ret.rolling(24).std()
        z = float(log_ret.iloc[-1] / roll_vol.iloc[-1]) if roll_vol.iloc[-1] > 0 else 0.0
        if z < -2.0:
            v = +2
        elif z < -1.5:
            v = +1
        elif z > 2.0:
            v = -2
        elif z > 1.5:
            v = -1
        else:
            v = 0
        rev_score += v
        details["z_score"] = round(z, 3)
        details["z_score_vote"] = v
    except Exception:
        details["z_score_vote"] = None

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = trend_score + rev_score

    if combined > BULL_THRESHOLD:
        label = "bullish"
    elif combined < BEAR_THRESHOLD:
        label = "bearish"
    else:
        label = "neutral"

    details["trend_score"] = trend_score
    details["rev_score"] = rev_score
    details["combined_score"] = combined
    details["label"] = label

    return trend_score, rev_score, combined, label, details


def side_from_direction(label: str, offset_pct: float) -> Optional[str]:
    """
    Map direction label + contract offset to preferred bet side.

    Returns:
        "yes"  — take YES bet
        "no"   — take NO bet
        None   — skip (neutral + ATM, no structural advantage)
    """
    if label == "bullish":
        return "yes"
    elif label == "bearish":
        return "no"
    else:  # neutral → range-bound preference
        if offset_pct > 0.002:    # strike above spot → OTM NO wins if price stays below
            return "no"
        elif offset_pct < -0.002: # strike below spot → ITM YES wins if price stays above
            return "yes"
        else:
            return None           # ATM in neutral → no structural edge
