"""
Confirmation indicators module: EMA alignment, RSI regime, and directional volume
on 1-hour candles.

Each indicator contributes -1, 0, or +1 to a combined score (-3 to +3).
Directional volume: high volume on an up candle = +1, high volume on a down
candle = -1, low volume either way = 0.

confirmation_score and no_score are identical (same 3 indicators, same direction).
Both are retained in the output for CSV compatibility.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


MIN_CANDLES = 60
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 21
VOLUME_MA_PERIOD = 20
EMA_CONFIRM_BARS = 3


@dataclass
class ConfirmationResult:
    confirmation_bias: int    # +1 bullish, -1 bearish, 0 neutral (score >= 2 / <= -2)
    no_bias: int              # same as confirmation_bias (identical 3-indicator model)
    confirmation_score: int   # sum of 3 indicator scores, range -3 to +3
    no_score: int             # same as confirmation_score
    ema_alignment: str        # "bullish", "bearish", or "neutral"
    rsi_regime: str           # "bullish", "bearish", or "neutral"
    rsi_value: float
    volume_confirmed: bool    # True if latest volume exceeds 20-period SMA
    ema_20_current: float
    ema_50_current: float
    reason: str


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_confirmation(df: pd.DataFrame, hist_1m: pd.DataFrame = None) -> ConfirmationResult:
    """
    Compute 3-indicator confirmation score from 1-hour bars.

    Indicators:
      EMA:    20-EMA above 50-EMA for 3 consecutive bars = +1; below = -1; else 0
      RSI:    >= 55 = +1; <= 45 = -1; 45-55 = 0
      Volume: high volume + price up = +1; high volume + price down = -1; low vol = 0

    confirmation_bias / no_bias = +1 if score >= 2, -1 if score <= -2, else 0.

    hist_1m is accepted but ignored (kept for call-site compatibility).
    """
    if len(df) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} candles, got {len(df)}.")

    df = df.copy()
    df.columns = df.columns.str.lower()
    close  = df["close"]
    volume = df["volume"]

    # --- EMA ---
    ema_20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_50 = close.ewm(span=EMA_SLOW, adjust=False).mean()
    ema_20_current = float(ema_20.iloc[-1])
    ema_50_current = float(ema_50.iloc[-1])

    last_ema20 = ema_20.iloc[-EMA_CONFIRM_BARS:].values
    last_ema50 = ema_50.iloc[-EMA_CONFIRM_BARS:].values

    if all(last_ema20 > last_ema50):
        ema_alignment = "bullish"
        ema_score = 1
    elif all(last_ema20 < last_ema50):
        ema_alignment = "bearish"
        ema_score = -1
    else:
        ema_alignment = "neutral"
        ema_score = 0

    # --- RSI ---
    rsi_series = _compute_rsi(close, RSI_PERIOD)
    rsi_value  = float(rsi_series.iloc[-1])
    if rsi_value >= 55:
        rsi_regime = "bullish"
        rsi_score  = 1
    elif rsi_value <= 45:
        rsi_regime = "bearish"
        rsi_score  = -1
    else:
        rsi_regime = "neutral"
        rsi_score  = 0

    # --- Directional volume ---
    vol_sma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
    high_vol = bool(float(volume.iloc[-1]) > float(vol_sma.iloc[-1]))
    price_up = bool(float(close.iloc[-1]) > float(close.iloc[-2]))
    volume_confirmed = high_vol

    if high_vol and price_up:
        vol_score = 1
    elif high_vol and not price_up:
        vol_score = -1
    else:
        vol_score = 0

    # --- Combined score ---
    score = ema_score + rsi_score + vol_score  # -3 to +3

    if score >= 2:
        bias = 1
        label = "Bullish"
    elif score <= -2:
        bias = -1
        label = "Bearish"
    else:
        bias = 0
        label = "Neutral"

    reason = (
        f"{label}: score={score:+d} (EMA={ema_score:+d}, RSI={rsi_score:+d}, Vol={vol_score:+d}). "
        f"20-EMA={ema_20_current:.2f} vs 50-EMA={ema_50_current:.2f}; "
        f"RSI={rsi_value:.1f} ({rsi_regime}); "
        f"vol {'above' if high_vol else 'below'} avg, price {'up' if price_up else 'down'}."
    )

    return ConfirmationResult(
        confirmation_bias=bias,
        no_bias=bias,
        confirmation_score=score,
        no_score=score,
        ema_alignment=ema_alignment,
        rsi_regime=rsi_regime,
        rsi_value=rsi_value,
        volume_confirmed=volume_confirmed,
        ema_20_current=ema_20_current,
        ema_50_current=ema_50_current,
        reason=reason,
    )
