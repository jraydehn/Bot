"""
Confirmation indicators module: EMA alignment, RSI regime, and volume confirmation
on 1-hour candles. All three must agree for a directional confirmation signal.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


MIN_CANDLES = 60
EMA_FAST = 20            # fast EMA period for trend alignment
EMA_SLOW = 50            # slow EMA period for trend alignment
RSI_PERIOD = 21          # RSI lookback period
VOLUME_MA_PERIOD = 20    # simple moving average period for volume baseline
EMA_CONFIRM_BARS = 3     # consecutive bars the EMA spread must hold to confirm


@dataclass
class ConfirmationResult:
    """Output of the confirmation indicators module."""

    confirmation_bias: int    # +1 bullish, -1 bearish, 0 neutral
    confirmation_score: int   # sum of 3 indicator scores, range -3 to +3
    no_score: int             # same as confirmation_score (3-indicator model)
    ema_alignment: str        # "bullish", "bearish", or "neutral"
    rsi_regime: str           # "bullish", "bearish", or "neutral"
    rsi_value: float          # current RSI reading
    volume_confirmed: bool    # True if latest volume exceeds its 20-period average
    ema_20_current: float     # current value of the 20-period EMA
    ema_50_current: float     # current value of the 50-period EMA
    reason: str               # plain-English explanation of the classification


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's exponential smoothing method.

    Uses EWM with alpha=1/period, which approximates Wilder's smoothing
    and is consistent with most charting platforms.

    Args:
        series: Closing price series.
        period: Lookback period for RSI (typically 14 or 21).

    Returns:
        RSI series on the same index as the input.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing: exponential weighted mean with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero when there are no losing periods
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_confirmation(df: pd.DataFrame, hist_1m: pd.DataFrame = None) -> ConfirmationResult:
    """
    Compute EMA alignment, RSI regime, and volume confirmation from 1-hour bars.

    EMA Alignment: 20-EMA must have been strictly above (or below) 50-EMA
    for at least the last 3 consecutive candles.

    RSI Regime: RSI(21) >= 55 is bullish, <= 45 is bearish, 45-55 is neutral.

    Volume Confirmation: latest candle volume must strictly exceed its
    20-period simple moving average.

    Confirmation bias is +1 when EMA is bullish, RSI is not bearish, and volume
    confirms. It is -1 when EMA is bearish and RSI is not bullish — volume is not
    required for bearish confirmation, as low volume in a bearish structure reflects
    absence of buying pressure and supports NO bets. Neutral RSI (45-55) does not
    block either direction.

    confirmation_score = EMA(±1) + RSI(±1) + Vol(±1), range -3 to +3.
    no_score is identical to confirmation_score in this 3-indicator model.

    Args:
        df: 1-hour OHLCV DataFrame. Must have at least 60 candles.
            Required columns: open, high, low, close, volume (case-insensitive).
        hist_1m: Unused in this model; accepted for API compatibility.

    Returns:
        ConfirmationResult dataclass with all indicator values and combined bias.

    Raises:
        ValueError: If fewer than 60 candles are provided.
    """
    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"DataFrame must have at least {MIN_CANDLES} candles, got {len(df)}."
        )

    df = df.copy()
    df.columns = df.columns.str.lower()
    close = df["close"]
    volume = df["volume"]

    # --- EMA Alignment ---
    ema_20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_50 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    ema_20_current = float(ema_20.iloc[-1])
    ema_50_current = float(ema_50.iloc[-1])

    last_ema20 = ema_20.iloc[-EMA_CONFIRM_BARS:].values
    last_ema50 = ema_50.iloc[-EMA_CONFIRM_BARS:].values

    if all(last_ema20 > last_ema50):
        ema_alignment = "bullish"
    elif all(last_ema20 < last_ema50):
        ema_alignment = "bearish"
    else:
        ema_alignment = "neutral"

    # --- RSI Regime ---
    rsi_series = _compute_rsi(close, RSI_PERIOD)
    rsi_value = float(rsi_series.iloc[-1])

    if rsi_value >= 55:
        rsi_regime = "bullish"
    elif rsi_value <= 45:
        rsi_regime = "bearish"
    else:
        rsi_regime = "neutral"

    # --- Volume Confirmation ---
    vol_sma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
    volume_confirmed = bool(float(volume.iloc[-1]) > float(vol_sma.iloc[-1]))

    # --- Score each indicator (-1 / 0 / +1) ---
    ema_score = 1 if ema_alignment == "bullish" else (-1 if ema_alignment == "bearish" else 0)
    rsi_score  = 1 if rsi_regime == "bullish" else (-1 if rsi_regime == "bearish" else 0)
    vol_score  = 1 if volume_confirmed else -1

    confirmation_score = ema_score + rsi_score + vol_score  # range -3 to +3
    no_score = confirmation_score                            # same model for both directions

    # --- Combine into confirmation_bias ---
    if ema_alignment == "bullish" and rsi_regime != "bearish" and volume_confirmed:
        confirmation_bias = +1
        label = "Bullish"
        detail = (
            f"EMA bullish + RSI not bearish + volume confirmed → confirmed. "
            f"20-EMA ({ema_20_current:.2f}) above 50-EMA ({ema_50_current:.2f}) "
            f"for last {EMA_CONFIRM_BARS} bars; RSI={rsi_value:.1f} ({rsi_regime})."
        )
    elif ema_alignment == "bearish" and rsi_regime != "bullish":
        confirmation_bias = -1
        label = "Bearish"
        detail = (
            f"EMA bearish + RSI not bullish → confirmed (volume not required "
            f"for NO trades). 20-EMA ({ema_20_current:.2f}) below 50-EMA "
            f"({ema_50_current:.2f}) for last {EMA_CONFIRM_BARS} bars; "
            f"RSI={rsi_value:.1f} ({rsi_regime})."
        )
    else:
        confirmation_bias = 0
        label = "Neutral"
        parts = []
        if ema_alignment == "neutral":
            parts.append("EMA alignment is neutral (crossed within last 3 bars)")
        elif ema_alignment == "bullish" and rsi_regime != "bullish":
            parts.append(f"EMA bullish but RSI={rsi_value:.1f} is not bullish")
        elif ema_alignment == "bearish" and rsi_regime != "bearish":
            parts.append(f"EMA bearish but RSI={rsi_value:.1f} is not bearish")
        if not volume_confirmed:
            parts.append("volume is below 20-period average")
        detail = ("; ".join(parts) if parts else "mixed signals") + "."

    reason = (
        "{}: score={:+d}/3 (EMA={:+d}, RSI={:+d}, Vol={:+d}) → {}. "
        "20-EMA ({:.2f}) vs 50-EMA ({:.2f}); RSI={:.1f} ({}).".format(
            label, confirmation_score,
            ema_score, rsi_score, vol_score, detail,
            ema_20_current, ema_50_current, rsi_value, rsi_regime,
        )
    )

    return ConfirmationResult(
        confirmation_bias=confirmation_bias,
        confirmation_score=confirmation_score,
        no_score=no_score,
        ema_alignment=ema_alignment,
        rsi_regime=rsi_regime,
        rsi_value=rsi_value,
        volume_confirmed=volume_confirmed,
        ema_20_current=ema_20_current,
        ema_50_current=ema_50_current,
        reason=reason,
    )
