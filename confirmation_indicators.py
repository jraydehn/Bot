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

    confirmation_bias: int    # +1 bullish, -1 bearish, 0 neutral — from 7-indicator score (YES direction)
    no_bias: int              # +1 bullish, -1 bearish, 0 neutral — from 3-indicator no_score (NO direction)
    confirmation_score: int   # sum of up to 7 indicator scores, range -9 to +9
    no_score: int             # 3-indicator score for NO direction (EMA+RSI+Vol, -3 to +3)
    ema_alignment: str        # "bullish", "bearish", or "neutral"
    rsi_regime: str           # "bullish", "bearish", or "neutral"
    rsi_value: float          # current RSI reading
    volume_confirmed: bool    # True if latest volume exceeds its 20-period average
    ema_20_current: float     # current value of the 20-period EMA
    ema_50_current: float     # current value of the 50-period EMA
    mom_15m_score: int        # +1 if 15-min price change > 0, -1 if < 0, 0 if unavailable
    mom_60m_score: int        # +1 if 60-min price change > 0, -1 if < 0, 0 if unavailable
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

    RSI Regime: RSI(21) >= 55 is bullish, <= 45 is bearish, 45–55 is neutral.

    Volume Confirmation: latest candle volume must strictly exceed its
    20-period simple moving average.

    Confirmation bias is +1 when EMA is bullish, RSI is not bearish, and volume
    confirms. It is -1 when EMA is bearish and RSI is not bullish — volume is not
    required for bearish confirmation, as low volume in a bearish structure reflects
    absence of buying pressure and supports NO bets. Neutral RSI (45–55) does not
    block either direction.

    Args:
        df: 1-hour OHLCV DataFrame. Must have at least 60 candles.
            Required columns: open, high, low, close, volume (case-insensitive).
        hist_1m: Optional 1-minute OHLCV DataFrame with at least 62 candles.
            If provided, adds 15-minute and 60-minute price momentum scores.

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
    # Exponential moving averages on closing prices
    ema_20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_50 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    ema_20_current = float(ema_20.iloc[-1])
    ema_50_current = float(ema_50.iloc[-1])

    # Check the last EMA_CONFIRM_BARS candles for a consistent spread.
    # Three conditions determine alignment:
    #   Bullish: EMA-20 > EMA-50 (golden cross) AND price > EMA-20 (above both)
    #   Bearish: EMA-20 < EMA-50 (death cross) OR price < EMA-50 (below both EMAs)
    #   Neutral: golden cross but price trapped between the two EMAs
    # Price below both EMAs is bearish regardless of crossover status — both EMAs
    # act as overhead resistance when price is underneath them.
    last_ema20  = ema_20.iloc[-EMA_CONFIRM_BARS:].values
    last_ema50  = ema_50.iloc[-EMA_CONFIRM_BARS:].values
    last_close  = close.iloc[-EMA_CONFIRM_BARS:].values

    golden_cross  = all(last_ema20 > last_ema50)
    death_cross   = all(last_ema20 < last_ema50)
    price_above   = all(last_close > last_ema20)   # price above faster EMA (above both)
    price_below   = all(last_close < last_ema50)   # price below slower EMA (below both)

    if golden_cross and price_above:
        ema_alignment = "bullish"
    elif death_cross or price_below:
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
    # Directional volume: high volume on an up candle = bullish, high volume on a down
    # candle = bearish, low volume either way = neutral (0). This correctly handles
    # selling pressure (high volume + price down) as bearish rather than penalizing
    # any high-volume candle regardless of direction.
    vol_sma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
    high_vol = bool(float(volume.iloc[-1]) > float(vol_sma.iloc[-1]))
    price_up = bool(float(close.iloc[-1]) > float(close.iloc[-2]))
    volume_confirmed = high_vol  # kept for display: True if volume above average

    # --- MACD (12, 26, 9) ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_score = 1 if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) else -1

    # --- Rolling VWAP (24-period) ---
    if all(c in df.columns for c in ["high", "low"]):
        typical_price = (df["high"] + df["low"] + close) / 3
        tp_x_vol = typical_price * volume
        vwap = tp_x_vol.rolling(24).sum() / volume.rolling(24).sum()
        vwap_score = 1 if float(close.iloc[-1]) > float(vwap.iloc[-1]) else -1
    else:
        vwap_score = 0

    # --- Score each indicator ---
    ema_score = 1 if ema_alignment == "bullish" else (-1 if ema_alignment == "bearish" else 0)
    rsi_score  = 1 if rsi_regime == "bullish" else (-1 if rsi_regime == "bearish" else 0)
    if high_vol and price_up:
        vol_score = 1
    elif high_vol and not price_up:
        vol_score = -1
    else:
        vol_score = 0  # low volume = neutral signal

    # --- Short-term 1m momentum scores (if hist_1m provided) ---
    # Weight 2 (vs weight 1 for hourly indicators) so that falling short-term momentum
    # overrides a bullish hourly signal in Gate P. Retroactive analysis showed that
    # chg_15m > 0 AND chg_60m > 0 is the strongest predictor of Gate P YES wins.
    mom_15m_score = 0
    mom_60m_score = 0
    if hist_1m is not None and len(hist_1m) >= 62:
        hist_1m_c = hist_1m.copy()
        hist_1m_c.columns = hist_1m_c.columns.str.lower()
        c1m = hist_1m_c["close"]
        last_close_1m = float(c1m.iloc[-1])
        close_15_ago  = float(c1m.iloc[-16])   # ~15 min ago
        close_60_ago  = float(c1m.iloc[-61])   # ~60 min ago
        mom_15m = (last_close_1m - close_15_ago) / close_15_ago
        mom_60m = (last_close_1m - close_60_ago) / close_60_ago
        mom_15m_score = 2 if mom_15m > 0 else -2
        mom_60m_score = 2 if mom_60m > 0 else -2

    # --- 3-indicator score for NO direction (original model, unchanged) ---
    # EMA + RSI + Vol only — these were the working NO indicators from the baseline model.
    # Momentum scores are intentionally excluded from NO direction to avoid short-term
    # bounces flipping the NO signal during a broader bearish structure.
    no_score = ema_score + rsi_score + vol_score  # range -3 to +3

    if no_score >= 2:
        no_bias = +1
    elif no_score <= -2:
        no_bias = -1
    else:
        no_bias = 0

    # --- 7-indicator score for YES direction (MACD, VWAP, momentum added) ---
    # Hourly indicators: -5 to +5; momentum (weight 2 each): -4 to +4 → total -9 to +9
    confirmation_score = ema_score + rsi_score + vol_score + macd_score + vwap_score + mom_15m_score + mom_60m_score
    max_score = 5 + (4 if mom_15m_score != 0 or mom_60m_score != 0 else 0)

    # --- confirmation_bias: YES direction uses full 7-indicator score ---
    if confirmation_score >= 2:
        confirmation_bias = +1
        label = "Bullish"
    elif confirmation_score <= -2:
        confirmation_bias = -1
        label = "Bearish"
    else:
        confirmation_bias = 0
        label = "Neutral"

    mom_str = ""
    if mom_15m_score != 0 or mom_60m_score != 0:
        mom_str = ", Mom15m={:+d}, Mom60m={:+d}".format(mom_15m_score, mom_60m_score)
    tag = "confirmed" if confirmation_bias != 0 else "mixed signals"
    reason = (
        "{}: score={:+d}/{} (EMA={:+d}, RSI={:+d}, Vol={:+d}, MACD={:+d}, VWAP={:+d}{}) → {} (YES bias). "
        "NO bias from 3-indicator no_score={:+d} (no_bias={:+d}). "
        "20-EMA ({:.2f}) vs 50-EMA ({:.2f}); RSI={:.1f} ({}).".format(
            label, confirmation_score, max_score,
            ema_score, rsi_score, vol_score, macd_score, vwap_score, mom_str, tag,
            no_score, no_bias,
            ema_20_current, ema_50_current, rsi_value, rsi_regime
        )
    )

    return ConfirmationResult(
        confirmation_bias=confirmation_bias,
        no_bias=no_bias,
        confirmation_score=confirmation_score,
        no_score=no_score,
        ema_alignment=ema_alignment,
        rsi_regime=rsi_regime,
        rsi_value=rsi_value,
        volume_confirmed=volume_confirmed,
        ema_20_current=ema_20_current,
        ema_50_current=ema_50_current,
        mom_15m_score=mom_15m_score,
        mom_60m_score=mom_60m_score,
        reason=reason,
    )
