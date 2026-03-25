"""
Market structure module for detecting swing highs/lows on 15-minute candles
and classifying the prevailing structural trend bias.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List


MIN_CANDLES = 90
PIVOT_LOOKBACK = 3  # candles on each side required to confirm a swing point


def resample_to_15min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to 15-minute candles.

    Aggregation logic:
        open   = first 1-min open in the 15-min window  (bar open price)
        high   = max of all 1-min highs in the window   (true bar high)
        low    = min of all 1-min lows  in the window   (true bar low)
        close  = last 1-min close in the window         (bar close price)
        volume = sum of all 1-min volumes               (total activity)

    Each bar's timestamp is the window's open time (label='left').
    Incomplete bars at the boundary are retained; entirely empty bars
    (from data gaps) are dropped via dropna(subset=['close']).

    Args:
        df_1m: 1-minute OHLCV DataFrame with a DatetimeIndex.
               Required columns: open, high, low, close, volume (case-insensitive).

    Returns:
        15-minute OHLCV DataFrame with DatetimeIndex.
    """
    df = df_1m.copy()
    df.columns = df.columns.str.lower()
    return df.resample("15min", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])


@dataclass
class StructureResult:
    """Output of the market structure module."""

    structure_bias: int          # +1 bullish, -1 bearish, 0 neutral
    swing_highs: List[float]     # up to 3 most recent swing high prices
    swing_lows: List[float]      # up to 3 most recent swing low prices
    reason: str                  # plain-English explanation of the classification


def detect_market_structure(df: pd.DataFrame) -> StructureResult:
    """
    Detect swing pivot points and classify market structure from 15-minute OHLCV data.

    A swing high is a candle whose high is strictly greater than the highs of
    the 3 candles immediately before it and the 3 candles immediately after it.
    A swing low is symmetric: low strictly less than the 3 candles on each side.

    Only the 2 most recent swing highs and swing lows are used for classification.

    Structure classification rules:
        - Bullish (+1): last 2 swing highs strictly ascending AND last 2 swing
          lows strictly ascending (higher highs and higher lows).
        - Bearish (-1): last 2 swing highs strictly descending AND last 2 swing
          lows strictly descending (lower highs and lower lows).
        - Neutral (0): anything else, or fewer than 2 swing points detected.

    Args:
        df: 4-hour OHLCV DataFrame. Must have at least 90 candles (15 days).
            Required columns: open, high, low, close, volume (case-insensitive).

    Returns:
        StructureResult with structure_bias, swing_highs, swing_lows, and reason.

    Raises:
        ValueError: If fewer than 90 candles are provided.
    """
    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"DataFrame must have at least {MIN_CANDLES} candles, got {len(df)}."
        )

    df = df.copy()
    df.columns = df.columns.str.lower()
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)

    swing_high_prices: List[float] = []
    swing_low_prices: List[float] = []

    # Scan each candle that has at least PIVOT_LOOKBACK candles on both sides
    for i in range(PIVOT_LOOKBACK, n - PIVOT_LOOKBACK):
        left_highs = highs[i - PIVOT_LOOKBACK: i]
        right_highs = highs[i + 1: i + PIVOT_LOOKBACK + 1]

        # Swing high: strictly greater than all neighbours on both sides
        if highs[i] > max(left_highs) and highs[i] > max(right_highs):
            swing_high_prices.append(float(highs[i]))

        left_lows = lows[i - PIVOT_LOOKBACK: i]
        right_lows = lows[i + 1: i + PIVOT_LOOKBACK + 1]

        # Swing low: strictly less than all neighbours on both sides
        if lows[i] < min(left_lows) and lows[i] < min(right_lows):
            swing_low_prices.append(float(lows[i]))

    # Use only the 2 most recent pivots for classification
    recent_highs = swing_high_prices[-2:] if len(swing_high_prices) >= 2 else swing_high_prices
    recent_lows = swing_low_prices[-2:] if len(swing_low_prices) >= 2 else swing_low_prices

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return StructureResult(
            structure_bias=0,
            swing_highs=recent_highs,
            swing_lows=recent_lows,
            reason=(
                f"Neutral: fewer than 2 swing pivots detected "
                f"(highs={len(recent_highs)}, lows={len(recent_lows)})."
            ),
        )

    # Check whether swing points form a consistent ascending or descending sequence
    highs_ascending = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, 2))
    highs_descending = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, 2))
    lows_ascending = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, 2))
    lows_descending = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, 2))

    if highs_ascending and lows_ascending:
        bias = +1
        reason = (
            "Bullish: 2 of 2 swing highs ascending, 2 of 2 swing lows ascending "
            "on 15-min chart (higher highs and higher lows)."
        )
    elif highs_descending and lows_descending:
        bias = -1
        reason = (
            "Bearish: 2 of 2 swing highs descending, 2 of 2 swing lows descending "
            "on 15-min chart (lower highs and lower lows)."
        )
    else:
        bias = 0
        reason = (
            "Neutral: swing highs and lows do not form a consistent higher-high/"
            "higher-low or lower-high/lower-low sequence on 15-min chart."
        )

    return StructureResult(
        structure_bias=bias,
        swing_highs=recent_highs,
        swing_lows=recent_lows,
        reason=reason,
    )
