"""
Market data module for computing realized volatility from 1-minute OHLCV data.

Scope: 1-hour expiry contracts on Kalshi. Volatility windows (30m, 60m, 120m)
are calibrated for this timeframe.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
MIN_ROWS = 120


@dataclass
class VolatilityResult:
    """Realized volatility per minute for three rolling windows."""

    vol_30m: float   # std of log returns over trailing 30 minutes
    vol_60m: float   # std of log returns over trailing 60 minutes
    vol_120m: float  # std of log returns over trailing 120 minutes
    log_returns: pd.Series  # full series of per-minute log returns


def compute_realized_volatility(df: pd.DataFrame) -> VolatilityResult:
    """
    Compute realized volatility from 1-minute OHLCV price data.

    Realized volatility is the standard deviation of log returns over a
    rolling window. Each value is expressed in per-minute units so it can
    be directly scaled to any horizon via sqrt(tau).

    Args:
        df: DataFrame of 1-minute bars. Must contain columns:
            open, high, low, close, volume (case-insensitive).
            Must have at least 120 rows.

    Returns:
        VolatilityResult with per-minute realized volatility for 30m, 60m,
        and 120m windows, plus the full log-return series.

    Raises:
        ValueError: If fewer than 120 rows are provided or required
            columns are missing.
    """
    # Normalise column names to lowercase for consistent access
    df = df.copy()
    df.columns = df.columns.str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"DataFrame must have at least {MIN_ROWS} rows, got {len(df)}."
        )

    # Log return: ln(close_t / close_{t-1}) — measures proportional price change
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    # Rolling std of log returns gives realized volatility per minute
    # ddof=1 applies the sample correction (Bessel's correction), standard practice
    vol_30m = float(log_returns.rolling(window=30).std().iloc[-1])
    vol_60m = float(log_returns.rolling(window=60).std().iloc[-1])
    vol_120m = float(log_returns.rolling(window=120).std().iloc[-1])

    return VolatilityResult(
        vol_30m=vol_30m,
        vol_60m=vol_60m,
        vol_120m=vol_120m,
        log_returns=log_returns,
    )
