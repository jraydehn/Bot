"""
Market data module for computing realized volatility from 1-minute OHLCV data.

Scope: 1-hour expiry contracts on Kalshi. Volatility windows (30m, 60m, 120m)
are calibrated for this timeframe.

Vol estimator: Rogers-Satchell (2013-06-13 reform).
  RS = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)  per bar
  Rolling mean of RS variance → sqrt → per-minute vol.
  5-8× more efficient than close-to-close; drift-robust.
  Previous estimator: rolling std of log(close/close[-1]).

Seasonality: intraday_vol_seasonality_btc.json loaded at import.
  vol_multi is scaled by the hour-of-day multiplier before returning.
  Multipliers calibrated on 2024-01-01→2026-06-10 (1m BTC, RS estimator).
  Range: 0.60× (hour 10 UTC) to 1.59× (hour 14 UTC).
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
MIN_ROWS = 120


# Weights for multi-horizon realized vol blend.
# Favors short-term windows to react quickly to vol regime changes.
MULTI_VOL_WEIGHTS = {"vol_15m": 0.50, "vol_30m": 0.30, "vol_60m": 0.20}

# Intraday seasonality multipliers — loaded once at import.
# Keys are UTC hour strings "0"–"23"; values are float multipliers.
_SEASONALITY_PATH = Path(__file__).parent / "intraday_vol_seasonality_btc.json"
_SEASONALITY: dict = {}
try:
    with open(_SEASONALITY_PATH) as _f:
        _SEASONALITY = {int(k): float(v) for k, v in json.load(_f).items()}
except Exception:
    pass  # falls back to no adjustment (multiplier = 1.0)


@dataclass
class VolatilityResult:
    """Realized volatility per minute for four rolling windows."""

    vol_15m: float   # RS vol over trailing 15 minutes (per-minute units)
    vol_30m: float   # RS vol over trailing 30 minutes
    vol_60m: float   # RS vol over trailing 60 minutes
    vol_120m: float  # RS vol over trailing 120 minutes
    vol_multi: float # weighted blend × intraday seasonality multiplier
    log_returns: pd.Series  # close-to-close log returns (kept for compatibility)


def _rogers_satchell_var(df: pd.DataFrame) -> pd.Series:
    """
    Rogers-Satchell variance per 1-minute bar.

    RS = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)

    Drift-robust and uses the full intrabar OHLC path. Guaranteed >= 0
    for valid OHLC data; clipped for safety.
    """
    h = np.log(df["high"]  / df["open"])
    l = np.log(df["low"]   / df["open"])
    c = np.log(df["close"] / df["open"])
    return (h * (h - c) + l * (l - c)).clip(lower=0)


def compute_realized_volatility(df: pd.DataFrame) -> VolatilityResult:
    """
    Compute realized volatility from 1-minute OHLCV price data.

    Uses Rogers-Satchell variance averaged over rolling windows.
    Each value is expressed in per-minute units so it can be directly
    scaled to any horizon via sqrt(tau).

    vol_multi is further scaled by an intraday seasonality multiplier
    (hour-of-day UTC) calibrated on 2+ years of historical 1m BTC data.
    The multiplier corrects for the structural 2.65× vol range across
    the trading day (quietest: 10 UTC, busiest: 14 UTC).

    Args:
        df: DataFrame of 1-minute bars. Must contain columns:
            open, high, low, close, volume (case-insensitive).
            Must have at least 120 rows. Index should be DatetimeIndex
            or contain a timestamp column; used only to read current hour.

    Returns:
        VolatilityResult with per-minute realized volatility for 15m, 30m,
        60m and 120m windows, a seasonality-adjusted vol_multi, and the
        close-to-close log-return series for backward compatibility.

    Raises:
        ValueError: If fewer than 120 rows are provided or required
            columns are missing.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"DataFrame must have at least {MIN_ROWS} rows, got {len(df)}."
        )

    # Rogers-Satchell variance per bar
    rs_var = _rogers_satchell_var(df)

    # Rolling mean of RS variance → sqrt → per-minute vol
    vol_15m  = float(np.sqrt(rs_var.rolling(window=15).mean().iloc[-1]))
    vol_30m  = float(np.sqrt(rs_var.rolling(window=30).mean().iloc[-1]))
    vol_60m  = float(np.sqrt(rs_var.rolling(window=60).mean().iloc[-1]))
    vol_120m = float(np.sqrt(rs_var.rolling(window=120).mean().iloc[-1]))

    # Multi-horizon blend (same weights as before)
    vols    = {"vol_15m": vol_15m, "vol_30m": vol_30m, "vol_60m": vol_60m}
    w_total = sum(w for k, w in MULTI_VOL_WEIGHTS.items() if not np.isnan(vols[k]))
    vol_multi = (
        sum(MULTI_VOL_WEIGHTS[k] * vols[k] for k in vols if not np.isnan(vols[k])) / w_total
        if w_total > 0 else vol_60m
    )

    # Intraday seasonality adjustment — scale by current UTC hour multiplier.
    # Uses the last bar's timestamp if the index is datetime, else system clock.
    try:
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
            hour_utc = int(df.index[-1].hour)
        else:
            hour_utc = pd.Timestamp.now("UTC").hour
        seasonality_mult = _SEASONALITY.get(hour_utc, 1.0)
    except Exception:
        seasonality_mult = 1.0

    vol_multi *= seasonality_mult

    # Close-to-close log returns retained for backward compatibility
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    return VolatilityResult(
        vol_15m=vol_15m,
        vol_30m=vol_30m,
        vol_60m=vol_60m,
        vol_120m=vol_120m,
        vol_multi=vol_multi,
        log_returns=log_returns,
    )
