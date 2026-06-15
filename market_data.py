"""
Market data module for computing realized volatility from 1-minute OHLCV data.

Scope: 1-hour expiry contracts on Kalshi. Volatility windows (15m, 30m, 60m, 120m)
are calibrated for this timeframe.

Vol estimator: close-to-close squared log-returns with EWMA smoothing.

Why not Rogers-Satchell (reverted 2026-06-13):
  Binance US data has 65-80% flat bars (H=L=O=C) per minute due to low
  exchange liquidity. Range estimators (RS, Garman-Klass) assume a
  continuously observed intrabar path. On sparse data the observed H/L
  are drawn from far fewer ticks than the true extremes — RS systematically
  underestimates vol by ~5× on this data. Close-to-close only needs the
  close price, which is reliably populated regardless of tick sparsity.

Seasonality: intraday_vol_seasonality_{asset}.json loaded at import for
  each asset (BTC/ETH/SOL). vol_multi is scaled by the UTC hour-of-day
  multiplier before returning. Multipliers calibrated from C2C variance
  on 2024-01-01→2026-06-10 1m data. Ranges:
    BTC: 0.78× (10 UTC) to 1.35× (14 UTC)
    ETH: 0.77× (10 UTC) to 1.41× (14 UTC)
    SOL: 0.81× (10 UTC) to 1.36× (14 UTC)
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
MIN_ROWS = 120

# Weights for multi-horizon realized vol blend.
MULTI_VOL_WEIGHTS = {"vol_15m": 0.50, "vol_30m": 0.30, "vol_60m": 0.20}

# EWMA half-life for smoothing squared returns before vol_multi blend.
# Reduces noise from single violent bars without introducing meaningful lag.
_EWMA_HALFLIFE = 30

# Load all seasonality JSONs once at import.
_SEASONALITY: dict[str, dict[int, float]] = {}
_SEASON_DIR = Path(__file__).parent
for _asset in ("btc", "eth", "sol"):
    _path = _SEASON_DIR / f"intraday_vol_seasonality_{_asset}.json"
    try:
        with open(_path) as _f:
            _SEASONALITY[_asset.upper()] = {int(k): float(v) for k, v in json.load(_f).items()}
    except Exception:
        _SEASONALITY[_asset.upper()] = {}   # falls back to no adjustment


@dataclass
class VolatilityResult:
    """Realized volatility per minute for four rolling windows."""

    vol_15m: float    # C2C vol over trailing 15 minutes (per-minute units)
    vol_30m: float    # C2C vol over trailing 30 minutes
    vol_60m: float    # C2C vol over trailing 60 minutes
    vol_120m: float   # C2C vol over trailing 120 minutes
    vol_multi: float  # EWMA-smoothed blend × intraday seasonality multiplier
    log_returns: pd.Series  # close-to-close log returns (backward compatibility)


def compute_realized_volatility(df: pd.DataFrame, asset: str = "BTC") -> VolatilityResult:
    """
    Compute realized volatility from 1-minute OHLCV price data.

    Uses close-to-close squared log-returns averaged over rolling windows.
    Each value is expressed in per-minute units so it can be scaled to any
    horizon via sqrt(tau).

    vol_multi uses an EWMA (half-life=30) over squared returns for
    smoothing, then applies an intraday seasonality multiplier calibrated
    on 2+ years of 1m data per asset. The multiplier corrects for the
    structural ~1.7× vol range across the trading day.

    Args:
        df:    DataFrame of 1-minute bars. Must contain open, high, low,
               close, volume (case-insensitive). At least 120 rows.
               Index should be DatetimeIndex in UTC; used to read the
               current UTC hour for seasonality.
        asset: "BTC", "ETH", or "SOL". Selects the seasonality table.

    Returns:
        VolatilityResult with per-minute vol for 15m, 30m, 60m, 120m
        windows, a seasonality-adjusted EWMA vol_multi, and the
        close-to-close log-return series for backward compatibility.

    Raises:
        ValueError: If fewer than 120 rows or required columns are missing.
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

    log_returns = np.log(df["close"] / df["close"].shift(1))
    sq_returns  = log_returns ** 2

    # Rolling realized vol for each window (sqrt of mean squared return)
    vol_15m  = float(np.sqrt(sq_returns.rolling(15,  min_periods=5).mean().iloc[-1]))
    vol_30m  = float(np.sqrt(sq_returns.rolling(30,  min_periods=10).mean().iloc[-1]))
    vol_60m  = float(np.sqrt(sq_returns.rolling(60,  min_periods=20).mean().iloc[-1]))
    vol_120m = float(np.sqrt(sq_returns.rolling(120, min_periods=40).mean().iloc[-1]))

    # EWMA variance for vol_multi — exponentially weighted average of squared returns.
    # halflife=30 ≈ 30-bar effective window; de-noises single violent bars
    # without accumulating two layers of smoothing.
    ewma_var  = float(sq_returns.ewm(halflife=_EWMA_HALFLIFE, min_periods=10).mean().iloc[-1])
    vol_multi = float(np.sqrt(max(ewma_var, 0.0)))

    # Intraday seasonality: scale by current UTC hour multiplier
    try:
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
            hour_utc = int(df.index[-1].hour)
        else:
            hour_utc = pd.Timestamp.now("UTC").hour
        season_table = _SEASONALITY.get(asset.upper(), {})
        seasonality_mult = season_table.get(hour_utc, 1.0)
    except Exception:
        seasonality_mult = 1.0

    vol_multi *= seasonality_mult

    return VolatilityResult(
        vol_15m=vol_15m,
        vol_30m=vol_30m,
        vol_60m=vol_60m,
        vol_120m=vol_120m,
        vol_multi=vol_multi,
        log_returns=log_returns.dropna(),
    )
