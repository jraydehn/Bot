"""
direct_p_model.py — Direct strike-hit prediction for ETH and SOL.

Replaces score_to_p_model()'s log-normal + drift formula with a HistGradientBoosting
model trained on (indicator values, offset_pct, vol_pm) → P(strike crossed next hour).

Used for ETH and SOL only (validated on test PnL). BTC remains on legacy score_to_p_model.

Models trained in direct_strike_hit_model.py. Pickle files loaded lazily at first call.
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from composite_scorer import (
    _stoch_k, _rsi, _bb_pct, _keltner_pct, _wpr, _macd_cross, _vol_signal_4h, _dc_pct,
)

_MODEL_DIR = Path(__file__).parent / "reform_results"
_ASSETS_WITH_DIRECT = {"ETH", "SOL"}   # BTC excluded — regressed on test
_FEATURE_COLUMNS = [
    "trend_stoch_4h", "trend_bb_4h", "trend_keltner_4h", "trend_wpr_4h",
    "trend_macd_4h", "trend_vol_4h",
    "rev_rsi_1h", "rev_rsi_4h", "rev_stoch_15m", "rev_stoch_1h",
    "rev_keltner_15m", "rev_dc_15m", "rev_wpr_1h", "rev_move_z",
    "offset_pct", "vol_pm",
]

_pipelines: dict = {}   # {asset: {"clf", "iso", "features"}}


def _load_pipeline(asset: str) -> Optional[dict]:
    """Lazy-load pickle for an asset. Returns None if unavailable."""
    if asset in _pipelines:
        return _pipelines[asset]
    path = _MODEL_DIR / f"direct_model_{asset}.pkl"
    if not path.exists():
        _pipelines[asset] = None
        return None
    try:
        with open(path, "rb") as f:
            _pipelines[asset] = pickle.load(f)
    except Exception as e:
        print(f"  [direct_p_model] Failed to load {path.name}: {e}")
        _pipelines[asset] = None
    return _pipelines[asset]


def asset_supported(asset: str) -> bool:
    """Check if direct model is enabled + trained for this asset."""
    if asset not in _ASSETS_WITH_DIRECT:
        return False
    return _load_pipeline(asset) is not None


def _latest_indicator_values(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                              df_15m: pd.DataFrame) -> dict:
    """Compute the 14 composite indicator continuous values at the latest 1h bar."""
    vals: dict = {}

    # 4h trend
    vals["trend_stoch_4h"]    = float(_stoch_k(df_4h["high"], df_4h["low"], df_4h["close"], 14).iloc[-1])
    vals["trend_bb_4h"]       = float(_bb_pct(df_4h["high"], df_4h["low"], df_4h["close"], 20).iloc[-1])
    kc4_pct, _, _             = _keltner_pct(df_4h["high"], df_4h["low"], df_4h["close"], 20, 2)
    vals["trend_keltner_4h"]  = float(kc4_pct.iloc[-1])
    vals["trend_wpr_4h"]      = float(_wpr(df_4h["high"], df_4h["low"], df_4h["close"], 14).iloc[-1])
    macd_state = _macd_cross(df_4h["close"]).iloc[-1]
    vals["trend_macd_4h"]     = {"crossed_up":2, "up_lag":1, "none":0, "down_lag":-1, "crossed_down":-2}.get(macd_state, 0)
    vsig = _vol_signal_4h(df_4h["close"], df_4h["volume"]).iloc[-1]
    vals["trend_vol_4h"]      = {"high_vol_up":1, "avg":0, "low_vol":0, "high_vol_down":-1}.get(vsig, 0)

    # Reversion (1h/15m)
    vals["rev_rsi_1h"]    = float(_rsi(df_1h["close"], 14).iloc[-1])
    vals["rev_rsi_4h"]    = float(_rsi(df_4h["close"], 14).iloc[-1])
    vals["rev_stoch_15m"] = float(_stoch_k(df_15m["high"], df_15m["low"], df_15m["close"], 14).iloc[-1])
    vals["rev_stoch_1h"]  = float(_stoch_k(df_1h["high"], df_1h["low"], df_1h["close"], 14).iloc[-1])
    kc15_pct, _, _        = _keltner_pct(df_15m["high"], df_15m["low"], df_15m["close"], 20, 2)
    vals["rev_keltner_15m"] = float(kc15_pct.iloc[-1])
    vals["rev_dc_15m"]    = float(_dc_pct(df_15m["high"], df_15m["low"], df_15m["close"], 20).iloc[-1])
    vals["rev_wpr_1h"]    = float(_wpr(df_1h["high"], df_1h["low"], df_1h["close"], 14).iloc[-1])

    # Move z-score (1h return / 24h rolling std)
    lr = np.log(df_1h["close"] / df_1h["close"].shift(1))
    roll_vol = lr.rolling(24).std()
    mz_series = lr / roll_vol.replace(0, float("nan"))
    vals["rev_move_z"] = float(mz_series.iloc[-1]) if not pd.isna(mz_series.iloc[-1]) else 0.0

    return vals


def compute_p_model_direct(asset: str, df_1m: pd.DataFrame, df_1h: pd.DataFrame,
                            df_4h: pd.DataFrame, df_15m: pd.DataFrame,
                            offset_pct: float) -> Optional[float]:
    """
    Predict P(close > strike at next-hour expiry) for a specific strike-offset.

    Computes vol_pm internally (60-bar rolling std of 1m log returns) to exactly
    match the training feature definition. Don't pass a vol from the runner's
    blended/scaled vol variables.

    Returns None if: asset unsupported, model unavailable, or features contain NaN.
    Caller falls back to legacy score_to_p_model.
    """
    pipe = _load_pipeline(asset)
    if pipe is None:
        return None

    try:
        vals = _latest_indicator_values(df_1h, df_4h, df_15m)
        # Match training: 60-min rolling std of 1m log returns
        lr_1m = np.log(df_1m["close"] / df_1m["close"].shift(1))
        vol_pm = float(lr_1m.rolling(60).std().iloc[-1])
    except Exception as e:
        print(f"  [direct_p_model] Feature extraction failed ({asset}): {e}")
        return None

    vals["offset_pct"] = float(offset_pct)
    vals["vol_pm"] = vol_pm if vol_pm and vol_pm > 0 else 0.0

    vec = np.array([[vals[c] for c in _FEATURE_COLUMNS]])
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        return None

    try:
        p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])
        p_cal = float(pipe["iso"].predict([p_raw])[0])
        return float(np.clip(p_cal, 0.01, 0.99))
    except Exception as e:
        print(f"  [direct_p_model] Inference failed ({asset}): {e}")
        return None
