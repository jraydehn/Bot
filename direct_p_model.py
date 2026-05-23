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
from scipy.stats import norm

from composite_scorer import (
    _stoch_k, _rsi, _bb_pct, _keltner_pct, _wpr, _macd_cross, _vol_signal_4h, _dc_pct,
    compute_scores,
)

_MODEL_DIR = Path(__file__).parent / "reform_results"
_ASSETS_WITH_DIRECT = {"ETH"}   # BTC excluded — regressed on test; SOL reverted to legacy score_to_p_model (2026-05-04: direct model degraded live, pre-direct $25.84/t vs $12.55/t)
_FEATURE_COLUMNS = [
    "trend_stoch_4h", "trend_bb_4h", "trend_keltner_4h", "trend_wpr_4h",
    "trend_macd_4h", "trend_vol_4h",
    "rev_rsi_1h", "rev_rsi_4h", "rev_stoch_15m", "rev_stoch_1h",
    "rev_keltner_15m", "rev_dc_15m", "rev_wpr_1h", "rev_move_z",
    "offset_pct", "vol_pm",
    "z_strike", "composite_trend", "composite_rev", "trend_z_24h",
]

# Training-period vol_pm baselines (Jan 2025 – Jan 2026, 60-bar rolling 1m std).
# Used to correct OTM YES p_model when live vol has drifted from training distribution.
# Recompute if model is retrained on a different window.
_VOL_PM_TRAINING = {"ETH": 0.001023, "SOL": 0.001069}

_pipelines: dict = {}       # {asset: pipe} for YES models (direct_model_<ASSET>.pkl)
_no_pipelines: dict = {}    # {asset: pipe} for NO models  (direct_no_model_<ASSET>.pkl)


def _load_pipeline(asset: str) -> Optional[dict]:
    """Lazy-load YES pickle for an asset. Returns None if unavailable."""
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


def _load_no_pipeline(asset: str) -> Optional[dict]:
    """Lazy-load NO-specific pickle for an asset. Returns None if unavailable."""
    if asset in _no_pipelines:
        return _no_pipelines[asset]
    path = _MODEL_DIR / f"direct_no_model_{asset}.pkl"
    if not path.exists():
        _no_pipelines[asset] = None
        return None
    try:
        with open(path, "rb") as f:
            _no_pipelines[asset] = pickle.load(f)
        print(f"  [direct_p_model] Loaded NO model for {asset}")
    except Exception as e:
        print(f"  [direct_p_model] Failed to load {path.name}: {e}")
        _no_pipelines[asset] = None
    return _no_pipelines[asset]


def no_model_supported(asset: str) -> bool:
    """Check if a NO-specific model is available for this asset."""
    return _load_no_pipeline(asset) is not None


def asset_supported(asset: str) -> bool:
    """Check if direct model is enabled + trained for this asset."""
    if asset not in _ASSETS_WITH_DIRECT:
        return False
    return _load_pipeline(asset) is not None


def _latest_indicator_values(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                              df_15m: pd.DataFrame, df_1m: pd.DataFrame,
                              composite_trend: Optional[float] = None,
                              composite_rev: Optional[float] = None) -> dict:
    """Compute composite indicator values at the latest 1h bar."""
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
    lr_1h = np.log(df_1h["close"] / df_1h["close"].shift(1))
    roll_vol = lr_1h.rolling(24).std()
    mz_series = lr_1h / roll_vol.replace(0, float("nan"))
    vals["rev_move_z"] = float(mz_series.iloc[-1]) if not pd.isna(mz_series.iloc[-1]) else 0.0

    # trend_z_24h: 24h log return normalised by 24h rolling vol × sqrt(24)
    lr_24h = np.log(df_1h["close"] / df_1h["close"].shift(24))
    tz24 = lr_24h / (roll_vol.replace(0, float("nan")) * math.sqrt(24))
    vals["trend_z_24h"] = float(tz24.iloc[-1]) if not pd.isna(tz24.iloc[-1]) else 0.0

    # composite_trend / composite_rev: use caller-supplied values when available (avoids
    # recomputing VWAP over full 1m history on every scan); fall back to compute_scores.
    if composite_trend is not None and composite_rev is not None:
        vals["composite_trend"] = float(composite_trend)
        vals["composite_rev"]   = float(composite_rev)
    else:
        try:
            tr_s, rv_s = compute_scores(
                df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"],
                df_4h["close"], df_4h["high"], df_4h["low"], df_4h["volume"],
                df_15m["close"], df_15m["high"], df_15m["low"],
                df_1m["close"], df_1m["volume"],
                ts_1h=df_1h.index,
            )
            vals["composite_trend"] = float(tr_s.iloc[-1]) if len(tr_s) > 0 else 0.0
            vals["composite_rev"]   = float(rv_s.iloc[-1]) if len(rv_s) > 0 else 0.0
        except Exception:
            vals["composite_trend"] = 0.0
            vals["composite_rev"]   = 0.0

    return vals


def _vol_correction(offset_pct: float, vol_pm_current: float, vol_pm_training: float) -> float:
    """
    Multiplicative correction for OTM YES p_model when live vol differs from training vol.

    The direct ML model was trained on vol_pm avg ~0.00102 (ETH) / ~0.00107 (SOL).
    When live vol drifts lower, the model overestimates OTM strike-hit probabilities
    because the lognormal relationship between vol and tail probability is nonlinear.

    Correction = P(hit | current_vol) / P(hit | training_vol) via lognormal formula.
    Only applied to OTM YES (offset_pct > 0). ITM and NO sides are unaffected.
    Clipped to [0.20, 5.0] to prevent extreme adjustments on edge cases.

    vol_pm is per-minute std; 1h sigma = vol_pm * sqrt(60).
    """
    if offset_pct <= 0 or vol_pm_training <= 0 or vol_pm_current <= 0:
        return 1.0
    sigma_train = vol_pm_training * math.sqrt(60)
    sigma_curr  = vol_pm_current  * math.sqrt(60)
    log_off     = math.log(1.0 + offset_pct)
    p_train = max(float(1 - norm.cdf(log_off / sigma_train)), 1e-6)
    p_curr  = max(float(1 - norm.cdf(log_off / sigma_curr)),  1e-6)
    return float(np.clip(p_curr / p_train, 0.20, 5.0))


def compute_p_model_direct(asset: str, df_1m: pd.DataFrame, df_1h: pd.DataFrame,
                            df_4h: pd.DataFrame, df_15m: pd.DataFrame,
                            offset_pct: float,
                            composite_trend: Optional[float] = None,
                            composite_rev: Optional[float] = None) -> Optional[float]:
    """
    Predict P(close > strike at next-hour expiry) for a specific strike-offset.

    Computes vol_pm internally (60-bar rolling std of 1m log returns) to exactly
    match the training feature definition. Don't pass a vol from the runner's
    blended/scaled vol variables.

    composite_trend / composite_rev: pass the already-computed values from the runner
    to avoid recomputing VWAP over the full 1m history on every scan call.

    Returns None if: asset unsupported, model unavailable, or features contain NaN.
    Caller falls back to legacy score_to_p_model.
    """
    pipe = _load_pipeline(asset)
    if pipe is None:
        return None

    try:
        vals = _latest_indicator_values(df_1h, df_4h, df_15m, df_1m,
                                        composite_trend=composite_trend,
                                        composite_rev=composite_rev)
        # Match training: 60-min rolling std of 1m log returns
        lr_1m = np.log(df_1m["close"] / df_1m["close"].shift(1))
        vol_pm = float(lr_1m.rolling(60).std().iloc[-1])
    except Exception as e:
        print(f"  [direct_p_model] Feature extraction failed ({asset}): {e}")
        return None

    vals["offset_pct"] = float(offset_pct)
    vals["vol_pm"] = vol_pm if vol_pm and vol_pm > 0 else 0.0

    # z_strike: normalised distance to strike in 1h sigma units
    sigma_1h = vals["vol_pm"] * math.sqrt(60)
    if sigma_1h > 0 and offset_pct > -1.0:
        vals["z_strike"] = math.log(1.0 + offset_pct) / sigma_1h
    else:
        vals["z_strike"] = 0.0

    vec = np.array([[vals[c] for c in _FEATURE_COLUMNS]])
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        return None

    try:
        p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])

        # Split isotonic calibration: OTM and ITM bets have different hit-rate dynamics.
        # Use offset-specific calibrator when available, fall back to unified iso.
        if offset_pct > 0 and pipe.get("iso_otm") is not None:
            p_cal = float(pipe["iso_otm"].predict([p_raw])[0])
        elif offset_pct <= 0 and pipe.get("iso_itm") is not None:
            p_cal = float(pipe["iso_itm"].predict([p_raw])[0])
        else:
            p_cal = float(pipe["iso"].predict([p_raw])[0])

        # Second-stage Platt scaling (SOL only — ETH uses split calibration instead).
        if pipe.get("platt") is not None:
            try:
                p_clipped = float(np.clip(p_cal, 1e-6, 1 - 1e-6))
                log_o = math.log(p_clipped / (1 - p_clipped))
                p_cal = float(pipe["platt"].predict_proba([[log_o]])[0, 1])
            except Exception:
                pass

        # Vol correction: adjust OTM YES p_model for drift between training vol and live vol.
        vol_training = _VOL_PM_TRAINING.get(asset)
        if vol_training and offset_pct > 0:
            corr = _vol_correction(offset_pct, vol_pm, vol_training)
            if abs(corr - 1.0) > 0.05:   # only log when correction is meaningful
                print(f"  [direct_p_model] vol_corr={corr:.3f}  "
                      f"vol_pm={vol_pm:.5f} vs train={vol_training:.5f}  "
                      f"offset={offset_pct*100:.2f}%  "
                      f"p_cal: {p_cal:.4f} → {p_cal*corr:.4f}")
            p_cal = p_cal * corr

        return float(np.clip(p_cal, 0.01, 0.99))
    except Exception as e:
        print(f"  [direct_p_model] Inference failed ({asset}): {e}")
        return None


def compute_p_no_direct(asset: str, df_1m: pd.DataFrame, df_1h: pd.DataFrame,
                         df_4h: pd.DataFrame, df_15m: pd.DataFrame,
                         offset_pct: float,
                         composite_trend: Optional[float] = None,
                         composite_rev: Optional[float] = None) -> Optional[float]:
    """
    Predict P(NO resolves) = P(close <= strike at next-hour expiry).

    Uses the NO-specific model (direct_no_model_<ASSET>.pkl) trained with a
    flipped target and calibration fit against NO outcome distributions.
    Returns P(NO) directly — no 1-p_yes translation needed.

    No vol correction applied: the YES-side OTM vol correction is not relevant
    for the NO model which was trained on actual NO outcome rates.

    Returns None if: NO model unavailable, features contain NaN.
    Caller falls back to 1 - compute_p_model_direct() in that case.
    """
    pipe = _load_no_pipeline(asset)
    if pipe is None:
        return None

    try:
        vals = _latest_indicator_values(df_1h, df_4h, df_15m, df_1m,
                                        composite_trend=composite_trend,
                                        composite_rev=composite_rev)
        lr_1m = np.log(df_1m["close"] / df_1m["close"].shift(1))
        vol_pm = float(lr_1m.rolling(60).std().iloc[-1])
    except Exception as e:
        print(f"  [direct_p_model] NO feature extraction failed ({asset}): {e}")
        return None

    vals["offset_pct"] = float(offset_pct)
    vals["vol_pm"]     = vol_pm if vol_pm and vol_pm > 0 else 0.0

    sigma_1h = vals["vol_pm"] * math.sqrt(60)
    if sigma_1h > 0 and offset_pct > -1.0:
        vals["z_strike"] = math.log(1.0 + offset_pct) / sigma_1h
    else:
        vals["z_strike"] = 0.0

    features = pipe.get("features", _FEATURE_COLUMNS)
    vec = np.array([[vals[c] for c in features]])
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        return None

    try:
        p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])

        # Split isotonic: positive offset = YES OTM / NO ITM; negative = YES ITM / NO OTM
        if offset_pct > 0 and pipe.get("iso_pos") is not None:
            p_cal = float(pipe["iso_pos"].predict([p_raw])[0])
        elif offset_pct <= 0 and pipe.get("iso_neg") is not None:
            p_cal = float(pipe["iso_neg"].predict([p_raw])[0])
        else:
            p_cal = float(pipe["iso"].predict([p_raw])[0])

        if pipe.get("platt") is not None:
            try:
                p_clipped = float(np.clip(p_cal, 1e-6, 1 - 1e-6))
                log_o = math.log(p_clipped / (1 - p_clipped))
                p_cal = float(pipe["platt"].predict_proba([[log_o]])[0, 1])
            except Exception:
                pass

        return float(np.clip(p_cal, 0.01, 0.99))
    except Exception as e:
        print(f"  [direct_p_model] NO inference failed ({asset}): {e}")
        return None
