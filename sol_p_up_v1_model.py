"""
sol_p_up_v1_model.py — SOL honest p_up model, SHADOW-ONLY inference.

Model: reform_results/sol_pup_rebuild_20260706/sol_p_up_v1_20260706.pkl
(16 features = A ONLY: SOL's own lag-corrected price/technicals). Unlike
BOTH BTC (A+B+M+Cs) and ETH (A+C), NONE of cross-asset (B), alt-coin/
dominance/session (C), regime/Donchian (R), or intra-hour 1m microstructure
(M) cleared significance for SOL -- all p_boot>=0.30 vs the AB baseline,
and ABM was confidently WORSE (p=0.99). SOL's honest signal is entirely
self-contained: no BTC/ETH/alt-coin fetch needed at all, unlike the other
two models. Honest walk-forward OOS: AUC 0.5304, weekly-mean 0.5398,
IC t=15.2, positive every year 2021-2026. Output range ~[0.434, 0.553].
Beats the existing leaky sol_p_up_v2_new.pkl's honest AUC~0.503.

Feature construction mirrors reform_results/sol_pup_rebuild_20260706/
s1_build_dataset.py exactly:
  * every feature from COMPLETED bars only; row T = last completed 1h
    bar, decision time T+1h
  * 4h bars = plain resample of 1h (no lookahead: only uses bars <= T)
  * label convention: direction of bar T+1 close vs bar T close

Data sources: SOL 1h only, via live_signal.fetch_recent_candles (SOL IS
in ASSET_CONFIG so this is safe).

Graceful degradation: any feature-group failure -> NaN for that group
(LGBM handles NaN natively); load/inference failure -> None. Never
raises into the scan loop; never read by any decision path.
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import train_btc_p_up_v2 as _T  # generic indicator helpers, asset-agnostic

_MODEL_PATH = _PROJ / "reform_results" / "sol_pup_rebuild_20260706" / "sol_p_up_v1_20260706.pkl"
_CACHE: "dict | None | str" = "unloaded"
NAN = float("nan")

_last = {"bar_ts": None, "p": None}


def _load() -> "dict | None":
    global _CACHE
    if _CACHE != "unloaded":
        return _CACHE
    try:
        with open(_MODEL_PATH, "rb") as f:
            _CACHE = pickle.load(f)
    except Exception:
        _CACHE = None
    return _CACHE


def _fetch_asset_configured(interval: str, bars: int, asset: str) -> "pd.DataFrame | None":
    """Safe ONLY for assets actually in live_signal.ASSET_CONFIG (BTC/ETH/SOL)."""
    try:
        from live_signal import fetch_recent_candles
        return fetch_recent_candles(interval, lookback_bars=bars, asset=asset)
    except Exception:
        return None


# ── feature assembly (exact port of sol_pup_rebuild_20260706/s1_build_dataset.py) ──

def _assemble(T: pd.Timestamp, b1: pd.DataFrame) -> dict:
    """Feature dict for bar T. b1 = completed SOL 1h bars ending at T."""
    feats = {}
    c1h = b1["close"]
    h1h = b1["high"]
    l1h = b1["low"]
    v1h = b1["volume"]

    # -- A: SOL 1h indicators (exact port of s1_build_dataset.py) ------------
    lr = np.log(c1h / c1h.shift(1))
    try:
        delta1 = c1h.diff()
        gain1 = delta1.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss1 = (-delta1.clip(upper=0)).ewm(span=14, adjust=False).mean()
        feats["rsi_14"] = float((100 - 100 / (1 + gain1 / loss1.replace(0, np.nan))).iloc[-1])
        feats["macd_hist_1h"] = float(_T.macd_hist_series(c1h).iloc[-1])
        feats["bb_pct"] = float(_T.bb_pct_series(c1h).iloc[-1])
        feats["ema50_dist"] = float(_T.ema50_dist_series(c1h).iloc[-1])
        feats["rvol_1h"] = float((lr.rolling(24).std() / lr.rolling(168).std().replace(0, np.nan)).iloc[-1])
        ll14 = l1h.rolling(14).min(); hh14 = h1h.rolling(14).max()
        feats["stoch_k"] = float((((c1h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100).iloc[-1])
        ema9, ema21, ema55 = c1h.ewm(span=9, adjust=False).mean(), c1h.ewm(span=21, adjust=False).mean(), c1h.ewm(span=55, adjust=False).mean()
        ema_stack = np.sign(ema9 - ema21) + np.sign(ema21 - ema55)
        feats["ema_stack_bias"] = float(ema_stack.iloc[-1])
        stretch = ((c1h - ema9) / ema9).clip(-0.05, 0.05) * 20
        feats["ema_stretch_score"] = float(stretch.iloc[-1])
        vwap_num = (c1h * v1h).rolling(24).sum(); vwap_den = v1h.rolling(24).sum().replace(0, np.nan)
        vwap = vwap_num / vwap_den
        feats["vwap_distance_pct"] = float(((c1h - vwap) / vwap).iloc[-1])
        vwap_std = ((c1h - vwap) ** 2).rolling(24).mean() ** 0.5
        feats["vwap_stretch_score"] = float(((c1h - vwap) / vwap_std.replace(0, np.nan)).clip(-3, 3).iloc[-1])
        atr = (h1h - l1h).rolling(14).mean()
        feats["chg_4h_atr"] = float(((c1h - c1h.shift(4)) / atr.replace(0, np.nan)).iloc[-1])
    except Exception:
        for k in ("rsi_14", "macd_hist_1h", "bb_pct", "ema50_dist", "rvol_1h", "stoch_k",
                  "ema_stack_bias", "ema_stretch_score", "vwap_distance_pct", "vwap_stretch_score",
                  "chg_4h_atr"):
            feats.setdefault(k, NAN)

    # -- A: SOL 4h indicators -- exact port of training's resample+shift(3)
    # quirk: shift(3) on an already-4h-spaced series moves by 3 PERIODS
    # (12 CALENDAR HOURS), not 3 hours despite the wording. The model was
    # trained on this exact transform; must replicate as-is for parity.
    try:
        b4 = b1[b1.index <= T]
        c4h_full = b4["close"].resample("4h").last()
        c4h_shifted = c4h_full.shift(3)
        delta4 = c4h_shifted.diff()
        gain4 = delta4.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss4 = (-delta4.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi_4h_series = (100 - 100 / (1 + gain4 / loss4.replace(0, np.nan))).reindex(
            pd.date_range(c4h_full.index[0], T, freq="h", tz="UTC"), method="ffill")
        feats["rsi_4h"] = float(rsi_4h_series.iloc[-1])

        ll14_4h = c4h_full.rolling(14).min(); hh14_4h = c4h_full.rolling(14).max()
        stoch_k_4h_series = (((c4h_full - ll14_4h) / (hh14_4h - ll14_4h).replace(0, np.nan)) * 100).shift(3)
        stoch_k_4h_hourly = stoch_k_4h_series.reindex(
            pd.date_range(c4h_full.index[0], T, freq="h", tz="UTC"), method="ffill")
        feats["stoch_k_4h"] = float(stoch_k_4h_hourly.iloc[-1])
    except Exception:
        for k in ("stoch_k_4h", "rsi_4h"):
            feats[k] = NAN

    # -- A: crude standalone composite trend/rev/p_up (SOL-specific, NOT the
    #    BTC-calibrated composite_scorer -- matches training s1 exactly) -----
    try:
        ema9, ema21, ema55 = c1h.ewm(span=9, adjust=False).mean(), c1h.ewm(span=21, adjust=False).mean(), c1h.ewm(span=55, adjust=False).mean()
        trend_votes = (np.sign(ema9 - ema21) + np.sign(ema21 - ema55) +
                      np.sign(c1h - c1h.shift(4)) + np.sign(c1h - c1h.shift(24)))
        feats["composite_trend"] = float(trend_votes.iloc[-1])
        # training's composite_rev uses the manual span=14 EWM rsi_14 computed
        # above (s1_build_dataset.py), NOT _T.rsi_series (com=13, BTC-style) --
        # same mismatch caught in ETH's parity check, fixed here directly.
        delta1r = c1h.diff()
        gain1r = delta1r.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss1r = (-delta1r.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi14 = 100 - 100 / (1 + gain1r / loss1r.replace(0, np.nan))
        ll14 = l1h.rolling(14).min(); hh14 = h1h.rolling(14).max()
        stoch_k = ((c1h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100
        rev = ((rsi14 < 30).astype(int) - (rsi14 > 70).astype(int) +
              (stoch_k < 20).astype(int) - (stoch_k > 80).astype(int))
        feats["composite_rev"] = float(rev.iloc[-1])
        feats["composite_p_up"] = float(0.5 + 0.02 * np.clip(trend_votes.iloc[-1], -4, 4))
    except Exception:
        feats["composite_trend"] = NAN
        feats["composite_rev"] = NAN
        feats["composite_p_up"] = 0.5

    return feats


# ── public API ─────────────────────────────────────────────────────────────

def compute_sol_p_up(df_1h: "pd.DataFrame | None" = None,
                     now: "datetime | None" = None) -> "float | None":
    """SHADOW-ONLY SOL p_up score. Cached per completed 1h bar.

    df_1h : runner's SOL 1h frame -- fallback if this module's own fetch fails.
    Returns float or None. Never raises.
    """
    try:
        pipe = _load()
        if pipe is None:
            return None
        now = now or datetime.now(timezone.utc)
        now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo \
            else pd.Timestamp(now, tz="UTC")
        T = now_ts.floor("h") - pd.Timedelta(hours=1)
        if _last["bar_ts"] == T and _last["p"] is not None:
            return _last["p"]

        b1 = _fetch_asset_configured("1h", 1000, "SOL")
        if b1 is None or len(b1) < 200:
            b1 = df_1h
        if b1 is None or len(b1) < 200:
            return None
        if b1.index.tz is None:
            b1 = b1.copy(); b1.index = b1.index.tz_localize("UTC")
        b1 = b1[b1.index <= T]
        if not len(b1) or (T - b1.index[-1]) > pd.Timedelta(hours=2):
            return None
        T = b1.index[-1]

        feats = _assemble(T, b1)
        vec = np.array([[feats.get(f, NAN) for f in pipe["features"]]], dtype=float)
        if np.isnan(vec).all():
            return None
        p = float(pipe["clf"].predict_proba(vec)[0, 1])
        _last["bar_ts"], _last["p"] = T, p
        return p
    except Exception:
        return None
