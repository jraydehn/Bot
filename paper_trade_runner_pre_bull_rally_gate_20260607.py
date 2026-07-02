"""
Paper trading runner — executes the full live signal pipeline and logs the result.

Appends one row per run to results/paper_trades.csv. Resolution (resolved_yes,
would_win, would_pnl) is filled in later by outcome_checker.py after the
contract expires.

Usage:
    export KALSHI_KEY_ID=your-key-id
    export KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
    python3 paper_trade_runner.py
    python3 paper_trade_runner.py --bankroll 10000
    python3 paper_trade_runner.py --sim   # simulated p_market (no auth needed)
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import ta.volatility as _ta_vol

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from evaluate_point import load_data
from market_data import compute_realized_volatility
from probability_engine import estimate_probability, implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT, REALIZED_VOL_WEIGHT_BY_ASSET
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from order_book import fetch_order_book_imbalance
import btc_p_up_model as _btc_p_up_model
import asset_p_up_model as _asset_p_up_model
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from decision import evaluate_trade
from funding_rate import fetch_funding_rate, FundingRateResult
import outcome_checker
import update_data
import live_trading
import gate_audit_logger
from kelly_sizing import compute_kelly_size
from composite_scorer import compute_current_scores, compute_current_scores_30m, compute_current_scores_90m, score_to_p_model, score_to_p_no_model, composite_to_confirmation, lookup_p_up, lookup_p_up_blended, K_DRIFT_NO_BTC, K_DRIFT_NO_ETH
import direct_p_model
from direct_p_model import compute_p_no_direct, no_model_supported
import pickle as _pickle
from vol_layer import compute_vol_regime_factor
import deribit_iv
import coinalyze_liq
import orderbook_depth
import coinglass_data

# BTC isotonic calibration: corrects lognormal p_model overconfidence at extremes.
# Trained on 490 resolved BTC paper trades. Reduces NO bet losses in 20–30% p_market
# range where formula overestimates P(NO wins). Loaded lazily at first BTC scan.
_BTC_ISO_CAL: "dict | None | str" = "unloaded"

# Daily Markov regime cache — refreshed once per UTC calendar day.
_MARKOV_DAILY_CACHE: dict = {"date": None, "regime": None}

def _get_btc_daily_markov_regime() -> "str | None":
    """Return today's BTC daily Markov regime: 'Bull', 'Bear', 'Sideways', or None on error.

    Loads 3-state GaussianHMM (features: log_ret, realized_vol_20d, ret_5d) from
    results/hmm_3state_btc.pkl. More precise than ±2% threshold: separates low-vol
    steady markets (Bull) from medium-vol flat periods (Sideways) from high-vol
    crashes (Bear), preventing false Sideways blocks in calm low-vol markets.
    Result is cached by UTC date — one yfinance call per calendar day maximum.
    Returns None on any fetch/compute failure so the gate is skipped rather than
    blocking incorrectly.
    """
    global _MARKOV_DAILY_CACHE
    today = datetime.now(timezone.utc).date()
    if _MARKOV_DAILY_CACHE["date"] == today:
        return _MARKOV_DAILY_CACHE["regime"]
    try:
        import yfinance as _yf
        import numpy as _np_m
        import pandas as _pd_m
        _end   = _pd_m.Timestamp.now("UTC").normalize()
        _start = _end - _pd_m.DateOffset(days=90)
        _df = _yf.download(
            "BTC-USD",
            start=_start.strftime("%Y-%m-%d"),
            end=(_end + _pd_m.DateOffset(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if isinstance(_df.columns, _pd_m.MultiIndex):
            _df.columns = _df.columns.get_level_values(0)
        _close = _df["Close"].dropna()
        if len(_close) < 25:
            return None
        _lr    = _np_m.log(_close / _close.shift(1))
        _rv    = _lr.rolling(20, min_periods=10).std()
        _r5    = _np_m.log(_close / _close.shift(5))
        _feats = _pd_m.DataFrame({"log_ret": _lr, "realized_vol": _rv, "ret_5d": _r5}).dropna()
        if len(_feats) < 10:
            return None
        _X        = _feats[["log_ret", "realized_vol", "ret_5d"]].values
        _pkl_path = Path(__file__).parent / "results" / "hmm_3state_btc.pkl"
        _payload  = _pickle.load(open(_pkl_path, "rb"))
        _states   = _payload["model"].predict(_X)
        _regime   = _payload["state_to_name"][int(_states[-1])]
        _MARKOV_DAILY_CACHE["date"]   = today
        _MARKOV_DAILY_CACHE["regime"] = _regime
        _rv_last = float(_feats["realized_vol"].iloc[-1])
        _r5_last = float(_feats["ret_5d"].iloc[-1])
        print(f"  [markov_daily] BTC HMM 3-state → regime={_regime}  "
              f"(vol={_rv_last:.4f}  ret5d={_r5_last*100:+.2f}%)")
        return _regime
    except Exception as _exc:
        print(f"  [markov_daily] Fetch failed (gate skipped): {_exc}")
        return None

# ── 7-state BTC daily Markov regime ──────────────────────────────────────────
# More granular than 3-state: Crash_Bear / Correction / Consolidation / Recovery
#   / Slow_Bull / Bull / Explosive_Bull.  Uses 4 features (adds ret_20d).
# Validated 2026-06-04 on scan archive:
#   Correction: YES edge=-9.4% (block), NO edge=+6.7% (allow)
#   Consolidation: YES edge=-2.1% (soft), NO edge=+2.7% (allow)
_MARKOV_7STATE_CACHE: dict = {"date": None, "regime": None}

def _get_btc_7state_regime() -> "str | None":
    """Return today's BTC 7-state HMM regime or None on error (gate skipped)."""
    global _MARKOV_7STATE_CACHE
    today = datetime.now(timezone.utc).date()
    if _MARKOV_7STATE_CACHE["date"] == today:
        return _MARKOV_7STATE_CACHE["regime"]
    try:
        import yfinance as _yf
        import numpy as _np_7
        import pandas as _pd_7
        _end   = _pd_7.Timestamp.now("UTC").normalize()
        _start = _end - _pd_7.DateOffset(days=120)
        _df = _yf.download(
            "BTC-USD",
            start=_start.strftime("%Y-%m-%d"),
            end=(_end + _pd_7.DateOffset(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if isinstance(_df.columns, _pd_7.MultiIndex):
            _df.columns = _df.columns.get_level_values(0)
        _close = _df["Close"].dropna()
        if len(_close) < 25:
            return None
        _lr   = _np_7.log(_close / _close.shift(1))
        _rv   = _lr.rolling(20, min_periods=10).std()
        _r5   = _np_7.log(_close / _close.shift(5))
        _r20  = _np_7.log(_close / _close.shift(20))
        _feats = _pd_7.DataFrame({"log_ret": _lr, "realized_vol": _rv,
                                   "ret_5d": _r5, "ret_20d": _r20}).dropna()
        if len(_feats) < 10:
            return None
        _X = _feats[["log_ret", "realized_vol", "ret_5d", "ret_20d"]].values
        _pkl_path = Path(__file__).parent / "results" / "hmm_7state_btc.pkl"
        _payload  = _pickle.load(open(_pkl_path, "rb"))
        _states   = _payload["model"].predict(_X)
        _regime   = _payload["state_to_name"][int(_states[-1])]
        _MARKOV_7STATE_CACHE["date"]   = today
        _MARKOV_7STATE_CACHE["regime"] = _regime
        _rv_last  = float(_feats["realized_vol"].iloc[-1])
        _r5_last  = float(_feats["ret_5d"].iloc[-1])
        _r20_last = float(_feats["ret_20d"].iloc[-1])
        print(f"  [markov_7state] BTC HMM 7-state → regime={_regime}  "
              f"(vol={_rv_last:.4f}  ret5d={_r5_last*100:+.2f}%  ret20d={_r20_last*100:+.2f}%)")
        return _regime
    except Exception as _exc:
        print(f"  [markov_7state] Fetch failed (gate skipped): {_exc}")
        return None


# ── HMM SMC Phase model (4 states: Markup/Markdown/Transition/Divergence) ────
# Trained on 21,114 hourly SMC observations (Jan 2024 – Jun 2026).
# Features: bos_4h/1h ±1, choch_4h/1h, supply_pct, demand_pct, zone_imbal, in_supply/demand.
# States: 0=Markup, 1=Markdown, 2=Transition(ChoCH), 3=Divergence(4h-bear,1h-bull)
# Key outcome: State 2 NO bets at pm>0.80 → WR=8%, PnL=-$2,398 (primary block target).
_HMM_SMC_PKL        = Path(__file__).parent / "hmm_smc_phases.pkl"
_hmm_smc_model      = None
_hmm_smc_scaler     = None
_HMM_SMC_SUPPLY_MED = 3.064   # median supply_pct from training data (fill for None)
_HMM_SMC_DEMAND_MED = 3.950   # median demand_pct from training data
from collections import deque as _deque_smc
_hmm_smc_buf = _deque_smc(maxlen=24)   # rolling 24h of scaled observations

try:
    _hmm_smc_pkg    = _pickle.load(open(_HMM_SMC_PKL, "rb"))
    _hmm_smc_model  = _hmm_smc_pkg["model"]
    _hmm_smc_scaler = _hmm_smc_pkg["scaler"]
    print(f"  [hmm_smc] Loaded {_HMM_SMC_PKL.name}  "
          f"({_hmm_smc_pkg['n_states']} states, {len(_hmm_smc_pkg['feat_cols'])} features)")
except Exception as _hmm_smc_e:
    print(f"  [hmm_smc] WARNING: could not load {_HMM_SMC_PKL.name}: {_hmm_smc_e}")

# ── Vol-regime HMM (BTC 1h runner) ─────────────────────────────────────────
# 2-state ergodic Gaussian HMM on 15m returns: R0=low-vol, R1=high-vol.
# R0: σ_ann=32%, self-trans=0.978, mean residence ~11h (near-absorbing).
# R1: σ_ann=88%, self-trans=0.923, mean residence ~3h  (near-absorbing).
# Full-history aggregate: R0=$0.84/trade, R1=$0.04/trade (-$2.69 May+).
#
# Three signals logged (soft posterior, not just hard Viterbi):
#   hmm_vol_state  — hard Viterbi rank (0=R0, 1=R1) for backwards compat
#   hmm_r1_prob    — P(R1|observations): soft posterior; catches transitions ~5 steps early
#   hmm_vol_k10    — P(R1 in 10 steps): forward forecast using P^10; regime persistence signal
#
# Gate readiness: 100+ R1 obs → run analyze_hmm_vol_state_otm_itm.py
_VOL_HMM_PKL_1H   = Path(__file__).parent / "models" / "hmm_ergodic_2state_btc_15m.pkl"
_vol1h_hmm_model  = None
_vol1h_hmm_order  = None
_vol1h_hmm_rankof: "dict | None" = None
_vol1h_P10:        "object | None" = None   # P^10 pre-computed at load time

try:
    import numpy as _np_volhmm
    with open(_VOL_HMM_PKL_1H, "rb") as _vhf:
        _vol1h_pkg = _pickle.load(_vhf)
    _vol1h_hmm_model  = _vol1h_pkg["model"]
    _n_vol1h          = _vol1h_pkg["n_states"]
    _vol1h_hmm_order  = sorted(range(_n_vol1h),
                               key=lambda s: float(
                                   _np_volhmm.sqrt(_vol1h_hmm_model.covars_[s, 0, 0])))
    _vol1h_hmm_rankof = {s: i for i, s in enumerate(_vol1h_hmm_order)}
    # Pre-compute P^10 for forward forecast (rank-ordered rows/cols)
    _P_raw = _vol1h_hmm_model.transmat_
    _P_ord = _np_volhmm.array([[_P_raw[_vol1h_hmm_order[i], _vol1h_hmm_order[j]]
                                 for j in range(_n_vol1h)]
                                for i in range(_n_vol1h)])
    _vol1h_P10 = _np_volhmm.linalg.matrix_power(_P_ord, 10)
    print(f"  [vol_hmm_1h] Loaded {_VOL_HMM_PKL_1H.name}  ({_n_vol1h} states)  "
          f"P10[R1→R1]={_vol1h_P10[1,1]:.3f}")
except Exception as _ve1h:
    print(f"  [vol_hmm_1h] WARNING: {_ve1h}")

# [vol_hmm ETH] 2-state ergodic HMM for ETH vol regime.
# Trained on ETH 15m log-returns (2025-01-01+). R0=low-vol (80%), R1=high-vol (20%).
# Gate: block ETH OTM YES (offset>0) in R1; rescue when composite_trend>=2.
# Validated: MCPT p=0.005, perm p=0.002, n=606, edge=-7.2% (2026-06-04).
_VOL_HMM_PKL_ETH    = Path(__file__).parent / "models" / "hmm_ergodic_2state_eth_15m.pkl"
_vol_hmm_model_eth  = None
_vol_hmm_rankof_eth: "dict | None" = None
_vol_hmm_order_eth  = None
_vol_hmm_P10_eth:   "object | None" = None
_n_vol_eth          = 0

try:
    import numpy as _np_volhmm_eth
    with open(_VOL_HMM_PKL_ETH, "rb") as _vhf_eth:
        _vol_eth_pkg = _pickle.load(_vhf_eth)
    _vol_hmm_model_eth  = _vol_eth_pkg["model"]
    _n_vol_eth          = _vol_eth_pkg["n_states"]
    _vol_hmm_order_eth  = sorted(range(_n_vol_eth),
                                 key=lambda s: float(
                                     _np_volhmm_eth.sqrt(_vol_hmm_model_eth.covars_[s, 0, 0])))
    _vol_hmm_rankof_eth = {s: i for i, s in enumerate(_vol_hmm_order_eth)}
    _P_raw_eth = _vol_hmm_model_eth.transmat_
    _P_ord_eth = _np_volhmm_eth.array([[_P_raw_eth[_vol_hmm_order_eth[i], _vol_hmm_order_eth[j]]
                                        for j in range(_n_vol_eth)]
                                       for i in range(_n_vol_eth)])
    _vol_hmm_P10_eth = _np_volhmm_eth.linalg.matrix_power(_P_ord_eth, 10)
    print(f"  [vol_hmm_eth] Loaded {_VOL_HMM_PKL_ETH.name}  ({_n_vol_eth} states)  "
          f"P10[R1→R1]={_vol_hmm_P10_eth[1,1]:.3f}")
except Exception as _ve_eth:
    print(f"  [vol_hmm_eth] WARNING: {_ve_eth}")

# [vol_hmm SOL] 2-state ergodic HMM for SOL vol regime (shadow only — MCPT p=0.714).
# R1 OTM YES gate fails MCPT on 2 weeks of data; logging only until data accumulates.
_VOL_HMM_PKL_SOL    = Path(__file__).parent / "models" / "hmm_ergodic_2state_sol_15m.pkl"
_vol_hmm_model_sol  = None
_vol_hmm_rankof_sol: "dict | None" = None
_vol_hmm_order_sol  = None
_n_vol_sol          = 0

try:
    import numpy as _np_volhmm_sol
    with open(_VOL_HMM_PKL_SOL, "rb") as _vhf_sol:
        _vol_sol_pkg = _pickle.load(_vhf_sol)
    _vol_hmm_model_sol  = _vol_sol_pkg["model"]
    _n_vol_sol          = _vol_sol_pkg["n_states"]
    _vol_hmm_order_sol  = sorted(range(_n_vol_sol),
                                 key=lambda s: float(
                                     _np_volhmm_sol.sqrt(_vol_hmm_model_sol.covars_[s, 0, 0])))
    _vol_hmm_rankof_sol = {s: i for i, s in enumerate(_vol_hmm_order_sol)}
    print(f"  [vol_hmm_sol] Loaded {_VOL_HMM_PKL_SOL.name}  ({_n_vol_sol} states)  [shadow only]")
except Exception as _ve_sol:
    print(f"  [vol_hmm_sol] WARNING: {_ve_sol}")

# [hmm_mtf_momentum] Multi-timeframe momentum HMM (9-state, BTC-only).
# State 3: stoch_k_1h≈64, rsi_1h≈57, bp_1h≈0.86, macd_hist_1h≈42, adx_1h≈32
#          — moderate uptrend/bullish momentum; WR=24.1%, ppt=-$149.7.
# Gate: block BTC NO when St3; rescue if offset∈[0,5%) + macd_hist_1h<-50.
_MTF_HMM_PKL      = Path(__file__).parent / "models" / "hmm_mtf_momentum_btc15m.pkl"
_mtf_hmm_model    = None
_mtf_hmm_scaler   = None

try:
    import numpy as _np_mtf
    with open(_MTF_HMM_PKL, "rb") as _mf:
        _mtf_pkg = _pickle.load(_mf)
    _mtf_hmm_model  = _mtf_pkg["model"]
    _mtf_hmm_scaler = _mtf_pkg["scaler"]
    print(f"  [hmm_mtf] Loaded {_MTF_HMM_PKL.name}  ({_mtf_pkg['n_states']} states, St3 WR=24.1%)")
except Exception as _mtf_load_e:
    print(f"  [hmm_mtf] WARNING: {_mtf_load_e}")


def _vol1h_hmm_probs(live_1m: "pd.DataFrame") -> "tuple[int,float,float,int] | None":
    """Return (hard_rank, r1_prob, r1_prob_k10, time_in_state) from live 1m data.

    hard_rank      : Viterbi hard state rank (0=R0 low-vol, 1=R1 high-vol)
    r1_prob        : soft posterior P(R1|observations) — catches transitions ~5 bars early
    r1_prob_k10    : 10-step forward P(R1) = posterior @ P^10 @ e_R1
    time_in_state  : consecutive trailing bars in current hard state (semi-Markov sojourn depth)
                     R1 early=1-3 (spike), mid=4-15 (settling), deep=16+ (committed episode)
    """
    if _vol1h_hmm_model is None or _vol1h_hmm_rankof is None or _vol1h_P10 is None:
        return None
    try:
        c15 = live_1m["close"].resample("15min").last().dropna()
        if len(c15) < 22:
            return None
        lr       = _np_volhmm.log(c15 / c15.shift(1)).dropna().values[-20:]
        obs      = lr.reshape(-1, 1)
        # Hard Viterbi sequence — use full sequence for time-in-state
        raw_seq   = _vol1h_hmm_model.predict(obs)
        hard_rank = _vol1h_hmm_rankof[int(raw_seq[-1])]
        # Count trailing bars in current state (semi-Markov sojourn depth)
        time_in_state = 0
        for _rs in reversed(raw_seq):
            if _vol1h_hmm_rankof[int(_rs)] == hard_rank:
                time_in_state += 1
            else:
                break
        # Soft posterior
        post_raw  = _vol1h_hmm_model.predict_proba(obs)[-1]
        post_ord  = _np_volhmm.array([post_raw[_vol1h_hmm_order[i]] for i in range(_n_vol1h)])
        r1_prob   = float(post_ord[1])
        r1_k10    = float(post_ord @ _vol1h_P10[:, 1])
        return (hard_rank, round(r1_prob, 4), round(r1_k10, 4), time_in_state)
    except Exception:
        return None


def _compute_sigma_swing_high(
    live_1m: "pd.DataFrame",
    sigma: float = 0.01,
    lookback_bars: int = 2000,
) -> "tuple[float | None, float | None]":
    """Return (last_confirmed_swing_high, last_confirmed_swing_low) from
    percentage-based Directional Change on the most recent `lookback_bars`
    of 1m data.  sigma=0.01 → 1% retracement required to confirm.
    Returns (None, None) if insufficient data or no swings confirmed.
    """
    try:
        df = live_1m.tail(lookback_bars)
        if len(df) < 50:
            return None, None
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        c = df["close"].astype(float).values

        up_zig = True
        tmp_max = h[0]; tmp_max_i = 0
        tmp_min = l[0]; tmp_min_i = 0
        last_high = None
        last_low  = None

        for i in range(len(c)):
            if up_zig:
                if h[i] > tmp_max:
                    tmp_max = h[i]; tmp_max_i = i
                elif c[i] < tmp_max * (1.0 - sigma):
                    last_high = tmp_max
                    up_zig    = False
                    tmp_min   = l[i]; tmp_min_i = i
            else:
                if l[i] < tmp_min:
                    tmp_min = l[i]; tmp_min_i = i
                elif c[i] > tmp_min * (1.0 + sigma):
                    last_low = tmp_min
                    up_zig   = True
                    tmp_max  = h[i]; tmp_max_i = i

        return last_high, last_low
    except Exception:
        return None, None


def _compute_rw_tops_1h(
    live_1m: "pd.DataFrame",
    order: int = 5,
    lookback_bars: int = 200,
) -> "list[tuple[float, pd.Timestamp]] | None":
    """Return list of (top_price, conf_timestamp) for RW order=5 tops confirmed
    in the last `lookback_bars` 1h bars, sorted by confirmation time ascending.
    Uses close prices resampled to 1h.  Returns None on failure.
    """
    try:
        c1h = live_1m["close"].resample("1h").last().dropna()
        c1h = c1h.iloc[-lookback_bars - order * 2:]   # extra bars for edge warmup
        if len(c1h) < order * 2 + 2:
            return None
        arr  = c1h.to_numpy(dtype=float)
        times = c1h.index
        tops = []
        for i in range(len(arr)):
            if i < order * 2 + 1:
                continue
            k = i - order
            v = arr[k]
            is_top = True
            for j in range(1, order + 1):
                if arr[k + j] > v or arr[k - j] > v:
                    is_top = False
                    break
            if is_top:
                tops.append((float(arr[k]), times[i]))   # (price, conf_time)
        return tops if tops else None
    except Exception:
        return None


def _vol_hmm_probs_eth(live_1m: "pd.DataFrame") -> "tuple[int,float,int] | None":
    """Return (hard_rank, r1_prob, time_in_state) for ETH using ETH-specific HMM."""
    if _vol_hmm_model_eth is None or _vol_hmm_rankof_eth is None:
        return None
    try:
        c15 = live_1m["close"].resample("15min").last().dropna()
        if len(c15) < 22:
            return None
        lr  = _np_volhmm_eth.log(c15 / c15.shift(1)).dropna().values[-20:]
        obs = lr.reshape(-1, 1)
        raw_seq   = _vol_hmm_model_eth.predict(obs)
        hard_rank = _vol_hmm_rankof_eth[int(raw_seq[-1])]
        time_in_state = 0
        for _rs in reversed(raw_seq):
            if _vol_hmm_rankof_eth[int(_rs)] == hard_rank:
                time_in_state += 1
            else:
                break
        post_raw = _vol_hmm_model_eth.predict_proba(obs)[-1]
        post_ord = _np_volhmm_eth.array([post_raw[_vol_hmm_order_eth[i]] for i in range(_n_vol_eth)])
        r1_prob  = float(post_ord[1])
        return (hard_rank, round(r1_prob, 4), time_in_state)
    except Exception:
        return None


def _vol_hmm_probs_sol(live_1m: "pd.DataFrame") -> "tuple[int,float,int] | None":
    """Return (hard_rank, r1_prob, time_in_state) for SOL using SOL-specific HMM."""
    if _vol_hmm_model_sol is None or _vol_hmm_rankof_sol is None:
        return None
    try:
        c15 = live_1m["close"].resample("15min").last().dropna()
        if len(c15) < 22:
            return None
        lr  = _np_volhmm_sol.log(c15 / c15.shift(1)).dropna().values[-20:]
        obs = lr.reshape(-1, 1)
        raw_seq   = _vol_hmm_model_sol.predict(obs)
        hard_rank = _vol_hmm_rankof_sol[int(raw_seq[-1])]
        time_in_state = 0
        for _rs in reversed(raw_seq):
            if _vol_hmm_rankof_sol[int(_rs)] == hard_rank:
                time_in_state += 1
            else:
                break
        post_raw = _vol_hmm_model_sol.predict_proba(obs)[-1]
        post_ord = _np_volhmm_sol.array([post_raw[_vol_hmm_order_sol[i]] for i in range(_n_vol_sol)])
        r1_prob  = float(post_ord[1])
        return (hard_rank, round(r1_prob, 4), time_in_state)
    except Exception:
        return None


_HMM_SMC_STATE_LABELS = {
    0: "Markup",
    1: "Markdown",
    2: "Transition(ChoCH)",
    3: "Divergence",
}

def _hmm_smc_predict(smc_result) -> int:
    """Predict HMM SMC phase state from a SMCResult. Returns -1 on load error."""
    if _hmm_smc_model is None or _hmm_smc_scaler is None or smc_result is None:
        return -1
    import numpy as _np_hmm
    _bos_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    _vec = [
        _bos_map.get(smc_result.bos_4h, 0.0),
        _bos_map.get(smc_result.bos_1h, 0.0),
        float(smc_result.choch_4h),
        float(smc_result.choch_1h),
        float(min(20.0, max(0.0, smc_result.nearest_supply_pct))
              if smc_result.nearest_supply_pct is not None else _HMM_SMC_SUPPLY_MED),
        float(min(20.0, max(0.0, smc_result.nearest_demand_pct))
              if smc_result.nearest_demand_pct is not None else _HMM_SMC_DEMAND_MED),
        float(smc_result.n_supply_zones - smc_result.n_demand_zones),
        float(smc_result.in_supply_zone),
        float(smc_result.in_demand_zone),
    ]
    try:
        _scaled = _hmm_smc_scaler.transform([_vec])[0]
        _hmm_smc_buf.append(_scaled)
        if len(_hmm_smc_buf) < 3:
            return -1
        _buf_arr = _np_hmm.array(list(_hmm_smc_buf))
        return int(_hmm_smc_model.predict(_buf_arr)[-1])
    except Exception:
        return -1


def _get_daily_markov_regime(asset: str) -> "str | None":
    """Return today's daily Markov regime for BTC, ETH, or SOL.

    BTC: 20-day rolling return on daily closes, ±2% threshold (existing logic).
    ETH: delegates to _get_markov_regimes_yf("ETH")["1d"] (±3%, cached per hour).
    SOL: delegates to _get_markov_regimes_yf("SOL")["6h"] (±3%, cached per hour).
    Returns None on any failure so callers skip regime-dependent logic rather than
    blocking incorrectly.
    """
    if asset == "BTC":
        return _get_btc_daily_markov_regime()
    try:
        _regs = _get_markov_regimes_yf(asset)
        if asset == "ETH":
            return _regs.get("1d")
        if asset == "SOL":
            return _regs.get("6h")
    except Exception as _exc:
        print(f"  [markov_daily] {asset} fetch failed: {_exc}")
    return None


# GARCH(1,1) conditional vol ratio cache — recomputed once per hour.
# ratio = cond_vol / long_run_vol; > 1.5 signals high vol regime (bull-trap risk on YES).
_GARCH_RATIO_CACHE: dict = {"hour": None, "ratio": None, "cond_ve": None}


def _compute_v_hawk(df_1h: pd.DataFrame, kappa: float = 0.01,
                    norm_lb: int = 336, roll_lb: int = 336) -> "tuple[float, str]":
    """Return (v_hawk_last, regime) from 1h OHLC.

    kappa=0.01, norm_lb=336, roll_lb=336 chosen from sweep (best PF).
    Regime: quiet / low / mid / elevated / spike (rolling q25/50/75/90 thresholds).
    Returns (nan, '') on failure or insufficient data.
    """
    try:
        import numpy as _np_vhawk
        if len(df_1h) < norm_lb + roll_lb:
            return float("nan"), ""
        hi  = _np_vhawk.log(df_1h["high"].astype(float))
        lo  = _np_vhawk.log(df_1h["low"].astype(float))
        cl  = _np_vhawk.log(df_1h["close"].astype(float))
        atr = _ta_vol.average_true_range(hi, lo, cl, window=norm_lb)
        nr  = (hi - lo) / atr
        nr  = nr.replace([_np_vhawk.inf, -_np_vhawk.inf], float("nan"))
        alpha = math.exp(-kappa)
        arr   = nr.to_numpy()
        out   = _np_vhawk.full(len(arr), float("nan"))
        for i in range(1, len(arr)):
            out[i] = arr[i] if _np_vhawk.isnan(out[i - 1]) else out[i - 1] * alpha + arr[i]
        vhawk = pd.Series(out * kappa, index=df_1h.index)
        q25 = vhawk.rolling(roll_lb).quantile(0.25).iloc[-1]
        q50 = vhawk.rolling(roll_lb).quantile(0.50).iloc[-1]
        q75 = vhawk.rolling(roll_lb).quantile(0.75).iloc[-1]
        q90 = vhawk.rolling(roll_lb).quantile(0.90).iloc[-1]
        vh  = vhawk.iloc[-1]
        if _np_vhawk.isnan(vh) or _np_vhawk.isnan(q25):
            return float("nan"), ""
        if vh < q25:   regime = "quiet"
        elif vh < q50: regime = "low"
        elif vh < q75: regime = "mid"
        elif vh < q90: regime = "elevated"
        else:          regime = "spike"
        return round(float(vh), 6), regime
    except Exception as _e:
        print(f"  [v_hawk] ERROR: {_e}")
        return float("nan"), ""


# PC1-RSI gate constants — trained on 2024-2025 BTC 1h data (RSI periods 2-24).
# PC1 = fast-vs-slow RSI divergence: negative = fast RSI << slow RSI (short-term weakness).
# Threshold q10 = -34.93 (bottom decile of training distribution).
# Backup: paper_trade_runner_pre_pc1_gate.py
_PC1_RSI_MEANS = [
    51.405519, 51.243366, 51.150051, 51.096481, 51.063375, 51.041335, 51.025548,
    51.013425, 51.003536, 50.995075, 50.987578, 50.980777, 50.974506, 50.968663,
    50.963177, 50.958002, 50.953099, 50.948442, 50.944006, 50.939774, 50.935727,
    50.931852, 50.928135,
]
_PC1_EVEC = [
    -0.60859181, -0.35463120, -0.19336223, -0.08657594, -0.01300311,  0.03922117,
     0.07712018,  0.10507346,  0.12592809,  0.14159878,  0.15341033,  0.16230187,
     0.16895327,  0.17386590,  0.17741545,  0.17988756,  0.18150210,  0.18243029,
     0.18280675,  0.18273828,  0.18231029,  0.18159156,  0.18063782,
]
_PC1_Q10 = -34.927972   # training q10; gate fires when pc1 <= this


def _compute_pc1(df_1h: pd.DataFrame) -> float:
    """Return PC1 RSI score from 1h close.

    PC1 = dot(centered_rsi_vec, _PC1_EVEC).  Negative = fast RSI below slow RSI.
    Returns nan on failure or insufficient data (need >= 25 bars).
    """
    try:
        import numpy as _np_pc1
        from ta.momentum import RSIIndicator as _RSI
        if len(df_1h) < 25:
            return float("nan")
        rsi_vals = []
        for _p, _mean, _w in zip(range(2, 25), _PC1_RSI_MEANS, _PC1_EVEC):
            _r = _RSI(close=df_1h["close"].astype(float), window=_p).rsi()
            _last = _r.iloc[-1]
            if _np_pc1.isnan(_last):
                return float("nan")
            rsi_vals.append((_last - _mean) * _w)
        return round(float(sum(rsi_vals)), 6)
    except Exception as _e:
        print(f"  [pc1_rsi] ERROR: {_e}")
        return float("nan")


def _get_garch_ratio(df_1h: pd.DataFrame, asset: str = "BTC") -> "float | None":
    """Return GARCH(1,1) conditional vol ratio for the most recent 1h bar.

    Fits on the last 500 bars of log returns. Caches per UTC hour — one fit per hour max.
    Returns None on any failure so callers skip the gate rather than blocking incorrectly.
    """
    global _GARCH_RATIO_CACHE
    if asset != "BTC":
        return None
    current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if _GARCH_RATIO_CACHE["hour"] == current_hour and _GARCH_RATIO_CACHE["ratio"] is not None:
        return _GARCH_RATIO_CACHE["ratio"]
    try:
        import numpy as _np_g
        from arch import arch_model as _arch_model
        import warnings as _w
        _w.filterwarnings("ignore")
        _close = df_1h["close"].astype(float).dropna()
        if len(_close) < 502:
            return None
        _window = _np_g.log(_close / _close.shift(1)).dropna() * 100
        _w500   = _window.iloc[-500:]
        _am     = _arch_model(_w500, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        _res    = _am.fit(disp="off", show_warning=False)
        _cond_v = float(_res.conditional_volatility.iloc[-1])
        _omega  = float(_res.params["omega"])
        _alpha  = float(_res.params["alpha[1]"])
        _beta   = float(_res.params["beta[1]"])
        _persist = _alpha + _beta
        _lr_vol  = (float(_np_g.sqrt(_omega / (1.0 - _persist)))
                    if _persist < 1.0 else float(_w500.std()))
        _ratio   = _cond_v / _lr_vol if _lr_vol > 0 else 1.0
        _GARCH_RATIO_CACHE["hour"]    = current_hour
        _GARCH_RATIO_CACHE["ratio"]   = _ratio
        _GARCH_RATIO_CACHE["cond_ve"] = _cond_v / 100.0 / math.sqrt(60.0)
        print(f"  [garch] BTC cond_vol={_cond_v:.4f}% lr_vol={_lr_vol:.4f}% ratio={_ratio:.3f} persist={_persist:.3f}")
        return _ratio
    except Exception as _exc:
        print(f"  [garch] Fit failed (gate skipped): {_exc}")
        return None


def _get_garch_cond_ve(df_1h: pd.DataFrame) -> float:
    """Return GARCH(1,1) conditional vol in vol_eff units (per sqrt-minute).
    Reuses the hourly-cached fit from _get_garch_ratio — no extra fitting cost."""
    _get_garch_ratio(df_1h, "BTC")
    cve = _GARCH_RATIO_CACHE.get("cond_ve")
    return float(cve) if cve is not None else float("nan")


def _load_btc_iso() -> "dict | None":
    global _BTC_ISO_CAL
    if _BTC_ISO_CAL != "unloaded":
        return _BTC_ISO_CAL
    path = Path(__file__).parent / "reform_results" / "btc_iso_calibration.pkl"
    if not path.exists():
        _BTC_ISO_CAL = None
        return None
    try:
        with open(path, "rb") as _f:
            _BTC_ISO_CAL = _pickle.load(_f)
        print(f"  [btc_iso] Loaded calibrator (n={_BTC_ISO_CAL['n_train']} trades)")
    except Exception as _e:
        print(f"  [btc_iso] Failed to load: {_e}")
        _BTC_ISO_CAL = None
    return _BTC_ISO_CAL

# BTC LGBM shadow model: trained on BTC paper trade archive (signals → would_win).
# Shadow mode only — logs p_gbdt alongside p_yes_model without changing trade logic.
# Retrain with: python3 train_btc_lgbm.py
_BTC_LGBM: "dict | None | str" = "unloaded"

# Gate meta-model shadow mode — predicts p(gate block was correct) for each BTC gate fire.
# Retrain with: python3 train_btc_gate_meta.py  (needs ~20+ resolved rows per gate)
_BTC_GATE_META: "dict | None | str" = "unloaded"


def _load_gate_meta_lgbm() -> "dict | None":
    global _BTC_GATE_META
    if _BTC_GATE_META != "unloaded":
        return _BTC_GATE_META
    path = Path(__file__).parent / "reform_results" / "btc_gate_meta.pkl"
    if not path.exists():
        _BTC_GATE_META = None
        return None
    try:
        with open(path, "rb") as _f:
            _BTC_GATE_META = _pickle.load(_f)
        _feat_n = len(_BTC_GATE_META.get("features", []))
        _auc    = _BTC_GATE_META.get("auc_te", float("nan"))
        print(f"  [btc_gate_meta] Loaded shadow model ({_feat_n} features, test AUC={_auc:.3f})")
    except Exception as _e:
        print(f"  [btc_gate_meta] Failed to load: {_e}")
        _BTC_GATE_META = None
    return _BTC_GATE_META


def _infer_gate_meta(pipe: dict, gate_name: str, feat_vals: dict) -> "float | None":
    """Run one inference pass through the gate meta-model. Returns p(block correct)."""
    import numpy as _np
    try:
        gate_to_int = pipe.get("gate_to_int", {})
        if gate_name not in gate_to_int:
            return None
        feat_vals = dict(feat_vals)
        feat_vals["gate_enc"] = float(gate_to_int[gate_name])
        feats = pipe["features"]
        vec   = _np.array([[feat_vals.get(f, _np.nan) for f in feats]])
        return float(_np.clip(pipe["clf"].predict_proba(vec)[0, 1], 0.01, 0.99))
    except Exception as _e:
        print(f"  [btc_gate_meta] inference error: {_e}")
        return None

def _load_btc_lgbm() -> "dict | None":
    global _BTC_LGBM
    if _BTC_LGBM != "unloaded":
        return _BTC_LGBM
    path = Path(__file__).parent / "reform_results" / "btc_lgbm.pkl"
    if not path.exists():
        _BTC_LGBM = None
        return None
    try:
        with open(path, "rb") as _f:
            _BTC_LGBM = _pickle.load(_f)
        _feat_n = len(_BTC_LGBM.get("features", []))
        _auc    = _BTC_LGBM.get("auc_te", float("nan"))
        print(f"  [btc_lgbm] Loaded shadow model ({_feat_n} features, test AUC={_auc:.3f})")
    except Exception as _e:
        print(f"  [btc_lgbm] Failed to load: {_e}")
        _BTC_LGBM = None
    return _BTC_LGBM


def _infer_btc_lgbm(pipe: dict, feat_vals: dict) -> "float | None":
    """Run one inference pass through the BTC LGBM shadow model."""
    import numpy as _np
    try:
        feats  = pipe["features"]
        vec    = _np.array([[feat_vals.get(f, _np.nan) for f in feats]])
        p_raw  = float(pipe["clf"].predict_proba(vec)[0, 1])
        platt  = pipe.get("platt")
        if platt is not None:
            import math as _math
            logit = _math.log(max(p_raw, 1e-6) / max(1 - p_raw, 1e-6))
            p_cal = float(platt.predict_proba([[logit]])[0, 1])
        else:
            p_cal = p_raw
        return float(_np.clip(p_cal, 0.01, 0.99))
    except Exception as _e:
        print(f"  [btc_lgbm] inference error: {_e}")
        return None


# ETH and SOL LGBM shadow models — same architecture, separate archives.
# Retrain with: python3 train_eth_sol_lgbm.py
_ETH_LGBM: "dict | None | str" = "unloaded"
_SOL_LGBM: "dict | None | str" = "unloaded"

def _load_asset_lgbm(asset: str) -> "dict | None":
    global _ETH_LGBM, _SOL_LGBM
    _ref = _ETH_LGBM if asset == "ETH" else _SOL_LGBM
    if _ref != "unloaded":
        return _ref
    fname = "eth_lgbm.pkl" if asset == "ETH" else "sol_lgbm.pkl"
    path  = Path(__file__).parent / "reform_results" / fname
    if not path.exists():
        result = None
    else:
        try:
            with open(path, "rb") as _f:
                result = _pickle.load(_f)
            _feat_n = len(result.get("features", []))
            _auc    = result.get("auc_te", float("nan"))
            print(f"  [{asset.lower()}_lgbm] Loaded shadow model ({_feat_n} features, test AUC={_auc:.3f})")
        except Exception as _e:
            print(f"  [{asset.lower()}_lgbm] Failed to load: {_e}")
            result = None
    if asset == "ETH":
        _ETH_LGBM = result
    else:
        _SOL_LGBM = result
    return result

def _infer_asset_lgbm(pipe: dict, feat_vals: dict, asset: str) -> "float | None":
    import numpy as _np
    try:
        feats = pipe["features"]
        vec   = _np.array([[feat_vals.get(f, _np.nan) for f in feats]])
        p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])
        platt = pipe.get("platt")
        if platt is not None:
            import math as _math
            logit = _math.log(max(p_raw, 1e-6) / max(1 - p_raw, 1e-6))
            p_cal = float(platt.predict_proba([[logit]])[0, 1])
        else:
            p_cal = p_raw
        return float(_np.clip(p_cal, 0.01, 0.99))
    except Exception as _e:
        print(f"  [{asset.lower()}_lgbm] inference error: {_e}")
        return None


# ETH isotonic calibration: log-normal p_yes → actual WR mapping.
# Trained on 18,836 reconstructed ETH outcomes (Apr 15 – May 3 2026).
# Key effect: p_ln < 0.20 → iso ≈ 0.04, eliminating phantom OTM YES edge
# that HistGBM overestimates. Calibration tracks actual WR within ~2pp.
_ETH_ISO_CAL: "object | None | str" = "unloaded"

def _load_eth_iso():
    global _ETH_ISO_CAL
    if _ETH_ISO_CAL != "unloaded":
        return _ETH_ISO_CAL
    path = Path(__file__).parent / "models" / "eth_iso_cal.pkl"
    if not path.exists():
        _ETH_ISO_CAL = None
        return None
    try:
        with open(path, "rb") as _f:
            _ETH_ISO_CAL = _pickle.load(_f)
        print(f"  [eth_iso] Loaded ETH isotonic calibrator")
    except Exception as _e:
        print(f"  [eth_iso] Failed to load: {_e}")
        _ETH_ISO_CAL = None
    return _ETH_ISO_CAL

def _compute_eth_p_ln(spot, strike, vol_eff, tau_min, p_up, k_drift=0.80):
    """Log-normal YES probability for ETH with k_drift=0.80."""
    if vol_eff <= 0 or tau_min <= 0 or spot <= 0:
        return None
    import math as _math
    from scipy.stats import norm as _norm
    sigma_tau = vol_eff * _math.sqrt(tau_min)
    if sigma_tau <= 0:
        return None
    z = _math.log(strike / spot) / sigma_tau
    z_adj = z - _norm.ppf(p_up) * k_drift
    return float(max(0.01, min(0.99, 1 - _norm.cdf(z_adj))))

from live_signal import (
    load_auth, kalshi_get, fetch_live_spot, fetch_current_price, find_live_contract,
    fetch_contracts_for_nearest_expiry, fetch_recent_1m_candles, minutes_to_expiry,
    BASE_URL, SERIES_TICKER, CANDLE_WINDOW, TAU, ASSET_CONFIG,
)

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"

# Funding rate cache — funding updates every 8 hours so re-fetching once per
# minute is wasteful. Cache the result for 5 minutes (300 seconds).
_funding_cache: "FundingRateResult | None" = None
_funding_cache_ts: float = 0.0
_FUNDING_CACHE_TTL = 300  # seconds

# In-memory dict of tickers traded this process run: {ticker: net_edge_at_trade}.
# Tickers traded this session: {ticker: net_edge}. Hard-blocks re-entry.
# Seeded once from CSV at startup to survive restarts, then cleared each hour.
_SESSION_TRADED: dict = {}
_SESSION_SEEDED: bool = False  # ensures CSV seed only runs once per process
_SIDE_COOLDOWN: dict = {}  # {(expiry_key, side): datetime} — last trade time per expiry+direction
from collections import deque
_pm_history: dict = {}  # {ticker: deque(maxlen=6)} — rolling p_market per contract for 5m drift
_pup_v2_buf: deque = deque(maxlen=4)          # rolling 4h p_up_v2 for BTC NO regime detection
_pup_v2_regime_state: dict = {"last_bar_ts": None}  # mutable state for in-function update


def _expiry_prefix(ticker: str) -> str:
    """Extract the expiry portion of a contract ticker.

    e.g. 'KXETHD-26APR0701-T2119.99' → 'KXETHD-26APR0701'
    Used as a consistent key for per-expiry trade counting across live and paper runners.
    """
    parts = ticker.rsplit("-T", 1)
    return parts[0] if len(parts) == 2 else ticker


def compute_zdrift_empirical(
    df_resolved: pd.DataFrame,
    df_confirm_1h: pd.DataFrame,
    w_short: int = 10,
    w_long: int = 30,
    alpha: float = 0.6,
    cap: float = 0.5,
) -> float:
    """
    Rolling empirical z-drift from resolved BTC trades.

    For each resolved trade: actual_z = log(btc_at_expiry / spot) / (vol_eff * sqrt(tau_min))
    Drift = alpha * mean(actual_z[-w_short:]) + (1-alpha) * mean(actual_z[-w_long:]), capped at ±cap.

    Uses the open price of the 1h candle at close_ts as btc_at_expiry (first tick of the
    new hour ≈ settlement price for Kalshi hourly contracts).

    Returns 0.0 on insufficient data or error.
    """
    try:
        needed = ["spot", "vol_eff", "tau_minutes", "close_ts", "resolved_yes"]
        df = df_resolved.dropna(subset=needed).copy()
        if len(df) < w_short:
            return 0.0
        df = df.tail(max(w_long, 50))
        df["_close_ts_utc"] = pd.to_datetime(df["close_ts"], utc=True)
        confirm_idx = df_confirm_1h.index
        actual_z_list: list[float] = []
        for _, row in df.iterrows():
            try:
                ts = row["_close_ts_utc"]
                spot_val = float(row["spot"])
                vol_eff  = float(row["vol_eff"])
                tau_min  = float(row["tau_minutes"])
                if spot_val <= 0 or vol_eff <= 0 or tau_min <= 0:
                    continue
                sigma_tau = vol_eff * math.sqrt(tau_min)
                if sigma_tau <= 0:
                    continue
                if ts in confirm_idx:
                    btc_expiry = float(df_confirm_1h.loc[ts, "open"])
                else:
                    idx = confirm_idx.searchsorted(ts)
                    if idx >= len(confirm_idx):
                        continue
                    btc_expiry = float(df_confirm_1h.iloc[idx]["open"])
                actual_z_list.append(math.log(btc_expiry / spot_val) / sigma_tau)
            except Exception:
                continue
        if len(actual_z_list) < w_short:
            return 0.0
        z_short = sum(actual_z_list[-w_short:]) / w_short
        z_long  = sum(actual_z_list[-w_long:]) / len(actual_z_list[-w_long:])
        return float(max(-cap, min(cap, alpha * z_short + (1 - alpha) * z_long)))
    except Exception:
        return 0.0


def get_csv_path(asset: str = "BTC", shadow: bool = False) -> Path:
    """Return the asset-specific paper trades CSV path.

    shadow=True: pure paper runner writes here so it doesn't pollute the live/dual CSV.
    """
    asset = asset.upper()
    base = Path(__file__).parent / "results"
    if shadow:
        if asset == "BTC":
            return base / "paper_trades_shadow.csv"
        return base / f"paper_trades_{asset.lower()}_shadow.csv"
    if asset == "BTC":
        return PAPER_TRADES_CSV  # keep existing BTC file unchanged
    return base / f"paper_trades_{asset.lower()}.csv"
DEFAULT_BANKROLL  = 1_000.0

CSV_COLUMNS = [
    "logged_at",
    "decision_time",
    "contract_ticker",
    "close_ts",
    "spot",
    "strike",
    "offset_pct",
    "p_market",
    "p_market_source",
    "p_yes_model",
    "z_score",
    "vol_60m",
    "vol_60m_model",
    "vol_implied_kalshi",
    "vol_ratio",
    "spread",
    "vol_eff",
    "structure_bias",
    "confirmation_bias",
    "confirmation_score",
    "no_score",
    "obi_score",
    "obi_raw",
    "obi_exchanges",
    "vpin_score",
    "vpin_raw",
    "funding_bias",
    "avg_funding_rate",
    "vol_score",
    "cmf_raw",
    "cmf_score",
    "vwap_score",
    "vwap_signal",
    "vwap_total",
    "vwap_stretch_score",
    "vwap_distance_pct",
    "bearish_rejection",
    "bullish_rejection",
    "ema_stretch_score",
    "stoch_bias",
    "stoch_k",
    "stoch_k_4h",
    "stoch_d",
    "stoch_crossover_active",
    "ema_stack_bias",
    "ema_alignment",
    "z_shift",
    "direction_strength",
    "raw_edge",
    "net_edge",
    "decision",
    "side",
    "neutral_gate",    # True if trade passed via neutral structure path (+0.02 edge premium)
    "pure_edge_gate",  # True if trade passed via pure-edge override (Gate P, 1/8 Kelly)
    "contracts_scanned",  # number of contracts with real bid/ask evaluated at this decision point
    "tau_minutes",        # minutes to expiry at decision time (used in probability engine)
    "gate_blocked",       # which gate blocked a no_trade (Gate 1/2/3); empty for trades
    "kelly_fraction",
    "bet_fraction",
    "bet_amount",
    "bankroll",
    "composite_trend",    # trend score from composite_scorer (-6 to +6)
    "composite_rev",      # reversion score from composite_scorer (-15 to +15)
    "composite_p_up",     # calibrated directional probability from composite scorer (lookup table)
    "p_up_v2",            # BTC p_up v2 LightGBM model value (overrides composite_p_up for BTC)
    "chg_30m",            # 30-minute price change fraction at decision time
    "chg_10m",            # 10-minute price change fraction at decision time
    "chg_5m",             # 5-minute price change fraction at decision time
    "bp_5m",              # buying pressure on last completed 5m bar: (close-low)/(high-low)
    "bp_1h",              # buying pressure on last completed 1h bar: (close-low)/(high-low)
    "chg_1h",             # 1-hour close pct-change (%) at decision time
    "chg_2h",             # 2-hour cumulative pct-change (%) — sustained rally/selloff detection
    "chg_3h",             # 3-hour cumulative pct-change (%) — regime momentum
    "body_15m",           # body ratio on last completed 15m bar: |close-open|/(high-low)
    "dir_15m",            # direction of last completed 15m bar: +1=bullish, -1=bearish
    "p_gbdt",             # BTC/ETH/SOL LGBM shadow model probability [SHADOW — gate eval after 2,000+ scan archive rows + retrain]
    "sharp_move_active",  # True if sharp move inversion was applied this cycle
    "smc_4h",             # SMC 4h structure: bullish / bearish / neutral
    "smc_1h",             # SMC 1h structure: bullish / bearish / neutral
    "choch_1h",           # True if 1h ChoCH fired in the last 5 bars (regime flip)
    "choch_4h",           # True if 4h ChoCH fired in the last 3 bars (regime flip)
    "supply_pct",         # % above nearest supply zone (None if no zone)
    "demand_pct",         # % below nearest demand zone (None if no zone)
    "in_supply_zone",     # True if price is currently inside a supply zone
    "in_demand_zone",     # True if price is currently inside a demand zone
    "stoch_flipped",      # retained for backward compatibility
    "squeeze_1h",         # True if BB width < KC width (volatility compression before breakout)
    "adx_1h",            # 14-period ADX on 1h bars (trend strength; >25=trending, <20=ranging)
    "rvol_1h",           # relative volume: current 1h vol / 30-bar avg for this UTC hour
    "pm_drift_5m",       # p_market change over last 5 minutes for this contract
    "hour_utc",          # UTC hour of decision (0-23) — calendar seasonality analysis
    "liq_score",         # Coinalyze: composite liquidation+positioning score (-2 to +2)
    "liq_bias",          # Coinalyze: (short_liqs - long_liqs) / total_liqs; +1=squeeze, -1=cascade
    "ls_long_pct",       # Coinalyze: % of open perp positions that are long (crowding signal)
    "oi_chg_pct",        # Coinalyze: open interest % change over last completed 15m bar
    "arima_forecast_1h",    # ARIMA(2,0,1) 1-step-ahead forecast of next 1h log return [SHADOW — gate eval after 500+ obs; fixed 2026-06-02 (disp arg)]
    "markov_regime_daily",   # BTC 3-state HMM: Bull / Bear / Sideways
    "markov_regime_7state",  # BTC 7-state HMM: Correction / Consolidation / Bull / etc.
    "ob_imbalance",      # Coinbase spot OB: (bid-ask)/(bid+ask) in 0.5% window around strike [SHADOW — gate eval after 200+ obs]
    "ob_path_ask_usd",   # USD ask notional between spot and strike (OTM YES resistance to clear) [SHADOW — same batch as ob_imbalance]
    "ob_path_bid_usd",   # USD bid notional between strike and spot (ITM YES / OTM NO floor support) [SHADOW — same batch as ob_imbalance]
    "ob_ask_frac",       # ask_mass at strike / total book ask (normalized resistance) [SHADOW — same batch as ob_imbalance]
    "ob_bid_wall_pct",   # distance to nearest $500k+ bid wall below spot (fraction of spot, negative) [SHADOW — same batch as ob_imbalance]
    "ob_ask_wall_pct",   # distance to nearest $500k+ ask wall above spot (fraction of spot, positive) [SHADOW — same batch as ob_imbalance]
    "hmm_vol_state",     # hard Viterbi rank: 0=R0 low-vol, 1=R1 high-vol
    "hmm_r1_prob",       # soft posterior P(R1|data) 0-1; catches transitions ~5 bars early [SHADOW]
    "hmm_vol_k10",       # 10-step forward P(R1): posterior @ P^10 [SHADOW]
    "hmm_time_in_state", # sojourn depth in bars: early=1-3 (spike), mid=4-15, deep=16+ (committed) [SHADOW]
    "ou_z_score",        # OU AR(1) fit: (spot - ou_mean) / ou_sigma; +ve=extended up, -ve=extended down [SHADOW]
    "ou_halflife_min",   # OU expected reversion half-life in minutes [SHADOW]
    "ou_tau_drift",      # OU expected log-return over contract tau: mu+(spot-mu)*exp(-theta*tau_h); tau-aware [SHADOW]
    "hs_pattern_type",   # most recent H&S pattern on 1h: 'hs' (bearish) or 'ihs' (bullish) [SHADOW — gate eval after 200+ obs]
    "hs_bars_since_break", # 1h bars elapsed since that pattern broke [SHADOW]
    "hs_r2",             # H&S pattern R² fit quality [SHADOW]
    "hs_neck_slope",     # H&S neckline slope (log price / bar) [SHADOW]
    "hs_head_height",    # H&S head height in log price units [SHADOW]
    "hs_head_width",     # H&S head width in bars [SHADOW]
    "flag_signal",        # +1=recent bull flag/pennant, -1=bear, 0=none [SHADOW — gate eval after 200+ obs]
    "flag_bull_bars_ago", # 1h bars since last confirmed bull flag/pennant (-1=none in lookback)
    "flag_bear_bars_ago", # 1h bars since last confirmed bear flag/pennant (-1=none)
    "flag_bull_tip_y",    # real price at top of bull pole
    "flag_bear_tip_y",    # real price at bottom of bear pole
    "flag_bull_pole_pct", # bull pole height as % of base price
    "flag_bear_pole_pct", # bear pole depth as % of base price
    "pip_last_slope",     # slope of last PIP segment on 1h (log-price/bar); +ve=up, -ve=down [SHADOW]
    "pip_up_frac",        # fraction of total PIP amplitude from upward legs [0,1] [SHADOW]
    "pip_n_turns",        # direction changes in 5-PIP skeleton [0,4] [SHADOW]
    "v_hawk",             # Hawkes vol intensity on 1h norm_range (kappa=0.01, lb=336) [SHADOW — gate eval after 200+ elevated-regime trades]
    "hawk_vol_regime",    # rolling vol regime: quiet/low/mid/elevated/spike (q25/50/75/90 thresholds) [SHADOW]
    "pc1_rsi",            # PC1 RSI divergence score: fast-vs-slow RSI (2-24 periods); gate fires NO block when <= -34.93
    "resolved_yes",   # filled by outcome_checker.py
    "would_win",      # filled by outcome_checker.py
    "would_pnl",      # filled by outcome_checker.py
    "spot_at_expiry", "price_move_pct", "miss_pct",  # filled by outcome_checker.py
    "loss_margin_pct", "loss_category",              # filled by outcome_checker.py; tau-scaled quality labels
]


def ensure_csv_exists(csv_path: Path = None) -> None:
    path = csv_path or PAPER_TRADES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"  Created {path}")
        return
    # Migrate if the file's header is missing any columns in CSV_COLUMNS
    with open(path, newline="") as f:
        existing_cols = (csv.DictReader(f).fieldnames or [])
    new_cols = [c for c in CSV_COLUMNS if c not in existing_cols]
    if new_cols:
        print(f"  [migrate] Adding columns to {path.name}: {new_cols}")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in new_cols:
                row.setdefault(col, "")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [migrate] Migrated {len(rows)} rows.")


def append_row(row: dict, csv_path: Path = None) -> None:
    path = csv_path or PAPER_TRADES_CSV
    # Sanitize string values: newlines in a field break CSV row alignment.
    clean = {k: (v.replace("\n", " ").replace("\r", " ") if isinstance(v, str) else v)
             for k, v in row.items()}
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(clean)
    print(f"  Logged → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market (no auth needed)")
    parser.add_argument("--asset", type=str, default=None, required=True,
                        help="Asset to trade: BTC, ETH, or SOL (required)")
    parser.add_argument("--live", action="store_true",
                        help="Place real orders on Kalshi (default: paper-trade only)")
    parser.add_argument("--dual", action="store_true",
                        help="Single process: fetch data once, log to both paper and live CSVs, place real orders")
    parser.add_argument("--daily-loss-limit", type=float, default=None,
                        help="Max dollars to lose live per calendar day before halting "
                             "(defaults: BTC=$250, ETH=$120, SOL=$120)")
    parser.add_argument("--max-contracts", type=int, default=500,
                        help="Hard cap on contracts per live order (default: 500 — size controlled by Kelly dollar amount)")
    args = parser.parse_args()
    args.asset = args.asset.upper()
    if args.daily_loss_limit is None:
        args.daily_loss_limit = {"BTC": 250.0, "ETH": 120.0, "SOL": 120.0}.get(args.asset, 120.0)

    now_utc = datetime.now(timezone.utc)
    print(f"\n  Run time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- US-session volatility filter (per-asset) ---
    # Bad hours derived from paper trade P&L by UTC hour (empirical, updated 2026-05-04).
    # BTC: 07 (48.9% WR, -$386), 16/17 (55-56% WR, -$168/-$220), 22 (52% WR, -$294)
    # ETH: 00/01 (60-67% WR, -$66/-$108), 16 (47% WR, -$184)
    # SOL: 15 (53% WR, -$153), 18 (59% WR, -$110)
    # ETH hour 16 removed from SKIP_HOURS 2026-05-17: NO bets at 16UTC have WR=87.5%, Edge=+21.2%, p=0.000.
    # BTC hour 22 removed from SKIP_HOURS 2026-06-03: 5 wins missed ($1,404) during May 25-27
    #   live outage; original -$294 figure is stale. Revisit after 30+ hour-22 live observations.
    # YES bets at hour 16 are now blocked per-contract by hour_yes_gate (all assets).
    SKIP_HOURS = {
        "BTC": set(),
        "ETH": set(),
        "SOL": set(),
    }.get(args.asset, set())
    _vol_skip_live = False
    if now_utc.hour in SKIP_HOURS and now_utc.weekday() < 5:  # 0=Mon…4=Fri; skip filter on weekends
        if args.live and not getattr(args, 'dual', False):
            # Pure live mode: skip entirely
            print(f"  [vol-filter] Skipping — UTC hour {now_utc.hour} is in high-volatility window {SKIP_HOURS}.")
            return
        elif getattr(args, 'dual', False):
            # Dual mode: skip live order but continue for paper data collection
            _vol_skip_live = True
            print(f"  [vol-filter] Skipping live order — UTC hour {now_utc.hour} in {SKIP_HOURS}. Paper continues.")
        else:
            print(f"  [vol-filter] PAPER continuing — collecting data in high-volatility window {SKIP_HOURS}.")

    _is_live_or_dual = args.live or getattr(args, 'dual', False)
    if _is_live_or_dual:
        _mode_label = "DUAL" if getattr(args, 'dual', False) else "LIVE"
        print(f"  *** {_mode_label} MODE *** daily_loss_limit=${args.daily_loss_limit:.0f}  max_contracts={args.max_contracts}")

    # Auth
    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            if _is_live_or_dual:
                print("  ERROR: --live/--dual requires Kalshi credentials. Set KALSHI_KEY_ID / KALSHI_KEY_PATH.")
                return
            print("  WARNING: No Kalshi credentials — using simulated p_market.")

    # Load OHLCV
    cfg        = ASSET_CONFIG.get(args.asset, ASSET_CONFIG["BTC"])
    tau        = cfg.get("tau", TAU)
    ema_fast   = cfg.get("ema_fast", 20)
    ema_slow   = cfg.get("ema_slow", 50)
    rsi_period = cfg.get("rsi_period", 21)
    vol_bars   = cfg.get("vol_lookback_bars", 60)
    confirm_iv = cfg.get("confirmation_interval", "1h")

    print(f"  Loading OHLCV data ({args.asset})...")
    df_vol, df_confirm, df_struct = load_data(asset=args.asset)

    ts = df_confirm.index[-1]

    # --- Hawkes vol intensity (BTC only; shadow log for gate research) ---
    _v_hawk_val, _hawk_vol_regime_val = (float("nan"), "")
    _pc1_rsi_val = float("nan")
    if args.asset == "BTC":
        _v_hawk_val, _hawk_vol_regime_val = _compute_v_hawk(df_confirm)
        if not math.isnan(_v_hawk_val):
            print(f"  [v_hawk] {_v_hawk_val:.5f}  regime={_hawk_vol_regime_val}")
        _pc1_rsi_val = _compute_pc1(df_confirm)
        if not math.isnan(_pc1_rsi_val):
            _gate_str = " [GATE: NO block]" if _pc1_rsi_val <= _PC1_Q10 else ""
            print(f"  [pc1_rsi] {_pc1_rsi_val:.4f}{_gate_str}")

    # --- Relative Volume 1h (RVOL) ---
    # Current 1h bar volume vs the 30-bar historical average for this UTC hour.
    # Captures whether this specific hour is busier or quieter than usual — distinct
    # from vol_score which compares to recent bars regardless of time-of-day.
    _rvol_1h = float("nan")
    _sk4h_bounce = float("nan")
    _hmm_mtf_state    = -1      # -1 = model unavailable / BTC only
    _macd_hist_1h_mtf = float("nan")
    _bp_1h_mtf  = float("nan")
    _chg_1h_mtf = 0.0
    _chg_2h_mtf = 0.0
    _chg_3h_mtf = 0.0
    try:
        _same_hour_vol = df_confirm[df_confirm.index.hour == now_utc.hour]["volume"]
        _avg_vol_hour  = float(_same_hour_vol.iloc[:-1].tail(30).mean())
        _cur_vol_1h    = float(df_confirm["volume"].iloc[-1])
        if _avg_vol_hour > 0:
            _rvol_1h = round(_cur_vol_1h / _avg_vol_hour, 4)
    except Exception:
        pass

    # --- VSA pressure (BTC only) — detects bullish absorption failures for NO flip ---
    # pressure_24 = 24h cumulative signed volume-spread z-score.
    # Threshold _VSA_P24_Q80 = 4.73 (empirical q80 of YES contracts, archive 2026-05/06).
    # Gate: sdz>=2.0 + p24>=q80 + p_market>0.50 → block YES, flip to NO.
    # Backtest: n=116 distinct contracts, WR_no=51.7% vs BEV=8.6% (+43.1% edge), MCPT z=+14, p=0.000.
    # Decision tree (per contract): skip if rvol>=1.5+ct>0 OR cpu>=0.60 OR pm_no<0.03.
    # Tiered sizing: cpu<0.40→$25 face, cpu<0.50→$15 face, else→$10 face.
    _VSA_P24_Q80     = 4.73
    _VSA_SDZ_MIN     = 2.0
    _VSA_PM_NO_MIN   = 0.03   # Kalshi liquidity floor for NO flip
    _vsa_pressure24  = float("nan")
    _vsa_sdz         = float("nan")
    if args.asset == "BTC" and len(df_confirm) >= 504:
        try:
            import scipy.stats as _scipy_stats
            import numpy as _np_vsa
            _vdf = df_confirm[["high", "low", "close", "volume"]].astype(float).copy()
            _vp  = _vdf["close"].shift(1)
            _vtr = pd.concat([_vdf["high"]-_vdf["low"],
                               (_vdf["high"]-_vp).abs(),
                               (_vdf["low"]-_vp).abs()], axis=1).max(axis=1)
            _vdf["_nr"] = (_vdf["high"]-_vdf["low"]) / _vtr.rolling(168).mean()
            _vdf["_nv"] = _vdf["volume"] / _vdf["volume"].rolling(168).median()
            _vdf["_cp"] = _np_vsa.where(
                (_vdf["high"]-_vdf["low"]) > 0,
                (_vdf["close"]-_vdf["low"]) / (_vdf["high"]-_vdf["low"]), 0.5)
            _nr_v = _vdf["_nr"].to_numpy(); _nv_v = _vdf["_nv"].to_numpy()
            _dev_v = _np_vsa.full(len(_vdf), _np_vsa.nan)
            for _vi in range(336, len(_vdf)):
                _vw = _vdf.iloc[_vi-167:_vi+1]
                _vm = _vw["_nr"].notna() & _vw["_nv"].notna()
                if _vm.sum() < 20:
                    continue
                _vsl, _vic, _vrv, _, _ = _scipy_stats.linregress(
                    _vw.loc[_vm, "_nv"], _vw.loc[_vm, "_nr"])
                if _vsl <= 0 or _vrv < 0.2:
                    _dev_v[_vi] = 0.0
                    continue
                _dev_v[_vi] = _nr_v[_vi] - (_vic + _vsl * _nv_v[_vi])
            _vdf["_dev"] = _dev_v
            _vdf["_sd"]  = -_vdf["_dev"] * (2 * _vdf["_cp"] - 1)
            _std_v = _vdf["_sd"].rolling(168).std()
            _vdf["_sdz"] = _vdf["_sd"] / _std_v
            _vdf["_p24"] = _vdf["_sdz"].rolling(24).sum()
            _vp24_last = float(_vdf["_p24"].iloc[-1])
            _vsdz_last = float(_vdf["_sdz"].iloc[-1])
            if not math.isnan(_vp24_last):
                _vsa_pressure24 = _vp24_last
            if not math.isnan(_vsdz_last):
                _vsa_sdz = _vsdz_last
            print(f"  [vsa] pressure_24={_vsa_pressure24:.2f}  sdz={_vsa_sdz:.2f}"
                  f"  (threshold: p24>={_VSA_P24_Q80}, sdz>={_VSA_SDZ_MIN})")
        except Exception as _ve:
            print(f"  [vsa] compute error: {_ve}")

    # Live spot
    live_spot = fetch_live_spot(asset=args.asset)
    spot = live_spot if live_spot is not None else float(df_confirm["close"].iloc[-1])

    # Signals
    hist_confirm = df_confirm.iloc[-100:]
    hist_struct  = df_struct.iloc[-120:]

    # Fetch fresh 1m candles for realized vol
    # BTC needs 1700 bars: 1440 (σ_kalshi window) + 120 (lag) + buffer for Gate VR
    _1m_lookback = 1700 if args.asset == "BTC" else max(vol_bars * 2, 800)
    live_1m = fetch_recent_1m_candles(lookback_bars=_1m_lookback, asset=args.asset)
    vol_src = live_1m if live_1m is not None and len(live_1m) >= vol_bars else df_vol.iloc[-200:]
    vol     = compute_realized_volatility(vol_src)

    # --- Gate VR (BTC only): vol_ratio = σ_model / σ_kalshi > 1.20 → skip scan ---
    # σ_model  = 60-bar rolling std of 1m log returns (current realized vol)
    # σ_kalshi = 1440-bar rolling std of 1m log returns, lagged 120 bars
    #            (simulates Kalshi's 24h implied vol with ~2h delayed update)
    # When σ_model > σ_kalshi, current vol has spiked above Kalshi's estimate.
    # In this regime: NO bets are not cheap; edge flips against us.
    # Out-of-sample backtest (Jan 2025–Apr 2026):
    #   vol_ratio < 1.20 → 89.8% win rate, +$26,212   (16/16 months profitable)
    #   vol_ratio > 1.20 → 22.9% win rate, -$15,420   (5/16 months profitable)
    if args.asset == "BTC" and live_1m is not None and len(live_1m) >= 1600:
        import numpy as _np
        _closes = live_1m["close"].values.astype(float)
        _lr = pd.Series(_np.diff(_np.log(_np.maximum(_closes, 1e-8)), prepend=0.0))
        _sig_m = float(_lr.rolling(60).std().iloc[-1])
        _sig_k = float(_lr.rolling(1440).std().iloc[-121])  # 120-bar lag
        _vr = _sig_m / _sig_k if _sig_k > 0 else 0.0
        print(f"  [Gate VR] BTC vol_ratio={_vr:.3f} (σ_model/σ_kalshi, threshold=1.20)")
        if _vr > 1.20:
            print(f"  [Gate VR] BLOCKED — current vol > Kalshi's lagged vol. Edge flipped. Skipping BTC scan.")
            return
    struct  = detect_market_structure(hist_struct)
    obi     = fetch_order_book_imbalance(asset=args.asset)
    print(f"  OBI: {obi.obi:+.4f}  score={obi.obi_score:+d}  exchanges={obi.exchanges_used}")

    # Fetch funding rate — cached for 5 minutes since it updates every 8 hours.
    # Falls back to neutral (funding_bias=0) on failure; never crashes the loop.
    global _funding_cache, _funding_cache_ts
    import time as _time
    if _funding_cache is None or (_time.time() - _funding_cache_ts) > _FUNDING_CACHE_TTL:
        try:
            _funding_cache    = fetch_funding_rate(asset=args.asset)
            _funding_cache_ts = _time.time()
        except Exception as exc:
            print(f"  [funding] Fetch error: {exc} — using neutral fallback")
            from funding_rate import _FALLBACK
            _funding_cache    = _FALLBACK
            _funding_cache_ts = _time.time()
    funding = _funding_cache
    print(f"  Funding: {funding.avg_funding_rate*100:+.4f}%/8h  bias={funding.funding_bias:+d}  ({', '.join(funding.exchanges_used) or 'none'})")

    confirm = compute_confirmation(hist_confirm, hist_1m=live_1m, obi_score=obi.obi_score, momentum_enabled=False,
                                   funding_bias=funding.funding_bias, avg_funding_rate=funding.avg_funding_rate)

    # --- Composite directional scores ---
    # Compute validated trend (4h) + reversion (1h/15m) scores from historical data.
    # These replace the unvalidated confirmation_score/no_score in the contract loop.
    _comp_trend, _comp_rev = 0, 0
    _comp_trend_30m, _comp_rev_30m = 0, 0
    _comp_trend_4h, _comp_rev_4h = 0, 0
    _asset_baseline = {"BTC": 0.504, "ETH": 0.509, "SOL": 0.500}.get(args.asset, 0.504)
    _comp_p_up = _asset_baseline
    _composite_computed = False
    # _df_4h_comp computed unconditionally so SMC can always use it
    _df_4h_comp = None
    try:
        _df_4h_comp = df_confirm.resample("4h", origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])
    except Exception:
        pass
    if live_1m is not None and len(live_1m) >= 400:
        try:
            _df_15m_comp = live_1m.resample("15min", origin="start_day").agg(
                {"high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["close"])
            _comp_trend, _comp_rev = compute_current_scores(
                df_confirm, _df_4h_comp, _df_15m_comp,
                live_1m["close"].astype(float), live_1m["volume"].astype(float),
            )
            _comp_p_up = lookup_p_up(_comp_trend, _comp_rev, asset=args.asset)
            _comp_trend_30m, _comp_rev_30m = compute_current_scores_30m(
                df_confirm, _df_15m_comp,
                live_1m["close"].astype(float), live_1m["volume"].astype(float),
            )
            _comp_trend_4h, _comp_rev_4h = compute_current_scores_90m(
                df_confirm,
                live_1m["close"].astype(float), live_1m["volume"].astype(float),
            )
            _composite_computed = True
            print(f"  [composite] trend={_comp_trend:+d}  rev={_comp_rev:+d}  p_up={_comp_p_up:.1%}"
                  f"  [30m] trend={_comp_trend_30m:+d}  rev={_comp_rev_30m:+d}"
                  f"  [4h] trend={_comp_trend_4h:+d}  rev={_comp_rev_4h:+d}")
        except Exception as _exc:
            print(f"  [composite] Score error: {_exc} — using pure log-normal fallback")

    # HMM MTF momentum state (BTC only) — classify current market regime via
    # the 9-state multi-timeframe momentum model.  Single-observation inference
    # (argmax emission) using features computed from live OHLCV data.
    if args.asset == "BTC" and _mtf_hmm_model is not None and _mtf_hmm_scaler is not None:
        try:
            _c1h_mtf = df_confirm["close"].astype(float)
            # stoch_k_1h
            _sk_1h_mtf = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else 50.0
            # rsi_1h
            _delta_1h = _c1h_mtf.diff()
            _gain_1h  = _delta_1h.clip(lower=0).ewm(span=14, adjust=False).mean()
            _loss_1h  = (-_delta_1h.clip(upper=0)).ewm(span=14, adjust=False).mean()
            _rsi_val  = (100 - 100 / (1 + _gain_1h / _loss_1h.replace(0, float("nan")))).iloc[-1]
            _rsi_1h_mtf = float(_rsi_val) if not math.isnan(float(_rsi_val)) else 50.0
            # bp_1h: (close-low)/(high-low) of last completed 1h bar
            _h1 = float(df_confirm["high"].iloc[-1]); _l1 = float(df_confirm["low"].iloc[-1])
            _bp_1h_mtf = (_c1h_mtf.iloc[-1] - _l1) / (_h1 - _l1) if (_h1 - _l1) > 0 else 0.5
            # chg_1h/2h/3h: rolling hourly momentum (% change over last N completed bars)
            _chg_1h_mtf = float(_c1h_mtf.pct_change().iloc[-1] * 100) if len(_c1h_mtf) >= 2 else 0.0
            if math.isnan(_chg_1h_mtf): _chg_1h_mtf = 0.0
            _chg_2h_mtf = float((_c1h_mtf.iloc[-1] / _c1h_mtf.iloc[-3] - 1) * 100) if len(_c1h_mtf) >= 3 else 0.0
            if math.isnan(_chg_2h_mtf): _chg_2h_mtf = 0.0
            _chg_3h_mtf = float((_c1h_mtf.iloc[-1] / _c1h_mtf.iloc[-4] - 1) * 100) if len(_c1h_mtf) >= 4 else 0.0
            if math.isnan(_chg_3h_mtf): _chg_3h_mtf = 0.0
            # macd_hist_1h (kept for rescue check below)
            _ema12_1h = _c1h_mtf.ewm(span=12, adjust=False).mean()
            _ema26_1h = _c1h_mtf.ewm(span=26, adjust=False).mean()
            _macd_1h  = _ema12_1h - _ema26_1h
            _macd_sig_1h = _macd_1h.ewm(span=9, adjust=False).mean()
            _macd_hist_1h_mtf = float((_macd_1h - _macd_sig_1h).iloc[-1])
            if math.isnan(_macd_hist_1h_mtf): _macd_hist_1h_mtf = 0.0
            # adx_1h (from confirm)
            _adx_1h_mtf = float(confirm.adx_1h) if (hasattr(confirm, "adx_1h") and
                                                     confirm.adx_1h == confirm.adx_1h) else 25.0
            # stoch_k_5m from live_1m (last completed 5m bar)
            _sk_5m_mtf = 50.0
            if live_1m is not None and len(live_1m) >= 20:
                _df5m = live_1m.resample("5min").agg(
                    {"high": "max", "low": "min", "close": "last"}).dropna()
                if len(_df5m) >= 14:
                    _ll5 = _df5m["low"].rolling(14).min()
                    _hh5 = _df5m["high"].rolling(14).max()
                    _rng5 = (_hh5 - _ll5).replace(0, float("nan"))
                    _s5 = ((_df5m["close"] - _ll5) / _rng5 * 100).iloc[-1]
                    _sk_5m_mtf = float(_s5) if not math.isnan(float(_s5)) else 50.0
            # stoch_k_15m from _df_15m_comp
            _sk_15m_mtf = 50.0
            _df_15m_mtf = locals().get("_df_15m_comp")
            if _df_15m_mtf is not None and len(_df_15m_mtf) >= 14:
                _ll15 = _df_15m_mtf["low"].rolling(14).min()
                _hh15 = _df_15m_mtf["high"].rolling(14).max()
                _rng15 = (_hh15 - _ll15).replace(0, float("nan"))
                _s15 = ((_df_15m_mtf["close"] - _ll15) / _rng15 * 100).iloc[-1]
                _sk_15m_mtf = float(_s15) if not math.isnan(float(_s15)) else 50.0
            # macd_hist_4h and adx_4h
            _macd_hist_4h_mtf = 0.0
            _adx_4h_mtf = 30.0
            if _df_4h_comp is not None and len(_df_4h_comp) >= 26:
                _c4 = _df_4h_comp["close"].astype(float)
                _h4 = _df_4h_comp["high"].astype(float)
                _l4 = _df_4h_comp["low"].astype(float)
                _e12_4 = _c4.ewm(span=12, adjust=False).mean()
                _e26_4 = _c4.ewm(span=26, adjust=False).mean()
                _m4 = _e12_4 - _e26_4
                _ms4 = _m4.ewm(span=9, adjust=False).mean()
                _mh4 = float((_m4 - _ms4).iloc[-1])
                _macd_hist_4h_mtf = _mh4 if not math.isnan(_mh4) else 0.0
                # ADX 4h
                _tr4 = pd.concat([_h4 - _l4, (_h4 - _c4.shift()).abs(),
                                   (_l4 - _c4.shift()).abs()], axis=1).max(axis=1)
                _dp4 = _h4.diff().clip(lower=0)
                _dm4 = (-_l4.diff()).clip(lower=0)
                _dp4 = _dp4.where(_dp4 > _dm4, 0)
                _dm4 = _dm4.where(_dm4 > _h4.diff().clip(lower=0), 0)
                _atr4 = _tr4.ewm(span=14, adjust=False).mean()
                _pdi4 = _dp4.ewm(span=14, adjust=False).mean() / _atr4.replace(0, float("nan")) * 100
                _mdi4 = _dm4.ewm(span=14, adjust=False).mean() / _atr4.replace(0, float("nan")) * 100
                _dx4  = ((_pdi4 - _mdi4).abs() / (_pdi4 + _mdi4).replace(0, float("nan")) * 100)
                _adx4_val = float(_dx4.ewm(span=14, adjust=False).mean().iloc[-1])
                _adx_4h_mtf = _adx4_val if not math.isnan(_adx4_val) else 30.0
            # Build feature vector and classify state
            import numpy as _np_mtf_local
            _fv = _np_mtf_local.array([[
                _sk_5m_mtf, _sk_15m_mtf, _sk_1h_mtf, _rsi_1h_mtf,
                _bp_1h_mtf, _chg_1h_mtf, _macd_hist_1h_mtf, _adx_1h_mtf,
                _macd_hist_4h_mtf, _adx_4h_mtf,
            ]])
            _fv_scaled = _mtf_hmm_scaler.transform(_fv)
            _hmm_mtf_state = int(_mtf_hmm_model.predict(_fv_scaled)[0])
        except Exception:
            _hmm_mtf_state = -1

    # 1h candle direction: last completed bar (iloc[-2]; iloc[-1] may be partial)
    _1h_candle_green = False
    _1h_candle_red   = False
    try:
        _lc1h = df_confirm.iloc[-2]
        _1h_candle_green = float(_lc1h["close"]) >= float(_lc1h["open"])
        _1h_candle_red   = not _1h_candle_green
    except Exception:
        pass

    # --- Trend Z (diagnostic) ---
    # Multi-timeframe trend strength: log-return over N bars / rolling vol (Z-score).
    # Captures sustained directional displacement at 12h, 24h, 48h horizons.
    # Positive = sustained upward pressure; negative = sustained downward pressure.
    # Logged only — no gating yet. Used to develop regime-adaptive model adjustments.
    _trend_z = float("nan")
    try:
        import numpy as _tz_np
        _tz_close = df_confirm["close"]
        _tz_log   = _tz_np.log(_tz_close / _tz_close.shift(1))
        _tzs = []
        for _N in [12, 24, 48]:
            _lr    = _tz_np.log(_tz_close.iloc[-1] / _tz_close.iloc[-1 - _N])
            _vol_N = _tz_log.iloc[-_N:].std() * (_N ** 0.5)
            if _vol_N > 0:
                _tzs.append(_lr / _vol_N)
        if _tzs:
            _trend_z = sum(_tzs) / len(_tzs)
        print(f"  [trend_z] {_trend_z:+.3f}  (12h/24h/48h composite — diagnostic only)")
    except Exception as _exc:
        print(f"  [trend_z] Error: {_exc}")

    # --- SMC signals ---
    # Smart Money Concepts: Break of Structure, Change of Character, Supply/Demand Zones.
    # 4h BOS = structural regime (persistent, changes rarely).
    # 1h BOS = tactical signal (changes within a session).
    # ChoCH: logged as persistent state (last two BOS events reversed) — see smc_signals.py.
    # Computed unconditionally (not gated on _composite_computed) so all CSV rows are populated.
    # All fields written to CSV for post-hoc correlation analysis.
    _smc = None
    if _df_4h_comp is not None:
        try:
            from smc_signals import get_smc_signals as _get_smc
            _smc = _get_smc(df_confirm, _df_4h_comp, spot)
            _choch_str = ""
            if _smc.choch_4h and _smc.choch_1h:
                _choch_str = "  *** ChoCH BOTH tf ***"
            elif _smc.choch_4h:
                _choch_str = "  * ChoCH 4h"
            elif _smc.choch_1h:
                _choch_str = "  * ChoCH 1h"
            print(f"  [smc] 4h={_smc.bos_4h}  1h={_smc.bos_1h}{_choch_str}")
            print(f"  [smc] sh_4h={_smc.swing_high_4h}  sl_4h={_smc.swing_low_4h}  "
                  f"sh_1h={_smc.swing_high_1h}  sl_1h={_smc.swing_low_1h}")
            _sup_str = f"+{_smc.nearest_supply_pct:.2f}%" if _smc.nearest_supply_pct is not None else "none"
            _dem_str = f"-{_smc.nearest_demand_pct:.2f}%" if _smc.nearest_demand_pct is not None else "none"
            _zone_flags = []
            if _smc.in_supply_zone:
                _zone_flags.append("IN_SUPPLY")
            if _smc.in_demand_zone:
                _zone_flags.append("IN_DEMAND")
            _zone_str = "  [" + ", ".join(_zone_flags) + "]" if _zone_flags else ""
            print(f"  [smc] supply={_sup_str} ({_smc.n_supply_zones} zones)  "
                  f"demand={_dem_str} ({_smc.n_demand_zones} zones){_zone_str}")
        except Exception as _smc_exc:
            print(f"  [smc] Error: {_smc_exc}")
            _smc = None

    # HMM SMC phase state — predicted once per scan from rolling 24h buffer
    _hmm_smc_state = _hmm_smc_predict(_smc) if (args.asset == "BTC" and _smc is not None) else -1
    if _hmm_smc_state >= 0:
        print(f"  [hmm_smc] state={_hmm_smc_state}  ({_HMM_SMC_STATE_LABELS.get(_hmm_smc_state, '?')})")

    # --- Vol regime factor ---
    # Scales blended sigma before score_to_p_model. Validated on 19,947h of OHLCV data.
    # High-vol regime → factor > 1.0 → wider sigma → OTM strikes more reachable.
    # Low-vol regime  → factor < 1.0 → tighter sigma → edge concentrates near ATM.
    _markov_regime  = _get_daily_markov_regime(args.asset)  # fetched early — needed by garch_markov_vol_adjust below
    _markov_7state  = _get_btc_7state_regime() if args.asset == "BTC" else None
    _vol_factor = 1.0
    _vol_score_dir = 0
    if live_1m is not None and len(live_1m) >= 400:
        try:
            _vol_factor, _vol_score_dir, _vol_details = compute_vol_regime_factor(df_confirm, live_1m, asset=args.asset)
            print(f"  [vol_layer] score={_vol_score_dir:+d}  factor={_vol_factor:.3f}  {_vol_details.get('votes', {})}")
        except Exception as _exc:
            print(f"  [vol_layer] Error: {_exc} — using factor=1.0")

    # [garch_markov_vol_adjust] BTC only: deflate sigma by one vote step when GARCH ratio is
    # suppressed (<0.67) AND daily Markov is Sideways — the quietest vol regime across all assets
    # (BigMove% = 11.8% vs 19.8% baseline). In this regime the model overestimates sigma,
    # accepting YES bets that are below breakeven (paper_trades: n=146 YES, WR=43.8% vs BE=48.9%,
    # -$7.34). NOT applied to ETH/SOL: both are profitable in LOW+Sideways (ETH +$15.29, SOL n=13).
    if args.asset == "BTC":
        _garch_ratio_vol = _get_garch_ratio(df_confirm, "BTC")
        if (_garch_ratio_vol is not None
                and _garch_ratio_vol < 0.67
                and _markov_regime == "Sideways"):
            from vol_layer import VOL_VOTE_STEP, VOL_FACTOR_MIN
            _vol_factor_pre = _vol_factor
            _vol_factor = max(VOL_FACTOR_MIN, _vol_factor - VOL_VOTE_STEP)
            _vol_score_dir -= 1
            print(f"  [garch_markov_vol] BTC LOW+Sideways → σ deflated "
                  f"(ratio={_garch_ratio_vol:.3f}, factor {_vol_factor_pre:.3f}→{_vol_factor:.3f})")

    # --- Deribit IV (BTC/ETH only) ---
    # Fetch once per scan; 5-min cache. Replaces noisy Kalshi back-computed IV as the
    # implied-vol input to blend_vol(). SOL returns None — pipeline falls back to realized.
    _deribit_dvol = deribit_iv.fetch_dvol(args.asset)
    _deribit_sigma_per_min = (deribit_iv.dvol_to_sigma_per_min(_deribit_dvol)
                              if _deribit_dvol is not None else None)
    if _deribit_dvol is not None:
        print(f"  [deribit_iv] DVOL={_deribit_dvol*100:.1f}%  σ/min={_deribit_sigma_per_min:.6f}")
    else:
        print(f"  [deribit_iv] unavailable — falling back to Kalshi implied vol")

    # --- Liquidation signal (BTC/ETH only, Coinalyze) ---
    # Fetched once per scan, 5-min cache. liq_score: +2 = strong short squeeze (bullish),
    # -2 = strong long cascade (bearish). Used as a rescue signal in bearish gates.
    _liq_signal = coinalyze_liq.fetch_liq_signal(args.asset)
    if _liq_signal is not None:
        print(f"  [liq_signal] bias={_liq_signal.liq_bias:+.2f}  "
              f"long={_liq_signal.ls_long_pct:.1f}%  short={_liq_signal.ls_short_pct:.1f}%  "
              f"score={_liq_signal.liq_score:+d}  [{_liq_signal.label}]")
    else:
        print(f"  [liq_signal] unavailable")

    # --- CoinGlass signals: exchange flows, options OI, spot taker, fear & greed ---
    _cg = coinglass_data.fetch_coinglass_signals(args.asset)
    if _cg is not None:
        print(f"  [coinglass] flow={_cg.exchange_flow_1d:+.0f} ({_cg.exchange_flow_1d_pct:+.3f}%/d)"
              f"  options_oi_chg={_cg.options_oi_change_24h:+.1f}%"
              f"  taker={_cg.spot_taker_ratio:.3f}"
              f"  F&G={_cg.fg_value:.0f}({_cg.fg_regime})"
              f"  composite={_cg.composite_score:+d}")
    else:
        print(f"  [coinglass] unavailable")

    # --- Sharp move detection ---
    # Compute 30-minute and 10-minute price changes from live 1m candles.
    # During sharp rallies the composite lags (1h/4h data) and generates NO edge
    # from reversion signals while price is actually continuing up — and vice versa.
    # Gate: block the counter-trend bet unless edge >= 8% override.
    #   Sharp rally (chg > +thresh) → skip NO  (continuation, not reversion)
    #   Sharp drop  (chg < -thresh) → skip YES (continuation, not reversion)
    # Two windows are checked: 30m (catches sustained moves) and 10m (catches
    # sharp moves masked by a prior move in the opposite direction within the 30m
    # window, e.g. a rally then sharp drop netting only ~0% over 30m).
    _sharp_move_pct = 0.0
    _sharp_move_pct_10m = 0.0
    _sharp_move_pct_5m = 0.0
    if live_1m is not None and len(live_1m) >= 31:
        try:
            _sm_close = live_1m["close"].astype(float)
            _sharp_move_pct = float(_sm_close.iloc[-1] / _sm_close.iloc[-31] - 1)
            if len(_sm_close) >= 11:
                _sharp_move_pct_10m = float(_sm_close.iloc[-1] / _sm_close.iloc[-11] - 1)
            if len(_sm_close) >= 6:
                _sharp_move_pct_5m = float(_sm_close.iloc[-1] / _sm_close.iloc[-6] - 1)
        except Exception:
            pass
    # For ETH/SOL: also fetch BTC 1m and check BTC's own sharp move thresholds.
    # If BTC fires, propagate the same direction to the alt (BTC leads).
    _btc_sharp_up = False
    _btc_sharp_down = False
    if args.asset in ("ETH", "SOL"):
        try:
            _btc_1m = fetch_recent_1m_candles(lookback_bars=35, asset="BTC")
            if _btc_1m is not None and len(_btc_1m) >= 31:
                _btc_close = _btc_1m["close"].astype(float)
                _btc_chg_30m = float(_btc_close.iloc[-1] / _btc_close.iloc[-31] - 1)
                _btc_chg_10m = float(_btc_close.iloc[-1] / _btc_close.iloc[-11] - 1) \
                               if len(_btc_close) >= 11 else 0.0
                _btc_sharp_up   = _btc_chg_30m > 0.008 or _btc_chg_10m > 0.005
                _btc_sharp_down = _btc_chg_30m < -0.008 or _btc_chg_10m < -0.005
                if _btc_sharp_up or _btc_sharp_down:
                    _btc_dir = "rally" if _btc_sharp_up else "drop"
                    _btc_win = "10m" if (abs(_btc_chg_10m) >= 0.005 and abs(_btc_chg_30m) < 0.008) else "30m"
                    _btc_pct = _btc_chg_10m if _btc_win == "10m" else _btc_chg_30m
                    print(f"  [sharp_move] BTC {_btc_pct*100:+.2f}% {_btc_win} — leading {_btc_dir} detected for {args.asset}")
        except Exception:
            pass
    _SHARP_THRESHOLDS     = {"BTC": 0.008, "ETH": 0.015, "SOL": 0.020}
    _SHARP_THRESHOLDS_10M = {"BTC": 0.005, "ETH": 0.010, "SOL": 0.013}
    _sharp_thresh     = _SHARP_THRESHOLDS.get(args.asset, 0.008)
    _sharp_thresh_10m = _SHARP_THRESHOLDS_10M.get(args.asset, 0.005)
    _sharp_up   = (_sharp_move_pct >  _sharp_thresh or _sharp_move_pct_10m >  _sharp_thresh_10m
                   or _btc_sharp_up)
    _sharp_down = (_sharp_move_pct < -_sharp_thresh or _sharp_move_pct_10m < -_sharp_thresh_10m
                   or _btc_sharp_down)
    _sharp_move_active = _sharp_up or _sharp_down
    # When a sharp move is detected, invert the composite scores before feeding
    # into the pipeline.  The composite uses 1h/4h data and lags sharp moves —
    # its "reversion" signal is systematically wrong in those periods.
    # Negating (trend, rev) flips p_up through the calibrated lookup table,
    # which reverses the drift term in score_to_p_model and swaps YES/NO bias.
    if _sharp_move_active and _composite_computed:
        _active_trend = -_comp_trend
        _active_rev   = -_comp_rev
        _direction    = "rally" if _sharp_up else "drop"
        _asset_fired  = (_sharp_move_pct >= _sharp_thresh or
                         _sharp_move_pct_10m >= _sharp_thresh_10m or
                         _sharp_move_pct <= -_sharp_thresh or
                         _sharp_move_pct_10m <= -_sharp_thresh_10m)
        _trigger_src    = args.asset if _asset_fired else "BTC"
        _trigger_window = "10m" if (abs(_sharp_move_pct_10m) >= _sharp_thresh_10m and
                                    abs(_sharp_move_pct) < _sharp_thresh) else "30m"
        _trigger_pct    = _sharp_move_pct_10m if _trigger_window == "10m" else _sharp_move_pct
        print(f"  [sharp_move] {_trigger_pct*100:+.2f}% {_trigger_window} ({_trigger_src}) — sharp {_direction} detected, inverting composite scores ({_comp_trend:+d},{_comp_rev:+d}) → ({_active_trend:+d},{_active_rev:+d})")
    else:
        _active_trend = _comp_trend
        _active_rev   = _comp_rev

    # --- Funding rate probability adjustment ---
    # Nudge p_yes_model ±1.5% based on funding bias before edge calculation.
    # Bullish funding (overcrowded shorts → squeeze): p_yes up → YES edge grows.
    # Bearish funding (overcrowded longs → unwind): p_yes down → NO edge grows.
    # Applied symmetrically — does not hardcode a directional preference.
    FUNDING_P_YES_DELTA = 0.015
    funding_delta = FUNDING_P_YES_DELTA * funding.funding_bias
    if funding_delta != 0:
        print(f"  Funding adj: p_yes {'+' if funding_delta > 0 else ''}{funding_delta:.3f} (bias={funding.funding_bias:+d})")

    gate_side = "yes" if struct.structure_bias == 1 else "no"

    # --- 30m streak trend gate (BTC only) ---
    # Block YES when 2 consecutive bearish 30m closes and stoch_k <= 70.
    # Block NO when 2 consecutive bullish 30m closes and stoch_k in [30, 60].
    # Excludes the last (possibly incomplete) 30m bar; uses the 2 bars before it.
    _streak30 = None  # 'bearish', 'bullish', or None
    if args.asset == "BTC" and live_1m is not None and len(live_1m) >= 62:
        try:
            _df30_close = live_1m['close'].resample('30min').last()
            _chg30 = _df30_close.pct_change()
            _last2 = _chg30.iloc[-3:-1]
            if len(_last2) == 2:
                if all(x < -0.0005 for x in _last2):
                    _streak30 = 'bearish'
                elif all(x > 0.0005 for x in _last2):
                    _streak30 = 'bullish'
        except Exception as _exc:
            print(f"  [streak_gate] Error computing 30m streak: {_exc}")
    if _streak30:
        _stoch_k_disp = f"{confirm.stoch_k:.1f}" if confirm.stoch_k == confirm.stoch_k else "NaN"
        print(f"  [streak_gate] BTC 30m streak: {_streak30} | stoch_k={_stoch_k_disp}")

    # --- 5m ADX + lower-highs/lows signals (BTC YES OTM downtrend gate) ---
    # Computed once per scan; used inside the contract loop below.
    _adx_5m   = None   # float or None
    _di_p_5m  = None
    _di_m_5m  = None
    _lower_hl = False  # True if lower-high AND lower-low in last 40 1m bars vs prior 20

    if args.asset == "BTC" and live_1m is not None and len(live_1m) >= 40:
        try:
            _h_arr = live_1m["high"].values.astype(float)
            _l_arr = live_1m["low"].values.astype(float)
            _lower_hl = (
                float(_h_arr[-20:].max()) < float(_h_arr[-40:-20].max()) and
                float(_l_arr[-20:].min()) < float(_l_arr[-40:-20].min())
            )
        except Exception as _exc:
            print(f"  [adx5_gate] LHL error: {_exc}")

    if args.asset == "BTC" and live_1m is not None and len(live_1m) >= 75:
        try:
            import numpy as _np_adx
            _df5 = live_1m.resample("5min").agg(
                {"high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(_df5) >= 15:
                _h5 = _df5["high"].astype(float)
                _l5 = _df5["low"].astype(float)
                _c5 = _df5["close"].astype(float)
                _cp, _hp, _lp = _c5.shift(1), _h5.shift(1), _l5.shift(1)
                _tr  = pd.concat([_h5 - _l5, (_h5 - _cp).abs(), (_l5 - _cp).abs()], axis=1).max(axis=1)
                _dmp = pd.Series(
                    _np_adx.where((_h5 - _hp).values > (_lp - _l5).values,
                                  _np_adx.maximum((_h5 - _hp).values, 0.0), 0.0),
                    index=_df5.index)
                _dmm = pd.Series(
                    _np_adx.where((_lp - _l5).values > (_h5 - _hp).values,
                                  _np_adx.maximum((_lp - _l5).values, 0.0), 0.0),
                    index=_df5.index)
                _atr  = _tr.ewm(span=14, adjust=False).mean()
                _di_p = _dmp.ewm(span=14, adjust=False).mean() / _atr * 100
                _di_m = _dmm.ewm(span=14, adjust=False).mean() / _atr * 100
                _dx   = (_di_p - _di_m).abs() / (_di_p + _di_m).replace(0.0, float("nan")) * 100
                _adx_5m  = float(_dx.ewm(span=14, adjust=False).mean().iloc[-1])
                _di_p_5m = float(_di_p.iloc[-1])
                _di_m_5m = float(_di_m.iloc[-1])
        except Exception as _exc:
            print(f"  [adx5_gate] ADX error: {_exc}")

    if args.asset == "BTC" and (_adx_5m is not None or _lower_hl):
        _adx_disp = (f"ADX={_adx_5m:.1f} DI+={_di_p_5m:.1f} DI-={_di_m_5m:.1f}"
                     if _adx_5m is not None else "ADX=n/a")
        print(f"  [adx5_gate] BTC 5m: {_adx_disp} | lower_hl={_lower_hl}")

    # --- bp_5m + body_15m: buying pressure and body ratio signals ---
    # bp_5m   = (close - low) / (high - low) on last completed 5m bar.
    #           0 = sellers won (close at low), 1 = buyers won (close at high).
    # body_15m = |close - open| / (high - low) on last completed 15m bar.
    #           0 = doji (indecision), 1 = full marubozu (commitment).
    # Use iloc[-2] to skip the current (incomplete) bar.
    _bp_5m    = None   # float [0, 1] or None
    _body_15m = None   # float [0, 1] or None
    _dir_15m  = None   # int +1 (bullish) or -1 (bearish) or None
    if live_1m is not None and len(live_1m) >= 20:
        try:
            _df5b = live_1m.resample("5min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(_df5b) >= 2:
                _r5 = float(_df5b["high"].iloc[-2]) - float(_df5b["low"].iloc[-2])
                if _r5 > 0:
                    _bp_5m = (float(_df5b["close"].iloc[-2]) - float(_df5b["low"].iloc[-2])) / _r5
                else:
                    _bp_5m = 0.5

            _df15b = live_1m.resample("15min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(_df15b) >= 2:
                _r15   = float(_df15b["high"].iloc[-2]) - float(_df15b["low"].iloc[-2])
                _c15   = float(_df15b["close"].iloc[-2])
                _o15   = float(_df15b["open"].iloc[-2])
                if _r15 > 0:
                    _body_15m = abs(_c15 - _o15) / _r15
                else:
                    _body_15m = 0.0
                _dir_15m = 1 if _c15 >= _o15 else -1
        except Exception as _exc:
            print(f"  [body_bp] compute error: {_exc}")

    if _bp_5m is not None or _body_15m is not None:
        _bp_disp   = f"{_bp_5m:.3f}"   if _bp_5m   is not None else "n/a"
        _body_disp = f"{_body_15m:.3f}" if _body_15m is not None else "n/a"
        _dir_disp  = f"{_dir_15m:+d}"   if _dir_15m  is not None else "n/a"
        print(f"  [body_bp] bp_5m={_bp_disp}  body_15m={_body_disp}  dir_15m={_dir_disp}")

    # --- 1h EMA stack count (EMA20/50/100/200 below spot) ---
    # _ema_stack_liq_1h=3 means EMA20/50/100 are below spot but EMA200 is above —
    # price is in a local uptrend but trapped below long-term resistance.
    # Sim (2834 trades): YES with stack=3 → WR=49.4%, PnL=-$1,101.
    _ema_stack_liq_1h = 0
    try:
        _1h_cls = df_confirm["close"].astype(float)
        _e20  = float(_1h_cls.ewm(span=20,  adjust=False).mean().iloc[-1])
        _e50  = float(_1h_cls.ewm(span=50,  adjust=False).mean().iloc[-1])
        _e100 = float(_1h_cls.ewm(span=100, adjust=False).mean().iloc[-1])
        _e200 = float(_1h_cls.ewm(span=200, adjust=False).mean().iloc[-1])
        _ema_stack_liq_1h = sum(1 for _ev in [_e20, _e50, _e100, _e200] if _ev < spot)
        print(f"  [ema_stack_liq_1h] stack={_ema_stack_liq_1h}  "
              f"EMA20={_e20:.0f}  EMA50={_e50:.0f}  EMA100={_e100:.0f}  EMA200={_e200:.0f}")
    except Exception as _esl_exc:
        print(f"  [ema_stack_liq_1h] Error: {_esl_exc}")

    # --- Counter-tape severity gate (hybrid hard-block + Kelly dampener) ---
    # Block or shrink bets that fight recent realized price movement. Addresses
    # slow-grind regime mismatches that the streak gate (2-consecutive-bar pattern)
    # misses when the grind has alternating sub-threshold candles.
    #
    # severity = max over 5m/10m/30m windows of (counter-tape fraction / threshold)
    #   severity < 0.5           → full Kelly
    #   0.5 ≤ severity < 1.5     → Kelly scaled to max(0.25, 1 - (sev-0.5)*0.75)
    #   severity ≥ 1.5           → hard block
    #
    # Thresholds calibrated against paper-trade archive:
    #   BTC: +$172 delta, blocks 10 (4W/6L)
    #   ETH: +$192 delta, blocks 17 (8W/9L)
    #   SOL: +$156 delta, blocks 1 — naturally quiet at wider thresholds
    #
    # 2026-04-28 retune (gate_attribution.py per-asset threshold sweep):
    # Original ×1.0 appeared to sit at a local minimum on BTC and too loose on ETH.
    # Multipliers tightened ×0.75 for BTC/ETH.
    #
    # 2026-04-28 PARTIAL REVERT — the v1 harness was using recomputed log-normal+drift
    # for ETH/SOL p_model, but production uses HistGradientBoosting (direct_p_model.py)
    # for those assets. v2 harness (gate_attribution_v2.py) using LOGGED p_yes_model
    # from the archive (the actual production model output at decision time) plus a
    # gate_attribution.py v2 (logged p_model, flat $1k bankroll, /100 at load_archive):
    #   BTC: ×1.0 near-optimal; ×0.75 was sub-optimal.  → BTC at ×1.0 thresholds.
    #   ETH: OFF (+$1,570) > ×1.0 (+$1,451). Every block the gate makes is net-negative.
    #        → ETH disabled (not in dict; severity returns 0.0).
    #   SOL: gate is flat — unchanged.
    # Note: gate_attribution.py divides CSV chg values by 100 at load time so units
    # match raw-decimal thresholds below. ~2% of ETH surviving candidates are hard-blocked.
    _COUNTER_TAPE_THR = {
        "BTC": (0.0016, 0.0024, 0.0040),
        # ETH: disabled — Opus v2 harness (correct units) shows OFF beats every multiplier
        "SOL": (0.0025, 0.0040, 0.0065),
    }

    def _counter_tape_severity(side: str) -> float:
        thr = _COUNTER_TAPE_THR.get(args.asset)
        if thr is None:
            return 0.0
        sign = -1.0 if side == "yes" else 1.0
        c5, c10, c30 = sign * _sharp_move_pct_5m, sign * _sharp_move_pct_10m, sign * _sharp_move_pct
        return max(0.0, c5 / thr[0], c10 / thr[1], c30 / thr[2])

    # Scan all contracts for nearest expiry; select highest net_edge trade.
    # Falls back to simulated p_market on nearest OTM contract when no auth.
    contracts_scanned = 0
    p_market_source   = "simulated"
    contract_ticker   = ""
    close_ts          = ""
    strike            = spot * 1.005   # fallback

    best_trade_dec    = None           # best DecisionResult with decision=="trade"
    best_trade_meta   = {}             # {strike, p_market, prob, contract_ticker, close_ts}
    best_any_dec      = None           # best DecisionResult across all contracts (for no_trade log)
    best_any_meta     = {}
    best_no_trade_dec = None           # best no_trade-only result (safe fallback when trade is cooldown-blocked)
    best_no_trade_meta = {}

    if auth is not None:
        ladder = fetch_contracts_for_nearest_expiry(auth, spot, asset=args.asset)
        contracts_scanned = len(ladder)
        print(f"  [scan] {contracts_scanned} liquid contracts in nearest expiry")

        # Load already-traded tickers and strike positions per expiry to prevent conflicting bets
        _is_pure_paper = not (args.live or getattr(args, 'dual', False))
        csv_path_check = get_csv_path(args.asset, shadow=_is_pure_paper)
        # Live runner tracks its own positions from live_trades.csv to avoid being
        # blocked by paper-only trades. Paper runner uses paper_trades_shadow.csv.
        if args.live or getattr(args, 'dual', False):
            expiry_source_path = live_trading.get_live_csv_path(args.asset)
            expiry_source_is_live = True
        else:
            expiry_source_path = csv_path_check
            expiry_source_is_live = False
        already_traded = _SESSION_TRADED  # always use session set; CSV failure cannot bypass it
        already_traded_expiries = {}  # {close_ts: {"yes": [strikes], "no": [strikes]}}
        _mu_6h_btc = _mu_12h_btc = _mu_24h_btc = _regime_z_btc = 0.0
        _rvol_inv_btc = 1.0
        _garch_ve_btc = float("nan")
        _arima_forecast_btc = float("nan")
        # H&S shadow state — scan-level, reset each cycle
        _hs_pat_type        = ""
        _hs_bars_since      = ""
        _hs_r2              = ""
        _hs_neck_slope      = ""
        _hs_head_height     = ""
        _hs_head_width      = ""
        # PIP shape state — scan-level, reset each cycle
        _pip_last_slope = float("nan")
        _pip_up_frac    = float("nan")
        _pip_n_turns    = -1
        # Hawkes vol — carried from top-of-loop computation; reset here for safety
        _v_hawk      = _v_hawk_val
        _hawk_regime = _hawk_vol_regime_val
        _pc1_rsi     = _pc1_rsi_val
        # OU mean reversion — scan-level, all assets
        _ou_z_score      = float("nan")   # (spot - ou_mean) / ou_sigma; +ve = price extended above mean
        _ou_halflife_min = float("nan")   # expected reversion half-life in minutes
        _ou_theta        = float("nan")   # speed of reversion (per hour), cached for tau_drift
        _ou_mu_val       = float("nan")   # OU long-run mean (price level)
        _ou_sigma_s      = float("nan")   # stationary std (price level)
        if csv_path_check.exists():
            try:
                df_existing = pd.read_csv(csv_path_check)
                # already_traded_expiries: only active (not yet expired) contracts
                # Expired contracts have settled and cannot conflict with new trades
                traded_rows_all = df_existing[df_existing["decision"] == "trade"].copy()
                traded_rows_all = traded_rows_all[
                    pd.to_datetime(traded_rows_all["close_ts"], utc=True) > pd.Timestamp(now_utc)
                ]
                # Build expiry counts from the runner-specific source
                if expiry_source_is_live and expiry_source_path.exists():
                    try:
                        df_live_exp = pd.read_csv(expiry_source_path)
                        df_live_exp = df_live_exp[
                            pd.to_datetime(df_live_exp["logged_at"], utc=True) > pd.Timestamp(now_utc) - pd.Timedelta(hours=2)
                        ]
                        for _, r in df_live_exp[["contract_ticker", "side", "strike"]].dropna().iterrows():
                            key = _expiry_prefix(str(r["contract_ticker"]))
                            bucket = already_traded_expiries.setdefault(key, {"yes": [], "no": []})
                            try:
                                bucket[r["side"]].append(float(r["strike"]))
                            except (ValueError, TypeError):
                                bucket[r["side"]].append(0.0)
                    except Exception:
                        pass
                else:
                    for _, r in traded_rows_all[["contract_ticker", "side", "strike", "logged_at"]].dropna().iterrows():
                        key = _expiry_prefix(str(r["contract_ticker"]))
                        bucket = already_traded_expiries.setdefault(key, {"yes": [], "no": []})
                        bucket[r["side"]].append(float(r["strike"]))
                # Sync _SESSION_TRADED from CSV every cycle using the 2-hour window.
                # Running every cycle (not just once at startup) prevents re-entry after
                # restarts, concurrent processes, or when a prior scan produced no_trade
                # for a contract that later qualifies as trade in the next scan.
                # Live runner syncs from live_trades.csv only to avoid being blocked by
                # paper-only trades.
                try:
                    if args.live or getattr(args, 'dual', False):
                        seed_path = live_trading.get_live_csv_path(args.asset)
                        if seed_path.exists():
                            df_live = pd.read_csv(seed_path)
                            df_live = df_live[
                                pd.to_datetime(df_live["logged_at"], utc=True) >
                                pd.Timestamp(now_utc) - pd.Timedelta(hours=2)
                            ]
                            for ticker in df_live["contract_ticker"].dropna().unique():
                                if ticker not in _SESSION_TRADED:
                                    _SESSION_TRADED[ticker] = 0.0
                    else:
                        for ticker in traded_rows_all["contract_ticker"].dropna().unique():
                            if ticker not in _SESSION_TRADED:
                                _SESSION_TRADED[ticker] = 0.0
                except Exception:
                    pass
                # Re-seed _SIDE_COOLDOWN from CSV every cycle so concurrent runner instances
                # (e.g. two paper runners, or paper + live dual) see each other's recent
                # trades and respect the same 300s cooldown. traded_rows_all is re-read
                # from disk each cycle so this is safe across processes.
                # Live runner seeds from live_trades.csv only — paper trades must not
                # influence live cooldowns (they are independent processes).
                try:
                    _cooldown_window = pd.Timestamp(now_utc) - pd.Timedelta(seconds=300)
                    if args.live or getattr(args, 'dual', False):
                        _cd_source = pd.DataFrame()
                        _cd_path = live_trading.get_live_csv_path(args.asset)
                        if _cd_path.exists():
                            _cd_df = pd.read_csv(_cd_path)
                            _cd_source = _cd_df[
                                pd.to_datetime(_cd_df["logged_at"], utc=True) >= _cooldown_window
                            ]
                        _cd_rows = _cd_source[["contract_ticker", "side", "logged_at"]].dropna() if not _cd_source.empty else pd.DataFrame()
                    else:
                        _cd_rows = traded_rows_all[["contract_ticker", "side", "logged_at"]].dropna()
                    for _, r in _cd_rows.iterrows():
                        _ts = pd.to_datetime(r["logged_at"], utc=True)
                        if _ts >= _cooldown_window:
                            _key = (_expiry_prefix(str(r["contract_ticker"])), r["side"])
                            if _key not in _SIDE_COOLDOWN or _ts > pd.Timestamp(_SIDE_COOLDOWN[_key]):
                                _SIDE_COOLDOWN[_key] = _ts.to_pydatetime()
                except Exception:
                    pass
                global _SESSION_SEEDED
                if not _SESSION_SEEDED:
                    if _SIDE_COOLDOWN:
                        print(f"  [session] Seeded {len(_SIDE_COOLDOWN)} cooldown entries from CSV")
                    _SESSION_SEEDED = True
                    if _SESSION_TRADED:
                        print(f"  [session] Seeded {len(_SESSION_TRADED)} open tickers from CSV")
                # Multi-window rolling drift + regime_z + GARCH ve for branched YES/NO model.
                try:
                    import numpy as _np_drift
                    _close_1h = df_confirm["close"].astype(float).sort_index()
                    _lr_1h = _np_drift.log(_close_1h / _close_1h.shift(1))
                    def _safe_roll(s, w):
                        v = float(s.rolling(w).mean().iloc[-1])
                        return 0.0 if _np_drift.isnan(v) else v
                    _mu_6h_btc  = _safe_roll(_lr_1h, 6)
                    _mu_12h_btc = _safe_roll(_lr_1h, 12)
                    _mu_24h_btc = _safe_roll(_lr_1h, 24)
                    _ewm_mean = float(_lr_1h.ewm(span=12).mean().iloc[-1])
                    _ewm_std  = float(_lr_1h.ewm(span=24).std().iloc[-1])
                    _regime_z_btc = float(_np_drift.clip(
                        _ewm_mean / _ewm_std if _ewm_std > 0 else 0.0, -3.0, 3.0))
                    _vol_24h_std  = float(_lr_1h.rolling(24,  min_periods=4).std().iloc[-1])
                    _vol_168h_std = float(_lr_1h.rolling(168, min_periods=24).std().iloc[-1])
                    if _vol_24h_std > 0 and not _np_drift.isnan(_vol_24h_std) and not _np_drift.isnan(_vol_168h_std):
                        _rvol_inv_btc = float(_np_drift.clip(_vol_168h_std / _vol_24h_std, 0.3, 2.0))
                    _garch_ve_btc = _get_garch_cond_ve(df_confirm)
                    try:
                        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                        _lr_arima = _lr_1h.dropna()
                        _arima_forecast_btc = float(
                            _ARIMA(_lr_arima, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
                    except Exception:
                        _arima_forecast_btc = float("nan")
                    print(f"  [drift] rvol_inv={_rvol_inv_btc:.3f} garch_ve={_garch_ve_btc:.6f}")
                except Exception as _drift_exc:
                    print(f"  [drift] compute failed: {_drift_exc}")
                # ── H&S shadow logging (1h close, last 500 bars) ─────────────────────
                try:
                    import sys as _hs_sys, os as _hs_os
                    _hs_sys.path.insert(0, str(Path(__file__).parent))
                    from hs_pattern import find_hs_patterns as _find_hs
                    import numpy as _np_hs
                    _hs_close = df_confirm["close"].astype(float).dropna().values[-500:]
                    _hs_log   = _np_hs.log(_hs_close)
                    _hs_all, _ihs_all = _find_hs(_hs_log, order=6)
                    _hs_combined = [(p, "hs") for p in _hs_all] + [(p, "ihs") for p in _ihs_all]
                    if _hs_combined:
                        _hs_latest, _hs_ltype = max(_hs_combined, key=lambda x: x[0].break_i)
                        _hs_pat_type   = _hs_ltype
                        _hs_bars_since = len(_hs_log) - 1 - _hs_latest.break_i
                        _hs_r2         = round(_hs_latest.pattern_r2, 4)
                        _hs_neck_slope = round(_hs_latest.neck_slope, 6)
                        _hs_head_height = round(_hs_latest.head_height, 5)
                        _hs_head_width  = int(_hs_latest.head_width)
                        print(f"  [hs_shadow] {_hs_ltype.upper()}  break={_hs_bars_since}bars ago  "
                              f"r2={_hs_r2:.2f}  slope={_hs_neck_slope:.5f}")
                except Exception as _hs_exc:
                    print(f"  [hs_shadow] compute failed: {_hs_exc}")
                # ── PIP shape features (1h close, last 50 bars) ──────────────────────
                try:
                    import sys as _pip_sys
                    _pip_sys.path.insert(0, str(Path(__file__).parent))
                    from pip_features import compute_pip_shape as _compute_pip_shape
                    import numpy as _np_pip
                    _pip_close = df_confirm["close"].astype(float).dropna().values
                    _pip_last_slope, _pip_up_frac, _pip_n_turns = _compute_pip_shape(
                        _pip_close, n_pips=5, n_bars=50
                    )
                    if not (_pip_last_slope != _pip_last_slope):  # not nan
                        print(f"  [pip_shadow] slope={_pip_last_slope:+.5f}  "
                              f"up_frac={_pip_up_frac:.3f}  turns={_pip_n_turns}")
                except Exception as _pip_exc:
                    print(f"  [pip_shadow] compute failed: {_pip_exc}")
            except Exception:
                pass

        # ── Ornstein-Uhlenbeck mean reversion fit (log-price, 24h rolling) ──────
        # AR(1) on log-prices over last 24 bars: log(X_t) = a + b*log(X_{t-1}) + ε
        # 24-bar window targets intraday reversion relevant to Kalshi tau (5-150 min).
        # Full-history fit gives half-life ~73k min (multi-year trend) — useless here.
        # b = e^{-θ}; θ = -ln(b) per hour; mu_log = a/(1-b) (OU log-price mean)
        # ou_z_score:    (log(spot) - mu_log) / sigma_log — signed extension in log units
        # ou_halflife_min: ln(2)/θ × 60 — minutes until half the deviation reverts
        # ou_tau_drift:  E[log_return over τ] = (mu_log - log(spot)) × (1 - e^{-θτ/60})
        #                negative z → positive tau_drift → supports YES; and vice versa
        try:
            import numpy as _np_ou
            _ou_lp = _np_ou.log(df_confirm["close"].astype(float).dropna().values[-25:])
            if len(_ou_lp) >= 20:
                _ou_lx = _ou_lp[:-1]; _ou_ly = _ou_lp[1:]
                _ou_b  = float(_np_ou.clip(
                    _np_ou.cov(_ou_lx, _ou_ly)[0, 1] / (_np_ou.var(_ou_lx) + 1e-12),
                    0.001, 0.9999))
                _ou_a      = float(_np_ou.mean(_ou_ly) - _ou_b * _np_ou.mean(_ou_lx))
                _ou_mu_val = _ou_a / (1.0 - _ou_b)        # OU mean in log-price space
                _ou_theta  = -float(_np_ou.log(_ou_b))    # reversion speed per hour
                _ou_sigma_s = float(_np_ou.std(_ou_lp))   # log-price stationary std
                if _ou_sigma_s > 0 and _ou_theta > 0:
                    _ou_z_score      = round((math.log(spot) - _ou_mu_val) / _ou_sigma_s, 4)
                    _ou_halflife_min = round(math.log(2) / _ou_theta * 60.0, 1)
                    print(f"  [ou] z={_ou_z_score:+.3f}  mu_px={math.exp(_ou_mu_val):,.0f}  "
                          f"hl={_ou_halflife_min:.0f}min  "
                          f"({'YES' if _ou_z_score < -0.5 else 'NO' if _ou_z_score > 0.5 else 'neutral'} bias)")
        except Exception:
            pass

        # _markov_regime already fetched above (before garch_markov_vol_adjust block).

        # Semi-Markov vol-regime signal — computed once per scan cycle, used by gate + CSV.
        # Returns (hard_rank, r1_prob, r1_k10, time_in_state); None if model unavailable.
        _hmm_vol_probs = _vol1h_hmm_probs(live_1m) if (args.asset == "BTC" and live_1m is not None) else None
        if _hmm_vol_probs is not None:
            _hmm_r, _hmm_tis = _hmm_vol_probs[0], _hmm_vol_probs[3]
            _tis_zone = ("early" if _hmm_tis <= 3 else "mid" if _hmm_tis <= 15 else "deep")
            print(f"  [vol_hmm] R{_hmm_r}  t={_hmm_tis} bars ({_tis_zone})  "
                  f"r1_prob={_hmm_vol_probs[1]:.3f}  k10={_hmm_vol_probs[2]:.3f}")

        # ETH vol regime (live gate) + SOL vol regime (shadow only)
        _hmm_vol_probs_eth_live = None
        _hmm_vol_probs_sol_live = None
        if args.asset == "ETH" and live_1m is not None:
            _hmm_vol_probs_eth_live = _vol_hmm_probs_eth(live_1m)
            if _hmm_vol_probs_eth_live is not None:
                _eth_hmm_r, _eth_r1_prob, _eth_hmm_tis = _hmm_vol_probs_eth_live
                _eth_tis_zone = ("early" if _eth_hmm_tis <= 3 else "mid" if _eth_hmm_tis <= 32 else "deep")
                print(f"  [vol_hmm_eth] R{_eth_hmm_r}  t={_eth_hmm_tis} bars ({_eth_tis_zone})  "
                      f"r1_prob={_eth_r1_prob:.3f}")
        elif args.asset == "SOL" and live_1m is not None:
            _hmm_vol_probs_sol_live = _vol_hmm_probs_sol(live_1m)
            if _hmm_vol_probs_sol_live is not None:
                _sol_hmm_r, _sol_r1_prob, _sol_hmm_tis = _hmm_vol_probs_sol_live
                _sol_tis_zone = ("early" if _sol_hmm_tis <= 4 else "mid" if _sol_hmm_tis <= 43 else "deep")
                print(f"  [vol_hmm_sol] R{_sol_hmm_r}  t={_sol_hmm_tis} bars ({_sol_tis_zone})  "
                      f"r1_prob={_sol_r1_prob:.3f}  [shadow]")

        # Flag/pennant signal — computed once per scan cycle from live 1h data (BTC only).
        # Shadow logging only; no gate wired until backtest validates edge.
        _flag_signal         = 0
        _flag_bull_bars_ago  = -1
        _flag_bear_bars_ago  = -1
        _flag_bull_tip_y     = float("nan")
        _flag_bear_tip_y     = float("nan")
        _flag_bull_pole_pct  = float("nan")
        _flag_bear_pole_pct  = float("nan")
        if args.asset == "BTC" and df_confirm is not None and len(df_confirm) >= 30:
            try:
                from flag_pennant import build_signal_series as _build_flags
                _flag_sig_df = _build_flags(df_confirm["close"], order=10, lookback_bars=48)
                if len(_flag_sig_df):
                    _last = _flag_sig_df.iloc[-1]
                    _flag_signal        = int(_last["flag_signal"])
                    _flag_bull_bars_ago = int(_last["flag_bull_bars_ago"])
                    _flag_bear_bars_ago = int(_last["flag_bear_bars_ago"])
                    _flag_bull_tip_y    = float(_last["flag_bull_tip_y"]) if not (isinstance(_last["flag_bull_tip_y"], float) and _last["flag_bull_tip_y"] != _last["flag_bull_tip_y"]) else float("nan")
                    _flag_bear_tip_y    = float(_last["flag_bear_tip_y"]) if not (isinstance(_last["flag_bear_tip_y"], float) and _last["flag_bear_tip_y"] != _last["flag_bear_tip_y"]) else float("nan")
                    _flag_bull_pole_pct = float(_last["flag_bull_pole_pct"]) if not (isinstance(_last["flag_bull_pole_pct"], float) and _last["flag_bull_pole_pct"] != _last["flag_bull_pole_pct"]) else float("nan")
                    _flag_bear_pole_pct = float(_last["flag_bear_pole_pct"]) if not (isinstance(_last["flag_bear_pole_pct"], float) and _last["flag_bear_pole_pct"] != _last["flag_bear_pole_pct"]) else float("nan")
                    _flag_lbl = ("BULL" if _flag_signal == 1 else "BEAR" if _flag_signal == -1 else "none")
                    _flag_ago = _flag_bull_bars_ago if _flag_signal == 1 else _flag_bear_bars_ago if _flag_signal == -1 else -1
                    _flag_tip = _flag_bull_tip_y if _flag_signal == 1 else _flag_bear_tip_y if _flag_signal == -1 else float("nan")
                    if _flag_signal != 0:
                        print(f"  [flag_pennant] {_flag_lbl}  {_flag_ago}h ago  tip=${_flag_tip:,.0f}  [shadow]")
            except Exception as _fe:
                pass  # silent — shadow signal only

        # Sigma 1% swing high — computed once per scan cycle for BTC.
        # Supplements ATR swing_high_gate: catches resistance levels ATR misses in low-vol.
        # Additive value validated: +$1,050 on top of ATR gate (MCPT p=0.000, n=698 new blocks).
        _sigma_swing_high: "float | None" = None
        _sigma_swing_low:  "float | None" = None
        _sigma_dist_high:  "float | None" = None
        if args.asset == "BTC" and live_1m is not None:
            _sigma_swing_high, _sigma_swing_low = _compute_sigma_swing_high(live_1m, sigma=0.01)
            if _sigma_swing_high is not None:
                _sigma_dist_high = (spot - _sigma_swing_high) / spot * 100
                print(f"  [sigma_dc_1%] sh=${_sigma_swing_high:,.0f}  dist={_sigma_dist_high:+.2f}%"
                      + (f"  sl=${_sigma_swing_low:,.0f}" if _sigma_swing_low else ""))

        # RW order=5 tops — computed once per scan cycle for BTC.
        # Used by itm_yes_rw_gate (itm_yes_sh_gate complement): catches local resistance
        # levels the sigma-DC misses. Tops confirmed in last 200 1h bars, sorted by time.
        _rw_tops: "list[tuple[float, pd.Timestamp]] | None" = None
        if args.asset == "BTC" and live_1m is not None:
            _rw_tops = _compute_rw_tops_1h(live_1m, order=5, lookback_bars=200)
            if _rw_tops:
                print(f"  [rw_tops] {len(_rw_tops)} confirmed tops (order=5, 200h lookback)  "
                      f"last=${_rw_tops[-1][0]:,.0f} @ {_rw_tops[-1][1].strftime('%m-%d %H:%M')}")

        for c in ladder:
            _p_gbdt_c     = None   # BTC LGBM p(YES) for this contract (shadow mode only)
            s_k       = c["floor_strike"]
            pm        = c["p_market"]
            _offset_limit = 0.01 if args.asset == "BTC" else 0.05
            if abs(s_k / spot - 1) > _offset_limit:
                continue
            # BTC: skip ITM contracts — ITM NO wins only 12%; ITM YES caught by Gate 0.
            # ETH: now matches SOL — ITM contracts allowed (trial; revert by changing
            #      condition back to: args.asset in ("BTC", "ETH"))
            # SOL: ITM YES wins 90.5%, OTM YES wins 80.6% — both regimes valid.
            offset_c = s_k / spot - 1
            spread_c  = c["ask"] - c["bid"]
            # Per-asset spread limits: SOL/ETH naturally wider in volatile conditions
            _spread_limit = 0.08 if args.asset == "BTC" else (0.30 if args.asset == "SOL" else 0.25)
            if spread_c > _spread_limit:
                print(f"  [scan] Skipping {c['ticker']} — spread={spread_c:.3f} (stale/illiquid, limit={_spread_limit})")
                continue
            tau_c     = minutes_to_expiry(c["close_time"])
            vol_imp_c = implied_vol_from_price(pm, spot, s_k, tau_c)
            vol_ratio_c = vol.vol_multi / vol_imp_c if vol_imp_c and vol_imp_c > 0 else None
            _vol_ratio_limit = 1.5 if args.asset == "BTC" else 5.0
            # Implied vol for blend: 50/50 DVOL + Kalshi when both are valid.
            # DVOL (Deribit 30-day ATM) provides stability and fills gaps; Kalshi back-computed
            # IV adds contract-specific moneyness+term-structure signal when available.
            # When Kalshi fails (OTM/ITM extremes → 0 or NaN), DVOL carries the full implied weight.
            # When DVOL unavailable (SOL), Kalshi alone is used.
            _kalshi_iv_valid = vol_imp_c is not None and vol_imp_c > 0
            if _deribit_sigma_per_min is not None and _kalshi_iv_valid:
                _vol_imp_for_blend = 0.5 * _deribit_sigma_per_min + 0.5 * vol_imp_c
            elif _deribit_sigma_per_min is not None:
                _vol_imp_for_blend = _deribit_sigma_per_min
            else:
                _vol_imp_for_blend = vol_imp_c
            _vol_weight = REALIZED_VOL_WEIGHT_BY_ASSET.get(args.asset, REALIZED_VOL_WEIGHT)
            vol_eff_c = blend_vol(vol.vol_multi, _vol_imp_for_blend, weight=_vol_weight)
            # BTC LGBM shadow inference — runs on ALL scanned contracts (including vol_ratio rejects)
            # so the model can learn whether vol_ratio is a useful filter.
            if args.asset == "BTC" and _composite_computed:
                _gbdt_feats_c = {
                    "offset_pct":         offset_c,
                    "p_market":           pm,
                    "tau_minutes":        tau_c,
                    "side_enc":           0.5,
                    "composite_p_up":     _comp_p_up if _comp_p_up is not None else float("nan"),
                    "composite_trend":    float(_active_trend),
                    "composite_rev":      float(_active_rev),
                    "ema_stack_bias":     float(confirm.ema_stack_bias) if confirm.ema_stack_bias is not None else float("nan"),
                    "ema_stretch_score":  float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else float("nan"),
                    "vwap_stretch_score": float(confirm.stretch_score) if confirm.stretch_score is not None else float("nan"),
                    "vwap_distance_pct":  confirm.distance_pct * 100 if confirm.distance_pct == confirm.distance_pct else float("nan"),
                    "stoch_k":            float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else float("nan"),
                    "chg_30m":            _sharp_move_pct * 100,
                    "chg_10m":            _sharp_move_pct_10m * 100,
                    "chg_5m":             _sharp_move_pct_5m * 100,
                    "bp_5m":              _bp_5m if _bp_5m is not None else float("nan"),
                    "body_15m":           _body_15m if _body_15m is not None else float("nan"),
                    "dir_15m":            float(_dir_15m) if _dir_15m is not None else float("nan"),
                    "vol_score":          float(confirm.vol_score) if confirm.vol_score is not None else float("nan"),
                    "vpin_score":         float(confirm.vpin_score) if confirm.vpin_score is not None else float("nan"),
                    "obi_score":          float(confirm.obi_score) if confirm.obi_score is not None else float("nan"),
                    "confirmation_score": float(confirm.confirmation_score) if confirm.confirmation_score is not None else float("nan"),
                    "no_score":           float(confirm.no_score) if confirm.no_score is not None else float("nan"),
                    "funding_bias":       float(confirm.funding_bias) if confirm.funding_bias is not None else float("nan"),
                    "vol_eff":            vol_eff_c,
                    "pm_drift_5m":        (pm - list(_pm_history[c["ticker"]])[0]) if len(_pm_history.get(c["ticker"], [])) >= 5 else float("nan"),
                    "adx_1h":             float(confirm.adx_1h) if hasattr(confirm, "adx_1h") and confirm.adx_1h == confirm.adx_1h else float("nan"),
                    "rvol_1h":            _rvol_1h if not math.isnan(_rvol_1h) else float("nan"),
                    "squeeze_1h":         float(getattr(confirm, "squeeze_1h", float("nan"))),
                    "liq_score":          float(_liq_signal.liq_score) if _liq_signal is not None else float("nan"),
                    "liq_bias":           float(_liq_signal.liq_bias) if _liq_signal is not None else float("nan"),
                    "ls_long_pct":        float(_liq_signal.ls_long_pct) if _liq_signal is not None else float("nan"),
                    "oi_chg_pct":         float(_liq_signal.oi_chg_pct) if _liq_signal is not None else float("nan"),
                    "sigma_swing_high_1pct": round(_sigma_swing_high, 2) if _sigma_swing_high is not None else float("nan"),
                    "sigma_dist_high_1pct":  round(_sigma_dist_high, 2)  if _sigma_dist_high  is not None else float("nan"),
                    "flag_signal":           _flag_signal,
                    "flag_bull_bars_ago":    _flag_bull_bars_ago,
                    "flag_bear_bars_ago":    _flag_bear_bars_ago,
                    "flag_bull_tip_y":       _flag_bull_tip_y,
                    "flag_bear_tip_y":       _flag_bear_tip_y,
                    "flag_bull_pole_pct":    _flag_bull_pole_pct,
                    "flag_bear_pole_pct":    _flag_bear_pole_pct,
                    "pip_last_slope":        _pip_last_slope,
                    "pip_up_frac":           _pip_up_frac,
                    "pip_n_turns":           _pip_n_turns,
                }
                _btc_lgbm_c = _load_btc_lgbm()
                if _btc_lgbm_c is not None:
                    _p_gbdt_c = _infer_btc_lgbm(_btc_lgbm_c, _gbdt_feats_c)
                try:
                    import scan_archive as _sa
                    _sa.log_scan_row(
                        ticker=c["ticker"], close_ts=c["close_time"],
                        spot=spot, strike=s_k, p_market=pm, tau_minutes=tau_c,
                        features=_gbdt_feats_c, p_gbdt=_p_gbdt_c,
                        asset=args.asset, now_utc=now_utc,
                    )
                except Exception:
                    pass
            elif _composite_computed:
                # ETH / SOL scan archive — same feature set, shadow LGBM inference.
                _scan_feats_c = {
                    "offset_pct":         offset_c,
                    "composite_p_up":     _comp_p_up if _comp_p_up is not None else float("nan"),
                    "composite_trend":    float(_active_trend),
                    "composite_rev":      float(_active_rev),
                    "ema_stack_bias":     float(confirm.ema_stack_bias) if confirm.ema_stack_bias is not None else float("nan"),
                    "ema_stretch_score":  float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else float("nan"),
                    "vwap_stretch_score": float(confirm.stretch_score) if confirm.stretch_score is not None else float("nan"),
                    "vwap_distance_pct":  confirm.distance_pct * 100 if confirm.distance_pct == confirm.distance_pct else float("nan"),
                    "stoch_k":            float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else float("nan"),
                    "chg_30m":            _sharp_move_pct * 100,
                    "chg_10m":            _sharp_move_pct_10m * 100,
                    "chg_5m":             _sharp_move_pct_5m * 100,
                    "bp_5m":              _bp_5m if _bp_5m is not None else float("nan"),
                    "body_15m":           _body_15m if _body_15m is not None else float("nan"),
                    "dir_15m":            float(_dir_15m) if _dir_15m is not None else float("nan"),
                    "vol_score":          float(confirm.vol_score) if confirm.vol_score is not None else float("nan"),
                    "vpin_score":         float(confirm.vpin_score) if confirm.vpin_score is not None else float("nan"),
                    "obi_score":          float(confirm.obi_score) if confirm.obi_score is not None else float("nan"),
                    "confirmation_score": float(confirm.confirmation_score) if confirm.confirmation_score is not None else float("nan"),
                    "no_score":           float(confirm.no_score) if confirm.no_score is not None else float("nan"),
                    "funding_bias":       float(confirm.funding_bias) if confirm.funding_bias is not None else float("nan"),
                    "vol_eff":            vol_eff_c,
                    "pm_drift_5m":        (pm - list(_pm_history[c["ticker"]])[0]) if len(_pm_history.get(c["ticker"], [])) >= 5 else float("nan"),
                    "adx_1h":             float(confirm.adx_1h) if hasattr(confirm, "adx_1h") and confirm.adx_1h == confirm.adx_1h else float("nan"),
                    "rvol_1h":            _rvol_1h if not math.isnan(_rvol_1h) else float("nan"),
                    "squeeze_1h":         float(getattr(confirm, "squeeze_1h", float("nan"))),
                    "liq_score":          float(_liq_signal.liq_score) if _liq_signal is not None else float("nan"),
                    "liq_bias":           float(_liq_signal.liq_bias) if _liq_signal is not None else float("nan"),
                    "ls_long_pct":        float(_liq_signal.ls_long_pct) if _liq_signal is not None else float("nan"),
                    "oi_chg_pct":         float(_liq_signal.oi_chg_pct) if _liq_signal is not None else float("nan"),
                }
                _asset_lgbm_c = _load_asset_lgbm(args.asset)
                if _asset_lgbm_c is not None:
                    _p_gbdt_c = _infer_asset_lgbm(_asset_lgbm_c, _scan_feats_c, args.asset)
                _p_up_v2_c = None
                if _df_4h_comp is not None:
                    try:
                        _p_up_v2_c = _asset_p_up_model.compute_p_up(
                            args.asset, df_confirm, _df_4h_comp, confirm,
                            composite_trend=float(_active_trend),
                            composite_rev=float(_active_rev),
                            composite_p_up_1h=_comp_p_up if _comp_p_up is not None else 0.504,
                            pm_drift_5m=_scan_feats_c.get("pm_drift_5m", float("nan")),
                        )
                        if _p_up_v2_c is not None:
                            print(f"  [{args.asset.lower()}_p_up_v2] {_p_up_v2_c:.3f}  (shadow)")
                    except Exception:
                        pass
                try:
                    import scan_archive as _sa
                    _sa.log_scan_row(
                        ticker=c["ticker"], close_ts=c["close_time"],
                        spot=spot, strike=s_k, p_market=pm, tau_minutes=tau_c,
                        features=_scan_feats_c, p_gbdt=_p_gbdt_c,
                        asset=args.asset, now_utc=now_utc,
                        p_up_v2=_p_up_v2_c,
                    )
                except Exception:
                    pass
            if vol_ratio_c is not None and vol_ratio_c > _vol_ratio_limit:
                print(f"  [scan] Skipping {c['ticker']} — vol_ratio={vol_ratio_c:.2f} (realized >> implied, limit={_vol_ratio_limit})")
                continue
            vol_adj_c = vol_eff_c * _vol_factor   # vol regime scaling
            prob_c    = estimate_probability(spot, s_k, tau_c, vol_adj_c,
                                               structure_bias=0,
                                               confirmation_score=0)  # kept for diagnostic fields
            # Composite-adjusted p_model: composite scores shift the log-normal distribution
            # by a calibrated drift derived from empirical win rates on 11,108 test hours.
            # Falls back to pure log-normal (prob_c.p_yes) when composite is unavailable.
            if _composite_computed:
                # BTC reform: vol_factor removed from sigma — use vol_eff_c (not vol_adj_c).
                # vol_factor is now used as a reachability gate (see below), not a sigma scaler.
                _sigma_base   = vol_eff_c if args.asset == "BTC" else vol_adj_c
                sigma_tau_c   = _sigma_base * math.sqrt(tau_c)
                # YES drift computed below after p_up_v2 is available
                _sq = math.sqrt(tau_c / 60.0) if args.asset == "BTC" else 0.0
                _z_drift_6h = 0.0
                # Tau-blended p_up: for BTC, interpolate between 1h and 30m calibration
                # tables based on how much time remains. At tau>=60 pure 1h; at tau<=30
                # pure 30m; linear blend between. Falls back to 1h for assets
                # without a 30m calibration file (lookup_p_up_blended handles this).
                _comp_p_up_c  = lookup_p_up_blended(
                    _comp_trend, _comp_rev, _comp_trend_30m, _comp_rev_30m,
                    tau_c, asset=args.asset,
                    trend_4h=_comp_trend_4h, rev_4h=_comp_rev_4h,
                )
                # BTC p_up v2: ML model combining price indicators + live signals.
                # Overrides the lookup table when model file exists.
                if args.asset == "BTC" and _composite_computed:
                    _p_up_v2 = _btc_p_up_model.compute_p_up(
                        df_confirm, _df_4h_comp,
                        confirm,
                        composite_trend=float(_active_trend),
                        composite_rev=float(_active_rev),
                        composite_p_up_1h=_comp_p_up,
                        pm_drift_5m=float("nan"),
                    )
                    if _p_up_v2 is not None:
                        print(f"  [p_up_v2] {_p_up_v2:.3f}")
                        # Update rolling regime buffer once per new 1h bar
                        _bar_ts_now = df_confirm.index[-1]
                        if _bar_ts_now != _pup_v2_regime_state["last_bar_ts"]:
                            _pup_v2_buf.append(_p_up_v2)
                            _pup_v2_regime_state["last_bar_ts"] = _bar_ts_now
                        # YES drift: norm.ppf(p_up_v2) × rvol_inv × k=0.3 × √(τ/60)
                        if args.asset == "BTC" and sigma_tau_c > 0:
                            from scipy.stats import norm as _norm_pup
                            _z_drift_6h = float(_norm_pup.ppf(max(0.01, min(0.99, _p_up_v2)))) * _rvol_inv_btc * 0.3 * _sq
                            print(f"  [yes_drift] z={_z_drift_6h:+.4f}  (pup={_p_up_v2:.3f} rvol_inv={_rvol_inv_btc:.3f})")
                p_model_comp  = None
                _p_no_eth     = None
                # ETH HYBRID: YES → score_to_p_model (k=0.80, tau-blended p_up)
                #             NO  → compute_p_no_direct (NO-specific ML model)
                # SOL: direct_p_model (strike-hit ML, validated)
                # BTC: score_to_p_model (k=1.40) via fallback below
                if args.asset == "ETH" and _composite_computed and sigma_tau_c > 0:
                    p_model_comp = score_to_p_model(
                        _active_trend, _active_rev, spot, s_k, sigma_tau_c,
                        asset="ETH", p_up_override=_comp_p_up_c,
                    )
                    # [2026-05-08] ETH NO: switched from direct ML model to log-drift
                    # (K=0.20). Simulation sweep showed K=0.20 is best-calibrated at
                    # pm [0.25,0.45) — WR≈BE=-0.2pp. Direct model had severe
                    # miscalibration at p_yes_model [0.25,0.35) → actual_YES=54.5%.
                    _p_no_eth = score_to_p_no_model(
                        _active_trend, _active_rev, spot, s_k, sigma_tau_c,
                        asset="ETH", p_up_override=_comp_p_up_c,
                    )
                elif direct_p_model.asset_supported(args.asset):
                    try:
                        p_model_comp = direct_p_model.compute_p_model_direct(
                            asset=args.asset,
                            df_1m=live_1m, df_1h=df_confirm,
                            df_4h=_df_4h_comp, df_15m=_df_15m_comp,
                            offset_pct=offset_c,
                            composite_trend=float(_active_trend),
                            composite_rev=float(_active_rev),
                        )
                    except Exception as _e:
                        print(f"  [direct_p_model] inference error ({args.asset}): {_e} — falling back")
                        p_model_comp = None
                if p_model_comp is None:
                    p_model_comp = score_to_p_model(_active_trend, _active_rev, spot, s_k, sigma_tau_c, asset=args.asset, p_up_override=_comp_p_up_c,
                                                    z_drift_override=_z_drift_6h if args.asset == "BTC" else None)
            else:
                p_model_comp  = prob_c.p_yes

            # [BTC vol gate] For OTM YES only (offset > 0): block if |z_strike| > 2.0 × vol_factor.
            # Only OTM YES bets need the reachability gate — ITM YES bets are already in the money,
            # and NO bets are governed by z_abs_no_min below. vol_factor widens/narrows the
            # band with the vol regime. BASE_Z=2.0 gives a 1.2–2.8σ range across vol_factor [0.60,1.40].
            _otm_yes_blocked    = False
            _otm_yes_block_gate = ""
            _smc_yes_blocked    = False
            _vsa_no_flip        = False   # True when VSA YES block + NO flip triggered
            if args.asset == "BTC" and _composite_computed and sigma_tau_c > 0 and offset_c > 0:
                _z_strike_abs = abs(math.log(s_k / spot) / sigma_tau_c)
                _btc_vol_gate_z = 2.0 * _vol_factor
                if _z_strike_abs > _btc_vol_gate_z:
                    print(f"  [btc_vol_gate] BLOCK YES {c['ticker']} — |z|={_z_strike_abs:.3f} > {_btc_vol_gate_z:.3f} (vol_factor={_vol_factor:.3f})")
                    _otm_yes_blocked = True; _otm_yes_block_gate = "btc_vol_gate"

            # [BTC isotonic calibration DISABLED — trained on drift-biased p_model values.
            # Reform uses k_drift=0.8 + vol_factor-as-gate; isotonic retrain required
            # before re-enabling. Remove this comment once retrained.]

            # [OTM YES momentum exhaustion gate — REMOVED 2026-05-16]
            # Simulation on 1,018 unique blocked OTM YES contracts showed all pm buckets
            # profitable when z_drift filters (WR=21.2% vs BE=6.1%, +$3,108 flat $10/trade).
            # z_drift already encodes direction — stoch/ema momentum checks are redundant.
            # Gate OTM in decision.py (4% net_edge floor for pm<0.15) remains as backstop.

            # [Near-ITM YES gate — BTC composite only]
            # Block YES when pm > 0.50 (strike already below spot) AND 4h timeframe is overbought/extended.
            # Analysis (88 live Near-ITM YES trades, May 2026):
            #   4h RSI > 62 OR 4h MACD hist > 80 → WR=34.5%, PnL=-$778 (58 trades)
            #   Neither condition                 → WR=76.7%, PnL=+$216 (30 trades)
            #   No rescue condition found within blocked group (best: 1h MACD neg at 50% WR = still losing).
            # Rationale: Near-ITM YES fails when 4h is extended because BTC retraces before expiry.
            #   The 1h composite looks bullish (high p_up) but the 4h exhaustion overrides it.
            # 2026-05-30 reform: changed OR → AND. MACD hist is in absolute $ terms and fires constantly
            #   at mild uptrends (RSI=57, MACD=126 = only 0.17% of price). MACD-only blocks have
            #   76.8% YES rate (+3.6% edge) vs RSI+MACD blocks at 63.5% (-9.3% edge). Scan archive
            #   simulation (250 expiry cycles, May 18-30): OR=$164, AND=$545, improvement=+$381.
            if args.asset == "BTC" and _composite_computed and pm > 0.50 and not _otm_yes_blocked:
                if _df_4h_comp is not None and len(_df_4h_comp) >= 26:
                    _c4h = _df_4h_comp["close"]
                    _delta4h = _c4h.diff()
                    _gain4h  = _delta4h.clip(lower=0).rolling(14).mean()
                    _loss4h  = (-_delta4h.clip(upper=0)).rolling(14).mean()
                    _rsi_4h  = float((100 - 100 / (1 + _gain4h / _loss4h.replace(0, float("nan")))).iloc[-2])
                    _ema12_4h   = _c4h.ewm(span=12, adjust=False).mean()
                    _ema26_4h   = _c4h.ewm(span=26, adjust=False).mean()
                    _macd_4h    = _ema12_4h - _ema26_4h
                    _macd_sig_4h= _macd_4h.ewm(span=9, adjust=False).mean()
                    _macd_hist_4h = float((_macd_4h - _macd_sig_4h).iloc[-2])
                    if _rsi_4h > 62 and _macd_hist_4h > 80:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "near_itm_gate"
                        print(f"  [near_itm_gate] BLOCK YES {c['ticker']} — pm={pm:.3f}>0.50, "
                              f"4h_rsi={_rsi_4h:.1f}, 4h_macd_hist={_macd_hist_4h:.1f} "
                              f"(4h overbought, Near-ITM YES gate)")
                        gate_audit_logger.log_block(
                            gate_name="near_itm_gate",
                            ticker=c["ticker"], asset=args.asset, side="yes",
                            pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot,
                            tau_minutes=tau_c, count=0, kelly_fraction=0.0,
                            close_ts=c.get("close_time", ""),
                            signals={"rsi_4h": round(_rsi_4h, 1), "macd_hist_4h": round(_macd_hist_4h, 1)},
                            now_utc=now_utc, bankroll=args.bankroll,
                        )

            # [cg_fr_gate REMOVED 2026-05-17]
            # gate fired on fr_vol>0 (positive funding = crowded longs).
            # Live data (n=208, funding_bias=+1, pm<0.60, vpin=0): WR=38.5% vs BE=38.4%, Δ=+0.1% — neutral.
            # Gate had no blocked_trades audit log (fired pre-evaluate_trade, never called _log_block).
            # Was silently killing all YES for 12+ hrs in positive-funding regimes with no P&L benefit.

            # [BTC reversal-divergence gate — BTC YES only]
            # Block YES when EMA stack is bullish (+1) but composite_rev is cratering (<=−4)
            # and stoch_k is overbought (>55) — trend/reversal divergence that precedes sharp drops.
            # Backtest (1h paper trades): blocks 76 YES at 36.8% WR → PnL improvement +$543.
            # Rescue: pm>0.65 (deeply ITM; market conviction overrides reversal pressure at 70.3% WR).
            if args.asset == "BTC" and _composite_computed and not _otm_yes_blocked:
                _sk_rev = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else 50.0
                if confirm.ema_stack_bias == 1 and _active_rev <= -4 and _sk_rev > 55:
                    _rev_rescue = (pm > 0.65)
                    if not _rev_rescue:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "rev_div_gate"
                        print(f"  [rev_div_gate] BLOCK YES {c['ticker']} — ema=+1, rev={_active_rev}<=-4, stoch_k={_sk_rev:.1f}>55, pm={pm:.3f}")
                    else:
                        print(f"  [rev_div_gate] RESCUE YES {c['ticker']} — ema=+1, rev={_active_rev}, stoch_k={_sk_rev:.1f} BUT pm={pm:.3f}>0.65 (deep ITM)")

            # [CoinGlass stablecoin OI 4h gate — BTC YES, OTM only]
            # Backtest (1h paper trades): oi_stable_chg_4h>2% blocks 110 YES → WR=44.5%, -$678.
            # Rescue: pm>=0.50 (ITM, WR=66.7%, pnl=+$178); block pm<0.50 (WR=18%, pnl=-$856).
            if args.asset == "BTC" and _cg is not None and not _otm_yes_blocked and _cg.oi_stable_pct_4h > 2.0:
                if pm >= 0.50:
                    print(f"  [cg_oi_stable_yes_gate] RESCUE YES {c['ticker']} — oi_stable_4h={_cg.oi_stable_pct_4h:+.2f}%>2% BUT pm={pm:.3f}>=0.50 (ITM)")
                else:
                    _otm_yes_blocked = True; _otm_yes_block_gate = "cg_oi_stable_yes_gate"
                    print(f"  [cg_oi_stable_yes_gate] BLOCK YES {c['ticker']} — oi_stable_4h={_cg.oi_stable_pct_4h:+.2f}%>2%, pm={pm:.3f}<0.50 (OTM, longs crowding)")

            # [vol_eff_low_yes_gate — BTC YES]
            # Block YES when vol_eff is below the bottom quartile (0.000318) AND z_score > -0.20.
            # Analysis (1625 BTC YES trades, bottom Q1 = 270 blocked):
            #   z_score <= -0.20 → WR=69.6%, +$86  (rescue: allow — ITM-leaning, strike reachable)
            #   z_score >  -0.20 → WR=32%,  ≈-$440  (hard block — market already moved, outcome random)
            # Rationale: low vol efficiency + positive z_score = indecisive regime where the market
            # has already priced the direction; outcome is near-random. Negative z_score = strike is
            # ITM-leaning even in low-vol regime, keeps enough edge to take.
            if args.asset == "BTC" and not _otm_yes_blocked and vol_eff_c < 0.000318:
                _veff_rescue = (prob_c.z_score <= -0.20)
                if not _veff_rescue:
                    _otm_yes_blocked = True; _otm_yes_block_gate = "vol_eff_low_yes_gate"
                    print(f"  [vol_eff_low_yes_gate] BLOCK YES {c['ticker']} — "
                          f"vol_eff={vol_eff_c:.6f}<0.000318, z_score={prob_c.z_score:.3f}>-0.20 "
                          f"(low-vol + OTM-leaning, outcome indecisive)")
                else:
                    print(f"  [vol_eff_low_yes_gate] RESCUE YES {c['ticker']} — "
                          f"vol_eff={vol_eff_c:.6f}<0.000318 but z_score={prob_c.z_score:.3f}<=-0.20 "
                          f"(ITM-leaning, strike still reachable)")

            # [Neutral-EMA YES gates — BTC only]
            # G1 REMOVED 2026-05-16: ema=0 + comp_p_up>=0.60 + stoch_k<40 was blocking
            #   profitable trades. Simulation (n=31, 10 days): WR=67.7% vs BE=46%, +$550 PnL.
            #   stoch_k<40 with high p_up = price hasn't moved yet (lagging momentum signal),
            #   not a failed move. z_drift already encodes direction for these cases.
            # G2: bearish VWAP (vwap=-1) + OTM (pm<0.60); WR=16.0%, -$135 → KEEP
            # G3: declining composite trend (-1); WR=56.3% < BE=61.5% → KEEP
            if args.asset == "BTC" and _composite_computed and not _otm_yes_blocked and confirm.ema_stack_bias == 0:
                # G2: active selling pressure below VWAP + OTM contract
                if confirm.vwap_score == -1 and pm < 0.60:
                    _otm_yes_blocked = True; _otm_yes_block_gate = "neutral_ema_g2"
                    print(f"  [neutral_ema_g2] BLOCK YES {c['ticker']} — ema=0, vwap=-1, pm={pm:.3f}<0.60 (bearish VWAP, OTM)")

                # G3: equilibrium tipping bearish (composite trend rolling over)
                if not _otm_yes_blocked and _active_trend == -1:
                    _otm_yes_blocked = True; _otm_yes_block_gate = "neutral_ema_g3"
                    print(f"  [neutral_ema_g3] BLOCK YES {c['ticker']} — ema=0, comp_trend=-1 (equilibrium tipping bearish)")

            # [swing_high_gate — BTC YES, composite only]
            # Primary: block YES when strike > 4h swing high AND c_trend <= 1.
            #   Swing high is structural resistance in neutral/bear context; no breakout momentum.
            # Secondary: block YES when spot is 0–1% above swing high AND c_trend <= 1.
            #   dist_high [0,+1%) = exhaustion — spot just tagged resistance, affects ALL YES.
            # Rescue (both): c_trend >= 2 — bull momentum turns resistance into breakout level.
            # Backtest: primary +$3,023 p=0.000 (n=3,561 blocked); secondary +$1,033 p=0.000 (n=981).
            if (args.asset == "BTC" and _composite_computed and not _otm_yes_blocked
                    and _smc is not None and _smc.swing_high_4h is not None):
                _sh4h = _smc.swing_high_4h
                _dist_high_4h = (spot - _sh4h) / spot * 100
                _sh_c_trend_rescue = (_active_trend >= 2)
                if not _sh_c_trend_rescue:
                    if s_k > _sh4h:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "swing_high_gate"
                        print(f"  [swing_high_gate] BLOCK YES {c['ticker']} — "
                              f"strike={s_k:.0f} > sh4h={_sh4h:.0f}, c_trend={_active_trend}<=1 (resistance)")
                        gate_audit_logger.log_block(
                            gate_name="swing_high_gate", ticker=c["ticker"], asset=args.asset,
                            side="yes", pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                            count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                            signals={"swing_high_4h": _sh4h, "dist_high_4h": round(_dist_high_4h, 2),
                                     "composite_trend": _active_trend, "reason": "strike_above_resistance"},
                            now_utc=now_utc, bankroll=args.bankroll,
                        )
                    elif 0.0 <= _dist_high_4h < 1.0:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "swing_high_gate"
                        print(f"  [swing_high_gate] BLOCK YES {c['ticker']} — "
                              f"dist_high={_dist_high_4h:.2f}%∈[0,+1%), c_trend={_active_trend}<=1 (spot exhaustion)")
                        gate_audit_logger.log_block(
                            gate_name="swing_high_gate", ticker=c["ticker"], asset=args.asset,
                            side="yes", pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                            count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                            signals={"swing_high_4h": _sh4h, "dist_high_4h": round(_dist_high_4h, 2),
                                     "composite_trend": _active_trend, "reason": "spot_exhaustion"},
                            now_utc=now_utc, bankroll=args.bankroll,
                        )
                else:
                    if s_k > _sh4h or 0.0 <= _dist_high_4h < 1.0:
                        print(f"  [swing_high_gate] RESCUE YES {c['ticker']} — "
                              f"sh4h={_sh4h:.0f} dist={_dist_high_4h:.2f}% BUT c_trend={_active_trend}>=2 (breakout)")

            # [swing_high_gate sigma_1% — BTC YES, additive to ATR gate]
            # Blocks 698 additional YES bets not caught by ATR(240m) gate.
            # sigma_1% swing high confirmed after 1% retracement — more sensitive than ATR
            # in low-vol regimes where ATR threshold shrinks below 1%. Same c_trend rescue.
            # Additive validation: +$1,050 on top of ATR gate, MCPT p=0.000 (2026-06-04).
            if (args.asset == "BTC" and _composite_computed and not _otm_yes_blocked
                    and _sigma_swing_high is not None and _active_trend < 2):
                _sigma_dist_h_c = (spot - _sigma_swing_high) / spot * 100
                _sigma_block_reason = None
                if s_k > _sigma_swing_high:
                    _sigma_block_reason = "strike_above_sigma1%"
                elif 0.0 <= _sigma_dist_h_c < 1.0:
                    _sigma_block_reason = "spot_exhaustion_sigma1%"
                if _sigma_block_reason is not None:
                    _otm_yes_blocked = True; _otm_yes_block_gate = "swing_high_gate"
                    print(f"  [swing_high_gate_sigma] BLOCK YES {c['ticker']} — "
                          f"sigma_1% sh=${_sigma_swing_high:,.0f} "
                          f"dist={_sigma_dist_h_c:+.2f}%, c_trend={_active_trend}<=1 "
                          f"({_sigma_block_reason})")
                    gate_audit_logger.log_block(
                        gate_name="swing_high_gate",
                        ticker=c["ticker"], asset=args.asset, side="yes",
                        pm=pm, p_model=p_model_comp or float("nan"),
                        net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                        offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                        count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                        signals={"sigma_swing_high_1pct": round(_sigma_swing_high, 2),
                                 "sigma_dist_high_1pct":  round(_sigma_dist_h_c, 2),
                                 "composite_trend":       _active_trend,
                                 "reason":                _sigma_block_reason},
                        now_utc=now_utc, bankroll=args.bankroll,
                    )

            # [itm_yes_sh_gate — BTC ITM YES, DC swing high proximity]
            # Block YES when ITM strike is within 0.05% of the last confirmed sigma-1% swing high.
            # Mechanism: price broke above prior swing high (making YES ITM), but the broken
            # resistance acts as a snap-back magnet within the 2h window.
            # Backtest (base): n=262, WR=39.3% vs p_mkt=59.2%, resid=-19.84%, PnL=+$5,198 flat.
            # Rescue bp_1h>=0.6 (strong buying pressure holds the break): gates tighten to n=134,
            # WR=9.7%, PnL=+$6,498 (+$1,300 delta). Rescued set WR=70.3% (positive EV YES bets).
            # MCPT p=0.0000. (2026-06-04)
            if (args.asset == "BTC" and not _otm_yes_blocked
                    and _sigma_swing_high is not None
                    and _sigma_swing_low is not None
                    and offset_c < 0          # ITM YES: strike below spot
                    and 0.40 <= pm < 0.70):
                _itm_d_to_sh = abs(_sigma_swing_high - s_k) / spot
                _itm_d_to_sl = abs(s_k - _sigma_swing_low) / spot
                if _itm_d_to_sh <= 0.0005 and _itm_d_to_sh < _itm_d_to_sl:
                    _bp1h_itm = _bp_1h_mtf if not math.isnan(_bp_1h_mtf) else 0.5
                    _itm_sh_rescued = _bp1h_itm >= 0.6
                    if _itm_sh_rescued:
                        print(f"  [itm_yes_sh_gate] RESCUE YES {c['ticker']} — "
                              f"sh=${_sigma_swing_high:,.0f} d={_itm_d_to_sh*100:.3f}% "
                              f"bp_1h={_bp1h_itm:.3f}>=0.6 (buyers holding break)")
                    else:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "itm_yes_sh_gate"
                        print(f"  [itm_yes_sh_gate] BLOCK YES {c['ticker']} — "
                              f"ITM strike=${s_k:,.0f} within {_itm_d_to_sh*100:.3f}% of "
                              f"sh=${_sigma_swing_high:,.0f}, pm={pm:.3f}, "
                              f"bp_1h={_bp1h_itm:.3f}<0.6 (snap-back risk)")
                        gate_audit_logger.log_block(
                            gate_name="itm_yes_sh_gate",
                            ticker=c["ticker"], asset=args.asset, side="yes",
                            pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                            count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                            signals={
                                "sigma_swing_high_1pct": round(_sigma_swing_high, 2),
                                "d_to_sh_pct":           round(_itm_d_to_sh * 100, 4),
                                "d_to_sl_pct":           round(_itm_d_to_sl * 100, 4),
                                "bp_1h":                 round(_bp1h_itm, 4),
                            },
                            now_utc=now_utc, bankroll=args.bankroll,
                        )

            # [itm_yes_rw_gate — BTC ITM YES, RW order=5 nearest-by-price top]
            # Complements itm_yes_sh_gate: catches short-term local resistance tops that
            # sigma-DC misses (only 30/908 overlap). Same snap-back mechanism: strike near
            # a recently confirmed top with no buying pressure → price reverts.
            # Backtest: n=620, WR=38.1% vs p_mkt=60.7%, resid=-22.64%, PnL=+$14,036 flat.
            # MCPT p=0.0000 (z=+46.4). Temporal stable: early -23.5%, late -21.8%.
            # Rescue: bp_1h>=0.6 (buyers defending the break → don't block).
            # (2026-06-04)
            if (args.asset == "BTC" and not _otm_yes_blocked
                    and _rw_tops is not None
                    and offset_c < 0          # ITM YES: strike below spot
                    and 0.40 <= pm < 0.70):
                _rw_min_dist: "float | None" = None
                _rw_nearest_top: "float | None" = None
                for _rw_top_price, _rw_conf_ts in reversed(_rw_tops):
                    if _rw_top_price <= s_k:
                        continue                           # only tops ABOVE strike
                    _d = abs(_rw_top_price - s_k) / spot
                    if _rw_min_dist is None or _d < _rw_min_dist:
                        _rw_min_dist    = _d
                        _rw_nearest_top = _rw_top_price
                    if _d > 0.005:                        # stop early — tops sorted by time,
                        pass                              # but prices vary; scan all in window
                if _rw_min_dist is not None and _rw_min_dist <= 0.0005:
                    _bp1h_rw = _bp_1h_mtf if not math.isnan(_bp_1h_mtf) else 0.5
                    _rw_rescued = _bp1h_rw >= 0.6
                    if _rw_rescued:
                        print(f"  [itm_yes_rw_gate] RESCUE YES {c['ticker']} — "
                              f"rw_top=${_rw_nearest_top:,.0f} d={_rw_min_dist*100:.3f}% "
                              f"bp_1h={_bp1h_rw:.3f}>=0.6 (buyers holding break)")
                    else:
                        _otm_yes_blocked = True; _otm_yes_block_gate = "itm_yes_rw_gate"
                        print(f"  [itm_yes_rw_gate] BLOCK YES {c['ticker']} — "
                              f"ITM strike=${s_k:,.0f} within {_rw_min_dist*100:.3f}% of "
                              f"rw_top=${_rw_nearest_top:,.0f}, pm={pm:.3f}, "
                              f"bp_1h={_bp1h_rw:.3f}<0.6")
                        gate_audit_logger.log_block(
                            gate_name="itm_yes_rw_gate",
                            ticker=c["ticker"], asset=args.asset, side="yes",
                            pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                            count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                            signals={
                                "rw_nearest_top":  round(_rw_nearest_top, 2),
                                "rw_d_to_top_pct": round(_rw_min_dist * 100, 4),
                                "bp_1h":           round(_bp1h_rw, 4),
                            },
                            now_utc=now_utc, bankroll=args.bankroll,
                        )

            # [bull_flag_ct2_yes_gate — BTC YES]
            # Block YES when a bull flag/pennant is active AND composite_trend>=2.
            # After a bull flag breaks out, the market prices in continuation immediately;
            # WR=34.7% vs bkev=40.8% (edge=-6.1%) for ct>=2 rows the ct<=1 gate misses.
            # Additive value: +$575 on top of existing gate stack (p=0.059, n=1,006).
            # Causal story: flag breakout = short-term exhaustion; post-breakout reversion
            # likely within the 1h contract window even in a strong bull trend.
            # Audit at 50+ fires: verify WR<bkev, check if freshness (bars_ago) matters.
            if (args.asset == "BTC" and _composite_computed and not _otm_yes_blocked
                    and _flag_signal == 1 and _active_trend >= 2):
                _otm_yes_blocked = True; _otm_yes_block_gate = "bull_flag_ct2_yes_gate"
                print(f"  [bull_flag_ct2_yes_gate] BLOCK YES {c['ticker']} — "
                      f"bull_flag {_flag_bull_bars_ago}h ago "
                      f"pole={_flag_bull_pole_pct:.1f}%, c_trend={_active_trend}>=2 "
                      f"(post-breakout reversion)")
                gate_audit_logger.log_block(
                    gate_name="bull_flag_ct2_yes_gate",
                    ticker=c["ticker"], asset=args.asset, side="yes",
                    pm=pm, p_model=p_model_comp or float("nan"),
                    net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                    offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                    count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                    signals={
                        "flag_signal":        _flag_signal,
                        "flag_bull_bars_ago": _flag_bull_bars_ago,
                        "flag_bull_pole_pct": round(_flag_bull_pole_pct, 2),
                        "composite_trend":    _active_trend,
                    },
                    now_utc=now_utc, bankroll=args.bankroll,
                )

            # [adx_mid_ct_neg_yes_gate — BTC YES]
            # Block YES when ADX ∈ [30,40) AND composite_trend < 0.
            # Backtest: n=8,081, WR=80.7%, BEV=88.3%, Edge=-7.6%, PnL=-$614, MCPT z=+20.81 p=0.0000.
            # Logic: moderate trend (ADX 30-40) + bearish composite = YES priced for upside that
            # the composite signals don't support.  No rescue needed — ct>=0 is already excluded.
            if (args.asset == "BTC" and _composite_computed and not _otm_yes_blocked
                    and hasattr(confirm, "adx_1h") and confirm.adx_1h == confirm.adx_1h
                    and 30.0 <= float(confirm.adx_1h) < 40.0
                    and _active_trend < 0):
                _adx_mid = float(confirm.adx_1h)
                _otm_yes_blocked = True
                _otm_yes_block_gate = "adx_mid_ct_neg_yes_gate"
                print(f"  [adx_mid_ct_neg_yes_gate] BLOCK YES {c['ticker']} — "
                      f"adx={_adx_mid:.1f}∈[30,40) c_trend={_active_trend}<0 "
                      f"(moderate trend against YES direction)")
                gate_audit_logger.log_block(
                    gate_name="adx_mid_ct_neg_yes_gate",
                    ticker=c["ticker"], asset=args.asset, side="yes",
                    pm=pm, p_model=p_model_comp or float("nan"),
                    net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                    offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                    count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                    signals={
                        "adx_1h":          round(_adx_mid, 1),
                        "composite_trend": _active_trend,
                    },
                    now_utc=now_utc, bankroll=args.bankroll,
                )

            # [vsa_no_flip_gate — BTC YES block + NO flip]
            # VSA (Volume Spread Analysis) detects bullish absorption failures: high buying volume
            # but the range is too wide = distribution, not accumulation. Market prices YES at ~91%
            # but these contracts resolve YES only 44% of the time when VSA fires.
            # Backtest: n=116 distinct, WR_no=51.7% vs BEV=8.6%, MCPT z=+14.06 p=0.0000.
            # NO-side decision tree applied at no_pm_floor bypass below.
            # Backup: paper_trade_runner_pre_vsa_no_flip_20260605.py
            if (args.asset == "BTC" and _composite_computed and not _otm_yes_blocked
                    and pm > 0.50
                    and not math.isnan(_vsa_pressure24) and not math.isnan(_vsa_sdz)
                    and _vsa_pressure24 >= _VSA_P24_Q80 and _vsa_sdz >= _VSA_SDZ_MIN):
                _otm_yes_blocked = True
                _otm_yes_block_gate = "vsa_no_flip_gate"
                _vsa_no_flip = True
                print(f"  [vsa_no_flip_gate] BLOCK YES {c['ticker']} — "
                      f"pressure_24={_vsa_pressure24:.2f}>={_VSA_P24_Q80} "
                      f"sdz={_vsa_sdz:.2f}>={_VSA_SDZ_MIN} pm={pm:.3f}>0.50 "
                      f"(bullish absorption failure → flip to NO)")
                gate_audit_logger.log_block(
                    gate_name="vsa_no_flip_gate",
                    ticker=c["ticker"], asset=args.asset, side="yes",
                    pm=pm, p_model=p_model_comp or float("nan"),
                    net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                    offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                    count=0, kelly_fraction=0.0, close_ts=c.get("close_time", ""),
                    signals={
                        "vsa_pressure24":  round(_vsa_pressure24, 3),
                        "vsa_sdz":         round(_vsa_sdz, 3),
                        "composite_p_up":  round(_comp_p_up, 4),
                        "composite_trend": _active_trend,
                        "rvol_1h":         round(_rvol_1h, 4) if not math.isnan(_rvol_1h) else None,
                    },
                    now_utc=now_utc, bankroll=args.bankroll,
                )

            # [SMC YES gate — BTC composite only]
            # Block YES when 4h structural context is bearish or spot is at structural resistance.
            # ChoCH alone requires 1h also bearish — if 1h has recovered to bullish, allow YES.
            # bos_4h=bearish (confirmed multi-bar breakdown) blocks regardless of 1h.
            # Reform (2026-05-16): bearish structure block scoped to OTM YES (pm < 0.35) only.
            # Sim n=1,075 (8 days): near-ATM YES (pm>=0.35) WR=60.1% vs BE=54.3%, +$2,789 delta.
            # z_drift already suppresses YES in bearish conditions via lower p_model; hard block
            # at pm>=0.35 over-corrects and cuts profitable near-ATM entries.
            # Supply zone and swing-low proximity blocks unchanged (structural resistance, not trend).
            _SMC_BEARISH_PM_MAX = 0.35  # only block OTM YES in bearish structure
            if args.asset == "BTC" and _smc is not None:
                _smc_net_edge = max(0.0, (p_model_comp or 0.0) - pm - 0.04)
                _smc_log = lambda reason: gate_audit_logger.log_block(
                    gate_name="smc_gate", ticker=c["ticker"], asset=args.asset,
                    side="yes", pm=pm, p_model=p_model_comp, net_edge=_smc_net_edge,
                    offset_pct=offset_c, strike=s_k, spot=spot, tau_minutes=tau_c,
                    count=0, kelly_fraction=0.0, close_ts=c.get("close_time",""),
                    signals={
                        "ema_stack_bias":    confirm.ema_stack_bias,
                        "composite_trend":   _active_trend,
                        "composite_rev":     _active_rev,
                        "composite_p_up":    _comp_p_up,
                        "stoch_k":           confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else None,
                        "vwap_stretch":      confirm.stretch_score,
                        "vol_score":         confirm.vol_score,
                        "vpin_score":        confirm.vpin_score,
                        "obi_score":         confirm.obi_score,
                        "ema_stretch":       confirm.ema_stretch_score,
                        "structure_bias":    struct.structure_bias,
                        "funding_bias":      confirm.funding_bias,
                        "sharp_move_active": _sharp_move_active,
                        "smc_reason":        reason,
                    },
                    now_utc=now_utc,
                    bankroll=args.bankroll,
                )
                # [interaction_rescue] full_opportunity_analysis 2026-05-17 — combos that override
                # SMC blocking across 99k blocked trades. Checked once, applied to all smc_gate paths.
                # Tier 1 (apply to all block paths): extreme-conviction setups market doesn't price.
                # Tier 2 (apply to structural blocks only, not supply/swing-low levels): high-conviction.
                _smc_sk = confirm.stoch_k if (confirm.stoch_k == confirm.stoch_k) else 0.0
                _smc_rescue_why = None
                if confirm.ema_stack_bias == 1 and confirm.funding_bias == -1:
                    _smc_rescue_why = "ema=1+fund=-1"        # WR=96.3% edge=+41.8% n=459
                elif struct.structure_bias == 0 and confirm.funding_bias == -1:
                    _smc_rescue_why = "struct=0+fund=-1"     # WR=77.9% edge=+19.9% n=1616
                elif _smc_sk >= 70.0 and struct.structure_bias == 0:
                    _smc_rescue_why = "stoch>=70+struct=0"   # WR=75.5%@>=80 / 40.3%@[70,80)+struct=0; BE=36.2%; +$4,563 would_pnl
                elif _smc_sk >= 70.0 and confirm.ema_stack_bias == 1:
                    _smc_rescue_why = "stoch>=70+ema=1"      # WR=72.2%@>=80 / 49.8%@[70,80)+ema=1;   BE=49.2%; +$2,821 would_pnl
                elif _active_rev <= -2 and confirm.ema_stack_bias == 1:
                    _smc_rescue_why = "cr<=-2+ema=1"         # WR=68.0% edge=+7.0%  n=3888
                elif _active_trend == 0 and confirm.funding_bias == -1:
                    _smc_rescue_why = "ct=0+fund=-1"         # WR=69.7% edge=+9.3%  n=2664
                # Tier 1 = ema=1+fund=-1 or struct=0+fund=-1 (override supply/swing-low too)
                _smc_tier1_rescue = _smc_rescue_why in ("ema=1+fund=-1", "struct=0+fund=-1")

                if (_smc.choch_4h and _smc.bos_1h == "bearish") or _smc.bos_4h == "bearish":
                    if pm < _SMC_BEARISH_PM_MAX:
                        if _smc_rescue_why:
                            print(f"  [smc_gate] RESCUED YES {c['ticker']} — {_smc_rescue_why} "
                                  f"overrides bearish structure (pm={pm:.3f})")
                        else:
                            _smc_yes_blocked = True
                            print(f"  [smc_gate] BLOCK YES {c['ticker']} — 4h bearish structure + OTM "
                                  f"(pm={pm:.3f}<{_SMC_BEARISH_PM_MAX}, choch_4h={_smc.choch_4h}, bos_4h={_smc.bos_4h})")
                            _smc_log("bearish_structure_otm")
                    else:
                        print(f"  [smc_gate] ALLOW YES {c['ticker']} — 4h bearish but pm={pm:.3f}>={_SMC_BEARISH_PM_MAX} (near-ATM, z_drift handles)")
                elif _smc.in_supply_zone:
                    if _smc_tier1_rescue:
                        print(f"  [smc_gate] RESCUED YES {c['ticker']} — {_smc_rescue_why} overrides supply zone")
                    else:
                        _smc_yes_blocked = True
                        print(f"  [smc_gate] BLOCK YES {c['ticker']} — spot in 4h supply zone")
                        _smc_log("supply_zone")
                elif _smc.swing_low_1h is not None and spot > _smc.swing_low_1h:
                    _sl1h_dist = (spot - _smc.swing_low_1h) / _smc.swing_low_1h
                    if _sl1h_dist < 0.003:
                        if _smc_tier1_rescue:
                            print(f"  [smc_gate] RESCUED YES {c['ticker']} — {_smc_rescue_why} "
                                  f"overrides swing_low_proximity ({_sl1h_dist*100:.2f}% above sl_1h)")
                        else:
                            _smc_yes_blocked = True
                            print(f"  [smc_gate] BLOCK YES {c['ticker']} — spot {_sl1h_dist*100:.2f}% "
                                  f"above sl_1h={_smc.swing_low_1h:.2f} (<0.30%)")
                            _smc_log("swing_low_proximity")

            p_yes_adj_c = max(0.03, min(0.97, p_model_comp + funding_delta))
            if c["ticker"] in already_traded:
                print(f"  [scan] Skipping {c['ticker']} — already traded this session")
                continue
            expiry_key = _expiry_prefix(c["ticker"])
            expiry_positions = already_traded_expiries.get(expiry_key, {"yes": [], "no": []})
            expiry_trade_count = len(expiry_positions["yes"]) + len(expiry_positions["no"])
            if expiry_trade_count >= 3:
                print(f"  [scan] Skipping {c['ticker']} — expiry limit reached ({expiry_trade_count}/3 trades)")
                continue
            # Use composite-derived confirmation scores for gate evaluation.
            # composite_to_confirmation() maps validated (trend, rev) signals to the
            # confirmation_score / no_score / ema_alignment API that evaluate_trade() expects.
            # Falls back to legacy confirm scores when composite is unavailable (non-BTC).
            # Use composite confirmation only when composite scoring actually ran
            # composite_active=True whenever composite successfully computed — including
            # (0,0) scores. The (0,0) path uses p_up=0.4788 (below baseline) which
            # produces minimal edge, naturally failing Gate 3 instead of getting a
            # phantom 12% edge from the legacy 0.65× calibration correction.
            _composite_active = _composite_computed
            if _composite_active:
                _cscore, _nscore, _ema_align = composite_to_confirmation(_active_trend, _active_rev)
            else:
                _cscore     = confirm.confirmation_score
                _nscore     = confirm.no_score
                _ema_align  = confirm.ema_alignment

            # [Dual YES/NO model — BTC composite only]
            # YES uses k_drift_yes=2.00 (via DRIFT_MULTIPLIER in composite_scorer.py).
            # NO uses an independent k_drift_no=0.30 model — NOT 1-p_yes.
            # We pass (1 - p_no_model) to evaluate_trade for NO so the formula
            # p_market - p_model gives the correct NO edge:
            #   edge = p_no_model - (1 - p_yes_market)  =  p_yes_market - (1 - p_no_model) ✓
            # Kelly sizing also works: p_no_kelly = 1 - (1 - p_no_model) = p_no_model ✓
            _p_no_btc = None
            if args.asset == "BTC" and _composite_computed and sigma_tau_c > 0:
                # NO drift: norm.ppf(p_up_v2) × rvol_inv × k=0.3 × √(τ/60); GARCH σ override
                _ve_no = _garch_ve_btc if (not math.isnan(_garch_ve_btc) and _garch_ve_btc > 0) else vol_eff_c
                _sigma_tau_no = _ve_no * math.sqrt(tau_c)
                _sq_no = math.sqrt(tau_c / 60.0)
                if _p_up_v2 is not None and _sigma_tau_no > 0:
                    from scipy.stats import norm as _norm_no
                    _z_drift_no = float(_norm_no.ppf(max(0.01, min(0.99, _p_up_v2)))) * _rvol_inv_btc * 0.3 * _sq_no
                else:
                    _z_drift_no = 0.0
                _p_no_btc = score_to_p_no_model(
                    _active_trend, _active_rev, spot, s_k,
                    _sigma_tau_no if _sigma_tau_no > 0 else sigma_tau_c,
                    asset="BTC", p_up_override=_comp_p_up_c, z_drift_override=_z_drift_no,
                )
                _pm_ask = c["ask"]
                _pm_bid = c["bid"]
                # [stoch_bounce — BTC YES/NO extreme-stoch trigger, MT_1h17_4h40]
                # When stoch_k(1h) < 17 + stoch_k(4h) < 40 + pm < 0.60 → YES pure lognormal.
                # When stoch_k(1h) > 83 + stoch_k(4h) > 60 + pm > 0.40 → NO pure lognormal.
                # 4h confirmation filters noise: backtest 84.2% WR, +$32.75/trade (19 trades).
                # Bypasses composite drift and most early gates; late gates (BearDrift, rvol, etc.) apply.
                _sk_bounce = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else 50.0
                if _df_4h_comp is not None and len(_df_4h_comp) >= 14:
                    _l4h = _df_4h_comp["low"]
                    _h4h = _df_4h_comp["high"]
                    _c4h_b = _df_4h_comp["close"]
                    _ll14 = _l4h.rolling(14, min_periods=14).min()
                    _hh14 = _h4h.rolling(14, min_periods=14).max()
                    _denom4h = (_hh14 - _ll14).replace(0, float("nan"))
                    _sk4h_raw = 100.0 * (_c4h_b - _ll14) / _denom4h
                    _sk4h_last = float(_sk4h_raw.iloc[-1])
                    _sk4h_bounce = _sk4h_last if not math.isnan(_sk4h_last) else 50.0
                else:
                    _sk4h_bounce = 50.0
                _stoch_bounce_yes = _sk_bounce < 17.0 and _sk4h_bounce < 50.0 and pm < 0.60
                _stoch_bounce_no  = _sk_bounce > 83.0 and _sk4h_bounce > 60.0 and pm > 0.40
                _p_bounce = max(0.03, min(0.97, prob_c.p_yes))
                if _stoch_bounce_yes:
                    _bounce_ctx = "RESCUE" if (_otm_yes_blocked or _smc_yes_blocked) else "TRIGGER"
                    print(f"  [stoch_bounce] {_bounce_ctx} YES {c['ticker']} — "
                          f"stoch_k={_sk_bounce:.1f}<17, stoch_k_4h={_sk4h_bounce:.1f}<50, "
                          f"pm={pm:.3f}<0.60, p_lognorm={_p_bounce:.3f}")
                    p_model_comp = _p_bounce
                if _stoch_bounce_no:
                    print(f"  [stoch_bounce] TRIGGER NO {c['ticker']} — "
                          f"stoch_k={_sk_bounce:.1f}>83, stoch_k_4h={_sk4h_bounce:.1f}>60, "
                          f"pm={pm:.3f}>0.40, p_lognorm={1-_p_bounce:.3f}")
                _dec_yes = None
                if _stoch_bounce_yes or (not _otm_yes_blocked and not _smc_yes_blocked):
                    _p_yes_eval = _p_bounce if _stoch_bounce_yes else p_yes_adj_c
                    _dec_yes = evaluate_trade(
                        struct.structure_bias, confirm.confirmation_bias,
                        _p_yes_eval, _pm_ask, args.bankroll,
                        confirmation_score=_cscore, no_score=_nscore,
                        obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                        ema_alignment=_ema_align, asset=args.asset,
                        composite_active=_composite_active, composite_p_up=_comp_p_up,
                        offset_pct=offset_c, force_side="yes")
                # [CoinGlass stablecoin OI 4h gate — BTC NO]
                # Backtest (1h paper trades): oi_stable_chg_4h>1% blocks 82 NO → WR=56.1%, -$545.
                # Rescue: pm<0.50 (NO is ITM — strike above spot; longs must push BTC up to flip outcome).
                # stoch_bounce rescues from this gate when overbought (stoch>83).
                _btc_oi_no_blocked = (not _stoch_bounce_no) and (not _vsa_no_flip) and (_cg is not None and _cg.oi_stable_pct_4h > 1.0 and pm >= 0.50)
                if _btc_oi_no_blocked:
                    print(f"  [cg_oi_stable_no_gate] BLOCK NO {c['ticker']} — oi_stable_4h={_cg.oi_stable_pct_4h:+.2f}%>1%, pm={pm:.3f}>=0.50 (OTM NO, crowded longs)")
                elif _cg is not None and _cg.oi_stable_pct_4h > 1.0 and pm < 0.50:
                    print(f"  [cg_oi_stable_no_gate] RESCUE NO {c['ticker']} — oi_stable_4h={_cg.oi_stable_pct_4h:+.2f}%>1% BUT pm={pm:.3f}<0.50 (ITM NO, longs need to push BTC up)")
                # [hmm_smc_s2_no_gate — BTC NO, State 2 Transition/ChoCH]
                # During ChoCH, NO bets at pm>0.80 (deep ITM YES) fail catastrophically:
                # n=112, WR=8%, PnL=-$2,398. Contract expires YES before structure flip completes.
                # Rescue: ema_stack=-1 + rvol_inv>0.8 (genuine orderly breakdown) OR bp_5m<0.30 (real sellers).
                # Without rescue: saves full -$2,398; rescued bucket (ema=-1+low-rvol): WR=66.1%, +$1,551.
                if not _btc_oi_no_blocked and not _stoch_bounce_no and not _vsa_no_flip and _hmm_smc_state == 2 and pm > 0.80:
                    _bp5m_s2   = _bp_5m if _bp_5m is not None else 0.5
                    _s2_rescue = (confirm.ema_stack_bias == -1 and _rvol_inv_btc > 0.8) or (_bp5m_s2 < 0.30)
                    if _s2_rescue:
                        _s2_why = ("ema=-1+rvol>0.8" if confirm.ema_stack_bias == -1 and _rvol_inv_btc > 0.8
                                   else "bp_5m<0.30")
                        print(f"  [hmm_smc_s2_no_gate] RESCUE NO {c['ticker']} — "
                              f"State2(ChoCH) pm={pm:.3f}>0.80 BUT {_s2_why} (genuine breakdown)")
                    else:
                        _btc_oi_no_blocked = True
                        print(f"  [hmm_smc_s2_no_gate] BLOCK NO {c['ticker']} — "
                              f"State2(ChoCH), pm={pm:.3f}>0.80 (WR=8% historically, structure flip too slow)")
                _p_no_eval = _p_bounce if _stoch_bounce_no else (1.0 - _p_no_btc)
                _dec_no = None if _btc_oi_no_blocked else evaluate_trade(
                    struct.structure_bias, confirm.confirmation_bias,
                    _p_no_eval, _pm_bid, args.bankroll,
                    confirmation_score=_cscore, no_score=_nscore,
                    obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                    ema_alignment=_ema_align, asset=args.asset,
                    composite_active=_composite_active, composite_p_up=_comp_p_up,
                    offset_pct=offset_c, force_side="no")
                if _dec_yes is not None and _dec_yes.decision == "trade" and _dec_no is not None and _dec_no.decision == "trade":
                    dec_c = _dec_yes if _dec_yes.net_edge >= _dec_no.net_edge else _dec_no
                elif _dec_yes is not None and _dec_yes.decision == "trade":
                    dec_c = _dec_yes
                elif _dec_no is not None and _dec_no.decision == "trade":
                    dec_c = _dec_no
                elif _dec_no is not None:
                    dec_c = _dec_no
                else:
                    # Both YES and NO hard-blocked before dec_c was computed — log each
                    # side to blocked_trades.csv so these contracts are visible in analysis.
                    _bt_signals = {
                        "ema_stack_bias":    confirm.ema_stack_bias,
                        "composite_trend":   _active_trend,
                        "composite_rev":     _active_rev,
                        "composite_p_up":    _comp_p_up_c,
                        "stoch_k":           confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else None,
                        "vwap_stretch":      confirm.stretch_score,
                        "vol_score":         confirm.vol_score,
                        "vpin_score":        confirm.vpin_score,
                        "obi_score":         confirm.obi_score,
                        "ema_stretch":       confirm.ema_stretch_score,
                        "structure_bias":    struct.structure_bias,
                        "funding_bias":      confirm.funding_bias,
                        "sharp_move_active": _sharp_move_active,
                        "liq_score":         _liq_signal.liq_score             if _liq_signal else "",
                        "liq_bias":          round(_liq_signal.liq_bias, 4)    if _liq_signal else "",
                        "ls_long_pct":       round(_liq_signal.ls_long_pct, 2) if _liq_signal else "",
                        "oi_chg_pct":        round(_liq_signal.oi_chg_pct, 4)  if _liq_signal else "",
                    }
                    _yes_edge = p_yes_adj_c - pm
                    # near_itm_gate already logged at fire point; skip to avoid double-log.
                    if _otm_yes_block_gate != "near_itm_gate":
                        gate_audit_logger.log_block(
                            gate_name=_otm_yes_block_gate or "btc_otm_yes_hardblock",
                            ticker=c["ticker"], asset=args.asset, side="yes",
                            pm=pm, p_model=p_model_comp, net_edge=_yes_edge,
                            offset_pct=offset_c, strike=s_k, spot=spot,
                            tau_minutes=tau_c, count=0, kelly_fraction=0.0,
                            close_ts=c.get("close_time", ""),
                            signals=_bt_signals, now_utc=now_utc, bankroll=args.bankroll,
                        )
                    _no_edge = (_p_no_btc - (1.0 - pm)) if _p_no_btc is not None else float("nan")
                    gate_audit_logger.log_block(
                        gate_name="cg_oi_stable_no_gate",
                        ticker=c["ticker"], asset=args.asset, side="no",
                        pm=pm,
                        p_model=(1.0 - _p_no_btc) if _p_no_btc is not None else None,
                        net_edge=_no_edge,
                        offset_pct=offset_c, strike=s_k, spot=spot,
                        tau_minutes=tau_c, count=0, kelly_fraction=0.0,
                        close_ts=c.get("close_time", ""),
                        signals=_bt_signals, now_utc=now_utc, bankroll=args.bankroll,
                    )
                    continue  # both YES and NO hard-blocked — GBDT does not override
            elif args.asset == "ETH" and _p_no_eth is not None:
                # ETH HYBRID dual-eval:
                #   YES → score_to_p_model (k=0.80, blended p_up) via p_yes_adj_c
                #   NO  → compute_p_no_direct (NO-specific ML model) via _p_no_eth
                # Pattern mirrors BTC: pass (1 - p_no) to evaluate_trade so the
                # internal edge formula p_model - p_market gives correct NO edge.
                _pm_ask = c["ask"]
                _pm_bid = c["bid"]
                # OTM YES gate: block pm<0.40 bets unless model edge is strong (net_edge>0.25).
                # Sweep on 566 paper YES trades: pm<0.40 segment n=119, P&L=-$3,414.
                # Rescue net_edge>0.25 recovers +$496; block remainder saves -$3,910.
                _eth_net_edge_yes = p_model_comp - pm
                _eth_otm_yes_blocked = pm < 0.40 and _eth_net_edge_yes <= 0.25
                if _eth_otm_yes_blocked:
                    print(f"  [eth_otm_yes_gate] BLOCK YES {c['ticker']} — pm={pm:.3f}<0.40, net_edge={_eth_net_edge_yes:.3f}<=0.25")
                # [G1] Hard block: offset >= 5% OTM. Data: n=30, WR=0.100 vs BE=0.198, -$1,041.
                # No rescue found — asymmetric face losses overwhelm any partial WR rescue.
                if not _eth_otm_yes_blocked and offset_c >= 0.05:
                    print(f"  [eth_g1_otm_gate] BLOCK YES {c['ticker']} — offset={offset_c*100:+.2f}%>=5% (deep OTM hard block)")
                    _eth_otm_yes_blocked = True
                # [eth_cheap_otm_gate] Block OTM YES when market prices it below 50¢.
                # Data (375 resolved): pm<0.50 + z>0 → N=35, WR=17%, PnL=-$1,164.
                # Rescue: z<=0 (ITM) → N=21, WR=55%, PnL=+$343 at same price range — allow.
                # Market correctly prices cheap OTM YES as difficult; lognormal edge is illusory.
                if not _eth_otm_yes_blocked and pm < 0.50 and prob_c.z_score > 0:
                    print(f"  [eth_cheap_otm_gate] BLOCK YES {c['ticker']} — pm={pm:.3f}<0.50, z={prob_c.z_score:.3f}>0 (cheap OTM, market is right)")
                    _eth_otm_yes_blocked = True
                # [G2] Block structure_bias=0 with no signal confirmation.
                # Data: n=50 blocked WR=0.440 vs BE=0.607; rescue n=23 WR=0.739 via vwap≠0 OR stoch_bias=-1.
                if (not _eth_otm_yes_blocked
                        and struct.structure_bias == 0
                        and confirm.vwap_score == 0
                        and confirm.stoch_bias != -1
                        and (_comp_p_up is None or _comp_p_up < 0.55)):
                    print(f"  [eth_g2_struct_gate] BLOCK YES {c['ticker']} — struct=0, vwap={confirm.vwap_score}, stoch_bias={confirm.stoch_bias}, p_up={_comp_p_up:.3f}<0.55 (no signal confirmation)")
                    _eth_otm_yes_blocked = True
                # ETH vol gate (YES) — block OTM YES when strike is unreachable given vol regime.
                # Mirrors btc_vol_gate: same BASE_Z=2.0, same _vol_factor already computed above.
                _eth_vol_gate_yes_blocked = False
                if offset_c > 0 and sigma_tau_c > 0:
                    _z_strike_abs = abs(math.log(s_k / spot) / sigma_tau_c)
                    _eth_vol_gate_z = 2.0 * _vol_factor
                    if _z_strike_abs > _eth_vol_gate_z:
                        print(f"  [eth_vol_gate] BLOCK YES {c['ticker']} — |z|={_z_strike_abs:.3f} > {_eth_vol_gate_z:.3f} (vol_factor={_vol_factor:.3f})")
                        _eth_vol_gate_yes_blocked = True
                # [CoinGlass exchange flow gate — ETH YES only]
                # Backtest (1h paper trades): flow>0.10% blocks 22 YES → WR=5.9%, PnL improvement +$673.
                # Rescue: z_score < 0 (strike below spot = ITM; ITM contracts survive inflow pressure).
                if _cg is not None and not _eth_otm_yes_blocked and _cg.exchange_flow_1d_pct > 0.10:
                    _cg_eth_rescue = (prob_c.z_score < 0)
                    if not _cg_eth_rescue:
                        _eth_otm_yes_blocked = True
                        print(f"  [cg_flow_gate] BLOCK YES {c['ticker']} — flow={_cg.exchange_flow_1d_pct:+.3f}%>0.10%, z={prob_c.z_score:.3f}>=0 (OTM)")
                    else:
                        print(f"  [cg_flow_gate] RESCUE YES {c['ticker']} — flow={_cg.exchange_flow_1d_pct:+.3f}% BUT z={prob_c.z_score:.3f}<0 (ITM)")
                # [CoinGlass taker 4h gate — ETH YES, OTM only]
                # Backtest (1h paper trades): taker_4h<1.00 blocks 169 YES → WR=60.9%, -$254.
                # Rescue: pm>=0.45 (WR=71.4%, pnl=+$561); block pm<0.45 (WR=10.3%, pnl=-$814).
                if _cg is not None and not _eth_otm_yes_blocked and _cg.taker_ratio_4h < 1.00:
                    if pm >= 0.45:
                        print(f"  [cg_taker_yes_gate] RESCUE YES {c['ticker']} — taker_4h={_cg.taker_ratio_4h:.3f}<1.00 BUT pm={pm:.3f}>=0.45 (ITM)")
                    else:
                        _eth_otm_yes_blocked = True
                        print(f"  [cg_taker_yes_gate] BLOCK YES {c['ticker']} — taker_4h={_cg.taker_ratio_4h:.3f}<1.00, pm={pm:.3f}<0.45 (OTM sellers dominant)")
                # [eth_no_squeeze_bull_gate — ETH YES]
                # Block YES when squeeze is NOT active AND composite_trend is bullish (+1).
                # Analysis (191 ETH YES trades with squeeze data logged, 36 in blocked set):
                #   squeeze=0 + trend>0  → WR=46.7%, ≈-$118 (hard block: 15 trades — trend-chasing)
                #   squeeze=0 + trend<=0 → WR=71.4%, +$45  (rescue: 21 trades — mean-reversion edge)
                # Rationale: without squeeze (no vol-compression breakout pending), bullish YES is
                # pure trend-chasing; the market already priced it. Bearish/neutral trend with no
                # squeeze = oversold setup where YES has mean-reversion edge despite no momentum.
                if not _eth_otm_yes_blocked and not _eth_vol_gate_yes_blocked:
                    _squeeze_on = bool(getattr(confirm, "squeeze_1h", False))
                    if not _squeeze_on and _active_trend > 0:
                        _eth_otm_yes_blocked = True
                        print(f"  [eth_no_squeeze_bull_gate] BLOCK YES {c['ticker']} — "
                              f"squeeze=off, trend={_active_trend}>0 "
                              f"(bullish trend-chase without breakout catalyst)")
                    elif not _squeeze_on and _active_trend <= 0:
                        print(f"  [eth_no_squeeze_bull_gate] RESCUE YES {c['ticker']} — "
                              f"squeeze=off but trend={_active_trend}<=0 "
                              f"(mean-reversion setup, allow)")

                # [eth_r1_otm_yes_gate — ETH YES, OTM only, semi-Markov zones]
                # Block OTM YES (offset>0) when vol regime is R1 (high-vol).
                # In R1, model underestimates σ for OTM strikes; market already prices elevated vol.
                #
                # Semi-Markov structure (ETH-specific, from sojourn p25/p90):
                #   Early R1 (1–3 bars, ~45min): vol just spiked. Block ALL — ct>=2 rescue
                #     still loses (WR=16.9%, edge=−11.4%); no rescue worth taking.
                #   Mid R1 (4–32 bars): settled episode. Rescue ct>=2 (WR=45.6%, edge=+8.8%).
                #   Deep R1 (33+ bars): too few observations; no gate applied yet.
                #
                # Validation: MCPT p=0.000, refined A+B delta=+$527.96 (n=457 blocked).
                # Backup: paper_trade_runner_pre_eth_sol_hmm_20260604.py
                if (not _eth_otm_yes_blocked and not _eth_vol_gate_yes_blocked
                        and offset_c > 0
                        and _hmm_vol_probs_eth_live is not None
                        and _hmm_vol_probs_eth_live[0] == 1):
                    _eth_r1_tis  = _hmm_vol_probs_eth_live[2]
                    _eth_r1_zone = ("early" if _eth_r1_tis <= 3
                                    else "mid"  if _eth_r1_tis <= 32
                                    else "deep")
                    _block_reason = None
                    if _eth_r1_zone == "early":
                        # No rescue in early R1 — trend rescue also loses
                        _block_reason = f"early R1 (t={_eth_r1_tis}≤3 bars), no rescue"
                    elif _eth_r1_zone == "mid" and _active_trend < 2:
                        _block_reason = f"mid R1 (t={_eth_r1_tis}, 4–32 bars), c_trend={_active_trend}<2"
                    elif _eth_r1_zone == "mid" and _active_trend >= 2:
                        print(f"  [eth_r1_otm_yes_gate] RESCUE YES {c['ticker']} — "
                              f"mid R1 (t={_eth_r1_tis}) BUT c_trend={_active_trend}>=2 "
                              f"(bull momentum, WR=45.6% in mid-R1 rescues)")
                    # deep R1: no gate (insufficient data)

                    if _block_reason is not None:
                        _eth_otm_yes_blocked = True
                        print(f"  [eth_r1_otm_yes_gate] BLOCK YES {c['ticker']} — "
                              f"R1 vol regime, offset={offset_c*100:+.2f}%>0, {_block_reason}")
                        gate_audit_logger.log_block(
                            gate_name="eth_r1_otm_yes_gate",
                            ticker=c["ticker"], asset=args.asset, side="yes",
                            pm=pm, p_model=p_model_comp or float("nan"),
                            net_edge=max(0.0, (p_model_comp or 0.0) - pm - 0.04),
                            offset_pct=offset_c, strike=s_k, spot=spot,
                            tau_minutes=tau_c, count=0, kelly_fraction=0.0,
                            close_ts=c.get("close_time", ""),
                            signals={
                                "hmm_vol_state":   _hmm_vol_probs_eth_live[0],
                                "hmm_r1_prob":     _hmm_vol_probs_eth_live[1],
                                "hmm_tis":         _eth_r1_tis,
                                "hmm_zone":        _eth_r1_zone,
                                "composite_trend": _active_trend,
                            },
                            now_utc=now_utc, bankroll=args.bankroll,
                        )

                _dec_yes = None
                if not _eth_otm_yes_blocked and not _eth_vol_gate_yes_blocked:
                    _dec_yes = evaluate_trade(
                        struct.structure_bias, confirm.confirmation_bias,
                        p_yes_adj_c, _pm_ask, args.bankroll,
                        confirmation_score=_cscore, no_score=_nscore,
                        obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                        ema_alignment=_ema_align, asset=args.asset,
                        composite_active=_composite_active, composite_p_up=_comp_p_up,
                        offset_pct=offset_c, force_side="yes")
                # ETH vol gate (NO) — block OTM NO when required downward move is unreachable.
                # OTM NO = offset < 0 (strike below spot, NO needs price to drop past strike).
                _eth_vol_gate_no_blocked = False
                if offset_c < 0 and sigma_tau_c > 0:
                    _z_no_otm = abs(math.log(s_k / spot) / sigma_tau_c)
                    if _z_no_otm > 2.0 * _vol_factor:
                        print(f"  [eth_no_vol_gate] BLOCK NO {c['ticker']} — |z|={_z_no_otm:.3f} > {2.0*_vol_factor:.3f} (OTM NO, vol_factor={_vol_factor:.3f})")
                        _eth_vol_gate_no_blocked = True
                _p_no_eth_adj = max(0.01, min(0.99, _p_no_eth - funding_delta))
                _dec_no = None if _eth_vol_gate_no_blocked else evaluate_trade(
                    struct.structure_bias, confirm.confirmation_bias,
                    1.0 - _p_no_eth_adj, _pm_bid, args.bankroll,
                    confirmation_score=_cscore, no_score=_nscore,
                    obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                    ema_alignment=_ema_align, asset=args.asset,
                    composite_active=_composite_active, composite_p_up=_comp_p_up,
                    offset_pct=offset_c, force_side="no")
                if _dec_yes is not None and _dec_yes.decision == "trade" and _dec_no is not None and _dec_no.decision == "trade":
                    dec_c = _dec_yes if _dec_yes.net_edge >= _dec_no.net_edge else _dec_no
                elif _dec_yes is not None and _dec_yes.decision == "trade":
                    dec_c = _dec_yes
                elif _dec_no is not None and _dec_no.decision == "trade":
                    dec_c = _dec_no
                elif _dec_no is not None:
                    dec_c = _dec_no
                else:
                    continue  # both sides vol-gate blocked (structurally impossible — YES/NO gates require opposite offset signs)
            else:
                dec_c = evaluate_trade(
                    struct.structure_bias, confirm.confirmation_bias,
                    p_yes_adj_c, pm, args.bankroll,
                    confirmation_score=_cscore, no_score=_nscore,
                    obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                    ema_alignment=_ema_align, asset=args.asset,
                    composite_active=_composite_active, composite_p_up=_comp_p_up,
                    offset_pct=offset_c, p_market_bid=c["bid"], p_market_ask=c["ask"])
            # Update pm to side-specific fill-price reference:
            # YES bet fills at YES ask; NO bet fills at 1 - YES bid (so YES bid is reference).
            # Using bid/ask (not mid) prevents edge inflation on wide-spread contracts.
            pm = c["bid"] if dec_c.side == "no" else c["ask"]

            # Order book depth signal — fetches Coinbase spot book (30s cache); per-contract strike.
            _ob = orderbook_depth.fetch_ob_signal(s_k, spot, asset=args.asset)

            # Signal snapshot for gate audit logging — captured once per contract evaluation.
            _gate_signals = {
                "ema_stack_bias":    confirm.ema_stack_bias,
                "composite_trend":   _active_trend,
                "composite_rev":     _active_rev,
                "composite_p_up":    _comp_p_up,
                "stoch_k":           confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else None,
                "vwap_stretch":      confirm.stretch_score,
                "vol_score":         confirm.vol_score,
                "vpin_score":        confirm.vpin_score,
                "obi_score":         confirm.obi_score,
                "ema_stretch":       confirm.ema_stretch_score,
                "structure_bias":    struct.structure_bias,
                "funding_bias":      confirm.funding_bias,
                "sharp_move_active": _sharp_move_active,
                "liq_score":         _liq_signal.liq_score             if _liq_signal else "",
                "liq_bias":          round(_liq_signal.liq_bias, 4)    if _liq_signal else "",
                "ls_long_pct":       round(_liq_signal.ls_long_pct, 2) if _liq_signal else "",
                "oi_chg_pct":        round(_liq_signal.oi_chg_pct, 4)  if _liq_signal else "",
                "ob_imbalance":      _ob.imbalance      if _ob else "",
                "ob_path_ask_usd":   _ob.path_ask_usd   if _ob else "",
                "ob_path_bid_usd":   _ob.path_bid_usd   if _ob else "",
                "ob_ask_frac":       _ob.ask_frac        if _ob else "",
                "ob_bid_wall_pct":   _ob.bid_wall_pct    if _ob else "",
                "ob_ask_wall_pct":   _ob.ask_wall_pct    if _ob else "",
                "pc1_rsi":           _pc1_rsi if not math.isnan(_pc1_rsi) else "",
            }

            def _log_block(gate_name: str) -> None:
                # count is only computed at order execution; estimate from bet_amount / cost_per_contract
                _cost_per  = pm if dec_c.side == "yes" else max(1.0 - pm, 0.01)
                _est_count = min(int(dec_c.bet_amount / _cost_per), args.max_contracts) if _cost_per > 0 else 0
                gate_audit_logger.log_block(
                    gate_name=gate_name,
                    ticker=c["ticker"],
                    asset=args.asset,
                    side=dec_c.side,
                    pm=pm,
                    p_model=p_model_comp,
                    net_edge=dec_c.net_edge,
                    offset_pct=offset_c,
                    strike=s_k,
                    spot=spot,
                    tau_minutes=tau_c,
                    count=_est_count,
                    kelly_fraction=dec_c.kelly_fraction,
                    close_ts=c.get("close_time", ""),
                    signals=_gate_signals,
                    now_utc=now_utc,
                    bankroll=args.bankroll,
                )
                if args.asset == "BTC":
                    _gm_pipe = _load_gate_meta_lgbm()
                    if _gm_pipe is not None:
                        _gm_feats = {
                            "side_enc":        1.0 if dec_c.side == "yes" else 0.0,
                            "pm":              pm,
                            "p_model":         p_model_comp or float("nan"),
                            "net_edge":        dec_c.net_edge,
                            "offset_pct":      offset_c,
                            "tau_minutes":     tau_c,
                            **{k: (float(v) if v is not None and v != "" else float("nan"))
                               for k, v in _gate_signals.items()
                               if k in ("ema_stack_bias","composite_trend","composite_rev",
                                        "composite_p_up","stoch_k","vwap_stretch","vol_score",
                                        "vpin_score","obi_score","ema_stretch","structure_bias",
                                        "funding_bias","liq_score","liq_bias","ls_long_pct","oi_chg_pct")},
                        }
                        _p_gm = _infer_gate_meta(_gm_pipe, gate_name, _gm_feats)
                        if _p_gm is not None:
                            _warn = "  ⚠ LIKELY WRONG" if _p_gm < 0.40 else ""
                            print(f"  [btc_gate_meta] {gate_name} p_correct={_p_gm:.3f}{_warn}")

            # ITM NO gate disabled 2026-05-03 — NO means BTC stays above strike, not drops
            # if dec_c.side == "no" and offset_c <= 0:
            #     print(f"  [scan] Skipping {c['ticker']} — ITM NO (offset={offset_c*100:+.3f}%, price already above strike)")
            #     continue
            # Minimum offset filters for NO bets — based on real Kalshi p_market analysis
            # (2026-04-07 backtest + paper trade archive, real pricing confirmed):
            #
            # BTC NO: < 0.10% — live win 54%, need 61%+. Min = 0.10%.
            # ETH NO: < 0.10% — near-ATM NO consistently loses. Min = 0.10%.
            #                   NOTE: 0.20% was tested but blocked all ETH trades in practice;
            #                   keeping at 0.10% to allow trade flow while building data.
            #                   Gate PM (p_market ≤ 0.35) provides the primary ETH NO filter.
            # SOL NO: < 0.20% — real Kalshi p_mkt 0.35-0.43 at < 0.20%, win rate 25-44% → losing.
            #                   ≥ 0.20%: live winners at 0.23-0.24% offset (n=2, 100% win).
            #                   NOTE: 0.50% was too aggressive — blocked all available SOL contracts.
            # Revert: copy paper_trade_runner_v3.py → paper_trade_runner.py
            # BTC NO minimum offset filter removed 2026-05-03 — let gate stack evaluate edge
            if dec_c.side == "no" and args.asset == "ETH" and offset_c < 0.001:
                print(f"  [scan] Skipping {c['ticker']} — ETH NO offset={offset_c*100:+.3f}% < 0.10% minimum")
                continue

            # Hard block: NO bets where YES price < 10¢ (R:R ≥ 9:1 unfavorable).
            # pm is c["bid"] for NO side (actual cost = 1 - pm; win = pm).
            # At pm=0.04: pay 96¢ to win 4¢ = 24:1 R:R, breakeven WR = 96% — structurally unfishable.
            # Decision.py has P_ETH_NO_PM_MIN for ETH, but this belt-and-suspenders runner check
            # catches any edge-case bypass path in the dual-eval routing.
            # VSA flip targets deep ITM YES (pm_bid ≈ 0.89), so this gate never fires for those.
            if dec_c.side == "no" and pm < 0.10:
                print(f"  [no_pm_floor] BLOCK NO {c['ticker']} — pm={pm:.3f}<0.10 "
                      f"(R:R={(1-pm)/max(pm,0.01):.0f}:1 unfavorable)")
                _log_block("no_pm_floor")
                continue

            # [vsa_no_flip_sizing] Force NO trade with tiered sizing when VSA absorption signal fires.
            # Upstream fixes (oi_stable, hmm_smc_s2) ensure _dec_no is computed. Here we override
            # the model edge (near-zero for deep ITM YES: p_model≈0.90 vs pm_bid≈0.89) and apply
            # conviction-based sizing. Backtest: n=51, WR=51.7%, MCPT z=+14.06, p=0.0000.
            # pm is c["bid"] ≈ YES_bid ≈ 0.89; cost = 1-pm ≈ 0.11; face × (1-pm) = bet_amount.
            if _vsa_no_flip and dec_c.side == "no":
                _vsa_rvol = _rvol_1h if not math.isnan(_rvol_1h) else 1.0
                _vsa_skip_reason = None
                if _vsa_rvol >= 1.5 and _active_trend > 0:
                    _vsa_skip_reason = (f"rvol_1h={_vsa_rvol:.2f}>=1.5 + ct={_active_trend}>0 "
                                        f"(real momentum, not absorption failure)")
                elif _comp_p_up >= 0.60:
                    _vsa_skip_reason = (f"cpu={_comp_p_up:.3f}>=0.60 "
                                        f"(YES model agrees with market)")
                if _vsa_skip_reason:
                    print(f"  [vsa_no_flip] SKIP NO {c['ticker']} — {_vsa_skip_reason}")
                    _log_block("vsa_no_flip_skip")
                    continue
                _vsa_face = 25.0 if _comp_p_up < 0.40 else (15.0 if _comp_p_up < 0.50 else 10.0)
                _vsa_bet  = round(_vsa_face * max(1.0 - pm, 0.01), 2)
                dec_c.decision       = "trade"
                dec_c.bet_amount     = _vsa_bet
                dec_c.kelly_fraction = _vsa_bet / max(args.bankroll, 1.0)
                dec_c.bet_fraction   = dec_c.kelly_fraction
                print(f"  [vsa_no_flip] FORCE NO trade {c['ticker']} — "
                      f"face=${_vsa_face:.0f} bet=${_vsa_bet:.2f} "
                      f"pm={pm:.3f} cpu={_comp_p_up:.3f} rvol={_vsa_rvol:.2f} ct={_active_trend}")

            # [hour_yes_gate] Block YES at UTC hours 13 and 16 — systematic YES losses.
            # mispricing_analysis.txt (2026-05-17): ALL assets executed YES trades:
            #   Hour 13: n=52, WR=40.4%, BE=58.2%, Edge=-17.8%, PnL=$-93, p=0.012
            #   Hour 16: n=55, WR=38.2%, BE=53.1%, Edge=-14.9%, PnL=$-82, p=0.028
            # Both are statistically significant across BTC+ETH+SOL.
            # NO bets at these hours are neutral/positive — only YES is affected.
            if dec_c.side == "yes" and now_utc.hour in {13, 16}:
                print(f"  [hour_yes_gate] BLOCK YES {c['ticker']} — "
                      f"hour={now_utc.hour}UTC (WR≈39% vs BE≈55%, p<0.03)")
                _log_block("hour_yes_gate")
                continue

            # [eth_vol_regime_gate] Block ETH when vol_ratio>1.20 AND ema_alignment != bearish.
            # High realized vol inflates sigma_tau → z_strike shrinks → model overconfident.
            # In neutral/bullish EMA alignment there is no directional anchor to overcome the
            # miscalibrated sigma. In bearish alignment YES bets are lower-strike (ITM) and
            # NO edge is real. n=78 blocked: WR=51.3%, BE=58.6%, P&L=-$810.
            # Rescue n=63 bearish: WR=73.0%, BE=65.8%, P&L=+$97. No secondary rescue found.
            if (args.asset == "ETH"
                    and vol_ratio_c is not None and vol_ratio_c > 1.20
                    and _ema_align != "bearish"):
                print(f"  [eth_vol_regime_gate] BLOCK {dec_c.side.upper()} {c['ticker']} — "
                      f"vol_ratio={vol_ratio_c:.3f}>1.20, ema_alignment={_ema_align} (not bearish)")
                _log_block("eth_vol_regime_gate")
                continue

            # [eth_squeeze_gate] Block ETH when squeeze_1h=True AND vol_ratio<0.80 AND stoch outside [30,60).
            # Compressed realized vol during a squeeze makes sigma tiny → z_strike inflated →
            # model near-certain NO. Stochastic reads are unreliable (false oversold/overbought).
            # stoch [30,60) is the neutral zone unaffected by squeeze distortion: WR=80%, +$20.
            # Blocked: n=7, WR=28.6%, BE=68.0%, P&L=-$121. Sample thin — monitor accumulation.
            if (args.asset == "ETH"
                    and getattr(confirm, "squeeze_1h", False)
                    and vol_ratio_c is not None and vol_ratio_c < 0.80):
                _sk_sq = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _in_rescue_zone = 30.0 <= _sk_sq < 60.0
                if not _in_rescue_zone:
                    print(f"  [eth_squeeze_gate] BLOCK {dec_c.side.upper()} {c['ticker']} — "
                          f"squeeze=True, vol_ratio={vol_ratio_c:.3f}<0.80, stoch_k={_sk_sq:.1f} not in [30,60)")
                    _log_block("eth_squeeze_gate")
                    continue
                else:
                    print(f"  [eth_squeeze_gate] RESCUED {dec_c.side.upper()} {c['ticker']} — "
                          f"squeeze=True, vol_ratio={vol_ratio_c:.3f}<0.80, stoch_k={_sk_sq:.1f} in [30,60)")

            # [eth_vol_yes_bearish_gate] Block YES when vol_ratio in [0.80,1.20) AND ema_alignment=bearish.
            # In transition vol zone, bearish EMA + YES bets without stoch/structure justification
            # are momentum-chasing into a headwind: 8 blocked trades, WR=12.5% (-45.6pp below BE),
            # P&L=-$260. chg_30m is positive for all 8 — price was rising into each bet.
            # Rescue (stoch_bias=1 OR structure_bias=-1): n=11, WR=81.8%, BE=67.5%, +$113.
            #   stoch_bias=1: deeply oversold stochastic (stoch_k 0-22) + falling price = mean-reversion.
            #   structure_bias=-1: confirmed bearish structure with ITM YES = floor already established.
            # Note: vol>1.20 zone is handled separately by eth_vol_regime_gate (opposite rescue).
            if (args.asset == "ETH"
                    and dec_c.side == "yes"
                    and vol_ratio_c is not None and 0.80 <= vol_ratio_c < 1.20
                    and _ema_align == "bearish"):
                _rescued = (confirm.stoch_bias == 1 or struct.structure_bias == -1)
                if not _rescued:
                    print(f"  [eth_vol_yes_bearish_gate] BLOCK YES {c['ticker']} — "
                          f"vol_ratio={vol_ratio_c:.3f} in [0.80,1.20), ema=bearish, "
                          f"stoch_bias={confirm.stoch_bias}, structure_bias={struct.structure_bias}")
                    _log_block("eth_vol_yes_bearish_gate")
                    continue
                else:
                    print(f"  [eth_vol_yes_bearish_gate] RESCUED YES {c['ticker']} — "
                          f"vol_ratio={vol_ratio_c:.3f} in [0.80,1.20), ema=bearish, "
                          f"stoch_bias={confirm.stoch_bias}, structure_bias={struct.structure_bias}")

            # [eth_funding_vol_yes_gate] Block ETH YES when funding_bias=-1 (shorts paying longs,
            # bearish sentiment) AND vol_ratio>=0.70 (realized vol not compressed).
            # Rescue: vol_ratio<0.70 = compressed vol makes strike more reachable despite headwind.
            # Analysis: funding_bias=-1 & vol_ratio>=0.70 saves ~$1,228; rescue recovers meaningful wins.
            if (args.asset == "ETH"
                    and dec_c.side == "yes"
                    and confirm.funding_bias == -1
                    and vol_ratio_c is not None):
                if vol_ratio_c >= 0.70:
                    print(f"  [eth_funding_vol_yes_gate] BLOCK YES {c['ticker']} — "
                          f"funding_bias=-1 (bearish), vol_ratio={vol_ratio_c:.3f}>=0.70")
                    _log_block("eth_funding_vol_yes_gate")
                    continue
                else:
                    print(f"  [eth_funding_vol_yes_gate] RESCUED YES {c['ticker']} — "
                          f"funding_bias=-1 but vol_ratio={vol_ratio_c:.3f}<0.70 (compressed)")

            # [eth_rev_vol_yes_gate] Block ETH YES when composite_rev in [3,5] AND vol_ratio>=0.70.
            # Moderate-high rev score loses when realized vol elevated — reversion move fails to
            # materialize cleanly. Rescue: vol_ratio<0.70 = compressed vol supports mean-reversion.
            # Analysis: rev [3,5] & vol_ratio>=0.70 saves ~$1,090.
            if (args.asset == "ETH"
                    and dec_c.side == "yes"
                    and 3 <= _active_rev <= 5
                    and vol_ratio_c is not None):
                if vol_ratio_c >= 0.70:
                    print(f"  [eth_rev_vol_yes_gate] BLOCK YES {c['ticker']} — "
                          f"rev={_active_rev} in [3,5], vol_ratio={vol_ratio_c:.3f}>=0.70")
                    _log_block("eth_rev_vol_yes_gate")
                    continue
                else:
                    print(f"  [eth_rev_vol_yes_gate] RESCUED YES {c['ticker']} — "
                          f"rev={_active_rev} in [3,5] but vol_ratio={vol_ratio_c:.3f}<0.70 (compressed)")

            # [eth_yesitm_no_gate] Block YES-ITM NO bets unless blow-off fade conditions.
            # s_k < spot = YES already in the money; NO needs full price reversal.
            # Rescue: rev>=2 (model sees overextension) AND 1h candle green (price actively
            # rose this hour, at peak of the extension move).
            # Sim n=18 (GREEN+rev>=2): WR=55.6%, BE=27.4%, WR-BE=+28.2pp, $/t=+$0.277.
            if (args.asset == "ETH" and dec_c.side == "no"
                    and dec_c.decision == "trade" and s_k < spot):
                _yesitm_rescue = (_active_rev >= 2 and _1h_candle_green
                                  and not _sharp_move_active)
                if _yesitm_rescue:
                    print(f"  [eth_yesitm_no_gate] RESCUED YES-ITM NO {c['ticker']} — "
                          f"rev={_active_rev}>=2 + GREEN candle (s_k={s_k:.2f}<spot={spot:.2f})")
                else:
                    _why = []
                    if _active_rev < 2:       _why.append(f"rev={_active_rev}<2")
                    if not _1h_candle_green:  _why.append("candle not green")
                    if _sharp_move_active:    _why.append("sharp_move")
                    print(f"  [eth_yesitm_no_gate] BLOCK YES-ITM NO {c['ticker']} — "
                          f"s_k={s_k:.2f}<spot={spot:.2f} ({', '.join(_why)})")
                    _log_block("eth_yesitm_no_gate")
                    continue

            # eth_bp_gate REMOVED 2026-05-31: blocked 82% WR trades (+$1,109 missed alltime).
            # Original backtest was on a different market distribution. Live tracking since
            # May 21 showed 128 NO blocks at 84% WR (+$983 missed), 20 YES blocks at 70% WR.
            # Near-ATM NO bets (offset<2%) never appeared in the blocked set — the gate only
            # fired on deep-OTM NO bets where 5m bp is irrelevant to 1h strike outcome.

            # [eth_yes_deepotm_gate] Block YES when offset_c>=0.001 (>=0.10% OTM = strike >$2+ above spot).
            # Analysis (2026-05-23, n=25): WR=0.0%, BE=15.3%, P&L=-$1,068.
            # ETH needs to rise $3-22 in one hour — exceeds typical hourly range.
            # Tested in daily-Bull (n=14, WR=0%) and Sideways (n=11, WR=0%) — regime-independent.
            # 4h/1h Bull Markov never observed in data window — rescue slot reserved, pending.
            # offset∈(0,0.001)+pm>=0.40: n=10, WR=60%, +$78 — barely-OTM stays allowed.
            if (args.asset == "ETH"
                    and dec_c.side == "yes"
                    and offset_c >= 0.001):
                print(f"  [eth_yes_deepotm_gate] BLOCK YES {c['ticker']} — "
                      f"offset={offset_c*100:+.3f}%>=0.10 "
                      f"(requires >${offset_c*spot:.0f} hourly move, 0-for-25 historically)")
                _log_block("eth_yes_deepotm_gate")
                continue

            # [eth_yes_chg5m_vwap_gate] Block YES when 5m rising + neutral/below VWAP + ITM.
            # Analysis (2026-05-23, n=44): WR=40.9%, BE=60.6%, P&L=-$635.
            # vwap_stretch=-2 rescue does not exist within this gate's population filter.
            # Only marginal rescue (confirmation_score>0, n=8, WR=62.5%) — insufficient.
            # confirmation_score==0 (n=36): WR=36.1%, -$659 — dominant driver.
            if (args.asset == "ETH"
                    and dec_c.side == "yes"
                    and _sharp_move_pct_5m > 0
                    and getattr(confirm, "stretch_score", None) in (-1, 0)
                    and offset_c <= 0):
                print(f"  [eth_yes_chg5m_vwap_gate] BLOCK YES {c['ticker']} — "
                      f"chg_5m={_sharp_move_pct_5m*100:+.3f}%>0, "
                      f"vwap_stretch={getattr(confirm,'stretch_score',None)}, "
                      f"offset={offset_c*100:+.3f}% ITM "
                      f"(rising into neutral VWAP, WR=40.9% vs BE=60.6%)")
                _log_block("eth_yes_chg5m_vwap_gate")
                continue

            # [eth_no_vwap_stretch2_gate] Block NO when vwap_stretch==2 AND composite_rev<=-1.
            # Refined 2026-06-04: adding rev<=-1 condition (overbought context).
            # Original gate (stretch==2 all) failed perm p=0.996 (WR≈bkev, no signal).
            # Refined (stretch==2 + rev<=-1): perm p=0.000, WF p=0.000; OOS edge=-3.3%.
            #   n=1411 (36 expiries): WR=21.3% vs bkev=23.9%; rev>-1 rows (8854) have edge=+0.6% — pass through.
            # Rescue: tau>40 + vol_score=0 → still valid (WR=18.3% vs bkev=26.7% within refined set).
            if (args.asset == "ETH"
                    and dec_c.side == "no"
                    and getattr(confirm, "stretch_score", None) == 2
                    and _active_rev <= -1):
                _tau_eth = float(tau_c) if tau_c is not None else 0.0
                _vol_sc_eth = getattr(confirm, "vol_score", None)
                _stretch2_rescue = (_tau_eth > 40 and _vol_sc_eth == 0)
                if not _stretch2_rescue:
                    print(f"  [eth_no_vwap_stretch2_gate] BLOCK NO {c['ticker']} — "
                          f"vwap_stretch=2, rev={_active_rev}<=-1, tau={_tau_eth:.0f}min, vol_score={_vol_sc_eth}")
                    _log_block("eth_no_vwap_stretch2_gate")
                    continue
                else:
                    print(f"  [eth_no_vwap_stretch2_gate] RESCUE NO {c['ticker']} — "
                          f"vwap_stretch=2+rev<=-1 but tau={_tau_eth:.0f}>40+vol=0")

            # [eth_no_adx_gate] Block NO when adx_1h>60 unless ema==-1 OR vwap_stretch==-2.
            # Refined 2026-06-04: raised threshold 40→60; replaced vol_ratio rescue with vwap_stretch==-2.
            # Original gate (adx>40) failed perm p=0.980 (adx 40-60 rows had edge≈0, diluting signal).
            # Refined (adx>60): perm p=0.000, WF p=0.000; blocked edge=-3.9% (1423 rows, 8 expiries).
            # Rescue ema==-1: edge=+2.8% (950 rows, 5 exp) — strong downtrend, NO bets win.
            # Rescue vwap_stretch==-2: edge=+5.3% (65 rows, 3 exp) — extended below VWAP, bounce risk.
            _adx_eth = (float(confirm.adx_1h)
                        if hasattr(confirm, "adx_1h")
                        and confirm.adx_1h is not None
                        and confirm.adx_1h == confirm.adx_1h
                        else None)
            if (args.asset == "ETH"
                    and dec_c.side == "no"
                    and _adx_eth is not None
                    and _adx_eth > 60):
                _ema_eth    = getattr(confirm, "ema_stack_bias", None)
                _stretch_eth = getattr(confirm, "stretch_score", None)
                _adx_rescue = (_ema_eth == -1 or _stretch_eth == -2)
                if not _adx_rescue:
                    print(f"  [eth_no_adx_gate] BLOCK NO {c['ticker']} — "
                          f"adx_1h={_adx_eth:.1f}>60, ema={_ema_eth}, vwap_stretch={_stretch_eth} "
                          f"(rescue: ema=-1 or vwap_stretch=-2)")
                    _log_block("eth_no_adx_gate")
                    continue
                else:
                    _rsrc_eth = "ema=-1" if _ema_eth == -1 else "vwap_stretch=-2"
                    print(f"  [eth_no_adx_gate] RESCUE NO {c['ticker']} — "
                          f"adx_1h={_adx_eth:.1f}>60 but {_rsrc_eth}")

            if dec_c.side == "no" and args.asset == "SOL" and offset_c < 0.002:
                print(f"  [scan] Skipping {c['ticker']} — SOL NO offset={offset_c*100:+.3f}% < 0.20% minimum")
                continue
            if _sharp_move_active and dec_c.decision == "trade":
                print(f"  [sharp_move] {c['ticker']} — inverted composite: side={dec_c.side.upper()} net={dec_c.net_edge:+.4f}")

            # btc_pup_gate removed: replaced by vol_factor-as-gate + k_drift=0.8 reform.

            # [markov_7state_gate — BTC Correction + Consolidation]
            # Replaces 3-state markov_sideways_gate. Uses 7-state HMM (log_ret, rv20d, ret5d, ret20d).
            # Validated 2026-06-04 on scan archive (real Kalshi payoffs):
            #
            # CORRECTION (YES=-9.4%, NO=+6.7% baseline):
            #   YES BLOCK by default.
            #   Rescue YES if (c_trend==0 AND ema∈{-1,0}) [edge=+8.0%] OR rvol∈[0.7,1.0) [+6.4%].
            #     Causal: neutral 4h trend = correction paused, price may bounce.
            #   NO BLOCK if (c_trend==0 OR rvol∈[0.7,1.0)) AND pm>=0.70.
            #     pm<0.70 RESCUED 2026-06-06: live analysis n=665 shows WR=79-94% vs BEV=18-82%,
            #     edge=+11-18%. OTM/near-ITM NO in Correction correctly wins (price falls away from strike).
            #     Deep-ITM NO pm>=0.70 remains blocked: edge=-9.9%, these lose even in Correction.
            #
            # CONSOLIDATION (YES=-2.1%, NO=+2.7% baseline):
            #   YES HARD-BLOCK if ema==+1 [edge=-10.6%, perm p=0.000] OR c_trend==2 [-10.1%, WF p=0.006].
            #     Causal: bullish EMA / mild uptrend in low-vol range → price extended, YES fails.
            #   YES SOFT otherwise (let normal edge gates filter).
            #   NO ALLOW across all conditions (edge positive everywhere).
            #
            # Other states (Recovery, Slow_Bull, Bull, Explosive_Bull, Crash_Bear):
            #   No additional gate — unvalidated in current archive. Monitor as regimes occur.
            if args.asset == "BTC" and _markov_7state is not None:
                _7st = _markov_7state
                _ct  = _active_trend                                           # composite_trend int
                _ema = confirm.ema_stack_bias                                  # -1 / 0 / +1
                _rv1 = _rvol_1h if (_rvol_1h == _rvol_1h) else 1.0           # relative vol

                if _7st == "Correction":
                    if dec_c.side == "yes":
                        _rescue_ct  = (_ct == 0 and _ema in (-1, 0))
                        _rescue_rv  = (0.7 <= _rv1 < 1.0)
                        if not (_rescue_ct or _rescue_rv):
                            print(f"  [markov_7state_gate] BLOCK YES {c['ticker']} — "
                                  f"Correction, c_trend={_ct}, ema={_ema:+d}, rvol={_rv1:.2f} "
                                  f"(YES edge=-9.4%; rescue needs c_trend=0+ema∈{{-1,0}} or rvol[0.7,1.0))")
                            _log_block("markov_7state_gate")
                            continue
                        else:
                            _why = ("c_trend=0+ema∈{-1,0}" if _rescue_ct else f"rvol={_rv1:.2f}∈[0.7,1.0)")
                            print(f"  [markov_7state_gate] RESCUE YES {c['ticker']} — "
                                  f"Correction but {_why} (edge=+6–15%)")
                    elif dec_c.side == "no":
                        _block_ct = (_ct == 0 and pm >= 0.70)
                        _block_rv = (0.7 <= _rv1 < 1.0 and pm >= 0.70)
                        if _block_ct or _block_rv:
                            _why = ("c_trend=0" if _block_ct else f"rvol={_rv1:.2f}∈[0.7,1.0)")
                            print(f"  [markov_7state_gate] BLOCK NO {c['ticker']} — "
                                  f"Correction+{_why}+pm={pm:.3f}>=0.70 (deep-ITM NO edge=-5 to -9%; perm p=0.000)")
                            _log_block("markov_7state_gate")
                            continue
                        elif pm < 0.70:
                            print(f"  [markov_7state_gate] RESCUE NO {c['ticker']} — "
                                  f"Correction but pm={pm:.3f}<0.70 (OTM/near-ITM NO, edge=+11-18%)")

                elif _7st == "Consolidation":
                    if dec_c.side == "yes":
                        _hard = (_ema == 1 or _ct == 2)
                        if _hard:
                            _why = (f"ema={_ema:+d}=+1" if _ema == 1 else f"c_trend={_ct}=2")
                            print(f"  [markov_7state_gate] BLOCK YES {c['ticker']} — "
                                  f"Consolidation+{_why} (YES edge=-10%; perm p=0.000)")
                            _log_block("markov_7state_gate")
                            continue
                        # else: SOFT — let normal edge gates handle
                    # NO: ALLOW (edge=+2.7% across all conditions)

            # [btc_highpm_no_gate] Block BTC NO when pm>=0.90 AND composite_rev>=0.
            # Analysis (2026-05-23, n=115 resolved): BTC NO bets at pm>0.70 → WR=20.0%, PnL=-$567.
            # Split by composite_rev:
            #   comp_rev <  0 (n=36): WR=33.3%, PnL=+$1,005 — no upward reversal signal → ALLOW
            #   comp_rev >= 0 (n=79): WR=14.0%, PnL=-$1,572 — reversal firing into near-ITM NO → BLOCK
            # Mechanism: pm>=0.90 = deep ITM YES; composite_rev>=0 = upward momentum active.
            # pm[0.70,0.90) RESCUED 2026-06-06: all-time analysis (n=4,445) shows WR=28% vs BEV=18%,
            # edge=+9.9%, +$4,405 blocked profit — gate was over-blocking near-ITM NO bets.
            # pm>=0.90 remains blocked: pm[0.90,0.95) WR=14% vs BEV=8% (small edge, marginal);
            # pm>=0.95 WR=3% vs BEV=2% (barely positive, not worth the tail risk).
            # Revert: paper_trade_runner_pre_gate_repair_20260606.py
            if (args.asset == "BTC"
                    and dec_c.side == "no"
                    and pm >= 0.90
                    and _active_rev >= 0):
                print(f"  [btc_highpm_no_gate] BLOCK NO {c['ticker']} — "
                      f"pm={pm:.3f}>=0.90, composite_rev={_active_rev:+d}>=0 "
                      f"(deep-ITM YES + reversal signal; pm[0.70,0.90) rescued 2026-06-06)")
                _log_block("btc_highpm_no_gate")
                continue

            # [btc_pup_direction_gate] Block when p_up_v2 is at extremes — strongly bullish
            # signal should block NO bets; strongly bearish should block YES bets.
            # Analysis (98 resolved BTC NO bets with p_up_v2 logged):
            #   p_up_v2>=0.70: N=2, WR=0%, PnL=-$100 — both lost, 0 wins
            #   p_up_v2>=0.65: N=3, WR=33%, net negative at extremes
            # Symmetric: p_up_v2<=0.35 applied to YES (mirror logic, thin data — monitor).
            # Causal: at extremes the LightGBM direction model strongly predicts movement
            # that fights the bet side. Not a drift multiplier — a hard directional filter.
            if args.asset == "BTC" and _p_up_v2 is not None:
                if dec_c.side == "no" and _p_up_v2 >= 0.65:
                    print(f"  [btc_pup_direction_gate] BLOCK NO {c['ticker']} — "
                          f"p_up_v2={_p_up_v2:.3f}>=0.65 (strongly bullish, NO fights trend)")
                    _log_block("btc_pup_direction_gate")
                    continue
                if dec_c.side == "yes" and _p_up_v2 <= 0.35:
                    print(f"  [btc_pup_direction_gate] BLOCK YES {c['ticker']} — "
                          f"p_up_v2={_p_up_v2:.3f}<=0.35 (strongly bearish, YES fights trend)")
                    _log_block("btc_pup_direction_gate")
                    continue

            # [rvol_gate] REMOVED 2026-05-28: fired on 97% of YES candidates (rvol mean=0.27),
            # effectively a permanent YES ban. blocked_trades: n=516, WR=46.3%, PnL=+$4,890
            # (would have profited). Original -$105 analysis (n=128, May-17) contradicted
            # by larger dataset. Deep OTM YES pm[0,0.3) alone: +$6,632 blocked profit.

            # [btc_garch_highvol_yes_gate] Block BTC YES when GARCH(1,1) cond vol ratio > 1.5.
            # ALL 424 HIGH GARCH trades share an identical Markov fingerprint: 1h=Bull inside
            # 4h/6h=Sideways inside 12h/daily=Bear — a bull-trap bounce in a sticky-vol bear market.
            # Persistence α+β=0.935 means the vol shock does not resolve within contract life.
            # Validated: 308 blocked (WR=25%), 231 losses saved, 77 wins forgone → +$44.94/$ net.
            # Rescue: pm≥0.80 AND tau<45 min — deep-ITM contract near expiry, GARCH vol can't
            # bridge the gap to strike; WR=100% on n=116 with +4.7pp above breakeven.
            if args.asset == "BTC" and dec_c.side == "yes":
                _garch_ratio = _get_garch_ratio(df_confirm, "BTC")
                if _garch_ratio is not None and _garch_ratio > 1.5:
                    _garch_rescue = (pm >= 0.80 and tau_c < 45)
                    if _garch_rescue:
                        print(f"  [btc_garch_highvol_yes_gate] RESCUE YES {c['ticker']} — "
                              f"ratio={_garch_ratio:.3f}>1.5 BUT pm={pm:.3f}>=0.80 + tau={tau_c:.0f}<45 (deep-ITM near-expiry)")
                    else:
                        print(f"  [btc_garch_highvol_yes_gate] BLOCK YES {c['ticker']} — "
                              f"ratio={_garch_ratio:.3f}>1.5, pm={pm:.3f}, tau={tau_c:.0f} (bull-trap vol regime)")
                        _log_block("btc_garch_highvol_yes_gate")
                        continue

            # [btc_adx_gate] Block BTC YES when ADX_1h is in moderate trending range [20,40).
            # mispricing_analysis [21a] 2026-05-17: adx_mod(20-40) YES: n=153, WR=36.6%,
            # BE=46.4%, Edge=-9.8%, PnL=-$150, p=0.013.
            # Breakdown by ema_stack: ema=0 → n=69, WR=36.2%, -$87; ema=+1 → n=74, WR=32.4%, -$82.
            # Rescue: ema_stack_bias=-1 (n=10, WR=70.0%, Edge=+18.7%, +$19) — bearish EMA stack
            # with YES bet = deep-ITM or reversal setup with high hit rate in moderate-trend conditions.
            if (args.asset == "BTC" and dec_c.side == "yes"
                    and confirm.adx_1h == confirm.adx_1h  # not NaN
                    and 20.0 <= confirm.adx_1h < 40.0):
                _adx_btc_rescue = (confirm.ema_stack_bias == -1)
                if not _adx_btc_rescue:
                    print(f"  [btc_adx_gate] BLOCK YES {c['ticker']} — "
                          f"adx_1h={confirm.adx_1h:.1f} in [20,40), ema_stack={confirm.ema_stack_bias} (not bearish stack rescue)")
                    _log_block("btc_adx_gate")
                    continue
                else:
                    print(f"  [btc_adx_gate] RESCUE YES {c['ticker']} — "
                          f"adx_1h={confirm.adx_1h:.1f} in [20,40), ema_stack={confirm.ema_stack_bias}=-1 (bearish stack rescue)")

            # [btc_deepno_neutral_gate] Block BTC YES when pm < 0.35 AND ema_stack_bias=0.
            # mispricing_analysis [16a] 2026-05-17: pm=deep_NO + ema=neutral YES:
            # n=48, WR=10.4%, BE=21.3%, Edge=-10.9%, PnL=-$52, p=0.018.
            # Full verified: pm<0.35 + stack=0 → n=63, PnL=-$52. Non-ADX subset (no adx_1h
            # data) → n=54, PnL=-$68. ADX-mod subset already caught by btc_adx_gate.
            # Mechanism: low-probability YES (market disagrees strongly) + no EMA direction
            # = chasing nothing. Passes bullish stack (+$29) and bearish stack (+$2) through.
            if (args.asset == "BTC" and dec_c.side == "yes"
                    and pm < 0.35
                    and confirm.ema_stack_bias == 0):
                print(f"  [btc_deepno_neutral_gate] BLOCK YES {c['ticker']} — "
                      f"pm={pm:.3f}<0.35, ema_stack=0 (no directional signal for low-pm bet)")
                _log_block("btc_deepno_neutral_gate")
                continue

            # [near_atm_ema_gate] Block BTC YES when pm [0.50, 0.60) AND ema_stack ∈ {0, +1}.
            # Resim 2026-05-17 (n=977 resolved YES):
            #   ema=0  → n=64, WR=43.8%, BE=54.4%, edge=-10.7%, PnL=-$631
            #   ema=+1 → n=65, WR=50.8%, BE=54.6%, edge=-3.8%,  PnL=-$220
            #   ema=-1 → n=38, WR=71.1%, BE=55.2%, edge=+15.8%, PnL=+$481  ← rescue
            # Mechanism: near-ATM YES needs price to keep moving up to win; with neutral or
            # bullish EMA structure, that move is already priced in — no edge remains.
            # ema=-1 is a mean-reversion setup (BTC below EMAs, bounce candidate) — rescue.
            # Revert: remove this block.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and 0.50 <= pm < 0.60
                    and confirm.ema_stack_bias in (0, 1)):
                print(f"  [near_atm_ema_gate] BLOCK YES {c['ticker']} — "
                      f"pm={pm:.3f} in [0.50,0.60), ema_stack={confirm.ema_stack_bias:+d} "
                      f"(no reversal setup, near-ATM edge spent)")
                _log_block("near_atm_ema_gate")
                continue

            # [strong_trend_nearatm_gate] REMOVED 2026-06-04.
            # Permutation + walk-forward validation: OOS blocked WR=62.7% > bkev=57.2% —
            # gate was blocking winning YES bets. Only 1 live fire since May-18.
            # Shadow observe refined sub-condition: stoch_1h[40,60) + ema≠-1
            # Scan archive: WR=0% but only 3 expiry events (n=12) — too thin to gate yet.
            # Grep logs for "[shadow/st_nearatm]" to accumulate 20+ expiry events.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and 0.55 <= pm < 0.60
                    and _active_trend >= 3
                    and confirm.ema_stack_bias != -1):
                _c1h  = df_confirm["close"].astype(float)
                _ll1h = df_confirm["low"].astype(float).rolling(14).min()
                _hh1h = df_confirm["high"].astype(float).rolling(14).max()
                _sk_1h = float(((_c1h - _ll1h) / (_hh1h - _ll1h).replace(0, float("nan")) * 100).iloc[-2])
                if 40 <= _sk_1h < 60:
                    print(f"  [shadow/st_nearatm] YES {c['ticker']} — pm={pm:.3f}, "
                          f"c_trend={_active_trend}, stoch_1h={_sk_1h:.1f}, ema={confirm.ema_stack_bias:+d}")

            # [btc_stoch_no_gate] Block BTC NO when stoch_k < 20 (oversold, bounce risk).
            # mispricing_analysis [Section 9] 2026-05-17: stoch_k<20 NO: n=49, WR=49.0%,
            # BE=66.3%, Edge=-17.3%, PnL=-$85, p=0.021.
            # Mechanism: stoch<20 = price has fallen hard; oversold condition creates mean-
            # reversion bounce risk that sends BTC above the NO strike.
            # By pm/offset: pm[0.30,0.40) → WR=37.5%, -$47 (worst); offset>0 (ITM NO) → -$84.
            # Rescue 1 (2026-05-17): ct<=-3 AND fund=-1 — strong bearish trend + crowded shorts
            # confirms downtrend continuation even from oversold; WR=88.5% vs 83% implied (n=3222).
            # Rescue 2 (2026-05-24): vwap_stretch_score==1 — price above VWAP despite stoch dip;
            # temporary oversold in elevated VWAP context, NO strike stays out of reach; n=11, WR=81.8%.
            if (args.asset == "BTC" and dec_c.side == "no"
                    and confirm.stoch_k == confirm.stoch_k  # not NaN
                    and confirm.stoch_k < 20.0):
                _stoch_no_rescue = (
                    (_active_trend <= -3 and confirm.funding_bias == -1)
                    or getattr(confirm, "stretch_score", None) == 1
                )
                if _stoch_no_rescue:
                    _rsrc_stoch = ("ct<=-3+fund=-1" if (_active_trend <= -3 and confirm.funding_bias == -1)
                                   else "vwap_stretch=1")
                    print(f"  [btc_stoch_no_gate] RESCUED NO {c['ticker']} — "
                          f"{_rsrc_stoch} overrides stoch<20")
                else:
                    print(f"  [btc_stoch_no_gate] BLOCK NO {c['ticker']} — "
                          f"stoch_k={confirm.stoch_k:.1f}<20 (oversold, bounce risk for NO)")
                    _log_block("btc_stoch_no_gate")
                    continue

            # [itm_no_neutral_stoch_gate] Block BTC ITM NO when stoch_k in [40,60] (neutral zone).
            # Archive analysis (2026-06-05, n=122,963 ITM NO resolved, offset<0, pm>0.50):
            #   stoch_k [40,60]: n=23,763, WR=7.3%, PnL=-$10,345, MCPT p=0.0000, z=+9.92
            #   Rescue (ema_stack<=-1 inside block): n=5,620, WR=13.4%, mean=+$4.57/bet
            #   Blocked pool w/ rescue applied: PnL=-$36,055 saved (3.5× better than no rescue)
            # Causal: neutral stoch [40,60] = no directional conviction; ITM NO needs price to
            # fall further below strike but neutral momentum argues for mean-reversion upward.
            # ema_stack<=-1 = bearish EMA alignment adds cross-timeframe directional confirmation
            # that overrides the stoch neutrality → rescue.
            # Confirmed by last 24h: 4 of 9 ITM NO losses had stoch [40,60] (+$156 saved).
            if (args.asset == "BTC"
                    and dec_c.side == "no"
                    and not _vsa_no_flip
                    and offset_c < 0
                    and pm > 0.50
                    and confirm.stoch_k == confirm.stoch_k  # not NaN
                    and 40.0 <= confirm.stoch_k <= 60.0):
                _itm_neutral_rescue = (confirm.ema_stack_bias <= -1)
                if _itm_neutral_rescue:
                    print(f"  [itm_no_neutral_stoch_gate] RESCUE NO {c['ticker']} — "
                          f"stoch={confirm.stoch_k:.1f} in [40,60] but ema_stack={confirm.ema_stack_bias:+d}<=-1 (bearish alignment)")
                else:
                    print(f"  [itm_no_neutral_stoch_gate] BLOCK NO {c['ticker']} — "
                          f"stoch={confirm.stoch_k:.1f} in [40,60], ema_stack={confirm.ema_stack_bias:+d}>-1 "
                          f"(neutral stoch + no bearish EMA alignment, WR=7.3% historically)")
                    _log_block("itm_no_neutral_stoch_gate")
                    continue

            # [EXPERIMENTAL — 2026-04-25] BTC YES vol_score=1 gate with rescue.
            # Block YES bets when vol_score=1 (last completed 1h bar: high volume + price up).
            # Mechanism: high-vol up bar = move already happened; YES bet is chasing into a
            # likely fade. All-time: 33 trades at 30.3% WR (-$644) vs 61.6% WR otherwise.
            #
            # Rescue (allow through) when:
            #   ema_stack_bias == 1 AND (confirmation_score == 0 OR funding_bias == 0)
            # Logic: bullish EMA structure (trend intact) + either no directional noise
            # (conf=0 = pure ITM price-proximity bet) OR clean funding (no crowded longs).
            # In-sample (calibration period): blocked 29 trades (WR=20.7%, -$733),
            # rescued 4 (WR=100%, +$89). Net vs baseline: +$733.
            #
            # [DISABLED 2026-04-30] gate_attribution_v3 LOO replay (logged p_yes_model,
            # flat $1k bankroll, BTC rescues active) showed gate now costs −$197 in
            # archive PnL. 10 trades blocked-after-rescue: 4W/6L (40%), net +$197.
            # The cell the gate carves out has shifted from −$733 (calibration window)
            # to +$197 (recent), suggesting regime change OR original calibration noise.
            # Disabling for live observation. Revert: uncomment the block below.
            #
            # if (args.asset == "BTC" and dec_c.side == "yes"
            #         and _vol_score_dir == 1):
            #     _ema_bullish  = (confirm.ema_stack_bias == 1)
            #     _conf_zero    = (_cscore == 0)
            #     _fund_neutral = (confirm.funding_bias == 0)
            #     _vol_rescue   = _ema_bullish and (_conf_zero or _fund_neutral)
            #     if not _vol_rescue:
            #         print(f"  [btc_vol1_gate] BLOCK YES vol=1 "
            #               f"ema_stack={confirm.ema_stack_bias} "
            #               f"conf={_cscore} fund={confirm.funding_bias}")
            #         continue
            #     else:
            #         _vol_rescue_reason = []
            #         if _conf_zero:
            #             _vol_rescue_reason.append("conf=0")
            #         if _fund_neutral:
            #             _vol_rescue_reason.append("fund=0")
            #         print(f"  [btc_vol1_gate] RESCUE YES vol=1 "
            #               f"via ema=1+{'+'.join(_vol_rescue_reason)} "
            #               f"ema_stack={confirm.ema_stack_bias} "
            #               f"conf={_cscore} fund={confirm.funding_bias}")

            # btc_otm_gate (pm<0.20 YES block) removed: vol_factor gate (|z|>1.0×vf)
            # naturally blocks unreachable deep OTM strikes without a hard pm cutoff.

            # [2026-04-27] ETH YES OTM hard block: p_market < 0.45 when strike > spot.
            # 32 historical OTM YES trades at pm<0.45 had 6.2% WR (-$1,203 net) across all
            # p_up levels (0.55–0.70+). High composite_p_up does not rescue these — model is
            # directionally correct but strikes 10–52% above spot are unreachable in ~50m.
            # Conditioned on offset_pct > 0 (OTM) to protect future ITM edge cases where a
            # sharp drop pushes pm below 0.45 on a technically in-the-money contract.
            # Simulation: +$1,203 (32 losses blocked, 0 winners). Revert: remove this block.
            if args.asset == "ETH" and dec_c.side == "yes" and pm < 0.40 and dec_c.net_edge <= 0.25:
                print(f"  [eth_otm_gate] BLOCK YES p_market={pm:.3f}<0.40, net_edge={dec_c.net_edge:.3f}<=0.25")
                _log_block("eth_otm_gate")
                continue

            # [G3 — ETH OTM YES pm 0.45-0.50 ema filter]
            # Analysis: pm 0.45-0.50 YES splits cleanly by ema_alignment.
            # ema=bull: n=7, 85.7% WR, +$32.78/t.  ema=notbull: n=8, 50% WR, -$4.24/t.
            # Block non-bullish alignment in this marginal zone; rescue when trend confirmed bullish.
            if args.asset == "ETH" and dec_c.side == "yes" and offset_c > 0 and 0.45 <= pm < 0.50:
                _g3_ema_bull = "bull" in _ema_align.lower()
                if not _g3_ema_bull:
                    print(f"  [eth_otm_gate] BLOCK OTM YES pm={pm:.3f} in [0.45,0.50) ema={_ema_align} — rescue requires bullish alignment")
                    _log_block("eth_otm_gate_ema")
                    continue
                else:
                    print(f"  [eth_otm_gate] RESCUE OTM YES pm={pm:.3f} in [0.45,0.50) ema={_ema_align}")

            # [G4 — sol_yes_gate REMOVED 2026-05-17]
            # blocked_trades.csv retrospective (n=283): WR=48.4%, BE=37.7%, Edge=+10.8%, $304 profit blocked.
            # Original n=12 analysis (WR=16.7%) no longer valid with full data. Gate was over-blocking.

            # [sol_yes_deepotm_gate] Hard block SOL YES when offset_c > 0.001 (>0.10% OTM).
            # Analysis (2026-05-23, n=18): WR=5.6%, BE≈20%, P&L≈-$450.
            # Existing struct=1 rescue only protects 4 of 18 — 14 unprotected trades structurally unwinnable.
            # Strike >0.10% above spot requires a large hourly move SOL rarely makes in range.
            if (args.asset == "SOL"
                    and dec_c.side == "yes"
                    and offset_c > 0.001):
                print(f"  [sol_yes_deepotm_gate] BLOCK YES {c['ticker']} — "
                      f"offset={offset_c*100:+.3f}%>0.10 "
                      f"(OTM SOL YES, n=18 WR=5.6%, no viable rescue)")
                _log_block("sol_yes_deepotm_gate")
                continue

            # [sol_yes_structure_pos_gate] Hard block SOL YES when structure_bias=1 AND (no_score=2 OR composite_rev=0).
            # Analysis (2026-05-23, n=24): WR=25.0%, BE≈55.0%, P&L=-$313.
            # no_score=2 alone: n=11, WR=18.2%. composite_rev=0 alone: n=15, WR=26.7%.
            # Baseline (neither condition): n=187, WR=77.5%, +$1,128 — stark contrast validates gate.
            # No rescue found — max WR=25% at n=12 across all tested features.
            if (args.asset == "SOL"
                    and dec_c.side == "yes"
                    and struct.structure_bias == 1
                    and (confirm.no_score == 2 or _active_rev == 0)):
                print(f"  [sol_yes_structure_pos_gate] BLOCK YES {c['ticker']} — "
                      f"struct=1, no_score={confirm.no_score}, composite_rev={_active_rev} "
                      f"(bullish structure but conflicting NO signal or no rev, WR=25% vs BE=55%)")
                _log_block("sol_yes_structure_pos_gate")
                continue

            # [sol_no_vwap_neutral_gate] Hard block SOL NO when vwap_stretch=0 AND (ema_stretch=1 OR stoch_k<40).
            # Analysis (2026-05-23, n=38): WR=52.6%, BE≈75.8%, P&L=-$374.
            # Baseline NO (not blocked): n=193, WR=85.0%, +$1,004 — confirms gate carves out losers.
            # ema_stretch=1 = EMA bullishly extended (trend against NO); stoch<40 = oversold (bounce imminent).
            # No rescue — best candidate (no_score>=1) only reached 60.9% vs 75.8% breakeven.
            _vwap_str_sol = getattr(confirm, "stretch_score", None)
            _ema_str_sol  = getattr(confirm, "ema_stretch_score", None)
            _stoch_k_sol  = (float(confirm.stoch_k)
                             if confirm.stoch_k == confirm.stoch_k
                             else 50.0)
            if (args.asset == "SOL"
                    and dec_c.side == "no"
                    and _vwap_str_sol == 0
                    and (_ema_str_sol == 1 or _stoch_k_sol < 40.0)):
                print(f"  [sol_no_vwap_neutral_gate] BLOCK NO {c['ticker']} — "
                      f"vwap_stretch=0, ema_stretch={_ema_str_sol}, stoch_k={_stoch_k_sol:.1f} "
                      f"(neutral VWAP+extended EMA or oversold, WR=52.6% vs BE=75.8%, no rescue)")
                _log_block("sol_no_vwap_neutral_gate")
                continue

            # [sol_no_structure_neg_gate] Hard block SOL NO when structure_bias=-1 AND pm>=0.30.
            # Analysis (2026-05-23, n=17): WR=41.2%, BE≈64.0%, P&L=-$227.
            # structure_bias=-1 AND pm<0.30 (excluded): n=37, WR=86.5% — correctly not blocked.
            # Baseline NO not blocked: n=214, WR=82.7%, +$856.
            # composite_rev>=3 shows WR=83% but n=6 — re-evaluate rescue after 30+ blocked trades.
            if (args.asset == "SOL"
                    and dec_c.side == "no"
                    and struct.structure_bias == -1
                    and pm >= 0.30):
                print(f"  [sol_no_structure_neg_gate] BLOCK NO {c['ticker']} — "
                      f"struct=-1, pm={pm:.3f}>=0.30 "
                      f"(bearish structure at fair price, WR=41.2% vs BE=64%, no rescue)")
                _log_block("sol_no_structure_neg_gate")
                continue

            # [G5 — SOL NO pm 0.30-0.50 mid-range gate]
            # Zone sits barely below breakeven: n=50, 62% WR, bkeven=64%, -$2.53/t (-$126 total).
            # structure_bias=1 (bullish structure) rescue: n=23, 78.3% WR, +$12.36/t.
            # ema_stack=-1 rescue: n=11, 81.8% WR, +$13.24/t.
            # Combined (struct=1 OR ema=-1): n=28, 82.1% WR, +$13.81/t.
            # Block (neither): n=22, 36.4% WR, -$23.32/t. Net save: +$513.
            if args.asset == "SOL" and dec_c.side == "no" and 0.30 <= pm < 0.50:
                _g5_struct_bull = (struct.structure_bias == 1)
                _g5_ema_bear    = (confirm.ema_stack_bias == -1)
                if not _g5_struct_bull and not _g5_ema_bear:
                    print(f"  [sol_no_gate] BLOCK NO pm={pm:.3f} in [0.30,0.50) struct={struct.structure_bias} ema_stack={confirm.ema_stack_bias} — rescue requires struct=1 OR ema_stack=-1")
                    _log_block("sol_no_gate")
                    continue
                else:
                    _g5_reasons = []
                    if _g5_struct_bull: _g5_reasons.append(f"struct={struct.structure_bias}")
                    if _g5_ema_bear:    _g5_reasons.append(f"ema_stack={confirm.ema_stack_bias}")
                    print(f"  [sol_no_gate] RESCUE NO pm={pm:.3f} via {'+'.join(_g5_reasons)}")

            # [sol_no_struct_ema_gate REMOVED 2026-05-17]
            # blocked_trades.csv retrospective (n=128): WR=100%, BE=83.4%, Edge=+16.6%, $213 profit blocked.
            # Gate was completely wrong — all blocked NO trades won. Removed.

            # [sol_no_trend_struct_gate REMOVED 2026-05-17]
            # blocked_trades.csv retrospective (n=31): WR=87.1%, BE=79.5%, Edge=+7.6%, $24 profit blocked.
            # Gate was over-blocking profitable SOL NO trades. Removed.

            # [SOL vol gate] Block OTM YES/NO when strike is unreachable given vol regime.
            # Same BASE_Z=2.0 and _vol_factor as BTC/ETH gates; SOL _VOL_CONFIGS has SOL-specific thresholds.
            if args.asset == "SOL" and dec_c.decision == "trade" and sigma_tau_c > 0:
                _z_sol = abs(math.log(s_k / spot) / sigma_tau_c)
                _sol_vol_z = 2.0 * _vol_factor
                _sol_otm = (dec_c.side == "yes" and offset_c > 0) or (dec_c.side == "no" and offset_c < 0)
                if _sol_otm and _z_sol > _sol_vol_z:
                    print(f"  [sol_vol_gate] BLOCK {dec_c.side.upper()} {c['ticker']} — |z|={_z_sol:.3f} > {_sol_vol_z:.3f} (vol_factor={_vol_factor:.3f})")
                    _log_block("sol_vol_gate")
                    continue

            # [sol_bp_body_gate] Block SOL bets when 5m microstructure contradicts direction.
            # bp_5m < 0.35 = sellers dominated last 5m bar; body_15m > 0.40 = conviction move.
            # Together: strong short-term directional pressure against the bet.
            # Rescue YES: ema_stack_bias == +1 (trend is bullish, mean-reversion bounce plausible).
            # Rescue NO:  ema_stack_bias == -1 (trend is bearish, confirms NO direction).
            # Symmetric: mirror applied to NO (high bp + bullish body fighting a NO bet).
            if (args.asset == "SOL" and dec_c.decision == "trade"
                    and _bp_5m is not None and _body_15m is not None):
                if dec_c.side == "yes" and _bp_5m < 0.35 and _body_15m > 0.40:
                    if confirm.ema_stack_bias == 1:
                        print(f"  [sol_bp_body_gate] RESCUED YES {c['ticker']} — "
                              f"bp={_bp_5m:.3f}<0.35 body={_body_15m:.3f}>0.40 "
                              f"but ema_stack=+1 (trend bullish, bounce plausible)")
                    else:
                        print(f"  [sol_bp_body_gate] BLOCK YES {c['ticker']} — "
                              f"bp={_bp_5m:.3f}<0.35 body={_body_15m:.3f}>0.40 "
                              f"(bearish microstructure, ema_stack={confirm.ema_stack_bias})")
                        _log_block("sol_bp_body_gate")
                        continue
                elif dec_c.side == "no" and _bp_5m > 0.65 and _body_15m > 0.40:
                    if confirm.ema_stack_bias == -1:
                        print(f"  [sol_bp_body_gate] RESCUED NO {c['ticker']} — "
                              f"bp={_bp_5m:.3f}>0.65 body={_body_15m:.3f}>0.40 "
                              f"but ema_stack=-1 (trend bearish, confirms NO)")
                    else:
                        print(f"  [sol_bp_body_gate] BLOCK NO {c['ticker']} — "
                              f"bp={_bp_5m:.3f}>0.65 body={_body_15m:.3f}>0.40 "
                              f"(bullish microstructure vs NO bet, ema_stack={confirm.ema_stack_bias})")
                        _log_block("sol_bp_body_gate")
                        continue

            # [eth_bp_body_gate] Block ETH YES when 5m buying pressure is high but 15m conviction
            # is in the ambiguous "medium body" zone.
            # Pattern: bp>0.60 (buyers dominated last 5m bar) + body in [0.25,0.55) (15m candle
            # is neither a doji nor a committed marubozu — medium conviction only).
            # This is a local exhaustion signature: the 5m surge was real but unsupported by the
            # 15m timeframe, meaning price is likely topping out.
            # Backtest: n=37, WR=35.1%, BE=50.2%, P&L=-$695. No rescue condition improves it.
            if (args.asset == "ETH" and dec_c.decision == "trade"
                    and dec_c.side == "yes"
                    and _bp_5m is not None and _body_15m is not None
                    and _bp_5m > 0.60 and 0.25 <= _body_15m < 0.55):
                print(f"  [eth_bp_body_gate] BLOCK YES {c['ticker']} — "
                      f"bp={_bp_5m:.3f}>0.60 body={_body_15m:.3f} in [0.25,0.55) "
                      f"(local exhaustion: 5m spike without 15m commitment)")
                _log_block("eth_bp_body_gate")
                continue

            # btc_no_pup_gate and btc_no_edge_gate removed: replaced by z_abs_no_min gate below.

            # [BTC NO z_abs gate] Block NO bets where the strike is < 0.30σ from spot.
            # History: 0.30 → 0.60 (dual-model reform), 0.60 → 0.45 (2026-05-07 sim),
            # 0.45 → 0.30 (2026-05-16 blocked_trades sim, n=1134):
            # z[0.30,0.45) bucket: n=249, WR=53.7% vs BE=46.6%, +9.4pp → profitable, gate over-blocked.
            # z<0.30 bucket: n=615, WR=46.2% vs BE=47.6%, -2.0pp → still correctly blocked.
            if (args.asset == "BTC" and dec_c.side == "no"
                    and dec_c.decision == "trade" and _composite_computed and sigma_tau_c > 0):
                _z_no = abs(math.log(s_k / spot) / sigma_tau_c)
                if _z_no < 0.30:
                    # Rescue: ct<=-3+fund=-1 provides structural edge the z_score alone misses.
                    # full_opportunity_analysis: ct<=-3+fund=-1 NO WR=88.5% vs 83.0% implied, n=3222.
                    _zngate_rescue = (_active_trend <= -3 and confirm.funding_bias == -1)
                    if _zngate_rescue:
                        print(f"  [btc_no_z_gate] RESCUED NO {c['ticker']} — "
                              f"ct={_active_trend}<=-3+fund=-1 (bearish trend+funding > z={_z_no:.3f} signal)")
                    else:
                        print(f"  [btc_no_z_gate] BLOCK NO {c['ticker']} — |z|={_z_no:.3f} < 0.30 (near-ATM, no structural edge)")
                        _log_block("btc_no_z_gate")
                        continue
                # [BTC NO OTM vol gate] Mirror of btc_vol_gate for the downside.
                # Block OTM NO (offset < 0) when required drop exceeds vol regime reach.
                if offset_c < 0 and _z_no > 2.0 * _vol_factor:
                    print(f"  [btc_no_vol_gate] BLOCK NO {c['ticker']} — |z|={_z_no:.3f} > {2.0*_vol_factor:.3f} (OTM NO, vol_factor={_vol_factor:.3f})")
                    _log_block("btc_no_vol_gate")
                    continue

                # [2026-05-08] BTC NO wrong-direction gate: bullish EMA stack + price extended
                # above VWAP when pm≥0.65 — market already prices YES at 65%+, trending up
                # into that is a clear counter-signal for NO.
                # Sim (n=2, pm≥0.65): WR=0% vs BE=25%, PnL=-$56.81 → net +$56.81 blocked, 0 wins lost.
                # pm<0.65: same condition is net profitable — do NOT gate there.
                if pm >= 0.65 and confirm.ema_stack_bias == 1 and confirm.stretch_score <= -2:
                    print(f"  [btc_no_wrongdir_gate] BLOCK NO {c['ticker']} — pm={pm:.3f}≥0.65, "
                          f"ema_stack=1 (bullish) + vwap_stretch={confirm.stretch_score} (price extended above VWAP)")
                    _log_block("btc_no_wrongdir_gate")
                    continue

            # [2026-05-08] BTC NO SMC demand zone gate: block when bearish 1h SMC AND demand
            # zone is close below (<1.2% from spot) — structural support likely prevents further drop.
            # Rescue: allow if supply zone is close above (<1.0%) — price compressed between zones.
            # Backtest: demand_pct<1.2%, n=11, WR=45.5%, BE=62.4%, PnL=-$188.05.
            # Rescue supply_pct<1.0%: n=8, WR=75.0%, BE=56%, PnL=+$118.40.
            if (args.asset == "BTC" and dec_c.side == "no"
                    and dec_c.decision == "trade"
                    and _smc is not None and _smc.bos_1h == "bearish"
                    and _smc.nearest_demand_pct is not None
                    and _smc.nearest_demand_pct < 1.2):
                _smc_no_rescue = (
                    _smc.nearest_supply_pct is not None and _smc.nearest_supply_pct < 1.0
                )
                if _smc_no_rescue:
                    print(f"  [btc_no_smc_demand_gate] RESCUED NO {c['ticker']} — "
                          f"demand_pct={_smc.nearest_demand_pct:.2f}%<1.2% but supply_pct={_smc.nearest_supply_pct:.2f}%<1.0% (compressed)")
                else:
                    print(f"  [btc_no_smc_demand_gate] BLOCK NO {c['ticker']} — "
                          f"bearish SMC 1h + demand_pct={_smc.nearest_demand_pct:.2f}%<1.2% (support too close below, no supply rescue)")
                    _log_block("btc_no_smc_demand_gate")
                    continue

            # [2026-04-28] BTC spread tightness gate with rescue.
            # Sim: trades with spread >= 0.04 → WR deteriorates, P&L=-$242 (49 trades).
            # Rescue: chg_10m direction-aligned AND net_edge >= 0.07 → 7W 2L (77.8%, +$64).
            # Revert: remove this block.
            if args.asset == "BTC" and dec_c.decision == "trade" and spread_c >= 0.04:
                _chg10m_aligned = (
                    (dec_c.side == "yes" and _sharp_move_pct_10m > 0) or
                    (dec_c.side == "no"  and _sharp_move_pct_10m < 0)
                )
                _spread_rescue = _chg10m_aligned and dec_c.net_edge >= 0.07
                if not _spread_rescue:
                    print(f"  [btc_spread_gate] BLOCK {dec_c.side.upper()} spread={spread_c:.3f}>=0.04 "
                          f"chg_10m={_sharp_move_pct_10m*100:+.2f}% net_edge={dec_c.net_edge:.4f}")
                    _log_block("btc_spread_gate")
                    continue
                else:
                    print(f"  [btc_spread_gate] RESCUED {dec_c.side.upper()} spread={spread_c:.3f} "
                          f"chg_10m={_sharp_move_pct_10m*100:+.2f}% aligned, net_edge={dec_c.net_edge:.4f}>=0.07")

            # Tau confidence decay — all 1h assets, kicks in below 6 minutes remaining.
            # Mirrors 15m runner's (tau/5)^2 pattern. At tau<6min the 1h signals are
            # largely stale; only high-conviction trades clear the Gate 3 floor once
            # the quadratic confidence factor is applied.
            # Effect: edge×(tau/6)² must still exceed MIN_NET_EDGE (1%).
            #   tau=5: conf=0.694 → need raw edge ≥1.44%
            #   tau=4: conf=0.444 → need raw edge ≥2.25%
            #   tau=3: conf=0.250 → need raw edge ≥4.0%
            #   tau=2: conf=0.111 → need raw edge ≥9.0%
            #   tau=1: conf=0.028 → need raw edge ≥36%
            if dec_c.decision == "trade" and tau_c < 6.0:
                _tau_conf     = (tau_c / 6.0) ** 2
                _tau_adj_edge = dec_c.net_edge * _tau_conf
                if _tau_adj_edge < MIN_NET_EDGE:
                    print(f"  [tau_decay] BLOCK {dec_c.side.upper()} {c['ticker']} — "
                          f"tau={tau_c:.1f}min  raw_edge={dec_c.net_edge:.3f}×"
                          f"conf={_tau_conf:.3f}={_tau_adj_edge:.4f}<{MIN_NET_EDGE:.3f}")
                    _log_block("tau_decay")
                    continue
                else:
                    print(f"  [tau_decay] PASS {dec_c.side.upper()} {c['ticker']} — "
                          f"tau={tau_c:.1f}min  adj_edge={_tau_adj_edge:.4f} (conf={_tau_conf:.3f})")

            # [2026-05-05] BTC tau gate — YES OTM hard block + NO conviction filter.
            # YES block: p_market<0.40 AND tau<30 — no rescue.
            #   Backtest n=46, WR=8.7% (BE=22.9%), PnL=-$1,217. Holds in both Apr (0% WR, n=24)
            #   and May (18.2% WR, n=22) halves. Old p_up>=0.52 rescue invalidated — all
            #   post-Apr-28 slippage had p_up>=0.52 but still lost. Physical constraint: OTM
            #   with <30min leaves no time for price to reach strike.
            # NO filter: block when lacking directional conviction (p_up>0.48) unless high Kelly+tight spread.
            # Revert: restore the [2026-04-28] block above.
            if args.asset == "BTC" and dec_c.decision == "trade" and tau_c < 30:
                # Near-ATM at very low tau: medium-term signals don't predict 5-min moves.
                # offset > -0.10% means BTC is within $80 of the strike — basically a coin flip.
                # Backtest: 0/2 WR at tau<7 + near-ATM, saves $72. Deep ITM (offset<=-0.10%) passes.
                if dec_c.side == "yes" and tau_c < 7 and offset_c > -0.001:
                    print(f"  [btc_tau_gate] BLOCK YES tau={tau_c:.1f}min<7 "
                          f"near-ATM offset={offset_c*100:+.3f}%>-0.10% (5-min noise dominates)")
                    _log_block("btc_tau_gate")
                    continue
                if dec_c.side == "yes" and pm < 0.40:
                    print(f"  [btc_tau_gate] BLOCK YES tau={tau_c:.1f}min<30 "
                          f"p_market={pm:.3f}<0.40 (OTM+low-tau, no rescue)")
                    _log_block("btc_tau_gate")
                    continue
                elif dec_c.side == "no":
                    _tau_conviction = _comp_p_up <= 0.48
                    _tau_rescue     = dec_c.kelly_fraction >= 0.15 and spread_c <= 0.02
                    if not _tau_conviction and not _tau_rescue:
                        print(f"  [btc_tau_gate] BLOCK NO tau={tau_c:.1f}min<30 "
                              f"p_up={_comp_p_up:.3f} kelly={dec_c.kelly_fraction:.3f} spread={spread_c:.3f}")
                        _log_block("btc_tau_gate")
                        continue
                    elif _tau_conviction:
                        print(f"  [btc_tau_gate] PASS NO tau={tau_c:.1f}min<30 "
                              f"p_up={_comp_p_up:.3f} (directional conviction)")
                    else:
                        print(f"  [btc_tau_gate] RESCUED NO tau={tau_c:.1f}min<30 "
                              f"p_up={_comp_p_up:.3f} kelly={dec_c.kelly_fraction:.3f}>=0.15 spread={spread_c:.3f}<=0.02")

            positions = already_traded_expiries.get(expiry_key, {"yes": [], "no": []})
            if dec_c.side == "yes" and any(no_k < s_k for no_k in positions["no"]):
                print(f"  [scan] Skipping {c['ticker']} — YES@{s_k} conflicts with existing NO below it")
                continue
            if dec_c.side == "no" and any(yes_k > s_k for yes_k in positions["yes"]):
                print(f"  [scan] Skipping {c['ticker']} — NO@{s_k} conflicts with existing YES above it")
                continue

            # For BTC NO trades using dual model, p_yes_model logs the YES probability
            # (p_model_comp) for consistency. The independent NO probability (_p_no_btc)
            # is logged separately via p_no_model_comp for future analysis.
            # Track p_market history for 5-minute drift (one reading per scan = 1 minute)
            _tk = c["ticker"]
            if _tk not in _pm_history:
                _pm_history[_tk] = deque(maxlen=6)
            _pm_history[_tk].append(pm)
            _pm_drift_c = (pm - list(_pm_history[_tk])[0]) if len(_pm_history[_tk]) >= 6 else float("nan")

            meta_c    = {"strike": s_k, "p_market": pm, "prob": prob_c,
                         "contract_ticker": c["ticker"], "close_ts": c["close_time"],
                         "vol_eff": vol_eff_c, "bid": c["bid"], "ask": c["ask"],
                         "p_model_comp": p_model_comp,
                         "p_no_model_comp": _p_no_btc if (args.asset == "BTC" and _composite_computed) else None,
                         "pm_drift_5m": _pm_drift_c,
                         "p_gbdt": _p_gbdt_c,
                         "p_up_v2": _comp_p_up_c if _composite_computed else None}

            if best_any_dec is None or dec_c.net_edge > best_any_dec.net_edge:
                best_any_dec  = dec_c
                best_any_meta = meta_c

            if dec_c.decision == "no_trade":
                if best_no_trade_dec is None or dec_c.net_edge > best_no_trade_dec.net_edge:
                    best_no_trade_dec  = dec_c
                    best_no_trade_meta = meta_c

            # 30m streak gate: only blocks contracts that would trade
            # Gate 1 (YES bearish streak rescue): BTC + ETH only — SOL wins at 80%+ WR when stoch<70
            # Gate 2 (NO bullish streak stoch 30-60): BTC + ETH only — SOL 94.4% WR when blocked
            #   BTC rescue: chg_5m < 0 (5m already reversing) → 83.3% WR, +$122
            #   ETH rescue: stoch_k >= 45 (upper band) → 83.3% WR, +$75
            # Gate 3 (NO stoch<20 block): BTC only — ETH/SOL both win 79-83% WR in this bucket
            #   Rescue (2026-05-16): composite_rev >= 2 — oversold model signal overrides the gate.
            #   blocked_trades sim (n=16, all stoch<30 + rev>=2): WR=87.5% BE=69.7% +17.8pp +$28.
            if dec_c.decision == "trade" and _streak30 is not None:
                _sk = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _gate = False
                _gate_reason = ""
                if dec_c.side == "yes" and _streak30 == "bearish" and _sk <= 70 and args.asset != "SOL":
                    if struct.structure_bias == 1:
                        # Rescue: bearish streak into bullish structure = counter-trend pullback.
                        # Data: n=60, WR=0.883 vs BE=0.542. Streak is corrective, not continuation.
                        print(f"  [streak_gate] RESCUED YES {c['ticker']} — streak30=bearish but structure_bias=1 (bullish structure, pullback into demand)")
                    elif confirm.vpin_score == -1 and _sharp_move_pct_10m <= 0:
                        # Rescue: vpin=-1 during bearish streak with no bounce = informed-seller compression,
                        # not continuation. blocked_trades sweep 2026-05-24: n=18 WR=72.2% +$220.
                        print(f"  [streak_gate] RESCUED YES {c['ticker']} — streak30=bearish, vpin=-1 (compression not continuation), stoch_k={_sk:.1f}")
                    elif _sharp_move_pct_10m <= 0:
                        _gate = True
                        _gate_reason = f"streak30=bearish, stoch_k={_sk:.1f}, chg_10m={_sharp_move_pct_10m*100:+.2f}% (no bounce)"
                    else:
                        print(f"  [streak_gate] RESCUED YES {c['ticker']} — streak30=bearish, stoch_k={_sk:.1f}, chg_10m={_sharp_move_pct_10m*100:+.2f}% (bounce active)")
                elif dec_c.side == "no" and _streak30 == "bullish" and 30 <= _sk <= 60 and args.asset != "SOL":
                    _rescued = False
                    if args.asset == "BTC" and _sharp_move_pct_5m < 0:
                        _rescued = True
                        print(f"  [streak_gate] RESCUED NO {c['ticker']} — streak30=bullish, stoch_k={_sk:.1f}, chg_5m={_sharp_move_pct_5m*100:+.2f}% (reversing)")
                    elif args.asset == "ETH" and _sk >= 45:
                        _rescued = True
                        print(f"  [streak_gate] RESCUED NO {c['ticker']} — streak30=bullish, stoch_k={_sk:.1f}>=45 (upper band, mean-reversion entry)")
                    if not _rescued:
                        _gate = True
                        _gate_reason = f"streak30=bullish, stoch_k={_sk:.1f}, chg_5m={_sharp_move_pct_5m*100:+.2f}%"
                elif dec_c.side == "no" and _sk < 20 and args.asset == "BTC":
                    if _sharp_move_pct <= 0:
                        if _active_rev >= 2:
                            print(f"  [streak_gate] RESCUED NO {c['ticker']} — stoch_k={_sk:.1f}<20, chg_30m={_sharp_move_pct*100:+.2f}% BUT rev={_active_rev}>=2 (oversold, bounce imminent)")
                        else:
                            _gate = True
                            _gate_reason = f"stoch_k={_sk:.1f}<20, chg_30m={_sharp_move_pct*100:+.2f}% (no bounce, rev={_active_rev}<2)"
                    else:
                        print(f"  [streak_gate] RESCUED NO {c['ticker']} — stoch_k={_sk:.1f}<20, chg_30m={_sharp_move_pct*100:+.2f}% (bounce active)")
                if _gate:
                    print(f"  [streak_gate] Blocked {dec_c.side.upper()} {c['ticker']} — {_gate_reason}")
                    _log_block("streak_gate")
                    continue

            # [2026-05-17] BTC YES liq cascade gate:
            # Block YES when liq_score <= -1 (long cascade active) — all YES, not just OTM.
            # Sim (paper_trades): ITM YES at liq=-1 → n=73, edge=-35.2%, saves $257.
            # Rescue A: vpin_raw >= 0.75 AND p_market >= 0.38 AND offset_pct <= 0.08
            #   (smart money absorbing cascade at near-ATM level — revival possible)
            # Rescue B: stoch_k < 35 OR composite_rev >= 2
            #   Cascade exhaustion signal: deeply oversold stoch or model detects bounce setup.
            #   blocked_trades sim (n=50): stoch<35 → WR=85.7% BE=43.5% +42pp +$118;
            #   rev>=2 → WR=82.6-100% +38-58pp +$124. stoch>=35 AND rev<2 → WR=0-13% -40pp -$65.
            # Log: [liq_cascade_gate] BLOCK / RESCUED
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and _liq_signal is not None and _liq_signal.liq_score <= -1):
                _lc_vpin     = (confirm.vpin_raw == confirm.vpin_raw
                                and confirm.vpin_raw >= 0.75)
                _lc_sk       = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _lc_oversold = (_lc_sk < 35 or _active_rev >= 2)
                _lc_rescue   = (_lc_vpin and pm >= 0.38 and offset_c <= 0.08) or _lc_oversold
                if _lc_rescue:
                    if _lc_oversold:
                        _lc_why = f"stoch={_lc_sk:.1f}<35" if _lc_sk < 35 else f"rev={_active_rev}>=2"
                        print(f"  [liq_cascade_gate] RESCUED YES {c['ticker']} — "
                              f"cascade={_liq_signal.liq_score:+d} offset={offset_c*100:.2f}% "
                              f"but cascade exhaustion: {_lc_why}")
                    else:
                        print(f"  [liq_cascade_gate] RESCUED YES {c['ticker']} — "
                              f"cascade={_liq_signal.liq_score:+d} offset={offset_c*100:.2f}% "
                              f"but vpin_raw={confirm.vpin_raw:.3f}>=0.75 pm={pm:.3f}>=0.38 off<=8%")
                else:
                    print(f"  [liq_cascade_gate] BLOCK YES {c['ticker']} — "
                          f"liq_score={_liq_signal.liq_score:+d} ({_liq_signal.label}) "
                          f"offset={offset_c*100:.2f}% stoch={_lc_sk:.1f}>=35 rev={_active_rev}<2")
                    _log_block("liq_cascade_gate")
                    continue

            # BearDrift gate (BTC YES only):
            # Arm 1 — block when ema_stack=-1 + composite_rev<=3 + stoch_k>=35
            #          Rescues: vpin_score=1, ema_stretch_score=1
            # Arm 2 — block when ema_stack=-1 + composite_rev<=3 + stoch_k<25 + OTM (offset>0)
            #          Backtest: n=19, WR=5.3%, BE=75%, Δ=-69.7pp, net=-$629; no rescue found
            #          ITM stoch<35 (n=31, WR=83.9%) and stoch 25-35 OTM (n=10, WR=70%) pass through
            if dec_c.decision == "trade" and args.asset == "BTC" and dec_c.side == "yes":
                _bd_sk  = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _bd_ema = (confirm.ema_stack_bias == -1)
                _bd_rev = (_active_rev <= 3)
                if _bd_ema and _bd_rev:
                    # Arm 1: stoch>=35, gate fires
                    if _bd_sk >= 35:
                        _bd_liq_squeeze = (_liq_signal is not None and _liq_signal.liq_score >= 1)
                        _bd_rescued = (confirm.vpin_score == 1 or confirm.ema_stretch_score == 1
                                       or _bd_liq_squeeze)
                        if _bd_rescued:
                            if confirm.vpin_score == 1:
                                _bd_why = "vpin=1"
                            elif confirm.ema_stretch_score == 1:
                                _bd_why = "ema_stretch=1"
                            else:
                                _bd_why = f"liq_squeeze={_liq_signal.liq_score:+d} ({_liq_signal.label})"
                            print(f"  [bear_drift] RESCUED YES {c['ticker']} — ema_stack=-1, rev={_active_rev}, stoch_k={_bd_sk:.1f} ({_bd_why})")
                        else:
                            print(f"  [bear_drift] BLOCK YES {c['ticker']} — ema_stack=-1, rev={_active_rev}, stoch_k={_bd_sk:.1f}")
                            _log_block("bear_drift")
                            continue
                    # Arm 2: stoch<25 + OTM — extreme oversold in structural downtrend, not a reversal
                    elif _bd_sk < 25 and offset_c > 0:
                        print(f"  [bear_drift] BLOCK YES {c['ticker']} — ema_stack=-1, rev={_active_rev}, stoch_k={_bd_sk:.1f}<25 OTM (arm2)")
                        _log_block("bear_drift")
                        continue

            # [2026-05-17] ETH YES liq cascade gate:
            # Block YES when liq_score <= -1 (long cascade active).
            # Sim (paper_trades): ETH YES at liq=-1 → n=56, edge=-18.0%, saves ~$101.
            # Rescue: stoch_k < 35 OR composite_rev >= 2 (cascade exhaustion logic, same as BTC).
            # Log: [eth_liq_cascade_gate] BLOCK / RESCUED
            if (dec_c.decision == "trade" and args.asset == "ETH"
                    and dec_c.side == "yes"
                    and _liq_signal is not None and _liq_signal.liq_score <= -1):
                _elc_sk      = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _elc_oversold = (_elc_sk < 35 or _active_rev >= 2)
                if _elc_oversold:
                    _elc_why = f"stoch={_elc_sk:.1f}<35" if _elc_sk < 35 else f"rev={_active_rev}>=2"
                    print(f"  [eth_liq_cascade_gate] RESCUED YES {c['ticker']} — "
                          f"cascade={_liq_signal.liq_score:+d} but exhaustion: {_elc_why}")
                else:
                    print(f"  [eth_liq_cascade_gate] BLOCK YES {c['ticker']} — "
                          f"liq_score={_liq_signal.liq_score:+d} ({_liq_signal.label}) "
                          f"stoch={_elc_sk:.1f}>=35 rev={_active_rev}<2")
                    _log_block("eth_liq_cascade_gate")
                    continue

            # [2026-05-17] BTC NO liq squeeze gate:
            # Block NO when liq_score >= +1 (short squeeze active — price likely rising).
            # Sim (paper_trades): BTC NO at liq=+1 → n=114, edge=-19.2%, saves ~$139.
            # Rescue: stoch_k >= 80 (extreme overbought in squeeze = near exhaustion)
            #         OR composite_rev >= 3 (model detects reversal setup within squeeze).
            # Log: [btc_liq_squeeze_gate] BLOCK / RESCUED
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "no"
                    and _liq_signal is not None and _liq_signal.liq_score >= 1):
                _bsq_sk      = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _bsq_rescue  = (_bsq_sk >= 80 or _active_rev >= 3)
                if _bsq_rescue:
                    _bsq_why = f"stoch={_bsq_sk:.1f}>=80 (overbought)" if _bsq_sk >= 80 else f"rev={_active_rev}>=3"
                    print(f"  [btc_liq_squeeze_gate] RESCUED NO {c['ticker']} — "
                          f"squeeze={_liq_signal.liq_score:+d} but exhaustion: {_bsq_why}")
                else:
                    print(f"  [btc_liq_squeeze_gate] BLOCK NO {c['ticker']} — "
                          f"liq_score={_liq_signal.liq_score:+d} ({_liq_signal.label}) "
                          f"stoch={_bsq_sk:.1f}<80 rev={_active_rev}<3")
                    _log_block("btc_liq_squeeze_gate")
                    continue

            # [2026-05-17] ETH NO liq squeeze gate:
            # Block NO when liq_score >= +1 (short squeeze active).
            # Sim (paper_trades): ETH NO at liq=+1 → n=29, edge=-28.4%, saves ~$72.
            # Rescue: stoch_k >= 80 OR composite_rev >= 3 (same squeeze exhaustion logic as BTC).
            # Log: [eth_liq_squeeze_gate] BLOCK / RESCUED
            if (dec_c.decision == "trade" and args.asset == "ETH"
                    and dec_c.side == "no"
                    and _liq_signal is not None and _liq_signal.liq_score >= 1):
                _esq_sk      = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _esq_rescue  = (_esq_sk >= 80 or _active_rev >= 3)
                if _esq_rescue:
                    _esq_why = f"stoch={_esq_sk:.1f}>=80 (overbought)" if _esq_sk >= 80 else f"rev={_active_rev}>=3"
                    print(f"  [eth_liq_squeeze_gate] RESCUED NO {c['ticker']} — "
                          f"squeeze={_liq_signal.liq_score:+d} but exhaustion: {_esq_why}")
                else:
                    print(f"  [eth_liq_squeeze_gate] BLOCK NO {c['ticker']} — "
                          f"liq_score={_liq_signal.liq_score:+d} ({_liq_signal.label}) "
                          f"stoch={_esq_sk:.1f}<80 rev={_active_rev}<3")
                    _log_block("eth_liq_squeeze_gate")
                    continue

            # [2026-05-12] 1h EMA stack=3 YES gate (all assets):
            # Block YES when EMA20/50/100 are all below spot but EMA200 is still above —
            # price is in a local uptrend but running into long-term overhead resistance.
            # Sim (2834 trades, 1h+15m): YES ema_stack=3 → WR=49.4%, PnL=-$1,101.
            # Revert: remove this block.
            if dec_c.decision == "trade" and dec_c.side == "yes" and _ema_stack_liq_1h == 3:
                print(f"  [ema_stack3_gate] BLOCK YES {c['ticker']} — "
                      f"ema_stack_liq=3 (EMA20/50/100 below spot, EMA200 overhead)")
                _log_block("ema_stack3_gate")
                continue

            # [Gate 1b — BTC deep-OTM YES block, 2026-05-31]
            # Hard block YES when pm < 0.15; rescue only on genuine reversal/bounce signals.
            # Simulation (Apr–May 2026, 63 bets at pm<0.15):
            #   Hard block (no rescue): n=63, WR=7.8% (BE~10%), Δ=+$1,211
            #   With rescue: blocked=43 WR=2.3%, rescued=20 kept, Δ=+$1,314 (+$103 vs hard block)
            # Rescue logic: composite_rev<-2 = multiple reversal indicators firing (mean-reversion
            #   signal overrides OTM barrier); ema_stack=-1+stoch<15 = stoch_bounce in bear trend.
            # Backup: paper_trade_runner_pre_yes_gates_20260531.py
            if dec_c.decision == "trade" and args.asset == "BTC" and dec_c.side == "yes":
                if pm < 0.15:
                    _sk_g1 = float(confirm.stoch_k) if confirm.stoch_k == confirm.stoch_k else 50.0
                    _rev_rescue  = (_active_rev < -2)
                    _bnc_rescue  = (confirm.ema_stack_bias == -1 and _sk_g1 < 15.0)
                    if not (_rev_rescue or _bnc_rescue):
                        print(f"  [btc_deepotm_gate] BLOCK YES {c['ticker']} — "
                              f"pm={pm:.3f}<0.15, rev={_active_rev}, ema={confirm.ema_stack_bias}, "
                              f"stoch={_sk_g1:.1f} (no reversal/bounce rescue)")
                        _log_block("btc_deepotm_gate")
                        continue
                    else:
                        _why = []
                        if _rev_rescue: _why.append(f"rev={_active_rev}<-2")
                        if _bnc_rescue: _why.append(f"ema=-1+stoch={_sk_g1:.1f}<15")
                        print(f"  [btc_deepotm_gate] RESCUE YES {c['ticker']} — "
                              f"pm={pm:.3f}<0.15 but {'+'.join(_why)}")
                # Original VPIN-based gate: retain for pm [0.15, 0.20) with vpin=0
                elif pm < 0.20 and confirm.vpin_score == 0:
                    print(f"  [btc_otmlow_gate] BLOCK YES {c['ticker']} — "
                          f"p_market={pm:.3f}<0.20 vpin=0 (deep OTM, no smart money)")
                    _log_block("btc_otmlow_gate")
                    continue

            # [Gate mid — BTC OTM YES in neutral EMA trend, 2026-05-31]
            # Block YES when 0.15 ≤ pm < 0.30 AND ema_stack_bias == 0 (no trend alignment).
            # Simulation (Apr–May 2026, 100 bets): WR=12.0% (BE~22–25%), PnL=-$1,217.
            # Full rescue search across 22 signals (stoch_k_4h, rsi_4h, p_up_v2, HMM 3-state,
            # HMM 7-state, markov_daily, ema20_dist, bb_pct, adx, rvol, composite_rev, etc.)
            # found no stable cross-month rescue: best candidate (p_yes>=0.40 in Recovery HMM)
            # is entirely from 9 May 3–6 bets — one market episode, not a pattern.
            # Losses avoided=88, wins sacrificed=0, Δ=+$1,217.
            # Backup: paper_trade_runner_pre_yes_gates_20260531.py
            if (dec_c.decision == "trade" and args.asset == "BTC" and dec_c.side == "yes"
                    and 0.15 <= pm < 0.30 and confirm.ema_stack_bias == 0):
                print(f"  [btc_otm_neutral_ema_gate] BLOCK YES {c['ticker']} — "
                      f"pm={pm:.3f} in [0.15,0.30), ema_stack=0 (no trend, OTM WR=12% vs BE~23%)")
                _log_block("btc_otm_neutral_ema_gate")
                continue

            # [2026-05-05] ETH/SOL OTM-low gate: hard block YES when p_market<0.20 AND vpin=0.
            # ETH: n=14, WR=0% (BE=11.1%), PnL=-$631 — saves +$631. No rescue warranted.
            # SOL: n=3, WR=0% (BE=9.5%), PnL=-$106 — small sample but consistent 0% WR pattern.
            # Same logic as BTC gate: market below 20¢ AND no smart money = no edge.
            # Revert: remove this block.
            if dec_c.decision == "trade" and args.asset in ("ETH", "SOL") and dec_c.side == "yes":
                if pm < 0.20 and confirm.vpin_score == 0:
                    print(f"  [otmlow_gate] BLOCK YES {c['ticker']} — "
                          f"p_market={pm:.3f}<0.20 vpin=0 (deep OTM, no smart money)")
                    _log_block("otmlow_gate")
                    continue

            # [2026-05-05] BTC structure gate: block YES when structure_bias=-1 unless a
            # reversal catalyst is present. Backtest: base n=121 (WR=47.9%, -$505);
            # rescue n=53 (WR=60.4%, +$309); block n=68 (WR=38.2%, -$814, saves +$814).
            # Rescues (any one sufficient):
            #   chg_5m>=+0.05%  — local up impulse (reversal in progress)
            #   vwap_score=-1   — price elevated above VWAP within bearish structure
            #   chg_30m<-0.20%  — sharp recent drop sets up technical bounce
            # Revert: remove this block.
            if dec_c.decision == "trade" and args.asset == "BTC" and dec_c.side == "yes":
                if struct.structure_bias == -1:
                    _sg_chg5m_up = _sharp_move_pct_5m >= 0.0005   # chg_5m >= +0.05%
                    _sg_vwap_neg = confirm.vwap_score == -1
                    _sg_drop30m  = _sharp_move_pct < -0.002        # chg_30m < -0.20%
                    _sg_rescued  = _sg_chg5m_up or _sg_vwap_neg or _sg_drop30m
                    if _sg_rescued:
                        _sg_why = []
                        if _sg_chg5m_up: _sg_why.append(f"chg_5m={_sharp_move_pct_5m*100:+.3f}%>=0.05")
                        if _sg_vwap_neg: _sg_why.append("vwap_score=-1")
                        if _sg_drop30m:  _sg_why.append(f"chg_30m={_sharp_move_pct*100:+.3f}%<-0.20")
                        print(f"  [btc_struct_gate] RESCUED YES {c['ticker']} — struct=-1 ({', '.join(_sg_why)})")
                    else:
                        print(f"  [btc_struct_gate] BLOCK YES {c['ticker']} — struct=-1 "
                              f"chg_5m={_sharp_move_pct_5m*100:+.3f}% vwap={confirm.vwap_score} "
                              f"chg_30m={_sharp_move_pct*100:+.3f}%")
                        _log_block("btc_struct_gate")
                        continue

            # [BTC YES ema=0 deep-oversold gate] Block YES when EMA stack neutral + price 2σ+ below VWAP.
            # ema=0 (trend broken) + stretch=+2 (2σ+ below VWAP) = structural decline, not bounce setup.
            # Rescue: stoch_k >= 10 AND ITM (offset <= 0) — some momentum recovery + strike already above
            # price means BTC just needs to hold, not rally. Backtest n=8, WR=100%, +$151.
            # OTM stays blocked even with stoch>=10 (n=5, WR=40%, -$67). Flat block n=31, saves $327.
            # Backtest: n=39 total, WR=41.0%, BE=48.6%. Rescue adds +$302 vs flat block.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and confirm.ema_stack_bias == 0
                    and confirm.stretch_score == 2):
                _stoch_ok = confirm.stoch_k is not None and confirm.stoch_k >= 10
                _es2_rescue = _stoch_ok and offset_c <= 0
                if _es2_rescue:
                    print(f"  [btc_ema0_stretch2_gate] RESCUED YES {c['ticker']} — "
                          f"stoch={confirm.stoch_k:.0f}>=10 ITM offset={offset_c*100:+.2f}% "
                          f"rev={_active_rev} stretch=+2")
                else:
                    print(f"  [btc_ema0_stretch2_gate] BLOCK YES {c['ticker']} — "
                          f"ema=0 stretch=+2 rev={_active_rev} stoch={confirm.stoch_k:.0f} "
                          f"offset={offset_c*100:+.2f}%")
                    _log_block("btc_ema0_stretch2_gate")
                    continue

            # [BTC YES OTM neutral-EMA gate] Block YES when ema_stack=0 (no trend conviction) +
            # composite_p_up>=0.60 (model overconfident bullish) + OTM (BTC below strike).
            # Backtest (n=81 OTM blocked): WR=24.7% vs BE=32.6% (Δ=-7.9pp), saves $732.
            # ITM (offset<0, n=60): WR=63.3% ≈ BE=63.9% — passes through.
            # Time-split: Apr Δ=-13.3pp, May Δ=-4.6pp (consistent).
            # Logic: neutral EMA + overconfident upside model + BTC hasn't reached strike = unlikely to.
            # Rescue (2026-05-10): liq_score>=1 (short squeeze active) invalidates the neutral-EMA fade
            # thesis — the squeeze IS the directional catalyst that can push price through the strike.
            # liq>=2: n=11, WR=72.7%, BE=34.8%, Δ=+37.9%, +$539. liq>=1: n=34, WR=44.1%, Δ=+12.6%, +$438.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and confirm.ema_stack_bias == 0
                    and _comp_p_up is not None and _comp_p_up >= 0.60
                    and offset_c > 0):
                _otmn_liq_rescue = (_liq_signal is not None and _liq_signal.liq_score >= 1)
                _otmn_sk = confirm.stoch_k if (confirm.stoch_k == confirm.stoch_k) else 0.0
                # Interaction rescues (full_opportunity_analysis 2026-05-17): neutral EMA does not
                # override if funding + structure or momentum signals confirm YES continuation.
                _otmn_interaction_rescue = (
                    (struct.structure_bias == 0 and confirm.funding_bias == -1) or   # WR=77.9% n=1616
                    (_otmn_sk >= 80.0 and struct.structure_bias == 0) or             # WR=71.8% n=2669
                    (_active_trend == 0 and confirm.funding_bias == -1)              # WR=69.7% n=2664
                )
                if _otmn_liq_rescue:
                    print(f"  [btc_otm_neutral_gate] RESCUED YES {c['ticker']} — "
                          f"liq_score={_liq_signal.liq_score:+d} ({_liq_signal.label}) "
                          f"ema=0 p_up={_comp_p_up:.3f} offset={offset_c*100:+.2f}%")
                elif _otmn_interaction_rescue:
                    _otmn_why = ("struct=0+fund=-1" if struct.structure_bias == 0 and confirm.funding_bias == -1
                                 else "stoch>=80+struct=0" if _otmn_sk >= 80.0 and struct.structure_bias == 0
                                 else "ct=0+fund=-1")
                    print(f"  [btc_otm_neutral_gate] RESCUED YES {c['ticker']} — "
                          f"{_otmn_why} ema=0 p_up={_comp_p_up:.3f} offset={offset_c*100:+.2f}%")
                else:
                    print(f"  [btc_otm_neutral_gate] BLOCK YES {c['ticker']} — "
                          f"ema_stack=0 p_up={_comp_p_up:.3f} offset={offset_c*100:+.2f}% pm={pm:.3f}")
                    _log_block("btc_otm_neutral_gate")
                    continue

            # [BTC YES ema=0 ITM gate] Block YES when neutral EMA + strong bullish trend + rev=0 + ITM.
            # ema=0 = no trend conviction; rev=0 = no mean-reversion anchor either.
            # Backtest: blocked n=24 (WR=37.5% vs BE=59.6%, Δ=-22.1pp), saves $355.
            # Rescue: vwap_stretch=-1 (BTC 1-2σ above VWAP), n=4 WR=75% BE=61.7% Δ=+13.3pp, +$58.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and confirm.ema_stack_bias == 0
                    and offset_c <= 0
                    and _active_trend >= 3
                    and _active_rev == 0):
                if confirm.stretch_score == -1:
                    print(f"  [btc_ema0_itm_gate] RESCUED YES {c['ticker']} — "
                          f"vwap_stretch=-1 ema=0 trend={_active_trend:+d} rev=0 offset={offset_c*100:+.2f}%")
                else:
                    print(f"  [btc_ema0_itm_gate] BLOCK YES {c['ticker']} — "
                          f"ema=0 ITM trend={_active_trend:+d} rev=0 "
                          f"vwap_stretch={confirm.stretch_score} offset={offset_c*100:+.2f}%")
                    _log_block("btc_ema0_itm_gate")
                    continue

            # [BTC YES exhaustion gate] Block YES when bullish EMA but all three exhaustion signals fire:
            # composite_rev<=-4 (max reversal score), stoch>=75 (overbought), vwap_stretch<=-1 (≥1σ above VWAP).
            # stretch<=-1 covers both 1-2σ (==-1, WR=30.0%, -$335) and >2σ (==-2, WR=51.9%, -$102) zones.
            # Using <= not == to avoid exact-equality evaluation timing miss (gate fired 0x post-implementation).
            # Rescue search exhaustive — no sub-condition with sound causal logic found. Flat block.
            # Backtest: n=47, WR=42.6%, BE=54.4%, Δ=-11.8pp, -$437 saved (vs -$335 with ==−1 only).
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and confirm.ema_stack_bias == 1
                    and _active_rev <= -4
                    and confirm.stretch_score <= -1):
                _exh_sk = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                if _exh_sk >= 75:
                    # Rescue: fund=-1 = shorts paying longs even at exhaustion — squeeze not done.
                    # ema=1+fund=-1 WR=96.3% edge=+41.8% n=459 across all blocked trades.
                    _exh_rescue = (confirm.funding_bias == -1)
                    if _exh_rescue:
                        print(f"  [btc_exhaustion_gate] RESCUED YES {c['ticker']} — "
                              f"fund=-1 (shorts still paying) overrides exhaustion "
                              f"ema=1 rev={_active_rev} stoch={_exh_sk:.1f}")
                    else:
                        print(f"  [btc_exhaustion_gate] BLOCK YES {c['ticker']} — "
                              f"ema=1 rev={_active_rev} stoch={_exh_sk:.1f} vwap_stretch=-1")
                        _log_block("btc_exhaustion_gate")
                        continue

            # [2026-05-06] BTC YES OTM downtrend gate: block deeply-OTM YES bets when 5m trend
            # is bearish (ADX or lower-highs/lows structure) AND market is skeptical (pm<0.27).
            # Two independent trend signals — either one fires the gate (OR logic):
            #   ADX>20 + DI->DI+: directional trend confirmed bearish on 5m chart
            #   lower_hl: classic downtrend structure (lower-high + lower-low in last 40 1m bars)
            # Rescue: vwap_stretch_score==2 (price 2σ above VWAP — spike brings strike within reach).
            # Backtest (n=882 BTC YES resolved): hard-blocked n=59, WR=5.1%, BE=17.7%, saves $1,578.
            # Rescue n=14 WR=21.4% BE=12.8% Δ=+8.6pp net=+$245 (above breakeven — let through).
            # Revert: cp paper_trade_runner_pre_adx5_gate.py paper_trade_runner.py
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and offset_c > 0
                    and pm < 0.27):
                _adx5_bear = (_adx_5m is not None
                              and _adx_5m > 20
                              and _di_m_5m is not None
                              and _di_p_5m is not None
                              and _di_m_5m > _di_p_5m)
                _adx5_trend_dn = _adx5_bear or _lower_hl
                if _adx5_trend_dn:
                    if confirm.stretch_score == 2:
                        _adx5_why = "ADX_bear" if _adx5_bear else "lower_HL"
                        print(f"  [btc_adx5_gate] RESCUED YES {c['ticker']} — "
                              f"{_adx5_why} OTM pm={pm:.3f} vwap_stretch=2 (price spike near strike)")
                    else:
                        _adx5_why = []
                        if _adx5_bear:
                            _adx5_why.append(f"ADX={_adx_5m:.1f}>20 DI-={_di_m_5m:.1f}>DI+={_di_p_5m:.1f}")
                        if _lower_hl:
                            _adx5_why.append("lower_HL")
                        print(f"  [btc_adx5_gate] BLOCK YES {c['ticker']} — "
                              f"{' '.join(_adx5_why)} "
                              f"offset={offset_c*100:+.2f}% pm={pm:.3f} "
                              f"vwap_stretch={confirm.stretch_score}")
                        _log_block("btc_adx5_gate")
                        continue

            # [BTC YES falling knife gate] Block YES when composite_rev >= 4 AND chg_30m < -0.20%:
            # the 30m/15m scorer detects extreme oversold (model expects imminent bounce) while price
            # is still actively declining. This combination systematically produces falling-knife losses:
            #   Backtest (n=900 BTC YES resolved): blocked n=40, WR=32.5% vs BE~49%, PnL=-$146 saved.
            # Rescue: chg_5m > +0.10% (5m momentum just reversed — bounce may be starting)
            #         OR offset_pct < -0.10% (contract deeply ITM, large buffer absorbs further drop).
            #   Rescue n=16, WR=81.2%, PnL=+$89 — let through.
            # Revert: remove this block.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and _active_rev >= 4
                    and _sharp_move_pct < -0.0020):
                _fk_liq_cascade_itm = (_liq_signal is not None
                                       and _liq_signal.liq_score <= -1
                                       and offset_c <= 0)
                _fk_rescue = ((_sharp_move_pct_5m > 0.0010)
                              or (offset_c < -0.0010)
                              or _fk_liq_cascade_itm)
                if _fk_rescue:
                    if _sharp_move_pct_5m > 0.0010:
                        _fk_why = "chg_5m>+0.10% (5m reversing)"
                    elif offset_c < -0.0010:
                        _fk_why = f"offset={offset_c*100:.2f}%<-0.10% (deep ITM)"
                    else:
                        _fk_why = (f"liq_cascade={_liq_signal.liq_score:+d} ITM "
                                   f"offset={offset_c*100:.2f}% (cascade+oversold bounce)")
                    print(f"  [btc_falling_knife_gate] RESCUED YES {c['ticker']} — "
                          f"cr={_active_rev} chg_30m={_sharp_move_pct*100:+.3f}% {_fk_why}")
                else:
                    print(f"  [btc_falling_knife_gate] BLOCK YES {c['ticker']} — "
                          f"cr={_active_rev}>=4 chg_30m={_sharp_move_pct*100:+.3f}%<-0.20% "
                          f"chg_5m={_sharp_move_pct_5m*100:+.3f}% offset={offset_c*100:+.2f}%")
                    _log_block("btc_falling_knife_gate")
                    continue

            # [BTC YES body-bp gate] Block YES when body_15m in [0.50, 0.60) AND bp_5m < 0.55.
            # Causal story: intermediate 15m body (50-60% of range) = directional but uncommitted.
            # bp_5m tells you which way: if the most recent 5m bar closed below the midpoint
            # (bp<0.55), selling pressure is continuing → block YES. If the 5m bar closed
            # at/above the midpoint (bp>=0.55), selling stalled → rescue passes through.
            # Backtest: hard-block n=88 (WR=28.4%, BE=45.7%, Δ=-17.3%, saves $1,031);
            #           rescue n=45 (WR=55.6%, BE=44.4%, Δ=+11.2%, keeps $71);
            #           net +$1,102 vs no gate.
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "yes"
                    and _body_15m is not None and _bp_5m is not None
                    and 0.50 <= _body_15m < 0.60):
                if _bp_5m >= 0.55:
                    print(f"  [btc_body_bp_gate] RESCUED YES {c['ticker']} — "
                          f"body={_body_15m:.3f} in [0.50,0.60) but bp={_bp_5m:.3f}>=0.55 "
                          f"(5m sellers stalled)")
                else:
                    print(f"  [btc_body_bp_gate] BLOCK YES {c['ticker']} — "
                          f"body={_body_15m:.3f} in [0.50,0.60) AND bp={_bp_5m:.3f}<0.55 "
                          f"(sustained selling pressure)")
                    _log_block("btc_body_bp_gate")
                    continue

            # [btc_contra_bar_gate] Block contra-trend bets when the last 15m bar is large
            # and directional. A body>=0.70 bar means strong committed momentum; betting
            # against it (YES into a big red bar, NO into a big green bar) fights the tape.
            # Does NOT overlap with btc_body_bp_gate which handles body in [0.50, 0.60).
            # Backtest (1,461 resolved BTC hourly trades via parquet):
            #   Block YES when body>=0.70, dir=-1: +$1,446 vs baseline (+$3,938 vs $2,492)
            #   Block NO  when body>=0.70, dir=+1: +$90 contribution
            #   Combined both sides: N=1114, WR=62.2%, PnL=+$4,028 (+$1,536 vs $2,492)
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and _body_15m is not None and _dir_15m is not None
                    and _body_15m >= 0.70):
                if dec_c.side == "yes" and _dir_15m == -1:
                    print(f"  [btc_contra_bar_gate] BLOCK YES {c['ticker']} — "
                          f"body={_body_15m:.3f}>=0.70 dir=-1 (large bearish bar)")
                    _log_block("btc_contra_bar_gate")
                    continue
                elif dec_c.side == "no" and _dir_15m == 1:
                    print(f"  [btc_contra_bar_gate] BLOCK NO {c['ticker']} — "
                          f"body={_body_15m:.3f}>=0.70 dir=+1 (large bullish bar)")
                    _log_block("btc_contra_bar_gate")
                    continue

            # [btc_no_highpm_bearema_gate] Block NO bets when pm∈(0.70,0.75) + ema=-1 + ct=-1.
            # Refined 2026-06-06: original gate blocked ALL ema=-1+pm>0.70. Large-n analysis showed:
            #   pm[0.70,0.75) ema=-1 ct=-1: WR=24.4% vs BEV=27.2% (n=45) → CORRECTLY loses
            #   pm[0.70,0.75) ema=-1 ct=0:  WR=34.8% vs BEV=26.5% (n=46) → profitable, do NOT block
            #   pm[0.75,0.90) ema=-1:        WR=21-32% vs BEV=13-23% (n=2,968) → all profitable, do NOT block
            # Narrowed to the single losing bucket: pm<0.75 + ct=-1 (moderate bearish, not yet decisive).
            # Revert: paper_trade_runner_pre_rr_bearema_20260606.py
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "no"
                    and not _vsa_no_flip
                    and 0.70 < pm < 0.75
                    and confirm.ema_stack_bias == -1
                    and _active_trend == -1):
                print(f"  [btc_no_highpm_bearema_gate] BLOCK NO {c['ticker']} — "
                      f"pm={pm:.3f}∈(0.70,0.75) ema=-1 ct=-1 (moderate bearish, WR=24.4%<BEV=27.2%)")
                _log_block("btc_no_highpm_bearema_gate")
                continue

            # [BTC NO p_up extremes gate] Block NO bets when composite_p_up is at an extreme:
            #   p_up <= 0.36: model is extremely bearish — move already priced in, bounce risk.
            #   p_up >= 0.50: model and NO bet direction are in conflict (model says BTC up).
            # Gate scoped to pm >= 0.20: deep-OTM NO bets (pm<0.20) have enough strike cushion.
            # Rescue: stretch_score==1 (price 1-2σ below VWAP, downtrend is structural, not exhausted)
            #         OR vol_score==1 (high-vol bar already happened — move done, NO holds)
            # Backtest (BTC, n=1,282 resolved): hard-blocks 62 (WR=50% vs BE=73%), rescues 24 (WR=83%)
            # Net improvement: +$659 on NO model ($155 → $814 for these 93 trades).
            if (dec_c.decision == "trade" and args.asset == "BTC"
                    and dec_c.side == "no" and not _vsa_no_flip and pm >= 0.20
                    and _comp_p_up is not None
                    and (_comp_p_up <= 0.36 or _comp_p_up >= 0.50)):
                _nopup_stretch  = confirm.stretch_score == 1
                _nopup_vol      = confirm.vol_score == 1
                if _nopup_stretch or _nopup_vol:
                    _nopup_why = []
                    if _nopup_stretch: _nopup_why.append(f"stretch={confirm.stretch_score}")
                    if _nopup_vol:     _nopup_why.append(f"vol_score={confirm.vol_score}")
                    print(f"  [btc_nopup_gate] RESCUED NO {c['ticker']} — "
                          f"p_up={_comp_p_up:.3f} pm={pm:.3f} ({', '.join(_nopup_why)})")
                else:
                    _nopup_dir = "extreme_bear" if _comp_p_up <= 0.36 else "model_conflict"
                    print(f"  [btc_nopup_gate] BLOCK NO {c['ticker']} — "
                          f"p_up={_comp_p_up:.3f} ({_nopup_dir}) pm={pm:.3f} "
                          f"stretch={confirm.stretch_score} vol={confirm.vol_score}")
                    _log_block("btc_nopup_gate")
                    continue

            # [eth_kelly_tier] 1.5× Kelly for high-conviction candle+rev signals.
            # YES tier: RED candle + rev>=3 = oversold bounce entry.
            #   Live data n=50: WR=68.0%, BE=57.8%, WR-BE=+10.2pp, $/t=+$0.102 (vs +$0.047 base).
            # NO  tier: YES-ITM + GREEN candle + rev>=2 = blow-off top fade.
            #   Sim n=18: WR=55.6%, BE=27.4%, WR-BE=+28.2pp, $/t=+$0.277.
            # Counter-tape gate runs after and can still dampen.
            if (args.asset == "ETH" and dec_c.decision == "trade"
                    and not _sharp_move_active):
                _eth_boost     = 1.0
                _eth_boost_why = ""
                if dec_c.side == "yes" and _1h_candle_red and _active_rev >= 3:
                    _eth_boost     = 1.5
                    _eth_boost_why = f"YES RED+rev={_active_rev}>=3 (oversold bounce)"
                elif (dec_c.side == "no" and s_k < spot
                      and _1h_candle_green and _active_rev >= 2):
                    _eth_boost     = 1.5
                    _eth_boost_why = f"NO YES-ITM GREEN+rev={_active_rev}>=2 (blow-off fade)"
                if _eth_boost > 1.0:
                    _max_bet_f = 0.02 if (dec_c.side == "yes" and pm > 0.75) else 0.05
                    dec_c.bet_amount   = round(
                        min(dec_c.bet_amount * _eth_boost, args.bankroll * _max_bet_f), 2)
                    dec_c.bet_fraction = min(dec_c.bet_fraction * _eth_boost, _max_bet_f)
                    print(f"  [eth_kelly_tier] BOOST {dec_c.side.upper()} {c['ticker']} — "
                          f"{_eth_boost_why} 1.5× → ${dec_c.bet_amount:.2f}")

            # [semi_markov_vol_gate] Non-homogeneous semi-Markov R1 gate (BTC only)
            # Zone thresholds are macro-state-conditional (non-homogeneous Markov):
            #   P(R1→R0) varies 2.84× across macro states; R1 mean duration:
            #   Bear=29 bars, Sideways=17 bars, Bull=16 bars (empirical, 84k 15m bars).
            # Backup: paper_trade_runner_pre_semimarkov_20260602.py
            #
            # Lookup: (early_max, mid_max, rvol_rescue_ceil)
            # early_max: bar threshold for spike zone (hazard-peak region, universal ~3)
            # mid_max:   bar threshold for settling zone (scales with macro R1 mean duration)
            # rvol_rescue_ceil: max rvol for deep-R1 rescue (tighter in Bear — episodes genuinely long)
            _SM_ZONES = {
                "Bear":     (3, 21, 0.60),  # R1 mean ~29 bars; conservative rescue
                "Sideways": (3, 12, 1.00),  # R1 mean ~17 bars
                "Bull":     (3, 11, 1.00),  # R1 mean ~16 bars (weighted Recovery+Bull)
            }
            _sm_macro = _markov_regime if _markov_regime in _SM_ZONES else "Sideways"
            _sm_early_max, _sm_mid_max, _sm_rvol_ceil = _SM_ZONES[_sm_macro]

            if (args.asset == "BTC" and dec_c.decision == "trade"
                    and _hmm_vol_probs is not None and _hmm_vol_probs[0] == 1):
                _sm_tis   = _hmm_vol_probs[3]
                _sm_liq_s = _liq_signal.liq_score if _liq_signal is not None else 0
                _sm_liq_b = float(_liq_signal.liq_bias) if _liq_signal is not None else 0.0
                _sm_fund  = confirm.funding_bias
                _sm_rvol  = _rvol_1h if (_rvol_1h == _rvol_1h) else 1.0

                if _sm_tis <= _sm_early_max:
                    # Early R1: spike entry. YES always blocked.
                    # NO rescue: liq cascade (bearish liquidation = tailwind for NO).
                    if dec_c.side == "yes":
                        print(f"  [semi_markov_r1_early] BLOCK YES {c['ticker']} — "
                              f"R1 bar {_sm_tis}/{_sm_early_max} macro={_sm_macro}")
                        _log_block("semi_markov_r1_early_yes")
                        continue
                    if dec_c.side == "no" and not (_sm_liq_s <= -1 and _sm_liq_b < 0.0):
                        print(f"  [semi_markov_r1_early] BLOCK NO {c['ticker']} — "
                              f"R1 bar {_sm_tis}/{_sm_early_max} macro={_sm_macro}, "
                              f"no liq cascade (liq_score={_sm_liq_s} bias={_sm_liq_b:.2f})")
                        _log_block("semi_markov_r1_early_no")
                        continue
                    print(f"  [semi_markov_r1_early] RESCUE NO {c['ticker']} — "
                          f"liq cascade (liq_score={_sm_liq_s} bias={_sm_liq_b:.2f})")

                elif _sm_tis <= _sm_mid_max:
                    # Mid R1: settling. Block when funding neutral — no directional anchor.
                    if _sm_fund == 0:
                        print(f"  [semi_markov_r1_mid] BLOCK {dec_c.side.upper()} {c['ticker']} — "
                              f"R1 bar {_sm_tis}/{_sm_mid_max} macro={_sm_macro}, funding neutral")
                        _log_block("semi_markov_r1_mid_neutral_funding")
                        continue

                else:
                    # Deep R1: committed episode, hazard below geometric baseline.
                    # Rescue when rvol decayed below macro-adjusted ceil (HMM overhang).
                    # Bear ceil is tighter (0.6) — long episodes can have temporary rvol dips.
                    if _sm_rvol >= _sm_rvol_ceil:
                        print(f"  [semi_markov_r1_deep] BLOCK {dec_c.side.upper()} {c['ticker']} — "
                              f"R1 bar {_sm_tis} macro={_sm_macro}, "
                              f"rvol={_sm_rvol:.2f}>={_sm_rvol_ceil:.1f}")
                        _log_block("semi_markov_r1_deep_highrvol")
                        continue
                    print(f"  [semi_markov_r1_deep] RESCUE {dec_c.side.upper()} {c['ticker']} — "
                          f"R1 bar {_sm_tis} macro={_sm_macro}, "
                          f"rvol={_sm_rvol:.2f}<{_sm_rvol_ceil:.1f} (regime decaying)")

            # [hmm_mtf_st3_gate] Block BTC NO when HMM MTF State 3 (moderate bullish momentum).
            # State 3: stoch_k_1h≈64, rsi_1h≈57, bp_1h≈0.86, macd_hist_1h≈42 — uptrend/churn.
            # WR=24.1% across 10,174 obs; -$149.7/trade in model backtests.
            # Rescue: offset∈[0,5%) + macd_hist_1h<-50 (near-ATM + MACD already rolling over).
            if (args.asset == "BTC" and dec_c.decision == "trade"
                    and dec_c.side == "no" and _hmm_mtf_state == 3):
                _st3_off   = float(offset_c) if not math.isnan(float(offset_c)) else 0.10
                _st3_macd  = _macd_hist_1h_mtf
                _st3_rescue = (0.00 <= _st3_off < 0.05 and _st3_macd < -50.0)
                if _st3_rescue:
                    print(f"  [hmm_mtf_st3] RESCUE NO {c['ticker']} — "
                          f"St3 but offset={_st3_off*100:.2f}%∈[0,5%), "
                          f"macd_hist_1h={_st3_macd:.1f}<-50 (MACD rolling over)")
                else:
                    _st3_why = (f"offset={_st3_off*100:.2f}%∉[0,5%)"
                                if not (0.00 <= _st3_off < 0.05)
                                else f"macd_hist_1h={_st3_macd:.1f}>=-50")
                    print(f"  [hmm_mtf_st3] BLOCK NO {c['ticker']} — "
                          f"St3 (bullish momentum): {_st3_why}")
                    _log_block("hmm_mtf_st3")
                    continue

            # [bp_1h_no_gate] Block BTC NO when 1h bar closed near its high (strong buying
            # pressure) AND pm >= 0.40.  bp_1h = (close-low)/(high-low); >= 0.55 means close
            # is in the top 45% of the bar's range — momentum is upward against the NO.
            # Backtest: n=14,706, WR=7.7%, bkev=69.0%, edge=-61.3%, Δ=$+83,862 on 182k scan rows.
            # MCPT p=0.000 (500 perms, label-shuffle) — not optimization bias.
            # No rescue: MACD rescue tested and made performance worse on every threshold.
            # Backup: paper_trade_runner_pre_hmm_mtf_gate_20260603.py
            if (args.asset == "BTC" and dec_c.decision == "trade"
                    and dec_c.side == "no" and pm >= 0.40):
                _bp1h = _bp_1h_mtf if not math.isnan(_bp_1h_mtf) else 0.5
                if _bp1h >= 0.55:
                    print(f"  [bp_1h_no_gate] BLOCK NO {c['ticker']} — "
                          f"bp_1h={_bp1h:.3f}>=0.55, pm={pm:.3f}>=0.40 "
                          f"(bullish 1h bar, upward momentum against NO)")
                    _log_block("bp_1h_no_gate")
                    continue

            # [no_bp1h_chg1h] Block BTC NO (ema=-1) when no real downward momentum:
            # bp_1h>=0.45 (1h bar closed in upper half of range) OR chg_1h>=-0.002% (flat/rising hour).
            # ema=-1 is positioning without follow-through — market is not actually falling.
            # MCPT walkforward p=0.000, rank=2000/2000 (2,256 OOS blocked, WR=11.7% vs BEV=22.9%).
            # Rescue (allow through): bp_1h<0.45 AND chg_1h<-0.002% (confirmed downward momentum).
            # Backup: paper_trade_runner_pre_no_bp1h_chg1h_20260606.py
            if (args.asset == "BTC" and dec_c.decision == "trade"
                    and dec_c.side == "no"
                    and offset_c < 0
                    and 0.55 <= pm < 0.92
                    and confirm.ema_stack_bias == -1):
                _bp1h_nbp  = _bp_1h_mtf if not math.isnan(_bp_1h_mtf) else 0.5
                _chg1h_nbp = _chg_1h_mtf
                if _bp1h_nbp >= 0.45 or _chg1h_nbp >= -0.002:
                    print(f"  [no_bp1h_chg1h] BLOCK NO {c['ticker']} — "
                          f"bp_1h={_bp1h_nbp:.3f} chg_1h={_chg1h_nbp:+.3f}% "
                          f"(ema=-1 without momentum; no downward follow-through)")
                    _log_block("no_bp1h_chg1h")
                    continue

            # [pc1_rsi_no_gate] Block BTC NO when PC1 RSI score <= -34.93 (bottom decile of
            # training dist): fast RSI (2-8 bar) has diverged far below slow RSI (16-24 bar),
            # signalling short-term momentum collapse. Archive backtest: n=16,091, WR=76.9%,
            # BEV=84.4%, edge=-7.5%, saves $1,210. MCPT z=+46, p=0.0000.
            # Rescues (all validated, positive edge in blocked pool):
            #   no_score>=3:            WR=89.3% vs BEV=84.9% (+4.4%)
            #   vpin_score=1:           WR=88.6% vs BEV=84.5% (+4.0%)
            #   vwap_stretch<=-2:       WR=89.1% vs BEV=84.2% (+4.8%)
            #   rvol>2 + stoch>60:      WR=94.2% vs BEV=85.4% (+8.8%)  [strongest]
            # Expanded rescue set: n=3,259 spared (WR=90.5% vs BEV=84.9%), saves $1,392, z=+55.
            # PC1 is independent of existing gates: corr(PC1,ema_stack)=-0.02, corr(PC1,vwap)=-0.02.
            # Backup: paper_trade_runner_pre_pc1_gate.py
            if (args.asset == "BTC" and dec_c.decision == "trade"
                    and dec_c.side == "no"
                    and not math.isnan(_pc1_rsi)
                    and _pc1_rsi <= _PC1_Q10):
                _sk_pc1       = confirm.stoch_k if (confirm.stoch_k == confirm.stoch_k) else 0.0
                _vwap_str_pc1 = confirm.stretch_score if confirm.stretch_score is not None else 0
                _rvol_pc1     = _rvol_1h if not math.isnan(_rvol_1h) else 0.0
                _pc1_rescued = (
                    (confirm.no_score  is not None and confirm.no_score  >= 3)
                    or (confirm.vpin_score is not None and confirm.vpin_score == 1)
                    or (_vwap_str_pc1 <= -2)
                    or (_rvol_pc1 > 2.0 and _sk_pc1 > 60)
                )
                _rescue_reason = (
                    f"no_score={confirm.no_score}" if (confirm.no_score is not None and confirm.no_score >= 3)
                    else f"vpin={confirm.vpin_score}" if (confirm.vpin_score is not None and confirm.vpin_score == 1)
                    else f"vwap_stretch={_vwap_str_pc1}<=-2" if _vwap_str_pc1 <= -2
                    else f"rvol={_rvol_pc1:.2f}>2+stoch={_sk_pc1:.0f}>60"
                )
                if _pc1_rescued:
                    print(f"  [pc1_rsi_no_gate] RESCUE NO {c['ticker']} — "
                          f"pc1={_pc1_rsi:.3f}<=q10 BUT {_rescue_reason}")
                else:
                    print(f"  [pc1_rsi_no_gate] BLOCK NO {c['ticker']} — "
                          f"pc1={_pc1_rsi:.3f}<={_PC1_Q10:.2f} "
                          f"no_score={confirm.no_score} vpin={confirm.vpin_score} "
                          f"vwap_str={_vwap_str_pc1} rvol={_rvol_pc1:.2f} stoch={_sk_pc1:.0f}")
                    _log_block("pc1_rsi_no_gate")
                    continue

            # Counter-tape severity gate: hard block or dampen by severity zone
            if dec_c.decision == "trade":
                _sev = _counter_tape_severity(dec_c.side)
                if _sev >= 1.5:
                    print(f"  [counter_tape] BLOCK {dec_c.side.upper()} {c['ticker']} — severity={_sev:.2f} "
                          f"(chg_5m={_sharp_move_pct_5m*100:+.2f}% 10m={_sharp_move_pct_10m*100:+.2f}% 30m={_sharp_move_pct*100:+.2f}%)")
                    continue
                elif _sev >= 0.5:
                    _scale = max(0.25, 1.0 - (_sev - 0.5) * 0.75)
                    dec_c.bet_amount   = round(dec_c.bet_amount * _scale, 2)
                    dec_c.bet_fraction = dec_c.bet_fraction * _scale
                    print(f"  [counter_tape] DAMPEN {dec_c.side.upper()} {c['ticker']} — severity={_sev:.2f} "
                          f"kelly_scale={_scale:.2f} → bet=${dec_c.bet_amount:.2f}")

            if dec_c.decision == "trade":
                if best_trade_dec is None or dec_c.net_edge > best_trade_dec.net_edge:
                    best_trade_dec  = dec_c
                    best_trade_meta = meta_c

    # Select final decision
    if best_trade_dec is not None:
        # Enforce 10-minute same-direction cooldown per expiry to prevent clustering
        _best_expiry_key = _expiry_prefix(best_trade_meta["contract_ticker"])
        _last_same = _SIDE_COOLDOWN.get((_best_expiry_key, best_trade_dec.side))
        if _last_same is not None:
            _elapsed = (now_utc - _last_same).total_seconds()
            if _elapsed < 300:
                print(f"  [scan] Cooldown active — same-side {best_trade_dec.side.upper()} "
                      f"traded {_elapsed:.0f}s ago in expiry {_best_expiry_key} (cooldown=300s). Skipping.")
                best_trade_dec = None
        if best_trade_dec is None and best_no_trade_dec is not None:
            dec              = best_no_trade_dec
            chosen           = best_no_trade_meta
            p_market_source  = "real"
            print(f"  [scan] Cooldown blocked trade. Best no_trade: {chosen['contract_ticker']}  "
                  f"net_edge={dec.net_edge:+.4f}")
        elif best_trade_dec is None:
            print("  [scan] Cooldown blocked trade — no fallback no_trade available. Skipping.")
            return
        else:
            dec              = best_trade_dec
            chosen           = best_trade_meta
            p_market_source  = "real"
    else:
        if best_any_dec is not None:
            # Re-check streak gate before using fallback — the streak gate uses continue
            # to skip best_trade_dec, but best_any_dec was already set before the gate ran.
            _sk_fb = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
            if (best_any_dec.side == "yes" and _streak30 == "bearish" and _sk_fb <= 70) or \
               (best_any_dec.side == "no"  and _streak30 == "bullish" and 30 <= _sk_fb <= 60):
                print(f"  [scan] Streak gate blocked fallback {best_any_dec.side.upper()} "
                      f"({best_any_meta['contract_ticker'] if best_any_meta else ''}) — skipping.")
                return
            # Re-check counter-tape severity on fallback (hard block only; dampening doesn't apply to no_trade)
            _sev_fb = _counter_tape_severity(best_any_dec.side)
            if _sev_fb >= 1.5:
                print(f"  [scan] Counter-tape blocked fallback {best_any_dec.side.upper()} "
                      f"({best_any_meta['contract_ticker'] if best_any_meta else ''}) — severity={_sev_fb:.2f}. Skipping.")
                return
            # Re-check cooldown on fallback — best_any_dec bypasses the trade-path cooldown above.
            if best_any_dec.decision == "trade" and best_any_meta is not None:
                _fb_expiry = _expiry_prefix(best_any_meta["contract_ticker"])
                _fb_last   = _SIDE_COOLDOWN.get((_fb_expiry, best_any_dec.side))
                if _fb_last is not None:
                    _fb_elapsed = (now_utc - _fb_last).total_seconds()
                    if _fb_elapsed < 300:
                        print(f"  [scan] Cooldown blocked fallback {best_any_dec.side.upper()} "
                              f"{best_any_meta['contract_ticker']} — {_fb_elapsed:.0f}s ago (cooldown=300s). Skipping.")
                        return
            dec             = best_any_dec
            chosen          = best_any_meta
            p_market_source = "real"
            print(f"  [scan] No trade passes gates. Best seen: {chosen['contract_ticker']}  "
                  f"net_edge={dec.net_edge:+.4f}")
        else:
            print("  [scan] No real contracts available (auth failed or empty ladder) — skipping.")
            # Still run position monitor even when no new contracts are available.
            if auth is not None:
                try:
                    import position_monitor as _pm
                    _pm.evaluate_open_positions(
                        auth=auth,
                        asset=args.asset,
                        spot=spot,
                        vol_multi=vol.vol_multi,
                        composite_trend=float(_comp_trend),
                        composite_rev=float(_comp_rev),
                        df_1m=live_1m,
                        df_1h=df_confirm,
                        df_4h=_df_4h_comp,
                        df_15m=locals().get("_df_15m_comp"),
                        now_utc=now_utc,
                    )
                except Exception as _pm_exc:
                    print(f"  [pos_monitor] Error: {_pm_exc}")
            return

    strike          = chosen["strike"]
    p_market        = chosen["p_market"]
    prob            = chosen["prob"]
    contract_ticker = chosen["contract_ticker"]
    close_ts        = chosen["close_ts"]
    effective_offset = strike / spot - 1
    p_yes_adj = max(0.03, min(0.97, prob.p_yes + funding_delta))
    pricing = evaluate_edge(p_yes_adj, p_market)

    vol_eff  = chosen.get("vol_eff", vol.vol_multi)
    vol_impl = implied_vol_from_price(p_market, spot, strike, minutes_to_expiry(close_ts))
    vol_ratio = round(vol.vol_multi / vol_impl, 4) if vol_impl > 0 else ""
    spread    = round(chosen.get("ask", 0) - chosen.get("bid", 0), 4) if chosen.get("ask") else ""

    # LGBM shadow model — compute p_gbdt for logging (no trade logic effect).
    # BTC uses dedicated loader; ETH/SOL share _load_asset_lgbm.
    # Retrain: python3 train_btc_lgbm.py  (BTC)
    #          python3 train_eth_sol_lgbm.py  (ETH, SOL)
    _p_gbdt_log = ""
    _lgbm_feats_common = {
        "offset_pct":         effective_offset,
        "p_market":           p_market,
        "tau_minutes":        minutes_to_expiry(close_ts),
        "side_enc":           1.0 if dec.side == "yes" else 0.0,
        "composite_p_up":     _comp_p_up if _comp_p_up is not None else float("nan"),
        "composite_trend":    float(_comp_trend) if _comp_trend is not None else float("nan"),
        "composite_rev":      float(_comp_rev) if _comp_rev is not None else float("nan"),
        "ema_stack_bias":     float(confirm.ema_stack_bias) if confirm.ema_stack_bias is not None else float("nan"),
        "ema_stretch_score":  float(confirm.ema_stretch_score) if confirm.ema_stretch_score is not None else float("nan"),
        "stoch_k":            float(confirm.stoch_k) if confirm.stoch_k is not None else float("nan"),
        "vwap_stretch_score": float(confirm.stretch_score) if confirm.stretch_score is not None else float("nan"),
        "vwap_distance_pct":  float(confirm.vwap_distance_pct) if hasattr(confirm, "vwap_distance_pct") and confirm.vwap_distance_pct is not None else float("nan"),
        "vol_score":          float(confirm.vol_score) if confirm.vol_score is not None else float("nan"),
        "vpin_score":         float(confirm.vpin_score) if confirm.vpin_score is not None else float("nan"),
        "confirmation_score": float(confirm.confirmation_score) if confirm.confirmation_score is not None else float("nan"),
        "funding_bias":       float(confirm.funding_bias) if confirm.funding_bias is not None else float("nan"),
        "chg_30m":            _sharp_move_pct * 100,
        "chg_10m":            _sharp_move_pct_10m * 100,
        "chg_5m":             _sharp_move_pct_5m * 100,
        "bp_5m":              _bp_5m if _bp_5m is not None else float("nan"),
        "body_15m":           _body_15m if _body_15m is not None else float("nan"),
        "dir_15m":            float(_dir_15m) if _dir_15m is not None else float("nan"),
        "vol_eff":            vol_eff,
    }
    if args.asset == "BTC":
        # p_gbdt was already computed per-contract inside the scan loop; reuse it.
        _p_gbdt_chosen = chosen.get("p_gbdt")
        if _p_gbdt_chosen is not None:
            _p_gbdt_log = round(_p_gbdt_chosen, 4)
            print(f"  [btc_lgbm] p_gbdt={_p_gbdt_chosen:.3f}  p_yes_model={prob.p_yes:.3f}  "
                  f"Δ={_p_gbdt_chosen - prob.p_yes:+.3f}")
        _lgbm_pipe = None
        _infer_fn  = None
        _lgbm_tag  = "btc_lgbm"
    elif args.asset in ("ETH", "SOL"):
        _lgbm_pipe = _load_asset_lgbm(args.asset)
        _infer_fn  = lambda pipe, feats: _infer_asset_lgbm(pipe, feats, args.asset)
        _lgbm_tag  = f"{args.asset.lower()}_lgbm"
    else:
        _lgbm_pipe = None
        _infer_fn  = None
        _lgbm_tag  = ""
    if _lgbm_pipe is not None and _infer_fn is not None:
        _p_gbdt = _infer_fn(_lgbm_pipe, _lgbm_feats_common)
        if _p_gbdt is not None:
            _p_gbdt_log = round(_p_gbdt, 4)
            print(f"  [{_lgbm_tag}] p_gbdt={_p_gbdt:.3f}  p_yes_model={prob.p_yes:.3f}  "
                  f"Δ={_p_gbdt - prob.p_yes:+.3f}")

    # Build row
    row = {
        "logged_at":          now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_time":      ts.strftime("%Y-%m-%d %H:%M"),
        "contract_ticker":    contract_ticker,
        "close_ts":           close_ts,
        "spot":               round(spot, 2),
        "strike":             round(strike, 2),
        "offset_pct":         round(effective_offset * 100, 4),
        "p_market":           round(p_market, 6),
        "p_market_source":    p_market_source,
        "p_yes_model":        round(chosen.get("p_model_comp", prob.p_yes), 6),
        "z_score":            round(prob.z_score, 4),
        "vol_60m":            round(vol.vol_60m, 8),
        "vol_60m_model":      round(vol.vol_multi, 8),
        "vol_implied_kalshi": round(vol_impl, 8) if vol_impl == vol_impl else "",
        "vol_ratio":          vol_ratio,
        "spread":             spread,
        "vol_eff":            round(vol_eff, 8),
        "structure_bias":     struct.structure_bias,
        "confirmation_bias":  confirm.confirmation_bias,
        "confirmation_score": confirm.confirmation_score,
        "no_score":           confirm.no_score,
        "obi_score":          confirm.obi_score,
        "obi_raw":            round(obi.obi, 4) if obi.obi == obi.obi else "",
        "obi_exchanges":      obi.exchanges_used,
        "vpin_score":         confirm.vpin_score,
        "vpin_raw":           round(confirm.vpin_raw, 4) if confirm.vpin_raw == confirm.vpin_raw else "",
        "funding_bias":       confirm.funding_bias,
        "avg_funding_rate":   round(confirm.avg_funding_rate, 8),
        "vol_score":          confirm.vol_score,
        "cmf_raw":            round(confirm.cmf_raw, 4) if confirm.cmf_raw == confirm.cmf_raw else "",
        "cmf_score":          confirm.cmf_score,
        "vwap_score":         confirm.vwap_score,
        "vwap_signal":        confirm.vwap_signal,
        "vwap_total":         confirm.vwap_total,
        "vwap_stretch_score": confirm.stretch_score,
        "vwap_distance_pct":  round(confirm.distance_pct * 100, 4) if confirm.distance_pct == confirm.distance_pct else "",
        "bearish_rejection":  confirm.bearish_rejection,
        "bullish_rejection":  confirm.bullish_rejection,
        "ema_stretch_score":      confirm.ema_stretch_score,
        "stoch_bias":             confirm.stoch_bias,
        "stoch_k":                round(confirm.stoch_k, 2) if confirm.stoch_k == confirm.stoch_k else "",
        "stoch_k_4h":             round(_sk4h_bounce, 2) if not math.isnan(_sk4h_bounce) else "",
        "stoch_d":                round(confirm.stoch_d, 2) if confirm.stoch_d == confirm.stoch_d else "",
        "stoch_crossover_active": confirm.stoch_crossover_active,
        "ema_stack_bias":         confirm.ema_stack_bias,
        "ema_alignment":          confirm.ema_alignment,
        "z_shift":            round(prob.z_shift, 6),
        "direction_strength": round(prob.direction_strength, 4),
        "raw_edge":           round(dec.raw_edge, 6),
        "net_edge":           round(dec.net_edge, 6),
        "decision":           dec.decision,
        "side":               dec.side,
        "neutral_gate":       struct.structure_bias == 0 and any("Gate 1 PASSED (neutral)" in r for r in dec.reasons),
        "pure_edge_gate":     any("Gate P PASSED" in r for r in dec.reasons),
        "contracts_scanned":  contracts_scanned,
        "tau_minutes":        round(minutes_to_expiry(close_ts), 2),
        "gate_blocked":       next((r.split(":")[0] for r in dec.reasons if "FAILED" in r), "") if dec.decision == "no_trade" else "",
        "kelly_fraction":     round(dec.kelly_fraction, 6),
        "bet_fraction":       round(dec.bet_fraction, 6),
        "bet_amount":         round(dec.bet_amount, 2),
        "bankroll":           round(args.bankroll, 2),
        "composite_trend":    _comp_trend,
        "composite_rev":      _comp_rev,
        "composite_p_up":     round(_comp_p_up, 4),
        "p_up_v2":            round(chosen.get("p_up_v2"), 4) if chosen.get("p_up_v2") is not None else "",
        "chg_30m":            round(_sharp_move_pct * 100, 4),
        "chg_10m":            round(_sharp_move_pct_10m * 100, 4),
        "chg_5m":             round(_sharp_move_pct_5m * 100, 4),
        "bp_5m":              round(_bp_5m, 4) if _bp_5m is not None else "",
        "bp_1h":              round(_bp_1h_mtf, 4) if not math.isnan(_bp_1h_mtf) else "",
        "chg_1h":             round(_chg_1h_mtf, 4),
        "chg_2h":             round(_chg_2h_mtf, 4),
        "chg_3h":             round(_chg_3h_mtf, 4),
        "body_15m":           round(_body_15m, 4) if _body_15m is not None else "",
        "dir_15m":            _dir_15m if _dir_15m is not None else "",
        "p_gbdt":             _p_gbdt_log,
        "sharp_move_active":  _sharp_move_active,
        "smc_4h":             _smc.bos_4h if _smc else "",
        "smc_1h":             _smc.bos_1h if _smc else "",
        "choch_1h":           _smc.choch_1h if _smc else "",
        "choch_4h":           _smc.choch_4h if _smc else "",
        "supply_pct":         round(_smc.nearest_supply_pct, 4) if (_smc and _smc.nearest_supply_pct is not None) else "",
        "demand_pct":         round(_smc.nearest_demand_pct, 4) if (_smc and _smc.nearest_demand_pct is not None) else "",
        "in_supply_zone":     _smc.in_supply_zone if _smc else "",
        "in_demand_zone":     _smc.in_demand_zone if _smc else "",
        "stoch_flipped":      "",
        "squeeze_1h":         confirm.squeeze_1h,
        "adx_1h":             round(confirm.adx_1h, 2) if confirm.adx_1h == confirm.adx_1h else "",
        "rvol_1h":            _rvol_1h if _rvol_1h == _rvol_1h else "",
        "pm_drift_5m":        round(chosen.get("pm_drift_5m", float("nan")), 5) if chosen.get("pm_drift_5m") == chosen.get("pm_drift_5m") else "",
        "hour_utc":           now_utc.hour,
        "liq_score":          _liq_signal.liq_score   if _liq_signal else "",
        "liq_bias":           round(_liq_signal.liq_bias, 4)    if _liq_signal else "",
        "ls_long_pct":        round(_liq_signal.ls_long_pct, 2) if _liq_signal else "",
        "oi_chg_pct":         round(_liq_signal.oi_chg_pct, 4)  if _liq_signal else "",
        "arima_forecast_1h":  round(_arima_forecast_btc, 7) if not math.isnan(_arima_forecast_btc) else "",
        "markov_regime_daily":  _markov_regime  or "",
        "markov_regime_7state": _markov_7state  or "",
        "ob_imbalance":       _gate_signals.get("ob_imbalance", ""),
        "ob_path_ask_usd":    _gate_signals.get("ob_path_ask_usd", ""),
        "ob_path_bid_usd":    _gate_signals.get("ob_path_bid_usd", ""),
        "ob_ask_frac":        _gate_signals.get("ob_ask_frac", ""),
        "ob_bid_wall_pct":    _gate_signals.get("ob_bid_wall_pct", ""),
        "ob_ask_wall_pct":    _gate_signals.get("ob_ask_wall_pct", ""),
        "hmm_vol_state":      (_hmm_vol_probs[0]           if _hmm_vol_probs is not None
                               else _hmm_vol_probs_eth_live[0] if _hmm_vol_probs_eth_live is not None
                               else _hmm_vol_probs_sol_live[0] if _hmm_vol_probs_sol_live is not None
                               else ""),
        "hmm_r1_prob":        (_hmm_vol_probs[1]           if _hmm_vol_probs is not None
                               else _hmm_vol_probs_eth_live[1] if _hmm_vol_probs_eth_live is not None
                               else _hmm_vol_probs_sol_live[1] if _hmm_vol_probs_sol_live is not None
                               else ""),
        "hmm_vol_k10":        _hmm_vol_probs[2] if _hmm_vol_probs is not None else "",
        "hmm_time_in_state":  (_hmm_vol_probs[3]           if _hmm_vol_probs is not None
                               else _hmm_vol_probs_eth_live[2] if _hmm_vol_probs_eth_live is not None
                               else _hmm_vol_probs_sol_live[2] if _hmm_vol_probs_sol_live is not None
                               else ""),
        "ou_z_score":         _ou_z_score      if not math.isnan(_ou_z_score)      else "",
        "ou_halflife_min":    _ou_halflife_min  if not math.isnan(_ou_halflife_min) else "",
        "ou_tau_drift":       round(
            # E[log_return | τ] = (mu_log - log(spot)) × (1 - exp(-θ × τ_hours))
            (_ou_mu_val - math.log(spot)) * (1.0 - math.exp(-_ou_theta * minutes_to_expiry(close_ts) / 60.0)), 6
        ) if (not math.isnan(_ou_theta) and not math.isnan(_ou_mu_val) and spot > 0 and _ou_theta > 0) else "",
        "hs_pattern_type":    _hs_pat_type,
        "hs_bars_since_break": _hs_bars_since,
        "hs_r2":              _hs_r2,
        "hs_neck_slope":      _hs_neck_slope,
        "hs_head_height":     _hs_head_height,
        "hs_head_width":      _hs_head_width,
        "v_hawk":             _v_hawk      if not math.isnan(_v_hawk)      else "",
        "hawk_vol_regime":    _hawk_regime if _hawk_regime                 else "",
        "pc1_rsi":            _pc1_rsi     if not math.isnan(_pc1_rsi)     else "",
        "resolved_yes":       "",
        "would_win":          "",
        "would_pnl":          "",
    }

    # Print summary
    print(f"\n  Decision: {dec.decision.upper()}  side={dec.side.upper()}")
    print(f"  p_yes={prob.p_yes:.4f}  p_market={p_market:.4f} ({p_market_source})")
    print(f"  net_edge={dec.net_edge:+.4f}  bet_amount=${dec.bet_amount:,.2f}")
    if contract_ticker:
        print(f"  Contract: {contract_ticker}  close_ts={close_ts}")

    # --- Logging and live order placement ---
    # Dual mode: paper and live must always match.
    #   - no_trade: log to paper immediately (no live action needed).
    #   - trade: validate all live checks first, then mark session state + log both CSVs.
    #            If any live check fails, skip session state update too — contract stays
    #            eligible for future cycles.
    # Paper-only mode: always log to paper and update session state.
    # Pure live mode: never log to paper (separate paper process handles it).
    _is_dual = getattr(args, 'dual', False)

    _live_order_placed = False  # True only when a real Kalshi order is confirmed placed

    if _is_live_or_dual and dec.decision == "trade" and auth is not None:
        _live_csv = live_trading.get_live_csv_path(args.asset)
        _live_limit_ok = live_trading.check_daily_loss_limit(args.daily_loss_limit, _live_csv)

        if _vol_skip_live:
            print("  [live] Vol-filter hour — skipping live order only.")
        elif not _live_limit_ok:
            print("  [live] Daily loss limit reached — skipping live order only.")
        else:
            bid_c = chosen.get("bid", p_market - 0.01)
            ask_c = chosen.get("ask", p_market + 0.01)
            yes_price_cents, count = live_trading.compute_order_params(
                side=dec.side,
                bet_amount=dec.bet_amount,
                bid=bid_c,
                ask=ask_c,
                max_contracts=args.max_contracts,
            )
            if count == 0:
                print(f"  [live] Bet amount ${dec.bet_amount:.2f} < single contract cost — skipping order")
            else:
                # Confirm balance before placing
                balance = live_trading.get_balance(auth)
                _balance_ok = True
                if balance is not None:
                    order_cost = count * (yes_price_cents if dec.side == "yes" else (100 - yes_price_cents)) / 100.0
                    print(f"  [live] Balance: ${balance:.2f}  order cost ≈ ${order_cost:.2f}")
                    if order_cost > balance:
                        print(f"  [live] Insufficient balance — skipping order")
                        _balance_ok = False

                if _balance_ok:
                    order_result = live_trading.place_order(
                        auth=auth,
                        ticker=contract_ticker,
                        side=dec.side,
                        count=count,
                        yes_price=yes_price_cents,
                    )
                    live_trading.log_live_trade(
                        row=row,
                        order_result=order_result,
                        yes_price_cents=yes_price_cents,
                        count=count,
                        side=dec.side,
                        asset=args.asset,
                        csv_path=_live_csv,
                    )
                    # Update session state only after live order is confirmed placed
                    _SESSION_TRADED[contract_ticker] = dec.net_edge
                    _SIDE_COOLDOWN[(_expiry_prefix(contract_ticker), dec.side)] = now_utc
                    _live_order_placed = True

    # --- Ghost exit monitor (no orders placed — data collection only) ---
    if auth is not None:
        try:
            import position_monitor as _pm
            _pm.evaluate_open_positions(
                auth=auth,
                asset=args.asset,
                spot=spot,
                vol_multi=vol.vol_multi,
                composite_trend=float(_comp_trend),
                composite_rev=float(_comp_rev),
                df_1m=live_1m,
                df_1h=df_confirm,
                df_4h=_df_4h_comp,
                df_15m=locals().get("_df_15m_comp"),
                now_utc=now_utc,
            )
        except Exception as _pm_exc:
            print(f"  [pos_monitor] Error: {_pm_exc}")

    # Paper CSV logging: dual mode logs both trades and no_trades; pure paper mode logs everything
    if _is_dual:
        if dec.decision == "trade":
            if _vol_skip_live:
                # Skip-hour: paper continues for data collection; advance paper session state
                # independently so paper doesn't re-trade the same ticker in subsequent scans.
                _SESSION_TRADED[contract_ticker] = dec.net_edge
                _SIDE_COOLDOWN[(_expiry_prefix(contract_ticker), dec.side)] = now_utc
            elif _live_order_placed:
                # Live placed: session state already advanced at line 1831-1832; paper mirrors.
                pass
            else:
                # Live couldn't place (count=0, balance, API error) outside skip-hours.
                # Paper mirrors live: downgrade to no_trade so logs stay in sync.
                row["decision"] = "no_trade"
                print("  [dual] Live order not placed — paper logging as no_trade to stay in sync.")
        csv_path = get_csv_path(args.asset)
        ensure_csv_exists(csv_path)
        append_row(row, csv_path)
    elif not args.live:
        # Pure paper mode: log everything to shadow CSV (keeps paper trades separate from live/dual)
        if dec.decision == "trade":
            _SESSION_TRADED[contract_ticker] = dec.net_edge
            _SIDE_COOLDOWN[(_expiry_prefix(contract_ticker), dec.side)] = now_utc
        csv_path = get_csv_path(args.asset, shadow=True)
        ensure_csv_exists(csv_path)
        append_row(row, csv_path)


if __name__ == "__main__":
    import argparse as _ap
    import fcntl as _fcntl

    _loop_parser = _ap.ArgumentParser(add_help=False)
    _loop_parser.add_argument("--asset", type=str, default="BTC")
    _loop_parser.add_argument("--live", action="store_true")
    _loop_parser.add_argument("--dual", action="store_true")
    _loop_args, _ = _loop_parser.parse_known_args()
    _loop_asset = _loop_args.asset.upper()
    _loop_live  = _loop_args.live
    _loop_dual  = getattr(_loop_args, 'dual', False)
    _loop_is_live_mode = _loop_live or _loop_dual  # dual places real orders like live

    # Enforce single-process-per-asset via lockfile.
    # Dual mode uses "live_trade" prefix — it places real orders and is the authoritative process.
    # A second launch for the same asset exits immediately with a clear error.
    _lock_prefix = "live_trade" if _loop_is_live_mode else "paper_trade"
    _lock_path = Path(__file__).parent / f".{_lock_prefix}_{_loop_asset}.lock"
    _lock_fd = open(_lock_path, "w")
    try:
        _fcntl.flock(_lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError:
        _mode_str = "dual" if _loop_dual else ("live" if _loop_live else "paper")
        print(f"ERROR: Another {_loop_asset} {_mode_str} trade process is already running. Exiting.")
        sys.exit(1)

    loop_count = 0
    _last_hour = datetime.now(timezone.utc).hour
    while True:
        # Reset session-traded set at the top of each new clock hour
        _current_hour = datetime.now(timezone.utc).hour
        if _current_hour != _last_hour:
            _SESSION_TRADED.clear()
            _SESSION_SEEDED = False  # allow CSV re-seed so still-open contracts stay blocked
            print(f"  [session] New hour — already_traded reset.")
            _last_hour = _current_hour
        # Data update: dual/paper runner always updates. Pure live runner defers to paper runner.
        _should_update = not _loop_live or _loop_dual or loop_count % 30 == 0
        if _loop_live and not _loop_dual and loop_count % 30 == 0:
            # Pure live runner: check age of most recent 1m parquet before updating
            from live_signal import ASSET_CONFIG as _AC
            _sym = _AC.get(_loop_asset, _AC["BTC"])["binance_symbol"]
            _parquets = sorted(
                (Path(__file__).parent / "data").glob(f"*{_sym}_1m_*.parquet"),
                key=lambda p: p.stat().st_mtime,
            )
            _parquets = [p for p in _parquets if ".ckpt." not in p.name]
            if _parquets:
                _age = time.time() - _parquets[-1].stat().st_mtime
                _should_update = _age > 300  # stale if paper runner hasn't updated in 5 min
                if not _should_update:
                    print(f"  [data] Skipping update — paper runner data is fresh ({_age:.0f}s old)")
        if _should_update:
            print(f"  [data] Updating OHLCV parquet files ({_loop_asset})...")
            try:
                update_data.main(asset=_loop_asset)
            except Exception as e:
                print(f"  [data] Update failed (will retry next cycle): {e}")
        ensure_csv_exists(get_csv_path(_loop_asset, shadow=not _loop_is_live_mode))
        if loop_count % 5 == 0:
            outcome_checker.main(get_csv_path(_loop_asset, shadow=not _loop_is_live_mode))
            if _loop_is_live_mode:
                _live_auth = load_auth()
                if _live_auth:
                    gate_audit_logger.fill_outcomes(_live_auth)
                    live_trading.settle_live_trades(_live_auth, live_trading.get_live_csv_path(_loop_asset))
                    try:
                        import scan_archive as _sa
                        _sa.fill_scan_outcomes(asset=_loop_asset, auth=_live_auth)
                    except Exception:
                        pass
            else:
                gate_audit_logger.fill_outcomes(None)
                try:
                    import scan_archive as _sa
                    _sa.fill_scan_outcomes(asset=_loop_asset, auth=None)
                except Exception:
                    pass
        main()
        loop_count += 1
        time.sleep(60)
