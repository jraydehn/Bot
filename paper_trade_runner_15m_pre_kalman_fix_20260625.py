#!/usr/bin/env python3
"""
paper_trade_runner_15m.py

Paper trading runner for 15-minute directional contracts (BTC, ETH, SOL).
  "Price up in next 15 mins?" — YES = asset ends above floor_strike at expiry.

Signals used:
  bp_5m       — buying pressure: (close-low)/(high-low) on last completed 5m bar
  body_15m    — body ratio: |close-open|/(high-low) on last completed 15m bar
  dir_15m     — +1 bullish / -1 bearish on last completed 15m bar
  stoch_k_5m  — 14-period stochastic %K on 5m bars
  stoch_k_15m — 14-period stochastic %K on 15m bars
  chg_1m      — % change on last completed 1m bar
  chg_5m      — % change over last 5m
  chg_15m     — % change over last 15m
  vwap_dist   — (spot - vwap_20bar_5m) / vwap * 100
  vol_ratio   — last completed 5m bar volume vs 20-bar avg
  ema_bias    — sign(ema_5 - ema_20) on 1m closes

Model: log-normal base P(S_T > floor_strike) + directional delta from signals.
Edge threshold: 0.04. Kelly multiplier: 0.30 (conservative for new model).

Usage:
    export KALSHI_KEY_ID=your-key-id
    export KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
    python3 paper_trade_runner_15m.py --asset BTC --bankroll 1000 --loop
    python3 paper_trade_runner_15m.py --asset ETH --bankroll 1000 --loop
    python3 paper_trade_runner_15m.py --asset SOL --bankroll 1000 --loop
"""

import argparse
import csv
import fcntl as _fcntl
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from live_signal import load_auth, kalshi_get, fetch_live_spot, fetch_recent_candles
from kelly_sizing import compute_kelly_size
from market_data import compute_realized_volatility
from probability_engine import implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT_BY_ASSET
from collections import deque as _deque

# ── HMM multi-timeframe regime model ─────────────────────────────────────────
# Trained on 2,290 BTC scans (5m+15m+1h features). BIC-optimal 8 states.
# State 0: stoch_5m=66 vs stoch_1h=11 divergence → WR=22%, block trades
# State 7: all-TF mid-high regime → +$116/trade, elevated Kelly cap
_HMM_PKL = Path(__file__).parent / "hmm_btc_multitf.pkl"
_hmm_model, _hmm_scaler, _hmm_feat_cols = None, None, None
_hmm_obs_buf: _deque = _deque(maxlen=20)   # rolling observation buffer

try:
    with open(_HMM_PKL, "rb") as _f:
        _hmm_pkg = pickle.load(_f)
    _hmm_model    = _hmm_pkg["model"]
    _hmm_scaler   = _hmm_pkg["scaler"]
    _hmm_feat_cols= _hmm_pkg["feat_cols"]
    print(f"  [hmm] Loaded {_HMM_PKL.name}  "
          f"({_hmm_pkg['n_states']} states, {len(_hmm_feat_cols)} features)")
except Exception as _hmm_e:
    print(f"  [hmm] WARNING: could not load {_HMM_PKL.name}: {_hmm_e}")

# Vol-regime HMM (BTC only): 3-state Gaussian HMM on 15m log returns.
# Rank 0=low-vol-bearish, 1=low-vol-bullish, 2=high-vol. Shadow-logged only —
# no gate applied until 4-6 weeks of live data confirm the Rank-2 edge gap.
_VOL_HMM_PKL = Path(__file__).parent / "models" / "hmm_vol_regime_btc_15m.pkl"
_vol_hmm_model   = None
_vol_hmm_order   = None   # state indices sorted by ascending sigma
_vol_hmm_rank_of: "dict | None" = None

try:
    with open(_VOL_HMM_PKL, "rb") as _vf:
        _vol_hmm_pkg  = pickle.load(_vf)
    _vol_hmm_model    = _vol_hmm_pkg["model"]
    _n_vol_states     = _vol_hmm_pkg["n_states"]
    _vol_hmm_order    = sorted(range(_n_vol_states),
                               key=lambda s: float(
                                   np.sqrt(_vol_hmm_model.covars_[s, 0, 0])))
    _vol_hmm_rank_of  = {s: i for i, s in enumerate(_vol_hmm_order)}
    print(f"  [vol_hmm] Loaded {_VOL_HMM_PKL.name}  ({_n_vol_states} states)")
except Exception as _ve:
    print(f"  [vol_hmm] WARNING: could not load vol-regime HMM: {_ve}")


def _vol_hmm_state(live_1m: "pd.DataFrame") -> "int | None":
    """Decode current vol-regime HMM rank (0=low-vol … 2=high-vol) from live 1m data."""
    if _vol_hmm_model is None or _vol_hmm_rank_of is None:
        return None
    try:
        c15 = live_1m["close"].resample("15min").last().dropna()
        if len(c15) < 22:
            return None
        lr  = np.log(c15 / c15.shift(1)).dropna().values[-20:]
        raw = int(_vol_hmm_model.predict(lr.reshape(-1, 1))[-1])
        return _vol_hmm_rank_of[raw]
    except Exception:
        return None


def _hmm_predict_state(sig: dict) -> int:
    """Build feature vector from sig, append to buffer, predict current HMM state.
    Returns -1 if model unavailable or buffer too short."""
    if _hmm_model is None or _hmm_scaler is None:
        return -1
    vec = []
    for col in _hmm_feat_cols:
        val = sig.get(col)
        try:
            vec.append(float(val) if val is not None and val == val else float("nan"))
        except (TypeError, ValueError):
            vec.append(float("nan"))
    if sum(1 for v in vec if v == v) < len(vec) * 0.75:
        return -1   # too many NaNs
    # Fill NaNs with 0 (scaler was fit on median-filled data)
    vec_arr = np.array([0.0 if v != v else v for v in vec], dtype=float).reshape(1, -1)
    try:
        vec_scaled = _hmm_scaler.transform(vec_arr)[0]
    except Exception:
        return -1
    _hmm_obs_buf.append(vec_scaled)
    if len(_hmm_obs_buf) < 3:
        return -1
    obs_seq = np.array(list(_hmm_obs_buf))
    try:
        states = _hmm_model.predict(obs_seq, lengths=[len(obs_seq)])
        return int(states[-1])
    except Exception:
        return -1
import coinalyze_liq
import coinglass_data
import live_trading

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASSET_CONFIG = {
    "BTC": {
        "series_ticker":      "KXBTC15M",
        "binance_symbol":     "BTCUSDT",
        "csv_name":           "paper_trades_btc15m.csv",
        "vol_fallback":       0.50,   # annual vol fallback when realized is unavailable
    },
    "ETH": {
        "series_ticker":      "KXETH15M",
        "binance_symbol":     "ETHUSDT",
        "csv_name":           "paper_trades_eth15m.csv",
        "vol_fallback":       0.65,
    },
    "SOL": {
        "series_ticker":      "KXSOL15M",
        "binance_symbol":     "SOLUSDT",
        "csv_name":           "paper_trades_sol15m.csv",
        "vol_fallback":       0.90,
    },
}

MIN_TAU_MIN      = 2.0    # skip contract if fewer than 2 minutes remain
MAX_TAU_MIN      = 14.0   # skip contract if more than 14 minutes remain (not opened yet)
EDGE_THRESHOLD   = 0.04   # minimum model-vs-market edge to bet
KELLY_MULT       = 0.08   # 8% of full-Kelly — allows signal variation (was 0.30, always hit cap)
MAX_BET_FRAC     = 0.06   # hard cap at 6% of bankroll per trade (was 0.03, 98% of bets hit cap)
MAX_BET_FRAC_ST7 = 0.09   # elevated cap for HMM State 7 (high-payout NO regime, +$2207 historical)
P_MARKET_VOL_MIN = 0.12   # block YES when p_market < this (deep OTM)
P_MARKET_VOL_MAX = 0.88   # block NO when p_market > this (deep OTM)
CANDLES_NEEDED   = 1500   # 1m candles (25h — need 20+ 1h bars for donchian/stoch_k_1h)
DEFAULT_BANKROLL = 1000.0

MINS_PER_YEAR    = 525600.0

# LightGBM models — loaded once per asset at startup, used in compute_p_model_15m
_LGBM_MODELS: dict = {}

# 1h Markov regime cache — refreshed once per UTC hour via yfinance
_MARKOV_1H_CACHE: dict = {"hour": None, "regime": None}
# Multi-asset Markov regime cache (ETH/SOL) — refreshed once per UTC hour
_MARKOV_ETH_SOL_CACHE: dict = {"hour": None, "regimes": {}}


def _get_btc_markov_regime_1h() -> "str | None":
    """Return the current 1h Markov regime (Bull/Bear/Sideways) for BTC.

    Uses a 20-bar rolling return on 1h yfinance data with ±0.8% threshold.
    Result cached for the current UTC hour; re-fetches at each new hour.
    Returns None on fetch failure.
    """
    global _MARKOV_1H_CACHE
    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if _MARKOV_1H_CACHE["hour"] == now_hour:
        return _MARKOV_1H_CACHE["regime"]
    try:
        import yfinance as _yf
        _end   = pd.Timestamp.now("UTC")
        _start = _end - pd.DateOffset(days=4)   # 4 days → ~96 1h bars (well over 20 needed)
        _df = _yf.download("BTC-USD", start=_start.strftime("%Y-%m-%d"),
                           end=(_end + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                           interval="1h", progress=False, auto_adjust=True)
        if isinstance(_df.columns, pd.MultiIndex):
            _df.columns = _df.columns.get_level_values(0)
        _close = _df["Close"].dropna()
        if len(_close) < 21:
            return None
        _rr = float(_close.pct_change(20).iloc[-1])
        _regime = "Bull" if _rr > 0.008 else "Bear" if _rr < -0.008 else "Sideways"
        _MARKOV_1H_CACHE["hour"]   = now_hour
        _MARKOV_1H_CACHE["regime"] = _regime
        return _regime
    except Exception as _e:
        print(f"  [markov_1h] fetch error: {_e}")
        return None


def _get_markov_regimes_yf(asset: str) -> dict:
    """Return Markov regime dict for ETH or SOL, cached per UTC hour.

    ETH: {"1d": regime_str}
    SOL: {"6h": ..., "4h": ..., "1h": ...}

    Thresholds (validated against paper-trade backtest):
      ETH daily  20-bar ±3.0%
      SOL 6h     20-bar ±3.0%
      SOL 4h     20-bar ±2.5%
      SOL 1h     20-bar ±1.5%
    """
    global _MARKOV_ETH_SOL_CACHE
    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if (_MARKOV_ETH_SOL_CACHE["hour"] == now_hour
            and asset in _MARKOV_ETH_SOL_CACHE["regimes"]):
        return _MARKOV_ETH_SOL_CACHE["regimes"][asset]
    try:
        import yfinance as _yf
        _ticker = "ETH-USD" if asset == "ETH" else "SOL-USD"
        _end    = pd.Timestamp.now("UTC")
        _start  = _end - pd.DateOffset(days=120)
        _raw = _yf.download(
            _ticker,
            start=_start.strftime("%Y-%m-%d"),
            end=(_end + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            interval="1h", progress=False, auto_adjust=True,
        )
        if isinstance(_raw.columns, pd.MultiIndex):
            _raw.columns = _raw.columns.get_level_values(0)
        _raw.index = pd.to_datetime(_raw.index, utc=True)
        _c1h = _raw["Close"].dropna()

        def _regime(close, window, thr):
            rr = close.pct_change(window)
            if len(rr.dropna()) == 0 or pd.isna(rr.iloc[-1]):
                return None
            v = float(rr.iloc[-1])
            return "Bull" if v > thr else "Bear" if v < -thr else "Sideways"

        result: dict = {}
        if asset == "ETH":
            _c1d = _c1h.resample("1d").last().dropna()
            result["1d"] = _regime(_c1d, 20, 0.030)
        else:  # SOL
            _c6h = _c1h.resample("6h").last().dropna()
            _c4h = _c1h.resample("4h").last().dropna()
            result["6h"] = _regime(_c6h, 20, 0.030)
            result["4h"] = _regime(_c4h, 20, 0.025)
            result["1h"] = _regime(_c1h, 20, 0.015)

        if _MARKOV_ETH_SOL_CACHE["hour"] != now_hour:
            _MARKOV_ETH_SOL_CACHE["hour"]    = now_hour
            _MARKOV_ETH_SOL_CACHE["regimes"] = {}
        _MARKOV_ETH_SOL_CACHE["regimes"][asset] = result
        return result
    except Exception as _e:
        print(f"  [markov_{asset.lower()}] fetch error: {_e}")
        return {}


def _load_15m_lgbm(asset: str) -> object:
    path = Path(__file__).parent / "models" / f"lgbm_15m_{asset.lower()}.pkl"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"  [{asset.lower()}_15m_lgbm] Loaded ({model.n_features_in_} features)")
        return model
    except Exception as e:
        print(f"  [{asset.lower()}_15m_lgbm] Failed to load: {e}")
        return None


CSV_COLUMNS = [
    "logged_at", "decision_time", "asset", "contract_ticker", "close_time",
    "spot", "floor_strike", "offset_pct", "tau_minutes", "spread",
    "p_market", "p_model_15m", "raw_edge", "side", "decision",
    # per-contract moneyness
    "z_score",
    # 5m signals
    "bp_5m", "body_5m", "dir_5m", "vol_ratio", "vol_ratio_5m",
    # 15m signals
    "body_15m", "bp_15m", "dir_15m",
    "upper_wick_15m", "lower_wick_15m",
    "atr_ratio_15m", "range_ratio_15m", "consec_dir_15m",
    "stoch_k_5m", "stoch_k_15m",
    # price changes
    "chg_1m", "chg_5m", "chg_15m",
    # vwap / ema
    "vwap_dist", "ema_bias", "ema_bias_1h", "ema20_dist_1h", "ema50_dist_1h",
    "nearest_res_dist_pct",
    # composite / vol
    "composite_p_up", "realized_vol_annual", "vol_ratio_1h",
    # 1h signals
    "bp_1h", "chg_1h", "dir_1h", "consec_dir_1h",
    "stoch_k_1h", "stoch_cross_1h", "rsi_1h", "macd_hist_1h",
    "donchian_breakout_1h", "engulfing_1h",
    "bb_pct_1h",
    # 1h rolling drift (matches branched 1h BTC model)
    "mu6h", "mu12h", "mu24h", "regime_z", "arima_forecast_1h",
    # 4h signals
    "stoch_k_4h", "rsi_4h", "chg_4h", "bp_4h",
    # Coinalyze
    "liq_score", "liq_bias", "oi_chg_pct", "ls_long_pct",
    # CoinGlass macro
    "fear_greed", "cg_composite",
    "kelly_fraction", "bet_fraction", "bet_amount", "bankroll",
    "resolved_yes", "would_win", "would_pnl",
    # Expiry price (backfilled at resolution)
    "spot_at_expiry", "price_move_pct", "miss_pct",
    # Markov regimes (BTC: 1h+15m; ETH: daily; SOL: 6h/4h/1h)
    "markov_regime_1h", "markov_regime_15m",
    "markov_eth_daily", "markov_sol_6h", "markov_sol_4h", "markov_sol_1h",
    # p_up_v2 drift model (BTC only)
    "p_up_v2_btc",
    # Rolling 6h empirical z_drift (for LGBM feature logging)
    "z_drift_6h",
    # Shadow LGBM output — lgbm_15m_{asset}.pkl runs alongside primary on every scan
    "p_gbdt",
    # Shadow stochastic signals (no gate logic — log-only for future analysis)
    "ou_theta",        # OU mean-reversion speed (higher = faster reversion)
    "ou_halflife",     # OU half-life in hours (ln2 / ou_theta)
    "ou_mu_distance",  # z-score: (current_price - OU long-run mean) / vol
    "hurst_exponent",  # H>0.5 trending, H<0.5 mean-reverting, H≈0.5 random walk
    "autocorr1_15",    # lag-1 autocorrelation of 15m log-returns (last 30 bars)
    "autocorr1_30",    # lag-1 autocorrelation of 30m log-returns (last 30 bars)
    "kalman_velocity", # Kalman-smoothed 1h price trend (return units, filtered)
    "kalman_residual", # Kalman residual: actual − filtered (mean-reversion signal)
]

RESULTS_DIR = Path(__file__).parent / "results"

HOURLY_CSV_MAP = {
    "BTC": RESULTS_DIR / "paper_trades.csv",
    "ETH": RESULTS_DIR / "paper_trades_eth.csv",
    "SOL": RESULTS_DIR / "paper_trades_sol.csv",
}


def _csv_path(asset: str) -> Path:
    return RESULTS_DIR / ASSET_CONFIG[asset.upper()]["csv_name"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv(asset: str) -> None:
    csv_path = _csv_path(asset)
    csv_path.parent.mkdir(exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        return
    with open(csv_path, "r", newline="") as f:
        existing = csv.DictReader(f).fieldnames or []
    new_cols = [c for c in CSV_COLUMNS if c not in existing]
    if new_cols:
        df = pd.read_csv(csv_path)
        for col in new_cols:
            df[col] = ""
        df = df.reindex(columns=CSV_COLUMNS)   # keep header order == DictWriter order
        df.to_csv(csv_path, index=False)


def append_row(row: dict, asset: str) -> None:
    csv_path = _csv_path(asset)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writerow(row)


# ---------------------------------------------------------------------------
# p_up_v2 drift model constants (calibrated 2026-05-22, territory-split, recent era)
# k_yes=1.40 on YES-territory (z<0), k_no=1.56 on NO-territory (z>0)
K_PUP_V2_YES = 1.40
K_PUP_V2_NO  = 0.12


# Composite p_up / p_up_v2 from hourly runner
# ---------------------------------------------------------------------------

def fetch_composite_p_up(asset: str) -> Optional[float]:
    """Read the most recent composite_p_up from the hourly paper trade CSV.

    Falls back to None (treated as 0.50 neutral drift) if the file is missing,
    the column is absent, or the last value is more than 2 hours old.
    """
    path = HOURLY_CSV_MAP.get(asset.upper())
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["logged_at", "composite_p_up"],
                         low_memory=False)
        df["composite_p_up"] = pd.to_numeric(df["composite_p_up"], errors="coerce")
        df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
        recent = df.dropna(subset=["composite_p_up", "logged_at"]).sort_values("logged_at")
        if recent.empty:
            return None
        last_time = recent["logged_at"].iloc[-1].to_pydatetime()
        age_min = (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0
        if age_min > 120:
            return None
        val = float(recent["composite_p_up"].iloc[-1])
        return val if 0.0 < val < 1.0 else None
    except Exception:
        return None


def fetch_p_up_v2(asset: str) -> Optional[float]:
    """Read most recent p_up_v2 from hourly paper trade CSV (same source as composite_p_up)."""
    path = HOURLY_CSV_MAP.get(asset.upper())
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["logged_at", "p_up_v2"], low_memory=False)
        df["p_up_v2"] = pd.to_numeric(df["p_up_v2"], errors="coerce")
        df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
        recent = df.dropna(subset=["p_up_v2", "logged_at"]).sort_values("logged_at")
        if recent.empty:
            return None
        last_time = recent["logged_at"].iloc[-1].to_pydatetime()
        age_min = (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0
        if age_min > 120:
            return None
        val = float(recent["p_up_v2"].iloc[-1])
        return val if 0.0 < val < 1.0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Kalshi contract fetching
# ---------------------------------------------------------------------------

def fetch_15m_contracts(auth: KalshiAuth, asset: str = "BTC") -> list[dict]:
    """
    Fetch active 15m contracts for the given asset. Returns list of dicts with:
      ticker, floor_strike, p_market, bid, ask, close_time, tau_minutes
    Filtered to contracts with tau in [MIN_TAU_MIN, MAX_TAU_MIN] and valid bid/ask.
    """
    now_ts = int(time.time())
    now_dt = datetime.now(timezone.utc)
    series = ASSET_CONFIG[asset.upper()]["series_ticker"]

    params = {
        "series_ticker": series,
        "min_close_ts":  now_ts,
        "max_close_ts":  now_ts + 1800,  # look 30 min ahead
        "limit":         50,
    }
    data = kalshi_get("/markets", params, auth)
    markets = data.get("markets") or []

    contracts = []
    for m in markets:
        status = m.get("status", "")
        if status in ("initialized", "closed", "settled", "resolved", "finalized"):
            continue

        fs = m.get("floor_strike")
        ct = m.get("close_time", "")
        if fs is None or not ct:
            continue
        try:
            fs = float(fs)
            close_dt = pd.Timestamp(ct).tz_convert("UTC")
        except Exception:
            continue

        tau_min = (close_dt - now_dt).total_seconds() / 60.0
        if tau_min < MIN_TAU_MIN or tau_min > MAX_TAU_MIN:
            continue

        try:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
        except (ValueError, TypeError):
            continue
        if bid <= 0 or ask <= 0:
            continue

        contracts.append({
            "ticker":       m.get("ticker", ""),
            "floor_strike": fs,
            "p_market":     (bid + ask) / 2.0,
            "bid":          bid,
            "ask":          ask,
            "close_time":   ct,
            "tau_minutes":  tau_min,
        })

    return contracts


# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------

def _fetch_spot_at_time(close_dt: datetime, asset: str = "BTC") -> Optional[float]:
    """Fetch Binance 1m close price at close_dt for expiry price logging."""
    from live_signal import ASSET_CONFIG
    symbol = ASSET_CONFIG.get(asset.upper(), ASSET_CONFIG["BTC"])["binance_symbol"]
    try:
        end_ms = int(close_dt.timestamp() * 1000)
        r = requests.get("https://api.binance.us/api/v3/klines", params={
            "symbol": symbol, "interval": "1m", "endTime": end_ms, "limit": 2,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[-1][4])  # close of last completed 1m bar at expiry
    except Exception:
        pass
    return None


def resolve_pending(auth: Optional[KalshiAuth], asset: str, is_live: bool = False) -> None:
    """Fill in resolved_yes / would_win / would_pnl for settled contracts."""
    if is_live and auth is not None:
        live_csv = live_trading.get_live_csv_path(asset)
        live_trading.settle_live_trades(auth, csv_path=live_csv)
    csv_path = _csv_path(asset)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    pending = df[df["resolved_yes"].isna() | (df["resolved_yes"].astype(str) == "")]
    if pending.empty:
        return

    now_dt = datetime.now(timezone.utc)
    updated = 0
    for idx, row in pending.iterrows():
        ct_str = str(row.get("close_time", ""))
        ticker = str(row.get("contract_ticker", ""))
        if not ct_str or not ticker:
            continue
        try:
            close_dt = pd.Timestamp(ct_str).tz_convert("UTC")
        except Exception:
            continue
        if close_dt > now_dt:
            continue  # contract not expired yet

        resolved_yes = None

        # Try Kalshi API first
        if auth is not None:
            try:
                mkt = kalshi_get(f"/markets/{ticker}", {}, auth)
                market = mkt.get("market") or {}
                status = market.get("status", "")
                if status in ("settled", "resolved", "finalized", "determined"):
                    result = (market.get("result") or "").lower()
                    if result == "yes":
                        resolved_yes = True
                    elif result == "no":
                        resolved_yes = False
            except Exception:
                pass

        # Fallback: compare floor_strike to live spot
        if resolved_yes is None:
            try:
                spot_now = fetch_live_spot(asset)
                floor = float(row.get("floor_strike", 0))
                if spot_now and floor > 0:
                    resolved_yes = (spot_now > floor)
            except Exception:
                pass

        if resolved_yes is None:
            continue

        side     = str(row.get("side", "yes")).lower()
        p_market = float(row.get("p_market", 0.5))
        bet_amt  = float(row.get("bet_amount", 0))

        if side == "yes":
            would_win = resolved_yes
            payout    = (1.0 - p_market) / p_market if p_market > 0 else 0
        else:
            would_win = not resolved_yes
            payout    = p_market / (1.0 - p_market) if p_market < 1 else 0

        would_pnl = round(bet_amt * payout if would_win else -bet_amt, 2)

        df.at[idx, "resolved_yes"] = int(resolved_yes)
        df.at[idx, "would_win"]    = int(would_win)
        df.at[idx, "would_pnl"]    = would_pnl

        # Log expiry price and move magnitude
        spot_scan = float(row.get("spot", 0) or 0)
        floor_s   = float(row.get("floor_strike", 0) or 0)
        spot_exp  = _fetch_spot_at_time(close_dt, asset)
        if spot_exp and spot_scan > 0:
            df.at[idx, "spot_at_expiry"] = round(spot_exp, 2)
            df.at[idx, "price_move_pct"] = round((spot_exp - spot_scan) / spot_scan * 100, 4)
        if spot_exp and floor_s > 0:
            df.at[idx, "miss_pct"] = round((spot_exp - floor_s) / floor_s * 100, 4)

        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False)
        print(f"  [resolve] Updated {updated} resolved trade(s).")


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _stoch_k(prices_high: pd.Series, prices_low: pd.Series,
              prices_close: pd.Series, period: int = 14) -> float:
    if len(prices_close) < period + 1:
        return 50.0
    ll = prices_low.rolling(period).min()
    hh = prices_high.rolling(period).max()
    rng = hh - ll
    sk = ((prices_close - ll) / rng.replace(0, float("nan"))) * 100
    val = sk.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-10)
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _macd_hist(prices: pd.Series, fast: int = 12, slow: int = 26,
               signal: int = 9) -> float:
    if len(prices) < slow + signal:
        return 0.0
    ema_fast  = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow  = prices.ewm(span=slow,   adjust=False).mean()
    macd      = ema_fast - ema_slow
    sig_line  = macd.ewm(span=signal, adjust=False).mean()
    val       = (macd - sig_line).iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def compute_signals(live_1m: pd.DataFrame, asset: str = "BTC") -> dict:
    """
    Compute microstructure signals from 1m OHLCV data.
    Uses iloc[-2] for completed bars (last bar may be incomplete).
    """
    sig = {}
    if live_1m is None or len(live_1m) < 20:
        return sig

    close = live_1m["close"].astype(float)
    high  = live_1m["high"].astype(float)
    low   = live_1m["low"].astype(float)
    vol   = live_1m["volume"].astype(float)

    # Realized vol — multi-window blend (50% 15m + 30% 30m + 20% 60m), per-minute units
    # Mirrors the 1h runner's vol_multi so sigma_tau is computed the same way.
    vol_fallback = ASSET_CONFIG.get(asset.upper(), ASSET_CONFIG["BTC"])["vol_fallback"]
    try:
        _vol_result = compute_realized_volatility(live_1m, asset=asset)
        sig["vol_multi"]          = _vol_result.vol_multi   # per-minute, for sigma_tau
        sig["realized_vol_annual"] = _vol_result.vol_multi * math.sqrt(MINS_PER_YEAR)
    except Exception:
        _fallback_pm = vol_fallback / math.sqrt(MINS_PER_YEAR)
        sig["vol_multi"]           = _fallback_pm
        sig["realized_vol_annual"] = vol_fallback

    # Raw price changes (using last BAR, not last tick; iloc[-1] is live bar)
    if len(close) >= 3:
        sig["chg_1m"] = float((close.iloc[-2] / close.iloc[-3] - 1) * 100)
    if len(close) >= 7:
        sig["chg_5m"] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
    if len(close) >= 16:
        sig["chg_15m"] = float((close.iloc[-1] / close.iloc[-16] - 1) * 100)

    # EMA bias on 1m closes (current bar ok — just a level comparison)
    ema5  = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    e5 = float(ema5.iloc[-1])
    e20 = float(ema20.iloc[-1])
    sig["ema_bias"] = 1 if e5 > e20 else -1 if e5 < e20 else 0

    # 5m bars (resample from 1m)
    df5 = live_1m.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()

    if len(df5) >= 2:
        # bp_5m, body_5m, dir_5m: last COMPLETED 5m bar
        r5  = float(df5["high"].iloc[-2]) - float(df5["low"].iloc[-2])
        c5  = float(df5["close"].iloc[-2])
        o5  = float(df5["open"].iloc[-2])
        l5  = float(df5["low"].iloc[-2])
        sig["bp_5m"]   = (c5 - l5) / r5 if r5 > 0 else 0.5
        sig["body_5m"] = abs(c5 - o5) / r5 if r5 > 0 else 0.0
        sig["dir_5m"]  = 1 if c5 > o5 else -1 if c5 < o5 else 0

        # vol_ratio: last completed 5m bar volume vs 20-bar mean
        if len(df5) >= 22:
            avg_vol = float(df5["volume"].iloc[-22:-2].mean())
            sig["vol_ratio"] = float(df5["volume"].iloc[-2]) / avg_vol if avg_vol > 0 else 1.0

        # vol_ratio_5m: price-based realized vol ratio (matches build_15m_model.py convention)
        # rv over last 12 5m bars vs 288-bar (24h) rolling median
        if len(df5) >= 300:
            _lr5 = np.log(df5["close"] / df5["close"].shift(1))
            _rv5 = _lr5.rolling(12).std()
            _med5 = _rv5.rolling(288).median()
            _last_rv5 = float(_rv5.iloc[-2])
            _last_med5 = float(_med5.iloc[-2])
            if _last_med5 > 0 and not np.isnan(_last_rv5) and not np.isnan(_last_med5):
                sig["vol_ratio_5m"] = float(np.clip(_last_rv5 / _last_med5, 0, 5))
            else:
                sig["vol_ratio_5m"] = 1.0
        else:
            sig["vol_ratio_5m"] = 1.0

        # VWAP on 5m bars (last 20 completed bars)
        df5_r = df5.iloc[max(0, len(df5) - 21):-1]  # up to but not including last (incomplete)
        if len(df5_r) >= 5:
            tp = (df5_r["high"] + df5_r["low"] + df5_r["close"]) / 3
            cum_v = df5_r["volume"].cumsum()
            cum_tv = (tp * df5_r["volume"]).cumsum()
            last_vol = float(cum_v.iloc[-1])
            if last_vol > 0:
                vwap = float(cum_tv.iloc[-1]) / last_vol
                spot_now = float(close.iloc[-1])
                sig["vwap_dist"] = (spot_now - vwap) / vwap * 100

        # stoch_k on 5m bars
        if len(df5) >= 16:
            sig["stoch_k_5m"] = _stoch_k(df5["high"], df5["low"], df5["close"], 14)

    # 15m bars (resample from 1m)
    df15 = live_1m.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()

    if len(df15) >= 2:
        # body_15m, bp_15m, dir_15m: last COMPLETED 15m bar
        r15 = float(df15["high"].iloc[-2]) - float(df15["low"].iloc[-2])
        c15 = float(df15["close"].iloc[-2])
        o15 = float(df15["open"].iloc[-2])
        l15 = float(df15["low"].iloc[-2])
        sig["body_15m"] = abs(c15 - o15) / r15 if r15 > 0 else 0.0
        sig["bp_15m"]   = (c15 - l15) / r15 if r15 > 0 else 0.5
        sig["dir_15m"]  = 1 if c15 > o15 else -1 if c15 < o15 else 0

        # wick ratios on last completed 15m bar
        h15 = l15 + r15
        if r15 > 0:
            _body_top = max(o15, c15)
            _body_bot = min(o15, c15)
            sig["upper_wick_15m"] = (h15 - _body_top) / r15
            sig["lower_wick_15m"] = (_body_bot - l15)  / r15
        else:
            sig["upper_wick_15m"] = 0.0
            sig["lower_wick_15m"] = 0.0

        # stoch_k on 15m bars
        if len(df15) >= 16:
            sig["stoch_k_15m"] = _stoch_k(df15["high"], df15["low"], df15["close"], 14)
            # ATR14 on 15m — atr_ratio (vs spot) and range_ratio (vs ATR)
            _tr15 = pd.concat([
                df15["high"] - df15["low"],
                (df15["high"] - df15["close"].shift(1)).abs(),
                (df15["low"]  - df15["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            _atr14_15m = float(_tr15.rolling(14).mean().iloc[-2])
            _spot_now   = float(close.iloc[-1])
            if not np.isnan(_atr14_15m) and _spot_now > 0:
                sig["atr_ratio_15m"]  = _atr14_15m / _spot_now
                sig["range_ratio_15m"] = r15 / _atr14_15m if _atr14_15m > 0 else float("nan")
            else:
                sig["atr_ratio_15m"]  = float("nan")
                sig["range_ratio_15m"] = float("nan")

    if len(df15) >= 3:
        # consecutive 15m direction streak (mirrors consec_dir_1h at 15m resolution)
        _dirs15 = np.sign(df15["close"].diff().dropna()).tolist()
        _str15  = 0.0
        for _d15 in reversed(_dirs15):
            if _d15 == 0:
                break
            if _str15 == 0:
                _str15 = _d15
            elif np.sign(_d15) == np.sign(_str15):
                _str15 += _d15
            else:
                break
        sig["consec_dir_15m"] = float(max(-5, min(5, _str15)))

    # ── 1h bars: primary model drivers ───────────────────────────────────────
    df1h = live_1m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()

    # Drop the current (potentially incomplete) 1h bar; use last COMPLETED bar
    df1h_c = df1h.iloc[:-1] if len(df1h) >= 2 else df1h

    if len(df1h_c) >= 2:
        h = df1h_c.iloc[-1]   # last completed 1h bar
        h_prev = df1h_c.iloc[-2]

        r1h = float(h["high"]) - float(h["low"])
        sig["bp_1h"]  = (float(h["close"]) - float(h["low"])) / r1h if r1h > 0 else 0.5
        sig["chg_1h"] = (float(h["close"]) - float(h["open"])) / float(h["open"]) * 100.0
        sig["dir_1h"] = 1 if float(h["close"]) > float(h["open"]) else -1

        # engulfing_1h: current body engulfs prior body in opposite direction
        cur_body  = float(h["close"]) - float(h["open"])
        prev_body = float(h_prev["close"]) - float(h_prev["open"])
        if cur_body > 0 and prev_body < 0 and abs(cur_body) > abs(prev_body):
            sig["engulfing_1h"] = 1
        elif cur_body < 0 and prev_body > 0 and abs(cur_body) > abs(prev_body):
            sig["engulfing_1h"] = -1
        else:
            sig["engulfing_1h"] = 0

    if len(df1h_c) >= 14:
        # stoch_k and stoch_d on completed 1h bars
        sk_arr = pd.Series(
            pd.Series(df1h_c["close"].values).values,
            index=df1h_c.index
        )  # use full df1h_c for rolling
        lo_1h  = df1h_c["low"].rolling(14).min()
        hi_1h  = df1h_c["high"].rolling(14).max()
        rng_1h = hi_1h - lo_1h
        sk_s   = ((df1h_c["close"] - lo_1h) / rng_1h.replace(0, float("nan"))) * 100.0
        sd_s   = sk_s.rolling(3).mean()

        sk_last = float(sk_s.iloc[-1]) if not pd.isna(sk_s.iloc[-1]) else 50.0
        sd_last = float(sd_s.iloc[-1]) if not pd.isna(sd_s.iloc[-1]) else sk_last
        sig["stoch_k_1h"] = sk_last

        # stoch crossover
        if len(sk_s) >= 2 and not pd.isna(sd_s.iloc[-2]):
            sk_prev = float(sk_s.iloc[-2])
            sd_prev = float(sd_s.iloc[-2])
            if sk_last > sd_last and sk_prev <= sd_prev:
                sig["stoch_cross_1h"] = 1
            elif sk_last < sd_last and sk_prev >= sd_prev:
                sig["stoch_cross_1h"] = -1
            else:
                sig["stoch_cross_1h"] = 0

    if len(df1h_c) >= 20:
        # Donchian 20-bar breakout
        dc_high = df1h_c["high"].rolling(20).max().iloc[-1]
        dc_low  = df1h_c["low"].rolling(20).min().iloc[-1]
        last_close_1h = float(df1h_c["close"].iloc[-1])
        if last_close_1h >= dc_high:
            sig["donchian_breakout_1h"] = 1
        elif last_close_1h <= dc_low:
            sig["donchian_breakout_1h"] = -1
        else:
            sig["donchian_breakout_1h"] = 0
        # 1h Donchian POSITION (0=at 20h low, 1=at 20h high) — for the donch_low_no_boost.
        sig["donch_1h_pos"] = ((last_close_1h - dc_low) / (dc_high - dc_low)
                               if dc_high > dc_low else float("nan"))

    if len(df1h_c) >= 3:
        # Consecutive 1h direction count (streak, capped at ±5)
        dirs = np.sign(df1h_c["close"].diff().dropna()).tolist()
        streak = 0.0
        for d in reversed(dirs):
            if d == 0:
                break
            if streak == 0:
                streak = d
            elif np.sign(d) == np.sign(streak):
                streak += d
            else:
                break
        sig["consec_dir_1h"] = float(max(-5, min(5, streak)))

    if len(df1h_c) >= 16:
        sig["rsi_1h"] = _rsi(df1h_c["close"], period=14)

    if len(df1h_c) >= 20:
        # ema_bias_1h: +1 if last completed 1h close > EMA20(1h), -1 otherwise
        _ema20_1h = df1h_c["close"].ewm(span=20, adjust=False).mean()
        sig["ema_bias_1h"] = 1.0 if float(df1h_c["close"].iloc[-1]) > float(_ema20_1h.iloc[-1]) else -1.0

    if len(df1h_c) >= 36:
        sig["macd_hist_1h"] = _macd_hist(df1h_c["close"])

    if len(df1h_c) >= 25:
        # vol_ratio_1h: 1h price-based realized vol ratio
        # rv over last 24 1h bars vs rolling median (matches mine_live_losses.py convention)
        _lr1h = np.log(df1h_c["close"] / df1h_c["close"].shift(1))
        _rv1h = _lr1h.rolling(24).std()
        _med1h = _rv1h.rolling(14 * 24).median()
        _last_rv1h  = float(_rv1h.iloc[-1])
        _last_med1h = float(_med1h.iloc[-1]) if len(_med1h.dropna()) > 0 else float("nan")
        if _last_med1h > 0 and not np.isnan(_last_rv1h) and not np.isnan(_last_med1h):
            sig["vol_ratio_1h"] = float(np.clip(_last_rv1h / _last_med1h, 0, 5))
        else:
            sig["vol_ratio_1h"] = 1.0

    # Nearest resistance above spot (1h EMA20/50/100/200 + daily VWAP).
    # Used by the YES Kelly dampener: halve bet when nearest overhead level <= 0.5%.
    # Sim (376 15m trades): YES with nearest_res<=0.5% → WR=43.8%, delta=+$329 when blocked.
    try:
        _cur_spot = float(close.iloc[-1])
        _res_lvls = []
        if len(df1h_c) >= 5:
            _1h_cls = df1h_c["close"].astype(float)
            for _ep in [20, 50, 100, 200]:
                if len(_1h_cls) >= _ep:
                    _ev = float(_1h_cls.ewm(span=_ep, adjust=False).mean().iloc[-1])
                    if _ev > _cur_spot:
                        _res_lvls.append(_ev)
        # Daily VWAP above spot
        if live_1m is not None and len(live_1m) >= 5:
            try:
                _day_mask = live_1m.index.normalize() == live_1m.index[-1].normalize()
                _day_bars = live_1m[_day_mask]
                if len(_day_bars) >= 5:
                    _dtp = (_day_bars["high"] + _day_bars["low"] + _day_bars["close"]) / 3
                    _dvwap = float((_dtp * _day_bars["volume"]).sum() / _day_bars["volume"].sum())
                    if _dvwap > _cur_spot:
                        _res_lvls.append(_dvwap)
            except Exception:
                pass
        sig["nearest_res_dist_pct"] = (
            min((_r - _cur_spot) / _cur_spot * 100 for _r in _res_lvls)
            if _res_lvls else 999.0
        )
    except Exception:
        sig["nearest_res_dist_pct"] = 999.0

    # ── EMA20 / EMA50 distance from 1h close (%) ─────────────────────────────
    # EWM works with any bar count; guard only against degenerate empty df.
    if len(df1h_c) >= 5:
        _c1h_f = df1h_c["close"].astype(float)
        _ema20_1h = _c1h_f.ewm(span=20, adjust=False).mean()
        _ema20_val = float(_ema20_1h.iloc[-1])
        if _ema20_val > 0:
            sig["ema20_dist_1h"] = float((_c1h_f.iloc[-1] - _ema20_val) / _ema20_val * 100)
        if len(df1h_c) >= 20:
            _ema50_1h = _c1h_f.ewm(span=50, adjust=False).mean()
            _ema50_val = float(_ema50_1h.iloc[-1])
            if _ema50_val > 0:
                sig["ema50_dist_1h"] = float((_c1h_f.iloc[-1] - _ema50_val) / _ema50_val * 100)

    # ── Bollinger Band %B from 1h close (min_periods=5 for short histories) ──
    if len(df1h_c) >= 5:
        _c1h_f = df1h_c["close"].astype(float)
        _n_bb   = min(20, len(df1h_c))
        _bb_mid = _c1h_f.rolling(_n_bb, min_periods=5).mean()
        _bb_std = _c1h_f.rolling(_n_bb, min_periods=5).std()
        _bb_lo  = _bb_mid - 2 * _bb_std
        _bb_hi  = _bb_mid + 2 * _bb_std
        _bb_rng = float((_bb_hi - _bb_lo).iloc[-1])
        if _bb_rng > 0 and not np.isnan(_bb_rng):
            sig["bb_pct_1h"] = float(
                (_c1h_f.iloc[-1] - float(_bb_lo.iloc[-1])) / _bb_rng)

    # ── Rolling drift mu + regime_z from 1h log returns ──────────────────────
    if len(df1h_c) >= 7:
        _lr_1h = np.log(df1h_c["close"] / df1h_c["close"].shift(1))
        def _roll_mean(s, w):
            # min_periods=1 gives partial-window mean rather than NaN → 0.0 fallback
            v = float(s.rolling(w, min_periods=1).mean().iloc[-1])
            return 0.0 if np.isnan(v) else v
        sig["mu6h"]  = _roll_mean(_lr_1h, 6)
        sig["mu12h"] = _roll_mean(_lr_1h, 12)
        sig["mu24h"] = _roll_mean(_lr_1h, 24)
        _ewm_mean = float(_lr_1h.ewm(span=12).mean().iloc[-1])
        _ewm_std  = float(_lr_1h.ewm(span=24).std().iloc[-1])
        sig["regime_z"] = float(np.clip(
            _ewm_mean / _ewm_std if _ewm_std > 0 else 0.0, -3.0, 3.0))

    # ── ARIMA(2,0,1) 1-step forecast from 1h log returns ─────────────────────
    if len(df1h_c) >= 20:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            _lr_arima = np.log(df1h_c["close"] / df1h_c["close"].shift(1)).dropna()
            sig["arima_forecast_1h"] = float(
                _ARIMA(_lr_arima, order=(2, 0, 1)).fit(disp=False).forecast(steps=1).iloc[0])
        except Exception:
            pass

    # ── Shadow stochastic signals (log only — no gate logic) ─────────────────
    # Uses 1h completed bars; minimum 30 bars for reliable estimates.
    if len(df1h_c) >= 30:
        try:
            _cl_s = df1h_c["close"].values.astype(float)
            _lr_s = np.diff(np.log(_cl_s))  # log returns

            # autocorr1_15 / autocorr1_30 — lag-1 autocorrelation of 15m/30m
            # log-returns, sampled from the 1m series resampled to those intervals.
            # Using 1h bars as proxy: autocorr1_15 ≈ lag-1 on last 30 1h bars
            # (honest approximation without a separate 15m resample here).
            def _lag1_ac(arr):
                if len(arr) < 4:
                    return 0.0
                x, y = arr[:-1] - arr[:-1].mean(), arr[1:] - arr[1:].mean()
                denom = np.sqrt((x**2).sum() * (y**2).sum())
                return float(np.dot(x, y) / denom) if denom > 0 else 0.0

            sig["autocorr1_15"] = _lag1_ac(_lr_s[-30:])
            sig["autocorr1_30"] = _lag1_ac(_lr_s[-60:] if len(_lr_s) >= 60 else _lr_s)

            # hurst_exponent — rescaled range (R/S) on last 64 1h log-returns.
            # H > 0.5 = persistent/trending; H < 0.5 = mean-reverting.
            _h_wins = [8, 16, 32, 64]
            _rs_pts = []
            _h_lr = _lr_s[-64:] if len(_lr_s) >= 64 else _lr_s
            for _w in _h_wins:
                if len(_h_lr) < _w:
                    continue
                _seg = _h_lr[-_w:]
                _mean = _seg.mean()
                _dev  = np.cumsum(_seg - _mean)
                _r    = _dev.max() - _dev.min()
                _s    = _seg.std(ddof=1)
                if _s > 0:
                    _rs_pts.append((np.log(_w), np.log(_r / _s)))
            if len(_rs_pts) >= 2:
                _xs = np.array([p[0] for p in _rs_pts])
                _ys = np.array([p[1] for p in _rs_pts])
                _h  = float(np.polyfit(_xs, _ys, 1)[0])
                sig["hurst_exponent"] = round(np.clip(_h, 0.0, 1.0), 4)

            # Ornstein-Uhlenbeck fit via discrete AR(1) on 1h log-returns.
            # y_t = mu + phi*(y_{t-1} - mu) + eps  =>  theta = -ln(phi) / dt
            _ou_lr = _lr_s[-48:] if len(_lr_s) >= 48 else _lr_s
            if len(_ou_lr) >= 10:
                _y_ou  = _ou_lr
                _mu_ou = _y_ou.mean()
                _y_c   = _y_ou - _mu_ou
                _phi   = float(np.dot(_y_c[:-1], _y_c[1:]) /
                               (np.dot(_y_c[:-1], _y_c[:-1]) + 1e-12))
                _phi   = np.clip(_phi, -0.9999, 0.9999)
                # theta per hour (dt = 1h)
                _theta = float(-np.log(abs(_phi)))
                _theta = np.clip(_theta, 0.0, 10.0)
                sig["ou_theta"]   = round(_theta, 6)
                sig["ou_halflife"] = round(float(np.log(2) / _theta), 4) if _theta > 0 else 999.0
                # distance of current price from the OU mean (in vol units)
                _ou_std = float(_y_c.std(ddof=1)) + 1e-10
                _cur_lr = float(_lr_s[-1]) if len(_lr_s) > 0 else 0.0
                sig["ou_mu_distance"] = round((_cur_lr - _mu_ou) / _ou_std, 4)

            # Kalman filter on 1h log-returns (constant-velocity model).
            # State: [level, velocity]. Observation: log-return.
            _kl = _lr_s[-48:] if len(_lr_s) >= 48 else _lr_s
            if len(_kl) >= 5:
                _Q = np.array([[1e-5, 0.0], [0.0, 1e-5]])  # process noise
                _R = float(np.var(_kl)) + 1e-10              # obs noise
                _x = np.array([_kl[0], 0.0])
                _P = np.eye(2) * 0.1
                _F = np.array([[1.0, 1.0], [0.0, 1.0]])
                _H = np.array([[1.0, 0.0]])
                for _obs in _kl:
                    _x = _F @ _x
                    _P = _F @ _P @ _F.T + _Q
                    _K = _P @ _H.T / (float(_H @ _P @ _H.T) + _R)
                    _innov = _obs - float(_H @ _x)
                    _x = _x + _K.flatten() * _innov
                    _P = (np.eye(2) - np.outer(_K.flatten(), _H)) @ _P
                sig["kalman_velocity"] = round(float(_x[1]), 6)
                sig["kalman_residual"] = round(float(_kl[-1] - float(_H @ _x)), 6)
        except Exception:
            pass

    # ── 4h bars ───────────────────────────────────────────────────────────────
    df4h = live_1m.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    df4h_c = df4h.iloc[:-1] if len(df4h) >= 2 else df4h

    if len(df4h_c) >= 2:
        h4  = df4h_c.iloc[-1]
        r4h = float(h4["high"]) - float(h4["low"])
        sig["chg_4h"] = float((float(h4["close"]) / float(h4["open"]) - 1) * 100)
        sig["bp_4h"]  = float((float(h4["close"]) - float(h4["low"])) / r4h) if r4h > 0 else 0.5

    if len(df4h_c) >= 5:
        # Use period=5 when fewer than 14 4h bars available; period=14 when enough data
        _4h_period = 14 if len(df4h_c) >= 14 else 5
        sig["stoch_k_4h"] = _stoch_k(df4h_c["high"], df4h_c["low"], df4h_c["close"], _4h_period)
        sig["rsi_4h"]     = _rsi(df4h_c["close"], period=_4h_period)

    return sig


# ---------------------------------------------------------------------------
# p_model computation
# ---------------------------------------------------------------------------

def compute_zdrift_empirical_15m(
    df_resolved: pd.DataFrame,
    live_1m: pd.DataFrame,
    w_short: int = 10,
    w_long: int = 30,
    alpha: float = 0.6,
    cap: float = 0.5,
) -> Optional[float]:
    """
    Empirical z_drift from resolved 15m BTC trade history.
    Looks up BTC price at each close_time from live_1m (25h window).
    Returns None when fewer than w_short resolved trades are available.
    """
    try:
        needed = ["spot", "realized_vol_annual", "tau_minutes", "close_time", "resolved_yes"]
        df = df_resolved.dropna(subset=needed).copy()
        if len(df) < w_short:
            return None
        df = df.tail(max(w_long, 50))
        m1_idx = live_1m.index
        actual_z_list: list[float] = []
        for _, row in df.iterrows():
            try:
                spot_val = float(row["spot"])
                rv_ann   = float(row["realized_vol_annual"])
                tau_min  = float(row["tau_minutes"])
                if spot_val <= 0 or rv_ann <= 0 or tau_min <= 0:
                    continue
                vol_eff   = rv_ann / math.sqrt(MINS_PER_YEAR)
                sigma_tau = vol_eff * math.sqrt(tau_min)
                if sigma_tau <= 0:
                    continue
                close_ts = pd.Timestamp(row["close_time"]).tz_convert("UTC")
                if close_ts in m1_idx:
                    btc_expiry = float(live_1m.loc[close_ts, "open"])
                else:
                    idx = m1_idx.searchsorted(close_ts)
                    if idx >= len(m1_idx):
                        continue
                    btc_expiry = float(live_1m.iloc[idx]["open"])
                actual_z_list.append(math.log(btc_expiry / spot_val) / sigma_tau)
            except Exception:
                continue
        if len(actual_z_list) < w_short:
            return None
        z_short = sum(actual_z_list[-w_short:]) / w_short
        z_long  = sum(actual_z_list[-w_long:]) / len(actual_z_list[-w_long:])
        return float(max(-cap, min(cap, alpha * z_short + (1 - alpha) * z_long)))
    except Exception:
        return None


def compute_z_drift_6h(df_resolved: pd.DataFrame) -> "float | None":
    """
    Rolling 6h mean of actual z-scores from resolved trades with spot_at_expiry.
    Uses logged spot_at_expiry rather than live_1m lookup — cleaner and works offline.
    Returns None when fewer than 3 valid rows exist in the window.
    """
    try:
        needed = ["decision_time", "spot", "realized_vol_annual", "tau_minutes", "spot_at_expiry"]
        df = df_resolved.dropna(subset=needed).copy()
        if df.empty:
            return None
        df["decision_time"] = pd.to_datetime(df["decision_time"], format="ISO8601", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)  # wall-clock, not last-trade anchor
        window = df[df["decision_time"] >= cutoff]
        if len(window) < 3:
            return None
        sigma = window["realized_vol_annual"] * (window["tau_minutes"] / 525600.0) ** 0.5
        sigma = sigma.replace(0, float("nan"))
        actual_z = (window["spot_at_expiry"].astype(float) / window["spot"].astype(float)).apply(
            lambda x: x if x > 0 else float("nan")
        ).apply(lambda x: __import__("math").log(x)) / sigma
        valid = actual_z.dropna()
        if len(valid) < 3:
            return None
        return float(valid.mean())
    except Exception:
        return None


def compute_p_yes_zdrift_15m(
    spot: float, floor_strike: float, tau_min: float,
    sig: dict, z_drift: float, p_market: float,
) -> float:
    """
    Pure log-normal z_drift model for YES side (mirrors 1h BTC formula).
    p_yes = norm.cdf(z_drift - z_strike)
    Uses same blended vol as LightGBM path for sigma_tau consistency.
    """
    if tau_min <= 0.5 or spot <= 0 or floor_strike <= 0:
        return 0.5
    vol_realized = sig.get("vol_multi", None)
    if vol_realized is None or not (vol_realized > 0):
        rv_ann = sig.get("realized_vol_annual", 0.3)
        vol_realized = rv_ann / math.sqrt(MINS_PER_YEAR)
    vol_imp   = implied_vol_from_price(p_market, spot, floor_strike, tau_min)
    weight    = REALIZED_VOL_WEIGHT_BY_ASSET.get("BTC", 0.35)
    vol_eff   = blend_vol(vol_realized, vol_imp, weight=weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z_strike  = math.log(floor_strike / spot) / sigma_tau
    return float(np.clip(norm.cdf(z_drift - z_strike), 0.03, 0.97))


def compute_p_yes_pup_v2_15m(
    spot: float, floor_strike: float, tau_min: float,
    sig: dict, p_up_v2: float, p_market: float,
) -> float:
    """
    BTC YES prob: log-normal with p_up_v2 τ-scaled drift.
    z_drift = Φ⁻¹(p_up_v2) × K_PUP_V2_YES × √(τ/60)
    Calibrated 2026-05-22 on YES-territory (z<0), recent era: k=1.40.
    """
    if tau_min <= 0.5 or spot <= 0 or floor_strike <= 0:
        return 0.5
    vol_realized = sig.get("vol_multi", None)
    if vol_realized is None or not (vol_realized > 0):
        rv_ann = sig.get("realized_vol_annual", 0.3)
        vol_realized = rv_ann / math.sqrt(MINS_PER_YEAR)
    vol_imp   = implied_vol_from_price(p_market, spot, floor_strike, tau_min)
    weight    = REALIZED_VOL_WEIGHT_BY_ASSET.get("BTC", 0.35)
    vol_eff   = blend_vol(vol_realized, vol_imp, weight=weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z_strike  = math.log(floor_strike / spot) / sigma_tau
    tau_scale = math.sqrt(min(tau_min, 60.0) / 60.0)
    z_drift   = norm.ppf(float(np.clip(p_up_v2, 0.02, 0.98))) * K_PUP_V2_YES * tau_scale
    return float(np.clip(norm.cdf(z_drift - z_strike), 0.03, 0.97))


def compute_p_no_pup_v2_15m(
    spot: float, floor_strike: float, tau_min: float,
    sig: dict, p_up_v2: float, p_market: float,
) -> float:
    """
    BTC NO prob: complementary log-normal with p_up_v2 τ-scaled drift.
    z_drift = Φ⁻¹(p_up_v2) × K_PUP_V2_NO × √(τ/60)
    Calibrated 2026-05-22 on NO-territory (z>0), recent era: k=1.56.
    Both YES and NO share the same distribution → coherent cross-strike pricing.
    """
    if tau_min <= 0.5 or spot <= 0 or floor_strike <= 0:
        return 0.5
    vol_realized = sig.get("vol_multi", None)
    if vol_realized is None or not (vol_realized > 0):
        rv_ann = sig.get("realized_vol_annual", 0.3)
        vol_realized = rv_ann / math.sqrt(MINS_PER_YEAR)
    vol_imp   = implied_vol_from_price(p_market, spot, floor_strike, tau_min)
    weight    = REALIZED_VOL_WEIGHT_BY_ASSET.get("BTC", 0.35)
    vol_eff   = blend_vol(vol_realized, vol_imp, weight=weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z_strike  = math.log(floor_strike / spot) / sigma_tau
    tau_scale = math.sqrt(min(tau_min, 60.0) / 60.0)
    z_drift   = norm.ppf(float(np.clip(p_up_v2, 0.02, 0.98))) * K_PUP_V2_NO * tau_scale
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.03, 0.97))


def _compute_1h_drift(sig: dict, tau_min: float) -> float:
    """
    Composite 1h directional drift from 7 signals.
    Each component normalized via tanh. SCALE reduced to 0.10 (was 0.25)
    to limit directional influence while data accumulates for calibration.
    Max z_drift ≈ ±0.24 at tau=14min (was ±0.61), shifting p_model ~9pp max.
    """
    chg1h  = sig.get("chg_1h",              0.0)
    bp1h   = sig.get("bp_1h",               0.5)
    streak = sig.get("consec_dir_1h",        0.0)
    sk1h   = sig.get("stoch_k_1h",          50.0)
    dc_brk = sig.get("donchian_breakout_1h", 0.0)
    eng1h  = sig.get("engulfing_1h",         0.0)
    sc1h   = sig.get("stoch_cross_1h",       0.0)

    composite = (
        0.35 * math.tanh(chg1h  / 1.0) +
        0.26 * math.tanh((bp1h - 0.5) * 4.0) +
        0.20 * math.tanh(streak / 2.5) +
        0.15 * math.tanh((sk1h - 50.0) / 25.0) +
        0.18 * float(dc_brk) +
        0.16 * float(eng1h) +
        0.15 * float(sc1h)
    )
    SCALE = 0.10
    return composite * SCALE * math.sqrt(tau_min / 5.0)


def compute_p_model_15m(spot: float, floor_strike: float,
                         tau_min: float, sig: dict,
                         asset: str = "BTC",
                         p_market: float = 0.5) -> float:
    """
    Probability that price ends above floor_strike at expiry.

    Primary path: LightGBM model (lgbm_15m_{asset}.pkl) trained on 2yr backtest.
    Fallback: log-normal base + 7-signal 1h composite drift.

    LightGBM features (20):
      offset_pct, z_score, bp_15m, body_15m, dir_15m, chg_15m, stoch_k_15m,
      bp_5m, body_5m, dir_5m, chg_5m, stoch_k_5m, vol_ratio_5m,
      chg_1h, bp_1h, stoch_k_1h, ema_bias_1h, consec_dir_1h, vol_ratio_1h,
      realized_vol_annual
    """
    if tau_min <= 0.5 or spot <= 0 or floor_strike <= 0:
        return 0.5

    # ── Volatility (shared by both paths) ────────────────────────────────────
    vol_realized = sig.get("vol_multi", None)
    if vol_realized is None or not (vol_realized > 0):
        rv_ann = sig.get("realized_vol_annual", 0.3)
        vol_realized = rv_ann / math.sqrt(MINS_PER_YEAR)

    vol_imp   = implied_vol_from_price(p_market, spot, floor_strike, tau_min)
    weight    = REALIZED_VOL_WEIGHT_BY_ASSET.get(asset.upper(), 0.35)
    vol_eff   = blend_vol(vol_realized, vol_imp, weight=weight)
    sigma_tau = max(vol_eff * math.sqrt(tau_min), 1e-6)
    z_strike  = math.log(floor_strike / spot) / sigma_tau

    # ── LightGBM primary path ─────────────────────────────────────────────────
    lgbm_model = _LGBM_MODELS.get(asset.upper())
    if lgbm_model is not None:
        try:
            offset_pct = (floor_strike / spot - 1.0) * 100.0
            feat = pd.DataFrame([{
                "offset_pct":        offset_pct,
                "z_score":           z_strike,
                "bp_15m":            sig.get("bp_15m",            0.5),
                "body_15m":          sig.get("body_15m",          0.0),
                "dir_15m":           sig.get("dir_15m",           0.0),
                "chg_15m":           sig.get("chg_15m",           0.0),
                "stoch_k_15m":       sig.get("stoch_k_15m",      50.0),
                "bp_5m":             sig.get("bp_5m",             0.5),
                "body_5m":           sig.get("body_5m",           0.0),
                "dir_5m":            sig.get("dir_5m",            0.0),
                "chg_5m":            sig.get("chg_5m",            0.0),
                "stoch_k_5m":        sig.get("stoch_k_5m",       50.0),
                "vol_ratio_5m":      sig.get("vol_ratio_5m",      1.0),
                "chg_1h":            sig.get("chg_1h",            0.0),
                "bp_1h":             sig.get("bp_1h",             0.5),
                "stoch_k_1h":        sig.get("stoch_k_1h",       50.0),
                "ema_bias_1h":       sig.get("ema_bias_1h",       0.0),
                "consec_dir_1h":     sig.get("consec_dir_1h",     0.0),
                "vol_ratio_1h":      sig.get("vol_ratio_1h",      1.0),
                "realized_vol_annual": sig.get("realized_vol_annual", 0.3),
            }])
            p_lgbm = float(lgbm_model.predict_proba(feat)[0, 1])
            return max(0.05, min(0.96, p_lgbm))
        except Exception as _e:
            print(f"  [{asset.lower()}_15m_lgbm] inference error: {_e}")

    # ── Fallback: log-normal base + composite drift ───────────────────────────
    z_drift = _compute_1h_drift(sig, tau_min)
    p_model = float(norm.cdf(-z_strike + z_drift))
    return max(0.05, min(0.96, p_model))


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run_scan(auth: Optional[KalshiAuth], bankroll: float, asset: str = "BTC",
             already_bet: Optional[set] = None,
             is_live: bool = False, daily_loss_limit: Optional[float] = None) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    series  = ASSET_CONFIG[asset.upper()]["series_ticker"]
    print(f"\n{'=' * 60}")
    print(f"  {asset} 15M SCAN  {now_str} UTC")
    print("=" * 60)

    # Fetch spot
    spot = fetch_live_spot(asset)
    if spot is None:
        print(f"  [error] Could not fetch {asset} spot price. Aborting.")
        return
    print(f"  {asset} spot: ${spot:,.4f}")

    # Fetch 1m candles
    live_1m = fetch_recent_candles("1m", CANDLES_NEEDED, asset=asset)
    if live_1m is None or len(live_1m) < 20:
        n = len(live_1m) if live_1m is not None else 0
        print(f"  [error] Insufficient 1m candle data ({n} bars).")
        return

    # Compute signals
    sig = compute_signals(live_1m, asset=asset)

    # Inject composite_p_up from most recent hourly scan
    composite_p_up = fetch_composite_p_up(asset)
    sig["composite_p_up"] = composite_p_up  # None → neutral drift in model

    # Markov regimes — BTC uses live_1m + yfinance; ETH/SOL use yfinance only (cached per hour)
    _markov_1h: Optional[str] = None
    _markov_15m: Optional[str] = None
    _markov_eth_daily: Optional[str] = None
    _markov_sol_6h: Optional[str] = None
    _markov_sol_4h: Optional[str] = None
    _markov_sol_1h: Optional[str] = None
    # Always compute all asset regimes — cached per hour so overhead is minimal after first call.
    try:
        _markov_1h = _get_btc_markov_regime_1h()
        if asset == "BTC":
            _df15m_reg = live_1m.resample("15min").agg({"close": "last"}).dropna().iloc[:-1]
            _rr15m_s = _df15m_reg["close"].pct_change(20)
            if pd.notna(_rr15m_s.iloc[-1]):
                _rr15m = float(_rr15m_s.iloc[-1])
                _markov_15m = "Bull" if _rr15m > 0.004 else "Bear" if _rr15m < -0.004 else "Sideways"
        print(f"  [markov] BTC 1h={_markov_1h or 'n/a'}  15m={_markov_15m or 'n/a'}")
    except Exception as _me:
        print(f"  [markov] BTC error: {_me}")
    try:
        _eth_regs = _get_markov_regimes_yf("ETH")
        _markov_eth_daily = _eth_regs.get("1d")
        print(f"  [markov] ETH daily={_markov_eth_daily or 'n/a'}")
    except Exception as _me:
        print(f"  [markov] ETH error: {_me}")
    try:
        _sol_regs = _get_markov_regimes_yf("SOL")
        _markov_sol_6h = _sol_regs.get("6h")
        _markov_sol_4h = _sol_regs.get("4h")
        _markov_sol_1h = _sol_regs.get("1h")
        print(f"  [markov] SOL 6h={_markov_sol_6h or 'n/a'}  "
              f"4h={_markov_sol_4h or 'n/a'}  1h={_markov_sol_1h or 'n/a'}")
    except Exception as _me:
        print(f"  [markov] SOL error: {_me}")
    sig["markov_regime_1h"]  = _markov_1h       or ""
    sig["markov_regime_15m"] = _markov_15m      or ""
    sig["markov_eth_daily"]  = _markov_eth_daily or ""
    sig["markov_sol_6h"]     = _markov_sol_6h    or ""
    sig["markov_sol_4h"]     = _markov_sol_4h    or ""
    sig["markov_sol_1h"]     = _markov_sol_1h    or ""

    # p_up_v2 drift model for BTC (k_yes=1.40, k_no=1.56, τ-scaled)
    # Replaces empirical z_drift. Falls back to z_drift when unavailable.
    _p_up_v2_btc: Optional[float] = None
    _zdrift_15m:  Optional[float] = None
    if asset == "BTC":
        _p_up_v2_btc = fetch_p_up_v2("BTC")
        if _p_up_v2_btc is not None:
            print(f"  [p_up_v2_btc] {_p_up_v2_btc:.3f}  "
                  f"(k_yes={K_PUP_V2_YES}, k_no={K_PUP_V2_NO}, τ-scaled)")
        else:
            # Fallback to empirical z_drift when p_up_v2 unavailable
            _csv_15m = _csv_path(asset)
            if _csv_15m.exists():
                try:
                    _df_all  = pd.read_csv(_csv_15m, low_memory=False)
                    _df_res  = _df_all[_df_all["resolved_yes"].notna() &
                                       (_df_all["resolved_yes"].astype(str) != "")]
                    _zdrift_15m = compute_zdrift_empirical_15m(_df_res, live_1m)
                    _n_res = len(_df_res)
                    if _zdrift_15m is not None:
                        print(f"  [zdrift_15m] fallback z_drift={_zdrift_15m:+.4f}  ({_n_res} resolved)")
                    else:
                        print(f"  [zdrift_15m] fallback insufficient data ({_n_res} resolved, need 10)")
                except Exception as _ze:
                    print(f"  [zdrift_15m] fallback error: {_ze}")
    sig["p_up_v2_btc"] = _p_up_v2_btc if _p_up_v2_btc is not None else ""

    # Rolling 6h z_drift logged for all assets (LGBM feature)
    _z_drift_6h: Optional[float] = None
    try:
        _csv_15m_path = _csv_path(asset)
        if _csv_15m_path.exists():
            _df_log = pd.read_csv(_csv_15m_path, low_memory=False)
            _df_log_res = _df_log[
                _df_log["resolved_yes"].notna() &
                (_df_log["resolved_yes"].astype(str) != "") &
                _df_log["spot_at_expiry"].notna()
            ]
            _z_drift_6h = compute_z_drift_6h(_df_log_res)
    except Exception as _zdex:
        print(f"  [z_drift_6h] error: {_zdex}")
    sig["z_drift_6h"] = _z_drift_6h if _z_drift_6h is not None else ""
    if _z_drift_6h is not None:
        print(f"  [z_drift_6h] {_z_drift_6h:+.4f}  (6h rolling mean actual_z)")

    # Vol-regime HMM state (BTC only, shadow-logged — no gate applied yet)
    _vol_state: "int | None" = None
    if asset == "BTC" and live_1m is not None:
        _vol_state = _vol_hmm_state(live_1m)
    sig["hmm_vol_state"] = _vol_state if _vol_state is not None else ""
    if _vol_state is not None:
        _rank_labels = {0: "low-vol-bear", 1: "low-vol-bull", 2: "high-vol"}
        print(f"  [vol_hmm] rank={_vol_state}  ({_rank_labels.get(_vol_state, '?')})")

    def _fmt(v, fmt=".3f"):
        return format(v, fmt) if isinstance(v, (int, float)) and v == v else "n/a"

    print(f"  Signals:")
    print(f"    [1h primary] chg_1h={_fmt(sig.get('chg_1h', 0), '+.3f')}%  "
          f"bp_1h={_fmt(sig.get('bp_1h'))}  dir_1h={sig.get('dir_1h', 'n/a')}  "
          f"streak={sig.get('consec_dir_1h', 0):.0f}")
    print(f"    [1h primary] stoch_k_1h={_fmt(sig.get('stoch_k_1h', 50), '.1f')}  "
          f"stoch_cross={sig.get('stoch_cross_1h', 0):.0f}  "
          f"dc_breakout={sig.get('donchian_breakout_1h', 0):.0f}  "
          f"engulfing={sig.get('engulfing_1h', 0):.0f}")
    z_1h = _compute_1h_drift(sig, 5.0)  # show at tau=5 reference
    print(f"    [1h drift@tau=5] z_1h={z_1h:+.3f}  → p_model shift≈{float(norm.cdf(z_1h))-0.5:+.3f}")
    print(f"    [5m/15m context] bp_5m={_fmt(sig.get('bp_5m'))}  "
          f"stoch_k_5m={_fmt(sig.get('stoch_k_5m', 50), '.1f')}  "
          f"body_15m={_fmt(sig.get('body_15m'))}  dir_15m={sig.get('dir_15m', 'n/a')}")
    print(f"    chg_5m={_fmt(sig.get('chg_5m', 0), '+.3f')}%  "
          f"chg_15m={_fmt(sig.get('chg_15m', 0), '+.3f')}%  "
          f"ema_bias={sig.get('ema_bias', 0)}  "
          f"realized_vol={_fmt(sig.get('realized_vol_annual', 0), '.2%')}")

    # HMM multi-timeframe regime state (BTC only; shadow for ETH/SOL)
    _hmm_state = _hmm_predict_state(sig) if asset == "BTC" else -1
    if asset == "BTC":
        _st_label = {0:"DIVERGE(block)", 7:"HIGH-PM-NO(boost)"}.get(_hmm_state, str(_hmm_state))
        print(f"    [hmm_state] {_hmm_state}  ({_st_label})")

    # Fetch Coinalyze liquidation + OI signal (cached 5 min; None for SOL if unavailable)
    _liq_signal = coinalyze_liq.fetch_liq_signal(asset)
    if _liq_signal is not None:
        print(f"    liq_bias={_liq_signal.liq_bias:+.2f}  long={_liq_signal.ls_long_pct:.1f}%  "
              f"short={_liq_signal.ls_short_pct:.1f}%  score={_liq_signal.liq_score:+d}"
              f"  [{_liq_signal.label}]  oi_chg={_liq_signal.oi_chg_pct:+.3f}%")
    else:
        print(f"    [liq_signal] unavailable for {asset}")

    # --- CoinGlass signals: exchange flows, options OI, spot taker, fear & greed ---
    _cg = coinglass_data.fetch_coinglass_signals(asset)
    if _cg is not None:
        print(f"    [coinglass] flow={_cg.exchange_flow_1d:+.0f} ({_cg.exchange_flow_1d_pct:+.3f}%/d)"
              f"  options_oi_chg={_cg.options_oi_change_24h:+.1f}%"
              f"  taker={_cg.spot_taker_ratio:.3f}"
              f"  F&G={_cg.fg_value:.0f}({_cg.fg_regime})"
              f"  composite={_cg.composite_score:+d}")

    if auth is None:
        print("  [warn] No Kalshi auth — contract scan skipped.")
        return

    # Fetch active 15m contracts
    contracts = fetch_15m_contracts(auth, asset=asset)
    if not contracts:
        print(f"  [scan] No liquid {series} contracts with tau in [{MIN_TAU_MIN}, {MAX_TAU_MIN}] min.")
        return

    print(f"  [scan] Found {len(contracts)} liquid contract(s).")
    decision_time = datetime.now(timezone.utc).isoformat()
    csv_name = ASSET_CONFIG[asset.upper()]["csv_name"]

    # [z_drift_6h_no_gate — BTC NO suppression in extreme uptrend]
    # When z_drift_6h > 2.5 the model's p_no is structurally unreliable: high p_market
    # drives implied vol down → blended sigma_tau shrinks → p_no collapses → fake NO edge.
    # Simulation (205 resolved BTC 15m trades, May 25-30):
    #   z_drift > 2.5 blocked: n=16, WR=6.3%, PnL=-$539  → 15 losses avoided, 1 win sacrificed.
    #   All bets kept (z_drift ≤ 2.5): n=189, WR=37.6%, PnL=+$667 vs baseline +$128.
    # Backup: paper_trade_runner_15m_pre_zdrift_no_gate_20260530.py
    if asset.upper() == "BTC" and _z_drift_6h is not None and _z_drift_6h > 2.5:
        print(f"  [z_drift_no_gate] SKIP SCAN — z_drift_6h={_z_drift_6h:+.4f} > 2.5 "
              f"(extreme uptrend: model p_no unreliable, sim WR=6.3% in this regime)")
        return

    # ── Pass 1: evaluate all contracts, pick the single best edge ─────────────
    # For each contract determine the better side (YES or NO), then among all
    # qualifying contracts place exactly ONE bet on the one with the highest edge.
    candidates = []   # (edge, side, c, p_model, offset_pct)
    evaluated  = []   # same tuple for every contract (including non-qualifiers)
    _lgbm_shadows: dict = {}  # ticker → shadow LGBM p_yes (logged as p_gbdt)

    for c in contracts:
        ticker     = c["ticker"]
        floor_s    = c["floor_strike"]
        p_market   = c["p_market"]
        tau_min    = c["tau_minutes"]
        close_time = c["close_time"]
        offset_pct = (spot - floor_s) / floor_s * 100

        # Compute p_model before already_bet check so scan archive captures all contracts.
        # BTC: p_up_v2 τ-scaled drift (k_yes=1.40, k_no=1.56); fallback to z_drift / LGBM.
        if asset == "BTC" and _p_up_v2_btc is not None:
            p_model_yes = compute_p_yes_pup_v2_15m(spot, floor_s, tau_min, sig, _p_up_v2_btc, p_market)
            p_model_no  = compute_p_no_pup_v2_15m(spot, floor_s, tau_min, sig, _p_up_v2_btc, p_market)
        else:
            p_model_no  = compute_p_model_15m(spot, floor_s, tau_min, sig, asset=asset, p_market=p_market)
            if asset == "BTC" and _zdrift_15m is not None:
                p_model_yes = compute_p_yes_zdrift_15m(spot, floor_s, tau_min, sig, _zdrift_15m, p_market)
            else:
                p_model_yes = p_model_no

        # Shadow LGBM: always run lgbm_15m_{asset}.pkl regardless of primary model path.
        # For BTC this is separate from p_up_v2; for ETH/SOL it equals p_model_no (same model).
        _lgbm_shadows[ticker] = compute_p_model_15m(
            spot, floor_s, tau_min, sig, asset=asset, p_market=p_market)

        # Scan archive: log all evaluated contracts before any skips.
        try:
            import scan_archive_15m as _sa15
            _sa15.log_scan_row(
                ticker=ticker, close_ts=close_time,
                spot=spot, strike=floor_s, p_market=p_market,
                tau_minutes=tau_min, spread=c["ask"] - c["bid"],
                p_model_yes=p_model_yes, p_model_no=p_model_no,
                features={
                    **sig,
                    "offset_pct":    offset_pct,
                    "liq_score":     _liq_signal.liq_score    if _liq_signal else float("nan"),
                    "liq_bias":      _liq_signal.liq_bias     if _liq_signal else float("nan"),
                    "oi_chg_pct":    _liq_signal.oi_chg_pct   if _liq_signal else float("nan"),
                    "ls_long_pct":   _liq_signal.ls_long_pct  if _liq_signal else float("nan"),
                    "fear_greed":    _cg.fg_value              if _cg else float("nan"),
                    "cg_composite":  _cg.composite_score       if _cg else float("nan"),
                },
                asset=asset,
                now_utc=datetime.now(timezone.utc),
            )
        except Exception:
            pass

        if already_bet is not None and close_time in already_bet:
            print(f"  [skip] expiry {close_time} already bet this session.")
            continue

        edge_yes = p_model_yes - p_market
        # [edge_no fix 2026-06-22] The legacy formula `p_market - p_model_no` is the WRONG sign on
        # p_model_no (it rewards LOW model-NO-prob = near-money, penalizes HIGH = deep-OTM winners).
        # It only ever "worked" by canceling a p_model_no miscalibration. The 06-16 BTC regime-pup
        # reform CALIBRATED BTC's p_model_no -> the legacy formula broke for BTC (selected near-money
        # NO at WR 22-40%, -$1,005 over 06-21/22). The correct EV is p_model_no - (1-p_market), which
        # the HOURLY runner already uses (line ~4035). Validation on calibrated data (06-17+):
        #   BTC: OLD -$1,423 -> TRUE +$887 (deep-OTM re-selected). FIX BTC.
        #   ETH: OLD +$3,012 -> TRUE -$4,740 (ETH p_model NOT recalibrated -> legacy still matches).
        #   SOL: both lose (separate p_model problem).
        # So asset-aware: TRUE for BTC, legacy for ETH/SOL until their p_model is recalibrated.
        # Backup: paper_trade_runner_15m_pre_edge_no_fix_20260622.py
        if asset == "BTC":
            edge_no = p_model_no - (1.0 - p_market)   # correct EV (matches hourly _no_edge)
        else:
            edge_no = p_market - p_model_no           # legacy (ETH/SOL p_model uncalibrated; see above)

        # Best side is the one with higher (and positive) edge
        if edge_yes >= edge_no:
            best_side, best_edge = "yes", edge_yes
            p_model = p_model_yes
        else:
            best_side, best_edge = "no", edge_no
            p_model = p_model_no

        # [eth_markov_daily_sideways_gate — hard block ALL ETH trades when daily regime=Sideways]
        # Backtest: 114 trades WR=37.7% -$793; subsumes ETH 4h Bull (100% overlap).
        # No profitable rescue found (n<30 in all rescue subgroups); revisit at n≥30 for NO.
        if asset.upper() == "ETH" and _markov_eth_daily == "Sideways":
            print(f"    [eth_daily_sw_gate] BLOCK {best_side.upper()} → skip — "
                  f"ETH daily Markov=Sideways (n=114, WR=37.7%, -$793)")
            evaluated.append((best_edge, best_side, c, p_model, offset_pct))
            continue

        # ETH YES gate: block overbought YES bets (stoch_k_5m >= 44)
        if asset.upper() == "ETH" and best_side == "yes":
            stoch = sig.get("stoch_k_5m", 50.0)
            if stoch >= 44:
                best_side, best_edge = "no", edge_no

        # [eth_15m_yes_lowvol_gate] Block YES when vol_ratio<0.80 + pm<0.65 unless OTM+cpu>=0.45.
        # Analysis (2026-05-23, n=82): WR=34.1%, BE=45.5%, P&L=-$644.
        # offset<-0.10% deep-ITM (n=29): WR=10.3%, -$551 — structurally unwinnable in low-vol.
        # Rescue: offset_pct>=0 (ITM) + composite_p_up>=0.45 → n=30, WR=60.0%, +$182.
        # Best rescue: OTM + cpu>=0.45 + ema_bias=-1 → n=24, WR=66.7%, +$241.
        # Blocked remainder: WR=19.2%, -$825.
        if asset.upper() == "ETH" and best_side == "yes":
            _vr_eth = float(sig.get("vol_ratio", 1.0) or 1.0)
            if _vr_eth < 0.80 and p_market < 0.65:
                _cpu_eth = sig.get("composite_p_up")
                _cpu_eth_f = float(_cpu_eth) if _cpu_eth is not None else 0.0
                _lowvol_rescue = (offset_pct >= 0.0 and _cpu_eth_f >= 0.45)
                if not _lowvol_rescue:
                    print(f"    [eth_15m_yes_lowvol_gate] BLOCK YES→NO {ticker} — "
                          f"vol_ratio={_vr_eth:.2f}<0.80, pm={p_market:.3f}<0.65, "
                          f"offset={offset_pct:+.3f}%, cpu={_cpu_eth_f:.3f} "
                          f"(rescue needs offset>=0+cpu>=0.45)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [eth_15m_yes_lowvol_gate] RESCUE YES {ticker} — "
                          f"vol_ratio={_vr_eth:.2f}<0.80 but offset={offset_pct:+.3f}%>=0"
                          f"+cpu={_cpu_eth_f:.3f}>=0.45 (WR=60-67%)")

        # [eth_15m_yes_lowcpu_gate] Hard block YES when composite_p_up<0.40.
        # Analysis (2026-05-23, n=27): WR=33.3%, BE=63.3%, P&L=-$348.
        # Calibration: cpu<0.35 → WR=23.1%; cpu 0.35-0.40 → WR=42.9% — both far below 63% BE.
        # No rescue found across all features — model is systematically miscalibrated here.
        if asset.upper() == "ETH" and best_side == "yes":
            _cpu_eth2 = sig.get("composite_p_up")
            if _cpu_eth2 is not None and float(_cpu_eth2) < 0.40:
                print(f"    [eth_15m_yes_lowcpu_gate] BLOCK YES→NO {ticker} — "
                      f"composite_p_up={float(_cpu_eth2):.3f}<0.40 "
                      f"(model miscalibrated, WR=33% vs BE=63%, no rescue)")
                best_side, best_edge = "no", edge_no

        # [eth_15m_yes_stoch1h_gate] Block YES when 1h stoch in mid-range [30,70); rescue if rsi_1h<35.
        # Full paper trade history (n=953 with stoch, 2026-05-11 to 2026-06-01):
        #   stoch[30,70) + rsi>=35: n=59, WR=40.7%, BE≈63%, P&L=-$376 → saves +$376.
        #   stoch[30,70) + rsi<35 rescue: n=8, WR=100%, P&L=+$97 → kept.
        # Causal: mid-range stoch = no momentum conviction for YES. Edge only exists when
        # stoch<30 (oversold bounce) or stoch≥70 (momentum continuation).
        # rsi<35 rescue: deeply oversold RSI diverges from mid stoch → convergence signal.
        # Scan archive confirms: 218 blocked at 40.8% WR. Consistent Wk21-23.
        if asset.upper() == "ETH" and best_side == "yes":
            _sk1h_eth = float(sig.get("stoch_k_1h", 50.0) or 50.0)
            if 30.0 <= _sk1h_eth < 70.0:
                _rsi1h_raw = sig.get("rsi_1h")
                _rsi1h_f = float(_rsi1h_raw) if _rsi1h_raw is not None else 50.0
                _stoch1h_rescue = (_rsi1h_raw is not None and _rsi1h_f < 35.0)
                if not _stoch1h_rescue:
                    print(f"    [eth_15m_yes_stoch1h_gate] BLOCK YES→NO {ticker} — "
                          f"stoch_k_1h={_sk1h_eth:.1f}∈[30,70), rsi_1h={_rsi1h_f:.1f} "
                          f"(no conviction; rescue needs rsi<35)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [eth_15m_yes_stoch1h_gate] RESCUE YES {ticker} — "
                          f"stoch_k_1h={_sk1h_eth:.1f}∈[30,70) but rsi_1h={_rsi1h_f:.1f}<35 "
                          f"(oversold RSI divergence, WR=100% n=8)")

        # [BTC YES gate] Gate 3 only — offset flip.
        # Gates 1 (ema=-1 flip) and 2 (pm<0.40 flip) removed 2026-05-23:
        # Audit on 387 scan archive rows showed payout asymmetry destroys value.
        # At low p_market, YES wins pay 3-6x more than NO wins; blocking rare YES winners
        # costs more than the gate recovers regardless of win-rate improvement.
        # Gate 1 cost: -$1,309 vs baseline. Gate 2 cost: -$1,507 vs baseline.
        if asset.upper() == "BTC" and best_side == "yes":
            # [btc_15m_lowpm_gate] Block BTC YES when pm<0.35; rescue when 1h=Sideways AND tau≥10.
            # Analysis (2026-05-23, n=51 resolved YES at pm<0.35):
            #   Overall: WR=17.6%, BE=24.4%, PnL=-$576.
            #   markov_15m=Bear (n=15): WR=0.0%, -$363 — zero wins across every sub-slice.
            #   markov_1h=Bear (n=12): WR=8.3%, -$155 — nearly as bad.
            # Rescue: 1h=Sideways + tau≥10 → n=15, WR=46.7%, Edge=+17.5%, PnL=+$125.
            #   Causal: BTC not in macro downtrend, enough time for price to reach strike.
            #   ema=1 looked good (+$137) but shares 9/10 trades with tau≥10 — OR adds 1 losing trade.
            # Gate blocks n=35 (wins=2, losses=33, saves $671); rescue allows n=16 (+$95).
            # Net vs flat block: +$190 better (rescue earns $95 on top of savings).
            if best_side == "yes" and p_market < 0.35:
                _lowpm_rescue = (
                    _markov_1h == "Sideways"
                    and tau_min >= 10.0
                )
                if not _lowpm_rescue:
                    print(f"    [btc_15m_lowpm_gate] BLOCK YES→NO {ticker} — "
                          f"pm={p_market:.3f}<0.35, 1h_regime={_markov_1h}, tau={tau_min:.1f}min "
                          f"(rescue requires 1h=Sideways+tau≥10)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [btc_15m_lowpm_gate] RESCUE YES {ticker} — "
                          f"pm={p_market:.3f}<0.35 but 1h=Sideways+tau={tau_min:.1f}≥10 "
                          f"(WR=46.7% historically)")

            # Gate 3: insufficient ITM YES — block when offset_pct < 0.025%
            # Sim (1117 15m trades, split-half validated May 11-22):
            #   cutoff=+0.025 → Total=+$1,056 vs baseline +$334
            #   First half: +$652, Second half: +$404 (YES PnL=+$56, profitable)
            #   OTM/barely-ITM YES has no edge under any indicator condition tested.
            # No rescue — analysis found no indicator combination makes offset<0.025 YES profitable.
            if best_side == "yes" and offset_pct < 0.025:
                print(f"    [btc_yes_gate3] BLOCK YES → flip NO — "
                      f"offset_pct={offset_pct:+.3f}% < 0.025% ITM threshold")
                best_side, best_edge = "no", edge_no

            # [markov_1h_bear_gate — BTC YES hard block]
            # Gate A: 1h Bear YES → hard block; no rescue found profitable.
            # Backtest: 105 trades WR=48.6% -$758; hard block saves $783.
            # Best single-feature rescue (pm≥0.65 WR=73%) still barely above breakeven.
            if best_side == "yes" and _markov_1h == "Bear":
                print(f"    [markov_1h_bear_gate] BLOCK YES → flip NO — "
                      f"1h Markov=Bear (n=105 WR=48.6%, no profitable rescue)")
                best_side, best_edge = "no", edge_no

            # [markov_15m_bear_gate — BTC YES conditional block]
            # Gate B: 15m Bear YES → block unless composite_p_up ≤ 0.488 (mean-reversion rescue).
            # Backtest: 68 trades WR=47.1% -$529.
            #   p_up > 0.488 (trend-fighting): WR=27.7%, hard block saves $554.
            #   p_up ≤ 0.488 (subdued model + Bear regime): WR=75%, P&L=+$72 — keep.
            if best_side == "yes" and _markov_15m == "Bear":
                _cpu = sig.get("composite_p_up")
                if _cpu is None or float(_cpu) > 0.488:
                    print(f"    [markov_15m_bear_gate] BLOCK YES → flip NO — "
                          f"15m Markov=Bear AND composite_p_up={_cpu} > 0.488 (trend-fighting)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [markov_15m_bear_gate] RESCUE YES — 15m Markov=Bear but "
                          f"composite_p_up={float(_cpu):.3f} ≤ 0.488 (mean-reversion, WR=75% expected)")

            # [btc_15m_stoch_overbought_gate] Block YES when stoch_k_15m>=80 AND 15m regime != Bull.
            # Analysis (2026-05-23, n=120): WR=50.8%, BE=59.8%, Edge=-9.0%, PnL=-$535.
            # markov_15m=Bull (n=45): WR=62.2%, Edge=+0.6%, +$35 — overbought in Bull=momentum.
            # markov_15m=Sideways (n=62): WR=45.2%, Edge=-15.0%, -$471 — overbought=exhaustion.
            # Block n=75, saves $574; rescue n=45, earns +$35; net +$605 vs flat block +$535.
            _stoch15 = float(sig.get("stoch_k_15m", 50.0) or 50.0)
            if best_side == "yes" and _stoch15 >= 80.0:
                _stoch_rescue = (_markov_15m == "Bull")
                if not _stoch_rescue:
                    print(f"    [btc_15m_stoch_overbought_gate] BLOCK YES→NO {ticker} — "
                          f"stoch_k_15m={_stoch15:.1f}>=80, 15m_regime={_markov_15m} "
                          f"(overbought+non-Bull=exhaustion, WR=45-33%)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [btc_15m_stoch_overbought_gate] RESCUE YES {ticker} — "
                          f"stoch_k_15m={_stoch15:.1f}>=80 but 15m=Bull "
                          f"(overbought in Bull=momentum, WR=62.2%)")

            # [btc_15m_upcandle_midpm_gate] Block YES when dir=1+pm=[0.50,0.65) AND 1h != Bull.
            # Analysis (2026-05-23, n=71): WR=45.1%, BE=58.1%, Edge=-13.0%, PnL=-$430.
            # markov_1h=Bull (n=12): WR=66.7%, Edge=+7.1%, +$28 — trend supports up-candle YES.
            # markov_1h=Bear+15m=Bear (n=11): WR=63.6%, +$13 — all have stoch_1h<35 (oversold).
            # markov_1h=Bear+15m=Sideways (n=10): WR=0.0%, -$285 — death zone.
            # Block n=48, saves $471; rescue n=23, WR=65.2%, earns +$41.
            _dir15 = sig.get("dir_15m", 0)
            if (best_side == "yes"
                    and _dir15 == 1
                    and 0.50 <= p_market < 0.65):
                _stoch_1h = float(sig.get("stoch_k_1h", 50.0) or 50.0)
                _upcandle_rescue = (
                    _markov_1h == "Bull"
                    or (_markov_1h == "Bear" and _markov_15m == "Bear" and _stoch_1h < 35.0)
                )
                if not _upcandle_rescue:
                    print(f"    [btc_15m_upcandle_midpm_gate] BLOCK YES→NO {ticker} — "
                          f"dir=1, pm={p_market:.3f}∈[0.50,0.65), 1h={_markov_1h}, 15m={_markov_15m} "
                          f"(momentum spent, WR=43-33%)")
                    best_side, best_edge = "no", edge_no
                else:
                    _rsrc = "1h=Bull" if _markov_1h == "Bull" else f"Bear/Bear+stoch_1h={_stoch_1h:.0f}<35 (oversold bounce)"
                    print(f"    [btc_15m_upcandle_midpm_gate] RESCUE YES {ticker} — "
                          f"dir=1+pm={p_market:.3f}: {_rsrc} (WR=63-67%)")

        # [btc_15m_smallbody_sideways_gate] Block BTC YES when body_15m<0.30 AND 1h=Sideways.
        # Analysis (2026-05-24, n=54 Sideways_1h): WR=42.6%, BE=56.7%, GAP=-14.1%, P&L=-$477.
        # Bear_1h (n=17, +$15) excluded — marginally profitable; Bull_1h (n=11, +$6) excluded.
        # Bear_1h trades already handled by markov_1h_bear_gate (flipped to NO before this check).
        # Rescue: composite_p_up<0.40 OR stoch_k_15m∈[20,40) → n=13, WR=69.2%, +$22.
        # Non-rescued: n=41, WR=34.1%, GAP=-22.6%, P&L=-$499 — hard block.
        if asset.upper() == "BTC" and best_side == "yes" and _markov_1h == "Sideways":
            _body15_btc = float(sig.get("body_15m", 1.0) or 1.0)
            if _body15_btc < 0.30:
                _cpu_btc  = sig.get("composite_p_up")
                _sk15_btc = float(sig.get("stoch_k_15m", 50.0) or 50.0)
                _smallbody_rescue = (
                    (_cpu_btc is not None and float(_cpu_btc) < 0.40)
                    or 20.0 <= _sk15_btc < 40.0
                )
                if not _smallbody_rescue:
                    print(f"    [btc_15m_smallbody_sideways_gate] BLOCK YES→NO {ticker} — "
                          f"body_15m={_body15_btc:.2f}<0.30, 1h=Sideways, "
                          f"cpu={_cpu_btc}, stoch_k_15m={_sk15_btc:.1f} "
                          f"(indecisive candle in sideways regime, WR=34.1% vs BE=56.7%)")
                    best_side, best_edge = "no", edge_no
                else:
                    _rsrc_sb = ("cpu<0.40" if _cpu_btc is not None and float(_cpu_btc) < 0.40
                                else f"stoch_k_15m={_sk15_btc:.1f}∈[20,40)")
                    print(f"    [btc_15m_smallbody_sideways_gate] RESCUE YES {ticker} — "
                          f"body<0.30+1h=Sideways but {_rsrc_sb} (WR=69.2%)")

        # [btc_15m_yes_overbought_liq_gate] Block YES when stoch_k_1h>=95 AND liq_score=-1.
        # Archive (n=17, 5 distinct episodes, 2026-05-25→2026-06-11):
        #   WR=5.9%, BE=36.0%, edge=-30.1%, binomial p=0.005.
        # Causal: max-overbought 1h + active long liquidations = no remaining upside fuel.
        # No rescue: all pm/vol sub-buckets lose (pm<0.60: 0% WR, vol>=1.2: 0% WR). Pure block.
        if asset.upper() == "BTC" and best_side == "yes":
            _sk1h_obliq = float(sig.get("stoch_k_1h", 50.0) or 50.0)
            _liq_obliq  = sig.get("liq_score")
            if _sk1h_obliq >= 95.0 and _liq_obliq is not None and float(_liq_obliq) == -1:
                print(f"    [btc_15m_yes_overbought_liq_gate] BLOCK YES→NO {ticker} — "
                      f"stoch_k_1h={_sk1h_obliq:.1f}>=95 + liq_score=-1 "
                      f"(max-overbought+long-liq, WR=5.9% vs BE=36.0%, p=0.005)")
                best_side, best_edge = "no", edge_no

        # [btc_15m_hmm_state0_gate] Block BTC NO when HMM regime = State 0.
        # State 0: stoch_k_5m=66 (neutral-high) diverges from stoch_k_1h=11 (deeply oversold).
        # Sim (N=9, WR=22%, -$174): 7 losses blocked, 2 wins blocked.
        # Causal: 5m momentum fighting a deeply oversold 1h = potential reversal, NO has no anchor.
        if asset.upper() == "BTC" and best_side == "no" and _hmm_state == 0:
            print(f"    [btc_15m_hmm_state0_gate] BLOCK NO {ticker} — "
                  f"HMM State 0 (stoch_5m/1h divergence, WR=22% historically)")
            evaluated.append((best_edge, best_side, c, p_model, offset_pct))
            continue

        # [btc_15m_no_uptrend_neutral_gate] Block BTC NO when chg_1h>0 AND stoch_k_1h∈[30,70).
        # Deep gate analysis (2026-05-29, 143 resolved BTC 15m NO trades):
        #   chg_1h>0 + stoch_1h[30,70): N=22, WR=27.3%, PnL=-$288 → delta +$288
        #   Kept (neither): N=121, WR=42.1%, PnL=+$1,274
        # Causal: BTC rose over the last hour AND 1h momentum is neutral (not oversold).
        # No structural anchor for a reversal — uptrend intact, no oversold signal to
        # expect a bounce back below the strike. No rescue found (stoch<20 AND chg_1h>0
        # are contradictory — can't be oversold while simultaneously rising on the hour).
        if asset.upper() == "BTC" and best_side == "no":
            _chg_1h_btc  = float(sig.get("chg_1h",     0.0) or 0.0)
            _sk1h_btc_no = float(sig.get("stoch_k_1h", 50.0) or 50.0)
            if _chg_1h_btc > 0 and 30.0 <= _sk1h_btc_no < 70.0:
                print(f"    [btc_15m_no_uptrend_neutral_gate] BLOCK NO {ticker} — "
                      f"chg_1h={_chg_1h_btc:+.3f}%>0 + stoch_k_1h={_sk1h_btc_no:.1f}∈[30,70) "
                      f"(uptrend + neutral 1h, no reversal anchor, N=22 WR=27% PnL=-$288)")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue

        # [btc_15m_no_sideways_overbought_gate] Block BTC NO when markov_1h=Sideways + pm>=0.70 + stoch>=70.
        # Backtest (scan archive May 21-28 + paper trades May 25-Jun 1, combined n=121 gate candidates):
        #   All 3 conditions (block zone): n=44, WR=6.8%, P&L=-$981 → saves +$981.
        #   stoch<70 kept (rescue): n=77, WR=22.1%, P&L=+$1,271.
        #   Scan archive total NO: +$665 improvement. Paper trades: +$317 improvement.
        #   Weekly consistency: Wk21 block WR=10% (-$187), Wk22 block WR=6% (-$795) — both bad.
        #   Impossible rescue check: no sub-slice (n>=5) inside block reaches WR>=30%.
        # Causal: Sideways hourly regime (no trend) + near-ATM strike (pm>=0.70) + overbought
        # 1h stoch (>=70) = upward momentum actively pushing toward a nearby strike with no
        # directional anchor. Without overbought stoch, Sideways NO at high pm can still win
        # (stoch<70 keep zone: WR=22.1%, profitable via payout asymmetry at high p_market).
        if asset.upper() == "BTC" and best_side == "no":
            _sk1h_sw = float(sig.get("stoch_k_1h", 0.0) or 0.0)
            if (_markov_1h == "Sideways"
                    and p_market >= 0.70
                    and _sk1h_sw >= 70.0):
                print(f"    [btc_15m_no_sideways_overbought_gate] BLOCK NO {ticker} — "
                      f"markov=Sideways, pm={p_market:.3f}>=0.70, stoch_k_1h={_sk1h_sw:.1f}>=70 "
                      f"(overbought momentum into near-ATM strike, WR=6.8% historically)")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue

        # [btc_15m_no_sideways_sideways_gate] Block BTC NO when 1h=Sideways AND 15m=Sideways.
        # Paper trades (n=42): WR=19.0% vs BE≈31.8%, PnL=-$722 → block saves +$722.
        # Sub-buckets: pm<0.55 (n=3, WR=100%) rescued; pm 0.55-0.70 (WR=18%) + pm≥0.70 (WR=6%) blocked.
        # pm≥0.70 already partially caught by btc_15m_no_sideways_overbought_gate (stoch≥70 required);
        # this gate closes the gap for mid-pm range and stoch<70 Sideways cases.
        # Causal: both timeframes directionless → no momentum anchor; BTC drifts through YES strikes.
        if asset.upper() == "BTC" and best_side == "no":
            if _markov_1h == "Sideways" and _markov_15m == "Sideways" and p_market >= 0.55:
                print(f"    [btc_15m_no_sideways_sideways_gate] BLOCK NO {ticker} — "
                      f"1h=Sideways, 15m=Sideways, pm={p_market:.3f}>=0.55 "
                      f"(no directional anchor on either TF, WR=19% historically)")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue

        # [btc_15m_no_bear_bull_gate] Block BTC NO when 1h=Bear AND 15m=Bull (regime divergence).
        # Paper trades (n=19): WR=26.3% vs BE≈50.4%, PnL=-$322 → block saves +$322.
        # No sub-bucket above breakeven: pm<0.55 (WR=38.5% vs BE=58.5%), all others worse.
        # Causal: 1h downtrend but 15m momentum already turning up → mini-rally carries BTC above
        # strike before the larger trend resumes. NO bets lose to the bounce.
        if asset.upper() == "BTC" and best_side == "no":
            if _markov_1h == "Bear" and _markov_15m == "Bull":
                print(f"    [btc_15m_no_bear_bull_gate] BLOCK NO {ticker} — "
                      f"1h=Bear, 15m=Bull (regime divergence: 15m bounce kills NO bets, "
                      f"WR=26% vs BE=50% historically)")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue

        # [btc_15m_no_highpm_gate] Block NO when pm>0.85 (market deep ITM YES).
        # Paper trades n=18: WR=5.6% vs BE=13.5%, PnL=-$420.
        # liq_score=-1 does NOT rescue here (0/7 wins) — block all regardless of liq.
        # Cause: inflated edge formula (p_market - p_model_no) produces phantom 70%+ edges
        # on contracts where BTC is far above the strike; true edge is <1% but Kelly sizes ~$60.
        if asset.upper() == "BTC" and best_side == "no" and p_market > 0.85:
            print(f"    [btc_15m_no_highpm_gate] BLOCK NO {ticker} — "
                  f"pm={p_market:.3f}>0.85 (WR=5.6% vs BE=13.5%, PnL=-$420 historically)")
            evaluated.append((best_edge, best_side, c, p_model, offset_pct))
            continue

        # [btc_15m_overbought_momentum_no_gate] Block NO when 5m overbought AND still ticking up.
        # Deep gate analysis 2026-06-21 (comprehensive sweep of all signals + OOS archive validation):
        #   stoch_k_5m>76 & chg_5m>0 → live up-momentum runs the downside NO bet over.
        #   ARCHIVE (deduped OTM-NO, OOS from taken): NO-EV=-0.105 vs -0.010 base, MCPT p=0.0002,
        #     5/6 weeks negative. TAKEN n=106: WR=25%, -$1,475, negative EVERY week incl wk25
        #     (survives the selection test). Blocking lifts BTC 15m NO PnL +$1,126 → +$2,601.
        #   RESCUE: chg_5m<=0 (overbought but stalling/turning → mean-reverts, NO-EV=+0.045,
        #     WR=62%) is NOT blocked — only the overbought+continuation pocket loses.
        # Causal: overbought 5m + live up-tick = momentum continuation; overbought + stall = reversion.
        if asset.upper() == "BTC" and best_side == "no":
            _s5 = sig.get("stoch_k_5m")
            _c5 = sig.get("chg_5m")
            try:
                if _s5 is not None and _c5 is not None and float(_s5) > 76.0 and float(_c5) > 0.0:
                    print(f"    [btc_15m_overbought_momentum_no_gate] BLOCK NO {ticker} — "
                          f"stoch_k_5m={float(_s5):.1f}>76 & chg_5m={float(_c5):+.3f}%>0 "
                          f"(5m overbought + up-momentum; NO-EV=-0.105 OOS p=0.0002, -$1,475 taken)")
                    evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                    continue
            except (TypeError, ValueError):
                pass

        # [btc_15m_no_midpm_liq_gate] Block NO when pm∈[0.65,0.75) unless liq_score=-1.
        # Paper trades breakdown (n=63, WR=23.8%, BE=30.8%, PnL=-$549 total):
        #   liq=-1 (genuine selling/liquidation): n=21, WR=47.6%, +$241 → RESCUE (keep)
        #   liq= 0 (neutral):                     n=13, WR=23.1%,  -$69 → block
        #   liq=+1 (buyers dominant):              n=27, WR= 7.4%, -$638 → block
        # Causal: when buyers dominate (liq=+1), BTC momentum stays bullish and NO fails.
        # Rescue: liquidation pressure (liq=-1) = genuine sellers compressing price → NO works.
        if asset.upper() == "BTC" and best_side == "no" and 0.65 <= p_market < 0.75:
            _liq_s = sig.get("liq_score")
            _liq_s = float(_liq_s) if _liq_s is not None else 0.0
            if _liq_s >= 0:
                print(f"    [btc_15m_no_midpm_liq_gate] BLOCK NO {ticker} — "
                      f"pm={p_market:.3f}∈[0.65,0.75), liq_score={_liq_s:.0f}≥0 "
                      f"(buyers dominant or neutral, WR=7-23% historically)")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue
            else:
                print(f"    [btc_15m_no_midpm_liq_gate] RESCUE NO {ticker} — "
                      f"pm={p_market:.3f}∈[0.65,0.75) BUT liq_score={_liq_s:.0f}=-1 "
                      f"(genuine liquidation pressure, WR=47.6% historically)")

        # [eth_no_consec_gate — 15m ETH NO]
        # Block NO when in a sustained bearish streak AND stochastic is already oversold.
        # Analysis (26 blocked at consec <= -1, 93 total ETH NO trades with feature):
        #   consec <= -1, stoch_k_15m <= 40 → WR=23%,   catastrophic (hard block: 13 trades)
        #   consec <= -1, stoch_k_15m >  40 → WR=69.2%, +$42         (rescue: 13 trades)
        # Rationale: sustained bearish streak + oversold stochastic = mean reversion likely
        # (price bounces back → YES wins → NO fails). If stochastic is NOT yet oversold,
        # the breakdown is genuine and NO can still work.
        # Note: n=26 is small — treat as provisional until 150+ NO trades with feature logged.
        if asset.upper() == "ETH" and best_side == "no":
            _cd15 = sig.get("consec_dir_15m")
            _sk15 = float(sig.get("stoch_k_15m", 50.0) or 50.0)
            if _cd15 is not None and _cd15 <= -1:
                _no_rescue = (_sk15 > 40.0)
                if not _no_rescue:
                    print(f"    [eth_no_consec_gate] BLOCK NO → flip YES — "
                          f"consec_dir_15m={_cd15:.0f}<=-1, stoch_k_15m={_sk15:.1f}<=40 "
                          f"(bearish streak + oversold = mean reversion expected)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    print(f"    [eth_no_consec_gate] RESCUE NO — "
                          f"consec_dir_15m={_cd15:.0f}<=-1 but stoch_k_15m={_sk15:.1f}>40 "
                          f"(genuine breakdown, NO allowed)")

        # [eth_15m_no_downcandle_gate] Block NO when dir_15m==-1+pm>=0.50 unless strong body+pressure.
        # Analysis (2026-05-23, n=71): WR=25.4%, BE=34.5%, P&L=-$612.
        # liq_score=-2 (n=42): WR=16.7%, -$704 — hard blocker regardless of other signals.
        # Rescue: body_15m>0.60 + bp_5m<0.45 → n=24, WR=50.0%, +$260.
        # Three-feature: body>0.60 + bp<0.45 + liq!=-2 → n=12, WR=75.0%, +$393.
        if asset.upper() == "ETH" and best_side == "no":
            _d15_eth = sig.get("dir_15m", 0)
            if _d15_eth == -1 and p_market >= 0.50:
                _body_eth = float(sig.get("body_15m", 0.0) or 0.0)
                _bp_eth   = float(sig.get("bp_5m", 0.5) or 0.5)
                _liq_eth  = sig.get("liq_score")
                _liq_eth_ok = (_liq_eth is None or _liq_eth != _liq_eth or float(_liq_eth) != -2)
                _downcandle_rescue = (_body_eth > 0.60 and _bp_eth < 0.45 and _liq_eth_ok)
                if not _downcandle_rescue:
                    print(f"    [eth_15m_no_downcandle_gate] BLOCK NO→YES {ticker} — "
                          f"dir=-1, pm={p_market:.3f}>=0.50, body={_body_eth:.2f}, "
                          f"bp={_bp_eth:.2f}, liq={_liq_eth} "
                          f"(rescue needs body>0.60+bp<0.45+liq!=-2)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    print(f"    [eth_15m_no_downcandle_gate] RESCUE NO {ticker} — "
                          f"dir=-1+pm={p_market:.3f}: body={_body_eth:.2f}>0.60"
                          f"+bp={_bp_eth:.2f}<0.45+liq={_liq_eth}!=-2 (WR=50-75%)")

        # [eth_15m_no_stoch1h_gate] Block NO when stoch_k_1h>80 AND no bullish K>D crossover.
        # Full paper trade history (n=953 with stoch, stoch_cross sub-analysis):
        #   stoch>80 + cross!=1: n=69, WR=50.7%, BE≈59%, P&L=-$383 → saves +$383.
        #   stoch>80 + cross=1 rescue: n=28, WR=78.6%, P&L=+$363 → kept.
        #   stoch>80 + cross=null: n=38, P&L=+$107 → gate skipped (no data).
        # Causal: stoch>80 + no fresh crossover = overbought grind without breakout signal →
        # price may drift into strike. cross=1 = K>D just fired = active momentum → NO still wins.
        # Scan archive confirms: 39 blocked at 5.1% WR. Consistent Wk21-22.
        if asset.upper() == "ETH" and best_side == "no":
            _sk1h_no = float(sig.get("stoch_k_1h", 0.0) or 0.0)
            _sc1h_no = sig.get("stoch_cross_1h")
            if _sk1h_no > 80.0 and _sc1h_no is not None:
                _sc1h_no_f = float(_sc1h_no or 0)
                if _sc1h_no_f != 1.0:
                    print(f"    [eth_15m_no_stoch1h_gate] BLOCK NO→YES {ticker} — "
                          f"stoch_k_1h={_sk1h_no:.1f}>80, stoch_cross_1h={_sc1h_no_f:.0f}!=1 "
                          f"(overbought, no momentum cross; WR=50.7% vs BE≈59%)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    print(f"    [eth_15m_no_stoch1h_gate] RESCUE NO {ticker} — "
                          f"stoch_k_1h={_sk1h_no:.1f}>80 but stoch_cross_1h=1 "
                          f"(fresh bullish K>D cross = active momentum, WR=78.6%)")

        # [eth_no_stoch_oversold_gate — Gate C]
        # Block ETH NO when 1h stoch oversold (<20) AND BTC 1h regime is non-Bear.
        # Analysis (paper_trades_eth15m.csv, MCPT p=0.036, n=113 resolved NO):
        #   sk1h<20 + non-Bear: flat_pnl=-$0.112/trade, would_pnl=-$515 Kelly-sized.
        # Causal: 1h stoch<20 = ETH oversold → bounce imminent → YES wins → NO fails.
        # Bear rescue: in BTC Bear regime, oversold = continued downtrend → NO still works.
        if asset.upper() == "ETH" and best_side == "no":
            _sk1h_gc = float(sig.get("stoch_k_1h", 50.0) or 50.0)
            if _sk1h_gc < 20.0 and _markov_1h != "Bear":
                print(f"    [eth_no_stoch_oversold_gate] BLOCK NO→YES {ticker} — "
                      f"stoch_k_1h={_sk1h_gc:.1f}<20 + BTC_1h={_markov_1h!r} (non-Bear) "
                      f"(oversold bounce → YES; MCPT p=0.036, n=113)")
                best_side, best_edge = "yes", edge_yes

        # [sol_markov_gates — block SOL contracts in adverse Markov regimes]
        # Gates (validated on 784 resolved SOL 15m trades, flat $25 bet):
        #   6h Bull YES: block unless stoch_cross_1h=0           (rescue n=13, WR=62%, +$152)
        #   6h Bull NO:  block unless offset_pct ≤ −0.006        (rescue n=43, WR=72%, +$79)
        #   4h Sideways YES: hard block (no profitable rescue)
        #   4h Sideways NO:  block unless stoch_k_1h ≥ 86.1      (rescue n=28, WR=79%, +$183)
        #   1h Sideways YES: block unless oi_chg_pct ≥ 0.0535    (rescue n=43, WR=63%, +$145)
        # Rescues are OR-combined: any rescue condition saves the contract regardless of
        # which gate triggered the block (matches simulation Scen 2: Δ+$1,951, net +$148).
        if asset.upper() == "SOL":
            _sc1h = float(sig.get("stoch_cross_1h", 0) or 0)
            _sk1h = float(sig.get("stoch_k_1h", 50.0) or 50.0)
            _oi   = float((_liq_signal.oi_chg_pct if _liq_signal else None) or 0.0)
            _OFF_MED_SOL = -0.006  # median offset_pct from 120-day SOL backtest

            _gate_yes = (
                (_markov_sol_6h == "Bull"      and _sc1h != 0)
                or (_markov_sol_4h == "Sideways")
                or (_markov_sol_1h == "Sideways" and _oi < 0.0535)
            )
            _rescue_yes = (
                (_markov_sol_6h == "Bull"      and _sc1h == 0)
                or (_markov_sol_1h == "Sideways" and _oi >= 0.0535)
            )
            _gate_no = (
                (_markov_sol_6h == "Bull"      and offset_pct > _OFF_MED_SOL)
                or (_markov_sol_4h == "Sideways" and _sk1h < 86.1)
            )
            _rescue_no = (
                (_markov_sol_6h == "Bull"      and offset_pct <= _OFF_MED_SOL)
                or (_markov_sol_4h == "Sideways" and _sk1h >= 86.1)
            )

            _sol_skip = (
                (best_side == "yes" and _gate_yes and not _rescue_yes)
                or (best_side == "no"  and _gate_no  and not _rescue_no)
            )

            _regs = (f"6h={_markov_sol_6h or '?'} 4h={_markov_sol_4h or '?'} "
                     f"1h={_markov_sol_1h or '?'}")
            if _sol_skip:
                print(f"    [sol_markov_gate] BLOCK {best_side.upper()} → skip — {_regs}"
                      f"  sc={_sc1h:.0f} sk={_sk1h:.1f} oi={_oi:.4f} off={offset_pct:+.3f}%")
                evaluated.append((best_edge, best_side, c, p_model, offset_pct))
                continue
            elif best_side == "yes" and _gate_yes and _rescue_yes:
                _rsrc = ("stoch_cross_1h=0" if _markov_sol_6h == "Bull" and _sc1h == 0
                         else f"oi_chg={_oi:.4f}≥0.054")
                print(f"    [sol_markov_gate] RESCUE YES [{_rsrc}] ({_regs})")
            elif best_side == "no" and _gate_no and _rescue_no:
                _rsrc = (f"stoch_k_1h={_sk1h:.1f}≥86"
                         if _markov_sol_4h == "Sideways" and _sk1h >= 86.1
                         else f"offset={offset_pct:+.3f}%≤{_OFF_MED_SOL}")
                print(f"    [sol_markov_gate] RESCUE NO [{_rsrc}] ({_regs})")

        # [sol_15m_yes_stoch_cross_gate] Hard block YES when stoch_cross_1h != 0.
        # Analysis (2026-05-23, n=77): WR=40.3%, BE=55.7%, P&L=-$335.
        # stoch_cross_1h=+1: n=45, WR=44.4%, -$133. stoch_cross_1h=-1: n=32, WR=34.4%, -$203.
        # No rescue found above 58% at n>=10 across all tested features.
        if asset.upper() == "SOL" and best_side == "yes":
            if _sc1h != 0:
                print(f"    [sol_15m_yes_stoch_cross_gate] BLOCK YES→NO {ticker} — "
                      f"stoch_cross_1h={_sc1h:+.0f} (momentum reversing, "
                      f"n=77 WR=40.3% vs BE=55.7%, no rescue)")
                best_side, best_edge = "no", edge_no

        # [sol_15m_yes_oi_vwap_gate] Hard block YES when OI declining or below VWAP + low pm.
        # Analysis (2026-05-23, n=107): WR=24.3%, BE=36.5%, P&L=-$364.
        # oi_chg_pct<-0.005 (distribution): n=69, WR=21.7%. vwap_dist<-0.01 (below VWAP): n=80, WR=26.2%.
        # No rescue found — max WR=33.3% at n=9 across all tested features. Clean hard block.
        if asset.upper() == "SOL" and best_side == "yes":
            _oi_sol = float(sig.get("oi_chg_pct", 0) or 0)
            _vd_sol = float(sig.get("vwap_dist", 0) or 0)
            if (_oi_sol < -0.005 or _vd_sol < -0.01) and p_market < 0.55:
                print(f"    [sol_15m_yes_oi_vwap_gate] BLOCK YES→NO {ticker} — "
                      f"oi_chg={_oi_sol:.4f}, vwap_dist={_vd_sol:.4f}, pm={p_market:.3f} "
                      f"(distribution/below VWAP at low pm, WR=24.3% vs BE=36.5%, no rescue)")
                best_side, best_edge = "no", edge_no

        # [sol_15m_yes_highcpu_gate] Block YES when composite_p_up>0.55; rescue if bp_1h>0.55.
        # Analysis (2026-05-23, n=64): WR=40.6%, BE=53.6%, P&L=-$224.
        # Rescue bp_1h>0.55: n=10, WR=70.0%, +$29. Remainder n=54, WR=35.0%, -$240 — hard block.
        # composite_p_up>0.65: n=9, WR=0.0%, -$79 — worst sub-bucket.
        if asset.upper() == "SOL" and best_side == "yes":
            _cpu_sol = sig.get("composite_p_up")
            if _cpu_sol is not None and float(_cpu_sol) > 0.55:
                _bp1h_sol = float(sig.get("bp_1h", 0.5) or 0.5)
                _highcpu_rescue = (_bp1h_sol > 0.55)
                if not _highcpu_rescue:
                    print(f"    [sol_15m_yes_highcpu_gate] BLOCK YES→NO {ticker} — "
                          f"composite_p_up={float(_cpu_sol):.3f}>0.55, bp_1h={_bp1h_sol:.3f} "
                          f"(model overconfident, rescue needs bp_1h>0.55)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [sol_15m_yes_highcpu_gate] RESCUE YES {ticker} — "
                          f"cpu={float(_cpu_sol):.3f}>0.55 but bp_1h={_bp1h_sol:.3f}>0.55 "
                          f"(buy pressure confirms, WR=70.0%)")

        # [sol_15m_yes_offset_gate] Hard block YES when offset_pct in [-10%, 0%) — barely OTM zone.
        # Analysis (2026-05-23, n=34): WR=20.6%, BE≈50%, P&L≈-$500.
        # Spot is below floor strike — YES needs a sharp upward move in 15 min. No rescue.
        if asset.upper() == "SOL" and best_side == "yes":
            if -10.0 <= offset_pct < 0.0:
                print(f"    [sol_15m_yes_offset_gate] BLOCK YES→NO {ticker} — "
                      f"offset={offset_pct:+.3f}%∈[-10%,0) "
                      f"(OTM YES in barely-below-floor zone, WR=20.6%, no rescue)")
                best_side, best_edge = "no", edge_no

        # [sol_15m_yes_ou_theta_gate] Block YES when ou_theta>3.0 + pup<=0.55 + autocorr1_30<-0.008.
        # Analysis (2026-06-10, n=64 net-new): WR=32.8%, edge=-19.4%, PnL=-$488, MCPT p=0.0012.
        # Rescue (autocorr1_30>=-0.008): n=62, WR=56.5%, edge=+0.8%, p=0.724 — positive edge, allow.
        # Walk-forward: early p=0.012, late p=0.027 — both halves validate.
        # Causal: ou_theta>3.0 = fast OU mean-reversion speed; autocorr1_30<-0.008 = confirmed
        #         negative lag-1 autocorr at 30m level → agreement = price oscillates before
        #         contract expires → YES bet fails. autocorr>=-0.008 (non-negative momentum) = rescue.
        # Note: pup>0.55 already covered by sol_15m_yes_highcpu_gate — gate adds net-new blocks only.
        if asset.upper() == "SOL" and best_side == "yes":
            _ou_theta_g  = sig.get("ou_theta")
            _autocorr_yg = sig.get("autocorr1_30")
            _cpu_ou      = sig.get("composite_p_up")
            if (_ou_theta_g is not None and _autocorr_yg is not None
                    and float(_ou_theta_g) > 3.0
                    and (_cpu_ou is None or float(_cpu_ou) <= 0.55)
                    and float(_autocorr_yg) < -0.008):
                print(f"    [sol_15m_yes_ou_theta_gate] BLOCK YES→NO {ticker} — "
                      f"ou_theta={float(_ou_theta_g):.3f}>3.0, "
                      f"autocorr1_30={float(_autocorr_yg):+.4f}<-0.008 "
                      f"(fast mean-reversion+negative autocorr, WR=32.8%, edge=-19.4%)")
                best_side, best_edge = "no", edge_no

        # [sol_15m_no_stoch_gate] Block NO when stoch_k_1h∈[60,80); rescue on near-extreme/deep-OTM/low-pm.
        # Analysis (2026-05-23, non-rescued n=42): WR=28.6%, BE≈35%, P&L=-$532.
        # Rescue: (pm<0.55 AND cpu<0.45) = market+model both low YES → NO still has edge.
        #         stoch_k_1h>=75 = near-extreme overbought → mean-reversion thesis strengthens.
        #         offset_pct<-0.10 = spot >0.10% below floor → NO structurally strong.
        if asset.upper() == "SOL" and best_side == "no":
            if 60.0 <= _sk1h < 80.0:
                _cpu_sol2   = sig.get("composite_p_up")
                _cpu_sol2_f = float(_cpu_sol2) if _cpu_sol2 is not None else 0.5
                _nostoch_rescue = (
                    (p_market < 0.55 and _cpu_sol2_f < 0.45)
                    or _sk1h >= 75.0
                    or offset_pct < -0.10
                )
                if not _nostoch_rescue:
                    print(f"    [sol_15m_no_stoch_gate] BLOCK NO→YES {ticker} — "
                          f"stoch_k_1h={_sk1h:.1f}∈[60,80), pm={p_market:.3f}, "
                          f"cpu={_cpu_sol2_f:.3f}, offset={offset_pct:+.3f}% "
                          f"(momentum extended, no rescue)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    _rsrc_sol = ("near-extreme stoch>=75" if _sk1h >= 75.0
                                 else ("deep-OTM offset<-0.10" if offset_pct < -0.10
                                       else "pm<0.55+cpu<0.45"))
                    print(f"    [sol_15m_no_stoch_gate] RESCUE NO {ticker} — "
                          f"stoch_k_1h={_sk1h:.1f}: {_rsrc_sol}")

        # [sol_15m_no_kalman_gate] Block NO when kalman_velocity<-0.001; rescue BOTH hurst>=0.6+autocorr>=-0.008.
        # Analysis (2026-06-10, block n=179): WR=41.9%, edge=-7.7%, PnL=-$854, MCPT p=0.0000.
        # Rescue (BOTH hurst>=0.6 AND autocorr>=-0.008): n=64, WR=54.7%, edge=+2.6%, PnL=+$83, p=0.477.
        # Walk-forward block: T1 p=0.042, T2 p=0.078 (borderline), T3 p=0.0012 — strengthening.
        # Causal: kalman_velocity<-0.001 = declining Kalman trend. Without BOTH persistent Hurst
        #         (>=0.6 = trending) AND non-negative autocorr (>=-0.008 = momentum), the decline
        #         is oscillatory — price bounces before reaching floor → NO loses. Rescue = trending
        #         decline with momentum → floor may be breached → allow NO (edge=+2.6%).
        if asset.upper() == "SOL" and best_side == "no":
            _kv_g = sig.get("kalman_velocity")
            if _kv_g is not None and float(_kv_g) < -0.001:
                _hurst_kg    = sig.get("hurst_exponent")
                _autocorr_ng = sig.get("autocorr1_30")
                _kv_rescue   = (
                    _hurst_kg is not None and _autocorr_ng is not None
                    and float(_hurst_kg) >= 0.6
                    and float(_autocorr_ng) >= -0.008
                )
                if not _kv_rescue:
                    if _hurst_kg is None:
                        _kv_miss = "hurst=None"
                    elif float(_hurst_kg) < 0.6:
                        _kv_miss = f"hurst={float(_hurst_kg):.3f}<0.6"
                    elif _autocorr_ng is None:
                        _kv_miss = "autocorr=None"
                    else:
                        _kv_miss = f"autocorr={float(_autocorr_ng):+.4f}<-0.008"
                    print(f"    [sol_15m_no_kalman_gate] BLOCK NO→YES {ticker} — "
                          f"kalman_velocity={float(_kv_g):+.5f}<-0.001, {_kv_miss} "
                          f"(oscillatory decline, WR=41.9%, edge=-7.7%)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    print(f"    [sol_15m_no_kalman_gate] RESCUE NO {ticker} — "
                          f"kv={float(_kv_g):+.5f}, hurst={float(_hurst_kg):.3f}>=0.6, "
                          f"autocorr={float(_autocorr_ng):+.4f}>=-0.008 "
                          f"(persistent declining trend, WR=54.7%, edge=+2.6%)")

        # [sol_15m_no_stoch_oversold_gate] Block SOL NO when stoch_k_15m < 20 (oversold).
        # Mean-reversion bounce: oversold 15m stoch → price snaps back above strike → NO fails.
        # Archive n=252, WR=42.5% vs BE=51.0%, saves +$1,042; MCPT z=+3.18 p=0.0000; 6/7 wks pos.
        # Rescue chg_5m < -0.20%: price already dropping (n=35, WR=54.3%) — allow NO.
        # Implemented 2026-06-24. Backup paper_trade_runner_15m_pre_sol_stoch_gates_20260624.py
        if asset.upper() == "SOL" and best_side == "no":
            _sk15m_no = float(sig.get("stoch_k_15m", 50.0) or 50.0)
            if _sk15m_no < 20.0:
                _chg5m_no = float(sig.get("chg_5m", 0.0) or 0.0)
                if _chg5m_no >= -0.20:
                    print(f"    [sol_15m_no_stoch_oversold_gate] FLIP NO→YES {ticker} — "
                          f"stoch_k_15m={_sk15m_no:.1f}<20 (oversold bounce risk, WR=42.5%), "
                          f"chg_5m={_chg5m_no:+.3f}%>=-0.20 (no momentum rescue)")
                    best_side, best_edge = "yes", edge_yes
                else:
                    print(f"    [sol_15m_no_stoch_oversold_gate] RESCUE NO {ticker} — "
                          f"stoch_k_15m={_sk15m_no:.1f}<20 but chg_5m={_chg5m_no:+.3f}%<-0.20 "
                          f"(already dropping, NO directionally correct, WR=54.3%)")

        # [sol_15m_yes_stoch_oversold_gate] Block SOL YES when stoch_k_15m < 30 (falling market).
        # Oversold downtrend → model misprices low-pm YES as edge; price continues down → YES fails.
        # Archive n=157, WR=43.3% vs BE=58.0%, saves +$855; MCPT z=+2.45 p=0.009; 7/7 wks pos.
        # Rescue offset_pct > 0.07%: YES is already slightly ITM — structural advantage holds (n=41,WR=58.5%).
        # Implemented 2026-06-24. Backup paper_trade_runner_15m_pre_sol_stoch_gates_20260624.py
        if asset.upper() == "SOL" and best_side == "yes":
            _sk15m_yes = float(sig.get("stoch_k_15m", 50.0) or 50.0)
            if _sk15m_yes < 30.0:
                if offset_pct <= 0.07:
                    print(f"    [sol_15m_yes_stoch_oversold_gate] FLIP YES→NO {ticker} — "
                          f"stoch_k_15m={_sk15m_yes:.1f}<30 (oversold downtrend, WR=43.3%), "
                          f"offset={offset_pct:+.3f}%<=0.07 (no structural ITM rescue)")
                    best_side, best_edge = "no", edge_no
                else:
                    print(f"    [sol_15m_yes_stoch_oversold_gate] RESCUE YES {ticker} — "
                          f"stoch_k_15m={_sk15m_yes:.1f}<30 but offset={offset_pct:+.3f}%>0.07 "
                          f"(slightly ITM, structural advantage holds, WR=58.5%)")

        # P_MARKET VOLATILITY GATE: skip deep-OTM contracts on either side.
        # Sim (347 resolved trades): 0W/26L blocked at 0.12/0.88 → +$538 PnL delta.
        if p_market < P_MARKET_VOL_MIN or p_market > P_MARKET_VOL_MAX:
            print(f"    [vol_gate] p_market={p_market:.3f} — blocked (outside "
                  f"[{P_MARKET_VOL_MIN:.2f},{P_MARKET_VOL_MAX:.2f}])")
            evaluated.append((best_edge, best_side, c, p_model, offset_pct))
            continue

        # Tau confidence decay: edge scales as (tau/5)^2 below 5 min
        tau_conf      = (tau_min / 5.0) ** 2 if tau_min < 5.0 else 1.0
        adjusted_edge = best_edge * tau_conf

        print(f"\n  Contract: {ticker}")
        print(f"    floor_strike={floor_s:.4f}  offset={offset_pct:+.3f}%  "
              f"tau={tau_min:.1f}min  p_market={p_market:.3f}")
        if asset == "BTC" and _p_up_v2_btc is not None:
            _branch_str = f"  p_yes(pup_v2)={p_model_yes:.3f}  p_no(pup_v2)={p_model_no:.3f}"
        elif asset == "BTC" and _zdrift_15m is not None:
            _branch_str = f"  p_yes(zdrift)={p_model_yes:.3f}  p_no(lgbm)={p_model_no:.3f}"
        else:
            _branch_str = f"  p_model={p_model:.3f}"
        print(f"   {_branch_str}  edge_yes={edge_yes:+.3f}  edge_no={edge_no:+.3f}"
              f"  → best={best_side.upper()} ({best_edge:+.3f})"
              + (f"  tau_adj={adjusted_edge:+.3f}" if tau_min < 5.0 else ""))

        entry = (best_edge, best_side, c, p_model, offset_pct)
        evaluated.append(entry)
        if adjusted_edge >= EDGE_THRESHOLD:
            candidates.append(entry)

    # ── Pass 2: log pass rows for all non-winner contracts ────────────────────
    candidates.sort(key=lambda x: x[0], reverse=True)
    winner_ticker = candidates[0][2]["ticker"] if candidates else None

    for best_edge, best_side, c, p_model, offset_pct in evaluated:
        if c["ticker"] == winner_ticker:
            continue  # winner logged below
        p_market = c["p_market"]
        sig["p_gbdt"] = _lgbm_shadows.get(c["ticker"], "")
        row = _build_row(
            asset=asset, decision_time=decision_time, ticker=c["ticker"],
            close_time=c["close_time"], spot=spot, floor_s=c["floor_strike"],
            offset_pct=offset_pct, tau_min=c["tau_minutes"], p_market=p_market,
            p_model=p_model, raw_edge=best_edge, side="", decision="pass",
            sig=sig, kelly_fraction=0.0, bet_fraction=0.0,
            bet_amount=0.0, bankroll=bankroll, liq_signal=_liq_signal,
            cg=_cg, spread=c["ask"] - c["bid"],
        )
        append_row(row, asset=asset)

    if not candidates:
        print(f"\n  [scan] No qualifying trades (best edge below {EDGE_THRESHOLD:.2f}).")
        return

    # ── Pass 3: place the single best bet ─────────────────────────────────────
    best_edge, side, c, p_model, offset_pct = candidates[0]
    ticker     = c["ticker"]
    floor_s    = c["floor_strike"]
    p_market   = c["p_market"]
    tau_min    = c["tau_minutes"]
    close_time = c["close_time"]

    print(f"\n  [best] → {ticker} {side.upper()}  edge={best_edge:+.3f}"
          f"  ({len(candidates)} contract(s) qualified)")

    # HMM State 7 → elevated Kelly cap (high-payout NO regime, +$116/trade historical)
    _kelly_cap = MAX_BET_FRAC_ST7 if (asset == "BTC" and _hmm_state == 7) else MAX_BET_FRAC
    try:
        kelly = compute_kelly_size(
            p_model=p_model, p_market=p_market, bankroll=bankroll,
            kelly_multiplier=KELLY_MULT, side=side, max_bet_fraction=_kelly_cap,
        )
    except ValueError as e:
        print(f"    [kelly] Error: {e}. Skipping.")
        return

    if kelly.bet_amount <= 0:
        print(f"    [kelly] No positive Kelly sizing. Skipping.")
        return

    if kelly.bet_amount < 3.0:
        print(f"    [min_bet] Skipping — Kelly bet ${kelly.bet_amount:.2f} < $3 (thin edge at pm={p_market:.3f})")
        return

    # Nearest resistance dampener (YES only): halve Kelly when an overhead EMA/VWAP
    # level is within 0.5% above spot. Sim: nearest_res<=0.5% → WR=43.8%, delta=+$329
    # when fully blocked; dampener keeps volume while reducing exposure.
    # Revert: remove this block.
    if side == "yes":
        _res_dist = sig.get("nearest_res_dist_pct", 999.0)
        if isinstance(_res_dist, float) and _res_dist <= 0.5:
            _undampened = kelly.bet_amount
            kelly.bet_amount = round(kelly.bet_amount * 0.5, 2)
            print(f"    [res_damper] nearest_res={_res_dist:.2f}% ≤ 0.5% → Kelly ×0.5 "
                  f"(${_undampened:.2f} → ${kelly.bet_amount:.2f})")

    # ETH YES Kelly dampener: all YES edge bands above [0.04,0.05) have negative flat PnL.
    # Flat edge is inverted — higher raw_edge correlates with lower actual WR.
    # 50% reduction approximately halves YES Kelly losses while preserving the only
    # profitable YES band ([0.04,0.05), tiny Kelly anyway).
    # Revert: remove this block.
    if asset == "ETH" and side == "yes":
        _undampened_eth = kelly.bet_amount
        kelly.bet_amount = round(kelly.bet_amount * 0.5, 2)
        print(f"    [eth_yes_kelly_damp] ETH YES ×0.5 "
              f"(${_undampened_eth:.2f} → ${kelly.bet_amount:.2f})")

    # [donch_low_no_boost 2026-06-22] 1.5x Kelly (ceil 7.5%) for NO when price is LOW in its
    # 1h Donchian channel (<0.20 = bottom 20% of 20h range → range-bound, NO safe). Validated on
    # 15m TAKEN NO: donch<0.20 is the best zone — BTC +$3,254 (WR 47%), ETH +$516 (WR 71%).
    # Ports the hourly donch boost to 15m. NOTE: the hourly donch>0.80 BLOCK was NEUTRAL on 15m
    # (BTC -$8, ETH +$3) so it is deliberately NOT ported. BTC/ETH only (SOL unvalidated).
    if asset in ("BTC", "ETH") and side == "no":
        _dp = sig.get("donch_1h_pos")
        if isinstance(_dp, (int, float)) and _dp == _dp and _dp < 0.20:
            _boosted = round(min(kelly.bet_amount * 1.5, 0.075 * bankroll), 2)
            if _boosted > kelly.bet_amount:
                print(f"    [donch_low_no_boost] donch_1h={_dp:.3f}<0.20 → Kelly ×1.5 "
                      f"(${kelly.bet_amount:.2f} → ${_boosted:.2f})")
                kelly.bet_amount = _boosted

    n_contracts = max(1, round(kelly.bet_amount / p_market)) if side == "yes" \
                  else max(1, round(kelly.bet_amount / (1 - p_market)))
    cost = round(n_contracts * (p_market if side == "yes" else 1 - p_market), 2)

    print(f"    [TRADE] {side.upper()}  {n_contracts} contracts @ ${p_market:.3f}")
    print(f"    Kelly: frac={kelly.kelly_fraction:.4f}  bet_frac={kelly.bet_fraction:.4f}  "
          f"amount=${kelly.bet_amount:.2f}  cost=${cost:.2f}")

    sig["p_gbdt"] = _lgbm_shadows.get(ticker, "")
    row = _build_row(
        asset=asset, decision_time=decision_time, ticker=ticker,
        close_time=close_time, spot=spot, floor_s=floor_s,
        offset_pct=offset_pct, tau_min=tau_min, p_market=p_market,
        p_model=p_model, raw_edge=best_edge, side=side, decision="trade",
        sig=sig, kelly_fraction=kelly.kelly_fraction,
        bet_fraction=kelly.bet_fraction, bet_amount=cost, bankroll=bankroll,
        liq_signal=_liq_signal, cg=_cg, spread=c["ask"] - c["bid"],
    )
    append_row(row, asset=asset)
    if already_bet is not None:
        already_bet.add(close_time)
    print(f"    [logged] Row written to {csv_name}.")

    # ── Live order placement ──────────────────────────────────────────────────
    if is_live and auth is not None:
        _live_csv = live_trading.get_live_csv_path(asset)
        if not live_trading.check_daily_loss_limit(daily_loss_limit or 150.0, _live_csv):
            print("  [live] Daily loss limit reached — skipping live order.")
        else:
            bid_pm = c.get("bid", p_market - 0.01)
            ask_pm = c.get("ask", p_market + 0.01)
            yes_price_cents, live_count = live_trading.compute_order_params(
                side=side,
                bet_amount=kelly.bet_amount,
                bid=bid_pm,
                ask=ask_pm,
                max_contracts=500,
            )
            if live_count == 0:
                print(f"  [live] Bet ${kelly.bet_amount:.2f} too small for one contract — skipping.")
            else:
                balance = live_trading.get_balance(auth)
                order_cost = live_count * (yes_price_cents if side == "yes" else (100 - yes_price_cents)) / 100.0
                if balance is not None:
                    print(f"  [live] Balance: ${balance:.2f}  order cost ≈ ${order_cost:.2f}")
                    if order_cost > balance:
                        print(f"  [live] Insufficient balance — skipping order.")
                        return
                order_result = live_trading.place_order(
                    auth=auth, ticker=ticker, side=side,
                    count=live_count, yes_price=yes_price_cents,
                )
                live_row = {
                    "contract_ticker": ticker,
                    "spot":            round(spot, 4),
                    "strike":          round(floor_s, 4),
                    "offset_pct":      round(offset_pct, 4),
                    "p_market":        round(p_market, 4),
                    "p_yes_model":     round(p_model, 4),
                    "net_edge":        round(best_edge, 4),
                    "bet_amount":      round(order_cost, 4),
                    "bankroll":        bankroll,
                }
                live_trading.log_live_trade(
                    row=live_row, order_result=order_result,
                    yes_price_cents=yes_price_cents, count=live_count,
                    side=side, asset=asset, csv_path=_live_csv,
                )


def _build_row(
    asset, decision_time, ticker, close_time, spot, floor_s, offset_pct,
    tau_min, p_market, p_model, raw_edge, side, decision, sig,
    kelly_fraction, bet_fraction, bet_amount, bankroll,
    liq_signal=None, cg=None, spread=0.0,
) -> dict:
    def _f(v, d=4):
        try:
            fv = float(v)
            return round(fv, d) if fv == fv else ""
        except (TypeError, ValueError):
            return ""
    # z_score: log(K/S) / sigma_tau using realized vol — captures moneyness for LGBM
    _vol_pm    = float(sig.get("vol_multi") or
                       sig.get("realized_vol_annual", 0.3) / math.sqrt(MINS_PER_YEAR))
    _sigma_tau = max(_vol_pm * math.sqrt(tau_min), 1e-6)
    _z_score   = round(math.log(floor_s / spot) / _sigma_tau, 4) if spot > 0 else ""
    return {
        "logged_at":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00"),
        "decision_time":        decision_time,
        "asset":                asset.upper(),
        "contract_ticker":      ticker,
        "close_time":           close_time,
        "spot":                 round(spot, 4),
        "floor_strike":         round(floor_s, 4),
        "offset_pct":           round(offset_pct, 4),
        "tau_minutes":          round(tau_min, 2),
        "spread":               _f(spread),
        "p_market":             round(p_market, 4),
        "p_model_15m":          round(p_model, 4),
        "raw_edge":             round(raw_edge, 4),
        "side":                 side,
        "decision":             decision,
        # per-contract moneyness
        "z_score":              _z_score,
        # 5m
        "bp_5m":                _f(sig.get("bp_5m")),
        "body_5m":              _f(sig.get("body_5m")),
        "dir_5m":               sig.get("dir_5m", ""),
        "vol_ratio":            _f(sig.get("vol_ratio"), 3),
        "vol_ratio_5m":         _f(sig.get("vol_ratio_5m"), 3),
        # 15m
        "body_15m":             _f(sig.get("body_15m")),
        "bp_15m":               _f(sig.get("bp_15m")),
        "dir_15m":              sig.get("dir_15m", ""),
        "upper_wick_15m":       _f(sig.get("upper_wick_15m")),
        "lower_wick_15m":       _f(sig.get("lower_wick_15m")),
        "atr_ratio_15m":        _f(sig.get("atr_ratio_15m"), 6),
        "range_ratio_15m":      _f(sig.get("range_ratio_15m")),
        "consec_dir_15m":       sig.get("consec_dir_15m", ""),
        "stoch_k_5m":           _f(sig.get("stoch_k_5m"), 2),
        "stoch_k_15m":          _f(sig.get("stoch_k_15m"), 2),
        # changes
        "chg_1m":               _f(sig.get("chg_1m")),
        "chg_5m":               _f(sig.get("chg_5m")),
        "chg_15m":              _f(sig.get("chg_15m")),
        # vwap / ema
        "vwap_dist":            _f(sig.get("vwap_dist")),
        "ema_bias":             sig.get("ema_bias", ""),
        "ema_bias_1h":          sig.get("ema_bias_1h", ""),
        "ema20_dist_1h":        _f(sig.get("ema20_dist_1h")),
        "ema50_dist_1h":        _f(sig.get("ema50_dist_1h")),
        "nearest_res_dist_pct": _f(sig.get("nearest_res_dist_pct")),
        # composite / vol
        "composite_p_up":       _f(sig.get("composite_p_up")),
        "realized_vol_annual":  _f(sig.get("realized_vol_annual")),
        "vol_ratio_1h":         _f(sig.get("vol_ratio_1h"), 3),
        # 1h
        "bp_1h":                _f(sig.get("bp_1h")),
        "chg_1h":               _f(sig.get("chg_1h")),
        "dir_1h":               sig.get("dir_1h", ""),
        "consec_dir_1h":        sig.get("consec_dir_1h", ""),
        "stoch_k_1h":           _f(sig.get("stoch_k_1h"), 2),
        "stoch_cross_1h":       sig.get("stoch_cross_1h", ""),
        "rsi_1h":               _f(sig.get("rsi_1h"), 2),
        "macd_hist_1h":         _f(sig.get("macd_hist_1h"), 6),
        "donchian_breakout_1h": sig.get("donchian_breakout_1h", ""),
        "engulfing_1h":         sig.get("engulfing_1h", ""),
        "bb_pct_1h":            _f(sig.get("bb_pct_1h")),
        # 1h rolling drift
        "mu6h":                 _f(sig.get("mu6h"), 7),
        "mu12h":                _f(sig.get("mu12h"), 7),
        "mu24h":                _f(sig.get("mu24h"), 7),
        "regime_z":             _f(sig.get("regime_z")),
        "arima_forecast_1h":    _f(sig.get("arima_forecast_1h"), 7),
        # 4h
        "stoch_k_4h":           _f(sig.get("stoch_k_4h"), 2),
        "rsi_4h":               _f(sig.get("rsi_4h"), 2),
        "chg_4h":               _f(sig.get("chg_4h")),
        "bp_4h":                _f(sig.get("bp_4h")),
        # Coinalyze
        "liq_score":            liq_signal.liq_score              if liq_signal is not None else "",
        "liq_bias":             _f(liq_signal.liq_bias)           if liq_signal is not None else "",
        "oi_chg_pct":           _f(liq_signal.oi_chg_pct)        if liq_signal is not None else "",
        "ls_long_pct":          _f(liq_signal.ls_long_pct, 2)    if liq_signal is not None else "",
        # CoinGlass macro
        "fear_greed":           _f(cg.fg_value, 1)               if cg is not None else "",
        "cg_composite":         cg.composite_score                if cg is not None else "",
        "kelly_fraction":       round(kelly_fraction, 4),
        "bet_fraction":         round(bet_fraction, 4),
        "bet_amount":           round(bet_amount, 2),
        "bankroll":             bankroll,
        "resolved_yes":         "",
        "would_win":            "",
        "would_pnl":            "",
        # Markov regimes
        "markov_regime_1h":     sig.get("markov_regime_1h",  ""),
        "markov_regime_15m":    sig.get("markov_regime_15m", ""),
        "markov_eth_daily":     sig.get("markov_eth_daily",  ""),
        "markov_sol_6h":        sig.get("markov_sol_6h",     ""),
        "markov_sol_4h":        sig.get("markov_sol_4h",     ""),
        "markov_sol_1h":        sig.get("markov_sol_1h",     ""),
        "p_up_v2_btc":          _f(sig.get("p_up_v2_btc")),
        "z_drift_6h":           _f(sig.get("z_drift_6h")),
        "p_gbdt":               _f(sig.get("p_gbdt")),
        # Shadow stochastic signals (log-only)
        "ou_theta":             _f(sig.get("ou_theta")),
        "ou_halflife":          _f(sig.get("ou_halflife")),
        "ou_mu_distance":       _f(sig.get("ou_mu_distance")),
        "hurst_exponent":       _f(sig.get("hurst_exponent")),
        "autocorr1_15":         _f(sig.get("autocorr1_15")),
        "autocorr1_30":         _f(sig.get("autocorr1_30")),
        "kalman_velocity":      _f(sig.get("kalman_velocity")),
        "kalman_residual":      _f(sig.get("kalman_residual")),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LOOP_INTERVAL_SEC = 300  # 5 minutes between scans in loop mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trading runner for 15-minute directional contracts (BTC, ETH, SOL)."
    )
    parser.add_argument("--asset", type=str, default="BTC",
                        choices=list(ASSET_CONFIG.keys()),
                        help="Asset to trade (default: BTC)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                        help=f"Bankroll for Kelly sizing (default ${DEFAULT_BANKROLL:,.0f})")
    parser.add_argument("--loop", action="store_true",
                        help=f"Run continuously, scanning every {LOOP_INTERVAL_SEC // 60} minutes")
    parser.add_argument("--live", action="store_true",
                        help="Place real orders on Kalshi (default: paper only)")
    parser.add_argument("--daily-loss-limit", type=float, default=150.0,
                        help="Max daily loss in dollars before halting live orders (default: $150)")
    args = parser.parse_args()
    asset = args.asset.upper()

    # Single-process-per-asset guard — prevents watchdog from spawning duplicates.
    _lock_prefix = "live_trade_15m" if args.live else "paper_trade_15m"
    _lock_path = Path(__file__).parent / f".{_lock_prefix}_{asset}.lock"
    _lock_fd = open(_lock_path, "w")
    try:
        _fcntl.flock(_lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"ERROR: Another {asset} 15m paper trade process is already running. Exiting.")
        sys.exit(1)

    series = ASSET_CONFIG[asset]["series_ticker"]
    print("=" * 60)
    print(f"  {asset} 15M PAPER TRADER  ({series})")
    print("=" * 60)
    print(f"  Bankroll: ${args.bankroll:,.2f}")
    print(f"  Edge threshold: {EDGE_THRESHOLD:.2f}")
    print(f"  Kelly multiplier: {KELLY_MULT:.0%}")
    print(f"  Mode: {'*** LIVE ***' if args.live else 'paper'}")
    if args.live:
        print(f"  Daily loss limit: ${args.daily_loss_limit:.0f}")
    if args.loop:
        print(f"  Loop mode: ON (every {LOOP_INTERVAL_SEC // 60} min)")

    auth = load_auth()
    if auth is None:
        print("\n  WARNING: No Kalshi credentials. Resolution check and contract scan require auth.")
    else:
        print("  Kalshi auth: loaded.")

    _LGBM_MODELS[asset] = _load_15m_lgbm(asset)

    if args.live:
        _live_csv = live_trading.get_live_csv_path(asset)
        live_trading.ensure_live_csv_exists(_live_csv)
        print(f"  Live CSV: {_live_csv}")

    ensure_csv(asset)

    already_bet: set = set()

    # Seed already_bet from the CSV so a freshly restarted process doesn't
    # re-trade a contract that was bet by a previous (or overlapping) instance.
    # Looks back 30 minutes to cover one full 15m contract window plus buffer.
    _csv_file = _csv_path(asset)
    if _csv_file.exists():
        try:
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(minutes=30)
            _seed_df = pd.read_csv(_csv_file, usecols=["logged_at", "close_time", "decision"])
            _seed_df = _seed_df[_seed_df["decision"] == "trade"]
            _seed_df["logged_at"] = pd.to_datetime(_seed_df["logged_at"], utc=True, errors="coerce")
            _seed_df = _seed_df[_seed_df["logged_at"] >= cutoff]
            for ct in _seed_df["close_time"].dropna():
                already_bet.add(ct)
            if already_bet:
                print(f"  [already_bet] Seeded {len(already_bet)} recent close_time(s) from CSV: {already_bet}")
        except Exception as _e:
            print(f"  [already_bet] Seed warning: {_e}")

    if not args.loop:
        print("\nChecking pending resolutions...")
        resolve_pending(auth, asset, is_live=args.live)
        run_scan(auth, args.bankroll, asset, already_bet=already_bet,
                 is_live=args.live, daily_loss_limit=args.daily_loss_limit)
        return

    scan_count = 0
    while True:
        scan_count += 1
        print(f"\n  [loop] Scan #{scan_count}  (session bets: {len(already_bet)})")
        try:
            resolve_pending(auth, asset, is_live=args.live)
            if scan_count % 3 == 0:
                try:
                    import scan_archive_15m as _sa15
                    _sa15.fill_scan_outcomes(asset=asset, auth=auth)
                except Exception:
                    pass
            run_scan(auth, args.bankroll, asset, already_bet=already_bet,
                     is_live=args.live, daily_loss_limit=args.daily_loss_limit)
        except KeyboardInterrupt:
            print("\n  [loop] Stopped by user.")
            break
        except Exception as exc:
            print(f"  [loop] Unhandled error: {exc}")

        print(f"\n  [loop] Sleeping {LOOP_INTERVAL_SEC // 60} min until next scan...")
        try:
            time.sleep(LOOP_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n  [loop] Stopped by user.")
            break


if __name__ == "__main__":
    main()
