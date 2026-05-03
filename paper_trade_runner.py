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

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from evaluate_point import load_data
from market_data import compute_realized_volatility
from probability_engine import estimate_probability, implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT, REALIZED_VOL_WEIGHT_BY_ASSET
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from order_book import fetch_order_book_imbalance
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from decision import evaluate_trade
from funding_rate import fetch_funding_rate, FundingRateResult
import outcome_checker
import update_data
import live_trading
from kelly_sizing import compute_kelly_size
from composite_scorer import compute_current_scores, score_to_p_model, score_to_p_no_model, composite_to_confirmation, lookup_p_up, K_DRIFT_NO_BTC
import direct_p_model
import pickle as _pickle
from vol_layer import compute_vol_regime_factor

# BTC isotonic calibration: corrects lognormal p_model overconfidence at extremes.
# Trained on 490 resolved BTC paper trades. Reduces NO bet losses in 20–30% p_market
# range where formula overestimates P(NO wins). Loaded lazily at first BTC scan.
_BTC_ISO_CAL: "dict | None | str" = "unloaded"

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


def _expiry_prefix(ticker: str) -> str:
    """Extract the expiry portion of a contract ticker.

    e.g. 'KXETHD-26APR0701-T2119.99' → 'KXETHD-26APR0701'
    Used as a consistent key for per-expiry trade counting across live and paper runners.
    """
    parts = ticker.rsplit("-T", 1)
    return parts[0] if len(parts) == 2 else ticker


def get_csv_path(asset: str = "BTC") -> Path:
    """Return the asset-specific paper trades CSV path."""
    asset = asset.upper()
    if asset == "BTC":
        return PAPER_TRADES_CSV  # keep existing BTC file unchanged
    return Path(__file__).parent / "results" / f"paper_trades_{asset.lower()}.csv"
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
    "composite_p_up",     # calibrated directional probability from composite scorer
    "chg_30m",            # 30-minute price change fraction at decision time
    "chg_10m",            # 10-minute price change fraction at decision time
    "chg_5m",             # 5-minute price change fraction at decision time
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
    "resolved_yes",   # filled by outcome_checker.py
    "would_win",      # filled by outcome_checker.py
    "would_pnl",      # filled by outcome_checker.py
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
    parser.add_argument("--daily-loss-limit", type=float, default=100.0,
                        help="Max dollars to lose live per calendar day before halting (default: 100)")
    parser.add_argument("--max-contracts", type=int, default=500,
                        help="Hard cap on contracts per live order (default: 500 — size controlled by Kelly dollar amount)")
    args = parser.parse_args()
    args.asset = args.asset.upper()

    now_utc = datetime.now(timezone.utc)
    print(f"\n  Run time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- US-session volatility filter (all assets) ---
    # Live PnL analysis (n=61 trades, 2026-04-07) shows 0% win rate during US market
    # open (13-15 UTC = 9-11 AM EST) and US afternoon (17-19 UTC = 1-3 PM EST).
    # All assets trend rather than range during these windows — extended to ETH/SOL
    # after ETH took a losing trade at 13:36 UTC on 2026-04-07.
    # Revert: copy paper_trade_runner_v1.py → paper_trade_runner.py
    SKIP_HOURS = {12, 13, 18}  # 12-13 UTC (pre/NY market open) + 18:00 UTC (afternoon peak)
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
            _composite_computed = True
            print(f"  [composite] trend={_comp_trend:+d}  rev={_comp_rev:+d}  p_up={_comp_p_up:.1%}")
        except Exception as _exc:
            print(f"  [composite] Score error: {_exc} — using pure log-normal fallback")

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

    # --- Vol regime factor ---
    # Scales blended sigma before score_to_p_model. Validated on 19,947h of OHLCV data.
    # High-vol regime → factor > 1.0 → wider sigma → OTM strikes more reachable.
    # Low-vol regime  → factor < 1.0 → tighter sigma → edge concentrates near ATM.
    _vol_factor = 1.0
    _vol_score_dir = 0
    if live_1m is not None and len(live_1m) >= 400:
        try:
            _vol_factor, _vol_score_dir, _vol_details = compute_vol_regime_factor(df_confirm, live_1m, asset=args.asset)
            print(f"  [vol_layer] score={_vol_score_dir:+d}  factor={_vol_factor:.3f}  {_vol_details.get('votes', {})}")
        except Exception as _exc:
            print(f"  [vol_layer] Error: {_exc} — using factor=1.0")

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
        csv_path_check = get_csv_path(args.asset)
        # Live runner tracks its own positions from live_trades.csv to avoid being
        # blocked by paper-only trades. Paper runner uses paper_trades.csv as before.
        if args.live or getattr(args, 'dual', False):
            expiry_source_path = live_trading.get_live_csv_path(args.asset)
            expiry_source_is_live = True
        else:
            expiry_source_path = csv_path_check
            expiry_source_is_live = False
        already_traded = _SESSION_TRADED  # always use session set; CSV failure cannot bypass it
        already_traded_expiries = {}  # {close_ts: {"yes": [strikes], "no": [strikes]}}
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
                # Seed _SIDE_COOLDOWN from recent trades so restarts preserve cooldown state.
                # Live runner seeds from live_trades.csv only — paper trades must not
                # influence live cooldowns (they are independent processes).
                global _SESSION_SEEDED
                if not _SESSION_SEEDED:
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
                        if _SIDE_COOLDOWN:
                            print(f"  [session] Seeded {len(_SIDE_COOLDOWN)} cooldown entries from CSV")
                    except Exception:
                        pass
                    _SESSION_SEEDED = True
                    if _SESSION_TRADED:
                        print(f"  [session] Seeded {len(_SESSION_TRADED)} open tickers from CSV")
            except Exception:
                pass

        for c in ladder:
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
            if vol_ratio_c is not None and vol_ratio_c > _vol_ratio_limit:
                print(f"  [scan] Skipping {c['ticker']} — vol_ratio={vol_ratio_c:.2f} (realized >> implied, limit={_vol_ratio_limit})")
                continue
            _vol_weight = REALIZED_VOL_WEIGHT_BY_ASSET.get(args.asset, REALIZED_VOL_WEIGHT)
            vol_eff_c = blend_vol(vol.vol_multi, vol_imp_c, weight=_vol_weight)
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
                p_model_comp  = None
                # ETH / SOL: use trained strike-hit model (validated +$467 ETH, +$3,780 SOL on test)
                # BTC: direct model regressed on test — stay on legacy score_to_p_model
                if direct_p_model.asset_supported(args.asset):
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
                    p_model_comp = score_to_p_model(_active_trend, _active_rev, spot, s_k, sigma_tau_c, asset=args.asset)

                # [ETH isotonic calibration] Override direct_p_model with log-normal + isotonic.
                # HistGBM overestimates p_yes on deep OTM ETH YES (pm<0.40: 0% WR, -$977 live).
                # Isotonic trained on 18,836 reconstructed outcomes; maps p_ln → actual WR.
                # OOS simulation: +$742 vs HistGBM +$528 (Apr 23 – May 3 2026, 191 slots).
                if args.asset == "ETH" and sigma_tau_c > 0:
                    _eth_p_ln = _compute_eth_p_ln(spot, s_k, vol_eff_c, tau_c, _comp_p_up)
                    if _eth_p_ln is not None:
                        _eth_iso = _load_eth_iso()
                        if _eth_iso is not None:
                            p_model_comp = float(max(0.01, min(0.99, _eth_iso.predict([_eth_p_ln])[0])))
            else:
                p_model_comp  = prob_c.p_yes

            # [BTC vol gate] For OTM YES only (offset > 0): block if |z_strike| > 2.0 × vol_factor.
            # Only OTM YES bets need the reachability gate — ITM YES bets are already in the money,
            # and NO bets are governed by z_abs_no_min below. vol_factor widens/narrows the
            # band with the vol regime. BASE_Z=2.0 gives a 1.2–2.8σ range across vol_factor [0.60,1.40].
            if args.asset == "BTC" and _composite_computed and sigma_tau_c > 0 and offset_c > 0:
                _z_strike_abs = abs(math.log(s_k / spot) / sigma_tau_c)
                _btc_vol_gate_z = 2.0 * _vol_factor
                if _z_strike_abs > _btc_vol_gate_z:
                    print(f"  [btc_vol_gate] BLOCK {c['ticker']} — |z|={_z_strike_abs:.3f} > {_btc_vol_gate_z:.3f} (vol_factor={_vol_factor:.3f})")
                    continue

            # [BTC isotonic calibration DISABLED — trained on drift-biased p_model values.
            # Reform uses k_drift=0.8 + vol_factor-as-gate; isotonic retrain required
            # before re-enabling. Remove this comment once retrained.]

            # [OTM YES momentum exhaustion gate — BTC composite only]
            # Archive analysis Apr 15–May 3 2026 (replay simulator):
            #   pm < 0.15              : 0–5% actual WR — hard block, no signal rescues
            #   pm [0.15,0.35)+ema>=1  : 0% WR on 35 trades (EMA already stretched bullish)
            #   pm [0.25,0.35)+stoch>70: 8% WR on 12 trades (overbought into deep OTM YES)
            # OOS impact: +$388 vs baseline (+101% PnL improvement) in replay simulation.
            _otm_yes_blocked = False
            if args.asset == "BTC" and _composite_computed and pm < 0.35:
                _ema_s    = confirm.ema_stretch_score
                _sk_gate  = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                if pm < 0.15:
                    _otm_yes_blocked = True
                    print(f"  [otm_yes_gate] BLOCK YES {c['ticker']} — pm={pm:.3f}<0.15 (hard block)")
                elif _ema_s >= 1:
                    _otm_yes_blocked = True
                    print(f"  [otm_yes_gate] BLOCK YES {c['ticker']} — pm={pm:.3f}<0.35, ema_stretch={int(_ema_s)}")
                elif pm >= 0.25 and _sk_gate > 70:
                    _otm_yes_blocked = True
                    print(f"  [otm_yes_gate] BLOCK YES {c['ticker']} — pm={pm:.3f} in [0.25,0.35), stoch_k={_sk_gate:.1f}>70")

            p_yes_adj_c = max(0.03, min(0.97, p_model_comp + funding_delta))
            if c["ticker"] in already_traded:
                print(f"  [scan] Skipping {c['ticker']} — already traded this session")
                continue
            expiry_key = _expiry_prefix(c["ticker"])
            expiry_positions = already_traded_expiries.get(expiry_key, {"yes": [], "no": []})
            expiry_trade_count = len(expiry_positions["yes"]) + len(expiry_positions["no"])
            if expiry_trade_count >= 6:
                print(f"  [scan] Skipping {c['ticker']} — expiry limit reached ({expiry_trade_count}/6 trades)")
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
                _p_no_btc = score_to_p_no_model(
                    _active_trend, _active_rev, spot, s_k, sigma_tau_c, asset="BTC"
                )
                _pm_ask = c["ask"]
                _pm_bid = c["bid"]
                _dec_yes = None
                if not _otm_yes_blocked:
                    _dec_yes = evaluate_trade(
                        struct.structure_bias, confirm.confirmation_bias,
                        p_yes_adj_c, _pm_ask, args.bankroll,
                        confirmation_score=_cscore, no_score=_nscore,
                        obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                        ema_alignment=_ema_align, asset=args.asset,
                        composite_active=_composite_active, composite_p_up=_comp_p_up,
                        offset_pct=offset_c, force_side="yes")
                _dec_no = evaluate_trade(
                    struct.structure_bias, confirm.confirmation_bias,
                    1.0 - _p_no_btc, _pm_bid, args.bankroll,
                    confirmation_score=_cscore, no_score=_nscore,
                    obi_score=confirm.obi_score, vol_score=confirm.vol_score,
                    ema_alignment=_ema_align, asset=args.asset,
                    composite_active=_composite_active, composite_p_up=_comp_p_up,
                    offset_pct=offset_c, force_side="no")
                if _dec_yes is not None and _dec_yes.decision == "trade" and _dec_no.decision == "trade":
                    dec_c = _dec_yes if _dec_yes.net_edge >= _dec_no.net_edge else _dec_no
                elif _dec_yes is not None and _dec_yes.decision == "trade":
                    dec_c = _dec_yes
                elif _dec_no.decision == "trade":
                    dec_c = _dec_no
                else:
                    dec_c = _dec_no
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
            if dec_c.side == "no" and offset_c <= 0:
                print(f"  [scan] Skipping {c['ticker']} — ITM NO (offset={offset_c*100:+.3f}%, price already above strike)")
                continue
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
            if dec_c.side == "no" and args.asset == "BTC" and offset_c < 0.001:
                print(f"  [scan] Skipping {c['ticker']} — BTC NO offset={offset_c*100:+.3f}% < 0.10% minimum")
                continue
            if dec_c.side == "no" and args.asset == "ETH" and offset_c < 0.001:
                print(f"  [scan] Skipping {c['ticker']} — ETH NO offset={offset_c*100:+.3f}% < 0.10% minimum")
                continue
            if dec_c.side == "no" and args.asset == "SOL" and offset_c < 0.002:
                print(f"  [scan] Skipping {c['ticker']} — SOL NO offset={offset_c*100:+.3f}% < 0.20% minimum")
                continue
            if _sharp_move_active and dec_c.decision == "trade":
                print(f"  [sharp_move] {c['ticker']} — inverted composite: side={dec_c.side.upper()} net={dec_c.net_edge:+.4f}")

            # btc_pup_gate removed: replaced by vol_factor-as-gate + k_drift=0.8 reform.

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
            if args.asset == "ETH" and dec_c.side == "yes" and offset_c > 0 and pm < 0.45:
                print(f"  [eth_otm_gate] BLOCK OTM YES p_market={pm:.3f}<0.45 offset={offset_c:+.3f} — unreachable strike")
                continue

            # btc_no_pup_gate and btc_no_edge_gate removed: replaced by z_abs_no_min gate below.

            # [BTC NO z_abs gate] Block NO bets where the strike is < 0.6σ from spot.
            # Raised from 0.30 → 0.60 with dual-model reform (k_drift_no=0.30):
            # backtest shows z_abs > 0.60 + k_no=0.30 gives 20 test trades, +$247 PnL
            # vs z_abs > 0.30 + k_no=0.30 which gives 21 test trades, +$187 PnL.
            # Near-ATM NO bets (z_abs < 0.6) have poor structural edge even with k_no=0.30.
            if (args.asset == "BTC" and dec_c.side == "no"
                    and dec_c.decision == "trade" and _composite_computed and sigma_tau_c > 0):
                _z_no = abs(math.log(s_k / spot) / sigma_tau_c)
                if _z_no < 0.6:
                    print(f"  [btc_no_z_gate] BLOCK NO {c['ticker']} — |z|={_z_no:.3f} < 0.60 (near-ATM, no structural edge)")
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
                    continue
                else:
                    print(f"  [btc_spread_gate] RESCUED {dec_c.side.upper()} spread={spread_c:.3f} "
                          f"chg_10m={_sharp_move_pct_10m*100:+.2f}% aligned, net_edge={dec_c.net_edge:.4f}>=0.07")

            # [2026-04-28] BTC tau < 30 directional conviction gate with rescue.
            # Sim: tau<30 trades = 75W 52L (59.1% WR), -$609. Rescue: trades with directional
            # composite conviction (p_up>=0.52 YES / <=0.48 NO) = 50W 3L (94.3% WR), +$745.
            # Near-neutral p_up rescue: kelly_fraction>=0.15 AND spread<=0.02 → 20W 5L (80%, +$99).
            # Revert: remove this block.
            if args.asset == "BTC" and dec_c.decision == "trade" and tau_c < 30:
                _tau_conviction = (
                    (dec_c.side == "yes" and _comp_p_up >= 0.52) or
                    (dec_c.side == "no"  and _comp_p_up <= 0.48)
                )
                _tau_rescue = dec_c.kelly_fraction >= 0.15 and spread_c <= 0.02
                if not _tau_conviction and not _tau_rescue:
                    print(f"  [btc_tau_gate] BLOCK {dec_c.side.upper()} tau={tau_c:.1f}min<30 "
                          f"p_up={_comp_p_up:.3f} kelly={dec_c.kelly_fraction:.3f} spread={spread_c:.3f}")
                    continue
                elif _tau_conviction:
                    print(f"  [btc_tau_gate] PASS {dec_c.side.upper()} tau={tau_c:.1f}min<30 "
                          f"p_up={_comp_p_up:.3f} (directional conviction)")
                else:
                    print(f"  [btc_tau_gate] RESCUED {dec_c.side.upper()} tau={tau_c:.1f}min<30 "
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
            meta_c    = {"strike": s_k, "p_market": pm, "prob": prob_c,
                         "contract_ticker": c["ticker"], "close_ts": c["close_time"],
                         "vol_eff": vol_eff_c, "bid": c["bid"], "ask": c["ask"],
                         "p_model_comp": p_model_comp,
                         "p_no_model_comp": _p_no_btc if (args.asset == "BTC" and _composite_computed) else None}

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
            if dec_c.decision == "trade" and _streak30 is not None:
                _sk = confirm.stoch_k if confirm.stoch_k == confirm.stoch_k else 50.0
                _gate = False
                _gate_reason = ""
                if dec_c.side == "yes" and _streak30 == "bearish" and _sk <= 70 and args.asset != "SOL":
                    if _sharp_move_pct_10m <= 0:
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
                        _gate = True
                        _gate_reason = f"stoch_k={_sk:.1f}<20, chg_30m={_sharp_move_pct*100:+.2f}% (no bounce)"
                    else:
                        print(f"  [streak_gate] RESCUED NO {c['ticker']} — stoch_k={_sk:.1f}<20, chg_30m={_sharp_move_pct*100:+.2f}% (bounce active)")
                if _gate:
                    print(f"  [streak_gate] Blocked {dec_c.side.upper()} {c['ticker']} — {_gate_reason}")
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
            dec             = best_any_dec
            chosen          = best_any_meta
            p_market_source = "real"
            print(f"  [scan] No trade passes gates. Best seen: {chosen['contract_ticker']}  "
                  f"net_edge={dec.net_edge:+.4f}")
        else:
            print("  [scan] No real contracts available (auth failed or empty ladder) — skipping.")
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
        "chg_30m":            round(_sharp_move_pct * 100, 4),
        "chg_10m":            round(_sharp_move_pct_10m * 100, 4),
        "chg_5m":             round(_sharp_move_pct_5m * 100, 4),
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

        # Paper always logs in dual mode regardless of live limit
        if _is_dual:
            if dec.decision == "trade":
                _SESSION_TRADED[contract_ticker] = dec.net_edge
                _SIDE_COOLDOWN[(_expiry_prefix(contract_ticker), dec.side)] = now_utc
            csv_path = get_csv_path(args.asset)
            ensure_csv_exists(csv_path)
            append_row(row, csv_path)

    elif not args.live:
        # Pure paper mode OR dual no_trade: log to paper CSV and update session state
        if dec.decision == "trade":
            _SESSION_TRADED[contract_ticker] = dec.net_edge
            _SIDE_COOLDOWN[(_expiry_prefix(contract_ticker), dec.side)] = now_utc
        csv_path = get_csv_path(args.asset)
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
        if loop_count % 5 == 0:
            outcome_checker.main(get_csv_path(_loop_asset))
            if _loop_is_live_mode:
                _live_auth = load_auth()
                if _live_auth:
                    live_trading.settle_live_trades(_live_auth, live_trading.get_live_csv_path(_loop_asset))
        main()
        loop_count += 1
        time.sleep(60)
