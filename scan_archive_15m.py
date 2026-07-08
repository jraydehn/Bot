"""
scan_archive_15m.py

Ghost-logs every evaluated contract in each 15m scan cycle to
results/{asset}_scan_archive_15m.csv. Captures the full 15m signal
snapshot at evaluation time. resolved_yes is backfilled post-expiry
by fill_scan_outcomes().

Covers ALL assets (BTC, ETH, SOL) so each can eventually train its
own unbiased LGBM model. BTC continues writing to btc_scan_archive_15m.csv
for backward compatibility; ETH/SOL write to eth_scan_archive_15m.csv /
sol_scan_archive_15m.csv.
"""

import csv
import fcntl
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_RESULTS_DIR = Path(__file__).parent / "results"


@contextmanager
def _archive_lock(path: Path):
    """Exclusive advisory lock serializing appends and rewrites per archive file.

    Same race as scan_archive.py (2026-07-02): unlocked full rewrite in
    fill_scan_outcomes could clobber rows appended during the API loop, or
    restore a stale copy after a hung call. See scan_archive._archive_lock.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

COLUMNS = [
    "logged_at",
    "contract_ticker",
    "close_ts",
    "spot",
    "strike",
    "p_market",
    "tau_minutes",
    "spread",
    "p_model_yes",
    "p_model_no",
    # 5m signals
    "bp_5m",
    "vol_ratio",
    "vol_ratio_5m",
    # 15m signals
    "body_15m",
    "bp_15m",
    "dir_15m",
    "upper_wick_15m",
    "lower_wick_15m",
    "atr_ratio_15m",
    "range_ratio_15m",
    "consec_dir_15m",
    "stoch_k_5m",
    "stoch_k_15m",
    # price changes
    "chg_1m",
    "chg_5m",
    "chg_15m",
    # vwap / ema
    "vwap_dist",
    "ema_bias",
    "ema_bias_1h",
    "nearest_res_dist_pct",
    # composite / vol
    "composite_p_up",
    "realized_vol_annual",
    "vol_ratio_1h",
    # 1h signals
    "bp_1h",
    "chg_1h",
    "dir_1h",
    "consec_dir_1h",
    "stoch_k_1h",
    "stoch_cross_1h",
    "rsi_1h",
    "macd_hist_1h",
    "donchian_breakout_1h",
    "engulfing_1h",
    # Coinalyze
    "liq_score",
    "liq_bias",
    "oi_chg_pct",
    "ls_long_pct",
    # CoinGlass macro
    "fear_greed",
    "cg_composite",
    # contract structure
    "offset_pct",
    # Outcome (backfilled)
    "resolved_yes",
    # Expiry price (backfilled at resolution)
    "spot_at_expiry", "price_move_pct", "miss_pct",
    # Drift model features (BTC: p_up_v2 + 6h rolling z_drift)
    "p_up_v2_btc", "z_drift_6h",
    # HMM vol-regime state (BTC only; 0=low-vol, 1=med-vol, 2=high-vol)
    "hmm_vol_state",
    # VWAP MTF HMM state (BTC + SOL, each asset's own model; 8-state on 1m/5m/15m VWAP distances)
    "vwap_hmm_state",
    # 2026-07-08: SOL short-timeframe (5m/15m) VWAP HMM rescue signals -- didn't
    # exist anywhere else in this codebase at 5m/15m before this build.
    "kc_pct_5m", "kc_bo_5m", "kc_pct_15m", "kc_bo_15m",
    "donch_breakout_5m", "donch_pos_5m", "donch_breakout_15m", "donch_pos_15m",
    "stoch_cross_5m", "stoch_cross_15m",
    "kalman_velocity_5m", "kalman_residual_5m", "hurst_exponent_5m", "ou_theta_5m",
    "kalman_velocity_15m", "kalman_residual_15m", "hurst_exponent_15m", "ou_theta_15m",
    "arima_forecast_15m",
]

_META_COLS = {
    "logged_at", "contract_ticker", "close_ts", "spot", "strike",
    "p_market", "tau_minutes", "spread", "p_model_yes", "p_model_no",
    "resolved_yes",
}
_FEATURE_COLS = [c for c in COLUMNS if c not in _META_COLS]


def get_archive_path(asset: str) -> Path:
    return _RESULTS_DIR / f"{asset.lower()}_scan_archive_15m.csv"


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        return
    with open(path, newline="") as f:
        existing = csv.DictReader(f).fieldnames or []
    new_cols = [c for c in COLUMNS if c not in existing]
    if not new_cols:
        return
    with _archive_lock(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in new_cols:
                row.setdefault(col, "")
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)


def _fmt(v, digits: int = 6):
    if v is None:
        return ""
    try:
        fv = float(v)
        return "" if math.isnan(fv) else round(fv, digits)
    except (TypeError, ValueError):
        return v


def log_scan_row(
    ticker: str,
    close_ts: str,
    spot: float,
    strike: float,
    p_market: float,
    tau_minutes: float,
    spread: float,
    p_model_yes: float,
    p_model_no: float,
    features: dict,
    asset: str = "BTC",
    now_utc: Optional[datetime] = None,
) -> None:
    """Append one evaluated-contract row to the asset's 15m scan archive."""
    path = get_archive_path(asset)
    _ensure_csv(path)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    row: dict = {
        "logged_at":       now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_ticker": ticker,
        "close_ts":        close_ts or "",
        "spot":            _fmt(spot, 2),
        "strike":          _fmt(strike, 2),
        "p_market":        _fmt(p_market, 6),
        "tau_minutes":     _fmt(tau_minutes, 1),
        "spread":          _fmt(spread, 4),
        "p_model_yes":     _fmt(p_model_yes, 4),
        "p_model_no":      _fmt(p_model_no, 4),
        "resolved_yes":    "",
    }
    for col in _FEATURE_COLS:
        row[col] = _fmt(features.get(col))

    with _archive_lock(path):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore").writerow(row)


def fill_scan_outcomes(asset: str = "BTC", auth=None) -> int:
    """Backfill resolved_yes for settled contracts. Returns rows updated."""
    path = get_archive_path(asset)
    if not path.exists():
        return 0

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if not (r.get("resolved_yes") or "").strip()]
    if not pending:
        return 0

    try:
        from outcome_checker import fetch_market, is_settled, parse_resolution
        from live_signal import load_auth as _load_auth
        _auth = auth or _load_auth()
        if _auth is None:
            print(f"  [scan_archive_15m:{asset}] No auth — cannot fill outcomes")
            return 0
    except Exception as e:
        print(f"  [scan_archive_15m:{asset}] Import error: {e}")
        return 0

    # Updates keyed by row identity; write path re-reads the file so rows
    # appended during the (slow) API loop are never lost. See scan_archive.py.
    _cache: dict[str, Optional[int]] = {}
    updates: dict[tuple, dict] = {}
    for row in pending:
        ticker = row.get("contract_ticker", "").strip()
        if not ticker:
            continue
        if ticker not in _cache:
            try:
                market = fetch_market(ticker, _auth)
                if market and is_settled(market):
                    _cache[ticker] = int(parse_resolution(market))
                else:
                    _cache[ticker] = None
            except Exception:
                _cache[ticker] = None
        if _cache[ticker] is not None:
            upd: dict = {"resolved_yes": str(_cache[ticker])}
            if not (row.get("spot_at_expiry") or "").strip():
                close_ts = row.get("close_ts", "")
                spot_scan = float(row.get("spot") or 0)
                strike    = float(row.get("strike") or 0)
                from live_signal import fetch_spot_at_time
                # NOTE: Binance-derived, ~+14bp above Kalshi settlement basis —
                # diagnostics only; never derive win/loss from spot_at_expiry.
                spot_exp  = fetch_spot_at_time(close_ts, asset) if close_ts else None
                if spot_exp and spot_scan > 0:
                    upd["spot_at_expiry"] = round(spot_exp, 2)
                    upd["price_move_pct"] = round((spot_exp - spot_scan) / spot_scan * 100, 4)
                if spot_exp and strike > 0:
                    upd["miss_pct"] = round((spot_exp - strike) / strike * 100, 4)
            updates[(row.get("logged_at", ""), ticker)] = upd

    updated = len(updates)
    if updated:
        with _archive_lock(path):
            with open(path, newline="") as f:
                fresh = list(csv.DictReader(f))
            applied = 0
            for row in fresh:
                upd = updates.get((row.get("logged_at", ""), row.get("contract_ticker", "")))
                if upd:
                    row.update(upd)
                    applied += 1
            tmp = path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(fresh)
            os.replace(tmp, path)
        print(f"  [scan_archive_15m:{asset}] Filled {applied} outcomes → {path.name}")

    return updated
