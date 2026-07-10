"""
scan_archive.py

Logs every evaluated contract in the scan loop to
results/{asset}_scan_archive.csv. Each row captures a 29-feature
signal snapshot at evaluation time. resolved_yes is backfilled
post-expiry by fill_scan_outcomes().

Covers ALL assets (BTC, ETH, SOL) so each can eventually train its
own unbiased LGBM model. BTC continues writing to btc_scan_archive.csv
for backward compatibility; ETH/SOL write to eth_scan_archive.csv /
sol_scan_archive.csv.
"""

import csv
import fcntl
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

_RESULTS_DIR = Path(__file__).parent / "results"


@contextmanager
def _archive_lock(path: Path):
    """Exclusive advisory lock serializing appends and rewrites per archive file.

    2026-07-02: fill_scan_outcomes used to read-all → slow API loop → full rewrite
    with no lock and no atomic replace. A hung call finishing late rewrote the file
    from a stale in-memory copy, which is how btc_scan_archive.csv lost all rows
    logged 06-04→06-24. The lock plus re-read-before-write plus os.replace closes
    every variant of that race.
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
    "p_gbdt",
    "p_up_v2",
    # Features
    "offset_pct",
    "composite_p_up",
    "composite_trend",
    "composite_rev",
    "ema_stack_bias",
    "ema_stretch_score",
    "vwap_stretch_score",
    "vwap_distance_pct",
    "stoch_k",
    "chg_30m",
    "chg_10m",
    "chg_5m",
    "bp_5m",
    "body_15m",
    "dir_15m",
    "vol_score",
    "vpin_score",
    "obi_score",
    "confirmation_score",
    "no_score",
    "funding_bias",
    "vol_eff",
    "pm_drift_5m",
    "adx_1h",
    "rvol_1h",
    "squeeze_1h",
    "liq_score",
    "liq_bias",
    "ls_long_pct",
    "oi_chg_pct",
    # Sigma DC 1% swing levels (BTC only, shadow signal)
    "sigma_swing_high_1pct",
    "sigma_dist_high_1pct",
    # Flag/pennant pattern signals (BTC only, shadow)
    "flag_signal",          # +1 bull active, -1 bear active, 0 none
    "flag_bull_bars_ago",   # bars since last confirmed bull flag/pennant (-1 = none)
    "flag_bear_bars_ago",   # bars since last confirmed bear flag/pennant (-1 = none)
    "flag_bull_tip_y",      # price at top of bull pole
    "flag_bear_tip_y",      # price at bottom of bear pole
    "flag_bull_pole_pct",   # pole height %
    "flag_bear_pole_pct",   # pole depth %
    # Ichimoku signals (BTC only, shadow)
    "ichi_bear",        # 1 = tenkan<kijun AND span_a<span_b on 1h; 0 otherwise
    "cloud_thick_pct",  # |span_a - span_b| / close × 100 (cloud thickness %)
    # Macro HMM regime posteriors (BTC only, shadow — 3-state: Bull/Sideways/Bear)
    "macro_regime_bull",   # P(Bull) from 1h directional HMM
    "macro_regime_sdwy",   # P(Sideways)
    "macro_regime_bear",   # P(Bear)
    # 15m directional p_up (BTC only, shadow 2026-07-10 — no consumer)
    "pup15m",              # P(next 15m bar up), (trend15, rev5) calibration tables
    # Outcome (backfilled)
    "resolved_yes",
    "spot_at_expiry", "price_move_pct", "miss_pct",
]

_FEATURE_COLS = {c for c in COLUMNS if c not in {
    "logged_at", "contract_ticker", "close_ts", "spot", "strike",
    "p_market", "tau_minutes", "p_gbdt", "p_up_v2", "resolved_yes",
}}
# p_up_v2 added to the exclusion set 2026-07-02: it is set explicitly from the
# p_up_v2 parameter in log_scan_row, but was missing here, so the feature loop
# overwrote it with features.get("p_up_v2") → None → "" — the column stayed
# 100% empty for every asset since it was introduced.


def get_archive_path(asset: str) -> Path:
    return _RESULTS_DIR / f"{asset.lower()}_scan_archive.csv"


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
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return round(v, digits)
    return v


def log_scan_row(
    ticker: str,
    close_ts: str,
    spot: float,
    strike: float,
    p_market: float,
    tau_minutes: float,
    features: dict,
    p_gbdt: Optional[float],
    asset: str = "BTC",
    now_utc: Optional[datetime] = None,
    p_up_v2: Optional[float] = None,
) -> None:
    """Append one evaluated-contract row to the asset's scan archive."""
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
        "p_gbdt":          _fmt(p_gbdt, 4) if p_gbdt is not None else "",
        "p_up_v2":         _fmt(p_up_v2, 4) if p_up_v2 is not None else "",
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
            print(f"  [scan_archive:{asset}] No auth — cannot fill outcomes")
            return 0
    except Exception as e:
        print(f"  [scan_archive:{asset}] Import error: {e}")
        return 0

    # Collect updates keyed by row identity instead of mutating the initial read.
    # The fetch loop below can run for a long time (one API call per ticker); rows
    # appended meanwhile must not be lost, so the write path re-reads the file.
    _cache: dict[str, Optional[int]] = {}
    updates: dict[tuple, dict] = {}  # (logged_at, contract_ticker) -> field updates
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
                from live_signal import fetch_spot_at_time
                close_ts  = row.get("close_ts", "")
                spot_scan = float(row.get("spot") or 0)
                strike    = float(row.get("strike") or 0)
                # NOTE (2026-07-01 audit): fetch_spot_at_time returns the Binance 1m
                # price, which sits ~+14bp above the Kalshi settlement source. Fine
                # for price_move/miss diagnostics, but do NOT derive win/loss labels
                # from spot_at_expiry vs strike — use resolved_yes (Kalshi API).
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
            # Re-read so rows appended during the fetch loop survive the rewrite.
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
        print(f"  [scan_archive:{asset}] Filled {applied} outcomes → {path.name}")

    return updated
