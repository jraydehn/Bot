"""
btc_scan_archive.py

Logs every BTC contract evaluated in the scan loop to
results/btc_scan_archive.csv. Each row captures the full 32-feature
LGBM signal snapshot at evaluation time. resolved_yes is backfilled
post-expiry by fill_scan_outcomes().

This gives the LGBM a training set covering ALL evaluated opportunities,
not just the one contract selected per scan cycle.

Retrain LGBM from this archive once enough resolved rows accumulate:
    python3 train_btc_lgbm.py  (update DATA_SOURCE to use archive)
"""

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ARCHIVE_CSV = Path(__file__).parent / "results" / "btc_scan_archive.csv"

COLUMNS = [
    "logged_at",
    "contract_ticker",
    "close_ts",
    "spot",
    "strike",
    "p_market",
    "tau_minutes",
    "p_gbdt",
    # LGBM features (32)
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
    "adx_1h",
    "rvol_1h",
    "squeeze_1h",
    "liq_score",
    "liq_bias",
    "ls_long_pct",
    "oi_chg_pct",
    # Outcome (backfilled)
    "resolved_yes",
]

_FEATURE_COLS = {c for c in COLUMNS if c not in {
    "logged_at", "contract_ticker", "close_ts", "spot", "strike",
    "p_market", "tau_minutes", "p_gbdt", "resolved_yes",
}}


def _ensure_csv() -> None:
    ARCHIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_CSV.exists():
        with open(ARCHIVE_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        return
    with open(ARCHIVE_CSV, newline="") as f:
        existing = csv.DictReader(f).fieldnames or []
    new_cols = [c for c in COLUMNS if c not in existing]
    if not new_cols:
        return
    with open(ARCHIVE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in new_cols:
            row.setdefault(col, "")
    with open(ARCHIVE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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
    now_utc: Optional[datetime] = None,
) -> None:
    """Append one ghost-eval row for a BTC contract."""
    _ensure_csv()
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
        "resolved_yes":    "",
    }
    for col in _FEATURE_COLS:
        row[col] = _fmt(features.get(col))

    with open(ARCHIVE_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore").writerow(row)


def fill_scan_outcomes(auth=None) -> int:
    """Backfill resolved_yes for settled contracts. Returns rows updated."""
    if not ARCHIVE_CSV.exists():
        return 0

    with open(ARCHIVE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if not (r.get("resolved_yes") or "").strip()]
    if not pending:
        return 0

    try:
        from outcome_checker import fetch_market, is_settled, parse_resolution
        from live_signal import load_auth
        _auth = auth or load_auth()
        if _auth is None:
            print("  [scan_archive] No auth — cannot fill outcomes")
            return 0
    except Exception as e:
        print(f"  [scan_archive] Import error: {e}")
        return 0

    # Cache per ticker to avoid redundant API calls
    _cache: dict[str, Optional[int]] = {}
    updated = 0
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
            row["resolved_yes"] = str(_cache[ticker])
            updated += 1

    if updated:
        with open(ARCHIVE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  [scan_archive] Filled {updated} outcomes → {ARCHIVE_CSV.name}")

    return updated
