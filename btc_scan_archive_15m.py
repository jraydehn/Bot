"""
btc_scan_archive_15m.py

Ghost-logs every BTC contract evaluated in each 15m scan cycle to
results/btc_scan_archive_15m.csv. Captures the full extended signal
snapshot at evaluation time. resolved_yes is backfilled post-expiry
by fill_scan_outcomes().

Gives the 15m LGBM a training set covering ALL evaluated opportunities,
not just one per scan. Retrain when 2,000+ resolved rows accumulate:
    python3 train_btc_lgbm_15m.py  (point DATA_PATH at this file)
"""

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ARCHIVE_CSV = Path(__file__).parent / "results" / "btc_scan_archive_15m.csv"

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
    # contract structure (also features for LGBM)
    "offset_pct",
    # Outcome (backfilled)
    "resolved_yes",
]

_META_COLS = {
    "logged_at", "contract_ticker", "close_ts", "spot", "strike",
    "p_market", "tau_minutes", "spread", "p_model_yes", "p_model_no",
    "resolved_yes",
}
_FEATURE_COLS = [c for c in COLUMNS if c not in _META_COLS]


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
    now_utc: Optional[datetime] = None,
) -> None:
    """Append one ghost-eval row for an evaluated BTC 15m contract."""
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
        "spread":          _fmt(spread, 4),
        "p_model_yes":     _fmt(p_model_yes, 4),
        "p_model_no":      _fmt(p_model_no, 4),
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
            print("  [scan_archive_15m] No auth — cannot fill outcomes")
            return 0
    except Exception as e:
        print(f"  [scan_archive_15m] Import error: {e}")
        return 0

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
        print(f"  [scan_archive_15m] Filled {updated} outcomes → {ARCHIVE_CSV.name}")

    return updated
