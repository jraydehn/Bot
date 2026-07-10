"""
gate_audit_logger.py — Persist every gate-blocked trade for post-expiry auditing.

Each time a gate fires and prevents a trade, log_block() appends a row to
results/blocked_trades.csv with all signal values at block time.

After contract expiry, fill_outcomes() resolves each row using Kalshi settlement
data and computes would_pnl (what P&L would have occurred if the gate had not fired).

gate_audit_report.py reads this CSV and produces per-gate accuracy stats.
"""

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BLOCKED_CSV = Path(__file__).parent / "results" / "blocked_trades.csv"

COLUMNS = [
    # Identity
    "logged_at",
    "gate_name",
    "contract_ticker",
    "asset",
    "side",
    # Pricing
    "pm",
    "p_model",
    "net_edge",
    "offset_pct",
    "strike",
    "spot",
    "tau_minutes",
    "count",
    "kelly_fraction",
    "bankroll",
    "close_ts",
    # Model signals at block time
    "ema_stack_bias",
    "composite_trend",
    "composite_rev",
    "composite_p_up",
    "stoch_k",
    "vwap_stretch",
    "vol_score",
    "vpin_score",
    "obi_score",
    "ema_stretch",
    "structure_bias",
    "funding_bias",
    "sharp_move_active",
    "pup15m",           # 15m directional p_up (BTC only, shadow 2026-07-10)
    # Coinalyze liquidation + positioning (logged at block time)
    "liq_score",
    "liq_bias",
    "ls_long_pct",
    "oi_chg_pct",
    # Order book depth (Coinbase spot, logged at block time)
    "ob_imbalance",
    "ob_path_ask_usd",
    "ob_path_bid_usd",
    "ob_ask_frac",
    "ob_bid_wall_pct",
    "ob_ask_wall_pct",
    # Filled post-expiry by fill_outcomes()
    "resolved_yes",
    "would_pnl",
]


def _ensure_csv() -> None:
    BLOCKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not BLOCKED_CSV.exists():
        with open(BLOCKED_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        return
    with open(BLOCKED_CSV, newline="") as f:
        existing = csv.DictReader(f).fieldnames or []
    new_cols = [c for c in COLUMNS if c not in existing]
    if not new_cols:
        return
    with open(BLOCKED_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in new_cols:
            row.setdefault(col, "")
    with open(BLOCKED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(v, digits=4):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return round(v, digits)
    return v


def log_block(
    gate_name: str,
    ticker: str,
    asset: str,
    side: str,
    pm: float,
    p_model: Optional[float],
    net_edge: float,
    offset_pct: float,
    strike: float,
    spot: float,
    tau_minutes: float,
    count: int,
    kelly_fraction: float,
    close_ts: str,
    signals: dict,
    now_utc: Optional[datetime] = None,
    bankroll: float = 0.0,
) -> None:
    """Append one blocked-trade row. Call this at every gate continue."""
    _ensure_csv()
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    row = {
        "logged_at":        now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "gate_name":        gate_name,
        "contract_ticker":  ticker,
        "asset":            asset,
        "side":             side,
        "pm":               _fmt(pm),
        "p_model":          _fmt(p_model),
        "net_edge":         _fmt(net_edge),
        "offset_pct":       _fmt(offset_pct * 100, 4),   # stored as percent
        "strike":           _fmt(strike, 2),
        "spot":             _fmt(spot, 2),
        "tau_minutes":      _fmt(tau_minutes, 1),
        "count":            count,
        "kelly_fraction":   _fmt(kelly_fraction),
        "bankroll":         _fmt(bankroll, 2),
        "close_ts":         close_ts or "",
        # signals
        "ema_stack_bias":   signals.get("ema_stack_bias", ""),
        "composite_trend":  signals.get("composite_trend", ""),
        "composite_rev":    signals.get("composite_rev", ""),
        "composite_p_up":   _fmt(signals.get("composite_p_up"), 4),
        "stoch_k":          _fmt(signals.get("stoch_k"), 2),
        "vwap_stretch":     signals.get("vwap_stretch", ""),
        "vol_score":        signals.get("vol_score", ""),
        "vpin_score":       signals.get("vpin_score", ""),
        "obi_score":        signals.get("obi_score", ""),
        "ema_stretch":      signals.get("ema_stretch", ""),
        "structure_bias":   signals.get("structure_bias", ""),
        "funding_bias":     signals.get("funding_bias", ""),
        "sharp_move_active": int(signals.get("sharp_move_active", 0)),
        "pup15m":           _fmt(signals.get("pup15m") if signals.get("pup15m") != "" else None, 4),
        # coinalyze
        "liq_score":        signals.get("liq_score", ""),
        "liq_bias":         signals.get("liq_bias", ""),
        "ls_long_pct":      signals.get("ls_long_pct", ""),
        "oi_chg_pct":       signals.get("oi_chg_pct", ""),
        # order book depth
        "ob_imbalance":     signals.get("ob_imbalance", ""),
        "ob_path_ask_usd":  signals.get("ob_path_ask_usd", ""),
        "ob_path_bid_usd":  signals.get("ob_path_bid_usd", ""),
        "ob_ask_frac":      signals.get("ob_ask_frac", ""),
        "ob_bid_wall_pct":  signals.get("ob_bid_wall_pct", ""),
        "ob_ask_wall_pct":  signals.get("ob_ask_wall_pct", ""),
        # filled later
        "resolved_yes":     "",
        "would_pnl":        "",
    }

    with open(BLOCKED_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore").writerow(row)


def fill_outcomes(auth=None) -> int:
    """
    Resolve pending blocked-trade rows after contract settlement.
    Fills resolved_yes and computes would_pnl.
    Returns number of rows updated.
    """
    if not BLOCKED_CSV.exists():
        return 0

    with open(BLOCKED_CSV, newline="", errors="replace") as f:
        import io as _io
        rows = list(csv.DictReader(_io.StringIO(f.read().replace("\x00", ""))))

    pending = [r for r in rows if not (r.get("resolved_yes") or "").strip()]
    if not pending:
        return 0

    try:
        from outcome_checker import fetch_market, is_settled, parse_resolution
        from live_signal import load_auth
        _auth = auth or load_auth()
        if _auth is None:
            print("  [gate_audit] No auth — cannot fill outcomes")
            return 0
    except Exception as e:
        print(f"  [gate_audit] Import error: {e}")
        return 0

    updated = 0
    for row in pending:
        ticker = row.get("contract_ticker", "").strip()
        if not ticker:
            continue
        try:
            market = fetch_market(ticker, _auth)
            if not market or not is_settled(market):
                continue
            resolved_yes = parse_resolution(market)
            row["resolved_yes"] = str(resolved_yes)

            # Compute counterfactual PnL if the trade had been taken.
            # For pre-evaluation gates (e.g. smc_gate) count=0 is logged because
            # Kelly sizing hasn't run yet. Estimate count from bankroll + net_edge
            # using quarter-Kelly so the dollar magnitude is meaningful for auditing.
            try:
                side_r      = row.get("side", "yes").strip().lower()
                pm_val      = float(row.get("pm") or 0)
                count       = int(float(row.get("count") or 0))
                if count == 0:
                    bankroll_r    = float(row.get("bankroll") or 0)
                    kelly_frac_r  = float(row.get("kelly_fraction") or 0)
                    net_edge_r    = float(row.get("net_edge") or 0)
                    if bankroll_r > 0 and pm_val > 0:
                        if kelly_frac_r > 0:
                            frac = kelly_frac_r
                        elif net_edge_r > 0:
                            # Quarter-Kelly estimate: conservative stand-in
                            frac = min(net_edge_r / max(1.0 - net_edge_r, 0.01) * 0.25, 0.25)
                        else:
                            frac = 0.0
                        if frac > 0:
                            cost_per = pm_val if side_r == "yes" else max(1.0 - pm_val, 0.01)
                            count = max(1, int(bankroll_r * frac / cost_per))
                if side_r == "yes":
                    row["would_pnl"] = str(round(count * (1.0 - pm_val), 2) if resolved_yes
                                           else round(-count * pm_val, 2))
                else:
                    row["would_pnl"] = str(round(count * pm_val, 2) if not resolved_yes
                                           else round(-count * (1.0 - pm_val), 2))
            except Exception:
                row["would_pnl"] = ""

            updated += 1
        except Exception as e:
            print(f"  [gate_audit] {ticker}: {e}")

    if updated:
        with open(BLOCKED_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  [gate_audit] Filled {updated} blocked-trade outcomes → {BLOCKED_CSV.name}")

    return updated
