"""
position_monitor.py — Ghost exit evaluator for open live positions.

Each cycle, for every open position in live_trades.csv, this module:
  1. Fetches current bid/ask from Kalshi
  2. Checks if the position has appreciated 3× or more
  3. Re-runs the same p_model the entry scanner uses (with current indicators)
  4. Logs the ghost exit decision to results/position_monitor.csv

NO orders are placed. The log accumulates over time and can be analyzed after
expiry to compare: when would_sell=True, did selling at current_bid beat
holding to expiry? Run outcome_checker.py to fill resolved_yes and held_pnl.
"""

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MONITOR_CSV = Path(__file__).parent / "results" / "position_monitor.csv"

MONITOR_COLUMNS = [
    "logged_at",
    "contract_ticker",
    "asset",
    "side",
    "count",
    "yes_price_cents",     # entry price in cents (for held_pnl computation)
    "entry_p_market",      # entry mid-price in [0,1]
    "entry_logged_at",     # when we entered the position
    "current_bid",         # current yes_bid in [0,1]
    "current_ask",         # current yes_ask in [0,1]
    "current_mid",         # (bid+ask)/2
    "appreciation_x",      # current_mid / entry_p_market
    "tau_remaining",       # minutes to expiry at evaluation time
    "p_model_rerun",       # model's current probability estimate
    "model_edge",          # p_model_rerun - current_mid (negative = sell signal)
    "threshold_met",       # True if appreciation_x >= 3.0
    "would_sell",          # True if any sell condition fires
    "sell_reason",         # which condition(s) triggered: "3x_edge", "pm_lock", or both
    "if_sold_pnl",         # locked-in PnL if sold at bid right now (deterministic)
    "would_add",           # True if model edge improved and price dropped below entry
    "resolved_yes",        # filled by outcome_checker.py after expiry
    "held_pnl",            # filled by outcome_checker.py: pnl from holding to expiry
]

SELL_SIGNAL_THRESHOLD  = 3.0    # appreciation_x threshold for edge-based sell signal
SELL_EDGE_THRESHOLD    = -0.05  # model_edge must be below this to flag would_sell (edge signal)
PM_LOCK_THRESHOLD      = 0.96   # p_market level at which locking in beats risking the remaining 4¢
PM_LOCK_MIN_TAU        = 5.0    # tau must be >= this (minutes) for pm_lock signal to fire
ADD_EDGE_THRESHOLD     = 0.10   # model_edge must exceed this to flag would_add (strong positive edge)
ADD_MIN_TAU            = 10.0   # tau must be >= this (minutes) to flag would_add

# Price+time exit signals (validated on n=27 ghost positions, 2026-05-07)
STOP_LOSS_APPX   = 0.30   # YES: exit if appreciation falls to ≤30% of entry near expiry
STOP_LOSS_TAU    = 15.0   # YES: stop-loss fires only inside this tau window (minutes)
PROFIT_LOCK_APPX = 1.5    # exit if appreciation ≥ 1.5× AND near-certain price AND short tau
PROFIT_LOCK_MID  = 0.85   # profit-lock requires mid ≥ 0.85 (near resolution, not mid-range)
PROFIT_LOCK_TAU  = 20.0   # profit-lock fires only inside this tau window (minutes)


def _ensure_monitor_csv() -> None:
    MONITOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not MONITOR_CSV.exists():
        with open(MONITOR_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MONITOR_COLUMNS).writeheader()
        return
    # Migrate new columns if needed
    with open(MONITOR_CSV, newline="") as f:
        existing = csv.DictReader(f).fieldnames or []
    new_cols = [c for c in MONITOR_COLUMNS if c not in existing]
    if new_cols:
        with open(MONITOR_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in new_cols:
                row.setdefault(col, "")
        with open(MONITOR_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MONITOR_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _fetch_market_prices(ticker: str, auth) -> tuple:
    """Return (yes_bid, yes_ask, close_time) in [0,1] or (None, None, None) on failure."""
    try:
        from live_signal import kalshi_get
        data = kalshi_get(f"/markets/{ticker}", {}, auth)
        market = data.get("market", {})
        # API returns yes_bid_dollars / yes_ask_dollars (in dollars, not cents)
        bid_d  = market.get("yes_bid_dollars")
        ask_d  = market.get("yes_ask_dollars")
        close_time = market.get("close_time", "")
        if bid_d is None or ask_d is None:
            return None, None, None
        return float(bid_d), float(ask_d), close_time
    except Exception:
        return None, None, None


def _read_open_positions(asset: str) -> list:
    """Read unresolved executed positions from live_trades CSV for this asset."""
    try:
        from live_trading import get_live_csv_path
        csv_path = get_live_csv_path(asset)
        if not csv_path.exists():
            return []
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return [
            r for r in rows
            if r.get("order_status", "").strip() == "executed"
            and not (r.get("resolved_yes") or "").strip()
        ]
    except Exception:
        return []


def _compute_p_model(
    asset: str,
    strike: float,
    spot: float,
    tau_minutes: float,
    vol_multi: float,
    current_mid: float,
    composite_trend: float,
    composite_rev: float,
    df_1m, df_1h, df_4h, df_15m,
) -> Optional[float]:
    """Re-run the entry p_model with current market state."""
    try:
        import direct_p_model as _dpm
        from composite_scorer import score_to_p_model
        from probability_engine import implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT_BY_ASSET, REALIZED_VOL_WEIGHT

        offset_pct  = strike / spot - 1
        vol_imp     = implied_vol_from_price(current_mid, spot, strike, tau_minutes)
        _vol_weight = REALIZED_VOL_WEIGHT_BY_ASSET.get(asset, REALIZED_VOL_WEIGHT)
        vol_eff     = blend_vol(vol_multi, vol_imp, weight=_vol_weight) if vol_imp and vol_imp > 0 else vol_multi
        sigma_tau   = vol_eff * math.sqrt(max(tau_minutes, 1))

        if _dpm.asset_supported(asset) and df_1m is not None and df_1h is not None:
            p = _dpm.compute_p_model_direct(
                asset=asset,
                df_1m=df_1m, df_1h=df_1h,
                df_4h=df_4h, df_15m=df_15m,
                offset_pct=offset_pct,
                composite_trend=composite_trend,
                composite_rev=composite_rev,
            )
            if p is not None:
                return float(p)

        return float(score_to_p_model(
            composite_trend, composite_rev, spot, strike, sigma_tau, asset=asset
        ))
    except Exception as e:
        print(f"  [position_monitor] p_model re-run failed ({asset}): {e}")
        return None


def evaluate_open_positions(
    auth,
    asset: str,
    spot: float,
    vol_multi: float,
    composite_trend: float,
    composite_rev: float,
    df_1m,
    df_1h,
    df_4h,
    df_15m,
    now_utc: datetime,
) -> None:
    """
    Evaluate all open live positions for this asset. Log ghost exit decisions.
    No orders are placed — data collection only.
    """
    if auth is None:
        return

    _ensure_monitor_csv()
    positions = _read_open_positions(asset)
    if not positions:
        return

    from live_signal import minutes_to_expiry

    new_rows = []
    for pos in positions:
        ticker = pos.get("contract_ticker", "").strip()
        if not ticker:
            continue

        try:
            entry_p_market  = float(pos.get("p_market", 0) or 0)
            yes_price_cents = float(pos.get("yes_price_cents", 0) or 0)
            count           = int(float(pos.get("count", 0) or 0))
            side            = pos.get("side", "yes").strip().lower()
            strike          = float(pos.get("strike", 0) or 0)
            entry_logged_at = pos.get("logged_at", "")
        except (ValueError, TypeError):
            continue

        if entry_p_market <= 0 or count == 0 or strike <= 0:
            continue

        bid, ask, close_time = _fetch_market_prices(ticker, auth)
        if bid is None:
            continue

        # Skip already-expired contracts
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                if close_dt <= now_utc:
                    continue
            except Exception:
                pass

        mid             = (bid + ask) / 2.0
        appreciation_x  = round(mid / entry_p_market, 3) if entry_p_market > 0 else 0.0
        tau_remaining   = minutes_to_expiry(close_time) if close_time else float("nan")

        p_model = _compute_p_model(
            asset=asset, strike=strike, spot=spot,
            tau_minutes=max(tau_remaining, 1) if tau_remaining == tau_remaining else 1,
            vol_multi=vol_multi, current_mid=mid,
            composite_trend=composite_trend, composite_rev=composite_rev,
            df_1m=df_1m, df_1h=df_1h, df_4h=df_4h, df_15m=df_15m,
        )

        model_edge    = round(p_model - mid, 4) if p_model is not None else None
        threshold_met = appreciation_x >= SELL_SIGNAL_THRESHOLD

        _tau_valid = tau_remaining == tau_remaining  # NaN check

        # Signal 1 (legacy): edge-based — 3× appreciation + model says price too high
        _sell_edge = (threshold_met
                      and model_edge is not None
                      and model_edge < SELL_EDGE_THRESHOLD)

        # Signal 2 (legacy): pm_lock — near certain resolution with reversal risk remaining
        _pm_near_resolution = (
            (side == "yes" and mid >= PM_LOCK_THRESHOLD) or
            (side == "no"  and mid <= (1.0 - PM_LOCK_THRESHOLD))
        )
        _sell_pm = (_tau_valid and _pm_near_resolution and tau_remaining >= PM_LOCK_MIN_TAU)

        # Signal 3: stop_loss — YES position deeply underwater with < 15 min left
        # Backtest n=27: 7/7 correct, +$67 delta. Cuts losses before riding to zero.
        _stop_loss = (side == "yes"
                      and _tau_valid
                      and appreciation_x <= STOP_LOSS_APPX
                      and tau_remaining < STOP_LOSS_TAU)

        # Signal 4: profit_lock — 1.5× gain AND mid ≥ 0.85 AND < 20 min left
        # Backtest n=27: 2 correct (+$70), 3 partial/wrong (-$30), net +$40.
        # Prevents a near-certain winner from reversing to worthless in the final minutes.
        _profit_lock = (_tau_valid
                        and appreciation_x >= PROFIT_LOCK_APPX
                        and mid >= PROFIT_LOCK_MID
                        and tau_remaining < PROFIT_LOCK_TAU)

        would_sell  = _sell_edge or _sell_pm or _stop_loss or _profit_lock
        sell_reason = ",".join(filter(None, [
            "3x_edge"     if _sell_edge    else "",
            "pm_lock"     if _sell_pm      else "",
            "stop_loss"   if _stop_loss    else "",
            "profit_lock" if _profit_lock  else "",
        ]))

        # Add signal: price dropped below entry AND model edge is now stronger
        would_add = (
            model_edge is not None
            and model_edge >= ADD_EDGE_THRESHOLD
            and appreciation_x < 1.0
            and _tau_valid
            and tau_remaining >= ADD_MIN_TAU
        )

        # Deterministic locked-in PnL if sold at bid now
        if side == "yes" and yes_price_cents > 0:
            if_sold_pnl = round(count * (bid - yes_price_cents / 100.0), 2)
        elif side == "no" and yes_price_cents > 0:
            # NO bid = 1 - YES ask; cost per NO = (100 - yes_price_cents) / 100
            if_sold_pnl = round(count * ((1.0 - ask) - (100.0 - yes_price_cents) / 100.0), 2)
        else:
            if_sold_pnl = ""

        row = {
            "logged_at":       now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "contract_ticker": ticker,
            "asset":           asset,
            "side":            side,
            "count":           count,
            "yes_price_cents": yes_price_cents,
            "entry_p_market":  round(entry_p_market, 4),
            "entry_logged_at": entry_logged_at,
            "current_bid":     round(bid,  4),
            "current_ask":     round(ask,  4),
            "current_mid":     round(mid,  4),
            "appreciation_x":  appreciation_x,
            "tau_remaining":   round(tau_remaining, 2) if tau_remaining == tau_remaining else "",
            "p_model_rerun":   round(p_model, 4) if p_model is not None else "",
            "model_edge":      model_edge if model_edge is not None else "",
            "threshold_met":   threshold_met,
            "would_sell":      would_sell,
            "sell_reason":     sell_reason,
            "if_sold_pnl":     if_sold_pnl,
            "would_add":       would_add,
            "resolved_yes":    "",
            "held_pnl":        "",
        }
        new_rows.append(row)

        if would_sell:
            status = f"WOULD_SELL ({sell_reason})"
        elif would_add:
            status = "WOULD_ADD"
        elif threshold_met:
            status = "watch"
        else:
            status = "hold"
        _pm_str  = f"{p_model:.3f}"    if p_model    is not None else "n/a"
        _edg_str = f"{model_edge:+.3f}" if model_edge is not None else "n/a"
        print(
            f"  [pos_monitor] {ticker}  {appreciation_x:.2f}×  "
            f"mid={mid:.3f}  p_model={_pm_str}  "
            f"edge={_edg_str}  "
            f"tau={tau_remaining:.0f}m  → {status}"
        )

    if new_rows:
        with open(MONITOR_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MONITOR_COLUMNS, extrasaction="ignore")
            writer.writerows(new_rows)


# ---------------------------------------------------------------------------
# Outcome filling — called by outcome_checker.py after contract settles
# ---------------------------------------------------------------------------

def fill_outcomes(auth=None) -> int:
    """
    Fill resolved_yes and held_pnl for settled positions in position_monitor.csv.
    Returns number of rows updated.
    """
    if not MONITOR_CSV.exists():
        return 0

    with open(MONITOR_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if not (r.get("resolved_yes") or "").strip()]
    if not pending:
        return 0

    from live_signal import kalshi_get
    from outcome_checker import fetch_market, is_settled, parse_resolution

    updated = 0
    for row in pending:
        ticker = row.get("contract_ticker", "").strip()
        if not ticker:
            continue
        try:
            from kalshi_python_sync import KalshiAuth
            from live_signal import load_auth
            _auth = auth or load_auth()
            if _auth is None:
                break
            market = fetch_market(ticker, _auth)
            if not market or not is_settled(market):
                continue
            resolved_yes = parse_resolution(market)
            row["resolved_yes"] = str(resolved_yes)

            # held_pnl: what holding to expiry gave
            try:
                count          = int(float(row.get("count", 0) or 0))
                yes_price_c    = float(row.get("yes_price_cents", 0) or 0)
                side           = row.get("side", "yes").strip().lower()
                entry_cost     = count * yes_price_c / 100.0
                if side == "yes":
                    row["held_pnl"] = str(round(count * (1.0 - yes_price_c / 100.0), 2) if resolved_yes
                                         else round(-entry_cost, 2))
                else:
                    no_cost = count * (100.0 - yes_price_c) / 100.0
                    row["held_pnl"] = str(round(count * (yes_price_c / 100.0), 2) if not resolved_yes
                                         else round(-no_cost, 2))
            except Exception:
                row["held_pnl"] = ""

            updated += 1
            print(f"  [pos_monitor] {ticker}: resolved_yes={resolved_yes}  held_pnl={row['held_pnl']}")
        except Exception as e:
            print(f"  [pos_monitor] {ticker}: error — {e}")

    if updated:
        with open(MONITOR_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MONITOR_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [pos_monitor] Updated {updated} rows in {MONITOR_CSV.name}")

    return updated
