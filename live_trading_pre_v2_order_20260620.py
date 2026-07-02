"""
Live order placement layer for Kalshi.

Provides place_order(), get_balance(), get_open_positions(), and daily
loss-limit helpers. Used by paper_trade_runner.py when --live is active.

All order amounts are in dollars; Kalshi API uses integer cents (0–99).
"""

import csv
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from kalshi_python_sync import KalshiAuth
from live_signal import BASE_URL

LIVE_TRADES_CSV = Path(__file__).parent / "results" / "live_trades.csv"  # BTC default


def get_live_csv_path(asset: str = "BTC") -> Path:
    """Return the asset-specific live trades CSV path."""
    asset = asset.upper()
    if asset == "BTC":
        return LIVE_TRADES_CSV
    return Path(__file__).parent / "results" / f"live_trades_{asset.lower()}.csv"

LIVE_CSV_COLUMNS = [
    "logged_at",
    "contract_ticker",
    "side",
    "count",
    "yes_price_cents",
    "live_cost",        # dollars paid (count * price / 100)
    "order_id",
    "order_status",
    "asset",
    # mirrors from paper row
    "spot",
    "strike",
    "offset_pct",
    "p_market",
    "p_yes_model",
    "net_edge",
    "bet_amount",
    "bankroll",
    "resolved_yes",     # filled by settle_live_trades()
    "live_pnl",         # filled by settle_live_trades(); unsettled = -live_cost (at risk)
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _kalshi_post(path: str, body: dict, auth: KalshiAuth) -> dict:
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    headers.update(auth.create_auth_headers("POST", url))
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        if not resp.ok:
            print(f"  [live][http {resp.status_code}] POST {path}: {resp.text[:300]}")
            return {}
        return resp.json()
    except Exception as exc:
        print(f"  [live][error] POST {path}: {exc}")
        return {}


def _kalshi_get(path: str, params: dict, auth: KalshiAuth) -> dict:
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    headers.update(auth.create_auth_headers("GET", url))
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if not resp.ok:
            print(f"  [live][http {resp.status_code}] GET {path}: {resp.text[:120]}")
            return {}
        return resp.json()
    except Exception as exc:
        print(f"  [live][error] GET {path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def get_balance(auth: KalshiAuth) -> Optional[float]:
    """Return available portfolio balance in dollars, or None on failure."""
    data = _kalshi_get("/portfolio/balance", {}, auth)
    cents = data.get("balance")
    if cents is None:
        return None
    return cents / 100.0


def get_open_positions(auth: KalshiAuth) -> list:
    """Return list of unsettled market position dicts."""
    data = _kalshi_get("/portfolio/positions", {"settlement_status": "unsettled"}, auth)
    return data.get("market_positions", [])


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def compute_order_params(
    side: str,
    bet_amount: float,
    bid: float,
    ask: float,
    max_contracts: int = 20,
) -> tuple:
    """
    Compute (yes_price_cents, count) for a limit buy order.

    YES buy: pay the ask → yes_price = ceil(ask * 100)
    NO buy:  pay the NO ask (= 100 - YES bid) → yes_price = floor(bid * 100)
             because submitting yes_price = bid is equivalent to paying
             (100 - bid) cents for NO, which matches the current NO ask.

    Returns (yes_price_cents: int, count: int).
    count is capped at max_contracts and minimum 1.
    """
    if side == "yes":
        # +1 cent above ask to sweep the next order-book level and improve fill rate.
        yes_price = math.ceil(ask * 100) + 1
        cost_per  = yes_price / 100.0
    else:
        yes_price = math.floor(bid * 100)
        no_price  = 100 - yes_price
        cost_per  = no_price / 100.0

    yes_price = max(1, min(99, yes_price))
    cost_per  = max(0.01, cost_per)
    count     = round(bet_amount / cost_per)
    if count == 0:
        return yes_price, 0  # bet_amount too small for one contract; caller should skip
    count     = min(count, max_contracts)
    return yes_price, count


def place_order(
    auth: KalshiAuth,
    ticker: str,
    side: str,
    count: int,
    yes_price: int,
) -> dict:
    """
    Submit a limit buy order to POST /portfolio/orders.

    Args:
        auth:       KalshiAuth instance
        ticker:     contract ticker, e.g. "KXBTCD-26APR0520-T69099.99"
        side:       "yes" or "no"
        count:      number of whole contracts to buy
        yes_price:  price in cents (1–99); for NO orders this is the YES-side
                    limit price (so NO executes at 100 - yes_price cents)

    Returns:
        Kalshi API response dict, or {} on failure.
    """
    if count < 1:
        print(f"  [live] Skipping — count={count} is 0")
        return {}

    body = {
        "ticker":    ticker,
        "action":    "buy",
        "side":      side,
        "count":     count,
        "type":      "limit",
        "yes_price": yes_price,
    }

    cost = count * (yes_price if side == "yes" else (100 - yes_price)) / 100.0
    print(f"  [live] Placing order: {ticker}  {side.upper()} x{count} @ {yes_price}¢"
          f"  (cost ≈ ${cost:.2f})")
    result = _kalshi_post("/portfolio/orders", body, auth)

    if result:
        order   = result.get("order", {})
        oid     = order.get("order_id", "?")
        status  = order.get("status", "?")
        filled  = order.get("count_filled", 0)
        print(f"  [live] Order result: id={oid}  status={status}  filled={filled}/{count}")
    else:
        print("  [live] Order placement FAILED — no response")

    return result


# ---------------------------------------------------------------------------
# Daily loss limit — file-based (survives restarts, accounts for settlements)
#
# Exposure = sum of today's trades where:
#   - Unsettled trade : -live_cost          (capital at risk)
#   - Settled win     : +profit             (count × yes_price_cents / 100)
#   - Settled loss    : -live_cost          (full cost lost)
#
# Halts when net exposure <= -limit (i.e. net losses + at-risk >= limit).
# Winning trades reduce the exposure so recovered capital doesn't count.
# ---------------------------------------------------------------------------

def compute_daily_exposure(csv_path: Path = None) -> float:
    """
    Return today's net P&L exposure from live_trades.csv.
    Negative = net loss / at-risk. Zero = break-even or no trades.
    """
    path = csv_path or LIVE_TRADES_CSV
    if not path.exists():
        return 0.0
    try:
        df = pd.read_csv(path)
        if df.empty:
            return 0.0
        _local_tz = datetime.now().astimezone().tzinfo
        df["_date"] = pd.to_datetime(df["logged_at"], utc=True).dt.tz_convert(_local_tz).dt.date
        today_df = df[df["_date"] == date.today()]
        if today_df.empty:
            return 0.0

        total = 0.0
        for _, r in today_df.iterrows():
            cost      = float(r["live_cost"])
            count     = int(r["count"])
            yes_price = int(r["yes_price_cents"])
            side      = str(r["side"])
            resolved  = str(r.get("resolved_yes", "")).strip()

            if resolved == "" or resolved.lower() == "nan":
                # Unsettled — full cost at risk
                total -= cost
            else:
                resolved_yes = resolved.lower() == "true"
                won = (side == "yes" and resolved_yes) or (side == "no" and not resolved_yes)
                if won:
                    # Profit = contracts × winning side price
                    profit = count * (yes_price if side == "no" else (100 - yes_price)) / 100.0
                    total += profit
                else:
                    total -= cost
        return round(total, 4)
    except Exception as exc:
        print(f"  [live] exposure calc error: {exc}")
        return 0.0


def record_wagered(amount: float) -> None:
    """No-op kept for backwards compatibility — exposure now computed from file."""
    pass


def check_daily_loss_limit(limit: float, csv_path: Path = None) -> bool:
    """
    Return True if today's net exposure is within the limit (safe to trade).
    Winning trades reduce exposure so recovered capital doesn't count.
    """
    exposure = compute_daily_exposure(csv_path)
    if exposure <= -abs(limit):
        print(f"  [live] Daily loss limit reached: net={exposure:.2f} <= -{abs(limit):.2f}"
              f" — no more live trades today")
        return False
    print(f"  [live] Daily exposure: ${exposure:.2f}  (limit: -${abs(limit):.2f})")
    return True


# ---------------------------------------------------------------------------
# Live trade logging
# ---------------------------------------------------------------------------

def ensure_live_csv_exists(csv_path: Path = None) -> None:
    path = csv_path or LIVE_TRADES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LIVE_CSV_COLUMNS).writeheader()
        print(f"  [live] Created {path}")


def log_live_trade(
    row: dict,
    order_result: dict,
    yes_price_cents: int,
    count: int,
    side: str,
    asset: str,
    csv_path: Path = None,
) -> None:
    """Append one row to the live trades log."""
    path = csv_path or LIVE_TRADES_CSV
    ensure_live_csv_exists(path)

    order        = order_result.get("order", {})
    count_filled = int(order.get("count_filled", 0) or 0)
    # Use actual filled count when the API reports at least one fill immediately.
    # If nothing filled yet (purely resting), log the intended count as potential exposure.
    actual_count = count_filled if count_filled > 0 else count
    cost         = actual_count * (yes_price_cents if side == "yes" else (100 - yes_price_cents)) / 100.0
    if count_filled > 0 and count_filled < count:
        print(f"  [live] Partial fill: {count_filled}/{count} contracts filled → logging ${cost:.2f}")

    live_row = {
        "logged_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "contract_ticker": row.get("contract_ticker", ""),
        "side":            side,
        "count":           actual_count,
        "yes_price_cents": yes_price_cents,
        "live_cost":       round(cost, 4),
        "order_id":        order.get("order_id", ""),
        "order_status":    order.get("status", "failed"),
        "asset":           asset,
        "spot":            row.get("spot", ""),
        "strike":          row.get("strike", ""),
        "offset_pct":      row.get("offset_pct", ""),
        "p_market":        row.get("p_market", ""),
        "p_yes_model":     row.get("p_yes_model", ""),
        "net_edge":        row.get("net_edge", ""),
        "bet_amount":      row.get("bet_amount", ""),
        "bankroll":        row.get("bankroll", ""),
        "resolved_yes":    "",
        "live_pnl":        "",
    }

    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LIVE_CSV_COLUMNS).writerow(live_row)
    print(f"  [live] Logged → {path.name}")


# ---------------------------------------------------------------------------
# Live trade settlement
# ---------------------------------------------------------------------------

def settle_live_trades(auth: KalshiAuth, csv_path: Path = None) -> int:
    """
    Check Kalshi for settled contracts in live_trades.csv and fill in
    resolved_yes and live_pnl. Mirrors outcome_checker logic for live trades.
    Returns number of rows updated.
    """
    from live_signal import kalshi_get
    from datetime import timezone

    path = csv_path or LIVE_TRADES_CSV
    if not path.exists():
        return 0

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    now_utc = datetime.now(timezone.utc)
    updated = 0

    for row in rows:
        ticker   = (row.get("contract_ticker") or "").strip()
        resolved = (row.get("resolved_yes") or "").strip()
        if not ticker or (resolved and resolved.lower() != "nan"):
            continue  # already settled or no ticker

        # Fetch market status
        data   = kalshi_get(f"/markets/{ticker}", {}, auth)
        market = data.get("market") or {}
        status = market.get("status", "")
        if status not in ("settled", "resolved", "finalized", "determined"):
            continue

        result       = (market.get("result") or "").lower()
        resolved_yes = result == "yes"
        row["resolved_yes"] = str(resolved_yes)

        # Compute live P&L
        count     = int(row["count"])
        yes_price = int(row["yes_price_cents"])
        side      = row["side"]
        cost      = float(row["live_cost"])
        won       = (side == "yes" and resolved_yes) or (side == "no" and not resolved_yes)

        if won:
            profit = count * (yes_price if side == "no" else (100 - yes_price)) / 100.0
            row["live_pnl"] = str(round(profit, 4))
        else:
            row["live_pnl"] = str(round(-cost, 4))

        updated += 1
        pnl_str = row["live_pnl"]
        print(f"  [live] Settled: {ticker}  resolved_yes={resolved_yes}  live_pnl={pnl_str}")

    if updated > 0:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LIVE_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [live] Settled {updated} live trade(s)")

    return updated
