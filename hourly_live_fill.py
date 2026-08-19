"""Shared live-fill helper for the hourly fav/rescue paper runners — 2026-08-19.

User directive: these paper books must record what a LIVE taker order
would actually get, not the archived mid. At trade time we fetch the
contract's current quote and book at the executable side (YES ask /
NO ask = 1 - yes_bid). If the price has run beyond the band cap (+1c
buffer) or no quote exists, the row is recorded with filled=0 and zero
PnL — the signal fired, the fill did not exist (attempts stay visible
per feedback_sim_methodology). One-shot per contract, no chasing —
matches the backtest's first-entry semantics and the delayed-fill sim.

No auth -> fatal at startup (feedback_no_sim_mode: never silently
degrade to mid-price accounting).
"""
import sys
import time

from live_signal import load_auth, kalshi_get


def load_auth_or_die(tag: str):
    auth = load_auth()
    if auth is None:
        print(f"{tag} FATAL: no Kalshi auth (env or .kalshi_config). "
              f"Live-fill accounting requires real quotes; refusing to run at mid.")
        sys.exit(1)
    return auth


def _price(v):
    if v is None:
        return None
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    if p > 1.5:  # cents-integer form from a non-dollars field
        p = p / 100.0
    return p if 0.0 < p < 1.0 else None


def fill_for(auth, ticker: str, side: str, cap_cost: float):
    """Executable cost for `side` on `ticker` right now.

    Returns (cost, filled): cost is the taker price (YES ask, or NO ask
    reconstructed as 1 - yes_bid); filled is False when there is no valid
    quote or the cost exceeds cap_cost (price ran away from the band).
    """
    time.sleep(0.15)  # polite pacing during cluster bursts
    resp = kalshi_get("/markets", {"tickers": ticker}, auth)
    mkts = resp.get("markets") if isinstance(resp, dict) else None
    if not mkts:
        return None, False
    q = mkts[0]
    ask = _price(q.get("yes_ask_dollars"))
    bid = _price(q.get("yes_bid_dollars"))
    if side == "yes":
        cost = ask
    else:
        cost = None if bid is None else round(1.0 - bid, 4)
    if cost is None or not (0.0 < cost < 1.0):
        return None, False
    return cost, cost <= cap_cost
