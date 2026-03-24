"""
Outcome checker — fetches Kalshi contract settlements and updates paper_trades.csv.

For each row in results/paper_trades.csv that has a contract_ticker and an empty
resolved_yes field, this script:
  1. Fetches the market detail from the Kalshi API.
  2. If the market is settled, reads the result (yes_sub_title indicates outcome).
  3. Fills in resolved_yes, would_win, and would_pnl.

Run this once after each contract expires (e.g. hourly, offset by 10 minutes):
    python3 outcome_checker.py

Requires KALSHI_KEY_ID and KALSHI_KEY_PATH env vars.
"""

import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from live_signal import load_auth, kalshi_get, BASE_URL

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"

CSV_COLUMNS = [
    "logged_at", "decision_time", "contract_ticker", "close_ts",
    "spot", "strike", "offset_pct", "p_market", "p_market_source",
    "p_yes_model", "z_score", "vol_60m", "vol_60m_model", "vol_implied_kalshi", "vol_ratio", "vol_eff",
    "structure_bias", "confirmation_bias", "confirmation_score",
    "ema_alignment", "rsi_value", "rsi_regime", "raw_edge", "net_edge",
    "decision", "side", "neutral_gate", "pure_edge_gate",
    "contracts_scanned", "tau_minutes", "gate_blocked",
    "kelly_fraction", "bet_fraction", "bet_amount", "bankroll",
    "resolved_yes", "would_win", "would_pnl",
]


def fetch_market(ticker: str, auth: KalshiAuth) -> dict:
    data = kalshi_get(f"/markets/{ticker}", {}, auth)
    return data.get("market") or {}


def is_settled(market: dict) -> bool:
    return market.get("status") in ("settled", "resolved", "finalized", "determined")


def parse_resolution(market: dict) -> bool:
    """
    Return True if the YES side won (i.e. BTC closed ABOVE the strike).
    Kalshi sets result to 'yes' or 'no'.
    """
    result = (market.get("result") or "").lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    # Fallback: check yes_sub_title for clues
    sub = (market.get("yes_sub_title") or "").lower()
    if "above" in sub or "higher" in sub:
        # YES condition met
        return True
    raise ValueError(f"Cannot determine resolution from market dict: result={result!r}")


def compute_pnl(row: dict, resolved_yes: bool) -> float:
    """
    Estimate paper P&L based on what would have happened.
    - YES trade: win +bet_amount * (1/p_market - 1), lose -bet_amount
    - NO trade : win +bet_amount * (1/(1-p_market) - 1), lose -bet_amount
    Does not deduct fees (already reflected in net_edge gate).
    """
    bet  = float(row["bet_amount"])
    pm   = float(row["p_market"])
    side = row["side"]

    if side == "yes":
        if resolved_yes:
            return round(bet * (1.0 / pm - 1.0), 2)
        else:
            return round(-bet, 2)
    else:  # NO trade
        if not resolved_yes:
            return round(bet * (1.0 / (1.0 - pm) - 1.0), 2)
        else:
            return round(-bet, 2)


def main() -> None:
    if not PAPER_TRADES_CSV.exists():
        print("No paper_trades.csv found. Run paper_trade_runner.py first.")
        return

    auth = load_auth()
    if auth is None:
        print("ERROR: KALSHI_KEY_ID / KALSHI_KEY_PATH not set.")
        sys.exit(1)

    # Read all rows
    with open(PAPER_TRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    skipped = 0

    for row in rows:
        ticker = row.get("contract_ticker", "").strip()

        # Skip rows without a ticker, already resolved, or not a trade
        if not ticker:
            skipped += 1
            continue
        if (row.get("resolved_yes") or "").strip():
            skipped += 1
            continue
        if row.get("decision", "").strip() != "trade":
            # Still log resolution for no_trade rows if we have a ticker
            # (useful for counterfactual analysis)
            pass

        # Check if contract close time has passed
        close_ts_str = row.get("close_ts", "").strip()
        if close_ts_str:
            try:
                close_dt = datetime.fromisoformat(close_ts_str.replace("Z", "+00:00"))
                if close_dt > datetime.now(timezone.utc):
                    skipped += 1
                    continue  # not expired yet
            except Exception:
                pass  # can't parse — try anyway

        market = fetch_market(ticker, auth)
        if not market:
            print(f"  {ticker}: could not fetch market data")
            skipped += 1
            continue

        if not is_settled(market):
            status = market.get("status", "unknown")
            print(f"  {ticker}: not yet settled (status={status})")
            skipped += 1
            continue

        try:
            resolved_yes = parse_resolution(market)
        except ValueError as e:
            print(f"  {ticker}: {e}")
            skipped += 1
            continue

        row["resolved_yes"] = str(resolved_yes)

        # Compute would_win only for trade rows with a valid bet_amount
        if row.get("decision", "").strip() == "trade" and (row.get("bet_amount") or "").strip():
            side = row.get("side", "yes")
            if side == "yes":
                would_win = resolved_yes
            else:
                would_win = not resolved_yes
            row["would_win"] = str(would_win)
            row["would_pnl"] = str(compute_pnl(row, resolved_yes))
        else:
            row["would_win"] = ""
            row["would_pnl"] = ""

        updated += 1
        print(f"  {ticker}: resolved_yes={resolved_yes}  would_win={row['would_win']}  "
              f"would_pnl={row['would_pnl']}")
        time.sleep(0.2)  # rate-limit courtesy

    # Write all rows back
    if updated > 0:
        with open(PAPER_TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Updated {updated} rows in {PAPER_TRADES_CSV}")
    else:
        print(f"\n  No rows updated (skipped={skipped}).")


if __name__ == "__main__":
    main()
