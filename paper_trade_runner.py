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
from probability_engine import estimate_probability
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from decision import evaluate_trade
from kelly_sizing import compute_kelly_size
from live_signal import (
    load_auth, kalshi_get, fetch_live_spot, fetch_current_price, find_live_contract,
    BASE_URL, SERIES_TICKER, CANDLE_WINDOW, TAU,
)

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"
DEFAULT_BANKROLL  = 10_000.0

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
    "structure_bias",
    "confirmation_bias",
    "ema_alignment",
    "rsi_value",
    "rsi_regime",
    "raw_edge",
    "net_edge",
    "decision",
    "side",
    "kelly_fraction",
    "bet_fraction",
    "bet_amount",
    "bankroll",
    "resolved_yes",   # filled by outcome_checker.py
    "would_win",      # filled by outcome_checker.py
    "would_pnl",      # filled by outcome_checker.py
]


def ensure_csv_exists() -> None:
    PAPER_TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not PAPER_TRADES_CSV.exists():
        with open(PAPER_TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"  Created {PAPER_TRADES_CSV}")


def append_row(row: dict) -> None:
    with open(PAPER_TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    print(f"  Logged → {PAPER_TRADES_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market (no auth needed)")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"\n  Run time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auth
    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            print("  WARNING: No Kalshi credentials — using simulated p_market.")

    # Load OHLCV
    print("  Loading OHLCV data...")
    df_1m, df_1h, df_4h = load_data()

    ts = df_1m.index[-1]

    # Live spot
    live_spot = fetch_live_spot()
    spot = live_spot if live_spot is not None else float(df_1m["close"].iloc[-1])

    # Signals
    hist_1m = df_1m.iloc[-200:]
    hist_1h = df_1h.iloc[-100:]
    hist_4h = df_4h.iloc[-120:]

    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_4h)
    confirm = compute_confirmation(hist_1h)
    gate_side = "yes" if struct.structure_bias == 1 else "no"

    # Contract & p_market
    contract       = None
    p_market       = None
    p_market_source = "simulated"
    contract_ticker = ""
    close_ts        = ""
    strike          = spot * 1.005  # fallback

    if auth is not None:
        contract = find_live_contract(auth, spot)
        if contract is not None:
            strike          = float(contract.get("floor_strike", strike))
            contract_ticker = contract.get("ticker", "")
            close_ts        = contract.get("close_time", "")
            live_price      = fetch_current_price(auth, contract_ticker)
            if live_price is not None:
                p_market        = live_price
                p_market_source = "real"

    effective_offset = strike / spot - 1
    prob = estimate_probability(spot, strike, TAU, vol.vol_60m)

    if p_market is None:
        p_market = simulate_p_market(effective_offset, side=gate_side)

    # Decision
    dec = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                         prob.p_yes, p_market, args.bankroll)
    pricing = evaluate_edge(prob.p_yes, p_market)

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
        "p_yes_model":        round(prob.p_yes, 6),
        "z_score":            round(prob.z_score, 4),
        "vol_60m":            round(vol.vol_60m, 8),
        "structure_bias":     struct.structure_bias,
        "confirmation_bias":  confirm.confirmation_bias,
        "ema_alignment":      confirm.ema_alignment,
        "rsi_value":          round(confirm.rsi_value, 2),
        "rsi_regime":         confirm.rsi_regime,
        "raw_edge":           round(dec.raw_edge, 6),
        "net_edge":           round(dec.net_edge, 6),
        "decision":           dec.decision,
        "side":               dec.side,
        "kelly_fraction":     round(dec.kelly_fraction, 6),
        "bet_fraction":       round(dec.bet_fraction, 6),
        "bet_amount":         round(dec.bet_amount, 2),
        "bankroll":           round(args.bankroll, 2),
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

    ensure_csv_exists()
    append_row(row)


if __name__ == "__main__":
    main()
