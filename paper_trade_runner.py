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
from probability_engine import estimate_probability, implied_vol_from_price, blend_vol, REALIZED_VOL_WEIGHT
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from decision import evaluate_trade
from kelly_sizing import compute_kelly_size
from live_signal import (
    load_auth, kalshi_get, fetch_live_spot, fetch_current_price, find_live_contract,
    fetch_contracts_for_nearest_expiry, fetch_recent_1m_candles, minutes_to_expiry,
    BASE_URL, SERIES_TICKER, CANDLE_WINDOW, TAU,
)

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"


def get_csv_path(asset: str = "BTC") -> Path:
    """Return the asset-specific paper trades CSV path."""
    asset = asset.upper()
    if asset == "BTC":
        return PAPER_TRADES_CSV  # keep existing BTC file unchanged
    return Path(__file__).parent / "results" / f"paper_trades_{asset.lower()}.csv"
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
    "vol_60m_model",
    "vol_implied_kalshi",
    "vol_ratio",
    "vol_eff",
    "structure_bias",
    "confirmation_bias",
    "confirmation_score",
    "ema_alignment",
    "rsi_value",
    "rsi_regime",
    "raw_edge",
    "net_edge",
    "decision",
    "side",
    "neutral_gate",    # True if trade passed via neutral structure path (+0.02 edge premium)
    "pure_edge_gate",  # True if trade passed via pure-edge override (Gate P, 1/8 Kelly)
    "contracts_scanned",  # number of contracts with real bid/ask evaluated at this decision point
    "tau_minutes",        # minutes to expiry at decision time (used in probability engine)
    "gate_blocked",       # which gate blocked a no_trade (Gate 1/2/3); empty for trades
    "kelly_fraction",
    "bet_fraction",
    "bet_amount",
    "bankroll",
    "resolved_yes",   # filled by outcome_checker.py
    "would_win",      # filled by outcome_checker.py
    "would_pnl",      # filled by outcome_checker.py
]


def ensure_csv_exists(csv_path: Path = None) -> None:
    path = csv_path or PAPER_TRADES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"  Created {path}")


def append_row(row: dict, csv_path: Path = None) -> None:
    path = csv_path or PAPER_TRADES_CSV
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    print(f"  Logged → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market (no auth needed)")
    parser.add_argument("--asset", type=str, default="BTC",
                        help="Asset to trade: BTC, ETH, or SOL (default: BTC)")
    args = parser.parse_args()
    args.asset = args.asset.upper()

    now_utc = datetime.now(timezone.utc)
    print(f"\n  Run time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auth
    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            print("  WARNING: No Kalshi credentials — using simulated p_market.")

    # Load OHLCV
    print(f"  Loading OHLCV data ({args.asset})...")
    df_1m, df_1h, df_4h = load_data(asset=args.asset)

    ts = df_1m.index[-1]

    # Live spot
    live_spot = fetch_live_spot(asset=args.asset)
    spot = live_spot if live_spot is not None else float(df_1m["close"].iloc[-1])

    # Signals
    hist_1m = df_1m.iloc[-200:]
    hist_1h = df_1h.iloc[-100:]
    hist_4h = df_4h.iloc[-120:]

    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_4h)

    # Fetch fresh 1m candles for real-time momentum scores.
    live_1m = fetch_recent_1m_candles(lookback_bars=70, asset=args.asset)
    confirm = compute_confirmation(hist_1h, hist_1m=live_1m if live_1m is not None else hist_1m)
    gate_side = "yes" if struct.structure_bias == 1 else "no"

    # Scan all contracts for nearest expiry; select highest net_edge trade.
    # Falls back to simulated p_market on nearest OTM contract when no auth.
    contracts_scanned = 0
    p_market_source   = "simulated"
    contract_ticker   = ""
    close_ts          = ""
    strike            = spot * 1.005   # fallback

    best_trade_dec    = None           # best DecisionResult with decision=="trade"
    best_trade_meta   = {}             # {strike, p_market, prob, contract_ticker, close_ts}
    best_any_dec      = None           # best DecisionResult across all contracts (for no_trade log)
    best_any_meta     = {}

    if auth is not None:
        ladder = fetch_contracts_for_nearest_expiry(auth, spot, asset=args.asset)
        contracts_scanned = len(ladder)
        print(f"  [scan] {contracts_scanned} liquid contracts in nearest expiry")

        for c in ladder:
            s_k       = c["floor_strike"]
            pm        = c["p_market"]
            tau_c     = minutes_to_expiry(c["close_time"])
            vol_imp_c = implied_vol_from_price(pm, spot, s_k, tau_c)
            vol_eff_c = blend_vol(vol.vol_60m, vol_imp_c)
            prob_c    = estimate_probability(spot, s_k, tau_c, vol_eff_c)
            dec_c     = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                       prob_c.p_yes, pm, args.bankroll,
                                       confirmation_score=confirm.confirmation_score, no_score=confirm.no_score)
            meta_c    = {"strike": s_k, "p_market": pm, "prob": prob_c,
                         "contract_ticker": c["ticker"], "close_ts": c["close_time"],
                         "vol_eff": vol_eff_c}

            if best_any_dec is None or dec_c.net_edge > best_any_dec.net_edge:
                best_any_dec  = dec_c
                best_any_meta = meta_c

            if dec_c.decision == "trade":
                if best_trade_dec is None or dec_c.net_edge > best_trade_dec.net_edge:
                    best_trade_dec  = dec_c
                    best_trade_meta = meta_c

    # Select final decision
    if best_trade_dec is not None:
        dec              = best_trade_dec
        chosen           = best_trade_meta
        p_market_source  = "real"
        print(f"  [scan] Best trade: {chosen['contract_ticker']}  "
              f"strike=${chosen['strike']:,.2f}  net_edge={dec.net_edge:+.4f}  side={dec.side.upper()}")
    elif best_any_dec is not None:
        dec              = best_any_dec
        chosen           = best_any_meta
        p_market_source  = "real"
        print(f"  [scan] No trade passes gates. Best seen: {chosen['contract_ticker']}  "
              f"net_edge={dec.net_edge:+.4f}")
    else:
        # No auth or empty ladder — simulate
        effective_offset = strike / spot - 1
        prob_c           = estimate_probability(spot, strike, TAU, vol.vol_60m)
        p_market_sim     = simulate_p_market(effective_offset, side=gate_side)
        dec              = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                          prob_c.p_yes, p_market_sim, args.bankroll,
                                          confirmation_score=confirm.confirmation_score, no_score=confirm.no_score)
        chosen           = {"strike": strike, "p_market": p_market_sim,
                            "prob": prob_c, "contract_ticker": "", "close_ts": "",
                            "vol_eff": vol.vol_60m}

    strike          = chosen["strike"]
    p_market        = chosen["p_market"]
    prob            = chosen["prob"]
    contract_ticker = chosen["contract_ticker"]
    close_ts        = chosen["close_ts"]
    effective_offset = strike / spot - 1
    pricing = evaluate_edge(prob.p_yes, p_market)

    vol_eff  = chosen.get("vol_eff", vol.vol_60m)
    vol_impl = implied_vol_from_price(p_market, spot, strike, minutes_to_expiry(close_ts))
    vol_ratio = round(vol.vol_60m / vol_impl, 4) if vol_impl > 0 else ""

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
        "vol_60m_model":      round(vol.vol_60m, 8),
        "vol_implied_kalshi": round(vol_impl, 8) if vol_impl == vol_impl else "",
        "vol_ratio":          vol_ratio,
        "vol_eff":            round(vol_eff, 8),
        "structure_bias":     struct.structure_bias,
        "confirmation_bias":  confirm.confirmation_bias,
        "confirmation_score": confirm.confirmation_score,
        "ema_alignment":      confirm.ema_alignment,
        "rsi_value":          round(confirm.rsi_value, 2),
        "rsi_regime":         confirm.rsi_regime,
        "raw_edge":           round(dec.raw_edge, 6),
        "net_edge":           round(dec.net_edge, 6),
        "decision":           dec.decision,
        "side":               dec.side,
        "neutral_gate":       struct.structure_bias == 0 and any("Gate 1 PASSED (neutral)" in r for r in dec.reasons),
        "pure_edge_gate":     any("Gate P PASSED" in r for r in dec.reasons),
        "contracts_scanned":  contracts_scanned,
        "tau_minutes":        round(minutes_to_expiry(close_ts), 2),
        "gate_blocked":       next((r.split(":")[0] for r in dec.reasons if "FAILED" in r), "") if dec.decision == "no_trade" else "",
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

    csv_path = get_csv_path(args.asset)
    ensure_csv_exists(csv_path)
    append_row(row, csv_path)


if __name__ == "__main__":
    main()
