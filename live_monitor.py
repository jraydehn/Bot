"""
Live monitor: runs the paper-trade signal pipeline on a short polling interval,
logging new opportunities as contracts approach expiration.

OHLCV data (vol, structure, confirmation) is loaded once at startup — these
signals change slowly (hourly/4h). Each iteration refreshes only the fast inputs:
BRTI spot price and Kalshi market price.

Each contract+side pair is only logged once. If a new contract opens or the
signal flips side, that is treated as a new opportunity.

Usage:
    python3 live_monitor.py                  # 2-minute interval, $10k bankroll
    python3 live_monitor.py --interval 1     # check every 60 seconds
    python3 live_monitor.py --bankroll 5000
    python3 live_monitor.py --sim            # simulated p_market (no auth needed)
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from evaluate_point import load_data
from market_data import compute_realized_volatility
from probability_engine import estimate_probability, implied_vol_from_price, blend_vol
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from decision import evaluate_trade
from kelly_sizing import compute_kelly_size
from live_signal import (
    load_auth, fetch_live_spot, fetch_current_price, find_live_contract,
    fetch_contracts_for_nearest_expiry, fetch_recent_1m_candles,
    extend_with_live_candles,
    minutes_to_expiry as tte_minutes,
    SERIES_TICKER, TAU,
)
from paper_trade_runner import ensure_csv_exists, append_row, DEFAULT_BANKROLL


def minutes_to_expiry(close_ts: str) -> float:
    """Return minutes until the contract expires (negative if already expired)."""
    if not close_ts:
        return float("inf")
    try:
        expiry = pd.Timestamp(close_ts).tz_convert("UTC")
        now    = pd.Timestamp.now(tz="UTC")
        return (expiry - now).total_seconds() / 60
    except Exception:
        return float("inf")


def run_once(auth, df_1m, df_1h, df_4h, bankroll: float, sim: bool):
    """
    Run the full signal pipeline once and return (contract_ticker, side, dec, row).
    Returns None for contract_ticker if no live contract was found.
    """
    now_utc = datetime.now(timezone.utc)

    # Signals — computed from cached OHLCV (reload unnecessary at 2-min cadence)
    hist_1m = df_1m.iloc[-200:]
    hist_1h = df_1h.iloc[-100:]
    hist_4h = df_4h.iloc[-120:]

    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_4h)

    # Fetch fresh 1m candles for real-time momentum scores (chg_15m, chg_60m).
    # Falls back to stale parquet hist_1m if Binance is unavailable.
    live_1m = fetch_recent_1m_candles(lookback_bars=70)
    confirm = compute_confirmation(hist_1h, hist_1m=live_1m if live_1m is not None else hist_1m)
    gate_side = "yes" if struct.structure_bias == 1 else "no"

    # Live spot (BRTI proxy)
    live_spot = fetch_live_spot()
    spot = live_spot if live_spot is not None else float(df_1m["close"].iloc[-1])

    # Scan all contracts for nearest expiry; select highest net_edge trade.
    contracts_scanned = 0
    p_market_source   = "simulated"
    strike            = spot * 1.005

    best_trade_dec  = None
    best_trade_meta = {}
    best_any_dec    = None
    best_any_meta   = {}

    if auth is not None and not sim:
        ladder = fetch_contracts_for_nearest_expiry(auth, spot)
        contracts_scanned = len(ladder)

        for c in ladder:
            s_k       = c["floor_strike"]
            pm        = c["p_market"]
            tau_c     = tte_minutes(c["close_time"])
            vol_imp_c = implied_vol_from_price(pm, spot, s_k, tau_c)
            vol_eff_c = blend_vol(vol.vol_60m, vol_imp_c)
            prob_c    = estimate_probability(spot, s_k, tau_c, vol_eff_c)
            dec_c     = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                       prob_c.p_yes, pm, bankroll,
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

    if best_trade_dec is not None:
        dec, chosen, p_market_source = best_trade_dec, best_trade_meta, "real"
    elif best_any_dec is not None:
        dec, chosen, p_market_source = best_any_dec, best_any_meta, "real"
    else:
        eff = strike / spot - 1
        prob_c   = estimate_probability(spot, strike, TAU, vol.vol_60m)
        p_market_sim = simulate_p_market(eff, side=gate_side)
        dec      = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                  prob_c.p_yes, p_market_sim, bankroll,
                                  confirmation_score=confirm.confirmation_score, no_score=confirm.no_score)
        chosen   = {"strike": strike, "p_market": p_market_sim, "prob": prob_c,
                    "contract_ticker": "", "close_ts": "", "vol_eff": vol.vol_60m}

    strike          = chosen["strike"]
    p_market        = chosen["p_market"]
    prob            = chosen["prob"]

    ticker       = chosen["contract_ticker"]
    close_ts     = chosen["close_ts"]
    eff_offset   = chosen["strike"] / spot - 1

    vol_eff   = chosen.get("vol_eff", vol.vol_60m)
    vol_impl  = implied_vol_from_price(p_market, spot, strike, tte_minutes(close_ts))
    vol_ratio = round(vol.vol_60m / vol_impl, 4) if vol_impl > 0 else ""

    row = {
        "logged_at":          now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_time":      df_1m.index[-1].strftime("%Y-%m-%d %H:%M"),
        "contract_ticker":    ticker,
        "close_ts":           close_ts,
        "spot":               round(spot, 2),
        "strike":             round(strike, 2),
        "offset_pct":         round(eff_offset * 100, 4),
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
        "tau_minutes":        round(tte_minutes(close_ts), 2),
        "gate_blocked":       next((r.split(":")[0] for r in dec.reasons if "FAILED" in r), "") if dec.decision == "no_trade" else "",
        "kelly_fraction":     round(dec.kelly_fraction, 6),
        "bet_fraction":       round(dec.bet_fraction, 6),
        "bet_amount":         round(dec.bet_amount, 2),
        "bankroll":           round(bankroll, 2),
        "resolved_yes":       "",
        "would_win":          "",
        "would_pnl":          "",
    }

    return ticker, close_ts, dec, prob, p_market, p_market_source, struct, row, contracts_scanned


def main() -> None:
    parser = argparse.ArgumentParser(description="Live paper-trade monitor")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Polling interval in minutes (default: 2)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market (no auth needed)")
    args = parser.parse_args()

    interval_sec = args.interval * 60

    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            print("  WARNING: No Kalshi credentials — using simulated p_market.")

    print("\n  Loading OHLCV data...")
    df_1m, df_1h, df_4h = load_data()
    df_1m = extend_with_live_candles(df_1m, "1m", 200)
    df_1h = extend_with_live_candles(df_1h, "1h", 120)
    df_4h = extend_with_live_candles(df_4h, "4h",  60)
    last_data_refresh = datetime.now(timezone.utc)
    soft_drawdown = args.bankroll * 0.20   # soft halt: only high-edge trades allowed
    hard_drawdown = args.bankroll * 0.35   # hard halt: all trading stopped
    high_edge_min = 0.08                   # minimum net_edge to trade through soft halt
    print(f"  Polling every {args.interval:.0f} min  |  bankroll=${args.bankroll:,.0f}  "
          f"|  soft halt=${soft_drawdown:,.0f} (edge>={high_edge_min:.0%})  |  hard halt=${hard_drawdown:,.0f}")
    print("  Press Ctrl+C to stop.\n")

    ensure_csv_exists()

    # Track which contract+side pairs have already been logged this session
    logged: set = set()
    # Track logged trades per expiry window: {expiry: [{side, strike}]}
    expiry_trades: dict = {}
    # Track session P&L for drawdown limit
    session_pnl: float = 0.0

    while True:
        now_utc = datetime.now(timezone.utc)
        print(f"{'='*60}")
        print(f"  {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Refresh OHLCV data every hour so confirmation signals stay current
        if (now_utc - last_data_refresh).total_seconds() >= 3600:
            print("  Refreshing OHLCV data...")
            try:
                df_1m, df_1h, df_4h = load_data()
                df_1m = extend_with_live_candles(df_1m, "1m", 200)
                df_1h = extend_with_live_candles(df_1h, "1h", 120)
                df_4h = extend_with_live_candles(df_4h, "4h",  60)
                last_data_refresh = now_utc
                print("  OHLCV refreshed.")
            except Exception as exc:
                print(f"  WARNING: OHLCV refresh failed ({exc}), using existing data.")

        try:
            ticker, close_ts, dec, prob, p_market, p_mkt_src, struct, row, n_scanned = run_once(
                auth, df_1m, df_1h, df_4h, args.bankroll, args.sim
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            time.sleep(interval_sec)
            continue

        tte = minutes_to_expiry(close_ts)
        tte_str = f"{tte:.1f} min to expiry" if tte < 999 else "no contract"

        # Extract expiry window from ticker (e.g. "KXBTCD-26MAR2302-T68899.99" -> "26MAR2302")
        expiry_window = ticker.split("-")[1] if ticker and ticker.count("-") >= 2 else ""

        print(f"  Scanned  : {n_scanned} contracts")
        print(f"  Contract : {ticker or '—'}  ({tte_str})")
        print(f"  Spot     : ${row['spot']:,.2f}  Strike: ${row['strike']:,.2f}")
        print(f"  p_yes    : {prob.p_yes:.4f}  p_market: {p_market:.4f} ({p_mkt_src})")
        print(f"  Decision : {dec.decision.upper()}  side={dec.side.upper()}  "
              f"net_edge={dec.net_edge:+.4f}  bet=${dec.bet_amount:,.2f}")

        # Check for contradictory positions within the same expiry window.
        # YES on strike A + NO on strike B is contradictory if B <= A
        # (both can't win — YES needs BTC > A, NO needs BTC < B).
        # YES on A + NO on B where B > A is fine — both win if A < BTC < B.
        log_key = f"{ticker}_{dec.side}"
        new_strike = row["strike"]
        prior_trades = expiry_trades.get(expiry_window, [])
        conflict_trade = None
        for pt in prior_trades:
            if dec.side == "no" and pt["side"] == "yes" and new_strike <= pt["strike"]:
                conflict_trade = pt
                break
            if dec.side == "yes" and pt["side"] == "no" and new_strike >= pt["strike"]:
                conflict_trade = pt
                break

        # Two-tier drawdown check before logging any new trade
        hard_halted = session_pnl <= -hard_drawdown
        soft_halted = session_pnl <= -soft_drawdown and dec.net_edge < high_edge_min

        if dec.decision == "trade" and hard_halted:
            print(f"  *** HARD HALT: session drawdown ${session_pnl:+,.2f} exceeds "
                  f"-${hard_drawdown:,.0f} limit (35%). All trading stopped. ***")
        elif dec.decision == "trade" and soft_halted:
            print(f"  *** SOFT HALT: session drawdown ${session_pnl:+,.2f} exceeds "
                  f"-${soft_drawdown:,.0f} (20%). Edge {dec.net_edge:+.4f} below {high_edge_min:.0%} threshold. ***")
        elif dec.decision == "trade" and log_key not in logged and conflict_trade is None:
            append_row(row)
            logged.add(log_key)
            expiry_trades.setdefault(expiry_window, []).append(
                {"side": dec.side, "strike": new_strike}
            )
            gate_tag = " [Gate P]" if row["pure_edge_gate"] else (" [Neutral]" if row["neutral_gate"] else "")
            # Deduct bet amount from session P&L (will be updated when resolved)
            session_pnl -= row["bet_amount"]
            print(f"  *** LOGGED{gate_tag} ***  (session P&L: ${session_pnl:+,.2f})")
        elif dec.decision == "trade" and conflict_trade is not None:
            print(f"  (contradictory: {dec.side.upper()} ${new_strike:,.2f} conflicts with "
                  f"logged {conflict_trade['side'].upper()} ${conflict_trade['strike']:,.2f})")
        elif dec.decision == "trade":
            print(f"  (already logged this contract+side)")
        else:
            print(f"  No trade.")

        # Clean up stale keys and expiry trades for expired contracts
        logged = {k for k in logged if not k.startswith(ticker + "_") or tte > 0}
        if expiry_window and tte <= 0:
            expiry_trades.pop(expiry_window, None)

        print()
        time.sleep(interval_sec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
