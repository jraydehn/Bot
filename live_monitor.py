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
    SERIES_TICKER, TAU, ASSET_CONFIG,
)
from paper_trade_runner import ensure_csv_exists, append_row, get_csv_path, DEFAULT_BANKROLL
from outcome_checker import main as resolve_outcomes


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


def run_once(auth, df_vol, df_confirm, df_struct, bankroll: float, sim: bool, asset: str = "BTC"):
    """
    Run the full signal pipeline once and return (contract_ticker, side, dec, row).
    Returns None for contract_ticker if no live contract was found.
    """
    now_utc = datetime.now(timezone.utc)
    cfg        = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    tau        = cfg.get("tau", TAU)
    ema_fast   = cfg.get("ema_fast", 20)
    ema_slow   = cfg.get("ema_slow", 50)
    rsi_period = cfg.get("rsi_period", 21)
    vol_bars   = cfg.get("vol_lookback_bars", 60)

    hist_confirm = df_confirm.iloc[-100:]
    hist_struct  = df_struct.iloc[-120:]
    hist_vol_fb  = df_vol.iloc[-200:]  # fallback if live 1m fetch fails

    struct  = detect_market_structure(hist_struct)

    # Fetch fresh 1m candles for realized vol.
    live_1m    = fetch_recent_1m_candles(lookback_bars=max(vol_bars * 2, 120), asset=asset)
    vol_src    = live_1m if live_1m is not None and len(live_1m) >= vol_bars else hist_vol_fb
    vol        = compute_realized_volatility(vol_src)
    confirm    = compute_confirmation(hist_confirm, ema_fast=ema_fast, ema_slow=ema_slow, rsi_period=rsi_period)
    gate_side  = "yes" if struct.structure_bias == 1 else "no"

    # Live spot
    live_spot = fetch_live_spot(asset=asset)
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
        ladder = fetch_contracts_for_nearest_expiry(auth, spot, asset=asset)
        contracts_scanned = len(ladder)

        for c in ladder:
            s_k       = c["floor_strike"]
            pm        = c["p_market"]
            tau_c     = tte_minutes(c["close_time"])
            vol_imp_c = implied_vol_from_price(pm, spot, s_k, tau_c)
            vol_eff_c = blend_vol(vol.vol_60m, vol_imp_c)
            prob_c    = estimate_probability(spot, s_k, tau_c, vol_eff_c)
            meta_c    = {"strike": s_k, "p_market": pm, "prob": prob_c,
                         "contract_ticker": c["ticker"], "close_ts": c["close_time"],
                         "vol_eff": vol_eff_c}

            # Evaluate both YES and NO for every contract — best qualifying trade wins
            for force_side in ["yes", "no"]:
                dec_c = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                       prob_c.p_yes, pm, bankroll,
                                       confirmation_score=confirm.confirmation_score,
                                       no_score=confirm.no_score, no_bias=confirm.no_bias,
                                       force_side=force_side)

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
        prob_c   = estimate_probability(spot, strike, tau, vol.vol_60m)
        p_market_sim = simulate_p_market(eff, side=gate_side)
        dec      = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                                  prob_c.p_yes, p_market_sim, bankroll,
                                  confirmation_score=confirm.confirmation_score,
                                  no_score=confirm.no_score, no_bias=confirm.no_bias)
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

    return ticker, close_ts, dec, prob, p_market, p_market_source, struct, confirm, row, contracts_scanned


def init_asset_state(asset: str, bankroll: float, session_start: str) -> dict:
    """Load OHLCV data and initialize per-asset tracking state."""
    cfg        = ASSET_CONFIG.get(asset, ASSET_CONFIG["BTC"])
    confirm_iv = cfg.get("confirmation_interval", "1h")
    struct_iv  = cfg.get("structure_interval", "4h")

    print(f"  Loading OHLCV data ({asset})...")
    df_vol, df_confirm, df_struct = load_data(asset=asset)
    df_vol     = extend_with_live_candles(df_vol,     "1m",       200, asset=asset)
    df_confirm = extend_with_live_candles(df_confirm, confirm_iv, 200, asset=asset)
    df_struct  = extend_with_live_candles(df_struct,  struct_iv,  120, asset=asset)
    csv_path   = get_csv_path(asset)
    ensure_csv_exists(csv_path)
    return {
        "asset":             asset,
        "df_vol":            df_vol,
        "df_confirm":        df_confirm,
        "df_struct":         df_struct,
        "confirm_iv":        confirm_iv,
        "struct_iv":         struct_iv,
        "csv_path":          csv_path,
        "last_data_refresh": datetime.now(timezone.utc),
        "logged":            set(),
        "expiry_trades":     {},
        "session_start":     session_start,
        "soft_drawdown":     bankroll * 0.20,
        "hard_drawdown":     bankroll * 0.35,
    }


def run_asset(state: dict, auth, bankroll: float, sim: bool, high_edge_min: float) -> None:
    """Run one evaluation cycle for a single asset using its state dict."""
    asset    = state["asset"]
    csv_path = state["csv_path"]
    now_utc  = datetime.now(timezone.utc)

    print(f"\n  [{asset}] {now_utc.strftime('%H:%M:%S UTC')}")

    # Refresh OHLCV every 10 minutes
    if (now_utc - state["last_data_refresh"]).total_seconds() >= 600:
        try:
            state["df_vol"]     = extend_with_live_candles(state["df_vol"],     "1m",                200, asset=asset)
            state["df_confirm"] = extend_with_live_candles(state["df_confirm"], state["confirm_iv"], 200, asset=asset)
            state["df_struct"]  = extend_with_live_candles(state["df_struct"],  state["struct_iv"],  120, asset=asset)
            state["last_data_refresh"] = now_utc
        except Exception as exc:
            print(f"  [{asset}] WARNING: OHLCV refresh failed ({exc}), using existing data.")

    # Resolve expired trades
    if auth is not None and not sim:
        try:
            resolve_outcomes(csv_path)
        except Exception as exc:
            print(f"  [{asset}] (outcome check skipped: {exc})")

    try:
        ticker, close_ts, dec, prob, p_market, p_mkt_src, struct, confirm, row, n_scanned = run_once(
            auth, state["df_vol"], state["df_confirm"], state["df_struct"], bankroll, sim, asset=asset
        )
    except Exception as exc:
        print(f"  [{asset}] ERROR: {exc}")
        return

    tte         = minutes_to_expiry(close_ts)
    tte_str     = f"{tte:.1f} min to expiry" if tte < 999 else "no contract"
    expiry_window = ticker.split("-")[1] if ticker and ticker.count("-") >= 2 else ""
    gate_blocked  = next((r.split(":")[0] for r in dec.reasons if "FAILED" in r), "")

    print(f"  [{asset}] Scanned  : {n_scanned} contracts")
    print(f"  [{asset}] Contract : {ticker or '—'}  ({tte_str})")
    print(f"  [{asset}] Spot     : ${row['spot']:,.2f}  Strike: ${row['strike']:,.2f}")
    print(f"  [{asset}] p_yes    : {prob.p_yes:.4f}  p_market: {p_market:.4f} ({p_mkt_src})")
    print(f"  [{asset}] Signals  : structure={struct.structure_bias:+d}  "
          f"yes_score={confirm.confirmation_score:+d}  no_score={confirm.no_score:+d}  "
          f"EMA={confirm.ema_alignment}  RSI={confirm.rsi_value:.1f}({confirm.rsi_regime})")
    print(f"  [{asset}] Decision : {dec.decision.upper()}  side={dec.side.upper()}  "
          f"net_edge={dec.net_edge:+.4f}  bet=${dec.bet_amount:,.2f}"
          + (f"  [{gate_blocked}]" if gate_blocked else ""))

    log_key    = f"{ticker}_{dec.side}"
    new_strike = row["strike"]
    prior_trades = state["expiry_trades"].get(expiry_window, [])
    conflict_trade = None
    for pt in prior_trades:
        if dec.side == "no" and pt["side"] == "yes" and new_strike <= pt["strike"]:
            conflict_trade = pt
            break
        if dec.side == "yes" and pt["side"] == "no" and new_strike >= pt["strike"]:
            conflict_trade = pt
            break

    try:
        csv_df = pd.read_csv(csv_path)
        session_rows = csv_df[
            (csv_df["decision"] == "trade") &
            (csv_df["logged_at"] >= state["session_start"])
        ]
        resolved_pnl = session_rows["would_pnl"].dropna().sum()
        pending_risk  = session_rows[session_rows["would_pnl"].isna()]["bet_amount"].sum()
        session_pnl   = resolved_pnl - pending_risk
    except Exception:
        session_pnl = 0.0

    hard_halted = session_pnl <= -state["hard_drawdown"]
    soft_halted = session_pnl <= -state["soft_drawdown"] and dec.net_edge < high_edge_min

    if dec.decision == "trade" and hard_halted:
        print(f"  [{asset}] *** HARD HALT: session drawdown ${session_pnl:+,.2f} exceeds "
              f"-${state['hard_drawdown']:,.0f} limit (35%). All trading stopped. ***")
    elif dec.decision == "trade" and soft_halted:
        print(f"  [{asset}] *** SOFT HALT: session drawdown ${session_pnl:+,.2f} exceeds "
              f"-${state['soft_drawdown']:,.0f} (20%). Edge {dec.net_edge:+.4f} below {high_edge_min:.0%} threshold. ***")
    elif dec.decision == "trade" and not ticker:
        print(f"  [{asset}] (no contract ticker — API timeout, trade not logged)")
    elif dec.decision == "trade" and log_key not in state["logged"] and conflict_trade is None:
        append_row(row, csv_path)
        state["logged"].add(log_key)
        state["expiry_trades"].setdefault(expiry_window, []).append(
            {"side": dec.side, "strike": new_strike}
        )
        gate_tag = " [Gate P]" if row["pure_edge_gate"] else (" [Neutral]" if row["neutral_gate"] else "")
        print(f"  [{asset}] *** LOGGED{gate_tag} ***  (session P&L: ${session_pnl:+,.2f})")
    elif dec.decision == "trade" and conflict_trade is not None:
        print(f"  [{asset}] (contradictory: {dec.side.upper()} ${new_strike:,.2f} conflicts with "
              f"logged {conflict_trade['side'].upper()} ${conflict_trade['strike']:,.2f})")
    elif dec.decision == "trade":
        print(f"  [{asset}] (already logged this contract+side)")
    else:
        print(f"  [{asset}] No trade.")

    # Clean up stale keys
    state["logged"] = {k for k in state["logged"] if not k.startswith(ticker + "_") or tte > 0}
    if expiry_window and tte <= 0:
        state["expiry_trades"].pop(expiry_window, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live paper-trade monitor")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Polling interval in minutes (default: 2)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market (no auth needed)")
    parser.add_argument("--asset", type=str, default="BTC",
                        help="Asset to trade: BTC, ETH, SOL, or ALL (default: BTC)")
    args = parser.parse_args()
    args.asset = args.asset.upper()

    interval_sec  = args.interval * 60
    high_edge_min = 0.08

    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            print("  WARNING: No Kalshi credentials — using simulated p_market.")

    assets = ["BTC", "ETH", "SOL"] if args.asset == "ALL" else [args.asset]
    session_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    asset_states = {}
    for a in assets:
        try:
            asset_states[a] = init_asset_state(a, args.bankroll, session_start)
        except Exception as exc:
            print(f"  WARNING: Could not initialize {a} ({exc}) — skipping.")

    print(f"\n  Monitoring: {', '.join(asset_states.keys())}")
    print(f"  Polling every {args.interval:.0f} min  |  bankroll=${args.bankroll:,.0f}  "
          f"|  soft halt=20% (edge>={high_edge_min:.0%})  |  hard halt=35%")
    print("  Press Ctrl+C to stop.\n")

    while True:
        print(f"{'='*60}")
        for a, state in asset_states.items():
            run_asset(state, auth, args.bankroll, args.sim, high_edge_min)
        print()
        time.sleep(interval_sec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
