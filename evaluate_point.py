"""
Single-point model evaluation.

Given a timestamp and a strike, runs every module against the data available
at that moment, then checks the actual BTC price 60 minutes later to see
whether the model's prediction was correct.

Usage:
    python evaluate_point.py --time "2025-06-15 14:00"
    python evaluate_point.py --time "2025-06-15 14:00" --strike 67500
    python evaluate_point.py --time "2025-06-15 14:00" --offset 0.005   # strike = spot * 1.005
    python evaluate_point.py --time "2025-06-15 14:00" --p-market 0.48  # custom market price

Strike K is always set above spot regardless of trade direction: K = spot * (1 + abs(offset)).
Trade side (YES/NO) is determined by the gate system from structure_bias and confirmation_bias.

Data is loaded from kalshi_btc/data/. Run fetch_data.py first to populate it.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path so modules resolve correctly when called from any directory
sys.path.insert(0, str(Path(__file__).parent))

from market_data import compute_realized_volatility
from probability_engine import estimate_probability
from pricing_comparison import evaluate_edge, simulate_p_market, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from kelly_sizing import compute_kelly_size
from decision import evaluate_trade


DATA_DIR = Path(__file__).parent / "data"
TAU = 60          # 1-hour expiry in minutes
BANKROLL = 10_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_parquet(interval: str, symbol: str = "BTCUSDT") -> Path:
    """
    Find the most recently modified Parquet file for the given interval and symbol.
    Raises FileNotFoundError if none exist — run fetch_data.py first.
    """
    matches = sorted(DATA_DIR.glob(f"*{symbol}*_{interval}_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No cached {interval} data found for {symbol} in {DATA_DIR}.\n"
            f"Run:  python fetch_data.py --symbol {symbol}  to download data first."
        )
    return matches[-1]  # most recently written file


def load_data(asset: str = "BTC") -> tuple:
    """
    Load OHLCV DataFrames from the local Parquet cache for the given asset.

    Returns (df_vol, df_confirm, df_struct) where:
      df_vol     : 1m candles (realized vol fallback)
      df_confirm : confirmation-interval candles (1h for hourly, 15m for BTC15)
      df_struct  : structure-interval candles (1h for hourly, 1h for BTC15)
    """
    from live_signal import ASSET_CONFIG
    cfg    = ASSET_CONFIG.get(asset.upper(), ASSET_CONFIG["BTC"])
    symbol = cfg["binance_symbol"]
    confirm_iv = cfg.get("confirmation_interval", "1h")
    struct_iv  = cfg.get("structure_interval", "4h")

    intervals = list(dict.fromkeys(["1m", confirm_iv, struct_iv]))  # deduplicated, ordered
    paths = {iv: _find_parquet(iv, symbol) for iv in intervals}
    dfs = {}
    for iv, path in paths.items():
        # Retry up to 3 times with a short delay to handle transient write collisions
        # where the data updater is mid-write when the runner tries to read.
        _last_exc = None
        for _attempt in range(3):
            try:
                dfs[iv] = pd.read_parquet(path)
                _last_exc = None
                break
            except Exception as _exc:
                _last_exc = _exc
                import time as _time
                print(f"  {iv:3s}: read error (attempt {_attempt+1}/3) — {_exc}. Retrying in 2s...")
                _time.sleep(2)
        if _last_exc is not None:
            raise _last_exc
        if dfs[iv].index.tz is None:
            dfs[iv].index = dfs[iv].index.tz_localize("UTC")
        print(f"  {iv:3s}: {path.name}  ({len(dfs[iv]):,} rows)")

    return dfs["1m"], dfs[confirm_iv], dfs[struct_iv]


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_point(
    ts: pd.Timestamp,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    strike: Optional[float] = None,
    strike_offset: float = 0.0,
    p_market: Optional[float] = None,
    bankroll: float = BANKROLL,
    tau: int = TAU,
) -> dict:
    """
    Run the full model at a single decision point, then check the actual outcome.

    Data is sliced strictly up to (but not including) `ts` to prevent any
    look-ahead bias. The outcome is checked at ts + tau minutes.

    Args:
        ts: Decision timestamp (UTC). Must be present in df_1m.
        df_1m: Full 1-minute OHLCV history.
        df_1h: Full 1-hour OHLCV history.
        df_4h: Full 4-hour OHLCV history.
        strike: Explicit strike price in USD. Must be above spot. If None,
            derived as spot * (1 + abs(strike_offset)).
        strike_offset: Magnitude of the offset from spot (sign is ignored).
            Ignored if strike is provided explicitly.
        p_market: Kalshi market-implied probability (simulated input).
        bankroll: Capital for Kelly sizing.
        tau: Minutes to expiry.

    Returns:
        Dict with all intermediate values, the model decision, and the actual outcome.

    Raises:
        ValueError: If ts is not in the data or there is insufficient history.
        KeyError: If the expiry timestamp is missing from df_1m.
    """
    # --- Slice data up to the decision timestamp (strict, no look-ahead) ---
    hist_1m = df_1m.loc[df_1m.index <= ts].iloc[-200:]
    hist_1h = df_1h.loc[df_1h.index <= ts].iloc[-100:]
    hist_4h = df_4h.loc[df_4h.index <= ts].iloc[-120:]

    if len(hist_1m) < 120:
        raise ValueError(f"Only {len(hist_1m)} 1m candles before {ts} — need ≥120.")
    if len(hist_1h) < 60:
        raise ValueError(f"Only {len(hist_1h)} 1h candles before {ts} — need ≥60.")
    if len(hist_4h) < 90:
        raise ValueError(f"Only {len(hist_4h)} 4h candles before {ts} — need ≥90.")

    spot = float(hist_1m["close"].iloc[-1])

    # --- Derive strike: always above spot regardless of trade direction ---
    if strike is None:
        strike = spot * (1 + abs(strike_offset))

    # --- Run structure first so side is known before simulating p_market ---
    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_4h)
    confirm = compute_confirmation(hist_1h, hist_1m=hist_1m)
    prob    = estimate_probability(spot, strike, tau, vol.vol_60m,
                                   confirmation_score=confirm.confirmation_score)

    # --- Resolve p_market: simulate from strike offset and gate-implied side ---
    effective_offset = strike / spot - 1
    gate_side = "yes" if struct.structure_bias == 1 else "no"
    if p_market is None:
        p_market = simulate_p_market(effective_offset, side=gate_side)
    pricing = evaluate_edge(prob.p_yes, p_market)
    kelly   = compute_kelly_size(prob.p_yes, p_market, bankroll, side=gate_side)
    dec     = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                             prob.p_yes, p_market, bankroll,
                             confirmation_score=confirm.confirmation_score,
                             no_score=confirm.no_score,
                             ema_alignment=confirm.ema_alignment)

    # --- Check actual outcome at expiry ---
    expiry_ts = ts + pd.Timedelta(minutes=tau)

    # Find the closest 1m candle at or after expiry (handles minor gaps)
    future = df_1m.loc[df_1m.index >= expiry_ts]
    if future.empty:
        raise KeyError(
            f"No data at or after expiry {expiry_ts}. "
            f"The dataset ends at {df_1m.index[-1]}. Fetch more data or choose an earlier timestamp."
        )
    actual_close = float(future.iloc[0]["close"])
    actual_ts    = future.index[0]
    resolved_yes = actual_close > strike

    # --- P&L if a trade was placed ---
    # Transaction cost is deducted on every trade regardless of outcome.
    # On a win the gross contract payout is reduced by the cost.
    # On a loss the stake is lost plus the cost (e.g. you still paid the spread).
    pnl = 0.0
    if dec.decision == "trade" and dec.bet_amount > 0:
        trade_cost = dec.bet_amount * (kalshi_fee(p_market) + DEFAULT_SLIPPAGE + DEFAULT_SPREAD)
        if dec.side == "yes" and resolved_yes:
            pnl = dec.bet_amount * (1 - p_market) / p_market - trade_cost
        elif dec.side == "yes" and not resolved_yes:
            pnl = -dec.bet_amount - trade_cost
        elif dec.side == "no" and not resolved_yes:
            pnl = dec.bet_amount * p_market / (1 - p_market) - trade_cost
        elif dec.side == "no" and resolved_yes:
            pnl = -dec.bet_amount - trade_cost

    # --- Model accuracy ---
    # Side is gate-determined: YES if structure+confirmation are bullish, NO if bearish.
    # A no_trade decision means no directional call was made — marked as None.
    model_called_yes = dec.side == "yes" if dec.decision == "trade" else None
    model_correct = (model_called_yes == resolved_yes) if model_called_yes is not None else None

    return {
        # Setup
        "decision_time":   ts,
        "expiry_time":     actual_ts,
        "spot":            spot,
        "strike":          strike,
        "strike_offset_pct": (strike / spot - 1) * 100,
        "tau":             tau,
        "p_market":        p_market,
        "bankroll":        bankroll,
        # Volatility
        "vol_30m":         vol.vol_30m,
        "vol_60m":         vol.vol_60m,
        "vol_120m":        vol.vol_120m,
        # Probability engine
        "p_yes":           prob.p_yes,
        "z_score":         prob.z_score,
        "sigma_tau":       prob.sigma_to_expiry,
        "expected_move_pct": prob.expected_move_pct,
        # Pricing
        "raw_edge":        pricing.raw_edge,
        "net_edge":        pricing.net_edge,
        "edge_qualifies":  pricing.qualifies,
        # Structure
        "structure_bias":  struct.structure_bias,
        "swing_highs":     struct.swing_highs,
        "swing_lows":      struct.swing_lows,
        "structure_reason": struct.reason,
        # Confirmation
        "confirmation_bias": confirm.confirmation_bias,
        "ema_alignment":   confirm.ema_alignment,
        "stoch_k":         confirm.stoch_k,
        "stoch_bias":      confirm.stoch_bias,
        "volume_confirmed": confirm.volume_confirmed,
        "confirm_reason":  confirm.reason,
        # Kelly
        "kelly_fraction":  kelly.kelly_fraction,
        "bet_fraction":    kelly.bet_fraction,
        "bet_amount":      kelly.bet_amount,
        "was_capped":      kelly.was_capped,
        # Decision
        "decision":        dec.decision,
        "side":            dec.side,
        "gate_reasons":    dec.reasons,
        # Outcome
        "actual_close":    actual_close,
        "resolved_yes":    resolved_yes,
        "model_called_yes": model_called_yes,
        "model_correct":   model_correct,
        "trade_cost":      trade_cost if dec.decision == "trade" and dec.bet_amount > 0 else 0.0,
        "pnl":             pnl,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(r: dict) -> None:
    """Print a full labeled evaluation report from the result dict."""

    def row(label, value, width=28):
        print(f"  {label:<{width}} {value}")

    W = 62
    print("\n" + "=" * W)
    print("  SINGLE-POINT MODEL EVALUATION")
    print("=" * W)

    print("\n── SETUP ──────────────────────────────────────────────")
    row("Decision time (UTC):", r["decision_time"].strftime("%Y-%m-%d %H:%M"))
    row("Expiry time (UTC):",   r["expiry_time"].strftime("%Y-%m-%d %H:%M"))
    row("Spot at decision:",    f"${r['spot']:,.2f}")
    row("Strike:",              f"${r['strike']:,.2f}  ({r['strike_offset_pct']:+.3f}% from spot)")
    row("Tau (min to expiry):", f"{r['tau']}")
    row("Kalshi p_market:",     f"{r['p_market']:.2%}")

    print("\n── MODULE 1: REALIZED VOLATILITY ───────────────────────")
    row("vol_30m (per min):",  f"{r['vol_30m']:.6f}")
    row("vol_60m (per min):",  f"{r['vol_60m']:.6f}  ← used for p_yes")
    row("vol_120m (per min):", f"{r['vol_120m']:.6f}")

    print("\n── MODULE 2: PROBABILITY ENGINE ────────────────────────")
    row("sigma_tau:",          f"{r['sigma_tau']:.6f}  (= vol_60m × √{r['tau']})")
    row("z_score:",            f"{r['z_score']:+.4f}")
    row("expected_move_pct:",  f"{r['expected_move_pct']:.4f}%  (1σ move)")
    row("p_yes (model):",      f"{r['p_yes']:.4f}  ({r['p_yes']:.2%})")

    print("\n── MODULE 3: PRICING COMPARISON ────────────────────────")
    row("p_market:",           f"{r['p_market']:.4f}")
    row("raw_edge:",           f"{r['raw_edge']:+.4f}")
    row("net_edge:",           f"{r['net_edge']:+.4f}")
    row("qualifies:",          str(r['edge_qualifies']))

    print("\n── MODULE 4: MARKET STRUCTURE (4h) ─────────────────────")
    row("structure_bias:",     f"{r['structure_bias']:+d}")
    highs = [f"${h:,.0f}" for h in r['swing_highs']]
    lows  = [f"${l:,.0f}" for l in r['swing_lows']]
    row("swing_highs:",        "  →  ".join(highs))
    row("swing_lows:",         "  →  ".join(lows))
    row("reason:",             r['structure_reason'][:55])

    print("\n── MODULE 5: CONFIRMATION INDICATORS (1h) ───────────────")
    row("ema_alignment:",      r['ema_alignment'])
    row("stoch_k:",             f"{r['stoch_k']:.2f}")
    row("stoch_bias:",         f"{r['stoch_bias']:+d}")
    row("volume_confirmed:",   str(r['volume_confirmed']))
    row("confirmation_bias:",  f"{r['confirmation_bias']:+d}")
    row("reason:",             r['confirm_reason'][:55])

    print("\n── MODULE 6: KELLY SIZING ───────────────────────────────")
    row("kelly_fraction:",     f"{r['kelly_fraction']:.4f}  ({r['kelly_fraction']:.2%})")
    row("bet_fraction:",       f"{r['bet_fraction']:.4f}  ({r['bet_fraction']:.2%})")
    row("was_capped:",         str(r['was_capped']))
    row("bet_amount:",         f"${r['bet_amount']:,.2f}")

    print("\n── MODULE 7: DECISION ───────────────────────────────────")
    row("decision:",           r['decision'].upper())
    row("side:",               r['side'].upper())
    for i, reason in enumerate(r['gate_reasons'], 1):
        print(f"    {i}. {reason}")

    print("\n── ACTUAL OUTCOME ───────────────────────────────────────")
    row("BTC at expiry:",      f"${r['actual_close']:,.2f}")
    row("Strike:",             f"${r['strike']:,.2f}")
    move = r['actual_close'] - r['spot']
    row("Price move:",         f"${move:+,.2f}  ({move/r['spot']*100:+.3f}%)")
    row("Resolved YES:",       str(r['resolved_yes']))

    print("\n── MODEL ACCURACY ───────────────────────────────────────")
    row("p_yes (model):",      f"{r['p_yes']:.2%}  (prob K not reached)")
    if r['model_called_yes'] is None:
        row("Model called:",   "— (no trade)")
        row("Actual result:",  "YES" if r['resolved_yes'] else "NO  (expiry < K)")
        row("Model correct:",  "— (no trade placed)")
        row("P&L:",            "$0.00")
    else:
        row("Model called:",   "YES (expiry > K)" if r['model_called_yes'] else "NO  (expiry < K)")
        row("Actual result:",  "YES" if r['resolved_yes'] else "NO  (expiry < K)")
        correct_str = "CORRECT ✓" if r['model_correct'] else "WRONG ✗"
        row("Model correct:",  correct_str)
        row("Transaction cost:", f"${r['trade_cost']:,.2f}  (fee+slippage+spread)")
        row("P&L (net):",      f"${r['pnl']:+,.2f}")

    print("\n" + "=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> pd.Timestamp:
    """Parse a timestamp string and attach UTC timezone."""
    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse timestamp '{ts_str}'. Use format: 'YYYY-MM-DD HH:MM'"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the model at a single timestamp and check the outcome."
    )
    parser.add_argument("--time", required=True,
                        help="Decision timestamp in UTC, e.g. '2025-06-15 14:00'")
    parser.add_argument("--strike", type=float, default=None,
                        help="Explicit strike price in USD (overrides --offset)")
    parser.add_argument("--offset", type=float, default=0.005,
                        help="Strike offset magnitude (e.g. 0.005 = K is 0.5%% above spot; sign ignored)")
    parser.add_argument("--p-market", type=float, default=None,
                        help="Kalshi market probability (default: dynamic simulation from strike offset)")
    parser.add_argument("--bankroll", type=float, default=BANKROLL,
                        help=f"Bankroll for Kelly sizing (default ${BANKROLL:,})")
    args = parser.parse_args()

    ts = _parse_ts(args.time)

    print("\nLoading cached OHLCV data...")
    df_1m, df_1h, df_4h = load_data()

    result = evaluate_point(
        ts=ts,
        df_1m=df_1m, df_1h=df_1h, df_4h=df_4h,
        strike=args.strike,
        strike_offset=args.offset,
        p_market=args.p_market,
        bankroll=args.bankroll,
    )

    print_report(result)


if __name__ == "__main__":
    main()
