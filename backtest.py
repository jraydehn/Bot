"""
Walk-forward backtest for the Kalshi BTC event trading model.

Evaluates the model at every hourly decision point in the loaded dataset,
using data available strictly before each timestamp (no look-ahead).

Strike K is always set above spot: K = spot * (1 + abs(strike_offset)).
Trade side (YES/NO) is determined by the gate system from structure_bias
and confirmation_bias — not from the sign of the strike offset.

    YES trade: model bets expiry price > K  (bullish gates: both biases = +1)
    NO  trade: model bets expiry price < K  (bearish gates: both biases = -1)

Usage:
    python backtest.py
    python backtest.py --offset 0.005 --p-market 0.45
    python backtest.py --start "2026-02-15" --end "2026-03-15"
    python backtest.py --offset 0.003 --p-market 0.48 --bankroll 50000
"""

import argparse
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_point import load_data, evaluate_point
from market_structure import resample_to_15min

DATA_DIR = Path(__file__).parent / "data"
TAU = 60
DEFAULT_OFFSET = 0.005
DEFAULT_BANKROLL = 10_000


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def run_backtest(
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame = None,  # retained for backward compatibility; unused
    strike_offset: float = DEFAULT_OFFSET,
    p_market: float = None,
    bankroll: float = DEFAULT_BANKROLL,
    start: str = None,
    end: str = None,
    flat_bet: float = None,
    structure_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Evaluate the model at every complete hourly candle in the dataset.

    Market structure is now derived from 15-minute candles resampled from df_1m.
    df_4h is accepted but ignored (kept for backward compatibility with callers
    that pass it positionally, e.g. monte_carlo.py).

    A decision point is skipped if:
      - There are fewer than 120 1m, 60 1h, or 120 15m candles before it.
      - The expiry timestamp (decision + 60 min) has no data.

    Args:
        df_1m: Full 1-minute OHLCV history.
        df_1h: Full 1-hour OHLCV history (used for decision timestamps).
        df_4h: Ignored. Retained for backward compatibility.
        strike_offset: Offset magnitude for strike above spot (sign ignored).
        p_market: Fixed Kalshi market probability. If None, simulated per-step
            from strike_offset via simulate_p_market() (default behaviour).
        bankroll: Starting capital. Updated after each trade to reflect P&L.
        start: Optional ISO date string to restrict backtest window start.
        end: Optional ISO date string to restrict backtest window end.
        flat_bet: If set, override Kelly sizing and bet this fixed dollar amount
            on every trade. Useful for measuring raw win rate without compounding
            distortion. Kelly sizing in evaluate_point is still computed but
            the P&L is recalculated here using flat_bet instead.

    Returns:
        DataFrame of per-decision results with columns for all key metrics.
    """
    # Determine which DataFrame to use for market structure detection.
    # structure_df allows callers (e.g. compare_structure.py) to supply a
    # pre-built DataFrame (e.g. real 4h bars) instead of the default 15m resample.
    if structure_df is not None:
        df_15m = structure_df
    else:
        df_15m = resample_to_15min(df_1m)

    # Build the list of hourly decision timestamps from the 1h index
    decision_times = df_1h.index.copy()

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        decision_times = decision_times[decision_times >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        decision_times = decision_times[decision_times <= end_ts]

    # Leave a warm-up buffer equal to the structure module's MIN_CANDLES requirement.
    # This ensures every decision point has enough history before the loop starts.
    from market_structure import MIN_CANDLES as _MS_MIN
    warmup_idx = min(_MS_MIN - 1, len(df_15m) - 1)
    min_history_ts = df_15m.index[warmup_idx]
    decision_times = decision_times[decision_times > min_history_ts]

    total = len(decision_times)
    print(f"\n  Decision points : {total:,}")
    print(f"  Strike offset   : {strike_offset:.3%} above spot")
    p_market_display = f"{p_market:.2%}" if p_market is not None else "dynamic (simulated per step)"
    print(f"  p_market        : {p_market_display}")
    if flat_bet is not None:
        print(f"  Sizing mode     : flat bet ${flat_bet:,.2f} per trade (no compounding)")
    print(f"  Starting bankroll: ${bankroll:,.2f}\n")

    rows = []
    current_bankroll = bankroll
    skipped = 0

    for i, ts in enumerate(decision_times):
        if i % 100 == 0 and i > 0:
            trades_so_far = sum(1 for r in rows if r["decision"] == "trade")
            print(f"  [{i:>4}/{total}]  trades={trades_so_far}  "
                  f"bankroll=${current_bankroll:,.0f}")

        try:
            r = evaluate_point(
                ts=ts,
                df_1m=df_1m, df_1h=df_1h, df_4h=df_15m,
                strike_offset=strike_offset,
                p_market=p_market,
                bankroll=current_bankroll,
                tau=TAU,
            )
        except (ValueError, KeyError):
            skipped += 1
            continue

        # Flat-bet override: replace Kelly-sized P&L with fixed bet amount.
        # Recalculates pnl using flat_bet so the bankroll curve reflects
        # pure signal quality without compounding distortion.
        if flat_bet is not None and r["decision"] == "trade":
            pm   = r["p_market"]
            side = r["side"]
            res  = r["resolved_yes"]
            from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
            cost = flat_bet * (kalshi_fee(pm) + DEFAULT_SLIPPAGE + DEFAULT_SPREAD)
            if side == "yes" and res:
                r["pnl"] = flat_bet * (1 - pm) / pm - cost
            elif side == "yes" and not res:
                r["pnl"] = -flat_bet - cost
            elif side == "no" and not res:
                r["pnl"] = flat_bet * pm / (1 - pm) - cost
            else:  # side == "no" and res
                r["pnl"] = -flat_bet - cost
            r["bet_amount"] = flat_bet
            r["trade_cost"] = cost

        # Update running bankroll after each closed trade
        current_bankroll += r["pnl"]
        r["bankroll_after"] = current_bankroll
        rows.append(r)

    print(f"\n  Complete. {len(rows):,} evaluated, {skipped} skipped.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a structured performance summary from the backtest results DataFrame."""

    def row(label, value, width=32):
        print(f"  {label:<{width}} {value}")

    W = 62
    print("\n" + "=" * W)
    print("  BACKTEST SUMMARY")
    print("=" * W)

    trades = df[df["decision"] == "trade"]
    no_trades = df[df["decision"] == "no_trade"]
    yes_trades = trades[trades["side"] == "yes"]
    no_trades_side = trades[trades["side"] == "no"]

    print("\n── COVERAGE ─────────────────────────────────────────────")
    row("Total decision points:", f"{len(df):,}")
    row("Trades taken:", f"{len(trades):,}  ({100*len(trades)/len(df):.1f}% of decisions)")
    row("  — YES trades:", f"{len(yes_trades):,}")
    row("  — NO trades:", f"{len(no_trades_side):,}")
    row("No-trades (gate blocked):", f"{len(no_trades):,}")

    if trades.empty:
        print("\n  No trades were taken.")
        return

    print("\n── OUTCOMES ─────────────────────────────────────────────")
    # A YES trade wins when resolved_yes is True; a NO trade wins when resolved_yes is False
    trades = trades.copy()
    trades["win"] = (
        ((trades["side"] == "yes") & (trades["resolved_yes"])) |
        ((trades["side"] == "no")  & (~trades["resolved_yes"]))
    )
    win_rate = trades["win"].mean()
    row("Win rate (all trades):", f"{win_rate:.2%}")
    if not yes_trades.empty:
        yes_trades = yes_trades.copy()
        yes_trades["win"] = yes_trades["resolved_yes"]
        row("  YES trade win rate:", f"{yes_trades['win'].mean():.2%}")
    if not no_trades_side.empty:
        no_trades_side = no_trades_side.copy()
        no_trades_side["win"] = ~no_trades_side["resolved_yes"]
        row("  NO  trade win rate:", f"{no_trades_side['win'].mean():.2%}")

    print("\n── P&L ──────────────────────────────────────────────────")
    total_pnl  = trades["pnl"].sum()
    total_cost = trades["trade_cost"].sum() if "trade_cost" in trades.columns else 0.0
    avg_win    = trades.loc[trades["win"], "pnl"].mean() if trades["win"].any() else 0
    avg_loss   = trades.loc[~trades["win"], "pnl"].mean() if (~trades["win"]).any() else 0
    starting_bankroll = df["bankroll"].iloc[0]
    final_bankroll    = df["bankroll_after"].iloc[-1]

    row("Total P&L (net):", f"${total_pnl:+,.2f}")
    row("Total transaction costs:", f"${total_cost:,.2f}")
    row("Starting bankroll:", f"${starting_bankroll:,.2f}")
    row("Final bankroll:", f"${final_bankroll:,.2f}")
    row("Return:", f"{100*(final_bankroll/starting_bankroll - 1):+.2f}%")
    row("Avg winning trade:", f"${avg_win:+,.2f}")
    row("Avg losing trade:", f"${avg_loss:+,.2f}")
    row("Profit factor:", (
        f"{abs(avg_win / avg_loss):.2f}"
        if avg_loss != 0 else "∞"
    ))

    print("\n── SIGNAL BREAKDOWN ─────────────────────────────────────")
    bias_counts = df["structure_bias"].value_counts().sort_index()
    for bias, count in bias_counts.items():
        label = {1: "Bullish (+1)", 0: "Neutral (0)", -1: "Bearish (-1)"}.get(bias, str(bias))
        row(f"  structure_bias {label}:", f"{count:,}  ({100*count/len(df):.1f}%)")

    print("\n── GATE FAILURE BREAKDOWN ───────────────────────────────")
    gate1_fails = no_trades[no_trades["gate_reasons"].apply(
        lambda r: any("Gate 1 FAILED" in x for x in r))]
    gate2_fails = no_trades[no_trades["gate_reasons"].apply(
        lambda r: any("Gate 2 FAILED" in x for x in r))]
    gate3_fails = no_trades[no_trades["gate_reasons"].apply(
        lambda r: any("Gate 3 FAILED" in x for x in r))]
    row("Gate 1 failures (structure):", f"{len(gate1_fails):,}")
    row("Gate 2 failures (confirmation):", f"{len(gate2_fails):,}")
    row("Gate 3 failures (edge):", f"{len(gate3_fails):,}")

    print("\n── EQUITY CURVE (terminal bankroll) ─────────────────────")
    # Simple text sparkline of monthly bankroll
    df_trades = df[df["decision"] == "trade"].copy()
    if not df_trades.empty:
        df_trades["month"] = df_trades["decision_time"].dt.to_period("M")
        monthly = df_trades.groupby("month")["pnl"].sum()
        for period, pnl in monthly.items():
            bar = "█" * int(abs(pnl) / 50) if abs(pnl) > 50 else "▏"
            sign = "+" if pnl >= 0 else "-"
            print(f"  {period}  {sign}${abs(pnl):>7,.0f}  {bar}")

    print("\n" + "=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of the Kalshi BTC event trading model."
    )
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET,
                        help=f"Strike offset magnitude above spot (default {DEFAULT_OFFSET})")
    parser.add_argument("--p-market", type=float, default=None,
                        help="Fixed Kalshi market probability (default: dynamic simulation from strike offset)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                        help=f"Starting bankroll in USD (default ${DEFAULT_BANKROLL:,})")
    parser.add_argument("--start", default=None,
                        help="Backtest window start date YYYY-MM-DD")
    parser.add_argument("--end", default=None,
                        help="Backtest window end date YYYY-MM-DD")
    parser.add_argument("--flat-bet", type=float, default=None,
                        help="Fixed dollar amount per trade (disables Kelly compounding)")
    parser.add_argument("--save", default=None,
                        help="Save results to this CSV path (e.g. results/backtest.csv)")
    args = parser.parse_args()

    print("=" * 62)
    print("  KALSHI BTC BACKTEST")
    print("=" * 62)

    print("\nLoading cached OHLCV data...")
    df_1m, df_1h, _ = load_data()

    results = run_backtest(
        df_1m=df_1m, df_1h=df_1h,
        strike_offset=args.offset,
        p_market=args.p_market,
        bankroll=args.bankroll,
        start=args.start,
        end=args.end,
        flat_bet=args.flat_bet,
    )

    if results.empty:
        print("No results — check that your date range overlaps with available data.")
        return

    print_summary(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop non-serialisable list columns before saving
        csv_df = results.drop(columns=["swing_highs", "swing_lows", "gate_reasons"])
        csv_df.to_csv(save_path, index=False)
        print(f"  Results saved to {save_path}")


if __name__ == "__main__":
    main()
