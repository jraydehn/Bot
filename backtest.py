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
    df_4h: pd.DataFrame,
    strike_offset: float = DEFAULT_OFFSET,
    p_market: float = None,
    bankroll: float = DEFAULT_BANKROLL,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Evaluate the model at every complete hourly candle in the dataset.

    A decision point is skipped if:
      - There are fewer than 120 1m, 60 1h, or 90 4h candles before it.
      - The expiry timestamp (decision + 60 min) has no data.

    Args:
        df_1m: Full 1-minute OHLCV history.
        df_1h: Full 1-hour OHLCV history (used for decision timestamps).
        df_4h: Full 4-hour OHLCV history.
        strike_offset: Offset magnitude for strike above spot (sign ignored).
        p_market: Fixed Kalshi market probability. If None, simulated per-step
            from strike_offset via simulate_p_market() (default behaviour).
        bankroll: Starting capital. Updated after each trade to reflect P&L.
        start: Optional ISO date string to restrict backtest window start.
        end: Optional ISO date string to restrict backtest window end.

    Returns:
        DataFrame of per-decision results with columns for all key metrics.
    """
    # Build the list of hourly decision timestamps from the 1h index
    decision_times = df_1h.index.copy()

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        decision_times = decision_times[decision_times >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        decision_times = decision_times[decision_times <= end_ts]

    # Leave a warm-up buffer: skip the first 90 4h candles (= 15 days)
    # to ensure every decision point has enough history for market structure.
    min_history_ts = df_4h.index[89]  # 90th 4h candle = 15 days of warm-up
    decision_times = decision_times[decision_times > min_history_ts]

    total = len(decision_times)
    print(f"\n  Decision points : {total:,}")
    print(f"  Strike offset   : {strike_offset:.3%} above spot")
    p_market_display = f"{p_market:.2%}" if p_market is not None else "dynamic (simulated per step)"
    print(f"  p_market        : {p_market_display}")
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
                df_1m=df_1m, df_1h=df_1h, df_4h=df_4h,
                strike_offset=strike_offset,
                p_market=p_market,
                bankroll=current_bankroll,
                tau=TAU,
            )
        except (ValueError, KeyError):
            skipped += 1
            continue

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
    parser.add_argument("--save", default=None,
                        help="Save results to this CSV path (e.g. results/backtest.csv)")
    args = parser.parse_args()

    print("=" * 62)
    print("  KALSHI BTC BACKTEST")
    print("=" * 62)

    print("\nLoading cached OHLCV data...")
    df_1m, df_1h, df_4h = load_data()

    results = run_backtest(
        df_1m=df_1m, df_1h=df_1h, df_4h=df_4h,
        strike_offset=args.offset,
        p_market=args.p_market,
        bankroll=args.bankroll,
        start=args.start,
        end=args.end,
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
