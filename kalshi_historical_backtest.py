"""
Historical backtest using real Kalshi contract data.

Combines real Kalshi market prices and settlement results with the model's
full signal chain (Gate EMA, Gate NO, 5% edge, current Kelly fractions).

For each decision point in the Kalshi history CSV:
  1. Find the contract whose strike is closest to spot * (1 + offset).
  2. Load historical OHLCV slices strictly before that timestamp.
  3. Run structure, confirmation, probability, and decision gates
     with the real Kalshi p_market_yes_open.
  4. Compute P&L from the real result (YES/NO) already in the CSV.

No future data look-ahead and no simulated p_market — this backtest
reflects what the model would have done with live Kalshi prices.

Usage:
    python3 kalshi_historical_backtest.py
    python3 kalshi_historical_backtest.py --kalshi results/kalshi_btc_history_authenticated.csv
    python3 kalshi_historical_backtest.py --offset 0.005 --flat-bet 100
    python3 kalshi_historical_backtest.py --save results/kalshi_backtest_out.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from market_data import compute_realized_volatility
from probability_engine import estimate_probability
from market_structure import detect_market_structure, resample_to_15min
from confirmation_indicators import compute_confirmation
from decision import evaluate_trade
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD


DATA_DIR       = Path(__file__).parent / "data"
DEFAULT_KALSHI = Path(__file__).parent / "results" / "kalshi_btc_history_authenticated.csv"
DEFAULT_OFFSET = 0.005
TAU            = 60       # Kalshi hourly contracts expire 60 minutes after open
DEFAULT_BK     = 10_000


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _latest_parquet(symbol: str, interval: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"*{symbol}*_{interval}_*.parquet"),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No {interval} data for {symbol} in {DATA_DIR}. "
            f"Run fetch_data.py first."
        )
    return matches[-1]


def load_ohlcv(symbol: str = "BTCUSDT") -> tuple:
    """Return (df_1m, df_1h) with UTC-aware DatetimeIndex."""
    paths = {iv: _latest_parquet(symbol, iv) for iv in ("1m", "1h")}
    dfs = {}
    for iv, path in paths.items():
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        dfs[iv] = df
        print(f"  {iv:3s}: {path.name}  ({len(df):,} rows)")
    return dfs["1m"], dfs["1h"]


# ---------------------------------------------------------------------------
# Core: evaluate one Kalshi contract row
# ---------------------------------------------------------------------------

def evaluate_contract(
    ts: pd.Timestamp,
    strike: float,
    p_market: float,
    resolved_yes: bool,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    bankroll: float,
    flat_bet: Optional[float],
) -> dict:
    """
    Run the full model for one Kalshi contract and return a result dict.

    Args:
        ts:           Contract open time (decision point).
        strike:       Kalshi contract strike (BTC price threshold).
        p_market:     Real Kalshi YES market price at open.
        resolved_yes: True iff the contract settled YES (expiry > strike).
        df_1m / df_1h / df_15m: Full OHLCV history (sliced internally).
        bankroll:     Current bankroll for Kelly sizing.
        flat_bet:     If set, override Kelly and use this fixed bet size.

    Returns:
        Dict with all intermediate values plus P&L.
    """
    # Slice strictly before decision time (no look-ahead)
    hist_1m  = df_1m.loc[df_1m.index  <= ts].iloc[-200:]
    hist_1h  = df_1h.loc[df_1h.index  <= ts].iloc[-100:]
    hist_15m = df_15m.loc[df_15m.index <= ts].iloc[-120:]

    if len(hist_1m) < 120 or len(hist_1h) < 60 or len(hist_15m) < 90:
        return None  # insufficient history

    spot = float(hist_1m["close"].iloc[-1])

    # Signal chain (same modules as live model)
    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_15m)
    confirm = compute_confirmation(hist_1h, hist_1m=hist_1m)
    prob    = estimate_probability(spot, strike, TAU, vol.vol_60m,
                                   confirmation_score=confirm.confirmation_score)

    dec = evaluate_trade(
        structure_bias    = struct.structure_bias,
        confirmation_bias = confirm.confirmation_bias,
        p_model           = prob.p_yes,
        p_market          = p_market,
        bankroll          = bankroll,
        confirmation_score= confirm.confirmation_score,
        no_score          = confirm.no_score,
        ema_alignment     = confirm.ema_alignment,
    )

    # P&L using real Kalshi result
    pnl       = 0.0
    trade_cost = 0.0
    if dec.decision == "trade" and dec.bet_amount > 0:
        bet        = flat_bet if flat_bet is not None else dec.bet_amount
        trade_cost = bet * (kalshi_fee(p_market) + DEFAULT_SLIPPAGE + DEFAULT_SPREAD)
        if dec.side == "yes" and resolved_yes:
            pnl = bet * (1 - p_market) / p_market - trade_cost
        elif dec.side == "yes" and not resolved_yes:
            pnl = -bet - trade_cost
        elif dec.side == "no" and not resolved_yes:
            pnl = bet * p_market / (1 - p_market) - trade_cost
        else:  # side == "no" and resolved_yes
            pnl = -bet - trade_cost

    return {
        "decision_time":      ts,
        "spot":               spot,
        "strike":             strike,
        "strike_offset_pct":  (strike / spot - 1) * 100,
        "p_market":           p_market,
        "p_model":            prob.p_yes,
        "raw_edge":           dec.raw_edge,
        "net_edge":           dec.net_edge,
        "vol_60m":            vol.vol_60m,
        "structure_bias":     struct.structure_bias,
        "ema_alignment":      confirm.ema_alignment,
        "confirmation_bias":  confirm.confirmation_bias,
        "confirmation_score": confirm.confirmation_score,
        "no_score":           confirm.no_score,
        "stoch_bias":         confirm.stoch_bias,
        "obi_score":          confirm.obi_score,
        "vwap_signal":        confirm.vwap_signal,
        "decision":           dec.decision,
        "side":               dec.side,
        "bet_amount":         dec.bet_amount if flat_bet is None else (flat_bet if dec.decision == "trade" else 0.0),
        "resolved_yes":       resolved_yes,
        "pnl":                pnl,
        "trade_cost":         trade_cost,
        "bankroll":           bankroll,
        "gate_reasons":       dec.reasons,
    }


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    kalshi_path: Path,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    offset: float = DEFAULT_OFFSET,
    bankroll: float = DEFAULT_BK,
    flat_bet: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run the historical backtest over all unique decision points in kalshi_path.

    For each timestamp, selects the contract whose strike is closest to
    spot * (1 + offset) and runs the full signal + decision chain.
    """
    kalshi = pd.read_csv(kalshi_path)
    kalshi["timestamp"] = pd.to_datetime(kalshi["timestamp"], utc=True)

    # Keep only rows with a real p_market and a settled result
    kalshi = kalshi.dropna(subset=["p_market_yes_open", "result"]).copy()
    kalshi["resolved_yes"] = kalshi["result"].str.lower() == "yes"

    # Precompute 15m resample once (structure module input)
    print("  Resampling 1m → 15m for market structure...")
    df_15m = resample_to_15min(df_1m)

    decision_times = sorted(kalshi["timestamp"].unique())
    total = len(decision_times)
    print(f"  Decision points : {total:,}")
    print(f"  Contracts total : {len(kalshi):,}")
    print(f"  Offset          : {offset:.3%} above spot")
    if flat_bet is not None:
        print(f"  Sizing          : flat ${flat_bet:,.0f} (no compounding)")
    print(f"  Starting bankroll: ${bankroll:,.2f}\n")

    rows = []
    current_bankroll = bankroll
    skipped = 0

    for i, ts in enumerate(decision_times, 1):
        # Spot price at this timestamp
        hist_spot = df_1m.loc[df_1m.index <= ts]
        if hist_spot.empty:
            skipped += 1
            continue
        spot = float(hist_spot["close"].iloc[-1])
        target_strike = spot * (1 + offset)

        # Select the contract nearest the target strike
        group = kalshi[kalshi["timestamp"] == ts].copy()
        group["strike_dist"] = (group["strike"] - target_strike).abs()
        best = group.nsmallest(1, "strike_dist").iloc[0]

        result = evaluate_contract(
            ts           = ts,
            strike       = float(best["strike"]),
            p_market     = float(best["p_market_yes_open"]),
            resolved_yes = bool(best["resolved_yes"]),
            df_1m        = df_1m,
            df_1h        = df_1h,
            df_15m       = df_15m,
            bankroll     = current_bankroll,
            flat_bet     = flat_bet,
        )

        if result is None:
            skipped += 1
            continue

        current_bankroll += result["pnl"]
        result["bankroll_after"] = current_bankroll
        rows.append(result)

        if result["decision"] == "trade":
            status = f"TRADE {result['side'].upper()} ${result['pnl']:+.2f}"
        else:
            # Show first gate that failed
            first_fail = next(
                (r for r in result["gate_reasons"] if "FAILED" in r), ""
            )
            # Extract "Gate X FAILED" → "X"
            m = re.match(r"Gate (\S+) FAILED", first_fail)
            gate_tag = m.group(1) if m else "no_trade"
            status = f"no_trade [{gate_tag}]"
        print(f"  [{i:>3}/{total}]  {ts}  spot=${spot:,.0f}  "
              f"strike=${best['strike']:,.0f}  p_mkt={best['p_market_yes_open']:.3f}"
              f"  {status}")

    print(f"\n  Evaluated: {len(rows):,}  |  Skipped: {skipped}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    def row(label, value, w=32):
        print(f"  {label:<{w}} {value}")

    W = 64
    print("\n" + "=" * W)
    print("  KALSHI HISTORICAL BACKTEST SUMMARY")
    print("=" * W)

    trades    = df[df["decision"] == "trade"].copy()
    no_trades = df[df["decision"] == "no_trade"]

    print("\n── COVERAGE ──────────────────────────────────────────────────")
    row("Decision points:", f"{len(df):,}")
    row("Trades taken:", f"{len(trades):,}  ({100*len(trades)/max(len(df),1):.1f}%)")
    row("No-trades (gate blocked):", f"{len(no_trades):,}")

    if trades.empty:
        print("\n  No trades taken.")
        return

    yes_t = trades[trades["side"] == "yes"]
    no_t  = trades[trades["side"] == "no"]
    row("  — YES trades:", f"{len(yes_t):,}")
    row("  — NO trades:", f"{len(no_t):,}")

    print("\n── OUTCOMES ──────────────────────────────────────────────────")
    trades["win"] = (
        ((trades["side"] == "yes") & (trades["resolved_yes"])) |
        ((trades["side"] == "no")  & (~trades["resolved_yes"]))
    )
    wr = trades["win"].mean()
    row("Win rate (all trades):", f"{wr:.2%}")
    if not yes_t.empty:
        yes_t = yes_t.copy(); yes_t["win"] = yes_t["resolved_yes"]
        row("  YES win rate:", f"{yes_t['win'].mean():.2%}  (n={len(yes_t)})")
    if not no_t.empty:
        no_t = no_t.copy(); no_t["win"] = ~no_t["resolved_yes"]
        row("  NO  win rate:", f"{no_t['win'].mean():.2%}  (n={len(no_t)})")

    print("\n── P&L ───────────────────────────────────────────────────────")
    total_pnl  = trades["pnl"].sum()
    total_cost = trades["trade_cost"].sum()
    avg_win    = trades.loc[trades["win"], "pnl"].mean() if trades["win"].any() else 0
    avg_loss   = trades.loc[~trades["win"], "pnl"].mean() if (~trades["win"]).any() else 0
    start_br   = df["bankroll"].iloc[0]
    final_br   = df["bankroll_after"].iloc[-1]

    row("Total P&L (net):", f"${total_pnl:+,.2f}")
    row("Total transaction costs:", f"${total_cost:,.2f}")
    row("Starting bankroll:", f"${start_br:,.2f}")
    row("Final bankroll:", f"${final_br:,.2f}")
    row("Return:", f"{100*(final_br/start_br - 1):+.2f}%")
    row("Avg winning trade:", f"${avg_win:+,.2f}")
    row("Avg losing trade:", f"${avg_loss:+,.2f}")
    if avg_loss != 0:
        row("Profit factor:", f"{abs(avg_win / avg_loss):.2f}")

    print("\n── GATE FAILURE BREAKDOWN ────────────────────────────────────")
    for gate in ("Gate EMA", "Gate NO", "Gate 3", "Gate R:R", "Gate 0"):
        fails = no_trades[no_trades["gate_reasons"].apply(
            lambda r: any(gate in x for x in r))]
        if len(fails):
            row(f"{gate} failures:", f"{len(fails):,}")

    print("\n── SIGNAL BREAKDOWN (all points) ────────────────────────────")
    for col in ("structure_bias", "ema_alignment", "confirmation_score", "no_score"):
        if col not in df.columns:
            continue
        print(f"\n  {col}:")
        for val, grp in df.groupby(df[col].astype(str)):
            t_sub = grp[grp["decision"] == "trade"]
            if len(t_sub) == 0:
                continue
            t_sub = t_sub.copy()
            t_sub["win"] = (
                ((t_sub["side"] == "yes") & (t_sub["resolved_yes"])) |
                ((t_sub["side"] == "no")  & (~t_sub["resolved_yes"]))
            )
            print(f"    {val:>10}  trades={len(t_sub):>3}  "
                  f"win={t_sub['win'].mean():.1%}  pnl=${t_sub['pnl'].sum():+,.2f}")

    print("\n" + "=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical backtest using real Kalshi KXBTCD contract data."
    )
    parser.add_argument("--kalshi", default=str(DEFAULT_KALSHI),
                        help=f"Path to Kalshi history CSV (default: {DEFAULT_KALSHI.name})")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET,
                        help=f"Target strike offset above spot (default {DEFAULT_OFFSET})")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BK,
                        help=f"Starting bankroll in USD (default ${DEFAULT_BK:,})")
    parser.add_argument("--flat-bet", type=float, default=None,
                        help="Fixed dollar bet per trade (disables Kelly)")
    parser.add_argument("--save", default=None,
                        help="Save result rows to this CSV path")
    args = parser.parse_args()

    kalshi_path = Path(args.kalshi)
    if not kalshi_path.exists():
        kalshi_path = Path(__file__).parent / args.kalshi
    if not kalshi_path.exists():
        print(f"ERROR: Kalshi CSV not found: {args.kalshi}")
        sys.exit(1)

    print("=" * 64)
    print("  KALSHI HISTORICAL BACKTEST")
    print("=" * 64)
    print(f"\nKalshi data: {kalshi_path}")
    print("\nLoading OHLCV data...")
    df_1m, df_1h = load_ohlcv()

    results = run_backtest(
        kalshi_path = kalshi_path,
        df_1m       = df_1m,
        df_1h       = df_1h,
        offset      = args.offset,
        bankroll    = args.bankroll,
        flat_bet    = args.flat_bet,
    )

    if results.empty:
        print("No results — check data availability and date range.")
        return

    print_summary(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        csv_df = results.drop(columns=["gate_reasons"], errors="ignore")
        csv_df.to_csv(save_path, index=False)
        print(f"  Results saved to {save_path}")


if __name__ == "__main__":
    main()
