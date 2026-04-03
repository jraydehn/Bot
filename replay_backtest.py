"""
Replay backtest against saved paper trade data.

Re-applies configurable gate logic to each resolved trade row using the
indicator values and real Kalshi p_market prices already recorded in the CSV.
Tests multiple gate configurations and ranks them by PnL and win rate.

Only resolved trade rows are used (decision=="trade" with resolved_yes filled).
No-trade rows cannot be evaluated as their outcomes were never recorded.

Usage:
    python3 replay_backtest.py
    python3 replay_backtest.py --csv results/paper_trades_sol.csv
    python3 replay_backtest.py --csv results/paper_trades.csv --flat-bet 50
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD


# ---------------------------------------------------------------------------
# PnL calculator (mirrors paper_trade_runner / evaluate_edge logic)
# ---------------------------------------------------------------------------

def compute_pnl(p_market: float, side: str, resolved_yes: bool, bet: float) -> float:
    """Compute net PnL for a single trade using real Kalshi p_market and outcome."""
    cost = bet * (kalshi_fee(p_market) + DEFAULT_SLIPPAGE + DEFAULT_SPREAD)
    if side == "yes":
        if resolved_yes:
            return bet * (1 - p_market) / p_market - cost
        else:
            return -bet - cost
    else:  # no
        if not resolved_yes:
            return bet * p_market / (1 - p_market) - cost
        else:
            return -bet - cost


# ---------------------------------------------------------------------------
# Gate logic — returns "yes", "no", or None (skip)
# ---------------------------------------------------------------------------

def apply_gates(row: pd.Series, cfg: dict):
    """
    Apply gate configuration to a single trade row.
    Returns "yes", "no", or None if the trade would be skipped.

    cfg keys (all optional, defaults are permissive):
        min_net_edge          float  minimum net_edge to trade (default 0.0)
        require_structure     bool   structure_bias must be nonzero (default False)
        block_ema_stack_opp   bool   block if ema_stack_bias opposes structure (default False)
        block_stoch_opp       bool   block if stoch_bias opposes structure (default False)
        invert_obi            bool   flip obi_score sign before scoring (default False)
        min_confirmation_score int   minimum confirmation_score to trade YES (default 0)
        min_no_score          int    minimum no_score to trade NO (default 0)
        require_vwap_align    bool   vwap_signal must align with trade side (default False)

        # New gates derived from EMA x side x p_market analysis:
        ema_dir_gate          bool   EMA must oppose trade side — bearish→YES, bullish→NO
                                     (ema=bullish+YES: 0% win; ema=bearish+NO: 37% win)
        pm_yes_min            float  minimum p_market for YES trades (default 0.0)
        pm_no_max             float  maximum p_market for NO trades (default 1.0)
    """
    structure  = _int(row.get("structure_bias", 0))
    stoch      = _int(row.get("stoch_bias", 0))
    ema_stack  = _int(row.get("ema_stack_bias", 0))
    obi        = _int(row.get("obi_score", 0))
    vwap       = _int(row.get("vwap_signal", 0))
    conf_score = _int(row.get("confirmation_score", 0))
    no_score   = _int(row.get("no_score", 0))
    net_edge   = float(row.get("net_edge", 0) or 0)
    ema_align  = str(row.get("ema_alignment", "") or "")
    p_market   = float(row.get("p_market", 0.5) or 0.5)

    if cfg.get("invert_obi"):
        obi = -obi

    # Determine trade side from structure bias (mirrors live model)
    if structure == 1:
        side = "yes"
    elif structure == -1:
        side = "no"
    else:
        if cfg.get("require_structure", False):
            return None
        # Neutral structure: use original logged side
        side = str(row.get("side", "yes"))

    # Net edge gate
    if net_edge < cfg.get("min_net_edge", 0.0):
        return None

    # EMA stack opposes structure
    if cfg.get("block_ema_stack_opp") and ema_stack != 0 and ema_stack != structure:
        return None

    # Stochastic opposes structure
    if cfg.get("block_stoch_opp") and stoch != 0 and stoch != structure:
        return None

    # VWAP alignment gate
    if cfg.get("require_vwap_align") and vwap != 0 and vwap != structure:
        return None

    # Side-specific score gates
    if side == "yes" and conf_score < cfg.get("min_confirmation_score", 0):
        return None
    if side == "no" and no_score < cfg.get("min_no_score", 0):
        return None

    # --- New: Gate EMA-Dir (ema must oppose trade direction) ---
    # ema=bullish+YES: 0% win (n=19). ema=bearish+NO: 37% win (n=46, below break-even).
    if cfg.get("ema_dir_gate") and ema_align:
        if ema_align == "bullish" and side == "yes":
            return None
        if ema_align == "bearish" and side == "no":
            return None

    # --- New: Gate PM (p_market direction range) ---
    # YES at p_market<0.55 (ema=bearish): 24% win. YES at p_market≥0.55: 96.4%.
    # NO at p_market>0.45 (ema=bullish): 33% win. NO at p_market≤0.45: 78%+.
    pm_yes_min = cfg.get("pm_yes_min", 0.0)
    pm_no_max  = cfg.get("pm_no_max", 1.0)
    if side == "yes" and p_market < pm_yes_min:
        return None
    if side == "no" and p_market > pm_no_max:
        return None

    return side


def _int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Evaluate one configuration
# ---------------------------------------------------------------------------

def evaluate_config(trades: pd.DataFrame, cfg: dict, flat_bet) -> dict:
    wins = losses = skipped = 0
    total_pnl = 0.0
    bankroll = float(trades["bankroll"].iloc[0]) if "bankroll" in trades.columns else 10_000.0

    for _, row in trades.iterrows():
        side = apply_gates(row, cfg)
        if side is None:
            skipped += 1
            continue

        resolved = bool(row["resolved_yes"])
        p_mkt = float(row["p_market"])
        bet = flat_bet if flat_bet is not None else float(row.get("bet_amount", 100) or 100)

        pnl = compute_pnl(p_mkt, side, resolved, bet)
        total_pnl += pnl

        won = (side == "yes" and resolved) or (side == "no" and not resolved)
        if won:
            wins += 1
        else:
            losses += 1

    taken = wins + losses
    return {
        "taken":     taken,
        "skipped":   skipped,
        "wins":      wins,
        "losses":    losses,
        "win_rate":  wins / taken if taken > 0 else float("nan"),
        "total_pnl": total_pnl,
        "cfg":       cfg,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(trades: pd.DataFrame, flat_bet) -> pd.DataFrame:
    """Test a grid of gate configurations and return ranked results."""

    grid = {
        "min_net_edge":          [0.0, 0.01, 0.02, 0.05],
        "require_structure":     [False, True],
        "block_ema_stack_opp":   [False, True],
        "block_stoch_opp":       [False, True],
        "invert_obi":            [False, True],
        "min_confirmation_score":[0, 1, 2],
        "min_no_score":          [0, 1, 2],
        # New gates from EMA-Dir + PM analysis
        "ema_dir_gate":          [False, True],
        "pm_yes_min":            [0.0, 0.45, 0.55],
        "pm_no_max":             [1.0, 0.55, 0.45],
    }

    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    print(f"  Testing {len(combos):,} configurations against {len(trades)} resolved trades...\n")

    rows = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        result = evaluate_config(trades, cfg, flat_bet)
        if result["taken"] >= 10:  # ignore configs that trade too rarely
            rows.append(result)

    df = pd.DataFrame(rows)
    df = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_results(df: pd.DataFrame, top_n: int = 20) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  REPLAY BACKTEST — TOP CONFIGURATIONS BY PnL")
    print("=" * W)
    print(f"  {'Rank':<5} {'Taken':<7} {'WinRate':<9} {'PnL':>9}  Configuration")
    print("  " + "-" * 68)

    for i, row in df.head(top_n).iterrows():
        cfg = row["cfg"]
        flags = []
        if cfg.get("require_structure"):      flags.append("struct_required")
        if cfg.get("block_ema_stack_opp"):    flags.append("ema_stack_gate")
        if cfg.get("block_stoch_opp"):        flags.append("stoch_gate")
        if cfg.get("invert_obi"):             flags.append("obi_inverted")
        if cfg.get("min_net_edge", 0) > 0:   flags.append(f"edge≥{cfg['min_net_edge']:.2f}")
        if cfg.get("min_confirmation_score", 0) > 0: flags.append(f"conf≥{cfg['min_confirmation_score']}")
        if cfg.get("min_no_score", 0) > 0:   flags.append(f"no_score≥{cfg['min_no_score']}")
        if cfg.get("ema_dir_gate"):           flags.append("ema_dir")
        if cfg.get("pm_yes_min", 0) > 0:     flags.append(f"pm_yes≥{cfg['pm_yes_min']:.2f}")
        if cfg.get("pm_no_max", 1) < 1:      flags.append(f"pm_no≤{cfg['pm_no_max']:.2f}")
        if not flags:                          flags.append("baseline")

        print(f"  {i+1:<5} {row['taken']:<7} {row['win_rate']:.1%}    "
              f"${row['total_pnl']:>+8,.2f}  {', '.join(flags)}")

    print()

    # Baseline (no gates)
    baseline_cfg = {k: ([False, 0, 0.0][0] if isinstance(v[0], bool) else 0)
                    for k, v in {"min_net_edge": [0.0], "require_structure": [False],
                                 "block_ema_stack_opp": [False], "block_stoch_opp": [False],
                                 "invert_obi": [False], "min_confirmation_score": [0],
                                 "min_no_score": [0]}.items()}
    # Find baseline in results
    baseline_rows = df[
        (df["cfg"].apply(lambda c: c.get("min_net_edge", 0) == 0.0)) &
        (df["cfg"].apply(lambda c: not c.get("require_structure"))) &
        (df["cfg"].apply(lambda c: not c.get("block_ema_stack_opp"))) &
        (df["cfg"].apply(lambda c: not c.get("block_stoch_opp"))) &
        (df["cfg"].apply(lambda c: not c.get("invert_obi"))) &
        (df["cfg"].apply(lambda c: c.get("min_confirmation_score", 0) == 0)) &
        (df["cfg"].apply(lambda c: c.get("min_no_score", 0) == 0)) &
        (df["cfg"].apply(lambda c: not c.get("ema_dir_gate"))) &
        (df["cfg"].apply(lambda c: c.get("pm_yes_min", 0.0) == 0.0)) &
        (df["cfg"].apply(lambda c: c.get("pm_no_max", 1.0) == 1.0))
    ]
    if not baseline_rows.empty:
        b = baseline_rows.iloc[0]
        print(f"  Baseline (no gates): {b['taken']} trades  "
              f"win={b['win_rate']:.1%}  PnL=${b['total_pnl']:+,.2f}")

    print("\n" + "=" * W + "\n")


# ---------------------------------------------------------------------------
# Indicator win rate breakdown
# ---------------------------------------------------------------------------

def print_indicator_breakdown(trades: pd.DataFrame) -> None:
    print("\n── INDICATOR WIN RATE BREAKDOWN ─────────────────────────────────")

    trades = trades.copy()
    trades["win"] = (
        ((trades["side"] == "yes") & (trades["resolved_yes"] == True)) |
        ((trades["side"] == "no")  & (trades["resolved_yes"] == False))
    )

    indicators = [
        ("structure_bias",        [-1, 0, 1]),
        ("stoch_bias",            [-1, 0, 1]),
        ("ema_stack_bias",        [-1, 0, 1]),
        ("ema_alignment",         ["bearish", "neutral", "bullish"]),
        ("obi_score",             [-1, 0, 1]),
        ("funding_bias",          [-1, 0, 1]),
        ("vwap_signal",           [-1, 0, 1]),
        ("confirmation_score",    [0, 1, 2, 3]),
        ("no_score",              [0, 1, 2, 3]),
    ]

    for col, values in indicators:
        if col not in trades.columns:
            continue
        print(f"\n  {col}:")
        for v in values:
            subset = trades[trades[col].astype(str).str.strip() == str(v)]
            if len(subset) < 3:
                continue
            wr = subset["win"].mean()
            print(f"    {str(v):>10}  n={len(subset):>4}  win={wr:.1%}  "
                  f"{'▓' * int(wr * 20)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay backtest against paper trade CSV")
    parser.add_argument("--csv", default="results/paper_trades.csv",
                        help="Path to paper trades CSV (default: results/paper_trades.csv)")
    parser.add_argument("--flat-bet", type=float, default=100.0,
                        help="Fixed bet size per trade in USD (default: 100)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to display (default: 20)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = Path(__file__).parent / args.csv
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    trades = df[df["decision"] == "trade"].copy()
    resolved = trades[
        trades["resolved_yes"].notna() &
        (trades["resolved_yes"].astype(str).str.strip().isin(["True", "False", "1", "0"]))
    ].copy()
    resolved["resolved_yes"] = resolved["resolved_yes"].astype(str).map(
        {"True": True, "False": False, "1": True, "0": False}
    )

    print(f"\n  CSV: {csv_path.name}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Trade rows: {len(trades):,}  |  Resolved: {len(resolved):,}")
    if len(resolved) < 10:
        print("  Not enough resolved trades for meaningful analysis (need ≥10).")
        sys.exit(0)

    overall_win = (
        ((resolved["side"] == "yes") & (resolved["resolved_yes"] == True)) |
        ((resolved["side"] == "no")  & (resolved["resolved_yes"] == False))
    ).mean()
    print(f"  Overall win rate: {overall_win:.1%}  |  flat-bet: ${args.flat_bet:,.0f}\n")

    print_indicator_breakdown(resolved)

    results = grid_search(resolved, flat_bet=args.flat_bet)
    print_results(results, top_n=args.top)


if __name__ == "__main__":
    main()
