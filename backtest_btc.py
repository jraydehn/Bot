"""
BTC Model Backtester — 2024-01-01 to present.

Walks through every hourly bar in historical OHLCV data and simulates trade
decisions using the current model logic (Gate 0, Gate EMA-Dir, Gate PM, Gate 3,
Gate R:R). Compares model predictions against actual BTC price outcomes.

Key outputs:
  1. Model calibration: does p_model=X mean X% actual win rate?
  2. EMA regime win rates: validates Gate EMA-Dir
  3. Gate PM threshold analysis: optimises p_market cutoffs
  4. Offset sweep: which strike distance has the best edge?
  5. Full P&L simulation with half-Kelly sizing

Usage:
    python3 backtest_btc.py
    python3 backtest_btc.py --start 2025-01-01    # shorter window
    python3 backtest_btc.py --offset 0.005        # single offset
    python3 backtest_btc.py --no-gates            # raw model, no filters
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probability_engine import estimate_probability
from pricing_comparison import (
    evaluate_edge, kalshi_fee,
    DEFAULT_SLIPPAGE, DEFAULT_SPREAD,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Strike offsets to sweep ────────────────────────────────────────────────
OFFSETS = [0.002, 0.003, 0.005, 0.008, 0.010, 0.015]

# ── Realistic deterministic p_market midpoints (from simulate_p_market ranges)
#    Used for gate evaluation and PnL simulation.
#    Keys are offset buckets; values are (p_yes_side, p_no_side).
PMARKET_MAP = {
    0.002: (0.30,  0.35),
    0.003: (0.30,  0.35),
    0.005: (0.225, 0.275),
    0.008: (0.125, 0.18),
    0.010: (0.125, 0.18),
    0.015: (0.018, 0.030),
}

# Gate constants (must match decision.py)
P_MODEL_MIN_BTC = 0.04
P_MODEL_MAX_BTC = 0.96
P_MARKET_YES_MIN = 0.55   # Gate PM: YES only when p_market >= this
P_MARKET_NO_MAX  = 0.45   # Gate PM: NO only when p_market <= this
MIN_NET_EDGE     = 0.03   # Gate 3: minimum net edge
RR_MIN           = 0.33   # Gate R:R lower bound
RR_MAX_NO        = 4.0    # Gate R:R upper bound for NO

TAU = 60  # minutes to expiry (hourly contracts)

EMA_FAST = 20
EMA_SLOW = 50
EMA_CONFIRM_BARS = 3


# ── Helpers ────────────────────────────────────────────────────────────────

def nearest_offset_bucket(offset: float) -> float:
    """Round to nearest key in PMARKET_MAP."""
    keys = sorted(PMARKET_MAP.keys())
    return min(keys, key=lambda k: abs(k - offset))


def compute_ema_alignment(close: pd.Series) -> pd.Series:
    """
    Vectorised EMA alignment: returns Series of 'bullish'/'bearish'/'neutral'
    at each bar using the same rules as confirmation_indicators.py.
    """
    ema_20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_50 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    alignments = []
    closes = close.values
    e20    = ema_20.values
    e50    = ema_50.values
    n      = len(closes)

    for i in range(n):
        if i < EMA_CONFIRM_BARS - 1:
            alignments.append("neutral")
            continue
        sl = slice(i - EMA_CONFIRM_BARS + 1, i + 1)
        lc  = closes[sl]
        l20 = e20[sl]
        l50 = e50[sl]

        golden_cross = bool(np.all(l20 > l50))
        death_cross  = bool(np.all(l20 < l50))
        price_above  = bool(np.all(lc > l20))
        price_below  = bool(np.all(lc < l50))

        if golden_cross and price_above:
            alignments.append("bullish")
        elif death_cross or price_below:
            alignments.append("bearish")
        else:
            alignments.append("neutral")

    return pd.Series(alignments, index=close.index)


def apply_gates_btc(
    p_model: float,
    p_market: float,
    ema_alignment: str,
    side: str,
) -> tuple[bool, str]:
    """
    Apply current BTC gates in order. Returns (passed: bool, blocker: str).
    blocker is the name of the gate that failed, or 'all' if all passed.
    """
    # Gate 0
    if not (P_MODEL_MIN_BTC <= p_model <= P_MODEL_MAX_BTC):
        return False, "Gate0"

    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD

    # Gate EMA-Dir (BTC only)
    if ema_alignment == "bullish" and side == "yes":
        return False, "GateEMADir"
    if ema_alignment == "bearish" and side == "no":
        return False, "GateEMADir"

    # Gate PM (BTC only)
    if side == "yes" and p_market < P_MARKET_YES_MIN:
        return False, "GatePM"
    if side == "no" and p_market > P_MARKET_NO_MAX:
        return False, "GatePM"

    # Gate 3 (min net edge)
    if net_edge < MIN_NET_EDGE:
        return False, "Gate3"

    # Gate R:R
    rr = p_market / (1 - p_market) if side == "yes" else (1 - p_market) / p_market
    if rr < RR_MIN or (side == "no" and rr > RR_MAX_NO):
        return False, "GateRR"

    return True, "all"


def simulate_pnl(side: str, p_market: float, outcome: int, bet_amount: float) -> float:
    """
    Compute PnL for a single trade.
    outcome: 1 = YES resolved, 0 = NO resolved
    """
    yes_wins = (outcome == 1)
    fee = kalshi_fee(p_market) * bet_amount

    if side == "yes":
        if yes_wins:
            return bet_amount * (1 - p_market) / p_market - fee
        else:
            return -bet_amount - fee
    else:
        if not yes_wins:
            return bet_amount * p_market / (1 - p_market) - fee
        else:
            return -bet_amount - fee


# ── Main backtest loop ─────────────────────────────────────────────────────

def run_backtest(
    start_date: str = "2024-01-01",
    offsets: list = None,
    apply_gates: bool = True,
    bankroll: float = 10_000,
) -> pd.DataFrame:
    """
    Run the backtest. Returns a DataFrame of all trade opportunities.
    """
    if offsets is None:
        offsets = OFFSETS

    # ── Load data ────────────────────────────────────────────────────────
    def latest_parquet(pattern):
        matches = sorted(
            DATA_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        matches = [m for m in matches if ".ckpt." not in m.name]
        return matches[-1] if matches else None

    p_1m = latest_parquet("*BTCUSDT_1m_2024-01-01*.parquet")
    p_1h = latest_parquet("*BTCUSDT_1h_2024-01-01*.parquet")
    if not p_1m or not p_1h:
        raise FileNotFoundError("BTC 1m / 1h parquet files not found. Run fetch_data.py first.")

    print(f"  Loading 1m  : {p_1m.name}")
    print(f"  Loading 1h  : {p_1h.name}")

    df_1m = pd.read_parquet(p_1m)
    df_1h = pd.read_parquet(p_1h)
    for df in [df_1m, df_1h]:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df_1m.columns = df_1m.columns.str.lower()
    df_1h.columns = df_1h.columns.str.lower()

    # Filter to start date
    start_ts = pd.Timestamp(start_date, tz="UTC")
    df_1h = df_1h[df_1h.index >= start_ts]
    df_1m = df_1m[df_1m.index >= start_ts - pd.Timedelta(hours=4)]  # keep some warmup

    print(f"  1h bars in window : {len(df_1h):,}  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
    print(f"  1m bars in window : {len(df_1m):,}")

    # ── Pre-compute EMA alignment on full 1h series ───────────────────────
    print("  Pre-computing EMA alignment...")
    full_1h = pd.read_parquet(p_1h)
    if full_1h.index.tz is None:
        full_1h.index = full_1h.index.tz_localize("UTC")
    full_1h.columns = full_1h.columns.str.lower()
    ema_series = compute_ema_alignment(full_1h["close"])
    ema_map = ema_series.to_dict()  # ts → 'bullish'/'bearish'/'neutral'

    # ── Pre-compute per-minute vol, then sample at each 1h bar ───────────
    print("  Pre-computing rolling vol_60m...")
    log_ret_1m = np.log(df_1m["close"] / df_1m["close"].shift(1)).dropna()
    vol_1m = log_ret_1m.rolling(60).std()  # per-minute vol, 60-bar window
    # Resample: take last 1m vol before each 1h bar close
    vol_1h = vol_1m.resample("1h", closed="right", label="right").last()
    vol_map = vol_1h.to_dict()  # 1h timestamp → vol_60m

    # ── Walk through hourly bars ─────────────────────────────────────────
    print(f"  Running hourly simulation ({len(df_1h)-1:,} bars × {len(offsets)} offsets × 2 sides)...")

    records = []
    bars = df_1h.reset_index()  # columns: open_time, open, high, low, close, volume
    n = len(bars) - 1  # last bar has no "next close"

    for i in range(200, n):  # 200-bar warmup for EMA stability
        row      = bars.iloc[i]
        next_row = bars.iloc[i + 1]
        ts       = row["open_time"]

        spot       = float(row["close"])
        next_close = float(next_row["close"])

        ema_alignment = ema_map.get(ts, "neutral")
        vol_60m = vol_map.get(ts, None)

        if vol_60m is None or np.isnan(vol_60m) or vol_60m <= 0:
            continue

        for offset in offsets:
            strike = spot * (1 + offset)

            # p_model: probability BTC ends above strike in 60 minutes
            try:
                prob = estimate_probability(spot, strike, TAU, vol_60m)
                p_model = prob.p_yes
            except Exception:
                continue

            # Actual outcome: did BTC close above strike one hour later?
            outcome = int(next_close > strike)  # 1 = YES resolved

            # Get simulated p_market for each side
            bucket = nearest_offset_bucket(offset)
            pm_yes, pm_no = PMARKET_MAP[bucket]

            for side, p_market in [("yes", pm_yes), ("no", pm_no)]:
                fee = kalshi_fee(p_market)
                raw_edge = p_model - p_market if side == "yes" else p_market - p_model
                net_edge = raw_edge - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD

                # Apply gates
                gate_passed, blocker = apply_gates_btc(p_model, p_market, ema_alignment, side)

                # Win condition for this side
                win = outcome if side == "yes" else (1 - outcome)

                # Kelly sizing (half-Kelly, capped at 5% bankroll)
                if gate_passed and apply_gates:
                    if side == "yes":
                        p_win = p_model
                        b = (1 - p_market) / p_market
                    else:
                        p_win = 1 - p_model  # P(BTC does NOT reach strike)
                        b = p_market / (1 - p_market)
                    q_win = 1 - p_win
                    kelly = (b * p_win - q_win) / b if b > 0 else 0
                    kelly = max(0.0, kelly * 0.5)  # half-Kelly
                    kelly = min(kelly, 0.05)        # cap at 5%
                    bet_amount = kelly * bankroll
                else:
                    bet_amount = 0.0

                records.append({
                    "ts":           ts,
                    "spot":         spot,
                    "offset":       offset,
                    "strike":       strike,
                    "next_close":   next_close,
                    "ema_alignment": ema_alignment,
                    "vol_60m":      vol_60m,
                    "p_model":      p_model,
                    "p_market":     p_market,
                    "side":         side,
                    "raw_edge":     raw_edge,
                    "net_edge":     net_edge,
                    "gate_passed":  gate_passed,
                    "blocker":      blocker,
                    "outcome":      outcome,
                    "win":          win,
                    "bet_amount":   bet_amount,
                })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} rows ({len(df[df['gate_passed']]):,} gate-passing trades)")
    return df


# ── Analysis functions ─────────────────────────────────────────────────────

def print_separator(title: str = "", width: int = 64):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n── {title} {'─' * pad}")
    else:
        print("─" * width)


def calibration_table(df: pd.DataFrame):
    """Bin p_model and show actual win rates — reveals model bias."""
    print_separator("MODEL CALIBRATION (p_model vs actual YES win rate)")
    bins   = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    labels = [f"{int(a*100)}-{int(b*100)}%" for a, b in zip(bins, bins[1:])]

    yes_df = df[df["side"] == "yes"].copy()
    yes_df["p_bin"] = pd.cut(yes_df["p_model"], bins=bins, labels=labels, right=False)
    tbl = yes_df.groupby("p_bin", observed=True).agg(
        n=("outcome", "count"),
        actual_win_pct=("outcome", "mean"),
        p_model_mean=("p_model", "mean"),
    ).dropna()
    tbl["bias"] = tbl["actual_win_pct"] - tbl["p_model_mean"]

    print(f"  {'p_model bin':<12}  {'n':>6}  {'model':>7}  {'actual':>7}  {'bias':>7}")
    print("  " + "-" * 48)
    for lbl, row in tbl.iterrows():
        b = row["bias"]
        flag = "▲" if b > 0.05 else ("▼" if b < -0.05 else " ")
        print(f"  {lbl:<12}  {int(row['n']):>6}  {row['p_model_mean']:>6.1%}  {row['actual_win_pct']:>6.1%}  {b:>+6.1%} {flag}")


def ema_regime_table(df: pd.DataFrame):
    """Win rate by EMA alignment × side — validates Gate EMA-Dir."""
    print_separator("EMA REGIME WIN RATES (Gate EMA-Dir validation)")

    # Use all trades at the primary offset (0.005) for a clean comparison
    sub = df[(df["offset"] == 0.005)].copy()

    tbl = sub.groupby(["ema_alignment", "side"]).agg(
        n=("win", "count"),
        win_pct=("win", "mean"),
    ).reset_index()

    print(f"  {'EMA':<10}  {'side':<5}  {'n':>6}  {'win%':>7}  gate_ema_dir")
    print("  " + "-" * 48)
    for _, row in tbl.iterrows():
        ema = row["ema_alignment"]
        s   = row["side"]
        # Gate EMA-Dir blocks bullish+YES and bearish+NO
        blocked = (ema == "bullish" and s == "yes") or (ema == "bearish" and s == "no")
        tag = "BLOCKED" if blocked else "allowed"
        print(f"  {ema:<10}  {s:<5}  {int(row['n']):>6}  {row['win_pct']:>6.1%}  {tag}")


def pmarket_threshold_table(df: pd.DataFrame):
    """Win rate by p_market bucket — validates Gate PM thresholds."""
    print_separator("GATE PM THRESHOLD ANALYSIS (p_market vs win rate)")

    for side in ["yes", "no"]:
        sub = df[(df["side"] == side) & (df["offset"] == 0.005)].copy()
        bins   = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0]
        labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(bins, bins[1:])]
        sub["pm_bin"] = pd.cut(sub["p_market"], bins=bins, labels=labels, right=False)

        tbl = sub.groupby("pm_bin", observed=True).agg(
            n=("win", "count"),
            win_pct=("win", "mean"),
        ).dropna()

        gate_threshold = P_MARKET_YES_MIN if side == "yes" else P_MARKET_NO_MAX
        print(f"\n  {side.upper()} — Gate PM threshold = {gate_threshold}")
        print(f"  {'p_market':<12}  {'n':>6}  {'win%':>7}  gate_pm")
        print("  " + "-" * 40)
        for lbl, row in tbl.iterrows():
            lo = float(str(lbl).split("-")[0])
            if side == "yes":
                tag = "PASS" if lo >= gate_threshold else "BLOCK"
            else:
                tag = "PASS" if lo + 0.1 <= gate_threshold else "BLOCK"
            print(f"  {lbl:<12}  {int(row['n']):>6}  {row['win_pct']:>6.1%}  {tag}")


def gate_funnel_table(df: pd.DataFrame):
    """How many trades each gate passes/blocks and the win rate at each stage."""
    print_separator("GATE FUNNEL (volume and win rate at each stage)")

    sub = df[df["offset"] == 0.005].copy()
    total = len(sub) // 2  # YES+NO pairs counted once each

    stages = [
        ("All signals",    sub),
        ("Gate0",         sub[~((sub["p_model"] < P_MODEL_MIN_BTC) | (sub["p_model"] > P_MODEL_MAX_BTC))]),
        ("GateEMADir",    sub[~(((sub["ema_alignment"] == "bullish") & (sub["side"] == "yes")) |
                               ((sub["ema_alignment"] == "bearish") & (sub["side"] == "no")))]),
        ("GatePM",        sub[((sub["side"] == "yes") & (sub["p_market"] >= P_MARKET_YES_MIN)) |
                              ((sub["side"] == "no")  & (sub["p_market"] <= P_MARKET_NO_MAX))]),
        ("Gate3",         sub[sub["net_edge"] >= MIN_NET_EDGE]),
        ("All gates",     sub[sub["gate_passed"]]),
    ]

    print(f"  {'Stage':<15}  {'n':>7}  {'win%':>7}  {'pass%':>7}")
    print("  " + "-" * 44)
    for name, stage_df in stages:
        n = len(stage_df)
        win = stage_df["win"].mean() if n > 0 else 0
        pct = n / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {name:<15}  {n:>7,}  {win:>6.1%}  {pct:>6.1f}%")


def offset_sweep_table(df: pd.DataFrame, apply_gates_flag: bool):
    """Win rate and edge by offset — which strike distance is best."""
    print_separator("OFFSET SWEEP (win rate and EV by strike distance)")

    mask = df["gate_passed"] if apply_gates_flag else pd.Series(True, index=df.index)
    sub  = df[mask].copy() if apply_gates_flag else df.copy()

    tbl = sub.groupby(["offset", "side"]).agg(
        n=("win", "count"),
        win_pct=("win", "mean"),
        net_edge_mean=("net_edge", "mean"),
    ).reset_index()

    print(f"  {'offset':>7}  {'side':<5}  {'n':>6}  {'win%':>7}  {'net_edge':>9}")
    print("  " + "-" * 46)
    for _, row in tbl.iterrows():
        print(f"  {row['offset']:>6.1%}  {row['side']:<5}  {int(row['n']):>6}  "
              f"{row['win_pct']:>6.1%}  {row['net_edge_mean']:>+8.3f}")


def pnl_summary(df: pd.DataFrame, bankroll: float):
    """Simulate P&L for gate-passing trades with half-Kelly sizing."""
    print_separator("P&L SIMULATION (gate-passing trades, half-Kelly)")

    trades = df[df["gate_passed"]].copy()
    if trades.empty:
        print("  No gate-passing trades.")
        return

    # PnL per trade
    pnls = []
    for _, row in trades.iterrows():
        pnl = simulate_pnl(row["side"], row["p_market"], row["outcome"], row["bet_amount"])
        pnls.append(pnl)
    trades["pnl"] = pnls
    trades["cumulative_pnl"] = trades["pnl"].cumsum()

    total_pnl  = trades["pnl"].sum()
    n_trades   = len(trades)
    win_rate   = trades["win"].mean()
    avg_bet    = trades["bet_amount"].mean()
    avg_pnl    = trades["pnl"].mean()
    sharpe     = trades["pnl"].mean() / trades["pnl"].std() * np.sqrt(8760) if trades["pnl"].std() > 0 else 0

    print(f"  Total trades     : {n_trades:,}")
    print(f"  Win rate         : {win_rate:.1%}")
    print(f"  Avg bet          : ${avg_bet:,.2f}")
    print(f"  Avg PnL/trade    : ${avg_pnl:+,.2f}")
    print(f"  Total PnL        : ${total_pnl:+,.2f}  ({total_pnl/bankroll:+.1%} of bankroll)")
    print(f"  Annualised Sharpe: {sharpe:.2f}")

    # By side
    for side in ["yes", "no"]:
        s = trades[trades["side"] == side]
        if len(s) == 0:
            continue
        print(f"\n  {side.upper()}: n={len(s):,}  win={s['win'].mean():.1%}  "
              f"PnL=${s['pnl'].sum():+,.2f}  avg_edge={s['net_edge'].mean():+.3f}")

    # Save trade log
    out = RESULTS_DIR / "backtest_trades.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    trades[["ts", "ema_alignment", "side", "offset", "p_model", "p_market",
            "net_edge", "win", "bet_amount", "pnl", "cumulative_pnl"]].to_csv(out, index=False)
    print(f"\n  Trade log saved → {out.name}")


def gate_contribution_table(df: pd.DataFrame):
    """For each gate, show what win rate the blocked trades would have had."""
    print_separator("GATE CONTRIBUTION (what win% would blocked trades have had?)")

    sub = df[df["offset"] == 0.005].copy()

    # Build incremental blocks
    g0_blocked  = sub[(sub["p_model"] < P_MODEL_MIN_BTC) | (sub["p_model"] > P_MODEL_MAX_BTC)]
    ema_blocked = sub[((sub["ema_alignment"] == "bullish") & (sub["side"] == "yes")) |
                      ((sub["ema_alignment"] == "bearish") & (sub["side"] == "no"))]
    pm_blocked  = sub[((sub["side"] == "yes") & (sub["p_market"] < P_MARKET_YES_MIN)) |
                      ((sub["side"] == "no")  & (sub["p_market"] > P_MARKET_NO_MAX))]
    g3_blocked  = sub[(sub["net_edge"] < MIN_NET_EDGE)]
    rr_blocked  = sub[sub["blocker"] == "GateRR"]
    passed      = sub[sub["gate_passed"]]

    rows = [
        ("Gate 0",     g0_blocked),
        ("Gate EMA-Dir", ema_blocked),
        ("Gate PM",    pm_blocked),
        ("Gate 3",     g3_blocked),
        ("Gate R:R",   rr_blocked),
        ("→ Passed",   passed),
    ]

    print(f"  {'Gate':<15}  {'n':>7}  {'win% if traded':>14}")
    print("  " + "-" * 44)
    for name, grp in rows:
        if len(grp) == 0:
            print(f"  {name:<15}  {'0':>7}  {'—':>14}")
        else:
            print(f"  {name:<15}  {len(grp):>7,}  {grp['win'].mean():>13.1%}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC model backtester")
    parser.add_argument("--start",     default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--offset",    type=float, default=None,
                        help="Single offset to test (e.g. 0.005). Default: sweep all.")
    parser.add_argument("--no-gates",  action="store_true", help="Disable all gates (raw model only)")
    parser.add_argument("--bankroll",  type=float, default=10_000)
    args = parser.parse_args()

    offsets = [args.offset] if args.offset else OFFSETS
    apply   = not args.no_gates

    print("=" * 64)
    print("  BTC MODEL BACKTEST")
    print("=" * 64)
    print(f"  Start date  : {args.start}")
    print(f"  Offsets     : {[f'{o:.1%}' for o in offsets]}")
    print(f"  Gates       : {'ON' if apply else 'OFF (raw model)'}")
    print(f"  Bankroll    : ${args.bankroll:,.0f}")
    print()

    df = run_backtest(start_date=args.start, offsets=offsets,
                      apply_gates=apply, bankroll=args.bankroll)

    if df.empty:
        print("  No data generated.")
        return

    calibration_table(df)
    ema_regime_table(df)
    pmarket_threshold_table(df)
    gate_contribution_table(df)
    gate_funnel_table(df)
    offset_sweep_table(df, apply)
    pnl_summary(df, args.bankroll)

    # Save full dataset
    out = RESULTS_DIR / "backtest_full.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Full dataset saved → {out.name}")
    print()


if __name__ == "__main__":
    main()
