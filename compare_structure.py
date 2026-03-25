"""
Side-by-side comparison: 4-hour structure model vs 15-minute structure model.

Both runs use identical settings:
  - p_market = 0.33 (real observed Kalshi mid-price)
  - flat $100 bet per trade (no compounding)
  - full dataset (Jan 2024 – present)

The only difference is the timeframe used for market structure detection.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import market_structure as ms
from evaluate_point import load_data
from backtest import run_backtest

P_MARKET   = 0.33
FLAT_BET   = 100.0
BANKROLL   = 10_000.0
OFFSET     = 0.005

print("Loading data...")
df_1m, df_1h, df_4h = load_data()
print()

# ---------------------------------------------------------------------------
# Helper: extract summary stats from a results DataFrame
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, label: str) -> dict:
    trades     = df[df["decision"] == "trade"]
    yes_trades = trades[trades["side"] == "yes"]
    no_trades  = trades[trades["side"] == "no"]

    trades = trades.copy()
    trades["win"] = (
        ((trades["side"] == "yes") &  trades["resolved_yes"]) |
        ((trades["side"] == "no")  & ~trades["resolved_yes"])
    )

    win_rate     = trades["win"].mean()      if len(trades) > 0 else 0.0
    yes_win_rate = yes_trades["resolved_yes"].mean() if len(yes_trades) > 0 else float("nan")
    no_win_rate  = (~no_trades["resolved_yes"]).mean() if len(no_trades) > 0 else float("nan")

    final_broll  = df["bankroll_after"].iloc[-1] if not df.empty else BANKROLL
    total_pnl    = trades["pnl"].sum()       if len(trades) > 0 else 0.0

    wins   = trades[trades["win"]]
    losses = trades[~trades["win"]]
    avg_win  = wins["pnl"].mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0

    bull = (df["structure_bias"] ==  1).sum()
    bear = (df["structure_bias"] == -1).sum()
    neut = (df["structure_bias"] ==  0).sum()

    return {
        "label":          label,
        "decisions":      len(df),
        "trades":         len(trades),
        "trade_pct":      len(trades) / len(df) * 100 if len(df) > 0 else 0,
        "yes_trades":     len(yes_trades),
        "no_trades":      len(no_trades),
        "win_rate":       win_rate * 100,
        "yes_win_rate":   yes_win_rate * 100,
        "no_win_rate":    no_win_rate * 100,
        "total_pnl":      total_pnl,
        "final_bankroll": final_broll,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "bias_bull_pct":  bull / len(df) * 100 if len(df) > 0 else 0,
        "bias_bear_pct":  bear / len(df) * 100 if len(df) > 0 else 0,
        "bias_neut_pct":  neut / len(df) * 100 if len(df) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Run 1: 4-hour structure  (PIVOT_LOOKBACK=3, MIN_CANDLES=90)
# ---------------------------------------------------------------------------

print("=" * 62)
print("  RUN 1 — 4-HOUR STRUCTURE  (pivot lookback=3, min=90)")
print("=" * 62)

_orig_lookback  = ms.PIVOT_LOOKBACK
_orig_min       = ms.MIN_CANDLES

ms.PIVOT_LOOKBACK = 3
ms.MIN_CANDLES    = 90

df_r1 = run_backtest(
    df_1m=df_1m, df_1h=df_1h,
    structure_df=df_4h,   # pass real 4h bars as the structure input
    strike_offset=OFFSET,
    p_market=P_MARKET,
    bankroll=BANKROLL,
    flat_bet=FLAT_BET,
)
df_r1.to_csv("results/backtest_4h_flat.csv", index=False)
print("  Saved → results/backtest_4h_flat.csv")

stats_4h = summarise(df_r1, "4h structure")

# Restore 15m params
ms.PIVOT_LOOKBACK = _orig_lookback
ms.MIN_CANDLES    = _orig_min


# ---------------------------------------------------------------------------
# Run 2: 15-minute structure  (PIVOT_LOOKBACK=2, MIN_CANDLES=120)
# ---------------------------------------------------------------------------

print()
print("=" * 62)
print("  RUN 2 — 15-MINUTE STRUCTURE  (pivot lookback=2, min=120)")
print("=" * 62)

ms.PIVOT_LOOKBACK = 2
ms.MIN_CANDLES    = 120

df_r2 = run_backtest(
    df_1m=df_1m, df_1h=df_1h,
    strike_offset=OFFSET,
    p_market=P_MARKET,
    bankroll=BANKROLL,
    flat_bet=FLAT_BET,
)
df_r2.to_csv("results/backtest_15m_lb2_flat.csv", index=False)
print("  Saved → results/backtest_15m_lb2_flat.csv")

stats_15m_lb2 = summarise(df_r2, "15m lb=2")

ms.PIVOT_LOOKBACK = _orig_lookback
ms.MIN_CANDLES    = _orig_min


# ---------------------------------------------------------------------------
# Run 3: 15-minute structure  (PIVOT_LOOKBACK=3, MIN_CANDLES=120)
# ---------------------------------------------------------------------------

print()
print("=" * 62)
print("  RUN 3 — 15-MINUTE STRUCTURE  (pivot lookback=3, min=120)")
print("=" * 62)

ms.PIVOT_LOOKBACK = 3
ms.MIN_CANDLES    = 120

df_r3 = run_backtest(
    df_1m=df_1m, df_1h=df_1h,
    strike_offset=OFFSET,
    p_market=P_MARKET,
    bankroll=BANKROLL,
    flat_bet=FLAT_BET,
)
df_r3.to_csv("results/backtest_15m_lb3_flat.csv", index=False)
print("  Saved → results/backtest_15m_lb3_flat.csv")

stats_15m_lb3 = summarise(df_r3, "15m lb=3")

ms.PIVOT_LOOKBACK = _orig_lookback
ms.MIN_CANDLES    = _orig_min


# ---------------------------------------------------------------------------
# Side-by-side comparison table
# ---------------------------------------------------------------------------

def pct(v):   return f"{v:.1f}%"
def usd(v):   return f"${v:+,.2f}"
def num(v):   return f"{v:,}"
def fpct(v):  return f"{v:.1f}%" if not (v != v) else "n/a"  # handles NaN

W = 84
s4, s2, s3 = stats_4h, stats_15m_lb2, stats_15m_lb3

print()
print("=" * W)
print("  COMPARISON: 4H (lb=3)  |  15M lb=2  |  15M lb=3")
print("  Settings: p_market=0.33  |  flat $100 bet  |  full dataset")
print("=" * W)

rows = [
    ("Decision points",   num(s4["decisions"]),                         num(s2["decisions"]),                         num(s3["decisions"])),
    ("Trades taken",      f"{num(s4['trades'])} ({pct(s4['trade_pct'])})", f"{num(s2['trades'])} ({pct(s2['trade_pct'])})", f"{num(s3['trades'])} ({pct(s3['trade_pct'])})"),
    ("  — YES trades",    num(s4["yes_trades"]),                        num(s2["yes_trades"]),                        num(s3["yes_trades"])),
    ("  — NO  trades",    num(s4["no_trades"]),                         num(s2["no_trades"]),                         num(s3["no_trades"])),
    ("Win rate (all)",    pct(s4["win_rate"]),                           pct(s2["win_rate"]),                           pct(s3["win_rate"])),
    ("  YES win rate",    fpct(s4["yes_win_rate"]),                      fpct(s2["yes_win_rate"]),                      fpct(s3["yes_win_rate"])),
    ("  NO  win rate",    fpct(s4["no_win_rate"]),                       fpct(s2["no_win_rate"]),                       fpct(s3["no_win_rate"])),
    ("Total net P&L",     usd(s4["total_pnl"]),                          usd(s2["total_pnl"]),                          usd(s3["total_pnl"])),
    ("Final bankroll",    usd(s4["final_bankroll"]),                      usd(s2["final_bankroll"]),                      usd(s3["final_bankroll"])),
    ("Avg winning trade", usd(s4["avg_win"]),                             usd(s2["avg_win"]),                             usd(s3["avg_win"])),
    ("Avg losing trade",  usd(s4["avg_loss"]),                            usd(s2["avg_loss"]),                            usd(s3["avg_loss"])),
    ("Structure bullish", pct(s4["bias_bull_pct"]),                       pct(s2["bias_bull_pct"]),                       pct(s3["bias_bull_pct"])),
    ("Structure bearish", pct(s4["bias_bear_pct"]),                       pct(s2["bias_bear_pct"]),                       pct(s3["bias_bear_pct"])),
    ("Structure neutral", pct(s4["bias_neut_pct"]),                       pct(s2["bias_neut_pct"]),                       pct(s3["bias_neut_pct"])),
]

header = f"  {'Metric':<28} {'4h (lb=3)':>14} {'15m (lb=2)':>14} {'15m (lb=3)':>14}"
print(header)
print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14}")
for label, v4, v2, v3 in rows:
    print(f"  {label:<28} {v4:>14} {v2:>14} {v3:>14}")

print("=" * W)
