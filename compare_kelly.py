"""
Three-way comparison: Full Kelly vs Quarter Kelly vs Split Kelly.
"""

import pandas as pd
import numpy as np

FILES = {
    "Full Kelly":    "results/backtest_pm15.csv",
    "Quarter Kelly": "results/backtest_qkelly.csv",
    "Split Kelly":   "results/backtest_splitkelly.csv",
}

def load(path):
    df = pd.read_csv(path, parse_dates=["decision_time"])
    return df

def max_drawdown(series):
    """Max peak-to-trough drawdown of a bankroll series."""
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()

def stats(df, label):
    trades = df[df["decision"] == "trade"].copy()
    if trades.empty:
        return {}

    trades["win"] = (
        ((trades["side"] == "yes") & (trades["resolved_yes"])) |
        ((trades["side"] == "no")  & (~trades["resolved_yes"]))
    )
    yes_t = trades[trades["side"] == "yes"]
    no_t  = trades[trades["side"] == "no"]

    starting = df["bankroll"].iloc[0]
    final    = df["bankroll_after"].iloc[-1]
    ret_pct  = (final / starting - 1) * 100

    bankroll_series = pd.concat([
        pd.Series([starting]),
        df["bankroll_after"]
    ]).reset_index(drop=True)
    mdd = max_drawdown(bankroll_series) * 100

    trades["month"] = trades["decision_time"].dt.to_period("M")
    monthly_pnl = trades.groupby("month")["pnl"].sum()
    monthly_std = monthly_pnl.std()

    return {
        "label":        label,
        "trades":       len(trades),
        "yes_trades":   len(yes_t),
        "no_trades":    len(no_t),
        "return_pct":   ret_pct,
        "final_bank":   final,
        "max_dd_pct":   mdd,
        "monthly_std":  monthly_std,
        "win_rate":     trades["win"].mean() * 100,
        "yes_win_rate": yes_t["win"].mean() * 100 if not yes_t.empty else float("nan"),
        "no_win_rate":  no_t["win"].mean() * 100 if not no_t.empty else float("nan"),
        "avg_yes_bet":  yes_t["bet_amount"].mean() if not yes_t.empty else 0,
        "avg_no_bet":   no_t["bet_amount"].mean() if not no_t.empty else 0,
        "monthly_pnl":  monthly_pnl,
    }

results = {}
for label, path in FILES.items():
    df = load(path)
    results[label] = stats(df, label)

# --- Header ---
W = 72
print("\n" + "=" * W)
print("  KELLY STRATEGY COMPARISON")
print("=" * W)

col_w = 18
labels = list(results.keys())

def hdr():
    h = " " * 28
    for lb in labels:
        h += f"{lb:>{col_w}}"
    print(h)

def row(name, key, fmt=".2f", suffix=""):
    line = f"  {name:<26}"
    for lb in labels:
        val = results[lb].get(key, float("nan"))
        try:
            line += f"{format(val, fmt) + suffix:>{col_w}}"
        except (ValueError, TypeError):
            line += f"{'N/A':>{col_w}}"
    print(line)

print("\n── OVERVIEW " + "─" * (W - 12))
hdr()
row("Trades taken",        "trades",      "d")
row("  YES trades",        "yes_trades",  "d")
row("  NO  trades",        "no_trades",   "d")

print("\n── RETURNS " + "─" * (W - 11))
hdr()
row("Total return",        "return_pct",   "+.2f", "%")
row("Final bankroll ($)",  "final_bank",   ",.2f")
row("Max drawdown",        "max_dd_pct",   "+.2f", "%")
row("Monthly PnL std ($)", "monthly_std",  ",.2f")

print("\n── WIN RATES " + "─" * (W - 13))
hdr()
row("Overall win rate",    "win_rate",     ".2f", "%")
row("YES win rate",        "yes_win_rate", ".2f", "%")
row("NO  win rate",        "no_win_rate",  ".2f", "%")

print("\n── BET SIZING " + "─" * (W - 14))
hdr()
row("Avg YES bet ($)",     "avg_yes_bet",  ",.2f")
row("Avg NO  bet ($)",     "avg_no_bet",   ",.2f")

# --- Monthly breakdown for Split Kelly ---
print("\n── MONTHLY P&L  (Split Kelly) " + "─" * (W - 30))
sk_monthly = results["Split Kelly"].get("monthly_pnl", pd.Series(dtype=float))
for period, pnl in sk_monthly.items():
    bar = "█" * int(abs(pnl) / 50) if abs(pnl) > 50 else "▏"
    sign = "+" if pnl >= 0 else "-"
    print(f"  {period}  {sign}${abs(pnl):>7,.0f}  {bar}")

print("\n" + "=" * W + "\n")
