"""
analyze_markov_regime_trades.py

Fetches BTC-USD daily data, computes the observable Markov regime label
(Bull / Bear / Sideways from 20-day rolling return), then joins to
paper_trades.csv by trade date and reports WR + P&L split by regime.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not found — run:  pip install yfinance")

TRADES_CSV   = "results/paper_trades.csv"
WINDOW       = 20       # 20-day rolling return for regime label
THRESHOLD    = 0.02     # >+2% = Bull, <-2% = Bear, else Sideways
FLAT_BET     = 25.0     # $25/trade for P&L estimate
KALSHI_TAKE  = 0.10     # 10% rake on winnings

SEP  = "=" * 68
SEP2 = "-" * 52

# ── 1. Fetch BTC daily ───────────────────────────────────────────────────────
print("\nFetching BTC-USD daily data from Yahoo Finance...")
df_d = yf.download("BTC-USD", start="2025-01-01", end="2026-05-23",
                   progress=False, auto_adjust=True)
if isinstance(df_d.columns, pd.MultiIndex):
    df_d.columns = df_d.columns.get_level_values(0)
close_d = df_d["Close"].dropna()
print(f"  {len(close_d)} daily bars: {close_d.index.min().date()} → {close_d.index.max().date()}")

# ── 2. Compute daily regime ──────────────────────────────────────────────────
roll_ret = close_d.pct_change(WINDOW)
regime = pd.Series("Sideways", index=close_d.index)
regime[roll_ret >  THRESHOLD] = "Bull"
regime[roll_ret < -THRESHOLD] = "Bear"
regime = regime.dropna()

# Map to UTC date for join
regime_by_date = regime.to_frame("regime")
regime_by_date.index = pd.to_datetime(regime_by_date.index).normalize()

print("\nDaily regime distribution (full history):")
vc = regime.value_counts()
for r in ("Bull", "Sideways", "Bear"):
    n = vc.get(r, 0)
    print(f"  {r:<10s}: {n:3d} days ({n/len(regime)*100:.1f}%)")

# Show last 30 days
recent = regime_by_date.tail(30)
print("\nLast 30 daily regime labels:")
for dt, row in recent.iterrows():
    print(f"  {dt.date()}  {row['regime']}")

# ── 3. Load BTC paper trades ─────────────────────────────────────────────────
print(f"\nLoading {TRADES_CSV}...")
df = pd.read_csv(TRADES_CSV, low_memory=False)
df = df[df["decision"] == "trade"].copy()
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df["p_market"]   = pd.to_numeric(df["p_market"],   errors="coerce")
df["net_edge"]   = pd.to_numeric(df["net_edge"],    errors="coerce")
df["bet_amount"] = pd.to_numeric(df["bet_amount"],  errors="coerce")

# Parse trade date from logged_at
df["trade_date"] = pd.to_datetime(df["logged_at"], utc=True).dt.normalize().dt.tz_localize(None)

# Join regime
df = df.join(regime_by_date, on="trade_date", how="left")
df["regime"] = df["regime"].fillna("Unknown")

# Compute flat P&L
def flat_pnl(row):
    if row["side"] == "yes":
        won = row["resolved_yes"] == 1
        p   = float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:  # no
        won = row["resolved_yes"] == 0
        p   = 1 - float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET

df["flat_pnl"] = df.apply(flat_pnl, axis=1)
df["won"] = (
    ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
    ((df["side"] == "no")  & (df["resolved_yes"] == 0))
).astype(int)

# ── 4. Report ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BTC PAPER TRADES — SPLIT BY DAILY MARKOV REGIME")
print(f"  Window={WINDOW}d, threshold=±{THRESHOLD*100:.0f}%   Flat bet ${FLAT_BET:.0f}")
print(SEP)


def report_split(sub, label):
    n = len(sub)
    if n == 0:
        print(f"\n  {label}: no trades")
        return
    wr   = sub["won"].mean()
    pnl  = sub["flat_pnl"].sum()
    bkev = 1 - (1 - sub["p_market"].mean()) if sub["side"].eq("yes").all() else None
    # breakeven WR = p_market / (p_market + p_market*(1-rake))  simplified:
    avg_pm = sub["p_market"].mean() if pd.notna(sub["p_market"].mean()) else 0.5
    sides  = sub["side"].value_counts().to_dict()
    print(f"\n  {label}  (n={n}, yes={sides.get('yes',0)}, no={sides.get('no',0)})")
    print(f"    Win rate:  {wr*100:.1f}%")
    print(f"    Flat P&L:  ${pnl:+,.2f}")
    avg_be = sub.apply(
        lambda r: r["p_market"] / (r["p_market"] + r["p_market"]*(1-KALSHI_TAKE))
        if r["side"]=="yes" and pd.notna(r["p_market"]) else
        (1-r["p_market"]) / ((1-r["p_market"]) + (1-r["p_market"])*(1-KALSHI_TAKE))
        if pd.notna(r["p_market"]) else 0.5,
        axis=1
    ).mean()
    print(f"    Avg breakeven WR: {avg_be*100:.1f}%")
    print(f"    WR vs breakeven:  {(wr - avg_be)*100:+.1f} pp")


# Overall
report_split(df, "ALL TRADES")

# By regime
print(f"\n{SEP2}")
print("  BY DAILY MARKOV REGIME (all sides combined):")
for reg in ("Bull", "Sideways", "Bear"):
    sub = df[df["regime"] == reg]
    report_split(sub, f"Regime = {reg}")

# YES only by regime
print(f"\n{SEP2}")
print("  YES TRADES BY DAILY MARKOV REGIME:")
df_yes = df[df["side"] == "yes"]
for reg in ("Bull", "Sideways", "Bear"):
    sub = df_yes[df_yes["regime"] == reg]
    report_split(sub, f"YES | Regime = {reg}")

# NO only by regime
print(f"\n{SEP2}")
print("  NO TRADES BY DAILY MARKOV REGIME:")
df_no = df[df["side"] == "no"]
for reg in ("Bull", "Sideways", "Bear"):
    sub = df_no[df_no["regime"] == reg]
    report_split(sub, f"NO  | Regime = {reg}")

# ── 5. Gate simulation: block YES in Bear regime ────────────────────────────
print(f"\n{SEP}")
print("  GATE SIMULATION: block YES when Markov regime = Bear")
print(SEP)

yes_bear = df_yes[df_yes["regime"] == "Bear"]
yes_not_bear = df_yes[df_yes["regime"] != "Bear"]

blocked_pnl  = yes_bear["flat_pnl"].sum()
baseline_pnl = df["flat_pnl"].sum()
new_pnl      = baseline_pnl - blocked_pnl  # removing those trades

print(f"  YES trades in Bear regime blocked: {len(yes_bear)}")
print(f"  Their WR:         {yes_bear['won'].mean()*100:.1f}%")
print(f"  Their flat P&L:   ${blocked_pnl:+,.2f}  (removed from book)")
print(f"  Baseline total P&L: ${baseline_pnl:+,.2f}")
print(f"  Post-gate P&L:      ${new_pnl:+,.2f}")
print(f"  Delta:              ${new_pnl - baseline_pnl:+,.2f}")

# ── 6. Breakout by sub-regime: Bear + composite_trend ───────────────────────
print(f"\n{SEP2}")
print("  YES IN BEAR REGIME — split by composite_trend:")
df["composite_trend"] = pd.to_numeric(df.get("composite_trend", pd.Series(dtype=float)), errors="coerce")
for ct_label, mask in [
    ("c_trend ≤ -1 (both bear)", yes_bear["composite_trend"] <= -1),
    ("c_trend = 0  (divergent)", yes_bear["composite_trend"] == 0),
    ("c_trend ≥ +1 (divergent)", yes_bear["composite_trend"] >= 1),
]:
    sub = yes_bear[mask]
    if len(sub) < 5:
        print(f"  {ct_label}: n={len(sub)} (too small)")
        continue
    print(f"  {ct_label}: n={len(sub)}, WR={sub['won'].mean()*100:.1f}%, "
          f"P&L=${sub['flat_pnl'].sum():+,.2f}")

# ── 7. Transition probability at current moment ─────────────────────────────
print(f"\n{SEP}")
print("  CURRENT MARKOV STATE (as of today)")
print(SEP)
from sklearn.preprocessing import LabelEncoder  # noqa — just for reference

# Build full transition matrix from history
states = ["Bear", "Sideways", "Bull"]
state_idx = {s: i for i, s in enumerate(states)}
arr = regime.to_numpy()
counts = np.zeros((3, 3), dtype=float)
for i in range(len(arr) - 1):
    counts[state_idx[arr[i]], state_idx[arr[i+1]]] += 1
row_sums = counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
P = counts / row_sums

current_regime = regime.iloc[-1]
current_date   = regime.index[-1].date()
current_idx    = state_idx[current_regime]
print(f"\n  Today's regime: {current_regime}  (as of {current_date})")
print(f"  Given {current_regime}, tomorrow's distribution:")
for j, s in enumerate(states):
    print(f"    → {s:<10s}: {P[current_idx, j]*100:.1f}%")

print(f"\n  Full transition matrix:")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states):
    row = "  ".join(f"{P[i,j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")
