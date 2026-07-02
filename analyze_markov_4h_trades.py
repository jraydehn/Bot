"""
analyze_markov_4h_trades.py

Fetches BTC-USD 1h data, resamples to 4h, computes a 20-bar rolling-return
Markov regime label (Bull / Bear / Sideways), joins to paper_trades.csv by
trade timestamp (last completed 4h bar before the trade), and reports WR +
P&L split by regime — same methodology as analyze_markov_regime_trades.py
but at 4h resolution.

Threshold sweep also runs to find the optimal Bull/Bear cutoff.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not found")

TRADES_CSV  = "results/paper_trades.csv"
WINDOW      = 20        # 20 × 4h bars = 80 hours of lookback
FLAT_BET    = 25.0
KALSHI_TAKE = 0.10

SEP  = "=" * 68
SEP2 = "-" * 52

# ── 1. Fetch 1h BTC, resample to 4h ─────────────────────────────────────────
print("\nFetching BTC-USD 1h data from Yahoo Finance (resampling to 4h)...")
df_1h = yf.download("BTC-USD", start="2025-01-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)

df_4h = df_1h["Close"].resample("4h").last().dropna()
print(f"  {len(df_4h)} 4h bars: {df_4h.index.min()} → {df_4h.index.max()}")

# ── 2. Sweep thresholds to find the best cutoff ──────────────────────────────
# We'll report multiple thresholds so we can see how sensitive the labels are.
print("\nRegime label counts at various thresholds (window=20 × 4h bars):")
print(f"  {'Threshold':>10s}  {'Bear':>6s}  {'Side':>6s}  {'Bull':>6s}  {'B/S/Bu%'}")
roll_ret = df_4h.pct_change(WINDOW)
for thr in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
    reg = pd.Series("Sideways", index=df_4h.index)
    reg[roll_ret >  thr] = "Bull"
    reg[roll_ret < -thr] = "Bear"
    reg = reg[roll_ret.notna()]
    bc = (reg == "Bear").sum()
    sc = (reg == "Sideways").sum()
    uc = (reg == "Bull").sum()
    n  = len(reg)
    print(f"  {thr*100:>8.1f}%   {bc:6d}  {sc:6d}  {uc:6d}  "
          f"({bc/n*100:.0f}% / {sc/n*100:.0f}% / {uc/n*100:.0f}%)")

# Use 1.0% as working threshold — reasonable split for 80h BTC lookback
THRESHOLD = 0.010
regime_4h = pd.Series("Sideways", index=df_4h.index)
regime_4h[roll_ret >  THRESHOLD] = "Bull"
regime_4h[roll_ret < -THRESHOLD] = "Bear"
regime_4h = regime_4h[roll_ret.notna()]
regime_4h.index = pd.to_datetime(regime_4h.index, utc=True)

print(f"\nWorking threshold: ±{THRESHOLD*100:.1f}%")
vc = regime_4h.value_counts()
for r in ("Bull", "Sideways", "Bear"):
    n = vc.get(r, 0)
    print(f"  {r:<10s}: {n:4d} bars ({n/len(regime_4h)*100:.1f}%)")

# ── 3. Build transition matrix ───────────────────────────────────────────────
states    = ["Bear", "Sideways", "Bull"]
state_idx = {s: i for i, s in enumerate(states)}
arr       = regime_4h.to_numpy()
counts    = np.zeros((3, 3), dtype=float)
for i in range(len(arr) - 1):
    counts[state_idx[arr[i]], state_idx[arr[i + 1]]] += 1
row_sums  = counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
P         = counts / row_sums

print(f"\n4h Transition matrix (rows=from, cols=to):")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states):
    row = "  ".join(f"{P[i,j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")
print(f"\nPersistence diagonal:")
for i, s in enumerate(states):
    print(f"  {s} -> {s}: {P[i,i]*100:.2f}%")

# ── 4. Load trades, join to 4h regime ───────────────────────────────────────
print(f"\nLoading {TRADES_CSV}...")
df = pd.read_csv(TRADES_CSV, low_memory=False)
df = df[df["decision"] == "trade"].copy()
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df["p_market"]     = pd.to_numeric(df["p_market"],   errors="coerce")
df["composite_trend"] = pd.to_numeric(df.get("composite_trend", pd.Series(dtype=float)), errors="coerce")

# Parse trade timestamp (UTC)
df["trade_ts"] = pd.to_datetime(df["logged_at"], utc=True)

# For each trade, find the last 4h bar whose close is <= trade_ts.
# regime_4h.index is the bar open/label timestamp after resample.
# We use merge_asof on sorted timestamps.
regime_df = regime_4h.reset_index()
regime_df.columns = ["bar_ts", "regime_4h"]
regime_df["bar_ts"] = pd.to_datetime(regime_df["bar_ts"]).dt.as_unit("us").dt.tz_localize("UTC") if regime_df["bar_ts"].dt.tz is None else regime_df["bar_ts"].dt.as_unit("us")
regime_df = regime_df.sort_values("bar_ts")
df = df.sort_values("trade_ts")
df["trade_ts"] = df["trade_ts"].dt.as_unit("us")

df = pd.merge_asof(
    df,
    regime_df,
    left_on="trade_ts",
    right_on="bar_ts",
    direction="backward",
)
df["regime_4h"] = df["regime_4h"].fillna("Unknown")

# Flat P&L
def flat_pnl(row):
    p = float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = 1 - p
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE) if p_no > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET

df["flat_pnl"] = df.apply(flat_pnl, axis=1)
df["won"]      = (
    ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
    ((df["side"] == "no")  & (df["resolved_yes"] == 0))
).astype(int)

# ── 5. Report ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BTC PAPER TRADES — SPLIT BY 4h MARKOV REGIME (20-bar × 4h, ±1%)")
print(f"  Flat bet ${FLAT_BET:.0f}   n={len(df)} resolved trades")
print(SEP)

def report(sub, label):
    n = len(sub)
    if n < 5:
        print(f"\n  {label}: n={n} (too small)")
        return
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    sides = sub["side"].value_counts().to_dict()
    avg_be = sub.apply(
        lambda r: r["p_market"] / (r["p_market"] + r["p_market"] * (1 - KALSHI_TAKE))
        if r["side"] == "yes" and pd.notna(r["p_market"]) else
        (1 - r["p_market"]) / ((1 - r["p_market"]) + (1 - r["p_market"]) * (1 - KALSHI_TAKE))
        if pd.notna(r["p_market"]) else 0.5,
        axis=1,
    ).mean()
    print(f"\n  {label}  (n={n}, yes={sides.get('yes',0)}, no={sides.get('no',0)})")
    print(f"    Win rate:         {wr*100:.1f}%")
    print(f"    Flat P&L:         ${pnl:+,.2f}")
    print(f"    Avg breakeven WR: {avg_be*100:.1f}%")
    print(f"    vs breakeven:     {(wr - avg_be)*100:+.1f} pp")

report(df, "ALL TRADES")

print(f"\n{SEP2}")
print("  BY 4h MARKOV REGIME (all sides):")
for reg in ("Bull", "Sideways", "Bear"):
    report(df[df["regime_4h"] == reg], f"Regime = {reg}")

print(f"\n{SEP2}")
print("  YES TRADES BY 4h MARKOV REGIME:")
dy = df[df["side"] == "yes"]
for reg in ("Bull", "Sideways", "Bear"):
    report(dy[dy["regime_4h"] == reg], f"YES | Regime = {reg}")

print(f"\n{SEP2}")
print("  NO TRADES BY 4h MARKOV REGIME:")
dn = df[df["side"] == "no"]
for reg in ("Bull", "Sideways", "Bear"):
    report(dn[dn["regime_4h"] == reg], f"NO  | Regime = {reg}")

# ── 6. Gate simulation ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  GATE SIMULATIONS (flat ${:.0f}/trade)".format(FLAT_BET))
print(SEP)

baseline = df["flat_pnl"].sum()
print(f"  Baseline P&L (all trades): ${baseline:+,.2f}")

for gate_label, mask in [
    ("Block ALL in Sideways",     df["regime_4h"] == "Sideways"),
    ("Block ALL in Bear",         df["regime_4h"] == "Bear"),
    ("Block YES in Sideways",     (df["regime_4h"] == "Sideways") & (df["side"] == "yes")),
    ("Block YES in Bear",         (df["regime_4h"] == "Bear")     & (df["side"] == "yes")),
    ("Block ALL in Bear+Sideways",(df["regime_4h"].isin(["Bear","Sideways"]))),
]:
    blocked = df[mask]
    if len(blocked) == 0:
        continue
    delta   = -blocked["flat_pnl"].sum()
    new_pnl = baseline + delta
    print(f"\n  {gate_label}")
    print(f"    Blocked n={len(blocked)}, WR={blocked['won'].mean()*100:.1f}%, "
          f"their P&L=${blocked['flat_pnl'].sum():+,.2f}")
    print(f"    Post-gate P&L: ${new_pnl:+,.2f}  (delta ${delta:+,.2f})")

# ── 7. Cross-tab: 4h regime × daily regime ───────────────────────────────────
print(f"\n{SEP}")
print("  4h REGIME × DAILY REGIME INTERACTION")
print("  (do they tell different stories, or are they the same signal?)")
print(SEP)

# Recompute daily regime and join
df_d = yf.download("BTC-USD", start="2025-01-01", end="2026-05-23",
                   progress=False, auto_adjust=True)
if isinstance(df_d.columns, pd.MultiIndex):
    df_d.columns = df_d.columns.get_level_values(0)
close_d  = df_d["Close"].dropna()
roll_d   = close_d.pct_change(20)
reg_d    = pd.Series("Sideways", index=close_d.index)
reg_d[roll_d >  0.02] = "Bull"
reg_d[roll_d < -0.02] = "Bear"
reg_d    = reg_d[roll_d.notna()]
reg_d.index = pd.to_datetime(reg_d.index, utc=True).normalize()

regime_daily_df = reg_d.reset_index()
regime_daily_df.columns = ["day_ts", "regime_daily"]
df["day_ts"] = df["trade_ts"].dt.normalize()
df = df.merge(regime_daily_df, on="day_ts", how="left")
df["regime_daily"] = df["regime_daily"].fillna("Unknown")

print("\n  Trade counts by (daily × 4h) regime cell:")
ct = pd.crosstab(df["regime_daily"], df["regime_4h"], margins=True)
print(ct.to_string())

print("\n  Win rate by (daily × 4h) regime cell:")
wr_ct = df.groupby(["regime_daily", "regime_4h"])["won"].agg(["mean","count"])
wr_ct.columns = ["WR", "n"]
wr_ct["WR"] = (wr_ct["WR"] * 100).round(1)
print(wr_ct.to_string())

print("\n  P&L by (daily × 4h) regime cell:")
pnl_ct = df.groupby(["regime_daily", "regime_4h"])["flat_pnl"].sum().round(2)
print(pnl_ct.to_string())

# ── 8. Current state ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  CURRENT 4h STATE")
print(SEP)
cur_4h  = regime_4h.iloc[-1]
cur_ts  = regime_4h.index[-1]
cur_idx = state_idx[cur_4h]
print(f"\n  Current 4h regime: {cur_4h}  (bar ending {cur_ts})")
print(f"  Given {cur_4h}, next-bar distribution:")
for j, s in enumerate(states):
    print(f"    → {s:<10s}: {P[cur_idx, j]*100:.1f}%")
