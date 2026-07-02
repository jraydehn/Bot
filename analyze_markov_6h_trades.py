"""
analyze_markov_6h_trades.py

Tests whether a 6h Markov regime has predictive or gating value for BTC Kalshi trades.

Two questions:
  1. Does the 6h regime predict the direction of the next 6h candle?
  2. Does the 6h regime predict P&L outcome of BTC Kalshi trades (1h model + 15m model)?

Regime definition: 20-bar rolling return on 6h BTC bars.
Thresholds swept to find best labeling.
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

TRADES_1H_CSV  = "results/paper_trades.csv"
TRADES_15M_CSV = "results/paper_trades_btc15m.csv"
FLAT_BET       = 25.0
KALSHI_TAKE    = 0.10
MIN_N          = 8
WINDOW         = 20   # 20 × 6h = 5 days rolling lookback
SEP  = "=" * 68
SEP2 = "-" * 52

# ── helpers ───────────────────────────────────────────────────────────────────
def flat_pnl(row):
    p = float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = max(1 - p, 1e-6)
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET

def show(sub, label):
    n = len(sub)
    if n < MIN_N:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    return {"label": label, "n": n, "wr": wr, "pnl": pnl}

def print_row(r):
    if r is None: return
    print(f"  {r['label']:<52s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}")

def make_regime_6h(close_6h, window, threshold):
    rr  = close_6h.pct_change(window)
    reg = pd.Series("Sideways", index=close_6h.index)
    reg[rr >  threshold] = "Bull"
    reg[rr < -threshold] = "Bear"
    reg = reg[rr.notna()]
    reg.index = pd.to_datetime(reg.index, utc=True)
    return reg

def join_regime(df_trades, regime_series, col_name, ts_col="trade_ts"):
    reg_df = regime_series.reset_index()
    reg_df.columns = ["_jts", col_name]
    reg_df["_jts"] = pd.to_datetime(reg_df["_jts"], utc=True).dt.as_unit("us")
    df_trades = df_trades.copy()
    df_trades[ts_col] = pd.to_datetime(df_trades[ts_col], utc=True).dt.as_unit("us")
    merged = pd.merge_asof(
        df_trades.sort_values(ts_col),
        reg_df.sort_values("_jts"),
        left_on=ts_col, right_on="_jts",
        direction="backward",
    )
    return merged.drop(columns=["_jts"], errors="ignore")

def load_trades(csv_path, ts_format="ISO8601"):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["decision"] == "trade"].copy()
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
    df["trade_ts"] = pd.to_datetime(df["logged_at"], format=ts_format, utc=True)
    df["flat_pnl"] = df.apply(flat_pnl, axis=1)
    df["won"] = (
        ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
        ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    ).astype(int)
    return df

# ── 1. Fetch 6h data ──────────────────────────────────────────────────────────
print("Fetching BTC-USD 1h data (to build 6h bars)...")
df_1h = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)
df_1h = df_1h[["Open", "High", "Low", "Close"]].dropna()

# Resample 1h → 6h
df_6h = df_1h.resample("6h").agg(
    {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
).dropna()
print(f"  {len(df_1h)} 1h bars → {len(df_6h)} 6h bars: {df_6h.index.min()} → {df_6h.index.max()}")

close_6h = df_6h["Close"]

# ── 2. Threshold sweep for labeling balance ────────────────────────────────────
print(f"\nRegime label counts at various thresholds (window={WINDOW} × 6h = {WINDOW*6}h = {WINDOW*6//24}d):")
roll_ret = close_6h.pct_change(WINDOW)
print(f"  {'Threshold':>10s}  {'Bear':>6s}  {'Side':>6s}  {'Bull':>6s}  {'B/S/Bu%'}")
for thr in [0.005, 0.008, 0.010, 0.015, 0.020, 0.030]:
    reg = pd.Series("Sideways", index=close_6h.index)
    reg[roll_ret >  thr] = "Bull"
    reg[roll_ret < -thr] = "Bear"
    reg = reg[roll_ret.notna()]
    bc, sc, uc, n = (reg == "Bear").sum(), (reg == "Sideways").sum(), (reg == "Bull").sum(), len(reg)
    print(f"  {thr*100:>8.2f}%   {bc:6d}  {sc:6d}  {uc:6d}  "
          f"({bc/n*100:.0f}% / {sc/n*100:.0f}% / {uc/n*100:.0f}%)")

# Working threshold: ±1.5% (30h rolling return on 6h bars = 5-day window)
THRESHOLD = 0.015
print(f"\nWorking threshold: ±{THRESHOLD*100:.1f}%")

reg_6h = make_regime_6h(close_6h, WINDOW, THRESHOLD)
states = ["Bear", "Sideways", "Bull"]
vc = reg_6h.value_counts()
for s in states:
    n = vc.get(s, 0)
    print(f"  {s:<10s}: {n:5d} bars ({n/len(reg_6h)*100:.1f}%)")

# ── 3. Transition matrix ───────────────────────────────────────────────────────
state_idx = {s: i for i, s in enumerate(states)}
arr = reg_6h.to_numpy()
counts = np.zeros((3, 3), dtype=float)
for i in range(len(arr) - 1):
    counts[state_idx[arr[i]], state_idx[arr[i + 1]]] += 1
rs = counts.sum(axis=1, keepdims=True)
rs[rs == 0] = 1.0
P = counts / rs

print(f"\n6h Transition matrix (rows=from, cols=to):")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states):
    row = "  ".join(f"{P[i, j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")
print("\nPersistence diagonal:")
for i, s in enumerate(states):
    print(f"  {s} → {s}: {P[i, i]*100:.2f}%")

# ── 4. Next-6h candle direction by regime ─────────────────────────────────────
print(f"\n{SEP}")
print("  NEXT-6h CANDLE DIRECTION  vs  6h MARKOV REGIME")
print(SEP)

df_6h2 = df_6h.copy()
df_6h2["regime"] = reg_6h.reindex(df_6h2.index)
df_6h2 = df_6h2.dropna(subset=["regime"]).sort_index()
df_6h2["next_close"] = df_6h2["Close"].shift(-1)
df_6h2["next_open"]  = df_6h2["Open"].shift(-1)
df_6h2["next_ret"]   = (df_6h2["next_close"] - df_6h2["next_open"]) / df_6h2["next_open"]
df_6h2["next_bull"]  = (df_6h2["next_close"] > df_6h2["next_open"]).astype(float)
df_6h2 = df_6h2.dropna(subset=["next_bull", "next_ret"])

base_rate = df_6h2["next_bull"].mean()
print(f"\nTotal labeled 6h bars: {len(df_6h2):,}   Base bullish rate: {base_rate*100:.2f}%")

for reg in states:
    sub = df_6h2[df_6h2["regime"] == reg]
    n   = len(sub)
    if n < 20:
        print(f"\n  {reg}: n={n} (too small)")
        continue
    acc   = sub["next_bull"].mean()
    avg_r = sub["next_ret"].mean() * 100
    std_r = sub["next_ret"].std()  * 100
    lift  = acc - base_rate
    print(f"\n  Regime = {reg:<10s}  (n={n:,})")
    print(f"    Next-6h bullish rate:        {acc*100:.2f}%  (lift={lift*100:+.2f}pp vs base {base_rate*100:.2f}%)")
    print(f"    Avg next-6h return:          {avg_r:+.4f}%  ± {std_r:.4f}%")

# Chi-square
bull_c = [(df_6h2[df_6h2["regime"] == s]["next_bull"] == 1).sum() for s in states]
bear_c = [(df_6h2[df_6h2["regime"] == s]["next_bull"] == 0).sum() for s in states]
chi2, p_chi, dof, _ = stats.chi2_contingency(np.array([bull_c, bear_c]))
print(f"\n  Chi-square: chi2={chi2:.3f}  p={p_chi:.5f}  "
      f"({'Significant' if p_chi < 0.05 else 'NOT significant'} at p<0.05)")

sub_b = df_6h2[df_6h2["regime"] == "Bull"]["next_bull"]
sub_r = df_6h2[df_6h2["regime"] == "Bear"]["next_bull"]
t, pt = stats.ttest_ind(sub_b, sub_r)
print(f"  Bull vs Bear t-test: t={t:.3f}  p={pt:.5f}")

# ── 5. Monthly stability ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MONTHLY STABILITY: bullish-rate of next 6h bar by regime")
print(SEP)
df_6h2["month"] = df_6h2.index.to_period("M")
mo   = df_6h2.groupby(["month", "regime"])["next_bull"].mean().unstack(fill_value=np.nan)
mo_n = df_6h2.groupby(["month", "regime"])["next_bull"].count().unstack(fill_value=0)
print(f"\n  {'Month':<10s}  {'Bear%':>7s}  {'Side%':>7s}  {'Bull%':>7s}  "
      f"{'Bear_n':>7s}  {'Side_n':>7s}  {'Bull_n':>7s}")
print("  " + "-" * 60)
for m in mo.index:
    def g(col):
        v = mo.loc[m, col] if col in mo.columns else np.nan
        n = int(mo_n.loc[m, col]) if col in mo_n.columns else 0
        return v, n
    bv, bn = g("Bear"); sv, sn = g("Sideways"); uv, un = g("Bull")
    print(f"  {str(m):<10s}  "
          f"{bv*100:>6.1f}%  {sv*100:>6.1f}%  {uv*100:>6.1f}%  "
          f"{bn:>7d}  {sn:>7d}  {un:>7d}")

# ── 6. Kalshi trade impact — 1h model ─────────────────────────────────────────
print(f"\n{SEP}")
print("  KALSHI TRADE IMPACT — 1h BTC model (paper_trades.csv)")
print(SEP)

if Path(TRADES_1H_CSV).exists():
    df1h = load_trades(TRADES_1H_CSV, ts_format="mixed")
    df1h = join_regime(df1h, reg_6h, "reg_6h")
    df1h["reg_6h"] = df1h["reg_6h"].fillna("Unknown")

    print(f"\n  Total resolved trades: {len(df1h)}  "
          f"WR={df1h['won'].mean()*100:.1f}%  P&L=${df1h['flat_pnl'].sum():+,.0f}")
    print(f"\n  {'Regime + Side':<40s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 64)
    for reg in ["Bull", "Sideways", "Bear", "Unknown"]:
        sub = df1h[df1h["reg_6h"] == reg]
        r = show(sub, f"6h={reg} (all)")
        print_row(r)
        for side in ["yes", "no"]:
            r2 = show(sub[sub["side"] == side], f"  6h={reg}  side={side}")
            print_row(r2)
else:
    print(f"  {TRADES_1H_CSV} not found — skipping")

# ── 7. Kalshi trade impact — 15m model ────────────────────────────────────────
print(f"\n{SEP}")
print("  KALSHI TRADE IMPACT — 15m BTC model (paper_trades_btc15m.csv)")
print(SEP)

if Path(TRADES_15M_CSV).exists():
    df15m = load_trades(TRADES_15M_CSV, ts_format="ISO8601")
    df15m = join_regime(df15m, reg_6h, "reg_6h")
    df15m["reg_6h"] = df15m["reg_6h"].fillna("Unknown")
    df15m["p_market"] = pd.to_numeric(df15m["p_market"], errors="coerce")

    print(f"\n  Total resolved trades: {len(df15m)}  "
          f"WR={df15m['won'].mean()*100:.1f}%  P&L=${df15m['flat_pnl'].sum():+,.0f}")
    print(f"\n  {'Regime + Side':<40s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 64)
    for reg in ["Bull", "Sideways", "Bear", "Unknown"]:
        sub = df15m[df15m["reg_6h"] == reg]
        r = show(sub, f"6h={reg} (all)")
        print_row(r)
        for side in ["yes", "no"]:
            r2 = show(sub[sub["side"] == side], f"  6h={reg}  side={side}")
            print_row(r2)

    # Sweep: gate candidates in 15m model
    print(f"\n  GATE SWEEP — 15m BTC YES under each 6h regime (WR≥55% shown):")
    print(f"\n  {'Condition':<52s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 74)

    yes_trades = df15m[df15m["side"] == "yes"]
    no_trades  = df15m[df15m["side"] == "no"]

    for reg in ["Bull", "Sideways", "Bear"]:
        sub_yes = yes_trades[yes_trades["reg_6h"] == reg]
        sub_no  = no_trades[no_trades["reg_6h"] == reg]

        # p_market splits within each regime
        for side_label, sub_s in [("YES", sub_yes), ("NO", sub_no)]:
            r_base = show(sub_s, f"6h={reg}  {side_label} (all)")
            print_row(r_base)
            for pm_cut, pm_label, pm_op in [
                (0.40, "pm≤0.40", "le"), (0.45, "pm≤0.45", "le"),
                (0.50, "pm≤0.50", "le"), (0.55, "pm≤0.55", "le"),
                (0.50, "pm≥0.50", "ge"), (0.55, "pm≥0.55", "ge"),
                (0.60, "pm≥0.60", "ge"), (0.65, "pm≥0.65", "ge"),
            ]:
                if pm_op == "le":
                    mask = sub_s["p_market"] <= pm_cut
                else:
                    mask = sub_s["p_market"] >= pm_cut
                r2 = show(sub_s[mask.fillna(False)],
                          f"  6h={reg}  {side_label}  {pm_label}")
                if r2 and r2["wr"] >= 0.55:
                    print_row(r2)
        print()

    # Overlap with 1h and 15m Markov regimes (if already computed from other scripts)
    # Cross-tab: how often does 6h Bear overlap with the hard-lose zones?
    print(f"\n  6h Bear YES: hard block saves how much?")
    b6_yes = yes_trades[yes_trades["reg_6h"] == "Bear"]
    r_b6 = show(b6_yes, "6h Bear YES (hard block scenario)")
    print_row(r_b6)
    if r_b6:
        print(f"  → hard block saves ${-b6_yes['flat_pnl'].sum():+,.0f} on {len(b6_yes)} trades")
    print(f"\n  6h Sideways YES:")
    s6_yes = yes_trades[yes_trades["reg_6h"] == "Sideways"]
    print_row(show(s6_yes, "6h Sideways YES (all)"))
    print(f"\n  6h Bull YES:")
    u6_yes = yes_trades[yes_trades["reg_6h"] == "Bull"]
    print_row(show(u6_yes, "6h Bull YES (all)"))
else:
    print(f"  {TRADES_15M_CSV} not found — skipping")

# ── 8. Current 6h state ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  CURRENT 6h STATE")
print(SEP)
cur_reg = reg_6h.iloc[-1]
cur_ts  = reg_6h.index[-1]
cur_idx = state_idx[cur_reg]
print(f"\n  Current 6h regime: {cur_reg}  ({cur_ts})")
print(f"  Given {cur_reg}, next-6h bar distribution:")
for j, s in enumerate(states):
    print(f"    → {s:<10s}: {P[cur_idx, j]*100:.1f}%")
print(f"\n  Directional bias: {P[cur_idx, 2] - P[cur_idx, 0]:+.4f}  "
      f"({'bullish' if P[cur_idx, 2] > P[cur_idx, 0] else 'bearish' if P[cur_idx, 0] > P[cur_idx, 2] else 'neutral'})")
