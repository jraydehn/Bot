"""
analyze_sol_gate_overlap.py

Check overlap between:
  SOL 4h Sideways (ALL — block YES + NO)
  SOL 1h Sideways YES (block YES only)
  SOL 6h Bull (block YES unless stoch_cross=0; block NO unless offset≤median)

Questions:
  1. How many trades are in each cell, and how many share both 4h+1h Sideways?
  2. Do the rescue conditions survive when populations are separated?
  3. What is the net P&L of applying all three SOL gates with rescues?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

FLAT_BET    = 25.0
KALSHI_TAKE = 0.10
SEP  = "=" * 72

DATA_DIR = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results")
SOL_CSV  = DATA_DIR / "paper_trades_sol15m.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def flat_pnl_row(row):
    try:
        p = float(row["p_market"])
        if not (0 < p < 1):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = max(1 - p, 1e-6)
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET

def prep_trades(df):
    df = df[df["decision"] == "trade"].copy()
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df["p_market"]  = pd.to_numeric(df["p_market"],  errors="coerce")
    df["trade_ts"]  = pd.to_datetime(df["logged_at"], format="ISO8601", utc=True)
    df = df.dropna(subset=["trade_ts", "p_market", "side"]).copy()
    df["flat_pnl"]  = df.apply(flat_pnl_row, axis=1)
    df["won"] = (
        ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
        ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    ).astype(int)
    return df

def join_regime(df_trades, regime_series, col):
    reg_df = regime_series.reset_index()
    reg_df.columns = ["_jts", col]
    reg_df["_jts"] = pd.to_datetime(reg_df["_jts"], utc=True).dt.as_unit("us")
    df = df_trades.copy()
    df["_ts"] = df["trade_ts"].dt.as_unit("us")
    merged = pd.merge_asof(
        df.sort_values("_ts"),
        reg_df.sort_values("_jts"),
        left_on="_ts", right_on="_jts",
        direction="backward",
    ).drop(columns=["_jts", "_ts"], errors="ignore")
    merged[col] = merged[col].fillna("Unknown")
    return merged

def show(sub, label):
    n = len(sub)
    if n < 5:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    return {"label": label, "n": n, "wr": wr, "pnl": pnl}

def pr(r, marker=""):
    if r is None: return
    flag = f"  ◄ {marker}" if marker else ""
    print(f"  {r['label']:<52s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}{flag}")

# ── load ──────────────────────────────────────────────────────────────────────

print(f"\nLoading {SOL_CSV.name}...")
raw = pd.read_csv(SOL_CSV, low_memory=False)
sol = prep_trades(raw)
print(f"  Resolved trades: {len(sol)}")

# ── build regimes ─────────────────────────────────────────────────────────────

print("\nFetching SOL price history from yfinance...")
raw_yf = yf.download(
    "SOL-USD",
    start=(pd.Timestamp.now("UTC") - pd.DateOffset(days=120)).strftime("%Y-%m-%d"),
    end=(pd.Timestamp.now("UTC") + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    interval="1h",
    progress=False,
    auto_adjust=True,
)
if isinstance(raw_yf.columns, pd.MultiIndex):
    raw_yf.columns = raw_yf.columns.get_level_values(0)
raw_yf.index = pd.to_datetime(raw_yf.index, utc=True)
close_1h = raw_yf["Close"].dropna()
print(f"  1h bars: {len(close_1h)}")

def make_regime(close, tf_label, period_bars, threshold):
    rr  = close.pct_change(period_bars)
    reg = pd.Series("Sideways", index=close.index)
    reg[rr >  threshold] = "Bull"
    reg[rr < -threshold] = "Bear"
    reg = reg[rr.notna()]
    cur = reg.iloc[-1] if len(reg) else "n/a"
    print(f"  {tf_label}: {len(reg)} bars, threshold=±{threshold*100:.1f}%, current={cur}")
    return reg

# 6h regime: resample 1h→6h, 20-bar window, ±3.0%
close_6h = close_1h.resample("6h").last().dropna()
reg_6h   = make_regime(close_6h, "6h", 20, 0.030)

# 4h regime: resample 1h→4h, 20-bar window, ±2.5%
close_4h = close_1h.resample("4h").last().dropna()
reg_4h   = make_regime(close_4h, "4h", 20, 0.025)

# 1h regime: 20-bar window, ±1.5%
reg_1h   = make_regime(close_1h, "1h", 20, 0.015)

# ── attach ────────────────────────────────────────────────────────────────────

sol = join_regime(sol, reg_6h, "r6h")
sol = join_regime(sol, reg_4h, "r4h")
sol = join_regime(sol, reg_1h, "r1h")

# ── overlap analysis ──────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  SOL GATE OVERLAP ANALYSIS")
print(SEP)

m6bull = sol["r6h"] == "Bull"
m4sw   = sol["r4h"] == "Sideways"
m1sw   = sol["r1h"] == "Sideways"

print(f"\n  Gate population sizes:")
print(f"  6h Bull:      n={m6bull.sum()}")
print(f"  4h Sideways:  n={m4sw.sum()}")
print(f"  1h Sideways:  n={m1sw.sum()}")
print(f"  4h SW ∩ 1h SW: n={(m4sw & m1sw).sum()}")
print(f"  6h Bull ∩ 4h SW: n={(m6bull & m4sw).sum()}")
print(f"  6h Bull ∩ 1h SW: n={(m6bull & m1sw).sum()}")
print(f"  All three:    n={(m6bull & m4sw & m1sw).sum()}")
print(f"  Union of all: n={(m6bull | m4sw | m1sw).sum()}")

print(f"\n  Venn breakdown (exclusive zones):")
zones = [
    ("6h Bull only",         m6bull & ~m4sw  & ~m1sw),
    ("4h SW only",          ~m6bull &  m4sw  & ~m1sw),
    ("1h SW only",          ~m6bull & ~m4sw  &  m1sw),
    ("6h Bull + 4h SW",      m6bull &  m4sw  & ~m1sw),
    ("6h Bull + 1h SW",      m6bull & ~m4sw  &  m1sw),
    ("4h SW + 1h SW",       ~m6bull &  m4sw  &  m1sw),
    ("All three",            m6bull &  m4sw  &  m1sw),
    ("None (clean trades)",  ~m6bull & ~m4sw & ~m1sw),
]
print(f"\n  {'Zone':<30s}  {'n':>5s}  {'WR':>6s}  {'YES P&L':>9s}  {'NO P&L':>9s}  {'Total P&L':>10s}")
print(f"  {'─'*78}")
for lbl, mask in zones:
    sub = sol[mask]
    if len(sub) == 0:
        print(f"  {lbl:<30s}  {0:5d}")
        continue
    wr   = sub["won"].mean() * 100
    y_pnl = sub[sub["side"]=="yes"]["flat_pnl"].sum()
    n_pnl = sub[sub["side"]=="no"]["flat_pnl"].sum()
    t_pnl = sub["flat_pnl"].sum()
    print(f"  {lbl:<30s}  {len(sub):5d}  {wr:5.1f}%  ${y_pnl:+8.0f}  ${n_pnl:+8.0f}  ${t_pnl:+9.0f}")

# ── rescue condition survival in each zone ────────────────────────────────────

print(f"\n{SEP}")
print("  RESCUE CONDITION SURVIVAL")
print(SEP)

# Get quantile values from full gate populations for reference
sc_col  = pd.to_numeric(sol.get("stoch_cross_1h", pd.Series(dtype=float)), errors="coerce")
off_col = pd.to_numeric(sol.get("offset_pct",     pd.Series(dtype=float)), errors="coerce")
sk_col  = pd.to_numeric(sol.get("stoch_k_1h",     pd.Series(dtype=float)), errors="coerce")
oi_col  = pd.to_numeric(sol.get("oi_chg_pct",     pd.Series(dtype=float)), errors="coerce")
off_med = float(off_col.median())

print(f"\n  [6h Bull YES rescue: stoch_cross_1h=0]")
for lbl, mask in [("6h Bull (full)", m6bull), ("6h Bull only (no 4h/1h SW)", m6bull & ~m4sw & ~m1sw)]:
    sub_yes = sol[mask & (sol["side"] == "yes")]
    if len(sub_yes) < 5: continue
    r = show(sub_yes, lbl + " YES")
    pr(r)
    r2 = show(sub_yes[sc_col.reindex(sub_yes.index, fill_value=np.nan) == 0], f"  stoch_cross=0")
    pr(r2, "RESCUE" if r2 and r2["pnl"] > 0 else "weak" if r2 else "")

print(f"\n  [6h Bull NO rescue: offset_pct ≤ {off_med:.4f} (median)]")
for lbl, mask in [("6h Bull (full)", m6bull), ("6h Bull only", m6bull & ~m4sw & ~m1sw)]:
    sub_no = sol[mask & (sol["side"] == "no")]
    if len(sub_no) < 5: continue
    r = show(sub_no, lbl + " NO")
    pr(r)
    r2 = show(sub_no[off_col.reindex(sub_no.index, fill_value=np.nan) <= off_med], f"  offset≤{off_med:.4f}")
    pr(r2, "RESCUE" if r2 and r2["pnl"] > 0 else "weak" if r2 else "")

print(f"\n  [4h Sideways NO rescue: stoch_k_1h ≥ 86.1]")
for lbl, mask in [("4h SW (full)", m4sw), ("4h SW only (no 1h SW)", m4sw & ~m1sw)]:
    sub_no = sol[mask & (sol["side"] == "no")]
    if len(sub_no) < 5: continue
    r = show(sub_no, lbl + " NO")
    pr(r)
    r2 = show(sub_no[sk_col.reindex(sub_no.index, fill_value=np.nan) >= 86.1], f"  stoch_k≥86")
    pr(r2, "RESCUE" if r2 and r2["pnl"] > 0 else "weak" if r2 else "")

print(f"\n  [1h Sideways YES rescue: oi_chg_pct ≥ 0.0535]")
for lbl, mask in [("1h SW (full)", m1sw), ("1h SW only (no 4h SW)", m1sw & ~m4sw)]:
    sub_yes = sol[mask & (sol["side"] == "yes")]
    if len(sub_yes) < 5: continue
    r = show(sub_yes, lbl + " YES")
    pr(r)
    r2 = show(sub_yes[oi_col.reindex(sub_yes.index, fill_value=np.nan) >= 0.0535], f"  oi_chg≥0.054")
    pr(r2, "RESCUE" if r2 and r2["pnl"] > 0 else "weak" if r2 else "")

# ── gate scenarios ────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  COMBINED GATE SCENARIOS (flat $25 bet)")
print(SEP)

base_pnl = sol["flat_pnl"].sum()
print(f"\n  Baseline: n={len(sol)}, WR={sol['won'].mean()*100:.1f}%, P&L=${base_pnl:+,.0f}")

def run_scenario(label, gate_mask, rescue_mask=None):
    """gate_mask = trades to block; rescue_mask = subset of gate_mask to keep."""
    if rescue_mask is None:
        rescue_mask = pd.Series(False, index=sol.index)
    blocked = sol[gate_mask & ~rescue_mask]
    kept    = sol[~gate_mask | rescue_mask]
    if len(kept) < 5:
        return
    wr_k  = kept["won"].mean() * 100
    pnl_k = kept["flat_pnl"].sum()
    improvement = pnl_k - base_pnl
    print(f"\n  {label}")
    print(f"  Blocked: {len(blocked)}, P&L blocked=${blocked['flat_pnl'].sum():+,.0f}")
    print(f"  Kept:    {len(kept)}, WR={wr_k:.1f}%, P&L=${pnl_k:+,.0f}  (Δ=${improvement:+,.0f})")

# Scenario 1: Hard block all three gates, no rescues
run_scenario(
    "Scen 1: Hard block 6h Bull + 4h SW + 1h SW YES (no rescues)",
    gate_mask=(m6bull | m4sw | (m1sw & (sol["side"] == "yes"))),
    rescue_mask=None,
)

# Scenario 2: Gates with rescues
rescued = (
    # 6h Bull YES rescue: stoch_cross=0
    (m6bull & (sol["side"] == "yes") & (sc_col == 0).fillna(False)) |
    # 6h Bull NO rescue: offset≤median
    (m6bull & (sol["side"] == "no")  & (off_col <= off_med).fillna(False)) |
    # 4h SW NO rescue: stoch_k≥86
    (m4sw   & (sol["side"] == "no")  & (sk_col >= 86.1).fillna(False)) |
    # 1h SW YES rescue: oi_chg≥0.054
    (m1sw   & (sol["side"] == "yes") & (oi_col >= 0.0535).fillna(False))
)
run_scenario(
    "Scen 2: All gates with rescues (recommended)",
    gate_mask=(m6bull | m4sw | (m1sw & (sol["side"] == "yes"))),
    rescue_mask=rescued,
)

# Scenario 3: Skip 1h SW gate (since heavily overlaps 4h SW)
run_scenario(
    "Scen 3: 6h Bull + 4h SW only (skip 1h SW gate)",
    gate_mask=(m6bull | m4sw),
    rescue_mask=(
        (m6bull & (sol["side"] == "yes") & (sc_col == 0).fillna(False)) |
        (m6bull & (sol["side"] == "no")  & (off_col <= off_med).fillna(False)) |
        (m4sw   & (sol["side"] == "no")  & (sk_col >= 86.1).fillna(False))
    ),
)

# Scenario 4: Skip 4h SW gate (use 1h SW instead, finer grained)
run_scenario(
    "Scen 4: 6h Bull + 1h SW YES only (skip 4h SW gate)",
    gate_mask=(m6bull | (m1sw & (sol["side"] == "yes"))),
    rescue_mask=(
        (m6bull & (sol["side"] == "yes") & (sc_col == 0).fillna(False)) |
        (m6bull & (sol["side"] == "no")  & (off_col <= off_med).fillna(False)) |
        (m1sw   & (sol["side"] == "yes") & (oi_col >= 0.0535).fillna(False))
    ),
)

# Scenario 5: 4h SW hard block + 1h SW YES rescue
run_scenario(
    "Scen 5: 4h SW hard block + 1h SW YES w/ rescue (no 6h Bull gate)",
    gate_mask=(m4sw | (m1sw & (sol["side"] == "yes"))),
    rescue_mask=(
        (m4sw & (sol["side"] == "no")  & (sk_col >= 86.1).fillna(False)) |
        (m1sw & (sol["side"] == "yes") & (oi_col >= 0.0535).fillna(False))
    ),
)

print(f"\n{SEP}")
print("  DONE")
print(SEP)
