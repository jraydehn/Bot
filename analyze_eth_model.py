"""
ETH model audit: reliability diagram + signal reconstruction + Gate A/B re-simulation.

Steps:
  1. Reliability diagram — p_yes_model calibration vs realized resolution rate
  2. YES ITM/OTM breakdown with WR + $ P&L + breakeven WR
  3. Reconstruct ema_alignment / stoch_bias / ema_stack_bias per trade from OHLCV history
  4. Gate A (ema_alignment==neutral), Gate B (stoch_bias==1 & ema_stack_bias==-1),
     Gate A+B combo — each with mandatory rescue search (framework Step 3)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

from confirmation_indicators import compute_confirmation

BASE = Path(__file__).parent
DATA = BASE / "data"
TRADES = BASE / "results" / "live_trades_eth.csv"

# ── load OHLCV ──────────────────────────────────────────────────────────────
print("Loading ETH 1h and 1m data …")
df_1h = pd.read_parquet(DATA / "binanceus_ETHUSDT_1h_2024-01-01_2026-05-07.parquet")
df_1m = pd.read_parquet(DATA / "binanceus_ETHUSDT_1m_2024-01-01_2026-05-07.parquet")

# Filter 1m to the period we actually need (reduces memory during per-trade slicing)
cutoff = pd.Timestamp("2026-03-01", tz="UTC")
df_1m = df_1m[df_1m.index >= cutoff]

# ── load trades ──────────────────────────────────────────────────────────────
df = pd.read_csv(TRADES)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
df["win"] = (
    ((df["side"] == "yes") & (df["resolved_yes"] == True)) |
    ((df["side"] == "no")  & (df["resolved_yes"] == False))
).astype(int)
n_total = len(df)
print(f"Loaded {n_total} ETH live trades\n")

# ── helper ──────────────────────────────────────────────────────────────────
def breakeven_wr(sub: pd.DataFrame) -> float:
    wins  = sub[sub["win"] == 1]["live_pnl"]
    loses = sub[sub["win"] == 0]["live_pnl"]
    if wins.empty or loses.empty:
        return float("nan")
    avg_win  = wins.mean()
    avg_loss = abs(loses.mean())
    return avg_loss / (avg_win + avg_loss)

def seg_stats(sub: pd.DataFrame, label: str) -> str:
    if sub.empty:
        return f"  {label}: n=0"
    n   = len(sub)
    wr  = sub["win"].mean()
    pnl = sub["live_pnl"].sum()
    bew = breakeven_wr(sub)
    pp  = (wr - bew) * 100
    return (f"  {label}: n={n}  WR={wr:.1%}  $P&L={pnl:+.2f}"
            f"  BEW={bew:.1%}  WRvsBEW={pp:+.1f}pp")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERALL + YES/NO SPLIT
# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SECTION 1 — OVERALL SUMMARY")
print("=" * 72)
print(seg_stats(df, "ALL"))
print(seg_stats(df[df["side"] == "yes"], "YES"))
print(seg_stats(df[df["side"] == "no"],  "NO "))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — YES ITM vs OTM (p_market threshold)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 2 — YES ITM vs OTM (split at p_market=0.50)")
print("=" * 72)
yes = df[df["side"] == "yes"]
yes_itm = yes[yes["p_market"] >= 0.50]
yes_otm = yes[yes["p_market"] <  0.50]
print(seg_stats(yes_itm, "YES ITM (pm≥0.50)"))
print(seg_stats(yes_otm, "YES OTM (pm<0.50)"))

# OTM sub-bands
for lo, hi in [(0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50)]:
    sub = yes[(yes["p_market"] >= lo) & (yes["p_market"] < hi)]
    print(seg_stats(sub, f"YES OTM pm=[{lo:.2f},{hi:.2f})"))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RELIABILITY DIAGRAM (p_yes_model vs realized)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 3 — RELIABILITY DIAGRAM (p_yes_model buckets)")
print("  bucket        n   mean_model  realized_YES%  mean_pnl/trade")
print("=" * 72)
bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
labels = ["<.10",".10-.20",".20-.30",".30-.40",".40-.50",
          ".50-.60",".60-.70",".70-.80",".80-.90",">.90"]
df["model_bucket"] = pd.cut(df["p_yes_model"], bins=bins, labels=labels)
for lab in labels:
    sub = df[df["model_bucket"] == lab]
    if sub.empty:
        continue
    mn  = sub["p_yes_model"].mean()
    rl  = sub["resolved_yes"].mean()
    mpnl = sub["live_pnl"].mean()
    print(f"  {lab:12s}  {len(sub):3d}   {mn:.3f}       {rl:.1%}        {mpnl:+.3f}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SIGNAL RECONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 4 — Reconstructing ema_alignment / stoch_bias / ema_stack_bias")
print("  (this may take ~1 min) …")
print("=" * 72)

ema_aligns   = []
stoch_biases = []
stack_biases = []
failed       = 0

for i, row in df.iterrows():
    ts = row["logged_at"]  # UTC

    # Last 90 1h bars ending at the bar that was open when the trade logged
    h1_end = ts.floor("1h")
    h1_slice = df_1h[df_1h.index < h1_end].tail(90)

    # Last 500 1m bars ending at trade minute (for stochastic + ema_stack)
    m1_slice = df_1m[df_1m.index < ts].tail(500)

    try:
        if len(h1_slice) < 60 or len(m1_slice) < 62:
            raise ValueError("insufficient bars")
        res = compute_confirmation(h1_slice, hist_1m=m1_slice)
        ema_aligns.append(res.ema_alignment)
        stoch_biases.append(res.stoch_bias)
        stack_biases.append(res.ema_stack_bias)
    except Exception as e:
        ema_aligns.append("neutral")
        stoch_biases.append(0)
        stack_biases.append(0)
        failed += 1

df["ema_alignment"]  = ema_aligns
df["stoch_bias"]     = stoch_biases
df["ema_stack_bias"] = stack_biases

print(f"  Reconstruction complete. Failures (fallback to neutral/0): {failed}/{n_total}")

# Signal distribution summary
print("\n  ema_alignment distribution:")
for v, cnt in df["ema_alignment"].value_counts().items():
    print(f"    {v}: {cnt} ({cnt/n_total:.1%})")
print("\n  stoch_bias distribution:")
for v, cnt in df["stoch_bias"].value_counts().items():
    print(f"    {v:+d}: {cnt} ({cnt/n_total:.1%})")
print("\n  ema_stack_bias distribution:")
for v, cnt in df["ema_stack_bias"].value_counts().items():
    print(f"    {v:+d}: {cnt} ({cnt/n_total:.1%})")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GATE A/B SIMULATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 5 — GATE SIMULATION (A, B, A+B) with rescue search")
print("=" * 72)

baseline_pnl = df["live_pnl"].sum()
print(f"\n  Baseline: n={n_total}  $P&L={baseline_pnl:+.2f}")

def gate_report(mask_block: pd.Series, label: str):
    blocked = df[mask_block]
    passed  = df[~mask_block]
    wins_b  = blocked[blocked["win"] == 1]["live_pnl"].sum()
    loss_b  = blocked[blocked["win"] == 0]["live_pnl"].sum()
    new_pnl = passed["live_pnl"].sum()
    delta   = new_pnl - baseline_pnl
    print(f"\n  [{label}]  blocks={len(blocked)}/{n_total}  wins_blocked=${wins_b:+.2f}  losses_blocked=${loss_b:+.2f}")
    print(f"  → new P&L={new_pnl:+.2f}  delta={delta:+.2f}")
    print(seg_stats(passed, "  passed"))
    print(seg_stats(blocked, "  blocked"))
    return mask_block

# Gate A: ema_alignment == "neutral"
mask_a = df["ema_alignment"] == "neutral"
gate_report(mask_a, "Gate A: ema_alignment==neutral")

# Gate B: stoch_bias==+1 AND ema_stack_bias==-1
mask_b = (df["stoch_bias"] == 1) & (df["ema_stack_bias"] == -1)
gate_report(mask_b, "Gate B: stoch_bias==+1 AND ema_stack_bias==-1")

# Gate A+B: either condition
mask_ab = mask_a | mask_b
gate_report(mask_ab, "Gate A+B combined (A OR B)")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RESCUE SEARCH on blocked segments
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SECTION 6 — RESCUE SEARCH within A+B-blocked trades")
print("  Slicing blocked trades by available signals to find profitable sub-sets")
print("=" * 72)

blocked_ab = df[mask_ab].copy()
print(f"\n  Blocked pool: n={len(blocked_ab)}  $P&L={blocked_ab['live_pnl'].sum():+.2f}")

# Slice by side
for side_val in ["yes", "no"]:
    sub = blocked_ab[blocked_ab["side"] == side_val]
    if not sub.empty:
        print(seg_stats(sub, f"  side={side_val}"))

# Slice by p_market band
print("\n  By p_market band within blocked:")
for lo, hi in [(0.05,0.20),(0.20,0.40),(0.40,0.60),(0.60,0.80),(0.80,0.95)]:
    sub = blocked_ab[(blocked_ab["p_market"] >= lo) & (blocked_ab["p_market"] < hi)]
    if not sub.empty:
        print(seg_stats(sub, f"  pm=[{lo:.2f},{hi:.2f})"))

# Slice by net_edge band within blocked
print("\n  By net_edge within blocked:")
for lo, hi in [(-1.0,-0.05),(-0.05,0.05),(0.05,0.15),(0.15,1.0)]:
    sub = blocked_ab[(blocked_ab["net_edge"] >= lo) & (blocked_ab["net_edge"] < hi)]
    if not sub.empty:
        print(seg_stats(sub, f"  edge=[{lo:.2f},{hi:.2f})"))

# Slice by stoch_bias within Gate-A-only block (neutral EMA, various stoch)
print("\n  Gate-A-blocked by stoch_bias (rescue candidate):")
ga_only = df[mask_a].copy()
for sv in [-1, 0, 1]:
    sub = ga_only[ga_only["stoch_bias"] == sv]
    if not sub.empty:
        print(seg_stats(sub, f"  ema=neutral stoch={sv:+d}"))

# Slice by ema_stack_bias within Gate-A block
print("\n  Gate-A-blocked by ema_stack_bias:")
for sv in [-1, 0, 1]:
    sub = ga_only[ga_only["ema_stack_bias"] == sv]
    if not sub.empty:
        print(seg_stats(sub, f"  ema=neutral stack={sv:+d}"))

# Within A+B block by side × stoch
print("\n  Gate-A+B-blocked by side × ema_stack_bias:")
for side_val in ["yes", "no"]:
    for sv in [-1, 0, 1]:
        sub = blocked_ab[(blocked_ab["side"] == side_val) & (blocked_ab["ema_stack_bias"] == sv)]
        if not sub.empty:
            print(seg_stats(sub, f"  {side_val} stack={sv:+d}"))

print("\nDone.")
