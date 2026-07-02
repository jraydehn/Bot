#!/usr/bin/env python3
"""
simulate_pm_ema_gate.py

Simulate a YES gate on pm [0.50,0.60) conditioned on ema_stack_bias.
Key finding: within pm [0.50,0.60), ema=-1 (bearish) has +19.8% edge (+$581),
while ema=0 and ema=1 both lose (-$601, -$220).

Gate proposal: BLOCK YES at pm [0.50,0.60) when ema_stack_bias IN {0, 1}
               PASS  YES at pm [0.50,0.60) when ema_stack_bias == -1

Also sweeps adjacent pm ranges and checks composite_trend as an alternative.
Uses flat BANKROLL sim with actual would_pnl for blocked vs kept trades.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

SEP = "=" * 72

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(
    "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv",
    low_memory=False,
)
df["logged_at"] = pd.to_datetime(df["logged_at"])

mask = (
    (df["decision"] == "trade") &
    (df["side"] == "yes") &
    (df["resolved_yes"].notna()) &
    (df["resolved_yes"] != "") &
    (df["would_pnl"].notna())
)
yes = df[mask].copy()
yes["resolved_yes"] = pd.to_numeric(yes["resolved_yes"], errors="coerce")
yes = yes[yes["resolved_yes"].isin([0, 1])].copy()
yes = yes.sort_values("logged_at").reset_index(drop=True)

print(f"Total resolved YES trades: {len(yes)}")
print()

# ── Section 1: ema breakdown across ALL pm buckets ────────────────────────
print(SEP)
print("EMA STACK BREAKDOWN ACROSS ALL pm BUCKETS  (actual would_pnl)")
print(SEP)
print(f"{'pm range':>14}  {'ema':>5}  {'n':>5}  {'WR':>6}  {'avg_pm':>7}  {'edge':>7}  {'P&L':>9}")
print("-" * 70)

for lo, hi in [(0.00,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,1.0)]:
    sub = yes[(yes["p_market"]>=lo) & (yes["p_market"]<hi)]
    for es in ["-1","0","1"]:
        g = sub[sub["ema_stack_bias"] == es]
        if len(g) < 8:
            continue
        wr  = g["resolved_yes"].mean()
        pm  = g["p_market"].mean()
        pnl = g["would_pnl"].sum()
        print(f"  [{lo:.2f},{hi:.2f})  ema={es}  {len(g):5d}  {wr:.3f}  {pm:.3f}  {wr-pm:>+7.3f}  ${pnl:>9.2f}")
    print()

# ── Section 2: Gate proposal — block ema=0 and ema=1 at pm [0.50,0.60) ──
print(SEP)
print("GATE SIMULATION: block YES at pm [0.50,0.60) when ema IN {0,1}")
print("                 (rescue: allow ema=-1 through)")
print(SEP)

bucket = yes[(yes["p_market"]>=0.50) & (yes["p_market"]<0.60)].copy()
kept   = bucket[bucket["ema_stack_bias"] == "-1"]
blocked= bucket[bucket["ema_stack_bias"].isin(["0","1"])]

print(f"  Total in bucket: {len(bucket)}")
print(f"  Kept  (ema=-1):  {len(kept):3d}  WR={kept['resolved_yes'].mean():.3f}  P&L=${kept['would_pnl'].sum():.2f}")
print(f"  Blocked (ema!=−1):{len(blocked):3d}  WR={blocked['resolved_yes'].mean():.3f}  P&L=${blocked['would_pnl'].sum():.2f}")
print()
print(f"  Bucket before gate:  P&L=${bucket['would_pnl'].sum():.2f}")
print(f"  Bucket after  gate:  P&L=${kept['would_pnl'].sum():.2f}")
print(f"  Gate delta:          ${kept['would_pnl'].sum() - bucket['would_pnl'].sum():+.2f}")
print()

# ── Section 3: Wins blocked vs losses blocked ─────────────────────────────
wins_blocked  = int((blocked["resolved_yes"]==1).sum())
losses_blocked= int((blocked["resolved_yes"]==0).sum())
pnl_wins_blk  = blocked[blocked["resolved_yes"]==1]["would_pnl"].sum()
pnl_loss_blk  = blocked[blocked["resolved_yes"]==0]["would_pnl"].sum()
print(f"  Trades blocked: {len(blocked)} ({wins_blocked} wins, {losses_blocked} losses)")
print(f"    Wins blocked cost:   ${pnl_wins_blk:.2f}  (foregone profit)")
print(f"    Losses blocked save: ${pnl_loss_blk:.2f}  (losses avoided)")
print(f"    Net delta:           ${pnl_wins_blk + pnl_loss_blk:+.2f}")
print()

# ── Section 4: Full model P&L — all YES trades ────────────────────────────
print(SEP)
print("FULL MODEL IMPACT  (all YES trades, gate only on pm [0.50,0.60) ema gate)")
print(SEP)

# Everything outside the bucket stays
outside = yes[(yes["p_market"]<0.50) | (yes["p_market"]>=0.60)]
gate_yes = pd.concat([outside, kept], ignore_index=True)

print(f"  Before gate: {len(yes):4d} trades  P&L=${yes['would_pnl'].sum():.2f}")
print(f"  After  gate: {len(gate_yes):4d} trades  P&L=${gate_yes['would_pnl'].sum():.2f}")
print(f"  Delta:       {len(gate_yes)-len(yes):+4d} trades  delta=${gate_yes['would_pnl'].sum()-yes['would_pnl'].sum():+.2f}")
print()

# ── Section 5: Weekly stability ───────────────────────────────────────────
print(SEP)
print("WEEKLY STABILITY of pm [0.50,0.60) ema gate")
print(SEP)
print(f"{'week':>6}  {'dates':>14}  {'kept_n':>7}  {'kept_WR':>8}  {'blk_n':>7}  {'blk_WR':>8}  {'delta':>9}")
print("-" * 75)

bucket["week"] = bucket["logged_at"].dt.isocalendar().week
for w, g in bucket.groupby("week"):
    k = g[g["ema_stack_bias"]=="-1"]
    b = g[g["ema_stack_bias"].isin(["0","1"])]
    dates = f"{g['logged_at'].min().strftime('%m/%d')}–{g['logged_at'].max().strftime('%m/%d')}"
    kwr = k["resolved_yes"].mean() if len(k)>0 else float("nan")
    bwr = b["resolved_yes"].mean() if len(b)>0 else float("nan")
    delta = -b["would_pnl"].sum()   # losses avoided (sign flip: we save these)
    print(f"  {w:4d}  {dates:>14}  {len(k):7d}  {kwr:>8.3f}  {len(b):7d}  {bwr:>8.3f}  ${delta:>+8.2f}")

print()

# ── Section 6: Composite trend as alternative discriminator ───────────────
print(SEP)
print("ALTERNATIVE DISCRIMINATOR: composite_trend  (within pm [0.50,0.60))")
print(SEP)
print(f"{'trend':>7}  {'n':>5}  {'WR':>6}  {'edge':>7}  {'P&L':>9}")
print("-" * 42)

for t, g in bucket.groupby("composite_trend"):
    if len(g) < 5: continue
    wr  = g["resolved_yes"].mean()
    pm  = g["p_market"].mean()
    pnl = g["would_pnl"].sum()
    print(f"  {t:>5.0f}  {len(g):5d}  {wr:.3f}  {wr-pm:>+7.3f}  ${pnl:>9.2f}")

print()

# ── Section 7: Combined condition — composite_trend AND ema ───────────────
print(SEP)
print("COMPOSITE_TREND >= -1  AND  ema_stack=-1  (best combo?)")
print(SEP)

for ema_val in ["-1","0","1"]:
    for trend_thresh in [-3, -2, -1, 0, 1, 2]:
        g = bucket[
            (bucket["ema_stack_bias"]==ema_val) &
            (bucket["composite_trend"]>=trend_thresh)
        ]
        if len(g) < 8: continue
        wr  = g["resolved_yes"].mean()
        pm  = g["p_market"].mean()
        pnl = g["would_pnl"].sum()
        if abs(wr - pm) > 0.05:
            print(f"  ema={ema_val}  trend>={trend_thresh:+d}  n={len(g):3d}  WR={wr:.3f}  edge={wr-pm:+.3f}  P&L=${pnl:.2f}")

print()
print("Done.")
