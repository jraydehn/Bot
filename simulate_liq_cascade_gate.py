"""
simulate_liq_cascade_gate.py
Analyse liq_cascade_gate blocked trades (n=50 BTC YES OTM).

Gate condition: liq_score <= -1 (long cascade) AND offset_pct >= 0 (OTM YES)
Rescue (allows through): vpin_raw >= 0.75 AND pm >= 0.38 AND offset_pct <= 0.08

These 50 were BLOCKED (no rescue). Simulation showed WR=52% vs BE=42%, +$51 if taken.
Goal: find whether any sub-condition explains the win rate — i.e., is there a tighter
block condition that correctly removes losers while letting profitable trades through?
"""

import pandas as pd
import numpy as np

FLAT = 10.0

df = pd.read_csv("results/blocked_trades.csv", low_memory=False)
g = df[(df["asset"] == "BTC") & (df["gate_name"] == "liq_cascade_gate") & df["resolved_yes"].notna()].copy()

for col in ["pm", "p_model", "offset_pct", "vpin_score", "composite_rev",
            "composite_trend", "ema_stack_bias", "stoch_k", "composite_p_up",
            "obi_score", "vol_score", "funding_bias", "structure_bias", "wap_stretch"]:
    if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")

def pnl_yes(sub):
    return sum((1-p)*FLAT if r==1 else -p*FLAT for r,p in zip(sub["resolved_yes"], sub["pm"]))

def row_stats(sub, label):
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr = sub["resolved_yes"].mean()
    be = sub["pm"].mean()
    pnl = pnl_yes(sub)
    delta = wr - be
    verdict = "KEEP-BLOCK" if delta < -0.04 else ("BORDERLINE" if abs(delta) <= 0.04 else "ALLOW")
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={delta:+.1%}, PnL={pnl:+.0f}  [{verdict}]"

print("="*65)
print(f"liq_cascade_gate blocked YES trades (n={len(g)})")
wr_all = g["resolved_yes"].mean()
be_all = g["pm"].mean()
print(f"Overall: WR={wr_all:.1%}, BE={be_all:.1%}, Δ={wr_all-be_all:+.1%}, PnL={pnl_yes(g):+.0f}")
print("="*65)

# --- pm buckets ---
print("\n[pm buckets]")
for lo, hi in [(0.25, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.55)]:
    m = (g["pm"] >= lo) & (g["pm"] < hi)
    print(row_stats(g[m], f"pm [{lo:.2f},{hi:.2f})"))

# --- offset_pct buckets ---
print("\n[offset_pct buckets]")
for lo, hi in [(0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10)]:
    m = (g["offset_pct"] >= lo) & (g["offset_pct"] < hi)
    print(row_stats(g[m], f"offset [{lo:.2f},{hi:.2f})"))

# --- ema_stack_bias ---
print("\n[ema_stack_bias]")
for val in [-1, 0, 1]:
    m = g["ema_stack_bias"] == val
    print(row_stats(g[m], f"ema_stack={val:+d}"))

# --- composite_rev ---
print("\n[composite_rev buckets]")
for lo, hi in [(-10, 0), (0, 2), (2, 4), (4, 10)]:
    m = (g["composite_rev"] >= lo) & (g["composite_rev"] < hi)
    print(row_stats(g[m], f"rev [{lo},{hi})"))

# --- composite_trend ---
print("\n[composite_trend buckets]")
for lo, hi in [(-5, -1), (-1, 1), (1, 5)]:
    m = (g["composite_trend"] >= lo) & (g["composite_trend"] < hi)
    print(row_stats(g[m], f"trend [{lo},{hi})"))

# --- stoch_k buckets ---
print("\n[stoch_k buckets]")
for lo, hi in [(0, 20), (20, 35), (35, 55), (55, 100)]:
    m = (g["stoch_k"] >= lo) & (g["stoch_k"] < hi)
    print(row_stats(g[m], f"stoch [{lo},{hi})"))

# --- composite_p_up ---
print("\n[composite_p_up buckets]")
for lo, hi in [(0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0)]:
    m = (g["composite_p_up"] >= lo) & (g["composite_p_up"] < hi)
    print(row_stats(g[m], f"p_up [{lo:.2f},{hi:.2f})"))

# --- vpin_score ---
print("\n[vpin_score]")
for val in [0, 1]:
    m = g["vpin_score"] == val
    print(row_stats(g[m], f"vpin={val}"))

# --- model edge ---
g["yes_edge"] = g["p_model"] - g["pm"]
print("\n[model edge (p_model - pm) buckets]")
for lo, hi in [(-0.5, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30)]:
    m = (g["yes_edge"] >= lo) & (g["yes_edge"] < hi)
    print(row_stats(g[m], f"edge [{lo:.2f},{hi:.2f})"))

# --- funding_bias ---
print("\n[funding_bias]")
for val in [-1, 0, 1]:
    m = g["funding_bias"] == val
    print(row_stats(g[m], f"funding={val:+d}"))

# --- structure_bias ---
print("\n[structure_bias]")
for val in [-1, 0, 1]:
    m = g["structure_bias"] == val
    print(row_stats(g[m], f"struct={val:+d}"))

# --- Cross-cuts: most promising splits ---
print("\n[Cross-cuts]")
m_ema_neg = g["ema_stack_bias"] == -1
m_stoch_low = g["stoch_k"] < 35
m_rev_low = g["composite_rev"] < 2
m_pm_high = g["pm"] >= 0.40

print(row_stats(g[m_ema_neg & m_stoch_low], "ema=-1 AND stoch<35"))
print(row_stats(g[m_ema_neg & ~m_stoch_low], "ema=-1 AND stoch>=35"))
print(row_stats(g[~m_ema_neg & m_stoch_low], "ema!=-1 AND stoch<35"))
print(row_stats(g[~m_ema_neg & ~m_stoch_low], "ema!=-1 AND stoch>=35"))
print()
print(row_stats(g[m_stoch_low & m_pm_high], "stoch<35 AND pm>=0.40"))
print(row_stats(g[m_stoch_low & ~m_pm_high], "stoch<35 AND pm<0.40"))
print(row_stats(g[~m_stoch_low & m_pm_high], "stoch>=35 AND pm>=0.40"))
print(row_stats(g[~m_stoch_low & ~m_pm_high], "stoch>=35 AND pm<0.40"))
print()
# Gate rescue condition was: vpin_raw>=0.75 AND pm>=0.38 AND offset<=0.08
# These went through anyway because they DID NOT meet rescue (so vpin<0.75 or pm<0.38 or off>0.08)
# Let's see: pm>=0.38 vs pm<0.38
m_pm38 = g["pm"] >= 0.38
print(row_stats(g[m_pm38], "pm>=0.38 (near rescue threshold)"))
print(row_stats(g[~m_pm38], "pm<0.38 (OTM)"))
print()
m_off8 = g["offset_pct"] <= 0.08
print(row_stats(g[m_off8], "offset<=0.08% (near rescue threshold)"))
print(row_stats(g[~m_off8], "offset>0.08%"))
print()
# stoch<35: historically a key separator for cascade gates
m_stoch35 = g["stoch_k"] < 35
print(row_stats(g[m_stoch35 & m_pm38], "stoch<35 AND pm>=0.38"))
print(row_stats(g[~m_stoch35 & m_pm38], "stoch>=35 AND pm>=0.38"))
print(row_stats(g[m_stoch35 & ~m_pm38], "stoch<35 AND pm<0.38"))
