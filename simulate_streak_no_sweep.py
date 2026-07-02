"""
simulate_streak_no_sweep.py
Analyse streak_gate NO blocks + dead gate candidates (bear_drift, btc_no_highpm_bearema).
"""

import pandas as pd
import numpy as np

FLAT = 10.0

df = pd.read_csv("results/blocked_trades.csv", low_memory=False)
df = df[df["asset"] == "BTC"].copy()
for col in ["pm","p_model","offset_pct","vpin_score","composite_rev","composite_trend",
            "ema_stack_bias","stoch_k","composite_p_up","obi_score","vol_score",
            "funding_bias","structure_bias","vwap_stretch","resolved_yes"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def pnl_no(sub):
    return sum(p*FLAT if r==0 else -(1-p)*FLAT
               for r,p in zip(sub["resolved_yes"], sub["pm"]))

def pnl_yes(sub):
    return sum((1-p)*FLAT if r==1 else -p*FLAT
               for r,p in zip(sub["resolved_yes"], sub["pm"]))

def stats_no(sub, label):
    sub = sub[sub["resolved_yes"].notna()]
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr  = (1 - sub["resolved_yes"]).mean()
    be  = (1 - sub["pm"]).mean()
    pnl = pnl_no(sub)
    d   = wr - be
    v   = "ALLOW" if d > 0.04 else ("KEEP-BLOCK" if d < -0.04 else "BORDERLINE")
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={d:+.1%}, PnL={pnl:+.0f}  [{v}]"

def stats_yes(sub, label):
    sub = sub[sub["resolved_yes"].notna()]
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr  = sub["resolved_yes"].mean()
    be  = sub["pm"].mean()
    pnl = pnl_yes(sub)
    d   = wr - be
    v   = "ALLOW" if d > 0.04 else ("KEEP-BLOCK" if d < -0.04 else "BORDERLINE")
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={d:+.1%}, PnL={pnl:+.0f}  [{v}]"

# ──────────────────────────────────────────────
# 1. STREAK_GATE NO
# ──────────────────────────────────────────────
sno = df[(df["gate_name"] == "streak_gate") & (df["side"] == "no")].copy()
print("=" * 60)
print(f"streak_gate NO blocks (n={len(sno)})")
print("=" * 60)
sno_r = sno[sno["resolved_yes"].notna()]
if len(sno_r):
    wr = (1 - sno_r["resolved_yes"]).mean()
    be = (1 - sno_r["pm"]).mean()
    print(f"Overall: WR={wr:.1%}, BE={be:.1%}, Δ={wr-be:+.1%}, PnL={pnl_no(sno_r):+.0f}")
    print(f"pm range: {sno_r['pm'].min():.3f}–{sno_r['pm'].max():.3f}")
    print(f"stoch_k range: {sno_r['stoch_k'].min():.1f}–{sno_r['stoch_k'].max():.1f}")
    print(f"composite_rev range: {sno_r['composite_rev'].min()}–{sno_r['composite_rev'].max()}")

    print("\n[pm buckets]")
    for lo, hi in [(0.3,0.5),(0.5,0.65),(0.65,0.80),(0.80,1.0)]:
        print(stats_no(sno_r[(sno_r["pm"]>=lo)&(sno_r["pm"]<hi)], f"pm [{lo:.2f},{hi:.2f})"))

    print("\n[stoch_k buckets]")
    for lo, hi in [(0,30),(30,40),(40,50),(50,70)]:
        print(stats_no(sno_r[(sno_r["stoch_k"]>=lo)&(sno_r["stoch_k"]<hi)], f"stoch [{lo},{hi})"))

    print("\n[composite_rev]")
    for lo, hi in [(-5,0),(0,2),(2,4),(4,10)]:
        print(stats_no(sno_r[(sno_r["composite_rev"]>=lo)&(sno_r["composite_rev"]<hi)], f"rev [{lo},{hi})"))

    print("\n[composite_trend]")
    for lo, hi in [(-5,-1),(-1,1),(1,3),(3,6)]:
        print(stats_no(sno_r[(sno_r["composite_trend"]>=lo)&(sno_r["composite_trend"]<hi)], f"trend [{lo},{hi})"))

    print("\n[ema_stack_bias]")
    for val in [-1,0,1]:
        print(stats_no(sno_r[sno_r["ema_stack_bias"]==val], f"ema_stack={val:+d}"))

    print("\n[structure_bias]")
    for val in [-1,0,1]:
        print(stats_no(sno_r[sno_r["structure_bias"]==val], f"struct={val:+d}"))

    print("\n[vwap_stretch]")
    for val in [-2,-1,0,1,2]:
        print(stats_no(sno_r[sno_r["vwap_stretch"]==val], f"vwap_stretch={val:+d}"))

    print("\n[model edge (pm - p_model)]")
    sno_r["no_edge"] = sno_r["pm"] - sno_r["p_model"]
    for lo, hi in [(-0.3,0),(0,0.10),(0.10,0.20),(0.20,0.50)]:
        print(stats_no(sno_r[(sno_r["no_edge"]>=lo)&(sno_r["no_edge"]<hi)], f"no_edge [{lo:.2f},{hi:.2f})"))

    print("\n[Cross-cuts]")
    m_rev  = sno_r["composite_rev"] >= 2
    m_ema1 = sno_r["ema_stack_bias"] == 1
    m_str1 = sno_r["structure_bias"] == 1
    m_sk40 = sno_r["stoch_k"] >= 40
    print(stats_no(sno_r[ m_rev],  "rev>=2 (bullish bounce)"))
    print(stats_no(sno_r[~m_rev],  "rev<2"))
    print(stats_no(sno_r[ m_ema1], "ema=+1 (bullish trend)"))
    print(stats_no(sno_r[~m_ema1], "ema!=+1"))
    print(stats_no(sno_r[ m_str1], "struct=+1 (bullish structure)"))
    print(stats_no(sno_r[~m_str1], "struct!=+1"))
    print(stats_no(sno_r[ m_sk40 &  m_ema1], "stoch>=40 AND ema=+1"))
    print(stats_no(sno_r[ m_sk40 & ~m_ema1], "stoch>=40 AND ema!=+1"))
    print(stats_no(sno_r[~m_sk40 &  m_ema1], "stoch<40 AND ema=+1"))
    print(stats_no(sno_r[~m_sk40 & ~m_ema1], "stoch<40 AND ema!=+1"))

# ──────────────────────────────────────────────
# 2. BEAR_DRIFT (YES side)
# ──────────────────────────────────────────────
bd = df[(df["gate_name"] == "bear_drift") & (df["side"] == "yes")].copy()
bd_r = bd[bd["resolved_yes"].notna()]
print("\n" + "=" * 60)
print(f"bear_drift YES blocks (n={len(bd_r)})")
print("=" * 60)
if len(bd_r) >= 3:
    wr = bd_r["resolved_yes"].mean()
    be = bd_r["pm"].mean()
    print(f"Overall: WR={wr:.1%}, BE={be:.1%}, Δ={wr-be:+.1%}, PnL={pnl_yes(bd_r):+.0f}")
    print("\n[stoch_k buckets]")
    for lo, hi in [(0,20),(20,35),(35,50),(50,100)]:
        print(stats_yes(bd_r[(bd_r["stoch_k"]>=lo)&(bd_r["stoch_k"]<hi)], f"stoch [{lo},{hi})"))
    print("\n[composite_rev]")
    for lo, hi in [(-5,0),(0,2),(2,4),(4,10)]:
        print(stats_yes(bd_r[(bd_r["composite_rev"]>=lo)&(bd_r["composite_rev"]<hi)], f"rev [{lo},{hi})"))
    print("\n[pm buckets]")
    for lo, hi in [(0.3,0.45),(0.45,0.55),(0.55,0.70)]:
        print(stats_yes(bd_r[(bd_r["pm"]>=lo)&(bd_r["pm"]<hi)], f"pm [{lo:.2f},{hi:.2f})"))
else:
    print("  (insufficient data)")

# ──────────────────────────────────────────────
# 3. BTC_NO_HIGHPM_BEAREMA_GATE (NO side)
# ──────────────────────────────────────────────
hp = df[(df["gate_name"] == "btc_no_highpm_bearema_gate") & (df["side"] == "no")].copy()
hp_r = hp[hp["resolved_yes"].notna()]
print("\n" + "=" * 60)
print(f"btc_no_highpm_bearema_gate NO blocks (n={len(hp_r)})")
print("=" * 60)
if len(hp_r) >= 3:
    wr = (1 - hp_r["resolved_yes"]).mean()
    be = (1 - hp_r["pm"]).mean()
    print(f"Overall: WR={wr:.1%}, BE={be:.1%}, Δ={wr-be:+.1%}, PnL={pnl_no(hp_r):+.0f}")
    print(f"pm range: {hp_r['pm'].min():.3f}–{hp_r['pm'].max():.3f}")
    print(f"p_model range: {hp_r['p_model'].min():.3f}–{hp_r['p_model'].max():.3f}")
    print("\n[pm buckets]")
    for lo, hi in [(0.70,0.80),(0.80,0.90),(0.90,1.0)]:
        print(stats_no(hp_r[(hp_r["pm"]>=lo)&(hp_r["pm"]<hi)], f"pm [{lo:.2f},{hi:.2f})"))
    print("\n[model NO edge (pm - p_model)]")
    hp_r["no_edge"] = hp_r["pm"] - hp_r["p_model"]
    for lo, hi in [(-0.2,0),(0,0.10),(0.10,0.20),(0.20,0.50)]:
        print(stats_no(hp_r[(hp_r["no_edge"]>=lo)&(hp_r["no_edge"]<hi)], f"no_edge [{lo:.2f},{hi:.2f})"))
    print("\n[stoch_k]")
    for lo, hi in [(0,30),(30,60),(60,100)]:
        print(stats_no(hp_r[(hp_r["stoch_k"]>=lo)&(hp_r["stoch_k"]<hi)], f"stoch [{lo},{hi})"))
else:
    print("  (insufficient data)")
