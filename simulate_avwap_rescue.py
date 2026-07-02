"""
simulate_avwap_rescue.py

Simulate replacing swing_high_gate c_trend>=2 rescue with below_avwap rescue.

Current gate: block YES if (strike > sh4h OR dist_high in [0,+1%)) AND c_trend < 2
Rescue: c_trend >= 2

Proposed options:
  A) rescue: below AVWAP only
  B) rescue: below AVWAP AND c_trend >= 2 (strictest)
  C) rescue: below AVWAP OR c_trend >= 2 (loosest / union)

Gate population: YES-side contracts near 4h swing high (proxy with detected 4h highs).
Uses flat $1 per-contract PnL (resolved_yes * (1-pm) - (1-resolved_yes) * pm).
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

# ── Load archive ─────────────────────────────────────────────────────────────
arc = pd.read_csv(RESULTS_DIR / "btc_scan_archive.csv", low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce")
arc = arc[arc["logged_at"].notna() & arc["resolved_yes"].notna()].copy()
arc["resolved_yes"] = arc["resolved_yes"].astype(int)
arc["strike"] = pd.to_numeric(arc["strike"], errors="coerce")

# ── Build AVWAP from 4h swing highs ──────────────────────────────────────────
df4h = pd.read_parquet(DATA_DIR / "binanceus_BTCUSDT_4h_1970-01-01_2026-06-14.parquet")
df4h.index = pd.to_datetime(df4h.index, utc=True)
df4h = df4h[df4h.index >= arc["logged_at"].min() - pd.Timedelta(days=30)].copy()

highs = df4h["high"].values
peaks, _ = find_peaks(highs, distance=4, prominence=highs * 0.008)

df4h["tp"] = (df4h["high"] + df4h["low"] + df4h["close"]) / 3
df4h["tv"] = df4h["tp"] * df4h["volume"]

avwap_arr = np.full(len(df4h), np.nan)
sh_arr    = np.full(len(df4h), np.nan)
cum_tv = cum_v = 0.0
last_sh = np.nan
last_sh_i = -1
peak_set = set(peaks)
for i in range(len(df4h)):
    if i in peak_set:
        cum_tv = cum_v = 0.0
        last_sh = df4h["high"].iloc[i]
        last_sh_i = i
    cum_tv += df4h["tv"].iloc[i]
    cum_v  += df4h["volume"].iloc[i]
    if cum_v > 0 and not np.isnan(last_sh):
        avwap_arr[i] = cum_tv / cum_v
        sh_arr[i]    = last_sh

df4h["avwap_sh"] = avwap_arr
df4h["sh_price"] = sh_arr

df4h_lu = df4h[["avwap_sh","sh_price"]].reset_index().rename(columns={"open_time":"bar_time"})
merged = pd.merge_asof(
    arc.sort_values("logged_at"),
    df4h_lu.sort_values("bar_time"),
    left_on="logged_at", right_on="bar_time",
    direction="backward",
)
merged["sh_dist_pct"]  = (merged["spot"] - merged["sh_price"]) / merged["sh_price"] * 100
merged["above_avwap"]  = merged["spot"] > merged["avwap_sh"]
merged["c_trend"]      = pd.to_numeric(merged["composite_trend"], errors="coerce").fillna(0)

# ── Gate population: YES-side contracts near the 4h swing high ───────────────
# "Near" = spot is within 3% below the swing high, OR strike is above the swing high.
# YES-side: p_market <= 0.5 (market implies YES is unlikely → we'd be buying YES at a discount)
yes_near = merged[
    (merged["p_market"] <= 0.5) &
    merged["sh_price"].notna() &
    (
        (merged["strike"] > merged["sh_price"]) |            # strike above resistance
        ((merged["sh_dist_pct"] >= 0.0) & (merged["sh_dist_pct"] < 1.0))   # spot just above sh
    )
].copy()

print(f"Gate population (YES-side near SH): {len(yes_near):,} rows")
print(f"  WR={yes_near['resolved_yes'].mean():.1%}  bkev={yes_near['p_market'].mean():.1%}")
print(f"  c_trend>=2: {(yes_near['c_trend']>=2).sum():,}  c_trend<2: {(yes_near['c_trend']<2).sum():,}")
print(f"  above_avwap: {yes_near['above_avwap'].sum():,}  below_avwap: {(~yes_near['above_avwap']).sum():,}")

# ── PnL helper ───────────────────────────────────────────────────────────────
def pnl(df):
    if len(df) == 0: return 0.0
    return float((df["resolved_yes"] * (1 - df["p_market"])
                  - (1 - df["resolved_yes"]) * df["p_market"]).sum())

def row(label, df):
    if len(df) == 0:
        print(f"  {label:<52s}  n=    0")
        return
    wr = df["resolved_yes"].mean()
    bk = df["p_market"].mean()
    w  = int(df["resolved_yes"].sum())
    l  = len(df) - w
    p  = pnl(df)
    print(f"  {label:<52s}  n={len(df):>5d}  wins={w:>4d}  losses={l:>4d}  WR={wr:.1%}  bkev={bk:.1%}  pnl=${p:+.0f}")

# ── Current gate ─────────────────────────────────────────────────────────────
cur_block = yes_near[yes_near["c_trend"] < 2]
cur_resc  = yes_near[yes_near["c_trend"] >= 2]

print(f"\n{'═'*70}")
print("CURRENT GATE  (block: c_trend < 2  |  rescue: c_trend >= 2)")
print(f"{'═'*70}")
row("Blocked (c_trend < 2)",  cur_block)
row("Rescued (c_trend >= 2)", cur_resc)
print(f"  Savings from blocking:  ${-pnl(cur_block):+.0f}")

# ── Option A: rescue = below AVWAP ───────────────────────────────────────────
a_block = yes_near[ yes_near["above_avwap"]]
a_resc  = yes_near[~yes_near["above_avwap"]]

print(f"\n{'═'*70}")
print("OPTION A  (block: above AVWAP  |  rescue: below AVWAP)")
print(f"{'═'*70}")
row("Blocked (above AVWAP)", a_block)
row("Rescued (below AVWAP)", a_resc)

# Marginal changes vs current
a_new_block = yes_near[ yes_near["above_avwap"] & (yes_near["c_trend"] >= 2)]  # c_trend rescued → now blocked
a_new_resc  = yes_near[~yes_near["above_avwap"] & (yes_near["c_trend"] <  2)]  # c_trend blocked → now rescued
print(f"\n  vs current — newly blocked: {len(a_new_block):,}  newly rescued: {len(a_new_resc):,}")
row("  Newly blocked (above AVWAP was rescued)", a_new_block)
row("  Newly rescued (below AVWAP was blocked)", a_new_resc)
saves_a = -pnl(a_new_block)   # avoiding losses from previously rescued bad bets
costs_a =  pnl(a_new_resc)    # allowing previously blocked bets (neg pnl = more losses)
print(f"  Savings from new blocks: ${saves_a:+.0f}")
print(f"  Cost of new rescues:     ${costs_a:+.0f}")
print(f"  NET vs current:          ${saves_a + costs_a:+.0f}")

# ── Option B: rescue = below AVWAP AND c_trend >= 2 ─────────────────────────
b_resc  = yes_near[~yes_near["above_avwap"] & (yes_near["c_trend"] >= 2)]
b_block = yes_near[ yes_near["above_avwap"] | (yes_near["c_trend"] <  2)]

print(f"\n{'═'*70}")
print("OPTION B  (rescue: below AVWAP AND c_trend >= 2)")
print(f"{'═'*70}")
row("Blocked (above AVWAP OR c_trend<2)", b_block)
row("Rescued (below AVWAP AND c_trend>=2)", b_resc)

b_new_block = yes_near[yes_near["above_avwap"] & (yes_near["c_trend"] >= 2)]
# B rescues subset of current rescues; nothing newly rescued vs current
saves_b = -pnl(b_new_block)
print(f"\n  vs current — additionally blocks {len(b_new_block):,} rows previously rescued by c_trend>=2")
row("  Newly blocked (above AVWAP, was rescued)", b_new_block)
print(f"  NET vs current:  ${saves_b:+.0f}  (no new rescues — stricter than current)")

# ── Option C: rescue = below AVWAP OR c_trend >= 2 ──────────────────────────
c_resc  = yes_near[~yes_near["above_avwap"] | (yes_near["c_trend"] >= 2)]
c_block = yes_near[ yes_near["above_avwap"] & (yes_near["c_trend"] <  2)]

print(f"\n{'═'*70}")
print("OPTION C  (rescue: below AVWAP OR c_trend >= 2)")
print(f"{'═'*70}")
row("Blocked (above AVWAP AND c_trend<2)", c_block)
row("Rescued (below AVWAP OR c_trend>=2)", c_resc)

c_new_resc = yes_near[~yes_near["above_avwap"] & (yes_near["c_trend"] < 2)]
costs_c = pnl(c_new_resc)
print(f"\n  vs current — additionally rescues {len(c_new_resc):,} rows previously blocked")
row("  Newly rescued (below AVWAP, was blocked)", c_new_resc)
print(f"  NET vs current:  ${costs_c:+.0f}  (no new blocks — looser than current)")

# ── Full comparison ──────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SUMMARY")
print(f"{'═'*70}")
print(f"  Current (c_trend<2 block):           baseline  "
      f"saves ${-pnl(cur_block):+.0f} by blocking {len(cur_block):,}")
print(f"  Option A (below_avwap rescue):        net {saves_a+costs_a:+.0f}  "
      f"blocks {len(a_block):,}  rescues {len(a_resc):,}")
print(f"  Option B (below_avwap AND c_trend≥2): net {saves_b:+.0f}  "
      f"blocks {len(b_block):,}  rescues {len(b_resc):,}  [strictest]")
print(f"  Option C (below_avwap OR c_trend≥2):  net {costs_c:+.0f}  "
      f"blocks {len(c_block):,}  rescues {len(c_resc):,}  [loosest]")
