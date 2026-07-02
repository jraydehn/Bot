#!/usr/bin/env python3
"""
OTM YES gate: block YES when offset_pct < 0 AND momentum indicators are weak.
ITM YES bets (offset_pct >= 0) pass freely.
OTM YES bets require confirmation from ITM-correlated indicators.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

EDGE_THRESH = 0.04
BET_SIZE    = 10.0

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)
for c in ["spot", "floor_strike", "offset_pct", "tau_minutes", "p_market",
          "p_model_15m", "realized_vol_annual", "chg_5m", "ema_bias",
          "stoch_k_5m", "vwap_dist", "chg_15m"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["offset_pct", "p_market", "p_model_15m"]).reset_index(drop=True)

# Load leaky LGBM for NO side
with open("models/lgbm_15m_btc.pkl", "rb") as f:
    leaky_clf = pickle.load(f)

LEAKY_FEATURES = [
    "offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m",
    "stoch_k_15m", "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m",
    "vol_ratio_5m", "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
    "consec_dir_1h", "vol_ratio_1h", "realized_vol_annual",
]
MINS_PER_YEAR = 252 * 390
df["sigma_tau"] = df["realized_vol_annual"] * np.sqrt(df["tau_minutes"] / MINS_PER_YEAR)
df["z_score"]   = np.log(df["floor_strike"] / df["spot"]) / df["sigma_tau"].replace(0, np.nan)
df["body_5m"]   = 0.0
df["dir_5m"]    = 0.0

def fill_X(df, feats):
    X = pd.DataFrame(index=df.index)
    for f in feats:
        X[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0) if f in df.columns else 0.0
    return X

p_leaky = np.clip(leaky_clf.predict_proba(fill_X(df, LEAKY_FEATURES))[:, 1], 0.01, 0.99)
df["p_leaky"] = p_leaky

def pnl_yes(pm, r): return BET_SIZE * (1/pm - 1) if r == 1 else -BET_SIZE
def pnl_no(pm, r):  return BET_SIZE * (1/(1-pm) - 1) if r == 0 else -BET_SIZE


def simulate(df, gate_fn, label):
    yes_n = yes_w = yes_pnl = 0
    no_n  = no_w  = no_pnl  = 0
    blocked = 0
    for _, row in df.iterrows():
        pm  = row["p_market"]
        pym = row["p_model_15m"]
        pnm = row["p_leaky"]
        r   = row["resolved_yes"]

        ey = pym - pm
        en = pm - pnm

        if ey >= EDGE_THRESH:
            if gate_fn(row):  # blocked
                blocked += 1
                # still check NO
                if en >= EDGE_THRESH:
                    no_n += 1; p = pnl_no(pm, r); no_pnl += p; no_w += (p > 0)
            else:
                yes_n += 1; p = pnl_yes(pm, r); yes_pnl += p; yes_w += (p > 0)
        elif en >= EDGE_THRESH:
            no_n += 1; p = pnl_no(pm, r); no_pnl += p; no_w += (p > 0)

    total = yes_pnl + no_pnl
    ye_wr = yes_w / yes_n if yes_n else 0
    no_wr = no_w  / no_n  if no_n  else 0
    print(f"\n  {label}")
    print(f"  YES: n={yes_n:3d}  WR={ye_wr:.1%}  PnL=${yes_pnl:+.0f}  (blocked={blocked})")
    print(f"  NO:  n={no_n:3d}  WR={no_wr:.1%}  PnL=${no_pnl:+.0f}")
    print(f"  TOTAL PnL: ${total:+.0f}")
    return total


# ── Baseline ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  OTM YES Gate Simulation")
print("=" * 60)
simulate(df, lambda r: False, "A: Baseline (no gate)")

# ── Gate 1: block all OTM YES (offset_pct < 0) ───────────────────────────────
simulate(df,
    lambda r: r["offset_pct"] < 0,
    "B: Block all OTM YES (offset_pct < 0)")

# ── Indicator agreement score for OTM confirmation ───────────────────────────
# Bullish = signal says price moving toward strike
def itm_score(row):
    score = 0
    if pd.notna(row.get("chg_5m"))    and row["chg_5m"]    > 0:        score += 1
    if pd.notna(row.get("ema_bias"))   and row["ema_bias"]  > 0:        score += 1
    if pd.notna(row.get("stoch_k_5m")) and row["stoch_k_5m"] > 50:     score += 1
    if pd.notna(row.get("vwap_dist"))  and row["vwap_dist"] > 0:        score += 1
    if pd.notna(row.get("chg_15m"))    and row["chg_15m"]   > 0:        score += 1
    return score  # 0-5

df["itm_score"] = df.apply(itm_score, axis=1)

print("\n  OTM YES breakdown by indicator agreement score:")
print(f"  {'Score':>6}  {'n':>5}  {'WR':>7}  {'BE':>7}")
otm = df[(df["offset_pct"] < 0) & (df["side"].str.upper() == "YES")]
be  = otm["p_market"].mean() if len(otm) else 0
for s in range(6):
    sub = otm[otm["itm_score"] == s]
    if len(sub) < 5:
        continue
    wr = sub["resolved_yes"].mean()
    flag = "  ✓" if wr > be + 0.03 else ("  ✗" if wr < be - 0.03 else "")
    print(f"  {s:>6}  {len(sub):>5}  {wr:>7.1%}  {be:>7.1%}{flag}")

# ── Gate 2: block OTM YES when score < N ─────────────────────────────────────
print()
print("  Sweep: block OTM YES when itm_score < threshold")
print(f"  {'Min score':>10}  {'YES_n':>6}  {'YES_WR':>7}  {'YES_PnL':>9}  {'Total':>9}")
print("  " + "-" * 50)
best = (None, None)
for min_score in range(6):
    def gate(row, ms=min_score):
        return row["offset_pct"] < 0 and row["itm_score"] < ms
    t = simulate(df, gate, f"")
    # recompute for table
    yes_n = yes_w = yes_pnl = no_n = no_w = no_pnl = bl = 0
    for _, row in df.iterrows():
        pm=row["p_market"]; pym=row["p_model_15m"]; pnm=row["p_leaky"]; r=row["resolved_yes"]
        ey=pym-pm; en=pm-pnm
        if ey >= EDGE_THRESH:
            if gate(row): bl+=1
            else: yes_n+=1; p=pnl_yes(pm,r); yes_pnl+=p; yes_w+=(p>0)
        if en >= EDGE_THRESH and not (ey >= EDGE_THRESH and not gate(row)):
            no_n+=1; p=pnl_no(pm,r); no_pnl+=p; no_w+=(p>0)
    ye_wr = yes_w/yes_n if yes_n else 0
    total = yes_pnl+no_pnl
    if best[0] is None or total > best[0]:
        best = (total, min_score)
    marker = " ◄" if total == best[0] else ""
    print(f"  {min_score:>10}  {yes_n:>6}  {ye_wr:>7.1%}  ${yes_pnl:>+8.0f}  ${total:>+8.0f}{marker}")

# ── Gate 3: best threshold with explicit rescue (offset_pct > 0) ─────────────
print()
ms = best[1]
simulate(df,
    lambda r, ms=ms: r["offset_pct"] < 0 and r["itm_score"] < ms,
    f"C: Block OTM YES when itm_score < {ms}  (rescue = already ITM)")
