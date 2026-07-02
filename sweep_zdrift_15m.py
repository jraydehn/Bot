#!/usr/bin/env python3
"""
sweep_zdrift_15m.py

Sweep YES-side z_drift values to find optimal cap.
YES: p_yes = norm.cdf(z_drift - z_strike)
NO:  leaky LGBM (unchanged)

Flat $10/trade, edge threshold 0.04.
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

EDGE_THRESH = 0.04
BET_SIZE    = 10.0
MINS_PER_YEAR = 252 * 390

# ── Load data ────────────────────────────────────────────────────────────────

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)

for c in ["spot", "floor_strike", "p_market", "realized_vol_annual", "tau_minutes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["spot", "floor_strike", "p_market",
                        "realized_vol_annual", "tau_minutes"]).reset_index(drop=True)

df["sigma_tau"] = df["realized_vol_annual"] * np.sqrt(df["tau_minutes"] / MINS_PER_YEAR)
df["z_strike"]  = np.log(df["floor_strike"] / df["spot"]) / df["sigma_tau"].replace(0, np.nan)
df = df.dropna(subset=["z_strike"]).reset_index(drop=True)
print(f"Rows: {len(df)}  YES rate: {df['resolved_yes'].mean():.1%}  "
      f"mean z_strike: {df['z_strike'].mean():.4f}")

# ── Load leaky LGBM for NO side ──────────────────────────────────────────────

with open("models/lgbm_15m_btc.pkl", "rb") as f:
    leaky_clf = pickle.load(f)

LEAKY_FEATURES = [
    "offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m",
    "stoch_k_15m", "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m",
    "vol_ratio_5m", "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
    "consec_dir_1h", "vol_ratio_1h", "realized_vol_annual",
]

df["z_score"]  = df["z_strike"].copy()
df["body_5m"]  = df.get("body_5m", pd.Series(0.0, index=df.index)).fillna(0.0)
df["dir_5m"]   = df.get("dir_5m",  pd.Series(0.0, index=df.index)).fillna(0.0)

def fill_features(df, features):
    X = pd.DataFrame(index=df.index)
    for f in features:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
        else:
            X[f] = 0.0
    return X

X_leaky  = fill_features(df, LEAKY_FEATURES)
p_leaky  = np.clip(leaky_clf.predict_proba(X_leaky)[:, 1], 0.01, 0.99)
df["p_leaky"] = p_leaky

# ── P&L helpers ──────────────────────────────────────────────────────────────

def pnl_yes(pm, resolved): return BET_SIZE * (1 / pm - 1) if resolved == 1 else -BET_SIZE
def pnl_no(pm, resolved):  return BET_SIZE * (1 / (1 - pm) - 1) if resolved == 0 else -BET_SIZE


def simulate_drift(z_drift_val: float):
    p_yes_arr = np.clip(norm.cdf(z_drift_val - df["z_strike"].values), 0.01, 0.99)
    pm_arr    = df["p_market"].values
    pl_arr    = df["p_leaky"].values
    res_arr   = df["resolved_yes"].values

    yes_n = yes_w = 0
    yes_pnl = no_n = no_w = 0.0
    no_pnl = 0.0

    for i in range(len(df)):
        pm  = pm_arr[i]
        pym = p_yes_arr[i]
        pnm = pl_arr[i]
        r   = res_arr[i]

        ey = pym - pm
        en = pm - pnm  # = (1 - pnm) - (1 - pm)

        if ey >= EDGE_THRESH:
            yes_n += 1
            p = pnl_yes(pm, r)
            yes_pnl += p
            yes_w += (p > 0)
        elif en >= EDGE_THRESH:
            no_n += 1
            p = pnl_no(pm, r)
            no_pnl += p
            no_w += (p > 0)

    yes_wr = yes_w / yes_n if yes_n else 0.0
    no_wr  = no_w  / no_n  if no_n  else 0.0
    return yes_n, yes_wr, yes_pnl, no_n, no_wr, no_pnl


# ── Sweep ─────────────────────────────────────────────────────────────────────

print()
print(f"{'z_drift':>8}  {'YES_n':>6}  {'YES_WR':>7}  {'YES_PnL':>9}  "
      f"{'NO_n':>6}  {'NO_WR':>7}  {'NO_PnL':>9}  {'Total':>9}")
print("─" * 78)

drift_vals = np.round(np.arange(-0.10, 0.61, 0.05), 2)
best = None
for z in drift_vals:
    yn, ywr, ypnl, nn, nwr, npnl = simulate_drift(z)
    total = ypnl + npnl
    marker = " ◄" if best is None or total > best[0] else ""
    if marker:
        best = (total, z)
    print(f"  {z:+.2f}   {int(yn):6d}  {ywr:7.1%}  ${ypnl:>+8.0f}  "
          f"{int(nn):6d}  {nwr:7.1%}  ${npnl:>+8.0f}  ${total:>+8.0f}{marker}")

print()
print(f"  Best total PnL: ${best[0]:+.0f} at z_drift = {best[1]:+.2f}")

# ── Fine-grained sweep around best ───────────────────────────────────────────

print()
print("Fine sweep ±0.05 around best:")
print(f"{'z_drift':>8}  {'YES_n':>6}  {'YES_WR':>7}  {'YES_PnL':>9}  {'Total':>9}")
print("─" * 55)
fine_vals = np.round(np.arange(max(-0.20, best[1] - 0.05), best[1] + 0.06, 0.01), 3)
best2 = None
for z in fine_vals:
    yn, ywr, ypnl, nn, nwr, npnl = simulate_drift(z)
    total = ypnl + npnl
    marker = " ◄" if best2 is None or total > best2[0] else ""
    if marker:
        best2 = (total, z)
    print(f"  {z:+.3f}   {int(yn):6d}  {ywr:7.1%}  ${ypnl:>+8.0f}  ${total:>+8.0f}{marker}")

print()
print(f"  Optimal z_drift = {best2[1]:+.3f}  →  Total PnL = ${best2[0]:+.0f}")
