"""
simulate_no_z_gate_sweep.py
Sweep btc_no_z_gate threshold (currently 0.45) to find optimal cutoff.

Uses blocked_trades.csv where gate_name='btc_no_z_gate' (all NO-side).
Estimates sigma_tau per contract from pm + spot + strike using inverse normal,
then computes actual z_no = |log(strike/spot)| / sigma_tau.
Sweeps threshold 0.10–0.70 and reports PnL at each level.
Flat $10/trade.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

FLAT = 10.0
CURRENT_THRESHOLD = 0.45

df = pd.read_csv("results/blocked_trades.csv", low_memory=False)
g = df[(df["asset"] == "BTC") & (df["gate_name"] == "btc_no_z_gate") & df["resolved_yes"].notna()].copy()

print(f"btc_no_z_gate blocked NO trades: {len(g)}")
print(f"pm range: {g['pm'].min():.3f}–{g['pm'].max():.3f}, mean={g['pm'].mean():.3f}")
print(f"offset_pct range: {g['offset_pct'].min():.4f}–{g['offset_pct'].max():.4f}")

# Estimate sigma_tau from market price using inverse normal
# For YES option (binary): pm ≈ Φ(log(spot/strike) / sigma_tau)  [ignoring drift]
# => sigma_tau = log(spot/strike) / Φ⁻¹(pm)
g["log_ratio"] = np.log(g["spot"] / g["strike"])
# Clip pm to avoid ±inf from norm.ppf
pm_clipped = g["pm"].clip(0.01, 0.99)
g["ppf_pm"] = norm.ppf(pm_clipped)
# Avoid division by near-zero ppf
valid = g["ppf_pm"].abs() > 0.05
g.loc[valid, "sigma_tau_est"] = g.loc[valid, "log_ratio"] / g.loc[valid, "ppf_pm"]
g.loc[~valid, "sigma_tau_est"] = np.nan

# Fallback: use typical sigma_tau for BTC hourly (vol~0.65 ann, tau~45min)
TYPICAL_SIGMA_TAU = 0.0065 * np.sqrt(45)
g["sigma_tau_est"] = g["sigma_tau_est"].fillna(TYPICAL_SIGMA_TAU).clip(0.001, 0.10)

# Compute actual z_no = |log(strike/spot)| / sigma_tau
g["z_no"] = np.abs(np.log(g["strike"] / g["spot"])) / g["sigma_tau_est"]

print(f"\nz_no distribution (blocked trades):")
print(g["z_no"].describe())

# Verify: all z_no should be < current threshold (they were blocked because z < 0.45)
below_thresh = (g["z_no"] < CURRENT_THRESHOLD).mean()
print(f"Fraction with z_no < {CURRENT_THRESHOLD}: {below_thresh:.1%}")
print()

# PnL helper for NO bets
def pnl_no(sub):
    return sum(p * FLAT if r == 0 else -(1-p) * FLAT
               for r, p in zip(sub["resolved_yes"], sub["pm"]))

# Bucket analysis
print("Performance by z_no bucket (blocked trades — PnL if taken):")
print(f"{'z_no bucket':<18} {'n':>6} {'WR':>7} {'BE':>7} {'Δ':>7} {'PnL/t':>8} {'PnL total':>10}")
print("-" * 70)

buckets = [0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0]
labels  = ["<0.10","0.10-0.20","0.20-0.30","0.30-0.40","0.40-0.45",
           "0.45-0.50","0.50-0.60","0.60-0.70",">0.70"]
g["z_bin"] = pd.cut(g["z_no"], bins=buckets, labels=labels)

for lbl in labels:
    sub = g[g["z_bin"] == lbl]
    if len(sub) < 3:
        print(f"  z{lbl:<15} {len(sub):>6}  (too small)")
        continue
    wr = (1 - sub["resolved_yes"]).mean()
    be = (1 - sub["pm"]).mean()
    pnl = pnl_no(sub)
    delta = wr - be
    ppt = pnl / len(sub)
    print(f"  z{lbl:<15} {len(sub):>6}  {wr:.1%}  {be:.1%}  {delta:>+.1%}  {ppt:>+8.2f}  {pnl:>+10.0f}")

# Threshold sweep
print(f"\nThreshold sweep (block z_no < threshold, allow z_no >= threshold):")
print(f"{'Threshold':<12} {'Blocked':>8} {'Allowed':>8} {'PnL blocked':>12} {'PnL allowed':>12} {'Net delta':>10}")
print("-" * 65)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
for t in thresholds:
    blocked = g[g["z_no"] < t]
    allowed = g[g["z_no"] >= t]
    pnl_b = pnl_no(blocked) if len(blocked) > 0 else 0
    pnl_a = pnl_no(allowed) if len(allowed) > 0 else 0
    marker = " <-- current" if t == CURRENT_THRESHOLD else ""
    print(f"  z < {t:.2f}      {len(blocked):>8} {len(allowed):>8}  {pnl_b:>+12.0f}  {pnl_a:>+12.0f}  {pnl_a:>+10.0f}{marker}")

# Recommendation
print(f"\nKey question: what threshold maximises PnL on currently-blocked trades?")
best_t = None
best_pnl = -999999
for t in np.arange(0.10, 0.61, 0.05):
    allowed = g[g["z_no"] >= t]
    if len(allowed) < 5:
        continue
    pnl = pnl_no(allowed)
    if pnl > best_pnl:
        best_pnl = pnl
        best_t = t
print(f"  Best: allow z_no >= {best_t:.2f} → PnL on newly-allowed = ${best_pnl:+.0f}")
print(f"  Current: block z_no < {CURRENT_THRESHOLD} → ${pnl_no(g[g['z_no']<CURRENT_THRESHOLD]):+.0f} (all blocked)")
print(f"  Δ from lowering threshold to {best_t:.2f}: ${best_pnl - pnl_no(g[g['z_no']<best_t]):+.0f}")

# Also: look at the p_model distribution — what does the current z_drift model say?
print(f"\nModel edge (pm - p_model) distribution for blocked NO trades:")
g["no_edge"] = g["pm"] - g["p_model"]  # NO edge = how much market overprices YES vs model
print(g["no_edge"].describe())
print()
print("By z_no bucket:")
for lbl in labels:
    sub = g[g["z_bin"] == lbl]
    if len(sub) < 3:
        continue
    avg_edge = sub["no_edge"].mean()
    wr = (1 - sub["resolved_yes"]).mean()
    be = (1 - sub["pm"]).mean()
    print(f"  z{lbl:<15} n={len(sub):>5}  avg_no_edge={avg_edge:+.3f}  WR={wr:.1%}  BE={be:.1%}  Δ={wr-be:+.1%}")
