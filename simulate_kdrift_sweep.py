#!/usr/bin/env python3
"""
simulate_kdrift_sweep.py

Sweep BTC YES base k_drift from 0.5 to 1.6 against live paper trades.
Uses actual trade rows (decision=='trade', side=='yes', resolved)
to see how different base multipliers affect:
  - p_model (and overestimation bias)
  - Kelly bet sizing
  - Total P&L

Formula mirrors score_to_p_model in composite_scorer.py:
    z_strike = log(K/S) / sigma_tau
    k_drift  = k_base * exp(-2.0 * max(0, z_strike))   # OTM dampening preserved
    z_drift  = ppf(p_up) * k_drift
    p_model  = 1 - Φ(z_strike - z_drift)
"""

import math, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pricing_comparison import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE

KALSHI_FEE_RATE = 0.07
BANKROLL        = 1000.0
KELLY_MULT      = 0.30
KELLY_CAP       = 0.06
EXP_DECAY_COEFF = 2.0     # keeps OTM dampening structure the same

K_BASE_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

SEP = "=" * 72

# ── Load trades ────────────────────────────────────────────────────────────
df = pd.read_csv(
    "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv",
    low_memory=False,
)

# Only resolved YES bets that actually traded
mask = (
    (df["decision"] == "trade") &
    (df["side"] == "yes") &
    (df["resolved_yes"].notna()) &
    (df["resolved_yes"] != "") &
    (df["spot"].notna()) &
    (df["strike"].notna()) &
    (df["vol_eff"].notna()) &
    (df["tau_minutes"].notna()) &
    (df["composite_p_up"].notna()) &
    (df["p_market"].notna())
)
yes = df[mask].copy()
yes["resolved_yes"] = pd.to_numeric(yes["resolved_yes"], errors="coerce")
yes = yes[yes["resolved_yes"].isin([0, 1])].copy()

print(f"Resolved YES trades: {len(yes)}")
print(f"Date range: {yes['logged_at'].min()} → {yes['logged_at'].max()}")
print()

def kalshi_fee(pm):
    return KALSHI_FEE_RATE * min(pm, 1 - pm)

def compute_p_model(row, k_base):
    sigma_tau = float(row["vol_eff"]) * math.sqrt(float(row["tau_minutes"]))
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(float(row["strike"]) / float(row["spot"])) / sigma_tau
    k_drift  = k_base * math.exp(-EXP_DECAY_COEFF * max(0.0, z_strike))
    p_up     = float(row["composite_p_up"])
    z_drift  = norm.ppf(max(0.01, min(0.99, p_up))) * k_drift
    z_adj    = z_strike - z_drift
    return float(np.clip(1 - norm.cdf(z_adj), 0.01, 0.99))

def compute_net_edge(p_model, pm, side="yes"):
    fee  = kalshi_fee(pm)
    if side == "yes":
        raw = p_model - pm
    else:
        raw = (1 - p_model) - (1 - pm)
    return raw - DEFAULT_SLIPPAGE - DEFAULT_SPREAD - fee

def kelly_bet(p_model, pm, side="yes"):
    if side == "yes":
        b = (1 - pm) / pm if pm > 0 else 0
    else:
        b = pm / (1 - pm) if pm < 1 else 0
    if b <= 0:
        return 0.0
    kf   = max(0.0, (b * p_model - (1 - p_model)) / b)
    frac = min(kf * KELLY_MULT, KELLY_CAP)
    return round(BANKROLL * frac, 2)

def trade_pnl(bet, pm, won, side="yes"):
    if bet <= 0:
        return 0.0
    fee = kalshi_fee(pm) * bet / pm   # fee on gross notional
    if side == "yes":
        if won:
            gross = bet * (1 - pm) / pm
            return gross - fee
        else:
            return -bet - fee
    else:
        if won:
            gross = bet * pm / (1 - pm)
            return gross - fee
        else:
            return -bet - fee

# ── Baseline: production p_model ──────────────────────────────────────────
print(SEP)
print("BASELINE (production p_yes_model)")
print(SEP)
base_pnl_total = 0.0
base_bets = 0
for _, row in yes.iterrows():
    pm      = float(row["p_market"])
    p_model = float(row["p_yes_model"])
    won     = int(row["resolved_yes"]) == 1
    net_e   = compute_net_edge(p_model, pm, "yes")
    if net_e < MIN_NET_EDGE:
        continue
    bet = kelly_bet(p_model, pm, "yes")
    pnl = trade_pnl(bet, pm, won, "yes")
    base_pnl_total += pnl
    base_bets += 1

base_wr  = yes["resolved_yes"].mean()
base_avg_pm = yes["p_market"].mean()
base_avg_pm_hat = yes["p_yes_model"].mean()
print(f"  N trades:        {len(yes)}")
print(f"  Actual WR:       {base_wr:.3f}")
print(f"  Avg pm:          {base_avg_pm:.3f}")
print(f"  Avg p_yes_model: {base_avg_pm_hat:.3f}  (bias = {base_avg_pm_hat - base_wr:+.3f})")
print(f"  Bets above MIN_EDGE: {base_bets}")
print(f"  Total P&L:       ${base_pnl_total:.2f}")
print()

# ── Sweep ─────────────────────────────────────────────────────────────────
print(SEP)
print(f"{'k_base':>8}  {'N_bets':>7}  {'Avg_pmodel':>11}  {'Bias':>7}  {'P&L':>9}  {'WR_bets':>8}")
print(SEP)

results = []
for k_base in K_BASE_SWEEP:
    pnl_total = 0.0
    wins_bets = 0
    n_bets    = 0
    sum_pm_hat = 0.0

    for _, row in yes.iterrows():
        pm      = float(row["p_market"])
        p_model = compute_p_model(row, k_base)
        won     = int(row["resolved_yes"]) == 1
        net_e   = compute_net_edge(p_model, pm, "yes")
        if net_e < MIN_NET_EDGE:
            continue
        bet  = kelly_bet(p_model, pm, "yes")
        pnl  = trade_pnl(bet, pm, won, "yes")
        pnl_total  += pnl
        wins_bets  += int(won)
        n_bets     += 1
        sum_pm_hat += p_model

    wr_bets  = wins_bets / n_bets if n_bets else 0
    avg_phat = sum_pm_hat / n_bets if n_bets else 0
    bias     = avg_phat - base_wr   # overestimation vs actual WR

    marker = " <-- current" if abs(k_base - 1.4) < 0.01 else ""
    print(f"  {k_base:>5.1f}   {n_bets:>7}   {avg_phat:>10.3f}   {bias:>+6.3f}   ${pnl_total:>8.2f}   {wr_bets:.3f}{marker}")
    results.append(dict(k_base=k_base, n_bets=n_bets, avg_phat=avg_phat, bias=bias, pnl=pnl_total, wr=wr_bets))

print()

# ── Per-pm-bucket breakdown at k=1.0 vs k=1.4 ─────────────────────────────
print(SEP)
print("P&L by pm bucket:  k=1.0  vs  k=1.4  (production)")
print(SEP)
buckets = [(0.00, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]

for lo, hi in buckets:
    sub = yes[(yes["p_market"] >= lo) & (yes["p_market"] < hi)]
    if len(sub) == 0:
        continue

    def sweep_pnl(sub, k):
        pnl, bets = 0.0, 0
        for _, row in sub.iterrows():
            pm      = float(row["p_market"])
            p_model = compute_p_model(row, k)
            won     = int(row["resolved_yes"]) == 1
            net_e   = compute_net_edge(p_model, pm)
            if net_e < MIN_NET_EDGE:
                continue
            bet  = kelly_bet(p_model, pm)
            pnl += trade_pnl(bet, pm, won)
            bets += 1
        return pnl, bets

    pnl_10, n10 = sweep_pnl(sub, 1.0)
    pnl_14, n14 = sweep_pnl(sub, 1.4)
    wr_sub = sub["resolved_yes"].mean()
    print(f"  pm [{lo:.2f},{hi:.2f})  n={len(sub):3d}  WR={wr_sub:.3f}  "
          f"k=1.0: ${pnl_10:>7.2f} ({n10} bets)  "
          f"k=1.4: ${pnl_14:>7.2f} ({n14} bets)")

print()
print("Best k_base by P&L:", max(results, key=lambda x: x["pnl"])["k_base"])
print("Done.")
