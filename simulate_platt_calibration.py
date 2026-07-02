#!/usr/bin/env python3
"""
simulate_platt_calibration.py

Calibrate BTC YES p_yes_model using:
  1. Platt scaling  — sigmoid fit: p_cal = σ(A + B * logit(p_raw))
  2. Isotonic regression — non-parametric monotone fit

Train on first 70% of resolved YES trades (by date), test on last 30%.
Shows: calibration curves, bias correction, and P&L delta on test set.

The goal is to find A,B such that the gate/Kelly machinery uses accurate
edge estimates, reducing phantom-edge betting.
"""

import math, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pricing_comparison import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE

KALSHI_FEE_RATE = 0.07
BANKROLL        = 1000.0
KELLY_MULT      = 0.30
KELLY_CAP       = 0.06
SEP = "=" * 72

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(
    "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv",
    low_memory=False,
)

mask = (
    (df["decision"] == "trade") &
    (df["side"] == "yes") &
    (df["resolved_yes"].notna()) &
    (df["resolved_yes"] != "") &
    (df["p_yes_model"].notna()) &
    (df["p_market"].notna()) &
    (df["would_pnl"].notna())
)
yes = df[mask].copy()
yes["resolved_yes"] = pd.to_numeric(yes["resolved_yes"], errors="coerce")
yes = yes[yes["resolved_yes"].isin([0, 1])].copy()
yes["logged_at"] = pd.to_datetime(yes["logged_at"])
yes = yes.sort_values("logged_at").reset_index(drop=True)

N = len(yes)
split = int(N * 0.70)
train = yes.iloc[:split].copy()
test  = yes.iloc[split:].copy()

print(f"Total resolved YES: {N}")
print(f"Train: {len(train)}  ({train['logged_at'].min().date()} → {train['logged_at'].max().date()})")
print(f"Test:  {len(test)}   ({test['logged_at'].min().date()} → {test['logged_at'].max().date()})")
print()

# ── Helper functions ───────────────────────────────────────────────────────
def kalshi_fee(pm):
    return KALSHI_FEE_RATE * min(pm, 1 - pm)

def net_edge(p_model, pm):
    return (p_model - pm) - DEFAULT_SLIPPAGE - DEFAULT_SPREAD - kalshi_fee(pm)

def kelly_bet(p_model, pm):
    b = (1 - pm) / pm if pm > 0 else 0
    if b <= 0:
        return 0.0
    kf   = max(0.0, (b * p_model - (1 - p_model)) / b)
    frac = min(kf * KELLY_MULT, KELLY_CAP)
    return round(BANKROLL * frac, 2)

def sim_pnl(rows, p_col="p_yes_model"):
    total, bets, wins = 0.0, 0, 0
    for _, r in rows.iterrows():
        pm      = float(r["p_market"])
        p_model = float(r[p_col])
        won     = int(r["resolved_yes"]) == 1
        ne      = net_edge(p_model, pm)
        if ne < MIN_NET_EDGE:
            continue
        bet = kelly_bet(p_model, pm)
        fee = kalshi_fee(pm)
        if won:
            pnl = bet * (1 - pm) / pm * (1 - KALSHI_FEE_RATE)
        else:
            pnl = -bet
        total += pnl
        bets  += 1
        wins  += int(won)
    wr = wins / bets if bets else 0
    return total, bets, wr

# ── Fit Platt scaling on train set ────────────────────────────────────────
print(SEP)
print("PLATT SCALING FIT  (70% train)")
print(SEP)

p_raw_train = np.clip(train["p_yes_model"].values.astype(float), 0.01, 0.99)
y_train     = train["resolved_yes"].values.astype(float)

def neg_log_loss(params):
    A, B = params
    logit_p = logit(p_raw_train)
    p_cal   = expit(A + B * logit_p)
    p_cal   = np.clip(p_cal, 1e-7, 1 - 1e-7)
    return -np.mean(y_train * np.log(p_cal) + (1 - y_train) * np.log(1 - p_cal))

res = minimize(neg_log_loss, x0=[-0.5, 0.7], method="Nelder-Mead",
               options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
A_platt, B_platt = res.x
print(f"  Platt params:  A={A_platt:.4f}  B={B_platt:.4f}")
print(f"  Interpretation: p_cal = σ({A_platt:.3f} + {B_platt:.3f} × logit(p_raw))")
print(f"  B<1 means compression (model over-confident), B>1 means expansion")
print()

def platt_calibrate(p_raw):
    return float(expit(A_platt + B_platt * logit(np.clip(p_raw, 0.01, 0.99))))

# Apply to test set
test["p_platt"] = test["p_yes_model"].apply(
    lambda x: platt_calibrate(float(x)) if pd.notna(x) else np.nan
)

# ── Fit Isotonic Regression on train set ──────────────────────────────────
print(SEP)
print("ISOTONIC REGRESSION FIT  (70% train)")
print(SEP)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_raw_train, y_train)

def iso_calibrate(p_raw):
    return float(iso.predict([np.clip(p_raw, 0.01, 0.99)])[0])

test["p_iso"] = test["p_yes_model"].apply(
    lambda x: iso_calibrate(float(x)) if pd.notna(x) else np.nan
)
print(f"  Isotonic breakpoints: {len(iso.X_thresholds_)} knots")
print()

# ── Calibration quality: bias by pm bucket on TEST set ───────────────────
print(SEP)
print("CALIBRATION BIAS BY pm BUCKET  (test set only)")
print(SEP)
print(f"{'pm range':>14}  {'n':>5}  {'WR':>6}  {'p_raw bias':>10}  {'p_platt bias':>12}  {'p_iso bias':>10}")
print("-" * 70)

for lo, hi in [(0.00,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,1.0)]:
    sub = test[(test["p_market"]>=lo) & (test["p_market"]<hi)]
    if len(sub) < 5:
        continue
    wr       = sub["resolved_yes"].mean()
    raw_bias = sub["p_yes_model"].mean() - wr
    plt_bias = sub["p_platt"].mean() - wr
    iso_bias = sub["p_iso"].mean() - wr
    print(f"  [{lo:.2f},{hi:.2f})  {len(sub):5d}  {wr:.3f}  {raw_bias:>+10.3f}  {plt_bias:>+12.3f}  {iso_bias:>+10.3f}")

print()

# ── P&L comparison on TEST set ─────────────────────────────────────────────
print(SEP)
print("P&L COMPARISON  (test set, 30%)")
print(SEP)

pnl_raw,   n_raw,   wr_raw   = sim_pnl(test, "p_yes_model")
pnl_platt, n_platt, wr_platt = sim_pnl(test, "p_platt")
pnl_iso,   n_iso,   wr_iso   = sim_pnl(test, "p_iso")

print(f"  Uncalibrated:  P&L=${pnl_raw:>7.2f}  bets={n_raw:3d}  WR={wr_raw:.3f}")
print(f"  Platt scaled:  P&L=${pnl_platt:>7.2f}  bets={n_platt:3d}  WR={wr_platt:.3f}  delta=${pnl_platt-pnl_raw:+.2f}")
print(f"  Isotonic:      P&L=${pnl_iso:>7.2f}  bets={n_iso:3d}  WR={wr_iso:.3f}  delta=${pnl_iso-pnl_raw:+.2f}")
print()

# ── P&L by pm bucket on test set ──────────────────────────────────────────
print(SEP)
print("P&L BY pm BUCKET  (test set)")
print(SEP)
print(f"{'pm range':>14}  {'n':>5}  {'WR':>6}  {'raw P&L':>10}  {'platt P&L':>10}  {'iso P&L':>10}")
print("-" * 68)

for lo, hi in [(0.00,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,1.0)]:
    sub = test[(test["p_market"]>=lo) & (test["p_market"]<hi)]
    if len(sub) < 5:
        continue
    wr     = sub["resolved_yes"].mean()
    pr, _nr, _ = sim_pnl(sub, "p_yes_model")
    pp, _np, _ = sim_pnl(sub, "p_platt")
    pi, _ni, _ = sim_pnl(sub, "p_iso")
    print(f"  [{lo:.2f},{hi:.2f})  {len(sub):5d}  {wr:.3f}  {pr:>+10.2f}  {pp:>+10.2f}  {pi:>+10.2f}")

print()

# ── What do Platt params mean for each pm bucket ─────────────────────────
print(SEP)
print("PLATT EFFECT: avg p_raw → avg p_cal  (full dataset)")
print(SEP)
print(f"  {'p_raw':>7}  →  {'p_platt':>7}   (shift)")
for p_raw in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    p_cal = platt_calibrate(p_raw)
    print(f"    {p_raw:.2f}   →   {p_cal:.3f}   ({p_cal-p_raw:+.3f})")

print()
print(f"Platt params for implementation:  A={A_platt:.6f}  B={B_platt:.6f}")
print("Done.")
