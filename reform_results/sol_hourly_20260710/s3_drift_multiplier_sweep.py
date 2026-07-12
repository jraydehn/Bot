"""
S3 -- sweep SOL's DRIFT_MULTIPLIER (currently 0.20 vs BTC's 1.40) against the
real hourly candidate population (results/sol_scan_archive.csv, 05-21 to
07-11, n~160k rows / ~7 weeks). For each candidate row, recompute p_model at
different k_drift values using the EXACT live formula (score_to_p_model),
holding composite_p_up / spot / strike / tau fixed, and score against the
actual resolved_yes outcome (Brier + a simple flat-stake PnL proxy: bet the
side p_model favors when it clears p_market by a fixed margin).

This directly tests whether SOL's multiplier is mis-set relative to its
(now confirmed, s2) BTC-comparable signal strength -- ground-truth-anchored
against the real candidate book, not a synthetic reconstruction.
"""
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "composite_p_up", "vol_eff", "resolved_yes"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "spot", "strike", "p_market", "tau_minutes",
                        "composite_p_up", "vol_eff", "resolved_yes"])
df = df[(df["vol_eff"] > 0) & (df["tau_minutes"] > 0) & (df["composite_p_up"].between(0.001, 0.999))]
print(f"candidates: {len(df)}  tickers: {df['contract_ticker'].nunique()}  "
      f"range: {df['logged_at'].min()} -> {df['logged_at'].max()}")

df["sigma_tau"] = df["vol_eff"] * np.sqrt(df["tau_minutes"])
df = df[df["sigma_tau"] > 0]
df["z_strike"] = np.log(df["strike"] / df["spot"]) / df["sigma_tau"]
df["z_pup"] = norm.ppf(df["composite_p_up"].clip(0.001, 0.999))

# ticker-level clustering for stats (avoid pseudo-replication across the same hour's strike ladder)
def tk_brier(p, y, ticker):
    t = pd.DataFrame({"p": p, "y": y, "tk": ticker})
    g = t.groupby("tk").agg(p=("p", "mean"), y=("y", "mean"))
    return float(np.mean((g["p"] - g["y"]) ** 2)), len(g)


print(f"\n=== DRIFT_MULTIPLIER sweep (ticker-clustered Brier score, lower=better) ===")
print(f"{'k_drift':>8s} {'brier':>8s} {'tickers':>8s}")
for k in [0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 2.00]:
    z_drift = df["z_pup"] * k
    z_adj = df["z_strike"] - z_drift
    p_model = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    brier, ntk = tk_brier(p_model, df["resolved_yes"], df["contract_ticker"])
    marker = " <- LIVE" if k == 0.20 else (" <- BTC's value" if k == 1.40 else "")
    print(f"{k:8.2f} {brier:8.4f} {ntk:8d}{marker}")

print(f"\n=== $ PnL proxy: bet side model favors when edge > margin, flat $100 stake ===")
print(f"{'k_drift':>8s} {'n_bets':>7s} {'WR':>7s} {'BE':>7s} {'edge':>8s} {'total_$':>10s} {'$/bet':>8s}")
for k in [0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 2.00]:
    z_drift = df["z_pup"] * k
    z_adj = df["z_strike"] - z_drift
    p_model = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    edge_yes = p_model - df["p_market"]
    edge_no = (1 - p_model) - (1 - df["p_market"])
    MARGIN = 0.03
    take_yes = edge_yes > MARGIN
    take_no = edge_no > MARGIN
    bets = []
    if take_yes.sum() > 0:
        sub = df[take_yes]
        bets.append(pd.DataFrame({"win": sub["resolved_yes"], "cost": sub["p_market"], "tk": sub["contract_ticker"]}))
    if take_no.sum() > 0:
        sub = df[take_no]
        bets.append(pd.DataFrame({"win": 1 - sub["resolved_yes"], "cost": 1 - sub["p_market"], "tk": sub["contract_ticker"]}))
    if not bets:
        print(f"{k:8.2f}  (no bets)")
        continue
    allbets = pd.concat(bets)
    tk = allbets.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    marker = " <- LIVE" if k == 0.20 else (" <- BTC's value" if k == 1.40 else "")
    print(f"{k:8.2f} {len(tk):7d} {tk['win'].mean():7.1%} {tk['cost'].mean():7.1%} "
          f"{tk['win'].mean()-tk['cost'].mean():+8.4f} {pnl.sum():10.2f} {pnl.sum()/len(tk):8.2f}{marker}")

print("\nDONE_S3")
