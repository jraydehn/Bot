"""
S4 -- s3 showed a flat/linear DRIFT_MULTIPLIER hurts SOL monotonically as it
rises (k=0 beats everything). s2 showed genuine, large, significant edge
concentrated in the TAIL buckets (rev>=4 -> +21.9pp, rev<=-4 -> -16.2pp)
while the neutral middle (rev in [-1,1], ~59% of hours) carries ~0 edge.

Hypothesis: a flat k applies drift to EVERY candidate uniformly, adding pure
noise to the ~85%+ of candidates sitting near p_up=0.5 while only the tail
candidates carry real signal. Test a GATED multiplier: k_drift=0 unless
|composite_p_up - 0.5| clears a threshold, then apply k_high only to that
subset. Sweep (threshold, k_high) jointly on the real candidate archive.
"""
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
df["sigma_tau"] = df["vol_eff"] * np.sqrt(df["tau_minutes"])
df = df[df["sigma_tau"] > 0]
df["z_strike"] = np.log(df["strike"] / df["spot"]) / df["sigma_tau"]
df["z_pup"] = norm.ppf(df["composite_p_up"].clip(0.001, 0.999))
df["pup_dev"] = (df["composite_p_up"] - 0.5).abs()
print(f"candidates: {len(df)}  tickers: {df['contract_ticker'].nunique()}")
print(f"pup_dev distribution: {df['pup_dev'].describe()}\n")


def tk_brier(p, y, ticker):
    t = pd.DataFrame({"p": p, "y": y, "tk": ticker})
    g = t.groupby("tk").agg(p=("p", "mean"), y=("y", "mean"))
    return float(np.mean((g["p"] - g["y"]) ** 2))


def sim_pnl(p_model, df_ref, margin=0.03):
    edge_yes = p_model - df_ref["p_market"]
    edge_no = (1 - p_model) - (1 - df_ref["p_market"])
    take_yes = edge_yes > margin
    take_no = edge_no > margin
    bets = []
    if take_yes.sum() > 0:
        sub = df_ref[take_yes]
        bets.append(pd.DataFrame({"win": sub["resolved_yes"], "cost": sub["p_market"], "tk": sub["contract_ticker"]}))
    if take_no.sum() > 0:
        sub = df_ref[take_no]
        bets.append(pd.DataFrame({"win": 1 - sub["resolved_yes"], "cost": 1 - sub["p_market"], "tk": sub["contract_ticker"]}))
    if not bets:
        return None
    allbets = pd.concat(bets)
    tk = allbets.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    return dict(n=len(tk), wr=tk["win"].mean(), be=tk["cost"].mean(), total=pnl.sum(), per_bet=pnl.sum() / len(tk))


baseline_p = np.clip(1 - norm.cdf(df["z_strike"]), 0.01, 0.99)  # k=0 reference
brier_base = tk_brier(baseline_p, df["resolved_yes"], df["contract_ticker"])
pnl_base = sim_pnl(baseline_p, df)
print(f"baseline (k=0, no drift at all): brier={brier_base:.4f}  "
      f"PnL n={pnl_base['n']} WR={pnl_base['wr']:.1%} BE={pnl_base['be']:.1%} total=${pnl_base['total']:.2f} $/bet={pnl_base['per_bet']:.2f}\n")

print(f"=== GATED multiplier sweep: k_drift=k_high only when |p_up-0.5| >= threshold, else k=0 ===")
print(f"{'thresh':>7s} {'k_high':>7s} {'n_gated':>8s} {'brier':>8s} {'n_bets':>7s} {'WR':>7s} {'BE':>7s} {'total_$':>10s} {'$/bet':>8s}")
for thresh in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    gated_mask = df["pup_dev"] >= thresh
    n_gated = gated_mask.sum()
    if n_gated < 100:
        continue
    for k_high in [0.5, 1.0, 1.4, 2.0, 3.0]:
        z_drift = np.where(gated_mask, df["z_pup"] * k_high, 0.0)
        z_adj = df["z_strike"] - z_drift
        p_model = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
        brier = tk_brier(p_model, df["resolved_yes"], df["contract_ticker"])
        pnl = sim_pnl(p_model, df)
        if pnl is None:
            continue
        print(f"{thresh:7.2f} {k_high:7.2f} {n_gated:8d} {brier:8.4f} {pnl['n']:7d} {pnl['wr']:7.1%} "
              f"{pnl['be']:7.1%} {pnl['total']:10.2f} {pnl['per_bet']:8.2f}")

print("\nDONE_S4")
