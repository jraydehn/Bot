"""
simulate_pmkt_base.py
---------------------
Simulate replacing the log-normal base probability with p_market as the
base, while keeping the same composite drift and gate stack.

Current model:
  z_base = log(K/S) / sigma_tau        ← log-normal
  p_model = Φ(-z_base + z_drift)       ← for YES
  p_no    = Φ(z_base - z_drift_no)     ← for NO (independent, k=0.30)

Market-base model:
  z_base = Φ⁻¹(1 − p_market)          ← implied from market price
  p_no_mkt = Φ(z_base − z_drift_no)   ← NO: same drift applied to market z

Drift extraction (NO):
  p_no_logged = raw_edge + (1 − p_market)
  z_drift_no  = z_lognorm − Φ⁻¹(p_no_logged)
  z_mkt       = Φ⁻¹(1 − p_market)     ← market's implied NO z
  p_no_mkt    = Φ(z_mkt − z_drift_no)

For YES (no independent logged model in BTC 1h backtest — uses p_model column):
  p_yes_logged = p_model (CSV column)
  z_drift_yes  = Φ⁻¹(1 − p_yes_logged) − z_lognorm
  p_yes_mkt    = 1 − Φ(Φ⁻¹(1−pm) − (−z_drift_yes))
               = 1 − Φ(z_mkt − z_drift_total_yes)

tau approximation: trades happen ~5 min after open → tau ≈ 55 min
sigma_tau ≈ vol_60m × sqrt(55/60)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

STAKE         = 50.0
EDGE_THRESH   = 0.04   # same threshold as live model
TAU_MIN       = 55.0   # approximate minutes to expiry at trade time

def sim_pmkt_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sigma_tau: vol_60m is per-minute vol; scale to tau_minutes
    df["sigma_tau"] = df["vol_60m"] * np.sqrt(TAU_MIN)
    df["sigma_tau"] = df["sigma_tau"].clip(lower=1e-6)

    # log-normal z: log(K/S) / sigma (positive = OTM above spot)
    df["z_lognorm"] = np.log(df["strike"] / df["spot"]) / df["sigma_tau"]

    pm    = df["p_market"].clip(1e-4, 1-1e-4)
    rake  = df["raw_edge"] - df["net_edge"]
    z_mkt = norm.ppf(1 - pm)   # market-implied z (Φ⁻¹(p_no_mkt))

    # ── NO trades (vectorized) ────────────────────────────────────────────
    no_mask = df["side"].str.lower() == "no"
    p_no_log = (df["raw_edge"] + (1 - pm)).clip(1e-4, 1-1e-4)
    z_no_adj  = norm.ppf(p_no_log)
    z_drift_no = df["z_lognorm"] - z_no_adj
    p_no_mkt   = norm.cdf(z_mkt - z_drift_no)
    new_raw_no = p_no_mkt - (1 - pm)
    new_net_no = new_raw_no - rake

    # ── YES trades (vectorized) ───────────────────────────────────────────
    p_yes_log  = df["p_model"].clip(1e-4, 1-1e-4)
    z_yes_adj  = norm.ppf(1 - p_yes_log)
    z_drift_yes = df["z_lognorm"] - z_yes_adj
    p_yes_mkt  = 1 - norm.cdf(z_mkt - z_drift_yes)
    new_raw_yes = p_yes_mkt - pm
    new_net_yes = new_raw_yes - rake

    df["new_net_edge"] = np.where(no_mask, new_net_no, new_net_yes)
    df["new_raw_edge"] = np.where(no_mask, new_raw_no, new_raw_yes)
    return df


def report(label, sub, new_thresh=EDGE_THRESH):
    base_trades  = sub[sub["net_edge"] >= EDGE_THRESH]
    new_trades   = sub[sub["new_net_edge"] >= new_thresh]
    both_trades  = sub[(sub["net_edge"] >= EDGE_THRESH) & (sub["new_net_edge"] >= new_thresh)]
    gained       = sub[(sub["net_edge"] < EDGE_THRESH) & (sub["new_net_edge"] >= new_thresh)]
    lost         = sub[(sub["net_edge"] >= EDGE_THRESH) & (sub["new_net_edge"] < new_thresh)]

    def flat_pnl(df):
        if len(df) == 0: return 0.0
        wins = (df["win"] == 1).sum()
        losses = len(df) - wins
        return wins * STAKE * 0.93 - losses * STAKE  # ~7% rake on wins

    def wr(df): return df["win"].mean() if len(df) else float("nan")

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Baseline  (current model): n={len(base_trades):>6,}  WR={wr(base_trades):.1%}  PnL=${flat_pnl(base_trades):>+12,.0f}")
    print(f"  Mkt-base  (new   model)  : n={len(new_trades):>6,}  WR={wr(new_trades):.1%}  PnL=${flat_pnl(new_trades):>+12,.0f}")
    print(f"  Both pass                : n={len(both_trades):>6,}  WR={wr(both_trades):.1%}  PnL=${flat_pnl(both_trades):>+12,.0f}")
    print(f"  Gained by mkt-base       : n={len(gained):>6,}  WR={wr(gained):.1%}  PnL=${flat_pnl(gained):>+12,.0f}  (new adds these)")
    print(f"  Lost by mkt-base         : n={len(lost):>6,}  WR={wr(lost):.1%}  PnL=${flat_pnl(lost):>+12,.0f}  (new drops these)")


def main():
    print("Loading backtest_full.csv (BTC 1h, all rows)...")
    df = pd.read_csv("results/backtest_full.csv")
    df["win"] = (df["win"] == 1) | (df["win"] == True)
    df["win"] = df["win"].astype(int)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    print(f"  Total rows: {len(df):,}  |  gate_passed=True: {df['gate_passed'].sum():,}")
    print("  Running market-base simulation...", end="", flush=True)
    df = sim_pmkt_base(df)
    print(" done.")

    # Only look at rows where gate_passed=True (gates already evaluated)
    gp = df[df["gate_passed"] == True].copy()

    # Overall
    report("BTC 1h — ALL sides (gate_passed=True)", gp)

    # By side
    for side in ["no", "yes"]:
        sub = gp[gp["side"] == side]
        if len(sub) > 0:
            report(f"BTC 1h — {side.upper()} only (gate_passed=True)", sub)

    # Fold breakdown for NO (the main side)
    no_rows = gp[gp["side"] == "no"].copy()
    fold_defs = [
        ("F1 2024H2", "2024-07-01", "2025-01-01"),
        ("F2 2025H1", "2025-01-01", "2025-07-01"),
        ("F3 2025H2", "2025-07-01", "2026-01-01"),
        ("F4 2026 YTD","2026-01-01", "2030-01-01"),
    ]
    print(f"\n{'='*60}")
    print("  NO side walk-forward fold breakdown")
    print(f"{'='*60}")
    print(f"  {'Fold':<12}  {'Base n':>7}  {'Base WR':>7}  {'Base PnL':>10}  {'Mkt n':>6}  {'Mkt WR':>6}  {'Mkt PnL':>10}  {'DELTA':>9}")
    for label, ts, te in fold_defs:
        mask = (no_rows["ts"] >= ts) & (no_rows["ts"] < te)
        fold = no_rows[mask]
        if len(fold) < 10: continue
        base = fold[fold["net_edge"]     >= EDGE_THRESH]
        mkt  = fold[fold["new_net_edge"] >= EDGE_THRESH]
        def fp(d):
            if len(d)==0: return 0.0
            return d["win"].sum()*STAKE*0.93 - (len(d)-d["win"].sum())*STAKE
        def wr2(d): return d["win"].mean() if len(d) else float("nan")
        delta = fp(mkt) - fp(base)
        print(f"  {label:<12}  {len(base):>7,}  {wr2(base):>7.1%}  ${fp(base):>+10,.0f}  {len(mkt):>6,}  {wr2(mkt):>6.1%}  ${fp(mkt):>+10,.0f}  ${delta:>+9,.0f}")

    # Edge distribution comparison
    print(f"\n{'='*60}")
    print("  Edge distribution (NO trades, gate_passed=True)")
    print(f"{'='*60}")
    no_g = gp[gp["side"] == "no"]
    print(f"  Current net_edge:   mean={no_g['net_edge'].mean():.4f}  std={no_g['net_edge'].std():.4f}  "
          f"min={no_g['net_edge'].min():.4f}  max={no_g['net_edge'].max():.4f}")
    print(f"  Mkt-base net_edge:  mean={no_g['new_net_edge'].mean():.4f}  std={no_g['new_net_edge'].std():.4f}  "
          f"min={no_g['new_net_edge'].min():.4f}  max={no_g['new_net_edge'].max():.4f}")
    corr = no_g[["net_edge","new_net_edge"]].corr().iloc[0,1]
    print(f"  Correlation (current vs mkt-base): {corr:.4f}")


if __name__ == "__main__":
    main()
