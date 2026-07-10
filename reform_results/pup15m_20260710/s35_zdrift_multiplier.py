"""
S35 -- test multiplying raw z_drift by a fraction instead of hard-capping.
Preserves the SHAPE of the drift signal (mild vs strong trend stay
distinguishable) rather than collapsing everything above 0.5 into one value
(the saturation problem: 73.8% of decisions hit the exact same +0.5).

Found + fixed a data-quality bug before testing: 3+ rows on 06-27 16:26-16:57
UTC have spot_at_expiry=2000.0 (same corrupted-CSV window flagged earlier
today for p_up_v2_btc="Bear" contamination) -- a 97% single-cycle "move"
that never happened, producing z values of -3772 to -4882. Filtered with a
sanity bound (|log(expiry/spot)| < 5%, generous vs BTC's real ~10min moves)
before reconstructing the drift trajectory.

Design: z_drift = clip(raw, -SAFETY, +SAFETY) * fraction. The safety clip
guards against real (non-corrupted) but still-extreme readings; the
fraction does the actual dampening work, preserving relative ordering
below that bound. Ground-truth-anchored test (s34's method): for the
REAL 317 taken YES trades, would this trade still qualify, and was it a
winner or loser?
"""
import warnings
import sys
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
OUT = "reform_results/pup15m_20260710"

import paper_trade_runner_15m as R  # noqa: E402

EDGE_THRESH = 0.04
SAFETY = 5.0  # generous bound against data-quality outliers, not a dampening mechanism
FRACTIONS = [1.0, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
res = df.dropna(subset=["resolved_yes", "spot", "realized_vol_annual", "tau_minutes",
                        "spot_at_expiry", "decision_time"]).copy()
res = res[(res["spot"] > 0) & (res["realized_vol_annual"] > 0) & (res["tau_minutes"] > 0) & (res["spot_at_expiry"] > 0)]
# data-quality filter: no real ~10min BTC move exceeds 5% (removes the 06-27 corruption)
log_move = np.log(res["spot_at_expiry"] / res["spot"])
n_before = len(res)
res = res[log_move.abs() < 0.05]
print(f"data-quality filter removed {n_before - len(res)} corrupted rows (06-27 spot_at_expiry=2000 glitch)")

res = res.sort_values("decision_time").reset_index(drop=True)
sigma_tau = res["realized_vol_annual"] * np.sqrt(res["tau_minutes"] / 525600.0)
res["actual_z"] = np.log(res["spot_at_expiry"] / res["spot"]) / sigma_tau
z_short = res["actual_z"].rolling(10).mean()
z_long = res["actual_z"].rolling(30, min_periods=30).mean()
res["z_drift_raw"] = 0.6 * z_short + 0.4 * z_long
zseries = res.dropna(subset=["z_drift_raw"])[["decision_time", "z_drift_raw"]].drop_duplicates("decision_time")
print(f"clean z_drift_raw trajectory: {len(zseries)} points  "
      f"median={zseries['z_drift_raw'].median():+.3f}  p99={zseries['z_drift_raw'].quantile(.99):+.3f}  "
      f"max_abs={zseries['z_drift_raw'].abs().max():.3f}")

t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes", "spot", "floor_strike", "tau_minutes", "realized_vol_annual"])
t = t.sort_values("decision_time")
t = pd.merge_asof(t, zseries.sort_values("decision_time"), on="decision_time", direction="backward")
t = t.dropna(subset=["z_drift_raw"])
t["win"] = t["resolved_yes"]
print(f"\nreal taken YES trades: {len(t)}")
gap = t["decision_time"].diff().dt.total_seconds() / 60
t["episode"] = (gap.isna() | (gap > 45)).cumsum()

print(f"\n{'frac':>5s} {'still_qual':>10s} {'filtered':>9s} {'kept_WR':>8s} {'kept_$':>9s} "
      f"{'net_$_delta':>12s} {'saturated%(|zd|>=0.45)':>24s}")
rows = {}
for frac in FRACTIONS:
    p_yes, zd_used = [], []
    for row in t.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        zd = float(np.clip(row.z_drift_raw, -SAFETY, SAFETY)) * frac
        zd_used.append(zd)
        py = R.compute_p_yes_zdrift_15m(row.spot, row.floor_strike, row.tau_minutes, sig, zd, row.p_market)
        p_yes.append(py)
    tt = t.copy()
    tt["zd_used"] = zd_used
    tt["p_model_yes_new"] = p_yes
    tt["edge_new"] = tt["p_model_yes_new"] - tt["p_market"]
    tt["still_qual"] = tt["edge_new"] >= EDGE_THRESH
    kept = tt[tt["still_qual"]]
    filtered = tt[~tt["still_qual"]]
    net_delta = -filtered["would_pnl"].sum()
    sat_pct = (tt["zd_used"].abs() >= 0.45).mean()
    rows[frac] = tt
    print(f"{frac:5.2f} {len(kept):10d} {len(filtered):9d} "
          f"{kept['win'].mean() if len(kept) else float('nan'):8.1%} "
          f"{kept['would_pnl'].sum() if len(kept) else 0:9.2f} "
          f"{net_delta:+12.2f} {sat_pct:24.1%}")

print("\n=== filtered-trade weekly detail for the most promising fractions ===")
for frac in [0.3, 0.25, 0.2]:
    tt = rows[frac]
    filt = tt[~tt["still_qual"]]
    if len(filt) == 0:
        print(f"  frac={frac}: nothing filtered")
        continue
    ep = filt.groupby("episode")["would_pnl"].sum()
    boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(3000)]
    print(f"\n  frac={frac}: {len(filt)} filtered, pnl sum={filt['would_pnl'].sum():+.2f} "
          f"(net delta {-filt['would_pnl'].sum():+.2f})  "
          f"P(filtering hurts)={np.mean(np.array(boots)>0):.4f}")
    wk = filt.groupby(filt["decision_time"].dt.to_period("W"))["would_pnl"].sum()
    print(f"    weekly: {wk.round(0).to_dict()}")

for frac in FRACTIONS:
    rows[frac].to_csv(f"{OUT}/s35_frac_{str(frac).replace('.','p')}.csv", index=False)
print("\nDONE_S35")
