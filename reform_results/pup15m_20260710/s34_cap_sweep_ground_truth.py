"""
S34 -- s33 was invalid: re-deriving "best edge across all scan-archive
candidates" without the runner's full gate stack (flip_guard,
markov_1h_bear_gate, btc_15m_lowpm_gate, btc_yes_gate3, VWAP/momentum
gates, cooldowns, etc.) picked a completely different, much riskier
population (avg p_market 0.45 vs the real book's ~0.73) -- not usable.

Ground-truth-anchored fix: use the ACTUAL taken trades (all real gates
already applied, real p_market/strike/outcome), and ask only "would a
tighter cap have reduced this SPECIFIC trade's edge below the qualifying
threshold, and was this trade a winner or a loser?" This sidesteps the
candidate-selection replication problem entirely -- no new contract needs
to be picked, just: would this one still have qualified?

Caveat (disclosed, not hidden): this only tells us about filtering out
currently-taken YES trades. It does NOT model whether a tighter cap would
promote a currently-unqualified NO candidate on the same ticker into the
best-edge slot instead (p_model_no = 1-p_model_yes rises as p_yes falls)
-- that would need the full gate-stack replication, out of scope here.
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

CAPS = [0.50, 0.35, 0.25, 0.20, 0.15, 0.10]
EDGE_THRESH = 0.04

# causal raw z_drift trajectory (same as s33)
df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
res = df.dropna(subset=["resolved_yes", "spot", "realized_vol_annual", "tau_minutes",
                        "spot_at_expiry", "decision_time"]).copy()
res = res[(res["spot"] > 0) & (res["realized_vol_annual"] > 0) & (res["tau_minutes"] > 0) & (res["spot_at_expiry"] > 0)]
res = res.sort_values("decision_time").reset_index(drop=True)
sigma_tau = res["realized_vol_annual"] * np.sqrt(res["tau_minutes"] / 525600.0)
res["actual_z"] = np.log(res["spot_at_expiry"] / res["spot"]) / sigma_tau
z_short = res["actual_z"].rolling(10).mean()
z_long = res["actual_z"].rolling(30, min_periods=30).mean()
res["z_drift_raw"] = 0.6 * z_short + 0.4 * z_long
zseries = res.dropna(subset=["z_drift_raw"])[["decision_time", "z_drift_raw"]].drop_duplicates("decision_time")

# the actual taken YES book (all real gates already applied)
t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes", "spot", "floor_strike", "tau_minutes", "realized_vol_annual"])
t = t.sort_values("decision_time")
t = pd.merge_asof(t, zseries.sort_values("decision_time"), on="decision_time", direction="backward")
t = t.dropna(subset=["z_drift_raw"])
t["win"] = t["resolved_yes"]
print(f"real taken YES trades with z_drift_raw available: {len(t)}")
gap = t["decision_time"].diff().dt.total_seconds() / 60
t["episode"] = (gap.isna() | (gap > 45)).cumsum()

print(f"\n{'cap':>5s} {'still_qual':>10s} {'filtered':>9s} {'kept_WR':>8s} {'kept_$':>9s} "
      f"{'filtered_WR':>12s} {'filtered_$_forfeited':>20s} {'net_$_delta':>12s}")
for cap in CAPS:
    p_yes = []
    for row in t.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        zd = float(np.clip(row.z_drift_raw, -cap, cap))
        py = R.compute_p_yes_zdrift_15m(row.spot, row.floor_strike, row.tau_minutes, sig, zd, row.p_market)
        p_yes.append(py)
    tt = t.copy()
    tt["p_model_yes_new"] = p_yes
    tt["edge_new"] = tt["p_model_yes_new"] - tt["p_market"]
    tt["still_qual"] = tt["edge_new"] >= EDGE_THRESH
    kept = tt[tt["still_qual"]]
    filtered = tt[~tt["still_qual"]]
    net_delta = -filtered["would_pnl"].sum()  # removing these trades changes book pnl by -sum(their pnl)
    print(f"{cap:5.2f} {len(kept):10d} {len(filtered):9d} "
          f"{kept['win'].mean() if len(kept) else float('nan'):8.1%} "
          f"{kept['would_pnl'].sum() if len(kept) else 0:9.2f} "
          f"{filtered['win'].mean() if len(filtered) else float('nan'):12.1%} "
          f"{filtered['would_pnl'].sum() if len(filtered) else 0:20.2f} "
          f"{net_delta:+12.2f}")

# episode-clustered significance of the filtered-trades pnl sign, for the two most realistic caps
print("\n=== filtered-trade detail + significance, cap=0.35 and cap=0.25 ===")
for cap in [0.35, 0.25]:
    p_yes = []
    for row in t.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        zd = float(np.clip(row.z_drift_raw, -cap, cap))
        py = R.compute_p_yes_zdrift_15m(row.spot, row.floor_strike, row.tau_minutes, sig, zd, row.p_market)
        p_yes.append(py)
    tt = t.copy()
    tt["p_model_yes_new"] = p_yes
    tt["edge_new"] = tt["p_model_yes_new"] - tt["p_market"]
    tt["still_qual"] = tt["edge_new"] >= EDGE_THRESH
    filt = tt[~tt["still_qual"]]
    print(f"\n  cap={cap}: {len(filt)} trades filtered out")
    if len(filt) == 0:
        continue
    ep = filt.groupby("episode")["would_pnl"].sum()
    boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(3000)]
    print(f"    their would_pnl sum: {filt['would_pnl'].sum():+.2f}  "
          f"(removing them changes the book by {-filt['would_pnl'].sum():+.2f})")
    print(f"    episode-clustered P(their pnl was net positive, i.e. filtering HURTS)="
          f"{np.mean(np.array(boots)>0):.4f}")
    filt["day"] = filt["decision_time"].dt.date
    wk = filt.groupby(filt["decision_time"].dt.to_period("W"))["would_pnl"].sum()
    print(f"    weekly breakdown of filtered trades' pnl: {wk.round(0).to_dict()}")
print("DONE_S34")
