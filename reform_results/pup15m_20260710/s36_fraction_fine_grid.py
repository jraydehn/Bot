"""
S36 -- finer grid between 0.25 and 0.30 (s35's two best fractions) to check
whether the peak sits precisely at 0.25 or somewhere in between, and how
sensitive the result is to the exact value. Same ground-truth-anchored
method as s35 (real 317 taken YES trades, cleaned z_drift_raw with the
06-27 corruption filter, safety_bound=5.0).
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
SAFETY = 5.0
FRACTIONS = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25]

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
res = df.dropna(subset=["resolved_yes", "spot", "realized_vol_annual", "tau_minutes",
                        "spot_at_expiry", "decision_time"]).copy()
res = res[(res["spot"] > 0) & (res["realized_vol_annual"] > 0) & (res["tau_minutes"] > 0) & (res["spot_at_expiry"] > 0)]
log_move = np.log(res["spot_at_expiry"] / res["spot"])
res = res[log_move.abs() < 0.05]  # 06-27 corruption filter
res = res.sort_values("decision_time").reset_index(drop=True)
sigma_tau = res["realized_vol_annual"] * np.sqrt(res["tau_minutes"] / 525600.0)
res["actual_z"] = np.log(res["spot_at_expiry"] / res["spot"]) / sigma_tau
z_short = res["actual_z"].rolling(10).mean()
z_long = res["actual_z"].rolling(30, min_periods=30).mean()
res["z_drift_raw"] = 0.6 * z_short + 0.4 * z_long
zseries = res.dropna(subset=["z_drift_raw"])[["decision_time", "z_drift_raw"]].drop_duplicates("decision_time")

t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes", "spot", "floor_strike", "tau_minutes", "realized_vol_annual"])
t = t.sort_values("decision_time")
t = pd.merge_asof(t, zseries.sort_values("decision_time"), on="decision_time", direction="backward")
t = t.dropna(subset=["z_drift_raw"])
t["win"] = t["resolved_yes"]
gap = t["decision_time"].diff().dt.total_seconds() / 60
t["episode"] = (gap.isna() | (gap > 45)).cumsum()
print(f"real taken YES trades: {len(t)}")

print(f"\n{'frac':>6s} {'still_qual':>10s} {'filtered':>9s} {'kept_WR':>8s} {'kept_$':>9s} "
      f"{'net_$_delta':>12s} {'P(hurts)':>9s}")
for frac in FRACTIONS:
    p_yes = []
    for row in t.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        zd = float(np.clip(row.z_drift_raw, -SAFETY, SAFETY)) * frac
        py = R.compute_p_yes_zdrift_15m(row.spot, row.floor_strike, row.tau_minutes, sig, zd, row.p_market)
        p_yes.append(py)
    tt = t.copy()
    tt["p_model_yes_new"] = p_yes
    tt["edge_new"] = tt["p_model_yes_new"] - tt["p_market"]
    tt["still_qual"] = tt["edge_new"] >= EDGE_THRESH
    kept = tt[tt["still_qual"]]
    filtered = tt[~tt["still_qual"]]
    net_delta = -filtered["would_pnl"].sum()
    if len(filtered):
        ep = filtered.groupby("episode")["would_pnl"].sum()
        boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(3000)]
        p_hurts = np.mean(np.array(boots) > 0)
    else:
        p_hurts = float("nan")
    print(f"{frac:6.3f} {len(kept):10d} {len(filtered):9d} "
          f"{kept['win'].mean() if len(kept) else float('nan'):8.1%} "
          f"{kept['would_pnl'].sum() if len(kept) else 0:9.2f} "
          f"{net_delta:+12.2f} {p_hurts:9.3f}")
print("DONE_S36")
