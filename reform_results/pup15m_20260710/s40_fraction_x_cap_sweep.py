"""
S40 -- 2D sweep: fraction x hard-cap, testing the user's original idea
("~0.75 factor, still hard-capped") against what actually got deployed
(0.28 factor, only a wide 5.0 safety bound -- not a real cap).

Design: zd = clip( clip(raw, -SAFETY, SAFETY) * frac , -cap, +cap ).
SAFETY=5.0 guards only against data-quality outliers (unchanged from s35).
cap is the REAL behavioral ceiling the user is asking about -- applied
AFTER the fraction, so it only binds on genuinely extreme readings, not
the everyday case (this preserves the differentiation the whole reform
was for, unlike capping BEFORE scaling which would reintroduce the
original saturation problem).

Same ground-truth-anchored method as s34/s35: real taken YES trades
(all actual gates already applied), recompute p_model_yes under each
(frac, cap), check whether the trade still clears the qualifying edge
threshold, and what removing disqualified trades does to real $ PnL.
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
FRACTIONS = [1.0, 0.75, 0.6, 0.5, 0.4, 0.3, 0.28]
CAPS = [0.35, 0.50, 0.75, 1.0, 5.0]  # 5.0 == "no real cap" (matches deployed config)

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
res = df.dropna(subset=["resolved_yes", "spot", "realized_vol_annual", "tau_minutes",
                        "spot_at_expiry", "decision_time"]).copy()
res = res[(res["spot"] > 0) & (res["realized_vol_annual"] > 0) & (res["tau_minutes"] > 0) & (res["spot_at_expiry"] > 0)]
log_move = np.log(res["spot_at_expiry"] / res["spot"])
res = res[log_move.abs() < 0.05]  # data-quality filter, same as s35

res = res.sort_values("decision_time").reset_index(drop=True)
sigma_tau = res["realized_vol_annual"] * np.sqrt(res["tau_minutes"] / 525600.0)
res["actual_z"] = np.log(res["spot_at_expiry"] / res["spot"]) / sigma_tau
z_short = res["actual_z"].rolling(10).mean()
z_long = res["actual_z"].rolling(30, min_periods=30).mean()
res["z_drift_raw"] = 0.6 * z_short + 0.4 * z_long
zseries = res.dropna(subset=["z_drift_raw"])[["decision_time", "z_drift_raw"]].drop_duplicates("decision_time")
print(f"z_drift_raw trajectory: {len(zseries)} pts  median={zseries['z_drift_raw'].median():+.3f}  "
      f"p90={zseries['z_drift_raw'].abs().quantile(.90):.3f}  max_abs={zseries['z_drift_raw'].abs().max():.3f}")

t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes", "spot", "floor_strike", "tau_minutes", "realized_vol_annual"])
t = t.sort_values("decision_time")
t = pd.merge_asof(t, zseries.sort_values("decision_time"), on="decision_time", direction="backward")
t = t.dropna(subset=["z_drift_raw"])
t["win"] = t["resolved_yes"]
print(f"real taken YES trades: {len(t)}\n")

results = {}
header = f"{'frac':>5s} {'cap':>5s} {'kept_n':>7s} {'kept_WR':>8s} {'kept_$':>10s} {'filt_n':>7s} {'net_$_delta':>12s} {'saturated%':>11s}"
print(header)
print("-" * len(header))
for frac in FRACTIONS:
    for cap in CAPS:
        p_yes, zd_used = [], []
        for row in t.itertuples(index=False):
            sig = {"realized_vol_annual": row.realized_vol_annual}
            zd_raw_safe = float(np.clip(row.z_drift_raw, -SAFETY, SAFETY)) * frac
            zd = float(np.clip(zd_raw_safe, -cap, cap))
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
        sat_pct = (tt["zd_used"].abs() >= cap - 1e-6).mean()
        results[(frac, cap)] = tt
        marker = ""
        if frac == 0.28 and cap == 5.0:
            marker = "  <- DEPLOYED"
        elif frac == 0.75 and cap == 0.50:
            marker = "  <- user's original idea"
        print(f"{frac:5.2f} {cap:5.2f} {len(kept):7d} "
              f"{kept['win'].mean() if len(kept) else float('nan'):8.1%} "
              f"{kept['would_pnl'].sum() if len(kept) else 0:10.2f} "
              f"{len(filtered):7d} {net_delta:+12.2f} {sat_pct:10.1%}{marker}")
    print()

print("\n=== weekly stability check for the top candidates ===")
for frac, cap in [(0.28, 5.0), (0.75, 0.50), (0.6, 0.50), (0.5, 0.50)]:
    tt = results[(frac, cap)]
    kept = tt[tt["still_qual"]]
    if len(kept) == 0:
        continue
    kept = kept.copy()
    kept["week"] = kept["decision_time"].dt.to_period("W").astype(str)
    wk = kept.groupby("week").agg(n=("win", "size"), wr=("win", "mean"), pnl=("would_pnl", "sum"))
    print(f"\n  frac={frac} cap={cap}: total kept n={len(kept)}, total_pnl=${kept['would_pnl'].sum():.2f}")
    print(wk.to_string())

print("\nDONE_S40")
