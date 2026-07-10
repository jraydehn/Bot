"""
S24 -- widen the pup15m_sc-as-K_YES/K_NO-drift test beyond the 9-day window
s23 was accidentally constrained to (by p_up_v2's coverage + a real gap in
btc_scan_archive_15m.csv: 6,229 of 9,413 rows have unparseable logged_at,
wiping out ~06-08->06-28). pup15m_sc doesn't need p_up_v2 to be present, so
evaluate it over EVERY clean week (05-18->06-07 AND 06-29->07-10) --
critically, the earlier weeks do NOT overlap with the days this whole
session has been focused on, so they're a genuine held-out check on
whether s23's marginal result (P=0.079) replicates or was a fluke of a
short, attention-biased window.
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

K_YES, K_NO = R.K_PUP_V2_YES, R.K_PUP_V2_NO
EDGE_THRESH = 0.04

sc = pd.read_csv(f"{OUT}/pup15m_sc_series_2026.csv", parse_dates=["effective"]).sort_values("effective")

arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "resolved_yes", "realized_vol_annual"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "tau_minutes",
                         "spot", "strike", "realized_vol_annual"])
arc = arc[arc["realized_vol_annual"] > 0].sort_values("logged_at")
print(f"clean candidate rows (all weeks): {len(arc)}  "
      f"{arc['logged_at'].min()} -> {arc['logged_at'].max()}")
arc["week"] = arc["logged_at"].dt.to_period("W").astype(str)
print("rows per week:", arc["week"].value_counts().sort_index().to_dict())

arc = pd.merge_asof(arc, sc[["effective", "p_sc"]], left_on="logged_at",
                    right_on="effective", direction="backward")
arc = arc.dropna(subset=["p_sc"])
print(f"with pup15m_sc joined: {len(arc)}")

p_yes, p_no = [], []
for row in arc.itertuples(index=False):
    sig = {"realized_vol_annual": row.realized_vol_annual}
    py = R.compute_p_yes_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, row.p_sc, row.p_market)
    pn = R.compute_p_no_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, row.p_sc, row.p_market)
    p_yes.append(py); p_no.append(pn)
arc["p_model_yes"], arc["p_model_no"] = p_yes, p_no
arc["edge_yes"] = arc["p_model_yes"] - arc["p_market"]
arc["edge_no"] = arc["p_model_no"] - (1 - arc["p_market"])
arc["best_side"] = np.where(arc["edge_yes"] >= arc["edge_no"], "yes", "no")
arc["best_edge"] = np.where(arc["edge_yes"] >= arc["edge_no"], arc["edge_yes"], arc["edge_no"])

qual = arc[arc["best_edge"] >= EDGE_THRESH].copy()
per_ticker = qual.sort_values("best_edge", ascending=False).drop_duplicates("contract_ticker")
per_ticker["win"] = np.where(per_ticker["best_side"] == "yes", per_ticker["resolved_yes"],
                             1 - per_ticker["resolved_yes"])
per_ticker["cost"] = np.where(per_ticker["best_side"] == "yes", per_ticker["p_market"],
                              1 - per_ticker["p_market"])
per_ticker["week"] = per_ticker["logged_at"].dt.to_period("W").astype(str)
print(f"\ntotal simulated trades: {len(per_ticker)}")
print(f"overall: WR={per_ticker['win'].mean():.3f}  BE={per_ticker['cost'].mean():.3f}  "
      f"edge={per_ticker['win'].mean()-per_ticker['cost'].mean():+.4f}")

print("\n=== per-week breakdown (the whole point: does the EARLY, non-overlapping era agree?) ===")
for wk, g in per_ticker.groupby("week"):
    print(f"  {wk}: n={len(g):3d}  side={g['best_side'].value_counts().to_dict()}  "
          f"WR={g['win'].mean():.3f}  BE={g['cost'].mean():.3f}  edge={g['win'].mean()-g['cost'].mean():+.4f}")

# split: pre-07-01 (genuinely held-out from today's focus) vs 07-01+ (s23's window)
per_ticker["era"] = np.where(per_ticker["logged_at"] < pd.Timestamp("2026-07-01", tz="UTC"),
                             "pre-07-01 (held-out)", "07-01+ (s23 window)")
print("\n=== era comparison ===")
rng = np.random.default_rng(29)
for era, g in per_ticker.groupby("era"):
    if len(g) < 10:
        print(f"  {era}: n={len(g)} too thin"); continue
    g2 = g.set_index("logged_at").sort_index()
    gap = g2.index.to_series().diff().dt.total_seconds() / 60
    g2["episode"] = (gap.isna() | (gap > 45)).cumsum()
    g2["pnl_proxy"] = np.where(g2["win"] == 1, 1 - g2["cost"], -g2["cost"])
    epsum = g2.groupby("episode")["pnl_proxy"].sum()
    boots = [epsum.sample(frac=1, replace=True, random_state=i).sum() for i in range(2000)]
    print(f"  {era}: n={len(g):3d}  WR={g['win'].mean():.3f} BE={g['cost'].mean():.3f} "
          f"edge={g['win'].mean()-g['cost'].mean():+.4f}  $1-proxy total={epsum.sum():+.2f}  "
          f"P(<=0)={np.mean(np.array(boots)<=0):.4f}")

per_ticker.to_csv(f"{OUT}/s24_trades_pup15msc_wide.csv", index=False)
print("DONE_S24")
