"""
S23 -- does pup15m_sc (regime-conditioned) work better than p_up_v2 as the
drift input to the K_YES/K_NO pricing model? Direct substitution test:
same candidates (btc_scan_archive_15m.csv), same functions
(compute_p_yes_pup_v2_15m / compute_p_no_pup_v2_15m, K_YES=0.50/K_NO=0.30
imported live), only the drift probability argument differs:
  A) corrected p_up_v2 (the s22 backtest, reproduced here for a clean
     side-by-side on the IDENTICAL candidate subset)
  B) pup15m_sc (state-conditioned 15m signal, this session's build)
Both joined zero-lookahead (effective time <= logged_at).
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

hourly = pd.read_csv("results/paper_trades.csv", usecols=["logged_at", "p_up_v2"], low_memory=False)
hourly["p_up_v2"] = pd.to_numeric(hourly["p_up_v2"], errors="coerce")
hourly["logged_at"] = pd.to_datetime(hourly["logged_at"], utc=True, errors="coerce", format="mixed")
hourly = hourly.dropna(subset=["p_up_v2", "logged_at"]).sort_values("logged_at").drop_duplicates("logged_at", keep="last")

sc = pd.read_csv(f"{OUT}/pup15m_sc_series_2026.csv", parse_dates=["effective"]).sort_values("effective")

arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "resolved_yes", "realized_vol_annual"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "tau_minutes",
                         "spot", "strike", "realized_vol_annual"])
arc = arc[(arc["realized_vol_annual"] > 0) & (arc["logged_at"] >= "2026-06-01")].sort_values("logged_at")

arc = pd.merge_asof(arc, hourly.rename(columns={"logged_at": "pu_ts"}),
                    left_on="logged_at", right_on="pu_ts", direction="backward",
                    tolerance=pd.Timedelta("120min"))
arc = pd.merge_asof(arc.sort_values("logged_at"), sc[["effective", "p_sc"]],
                    left_on="logged_at", right_on="effective", direction="backward")
# common population: BOTH signals available (fair side-by-side, no coverage-window bias)
common = arc.dropna(subset=["p_up_v2", "p_sc"]).copy()
print(f"common candidate population (both signals fresh): {len(common)} rows, "
      f"{common['contract_ticker'].nunique()} tickers")
print(f"window: {common['logged_at'].min()} -> {common['logged_at'].max()}")

def run_model(df, drift_col, label):
    d = df.copy()
    p_yes, p_no = [], []
    for row in d.itertuples(index=False):
        sig = {"realized_vol_annual": row.realized_vol_annual}
        drift = getattr(row, drift_col)
        py = R.compute_p_yes_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, drift, row.p_market)
        pn = R.compute_p_no_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, drift, row.p_market)
        p_yes.append(py); p_no.append(pn)
    d["p_model_yes"], d["p_model_no"] = p_yes, p_no
    d["edge_yes"] = d["p_model_yes"] - d["p_market"]
    d["edge_no"] = d["p_model_no"] - (1 - d["p_market"])
    d["best_side"] = np.where(d["edge_yes"] >= d["edge_no"], "yes", "no")
    d["best_edge"] = np.where(d["edge_yes"] >= d["edge_no"], d["edge_yes"], d["edge_no"])
    qual = d[d["best_edge"] >= EDGE_THRESH].copy()
    per_ticker = qual.sort_values("best_edge", ascending=False).drop_duplicates("contract_ticker")
    per_ticker["win"] = np.where(per_ticker["best_side"] == "yes", per_ticker["resolved_yes"],
                                 1 - per_ticker["resolved_yes"])
    per_ticker["cost"] = np.where(per_ticker["best_side"] == "yes", per_ticker["p_market"],
                                  1 - per_ticker["p_market"])
    per_ticker["day"] = per_ticker["logged_at"].dt.date
    n = len(per_ticker)
    if n == 0:
        print(f"\n=== {label}: 0 qualifying trades ===")
        return per_ticker
    wr, be = per_ticker["win"].mean(), per_ticker["cost"].mean()
    print(f"\n=== {label} ===")
    print(f"  qualifying candidates: {len(qual)} of {len(d)}  |  trades (1/ticker): {n}")
    print(f"  side mix: {per_ticker['best_side'].value_counts().to_dict()}")
    print(f"  overall: WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.4f}")
    for side, g in per_ticker.groupby("best_side"):
        print(f"    {side}: n={len(g):3d}  WR={g['win'].mean():.3f}  BE={g['cost'].mean():.3f}  "
              f"edge={g['win'].mean()-g['cost'].mean():+.4f}")
    ep = per_ticker.set_index("logged_at").sort_index()
    gap = ep.index.to_series().diff().dt.total_seconds() / 60
    ep["episode"] = (gap.isna() | (gap > 45)).cumsum()
    ep["pnl_proxy"] = np.where(ep["win"] == 1, 1 - ep["cost"], -ep["cost"])  # $1 stake proxy
    epsum = ep.groupby("episode")["pnl_proxy"].sum()
    rng = np.random.default_rng(13)
    boots = [epsum.sample(frac=1, replace=True, random_state=i).sum() for i in range(2000)]
    print(f"  episode-clustered ($1-stake proxy) total={epsum.sum():+.2f}  P(<=0)={np.mean(np.array(boots)<=0):.4f}")
    return per_ticker

t_v2 = run_model(common, "p_up_v2", "A) p_up_v2 (corrected fetch)")
t_sc = run_model(common, "p_sc", "B) pup15m_sc (regime-conditioned)")

t_v2.to_csv(f"{OUT}/s23_trades_pupv2.csv", index=False)
t_sc.to_csv(f"{OUT}/s23_trades_pupscsc.csv", index=False)
print("\nDONE_S23")
