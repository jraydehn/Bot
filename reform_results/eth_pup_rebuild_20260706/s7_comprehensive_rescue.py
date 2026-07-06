"""
S7 -- Comprehensive rescue search for the ETH p_up_v1 AGREE bucket
(n=514, WR=57.4%, -$3,782 -- the losing side of the contrarian finding).
Same exhaustive methodology as the BTC p_up_v3 gates (s14): every usable
signal column in paper_trades_eth.csv, fine decile grid, both directions,
categorical handling for low-cardinality columns.
"""
import numpy as np
import pandas as pd

OUT = "reform_results/eth_pup_rebuild_20260706"
EXCLUDE = {
    "logged_at", "decision_time", "contract_ticker", "close_ts", "p_market_source",
    "decision", "side", "gate_blocked", "kelly_fraction", "bet_fraction", "bet_amount",
    "bankroll", "contracts_scanned", "resolved_yes", "would_win", "would_pnl",
    "spot_at_expiry", "price_move_pct", "miss_pct", "loss_margin_pct", "loss_category",
    "logged_at_parsed", "week", "yw", "be", "agree", "p_eth", "asset", "spot", "strike",
}
CHAIN = [
    "results/paper_trades_eth_archive_20260415_1342_precal.csv",
    "results/paper_trades_eth.csv",
]


def load_chain():
    frames = []
    all_cols = set()
    for p in CHAIN:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        all_cols.update(cols)
    for p in CHAIN:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        df = pd.read_csv(p, usecols=cols, low_memory=False)
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    return full.sort_values("logged_at_parsed"), sorted(all_cols)


raw, ALL_COLS = load_chain()
CANDIDATE_COLS = [c for c in ALL_COLS if c not in EXCLUDE]
print(f"total columns: {len(ALL_COLS)}, candidate signal columns: {len(CANDIDATE_COLS)}")

wf = pd.read_parquet(f"{OUT}/wf_preds_AC.parquet").dropna(subset=["p"])
bar_index = wf.index
p_series = wf["p"]


def lookup_p(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (logged_at - bar_index[idx]) > pd.Timedelta(hours=2):
        return np.nan
    return float(p_series.iloc[idx])


taken = raw[raw["decision"] == "trade"].copy()
taken["p_eth"] = taken["logged_at_parsed"].apply(lookup_p)
taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")
taken["side"] = taken["side"].str.lower()
taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
taken["yw"] = (taken["logged_at_parsed"].dt.isocalendar().year.astype(str) + "-W" +
              taken["week"].astype(str).str.zfill(2))
covered = taken.dropna(subset=["p_eth", "would_pnl"]).copy()
covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])
is_yes = covered["side"] == "yes"
covered["agree"] = np.where(is_yes, covered["p_eth"] >= 0.50, covered["p_eth"] < 0.50)

agree_pop = covered[covered["agree"]].copy()
print(f"\nAGREE blocked population: n={len(agree_pop)}, base WR={agree_pop['would_win'].mean():.3f}, "
      f"base BE={agree_pop['be'].mean():.3f}, base PnL=${agree_pop['would_pnl'].sum():.2f}")

found = []
n_tests = 0
for feat in CANDIDATE_COLS:
    if feat not in agree_pop.columns:
        continue
    col = agree_pop[feat]
    if col.dtype == bool or col.dropna().nunique() <= 6:
        for val in col.dropna().unique():
            mask = col == val
            rescued = agree_pop[mask]; remainder = agree_pop[~mask]
            n_tests += 1
            if len(rescued) < 10 or len(remainder) < 10:
                continue
            r_edge = rescued["would_win"].mean() - rescued["be"].mean()
            rem_edge = remainder["would_win"].mean() - remainder["be"].mean()
            wk_pnl = rescued.groupby("yw")["would_pnl"].sum()
            worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
            found.append({"feature": feat, "split": f"=={val}", "n_rescued": len(rescued),
                         "wr_rescued": rescued["would_win"].mean(), "edge_rescued": r_edge,
                         "pnl_rescued": rescued["would_pnl"].sum(), "n_weeks": len(wk_pnl),
                         "worst_wk_share": worst_share, "edge_remainder": rem_edge,
                         "pnl_remainder": remainder["would_pnl"].sum()})
        continue
    vals = pd.to_numeric(col, errors="coerce")
    sub = agree_pop[vals.notna()].copy()
    if len(sub) < 20:
        continue
    vv = vals.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        thresh = vv.quantile(q)
        for direction, mask in [(">=", vv >= thresh), ("<", vv < thresh)]:
            n_tests += 1
            rescued = sub.loc[mask.index[mask]]; remainder = sub.loc[mask.index[~mask]]
            if len(rescued) < 10 or len(remainder) < 10:
                continue
            r_edge = rescued["would_win"].mean() - rescued["be"].mean()
            rem_edge = remainder["would_win"].mean() - remainder["be"].mean()
            wk_pnl = rescued.groupby("yw")["would_pnl"].sum()
            worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
            found.append({"feature": feat, "split": f"{direction}{thresh:.4g}(q{q:.1f})",
                         "n_rescued": len(rescued), "wr_rescued": rescued["would_win"].mean(),
                         "edge_rescued": r_edge, "pnl_rescued": rescued["would_pnl"].sum(),
                         "n_weeks": len(wk_pnl), "worst_wk_share": worst_share,
                         "edge_remainder": rem_edge, "pnl_remainder": remainder["would_pnl"].sum()})

df_found = pd.DataFrame(found)
print(f"\ntotal splits tested: {n_tests}  (across {len(CANDIDATE_COLS)} candidate columns)")
real_rescues = df_found[(df_found["edge_rescued"] > 0) & (df_found["n_rescued"] >= 15)]
print(f"splits where rescued subset is ACTUALLY above breakeven (edge>0, n>=15): {len(real_rescues)}")
if len(real_rescues):
    print(real_rescues.sort_values("edge_rescued", ascending=False).head(20).round(3).to_string(index=False))
else:
    print("\nBest 10 for reference:")
    print(df_found.sort_values("edge_rescued", ascending=False).head(10).round(3).to_string(index=False))

df_found.to_csv(f"{OUT}/rescue_sweep_agree_all.csv", index=False)
covered.to_csv(f"{OUT}/eth_agree_pop_for_reconstruction.csv", index=False)
print(f"\nsaved sweep results and covered population for further reconstruction")
