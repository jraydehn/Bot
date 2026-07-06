"""
S14 -- Genuinely comprehensive rescue search for the two p_up_v3 HMM
gates. s13 tested 19 hand-picked signals and found nothing; this tests
EVERY usable signal column in paper_trades.csv (~140 candidates after
excluding pure metadata/bookkeeping/outcome columns), at a fine
decile grid, in both directions, plus every boolean/categorical column
by direct category comparison, plus pairwise combinations of the
single best individual splits per population.

Excluded (not real candidate signals): logged_at, decision_time,
contract_ticker, close_ts, p_market_source, decision, side (the filter
itself), gate_blocked, kelly_fraction, bet_fraction, bet_amount,
bankroll, contracts_scanned (bookkeeping); resolved_yes, would_win,
would_pnl, spot_at_expiry, price_move_pct, miss_pct, loss_margin_pct,
loss_category (these ARE the outcome, or derived post-hoc from it --
using them would be circular, not a real rescue signal).
"""
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
EXCLUDE = {
    "logged_at", "decision_time", "contract_ticker", "close_ts", "p_market_source",
    "decision", "side", "gate_blocked", "kelly_fraction", "bet_fraction", "bet_amount",
    "bankroll", "contracts_scanned", "resolved_yes", "would_win", "would_pnl",
    "spot_at_expiry", "price_move_pct", "miss_pct", "loss_margin_pct", "loss_category",
    "logged_at_parsed", "state3", "week", "yw", "be", "asset", "spot", "strike",
}
CHAIN = [
    "results/paper_trades_archive_20260415_1342_precal.csv",
    "results/paper_trades_archive_20260525_1432_pre_branched_drift.csv",
    "results/paper_trades_pre_regime_pup_20260616.csv",
    "results/paper_trades.csv",
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
        if "asset" not in df.columns:
            df["asset"] = "BTC"
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[full["asset"] == "BTC"]
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    return full.sort_values("logged_at_parsed"), sorted(all_cols)


raw, ALL_COLS = load_chain()
CANDIDATE_COLS = [c for c in ALL_COLS if c not in EXCLUDE]
print(f"total columns available: {len(ALL_COLS)}, candidate signal columns to test: {len(CANDIDATE_COLS)}")

states = pd.read_parquet(f"{OUT}/pup_v3_hmm_states.parquet")
bar_index = states.index
state3_series = states["state"].map({0: "neutral", 1: "rising", 2: "neutral", 3: "crashing"})


def lookup_state(logged_at):
    if pd.isna(logged_at):
        return np.nan
    idx = bar_index.searchsorted(logged_at, side="right") - 1
    if idx < 0 or idx >= len(bar_index):
        return np.nan
    if (logged_at - bar_index[idx]) > pd.Timedelta(hours=2):
        return np.nan
    return state3_series.iloc[idx]


taken = raw[raw["decision"] == "trade"].copy()
taken["state3"] = taken["logged_at_parsed"].apply(lookup_state)
taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")
taken["side"] = taken["side"].str.lower()
taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
taken["yw"] = (taken["logged_at_parsed"].dt.isocalendar().year.astype(str) + "-W" +
              taken["week"].astype(str).str.zfill(2))
covered = taken.dropna(subset=["state3", "would_pnl"]).copy()
covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])


def sweep(pop, label):
    print(f"\n{'='*90}\nRESCUE SEARCH: {label}\n"
          f"blocked pop n={len(pop)}, base WR={pop['would_win'].mean():.3f}, "
          f"base BE={pop['be'].mean():.3f}, base PnL=${pop['would_pnl'].sum():.2f}\n{'='*90}")
    found = []
    n_tests = 0
    for feat in CANDIDATE_COLS:
        if feat not in pop.columns:
            continue
        col = pop[feat]
        # boolean / low-cardinality categorical: compare each category directly
        if col.dtype == bool or col.dropna().nunique() <= 6:
            for val in col.dropna().unique():
                mask = col == val
                rescued = pop[mask]
                remainder = pop[~mask]
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
                             "n_remainder": len(remainder), "pnl_remainder": remainder["would_pnl"].sum()})
            continue
        # continuous: fine decile grid, both directions
        vals = pd.to_numeric(col, errors="coerce")
        sub = pop[vals.notna()].copy()
        if len(sub) < 20:
            continue
        vv = vals.dropna()
        for q in np.arange(0.1, 1.0, 0.1):
            thresh = vv.quantile(q)
            for direction, mask in [(">=", vv >= thresh), ("<", vv < thresh)]:
                n_tests += 1
                rescued = sub.loc[mask.index[mask]]
                remainder = sub.loc[mask.index[~mask]]
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
                             "edge_remainder": rem_edge, "n_remainder": len(remainder),
                             "pnl_remainder": remainder["would_pnl"].sum()})
    df_found = pd.DataFrame(found)
    print(f"total splits tested: {n_tests}  (across {len(CANDIDATE_COLS)} candidate columns)")
    real_rescues = df_found[(df_found["edge_rescued"] > 0) & (df_found["n_rescued"] >= 15)]
    print(f"splits where rescued subset is ACTUALLY above breakeven (edge>0, n>=15): {len(real_rescues)}")
    if len(real_rescues):
        top = real_rescues.sort_values("edge_rescued", ascending=False).head(15)
        print(top.round(3).to_string(index=False))
    else:
        top10 = df_found.sort_values("edge_rescued", ascending=False).head(10)
        print("\nBest 10 splits found (none clear breakeven) for reference:")
        print(top10.round(3).to_string(index=False))
    return df_found


rising_yes = covered[(covered["state3"] == "rising") & (covered["side"] == "yes")]
crashing_no = covered[(covered["state3"] == "crashing") & (covered["side"] == "no")]

f1 = sweep(rising_yes, "hmm_pup_v3_rising_yes_gate blocked population")
f2 = sweep(crashing_no, "hmm_pup_v3_crashing_no_gate blocked population")

f1.to_csv(f"{OUT}/rescue_sweep_rising_yes_all.csv", index=False)
f2.to_csv(f"{OUT}/rescue_sweep_crashing_no_all.csv", index=False)
