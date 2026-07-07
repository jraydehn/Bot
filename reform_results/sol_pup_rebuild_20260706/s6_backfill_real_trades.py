"""
S6 -- Backfill sol_p_up_v1 against REAL taken SOL trades, using the honest
walk-forward OOS predictions (wf_preds_A.parquet, the winning final config)
causally joined to actual trade timestamps. Same methodology as BTC/ETH.

Archive chain: paper_trades_sol.csv itself already covers 2026-04-15 through
present continuously (never rotated after that point) -- confirmed by
checking date ranges of all rotated archives before including them. The
20260420_* archives are subsets already contained in the current file and
are NOT included (would just be redundant, deduped anyway but no reason to
carry the extra rows).
"""
import numpy as np
import pandas as pd

OUT = "reform_results/sol_pup_rebuild_20260706"
NEEDED = ["logged_at", "decision", "side", "would_win", "would_pnl", "p_market", "contract_ticker"]

CHAIN = [
    "results/paper_trades_sol_archive_20260407_122844.csv",
    "results/paper_trades_sol_archive_20260407_152310.csv",
    "results/paper_trades_sol_archive_20260415_1342_precal.csv",
    "results/paper_trades_sol.csv",
]


def load_chain():
    frames = []
    for p in CHAIN:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        use = [c for c in NEEDED if c in cols]
        df = pd.read_csv(p, usecols=use, low_memory=False)
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["logged_at_parsed", "contract_ticker"], keep="first")
    return full.sort_values("logged_at_parsed")


raw = load_chain()
print(f"combined SOL archive: {len(raw)} rows ({raw['logged_at_parsed'].min()} -> {raw['logged_at_parsed'].max()})")

wf = pd.read_parquet(f"{OUT}/wf_preds_A.parquet").dropna(subset=["p"])
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
taken["p_sol"] = taken["logged_at_parsed"].apply(lookup_p)
taken["would_win"] = taken["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
taken["would_pnl"] = pd.to_numeric(taken["would_pnl"], errors="coerce")
taken["p_market"] = pd.to_numeric(taken["p_market"], errors="coerce")
taken["side"] = taken["side"].str.lower()
taken["week"] = taken["logged_at_parsed"].dt.isocalendar().week
taken["yw"] = (taken["logged_at_parsed"].dt.isocalendar().year.astype(str) + "-W" +
              taken["week"].astype(str).str.zfill(2))
covered = taken.dropna(subset=["p_sol", "would_pnl"]).copy()
covered["be"] = np.where(covered["side"] == "yes", covered["p_market"], 1 - covered["p_market"])
is_yes = covered["side"] == "yes"
covered["agree"] = np.where(is_yes, covered["p_sol"] >= 0.50, covered["p_sol"] < 0.50)

print(f"\ntaken trades: {len(taken)}, covered by backfill: {len(covered)}")
print(f"coverage window: {covered['logged_at_parsed'].min()} -> {covered['logged_at_parsed'].max()}")


def report(sub, label):
    if len(sub) == 0:
        print(f"{label}: n=0"); return
    wr = sub["would_win"].mean(); be = sub["be"].mean(); pnl = sub["would_pnl"].sum()
    wk_pnl = sub.groupby("yw")["would_pnl"].sum()
    worst_share = (wk_pnl.abs().max() / wk_pnl.abs().sum() * 100) if wk_pnl.abs().sum() > 0 else np.nan
    print(f"{label:10s} n={len(sub):4d}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  "
          f"PnL=${pnl:8.2f}  weeks={len(wk_pnl)}  worst_wk_share={worst_share:.1f}%")


print("\n=== Agreement filter (p>=0.50 threshold) ===")
report(covered, "ALL")
report(covered[covered["agree"]], "AGREE")
report(covered[~covered["agree"]], "DISAGREE")

print("\n=== By side ===")
for side in ["yes", "no"]:
    sub = covered[covered["side"] == side]
    print(f"--- {side.upper()} --- n={len(sub)}")
    report(sub, "ALL")
    report(sub[sub["agree"]], "AGREE")
    report(sub[~sub["agree"]], "DISAGREE")

print("\n=== Per-week breakdown ===")
for wk in sorted(covered["yw"].unique()):
    wkdf = covered[covered["yw"] == wk]
    a = wkdf[wkdf["agree"]]; d = wkdf[~wkdf["agree"]]
    a_pnl = a["would_pnl"].sum() if len(a) else 0.0
    d_pnl = d["would_pnl"].sum() if len(d) else 0.0
    print(f"  {wk}: agree n={len(a):3d} pnl=${a_pnl:8.2f}   disagree n={len(d):3d} pnl=${d_pnl:8.2f}   "
          f"unfiltered=${wkdf['would_pnl'].sum():8.2f}")

covered.to_csv(f"{OUT}/sol_pup_v1_backfilled.csv", index=False)
print(f"\nsaved {OUT}/sol_pup_v1_backfilled.csv")
