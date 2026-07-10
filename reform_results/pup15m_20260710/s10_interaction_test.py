"""
S10 -- rigorous test of the S{1,6,7} x pup15m-oppose YES interaction
(user challenge: "seems gate worthy -- why is this not significant?").

The honest problem: the cells were picked from a 10x2 table AFTER seeing
outcomes. Tests, in order of strictness:
 1. The combined bucket's raw stats: WR/BE/$, weekly ledger, era split,
    episode-clustered bootstrap (as if it had been pre-registered).
 2. Candidate-level (archive) check of the same bucket, ticker-clustered.
 3. SELECTION-HONEST split-half: pick the worst-PnL cells using only H1
    trades, evaluate exactly those cells on H2 (blind). Repeat reversed.
 4. Multiplicity/permutation: circularly shift the cg30 state series by
    random offsets (preserves autocorrelation + the p15 join), recompute
    "sum of the 3 worst YES cells" each time -> how extreme is the observed
    -$587 under the null that states carry no interaction info?
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
REFORM = pd.Timestamp("2026-06-30", tz="UTC")

st = pd.read_csv(f"{OUT}/cg30m_states.csv", parse_dates=["bar_open", "effective"]).sort_values("effective")
sig = pd.read_csv(f"{OUT}/pup15m_series_2026.csv", parse_dates=["effective"]).sort_values("effective")

bk = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
bk["decision_time"] = pd.to_datetime(bk["decision_time"], utc=True, errors="coerce", format="mixed")
bk = bk[bk["side"].isin(["yes", "no"]) & (pd.to_numeric(bk["bet_amount"], errors="coerce") > 0)]
bk = bk.dropna(subset=["would_pnl", "resolved_yes", "decision_time"]).sort_values("decision_time")
bk = pd.merge_asof(bk, st[["effective", "cg30_state"]], left_on="decision_time",
                   right_on="effective", direction="backward").dropna(subset=["cg30_state"])
bk["cg30_state"] = bk["cg30_state"].astype(int)
bk = pd.merge_asof(bk.drop(columns=["effective"]), sig[["effective", "p15"]],
                   left_on="decision_time", right_on="effective", direction="backward").dropna(subset=["p15"])
bk["opp"] = bk["p15"] < 0.48
bk["win"] = np.where(bk["side"] == "yes", bk["resolved_yes"], 1 - bk["resolved_yes"])
bk["cost"] = np.where(bk["side"] == "yes", bk["p_market"], 1 - bk["p_market"])
bk["week"] = bk["decision_time"].dt.to_period("W").astype(str)
gap = bk["decision_time"].diff().dt.total_seconds() / 60
bk["episode"] = (gap.isna() | (gap > 45)).cumsum()
yes = bk[bk["side"] == "yes"].copy()
print(f"YES trades with both joins: {len(yes)}")

CELLS = {1, 6, 7}
b = yes[yes["cg30_state"].isin(CELLS) & yes["opp"]]
print(f"\n=== 1. combined bucket S{sorted(CELLS)} & p15<0.48 (as-if pre-registered) ===")
print(f"n={len(b)}  WR={b['win'].mean():.1%}  BE={b['cost'].mean():.1%}  "
      f"edge={b['win'].mean()-b['cost'].mean():+.3f}  $ {b['would_pnl'].sum():+.2f}")
print("weekly $ (block saves the negative):")
print((b.groupby("week")["would_pnl"].agg(["size", "sum"]).round(2)).to_string())
for era, m in [("pre", b["decision_time"] < REFORM), ("post", b["decision_time"] >= REFORM)]:
    e = b[m]
    print(f"  {era}: n={len(e)}  WR={e['win'].mean():.1%} BE={e['cost'].mean():.1%}  $ {e['would_pnl'].sum():+.2f}")
ep = b.groupby("episode")["would_pnl"].sum()
boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(4000)]
print(f"episodes={len(ep)}  episode-bootstrap P(bucket_pnl>=0) = {np.mean(np.array(boots) >= 0):.4f}")

print("\n=== 2. candidate level (archive), same bucket ===")
arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes"]).sort_values("logged_at")
arc = pd.merge_asof(arc, st[["effective", "cg30_state"]], left_on="logged_at",
                    right_on="effective", direction="backward").dropna(subset=["cg30_state"])
arc["cg30_state"] = arc["cg30_state"].astype(int)
arc = pd.merge_asof(arc.drop(columns=["effective"]), sig[["effective", "p15"]],
                    left_on="logged_at", right_on="effective", direction="backward").dropna(subset=["p15"])
arc["opp"] = arc["p15"] < 0.48
arc["week"] = arc["logged_at"].dt.to_period("W").astype(str)
ab = arc[arc["cg30_state"].isin(CELLS) & arc["opp"]]
e = (ab["resolved_yes"] - ab["p_market"])
g = ab.assign(edge=e).groupby("contract_ticker").agg(edge=("edge", "mean"), week=("week", "first"))
boots = [g["edge"].sample(frac=1, replace=True, random_state=i).mean() for i in range(2000)]
wk = g.groupby("week")["edge"].mean()
print(f"tickers={len(g)}  tk_edge(YES)={g['edge'].mean():+.4f}  "
      f"P(edge>=0)={np.mean(np.array(boots) >= 0):.4f}  wk_neg={int((wk < 0).sum())}/{wk.size}")

print("\n=== 3. selection-honest split-half ===")
mid = yes["decision_time"].median()
for name, sel_m, test_m in [("select H1 -> test H2", yes["decision_time"] <= mid, yes["decision_time"] > mid),
                            ("select H2 -> test H1", yes["decision_time"] > mid, yes["decision_time"] <= mid)]:
    selh = yes[sel_m]
    cell_pnl = selh.groupby(["cg30_state", "opp"])["would_pnl"].agg(["sum", "size"])
    cand = cell_pnl[(cell_pnl["size"] >= 5) & (cell_pnl["sum"] < 0)].nsmallest(3, "sum")
    picked = list(cand.index)
    testh = yes[test_m]
    tm = pd.Series(False, index=testh.index)
    for s_, o_ in picked:
        tm |= (testh["cg30_state"] == s_) & (testh["opp"] == o_)
    tb = testh[tm]
    if len(tb) == 0:
        print(f"  {name}: picked {picked} -> 0 test trades"); continue
    print(f"  {name}: picked {picked}")
    print(f"    test: n={len(tb)}  WR={tb['win'].mean():.1%} BE={tb['cost'].mean():.1%}  "
          f"$ {tb['would_pnl'].sum():+.2f}  (block would save {-tb['would_pnl'].sum():+.2f})")

print("\n=== 4. permutation (circular shift of state series), stat = sum of 3 worst YES cells ===")
def worst3(y):
    cp = y.groupby(["cg30_state", "opp"])["would_pnl"].agg(["sum", "size"])
    cp = cp[cp["size"] >= 5]
    return cp["sum"].nsmallest(3).sum()
obs = worst3(yes)
st_idx = st.reset_index(drop=True)
rng = np.random.default_rng(23)
null = []
for k in range(500):
    off = rng.integers(100, len(st_idx) - 100)
    sh = st_idx.copy()
    sh["cg30_state"] = np.roll(sh["cg30_state"].values, off)
    y2 = pd.merge_asof(yes.drop(columns=["cg30_state"]).sort_values("decision_time"),
                       sh[["effective", "cg30_state"]].sort_values("effective"),
                       left_on="decision_time", right_on="effective", direction="backward")
    null.append(worst3(y2))
null = np.array(null)
print(f"observed worst-3-cells sum = ${obs:+.2f}")
print(f"null (500 shifts): median ${np.median(null):+.2f}  5th pct ${np.percentile(null,5):+.2f}")
print(f"P(null <= observed) = {np.mean(null <= obs):.4f}   <- multiplicity-adjusted significance")
print("DONE_S10")
