"""
S9 -- validate the 30m CG flow HMM against the BTC 15m book.
Zero-lookahead join (state effective = bar_open + 30m <= decision_time).
Per state x side: n/WR/BE/$ on the taken book (911 trades, era-split at the
06-30 reform, episode-clustered bootstrap for any candidate cell) and
ticker-clustered candidate edge on the scan archive. Split-half on time for
any cell that looks actionable. Orthogonality vs pup15m opposition.
NO deployment -- research only; gate candidates get presented with the
mandatory rescue-search step still to come.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
REFORM = pd.Timestamp("2026-06-30", tz="UTC")

st = pd.read_csv(f"{OUT}/cg30m_states.csv", parse_dates=["effective"]).sort_values("effective")
sig = pd.read_csv(f"{OUT}/pup15m_series_2026.csv", parse_dates=["effective"]).sort_values("effective")
LBL = {0: "longliq-drift", 1: "shortliq-mild", 2: "shortliq-mild2", 3: "LONG-CASCADE",
       4: "quiet", 5: "SHORT-SQUEEZE", 6: "neutral", 7: "longliq-buylean",
       8: "buy-cvd", 9: "sell-cvd"}

bk = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
bk["decision_time"] = pd.to_datetime(bk["decision_time"], utc=True, errors="coerce", format="mixed")
bk = bk[bk["side"].isin(["yes", "no"]) & (pd.to_numeric(bk["bet_amount"], errors="coerce") > 0)]
bk = bk.dropna(subset=["would_pnl", "resolved_yes", "decision_time"]).sort_values("decision_time")
bk = pd.merge_asof(bk, st[["effective", "cg30_state"]], left_on="decision_time",
                   right_on="effective", direction="backward").dropna(subset=["cg30_state"])
bk["cg30_state"] = bk["cg30_state"].astype(int)
bk = pd.merge_asof(bk.drop(columns=["effective"]), sig[["effective", "p15"]],
                   left_on="decision_time", right_on="effective", direction="backward")
bk["win"] = np.where(bk["side"] == "yes", bk["resolved_yes"], 1 - bk["resolved_yes"])
bk["cost"] = np.where(bk["side"] == "yes", bk["p_market"], 1 - bk["p_market"])
bk["era"] = np.where(bk["decision_time"] < REFORM, "pre", "post")
bk["week"] = bk["decision_time"].dt.to_period("W").astype(str)
gap = bk["decision_time"].diff().dt.total_seconds() / 60
bk["episode"] = (gap.isna() | (gap > 45)).cumsum()
print(f"taken trades joined: {len(bk)}")

print("\n=== TAKEN BOOK: state x side (both eras pooled, then era detail for big cells) ===")
for side in ["yes", "no"]:
    e = bk[bk["side"] == side]
    print(f" side={side.upper()} (n={len(e)}):")
    for s_, g in sorted(e.groupby("cg30_state"), key=lambda kv: kv[1]["would_pnl"].sum()):
        pre = g[g["era"] == "pre"]; post = g[g["era"] == "post"]
        print(f"   S{s_} {LBL[s_]:16s}: n={len(g):3d}  WR={g['win'].mean():.1%} BE={g['cost'].mean():.1%} "
              f"edge={g['win'].mean()-g['cost'].mean():+.3f}  $ {g['would_pnl'].sum():+8.2f}  "
              f"[pre n={len(pre)} ${pre['would_pnl'].sum():+7.2f} | post n={len(post)} ${post['would_pnl'].sum():+7.2f}]")

# candidate level on the archive
arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes"]).sort_values("logged_at")
arc = pd.merge_asof(arc, st[["effective", "cg30_state"]], left_on="logged_at",
                    right_on="effective", direction="backward").dropna(subset=["cg30_state"])
arc["cg30_state"] = arc["cg30_state"].astype(int)
arc["week"] = arc["logged_at"].dt.to_period("W").astype(str)
print(f"\n=== CANDIDATES: archive rows joined: {len(arc)}, tickers {arc['contract_ticker'].nunique()} ===")
def tkstats(sub, side):
    e = (sub["resolved_yes"] - sub["p_market"]) if side == "yes" else (sub["p_market"] - sub["resolved_yes"])
    g = sub.assign(edge=e).groupby("contract_ticker").agg(edge=("edge", "mean"), week=("week", "first"))
    if len(g) < 15:
        return None
    boots = [g["edge"].sample(frac=1, replace=True, random_state=i).mean() for i in range(2000)]
    p = float(np.mean(np.array(boots) <= 0))
    wk = g.groupby("week")["edge"].mean()
    return len(g), g["edge"].mean(), p, int((wk > 0).sum()), wk.size
for side in ["yes", "no"]:
    print(f" side={side.upper()} candidate edge per state:")
    for s_ in sorted(arc["cg30_state"].unique()):
        r = tkstats(arc[arc["cg30_state"] == s_], side)
        if r:
            flag = " <<<" if (r[2] <= 0.02 or r[2] >= 0.98) and r[0] >= 40 else ""
            print(f"   S{s_} {LBL[s_]:16s}: tickers={r[0]:4d} tk_edge={r[1]:+.4f} P(<=0)={r[2]:.4f} wk+={r[3]}/{r[4]}{flag}")

# split-half stability for flagged candidate cells (time halves)
print("\n=== split-half (time) for cells with pooled P<=0.02 or >=0.98 ===")
mid = arc["logged_at"].median()
for side in ["yes", "no"]:
    for s_ in sorted(arc["cg30_state"].unique()):
        sub = arc[arc["cg30_state"] == s_]
        r = tkstats(sub, side)
        if not r or not ((r[2] <= 0.02 or r[2] >= 0.98) and r[0] >= 40):
            continue
        for h, m in [("H1", sub["logged_at"] <= mid), ("H2", sub["logged_at"] > mid)]:
            rh = tkstats(sub[m], side)
            if rh:
                print(f"   {side.upper()} S{s_} {h}: tickers={rh[0]:4d} tk_edge={rh[1]:+.4f} P(<=0)={rh[2]:.4f} wk+={rh[3]}/{rh[4]}")

# orthogonality: does the state add anything beyond pup15m opposition?
print("\n=== orthogonality: YES trades, state cells vs pup15m support ===")
yes = bk[bk["side"] == "yes"].dropna(subset=["p15"])
yes["p15_opp"] = yes["p15"] < 0.48
ct = yes.groupby(["cg30_state", "p15_opp"]).agg(n=("win", "size"), wr=("win", "mean"),
                                                be=("cost", "mean"), pnl=("would_pnl", "sum")).round(3)
print(ct.to_string())
print("DONE_S9")
