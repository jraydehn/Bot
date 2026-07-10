"""
S6 -- backfill pup15m over the ENTIRE BTC 15m book (user request 2026-07-10):
does a pup15m-opposition gate actually help the 15m model?

Signal: pup15m_series_2026.csv -- tables trained 2023-10..2025-12, so every
2026 value is out-of-sample. Zero-lookahead join: effective (bar close) <=
decision_time / logged_at.

Book: results/paper_trades_btc15m.csv taken resolved trades (2026-05-25 ->
now, 911 trades, spans pre/post the 06-30 non-coherent reform).
Candidates: results/btc_scan_archive_15m.csv (9.4k resolved rows).

Evaluation per house rules: WR + BE + $ together; era split at 06-30;
weekly sign counts; episode-clustered bootstrap (15m book: episodes =
consecutive trades <=45min apart); gate sims report wins blocked + losses
blocked + net $ delta. Flat sizes (logged would_pnl, no compounding).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
REFORM = pd.Timestamp("2026-06-30", tz="UTC")

sig = pd.read_csv(f"{OUT}/pup15m_series_2026.csv", parse_dates=["effective"]).sort_values("effective")

bk = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
bk["decision_time"] = pd.to_datetime(bk["decision_time"], utc=True, errors="coerce", format="mixed")
bk = bk[bk["side"].isin(["yes", "no"]) & (pd.to_numeric(bk["bet_amount"], errors="coerce") > 0)]
bk = bk.dropna(subset=["would_pnl", "resolved_yes", "decision_time"]).sort_values("decision_time")
bk = pd.merge_asof(bk, sig[["effective", "p15"]], left_on="decision_time",
                   right_on="effective", direction="backward")
bk = bk.dropna(subset=["p15"])
bk["stale_min"] = (bk["decision_time"] - bk["effective"]).dt.total_seconds() / 60
bk["want"] = np.where(bk["side"] == "yes", 1, -1)
bk["support"] = (bk["p15"] - 0.5) * bk["want"]
bk["win"] = np.where(bk["side"] == "yes", bk["resolved_yes"], 1 - bk["resolved_yes"])
bk["cost"] = np.where(bk["side"] == "yes", bk["p_market"], 1 - bk["p_market"])
bk["era"] = np.where(bk["decision_time"] < REFORM, "pre", "post")
bk["week"] = bk["decision_time"].dt.to_period("W").astype(str)
gap = bk["decision_time"].diff().dt.total_seconds() / 60
bk["episode"] = (gap.isna() | (gap > 45)).cumsum()
print(f"joined taken trades: {len(bk)}  staleness median={bk['stale_min'].median():.1f}min "
      f"p95={bk['stale_min'].quantile(0.95):.1f}min")
print(f"eras: {bk['era'].value_counts().to_dict()}  episodes: {bk['episode'].nunique()}")

def bucket_report(sub, label):
    if len(sub) == 0:
        print(f"    {label:26s}: n=0"); return
    print(f"    {label:26s}: n={len(sub):4d}  WR={sub['win'].mean():.1%}  "
          f"BE={sub['cost'].mean():.1%}  edge={sub['win'].mean()-sub['cost'].mean():+.3f}  "
          f"$ {sub['would_pnl'].sum():+9.2f}")

BUCKETS = [("supports (>+0.02)", lambda s: s > 0.02),
           ("neutral [-0.02,+0.02]", lambda s: (s >= -0.02) & (s <= 0.02)),
           ("opposes (-0.05,-0.02)", lambda s: (s < -0.02) & (s >= -0.05)),
           ("strong opposes (<-0.05)", lambda s: s < -0.05)]

print("\n=== TAKEN BOOK: support buckets per side x era ===")
for side in ["yes", "no"]:
    for era in ["pre", "post"]:
        e = bk[(bk["side"] == side) & (bk["era"] == era)]
        print(f"  {side.upper()} {era}-reform (n={len(e)}, ${e['would_pnl'].sum():+.2f}):")
        for lb, f in BUCKETS:
            bucket_report(e[f(e["support"])], lb)

print("\n=== GATE SIM: block YES when support <= -T (whole book + per era) ===")
yes = bk[bk["side"] == "yes"]
rng = np.random.default_rng(17)
for T in [0.02, 0.03, 0.04, 0.05, 0.06]:
    b = yes[yes["support"] <= -T]
    if len(b) < 10:
        continue
    wins, losses = int(b["win"].sum()), int((1 - b["win"]).sum())
    saved = -b["would_pnl"].sum()
    ep = b.groupby("episode")["would_pnl"].sum()
    boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(3000)]
    p_le0 = float(np.mean(-np.array(boots) <= 0))
    wk = b.groupby("week")["would_pnl"].sum()
    pre_s = -b[b["era"] == "pre"]["would_pnl"].sum()
    post_s = -b[b["era"] == "post"]["would_pnl"].sum()
    print(f"  T={T:.2f}: blocked n={len(b):3d} (ep={ep.size})  wins_blk={wins:3d} loss_blk={losses:3d}  "
          f"net saved=${saved:+8.2f} (pre ${pre_s:+7.2f} / post ${post_s:+7.2f})  "
          f"P(saved<=0)={p_le0:.4f}  wk+={int((-wk > 0).sum())}/{wk.size}")

print("\n=== DAMPENER SIM: halve YES stake when support <= -T (delta vs existing) ===")
for T in [0.02, 0.04]:
    b = yes[yes["support"] <= -T]
    delta = -0.5 * b["would_pnl"].sum()   # halving stake halves pnl of those trades
    ep = b.groupby("episode")["would_pnl"].sum() * -0.5
    boots = [ep.sample(frac=1, replace=True, random_state=i).sum() for i in range(3000)]
    print(f"  T={T:.2f}: n={len(b)}  delta=${delta:+8.2f}  P(delta<=0)={float(np.mean(np.array(boots)<=0)):.4f}")

print("\n=== weekly ledger: existing vs block-YES@T=0.02 ===")
bk["gated_pnl"] = np.where((bk["side"] == "yes") & (bk["support"] <= -0.02), 0.0, bk["would_pnl"])
wk = bk.groupby("week").agg(n=("win", "size"), existing=("would_pnl", "sum"), gated=("gated_pnl", "sum")).round(2)
wk["delta"] = (wk["gated"] - wk["existing"]).round(2)
print(wk.to_string())
print(f"\nTOTAL: existing ${bk['would_pnl'].sum():+.2f}  gated ${bk['gated_pnl'].sum():+.2f}  "
      f"delta ${bk['gated_pnl'].sum()-bk['would_pnl'].sum():+.2f}")

# ---- candidate level, full archive ----
arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes"]).sort_values("logged_at")
arc = pd.merge_asof(arc, sig[["effective", "p15"]], left_on="logged_at",
                    right_on="effective", direction="backward").dropna(subset=["p15"])
arc["era"] = np.where(arc["logged_at"] < REFORM, "pre", "post")
arc["week"] = arc["logged_at"].dt.to_period("W").astype(str)
arc["supp_yes"] = arc["p15"] - 0.5
print(f"\n=== CANDIDATE LEVEL (archive, {len(arc)} rows, {arc['contract_ticker'].nunique()} tickers) ===")
def tk(sub, side):
    e = (sub["resolved_yes"] - sub["p_market"]) if side == "yes" else (sub["p_market"] - sub["resolved_yes"])
    g = sub.assign(edge=e).groupby("contract_ticker").agg(edge=("edge", "mean"), week=("week", "first"))
    if len(g) < 10:
        return None
    boots = [g["edge"].sample(frac=1, replace=True, random_state=i).mean() for i in range(2000)]
    wkk = g.groupby("week")["edge"].mean()
    return len(g), g["edge"].mean(), float(np.mean(np.array(boots) <= 0)), int((wkk > 0).sum()), wkk.size
for era in ["pre", "post"]:
    e = arc[arc["era"] == era]
    for lb, m in [("YES supports (>0.02)", e["supp_yes"] > 0.02),
                  ("YES neutral", (e["supp_yes"] >= -0.02) & (e["supp_yes"] <= 0.02)),
                  ("YES opposes (<-0.02)", e["supp_yes"] < -0.02)]:
        r = tk(e[m], "yes")
        if r:
            print(f"  {era:4s} {lb:22s}: tickers={r[0]:4d} tk_edge={r[1]:+.4f} P(<=0)={r[2]:.4f} wk+={r[3]}/{r[4]}")
print("DONE_S6")
