"""
Sweep the p_up_v3 "agreement" cutoff instead of assuming the fixed 0.50 split.

Two designs tested:
  A) Shifted single threshold T: YES agrees iff v3>=T, NO agrees iff v3<T.
  B) Symmetric deadband margin d around 0.50: YES agrees iff v3>=0.50+d,
     NO agrees iff v3<=0.50-d, everything else is NEUTRAL (neither agrees
     nor disagrees) -- tests whether requiring more conviction from v3
     (not just direction) sharpens the split, at the cost of a growing
     neutral bucket that would need its own kept/dropped decision.

Read-only against paper_trades_btc15m.csv; writes results to this dir only.
"""
import pandas as pd
import numpy as np

bf = pd.read_csv("reform_results/pup_v3_backfill_20260706/btc_15m_backfilled.csv", low_memory=False)
bf["logged_at_parsed"] = pd.to_datetime(bf["logged_at_parsed"], utc=True, errors="coerce")

raw = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
raw["logged_at_parsed"] = pd.to_datetime(raw["logged_at"], format="mixed", utc=True, errors="coerce")

df = bf.merge(raw[["logged_at_parsed", "p_market"]], on="logged_at_parsed", how="left")
df["would_win"] = df["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")
df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
df = df.dropna(subset=["p_up_v3_backfill"])
df["week"] = df["logged_at_parsed"].dt.isocalendar().week
print(f"total covered trades: {len(df)}  (weeks {sorted(df['week'].unique())})")


def breakeven(sub):
    # per-trade implied breakeven: yes needs pm, no needs 1-pm
    be = np.where(sub["side"].str.lower() == "yes", sub["p_market"], 1 - sub["p_market"])
    return np.nanmean(be)


def bucket_stats(sub, label):
    if len(sub) == 0:
        return f"{label:10s} n=   0"
    wr = sub["would_win"].mean()
    pnl = sub["would_pnl"].sum()
    be = breakeven(sub)
    nweeks = sub["week"].nunique()
    return f"{label:10s} n={len(sub):4d}  WR={wr:.3f}  BE={be:.3f}  edge={wr-be:+.3f}  PnL=${pnl:8.2f}  weeks={nweeks}"


print("\n=== Design A: shifted single threshold (YES agrees v3>=T, NO agrees v3<T) ===")
for T in np.arange(0.44, 0.58, 0.01):
    agree = df[np.where(df["side"].str.lower() == "yes", df["p_up_v3_backfill"] >= T, df["p_up_v3_backfill"] < T)]
    disagree = df[~df.index.isin(agree.index)]
    print(f"T={T:.2f}  " + bucket_stats(agree, "AGREE") + "   |   " + bucket_stats(disagree, "DISAGREE"))

print("\n=== Design B: symmetric deadband margin d (neutral zone widens with d) ===")
for d in np.arange(0.00, 0.09, 0.01):
    is_yes = df["side"].str.lower() == "yes"
    agree = df[(is_yes & (df["p_up_v3_backfill"] >= 0.50 + d)) | (~is_yes & (df["p_up_v3_backfill"] <= 0.50 - d))]
    disagree = df[(is_yes & (df["p_up_v3_backfill"] < 0.50 - d)) | (~is_yes & (df["p_up_v3_backfill"] > 0.50 + d))]
    neutral = df[~df.index.isin(agree.index) & ~df.index.isin(disagree.index)]
    print(f"d={d:.2f}  " + bucket_stats(agree, "AGREE") + "   |   " + bucket_stats(disagree, "DISAGREE")
          + "   |   " + bucket_stats(neutral, "NEUTRAL"))
