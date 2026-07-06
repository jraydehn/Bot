"""
S6 -- Does p_w15 do better on the subset of real trades where the EXISTING
gate stack has no strong directional opinion of its own, vs the subset
where it does? Tests the hypothesis that the model's real AUC is being
buried by fighting against gates that already selected the population,
rather than by the model itself being useless.

Two proxies for "existing directional conviction", both fully populated:
  1. ema_bias (-1/0/+1): the runner's own simple trend lens. 0 = neutral,
     i.e. no existing trend opinion.
  2. raw_edge magnitude (terciles): low edge = a marginal/borderline bet
     the existing system wasn't strongly convicted about either.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr

df = pd.read_csv("reform_results/pup_v3_15m_window_sweep_20260706/btc_15m_w15_backfilled.csv", low_memory=False)
df["would_win"] = df["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")
is_yes = df["side"].str.lower() == "yes"
df["conviction"] = np.where(is_yes, df["p_w15"] - 0.50, 0.50 - df["p_w15"])


def test(sub, label):
    sub = sub.dropna(subset=["conviction", "would_pnl"])
    if len(sub) < 20:
        print(f"{label}: n={len(sub)} (too thin)")
        return
    ic_pnl = spearmanr(sub["conviction"], sub["would_pnl"]).statistic
    ic_win = pointbiserialr(sub["would_win"].astype(float), sub["conviction"]).statistic
    agree = sub[sub["conviction"] > 0]
    disagree = sub[sub["conviction"] <= 0]
    print(f"{label}: n={len(sub)}  corr(conv,pnl)={ic_pnl:+.4f}  corr(conv,win)={ic_win:+.4f}")
    print(f"    agree(n={len(agree):3d}) WR={agree['would_win'].mean():.3f} pnl=${agree['would_pnl'].sum():8.2f}   "
          f"disagree(n={len(disagree):3d}) WR={disagree['would_win'].mean():.3f} pnl=${disagree['would_pnl'].sum():8.2f}")


print("=== Split 1: ema_bias (existing trend lens) ===")
test(df[df["ema_bias"] == 0], "ema_bias=0 (NO existing trend opinion)")
test(df[df["ema_bias"] != 0], "ema_bias!=0 (existing trend opinion present)")

print("\n=== Split 2: raw_edge terciles (existing conviction strength) ===")
df["edge_tercile"] = pd.qcut(df["raw_edge"], 3, labels=["low", "mid", "high"])
for t in ["low", "mid", "high"]:
    test(df[df["edge_tercile"] == t], f"raw_edge={t}")

print("\n=== Combined: ema_bias=0 AND low raw_edge (weakest existing opinion) ===")
test(df[(df["ema_bias"] == 0) & (df["edge_tercile"] == "low")], "ema_bias=0 & low edge")
