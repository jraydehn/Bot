"""
S5 -- Does p_w15 correlate with real outcomes at all, continuously (not
via the 0.50 threshold), and is any correlation concentrated in a
particular moneyness regime (OTM vs near-ATM/ITM)? A genuine directional
edge that a crude agree/disagree split buries should still show up as a
monotonic relationship between "signed distance from 0.50 in the trade's
favor" and would_win/would_pnl.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"

df = pd.read_csv(f"{OUT}/btc_15m_w15_backfilled.csv", low_memory=False)
df["would_win"] = df["would_win"].astype(str).str.lower().isin(["true", "1", "1.0"])
df["would_pnl"] = pd.to_numeric(df["would_pnl"], errors="coerce")

df["offset_pct"] = pd.to_numeric(df["offset_pct"], errors="coerce")

# signed model conviction IN THE DIRECTION OF THE TRADE'S SIDE:
# for a yes bet, conviction = p_w15 - 0.50 (positive = model supports it)
# for a no bet,  conviction = 0.50 - p_w15 (positive = model supports it)
is_yes = df["side"].str.lower() == "yes"
df["conviction"] = np.where(is_yes, df["p_w15"] - 0.50, 0.50 - df["p_w15"])

print("=== Continuous correlation: conviction vs outcome (all covered trades) ===")
sub = df.dropna(subset=["conviction", "would_pnl"])
ic_pnl = spearmanr(sub["conviction"], sub["would_pnl"]).statistic
ic_win = pointbiserialr(sub["would_win"].astype(float), sub["conviction"]).statistic
print(f"n={len(sub)}  spearman(conviction, would_pnl)={ic_pnl:+.4f}  "
      f"point-biserial(conviction, would_win)={ic_win:+.4f}")

print("\n=== Conviction quintiles: does higher conviction -> better outcome? ===")
sub = sub.copy()
sub["q"] = pd.qcut(sub["conviction"], 5, labels=False, duplicates="drop")
qtab = sub.groupby("q").agg(n=("would_win", "size"), WR=("would_win", "mean"),
                            pnl=("would_pnl", "sum"),
                            conv_lo=("conviction", "min"), conv_hi=("conviction", "max"))
print(qtab.round(4).to_string())

print("\n=== Same test, split by moneyness (|offset_pct|) ===")
df["abs_offset"] = df["offset_pct"].abs()
for label, mask in [("near-ATM (|offset|<0.15%)", df["abs_offset"] < 0.0015),
                    ("mid (0.15-0.40%)", (df["abs_offset"] >= 0.0015) & (df["abs_offset"] < 0.0040)),
                    ("far-OTM (>=0.40%)", df["abs_offset"] >= 0.0040)]:
    s = df[mask].dropna(subset=["conviction", "would_pnl"])
    if len(s) < 20:
        print(f"{label}: n={len(s)} (too thin)")
        continue
    ic_p = spearmanr(s["conviction"], s["would_pnl"]).statistic
    ic_w = pointbiserialr(s["would_win"].astype(float), s["conviction"]).statistic
    agree = s[s["conviction"] > 0]
    disagree = s[s["conviction"] <= 0]
    print(f"{label}: n={len(s)}  corr(conv,pnl)={ic_p:+.4f}  corr(conv,win)={ic_w:+.4f}  "
          f"agree(n={len(agree)}) WR={agree['would_win'].mean():.3f} pnl=${agree['would_pnl'].sum():.2f}  "
          f"disagree(n={len(disagree)}) WR={disagree['would_win'].mean():.3f} pnl=${disagree['would_pnl'].sum():.2f}")
