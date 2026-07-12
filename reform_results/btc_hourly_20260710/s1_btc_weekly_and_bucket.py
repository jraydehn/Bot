"""
S1 -- BTC hourly loss autopsy, same methodology as the SOL investigation.
paper_trades.csv covers 06-17 -> now, a single clean era (post the 06-16
regime_pup reform, no resets in between). Weekly trend + AGREE/CONTRARIAN
bucket split (the exact split that found SOL's break) to see if BTC's
"slow grind down" is concentrated somewhere specific or spread evenly.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/paper_trades.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[pd.to_numeric(df["bet_amount"], errors="coerce") > 0].dropna(subset=["would_pnl", "resolved_yes", "side", "p_market"])
t["p_market"] = pd.to_numeric(t["p_market"], errors="coerce")
t["won"] = np.where(t["side"] == "yes", t["resolved_yes"] == True, t["resolved_yes"] == False)
t["cost"] = np.where(t["side"] == "yes", t["p_market"], 1 - t["p_market"])
t["contrarian"] = ((t.side == "yes") & (t.p_market < 0.5)) | ((t.side == "no") & (t.p_market > 0.5))
t["bucket"] = np.where(t["contrarian"], "CONTRARIAN", "AGREE")
print(f"total taken trades: {len(t)}  range: {t['decision_time'].min()} -> {t['decision_time'].max()}")
print(f"overall: WR={t['won'].mean():.1%}  BE={t['cost'].mean():.1%}  total_pnl=${t['would_pnl'].sum():.2f}")

print(f"\n=== weekly trend, overall ===")
t["week"] = t["decision_time"].dt.to_period("W").astype(str)
print(t.groupby("week").agg(n=("won", "size"), wr=("won", "mean"), be=("cost", "mean"), pnl=("would_pnl", "sum")).to_string())

print(f"\n=== weekly trend, by bucket ===")
for b, sub in t.groupby("bucket"):
    print(f"\n  --- {b} ---")
    print(sub.groupby("week").agg(n=("won", "size"), wr=("won", "mean"), be=("cost", "mean"), pnl=("would_pnl", "sum")).to_string())

print(f"\n=== by side ===")
print(t.groupby("side").agg(n=("won", "size"), wr=("won", "mean"), be=("cost", "mean"), pnl=("would_pnl", "sum")))

print(f"\n=== by side x bucket ===")
print(t.groupby(["side", "bucket"]).agg(n=("won", "size"), wr=("won", "mean"), be=("cost", "mean"), pnl=("would_pnl", "sum")))

print("\nDONE_S1")
