"""
S8 -- s5/s7 filtered on raw offset_pct to isolate "near-ATM" candidates, but
that's confounded by tau: a small % offset at low tau is still many sigmas
from the strike, so p_market averaged 0.75-0.99 in that "near-ATM" sample --
not actually uncertain contracts. Redo properly: filter on p_market itself
(the correct joint moneyness/tau proxy) to isolate genuinely uncertain
(0.35-0.65) contracts, where composite signal would need to add value for
trading to matter anyway (extreme-p_market contracts have poor payout math
regardless of model quality).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "offset_pct", "composite_p_up",
                            "composite_trend", "composite_rev", "p_market", "resolved_yes", "tau_minutes"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "offset_pct", "composite_p_up", "composite_trend",
                        "composite_rev", "p_market", "resolved_yes", "tau_minutes"])

unc = df[df["p_market"].between(0.35, 0.65)].copy()
print(f"genuinely uncertain candidates (p_market 0.35-0.65): {len(unc)}  tickers: {unc['contract_ticker'].nunique()}")
print(f"  avg offset_pct: {unc['offset_pct'].abs().mean():.4f}%  avg tau_minutes: {unc['tau_minutes'].mean():.1f}")

print("\n=== composite_trend bucket vs actual resolved_yes% (p_market 0.35-0.65, ticker-clustered) ===")
for t in sorted(unc["composite_trend"].unique()):
    sub = unc[unc["composite_trend"] == t]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    if len(tk) < 15:
        continue
    print(f"  trend={t:+.0f}  n={len(sub):6d}  tk={len(tk):5d}  actual_up%={tk['y'].mean():.3f}  avg_pm={tk['pm'].mean():.3f}")

print("\n=== composite_rev bucket vs actual resolved_yes% (p_market 0.35-0.65, ticker-clustered) ===")
for r in sorted(unc["composite_rev"].unique()):
    sub = unc[unc["composite_rev"] == r]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    if len(tk) < 15:
        continue
    print(f"  rev={r:+.0f}  n={len(sub):6d}  tk={len(tk):5d}  actual_up%={tk['y'].mean():.3f}  avg_pm={tk['pm'].mean():.3f}")

print("\n=== tail buckets (p_market 0.35-0.65) ===")
for lbl, mask in [
    ("Strong bullish (rev>=4,trend>=0)", (unc["composite_rev"] >= 4) & (unc["composite_trend"] >= 0)),
    ("Strong bearish (rev<=-4,trend<=0)", (unc["composite_rev"] <= -4) & (unc["composite_trend"] <= 0)),
    ("Neutral (rev in [-1,1])", unc["composite_rev"].between(-1, 1)),
]:
    sub = unc[mask]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    if len(tk) < 15:
        print(f"  {lbl:<34s}  (thin, tk={len(tk)})")
        continue
    print(f"  {lbl:<34s} n={len(sub):6d} tk={len(tk):5d} actual_up%={tk['y'].mean():.3f} avg_p_market={tk['pm'].mean():.3f}  edge={tk['y'].mean()-tk['pm'].mean():+.3f}")

print(f"\ncorr(composite_trend, resolved_yes) = {np.corrcoef(unc['composite_trend'], unc['resolved_yes'])[0,1]:+.4f}")
print(f"corr(composite_rev, resolved_yes)   = {np.corrcoef(unc['composite_rev'], unc['resolved_yes'])[0,1]:+.4f}")
print(f"corr(composite_p_up-0.5, resolved_yes) = {np.corrcoef(unc['composite_p_up']-0.5, unc['resolved_yes'])[0,1]:+.4f}")
print(f"corr(p_market-0.5, resolved_yes) = {np.corrcoef(unc['p_market']-0.5, unc['resolved_yes'])[0,1]:+.4f}")

print("\nDONE_S8")
