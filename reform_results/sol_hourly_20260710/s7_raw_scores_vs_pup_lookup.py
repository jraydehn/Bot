"""
S7 -- decisive check. s6 proved composite_trend/composite_rev (the raw
inputs) carry stable, undecayed IC~0.27 on next-hour direction throughout
2024-2026, including the exact recent window. s5 proved the DERIVED
composite_p_up (the lookup-table output fed into z_drift) shows ~zero
relationship with actual outcomes in that same recent window. The gap must
be in the lookup/blending layer (lookup_p_up_blended, stale tables per s1).

Test: bucket the archive's own LIVE-LOGGED composite_trend / composite_rev
(not the derived p_up) directly against resolved_yes. If raw buckets show
real edge where composite_p_up showed none, the calibration LOOKUP is the
broken link -- not the underlying signal, and not the DRIFT_MULTIPLIER.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "offset_pct", "composite_p_up",
                            "composite_trend", "composite_rev", "p_market", "resolved_yes"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "offset_pct", "composite_p_up", "composite_trend",
                        "composite_rev", "p_market", "resolved_yes"])

atm = df[df["offset_pct"].abs() < 0.05].copy()
print(f"near-ATM candidates: {len(atm)}  tickers: {atm['contract_ticker'].nunique()}")

print("\n=== raw composite_trend bucket vs actual resolved_yes% (near-ATM, ticker-clustered) ===")
for t in sorted(atm["composite_trend"].unique()):
    sub = atm[atm["composite_trend"] == t]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"))
    if len(tk) < 15:
        continue
    print(f"  trend={t:+.0f}  n={len(sub):6d}  tk={len(tk):5d}  actual_up%={tk['y'].mean():.3f}")

print("\n=== raw composite_rev bucket vs actual resolved_yes% (near-ATM, ticker-clustered) ===")
for r in sorted(atm["composite_rev"].unique()):
    sub = atm[atm["composite_rev"] == r]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"))
    if len(tk) < 15:
        continue
    print(f"  rev={r:+.0f}  n={len(sub):6d}  tk={len(tk):5d}  actual_up%={tk['y'].mean():.3f}")

print("\n=== combined tail buckets (near-ATM) ===")
for lbl, mask in [
    ("Strong bullish (rev>=4,trend>=0)", (atm["composite_rev"] >= 4) & (atm["composite_trend"] >= 0)),
    ("Strong bearish (rev<=-4,trend<=0)", (atm["composite_rev"] <= -4) & (atm["composite_trend"] <= 0)),
    ("Neutral (rev in [-1,1])", atm["composite_rev"].between(-1, 1)),
]:
    sub = atm[mask]
    tk = sub.groupby("contract_ticker").agg(y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    print(f"  {lbl:<34s} n={len(sub):6d} tk={len(tk):5d} actual_up%={tk['y'].mean():.3f} avg_p_market={tk['pm'].mean():.3f}")

print(f"\ncorr(composite_trend, resolved_yes) = {np.corrcoef(atm['composite_trend'], atm['resolved_yes'])[0,1]:+.4f}")
print(f"corr(composite_rev, resolved_yes)   = {np.corrcoef(atm['composite_rev'], atm['resolved_yes'])[0,1]:+.4f}")
print(f"corr(composite_p_up-0.5, resolved_yes) = {np.corrcoef(atm['composite_p_up']-0.5, atm['resolved_yes'])[0,1]:+.4f}  (s5 reference, ~0)")

# Does composite_p_up itself even track composite_trend/rev sensibly?
print(f"\ncorr(composite_trend, composite_p_up) = {np.corrcoef(atm['composite_trend'], atm['composite_p_up'])[0,1]:+.4f}")
print(f"corr(composite_rev, composite_p_up)   = {np.corrcoef(atm['composite_rev'], atm['composite_p_up'])[0,1]:+.4f}")

print("\nDONE_S7")
