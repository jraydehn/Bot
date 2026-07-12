"""
S5 -- s3/s4 found any z_drift application (flat or gated) hurts SOL strike-hit
calibration. Isolate whether composite_p_up itself is miscalibrated by
restricting to near-ATM candidates (|offset_pct| < 0.05%) where strike
distance barely matters and outcome should track composite_p_up almost
directly if the signal is real in strike-space.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "offset_pct", "composite_p_up",
                            "p_market", "resolved_yes", "tau_minutes"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "offset_pct", "composite_p_up", "p_market", "resolved_yes"])
df = df[df["composite_p_up"].between(0.001, 0.999)]

atm = df[df["offset_pct"].abs() < 0.05].copy()
print(f"near-ATM candidates (|offset|<0.05%): {len(atm)}  tickers: {atm['contract_ticker'].nunique()}")

atm["pup_decile"] = pd.qcut(atm["composite_p_up"], 10, duplicates="drop")
print(f"\n=== near-ATM: composite_p_up decile vs actual resolved_yes% (ticker-clustered) ===")
print(f"{'decile':>22s} {'n':>6s} {'tk':>5s} {'mean_pup':>9s} {'actual_up%':>11s} {'p_market':>9s}")
for d, sub in atm.groupby("pup_decile", observed=True):
    tk = sub.groupby("contract_ticker").agg(pup=("composite_p_up", "mean"), y=("resolved_yes", "mean"), pm=("p_market", "mean"))
    print(f"{str(d):>22s} {len(sub):6d} {len(tk):5d} {tk['pup'].mean():9.3f} {tk['y'].mean():11.3f} {tk['pm'].mean():9.3f}")

print(f"\n=== near-ATM: p_market decile vs actual resolved_yes% (is the MARKET well-calibrated here?) ===")
atm["pm_decile"] = pd.qcut(atm["p_market"], 10, duplicates="drop")
print(f"{'decile':>22s} {'n':>6s} {'tk':>5s} {'mean_pm':>9s} {'actual_up%':>11s}")
for d, sub in atm.groupby("pm_decile", observed=True):
    tk = sub.groupby("contract_ticker").agg(pm=("p_market", "mean"), y=("resolved_yes", "mean"))
    print(f"{str(d):>22s} {len(sub):6d} {len(tk):5d} {tk['pm'].mean():9.3f} {tk['y'].mean():11.3f}")

# Correlation: does composite_p_up add anything OVER p_market at near-ATM?
resid_pup = atm["composite_p_up"] - 0.5
resid_pm = atm["p_market"] - 0.5
print(f"\ncorr(composite_p_up-0.5, resolved_yes) = {np.corrcoef(resid_pup, atm['resolved_yes'])[0,1]:+.4f}")
print(f"corr(p_market-0.5, resolved_yes)       = {np.corrcoef(resid_pm, atm['resolved_yes'])[0,1]:+.4f}")
print(f"corr(composite_p_up-0.5, p_market-0.5) = {np.corrcoef(resid_pup, resid_pm)[0,1]:+.4f}  (redundancy check)")

print("\nDONE_S5")
