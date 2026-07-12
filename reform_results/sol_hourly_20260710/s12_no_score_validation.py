"""
S12 -- validate the no_score>=3 candidate found in s11 (uncertain zone,
p_market 0.35-0.65): n=336, WR(up)=35.6%, 95% CI [0.24,0.47] excludes 0.5.
no_score is an established signal (already validated as a BTC rescue
elsewhere in this codebase; one narrow existing SOL YES-block gate uses
no_score=2, but nothing exploits no_score generally as a NO-conviction
signal for SOL). Check: split-half stability, redundancy with the dead
composite_trend/rev signal, robustness at threshold >=2 (more data), and
$ PnL impact of a NO-side boost/floor gated on it, uncertain zone only.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("results/sol_scan_archive.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_market", "resolved_yes",
                            "no_score", "composite_trend", "composite_rev", "offset_pct"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "p_market", "resolved_yes", "no_score"])
unc = df[df["p_market"].between(0.35, 0.65)].copy()
print(f"uncertain zone: n={len(unc)}  tickers={unc['contract_ticker'].nunique()}")

print(f"\n=== no_score >= threshold, uncertain zone, ticker-clustered ===")
for thresh in [1, 2, 3]:
    sub = unc[unc["no_score"] >= thresh]
    tk = sub.groupby("contract_ticker")["resolved_yes"].mean()
    if len(tk) < 15:
        continue
    boots = [tk.sample(frac=1, replace=True, random_state=i).mean() for i in range(1500)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  no_score>={thresh}: n={len(sub):5d} tk={len(tk):4d}  up%={tk.mean():.1%}  95%CI=[{lo:.2f},{hi:.2f}]  "
          f"excl_0.5={'YES' if not (lo<=0.5<=hi) else 'no'}")

print(f"\n=== split-half stability: no_score>=3, uncertain zone ===")
mid = unc["logged_at"].quantile(0.5)
for label, sub in [("H1 (early)", unc[unc["logged_at"] < mid]), ("H2 (late)", unc[unc["logged_at"] >= mid])]:
    s = sub[sub["no_score"] >= 3]
    tk = s.groupby("contract_ticker")["resolved_yes"].mean()
    if len(tk) < 10:
        print(f"  {label}: too thin (tk={len(tk)})")
        continue
    print(f"  {label}: n={len(s):4d} tk={len(tk):3d}  up%={tk.mean():.1%}")

print(f"\n=== weekly stability: no_score>=3, uncertain zone ===")
unc["week"] = unc["logged_at"].dt.to_period("W").astype(str)
for wk, sub in unc[unc["no_score"] >= 3].groupby("week"):
    tk = sub.groupby("contract_ticker")["resolved_yes"].mean()
    print(f"  {wk}: n={len(sub):4d} tk={len(tk):3d}  up%={tk.mean() if len(tk) else float('nan'):.1%}")

print(f"\n=== redundancy check: does no_score correlate with the DEAD composite_trend/rev signal? ===")
print(f"corr(no_score, composite_trend) = {np.corrcoef(unc['no_score'], unc['composite_trend'])[0,1]:+.4f}")
print(f"corr(no_score, composite_rev)   = {np.corrcoef(unc['no_score'], unc['composite_rev'])[0,1]:+.4f}")
# is the edge still there conditioning OUT composite signals (i.e. within composite-neutral subset)?
neutral_composite = unc[unc["composite_rev"].between(-1, 1)]
sub = neutral_composite[neutral_composite["no_score"] >= 3]
tk = sub.groupby("contract_ticker")["resolved_yes"].mean()
print(f"no_score>=3 WITHIN composite-neutral subset (rev in [-1,1]): n={len(sub)} tk={len(tk)} "
      f"up%={tk.mean() if len(tk) else float('nan'):.1%}  (if similar to unconditional 35.6%, edge is independent of composite score)")

print(f"\n=== $ PnL: NO-side bet whenever no_score>=3 in uncertain zone, flat $100 stake, ticker-clustered ===")
for thresh in [2, 3]:
    sub = unc[unc["no_score"] >= thresh]
    tk = sub.groupby("contract_ticker").agg(win=("resolved_yes", "mean"), pm=("p_market", "mean"))
    tk["win_no"] = 1 - tk["win"]
    tk["cost_no"] = 1 - tk["pm"]
    n_contracts = 100.0 / tk["cost_no"]
    pnl = np.where(tk["win_no"] >= 0.5, n_contracts * (1 - tk["cost_no"]), -n_contracts * tk["cost_no"])
    boots = [pnl[np.random.default_rng(i).integers(0, len(pnl), len(pnl))].sum() for i in range(1500)]
    p_le0 = float(np.mean(np.array(boots) <= 0))
    print(f"  no_score>={thresh}: n={len(tk):4d}  NO_WR={tk['win_no'].mean():.1%}  NO_BE={tk['cost_no'].mean():.1%}  "
          f"total=${pnl.sum():.2f}  $/bet=${pnl.sum()/len(tk):.2f}  P(total<=0)={p_le0:.3f}")

print("\nDONE_S12")
