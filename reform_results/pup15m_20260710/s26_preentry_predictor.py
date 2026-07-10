"""
S26 -- s25 found the mechanism: trades where price ever touches/crosses the
strike during the ~10.5min window lose 85% of the time (vs 7.5% loss rate
when it doesn't). Daily vol_ratio only partially predicts this (flags the
07-06/07 crash, misses that 07-08->07-10am was ALSO calm-vol yet had
frequent pullbacks while the 07-10 overnight window had none).

This tests whether PRE-ENTRY path behavior (the 5/10/15 minutes BEFORE each
decision, fully causal -- no lookahead) predicts POST-entry MAE/outcome.
If recent choppiness/reversal-proneness carries forward, it's a genuinely
new, live-computable signal distinct from the daily vol_ratio regime and
distinct from the self-referential z_drift fallback.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
c1 = px["close"]

pt = pd.read_csv(f"{OUT}/s25_path_features.csv", parse_dates=["decision_time"])

pre_rv5, pre_rv15, pre_rev5, pre_rev15, pre_maxdd5 = [], [], [], [], []
for dt in pt["decision_time"]:
    hist = c1[(c1.index > dt - pd.Timedelta(minutes=16)) & (c1.index <= dt)]
    if len(hist) < 6:
        pre_rv5.append(np.nan); pre_rv15.append(np.nan)
        pre_rev5.append(np.nan); pre_rev15.append(np.nan); pre_maxdd5.append(np.nan)
        continue
    r = hist.pct_change().dropna()
    r5 = r.iloc[-5:]
    r15 = r
    pre_rv5.append(r5.std() if len(r5) > 1 else np.nan)
    pre_rv15.append(r15.std() if len(r15) > 1 else np.nan)
    pre_rev5.append(int((np.sign(r5) != np.sign(r5.shift(1))).sum()) if len(r5) > 1 else np.nan)
    pre_rev15.append(int((np.sign(r15) != np.sign(r15.shift(1))).sum()) if len(r15) > 1 else np.nan)
    last5 = hist.iloc[-6:]
    pre_maxdd5.append((last5.min() - last5.iloc[0]) / last5.iloc[0] * 100 if len(last5) > 1 else np.nan)

pt["pre_rv5"], pt["pre_rv15"] = pre_rv5, pre_rv15
pt["pre_rev5"], pt["pre_rev15"] = pre_rev5, pre_rev15
pt["pre_maxdd5"] = pre_maxdd5
pt = pt.dropna(subset=["pre_rv5", "pre_rv15"])
print(f"trades with valid pre-entry features: {len(pt)}")

print("\n=== does PRE-entry path behavior predict POST-entry MAE? (correlation) ===")
for col in ["pre_rv5", "pre_rv15", "pre_rev5", "pre_rev15", "pre_maxdd5"]:
    print(f"  corr({col}, mae_pct) = {pt[col].corr(pt['mae_pct']):+.4f}   "
          f"corr({col}, win) = {pt[col].corr(pt['win']):+.4f}")

print("\n=== quintile check: pre_rv15 (pre-entry 15min realized vol) vs outcome ===")
q = pt["pre_rv15"].quantile([0.33, 0.67])
for lbl, m in [("low (calm before entry)", pt["pre_rv15"] <= q[0.33]),
              ("mid", (pt["pre_rv15"] > q[0.33]) & (pt["pre_rv15"] <= q[0.67])),
              ("high (choppy before entry)", pt["pre_rv15"] > q[0.67])]:
    g = pt[m]
    print(f"  {lbl:28s}: n={len(g):3d}  WR={g['win'].mean():.1%}  mae_mean={g['mae_pct'].mean():+.4f}  "
          f"touched%={g['touched_strike'].mean():.1%}")

print("\n=== group means of pre-entry features (does overnight-win differ from bad stretch BEFORE entry?) ===")
groups = {
    "07-01->07-05 (good)":      (pt["decision_time"] >= "2026-07-01") & (pt["decision_time"] < "2026-07-06"),
    "07-06->07-07 (crash)":     (pt["decision_time"] >= "2026-07-06") & (pt["decision_time"] < "2026-07-08"),
    "07-08->07-10 07:50 (bad)": (pt["decision_time"] >= "2026-07-08") & (pt["decision_time"] < "2026-07-10 07:50:00+00:00"),
    "07-10 07:50-> (win)":      pt["decision_time"] >= "2026-07-10 07:50:00+00:00",
}
for lbl, m in groups.items():
    g = pt[m]
    if len(g) == 0:
        continue
    print(f"  {lbl:28s}: n={len(g):3d}  pre_rv15={g['pre_rv15'].mean()*100:.4f}%  "
          f"pre_rev15={g['pre_rev15'].mean():.2f}  pre_maxdd5={g['pre_maxdd5'].mean():+.4f}%")

# episode-clustered significance of the pre_rv15 relationship (avoid pseudo-replication)
pt = pt.sort_values("decision_time")
gap = pt["decision_time"].diff().dt.total_seconds() / 60
pt["episode"] = (gap.isna() | (gap > 45)).cumsum()
rng = np.random.default_rng(3)
lo = pt[pt["pre_rv15"] <= q[0.33]]
hi = pt[pt["pre_rv15"] > q[0.67]]
lo_ep = lo.groupby("episode")["win"].mean()
hi_ep = hi.groupby("episode")["win"].mean()
boots = [lo_ep.sample(frac=1, replace=True, random_state=i).mean() - hi_ep.sample(frac=1, replace=True, random_state=i).mean()
        for i in range(3000)]
print(f"\nepisode-clustered WR(low pre_rv15) - WR(high pre_rv15): "
      f"{lo['win'].mean()-hi['win'].mean():+.4f}  P(<=0)={np.mean(np.array(boots)<=0):.4f}  "
      f"(low n={len(lo)}/{lo_ep.size}eps, high n={len(hi)}/{hi_ep.size}eps)")
pt.to_csv(f"{OUT}/s26_preentry_features.csv", index=False)
print("DONE_S26")
