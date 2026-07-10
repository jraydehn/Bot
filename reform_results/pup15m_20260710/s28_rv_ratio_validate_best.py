"""
S28 -- the s27 grid found 2h/120h nominally best (mean |spread| 0.126 vs
6h/72h's 0.108), but ALL 34 combos were same-sign -- expected, since any
short/long vol-expansion measure captures overlapping information. Real
questions: (1) is 2h/120h materially different information from 6h/72h, or
just a noisier restatement of the same thing? (2) does it explain MORE of
the trade-level touch/MAE variance (6h/72h only got r^2=0.016)? (3) does it
hold up on the real 309-trade book with a threshold pre-registered from
2024-25 only?
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
px = px[px.index >= "2023-11-01"]
r1m = px["close"].pct_change()

rv2h = r1m.rolling(120).std()
rv120h = r1m.rolling(7200).std()
rv6h = r1m.rolling(360).std()
rv72h = r1m.rolling(4320).std()
new_ratio = (rv2h/rv120h.replace(0, np.nan)).rename("rv_new")
old_ratio = (rv6h/rv72h.replace(0, np.nan)).rename("rv_old")

both = pd.concat([new_ratio, old_ratio], axis=1).dropna()
print(f"=== correlation between 2h/120h and 6h/72h (same info or new?) ===")
print(f"  corr = {both['rv_new'].corr(both['rv_old']):.4f}")

syn = pd.read_csv(f"{OUT}/synthetic_yes_bets.csv", parse_dates=["dec"])[["dec","win","year","day"]]
nd = new_ratio.reset_index(); nd.columns = ["ts","rv_new"]
syn = pd.merge_asof(syn.sort_values("dec"), nd.sort_values("ts"), left_on="dec", right_on="ts", direction="backward")
od = old_ratio.reset_index(); od.columns = ["ts2","rv_old"]
syn = pd.merge_asof(syn.sort_values("dec"), od.sort_values("ts2"), left_on="dec", right_on="ts2", direction="backward")
syn = syn.dropna(subset=["rv_new","rv_old"])

print("\n=== day-clustered significance, 2026 holdout only (most relevant to recent behavior) ===")
y26 = syn[syn["year"]==2026]
for col in ["rv_new","rv_old"]:
    q = y26[col].quantile([0.2,0.8])
    lo_d = y26[y26[col]<=q[0.2]].groupby("day")["win"].mean()
    hi_d = y26[y26[col]>=q[0.8]].groupby("day")["win"].mean()
    boots = [hi_d.sample(frac=1,replace=True,random_state=i).mean()-lo_d.sample(frac=1,replace=True,random_state=i).mean()
             for i in range(2000)]
    arr = np.array(boots)
    print(f"  {col}: spread={hi_d.mean()-lo_d.mean():+.4f}  P(two-sided)={2*min(np.mean(arr<=0),np.mean(arr>=0)):.4f}")

print("\n=== does rv_new explain MORE trade-level touch/MAE variance than rv_old? ===")
pt = pd.read_csv(f"{OUT}/s26_preentry_features.csv", parse_dates=["decision_time"]).sort_values("decision_time")
nd2 = new_ratio.reset_index(); nd2.columns = ["ts3","rv_new"]
pt = pd.merge_asof(pt, nd2.sort_values("ts3"), left_on="decision_time", right_on="ts3", direction="backward")
od2 = old_ratio.reset_index(); od2.columns = ["ts4","rv_old"]
pt = pd.merge_asof(pt.drop(columns=["ts3"]), od2.sort_values("ts4"), left_on="decision_time", right_on="ts4", direction="backward")
pt = pt.dropna(subset=["rv_new","rv_old"])
for col in ["rv_new","rv_old"]:
    r_win, p_win = stats.pearsonr(pt[col], pt["win"])
    r_mae, p_mae = stats.pearsonr(pt[col], pt["mae_pct"])
    r_touch, p_touch = stats.pearsonr(pt[col], pt["touched_strike"].astype(float))
    print(f"  {col}: r^2(win)={r_win**2:.4f} (P={p_win:.3f})  r^2(mae)={r_mae**2:.4f} (P={p_mae:.3f})  "
          f"r^2(touched)={r_touch**2:.4f} (P={p_touch:.3f})")
q = pt["rv_new"].quantile([0.33,0.67])
for lbl,m in [("cool",pt["rv_new"]<=q[0.33]),("mid",(pt["rv_new"]>q[0.33])&(pt["rv_new"]<=q[0.67])),("hot",pt["rv_new"]>q[0.67])]:
    g = pt[m]
    print(f"  rv_new {lbl:5s}: n={len(g):3d} WR={g['win'].mean():.1%} touched%={g['touched_strike'].mean():.1%}")

print("\n=== real 309-trade YES book: rv_new with PRE-REGISTERED 2024-25 top-quintile threshold ===")
TH_NEW = syn[syn["year"]<=2025]["rv_new"].quantile(0.8)
print(f"  threshold: rv_new > {TH_NEW:.3f}")
df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[(df["side"]=="yes") & (pd.to_numeric(df["bet_amount"], errors="coerce")>0)].dropna(subset=["would_pnl"]).sort_values("decision_time")
nd3 = new_ratio.reset_index(); nd3.columns = ["ts5","rv_new"]
t = pd.merge_asof(t, nd3.sort_values("ts5"), left_on="decision_time", right_on="ts5", direction="backward").dropna(subset=["rv_new"])
t["win"] = t["resolved_yes"]; t["cost"] = t["p_market"]
t["hot"] = t["rv_new"] > TH_NEW
for hot, g in t.groupby("hot"):
    print(f"  hot={hot!s:5s}: n={len(g):3d}  WR={g['win'].mean():.1%}  BE={g['cost'].mean():.1%}  "
          f"edge={g['win'].mean()-g['cost'].mean():+.3f}  $ {g['would_pnl'].sum():+8.2f}")
print("DONE_S28")
