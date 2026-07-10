"""
S27 -- sweep a grid of rv_ratio(short, long) timeframe pairs on the SAME
2.5-year synthetic BTC 15m YES bet used in s19 (hold spot 0.0633% above
strike ~10.5min, cost 0.730). Same rigor as s19: same-sign quintile WR
spread required in ALL THREE years (2024/2025/2026) + day-clustered
significance, so nothing here is fit to this week's data.

Grid: short in {1h,2h,4h,6h,8h,12h}, long in {12h,24h,48h,72h,120h,168h},
short < long. rv_ratio(s,l) = realized_vol(last s hours) / realized_vol(last l hours)
of 1-minute log returns.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
px = px[px.index >= "2023-11-01"]
r1m = px["close"].pct_change()

SHORT_H = [1, 2, 4, 6, 8, 12]
LONG_H = [12, 24, 48, 72, 120, 168]

syn = pd.read_csv(f"{OUT}/synthetic_yes_bets.csv", parse_dates=["dec"])
syn = syn[["dec", "win", "year", "day"]].copy()
print(f"synthetic bets: {len(syn)}  years {sorted(syn['year'].unique())}")

# precompute rolling std at every needed hour-window once, reuse across pairs
rv_cache = {}
for h in sorted(set(SHORT_H + LONG_H)):
    rv_cache[h] = r1m.rolling(h * 60).std()
    print(f"  computed {h}h rolling vol", flush=True)

results = []
for s in SHORT_H:
    for l in LONG_H:
        if s >= l:
            continue
        ratio = (rv_cache[s] / rv_cache[l].replace(0, np.nan)).rename("rv")
        rv_df = ratio.reset_index()
        rv_df.columns = ["ts", "rv"]
        d = pd.merge_asof(syn.sort_values("dec"), rv_df.sort_values("ts"),
                          left_on="dec", right_on="ts", direction="backward").dropna(subset=["rv"])
        spreads = {}
        consistent = True
        for y in [2024, 2025, 2026]:
            yy = d[d["year"] == y]
            if len(yy) < 2000:
                consistent = False
                continue
            q = yy["rv"].quantile([0.2, 0.8])
            lo = yy[yy["rv"] <= q[0.2]]["win"].mean()
            hi = yy[yy["rv"] >= q[0.8]]["win"].mean()
            spreads[y] = hi - lo
        if len(spreads) < 3:
            continue
        signs = set(np.sign(v) for v in spreads.values() if v != 0)
        consistent = len(signs) == 1
        results.append({"short_h": s, "long_h": l, **{f"spread_{y}": round(v, 4) for y, v in spreads.items()},
                        "consistent": consistent, "mean_abs_spread": np.mean([abs(v) for v in spreads.values()])})
        print(f"  rv({s}h/{l}h): 2024={spreads.get(2024,float('nan')):+.4f} "
              f"2025={spreads.get(2025,float('nan')):+.4f} 2026={spreads.get(2026,float('nan')):+.4f} "
              f"consistent={consistent}", flush=True)

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/s27_rv_ratio_grid.csv", index=False)
print("\n=== consistent (same-sign all 3 years) candidates, ranked by mean |spread| ===")
cons = res[res["consistent"]].sort_values("mean_abs_spread", ascending=False)
print(cons.to_string(index=False))
print("\n=== inconsistent (sign flips across years -- likely noise, discard) ===")
incons = res[~res["consistent"]].sort_values("mean_abs_spread", ascending=False)
print(incons.head(10).to_string(index=False))
print("DONE_S27")
