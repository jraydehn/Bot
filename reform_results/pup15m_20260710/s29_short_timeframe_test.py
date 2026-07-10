"""
S29 -- user hypothesis: shorter timeframes should matter more for a bet
that only lives ~10.5 minutes. s27's grid ranked 1h/12h 31st of 35 on the
synthetic aggregate win-spread metric, but s28 already showed that metric
can be misleading -- 2h/120h ranked well on it yet was WORSE than 6h/72h at
predicting the actual touch/MAE mechanism. Testing properly this time:
short windows down to 15min, against the trade-level mechanism directly
(touch/MAE, the ground truth for "is the buffer at risk"), not just the
synthetic win-spread proxy. Also includes the already-established
candidates (6h/72h, 2h/120h) for direct side-by-side comparison.
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

CANDIDATES = [
    ("15m", "1h",  15,   60),
    ("15m", "2h",  15,  120),
    ("15m", "4h",  15,  240),
    ("30m", "2h",  30,  120),
    ("30m", "4h",  30,  240),
    ("30m", "6h",  30,  360),
    ("1h",  "4h",  60,  240),
    ("1h",  "6h",  60,  360),
    ("1h",  "12h", 60,  720),
    ("2h",  "12h", 120, 720),
    ("6h",  "72h", 360, 4320),   # established baseline
    ("2h",  "120h",120, 7200),   # s27's top synthetic-metric pick
]
mins_needed = sorted(set(m for _, _, s, l in CANDIDATES for m in (s, l)))
rv_cache = {m: r1m.rolling(m).std() for m in mins_needed}

pt = pd.read_csv(f"{OUT}/s26_preentry_features.csv", parse_dates=["decision_time"]).sort_values("decision_time")
syn = pd.read_csv(f"{OUT}/synthetic_yes_bets.csv", parse_dates=["dec"])[["dec", "win", "year", "day"]]

results = []
for slabel, llabel, s, l in CANDIDATES:
    ratio = (rv_cache[s] / rv_cache[l].replace(0, np.nan)).rename("rv")
    rv_df = ratio.reset_index(); rv_df.columns = ["ts", "rv"]

    # trade-level mechanism test (ground truth: does it predict touch/MAE/win?)
    d = pd.merge_asof(pt, rv_df.sort_values("ts"), left_on="decision_time", right_on="ts",
                      direction="backward").dropna(subset=["rv"])
    r_win, p_win = stats.pearsonr(d["rv"], d["win"])
    r_mae, p_mae = stats.pearsonr(d["rv"], d["mae_pct"])
    r_touch, p_touch = stats.pearsonr(d["rv"], d["touched_strike"].astype(float))

    # synthetic 2.5yr aggregate spread (2026 holdout, day-clustered significance)
    sd = pd.merge_asof(syn.sort_values("dec"), rv_df.sort_values("ts"), left_on="dec", right_on="ts",
                       direction="backward").dropna(subset=["rv"])
    y26 = sd[sd["year"] == 2026]
    q = y26["rv"].quantile([0.2, 0.8])
    lo_d = y26[y26["rv"] <= q[0.2]].groupby("day")["win"].mean()
    hi_d = y26[y26["rv"] >= q[0.8]].groupby("day")["win"].mean()
    boots = [hi_d.sample(frac=1, replace=True, random_state=i).mean() - lo_d.sample(frac=1, replace=True, random_state=i).mean()
            for i in range(1500)]
    arr = np.array(boots)
    spread_2026 = hi_d.mean() - lo_d.mean()
    p_2026 = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))

    results.append(dict(pair=f"{slabel}/{llabel}", r2_win=r_win**2, p_win=p_win, r2_mae=r_mae**2, p_mae=p_mae,
                        r2_touch=r_touch**2, p_touch=p_touch, spread_2026=spread_2026, p_spread_2026=p_2026,
                        n_trades=len(d)))
    print(f"{slabel:>4s}/{llabel:<5s}: trade-level r2(win)={r_win**2:.4f}(P={p_win:.3f})  "
          f"r2(mae)={r_mae**2:.4f}(P={p_mae:.3f})  r2(touch)={r_touch**2:.4f}(P={p_touch:.3f})  "
          f"|| synth 2026 spread={spread_2026:+.4f}(P={p_2026:.4f})", flush=True)

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/s29_short_timeframe_results.csv", index=False)
print("\n=== ranked by trade-level r2(win) -- the metric that actually matters (predicting the outcome) ===")
print(res.sort_values("r2_win", ascending=False)[["pair", "r2_win", "p_win", "r2_touch", "p_touch"]].to_string(index=False))
print("DONE_S29")
