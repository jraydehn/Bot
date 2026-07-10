"""
S30 -- pair rv_ratio (magnitude of vol expansion) with a DIRECTIONAL
EFFICIENCY measure (Kaufman-style): does price move efficiently in one
direction, or thrash back and forth with little net progress? Same vol
magnitude, very different implications for a thin ITM buffer -- a choppy
oscillation eats the buffer even at modest volatility; a clean trend
(especially one running the same direction as the position) may not.

efficiency_ratio(window) = |net move over window| / (sum of |each 1-min
move| over window)  -- 1.0 = perfectly directional, ~0 = pure noise/chop.

Tests, in order: (1) ER alone on the 2.5yr synthetic bet + trade-level
touch/MAE (same rigor as rv_ratio candidates); (2) rv_ratio x ER 2x2 grid
on both; (3) does conditioning on BOTH explain more touch/MAE variance
than rv_ratio alone (2h/120h, s28's best)?
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
c1 = px["close"]
r1m = c1.pct_change()

def efficiency_ratio(close, window_min):
    net = (close - close.shift(window_min)).abs()
    path = close.diff().abs().rolling(window_min).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)

# best rv_ratio candidate from s28 (2h/120h), plus 6h/72h for comparison
rv2h = r1m.rolling(120).std()
rv120h = r1m.rolling(7200).std()
rv_new = (rv2h / rv120h.replace(0, np.nan)).rename("rv")

ER_WINDOWS = [60, 120, 240, 360]  # 1h, 2h, 4h, 6h
er_series = {w: efficiency_ratio(c1, w) for w in ER_WINDOWS}

pt = pd.read_csv(f"{OUT}/s26_preentry_features.csv", parse_dates=["decision_time"]).sort_values("decision_time")
rv_df = rv_new.reset_index(); rv_df.columns = ["ts", "rv"]
pt = pd.merge_asof(pt, rv_df.sort_values("ts"), left_on="decision_time", right_on="ts", direction="backward")

print("=== ER alone: trade-level correlation with touch/MAE/win, each window ===")
for w in ER_WINDOWS:
    ed = er_series[w].rename("er").reset_index(); ed.columns = ["ts2", "er"]
    d = pd.merge_asof(pt.drop(columns=["ts"]) if "ts" in pt.columns else pt, ed.sort_values("ts2"),
                      left_on="decision_time", right_on="ts2", direction="backward").dropna(subset=["er"])
    r_win, p_win = stats.pearsonr(d["er"], d["win"])
    r_mae, p_mae = stats.pearsonr(d["er"], d["mae_pct"])
    r_touch, p_touch = stats.pearsonr(d["er"], d["touched_strike"].astype(float))
    print(f"  ER({w}min): r2(win)={r_win**2:.4f}(P={p_win:.3f})  r2(mae)={r_mae**2:.4f}(P={p_mae:.3f})  "
          f"r2(touch)={r_touch**2:.4f}(P={p_touch:.3f})")

# pick the best ER window for the interaction test
best_er_w = 120  # 2h, matches rv_new's short leg
ed = er_series[best_er_w].rename("er").reset_index(); ed.columns = ["ts2", "er"]
pt2 = pd.merge_asof(pt, ed.sort_values("ts2"), left_on="decision_time", right_on="ts2", direction="backward")
pt2 = pt2.dropna(subset=["rv", "er"])

print(f"\n=== rv_ratio(2h/120h) x ER({best_er_w}min) 2x2 interaction, trade-level (n={len(pt2)}) ===")
rv_med, er_med = pt2["rv"].median(), pt2["er"].median()
for rv_lbl, rv_m in [("rv_lo(cool)", pt2["rv"] <= rv_med), ("rv_hi(hot)", pt2["rv"] > rv_med)]:
    for er_lbl, er_m in [("er_lo(choppy)", pt2["er"] <= er_med), ("er_hi(trending)", pt2["er"] > er_med)]:
        g = pt2[rv_m & er_m]
        if len(g) == 0:
            continue
        print(f"  {rv_lbl:12s} x {er_lbl:16s}: n={len(g):3d}  WR={g['win'].mean():.1%}  "
              f"touched%={g['touched_strike'].mean():.1%}  mae_mean={g['mae_pct'].mean():+.4f}")

# does the interaction beat rv_ratio alone at explaining touch/MAE? (multiple regression R^2)
import numpy.linalg as la
X = np.column_stack([np.ones(len(pt2)), pt2["rv"], pt2["er"], pt2["rv"] * pt2["er"]])
for target in ["touched_strike", "mae_pct"]:
    y = pt2[target].astype(float).values
    beta, *_ = la.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_full = 1 - ss_res / ss_tot
    X_rv = np.column_stack([np.ones(len(pt2)), pt2["rv"]])
    beta_rv, *_ = la.lstsq(X_rv, y, rcond=None)
    r2_rv = 1 - ((y - X_rv @ beta_rv) ** 2).sum() / ss_tot
    print(f"\n  {target}: R2(rv_ratio alone)={r2_rv:.4f}   R2(rv_ratio + ER + interaction)={r2_full:.4f}   "
          f"gain={r2_full-r2_rv:+.4f}")

print("\n=== 2.5yr synthetic: ER alone, day-clustered, 2026 holdout ===")
syn = pd.read_csv(f"{OUT}/synthetic_yes_bets.csv", parse_dates=["dec"])[["dec", "win", "year", "day"]]
ed2 = er_series[best_er_w].rename("er").reset_index(); ed2.columns = ["ts3", "er"]
syn2 = pd.merge_asof(syn.sort_values("dec"), ed2.sort_values("ts3"), left_on="dec", right_on="ts3",
                     direction="backward").dropna(subset=["er"])
for y in [2024, 2025, 2026]:
    yy = syn2[syn2["year"] == y]
    q = yy["er"].quantile([0.2, 0.8])
    lo = yy[yy["er"] <= q[0.2]]["win"].mean()
    hi = yy[yy["er"] >= q[0.8]]["win"].mean()
    print(f"  {y}: ER low(choppy) WR={lo:.4f}  ER high(trending) WR={hi:.4f}  spread={hi-lo:+.4f}")

pt2.to_csv(f"{OUT}/s30_rv_x_er.csv", index=False)
print("DONE_S30")
