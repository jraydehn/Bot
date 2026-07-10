"""
S32 -- extend the touch/MAE mechanism study to the full ~8.5-week history
(05-11 -> now), combining paper_trades_btc15m_archive_20260525_1432_pre_branched_drift.csv
(05-11->05-25) + paper_trades_btc15m.csv (05-25->now). The pre-05-25 era
used a DIFFERENT bet structure (YES buffer ~0.02% vs ~0.08% now, pm~0.59 vs
~0.74, balanced side-mix vs YES-heavy) -- treated as a genuine held-out era,
not pooled blindly. Question: does the touch-strike mechanism and the
rv_ratio/ER prediction of it REPLICATE in this structurally different era?
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
c1 = px["close"]

frames = []
for f in ["results/paper_trades_btc15m_archive_20260525_1432_pre_branched_drift.csv",
          "results/paper_trades_btc15m.csv"]:
    d = pd.read_csv(f, low_memory=False)
    d["src"] = f.split("/")[-1]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)].dropna(subset=["would_pnl"])
t = t.drop_duplicates(subset=["contract_ticker", "decision_time"], keep="last").sort_values("decision_time")
t["win"] = t["resolved_yes"]
t["era"] = np.where(t["decision_time"] < "2026-05-25 21:50:00+00:00", "pre-branched-drift", "current")
print(f"combined YES trades: {len(t)}  {t['decision_time'].min()} -> {t['decision_time'].max()}")
print(t["era"].value_counts().to_dict())

rows = []
for row in t.itertuples(index=False):
    dt = row.decision_time
    tau = float(row.tau_minutes) if pd.notna(row.tau_minutes) else 10.5
    close_t = dt + pd.Timedelta(minutes=tau)
    path = c1[(c1.index >= dt) & (c1.index <= close_t + pd.Timedelta(minutes=1))]
    if len(path) < 3:
        continue
    decision_spot = float(row.spot)
    strike = float(row.floor_strike)
    net_move = (path.iloc[-1] - decision_spot) / decision_spot
    mae = (path.min() - decision_spot) / decision_spot
    touched = (path.min() <= strike)
    rows.append(dict(decision_time=dt, ticker=row.contract_ticker, win=row.win, era=row.era,
                     would_pnl=row.would_pnl, net_move_pct=net_move * 100, mae_pct=mae * 100,
                     touched_strike=touched, offset_pct=row.offset_pct, p_market=row.p_market))
pt = pd.DataFrame(rows)
print(f"\nreconstructed paths: {len(pt)}")

print("\n=== touch mechanism, era-split (does it replicate in the structurally different era?) ===")
for era, g in pt.groupby("era"):
    ct = g.groupby("touched_strike")["win"].agg(["size", "mean"])
    print(f"  {era} (n={len(g)}):")
    print(ct.to_string().replace("\n", "\n    "))

# join rv_ratio(2h/120h) and ER(2h/6h) -- same features as s28/s30, now across full history
r1m = c1.pct_change()
rv2h, rv120h = r1m.rolling(120).std(), r1m.rolling(7200).std()
rv_ratio = (rv2h / rv120h.replace(0, np.nan)).rename("rv")

def efficiency_ratio(close, w):
    net = (close - close.shift(w)).abs()
    path = close.diff().abs().rolling(w).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)
er2h = efficiency_ratio(c1, 120).rename("er2h")

rv_df = rv_ratio.reset_index(); rv_df.columns = ["ts", "rv"]
pt = pd.merge_asof(pt.sort_values("decision_time"), rv_df.sort_values("ts"),
                   left_on="decision_time", right_on="ts", direction="backward")
er_df = er2h.reset_index(); er_df.columns = ["ts2", "er2h"]
pt = pd.merge_asof(pt.sort_values("decision_time"), er_df.sort_values("ts2"),
                   left_on="decision_time", right_on="ts2", direction="backward")
pt = pt.dropna(subset=["rv", "er2h"])

print("\n=== does rv_ratio predict touch/MAE in the pre-branched-drift era too? ===")
for era, g in pt.groupby("era"):
    r_touch, p_touch = stats.pearsonr(g["rv"], g["touched_strike"].astype(float))
    r_mae, p_mae = stats.pearsonr(g["rv"], g["mae_pct"])
    r_win, p_win = stats.pearsonr(g["rv"], g["win"])
    print(f"  {era} (n={len(g)}): r2(touch)={r_touch**2:.4f}(P={p_touch:.3f})  "
          f"r2(mae)={r_mae**2:.4f}(P={p_mae:.3f})  r2(win)={r_win**2:.4f}(P={p_win:.3f})")
    q = g["rv"].quantile([0.33, 0.67])
    for lbl, m in [("cool", g["rv"] <= q[0.33]), ("hot", g["rv"] > q[0.67])]:
        s = g[m]
        print(f"    {lbl}: n={len(s):3d}  WR={s['win'].mean():.1%}  touched%={s['touched_strike'].mean():.1%}")

print(f"\n=== full combined population (n={len(pt)}), rv_ratio tercile ===")
q = pt["rv"].quantile([0.33, 0.67])
for lbl, m in [("cool", pt["rv"] <= q[0.33]), ("mid", (pt["rv"] > q[0.33]) & (pt["rv"] <= q[0.67])),
              ("hot", pt["rv"] > q[0.67])]:
    g = pt[m]
    print(f"  {lbl}: n={len(g):3d}  WR={g['win'].mean():.1%}  touched%={g['touched_strike'].mean():.1%}")
r_touch, p_touch = stats.pearsonr(pt["rv"], pt["touched_strike"].astype(float))
print(f"  full-pop r2(touch)={r_touch**2:.4f} P={p_touch:.4f}  (n={len(pt)})")

pt.to_csv(f"{OUT}/s32_full_history_paths.csv", index=False)
print("DONE_S32")
