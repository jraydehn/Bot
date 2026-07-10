"""
S25 -- what distinguishes the overnight winning stretch from the 07-06->07-10
degradation, at the PRICE-PATH level (not just daily vol ratios, which don't
separate 07-10-morning-loss from 07-10-overnight-win -- both were "calm").

For every taken YES trade, reconstruct the 1m price path from decision_time
to close_time (the ~10.5min window the contract actually lives) and compute:
  net_move_pct     : (close - decision_spot)/decision_spot, the resolved outcome
  mae_pct          : max adverse excursion -- how far BELOW decision_spot the
                     path dipped at its worst point (the buffer-eating metric)
  touched_strike   : did the path ever trade AT or BELOW the strike during
                     the window, even if it recovered by close?
  path_rv          : realized vol of 1-min log returns WITHIN the window
                     (choppiness/whipsaw, independent of net drift)
  reversals        : count of 1-min return sign flips within the window
  monotonic_frac   : fraction of 1-min bars moving in the SAME direction as
                     the window's eventual net move (trend "cleanliness")
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

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)].dropna(subset=["would_pnl"])
t = t[t["decision_time"] >= "2026-07-01"].sort_values("decision_time").copy()
t["win"] = t["resolved_yes"]

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
    r1 = path.pct_change().dropna()
    net_move = (path.iloc[-1] - decision_spot) / decision_spot
    mae = (path.min() - decision_spot) / decision_spot   # negative = dipped below entry
    touched = (path.min() <= strike)
    path_rv = r1.std() if len(r1) > 1 else np.nan
    reversals = int((np.sign(r1) != np.sign(r1.shift(1))).sum()) if len(r1) > 1 else 0
    net_sign = np.sign(net_move) if net_move != 0 else 1
    monotonic_frac = (np.sign(r1) == net_sign).mean() if len(r1) > 1 else np.nan
    rows.append(dict(decision_time=dt, ticker=row.contract_ticker, win=row.win,
                     would_pnl=row.would_pnl, net_move_pct=net_move * 100, mae_pct=mae * 100,
                     touched_strike=touched, path_rv=path_rv, reversals=reversals,
                     n_bars=len(r1), monotonic_frac=monotonic_frac))

pt = pd.DataFrame(rows)
print(f"reconstructed paths: {len(pt)} of {len(t)} trades")

groups = {
    "07-01->07-05 (good)":        (pt["decision_time"] >= "2026-07-01") & (pt["decision_time"] < "2026-07-06"),
    "07-06->07-07 (crash)":       (pt["decision_time"] >= "2026-07-06") & (pt["decision_time"] < "2026-07-08"),
    "07-08->07-10 07:50 (bad)":   (pt["decision_time"] >= "2026-07-08") & (pt["decision_time"] < "2026-07-10 07:50:00+00:00"),
    "07-10 07:50-> (win)":        pt["decision_time"] >= "2026-07-10 07:50:00+00:00",
}
print(f"\n{'group':30s} {'n':>3s} {'WR':>6s} {'mae_pct':>8s} {'path_rv%':>9s} {'revers':>7s} {'monofrac':>9s} {'touched%':>9s}")
for lbl, m in groups.items():
    g = pt[m]
    if len(g) == 0:
        continue
    print(f"{lbl:30s} {len(g):3d} {g['win'].mean():6.1%} {g['mae_pct'].mean():8.4f} "
          f"{g['path_rv'].mean()*100:9.4f} {g['reversals'].mean():7.2f} {g['monotonic_frac'].mean():9.3f} "
          f"{g['touched_strike'].mean():9.1%}")

print("\n=== correlation of path features with win/loss (whole sample) ===")
for col in ["mae_pct", "path_rv", "reversals", "monotonic_frac"]:
    x = pt[col]
    winner = pt[pt["win"] == 1][col]
    loser = pt[pt["win"] == 0][col]
    print(f"  {col:16s}: winners mean={winner.mean():+.4f}  losers mean={loser.mean():+.4f}  "
          f"diff={winner.mean()-loser.mean():+.4f}")

print("\n=== touched_strike vs outcome (the direct mechanism check) ===")
ct = pt.groupby("touched_strike")["win"].agg(["size", "mean"])
print(ct.to_string())

pt.to_csv(f"{OUT}/s25_path_features.csv", index=False)
print("DONE_S25")
