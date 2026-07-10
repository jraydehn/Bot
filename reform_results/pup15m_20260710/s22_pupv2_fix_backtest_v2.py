"""
S22 -- corrected backtest of the p_up_v2 fix. s21 hand-rewrote the vol
formulas and got them wrong (norm.ppf(pm) instead of norm.ppf(1-pm), and
skipped a sqrt(tau) step) -- produced a nonsensical ~28% WR. This version
imports compute_p_yes_pup_v2_15m / compute_p_no_pup_v2_15m and K_PUP_V2_YES/
K_PUP_V2_NO DIRECTLY from paper_trade_runner_15m.py so there is zero
translation risk, and sources realized_vol_annual from the scan archive's
own logged column (exactly what the live sig dict would have contained).
"""
import warnings
import sys
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
OUT = "reform_results/pup15m_20260710"

import paper_trade_runner_15m as R  # noqa: E402

K_YES, K_NO = R.K_PUP_V2_YES, R.K_PUP_V2_NO
EDGE_THRESH = 0.04
print(f"imported live constants: K_YES={K_YES}  K_NO={K_NO}")

hourly = pd.read_csv("results/paper_trades.csv", usecols=["logged_at", "p_up_v2"], low_memory=False)
hourly["p_up_v2"] = pd.to_numeric(hourly["p_up_v2"], errors="coerce")
hourly["logged_at"] = pd.to_datetime(hourly["logged_at"], utc=True, errors="coerce", format="mixed")
hourly = hourly.dropna(subset=["p_up_v2", "logged_at"]).sort_values("logged_at")
hourly = hourly.drop_duplicates(subset="logged_at", keep="last")

arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "resolved_yes", "realized_vol_annual"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "tau_minutes",
                         "spot", "strike", "realized_vol_annual"])
arc = arc[(arc["realized_vol_annual"] > 0) & (arc["logged_at"] >= "2026-06-01")].sort_values("logged_at")
print(f"candidate rows: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")

arc = pd.merge_asof(arc, hourly.rename(columns={"logged_at": "pu_ts"}),
                    left_on="logged_at", right_on="pu_ts", direction="backward",
                    tolerance=pd.Timedelta("120min"))
arc = arc.dropna(subset=["p_up_v2"])
print(f"candidate rows with valid (fresh, <=2h) p_up_v2: {len(arc)}")

p_yes, p_no = [], []
for row in arc.itertuples(index=False):
    sig = {"realized_vol_annual": row.realized_vol_annual}  # vol_multi absent -> uses this fallback, matches live when vol_multi unset
    py = R.compute_p_yes_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, row.p_up_v2, row.p_market)
    pn = R.compute_p_no_pup_v2_15m(row.spot, row.strike, row.tau_minutes, sig, row.p_up_v2, row.p_market)
    p_yes.append(py); p_no.append(pn)
arc["p_model_yes"] = p_yes
arc["p_model_no"] = p_no
arc["edge_yes"] = arc["p_model_yes"] - arc["p_market"]
arc["edge_no"] = arc["p_model_no"] - (1 - arc["p_market"])
arc["best_side"] = np.where(arc["edge_yes"] >= arc["edge_no"], "yes", "no")
arc["best_edge"] = np.where(arc["edge_yes"] >= arc["edge_no"], arc["edge_yes"], arc["edge_no"])

print("\nraw edge distribution (all candidates, both sides' best):")
print(arc["best_edge"].describe()[["mean", "min", "25%", "50%", "75%", "max"]].round(4).to_string())
print("p_model_yes describe:", arc["p_model_yes"].describe()[["mean", "min", "max"]].round(3).to_dict())
print("p_model_no describe:", arc["p_model_no"].describe()[["mean", "min", "max"]].round(3).to_dict())

qual = arc[arc["best_edge"] >= EDGE_THRESH].copy()
print(f"\nqualifying (edge>={EDGE_THRESH}): {len(qual)} of {len(arc)}")
print("side mix:", qual["best_side"].value_counts().to_dict())

per_ticker = qual.sort_values("best_edge", ascending=False).drop_duplicates("contract_ticker")
per_ticker["win"] = np.where(per_ticker["best_side"] == "yes", per_ticker["resolved_yes"],
                             1 - per_ticker["resolved_yes"])
per_ticker["cost"] = np.where(per_ticker["best_side"] == "yes", per_ticker["p_market"],
                              1 - per_ticker["p_market"])
per_ticker["day"] = per_ticker["logged_at"].dt.date
print(f"\nsimulated trades (one per ticker, best edge): {len(per_ticker)}")
print(per_ticker.groupby("best_side").agg(n=("win", "size"), wr=("win", "mean"),
     be=("cost", "mean")).round(3).to_string())
print(f"overall edge = WR - BE: yes={per_ticker[per_ticker.best_side=='yes']['win'].mean()-per_ticker[per_ticker.best_side=='yes']['cost'].mean():+.3f}"
      f"  no={per_ticker[per_ticker.best_side=='no']['win'].mean()-per_ticker[per_ticker.best_side=='no']['cost'].mean():+.3f}")

print("\nby day:")
d = per_ticker.groupby("day").agg(n=("win", "size"), n_yes=("best_side", lambda s: (s == "yes").sum()),
     n_no=("best_side", lambda s: (s == "no").sum()), wr=("win", "mean"),
     avg_cost=("cost", "mean")).round(3)
print(d.to_string())

real = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
real["decision_time"] = pd.to_datetime(real["decision_time"], utc=True, errors="coerce", format="mixed")
rt = real[real["side"].isin(["yes", "no"]) & (pd.to_numeric(real["bet_amount"], errors="coerce") > 0)]
rt = rt.dropna(subset=["would_pnl"])
rt["day"] = rt["decision_time"].dt.date
real_daily = rt.groupby("day").agg(real_n=("would_pnl", "size"), real_pnl=("would_pnl", "sum"))
print("\nJuly: simulated (fixed) vs actual (broken-fallback) book:")
jul = d[d.index >= pd.Timestamp("2026-07-01").date()].join(real_daily, how="left")
print(jul.to_string())

per_ticker.to_csv(f"{OUT}/pupv2_fix_backtest_v2.csv", index=False)
print("DONE_S22")
