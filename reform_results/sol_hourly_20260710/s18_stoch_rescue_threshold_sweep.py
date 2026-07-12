"""
S18 -- threshold sweep for the sol_markov_gate NO rescue (currently
stoch_k_1h>=86.1 under 4h=Sideways). s17 found this exact threshold was
strong in May (+12.7pp) but decayed to negative in June/July (-1.7pp,
-2.5pp). Sweep the threshold to see whether ANY cutoff is stable across all
three months, or whether the whole stoch_k_1h-as-rescue idea has decayed
regardless of the specific level chosen (i.e. don't just move the goalpost
to wherever happens to look good on the full-period average -- require
monthly consistency).
"""
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

p1h = sorted(glob.glob("data/binanceus_SOLUSDT_1h_2024-01-01_*.parquet"))[-1]
c1h = pd.read_parquet(p1h)["close"].astype(float)
c1h.index = pd.to_datetime(c1h.index, utc=True)
c1h = c1h.sort_index()


def regime_series(close_1h, rule, window, thr):
    c = close_1h.resample(rule, origin="start_day").last().dropna()
    rr = c.pct_change(window)
    reg = pd.Series(np.where(rr > thr, "Bull", np.where(rr < -thr, "Bear", "Sideways")), index=c.index)
    reg[rr.isna()] = None
    return reg


reg_4h = regime_series(c1h, "4h", 20, 0.025)

df = pd.read_csv("results/sol_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_market", "resolved_yes",
                            "offset_pct", "stoch_k_1h"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "p_market", "resolved_yes", "stoch_k_1h"])
df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
df["stoch_k_1h"] = pd.to_numeric(df["stoch_k_1h"], errors="coerce")
df = df.sort_values("logged_at")

r = reg_4h.reset_index(); r.columns = ["ts", "markov_4h"]; r = r.sort_values("ts")
df = pd.merge_asof(df, r, left_on="logged_at", right_on="ts", direction="backward").drop(columns=["ts"])
df = df.dropna(subset=["markov_4h"])
side4h = df[df["markov_4h"] == "Sideways"].copy()
side4h["month"] = side4h["logged_at"].dt.to_period("M").astype(str)
print(f"4h=Sideways candidates: {len(side4h)}  tickers: {side4h['contract_ticker'].nunique()}")


def no_stats(sub):
    win = 1 - sub["resolved_yes"]; cost = 1 - sub["p_market"]
    t = pd.DataFrame({"win": win, "cost": cost, "tk": sub["contract_ticker"]})
    t = t[t["cost"].between(0.05, 0.95)]
    tk = t.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    if len(tk) < 10:
        return None
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    return dict(n=len(t), tk=len(tk), wr=tk["win"].mean(), be=tk["cost"].mean(),
                edge=tk["win"].mean() - tk["cost"].mean(), pnl=pnl.sum())


THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 86.1, 90, 92, 94, 95, 96, 97, 98, 99]
print(f"\n{'thresh':>7s} {'overall':>28s} | {'2026-05':>22s} | {'2026-06':>22s} | {'2026-07':>22s} | consistent?")
for th in THRESHOLDS:
    sub = side4h[side4h["stoch_k_1h"] >= th]
    r_all = no_stats(sub)
    row = f"{th:7.1f} "
    if r_all:
        row += f"n={r_all['n']:4d} tk={r_all['tk']:3d} edge={r_all['edge']:+6.1%} pnl=${r_all['pnl']:8.0f} |"
    else:
        row += f"{'(thin)':>28s} |"
    monthly_edges = []
    for m in ["2026-05", "2026-06", "2026-07"]:
        gm = sub[sub["month"] == m]
        rm = no_stats(gm)
        if rm:
            row += f" edge={rm['edge']:+6.1%} pnl=${rm['pnl']:7.0f} |"
            monthly_edges.append(rm["edge"])
        else:
            row += f" {'(thin)':>20s} |"
            monthly_edges.append(None)
    valid = [e for e in monthly_edges if e is not None]
    consistent = len(valid) >= 2 and all(e > 0 for e in valid)
    row += f" {'YES' if consistent else 'no'}"
    print(row)

print("\nDONE_S18")
