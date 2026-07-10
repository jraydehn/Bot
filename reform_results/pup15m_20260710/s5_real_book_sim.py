"""
S5 -- real-book dollar simulation: EXISTING (enter at decision_time) vs
btc_1h_WAIT (if p15 disagrees/neutral at entry, wait for first agreeing
scan cycle of the same ticker; re-enter at that cycle's price with the SAME
bet_amount; if it never agrees, trade is missed -> $0).

Uses the ACTUAL taken BTC hourly book (results/paper_trades.csv, KXBTCD
tickers, resolved rows) with actual bet sizes and would_pnl. Prices along
the waiting path come from the scan archive (hourly_archive_p15.csv, which
already carries the zero-lookahead p15 join). Era-split at 06-30.
Prior benchmark (old signal, pre-reform only): EXISTING $+574 vs WAIT $+556.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
BAND = 0.02
REFORM = pd.Timestamp("2026-06-30 00:00:00", tz="UTC")

sig = pd.read_csv(f"{OUT}/pup15m_series_2026.csv", parse_dates=["bar_open", "effective"])
sig = sig.sort_values("effective")
arc = pd.read_csv(f"{OUT}/hourly_archive_p15.csv", parse_dates=["logged_at"], low_memory=False)
arc = arc.sort_values(["contract_ticker", "logged_at"])

bk = pd.read_csv("results/paper_trades.csv", low_memory=False)
bk = bk[bk["contract_ticker"].astype(str).str.startswith("KXBTCD")]
bk["decision_time"] = pd.to_datetime(bk["decision_time"], utc=True, errors="coerce", format="mixed")
bk = bk.dropna(subset=["decision_time", "would_pnl", "resolved_yes", "bet_amount", "p_market"])
bk = bk[bk["decision_time"] >= sig["effective"].min()]
print(f"real resolved BTC hourly trades in window: {len(bk)}  "
      f"{bk['decision_time'].min().date()} -> {bk['decision_time'].max().date()}")
print(f"actual-book would_pnl total: ${bk['would_pnl'].sum():+,.2f}")

bk = pd.merge_asof(bk.sort_values("decision_time"), sig[["effective", "p15"]],
                   left_on="decision_time", right_on="effective", direction="backward")
bk["dir15"] = np.where(bk["p15"] > 0.5 + BAND, 1, np.where(bk["p15"] < 0.5 - BAND, -1, 0))
bk["want"] = np.where(bk["side"] == "yes", 1, -1)
bk["era"] = np.where(bk["decision_time"] < REFORM, "pre", "post")

def simulate(row):
    """returns (action, wait_pnl)"""
    if row["dir15"] == row["want"]:
        return "unchanged", row["would_pnl"]
    g = arc[(arc["contract_ticker"] == row["contract_ticker"])
            & (arc["logged_at"] > row["decision_time"])
            & (arc["dir15"] == row["want"])]
    if not len(g):
        return "missed", 0.0
    pm_new = g.iloc[0]["p_market"]
    cost_new = pm_new if row["side"] == "yes" else 1 - pm_new
    if cost_new <= 0.01 or cost_new >= 0.99:
        return "missed", 0.0            # unfillable / degenerate quote
    win = row["resolved_yes"] if row["side"] == "yes" else 1 - row["resolved_yes"]
    n_contracts = row["bet_amount"] / (row["p_market"] if row["side"] == "yes" else 1 - row["p_market"])
    stake_new = n_contracts * cost_new  # same contract count, new price
    pnl = n_contracts * (1 - cost_new) if win else -stake_new
    return "re-entered", round(pnl, 2)

acts, pnls = [], []
for _, row in bk.iterrows():
    a, p = simulate(row)
    acts.append(a); pnls.append(p)
bk["action"], bk["wait_pnl"] = acts, pnls
bk.to_csv(f"{OUT}/real_book_wait_sim.csv", index=False)

print(f"\naction breakdown: {bk['action'].value_counts().to_dict()}")
print(f"disagree/neutral at entry: {(bk['action']!='unchanged').sum()} of {len(bk)}")

bk["week"] = bk["decision_time"].dt.to_period("W").astype(str)
print(f"\n{'week':26s} {'n':>4s} {'EXISTING $':>12s} {'WAIT $':>10s} {'delta':>9s}")
for (era, wk), g in bk.groupby(["era", "week"]):
    print(f"{wk:26s} {len(g):4d} {g['would_pnl'].sum():12.2f} {g['wait_pnl'].sum():10.2f} "
          f"{g['wait_pnl'].sum()-g['would_pnl'].sum():+9.2f}  [{era}]")
for era in ["pre", "post"]:
    g = bk[bk["era"] == era]
    print(f"\nTOTAL {era}-reform (n={len(g)}): EXISTING ${g['would_pnl'].sum():+,.2f}  "
          f"WAIT ${g['wait_pnl'].sum():+,.2f}  DELTA ${g['wait_pnl'].sum()-g['would_pnl'].sum():+,.2f}")
    m = g[g["action"] != "unchanged"]
    print(f"  modified trades only (n={len(m)}): existing ${m['would_pnl'].sum():+,.2f} "
          f"-> wait ${m['wait_pnl'].sum():+,.2f}")
    for a in ["re-entered", "missed"]:
        s = m[m["action"] == a]
        print(f"    {a}: n={len(s)}  their existing pnl ${s['would_pnl'].sum():+,.2f}"
              + (f"  their wait pnl ${s['wait_pnl'].sum():+,.2f}" if a == "re-entered" else "  (wait pnl $0)"))
print("DONE_S5")
