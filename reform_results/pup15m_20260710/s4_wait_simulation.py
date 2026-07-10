"""
S4 -- wait-for-agreement entry-timing simulation with the NEW p15 signal.
The user's mechanic: hourly candidate appears; if 15m direction disagrees
with the side, WAIT; re-check on each later scan cycle of the SAME contract;
enter at the first cycle where 15m agrees (or never, if it never agrees).

Per contract: entry-decision point = FIRST archive row of that ticker.
  ENTER-NOW:  cost = p_side at first row;      edge = win - cost
  WAIT:       cost = p_side at first agreeing row (missed if none);
              edge = win - cost;  strategy avg counts missed as 0.
Also the pure-timing probe: cost/edge at the first row >=15min later,
regardless of signal (does waiting alone help? -- prior answer: no).
Era split at the 06-30 reform. Ticker-clustered by construction (one
decision per ticker). p15 joined zero-lookahead (effective <= logged_at).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
BAND = 0.02
REFORM = pd.Timestamp("2026-06-30 00:00:00", tz="UTC")

arc = pd.read_csv(f"{OUT}/hourly_archive_p15.csv",
                  parse_dates=["logged_at"], low_memory=False)
arc = arc.sort_values(["contract_ticker", "logged_at"])
print(f"rows: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")

results = []
for tk, g in arc.groupby("contract_ticker", sort=False):
    g = g.reset_index(drop=True)
    first = g.iloc[0]
    win_yes = first["resolved_yes"]
    era = "pre" if first["logged_at"] < REFORM else "post"
    for side in ["yes", "no"]:
        want = 1 if side == "yes" else -1
        cost_now = first["p_market"] if side == "yes" else 1 - first["p_market"]
        win = win_yes if side == "yes" else 1 - win_yes
        agree_now = first["dir15"] == want
        neutral_now = first["dir15"] == 0
        # pure-timing probe: first row >= 15 min later
        later = g[g["logged_at"] >= first["logged_at"] + pd.Timedelta("15min")]
        cost_15 = (later.iloc[0]["p_market"] if side == "yes"
                   else 1 - later.iloc[0]["p_market"]) if len(later) else np.nan
        # wait-for-agreement (only meaningful when NOT agreeing now)
        cost_wait, waited_min = np.nan, np.nan
        if not agree_now:
            ag = g[(g["dir15"] == want) & (g["logged_at"] > first["logged_at"])]
            if len(ag):
                cost_wait = ag.iloc[0]["p_market"] if side == "yes" else 1 - ag.iloc[0]["p_market"]
                waited_min = (ag.iloc[0]["logged_at"] - first["logged_at"]).total_seconds() / 60
        results.append(dict(ticker=tk, side=side, era=era, win=win,
                            agree_now=agree_now, neutral_now=neutral_now,
                            cost_now=cost_now, cost_15=cost_15,
                            cost_wait=cost_wait, waited_min=waited_min,
                            week=str(first["logged_at"].to_period("W"))))
r = pd.DataFrame(results)
r.to_csv(f"{OUT}/wait_sim_decisions.csv", index=False)

for side in ["yes", "no"]:
    print(f"\n================ side={side.upper()} ================")
    for era in ["pre", "post"]:
        d = r[(r["side"] == side) & (r["era"] == era) & (~r["agree_now"])]
        if len(d) < 20:
            print(f"  {era}: n={len(d)} too thin"); continue
        edge_now = (d["win"] - d["cost_now"]).mean()
        h15 = d.dropna(subset=["cost_15"])
        d15c = (h15["cost_15"] - h15["cost_now"]).mean()
        e15 = (h15["win"] - h15["cost_15"]).mean()
        w = d.dropna(subset=["cost_wait"])
        missed = len(d) - len(w)
        dcost_w = (w["cost_wait"] - w["cost_now"]).mean() if len(w) else np.nan
        edge_w_entered = (w["win"] - w["cost_wait"]).mean() if len(w) else np.nan
        edge_now_of_entered = (w["win"] - w["cost_now"]).mean() if len(w) else np.nan
        strat_wait = (w["win"] - w["cost_wait"]).sum() / len(d)   # missed = 0
        print(f"  DISAGREE/NEUTRAL-at-entry {era}: n={len(d)}  edge_now={edge_now:+.4f}")
        print(f"    +15min fixed wait:        dcost={d15c:+.4f}  edge={e15:+.4f}")
        print(f"    wait-for-agree: entered {len(w)}/{len(d)} ({missed/len(d):.0%} missed)  "
              f"median_wait={w['waited_min'].median() if len(w) else float('nan'):.0f}min")
        print(f"      dcost={dcost_w:+.4f}  edge_when_entered={edge_w_entered:+.4f} "
              f"(same tickers enter-now: {edge_now_of_entered:+.4f})")
        print(f"    STRATEGY avg/candidate (missed=0): wait={strat_wait:+.4f} vs now={edge_now:+.4f}")
        # control: agree-at-entry population
        a = r[(r["side"] == side) & (r["era"] == era) & (r["agree_now"])]
        print(f"  AGREE-at-entry control {era}: n={len(a)}  edge_now={(a['win']-a['cost_now']).mean():+.4f}")
print("DONE_S4")
