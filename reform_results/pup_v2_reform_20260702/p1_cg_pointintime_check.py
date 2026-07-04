#!/usr/bin/env python3
"""Re-test the CG futures-flow residual IC with a strictly point-in-time series.

Why: backfill_cg_futures_cvd.py aligned rows to floor(logged_at,'4h') = the
CONTAINING 4h bar (future flow, same idiom as the p_up_v2 leak), while the live
logger uses the last COMPLETED bar (data[-2]). Most archive-era rows were
backfilled -> the naive residual IC (0.13) is contaminated.

Here: fetch the 4h aggregated futures taker history from CoinGlass, assign each
decision time L the last bar whose CLOSE (open+4h) <= L. Recompute rank-IC vs
the ATM residual (y - pm) and vs y, per week; also the same test with the
CONTAINING bar to quantify the contamination.
Output: cg_pointintime_ic.csv + stdout log.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
OUT = PROJ / "reform_results" / "pup_v2_reform_20260702"

_BASE = "https://open-api-v4.coinglass.com"
_KEY = os.environ.get("COINGLASS_API_KEY", "8f0a30c29a5e424ba2641f649051786b")

r = requests.get(f"{_BASE}/api/futures/aggregated-taker-buy-sell-volume/history",
                 headers={"CG-API-KEY": _KEY},
                 params={"symbol": "BTC", "interval": "4h", "limit": 500,
                         "exchange_list": "Binance,OKX,Bybit"}, timeout=20)
body = r.json()
assert body.get("code") == "0", body
cg = pd.DataFrame(body["data"])
cg["bar_open"] = pd.to_datetime(cg["time"], unit="ms", utc=True)
cg["buy"] = cg["aggregated_buy_volume_usd"].astype(float)
cg["sell"] = cg["aggregated_sell_volume_usd"].astype(float)
cg["delta"] = cg["buy"] - cg["sell"]
cg["ratio"] = cg["buy"] / cg["sell"].replace(0, np.nan)
cg = cg.sort_values("bar_open").drop_duplicates("bar_open").reset_index(drop=True)
cg["cvd_12h"] = cg["delta"].rolling(3).sum()
cg["bar_close"] = cg["bar_open"] + pd.Timedelta(hours=4)
print(f"CG bars: {len(cg)}  {cg.bar_open.min()} -> {cg.bar_open.max()}")

atm = pd.read_parquet(OUT / "archive_hourly_frame.parquet")
atm = atm.dropna(subset=["p_market"]).copy()
atm["resid"] = atm.y.astype(float) - atm.p_market
atm = atm.sort_values("logged_at")

cols = ["delta", "ratio", "cvd_12h"]
# point-in-time: last bar CLOSED at or before decision (logged_at)
pit = pd.merge_asof(atm, cg[["bar_close"] + cols].rename(columns={c: f"pit_{c}" for c in cols}),
                    left_on="logged_at", right_on="bar_close", direction="backward")
# containing bar (replicates the backfill leak)
pit["_floor"] = pit.logged_at.dt.floor("4h")
pit = pit.merge(cg[["bar_open"] + cols].rename(columns={c: f"cont_{c}" for c in cols}),
                left_on="_floor", right_on="bar_open", how="left")
pit["week"] = pit.close_ts.dt.to_period("W-WED").astype(str)

rows = []
for align in ("pit", "cont"):
    for c in cols:
        col = f"{align}_{c}"
        g = pit[[col, "resid", "y", "week"]].dropna()
        if len(g) < 100:
            continue
        icr = spearmanr(g[col], g.resid)
        icy = spearmanr(g[col], g.y)
        wk = g.groupby("week").apply(
            lambda x: spearmanr(x[col], x.resid).statistic if len(x) > 15 else np.nan).dropna()
        rows.append({"align": "point_in_time" if align == "pit" else "containing(leaky)",
                     "feature": f"cg_futures_{c}", "n": len(g),
                     "ic_vs_resid": icr.statistic, "p_resid": icr.pvalue,
                     "ic_vs_y": icy.statistic, "p_y": icy.pvalue,
                     "n_weeks": len(wk), "pct_weeks_pos": (wk > 0).mean()})
res = pd.DataFrame(rows)
res.to_csv(OUT / "cg_pointintime_ic.csv", index=False)
print(res.round(4).to_string(index=False))
