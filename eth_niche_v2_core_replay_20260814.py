"""ETH hourly niche v2 'CORE' — pre-registered gated variant, frozen 2026-08-14.

The v2 forward book is -$2,012 (all weeks red). A 509-test protocol sweep
(outcome columns excluded; spot/strike rejected as date proxies) found a
salvageable core: KEEP only trades where the decision-scan's body_15m >=
0.79 (decisive candle). Discovery: kept n=131, +$2,254, 3/3 weeks positive
(delta +$4,266, p=0.001). MINED -> this script is the referee; forward rows
(dt >= FREEZE) are the decision evidence. Evaluate ~08-25 alongside the BTC
rescue books. Also banked: v2's YES trades LOSE when composite_p_up >= 0.5
(inverse witness #5 this week — crowded-bull YES is the toxic bucket).
"""
import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
FREEZE = pd.Timestamp("2026-08-14 07:00", tz="UTC")
nv = pd.read_csv("results/paper_trades_eth_hourly_niche_v2.csv", low_memory=False)
nv["dt"] = pd.to_datetime(nv["logged_at"], errors="coerce", utc=True, format="mixed")
for c in ["p_market", "would_pnl_net", "resolved_yes"]:
    nv[c] = pd.to_numeric(nv[c], errors="coerce")
start = FREEZE if "--forward" in sys.argv else pd.Timestamp("2026-07-28 12:00", tz="UTC")
g = nv[nv["would_pnl_net"].notna() & (nv["dt"] >= start)].sort_values("dt").copy()
ar = pd.read_csv("results/eth_scan_archive.csv", low_memory=False,
                 usecols=["logged_at", "contract_ticker", "body_15m"])
ar["dt"] = pd.to_datetime(ar["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
ar = ar.dropna(subset=["dt"]).sort_values("dt")
ar["body_15m"] = pd.to_numeric(ar["body_15m"], errors="coerce")
amap = {tk: (grp["dt"].astype("int64").values / 1e9, grp["body_15m"].values)
        for tk, grp in ar.groupby("contract_ticker")}
body = []
for tk, t in zip(g["contract_ticker"], g["dt"].astype("int64").values / 1e9):
    a = amap.get(tk)
    i = np.searchsorted(a[0], t + 2, side="right") - 1 if a else -1
    body.append(a[1][i] if (a and i >= 0) else np.nan)
g["body_15m"] = body
core = g[(pd.Series(body, index=g.index) >= 0.79).fillna(False)]
for nm, b in [("v2 full book", g), ("v2 CORE (body>=0.79)", core)]:
    if not len(b):
        print(f"{nm}: no rows yet"); continue
    wk = b.groupby(b["dt"].dt.isocalendar().week)["would_pnl_net"].sum()
    wr = (b["resolved_yes"] == 1).mean()
    print(f"{nm}: n={len(b)} net=${b['would_pnl_net'].sum():+,.0f} WR={wr:.0%} "
          f"weekly={{{', '.join(f'{int(k)}: {v:+.0f}' for k, v in wk.items())}}}")
