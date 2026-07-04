#!/usr/bin/env python3
"""D(iii) rerun with point-in-time CG flow features (honest C group).

Replaces the backfilled (containing-bar, leaky) cg_futures_* columns with the
strictly point-in-time series (last CLOSED 4h bar <= logged_at) and repeats the
archive-era walk-forward combos + unit-$ backtest vs pm.
Fees note: Kalshi fee ~ 0.07*p*(1-p) per contract (~1.7c at p=0.5) is reported
alongside gross unit PnL.
Output: archive_wf_summary_pit.csv
"""
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
OUT = PROJ / "reform_results" / "pup_v2_reform_20260702"

_BASE = "https://open-api-v4.coinglass.com"
_KEY = os.environ.get("COINGLASS_API_KEY", "8f0a30c29a5e424ba2641f649051786b")
r = requests.get(f"{_BASE}/api/futures/aggregated-taker-buy-sell-volume/history",
                 headers={"CG-API-KEY": _KEY},
                 params={"symbol": "BTC", "interval": "4h", "limit": 500,
                         "exchange_list": "Binance,OKX,Bybit"}, timeout=20)
cg = pd.DataFrame(r.json()["data"])
cg["bar_open"] = pd.to_datetime(cg["time"], unit="ms", utc=True)
cg["delta"] = cg["aggregated_buy_volume_usd"].astype(float) - cg["aggregated_sell_volume_usd"].astype(float)
cg["ratio"] = cg["aggregated_buy_volume_usd"].astype(float) / cg["aggregated_sell_volume_usd"].astype(float).replace(0, np.nan)
cg = cg.sort_values("bar_open").drop_duplicates("bar_open")
cg["cvd_12h"] = cg["delta"].rolling(3).sum()
cg["bar_close"] = cg["bar_open"] + pd.Timedelta(hours=4)

atm = pd.read_parquet(OUT / "archive_hourly_frame.parquet").sort_values("logged_at")
atm = pd.merge_asof(atm, cg[["bar_close", "delta", "ratio", "cvd_12h"]].rename(
    columns={"delta": "pit_cg_delta", "ratio": "pit_cg_ratio", "cvd_12h": "pit_cg_cvd12"}),
    left_on="logged_at", right_on="bar_close", direction="backward")

C_PIT = ["funding_bias", "avg_funding_rate", "ls_long_pct", "oi_chg_pct",
         "liq_score", "liq_bias", "pit_cg_delta", "pit_cg_ratio", "pit_cg_cvd12"]
ev = atm.dropna(subset=["wf_p_A", "p_market"]).sort_values("close_ts").reset_index(drop=True)
ev["week"] = ev.close_ts.dt.to_period("W-WED").astype(str)
y = ev.y.values.astype(int)
ts = ev.close_ts
pm = ev.p_market.values

CONFS = {
    "pm_only":   ["p_market"],
    "Cpit_only": C_PIT,
    "A+D":       ["wf_p_A", "p_market"],
    "Cpit+D":    C_PIT + ["p_market"],
    "A+Cpit+D":  ["wf_p_A", "p_market"] + C_PIT,
}
week_starts = pd.date_range(ts.min().normalize() + pd.Timedelta(days=14), ts.max(), freq="7D")
rows = []
for name, feats in CONFS.items():
    X = ev[feats].values.astype(float)
    p = np.full(len(ev), np.nan)
    for ws in week_starts:
        te = np.where((ts >= ws) & (ts < ws + pd.Timedelta(days=7)))[0]
        tr = np.where(ts < ws - pd.Timedelta(hours=1))[0]
        if len(te) == 0 or len(tr) < 150:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                               num_leaves=7, min_child_samples=40, reg_lambda=5.0,
                               subsample=0.8, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(p)
    edge = p - pm
    bet = mask & (np.abs(edge) > 0.02)
    pnl = np.where(edge > 0, y - pm, pm - y)[bet]
    fee = (0.07 * pm * (1 - pm))[bet]
    rows.append({"config": name, "n_oos": int(mask.sum()),
                 "auc_oos": roc_auc_score(y[mask], p[mask]),
                 "n_bets": int(bet.sum()),
                 "unit_pnl_mean": pnl.mean() if bet.sum() else np.nan,
                 "unit_pnl_net_fee": (pnl - fee).mean() if bet.sum() else np.nan,
                 "unit_pnl_total": pnl.sum() if bet.sum() else np.nan})
rows.insert(0, {"config": "raw_pm_benchmark", "n_oos": len(ev),
                "auc_oos": roc_auc_score(y, pm), "n_bets": 0,
                "unit_pnl_mean": np.nan, "unit_pnl_net_fee": np.nan, "unit_pnl_total": np.nan})
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "archive_wf_summary_pit.csv", index=False)
print(summ.round(4).to_string(index=False))
