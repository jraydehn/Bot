#!/usr/bin/env python3
"""S8 — Deployment-role test: v3 honest score as an AGREEMENT FILTER on the
actual BTC 15m runner trades (the realistic marginal effect on top of the
existing gate stack, which s7 showed carries the PnL).

Rule: keep a taken trade iff the hour-level v3 score agrees with the side
(YES needs p_hat >= hi, NO needs p_hat <= lo). Standardized $100 stakes,
entry at p_market +/- spread/2, Kalshi 7% fee. One bet per ticker.

Result (2026-05-25 -> 07-04, n=806 actual trades):
  no filter:              WR=0.508  PnL=-$368
  agree @ 0.50/0.50:      kept n=250 WR=0.572 PnL=+$1,487 | dropped PnL=-$1,888
  agree @ 0.45/0.55(soft): kept n=733 WR=0.524 PnL=+$1,153
  weeks favoring agree-kept: 6/7
  YES: agree WR=0.864 +$1,339 vs disagree WR=0.701 -$857
  NO:  agree WR=0.432 +$148   vs disagree WR=0.393 -$1,030
  Old leaky p_up_v2 as the same filter: kept -$859 / dropped +$131 (WRONG sign)
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

df = pd.read_csv(PROJ / "results" / "paper_trades_btc15m.csv", low_memory=False)
df = df[df["asset"] == "BTC"].copy()
for c in ("spread", "p_market", "p_up_v2_btc"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["resolved"] = df["resolved_yes"].astype(str).str.lower().map(
    {"1.0": 1, "1": 1, "true": 1, "0.0": 0, "0": 0, "false": 0})
df["dt"] = pd.to_datetime(df["decision_time"], utc=True, format="mixed")
wf = pd.read_parquet(HERE / "wf_preds_FINAL.parquet")["p"]
df["p_hat"] = wf.reindex(pd.DatetimeIndex(df["dt"].dt.floor("h") - pd.Timedelta(hours=1), tz="UTC")).values

a = df[(df["decision"] == "trade") & df["resolved"].notna()].copy()
a["spread"] = a["spread"].fillna(0.01).clip(0, 0.1)
ask = np.where(a["side"] == "no", 1 - a["p_market"] + a["spread"] / 2,
               a["p_market"] + a["spread"] / 2)
price = np.clip(ask, 0.03, 0.99)
ct = 100 / price
fee = 0.07 * price * (1 - price) * ct
win = np.where(a["side"] == "no", 1 - a["resolved"], a["resolved"])
a["pnl"] = np.where(win == 1, ct * (1 - price) - fee, -100 - fee)
a["win"] = win
a = a.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
print(f"actual trades std-$100: n={len(a)} WR={a.win.mean():.3f} PnL=${a.pnl.sum():,.0f}")

for col, label in [("p_hat", "new v3"), ("p_up_v2_btc", "old leaky p_up_v2")]:
    print(f"\n--- {label} agreement filter ---")
    for lo, hi in [(0.50, 0.50), (0.48, 0.52), (0.45, 0.55), (0.46, 0.54)]:
        s = a[a[col].notna()]
        agree = np.where(s["side"] == "yes", s[col] >= hi, s[col] <= lo)
        k, d = s[agree], s[~agree]
        print(f"  yes>={hi} / no<={lo}: kept n={len(k)} WR={k.win.mean():.3f} "
              f"${k.pnl.sum():,.0f} | dropped n={len(d)} ${d.pnl.sum():,.0f}")

s = a[a["p_hat"].notna()].copy()
s["agree"] = np.where(s["side"] == "yes", s["p_hat"] >= 0.5, s["p_hat"] <= 0.5)
s["wk"] = s["dt"].dt.to_period("W-WED").astype(str)
g = s.groupby("wk").apply(lambda x: x[x.agree].pnl.mean() - x[~x.agree].pnl.mean())
print(f"\nweeks favoring agree: {(g > 0).sum()}/{g.notna().sum()}")
