"""Phase 1 — ETH hourly pm-band baseline map (both sides), fee-net.

Mirrors eth_hourly_fav_runner accounting exactly: flat $100, one bet per
contract (first scan entering the band), fee = (stake/cost)*0.07*pm*(1-pm).
Windows match the fav validation: May-Jun / July / Aug-forward.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
import sys as _sys
ASSET = _sys.argv[1].lower() if len(_sys.argv) > 1 else "eth"
ARCH = BASE / "results" / f"{ASSET}_scan_archive.csv"

USE = ["logged_at", "contract_ticker", "close_ts", "p_market", "tau_minutes",
       "resolved_yes"]
df = pd.read_csv(ARCH, usecols=USE, low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
for c in ["p_market", "tau_minutes", "resolved_yes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["dt", "p_market"])
print(f"scans: {len(df)}  range: {df['dt'].min()} .. {df['dt'].max()}")

# resolution map (last resolved row per contract)
res = (df.dropna(subset=["resolved_yes"])
         .drop_duplicates("contract_ticker", keep="last")
         .set_index("contract_ticker")["resolved_yes"])
print(f"contracts seen: {df['contract_ticker'].nunique()}  resolved: {len(res)}")

WINDOWS = [("May-Jun", "2026-05-01", "2026-07-01"),
           ("July",    "2026-07-01", "2026-08-01"),
           ("Aug-fwd", "2026-08-01", "2026-12-31")]

BANDS = [(0.03, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
         (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.97), (0.97, 0.995)]

def band_trades(side, lo, hi):
    m = (df["p_market"] >= lo) & (df["p_market"] < hi) & (df["tau_minutes"] > 0)
    t = df[m].sort_values("dt").drop_duplicates("contract_ticker", keep="first").copy()
    t["resolved_yes"] = t["contract_ticker"].map(res)
    t = t.dropna(subset=["resolved_yes"])
    pm = t["p_market"]
    if side == "yes":
        win = t["resolved_yes"] == 1
        cost = pm
    else:
        win = t["resolved_yes"] == 0
        cost = 1 - pm
    gross = np.where(win, 100 * (1 - cost) / cost, -100.0)
    fee = (100 / cost) * 0.07 * pm * (1 - pm)
    t["pnl_net"] = gross - fee
    t["win"] = win
    t["be"] = cost + 0.07 * pm * (1 - pm)   # breakeven win prob incl fee
    return t

rows = []
for side in ["yes", "no"]:
    for lo, hi in BANDS:
        t = band_trades(side, lo, hi)
        if t.empty:
            continue
        r = {"side": side, "band": f"{lo:.2f}-{hi:.2f}", "n": len(t),
             "events": t["close_ts"].nunique(),
             "wr": t["win"].mean(), "be": t["be"].mean(),
             "pnl": t["pnl_net"].sum()}
        for wname, ws, we in WINDOWS:
            w = t[(t["dt"] >= ws) & (t["dt"] < we)]
            r[wname] = round(w["pnl_net"].sum()) if len(w) else 0
            r[wname + "_n"] = len(w)
        rows.append(r)

out = pd.DataFrame(rows)
out["pnl"] = out["pnl"].round()
out["wr"] = (out["wr"] * 100).round(1)
out["be"] = (out["be"] * 100).round(1)
print(out.to_string(index=False))
