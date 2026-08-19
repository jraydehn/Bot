"""Phase 3 — harden the rescue candidates.

For each candidate: per-window PnL, weekly green count, MC p-value under a
market-calibrated null (win ~ Bernoulli(pm), fees included), threshold
neighbors, worst single expiry event, and a 1c adverse-fill stress (archive
pm is MID; live/paper fills pay spread).
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
BASE = Path(__file__).resolve().parent.parent
import sys as _sys
ASSET = _sys.argv[1].lower() if len(_sys.argv) > 1 else "eth"
ARCH = BASE / "results" / f"{ASSET}_scan_archive.csv"

USE = ["logged_at", "contract_ticker", "close_ts", "p_market", "tau_minutes",
       "resolved_yes", "liq_bias", "vwap_stretch_score"]
df = pd.read_csv(ARCH, usecols=USE, low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
for c in ["p_market", "resolved_yes", "liq_bias", "vwap_stretch_score", "tau_minutes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["dt", "p_market"])
df = df[df["tau_minutes"] > 0]
res = (df.dropna(subset=["resolved_yes"])
         .drop_duplicates("contract_ticker", keep="last")
         .set_index("contract_ticker")["resolved_yes"])
df["outcome"] = df["contract_ticker"].map(res)

WINDOWS = [("May-Jun", "2026-05-01", "2026-07-01"),
           ("July", "2026-07-01", "2026-08-01"),
           ("Aug-fwd", "2026-08-01", "2026-12-31")]

def make_trades(side, lo, hi, cond_col=None, cond_op=None, cond_val=None):
    m = (df["p_market"] >= lo) & (df["p_market"] < hi)
    if cond_col is not None:
        c = df[cond_col]
        m &= (c <= cond_val) if cond_op == "<=" else (c >= cond_val)
    t = df[m & df["outcome"].notna()].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first").copy()
    t["side"] = side
    return t

def pnl_vec(t, slip=0.0):
    pm = t["p_market"]
    yes = t["side"] == "yes"
    win = np.where(yes, t["outcome"] == 1, t["outcome"] == 0)
    cost = np.where(yes, pm, 1 - pm) + slip
    cost = np.clip(cost, 0.01, 0.99)
    gross = np.where(win, 100 * (1 - cost) / cost, -100.0)
    fee = (100 / cost) * 0.07 * pm * (1 - pm)
    return gross - fee, win, cost

def mc_pval(t, obs, nsim=20000):
    pm = t["p_market"].values
    yes = (t["side"] == "yes").values
    pwin = np.where(yes, pm, 1 - pm)
    cost = np.where(yes, pm, 1 - pm)
    fee = (100 / cost) * 0.07 * pm * (1 - pm)
    wamt = 100 * (1 - cost) / cost
    sims = np.empty(nsim)
    for i in range(nsim):
        w = rng.random(len(pm)) < pwin
        sims[i] = (np.where(w, wamt, -100.0) - fee).sum()
    return float((sims >= obs).mean())

def report(name, t):
    pnl, win, _ = pnl_vec(t)
    t = t.assign(pnl=pnl)
    print(f"\n### {name}  n={len(t)}  events={t['close_ts'].nunique()}  "
          f"WR={win.mean()*100:.1f}%  net={pnl.sum():+,.0f}  "
          f"avg/trade={pnl.mean():+.2f}")
    for wn, ws, we in WINDOWS:
        m = (t["dt"] >= ws) & (t["dt"] < we)
        sub = t[m]
        if not len(sub):
            continue
        p = mc_pval(sub, sub["pnl"].sum(), 5000)
        print(f"  {wn:8s} n={len(sub):5d} net={sub['pnl'].sum():+9,.0f}  mcP={p:.4f}")
    wk = t.set_index("dt")["pnl"].resample("W").sum()
    wk = wk[wk != 0]
    print(f"  weeks green: {(wk > 0).sum()}/{len(wk)}")
    ev = t.groupby("close_ts")["pnl"].sum()
    print(f"  worst expiry event: {ev.min():+,.0f}  best: {ev.max():+,.0f}")
    p_all = mc_pval(t, t["pnl"].sum(), 20000)
    print(f"  full-period mcP={p_all:.5f}")
    for slip in (0.005, 0.01):
        sp, _, _ = pnl_vec(t, slip)
        ts = t.assign(pnl=sp)
        ws_ = [ts[(ts['dt'] >= ws) & (ts['dt'] < we)]['pnl'].sum() for _, ws, we in WINDOWS]
        print(f"  slip {slip*100:.1f}c: net={sp.sum():+,.0f}  windows="
              + "/".join(f"{x:+,.0f}" for x in ws_))

print("=" * 70)
report("A. YES 0.97-0.995 unconditional (band extension)",
       make_trades("yes", 0.97, 0.995))
report("B. YES 0.70-0.80 & liq_bias <= 0",
       make_trades("yes", 0.70, 0.80, "liq_bias", "<=", 0.0))
report("C. NO 0.20-0.40 & vwap_stretch_score <= -1",
       make_trades("no", 0.20, 0.40, "vwap_stretch_score", "<=", -1.0))
report("D. NO 0.03-0.20 & vwap_stretch_score <= -1",
       make_trades("no", 0.03, 0.20, "vwap_stretch_score", "<=", -1.0))

print("\n--- threshold neighbors ---")
for v in (-1.0, -0.5, 0.0, 0.5):
    t = make_trades("yes", 0.70, 0.80, "liq_bias", "<=", v)
    pnl, _, _ = pnl_vec(t)
    print(f"B: liq_bias <= {v:+.1f}: n={len(t):5d} net={pnl.sum():+9,.0f}")
for lo in (0.65, 0.70, 0.75):
    t = make_trades("yes", lo, 0.80, "liq_bias", "<=", 0.0)
    pnl, _, _ = pnl_vec(t)
    print(f"B: floor {lo:.2f}: n={len(t):5d} net={pnl.sum():+9,.0f}")
for v in (-2.0, -1.0, 0.0):
    t = make_trades("no", 0.03, 0.40, "vwap_stretch_score", "<=", v)
    pnl, _, _ = pnl_vec(t)
    print(f"C+D: stretch <= {v:+.1f}: n={len(t):5d} net={pnl.sum():+9,.0f}")
