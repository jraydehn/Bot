"""Where does A's profit live, and does it survive real tick fills?

Slices YES 0.97-0.995 by pm sub-band. Fill models:
  mid   — archive pm as-is (what the fav book / prior numbers use)
  tick  — buy at ceil(pm to next cent) (realistic taker fill vs mid quote);
          fee charged on fill price; untradeable if fill >= 1.00
Windows as before; also per-window at tick fills.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ARCH = BASE / "results" / "eth_scan_archive.csv"
USE = ["logged_at", "contract_ticker", "close_ts", "p_market", "tau_minutes",
       "resolved_yes"]
df = pd.read_csv(ARCH, usecols=USE, low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(r"\+00:00$", "", regex=True),
                          errors="coerce", utc=True, format="mixed")
for c in ["p_market", "resolved_yes", "tau_minutes"]:
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

def trades(lo, hi):
    m = (df["p_market"] >= lo) & (df["p_market"] < hi) & df["outcome"].notna()
    return df[m].sort_values("dt").drop_duplicates("contract_ticker", keep="first").copy()

def pnl(t, fill):
    pm = t["p_market"].values
    if fill == "mid":
        cost = pm
    else:  # tick: next cent up (taker crossing from mid)
        cost = np.ceil(pm * 100 + 1e-9) / 100
        cost = np.where(np.isclose(cost, pm), pm + 0.01, cost)  # pm already on tick -> pay next tick
    ok = cost <= 0.99
    win = (t["outcome"].values == 1)
    gross = np.where(win, 100 * (1 - cost) / cost, -100.0)
    fee = (100 / cost) * 0.07 * cost * (1 - cost)
    out = np.where(ok, gross - fee, 0.0)
    return out, ok

print("sub-band  |  n  WR%  mid$  tick$  tick_tradeable")
for lo, hi in [(0.97, 0.975), (0.975, 0.98), (0.98, 0.985),
               (0.985, 0.99), (0.99, 0.995)]:
    t = trades(lo, hi)
    if t.empty:
        continue
    pm_, okm = pnl(t, "mid")
    pt_, okt = pnl(t, "tick")
    print(f"{lo:.3f}-{hi:.3f} | {len(t):5d} {100*(t['outcome']==1).mean():5.1f} "
          f"{pm_.sum():+9,.0f} {pt_.sum():+9,.0f}  {okt.sum():5d}")

print("\n--- trimmed band candidates at TICK fills, per window ---")
for lo, hi in [(0.97, 0.995), (0.97, 0.985), (0.97, 0.98)]:
    t = trades(lo, hi)
    pt_, okt = pnl(t, "tick")
    t = t.assign(p=pt_)
    line = f"YES {lo:.3f}-{hi:.3f}: n={okt.sum():5d} tick_net={pt_.sum():+9,.0f}  "
    wins = []
    for wn, ws, we in WINDOWS:
        m = (t["dt"] >= ws) & (t["dt"] < we)
        wins.append(f"{wn} {t.loc[m,'p'].sum():+8,.0f}")
    print(line + " | ".join(wins))

print("\n--- for scale: fav band 0.80-0.97 under same tick model ---")
t = trades(0.80, 0.97)
pm_, _ = pnl(t, "mid")
pt_, _ = pnl(t, "tick")
print(f"mid={pm_.sum():+,.0f}  tick={pt_.sum():+,.0f}  n={len(t)}")
