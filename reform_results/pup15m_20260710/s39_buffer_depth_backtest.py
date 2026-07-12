"""
S39 -- proper backtest of a flat, deeper ITM buffer target for BTC 15m YES.
s38 found the model's current ~0.06-0.08% buffer sits in a weak-edge zone
across every vol regime, with 0.10-0.20%+ looking consistently better --
not a regime interaction, a flat structural effect. This tests it properly:

1. Ticker-clustered WR/BE/edge by offset-depth zone, full history + era
   split (pre/post 06-30 reform), using the FULL candidate population
   (unconditional on the model's own edge check -- a clean, unconfounded
   read of "is this buffer zone inherently safer/more profitable").
2. A $ simulation at FIXED stake size (so zones are compared on equal
   dollar risk, not just percentage-point edge -- deeper buffers cost more
   per contract at a given payout, changing the $ math).
3. Compare the best zone against the model's actual real-book performance
   as a benchmark.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
REFORM = pd.Timestamp("2026-06-30", tz="UTC")
FIXED_STAKE = 100.0  # $ per contract-equivalent bet, for apples-to-apples $ comparison

arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_market", "tau_minutes",
                           "offset_pct", "resolved_yes"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "offset_pct"])
yes_cand = arc[arc["offset_pct"] > 0].copy()
yes_cand["win"] = yes_cand["resolved_yes"]
yes_cand["cost"] = yes_cand["p_market"]
yes_cand["era"] = np.where(yes_cand["logged_at"] < REFORM, "pre", "post")
yes_cand["week"] = yes_cand["logged_at"].dt.to_period("W").astype(str)
print(f"YES-side candidates: {len(yes_cand)}  tickers: {yes_cand['contract_ticker'].nunique()}")

bins = [0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 100]
labels = ["0-.04", ".04-.06", ".06-.08", ".08-.10", ".10-.12", ".12-.15",
         ".15-.20", ".20-.25", ".25-.30", ".30-.40", ".40+"]
yes_cand["zone"] = pd.cut(yes_cand["offset_pct"], bins, labels=labels)

def tk_stats(sub):
    if len(sub) < 15:
        return None
    tk = sub.groupby("contract_ticker").agg(win=("win", "mean"), cost=("cost", "mean"), week=("week", "first"))
    boots_edge = [(tk["win"].sample(frac=1, replace=True, random_state=i).mean()
                  - tk["cost"].sample(frac=1, replace=True, random_state=i).mean()) for i in range(1500)]
    wk = (tk["win"] - tk["cost"]).groupby(tk["week"]).mean()
    return dict(n=len(sub), tickers=len(tk), wr=tk["win"].mean(), be=tk["cost"].mean(),
               edge=tk["win"].mean() - tk["cost"].mean(),
               p_le0=float(np.mean(np.array(boots_edge) <= 0)),
               wk_pos=int((wk > 0).sum()), wk_n=len(wk))

print(f"\n=== ticker-clustered WR/BE/edge by buffer zone (FULL history) ===")
print(f"{'zone':>9s} {'n':>5s} {'tk':>4s} {'WR':>7s} {'BE':>7s} {'edge':>8s} {'P(<=0)':>7s} {'wk+':>6s}")
zone_stats = {}
for z in labels:
    sub = yes_cand[yes_cand["zone"] == z]
    r = tk_stats(sub)
    zone_stats[z] = r
    if r:
        print(f"{z:>9s} {r['n']:5d} {r['tickers']:4d} {r['wr']:7.1%} {r['be']:7.1%} "
              f"{r['edge']:+8.4f} {r['p_le0']:7.3f} {r['wk_pos']:3d}/{r['wk_n']:<3d}")
    else:
        print(f"{z:>9s}   (thin)")

print(f"\n=== era split ===")
for era in ["pre", "post"]:
    print(f"  --- {era}-reform ---")
    for z in labels:
        sub = yes_cand[(yes_cand["zone"] == z) & (yes_cand["era"] == era)]
        r = tk_stats(sub)
        if r:
            print(f"  {z:>9s} n={r['n']:4d} tk={r['tickers']:3d} WR={r['wr']:.1%} BE={r['be']:.1%} "
                  f"edge={r['edge']:+.4f} P={r['p_le0']:.3f}")

print(f"\n=== $ simulation at fixed ${FIXED_STAKE:.0f} stake per contract, ticker-clustered ===")
print(f"{'zone':>9s} {'tickers':>7s} {'total_$':>10s} {'$/ticket':>9s} {'P($<=0)':>8s}")
for z in labels:
    sub = yes_cand[yes_cand["zone"] == z]
    if len(sub) < 15:
        continue
    tk = sub.groupby("contract_ticker").agg(win=("win", "mean"), cost=("cost", "mean"))
    n_contracts = FIXED_STAKE / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    total = pnl.sum()
    boots = [pnl[np.random.default_rng(i).integers(0, len(pnl), len(pnl))].sum() for i in range(1500)]
    p_le0 = float(np.mean(np.array(boots) <= 0))
    print(f"{z:>9s} {len(tk):7d} {total:10.2f} {total/len(tk):9.2f} {p_le0:8.3f}")

print("\n=== benchmark: real actual BTC 15m YES book (for comparison) ===")
df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes"])
print(f"  n={len(t)}  WR={t['resolved_yes'].mean():.1%}  total $ {t['would_pnl'].sum():+.2f}  "
      f"$/trade={t['would_pnl'].sum()/len(t):+.2f}")
print("DONE_S39")
