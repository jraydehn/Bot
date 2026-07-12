"""
S17 -- does sol_markov_gate (blocking SOL 15m YES/NO during 4h/6h/1h "adverse"
Markov-labeled regimes) actually still work? It's not a real Markov/HMM model --
just a 20-bar % change threshold: 6h regime = 20-bar 6h-resampled return vs
+/-3.0%, 4h = vs +/-2.5%, 1h = vs +/-1.5%. Reconstructed exactly (same formula,
using Binance SOL 1h close as a proxy for the live yfinance SOL-USD feed --
highly correlated, and the +/-2.5-3% thresholds are well above any plausible
cross-exchange discrepancy).

Joins onto the FULL candidate population (sol_scan_archive_15m.csv, not just
taken trades -- since the gate already filters live decisions, only the full
candidate archive shows what WOULD be blocked). Checks whether the specific
current blocker (4h=Sideways, both the YES hard-block and the NO
stoch_k_1h<86.1 block) still holds up on real recent outcomes, ticker-clustered.
"""
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

print("Reconstructing SOL 6h/4h/1h regime history from Binance 1h data...")
p1h = sorted(glob.glob("data/binanceus_SOLUSDT_1h_2024-01-01_*.parquet"))[-1]
c1h = pd.read_parquet(p1h)["close"].astype(float)
c1h.index = pd.to_datetime(c1h.index, utc=True)
c1h = c1h.sort_index()


def regime_series(close_1h, rule, window, thr):
    c = close_1h.resample(rule, origin="start_day").last().dropna()
    rr = c.pct_change(window)
    reg = pd.Series(np.where(rr > thr, "Bull", np.where(rr < -thr, "Bear", "Sideways")), index=c.index)
    reg[rr.isna()] = None
    return reg


reg_6h = regime_series(c1h, "6h", 20, 0.030)
reg_4h = regime_series(c1h, "4h", 20, 0.025)
reg_1h = regime_series(c1h, "1h", 20, 0.015)
print(f"  6h regime points: {len(reg_6h)}  4h: {len(reg_4h)}  1h: {len(reg_1h)}")
print(f"  4h regime value counts:\n{reg_4h.value_counts()}")

print("\nLoading SOL 15m FULL candidate archive...")
df = pd.read_csv("results/sol_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "p_market", "resolved_yes",
                            "offset_pct", "stoch_k_1h", "stoch_cross_1h"])
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["logged_at", "p_market", "resolved_yes", "offset_pct"])
df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
df["offset_pct"] = pd.to_numeric(df["offset_pct"], errors="coerce")
df["stoch_k_1h"] = pd.to_numeric(df["stoch_k_1h"], errors="coerce")
df["stoch_cross_1h"] = pd.to_numeric(df["stoch_cross_1h"], errors="coerce")
df = df.sort_values("logged_at")
print(f"  rows: {len(df)}  tickers: {df['contract_ticker'].nunique()}  "
      f"range: {df['logged_at'].min()} -> {df['logged_at'].max()}")

for name, reg in [("markov_6h", reg_6h), ("markov_4h", reg_4h), ("markov_1h", reg_1h)]:
    r = reg.reset_index()
    r.columns = ["ts", name]
    r = r.sort_values("ts")
    df = pd.merge_asof(df, r, left_on="logged_at", right_on="ts", direction="backward")
    df = df.drop(columns=["ts"])

df = df.dropna(subset=["markov_4h"])
print(f"\nrows with regime data: {len(df)}  tickers: {df['contract_ticker'].nunique()}")

# infer side from offset_pct sign: YES = offset>0 candidates, NO = offset<0 -- but the
# archive logs EVERY evaluated candidate at whatever offset it has, both are present.
# For a strike with a given offset, "YES" resolves = resolved_yes; "NO" resolves = 1-resolved_yes,
# cost = 1-p_market. Evaluate both interpretations on the same rows (matches the archive's
# per-contract, not per-side, structure).


def tk_stats(sub, side):
    if side == "yes":
        win = sub["resolved_yes"]; cost = sub["p_market"]
    else:
        win = 1 - sub["resolved_yes"]; cost = 1 - sub["p_market"]
    t = pd.DataFrame({"win": win, "cost": cost, "tk": sub["contract_ticker"]})
    # sane-cost filter: exclude near-certain contracts (cost<0.05 or >0.95) that blow up
    # n_contracts=100/cost under flat-stake sizing and dominate the $ total with noise
    t = t[t["cost"].between(0.05, 0.95)]
    tk = t.groupby("tk").agg(win=("win", "mean"), cost=("cost", "mean"))
    if len(tk) < 10:
        return None
    n_contracts = 100.0 / tk["cost"]
    pnl = np.where(tk["win"] >= 0.5, n_contracts * (1 - tk["cost"]), -n_contracts * tk["cost"])
    return dict(n=len(t), tk=len(tk), wr=tk["win"].mean(), be=tk["cost"].mean(),
                edge=tk["win"].mean() - tk["cost"].mean(), pnl=pnl.sum())


print(f"\n=== 4h=Sideways YES: hard-blocked population (validate the 'no profitable rescue' claim) ===")
side4h = df[df["markov_4h"] == "Sideways"]
r = tk_stats(side4h, "yes")
if r:
    print(f"  n={r['n']} tk={r['tk']} WR={r['wr']:.1%} BE={r['be']:.1%} edge={r['edge']:+.1%} pnl=${r['pnl']:.2f}")

print(f"\n=== 4h=Sideways NO: split by stoch_k_1h>=86.1 rescue threshold ===")
side4h_stoch = side4h.dropna(subset=["stoch_k_1h"])
blocked_no = side4h_stoch[side4h_stoch["stoch_k_1h"] < 86.1]
rescued_no = side4h_stoch[side4h_stoch["stoch_k_1h"] >= 86.1]
r_b = tk_stats(blocked_no, "no")
r_r = tk_stats(rescued_no, "no")
print(f"  BLOCKED (stoch_k_1h<86.1):  n={r_b['n'] if r_b else 0} tk={r_b['tk'] if r_b else 0} "
      f"WR={r_b['wr']:.1%} BE={r_b['be']:.1%} edge={r_b['edge']:+.1%} pnl=${r_b['pnl']:.2f}" if r_b else "  BLOCKED: thin")
print(f"  RESCUED (stoch_k_1h>=86.1): n={r_r['n'] if r_r else 0} tk={r_r['tk'] if r_r else 0} "
      f"WR={r_r['wr']:.1%} BE={r_r['be']:.1%} edge={r_r['edge']:+.1%} pnl=${r_r['pnl']:.2f}" if r_r else "  RESCUED: thin (n<10)")

print(f"\n=== comparison: NOT 4h=Sideways (regime OK), both sides ===")
not_side = df[df["markov_4h"] != "Sideways"]
for side in ["yes", "no"]:
    r = tk_stats(not_side, side)
    if r:
        print(f"  {side.upper()}: n={r['n']} tk={r['tk']} WR={r['wr']:.1%} BE={r['be']:.1%} edge={r['edge']:+.1%} pnl=${r['pnl']:.2f}")

print(f"\n=== recency check: 4h=Sideways YES population, split by era ===")
side4h = side4h.copy()
side4h["month"] = side4h["logged_at"].dt.to_period("M").astype(str)
for m, g in side4h.groupby("month"):
    r = tk_stats(g, "yes")
    if r:
        print(f"  {m}: n={r['n']:5d} tk={r['tk']:4d}  WR={r['wr']:.1%}  BE={r['be']:.1%}  edge={r['edge']:+.1%}  pnl=${r['pnl']:.2f}")

print(f"\n=== recency check: 4h=Sideways NO (stoch_k_1h<86.1, the currently-active block), by month ===")
blocked_no = blocked_no.copy()
blocked_no["month"] = blocked_no["logged_at"].dt.to_period("M").astype(str)
for m, g in blocked_no.groupby("month"):
    r = tk_stats(g, "no")
    if r:
        print(f"  {m}: n={r['n']:5d} tk={r['tk']:4d}  WR={r['wr']:.1%}  BE={r['be']:.1%}  edge={r['edge']:+.1%}  pnl=${r['pnl']:.2f}")

print(f"\n=== recency check: 4h=Sideways NO RESCUED (stoch_k_1h>=86.1), by month ===")
rescued_no = rescued_no.copy()
rescued_no["month"] = rescued_no["logged_at"].dt.to_period("M").astype(str)
for m, g in rescued_no.groupby("month"):
    r = tk_stats(g, "no")
    if r:
        print(f"  {m}: n={r['n']:5d} tk={r['tk']:4d}  WR={r['wr']:.1%}  BE={r['be']:.1%}  edge={r['edge']:+.1%}  pnl=${r['pnl']:.2f}")

print("\nDONE_S17")
