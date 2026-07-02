#!/usr/bin/env python3
"""
analyze_true_mispricing.py

Treats every resolved contract as an unresolved data point and asks:
    "Given the signal state at decision time, what was the TRUE probability
     of YES resolution, and how did that compare to pm?"

edge_yes = mean(resolved_yes) - avg_pm   (market underpriced YES)
edge_no  = avg_pm - mean(resolved_yes)   (market overpriced YES)

Does NOT look at which side we bet or whether we won/lost.
Includes ALL resolved trades (YES bets + NO bets).

Key additions over the first edge map:
  - "Wrong side" analysis: bets we placed on the lower-edge side
  - Flip value: if we had bet the other side instead, what was the P&L delta?
  - Full signal × pm mispricing, not filtered by bet direction
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from pricing_comparison import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE

KALSHI_FEE_RATE = 0.07
BANKROLL        = 1000.0
KELLY_MULT      = 0.30
KELLY_CAP       = 0.06
MIN_N           = 15
SEP  = "=" * 76
SEP2 = "-" * 76

# ── Load all resolved trades ───────────────────────────────────────────────
df = pd.read_csv(
    "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv",
    low_memory=False,
)
df["logged_at"]    = pd.to_datetime(df["logged_at"])
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")

all_trades = df[
    df["resolved_yes"].isin([0.0, 1.0]) &
    df["p_market"].notna() &
    df["would_pnl"].notna() &
    (df["decision"] == "trade")
].copy()

# Signal cleanup
all_trades["ema_s"]    = all_trades["ema_stack_bias"].astype(str).str.strip()
all_trades["struct"]   = all_trades["structure_bias"].astype(str).str.strip()
all_trades["vwap_s"]   = pd.to_numeric(all_trades["vwap_score"],   errors="coerce")
all_trades["obi_s"]    = pd.to_numeric(all_trades["obi_score"],    errors="coerce")
all_trades["fund_s"]   = pd.to_numeric(all_trades["funding_bias"], errors="coerce")
all_trades["vol_s"]    = pd.to_numeric(all_trades["vol_score"],    errors="coerce")
all_trades["c_trend"]  = pd.to_numeric(all_trades["composite_trend"], errors="coerce")
all_trades["c_pup"]    = pd.to_numeric(all_trades["composite_p_up"],  errors="coerce")
all_trades["stoch_k_v"]= pd.to_numeric(all_trades["stoch_k"],     errors="coerce")
all_trades["rvol"]     = pd.to_numeric(all_trades["rvol_1h"],      errors="coerce")
all_trades["adx"]      = pd.to_numeric(all_trades["adx_1h"],       errors="coerce")
all_trades["chg30"]    = pd.to_numeric(all_trades["chg_30m"],      errors="coerce")
all_trades["chg10"]    = pd.to_numeric(all_trades["chg_10m"],      errors="coerce")
all_trades["chg5"]     = pd.to_numeric(all_trades["chg_5m"],       errors="coerce")
all_trades["pm_drift"] = pd.to_numeric(all_trades["pm_drift_5m"],  errors="coerce")
all_trades["tau_v"]    = pd.to_numeric(all_trades["tau_minutes"],  errors="coerce")
all_trades["hour"]     = all_trades["hour_utc"].fillna(
                             all_trades["logged_at"].dt.hour)
all_trades["smc4"]     = pd.to_numeric(all_trades["smc_4h"],       errors="coerce")
all_trades["week"]     = all_trades["logged_at"].dt.isocalendar().week.astype(int)

# pm buckets
PM_BINS  = [0.0, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.01]
PM_LBLS  = ["<.20",".20-.30",".30-.35",".35-.40",".40-.45",
            ".45-.50",".50-.55",".55-.60",".60-.70",".70-.80",">.80"]
all_trades["pm_b"] = pd.cut(all_trades["p_market"], bins=PM_BINS,
                            labels=PM_LBLS, right=False)

print(SEP)
print(f"ALL RESOLVED TRADES (both sides pooled): {len(all_trades)}")
print(f"  YES bets: {(all_trades['side']=='yes').sum()}")
print(f"  NO  bets: {(all_trades['side']=='no').sum()}")
print(f"Date range: {all_trades['logged_at'].min().date()} → "
      f"{all_trades['logged_at'].max().date()}")
print(SEP)

# ── Core mispricing cell ───────────────────────────────────────────────────
def misprice(grp):
    n       = len(grp)
    if n < MIN_N: return None
    true_p  = grp["resolved_yes"].mean()
    avg_pm  = grp["p_market"].mean()
    e_yes   = true_p - avg_pm
    e_no    = avg_pm - true_p
    se      = np.sqrt(true_p * (1 - true_p) / n)
    t       = e_yes / se if se > 0 else 0.0
    # P&L of actual bets
    pnl_act = grp["would_pnl"].sum()
    # P&L if we had bet the "correct" side on every trade
    fee_est = KALSHI_FEE_RATE * avg_pm * (1 - avg_pm)  # approximate
    bet_est = BANKROLL * min(KELLY_MULT * max(0, abs(e_yes) - DEFAULT_SLIPPAGE), KELLY_CAP)
    n_yes   = (grp["side"] == "yes").sum()
    n_no    = (grp["side"] == "no").sum()
    best    = "YES" if e_yes > 0 else "NO"
    return dict(n=n, true_p=true_p, avg_pm=avg_pm,
                e_yes=e_yes, e_no=e_no, t=t, se=se,
                best=best, n_yes=n_yes, n_no=n_no, pnl=pnl_act)

# ── 1. Baseline: pm bucket, side-agnostic ─────────────────────────────────
print()
print(SEP)
print("1. TRUE MISPRICING BY pm BUCKET  (all trades pooled, side-agnostic)")
print(SEP)
print(f"  {'pm':>10}  {'n':>5}  {'true_p':>7}  {'avg_pm':>7}  "
      f"{'e_YES':>7}  {'e_NO':>7}  {'t':>6}  {'best':>5}")
print(SEP2)
for pm_b, g in all_trades.groupby("pm_b", observed=True):
    r = misprice(g)
    if r is None: continue
    print(f"  {str(pm_b):>10}  {r['n']:>5}  {r['true_p']:.3f}  {r['avg_pm']:.3f}  "
          f"{r['e_yes']:>+7.3f}  {r['e_no']:>+7.3f}  {r['t']:>+6.2f}  {r['best']:>5}")

# ── 2. "Wrong side" — where did we bet the lower-edge side? ───────────────
print()
print(SEP)
print("2. WRONG-SIDE ANALYSIS — contracts where we bet the mispriced direction")
print("   (comparing actual bet to true edge; sorted by $ left on table)")
print(SEP)

wrong_rows = []
for pm_b, g in all_trades.groupby("pm_b", observed=True):
    r = misprice(g)
    if r is None: continue
    if r["best"] == "YES":
        wrong = g[g["side"] == "no"]
    else:
        wrong = g[g["side"] == "yes"]
    if len(wrong) < 5: continue
    true_p  = r["true_p"]
    avg_pm  = g["p_market"].mean()
    lost_edge = abs(r["e_yes"]) * len(wrong)  # proxy for dollars left on table
    wrong_rows.append(dict(
        pm=str(pm_b), best=r["best"], wrong_n=len(wrong), total_n=r["n"],
        true_p=true_p, avg_pm=avg_pm, e_best=max(r["e_yes"], r["e_no"]),
        actual_pnl=wrong["would_pnl"].sum(), left_on_table=lost_edge
    ))

wrong_rows.sort(key=lambda x: x["left_on_table"], reverse=True)
print(f"  {'pm':>10}  {'best':>5}  {'wrong_n':>8}  {'true_p':>7}  "
      f"{'avg_pm':>7}  {'e_best':>7}  {'actual_pnl':>11}  {'$_left':>9}")
print(SEP2)
for r in wrong_rows:
    print(f"  {r['pm']:>10}  {r['best']:>5}  {r['wrong_n']:>8}  "
          f"{r['true_p']:.3f}  {r['avg_pm']:.3f}  {r['e_best']:>+7.3f}  "
          f"${r['actual_pnl']:>10.2f}  {r['left_on_table']:>9.1f}")

# ── 3. Full signal × pm mispricing (all trades, both sides pooled) ─────────
print()
print(SEP)
print("3. SIGNAL × pm MISPRICING  (all trades pooled — best edge side shown)")
print("   min n=15, |e_yes|≥0.07, sorted by |e_yes|×√N")
print(SEP)

all_pockets = []

def run_signal_both(df_all, sig_col, sig_vals, label):
    rows = []
    for sv in sig_vals:
        sub = df_all[df_all[sig_col] == sv]
        if len(sub) < MIN_N: continue
        for pm_b, g in sub.groupby("pm_b", observed=True):
            r = misprice(g)
            if r is None or abs(r["e_yes"]) < 0.07: continue
            rows.append(dict(
                signal=f"{label}={sv}", pm=str(pm_b),
                **r, score=abs(r["e_yes"]) * np.sqrt(r["n"])
            ))
    return rows

def run_numeric_both(df_all, sig_col, bins, lbls, label):
    df2 = df_all.copy()
    df2["_sb"] = pd.cut(df2[sig_col], bins=bins, labels=lbls, right=False)
    rows = []
    for sv, sub_all in df2.groupby("_sb", observed=True):
        if len(sub_all) < MIN_N: continue
        for pm_b, g in sub_all.groupby("pm_b", observed=True):
            r = misprice(g)
            if r is None or abs(r["e_yes"]) < 0.07: continue
            rows.append(dict(
                signal=f"{label}={sv}", pm=str(pm_b),
                **r, score=abs(r["e_yes"]) * np.sqrt(r["n"])
            ))
    return rows

signals = [
    run_signal_both(all_trades, "ema_s",     ["-1","0","1"],   "ema"),
    run_signal_both(all_trades, "struct",    ["-1","0","1"],   "struct"),
    run_signal_both(all_trades, "vwap_s",    [-1.,0.,1.],      "vwap"),
    run_signal_both(all_trades, "obi_s",     [-1.,0.,1.],      "obi"),
    run_signal_both(all_trades, "fund_s",    [-1.,0.,1.],      "fund"),
    run_signal_both(all_trades, "vol_s",     [-1.,0.,1.],      "vol"),
    run_signal_both(all_trades, "smc4",      [-1.,0.,1.],      "smc4"),
    run_numeric_both(all_trades, "c_trend",
        [-7,-3,-1,0,1,3,7], ["≤-3","-3--1","-1-0","0-1","1-3","≥3"], "c_trend"),
    run_numeric_both(all_trades, "stoch_k_v",
        [0,20,40,60,80,100], ["<20","20-40","40-60","60-80","≥80"], "stoch_k"),
    run_numeric_both(all_trades, "c_pup",
        [0,.45,.50,.55,.60,.65,1.], ["<.45",".45-.50",".50-.55",".55-.60",".60-.65","≥.65"], "p_up"),
    run_numeric_both(all_trades, "rvol",
        [0,.8,1.2,2.0,10], ["<.80",".80-1.2","1.2-2.0","≥2.0"], "rvol"),
    run_numeric_both(all_trades, "adx",
        [0,20,30,50,200], ["<20","20-30","30-50","≥50"], "adx"),
    run_numeric_both(all_trades, "chg30",
        [-1,-.015,-.005,.005,.015,1],
        ["≤-1.5%","-1.5--.5%","-.5-.5%",".5-1.5%","≥1.5%"], "chg30"),
    run_numeric_both(all_trades, "chg5",
        [-1,-.005,-.001,.001,.005,1],
        ["≤-.5%","-.5--.1%","-.1-.1%",".1-.5%","≥.5%"], "chg5"),
    run_numeric_both(all_trades, "tau_v",
        [0,10,20,30,45,60,200],
        ["<10","10-20","20-30","30-45","45-60","≥60"], "tau"),
    run_numeric_both(all_trades, "hour",
        [0,4,8,12,16,20,24],
        ["0-4","4-8","8-12","12-16","16-20","20-24"], "hour"),
    run_numeric_both(all_trades, "pm_drift",
        [-1,-.02,-.005,.005,.02,1],
        ["≤-2%","-2--.5%","-.5-.5%",".5-2%","≥2%"], "pm_drift"),
]
for s in signals:
    all_pockets.extend(s)

all_pockets.sort(key=lambda x: x["score"], reverse=True)

print(f"  {'signal':>22}  {'pm':>10}  {'n_yes':>6}  {'n_no':>5}  "
      f"{'true_p':>7}  {'avg_pm':>7}  {'e_YES':>7}  {'t':>6}  {'best':>5}  {'score':>6}")
print(SEP2)
seen = set()
for p in all_pockets:
    key = (p["signal"], p["pm"])
    if key in seen: continue
    seen.add(key)
    print(f"  {p['signal']:>22}  {p['pm']:>10}  {p['n_yes']:>6}  {p['n_no']:>5}  "
          f"{p['true_p']:.3f}  {p['avg_pm']:.3f}  {p['e_yes']:>+7.3f}  "
          f"{p['t']:>+6.2f}  {p['best']:>5}  {p['score']:>6.1f}")

# ── 4. Two-signal combos (both sides pooled) ──────────────────────────────
print()
print(SEP)
print("4. TWO-SIGNAL COMBOS  (both sides pooled, |e_yes|≥0.10, sorted by score)")
print(SEP)

combo_rows = []
pairs = [
    ("ema_s",  ["-1","0","1"],  "ema"),
    ("vwap_s", [-1.,0.,1.],     "vwap"),
    ("fund_s", [-1.,0.,1.],     "fund"),
    ("vol_s",  [-1.,0.,1.],     "vol"),
    ("struct", ["-1","0","1"],  "struct"),
]
# numeric binned
def add_bucket_col(df, col, bins, lbls, new_col):
    df = df.copy()
    df[new_col] = pd.cut(df[col], bins=bins, labels=lbls, right=False).astype(str)
    return df

all_trades["c_trend_b"] = pd.cut(all_trades["c_trend"],
    [-7,-3,-1,0,1,3,7], labels=["≤-3","-3--1","-1-0","0-1","1-3","≥3"],
    right=False).astype(str)
all_trades["stoch_b"] = pd.cut(all_trades["stoch_k_v"],
    [0,20,40,60,80,100], labels=["<20","20-40","40-60","60-80","≥80"],
    right=False).astype(str)
all_trades["rvol_b"] = pd.cut(all_trades["rvol"],
    [0,.8,1.2,2.0,10], labels=["<.80",".80-1.2","1.2-2.0","≥2.0"],
    right=False).astype(str)

pair_signals = [
    ("ema_s",      ["-1","0","1"],                              "ema"),
    ("vwap_s",     [-1.,0.,1.],                                 "vwap"),
    ("fund_s",     [-1.,0.,1.],                                 "fund"),
    ("c_trend_b",  ["≤-3","-3--1","-1-0","0-1","1-3","≥3"],    "c_trend"),
    ("stoch_b",    ["<20","20-40","40-60","60-80","≥80"],       "stoch_k"),
    ("rvol_b",     ["<.80",".80-1.2","1.2-2.0","≥2.0"],        "rvol"),
]

for i, (sc1, sv1, l1) in enumerate(pair_signals):
    for sc2, sv2, l2 in pair_signals[i+1:]:
        for v1 in sv1:
            g1 = all_trades[all_trades[sc1].astype(str) == str(v1)]
            for v2 in sv2:
                combo = g1[g1[sc2].astype(str) == str(v2)]
                if len(combo) < MIN_N: continue
                for pm_b, g in combo.groupby("pm_b", observed=True):
                    r = misprice(g)
                    if r is None or abs(r["e_yes"]) < 0.10: continue
                    lbl = f"{l1}={v1} & {l2}={v2}"
                    combo_rows.append(dict(
                        signal=lbl, pm=str(pm_b),
                        **r, score=abs(r["e_yes"]) * np.sqrt(r["n"])
                    ))

combo_rows.sort(key=lambda x: x["score"], reverse=True)
seen2 = set()
print(f"  {'signal combo':>36}  {'pm':>10}  {'n':>5}  "
      f"{'true_p':>7}  {'avg_pm':>7}  {'e_YES':>7}  {'t':>6}  {'best':>5}")
print(SEP2)
for r in combo_rows[:35]:
    key = (r["signal"], r["pm"])
    if key in seen2: continue
    seen2.add(key)
    print(f"  {r['signal']:>36}  {r['pm']:>10}  {r['n']:>5}  "
          f"{r['true_p']:.3f}  {r['avg_pm']:.3f}  {r['e_yes']:>+7.3f}  "
          f"{r['t']:>+6.2f}  {r['best']:>5}")

# ── 5. "Flipped" P&L: what if we bet the TRUE edge side on every trade ────
print()
print(SEP)
print("5. FLIP VALUE BY pm BUCKET — P&L if we always bet the true-edge side")
print("   (hindsight optimal, conditioned on having the signal)")
print(SEP)

KALSHI_FEE = KALSHI_FEE_RATE

def hypothetical_pnl(row, bet_side):
    pm  = float(row["p_market"])
    won_yes = int(row["resolved_yes"]) == 1
    bet = BANKROLL * min(
        KELLY_MULT * max(0, abs(float(row["p_market"]) - 0.5) - DEFAULT_SLIPPAGE),
        KELLY_CAP
    )
    if bet <= 0: return 0.0
    if bet_side == "yes":
        if won_yes: return bet * (1 - pm) / pm * (1 - KALSHI_FEE)
        else:       return -bet
    else:
        if not won_yes: return bet * pm / (1 - pm) * (1 - KALSHI_FEE)
        else:           return -bet

for pm_b, g in all_trades.groupby("pm_b", observed=True):
    r = misprice(g)
    if r is None: continue
    true_p  = r["true_p"]
    avg_pm  = r["avg_pm"]
    e_yes   = r["e_yes"]
    actual_pnl = g["would_pnl"].sum()

    # Hypothetical: always bet true-edge side at flat Kelly
    hyp_pnl = sum(hypothetical_pnl(row, "yes" if e_yes > 0 else "no")
                  for _, row in g.iterrows())
    delta = hyp_pnl - actual_pnl
    yes_n = (g["side"] == "yes").sum()
    no_n  = (g["side"] == "no").sum()
    print(f"  {str(pm_b):>10}  n={r['n']:4d}  true_p={true_p:.3f}  "
          f"pm={avg_pm:.3f}  e_YES={e_yes:>+7.3f}  best={r['best']:>3s}  "
          f"actual=${actual_pnl:>7.2f}  hyp=${hyp_pnl:>7.2f}  Δ=${delta:>+8.2f}  "
          f"(YES:{yes_n} NO:{no_n})")

print()
print("Done.")
