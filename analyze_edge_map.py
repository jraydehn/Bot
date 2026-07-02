#!/usr/bin/env python3
"""
analyze_edge_map.py — Comprehensive BTC market mispricing map.

For every signal × pm bucket, computes:
  edge = actual_WR - avg_pm   (YES side)
  edge = (1 - actual_WR) - (1 - avg_pm) = avg_pm - actual_WR  (NO side)

Three layers:
  1. Single-signal breakdown (all available signals × pm bucket)
  2. Two-signal combination pockets (best single signals crossed)
  3. Time/tau/momentum structural analysis
  4. NO-side specific analysis

Ranked summary at the end: biggest exploitable pockets sorted by
  t_stat = edge / SE  where SE = sqrt(WR*(1-WR)/N)

Uses ALL resolved trades (YES + NO actual trades + hypothetical).
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

SEP  = "=" * 76
SEP2 = "-" * 76
MIN_N   = 15   # minimum cell size to report
TOP_N   = 30   # top ranked pockets to show in summary

# ── Load all resolved trades ───────────────────────────────────────────────
df = pd.read_csv(
    "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades.csv",
    low_memory=False,
)
df["logged_at"] = pd.to_datetime(df["logged_at"])
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")

resolved = df[
    df["resolved_yes"].isin([0.0, 1.0]) &
    df["p_market"].notna() &
    df["would_pnl"].notna() &
    (df["decision"] == "trade")
].copy()

resolved["week"]    = resolved["logged_at"].dt.isocalendar().week.astype(int)
resolved["hour"]    = resolved["hour_utc"].fillna(resolved["logged_at"].dt.hour)
resolved["ema_s"]   = resolved["ema_stack_bias"].astype(str).str.strip()
resolved["struct"]  = resolved["structure_bias"].astype(str).str.strip()
resolved["vwap_s"]  = pd.to_numeric(resolved["vwap_score"],  errors="coerce")
resolved["obi_s"]   = pd.to_numeric(resolved["obi_score"],   errors="coerce")
resolved["vol_s"]   = pd.to_numeric(resolved["vol_score"],   errors="coerce")
resolved["cmf_s"]   = pd.to_numeric(resolved["cmf_score"],   errors="coerce")
resolved["fund_s"]  = pd.to_numeric(resolved["funding_bias"],errors="coerce")
resolved["smc4"]    = pd.to_numeric(resolved["smc_4h"],      errors="coerce")
resolved["adx"]     = pd.to_numeric(resolved["adx_1h"],      errors="coerce")
resolved["rvol"]    = pd.to_numeric(resolved["rvol_1h"],     errors="coerce")
resolved["c_trend"] = pd.to_numeric(resolved["composite_trend"], errors="coerce")
resolved["c_rev"]   = pd.to_numeric(resolved["composite_rev"],   errors="coerce")
resolved["c_pup"]   = pd.to_numeric(resolved["composite_p_up"],  errors="coerce")
resolved["stoch_k_v"]= pd.to_numeric(resolved["stoch_k"],    errors="coerce")
resolved["chg30"]   = pd.to_numeric(resolved["chg_30m"],     errors="coerce")
resolved["chg10"]   = pd.to_numeric(resolved["chg_10m"],     errors="coerce")
resolved["chg5"]    = pd.to_numeric(resolved["chg_5m"],      errors="coerce")
resolved["supply"]  = pd.to_numeric(resolved["in_supply_zone"], errors="coerce")
resolved["demand"]  = pd.to_numeric(resolved["in_demand_zone"], errors="coerce")
resolved["squeeze"] = pd.to_numeric(resolved["squeeze_1h"],  errors="coerce")
resolved["choch1"]  = pd.to_numeric(resolved["choch_1h"],    errors="coerce")
resolved["choch4"]  = pd.to_numeric(resolved["choch_4h"],    errors="coerce")
resolved["pm_drift"]= pd.to_numeric(resolved["pm_drift_5m"], errors="coerce")

yes = resolved[resolved["side"] == "yes"].copy()
no  = resolved[resolved["side"] == "no"].copy()

print(SEP)
print(f"DATASET: {len(resolved)} resolved trades  ({len(yes)} YES, {len(no)} NO)")
print(f"Date range: {resolved['logged_at'].min().date()} → {resolved['logged_at'].max().date()}")
print(SEP)

# ── Edge computation helpers ───────────────────────────────────────────────
def edge_stats(grp, side="yes"):
    """Return (n, WR, avg_pm, edge, SE, t_stat, pnl) for a group."""
    n = len(grp)
    if n < MIN_N:
        return None
    wr     = grp["resolved_yes"].mean()
    avg_pm = grp["p_market"].mean()
    pnl    = grp["would_pnl"].sum()
    if side == "yes":
        edge = wr - avg_pm
    else:
        edge = avg_pm - wr   # NO: market says 1-pm, actual NO WR = 1-wr
    se     = np.sqrt(wr * (1 - wr) / n)
    t_stat = edge / se if se > 0 else 0.0
    return dict(n=n, wr=wr, avg_pm=avg_pm, edge=edge, se=se, t_stat=t_stat, pnl=pnl)

PM_BINS  = [0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.01]
PM_LABELS= ["<.20",".20-.30",".30-.40",".40-.45",".45-.50",
            ".50-.55",".55-.60",".60-.70",".70-.80",">.80"]

def pm_bucket(pm_series):
    return pd.cut(pm_series, bins=PM_BINS, labels=PM_LABELS, right=False)

yes["pm_b"] = pm_bucket(yes["p_market"])
no["pm_b"]  = pm_bucket(no["p_market"])

# ── Global edge by pm bucket ───────────────────────────────────────────────
print()
print(SEP)
print("1. BASELINE EDGE BY pm BUCKET  (no signal conditioning)")
print(SEP)

for side, df_s, label in [("yes", yes, "YES"), ("no", no, "NO")]:
    print(f"\n  {label} bets:")
    print(f"  {'pm':>10}  {'n':>5}  {'WR':>6}  {'avg_pm':>7}  {'edge':>7}  {'t':>6}  {'P&L':>9}")
    print(f"  {SEP2[:65]}")
    for pm_b, g in df_s.groupby("pm_b", observed=True):
        r = edge_stats(g, side)
        if r is None: continue
        print(f"  {str(pm_b):>10}  {r['n']:>5}  {r['wr']:.3f}  {r['avg_pm']:.3f}  "
              f"{r['edge']:>+7.3f}  {r['t_stat']:>+6.2f}  ${r['pnl']:>9.2f}")

# ── Single signal analysis ─────────────────────────────────────────────────
all_pockets = []   # collect all cells for ranked summary

def run_signal(df_s, sig_col, sig_vals, label, side):
    """Breakdown by signal state × pm bucket, collect pockets."""
    rows = []
    for sv in sig_vals:
        sub = df_s[df_s[sig_col] == sv]
        if len(sub) < MIN_N: continue
        for pm_b, g in sub.groupby("pm_b", observed=True):
            r = edge_stats(g, side)
            if r is None: continue
            rows.append(dict(
                signal=f"{label}={sv}", pm=str(pm_b), side=side, **r
            ))
    return rows

def run_numeric_signal(df_s, sig_col, bins, bin_labels, label, side):
    """Numeric signal bucketed × pm bucket."""
    df_s = df_s.copy()
    df_s["_sb"] = pd.cut(df_s[sig_col], bins=bins, labels=bin_labels, right=False)
    rows = []
    for sv, sub_all in df_s.groupby("_sb", observed=True):
        if len(sub_all) < MIN_N: continue
        for pm_b, g in sub_all.groupby("pm_b", observed=True):
            r = edge_stats(g, side)
            if r is None: continue
            rows.append(dict(
                signal=f"{label}={sv}", pm=str(pm_b), side=side, **r
            ))
    return rows

def print_signal_section(title, rows, min_edge=0.04):
    if not rows: return
    rows_s = sorted(rows, key=lambda x: abs(x["edge"]), reverse=True)
    interesting = [r for r in rows_s if abs(r["edge"]) >= min_edge]
    if not interesting: return
    print(f"\n  {title}")
    print(f"  {'signal':>22}  {'pm':>10}  {'n':>5}  {'WR':>6}  {'edge':>7}  {'t':>6}  {'P&L':>9}")
    print(f"  {'-'*72}")
    for r in interesting[:20]:
        print(f"  {r['signal']:>22}  {r['pm']:>10}  {r['n']:>5}  "
              f"{r['wr']:.3f}  {r['edge']:>+7.3f}  {r['t_stat']:>+6.2f}  ${r['pnl']:>9.2f}")

print()
print(SEP)
print("2. SINGLE-SIGNAL EDGE BREAKDOWN  (sorted by |edge|, min n=15, |edge|≥0.04)")
print(SEP)

# ── YES side signals ───────────────────────────────────────────────────────
pockets_yes = []

# EMA stack
r = run_signal(yes, "ema_s", ["-1","0","1"], "ema_stack", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("EMA STACK (YES)", r)

# Structure bias
r = run_signal(yes, "struct", ["-1","0","1"], "structure", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("STRUCTURE BIAS (YES)", r)

# VWAP score
r = run_signal(yes, "vwap_s", [-1.0,0.0,1.0], "vwap_score", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("VWAP SCORE (YES)", r)

# OBI score
r = run_signal(yes, "obi_s", [-1.0,0.0,1.0], "obi_score", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("OBI SCORE (YES)", r)

# Vol score
r = run_signal(yes, "vol_s", [-1.0,0.0,1.0], "vol_score", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("VOL SCORE (YES)", r)

# CMF score
r = run_signal(yes, "cmf_s", [-1.0,0.0,1.0], "cmf_score", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("CMF SCORE (YES)", r)

# Funding bias
r = run_signal(yes, "fund_s", [-1.0,0.0,1.0], "funding", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("FUNDING BIAS (YES)", r)

# SMC 4h
r = run_signal(yes, "smc4", [-1.0,0.0,1.0], "smc_4h", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("SMC 4H (YES)", r)

# Composite trend buckets
r = run_numeric_signal(yes, "c_trend",
    [-6,-3,-1,0,1,3,7], ["≤-3","-3--1","-1-0","0-1","1-3","≥3"], "c_trend", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("COMPOSITE TREND (YES)", r)

# Composite p_up buckets
r = run_numeric_signal(yes, "c_pup",
    [0,.45,.50,.55,.60,.65,1.0], ["<.45",".45-.50",".50-.55",".55-.60",".60-.65","≥.65"],
    "p_up", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("COMPOSITE P_UP (YES)", r)

# Stoch K
r = run_numeric_signal(yes, "stoch_k_v",
    [0,20,40,60,80,100], ["<20","20-40","40-60","60-80","≥80"],
    "stoch_k", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("STOCH K (YES)", r)

# ADX
r = run_numeric_signal(yes, "adx",
    [0,20,30,50,200], ["<20","20-30","30-50","≥50"],
    "adx_1h", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("ADX 1H (YES)", r)

# RVOL
r = run_numeric_signal(yes, "rvol",
    [0,.8,1.2,2.0,10], ["<.80",".80-1.2","1.2-2.0","≥2.0"],
    "rvol_1h", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("RVOL 1H (YES)", r)

# Supply / demand zone
r = run_signal(yes, "supply", [0.0,1.0], "supply_zone", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("SUPPLY ZONE (YES)", r)

r = run_signal(yes, "demand", [0.0,1.0], "demand_zone", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("DEMAND ZONE (YES)", r)

# Squeeze
r = run_signal(yes, "squeeze", [0.0,1.0], "squeeze_1h", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("SQUEEZE 1H (YES)", r)

# ChoCh 1h
r = run_signal(yes, "choch1", [-1.0,0.0,1.0], "choch_1h", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("CHOCH 1H (YES)", r)

# pm_drift_5m (pm momentum)
r = run_numeric_signal(yes, "pm_drift",
    [-1,-.02,-.005,.005,.02,1], ["≤-.02","-.02--.005","-.005-.005",".005-.02","≥.02"],
    "pm_drift5m", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("PM DRIFT 5M (YES)", r)

# Tau buckets
r = run_numeric_signal(yes, "tau_minutes",
    [0,10,20,30,45,60,200], ["<10","10-20","20-30","30-45","45-60","≥60"],
    "tau_min", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("TAU MINUTES (YES)", r)

# Hour of day (UTC)
r = run_numeric_signal(yes, "hour",
    [0,4,8,12,16,20,24], ["0-4","4-8","8-12","12-16","16-20","20-24"],
    "hour_utc", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("HOUR UTC (YES)", r)

# 30m price change
r = run_numeric_signal(yes, "chg30",
    [-1,-.015,-.005,.005,.015,1], ["≤-1.5%","-1.5--.5%","-.5-.5%",".5-1.5%","≥1.5%"],
    "chg_30m", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("CHG 30M (YES)", r)

# 5m price change
r = run_numeric_signal(yes, "chg5",
    [-1,-.005,-.001,.001,.005,1], ["≤-.5%","-.5--.1%","-.1-.1%",".1-.5%","≥.5%"],
    "chg_5m", "yes")
pockets_yes += r; all_pockets += r
print_signal_section("CHG 5M (YES)", r)

# ── NO side signals ────────────────────────────────────────────────────────
print()
print(SEP)
print("3. NO SIDE — SINGLE SIGNAL BREAKDOWN  (edge = avg_pm − WR_yes)")
print(SEP)
pockets_no = []

for sig_col, sig_vals, label in [
    ("ema_s",  ["-1","0","1"],       "ema_stack"),
    ("struct", ["-1","0","1"],       "structure"),
    ("vwap_s", [-1.,0.,1.],          "vwap_score"),
    ("obi_s",  [-1.,0.,1.],          "obi_score"),
    ("fund_s", [-1.,0.,1.],          "funding"),
    ("smc4",   [-1.,0.,1.],          "smc_4h"),
]:
    r = run_signal(no, sig_col, sig_vals, label, "no")
    pockets_no += r; all_pockets += r
    print_signal_section(f"{label.upper()} (NO)", r)

for sig_col, bins, labels, label in [
    ("c_trend",  [-7,-3,-1,0,1,3,7], ["≤-3","-3--1","-1-0","0-1","1-3","≥3"], "c_trend"),
    ("c_pup",    [0,.45,.50,.55,.60,.65,1.], ["<.45",".45-.50",".50-.55",".55-.60",".60-.65","≥.65"], "p_up"),
    ("stoch_k_v",[0,20,40,60,80,100], ["<20","20-40","40-60","60-80","≥80"], "stoch_k"),
    ("adx",      [0,20,30,50,200],    ["<20","20-30","30-50","≥50"],         "adx_1h"),
    ("rvol",     [0,.8,1.2,2.0,10],   ["<.80",".80-1.2","1.2-2.0","≥2.0"],  "rvol_1h"),
    ("tau_minutes",[0,10,20,30,45,60,200],["<10","10-20","20-30","30-45","45-60","≥60"],"tau_min"),
    ("hour",     [0,4,8,12,16,20,24], ["0-4","4-8","8-12","12-16","16-20","20-24"], "hour_utc"),
]:
    r = run_numeric_signal(no, sig_col, bins, labels, label, "no")
    pockets_no += r; all_pockets += r
    print_signal_section(f"{label.upper()} (NO)", r)

# ── Two-signal combinations (top YES signals) ─────────────────────────────
print()
print(SEP)
print("4. TWO-SIGNAL COMBINATIONS  (min n=15, |edge|≥0.07)")
print(SEP)

combo_pockets = []

cross_pairs_yes = [
    ("ema_s",   ["-1","0","1"],  "ema"),
    ("vwap_s",  [-1.,0.,1.],     "vwap"),
    ("c_trend", None,            "c_trend"),  # use bins
    ("struct",  ["-1","0","1"],  "struct"),
    ("fund_s",  [-1.,0.,1.],     "fund"),
]

def get_signal_groups(df_s, sig_col, sig_vals):
    if sig_vals is not None:
        return [(sv, df_s[df_s[sig_col]==sv]) for sv in sig_vals]
    # numeric: bucket into low/mid/high
    vals = pd.to_numeric(df_s[sig_col], errors="coerce")
    p33, p67 = np.nanpercentile(vals, 33), np.nanpercentile(vals, 67)
    groups = [
        (f"low(≤{p33:.1f})",  df_s[vals <= p33]),
        (f"mid",              df_s[(vals > p33) & (vals < p67)]),
        (f"hi(≥{p67:.1f})",   df_s[vals >= p67]),
    ]
    return groups

print_combo_rows = []
for i, (sc1, sv1, l1) in enumerate(cross_pairs_yes):
    for sc2, sv2, l2 in cross_pairs_yes[i+1:]:
        for v1_label, g1 in get_signal_groups(yes, sc1, sv1):
            for v2_label, g2 in get_signal_groups(yes, sc2, sv2):
                combo = yes[
                    yes.index.isin(g1.index) & yes.index.isin(g2.index)
                ]
                if len(combo) < MIN_N: continue
                for pm_b, g in combo.groupby("pm_b", observed=True):
                    r = edge_stats(g, "yes")
                    if r is None or abs(r["edge"]) < 0.07: continue
                    sig_label = f"{l1}={v1_label} & {l2}={v2_label}"
                    combo_pockets.append(dict(signal=sig_label, pm=str(pm_b), side="yes", **r))
                    print_combo_rows.append((r["edge"], r["t_stat"], r["n"],
                                             sig_label, str(pm_b), r["wr"], r["avg_pm"], r["pnl"]))

print_combo_rows.sort(key=lambda x: abs(x[0]), reverse=True)
print(f"  {'signal combo':>42}  {'pm':>10}  {'n':>5}  {'WR':>6}  {'edge':>7}  {'t':>6}  {'P&L':>9}")
print(f"  {'-'*76}")
for edge, t, n, sig, pm, wr, avg_pm, pnl in print_combo_rows[:30]:
    print(f"  {sig:>42}  {pm:>10}  {n:>5}  {wr:.3f}  {edge:>+7.3f}  {t:>+6.2f}  ${pnl:>9.2f}")

# ── Weekly stability of top pockets ───────────────────────────────────────
print()
print(SEP)
print("5. WEEKLY STABILITY CHECK  (top single-signal YES pockets, edge>0.08)")
print(SEP)

top_yes_pockets = sorted(
    [p for p in pockets_yes if abs(p["edge"]) >= 0.08 and p["n"] >= 20],
    key=lambda x: abs(x["edge"]) * x["n"],
    reverse=True
)[:8]

for pocket in top_yes_pockets:
    sig_col_map = {
        "ema_stack": "ema_s", "structure": "struct",
        "vwap_score": "vwap_s", "obi_score": "obi_s",
        "vol_score": "vol_s", "funding": "fund_s",
        "smc_4h": "smc4",
    }
    sig_str = pocket["signal"]
    pm_str  = pocket["pm"]

    # parse signal name and value
    parts = sig_str.split("=", 1)
    if len(parts) != 2: continue
    sig_name, sig_val_str = parts[0].strip(), parts[1].strip()
    col = sig_col_map.get(sig_name)
    if col is None: continue

    lo_pm = PM_BINS[PM_LABELS.index(pm_str)] if pm_str in PM_LABELS else None
    if lo_pm is None: continue
    hi_pm = PM_BINS[PM_LABELS.index(pm_str)+1]

    sub = yes[
        (yes[col].astype(str).str.strip() == sig_val_str) &
        (yes["p_market"] >= lo_pm) & (yes["p_market"] < hi_pm)
    ]
    print(f"\n  {sig_str}  pm={pm_str}  (overall n={pocket['n']} edge={pocket['edge']:+.3f})")
    print(f"  {'week':>5}  {'n':>4}  {'WR':>5}  {'edge':>7}  {'P&L':>9}")
    for w, g in sub.groupby("week"):
        if len(g) < 3: continue
        wr = g["resolved_yes"].mean()
        pm_ = g["p_market"].mean()
        print(f"  {w:5d}  {len(g):4d}  {wr:.3f}  {wr-pm_:>+7.3f}  ${g['would_pnl'].sum():>9.2f}")

# ── MASTER RANKED SUMMARY ─────────────────────────────────────────────────
print()
print(SEP)
print(f"6. MASTER EDGE POCKET RANKING  (all signals, min n={MIN_N}, ranked by |edge|×√N)")
print(SEP)
print("   Only showing |edge| ≥ 0.06")
print()

all_p = [p for p in all_pockets if abs(p["edge"]) >= 0.06]
for p in all_p:
    p["score"] = abs(p["edge"]) * np.sqrt(p["n"])
    p["info"]  = f"{p['signal']:30s} pm={p['pm']:10s} [{p['side'].upper():3s}]"

all_p.sort(key=lambda x: x["score"], reverse=True)

print(f"  {'signal':30s} {'pm':>10}  {'side':>4}  {'n':>5}  {'WR':>6}  "
      f"{'edge':>7}  {'t':>6}  {'score':>6}  {'P&L':>9}")
print(f"  {'-'*88}")
for p in all_p[:TOP_N]:
    print(f"  {p['signal']:30s} {p['pm']:>10}  {p['side']:>4}  {p['n']:>5}  "
          f"{p['wr']:.3f}  {p['edge']:>+7.3f}  {p['t_stat']:>+6.2f}  "
          f"{p['score']:>6.1f}  ${p['pnl']:>9.2f}")

# ── Hypothetical: pm-level edge map (both sides, signal-free) ─────────────
print()
print(SEP)
print("7. HYPOTHETICAL EDGE MAP  (if we could bet either side freely)")
print("   Shows raw mispricing by pm bucket regardless of which side was taken")
print(SEP)

all_resolved = resolved[resolved["p_market"].notna()].copy()
all_resolved["pm_b"] = pm_bucket(all_resolved["p_market"])
all_resolved["week"] = all_resolved["logged_at"].dt.isocalendar().week.astype(int)

print(f"  {'pm':>10}  {'n':>5}  {'WR_yes':>7}  {'avg_pm':>7}  "
      f"{'edge_yes':>9}  {'edge_no':>8}  {'best_side':>10}  {'P&L_best':>10}")
print(f"  {'-'*80}")
for pm_b, g in all_resolved.groupby("pm_b", observed=True):
    n   = len(g)
    if n < MIN_N: continue
    wr  = g["resolved_yes"].mean()
    pm_ = g["p_market"].mean()
    e_y = wr - pm_
    e_n = pm_ - wr
    best_side = "YES" if e_y > e_n else "NO"
    best_edge = max(e_y, e_n)
    best_pnl  = g[g["side"]==best_side.lower()]["would_pnl"].sum() if len(g[g["side"]==best_side.lower()]) else 0
    print(f"  {str(pm_b):>10}  {n:>5}  {wr:>7.3f}  {pm_:>7.3f}  "
          f"{e_y:>+9.3f}  {e_n:>+8.3f}  {best_side:>10}  ${best_pnl:>10.2f}")

print()
print("Done.")
