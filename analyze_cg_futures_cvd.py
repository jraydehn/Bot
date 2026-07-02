#!/usr/bin/env python3
"""
Deep analysis of CoinGlass futures CVD signals:
  - cg_futures_delta_4h
  - cg_futures_ratio_4h
  - cg_futures_cvd_12h
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results"

# ── helpers ───────────────────────────────────────────────────────────────────
def load_archive(fname, cols_needed=None):
    path = f"{RESULTS_DIR}/{fname}"
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    for c in ["resolved_yes", "p_market", "would_pnl",
              "cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # spot cvd if present
    for c in ["cvd_4h", "cvd_spot_4h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "close_ts" in df.columns:
        df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce", utc=True)
    elif "logged_at" in df.columns:
        df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    return df

def sep(title="", char="=", width=70):
    if title:
        print(f"\n{char*3} {title} {char*(width-len(title)-5)}")
    else:
        print(char * width)

def no_wr(df):
    if len(df) == 0: return np.nan
    return (df["resolved_yes"] == 0).mean()

def yes_wr(df):
    if len(df) == 0: return np.nan
    return (df["resolved_yes"] == 1).mean()

def pnl_sum(df):
    if "would_pnl" not in df.columns: return np.nan
    return df["would_pnl"].sum()

def quintile_analysis(df, col, label):
    """NO/YES WR by quintile of col."""
    sub = df.dropna(subset=[col, "resolved_yes"]).copy()
    if len(sub) < 50:
        print(f"  [SKIP] too few rows: {len(sub)}")
        return
    sub["q"] = pd.qcut(sub[col], 5, labels=["Q1(low)","Q2","Q3","Q4","Q5(high)"])
    base_no = no_wr(sub)
    base_yes = yes_wr(sub)
    print(f"\n  Baseline: NO_WR={base_no:.1%}  YES_WR={base_yes:.1%}  n={len(sub)}")
    print(f"  {'Quintile':<12} {'n':>6} {'avg_'+col[:15]:>18} {'avg_pm':>8} {'NO_WR':>8} {'Δ_NO':>8} {'YES_WR':>8}")
    for q in ["Q1(low)","Q2","Q3","Q4","Q5(high)"]:
        g = sub[sub["q"]==q]
        n = len(g)
        avg_sig = g[col].mean()
        avg_pm  = g["p_market"].mean() if "p_market" in g.columns else np.nan
        nwr = no_wr(g)
        ywr = yes_wr(g)
        delta = nwr - base_no
        print(f"  {q:<12} {n:>6} {avg_sig:>18.4f} {avg_pm:>8.3f} {nwr:>8.1%} {delta:>+8.1%} {ywr:>8.1%}")

def mcpt_test(df, mask, col="resolved_yes", n_perm=2000, seed=42):
    """Monte Carlo Permutation Test for NO_WR difference."""
    np.random.seed(seed)
    sub = df[mask].dropna(subset=[col])
    if len(sub) < 20:
        return np.nan, np.nan
    obs_nwr = (sub[col] == 0).mean()
    base_nwr = (df.dropna(subset=[col])[col] == 0).mean()
    obs_stat = obs_nwr - base_nwr
    all_y = df.dropna(subset=[col])[col].values
    perm_stats = []
    n = len(sub)
    for _ in range(n_perm):
        samp = np.random.choice(all_y, size=n, replace=False)
        perm_stats.append((samp == 0).mean() - base_nwr)
    perm_stats = np.array(perm_stats)
    # one-sided: obs_stat > 0 means NO_WR elevated above baseline
    if obs_stat >= 0:
        p = (perm_stats >= obs_stat).mean()
    else:
        p = (perm_stats <= obs_stat).mean()
    z = (obs_stat - perm_stats.mean()) / (perm_stats.std() + 1e-12)
    return p, z

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: BTC DATA CHECK
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 1: BTC DATA CHECK")
btc = load_archive("btc_scan_archive.csv")
print(f"Total rows: {len(btc):,}")
print(f"Columns with cg_futures: {[c for c in btc.columns if 'cg_futures' in c]}")

cg_cols = ["cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h"]
for c in cg_cols:
    if c in btc.columns:
        filled = btc[c].notna().sum()
        print(f"\n  {c}:")
        print(f"    Filled: {filled:,} / {len(btc):,} ({filled/len(btc):.1%})")

# Date range of filled data
if "cg_futures_ratio_4h" in btc.columns:
    sub_f = btc[btc["cg_futures_ratio_4h"].notna()]
    if "close_ts" in btc.columns:
        ts_col = "close_ts"
    elif "logged_at" in btc.columns:
        ts_col = "logged_at"
    else:
        ts_col = None
    if ts_col:
        sub_f[ts_col] = pd.to_datetime(sub_f[ts_col], errors="coerce", utc=True)
        print(f"\n  Date range (filled ratio_4h):")
        print(f"    First: {sub_f[ts_col].min()}")
        print(f"    Last:  {sub_f[ts_col].max()}")

# Distribution
print(f"\n  Percentile distribution:")
for c in cg_cols:
    if c not in btc.columns: continue
    s = btc[c].dropna()
    print(f"\n  {c} (n={len(s):,}):")
    print(f"    p5={s.quantile(0.05):.4f}  p25={s.quantile(0.25):.4f}  "
          f"p50={s.quantile(0.50):.4f}  p75={s.quantile(0.75):.4f}  p95={s.quantile(0.95):.4f}")
    print(f"    min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")

# Resolved rows
btc_res = btc[btc["resolved_yes"].notna()].copy()
print(f"\n  Resolved rows: {len(btc_res):,}")
for c in cg_cols:
    if c in btc.columns:
        n_both = btc_res[c].notna().sum()
        print(f"    {c} filled + resolved: {n_both:,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: SIGNAL CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 2: SIGNAL CORRELATION (BTC, resolved rows only)")
print(f"\n  Using {len(btc_res):,} resolved BTC rows")
print(f"  {'Signal':<30} {'Pearson r':>12}")
print(f"  {'-'*50}")

spot_cols = [c for c in ["cvd_4h","cvd_spot_4h"] if c in btc.columns]
for c in spot_cols + cg_cols:
    if c not in btc.columns: continue
    sub = btc_res.dropna(subset=[c,"resolved_yes"])
    if len(sub) < 20:
        print(f"  {c:<30} {'n too small':>12}")
        continue
    r = sub[c].corr(sub["resolved_yes"])
    print(f"  {c:<30} {r:>12.5f}  (n={len(sub):,})")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: NO WIN RATE BY RATIO QUINTILE
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 3: NO_WR BY RATIO_4H QUINTILE (BTC)")
quintile_analysis(btc_res, "cg_futures_ratio_4h", "BTC")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: YES WIN RATE BY RATIO QUINTILE
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 4: YES_WR BY RATIO_4H QUINTILE (BTC)")
# Already printed in quintile_analysis – just confirm monotonicity comment
sub = btc_res.dropna(subset=["cg_futures_ratio_4h","resolved_yes"]).copy()
sub["q"] = pd.qcut(sub["cg_futures_ratio_4h"], 5, labels=["Q1(low)","Q2","Q3","Q4","Q5(high)"])
q_yes = sub.groupby("q")["resolved_yes"].mean()
print(f"\n  YES_WR by quintile (for monotonicity check):")
for q,v in q_yes.items():
    print(f"    {q}: {v:.3f}")
q_no = 1 - q_yes
diffs = q_no.values[1:] - q_no.values[:-1]
print(f"  NO_WR monotonically decreasing Q1→Q5: {all(d<0 for d in diffs)}")
print(f"  NO_WR quintile diffs: {[f'{d:+.3f}' for d in diffs]}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: CVD × PM INTERACTION GRID
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 5: CVD × PM INTERACTION GRID (BTC)")

pm_bins = [0, 0.35, 0.50, 0.65, 0.80, 1.01]
pm_labels = ["<0.35","0.35-0.50","0.50-0.65","0.65-0.80",">0.80"]
ratio_bins = [0, 0.95, 0.99, 1.01, 1.05, 99]
ratio_labels = ["<0.95","0.95-0.99","0.99-1.01","1.01-1.05",">1.05"]

sub = btc_res.dropna(subset=["cg_futures_ratio_4h","resolved_yes","p_market"]).copy()
sub["pm_b"] = pd.cut(sub["p_market"], bins=pm_bins, labels=pm_labels, right=False)
sub["r_b"]  = pd.cut(sub["cg_futures_ratio_4h"], bins=ratio_bins, labels=ratio_labels, right=False)

print(f"\n  NO_WR grid (n in parens) — rows=pm bucket, cols=ratio bucket")
_hdr = "pm/ratio"
print(f"  {_hdr:<14}", end="")
for rl in ratio_labels:
    print(f"  {rl:>12}", end="")
print()
print("  " + "-"*80)
for pml in pm_labels:
    row_base = sub[sub["pm_b"]==pml]
    base_nwr = no_wr(row_base)
    print(f"  {pml:<14}  base={base_nwr:.1%}", end="")
    for rl in ratio_labels:
        cell = row_base[row_base["r_b"]==rl]
        n = len(cell)
        nwr = no_wr(cell)
        print(f"  {nwr:.0%}({n:4d})", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: CG GATE REPLICATION TEST
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 6: CG GATE REPLICATION TEST — pm[0.50,0.80) BTC")

btc_pm = btc_res[(btc_res["p_market"] >= 0.50) & (btc_res["p_market"] < 0.80)].copy()
btc_pm = btc_pm.dropna(subset=["resolved_yes"])
baseline_no = no_wr(btc_pm)
print(f"\n  pm[0.50,0.80) BTC resolved: n={len(btc_pm):,}  baseline_NO_WR={baseline_no:.1%}")
print(f"  Reference (dead CG stablecoin OI gate): baseline=44.9%, condition=33.1%, gap=-11.8pp\n")

conds = [
    ("ratio_4h < 1.00",  btc_pm["cg_futures_ratio_4h"] < 1.00),
    ("ratio_4h < 0.98",  btc_pm["cg_futures_ratio_4h"] < 0.98),
    ("ratio_4h < 0.95",  btc_pm["cg_futures_ratio_4h"] < 0.95),
    ("ratio_4h > 1.02",  btc_pm["cg_futures_ratio_4h"] > 1.02),
    ("ratio_4h > 1.05",  btc_pm["cg_futures_ratio_4h"] > 1.05),
    ("delta_4h < 0",     btc_pm["cg_futures_delta_4h"] < 0),
    ("delta_4h < -500M", btc_pm["cg_futures_delta_4h"] < -5e8),
    ("cvd_12h < 0",      btc_pm["cg_futures_cvd_12h"] < 0),
    ("cvd_12h < -1B",    btc_pm["cg_futures_cvd_12h"] < -1e9),
]

print(f"  {'Condition':<22} {'n':>6} {'NO_WR':>8} {'Δ_pp':>8} {'YES_WR':>8} {'PnL_sum':>12}")
print(f"  {'-'*72}")
for label, mask in conds:
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    n = len(sub)
    if n < 5:
        print(f"  {label:<22} {n:>6}  [insufficient data]")
        continue
    nwr = no_wr(sub)
    ywr = yes_wr(sub)
    delta = nwr - baseline_no
    pnl = pnl_sum(sub)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "    N/A"
    print(f"  {label:<22} {n:>6} {nwr:>8.1%} {delta:>+8.1%} {ywr:>8.1%} {pnl_str}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: WEEK-BY-WEEK STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 7: WEEK-BY-WEEK STABILITY — ratio_4h < 1.0, pm[0.50,0.80) BTC")

# find timestamp column
ts_col = None
for c in ["close_ts", "logged_at", "scan_ts"]:
    if c in btc.columns:
        ts_col = c
        break

if ts_col is None:
    print("  [ERROR] No timestamp column found")
else:
    btc_pm2 = btc_pm.copy()
    btc_pm2[ts_col] = pd.to_datetime(btc_pm2[ts_col], errors="coerce", utc=True)
    btc_pm2 = btc_pm2.dropna(subset=[ts_col])
    btc_pm2["isoweek"] = btc_pm2[ts_col].dt.isocalendar().week.astype(str) + \
                          "-" + btc_pm2[ts_col].dt.year.astype(str)

    mask = btc_pm2["cg_futures_ratio_4h"] < 1.0
    print(f"\n  {'Week':<12} {'n_cond':>8} {'n_base':>8} {'NO_WR_cond':>12} {'NO_WR_base':>12} {'Δ_pp':>8}")
    print(f"  {'-'*60}")
    weeks_pos = 0
    weeks_total = 0
    for wk in sorted(btc_pm2["isoweek"].dropna().unique()):
        wk_df = btc_pm2[btc_pm2["isoweek"]==wk]
        n_base = wk_df["resolved_yes"].notna().sum()
        sub_cond = wk_df[mask & wk_df["resolved_yes"].notna()]
        n_cond = len(sub_cond)
        if n_base < 5: continue
        base_nwr_wk = no_wr(wk_df.dropna(subset=["resolved_yes"]))
        cond_nwr = no_wr(sub_cond) if n_cond >= 3 else np.nan
        delta_wk = (cond_nwr - base_nwr_wk) if not np.isnan(cond_nwr) else np.nan
        flag = "✓" if (not np.isnan(delta_wk) and delta_wk > 0) else "✗"
        print(f"  {wk:<12} {n_cond:>8} {n_base:>8} {cond_nwr:>12.1%} {base_nwr_wk:>12.1%} "
              f"{delta_wk:>+8.1%}  {flag}")
        if not np.isnan(delta_wk):
            weeks_total += 1
            if delta_wk > 0: weeks_pos += 1
    print(f"\n  Weeks with elevated NO_WR: {weeks_pos}/{weeks_total}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: ETH AND SOL
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 8: ETH AND SOL — Ratio Quintile Analysis")

for asset, fname in [("ETH","eth_scan_archive.csv"),("SOL","sol_scan_archive.csv")]:
    sep(f"  {asset}", char="-")
    try:
        df = load_archive(fname)
        df_res = df[df["resolved_yes"].notna()].copy()
        print(f"  {asset}: total={len(df):,}  resolved={len(df_res):,}")
        has_sig = df_res["cg_futures_ratio_4h"].notna().sum() if "cg_futures_ratio_4h" in df_res.columns else 0
        print(f"  cg_futures_ratio_4h filled+resolved: {has_sig:,}")
        if has_sig < 50:
            print(f"  [SKIP] insufficient data")
            continue
        # correlation
        for c in cg_cols:
            if c not in df_res.columns: continue
            sub = df_res.dropna(subset=[c,"resolved_yes"])
            if len(sub) < 20: continue
            r = sub[c].corr(sub["resolved_yes"])
            print(f"  {c}: Pearson r = {r:.5f}  (n={len(sub):,})")
        # quintile
        quintile_analysis(df_res, "cg_futures_ratio_4h", asset)
    except Exception as e:
        print(f"  [ERROR] {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9: 15m TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 9: 15m TIMEFRAME — BTC, ETH, SOL paper trades")

for asset, fname in [("BTC","paper_trades_btc15m.csv"),
                     ("ETH","paper_trades_eth15m.csv"),
                     ("SOL","paper_trades_sol15m.csv")]:
    sep(f"  {asset} 15m", char="-")
    try:
        df15 = load_archive(fname)
        df15_res = df15[df15["resolved_yes"].notna()].copy()
        print(f"  {asset} 15m: total={len(df15):,}  resolved={len(df15_res):,}")
        has_sig = df15_res["cg_futures_ratio_4h"].notna().sum() if "cg_futures_ratio_4h" in df15_res.columns else 0
        print(f"  ratio_4h filled+resolved: {has_sig:,}")
        if has_sig < 30:
            print(f"  [SKIP] insufficient data ({has_sig})")
            continue
        quintile_analysis(df15_res, "cg_futures_ratio_4h", asset+" 15m")
    except Exception as e:
        print(f"  [ERROR] {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10: THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 10: THRESHOLD SEARCH — BTC pm[0.50,0.80)")

print(f"\n  Baseline n={len(btc_pm):,}  NO_WR={baseline_no:.1%}")
print(f"\n  --- ratio_4h thresholds ---")
print(f"  {'Threshold':<18} {'n':>6} {'NO_WR':>8} {'Δ_pp':>8} {'PnL_sum':>12}")
for thr in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
    if thr <= 1.00:
        mask = btc_pm["cg_futures_ratio_4h"] < thr
        label = f"ratio < {thr:.2f}"
    else:
        mask = btc_pm["cg_futures_ratio_4h"] > thr
        label = f"ratio > {thr:.2f}"
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    n = len(sub)
    if n < 5:
        print(f"  {label:<18} {n:>6}  [thin]")
        continue
    nwr = no_wr(sub)
    delta = nwr - baseline_no
    pnl = pnl_sum(sub)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "    N/A"
    print(f"  {label:<18} {n:>6} {nwr:>8.1%} {delta:>+8.1%} {pnl_str}")

print(f"\n  --- cvd_12h thresholds ---")
print(f"  {'Threshold':<20} {'n':>6} {'NO_WR':>8} {'Δ_pp':>8} {'PnL_sum':>12}")
for thr in [-2e9, -1e9, -5e8, 0, 5e8, 1e9]:
    if thr < 0:
        mask = btc_pm["cg_futures_cvd_12h"] < thr
        label = f"cvd_12h < {thr/1e9:.1f}B"
    else:
        mask = btc_pm["cg_futures_cvd_12h"] > thr
        label = f"cvd_12h > {thr/1e9:.1f}B"
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    n = len(sub)
    if n < 5:
        print(f"  {label:<20} {n:>6}  [thin]")
        continue
    nwr = no_wr(sub)
    delta = nwr - baseline_no
    pnl = pnl_sum(sub)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "    N/A"
    print(f"  {label:<20} {n:>6} {nwr:>8.1%} {delta:>+8.1%} {pnl_str}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11: MCPT SIGNIFICANCE TEST
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 11: MCPT SIGNIFICANCE TEST (2000 permutations, BTC pm[0.50,0.80))")

# Find best candidates from section 10
candidates = []
for thr in [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]:
    if thr <= 1.00:
        mask = btc_pm["cg_futures_ratio_4h"].notna() & (btc_pm["cg_futures_ratio_4h"] < thr)
        label = f"ratio < {thr:.2f}"
    else:
        mask = btc_pm["cg_futures_ratio_4h"].notna() & (btc_pm["cg_futures_ratio_4h"] > thr)
        label = f"ratio > {thr:.2f}"
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    if len(sub) >= 20:
        nwr = no_wr(sub)
        candidates.append((label, mask, nwr, nwr - baseline_no))

# Also cvd_12h
for thr in [-1e9, -5e8, 0]:
    if thr < 0:
        mask = btc_pm["cg_futures_cvd_12h"].notna() & (btc_pm["cg_futures_cvd_12h"] < thr)
        label = f"cvd_12h < {thr/1e9:.1f}B"
    else:
        mask = btc_pm["cg_futures_cvd_12h"].notna() & (btc_pm["cg_futures_cvd_12h"] > thr)
        label = f"cvd_12h > {thr/1e9:.1f}B"
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    if len(sub) >= 20:
        nwr = no_wr(sub)
        candidates.append((label, mask, nwr, nwr - baseline_no))

# Sort by abs(delta), test top 5
top_cands = sorted(candidates, key=lambda x: abs(x[3]), reverse=True)[:5]

print(f"\n  {'Candidate':<22} {'n':>5} {'NO_WR':>8} {'Δ_pp':>8} {'p_val':>8} {'z':>8}")
print(f"  {'-'*65}")
for label, mask, nwr, delta in top_cands:
    sub = btc_pm[mask].dropna(subset=["resolved_yes"])
    n = len(sub)
    p, z = mcpt_test(btc_pm.dropna(subset=["resolved_yes"]), mask[btc_pm.index], n_perm=2000)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    print(f"  {label:<22} {n:>5} {nwr:>8.1%} {delta:>+8.1%} {p:>8.4f} {z:>8.2f}  {sig}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 12: COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
sep("SECTION 12: FINAL COMPARISON TABLE")

print(f"\n  {'Signal':<35} {'Best_Δ_NO_WR':>14} {'MCPT_p':>10} {'Notes'}")
print(f"  {'-'*80}")

# Spot CVD
spot_col = next((c for c in ["cvd_4h","cvd_spot_4h"] if c in btc.columns), None)
if spot_col:
    sub_s = btc_pm.dropna(subset=[spot_col,"resolved_yes"])
    r_spot = sub_s[spot_col].corr(sub_s["resolved_yes"])
    # best threshold: < 0
    mask_s = btc_pm[spot_col].notna() & (btc_pm[spot_col] < 0)
    sub_s2 = btc_pm[mask_s].dropna(subset=["resolved_yes"])
    nwr_s = no_wr(sub_s2)
    delta_s = nwr_s - baseline_no
    p_s, z_s = mcpt_test(btc_pm.dropna(subset=["resolved_yes"]), mask_s[btc_pm.index], n_perm=1000)
    print(f"  {'Spot CVD ('+spot_col+')':<35} {delta_s:>+14.1%} {p_s:>10.4f}  r={r_spot:.4f}")
else:
    print(f"  {'Spot CVD':<35} {'N/A':>14} {'N/A':>10}  not in archive")

# Futures delta
mask_d = btc_pm["cg_futures_delta_4h"].notna() & (btc_pm["cg_futures_delta_4h"] < 0)
sub_d = btc_pm[mask_d].dropna(subset=["resolved_yes"])
nwr_d = no_wr(sub_d)
delta_d = nwr_d - baseline_no
r_d = btc_res.dropna(subset=["cg_futures_delta_4h","resolved_yes"])["cg_futures_delta_4h"].corr(
      btc_res.dropna(subset=["cg_futures_delta_4h","resolved_yes"])["resolved_yes"])
p_d, z_d = mcpt_test(btc_pm.dropna(subset=["resolved_yes"]), mask_d[btc_pm.index], n_perm=1000)
print(f"  {'Futures delta_4h < 0':<35} {delta_d:>+14.1%} {p_d:>10.4f}  r={r_d:.4f}")

# Futures ratio best
mask_r = btc_pm["cg_futures_ratio_4h"].notna() & (btc_pm["cg_futures_ratio_4h"] < 1.0)
sub_r = btc_pm[mask_r].dropna(subset=["resolved_yes"])
nwr_r = no_wr(sub_r)
delta_r = nwr_r - baseline_no
r_r = btc_res.dropna(subset=["cg_futures_ratio_4h","resolved_yes"])["cg_futures_ratio_4h"].corr(
      btc_res.dropna(subset=["cg_futures_ratio_4h","resolved_yes"])["resolved_yes"])
p_r, z_r = mcpt_test(btc_pm.dropna(subset=["resolved_yes"]), mask_r[btc_pm.index], n_perm=1000)
print(f"  {'Futures ratio_4h < 1.0':<35} {delta_r:>+14.1%} {p_r:>10.4f}  r={r_r:.4f}")

# CVD 12h
mask_c = btc_pm["cg_futures_cvd_12h"].notna() & (btc_pm["cg_futures_cvd_12h"] < 0)
sub_c = btc_pm[mask_c].dropna(subset=["resolved_yes"])
nwr_c = no_wr(sub_c)
delta_c = nwr_c - baseline_no
r_c = btc_res.dropna(subset=["cg_futures_cvd_12h","resolved_yes"])["cg_futures_cvd_12h"].corr(
      btc_res.dropna(subset=["cg_futures_cvd_12h","resolved_yes"])["resolved_yes"])
p_c, z_c = mcpt_test(btc_pm.dropna(subset=["resolved_yes"]), mask_c[btc_pm.index], n_perm=1000)
print(f"  {'Futures CVD_12h < 0':<35} {delta_c:>+14.1%} {p_c:>10.4f}  r={r_c:.4f}")

print(f"  {'Dead CG stablecoin OI gate (ref)':<35} {'-11.8pp':>14} {'0.0000':>10}  historical reference")

print(f"""
  INTERPRETATION:
    - Pearson r ~0 across all futures CVD variants → signal adds little to resolved_yes prediction
    - Best gate candidate from threshold sweep: see MCPT results above
    - If any p < 0.01 + 5+/7 weeks consistent → FLAG as gate candidate
    - Otherwise: REJECT / shadow-log only
""")

sep("END OF ANALYSIS")
