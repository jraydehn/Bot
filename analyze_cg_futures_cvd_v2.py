#!/usr/bin/env python3
"""
Deep analysis of CoinGlass futures CVD signals.
Data is in paper_trades files (scan archives don't have cg_futures columns).
Primary analysis file: paper_trades_btc15m.csv (4,770 resolved rows, ~1 month)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results"

CG_COLS = ["cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h"]

def load(fname):
    df = pd.read_csv(f"{RESULTS_DIR}/{fname}", low_memory=False)
    for c in CG_COLS + ["resolved_yes","p_market","would_pnl","side","decision"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for tc in ["logged_at","close_ts","decision_time"]:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], errors="coerce", utc=True)
    return df

def sep(title="", char="=", width=72):
    if title:
        print(f"\n{'='*3} {title} {'='*(width-len(title)-5)}")
    else:
        print("="*width)

def no_wr(df):
    s = df["resolved_yes"].dropna()
    return (s == 0).mean() if len(s) else np.nan

def yes_wr(df):
    s = df["resolved_yes"].dropna()
    return (s == 1).mean() if len(s) else np.nan

def pnl_sum(df):
    if "would_pnl" not in df.columns: return np.nan
    return df["would_pnl"].dropna().sum()

def quintile_analysis(sub, col, asset_label=""):
    sub = sub.dropna(subset=[col, "resolved_yes"]).copy()
    if len(sub) < 50:
        print(f"  [SKIP] too few rows: {len(sub)}")
        return
    sub["q"] = pd.qcut(sub[col], 5, labels=["Q1(low)","Q2","Q3","Q4","Q5(hi)"], duplicates="drop")
    base_no  = no_wr(sub)
    base_yes = yes_wr(sub)
    print(f"\n  {asset_label} | n={len(sub):,}  baseline NO_WR={base_no:.1%}  YES_WR={base_yes:.1%}")
    avg_pm_col = "p_market" if "p_market" in sub.columns else None
    print(f"  {'Quintile':<12} {'n':>6} {'avg_'+col[:14]:>18} {'avg_pm':>8} {'NO_WR':>8} {'Δ_NO':>8} {'YES_WR':>8}")
    q_no_wrs = []
    for q in ["Q1(low)","Q2","Q3","Q4","Q5(hi)"]:
        g = sub[sub["q"]==q]
        n = len(g)
        avg_sig = g[col].mean()
        avg_pm = g["p_market"].mean() if avg_pm_col else np.nan
        nwr = no_wr(g)
        ywr = yes_wr(g)
        delta = nwr - base_no
        print(f"  {q:<12} {n:>6} {avg_sig:>18.4f} {avg_pm:>8.3f} {nwr:>8.1%} {delta:>+8.1%} {ywr:>8.1%}")
        q_no_wrs.append(nwr)
    diffs = [q_no_wrs[i+1]-q_no_wrs[i] for i in range(4)]
    print(f"  Monotonic Q1→Q5 decrease (expected if signal works): {all(d<0 for d in diffs)}")
    print(f"  NO_WR diffs Q1→Q2→Q3→Q4→Q5: {[f'{d:+.3f}' for d in diffs]}")

def mcpt_test(full_sub, mask, n_perm=2000, seed=42):
    """MCPT for elevated/suppressed NO_WR within the filtered subset."""
    np.random.seed(seed)
    cond = full_sub[mask].dropna(subset=["resolved_yes"])
    base = full_sub.dropna(subset=["resolved_yes"])
    if len(cond) < 10 or len(base) < 20:
        return np.nan, np.nan, len(cond)
    obs_nwr   = (cond["resolved_yes"] == 0).mean()
    base_nwr  = (base["resolved_yes"] == 0).mean()
    obs_stat  = obs_nwr - base_nwr
    all_y = base["resolved_yes"].values
    n = len(cond)
    perm_stats = np.array([(np.random.choice(all_y, size=n, replace=False)==0).mean() - base_nwr
                           for _ in range(n_perm)])
    if obs_stat >= 0:
        p = (perm_stats >= obs_stat).mean()
    else:
        p = (perm_stats <= obs_stat).mean()
    z = (obs_stat - perm_stats.mean()) / (perm_stats.std() + 1e-12)
    return p, z, n

# ─────────────────────────────────────────────────────────────────────────────
sep("DATA INVENTORY")
# ─────────────────────────────────────────────────────────────────────────────
files = {
    "BTC_hourly": "paper_trades.csv",
    "ETH_hourly": "paper_trades_eth.csv",
    "SOL_hourly": "paper_trades_sol.csv",
    "BTC_15m":    "paper_trades_btc15m.csv",
    "ETH_15m":    "paper_trades_eth15m.csv",
    "SOL_15m":    "paper_trades_sol15m.csv",
}

datasets = {}
for name, fname in files.items():
    df = load(fname)
    sub = df[df["cg_futures_ratio_4h"].notna() & df["resolved_yes"].notna()]
    ts_col = next((c for c in ["logged_at","close_ts"] if c in df.columns), None)
    date_lo = df[ts_col].min().date() if ts_col else "?"
    date_hi = df[ts_col].max().date() if ts_col else "?"
    print(f"  {name:15s}: total={len(df):6,}  resolved+ratio={len(sub):6,}  "
          f"dates={date_lo}..{date_hi}")
    datasets[name] = sub

print(f"\n  NOTE: btc_scan_archive.csv (192k rows) does NOT contain cg_futures columns.")
print(f"  Analysis is based on paper_trades files (all scanned rows, both taken+rejected).")
print(f"  Primary file for BTC: paper_trades_btc15m.csv (4,770 resolved rows, ~1 month)")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 1: DATA CHECK — BTC 15m (primary dataset)")
# ─────────────────────────────────────────────────────────────────────────────
btc15 = datasets["BTC_15m"]
print(f"\n  n={len(btc15):,}")
print(f"  resolved_yes=1 (YES wins/price up): {btc15.resolved_yes.mean():.1%}")
print(f"  resolved_yes=0 (NO wins/price dn):  {(1-btc15.resolved_yes.mean()):.1%}")

print(f"\n  Signal distributions:")
for c in CG_COLS:
    if c not in btc15.columns: continue
    s = btc15[c].dropna()
    print(f"\n  {c} (n={len(s):,}):")
    print(f"    p5={s.quantile(0.05):,.0f}  p25={s.quantile(0.25):,.0f}  "
          f"p50={s.quantile(0.50):,.0f}  p75={s.quantile(0.75):,.0f}  p95={s.quantile(0.95):,.0f}")
    if c == "cg_futures_ratio_4h":
        print(f"    < 1.0 (net selling): {(s<1.0).sum():,} ({(s<1.0).mean():.1%})  "
              f"> 1.0 (net buying): {(s>1.0).sum():,} ({(s>1.0).mean():.1%})")

print(f"\n  p_market distribution:")
for lo, hi in [(0,0.35),(0.35,0.50),(0.50,0.65),(0.65,0.80),(0.80,1.01)]:
    m = (btc15["p_market"]>=lo) & (btc15["p_market"]<hi)
    sub_pm = btc15[m]
    print(f"    [{lo:.2f},{hi:.2f}): n={m.sum():4d}  NO_WR={no_wr(sub_pm):.1%}  YES_WR={yes_wr(sub_pm):.1%}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 2: SIGNAL CORRELATIONS")
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  {'Dataset':<18} {'Signal':<28} {'Pearson r':>12} {'n':>6}")
print(f"  {'-'*70}")
for ds_name, ds in datasets.items():
    for c in CG_COLS:
        if c not in ds.columns: continue
        sub = ds.dropna(subset=[c,"resolved_yes"])
        if len(sub) < 20:
            continue
        r = sub[c].corr(sub["resolved_yes"])
        print(f"  {ds_name:<18} {c:<28} {r:>12.5f} {len(sub):>6,}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 3+4: NO/YES WR BY RATIO QUINTILE")
# ─────────────────────────────────────────────────────────────────────────────

for ds_name, ds in datasets.items():
    sep(f"  {ds_name}", char="-")
    quintile_analysis(ds, "cg_futures_ratio_4h", ds_name)

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 5: CVD × PM INTERACTION GRID — BTC 15m")
# ─────────────────────────────────────────────────────────────────────────────
sub5 = btc15.dropna(subset=["cg_futures_ratio_4h","resolved_yes","p_market"]).copy()
pm_bins   = [0, 0.35, 0.50, 0.65, 0.80, 1.01]
pm_labels = ["<0.35","0.35-0.50","0.50-0.65","0.65-0.80",">0.80"]
r_bins    = [0, 0.95, 0.99, 1.01, 1.05, 99]
r_labels  = ["<0.95","0.95-0.99","0.99-1.01","1.01-1.05",">1.05"]

sub5["pm_b"] = pd.cut(sub5["p_market"], bins=pm_bins, labels=pm_labels, right=False)
sub5["r_b"]  = pd.cut(sub5["cg_futures_ratio_4h"], bins=r_bins, labels=r_labels, right=False)

print(f"\n  NO_WR grid  (n in parens) — rows=pm bucket, cols=ratio bucket")
hdr = "pm/ratio"
print(f"  {hdr:<14}", end="")
for rl in r_labels:
    print(f"  {rl:>12}", end="")
print()
print("  " + "-"*80)
for pml in pm_labels:
    row_data = sub5[sub5["pm_b"]==pml]
    base_nwr = no_wr(row_data)
    print(f"  {pml:<14}  base={base_nwr:.1%}", end="")
    for rl in r_labels:
        cell = row_data[row_data["r_b"]==rl]
        n = len(cell)
        nwr = no_wr(cell)
        nwr_str = f"{nwr:.0%}" if not np.isnan(nwr) else " N/A"
        print(f"  {nwr_str}({n:4d})", end="")
    print()

print(f"\n  YES_WR grid  (n in parens)")
hdr2 = "pm/ratio"
print(f"  {hdr2:<14}", end="")
for rl in r_labels:
    print(f"  {rl:>12}", end="")
print()
print("  " + "-"*80)
for pml in pm_labels:
    row_data = sub5[sub5["pm_b"]==pml]
    base_ywr = yes_wr(row_data)
    print(f"  {pml:<14}  base={base_ywr:.1%}", end="")
    for rl in r_labels:
        cell = row_data[row_data["r_b"]==rl]
        n = len(cell)
        ywr = yes_wr(cell)
        ywr_str = f"{ywr:.0%}" if not np.isnan(ywr) else " N/A"
        print(f"  {ywr_str}({n:4d})", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 6: CG GATE REPLICATION TEST — pm[0.50,0.80) BTC 15m")
# ─────────────────────────────────────────────────────────────────────────────
btc15_pm = btc15[(btc15["p_market"]>=0.50) & (btc15["p_market"]<0.80)].copy()
btc15_pm = btc15_pm.dropna(subset=["resolved_yes"])
baseline_no = no_wr(btc15_pm)
print(f"\n  pm[0.50,0.80) BTC 15m: n={len(btc15_pm):,}  baseline_NO_WR={baseline_no:.1%}")
print(f"  Reference (dead CG stablecoin OI gate): baseline=44.9%, cond=33.1%, gap=-11.8pp\n")

conds = [
    ("ratio_4h < 1.00",   btc15_pm["cg_futures_ratio_4h"] < 1.00),
    ("ratio_4h < 0.98",   btc15_pm["cg_futures_ratio_4h"] < 0.98),
    ("ratio_4h < 0.95",   btc15_pm["cg_futures_ratio_4h"] < 0.95),
    ("ratio_4h > 1.02",   btc15_pm["cg_futures_ratio_4h"] > 1.02),
    ("ratio_4h > 1.05",   btc15_pm["cg_futures_ratio_4h"] > 1.05),
    ("delta_4h < 0",      btc15_pm["cg_futures_delta_4h"] < 0),
    ("delta_4h < -500M",  btc15_pm["cg_futures_delta_4h"] < -5e8),
    ("cvd_12h < 0",       btc15_pm["cg_futures_cvd_12h"] < 0),
    ("cvd_12h < -1B",     btc15_pm["cg_futures_cvd_12h"] < -1e9),
]

print(f"  {'Condition':<22} {'n':>5} {'NO_WR':>7} {'Δ_pp':>7} {'YES_WR':>7} {'PnL_sum':>12}")
print(f"  {'-'*68}")
for label, mask in conds:
    sub = btc15_pm[mask].dropna(subset=["resolved_yes"])
    n = len(sub)
    if n < 5:
        print(f"  {label:<22} {n:>5}  [thin]")
        continue
    nwr  = no_wr(sub)
    ywr  = yes_wr(sub)
    delta = nwr - baseline_no
    pnl  = pnl_sum(sub)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "       N/A"
    print(f"  {label:<22} {n:>5} {nwr:>7.1%} {delta:>+7.1%} {ywr:>7.1%} {pnl_str}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 7: WEEK-BY-WEEK STABILITY — ratio_4h < 1.0, pm[0.50,0.80) BTC 15m")
# ─────────────────────────────────────────────────────────────────────────────
ts_col = "logged_at" if "logged_at" in btc15_pm.columns else "close_ts"
btc15_pm2 = btc15_pm.copy()
btc15_pm2["iso_week"] = btc15_pm2[ts_col].dt.isocalendar().week.astype(str) + \
                         "-" + btc15_pm2[ts_col].dt.year.astype(str)

mask_r1 = btc15_pm2["cg_futures_ratio_4h"] < 1.0
print(f"\n  {'Week':<12} {'n_cond':>8} {'n_base':>8} {'NO_WR_cond':>11} {'NO_WR_base':>11} {'Δ_pp':>8} {'sig'}")
print(f"  {'-'*65}")
weeks_pos = 0; weeks_total = 0
for wk in sorted(btc15_pm2["iso_week"].dropna().unique()):
    wk_df = btc15_pm2[btc15_pm2["iso_week"]==wk].dropna(subset=["resolved_yes"])
    n_base = len(wk_df)
    if n_base < 5: continue
    sub_cond = wk_df[mask_r1[btc15_pm2["iso_week"]==wk]]
    n_cond = len(sub_cond)
    base_nwr_wk = no_wr(wk_df)
    cond_nwr = no_wr(sub_cond) if n_cond >= 3 else np.nan
    delta_wk = (cond_nwr - base_nwr_wk) if not np.isnan(cond_nwr) else np.nan
    flag = "✓" if (not np.isnan(delta_wk) and delta_wk > 0) else ("✗" if not np.isnan(delta_wk) else "-")
    print(f"  {wk:<12} {n_cond:>8} {n_base:>8} {cond_nwr:>11.1%} {base_nwr_wk:>11.1%} "
          f"{delta_wk:>+8.1%}  {flag}")
    if not np.isnan(delta_wk):
        weeks_total += 1
        if delta_wk > 0: weeks_pos += 1
print(f"\n  Consistency: {weeks_pos}/{weeks_total} weeks elevated NO_WR")

# Also test ratio > 1.05 (strong buying → low NO_WR)
mask_r5 = btc15_pm2["cg_futures_ratio_4h"] > 1.05
print(f"\n  --- ratio > 1.05 (strong buying, expect LOW NO_WR) ---")
print(f"  {'Week':<12} {'n_cond':>8} {'n_base':>8} {'NO_WR_cond':>11} {'NO_WR_base':>11} {'Δ_pp':>8}")
for wk in sorted(btc15_pm2["iso_week"].dropna().unique()):
    wk_df = btc15_pm2[btc15_pm2["iso_week"]==wk].dropna(subset=["resolved_yes"])
    n_base = len(wk_df)
    if n_base < 5: continue
    sub_cond = wk_df[mask_r5[btc15_pm2["iso_week"]==wk]]
    n_cond = len(sub_cond)
    cond_nwr = no_wr(sub_cond) if n_cond >= 3 else np.nan
    delta_wk = (cond_nwr - no_wr(wk_df)) if not np.isnan(cond_nwr) else np.nan
    print(f"  {wk:<12} {n_cond:>8} {n_base:>8} {cond_nwr:>11.1%} {no_wr(wk_df):>11.1%} {delta_wk:>+8.1%}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 8: ETH and SOL — Core Analysis")
# ─────────────────────────────────────────────────────────────────────────────
for ds_name in ["ETH_hourly","SOL_hourly","ETH_15m","SOL_15m"]:
    sep(f"  {ds_name}", char="-")
    ds = datasets[ds_name]
    print(f"  n={len(ds):,}  YES_WR={yes_wr(ds):.1%}  NO_WR={no_wr(ds):.1%}")
    # correlation
    for c in CG_COLS:
        if c not in ds.columns: continue
        sub = ds.dropna(subset=[c,"resolved_yes"])
        if len(sub) < 20: continue
        r = sub[c].corr(sub["resolved_yes"])
        print(f"  {c}: r={r:.5f} (n={len(sub):,})")
    # quintile
    quintile_analysis(ds, "cg_futures_ratio_4h", ds_name)
    # pm[0.50,0.80) gate test
    pm_sub = ds[(ds["p_market"]>=0.50) & (ds["p_market"]<0.80)].dropna(subset=["resolved_yes"])
    if len(pm_sub) > 20:
        base = no_wr(pm_sub)
        print(f"\n  pm[0.50,0.80) n={len(pm_sub):,}  baseline_NO_WR={base:.1%}")
        for thr, op in [(1.00,"<"),(0.98,"<"),(1.02,">"),(1.05,">")]:
            if op == "<":
                m = pm_sub["cg_futures_ratio_4h"] < thr
            else:
                m = pm_sub["cg_futures_ratio_4h"] > thr
            s = pm_sub[m].dropna(subset=["resolved_yes"])
            if len(s) < 5: continue
            print(f"  ratio {op}{thr:.2f}: n={len(s):4d}  NO_WR={no_wr(s):.1%}  Δ={no_wr(s)-base:+.1%}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 9: 15m TIMEFRAME CONSISTENCY CHECK")
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [Already covered via BTC_15m as primary dataset — using ALL scanned rows]")
print(f"\n  Cross-15m summary at pm[0.50,0.80), ratio_4h < 1.00:")
print(f"  {'Asset':<12} {'n_cond':>8} {'NO_WR_cond':>12} {'baseline':>10} {'Δ_pp':>8}")
for ds_name in ["BTC_15m","ETH_15m","SOL_15m"]:
    ds = datasets[ds_name]
    pm_sub = ds[(ds["p_market"]>=0.50) & (ds["p_market"]<0.80)].dropna(subset=["resolved_yes"])
    if len(pm_sub) < 10: continue
    m = pm_sub["cg_futures_ratio_4h"] < 1.00
    s = pm_sub[m].dropna(subset=["resolved_yes"])
    b = no_wr(pm_sub)
    c = no_wr(s)
    print(f"  {ds_name:<12} {len(s):>8}  {c:>12.1%} {b:>10.1%} {c-b:>+8.1%}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 10: THRESHOLD SEARCH — BTC 15m pm[0.50,0.80)")
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Baseline n={len(btc15_pm):,}  NO_WR={baseline_no:.1%}")

print(f"\n  --- ratio_4h thresholds ---")
print(f"  {'Threshold':<18} {'n':>5} {'NO_WR':>7} {'Δ_pp':>8} {'PnL_sum':>12}")
ratio_results = []
for thr in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
    if thr <= 1.00:
        m = btc15_pm["cg_futures_ratio_4h"] < thr
        label = f"ratio < {thr:.2f}"
    else:
        m = btc15_pm["cg_futures_ratio_4h"] > thr
        label = f"ratio > {thr:.2f}"
    s = btc15_pm[m].dropna(subset=["resolved_yes"])
    n = len(s)
    if n < 5:
        print(f"  {label:<18} {n:>5}  [thin]")
        continue
    nwr = no_wr(s)
    delta = nwr - baseline_no
    pnl = pnl_sum(s)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "       N/A"
    print(f"  {label:<18} {n:>5} {nwr:>7.1%} {delta:>+8.1%} {pnl_str}")
    ratio_results.append((label, m, nwr, delta, n))

print(f"\n  --- cvd_12h thresholds ---")
print(f"  {'Threshold':<22} {'n':>5} {'NO_WR':>7} {'Δ_pp':>8} {'PnL_sum':>12}")
cvd_results = []
for thr in [-2e9, -1e9, -5e8, 0, 5e8, 1e9]:
    if thr < 0:
        m = btc15_pm["cg_futures_cvd_12h"] < thr
        label = f"cvd_12h < {thr/1e9:.1f}B"
    else:
        m = btc15_pm["cg_futures_cvd_12h"] > thr
        label = f"cvd_12h > {thr/1e9:.1f}B"
    s = btc15_pm[m].dropna(subset=["resolved_yes"])
    n = len(s)
    if n < 5:
        print(f"  {label:<22} {n:>5}  [thin]")
        continue
    nwr = no_wr(s)
    delta = nwr - baseline_no
    pnl = pnl_sum(s)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "       N/A"
    print(f"  {label:<22} {n:>5} {nwr:>7.1%} {delta:>+8.1%} {pnl_str}")
    cvd_results.append((label, m, nwr, delta, n))

print(f"\n  --- delta_4h thresholds ---")
print(f"  {'Threshold':<22} {'n':>5} {'NO_WR':>7} {'Δ_pp':>8} {'PnL_sum':>12}")
for thr in [-5e8, -2e8, -1e8, 0, 1e8, 2e8, 5e8]:
    if thr < 0:
        m = btc15_pm["cg_futures_delta_4h"] < thr
        label = f"delta_4h < {thr/1e8:.0f}*1e8"
    else:
        m = btc15_pm["cg_futures_delta_4h"] > thr
        label = f"delta_4h > {thr/1e8:.0f}*1e8"
    s = btc15_pm[m].dropna(subset=["resolved_yes"])
    n = len(s)
    if n < 5:
        print(f"  {label:<22} {n:>5}  [thin]")
        continue
    nwr = no_wr(s)
    delta = nwr - baseline_no
    pnl = pnl_sum(s)
    pnl_str = f"${pnl:>10,.0f}" if not np.isnan(pnl) else "       N/A"
    print(f"  {label:<22} {n:>5} {nwr:>7.1%} {delta:>+8.1%} {pnl_str}")

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 11: MCPT SIGNIFICANCE TEST (2000 permutations)")
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Testing top candidates from threshold sweep on BTC 15m pm[0.50,0.80)")
full_pm_sub = btc15_pm.dropna(subset=["resolved_yes"]).copy()
full_pm_sub = full_pm_sub.reset_index(drop=True)

# Build all candidates, sort by |delta|
all_cands = ratio_results + cvd_results
all_cands.sort(key=lambda x: abs(x[3]), reverse=True)
top = all_cands[:6]

print(f"\n  {'Candidate':<22} {'n':>5} {'NO_WR':>7} {'Δ_pp':>7} {'p_val':>8} {'z':>7}  {'sig'}")
print(f"  {'-'*68}")
best_cand = None
for label, m, nwr, delta, n in top:
    # Recompute mask on the reset index df
    if "ratio" in label:
        thr = float(label.split()[-1])
        op = label.split()[1]
        if op == "<":
            new_m = full_pm_sub["cg_futures_ratio_4h"] < thr
        else:
            new_m = full_pm_sub["cg_futures_ratio_4h"] > thr
    else:  # cvd
        thr = float(label.split()[-1].replace("B","")) * 1e9
        op = label.split()[1]
        if op == "<":
            new_m = full_pm_sub["cg_futures_cvd_12h"] < thr
        else:
            new_m = full_pm_sub["cg_futures_cvd_12h"] > thr

    p, z, n_test = mcpt_test(full_pm_sub, new_m, n_perm=2000)
    sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else "ns"))
    print(f"  {label:<22} {n_test:>5} {nwr:>7.1%} {delta:>+7.1%} {p:>8.4f} {z:>7.2f}  {sig}")
    if best_cand is None and p < 0.01:
        best_cand = (label, nwr, delta, p, z)

# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 12: FINAL COMPARISON TABLE")
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
  Signal comparisons at BTC pm[0.50,0.80), using best threshold each:

  {'Signal':<38} {'Best_Δ_NO_WR':>14} {'Notes'}
  {'-'*75}""")

# Check for spot CVD in btc15
spot_cvd = next((c for c in ["cvd_4h","cvd_spot_4h","spot_cvd"] if c in btc15.columns), None)
if spot_cvd:
    m_s = btc15_pm[spot_cvd] < 0
    s_s = btc15_pm[m_s].dropna(subset=["resolved_yes"])
    r_s = btc15.dropna(subset=[spot_cvd,"resolved_yes"])[spot_cvd].corr(
          btc15.dropna(subset=[spot_cvd,"resolved_yes"])["resolved_yes"])
    print(f"  {'Spot CVD < 0 ('+spot_cvd+')':<38} {no_wr(s_s)-baseline_no:>+14.1%}  r={r_s:.4f}")
else:
    print(f"  {'Spot CVD (not present in dataset)':<38} {'N/A':>14}  not in paper_trades_btc15m")

# delta_4h < 0
m_d = btc15_pm["cg_futures_delta_4h"] < 0
s_d = btc15_pm[m_d].dropna(subset=["resolved_yes"])
r_d = btc15.dropna(subset=["cg_futures_delta_4h","resolved_yes"])["cg_futures_delta_4h"].corr(
      btc15.dropna(subset=["cg_futures_delta_4h","resolved_yes"])["resolved_yes"])
print(f"  {'Futures delta_4h < 0':<38} {no_wr(s_d)-baseline_no:>+14.1%}  r={r_d:.4f}")

# ratio_4h < 1.0
m_r = btc15_pm["cg_futures_ratio_4h"] < 1.0
s_r = btc15_pm[m_r].dropna(subset=["resolved_yes"])
r_r = btc15.dropna(subset=["cg_futures_ratio_4h","resolved_yes"])["cg_futures_ratio_4h"].corr(
      btc15.dropna(subset=["cg_futures_ratio_4h","resolved_yes"])["resolved_yes"])
print(f"  {'Futures ratio_4h < 1.0':<38} {no_wr(s_r)-baseline_no:>+14.1%}  r={r_r:.4f}")

# cvd_12h < 0
m_c = btc15_pm["cg_futures_cvd_12h"] < 0
s_c = btc15_pm[m_c].dropna(subset=["resolved_yes"])
r_c = btc15.dropna(subset=["cg_futures_cvd_12h","resolved_yes"])["cg_futures_cvd_12h"].corr(
      btc15.dropna(subset=["cg_futures_cvd_12h","resolved_yes"])["resolved_yes"])
print(f"  {'Futures CVD_12h < 0':<38} {no_wr(s_c)-baseline_no:>+14.1%}  r={r_c:.4f}")

print(f"  {'Dead CG stablecoin OI gate (reference)':<38} {'-11.8pp':>14}  p=0.0000, historical")

# ─────────────────────────────────────────────────────────────────────────────
sep("VERDICT & RECOMMENDATION")
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
  PRIMARY ANALYSIS: BTC 15m (4,770 resolved rows, {btc15['logged_at'].min().date() if 'logged_at' in btc15.columns else '?'}..now)
  Secondary: ETH/SOL hourly (~500-800 resolved rows each)

  SIGNALS TESTED:
    cg_futures_delta_4h  (buy_usd - sell_usd, last 4h completed bar)
    cg_futures_ratio_4h  (buy/sell ratio, >1=buying)
    cg_futures_cvd_12h   (rolling 12h cumulative delta)

  KEY FINDINGS (see MCPT results above for significance):""")

if best_cand:
    lbl, nwr, delta, p, z = best_cand
    print(f"""
    BEST CANDIDATE: {lbl}
      NO_WR={nwr:.1%}  Δ={delta:>+.1%}  MCPT p={p:.4f}  z={z:.2f}
      → Compare to dead CG stablecoin gate: -11.8pp, p=0.000""")
else:
    print(f"""
    NO candidate passed MCPT p<0.01 threshold.""")

print(f"""
  RECOMMENDED ACTION:
    See MCPT section.  Criteria for gate candidacy:
      1. MCPT p < 0.01  (strong signal)
      2. 5+/7 weeks consistent direction (see Section 7)
      3. |Δ_pp| >= 5pp at pm[0.50,0.80)

    If criteria NOT met → REJECT as gate/boost.
    If criteria MET    → Shadow-log 2 more weeks before implementing.
    Note: 4,770 BTC 15m rows covers only ~1 month.  ETH/SOL hourly
    has similar data vol but even fewer resolved rows at pm[0.50,0.80).
    Any significant finding here is tentative — needs 12+ week history.
""")
sep("END")
