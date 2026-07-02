"""
BTC 15m Paper Trade Comprehensive Profitability Analysis
Run: python3 analyze_15m_btc.py
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "results/paper_trades_btc15m.csv"
MINS_PER_YEAR = 525600.0
BINANCE_URL = "https://api.binance.us/api/v3/klines"

# ── helpers ──────────────────────────────────────────────────────────────────

def pnl_stats(df_sub, label=""):
    """Return n, WR, P&L, breakeven WR for a subset of trade rows."""
    sub = df_sub[df_sub['decision'] == 'trade'].copy()
    sub = sub[sub['would_pnl'].notna()]
    if len(sub) == 0:
        return {"label": label, "n": 0, "wr": np.nan, "pnl": np.nan, "be_wr": np.nan}
    wins = (sub['would_win'] == 1).sum()
    wr = wins / len(sub)
    pnl = sub['would_pnl'].sum()
    # breakeven WR: avg bet on wins vs losses
    yes_rows = sub[sub['side'] == 'yes']
    no_rows  = sub[sub['side'] == 'no']
    # For YES: win pays p_market odds; lose costs bet
    # For NO: win pays (1-p_market) odds; lose costs bet
    # Approximate breakeven from actual data
    win_rows  = sub[sub['would_win'] == 1]
    loss_rows = sub[sub['would_win'] == 0]
    avg_win  = win_rows['would_pnl'].mean()  if len(win_rows)  else np.nan
    avg_loss = loss_rows['would_pnl'].mean() if len(loss_rows) else np.nan
    if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_win > 0 and avg_loss < 0:
        be_wr = -avg_loss / (avg_win - avg_loss)
    else:
        be_wr = np.nan
    return {"label": label, "n": len(sub), "wr": wr, "pnl": pnl, "be_wr": be_wr,
            "avg_win": avg_win, "avg_loss": avg_loss}

def fmt(d):
    n = d['n']
    wr = f"{d['wr']*100:.1f}%" if not np.isnan(d['wr']) else "N/A"
    pnl = f"${d['pnl']:+.2f}" if not np.isnan(d['pnl']) else "N/A"
    be = f"{d['be_wr']*100:.1f}%" if not np.isnan(d['be_wr']) else "N/A"
    return f"  n={n:4d}  WR={wr}  P&L={pnl}  BE_WR={be}"

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df['decision_time'] = pd.to_datetime(df['decision_time'], format='mixed', utc=True)
df['close_time_dt'] = pd.to_datetime(df['close_time'], format='mixed', utc=True)
df = df.sort_values('decision_time').reset_index(drop=True)

trades = df[df['decision'] == 'trade'].copy()
trades_pnl = trades[trades['would_pnl'].notna()].copy()

print("BTC 15m PAPER TRADE ANALYSIS")
print(f"Total rows: {len(df)} | Trades: {len(trades)} | With P&L: {len(trades_pnl)}")
print(f"Date range: {df['decision_time'].min().date()} to {df['decision_time'].max().date()}")
base = pnl_stats(df)
print(f"BASELINE: {fmt(base)}")
yes_base = pnl_stats(trades_pnl[trades_pnl['side']=='yes'])
no_base  = pnl_stats(trades_pnl[trades_pnl['side']=='no'])
print(f"  YES side:{fmt(yes_base)}")
print(f"  NO  side:{fmt(no_base)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
section("1. MODEL CALIBRATION (p_model_15m deciles)")

resolved = df[df['resolved_yes'].notna()].copy()
resolved['decile'] = pd.qcut(resolved['p_model_15m'], q=10, duplicates='drop')
cal = resolved.groupby('decile', observed=True).agg(
    n=('resolved_yes', 'count'),
    pred_mean=('p_model_15m', 'mean'),
    actual_wr=('resolved_yes', 'mean')
).reset_index()
cal['cal_error'] = cal['actual_wr'] - cal['pred_mean']
print(f"{'Decile':<28} {'n':>5} {'Pred':>6} {'ActualWR':>9} {'Error':>7}")
for _, row in cal.iterrows():
    print(f"  {str(row['decile']):<26} {row['n']:>5} {row['pred_mean']:>6.3f} {row['actual_wr']:>9.3f} {row['cal_error']:>+7.3f}")

print()
print("Trade-weighted calibration (trade rows only):")
tcal = trades_pnl.copy()
tcal['decile'] = pd.qcut(tcal['p_model_15m'], q=10, duplicates='drop')
tcal2 = tcal.groupby('decile', observed=True).agg(
    n=('resolved_yes', 'count'),
    pred_mean=('p_model_15m', 'mean'),
    actual_wr=('resolved_yes', 'mean'),
    pnl=('would_pnl', 'sum')
).reset_index()
tcal2['cal_error'] = tcal2['actual_wr'] - tcal2['pred_mean']
print(f"{'Decile':<28} {'n':>5} {'Pred':>6} {'ActualWR':>9} {'Error':>7} {'P&L':>8}")
for _, row in tcal2.iterrows():
    print(f"  {str(row['decile']):<26} {row['n']:>5} {row['pred_mean']:>6.3f} {row['actual_wr']:>9.3f} {row['cal_error']:>+7.3f} ${row['pnl']:>+7.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BREAKEVEN ANALYSIS BY p_market BUCKET
# ═══════════════════════════════════════════════════════════════════════════════
section("2. BREAKEVEN ANALYSIS BY p_market BUCKET")

pm_bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
pm_labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100']
trades_pnl['pm_bucket'] = pd.cut(trades_pnl['p_market'], bins=pm_bins, labels=pm_labels, right=False)

for side in ['yes', 'no', 'all']:
    print(f"\n--- {side.upper()} trades by p_market bucket ---")
    subset = trades_pnl if side == 'all' else trades_pnl[trades_pnl['side'] == side]
    print(f"{'pm_bucket':<12} {'n':>5} {'WR':>7} {'P&L':>9} {'BE_WR':>7} {'edge':>7}")
    for bkt in pm_labels:
        sub = subset[subset['pm_bucket'] == bkt]
        if len(sub) == 0:
            continue
        s = pnl_stats(sub)
        edge_str = f"{s['wr']-s['be_wr']:+.3f}" if not np.isnan(s['be_wr']) else "N/A"
        print(f"  {bkt:<10} {s['n']:>5} {s['wr']*100:>6.1f}% ${s['pnl']:>+8.2f} {s['be_wr']*100 if not np.isnan(s['be_wr']) else 0:>6.1f}% {edge_str:>7}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL CORRELATION WITH OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════════
section("3. SIGNAL CORRELATION WITH OUTCOMES")

signals = ['stoch_k_1h','chg_1h','bp_1h','consec_dir_1h','ema_bias',
           'stoch_k_5m','bp_5m','dir_15m','liq_score','vol_ratio','vwap_dist',
           'composite_p_up','dir_1h','stoch_k_15m','chg_15m','chg_5m']

print("\n--- Signal correlation with resolved_yes (all resolved rows) ---")
print(f"{'Signal':<22} {'n_nonull':>8} {'corr_w_yes':>11} {'trade_corr_pnl':>15}")
for sig in signals:
    sub = resolved[resolved[sig].notna()]
    n = len(sub)
    if n < 10:
        continue
    corr = sub[sig].corr(sub['resolved_yes'])
    # P&L correlation
    sub_t = trades_pnl[trades_pnl[sig].notna()]
    pnl_corr = sub_t[sig].corr(sub_t['would_pnl']) if len(sub_t) > 5 else np.nan
    print(f"  {sig:<22} {n:>8} {corr:>11.4f} {pnl_corr:>15.4f}")

print("\n--- P&L by signal quartile (trade rows only) ---")
for sig in signals:
    sub = trades_pnl[trades_pnl[sig].notna()].copy()
    if len(sub) < 20:
        continue
    try:
        sub['q'] = pd.qcut(sub[sig], q=4, duplicates='drop')
        grp = sub.groupby('q', observed=True).agg(
            n=('would_pnl', 'count'),
            wr=('would_win', 'mean'),
            pnl=('would_pnl', 'sum')
        ).reset_index()
        print(f"\n  {sig}:")
        for _, r in grp.iterrows():
            print(f"    {str(r['q']):<30} n={r['n']:>4}  WR={r['wr']*100:.1f}%  P&L=${r['pnl']:>+8.2f}")
    except Exception as e:
        print(f"  {sig}: skip ({e})")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. z_drift COMPUTATION (walk-forward, prior-only)
# ═══════════════════════════════════════════════════════════════════════════════
section("4. z_drift COMPUTATION AND GATE ANALYSIS")

print("Computing actual_z walk-forward from CSV data...")

# actual_z = log(floor_strike / spot) / (vol_eff * sqrt(tau_min))
# vol_eff = realized_vol_annual / sqrt(MINS_PER_YEAR)
# z_drift = 0.6 * mean(actual_z[-10:]) + 0.4 * mean(actual_z[-30:])

df2 = df.copy()
df2['vol_eff'] = df2['realized_vol_annual'] / np.sqrt(MINS_PER_YEAR)
df2['sigma_tau'] = df2['vol_eff'] * np.sqrt(df2['tau_minutes'])

# actual_z at RESOLUTION: log(floor_strike / expiry_price) / sigma_tau
# But we don't have expiry_price in CSV; we'll compute actual_z at ENTRY
# as log(floor_strike/spot) / sigma_tau (same sign convention as z_strike)
# This is the "standardized distance to strike" at entry time
df2['z_strike_entry'] = np.log(df2['floor_strike'] / df2['spot']) / df2['sigma_tau'].replace(0, np.nan)

# Walk-forward z_drift: for each row i, use rows 0..i-1
z_drift_vals = []
for i in range(len(df2)):
    prior = df2['z_strike_entry'].iloc[:i].dropna()
    if len(prior) < 5:
        z_drift_vals.append(np.nan)
    else:
        w10 = prior.iloc[-10:].mean() if len(prior) >= 10 else prior.mean()
        w30 = prior.iloc[-30:].mean() if len(prior) >= 30 else prior.mean()
        z_drift_vals.append(0.6 * w10 + 0.4 * w30)

df2['z_drift'] = z_drift_vals
trades2 = df2[df2['decision'] == 'trade'].copy()
trades2_pnl = trades2[trades2['would_pnl'].notna() & trades2['z_drift'].notna()].copy()

print(f"\nRows with z_drift computed: {trades2_pnl['z_drift'].notna().sum()}")
print(f"z_drift stats: mean={trades2_pnl['z_drift'].mean():.4f}  std={trades2_pnl['z_drift'].std():.4f}  "
      f"min={trades2_pnl['z_drift'].min():.4f}  max={trades2_pnl['z_drift'].max():.4f}")

print("\n--- 4a. YES trades: WR by z_drift direction ---")
yes2 = trades2_pnl[trades2_pnl['side'] == 'yes']
yes_pos = yes2[yes2['z_drift'] > 0]
yes_neg = yes2[yes2['z_drift'] <= 0]
s_pos = pnl_stats(yes_pos)
s_neg = pnl_stats(yes_neg)
print(f"  YES z_drift > 0 (tailwind): {fmt(s_pos)}")
print(f"  YES z_drift <=0 (headwind): {fmt(s_neg)}")

print("\n--- 4b. NO trades: WR by z_drift direction ---")
no2 = trades2_pnl[trades2_pnl['side'] == 'no']
no_pos = no2[no2['z_drift'] > 0]
no_neg = no2[no2['z_drift'] <= 0]
s_pos_no = pnl_stats(no_pos)
s_neg_no = pnl_stats(no_neg)
print(f"  NO  z_drift > 0 (bullish momentum): {fmt(s_pos_no)}")
print(f"  NO  z_drift <=0 (bearish momentum): {fmt(s_neg_no)}")

print("\n--- 4c. Hard gate: block YES when z_drift <= 0 ---")
yes_allowed = yes2[yes2['z_drift'] > 0]
yes_blocked = yes2[yes2['z_drift'] <= 0]
# Gate = keep only YES with z_drift>0, keep all NO
gated = pd.concat([yes_allowed, no2])
gated_stats = pnl_stats(gated)
baseline_yd = pnl_stats(trades2_pnl)
print(f"  Baseline (all trades w/ z_drift):  {fmt(baseline_yd)}")
print(f"  Gate YES z_drift>0 + all NO:       {fmt(gated_stats)}")
print(f"  Blocked: {len(yes_blocked)} YES trades  "
      f"WR={yes_blocked['would_win'].mean()*100:.1f}%  "
      f"P&L=${yes_blocked['would_pnl'].sum():+.2f}")
print(f"  P&L delta: ${gated_stats['pnl'] - baseline_yd['pnl']:+.2f}")

print("\n--- 4d. z_drift magnitude thresholds ---")
thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
for thr in thresholds:
    # Block YES when z_drift < -thr (only strong bearish drift)
    yes_allowed_t = yes2[(yes2['z_drift'] > -thr)]
    yes_blocked_t = yes2[(yes2['z_drift'] <= -thr)]
    gated_t = pd.concat([yes_allowed_t, no2])
    s = pnl_stats(gated_t)
    blocked_wr = yes_blocked_t['would_win'].mean() if len(yes_blocked_t) else np.nan
    blocked_pnl = yes_blocked_t['would_pnl'].sum() if len(yes_blocked_t) else 0
    delta = s['pnl'] - baseline_yd['pnl']
    print(f"  Block YES z_drift<-{thr:.2f}: blocked n={len(yes_blocked_t):3d} "
          f"WR={blocked_wr*100:.1f}% P&L_blocked=${blocked_pnl:+.2f} | "
          f"delta=${delta:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. z_drift AS STANDALONE MODEL + BLEND
# ═══════════════════════════════════════════════════════════════════════════════
section("5. z_drift STANDALONE MODEL AND BLEND WITH LGBM")

# p_zdrift = norm.cdf(-z_strike + z_drift)
# z_strike at entry = log(floor_strike/spot) / sigma_tau  (already in z_strike_entry)
# so p_zdrift = norm.cdf(-z_strike_entry + z_drift)  = P(BTC ends above strike)

trades2_pnl['p_zdrift'] = norm.cdf(-trades2_pnl['z_strike_entry'] + trades2_pnl['z_drift'])
print(f"p_zdrift stats: mean={trades2_pnl['p_zdrift'].mean():.3f}  "
      f"std={trades2_pnl['p_zdrift'].std():.3f}")

# Calibration of p_zdrift
print("\n--- p_zdrift calibration (resolved_yes) ---")
rz = df2[df2['resolved_yes'].notna() & df2['z_drift'].notna()].copy()
rz['sigma_tau'] = rz['vol_eff'] * np.sqrt(rz['tau_minutes'])
rz['z_strike_entry2'] = np.log(rz['floor_strike'] / rz['spot']) / rz['sigma_tau'].replace(0, np.nan)
rz['p_zdrift'] = norm.cdf(-rz['z_strike_entry2'] + rz['z_drift'])
rz['decile_zd'] = pd.qcut(rz['p_zdrift'], q=10, duplicates='drop')
cz = rz.groupby('decile_zd', observed=True).agg(
    n=('resolved_yes','count'),
    pred=('p_zdrift','mean'),
    actual=('resolved_yes','mean')
).reset_index()
cz['error'] = cz['actual'] - cz['pred']
print(f"{'Decile':<28} {'n':>5} {'Pred':>6} {'Actual':>8} {'Error':>7}")
for _, row in cz.iterrows():
    print(f"  {str(row['decile_zd']):<26} {row['n']:>5} {row['pred']:>6.3f} {row['actual']:>8.3f} {row['error']:>+7.3f}")

# Blend p_lgbm and p_zdrift, re-compute side/edge, simulate P&L
# For simplicity: use existing side/edge from CSV; test if blended p changes gate decisions
# More useful: compute blended edge and see if same trades are still above threshold
print("\n--- Blend: p_blend = (1-w)*p_lgbm + w*p_zdrift, re-evaluate edge ---")
print("    (Using same side as current trade; checking if blended edge > 0)")
blended_baseline = trades2_pnl['would_pnl'].sum()
print(f"  Baseline P&L (all trades w/ z_drift): ${blended_baseline:+.2f}  n={len(trades2_pnl)}")

for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    t = trades2_pnl.copy()
    t['p_blend'] = (1-w)*t['p_model_15m'] + w*t['p_zdrift']
    # Edge = p_blend - p_market for YES; (1-p_market) - (1-p_blend) = p_blend - p_market for NO too
    # Actually for NO: p_blend is P(YES), edge = (1-p_blend) - (1-p_market) = p_market - p_blend
    # Current decision already made; we test if edge still > 0 in blended model
    yes_mask = t['side'] == 'yes'
    no_mask  = t['side'] == 'no'
    t['blend_edge'] = 0.0
    t.loc[yes_mask, 'blend_edge'] = t.loc[yes_mask, 'p_blend'] - t.loc[yes_mask, 'p_market']
    t.loc[no_mask,  'blend_edge'] = t.loc[no_mask,  'p_market'] - t.loc[no_mask,  'p_blend']
    # Keep only trades where blended edge > 0
    keep = t[t['blend_edge'] > 0]
    dropped = t[t['blend_edge'] <= 0]
    s = pnl_stats(keep)
    dropped_pnl = dropped['would_pnl'].sum()
    delta = s['pnl'] - blended_baseline
    print(f"  w={w:.1f}: kept={s['n']:3d}  dropped={len(dropped):3d} (P&L dropped=${dropped_pnl:+.2f}) | "
          f"WR={s['wr']*100:.1f}%  P&L=${s['pnl']:+.2f}  delta=${delta:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. LOSS CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("6. LOSS CLUSTER ANALYSIS (top loss-generating conditions)")

t = trades_pnl.copy()

# Pre-compute common categoricals
t['pm_hi']     = t['p_market'] >= 0.60
t['pm_lo']     = t['p_market'] <= 0.40
t['stoch_hi']  = t['stoch_k_1h'] >= 70
t['stoch_lo']  = t['stoch_k_1h'] <= 30
t['ema_bull']  = t['ema_bias'] == 1
t['ema_bear']  = t['ema_bias'] == -1
t['chg_pos']   = t['chg_1h'] > 0
t['chg_neg']   = t['chg_1h'] < 0
t['vol_high']  = t['vol_ratio'] > 1.5
t['vol_low']   = t['vol_ratio'] < 0.7
t['vwap_pos']  = t['vwap_dist'] > 0
t['vwap_neg']  = t['vwap_dist'] < 0
t['liq_neg']   = t['liq_score'] < 0
t['stoch5_hi'] = t['stoch_k_5m'] >= 70
t['stoch5_lo'] = t['stoch_k_5m'] <= 30
t['bp5_neg']   = t['bp_5m'] < 0
t['bp5_pos']   = t['bp_5m'] > 0

print("Scanning signal combinations for loss clusters (YES losses only):")
yes_losses = t[(t['side']=='yes') & (t['would_win']==0)]
print(f"  Total YES losses: {len(yes_losses)}  P&L=${yes_losses['would_pnl'].sum():+.2f}")

conditions_yes = [
    ("YES + ema_bear",           t['side']=='yes', t['ema_bear']),
    ("YES + stoch_k_1h>=70",     t['side']=='yes', t['stoch_hi']),
    ("YES + vol_high>1.5",       t['side']=='yes', t['vol_high']),
    ("YES + vwap_neg",           t['side']=='yes', t['vwap_neg']),
    ("YES + chg_1h<0",           t['side']=='yes', t['chg_neg']),
    ("YES + pm>0.60",            t['side']=='yes', t['pm_hi']),
    ("YES + stoch_k_5m>=70",     t['side']=='yes', t['stoch5_hi']),
    ("YES + bp_5m<0",            t['side']=='yes', t['bp5_neg']),
    ("YES + ema_bear + stochhi", t['side']=='yes', t['ema_bear'] & t['stoch_hi']),
    ("YES + ema_bear + volhi",   t['side']=='yes', t['ema_bear'] & t['vol_high']),
    ("YES + ema_bear + chgneg",  t['side']=='yes', t['ema_bear'] & t['chg_neg']),
    ("YES + ema_bear + vwapneg", t['side']=='yes', t['ema_bear'] & t['vwap_neg']),
    ("YES + stochhi + volhi",    t['side']=='yes', t['stoch_hi'] & t['vol_high']),
    ("YES + chgneg + vwapneg",   t['side']=='yes', t['chg_neg'] & t['vwap_neg']),
    ("YES + pm_hi + ema_bear",   t['side']=='yes', t['pm_hi'] & t['ema_bear']),
    ("YES + liq_neg",            t['side']=='yes', t['liq_neg']),
    ("YES + liq_neg + ema_bear", t['side']=='yes', t['liq_neg'] & t['ema_bear']),
    ("YES + liq_neg + stoch_hi", t['side']=='yes', t['liq_neg'] & t['stoch_hi']),
]
print(f"\n{'Condition':<35} {'n_tot':>6} {'n_loss':>7} {'WR':>6} {'P&L_cond':>10} {'P&L_blocked':>12}")
results_yes = []
for label, side_mask, cond_mask in conditions_yes:
    cond = t[side_mask & cond_mask]
    if len(cond) == 0:
        continue
    s = pnl_stats(cond)
    losses = cond[cond['would_win']==0]
    # What if we blocked this condition?
    complement = t[side_mask & ~cond_mask]
    s_comp = pnl_stats(t[~(side_mask & cond_mask)])  # all except this condition
    blocked_pnl = cond['would_pnl'].sum()
    print(f"  {label:<35} {s['n']:>6} {len(losses):>7} {s['wr']*100:>5.1f}% ${s['pnl']:>+9.2f} ${-blocked_pnl:>+11.2f}")
    results_yes.append((label, s['n'], s['wr'], s['pnl'], blocked_pnl))

# Sort by worst P&L conditions
print("\n--- NO trade loss clusters ---")
no_losses = t[(t['side']=='no') & (t['would_win']==0)]
print(f"  Total NO losses: {len(no_losses)}  P&L=${no_losses['would_pnl'].sum():+.2f}")

conditions_no = [
    ("NO + ema_bull",            t['side']=='no',  t['ema_bull']),
    ("NO + stoch_k_1h<=30",      t['side']=='no',  t['stoch_lo']),
    ("NO + chg_1h>0",            t['side']=='no',  t['chg_pos']),
    ("NO + vwap_pos",            t['side']=='no',  t['vwap_pos']),
    ("NO + pm<0.40",             t['side']=='no',  t['pm_lo']),
    ("NO + stoch_k_5m<=30",      t['side']=='no',  t['stoch5_lo']),
    ("NO + bp_5m>0",             t['side']=='no',  t['bp5_pos']),
    ("NO + ema_bull + stochlo",  t['side']=='no',  t['ema_bull'] & t['stoch_lo']),
    ("NO + ema_bull + chgpos",   t['side']=='no',  t['ema_bull'] & t['chg_pos']),
    ("NO + ema_bull + vwappos",  t['side']=='no',  t['ema_bull'] & t['vwap_pos']),
    ("NO + liq_neg",             t['side']=='no',  t['liq_neg']),
    ("NO + vol_high",            t['side']=='no',  t['vol_high']),
    ("NO + vol_high + ema_bull", t['side']=='no',  t['vol_high'] & t['ema_bull']),
]
print(f"\n{'Condition':<35} {'n_tot':>6} {'n_loss':>7} {'WR':>6} {'P&L_cond':>10} {'P&L_blocked':>12}")
for label, side_mask, cond_mask in conditions_no:
    cond = t[side_mask & cond_mask]
    if len(cond) == 0:
        continue
    s = pnl_stats(cond)
    losses = cond[cond['would_win']==0]
    blocked_pnl = cond['would_pnl'].sum()
    print(f"  {label:<35} {s['n']:>6} {len(losses):>7} {s['wr']*100:>5.1f}% ${s['pnl']:>+9.2f} ${-blocked_pnl:>+11.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TIME OF DAY / TAU ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("7. TIME OF DAY / TAU ANALYSIS")

t2 = trades_pnl.copy()
t2['utc_hour'] = pd.to_datetime(t2['decision_time'], format='mixed', utc=True).dt.hour

print("\n--- P&L by UTC hour ---")
print(f"{'Hour':>6} {'n':>5} {'WR':>7} {'P&L':>9} {'BE_WR':>7}")
for h in sorted(t2['utc_hour'].unique()):
    sub = t2[t2['utc_hour'] == h]
    s = pnl_stats(sub)
    print(f"  {h:>4}  {s['n']:>5} {s['wr']*100:>6.1f}% ${s['pnl']:>+8.2f} "
          f"{s['be_wr']*100 if not np.isnan(s['be_wr']) else 0:>6.1f}%")

print("\n--- P&L by tau_minutes bucket ---")
tau_bins  = [0, 5, 10, 15, 20, 30, 60, 999]
tau_lbls  = ['0-5','5-10','10-15','15-20','20-30','30-60','60+']
t2['tau_bkt'] = pd.cut(t2['tau_minutes'], bins=tau_bins, labels=tau_lbls, right=False)
print(f"{'tau_bkt':<10} {'n':>5} {'WR':>7} {'P&L':>9} {'BE_WR':>7}")
for bkt in tau_lbls:
    sub = t2[t2['tau_bkt'] == bkt]
    if len(sub) == 0:
        continue
    s = pnl_stats(sub)
    print(f"  {bkt:<8} {s['n']:>5} {s['wr']*100:>6.1f}% ${s['pnl']:>+8.2f} "
          f"{s['be_wr']*100 if not np.isnan(s['be_wr']) else 0:>6.1f}%")

print("\n--- P&L by (side, tau_bkt) ---")
for side in ['yes','no']:
    print(f"\n  {side.upper()} side:")
    for bkt in tau_lbls:
        sub = t2[(t2['side']==side) & (t2['tau_bkt']==bkt)]
        if len(sub) < 3:
            continue
        s = pnl_stats(sub)
        print(f"    tau {bkt:<8} {s['n']:>5} WR={s['wr']*100:.1f}%  P&L=${s['pnl']:>+8.2f}  BE={s['be_wr']*100 if not np.isnan(s['be_wr']) else 0:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX: Raw edge distribution
# ═══════════════════════════════════════════════════════════════════════════════
section("APPENDIX: Edge and P_MODEL distribution for trade rows")

print("\n--- raw_edge distribution (trade rows) ---")
t3 = trades_pnl.copy()
edge_bins = [-0.5, -0.3, -0.1, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
edge_lbls = ['-0.5to-0.3','-0.3to-0.1','-0.1to0','0to0.05','0.05to0.1','0.1to0.15','0.15to0.2','0.2to0.3','0.3to0.5']
t3['edge_bkt'] = pd.cut(t3['raw_edge'], bins=edge_bins, labels=edge_lbls, right=False)
print(f"{'edge_bkt':<14} {'n':>5} {'WR':>7} {'P&L':>9}")
for bkt in edge_lbls:
    sub = t3[t3['edge_bkt'] == bkt]
    if len(sub) == 0:
        continue
    s = pnl_stats(sub)
    print(f"  {bkt:<14} {s['n']:>5} {s['wr']*100:>6.1f}% ${s['pnl']:>+8.2f}")

print("\n--- p_model_15m vs resolved_yes overall ---")
print(f"  All resolved: mean p_model={resolved['p_model_15m'].mean():.3f}  actual WR={resolved['resolved_yes'].mean():.3f}")
print(f"  Trade rows:   mean p_model={trades_pnl['p_model_15m'].mean():.3f}  actual WR={trades_pnl['resolved_yes'].mean():.3f}")
print(f"  YES trades:   mean p_model={trades_pnl[trades_pnl['side']=='yes']['p_model_15m'].mean():.3f}  "
      f"actual WR={trades_pnl[trades_pnl['side']=='yes']['resolved_yes'].mean():.3f}")
print(f"  NO  trades:   mean p_model={trades_pnl[trades_pnl['side']=='no']['p_model_15m'].mean():.3f}  "
      f"actual WR={trades_pnl[trades_pnl['side']=='no']['resolved_yes'].mean():.3f}  "
      f"(NO wins when resolved_yes=0, so NO WR={(1-trades_pnl[trades_pnl['side']=='no']['resolved_yes'].mean()):.3f})")

print("\n--- p_market analysis: where is the model agreeing with market? ---")
t3['pm_model_agree'] = ((t3['side']=='yes') & (t3['p_model_15m'] > 0.5)) | \
                        ((t3['side']=='no')  & (t3['p_model_15m'] < 0.5))
for agree in [True, False]:
    sub = t3[t3['pm_model_agree'] == agree]
    s = pnl_stats(sub)
    lbl = "Model direction matches p_market" if agree else "Model direction disagrees w/ p_market"
    print(f"  {lbl}: {fmt(s)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY: TOP ACTION ITEMS
# ═══════════════════════════════════════════════════════════════════════════════
section("SUMMARY TABLE: Candidate gate improvements")

print("""
Action                                  | Delta P&L  | Note
----------------------------------------|------------|-----
[See section 4c] z_drift YES gate       | see above  | block YES when z_drift<=0
[See section 4d] z_drift magnitude thr  | see above  | various thresholds
[See section 5]  z_drift blend weights  | see above  | blend 0.1-0.5
[See section 6]  YES loss clusters      | see above  | top ema/stoch conditions
[See section 7]  tau / time filtering   | see above  | time-of-day effects
""")
print("Analysis complete.")
