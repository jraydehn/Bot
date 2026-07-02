"""
Comprehensive Mispricing Analysis for Kalshi Binary Options
Analyzes BTC/ETH/SOL executed trades + blocked trades for exploitable market inefficiencies.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FILE = "results/mispricing_analysis.txt"
MIN_N = 20

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_trades(path, asset_label):
    df = pd.read_csv(path, low_memory=False)
    df['asset'] = asset_label
    resolved = df[df['resolved_yes'].notna()].copy()
    resolved['resolved_yes'] = pd.to_numeric(
        resolved['resolved_yes'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}),
        errors='coerce')
    resolved = resolved[resolved['resolved_yes'].notna()]
    # Derive hour_utc from decision_time or logged_at when missing
    resolved['dt'] = pd.to_datetime(resolved['decision_time'], errors='coerce')
    resolved.loc[resolved['dt'].isna(), 'dt'] = pd.to_datetime(
        resolved.loc[resolved['dt'].isna(), 'logged_at'], errors='coerce')
    resolved['hour_utc_derived'] = resolved['dt'].dt.hour
    resolved['dow'] = resolved['dt'].dt.dayofweek  # 0=Mon
    resolved['month_str'] = resolved['dt'].dt.to_period('M').astype(str)
    # Use derived hour if original is null
    resolved['hour_utc'] = resolved['hour_utc'].fillna(resolved['hour_utc_derived'])
    # Normalize ema_stack_bias to numeric
    resolved['ema_stack_bias'] = pd.to_numeric(resolved['ema_stack_bias'], errors='coerce')
    resolved['stoch_k'] = pd.to_numeric(resolved['stoch_k'], errors='coerce')
    resolved['p_market'] = pd.to_numeric(resolved['p_market'], errors='coerce')
    resolved['offset_pct'] = pd.to_numeric(resolved['offset_pct'], errors='coerce')
    resolved['tau_minutes'] = pd.to_numeric(resolved['tau_minutes'], errors='coerce')
    resolved['vol_eff'] = pd.to_numeric(resolved['vol_eff'], errors='coerce')
    resolved['rvol_1h'] = pd.to_numeric(resolved['rvol_1h'], errors='coerce')
    resolved['funding_bias'] = pd.to_numeric(resolved['funding_bias'], errors='coerce')
    resolved['obi_score'] = pd.to_numeric(resolved['obi_score'], errors='coerce')
    resolved['vpin_score'] = pd.to_numeric(resolved['vpin_score'], errors='coerce')
    resolved['composite_trend'] = pd.to_numeric(resolved['composite_trend'], errors='coerce')
    resolved['composite_rev'] = pd.to_numeric(resolved['composite_rev'], errors='coerce')
    resolved['composite_p_up'] = pd.to_numeric(resolved['composite_p_up'], errors='coerce')
    resolved['chg_5m'] = pd.to_numeric(resolved['chg_5m'], errors='coerce')
    resolved['chg_30m'] = pd.to_numeric(resolved['chg_30m'], errors='coerce')
    resolved['vol_score'] = pd.to_numeric(resolved['vol_score'], errors='coerce')
    resolved['vwap_stretch_score'] = pd.to_numeric(resolved['vwap_stretch_score'], errors='coerce')
    resolved['vwap_distance_pct'] = pd.to_numeric(resolved['vwap_distance_pct'], errors='coerce')
    resolved['adx_1h'] = pd.to_numeric(resolved['adx_1h'], errors='coerce')
    resolved['p_yes_model'] = pd.to_numeric(resolved['p_yes_model'], errors='coerce')
    return resolved

print("Loading data...")
btc = load_trades("results/paper_trades.csv", "BTC")
eth = load_trades("results/paper_trades_eth.csv", "ETH")
sol = load_trades("results/paper_trades_sol.csv", "SOL")
all_trades = pd.concat([btc, eth, sol], ignore_index=True)

print(f"BTC resolved: {len(btc)}, ETH: {len(eth)}, SOL: {len(sol)}, Total: {len(all_trades)}")

# Load blocked trades
print("Loading blocked trades...")
blocked = pd.read_csv("results/blocked_trades.csv", low_memory=False)
blocked['resolved_yes'] = pd.to_numeric(blocked['resolved_yes'], errors='coerce')
blocked = blocked[blocked['resolved_yes'].notna()]
blocked['pm'] = pd.to_numeric(blocked['pm'], errors='coerce')
blocked['offset_pct'] = pd.to_numeric(blocked['offset_pct'], errors='coerce')
blocked['tau_minutes'] = pd.to_numeric(blocked['tau_minutes'], errors='coerce')
blocked['stoch_k'] = pd.to_numeric(blocked['stoch_k'], errors='coerce')
blocked['ema_stack_bias'] = pd.to_numeric(blocked['ema_stack_bias'], errors='coerce')
blocked['composite_trend'] = pd.to_numeric(blocked['composite_trend'], errors='coerce')
blocked['composite_rev'] = pd.to_numeric(blocked['composite_rev'], errors='coerce')
blocked['composite_p_up'] = pd.to_numeric(blocked['composite_p_up'], errors='coerce')
blocked['vwap_stretch'] = pd.to_numeric(blocked['vwap_stretch'], errors='coerce') if 'vwap_stretch' in blocked.columns else np.nan
blocked['funding_bias'] = pd.to_numeric(blocked['funding_bias'], errors='coerce')
blocked['obi_score'] = pd.to_numeric(blocked['obi_score'], errors='coerce')
blocked['vpin_score'] = pd.to_numeric(blocked['vpin_score'], errors='coerce')
blocked['vol_score'] = pd.to_numeric(blocked['vol_score'], errors='coerce')
blocked['structure_bias'] = pd.to_numeric(blocked['structure_bias'], errors='coerce')
print(f"Blocked resolved: {len(blocked)}")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_edge(df, side_col='side', resolved_col='resolved_yes', pm_col='p_market'):
    """Given a filtered slice, compute WR, BE, edge, pnl per side."""
    results = {}
    for side in ['yes', 'no']:
        sub = df[df[side_col] == side] if side_col in df.columns else df
        n = len(sub)
        if n < MIN_N:
            continue
        if side == 'yes':
            wr = sub[resolved_col].mean()
            be = sub[pm_col].mean()
        else:
            wr = (1 - sub[resolved_col]).mean()
            be = (1 - sub[pm_col]).mean()
        edge = wr - be
        # Dollar PnL at flat $10/trade
        # YES win: +$10*(1-pm), lose: -$10*pm => expected = 10*(wr - be) / (1-be) * (1-be) roughly
        # Simplified: for each trade, pnl = 10*(1/pm - 1)*resolved - 10*(1-resolved) for YES
        # Use would_pnl if available, else approximate
        pnl_per_trade = edge * 10  # rough
        total_pnl = pnl_per_trade * n
        # stat test
        if side == 'yes':
            wins = sub[resolved_col].values
        else:
            wins = (1 - sub[resolved_col].values).astype(float)
        p_val = stats.ttest_1samp(wins - be, 0).pvalue if n >= 30 else np.nan
        results[side] = {
            'n': n, 'wr': wr, 'be': be, 'edge': edge,
            'pnl': total_pnl, 'p_val': p_val
        }
    return results

def fmt_result(side, r):
    p_str = f"p={r['p_val']:.3f}" if not np.isnan(r.get('p_val', np.nan)) else "p=n/a"
    tag = ""
    if abs(r['edge']) > 0.10:
        tag = " [HIGH VALUE]"
    elif abs(r['edge']) > 0.05:
        tag = " [MEDIUM VALUE]"
    return (f"  Side={side.upper()}: n={r['n']}, WR={r['wr']*100:.1f}%, "
            f"BE={r['be']*100:.1f}%, Edge={r['edge']*100:+.1f}%, "
            f"PnL@$10flat=${r['pnl']:.0f}, {p_str}{tag}")

# ──────────────────────────────────────────────────────────────────────────────
# COLLECT ALL FINDINGS
# ──────────────────────────────────────────────────────────────────────────────
findings = []

def add_finding(title, condition, sides_results, interpretation, extra=""):
    """Add a finding to the list."""
    for side, r in sides_results.items():
        findings.append({
            'title': title,
            'condition': condition,
            'side': side,
            'n': r['n'],
            'wr': r['wr'],
            'be': r['be'],
            'edge': r['edge'],
            'pnl': r['pnl'],
            'p_val': r.get('p_val', np.nan),
            'interpretation': interpretation,
            'extra': extra,
        })

lines = []
def out(s=""):
    lines.append(s)

out("=" * 80)
out("COMPREHENSIVE MISPRICING ANALYSIS — Kalshi Binary Options")
out("Date: 2026-05-17  |  BTC trades: 1564  ETH: 622  SOL: 412  Blocked: 98,459")
out("=" * 80)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: BASELINE BY ASSET × SIDE
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 1: BASELINE BY ASSET × SIDE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    res = compute_edge(df)
    out(f"\n[ASSET: {asset_name}]")
    for side, r in res.items():
        out(fmt_result(side, r))
        add_finding(f"{asset_name} Baseline", f"{asset_name} all trades", {side: r},
                    f"{asset_name} {side.upper()} baseline edge")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: PM BUCKETING (0.05 bins)
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 2: P_MARKET BUCKETING (0.05-wide bins)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol), ("ALL", all_trades)]:
    out(f"\n[{asset_name}] PM buckets")
    df2 = df.copy()
    df2['pm_bin'] = pd.cut(df2['p_market'], bins=np.arange(0, 1.05, 0.05), right=False)
    for pm_bin in sorted(df2['pm_bin'].dropna().unique(), key=lambda x: x.left):
        sub = df2[df2['pm_bin'] == pm_bin]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        bin_str = f"pm=[{pm_bin.left:.2f},{pm_bin.right:.2f})"
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {bin_str} {line}")
            add_finding(f"{asset_name} PM Bucket",
                        f"{asset_name} {bin_str}",
                        {side: r},
                        f"PM bucket {bin_str} {side} edge")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: OFFSET × SIDE
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 3: OFFSET_PCT × SIDE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    df2 = df.copy()
    # offset_pct: negative=below strike (YES likely wins), positive=above strike
    df2['offset_bin'] = pd.cut(df2['offset_pct'],
                                bins=[-1, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 1],
                                right=False)
    for ob in sorted(df2['offset_bin'].dropna().unique(), key=lambda x: x.left):
        sub = df2[df2['offset_bin'] == ob]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        ob_str = f"offset=[{ob.left:.2f},{ob.right:.2f})"
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {ob_str} {line}")
            add_finding(f"{asset_name} Offset",
                        f"{asset_name} {ob_str}",
                        {side: r},
                        f"Offset {ob_str} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: TAU (TIME TO EXPIRY)
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 4: TAU (MINUTES TO EXPIRY)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    df2 = df.copy()
    df2['tau_bin'] = pd.cut(df2['tau_minutes'],
                             bins=[0, 10, 20, 30, 45, 60, 90, 120, 300, 10000],
                             right=False)
    for tb in sorted(df2['tau_bin'].dropna().unique(), key=lambda x: x.left):
        sub = df2[df2['tau_bin'] == tb]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        tb_str = f"tau=[{int(tb.left)},{int(tb.right)}m)"
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {tb_str} {line}")
            add_finding(f"{asset_name} Tau",
                        f"{asset_name} {tb_str}",
                        {side: r},
                        f"Time to expiry {tb_str} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: VOL REGIME
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 5: VOL REGIME (vol_eff and rvol_1h)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}] vol_eff")
    df2 = df.copy()
    df2['vol_eff_n'] = pd.to_numeric(df2['vol_eff'], errors='coerce')
    valid = df2[df2['vol_eff_n'].notna()]
    if len(valid) >= MIN_N:
        tertiles = valid['vol_eff_n'].quantile([0.33, 0.67])
        def vol_regime(v):
            if v < tertiles[0.33]: return 'low_vol'
            elif v < tertiles[0.67]: return 'med_vol'
            else: return 'high_vol'
        df2['vol_regime'] = df2['vol_eff_n'].apply(lambda v: vol_regime(v) if not pd.isna(v) else np.nan)
        for regime in ['low_vol', 'med_vol', 'high_vol']:
            sub = df2[df2['vol_regime'] == regime]
            if len(sub) < MIN_N:
                continue
            res = compute_edge(sub)
            for side, r in res.items():
                line = fmt_result(side, r)
                out(f"  {regime} {line}")
                add_finding(f"{asset_name} Vol Regime",
                            f"{asset_name} {regime} (vol_eff)",
                            {side: r},
                            f"{regime} vol edge for {asset_name}")

    out(f"\n[{asset_name}] rvol_1h")
    valid_rv = df[df['rvol_1h'].notna() & (df['rvol_1h'] > 0)]
    if len(valid_rv) >= MIN_N * 2:
        med = valid_rv['rvol_1h'].median()
        for regime, sub in [('rvol_low', valid_rv[valid_rv['rvol_1h'] < med]),
                              ('rvol_high', valid_rv[valid_rv['rvol_1h'] >= med])]:
            if len(sub) < MIN_N:
                continue
            res = compute_edge(sub)
            for side, r in res.items():
                line = fmt_result(side, r)
                out(f"  {regime} {line}")
                add_finding(f"{asset_name} RVol",
                            f"{asset_name} {regime} (rvol_1h)",
                            {side: r},
                            f"RVol regime {regime} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: HOUR OF DAY
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 6: HOUR OF DAY (UTC)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ALL", all_trades)]:
    out(f"\n[{asset_name}]")
    df2 = df.copy()
    df2['hour_int'] = df2['hour_utc'].fillna(df2['hour_utc_derived']).astype(float).round().astype('Int64')
    for hour in range(24):
        sub = df2[df2['hour_int'] == hour]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  Hour={hour:02d}UTC {line}")
            add_finding(f"{asset_name} Hour of Day",
                        f"{asset_name} hour={hour:02d}UTC",
                        {side: r},
                        f"Hour {hour}UTC {side} edge for {asset_name}")

# Also group into sessions
out(f"\n[BTC] Trading Sessions")
df2 = btc.copy()
df2['hour_int'] = df2['hour_utc'].fillna(df2['hour_utc_derived']).astype(float).round().astype('Int64')
sessions = {
    'ASIA_00-08': (0, 8),
    'EUROPE_08-14': (8, 14),
    'US_14-20': (14, 20),
    'US_LATE_20-24': (20, 24)
}
for sess_name, (h_start, h_end) in sessions.items():
    sub = df2[(df2['hour_int'] >= h_start) & (df2['hour_int'] < h_end)]
    if len(sub) < MIN_N:
        continue
    res = compute_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r)
        out(f"  {sess_name} {line}")
        add_finding("BTC Session",
                    f"BTC {sess_name}",
                    {side: r},
                    f"Session {sess_name} {side} edge for BTC")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: DAY OF WEEK
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 7: DAY OF WEEK")
out("-" * 60)

dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
for asset_name, df in [("BTC", btc), ("ALL", all_trades)]:
    out(f"\n[{asset_name}]")
    for dow_num, dow_name in dow_names.items():
        sub = df[df['dow'] == dow_num]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {dow_name} {line}")
            add_finding(f"{asset_name} Day of Week",
                        f"{asset_name} {dow_name}",
                        {side: r},
                        f"{dow_name} {side} edge for {asset_name}")
    # Weekday vs weekend
    sub_wd = df[df['dow'] < 5]
    sub_we = df[df['dow'] >= 5]
    for label, sub in [('weekday', sub_wd), ('weekend', sub_we)]:
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {label} {line}")
            add_finding(f"{asset_name} Weekday/Weekend",
                        f"{asset_name} {label}",
                        {side: r},
                        f"{label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: EMA ALIGNMENT × SIDE
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 8: EMA ALIGNMENT × SIDE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    ema_labels = {-1: 'bearish', 0: 'neutral', 1: 'bullish'}
    for ema_val, ema_name in ema_labels.items():
        sub = df[df['ema_stack_bias'] == ema_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  EMA={ema_name} {line}")
            add_finding(f"{asset_name} EMA Alignment",
                        f"{asset_name} EMA={ema_name}",
                        {side: r},
                        f"EMA={ema_name} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: STOCH_K REGIME
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 9: STOCH_K REGIME")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    bins_stoch = [(0, 20, 'oversold<20'), (20, 40, '20-40'), (40, 60, '40-60'),
                  (60, 80, '60-80'), (80, 100.1, 'overbought>80')]
    for lo, hi, label in bins_stoch:
        sub = df[(df['stoch_k'] >= lo) & (df['stoch_k'] < hi)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  stoch_k={label} {line}")
            add_finding(f"{asset_name} Stoch_K",
                        f"{asset_name} stoch_k={label}",
                        {side: r},
                        f"Stoch_K {label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: VWAP STRETCH
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 10: VWAP STRETCH SCORE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    for vwap_val in sorted(df['vwap_stretch_score'].dropna().unique()):
        sub = df[df['vwap_stretch_score'] == vwap_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  vwap_stretch={vwap_val:.0f} {line}")
            add_finding(f"{asset_name} VWAP Stretch",
                        f"{asset_name} vwap_stretch={vwap_val:.0f}",
                        {side: r},
                        f"VWAP stretch={vwap_val} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: FUNDING BIAS
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 11: FUNDING BIAS × SIDE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    for fb_val, fb_label in [(-1, 'bearish_funding'), (0, 'neutral_funding'), (1, 'bullish_funding')]:
        sub = df[df['funding_bias'] == fb_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  funding={fb_label} {line}")
            add_finding(f"{asset_name} Funding Bias",
                        f"{asset_name} funding={fb_label}",
                        {side: r},
                        f"Funding {fb_label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: OBI + VPIN ORDER FLOW
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 12: ORDER FLOW (OBI + VPIN)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth)]:
    out(f"\n[{asset_name}] OBI score")
    for obi_val, obi_label in [(-1, 'sell_imbalance'), (0, 'neutral'), (1, 'buy_imbalance')]:
        sub = df[df['obi_score'] == obi_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  OBI={obi_label} {line}")
            add_finding(f"{asset_name} OBI",
                        f"{asset_name} OBI={obi_label}",
                        {side: r},
                        f"OBI {obi_label} {side} edge for {asset_name}")

    out(f"\n[{asset_name}] VPIN score")
    for vpin_val, vpin_label in [(-1, 'vpin_sell'), (0, 'vpin_neutral'), (1, 'vpin_buy')]:
        sub = df[df['vpin_score'] == vpin_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  VPIN={vpin_label} {line}")
            add_finding(f"{asset_name} VPIN",
                        f"{asset_name} VPIN={vpin_label}",
                        {side: r},
                        f"VPIN {vpin_label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13: COMPOSITE SCORES
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 13: COMPOSITE SCORES (trend, rev, p_up)")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}] composite_trend buckets")
    df2 = df.copy()
    df2['ctrend_bin'] = pd.cut(df2['composite_trend'],
                                bins=[-10, -3, -1, 1, 3, 10], right=True,
                                labels=['strong_bearish', 'bearish', 'neutral', 'bullish', 'strong_bullish'])
    for bin_label in ['strong_bearish', 'bearish', 'neutral', 'bullish', 'strong_bullish']:
        sub = df2[df2['ctrend_bin'] == bin_label]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  c_trend={bin_label} {line}")
            add_finding(f"{asset_name} Composite Trend",
                        f"{asset_name} c_trend={bin_label}",
                        {side: r},
                        f"Composite trend {bin_label} {side} edge for {asset_name}")

    out(f"\n[{asset_name}] composite_rev buckets")
    df2['crev_bin'] = pd.cut(df2['composite_rev'],
                              bins=[-15, -4, -2, 2, 4, 15], right=True,
                              labels=['strong_down', 'down', 'neutral', 'up', 'strong_up'])
    for bin_label in ['strong_down', 'down', 'neutral', 'up', 'strong_up']:
        sub = df2[df2['crev_bin'] == bin_label]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  c_rev={bin_label} {line}")
            add_finding(f"{asset_name} Composite Rev",
                        f"{asset_name} c_rev={bin_label}",
                        {side: r},
                        f"Composite rev {bin_label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14: MOMENTUM (chg_5m, chg_30m)
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 14: MOMENTUM (chg_5m, chg_30m) × SIDE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}] chg_5m")
    df2 = df.copy()
    valid = df2[df2['chg_5m'].notna()]
    if len(valid) >= MIN_N * 3:
        p33, p67 = valid['chg_5m'].quantile([0.33, 0.67])
        df2['chg5_regime'] = df2['chg_5m'].apply(
            lambda x: 'falling' if x < p33 else ('rising' if x > p67 else 'flat') if not pd.isna(x) else np.nan)
        for regime in ['falling', 'flat', 'rising']:
            sub = df2[df2['chg5_regime'] == regime]
            if len(sub) < MIN_N:
                continue
            res = compute_edge(sub)
            for side, r in res.items():
                line = fmt_result(side, r)
                out(f"  chg_5m={regime} {line}")
                add_finding(f"{asset_name} Momentum 5m",
                            f"{asset_name} chg_5m={regime}",
                            {side: r},
                            f"5m momentum {regime} {side} edge for {asset_name}")

    out(f"\n[{asset_name}] chg_30m")
    valid = df2[df2['chg_30m'].notna()]
    if len(valid) >= MIN_N * 3:
        p33, p67 = valid['chg_30m'].quantile([0.33, 0.67])
        df2['chg30_regime'] = df2['chg_30m'].apply(
            lambda x: 'falling' if x < p33 else ('rising' if x > p67 else 'flat') if not pd.isna(x) else np.nan)
        for regime in ['falling', 'flat', 'rising']:
            sub = df2[df2['chg30_regime'] == regime]
            if len(sub) < MIN_N:
                continue
            res = compute_edge(sub)
            for side, r in res.items():
                line = fmt_result(side, r)
                out(f"  chg_30m={regime} {line}")
                add_finding(f"{asset_name} Momentum 30m",
                            f"{asset_name} chg_30m={regime}",
                            {side: r},
                            f"30m momentum {regime} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15: VOL SCORE
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 15: VOL SCORE")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}]")
    for vs_val, vs_label in [(-1, 'low_vol'), (0, 'med_vol'), (1, 'high_vol')]:
        sub = df[df['vol_score'] == vs_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  vol_score={vs_label} {line}")
            add_finding(f"{asset_name} Vol Score",
                        f"{asset_name} vol_score={vs_label}",
                        {side: r},
                        f"Vol score {vs_label} {side} edge for {asset_name}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 16: MULTI-DIMENSIONAL CROSS-CUTS
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 16: MULTI-DIMENSIONAL CROSS-CUTS")
out("-" * 60)

# 16a: BTC pm × ema_stack_bias × side
out("\n[16a] BTC: pm_bin × ema_alignment × side")
df2 = btc.copy()
df2['pm_coarse'] = pd.cut(df2['p_market'], bins=[0, 0.30, 0.45, 0.55, 0.70, 1.0],
                           labels=['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES'])
for pm_label in ['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES']:
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = df2[(df2['pm_coarse'] == pm_label) & (df2['ema_stack_bias'] == ema_val)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.03:  # only output meaningful
                line = fmt_result(side, r)
                out(f"  pm={pm_label} ema={ema_name} {line}")
                add_finding("BTC pm×ema×side",
                            f"BTC pm={pm_label} ema={ema_name}",
                            {side: r},
                            f"PM×EMA combo pm={pm_label} ema={ema_name} {side}")

# 16b: BTC stoch × pm × side
out("\n[16b] BTC: stoch_regime × pm_coarse × side")
df2['stoch_regime'] = pd.cut(df2['stoch_k'], bins=[0, 20, 40, 60, 80, 101],
                              labels=['OS<20', '20-40', '40-60', '60-80', 'OB>80'])
for stoch_label in ['OS<20', '20-40', '40-60', '60-80', 'OB>80']:
    for pm_label in ['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES']:
        sub = df2[(df2['stoch_regime'] == stoch_label) & (df2['pm_coarse'] == pm_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  stoch={stoch_label} pm={pm_label} {line}")
                add_finding("BTC stoch×pm×side",
                            f"BTC stoch={stoch_label} pm={pm_label}",
                            {side: r},
                            f"Stoch×PM combo stoch={stoch_label} pm={pm_label} {side}")

# 16c: tau × vol_regime × side (BTC)
out("\n[16c] BTC: tau_bucket × vol_regime × side")
df2['tau_3way'] = pd.cut(df2['tau_minutes'], bins=[0, 20, 60, 10000],
                          labels=['short<20m', 'med20-60m', 'long>60m'])
df2['rvol_2way'] = df2['rvol_1h'].apply(
    lambda x: 'hi_rvol' if x > 1.5 else ('lo_rvol' if x < 1.5 else 'lo_rvol') if not pd.isna(x) else np.nan)
for tau_label in ['short<20m', 'med20-60m', 'long>60m']:
    for rvol_label in ['lo_rvol', 'hi_rvol']:
        sub = df2[(df2['tau_3way'] == tau_label) & (df2['rvol_2way'] == rvol_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.03:
                line = fmt_result(side, r)
                out(f"  tau={tau_label} rvol={rvol_label} {line}")
                add_finding("BTC tau×rvol×side",
                            f"BTC tau={tau_label} rvol={rvol_label}",
                            {side: r},
                            f"Tau×RVol combo tau={tau_label} rvol={rvol_label} {side}")

# 16d: BTC pm × chg_30m × side
out("\n[16d] BTC: pm × 30m_momentum × side")
df2['mom30'] = df2['chg_30m'].apply(
    lambda x: 'up30' if x > 0.005 else ('dn30' if x < -0.005 else 'flat30') if not pd.isna(x) else np.nan)
for pm_label in ['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES']:
    for mom_label in ['dn30', 'flat30', 'up30']:
        sub = df2[(df2['pm_coarse'] == pm_label) & (df2['mom30'] == mom_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  pm={pm_label} mom30={mom_label} {line}")
                add_finding("BTC pm×momentum×side",
                            f"BTC pm={pm_label} mom30={mom_label}",
                            {side: r},
                            f"PM×Momentum30 combo pm={pm_label} mom={mom_label} {side}")

# 16e: BTC: ema × composite_rev × side
out("\n[16e] BTC: ema_stack × c_rev_bucket × side")
df2['crev_3way'] = pd.cut(df2['composite_rev'], bins=[-15, -2, 2, 15],
                           labels=['rev_bearish', 'rev_neutral', 'rev_bullish'])
for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
    for crev_label in ['rev_bearish', 'rev_neutral', 'rev_bullish']:
        sub = df2[(df2['ema_stack_bias'] == ema_val) & (df2['crev_3way'] == crev_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  ema={ema_name} c_rev={crev_label} {line}")
                add_finding("BTC ema×c_rev×side",
                            f"BTC ema={ema_name} c_rev={crev_label}",
                            {side: r},
                            f"EMA×CRev combo ema={ema_name} crev={crev_label} {side}")

# 16f: BTC: stoch × ema × c_trend × side (3D)
out("\n[16f] BTC: stoch(OB/OS) × ema × side (3D)")
for stoch_extreme in ['OS<20', 'OB>80']:
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = df2[(df2['stoch_regime'] == stoch_extreme) & (df2['ema_stack_bias'] == ema_val)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  stoch={stoch_extreme} ema={ema_name} {line}")
            add_finding("BTC stoch_extreme×ema",
                        f"BTC stoch={stoch_extreme} ema={ema_name}",
                        {side: r},
                        f"StochExtreme×EMA stoch={stoch_extreme} ema={ema_name} {side}")

# 16g: BTC pm × offset × side
out("\n[16g] BTC: pm_coarse × offset_direction × side")
df2['offset_dir'] = df2['offset_pct'].apply(
    lambda x: 'below_strike' if x < -0.01 else ('above_strike' if x > 0.01 else 'at_strike') if not pd.isna(x) else np.nan)
for pm_label in ['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES']:
    for offset_label in ['below_strike', 'at_strike', 'above_strike']:
        sub = df2[(df2['pm_coarse'] == pm_label) & (df2['offset_dir'] == offset_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  pm={pm_label} offset={offset_label} {line}")
                add_finding("BTC pm×offset×side",
                            f"BTC pm={pm_label} offset={offset_label}",
                            {side: r},
                            f"PM×Offset combo pm={pm_label} offset={offset_label} {side}")

# 16h: BTC deep c_trend × stoch × side
out("\n[16h] BTC: deep_trend(>=3) × stoch_regime × side")
for ct_label, ct_filter in [('trend_bull_strong', df2['composite_trend'] >= 3),
                              ('trend_bear_strong', df2['composite_trend'] <= -3)]:
    for stoch_label in ['OS<20', '20-40', '40-60', '60-80', 'OB>80']:
        sub = df2[ct_filter & (df2['stoch_regime'] == stoch_label)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  c_trend={ct_label} stoch={stoch_label} {line}")
                add_finding("BTC c_trend×stoch",
                            f"BTC c_trend={ct_label} stoch={stoch_label}",
                            {side: r},
                            f"Trend×Stoch combo trend={ct_label} stoch={stoch_label} {side}")

# 16i: BTC composite_p_up × side × pm
out("\n[16i] BTC: composite_p_up bucket × side")
df2['pup_bin'] = pd.cut(df2['composite_p_up'], bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
                         labels=['strong_dn', 'lean_dn', 'neutral', 'lean_up', 'strong_up'])
for pup_label in ['strong_dn', 'lean_dn', 'neutral', 'lean_up', 'strong_up']:
    sub = df2[df2['pup_bin'] == pup_label]
    if len(sub) < MIN_N:
        continue
    res = compute_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r)
        out(f"  comp_p_up={pup_label} {line}")
        add_finding("BTC composite_p_up",
                    f"BTC comp_p_up={pup_label}",
                    {side: r},
                    f"Composite p_up={pup_label} {side} edge for BTC")

# 16j: BTC funding × ema × side
out("\n[16j] BTC: funding_bias × ema × side")
for fb_val, fb_label in [(0, 'funding_neutral'), (1, 'funding_bull')]:
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = df2[(df2['funding_bias'] == fb_val) & (df2['ema_stack_bias'] == ema_val)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.04:
                line = fmt_result(side, r)
                out(f"  funding={fb_label} ema={ema_name} {line}")
                add_finding("BTC funding×ema",
                            f"BTC funding={fb_label} ema={ema_name}",
                            {side: r},
                            f"Funding×EMA funding={fb_label} ema={ema_name} {side}")

# 16k: ETH multi-dimensional
out("\n[16k] ETH: pm_coarse × ema × side")
eth2 = eth.copy()
eth2['pm_coarse'] = pd.cut(eth2['p_market'], bins=[0, 0.30, 0.45, 0.55, 0.70, 1.0],
                            labels=['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES'])
for pm_label in ['deep_NO', 'lean_NO', 'near_even', 'lean_YES', 'deep_YES']:
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = eth2[(eth2['pm_coarse'] == pm_label) & (eth2['ema_stack_bias'] == ema_val)]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.05:
                line = fmt_result(side, r)
                out(f"  ETH pm={pm_label} ema={ema_name} {line}")
                add_finding("ETH pm×ema×side",
                            f"ETH pm={pm_label} ema={ema_name}",
                            {side: r},
                            f"ETH PM×EMA pm={pm_label} ema={ema_name} {side}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 17: BLOCKED TRADES ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 17: BLOCKED TRADES — GATES INCORRECTLY BLOCKING POSITIVE EDGE")
out("-" * 60)

def compute_blocked_edge(sub):
    """Compute edge for blocked trades (pm vs resolved_yes)."""
    results = {}
    for side in ['yes', 'no']:
        s = sub[sub['side'] == side]
        n = len(s)
        if n < MIN_N:
            continue
        if side == 'yes':
            wr = s['resolved_yes'].mean()
            be = s['pm'].mean()
        else:
            wr = (1 - s['resolved_yes']).mean()
            be = (1 - s['pm']).mean()
        edge = wr - be
        pnl = edge * 10 * n
        wins = s['resolved_yes'].values if side == 'yes' else (1 - s['resolved_yes']).values
        p_val = stats.ttest_1samp(wins - be, 0).pvalue if n >= 30 else np.nan
        results[side] = {'n': n, 'wr': wr, 'be': be, 'edge': edge, 'pnl': pnl, 'p_val': p_val}
    return results

out("\nEdge analysis by gate name (gates that might be wrongly blocking trades):")
gate_findings = []
for gate in sorted(blocked['gate_name'].dropna().unique()):
    sub = blocked[blocked['gate_name'] == gate]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        if r['edge'] > 0.05:  # positive edge = we SHOULD be taking this
            gate_findings.append((gate, side, r))

gate_findings.sort(key=lambda x: x[2]['edge'], reverse=True)
for gate, side, r in gate_findings:
    tag = "[HIGH VALUE - OVER-BLOCKING]" if r['edge'] > 0.10 else "[MEDIUM VALUE - OVER-BLOCKING]"
    out(f"  Gate={gate} Side={side.upper()}: n={r['n']}, WR={r['wr']*100:.1f}%, "
        f"BE={r['be']*100:.1f}%, Edge={r['edge']*100:+.1f}%, "
        f"PnL@$10flat=${r['pnl']:.0f} {tag}")
    add_finding(f"Blocked Gate: {gate}",
                f"Gate={gate} Side={side}",
                {side: r},
                f"Gate {gate} blocks {side} trades with positive edge {r['edge']*100:.1f}%")

# Also analyze gates with NEGATIVE edge (correctly blocked)
out("\nGates with large NEGATIVE edge (correctly blocking losses):")
neg_findings = []
for gate in sorted(blocked['gate_name'].dropna().unique()):
    sub = blocked[blocked['gate_name'] == gate]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        if r['edge'] < -0.05:
            neg_findings.append((gate, side, r))
neg_findings.sort(key=lambda x: x[2]['edge'])
for gate, side, r in neg_findings[:15]:
    out(f"  Gate={gate} Side={side.upper()}: n={r['n']}, WR={r['wr']*100:.1f}%, "
        f"BE={r['be']*100:.1f}%, Edge={r['edge']*100:+.1f}% [CORRECTLY BLOCKED]")

# Blocked trades by asset
out("\nBlocked trades edge by asset × side:")
for asset_name in ['BTC', 'ETH', 'SOL']:
    sub = blocked[blocked['asset'] == asset_name]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r).replace('p_market', 'pm')
        out(f"  {asset_name} Blocked {line}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 18: CALIBRATION (actual WR vs p_market)
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 18: CALIBRATION — ACTUAL WR vs P_MARKET BY PM BUCKET")
out("-" * 60)
out("  (positive = market underprices YES, negative = market overprices YES)")
out()

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol), ("ALL", all_trades)]:
    out(f"\n[{asset_name}] Calibration error by pm bucket")
    df2 = df.copy()
    df2['pm_bin'] = pd.cut(df2['p_market'], bins=np.arange(0, 1.05, 0.05), right=False)
    rows = []
    for pm_bin in sorted(df2['pm_bin'].dropna().unique(), key=lambda x: x.left):
        sub = df2[df2['pm_bin'] == pm_bin]
        if len(sub) < 10:
            continue
        actual_wr = sub['resolved_yes'].mean()
        pm_mid = (pm_bin.left + pm_bin.right) / 2
        calib_err = actual_wr - pm_mid
        rows.append((pm_bin, len(sub), actual_wr, pm_mid, calib_err))
    for pm_bin, n, actual_wr, pm_mid, calib_err in rows:
        tag = " *** LARGE BIAS ***" if abs(calib_err) > 0.10 else ""
        out(f"  pm=[{pm_bin.left:.2f},{pm_bin.right:.2f}) n={n:4d} "
            f"actual_WR={actual_wr*100:.1f}% pm_mid={pm_mid*100:.1f}% "
            f"calib_err={calib_err*100:+.1f}%{tag}")
        add_finding(f"{asset_name} Calibration",
                    f"{asset_name} pm=[{pm_bin.left:.2f},{pm_bin.right:.2f})",
                    {'yes': {'n': n, 'wr': actual_wr, 'be': pm_mid,
                              'edge': calib_err, 'pnl': calib_err * 10 * n,
                              'p_val': np.nan}},
                    f"Calibration error {calib_err*100:+.1f}% for pm=[{pm_bin.left:.2f},{pm_bin.right:.2f})")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 19: TIME-SERIES — MONTHLY EDGE STABILITY
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 19: TIME-SERIES — MONTHLY EDGE STABILITY")
out("-" * 60)

for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    out(f"\n[{asset_name}] Monthly edge")
    for month in sorted(df['month_str'].dropna().unique()):
        sub = df[df['month_str'] == month]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {month} {line}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 20: CROSS-ASSET COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 20: CROSS-ASSET MISPRICING COMPARISON")
out("-" * 60)

# Common factor comparisons across assets
out("\n[All Assets] EMA alignment × side")
for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = df[df['ema_stack_bias'] == ema_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.03:
                line = fmt_result(side, r)
                out(f"  {asset_name} EMA={ema_name} {line}")

out("\n[All Assets] PM > 0.70 vs PM < 0.30 (deep ITM both sides)")
for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    for label, cond in [('deep_YES(pm>0.70)', df['p_market'] > 0.70),
                         ('deep_NO(pm<0.30)', df['p_market'] < 0.30)]:
        sub = df[cond]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            if abs(r['edge']) > 0.03:
                line = fmt_result(side, r)
                out(f"  {asset_name} {label} {line}")

out("\n[All Assets] Composite trend distribution + edge")
for asset_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
    for label, cond in [('bull_trend(c_trend>=3)', df['composite_trend'] >= 3),
                         ('bear_trend(c_trend<=-3)', df['composite_trend'] <= -3)]:
        sub = df[cond]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {asset_name} {label} {line}")

# ──────────────────────────────────────────────────────────────────────────────
# ADDITIONAL DEEP DIVES
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 21: ADDITIONAL DEEP DIVES")
out("-" * 60)

# 21a: ADX (trend strength) for BTC where available
out("\n[21a] BTC: ADX_1h regime (where available)")
btc_adx = btc[btc['adx_1h'].notna() & (btc['adx_1h'] > 0)]
if len(btc_adx) >= MIN_N:
    for label, sub in [('adx_weak(<20)', btc_adx[btc_adx['adx_1h'] < 20]),
                        ('adx_mod(20-40)', btc_adx[(btc_adx['adx_1h'] >= 20) & (btc_adx['adx_1h'] < 40)]),
                        ('adx_strong(>40)', btc_adx[btc_adx['adx_1h'] >= 40])]:
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {label} {line}")
            add_finding("BTC ADX Regime",
                        f"BTC {label}",
                        {side: r},
                        f"ADX regime {label} {side} edge for BTC")

# 21b: BTC z_score buckets
out("\n[21b] BTC: z_score regime")
df2 = btc.copy()
df2['z_score'] = pd.to_numeric(df2['z_score'], errors='coerce')
valid = df2[df2['z_score'].notna()]
if len(valid) >= MIN_N:
    for label, sub in [('z_low(<-1)', valid[valid['z_score'] < -1]),
                        ('z_neutral(-1-1)', valid[(valid['z_score'] >= -1) & (valid['z_score'] <= 1)]),
                        ('z_high(>1)', valid[valid['z_score'] > 1])]:
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  z_score={label} {line}")
            add_finding("BTC Z-Score",
                        f"BTC z_score={label}",
                        {side: r},
                        f"Z-score {label} {side} edge for BTC")

# 21c: structure_bias analysis
out("\n[21c] BTC: structure_bias × side")
df2 = btc.copy()
df2['structure_bias'] = pd.to_numeric(df2['structure_bias'], errors='coerce')
for sb_val, sb_label in [(-1, 'struct_bear'), (0, 'struct_neutral'), (1, 'struct_bull')]:
    sub = df2[df2['structure_bias'] == sb_val]
    if len(sub) < MIN_N:
        continue
    res = compute_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r)
        out(f"  structure={sb_label} {line}")
        add_finding("BTC Structure Bias",
                    f"BTC structure={sb_label}",
                    {side: r},
                    f"Structure {sb_label} {side} edge for BTC")

# 21d: BTC: near-ITM specific (pm 0.45-0.55) deep-dive
out("\n[21d] BTC near-ITM (pm=[0.45,0.55]) × all signal combos")
near_itm = btc[(btc['p_market'] >= 0.45) & (btc['p_market'] < 0.55)].copy()
if len(near_itm) >= MIN_N:
    out(f"  Total near-ITM: {len(near_itm)}")
    for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
        sub = near_itm[near_itm['ema_stack_bias'] == ema_val]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  near_ITM ema={ema_name} {line}")
            add_finding("BTC Near-ITM EMA",
                        f"BTC near_ITM pm=[0.45,0.55) ema={ema_name}",
                        {side: r},
                        f"Near-ITM EMA={ema_name} {side} edge for BTC")

    # Near-ITM × stoch
    for stoch_label in ['OS<20', '20-40', '40-60', '60-80', 'OB>80']:
        near_itm['stoch_regime'] = pd.cut(near_itm['stoch_k'], bins=[0, 20, 40, 60, 80, 101],
                                           labels=['OS<20', '20-40', '40-60', '60-80', 'OB>80'])
        sub = near_itm[near_itm['stoch_regime'] == stoch_label]
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  near_ITM stoch={stoch_label} {line}")
            add_finding("BTC Near-ITM Stoch",
                        f"BTC near_ITM pm=[0.45,0.55) stoch={stoch_label}",
                        {side: r},
                        f"Near-ITM Stoch={stoch_label} {side} edge for BTC")

# 21e: BTC: confirmation_score
out("\n[21e] BTC: confirmation_score × side")
df2 = btc.copy()
df2['conf_score'] = pd.to_numeric(df2['confirmation_score'], errors='coerce')
for label, sub in [('conf_neg(<=0)', df2[df2['conf_score'] <= 0]),
                    ('conf_pos1(1-2)', df2[(df2['conf_score'] >= 1) & (df2['conf_score'] <= 2)]),
                    ('conf_pos2(3+)', df2[df2['conf_score'] >= 3])]:
    if len(sub) < MIN_N:
        continue
    res = compute_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r)
        out(f"  confirmation={label} {line}")
        add_finding("BTC Confirmation Score",
                    f"BTC confirmation={label}",
                    {side: r},
                    f"Confirmation {label} {side} edge for BTC")

# 21f: BTC: p_gbdt (if available)
out("\n[21f] BTC: p_gbdt regime")
df2 = btc.copy()
df2['p_gbdt'] = pd.to_numeric(df2['p_gbdt'], errors='coerce')
valid_gbdt = df2[df2['p_gbdt'].notna() & (df2['p_gbdt'] > 0)]
if len(valid_gbdt) >= MIN_N:
    for label, sub in [('gbdt_low(<0.40)', valid_gbdt[valid_gbdt['p_gbdt'] < 0.40]),
                        ('gbdt_mid(0.40-0.60)', valid_gbdt[(valid_gbdt['p_gbdt'] >= 0.40) & (valid_gbdt['p_gbdt'] < 0.60)]),
                        ('gbdt_high(>0.60)', valid_gbdt[valid_gbdt['p_gbdt'] >= 0.60])]:
        if len(sub) < MIN_N:
            continue
        res = compute_edge(sub)
        for side, r in res.items():
            line = fmt_result(side, r)
            out(f"  {label} {line}")
            add_finding("BTC GBDT Model",
                        f"BTC {label}",
                        {side: r},
                        f"GBDT regime {label} {side} edge for BTC")

# 21g: SOL detailed breakdown
out("\n[21g] SOL: key signal analysis")
for label, sub in [('SOL EMA bull', sol[sol['ema_stack_bias'] == 1]),
                    ('SOL EMA bear', sol[sol['ema_stack_bias'] == -1]),
                    ('SOL EMA neut', sol[sol['ema_stack_bias'] == 0]),
                    ('SOL pm<0.50', sol[sol['p_market'] < 0.50]),
                    ('SOL pm>=0.50', sol[sol['p_market'] >= 0.50])]:
    if len(sub) < MIN_N:
        continue
    res = compute_edge(sub)
    for side, r in res.items():
        if abs(r['edge']) > 0.03:
            line = fmt_result(side, r)
            out(f"  {label} {line}")
            add_finding("SOL Key Signals",
                        label,
                        {side: r},
                        f"SOL {label} {side} edge")

# ──────────────────────────────────────────────────────────────────────────────
# BLOCKED TRADES: DEEP DIVE BY GATE + SIGNALS
# ──────────────────────────────────────────────────────────────────────────────
out()
out("## SECTION 22: BLOCKED TRADES — SIGNAL PATTERN ANALYSIS")
out("-" * 60)

out("\n[Blocked] pm bucket analysis (all blocked trades)")
blocked2 = blocked.copy()
blocked2['pm_bin'] = pd.cut(blocked2['pm'], bins=np.arange(0, 1.05, 0.10), right=False)
for pm_bin in sorted(blocked2['pm_bin'].dropna().unique(), key=lambda x: x.left):
    sub = blocked2[blocked2['pm_bin'] == pm_bin]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        if abs(r['edge']) > 0.05:
            tag = " [POSITIVE EDGE - OVER-BLOCKING]" if r['edge'] > 0 else " [NEGATIVE EDGE - CORRECT BLOCK]"
            out(f"  Blocked pm=[{pm_bin.left:.1f},{pm_bin.right:.1f}) "
                f"n={r['n']}, WR={r['wr']*100:.1f}%, BE={r['be']*100:.1f}%, "
                f"Edge={r['edge']*100:+.1f}%{tag}")

out("\n[Blocked] EMA alignment analysis")
for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
    sub = blocked[blocked['ema_stack_bias'] == ema_val]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        line = fmt_result(side, r).replace('p_market', 'pm')
        out(f"  Blocked EMA={ema_name} {line}")
        add_finding("Blocked EMA Analysis",
                    f"Blocked EMA={ema_name}",
                    {side: r},
                    f"Blocked trades EMA={ema_name} {side} edge")

out("\n[Blocked] smc_gate deep dive (largest gate)")
smc_sub = blocked[blocked['gate_name'] == 'smc_gate']
out(f"  Total smc_gate blocked resolved: {len(smc_sub)}")
for ema_val, ema_name in [(-1, 'bear'), (0, 'neut'), (1, 'bull')]:
    sub = smc_sub[smc_sub['ema_stack_bias'] == ema_val]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        if abs(r['edge']) > 0.03:
            line = fmt_result(side, r).replace('p_market', 'pm')
            out(f"  smc_gate EMA={ema_name} {line}")

out("\n[Blocked] no_pm_floor deep dive")
npm_sub = blocked[blocked['gate_name'] == 'no_pm_floor']
out(f"  Total no_pm_floor blocked resolved: {len(npm_sub)}")
for pm_bin in [0.1, 0.2, 0.3, 0.4, 0.5]:
    sub = npm_sub[(npm_sub['pm'] >= pm_bin) & (npm_sub['pm'] < pm_bin + 0.1)]
    if len(sub) < MIN_N:
        continue
    res = compute_blocked_edge(sub)
    for side, r in res.items():
        if abs(r['edge']) > 0.03:
            out(f"  no_pm_floor pm=[{pm_bin:.1f},{pm_bin+0.1:.1f}) "
                f"n={r['n']}, WR={r['wr']*100:.1f}%, BE={r['be']*100:.1f}%, "
                f"Edge={r['edge']*100:+.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY: TOP 20 MOST ACTIONABLE FINDINGS
# ──────────────────────────────────────────────────────────────────────────────
out()
out("=" * 80)
out("TOP 20 MOST ACTIONABLE FINDINGS (sorted by |edge|)")
out("=" * 80)

# Build clean findings frame
findings_df = pd.DataFrame(findings)
# Remove duplicates, keep highest edge per condition
findings_df = findings_df.drop_duplicates(subset=['condition', 'side'])
findings_df['abs_edge'] = findings_df['edge'].abs()
findings_df = findings_df.sort_values('abs_edge', ascending=False)
# Remove degenerate cases where n is 0 or missing
findings_df = findings_df[findings_df['n'] >= MIN_N]

top20 = findings_df.head(40)  # get 40, filter for variety

seen_cats = {}
top_final = []
for _, row in top20.iterrows():
    cat = row['title'].split()[0]  # first word as category
    if cat not in seen_cats:
        seen_cats[cat] = 0
    if seen_cats[cat] < 4:  # max 4 per category
        top_final.append(row)
        seen_cats[cat] += 1
    if len(top_final) >= 20:
        break

for i, row in enumerate(top_final, 1):
    tag = ""
    if row['abs_edge'] > 0.10:
        tag = " *** HIGH VALUE ***"
    elif row['abs_edge'] > 0.05:
        tag = " ** MEDIUM VALUE **"
    p_str = f"p={row['p_val']:.3f}" if not pd.isna(row['p_val']) else "p=n/a"
    direction = "EXPLOIT" if row['edge'] > 0 else "AVOID"
    out()
    out(f"[FINDING {i:02d}] {row['title']}{tag}")
    out(f"  Condition: {row['condition']}")
    out(f"  n={row['n']}, WR={row['wr']*100:.1f}%, BE={row['be']*100:.1f}%, "
        f"Edge={row['edge']*100:+.1f}%, PnL@$10flat=${row['pnl']:.0f}")
    out(f"  {p_str} | Action: {direction}")
    out(f"  Interpretation: {row['interpretation']}")

out()
out("=" * 80)
out("END OF MISPRICING ANALYSIS")
out("=" * 80)

# Write output
output = "\n".join(lines)
with open(OUTPUT_FILE, 'w') as f:
    f.write(output)

print(f"\nOutput written to {OUTPUT_FILE}")
print(f"Total findings collected: {len(findings_df)}")
print(f"\nSummary statistics:")
print(f"  HIGH VALUE findings (|edge|>10%): {(findings_df['abs_edge']>0.10).sum()}")
print(f"  MEDIUM VALUE findings (|edge|>5%): {(findings_df['abs_edge']>0.05).sum()}")
print(f"  Total findings: {len(findings_df)}")

# Print top 20 to console as well
print("\n" + "="*60)
print("TOP 20 FINDINGS SUMMARY:")
print("="*60)
for i, row in enumerate(top_final, 1):
    direction = "EXPLOIT" if row['edge'] > 0 else "AVOID"
    tag = " [HIGH]" if row['abs_edge'] > 0.10 else (" [MED]" if row['abs_edge'] > 0.05 else "")
    print(f"{i:2d}. {row['title'][:45]:<45} "
          f"Edge={row['edge']*100:+.1f}%{tag} "
          f"n={row['n']:5d} "
          f"PnL=${row['pnl']:+8.0f} "
          f"({direction})")
