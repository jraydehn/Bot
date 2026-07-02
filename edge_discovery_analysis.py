#!/usr/bin/env python3
"""
Comprehensive BTC Scan Archive Edge Discovery Analysis
Parts 1-5: IC ranking, bucket sweeps, 2-way interactions, 3-way, summary
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD & FILTER
# ─────────────────────────────────────────────
df = pd.read_parquet('/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/btc_scan_archive_hmm.parquet')
df = df[(df['p_market'] >= 0.15) & (df['p_market'] <= 0.85) & (df['resolved_yes'].notna())].copy()
df['win_yes'] = (df['resolved_yes'] == 1.0).astype(int)
df['win_no']  = (df['resolved_yes'] == 0.0).astype(int)
df['pm'] = df['p_market']
print(f"=== BASE UNIVERSE: {len(df):,} rows | YES wins: {df['win_yes'].sum():,} | NO wins: {df['win_no'].sum():,}")
print(f"Overall YES WR: {df['win_yes'].mean():.3f} | Overall NO WR: {df['win_no'].mean():.3f}\n")

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def bucket_stats(sub, label=""):
    """Given a sub-df, return dict of n, yes_wr, yes_bkev, yes_edge, no_wr, no_bkev, no_edge"""
    n = len(sub)
    if n == 0:
        return dict(n=0, yes_wr=np.nan, yes_bkev=np.nan, yes_edge=np.nan,
                    no_wr=np.nan, no_bkev=np.nan, no_edge=np.nan)
    yes_wr   = sub['win_yes'].mean()
    yes_bkev = sub['pm'].mean()
    no_wr    = sub['win_no'].mean()
    no_bkev  = (1 - sub['pm']).mean()
    return dict(n=n,
                yes_wr=yes_wr, yes_bkev=yes_bkev, yes_edge=yes_wr-yes_bkev,
                no_wr=no_wr,  no_bkev=no_bkev,   no_edge=no_wr-no_bkev)

def print_bucket_table(results, feature_name, sig_n=100, sig_edge=0.05):
    """results: list of (label, stats_dict)"""
    print(f"\n{'='*80}")
    print(f"  FEATURE: {feature_name}")
    print(f"{'='*80}")
    hdr = f"{'Bucket':<25} {'N':>7} | {'YES_WR':>7} {'Y_bkev':>7} {'Y_edge':>8} | {'NO_WR':>7} {'N_bkev':>7} {'N_edge':>8} |"
    print(hdr)
    print('-'*len(hdr))
    for label, s in results:
        if s['n'] == 0:
            print(f"  {label:<23} {'0':>7} | {'—':>7} {'—':>7} {'—':>8} | {'—':>7} {'—':>7} {'—':>8} |")
            continue
        flags = []
        if s['n'] >= sig_n and abs(s['yes_edge']) >= sig_edge:
            flags.append(f"YES={'+'if s['yes_edge']>0 else ''}{s['yes_edge']:.3f}★")
        if s['n'] >= sig_n and abs(s['no_edge']) >= sig_edge:
            flags.append(f"NO={'+'if s['no_edge']>0 else ''}{s['no_edge']:.3f}★")
        flag_str = '  ' + ' '.join(flags) if flags else ''
        print(f"  {label:<23} {s['n']:>7,} | {s['yes_wr']:>7.3f} {s['yes_bkev']:>7.3f} {s['yes_edge']:>+8.3f} | "
              f"{s['no_wr']:>7.3f} {s['no_bkev']:>7.3f} {s['no_edge']:>+8.3f} |{flag_str}")


# ═══════════════════════════════════════════════════════════════
# PART 1: FEATURE IC RANKING
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*80)
print("  PART 1: FEATURE IC RANKING (Spearman correlation vs outcome)")
print("═"*80)

EXCLUDE = {'logged_at','contract_ticker','close_ts','spot','strike','p_market','pm',
           'resolved_yes','spot_at_expiry','price_move_pct','miss_pct','win_yes','win_no'}

numeric_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
print(f"Evaluating {len(numeric_cols)} numeric features...\n")

ic_records = []
for col in numeric_cols:
    sub = df[[col, 'win_yes', 'win_no']].dropna()
    if len(sub) < 200:
        continue
    ic_yes, p_yes = stats.spearmanr(sub[col], sub['win_yes'])
    ic_no,  p_no  = stats.spearmanr(sub[col], sub['win_no'])
    ic_records.append(dict(feature=col, n=len(sub),
                           IC_yes=ic_yes, p_yes=p_yes,
                           IC_no=ic_no,   p_no=p_no,
                           abs_IC_yes=abs(ic_yes), abs_IC_no=abs(ic_no)))

ic_df = pd.DataFrame(ic_records)

print(f"{'─'*70}")
print(f"TOP 20 BY |IC_YES| (correlation of feature with YES win):")
print(f"{'─'*70}")
top_yes = ic_df.nlargest(20, 'abs_IC_yes')
for _, r in top_yes.iterrows():
    sig = '***' if r['p_yes']<0.001 else '**' if r['p_yes']<0.01 else '*' if r['p_yes']<0.05 else ''
    print(f"  {r['feature']:<25} IC_yes={r['IC_yes']:+.4f}  IC_no={r['IC_no']:+.4f}  n={int(r['n']):,}  {sig}")

print(f"\n{'─'*70}")
print(f"TOP 20 BY |IC_NO| (correlation of feature with NO win):")
print(f"{'─'*70}")
top_no = ic_df.nlargest(20, 'abs_IC_no')
for _, r in top_no.iterrows():
    sig = '***' if r['p_no']<0.001 else '**' if r['p_no']<0.01 else '*' if r['p_no']<0.05 else ''
    print(f"  {r['feature']:<25} IC_no={r['IC_no']:+.4f}  IC_yes={r['IC_yes']:+.4f}  n={int(r['n']):,}  {sig}")

top5_yes = top_yes['feature'].tolist()[:5]
print(f"\nTop-5 by IC_YES: {top5_yes}")


# ═══════════════════════════════════════════════════════════════
# PART 2: SYSTEMATIC 1-FEATURE BUCKET SWEEPS
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "═"*80)
print("  PART 2: SYSTEMATIC 1-FEATURE BUCKET SWEEPS")
print("═"*80)

def run_quantile_buckets(feat, q=4):
    """Split feature into q quantile buckets, return labeled results"""
    sub = df[df[feat].notna()].copy()
    try:
        sub['bucket'] = pd.qcut(sub[feat], q=q, duplicates='drop')
    except Exception:
        return []
    results = []
    for b in sorted(sub['bucket'].unique()):
        g = sub[sub['bucket']==b]
        results.append((str(b), bucket_stats(g)))
    return results

def run_categorical_buckets(feat, cats=None):
    sub = df[df[feat].notna()].copy()
    if cats is None:
        cats = sorted(sub[feat].unique())
    results = []
    for c in cats:
        g = sub[sub[feat]==c]
        results.append((f"{feat}={c}", bucket_stats(g)))
    return results

def run_custom_buckets(feat, cuts, labels):
    sub = df[df[feat].notna()].copy()
    results = []
    for label, mask_fn in zip(labels, cuts):
        g = sub[mask_fn(sub[feat])]
        results.append((label, bucket_stats(g)))
    return results

# 1. offset_pct
cuts_off = [
    (lambda x: x < -0.01,               "<-1%"),
    (lambda x: (x>=-0.01)&(x<0),        "[-1%,0%)"),
    (lambda x: (x>=0)&(x<0.01),         "[0%,1%)"),
    (lambda x: (x>=0.01)&(x<0.03),      "[1%,3%)"),
    (lambda x: x>=0.03,                  ">=3%"),
]
res = [(lbl, bucket_stats(df[df['offset_pct'].notna()][fn(df[df['offset_pct'].notna()]['offset_pct'])]))
       for fn,lbl in [(c[0],c[1]) for c in cuts_off]]
# rebuild properly
off_res = []
sub_off = df[df['offset_pct'].notna()]
for fn, lbl in [(lambda x: x < -0.01, "<-1%"),
                (lambda x: (x>=-0.01)&(x<0), "[-1%,0%)"),
                (lambda x: (x>=0)&(x<0.01), "[0%,1%)"),
                (lambda x: (x>=0.01)&(x<0.03), "[1%,3%)"),
                (lambda x: x>=0.03, ">=3%")]:
    off_res.append((lbl, bucket_stats(sub_off[fn(sub_off['offset_pct'])])))
print_bucket_table(off_res, "1. offset_pct [(strike-spot)/spot]")

# 2. ema_stack_bias
print_bucket_table(run_categorical_buckets('ema_stack_bias', [-1,0,1]), "2. ema_stack_bias")

# 3. composite_trend
print_bucket_table(run_quantile_buckets('composite_trend', q=5), "3. composite_trend (quintile buckets)")

# 4. composite_rev
print_bucket_table(run_quantile_buckets('composite_rev', q=5), "4. composite_rev (quintile buckets)")

# 5. stoch_k
sub_sk = df[df['stoch_k'].notna()]
sk_res = []
for fn, lbl in [(lambda x: x < 20, "<20"),
                (lambda x: (x>=20)&(x<40), "[20,40)"),
                (lambda x: (x>=40)&(x<60), "[40,60)"),
                (lambda x: (x>=60)&(x<80), "[60,80)"),
                (lambda x: x>=80, ">=80")]:
    sk_res.append((lbl, bucket_stats(sub_sk[fn(sub_sk['stoch_k'])])))
print_bucket_table(sk_res, "5. stoch_k")

# 6. hmm_vol_state
print_bucket_table(run_categorical_buckets('hmm_vol_state', [0,1]), "6. hmm_vol_state")

# 7. rvol_1h
if 'rvol_1h' in df.columns:
    print_bucket_table(run_quantile_buckets('rvol_1h', q=5), "7. rvol_1h (quintile)")

# 8. vol_score
print_bucket_table(run_categorical_buckets('vol_score', [-1,0,1]), "8. vol_score")

# 9. chg_1h
sub_c1h = df[df['chg_1h'].notna()]
c1h_res = []
for fn, lbl in [(lambda x: x < -0.005, "<-0.5%"),
                (lambda x: (x>=-0.005)&(x<-0.001), "[-0.5%,-0.1%)"),
                (lambda x: (x>=-0.001)&(x<0.001), "[-0.1%,0.1%)"),
                (lambda x: (x>=0.001)&(x<0.005), "[0.1%,0.5%)"),
                (lambda x: x>=0.005, ">=0.5%")]:
    c1h_res.append((lbl, bucket_stats(sub_c1h[fn(sub_c1h['chg_1h'])])))
print_bucket_table(c1h_res, "9. chg_1h")

# 10. chg_30m
sub_c30m = df[df['chg_30m'].notna()]
c30m_res = []
for fn, lbl in [(lambda x: x < -0.005, "<-0.5%"),
                (lambda x: (x>=-0.005)&(x<-0.001), "[-0.5%,-0.1%)"),
                (lambda x: (x>=-0.001)&(x<0.001), "[-0.1%,0.1%)"),
                (lambda x: (x>=0.001)&(x<0.005), "[0.1%,0.5%)"),
                (lambda x: x>=0.005, ">=0.5%")]:
    c30m_res.append((lbl, bucket_stats(sub_c30m[fn(sub_c30m['chg_30m'])])))
print_bucket_table(c30m_res, "10. chg_30m")

# 11. chg_10m
sub_c10m = df[df['chg_10m'].notna()]
c10m_res = []
for fn, lbl in [(lambda x: x < -0.005, "<-0.5%"),
                (lambda x: (x>=-0.005)&(x<-0.001), "[-0.5%,-0.1%)"),
                (lambda x: (x>=-0.001)&(x<0.001), "[-0.1%,0.1%)"),
                (lambda x: (x>=0.001)&(x<0.005), "[0.1%,0.5%)"),
                (lambda x: x>=0.005, ">=0.5%")]:
    c10m_res.append((lbl, bucket_stats(sub_c10m[fn(sub_c10m['chg_10m'])])))
print_bucket_table(c10m_res, "11. chg_10m")

# 12. rsi_1h
sub_rsi1h = df[df['rsi_1h'].notna()]
rsi1h_res = []
for fn, lbl in [(lambda x: x < 30, "<30"),
                (lambda x: (x>=30)&(x<50), "[30,50)"),
                (lambda x: (x>=50)&(x<70), "[50,70)"),
                (lambda x: x>=70, ">=70")]:
    rsi1h_res.append((lbl, bucket_stats(sub_rsi1h[fn(sub_rsi1h['rsi_1h'])])))
print_bucket_table(rsi1h_res, "12. rsi_1h")

# 13. rsi_4h
sub_rsi4h = df[df['rsi_4h'].notna()]
rsi4h_res = []
for fn, lbl in [(lambda x: x < 30, "<30"),
                (lambda x: (x>=30)&(x<50), "[30,50)"),
                (lambda x: (x>=50)&(x<70), "[50,70)"),
                (lambda x: x>=70, ">=70")]:
    rsi4h_res.append((lbl, bucket_stats(sub_rsi4h[fn(sub_rsi4h['rsi_4h'])])))
print_bucket_table(rsi4h_res, "13. rsi_4h")

# 14. macd_hist_1h
print_bucket_table(run_quantile_buckets('macd_hist_1h', q=5), "14. macd_hist_1h (quintile)")

# 15. obi_score
print_bucket_table(run_categorical_buckets('obi_score', [-1,0,1]), "15. obi_score")

# 16. liq_score
print_bucket_table(run_quantile_buckets('liq_score', q=5), "16. liq_score (quintile)")

# 17. liq_bias  (many continuous-ish values — quantile)
print_bucket_table(run_quantile_buckets('liq_bias', q=5), "17. liq_bias (quintile)")

# 18. funding_bias
print_bucket_table(run_categorical_buckets('funding_bias', [-1,0,1]), "18. funding_bias")

# 19. bp_1h
if 'bp_1h' in df.columns:
    print_bucket_table(run_quantile_buckets('bp_1h', q=5), "19. bp_1h (quintile)")

# 20. adx_1h
sub_adx1h = df[df['adx_1h'].notna()]
adx1h_res = []
for fn, lbl in [(lambda x: x < 20, "<20"),
                (lambda x: (x>=20)&(x<30), "[20,30)"),
                (lambda x: (x>=30)&(x<40), "[30,40)"),
                (lambda x: (x>=40)&(x<60), "[40,60)"),
                (lambda x: x>=60, ">=60")]:
    adx1h_res.append((lbl, bucket_stats(sub_adx1h[fn(sub_adx1h['adx_1h'])])))
print_bucket_table(adx1h_res, "20. adx_1h")

# 21. dir_15m
print_bucket_table(run_categorical_buckets('dir_15m', [-1,1]), "21. dir_15m")

# 22. vpin_score
print_bucket_table(run_categorical_buckets('vpin_score', [-1,0,1]), "22. vpin_score")

# 23. body_15m
if 'body_15m' in df.columns:
    print_bucket_table(run_quantile_buckets('body_15m', q=5), "23. body_15m (quintile)")

# 24. p_up_v2
if 'p_up_v2' in df.columns:
    print_bucket_table(run_quantile_buckets('p_up_v2', q=5), "24. p_up_v2 (quintile)")

# 25. tau_minutes
sub_tau = df[df['tau_minutes'].notna()]
tau_res = []
for fn, lbl in [(lambda x: x < 15, "<15m"),
                (lambda x: (x>=15)&(x<30), "[15,30)"),
                (lambda x: (x>=30)&(x<45), "[30,45)"),
                (lambda x: x>=45, ">=45m")]:
    tau_res.append((lbl, bucket_stats(sub_tau[fn(sub_tau['tau_minutes'])])))
print_bucket_table(tau_res, "25. tau_minutes")

# Additional features found in archive worth sweeping
for feat in ['stoch_k_1h','stoch_k_4h','macd_hist_4h','adx_4h','ema_stretch_score',
             'vwap_stretch_score','composite_p_up','confirmation_score','no_score',
             'chg_5m','p_gbdt','pm_drift_5m','squeeze_1h','stoch_k_5m','stoch_k_15m']:
    if feat in df.columns and df[feat].notna().sum() >= 200:
        try:
            print_bucket_table(run_quantile_buckets(feat, q=5), f"EXTRA: {feat} (quintile)")
        except Exception as e:
            print(f"  SKIP {feat}: {e}")


# ═══════════════════════════════════════════════════════════════
# PART 3: KEY 2-WAY INTERACTIONS
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "═"*80)
print("  PART 3: KEY 2-WAY INTERACTIONS")
print("═"*80)

def offset_bucket(s):
    if pd.isna(s): return np.nan
    if s < -0.01: return "<-1%"
    if s < 0:     return "[-1%,0%)"
    if s < 0.01:  return "[0%,1%)"
    if s < 0.03:  return "[1%,3%)"
    return ">=3%"

def chg1h_bucket(s):
    if pd.isna(s): return np.nan
    if s < -0.005: return "<-0.5%"
    if s < -0.001: return "[-0.5%,-0.1%)"
    if s < 0.001:  return "[-0.1%,0.1%)"
    if s < 0.005:  return "[0.1%,0.5%)"
    return ">=0.5%"

def stochk_bucket(s):
    if pd.isna(s): return np.nan
    if s < 20: return "<20"
    if s < 40: return "[20,40)"
    if s < 60: return "[40,60)"
    if s < 80: return "[60,80)"
    return ">=80"

df['off_b']   = df['offset_pct'].apply(offset_bucket)
df['c1h_b']   = df['chg_1h'].apply(chg1h_bucket)
df['sk_b']    = df['stoch_k'].apply(stochk_bucket)

OFF_ORDER  = ["<-1%","[-1%,0%)","[0%,1%)","[1%,3%)",">=3%"]
C1H_ORDER  = ["<-0.5%","[-0.5%,-0.1%)","[-0.1%,0.1%)","[0.1%,0.5%)",">=0.5%"]
SK_ORDER   = ["<20","[20,40)","[40,60)","[60,80)",">=80"]
EMA_ORDER  = [-1.0, 0.0, 1.0]
HMM_ORDER  = [0.0, 1.0]

def print_2way(df_in, col1, col2, order1, order2, title, sig_n=50, sig_edge=0.08):
    print(f"\n{'─'*80}")
    print(f"  2-WAY: {title}")
    print(f"{'─'*80}")
    # header
    col2_vals = [v for v in order2 if v in df_in[col2].unique()]
    hdr = f"  {col1[:18]:<18}"
    for v in col2_vals:
        hdr += f" | {str(v)[:14]:^30}"
    print(hdr)
    sub_col2_label = f"  {'':18}" + "".join([f" | {'n':>4} Y_edge  N_edge       " for _ in col2_vals])
    print(sub_col2_label)
    print("  " + "─"*78)
    for v1 in order1:
        if v1 not in df_in[col1].unique(): continue
        row = f"  {str(v1)[:18]:<18}"
        for v2 in col2_vals:
            g = df_in[(df_in[col1]==v1)&(df_in[col2]==v2)]
            s = bucket_stats(g)
            if s['n'] == 0:
                row += f" | {'—':>4} {'—':>7} {'—':>7}      "
            else:
                flag = ""
                if s['n'] >= sig_n and (abs(s['yes_edge']) >= sig_edge or abs(s['no_edge']) >= sig_edge):
                    flag = "★"
                row += f" | {s['n']:>4} {s['yes_edge']:>+7.3f} {s['no_edge']:>+7.3f} {flag:<5}"
        print(row)

# A. offset_pct × ema_stack_bias
print_2way(df[df['off_b'].notna()&df['ema_stack_bias'].notna()],
           'off_b','ema_stack_bias', OFF_ORDER, EMA_ORDER,
           "offset_pct_bucket × ema_stack_bias")

# B. offset_pct × hmm_vol_state
sub_hmm = df[df['off_b'].notna()&df['hmm_vol_state'].notna()]
print_2way(sub_hmm, 'off_b','hmm_vol_state', OFF_ORDER, HMM_ORDER,
           "offset_pct_bucket × hmm_vol_state")

# C. offset_pct × chg_1h
print_2way(df[df['off_b'].notna()&df['c1h_b'].notna()],
           'off_b','c1h_b', OFF_ORDER, C1H_ORDER,
           "offset_pct_bucket × chg_1h_bucket")

# D. hmm_vol_state × chg_1h
sub_hc = df[df['hmm_vol_state'].notna()&df['c1h_b'].notna()]
print_2way(sub_hc, 'hmm_vol_state','c1h_b', HMM_ORDER, C1H_ORDER,
           "hmm_vol_state × chg_1h_bucket")

# E. ema_stack_bias × stoch_k
print_2way(df[df['ema_stack_bias'].notna()&df['sk_b'].notna()],
           'ema_stack_bias','sk_b', EMA_ORDER, SK_ORDER,
           "ema_stack_bias × stoch_k_bucket")


# ═══════════════════════════════════════════════════════════════
# PART 4: 3-WAY  hmm_vol_state × chg_1h × offset_pct
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "═"*80)
print("  PART 4: 3-WAY  hmm_vol_state × chg_1h_bucket × offset_pct_bucket")
print("═"*80)

sub3 = df[df['hmm_vol_state'].notna() & df['c1h_b'].notna() & df['off_b'].notna()].copy()

DIFF_THRESH = 0.05
notable = []

print(f"\n  {'HMM':>4} {'chg_1h':<16} {'offset':<12} {'n':>5} | {'Y_WR':>6} {'Y_edge':>8} | {'N_WR':>6} {'N_edge':>8}")
print("  " + "─"*70)
for hmm in HMM_ORDER:
    for c1h in C1H_ORDER:
        for off in OFF_ORDER:
            g = sub3[(sub3['hmm_vol_state']==hmm)&(sub3['c1h_b']==c1h)&(sub3['off_b']==off)]
            s = bucket_stats(g)
            if s['n'] < 30: continue
            flag = "★" if (abs(s['yes_edge'])>0.06 or abs(s['no_edge'])>0.06) else ""
            print(f"  R{int(hmm)}  {c1h:<16} {off:<12} {s['n']:>5,} | "
                  f"{s['yes_wr']:>6.3f} {s['yes_edge']:>+8.3f} | {s['no_wr']:>6.3f} {s['no_edge']:>+8.3f} {flag}")
            if flag:
                notable.append((hmm, c1h, off, s))

# Cross-regime comparison
print(f"\n  === R0 vs R1 DIFFERENCES (same chg_1h × offset, |edge diff| > 5pp) ===")
for c1h in C1H_ORDER:
    for off in OFF_ORDER:
        r0 = bucket_stats(sub3[(sub3['hmm_vol_state']==0.0)&(sub3['c1h_b']==c1h)&(sub3['off_b']==off)])
        r1 = bucket_stats(sub3[(sub3['hmm_vol_state']==1.0)&(sub3['c1h_b']==c1h)&(sub3['off_b']==off)])
        if r0['n'] < 30 or r1['n'] < 30: continue
        dy = r0['yes_edge'] - r1['yes_edge']
        dn = r0['no_edge']  - r1['no_edge']
        if abs(dy) > DIFF_THRESH or abs(dn) > DIFF_THRESH:
            print(f"  chg={c1h:<16} off={off:<12}  "
                  f"R0(n={r0['n']}) Y_edge={r0['yes_edge']:+.3f} N_edge={r0['no_edge']:+.3f}  |  "
                  f"R1(n={r1['n']}) Y_edge={r1['yes_edge']:+.3f} N_edge={r1['no_edge']:+.3f}  "
                  f"ΔY={dy:+.3f} ΔN={dn:+.3f} ★")


# ═══════════════════════════════════════════════════════════════
# PART 5: SUMMARY RANKINGS
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "═"*80)
print("  PART 5: SUMMARY RANKINGS (n>=100, |edge|>=5pp)")
print("═"*80)

# Collect all single-feature bucket results
all_yes = []  # (condition, n, yes_wr, yes_bkev, yes_edge)
all_no  = []  # (condition, n, no_wr, no_bkev, no_edge)

def collect(results, feature):
    for label, s in results:
        if s['n'] < 100: continue
        all_yes.append((f"{feature} [{label}]", s['n'], s['yes_wr'], s['yes_bkev'], s['yes_edge']))
        all_no.append( (f"{feature} [{label}]", s['n'], s['no_wr'],  s['no_bkev'],  s['no_edge']))

# Rebuild all bucket results for collection
collect(off_res, "offset_pct")
collect(run_categorical_buckets('ema_stack_bias', [-1,0,1]), "ema_stack_bias")
collect(run_quantile_buckets('composite_trend', q=5), "composite_trend")
collect(run_quantile_buckets('composite_rev', q=5), "composite_rev")
collect(sk_res, "stoch_k")
collect(run_categorical_buckets('hmm_vol_state', [0,1]), "hmm_vol_state")
if 'rvol_1h' in df.columns: collect(run_quantile_buckets('rvol_1h', q=5), "rvol_1h")
collect(run_categorical_buckets('vol_score', [-1,0,1]), "vol_score")
collect(c1h_res, "chg_1h")
collect(c30m_res, "chg_30m")
collect(c10m_res, "chg_10m")
collect(rsi1h_res, "rsi_1h")
collect(rsi4h_res, "rsi_4h")
collect(run_quantile_buckets('macd_hist_1h', q=5), "macd_hist_1h")
collect(run_categorical_buckets('obi_score', [-1,0,1]), "obi_score")
collect(run_quantile_buckets('liq_score', q=5), "liq_score")
collect(run_quantile_buckets('liq_bias', q=5), "liq_bias")
collect(run_categorical_buckets('funding_bias', [-1,0,1]), "funding_bias")
if 'bp_1h' in df.columns: collect(run_quantile_buckets('bp_1h', q=5), "bp_1h")
collect(adx1h_res, "adx_1h")
collect(run_categorical_buckets('dir_15m', [-1,1]), "dir_15m")
collect(run_categorical_buckets('vpin_score', [-1,0,1]), "vpin_score")
if 'body_15m' in df.columns: collect(run_quantile_buckets('body_15m', q=5), "body_15m")
if 'p_up_v2' in df.columns: collect(run_quantile_buckets('p_up_v2', q=5), "p_up_v2")
collect(tau_res, "tau_minutes")
for feat in ['stoch_k_1h','stoch_k_4h','macd_hist_4h','adx_4h','ema_stretch_score',
             'vwap_stretch_score','composite_p_up','confirmation_score','no_score',
             'chg_5m','p_gbdt','pm_drift_5m','squeeze_1h','stoch_k_5m','stoch_k_15m']:
    if feat in df.columns and df[feat].notna().sum() >= 200:
        try:
            collect(run_quantile_buckets(feat, q=5), feat)
        except: pass

def print_rank(lst, sort_key, title, top_n=10):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    lst_sorted = sorted(lst, key=sort_key, reverse=(sort_key.__name__!='<lambda>'))
    # handle ascending vs descending
    lst_sorted = sorted(lst, key=sort_key)
    if 'positive' in title.lower() or 'best' in title.lower():
        lst_sorted = sorted(lst, key=sort_key, reverse=True)
    else:
        lst_sorted = sorted(lst, key=sort_key, reverse=False)
    count = 0
    for row in lst_sorted:
        if count >= top_n: break
        cond, n, wr, bkev, edge = row
        if abs(edge) < 0.05: continue
        sig = "★★★" if abs(edge)>0.15 else "★★" if abs(edge)>0.10 else "★"
        print(f"  {sig} {cond[:45]:<45}  n={n:>5,}  WR={wr:.3f}  bkev={bkev:.3f}  edge={edge:+.3f}")
        count += 1

print_rank(all_yes, lambda r: r[4], "TOP 10 POSITIVE EDGE — YES BETS (best conditions to BET YES)")
print_rank(all_yes, lambda r: r[4], "TOP 10 NEGATIVE EDGE — YES BETS (worst conditions / gate candidates)")
print_rank(all_no,  lambda r: r[4], "TOP 10 POSITIVE EDGE — NO BETS (best conditions to BET NO)")
print_rank(all_no,  lambda r: r[4], "TOP 10 NEGATIVE EDGE — NO BETS (worst conditions / gate candidates)")

# Fix: sort correctly
print("\n\n=== CORRECTED RANKINGS ===")

yes_sig = [(c,n,wr,bk,e) for c,n,wr,bk,e in all_yes if abs(e)>=0.05]
no_sig  = [(c,n,wr,bk,e) for c,n,wr,bk,e in all_no  if abs(e)>=0.05]

print(f"\n{'─'*70}")
print(f"  TOP 10 POSITIVE EDGE — YES BETS (highest YES_edge)")
print(f"{'─'*70}")
for c,n,wr,bk,e in sorted(yes_sig, key=lambda r: r[4], reverse=True)[:10]:
    sig = "★★★" if e>0.15 else "★★" if e>0.10 else "★"
    print(f"  {sig} {c[:50]:<50}  n={n:>5,}  WR={wr:.3f}  bkev={bk:.3f}  edge={e:+.3f}")

print(f"\n{'─'*70}")
print(f"  TOP 10 NEGATIVE EDGE — YES BETS (lowest YES_edge = gate candidates)")
print(f"{'─'*70}")
for c,n,wr,bk,e in sorted(yes_sig, key=lambda r: r[4])[:10]:
    sig = "★★★" if e<-0.15 else "★★" if e<-0.10 else "★"
    print(f"  {sig} {c[:50]:<50}  n={n:>5,}  WR={wr:.3f}  bkev={bk:.3f}  edge={e:+.3f}")

print(f"\n{'─'*70}")
print(f"  TOP 10 POSITIVE EDGE — NO BETS (highest NO_edge)")
print(f"{'─'*70}")
for c,n,wr,bk,e in sorted(no_sig, key=lambda r: r[4], reverse=True)[:10]:
    sig = "★★★" if e>0.15 else "★★" if e>0.10 else "★"
    print(f"  {sig} {c[:50]:<50}  n={n:>5,}  WR={wr:.3f}  bkev={bk:.3f}  edge={e:+.3f}")

print(f"\n{'─'*70}")
print(f"  TOP 10 NEGATIVE EDGE — NO BETS (lowest NO_edge = gate candidates)")
print(f"{'─'*70}")
for c,n,wr,bk,e in sorted(no_sig, key=lambda r: r[4])[:10]:
    sig = "★★★" if e<-0.15 else "★★" if e<-0.10 else "★"
    print(f"  {sig} {c[:50]:<50}  n={n:>5,}  WR={wr:.3f}  bkev={bk:.3f}  edge={e:+.3f}")

print("\n\n=== ANALYSIS COMPLETE ===")
