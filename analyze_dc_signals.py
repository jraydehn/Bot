"""
analyze_dc_signals.py — Perm + WF + MCPT + independence check on DC swing signals.
"""
import numpy as np
import pandas as pd
from scipy import stats as ss

sa = pd.read_parquet('results/btc_scan_archive_dc.parquet')
sa = sa[sa['resolved_yes'].notna()].copy()
sa['logged_at']       = pd.to_datetime(sa['logged_at'],       errors='coerce', utc=True)
sa['p_market']        = pd.to_numeric(sa['p_market'],         errors='coerce')
sa['composite_trend'] = pd.to_numeric(sa.get('composite_trend', np.nan), errors='coerce')
sa['ema_stack_bias']  = pd.to_numeric(sa.get('ema_stack_bias', np.nan),  errors='coerce')

FEE=0.07; FLAT=10.0; N=500; TF=0.75
YES_LO, YES_HI = 0.20, 0.65

def pyes(pm, won, keep=None):
    k = keep if keep is not None else np.ones(len(pm), dtype=bool)
    return float((np.where(won[k], (1-pm[k]), -pm[k]) * (1-FEE)*FLAT).sum())

def shuf_bins(won, bins, rng):
    wp = won.copy()
    for b in range(11):
        idx = np.where(bins==b)[0]
        if len(idx)>1: v=wp[idx]; rng.shuffle(v); wp[idx]=v
    return wp

def shuf_oos(won, mask, bins, rng):
    wp = won.copy()
    for b in range(11):
        idx = np.where((bins==b)&mask)[0]
        if len(idx)>1: v=wp[idx]; rng.shuffle(v); wp[idx]=v
    return wp

def validate(label, sub, block_ser):
    pm   = sub['p_market'].astype(float).values
    won  = (sub['resolved_yes']==1).astype(float).values
    blk  = block_ser.reindex(sub.index).fillna(False).values
    keep = ~blk

    real_d  = pyes(pm, won, keep) - pyes(pm, won)
    wr_blk  = won[blk].mean()  if blk.sum() else float('nan')
    bk_blk  = pm[blk].mean()   if blk.sum() else float('nan')
    edge_blk= wr_blk - bk_blk
    n_exp   = sub['close_ts'].nunique() if 'close_ts' in sub.columns else '?'

    bp = ss.binomtest(int(won[blk].sum()), int(blk.sum()), float(bk_blk),
                      alternative='less').pvalue if blk.sum()>5 else 1.0

    bins = np.floor(pm*10).astype(int)
    rng  = np.random.default_rng(0); pd_ = []
    for _ in range(N):
        wp = shuf_bins(won, bins, rng)
        pd_.append(pyes(pm, wp, keep) - pyes(pm, wp))
    arr    = np.array(pd_)
    p_perm = float((arr >= real_d).mean())

    # WF
    sub_s = sub.sort_values('logged_at').reset_index(drop=True)
    sp    = int(len(sub_s) * TF)
    wf_str = 'WF: degenerate'
    if sp >= 20 and sp < len(sub_s)-20:
        pm_s   = sub_s['p_market'].astype(float).values
        won_s  = (sub_s['resolved_yes']==1).astype(float).values
        blk_s  = block_ser.reindex(sub_s.index).fillna(False).values
        bins_s = np.floor(pm_s*10).astype(int)
        ti     = np.arange(sp, len(sub_s))
        tm     = np.zeros(len(sub_s), dtype=bool); tm[sp:] = True
        koo    = ~blk_s[ti]
        oos_real = pyes(pm_s[ti], won_s[ti], koo) - pyes(pm_s[ti], won_s[ti])
        n_blk_oos= blk_s[ti].sum()
        wr_oos   = won_s[ti][blk_s[ti]].mean() if n_blk_oos else float('nan')
        e_oos    = wr_oos - pm_s[ti][blk_s[ti]].mean() if n_blk_oos else float('nan')
        rng2 = np.random.default_rng(99); po_ = []
        for _ in range(N):
            wp = shuf_oos(won_s, tm, bins_s, rng2)
            po_.append(pyes(pm_s[ti], wp[ti], koo) - pyes(pm_s[ti], wp[ti]))
        arr_o = np.array(po_)
        p_wf  = float((arr_o >= oos_real).mean())
        split_ts = sub_s['logged_at'].iloc[sp]
        sig_wf = 'SIG' if p_wf < 0.05 else 'FAIL'
        wf_str = (f'WF->{split_ts.date()} n_blk={n_blk_oos} '
                  f'WR={wr_oos:.1%} edge={e_oos:+.1%} p={p_wf:.3f} {sig_wf}')

    sig = 'SIG' if (p_perm < 0.05 or bp < 0.05) else 'FAIL'
    print(f'\n  {label}')
    print(f'    blocked n={blk.sum():,} ({n_exp} exp)  WR={wr_blk:.1%}  bkev={bk_blk:.1%}  '
          f'edge={edge_blk:+.1%}  delta=${real_d:+,.0f}')
    print(f'    Perm p={p_perm:.3f}  Binom p={bp:.4f}  [{sig}]')
    print(f'    {wf_str}')


# ── 1. Each lookback ──────────────────────────────────────────────────────────
print('='*70)
print('STRIKE-ABOVE-SWING-HIGH  Perm + WF + MCPT + Independence')
print('='*70)
print('\n-- Per-lookback validation --')
for lb in [60, 240, 1440]:
    col = f'strike_above_high_{lb}'
    if col not in sa.columns: continue
    sub = sa[(sa['p_market']>=YES_LO)&(sa['p_market']<YES_HI)&sa[col].notna()].copy()
    validate(f'lb={lb}m', sub, pd.Series(sub[col].astype(bool), index=sub.index))


# ── 2. MCPT: 3-lookback sweep ─────────────────────────────────────────────────
print('\n-- MCPT: 3-lookback sweep --')
LBS     = [60, 240, 1440]
sub_all = sa[(sa['p_market']>=YES_LO)&(sa['p_market']<YES_HI)].copy()
sub_all = sub_all.dropna(subset=[f'strike_above_high_{lb}' for lb in LBS]+['p_market'])
pm_m    = sub_all['p_market'].astype(float).values
won_m   = (sub_all['resolved_yes']==1).astype(float).values
bins_m  = np.floor(pm_m*10).astype(int)

def best_delta(pm, won):
    base = pyes(pm, won)
    best, best_lb = -np.inf, -1
    for lb in LBS:
        blk = sub_all[f'strike_above_high_{lb}'].astype(bool).values
        d   = pyes(pm, won, ~blk) - base
        if d > best: best, best_lb = d, lb
    return best, best_lb

real_best, real_lb = best_delta(pm_m, won_m)
print(f'  Real best: lb={real_lb}m  delta=${real_best:+,.0f}')
rng = np.random.default_rng(0); perm_bests = []
print(f'  Running {N} perms...', end=' ', flush=True)
for i in range(N):
    wp = shuf_bins(won_m, bins_m, rng)
    perm_bests.append(best_delta(pm_m, wp)[0])
    if (i+1) % 100 == 0: print(i+1, end=' ', flush=True)
print('done.')
parr   = np.array(perm_bests)
p_mcpt = float((parr >= real_best).mean())
sig_mc = 'SIG' if p_mcpt < 0.05 else 'FAIL'
print(f'  MCPT: p={p_mcpt:.3f}  null_p50=${np.percentile(parr,50):+,.0f}  [{sig_mc}]')
print(f'  Interpretation: best lookback ({real_lb}m) {"survives" if p_mcpt<0.05 else "does NOT survive"} '
      f'parameter search over {len(LBS)} candidates.')


# ── 3. Independence from composite_trend / ema_stack ─────────────────────────
print('\n-- Cross-cut: independence from composite_trend / ema_stack_bias --')
lb_use = 240
col    = f'strike_above_high_{lb_use}'
sub_x  = sa[(sa['p_market']>=YES_LO)&(sa['p_market']<YES_HI)
             &sa[col].notna()
             &sa['composite_trend'].notna()
             &sa['ema_stack_bias'].notna()].copy()

print(f'  (using lb={lb_use}m)')
print()
for ct_lo, ct_hi, ct_lbl in [(-6,-2,'c_trend<=-2'), (-1,1,'c_trend -1..+1'), (2,6,'c_trend>=2')]:
    for above in [True, False]:
        g  = sub_x[(sub_x['composite_trend']>=ct_lo)&(sub_x['composite_trend']<=ct_hi)
                   &(sub_x[col]==above)]
        if len(g) < 30: continue
        pm_ = g['p_market'].astype(float).values
        w_  = (g['resolved_yes']==1).astype(float).values
        wr_ = w_.mean(); bk_ = pm_.mean()
        abv = 'ABOVE_high' if above else 'below_high'
        print(f'  {ct_lbl:20s}  {abv}  n={len(g):5,}  '
              f'WR={wr_:.1%}  bkev={bk_:.1%}  edge={wr_-bk_:+.1%}')
    print()

print('\n-- dist_high buckets: YES edge by distance from spot to swing high --')
lb_use = 240
dh_col = f'dist_high_{lb_use}'
sub_d  = sa[(sa['p_market']>=YES_LO)&(sa['p_market']<YES_HI)&sa[dh_col].notna()].copy()
for lo, hi in [(-5,-2),(-2,-1),(-1,-0.5),(-0.5,0),(0,1),(1,2),(2,5)]:
    g = sub_d[(sub_d[dh_col]>=lo)&(sub_d[dh_col]<hi)]
    if len(g)<30: continue
    pm_=g['p_market'].astype(float).values; w_=(g['resolved_yes']==1).astype(float).values
    wr_=w_.mean(); bk_=pm_.mean()
    n_exp = g['close_ts'].nunique() if 'close_ts' in g.columns else '?'
    print(f'  dist_high [{lo:+.1f},{hi:+.1f}%): n={len(g):6,} ({n_exp} exp)  '
          f'WR={wr_:.1%}  bkev={bk_:.1%}  edge={wr_-bk_:+.1%}')

print('\nDone.')
