"""
validate_no_otm_signals.py  (scratch analysis — no live impact)

Validate the conditional NO-on-OTM signals on the repaired archive:
  adx_1h<15, ls_long_pct<=65, composite_trend non-flat (|trend|>=1).
Deduped to unique contracts (kills ~26x scan multi-count). Pre-specified
rules (no in-sample fitting). Reports per-week EV, MCPT p (label shuffle)
and a time-block MCPT (shuffle whole expiry-hour outcome blocks) to respect
that contracts settling the same hour share one BTC move.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
FEE = 0.07
rng = np.random.default_rng(42)

a = pd.read_csv("results/btc_scan_archive.csv", low_memory=False)
a['dt']  = pd.to_datetime(a['logged_at'], errors='coerce', utc=True, format='mixed')
a['cts'] = pd.to_datetime(a['close_ts'],  errors='coerce', utc=True, format='mixed')
for c in ['p_market','resolved_yes','ls_long_pct','adx_1h','composite_trend','tau_minutes']:
    a[c] = pd.to_numeric(a[c], errors='coerce')
a = a.dropna(subset=['p_market','resolved_yes','cts'])
a = a[(a['p_market']>0)&(a['p_market']<1)]

# dedupe to unique contract (median pm, first outcome/features/close-hour)
d = (a.groupby('contract_ticker')
       .agg(pm=('p_market','median'), ry=('resolved_yes','first'),
            ls=('ls_long_pct','median'), adx=('adx_1h','median'),
            trend=('composite_trend','median'), cts=('cts','first'),
            wk=('dt', lambda s: s.dropna().dt.isocalendar().week.iloc[0] if s.notna().any() else np.nan))
       .reset_index())
d['ry'] = d['ry'].round().astype(int)
d['ev_no'] = (d['pm'] - d['ry']) - FEE*np.minimum(d['pm'],1-d['pm'])
d['hour'] = d['cts'].dt.floor('h')

otm = d[(d['pm']>=0.10)&(d['pm']<0.40)].copy()
print(f"unique OTM contracts pm[0.10,0.40): {len(otm)}  "
      f"unique expiry hours: {otm['hour'].nunique()}")
base = otm['ev_no'].mean()
print(f"baseline unconditional OTM-NO  EV/ct={base:+.4f}  total=${otm['ev_no'].sum():+.0f}  n={len(otm)}")

def mcpt(sub_mask, n=3000):
    """Label-shuffle MCPT: is selected subset's mean EV > random subset of same size?"""
    sel = otm.loc[sub_mask,'ev_no']; k=len(sel); obs=sel.mean()
    ev = otm['ev_no'].values
    cnt=0
    for _ in range(n):
        cnt += rng.choice(ev,k,replace=False).mean() >= obs
    return obs, (cnt+1)/(n+1)

def mcpt_block(sub_mask, n=3000):
    """Time-block MCPT: shuffle outcomes among contracts within each expiry hour
    is destroyed; instead permute hour-level: reassign condition labels by whole
    hours to respect intra-hour outcome correlation."""
    hours = otm['hour'].values
    cond  = sub_mask.values
    ev    = otm['ev_no'].values
    obs   = ev[cond].mean(); k_hours = pd.Series(hours[cond]).nunique()
    uh = otm['hour'].unique()
    # fraction of condition contracts per hour, then permute which hours are "on"
    by_hour = otm.assign(c=cond).groupby('hour')
    hour_ev = by_hour['ev_no'].apply(list).to_dict()
    hour_con= by_hour['c'].apply(lambda s: s.values).to_dict()
    cnt=0
    for _ in range(n):
        perm = rng.permutation(uh)
        # pick hours until we cover ~k condition-contracts, take their condition-positions' ev
        picked=[]; need=cond.sum()
        for h in perm:
            picked.extend(hour_ev[h]);
            if len(picked)>=need: break
        cnt += np.mean(picked[:need]) >= obs
    return obs,(cnt+1)/(n+1)

conds = {
    'adx<15'                : otm['adx']<15,
    'ls_long<=65'           : otm['ls']<=65,
    '|trend|>=1 (non-flat)' : otm['trend'].abs()>=1,
    'adx<15 & ls<=65'       : (otm['adx']<15)&(otm['ls']<=65),
    'adx<15 & |trend|>=1'   : (otm['adx']<15)&(otm['trend'].abs()>=1),
    'ls<=65 & |trend|>=1'   : (otm['ls']<=65)&(otm['trend'].abs()>=1),
    'all three'             : (otm['adx']<15)&(otm['ls']<=65)&(otm['trend'].abs()>=1),
}
print(f"\n{'condition':<24}{'n':>6}{'EV/ct':>9}{'vs_base':>9}{'total$':>9}{'wk_signs':>14}{'MCPT_p':>9}{'block_p':>9}")
print("-"*100)
for name,m in conds.items():
    sub=otm[m]
    if len(sub)<30:
        print(f"{name:<24}{len(sub):>6}  (too few)"); continue
    ev=sub['ev_no'].mean(); tot=sub['ev_no'].sum()
    wk=sub.groupby('wk')['ev_no'].mean()
    signs=''.join('+' if v>0 else '-' for _,v in wk.items())
    _,p   = mcpt(m)
    _,pb  = mcpt_block(m)
    print(f"{name:<24}{len(sub):>6}{ev:>+9.4f}{ev-base:>+9.4f}{tot:>+9.0f}{signs:>14}{p:>9.4f}{pb:>9.4f}")
print("-"*100)
print("wk_signs = EV sign per week wk21..wk25 (want all '+' for robustness)")
print("MCPT_p   = P(random same-size subset >= observed EV)   block_p = time-block variant")
