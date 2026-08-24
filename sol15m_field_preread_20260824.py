"""Pre-08-25-read SOL 15m field sweep (dashboard-faithful replication).
Arms: old/slope model x {flat, SW-stack kelly, xHdamp, dipRESC-retro};
architectures: model-DUAL, mkt-fav+OUtau frozen transfer, TRIPLE. Analysis only."""
import pandas as pd, numpy as np
from scipy.stats import norm

R = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results'
sh = pd.read_csv(f'{R}/paper_trades_sol15m.csv', low_memory=False)
sh['dt'] = pd.to_datetime(sh['logged_at'], errors='coerce', utc=True, format='mixed')
for c in ['p_market','p_sol_old','p_sol_slope','resolved_yes','sol_persist_score','slope120_stoch_k_15m',
          'stoch_cross_1h','stoch_k_1h','oi_chg_pct','offset_pct','z_drift_6h','vol_ratio_1h',
          'hurst_exponent_5m','pm_path_drift','pm_path_vr3','stoch_k_5m','ou_theta','tau_minutes']:
    sh[c] = pd.to_numeric(sh.get(c), errors='coerce')
try:
    bf = pd.read_csv(f'{R}/sol15m_pmpath_backfill.csv')
    sh = sh.merge(bf, on=['logged_at','contract_ticker'], how='left')
    for pc, bc in [('pm_path_drift','pm_path_drift_bf'), ('pm_path_vr3','pm_path_vr3_bf')]:
        sh[pc] = pd.to_numeric(sh[pc], errors='coerce').fillna(pd.to_numeric(sh[bc], errors='coerce'))
except Exception:
    pass
START = pd.Timestamp('2026-07-30 01:00', tz='UTC')
sh = sh[(sh['dt'] >= START) & sh['resolved_yes'].notna() & sh['p_market'].between(0.03, 0.97)].copy()
TKT = pd.Timestamp('2026-08-20 05:30', tz='UTC')
RESC_T = pd.Timestamp('2026-08-18 04:27', tz='UTC')
DIP_T = pd.Timestamp('2026-08-23 04:20', tz='UTC')

def build(col):
    s = sh.dropna(subset=[col]).copy()
    fee = 0.07*s['p_market']*(1-s['p_market'])
    ey = s[col]-s['p_market']-fee; en = s['p_market']-s[col]-fee
    s['side'] = np.where(ey >= en, 'yes', 'no'); s['edge'] = np.maximum(ey, en)
    q = s[s['edge'] >= 0.04].sort_values('dt').drop_duplicates('contract_ticker', keep='first').copy()
    tkc = np.where(q['side']=='yes', q['p_market'], 1-q['p_market'])
    q = q[~((tkc <= 0.20) & (q['dt'] >= TKT))].copy()
    cost = np.where(q['side']=='yes', q['p_market'], 1-q['p_market'])
    win = np.where(q['side']=='yes', q['resolved_yes']==1, q['resolved_yes']==0)
    feeq = 0.07*q['p_market']*(1-q['p_market'])
    q['pnl_flat'] = np.where(win, 100*(1-cost)/cost, -100) - (100/cost)*feeq
    m6 = q['markov_sol_6h'].astype(str); m4 = q['markov_sol_4h'].astype(str); m1 = q['markov_sol_1h'].astype(str)
    sc1 = q['stoch_cross_1h'].fillna(0.0); sk1 = q['stoch_k_1h'].fillna(50.0)
    oi = q['oi_chg_pct'].fillna(0.0); off = q['offset_pct'].fillna(0.0); zd6 = q['z_drift_6h']
    gy = ((m6=='Bull')&(sc1!=0)) | (m4=='Sideways') | ((m1=='Sideways')&(oi<0.0535))
    ry = ((m6=='Bull')&(sc1==0)) | ((m1=='Sideways')&(oi>=0.0535))
    gn = ((m6=='Bull')&(off>-0.006)) | ((m4=='Sideways')&(sk1<90.0))
    rn = ((m6=='Bull')&(off<=-0.006)) | ((m4=='Sideways')&(sk1>=90.0))
    mkv = np.where(q['side']=='yes', ~(gy&~ry), ~(gn&~rn))
    zd65 = np.where(q['side']=='no', ~(zd6<0.65).fillna(False), True)
    flip = (m1=='Sideways')&(oi>=0.0535)&(zd6<0.55).fillna(False)
    offok = np.where(q['side']=='yes', ~((off>=-10.0)&(off<0.0)&~flip), True)
    v2 = np.where(q['side']=='yes', q['sol_persist_score']>=3,
                  ~((q['p_market']>0.8) | (q['p_market'].between(0.5,0.65) & ~(q['slope120_stoch_k_15m']>=40))))
    dip = (q['stoch_k_5m']<=30).fillna(False)
    core = np.asarray((q['side']=='yes') & ~(q['sol_persist_score']>=3).fillna(False)
                      & (zd6<0.59).fillna(False) & (q['dt']>=RESC_T), bool)
    rescY = core & np.asarray((q['dt']<DIP_T) | dip, bool)
    rescR = core & np.asarray(dip, bool)
    path = np.where(q['side']=='no', ~((q['pm_path_drift']*q['pm_path_vr3'])>0).fillna(False), True)
    swA = ~((q['side']=='no')&(m1=='Sideways')&(q['p_market']>=0.70)&(sk1>=70))
    swB = ~((q['side']=='no')&(m1=='Sideways')&(m4=='Sideways')&(q['p_market']>=0.55))
    q['m_stack'] = (v2|rescY)&(mkv|rescY)&zd65&(offok|rescY)&path&swA&swB
    q['m_stackR'] = (v2|rescR)&(mkv|rescR)&zd65&(offok|rescR)&path&swA&swB
    fr = np.where(q['side']=='yes', (q[col]-q['p_market']-feeq)/(1-q['p_market']),
                  (q['p_market']-q[col]-feeq)/q['p_market'])
    stk = 2500.0*np.clip(fr, 0, 0.10)
    q['pnl_k'] = np.where(win, stk*(1-cost)/cost, -stk) - (stk/cost)*feeq
    q['hmul'] = np.clip((q['hurst_exponent_5m']-0.4)/0.2, 0.25, 1.0).fillna(1.0)
    return q

old = build('p_sol_old'); slp = build('p_sol_slope')
# mkt-fav+OUtau frozen transfer (SOL raw mkt-fav was -$4,063)
mf = sh.copy()
pmf = norm.cdf(1.8*norm.ppf(mf['p_market'].clip(0.01, 0.99)))
fee = 0.07*mf['p_market']*(1-mf['p_market'])
eyf = pmf-mf['p_market']-fee; enf = mf['p_market']-pmf-fee
mf['side'] = np.where(eyf>=enf, 'yes', 'no'); mf['edge'] = np.maximum(eyf, enf)
mq = mf[mf['edge']>=0.04].sort_values('dt').drop_duplicates('contract_ticker', keep='first').copy()
for lab, mm in [('mktfav RAW (SOL)', mq),
                ('mktfav+OUtau (SOL, frozen transfer)',
                 mq[~(((mq['ou_theta']<2.2243).fillna(False)) | ((mq['tau_minutes']<8.64).fillna(False)))])]:
    c = np.where(mm['side']=='yes', mm['p_market'], 1-mm['p_market'])
    w = np.where(mm['side']=='yes', mm['resolved_yes']==1, mm['resolved_yes']==0)
    f2 = 0.07*mm['p_market']*(1-mm['p_market'])
    mm = mm.copy(); mm['pnl'] = np.where(w, 100*(1-c)/c, -100) - (100/c)*f2
    if 'OUtau' in lab: mo = mm
    d = mm.set_index('dt')['pnl'].resample('D').sum(); cum = mm['pnl'].cumsum()
    print(f'{lab:42s} n={len(mm):4d} ${mm.pnl.sum():+8,.0f} S={d.mean()/d.std():5.2f} DD=${float((cum.cummax()-cum).max()):7,.0f}')
print()
R7 = pd.Timestamp('2026-08-17', tz='UTC')
def score(parts, label):
    rows = []
    for df, mask, wexpr in parts:
        g = df[mask] if mask is not None else df
        p = wexpr(g)
        rows.append(pd.DataFrame({'dt': g['dt'], 'pnl': p}))
    P = pd.concat(rows).sort_values('dt')
    out = []
    for lab2, PP in [('full', P), ('7d', P[P['dt']>=R7])]:
        d = PP.set_index('dt')['pnl'].resample('D').sum(); cum = PP['pnl'].cumsum()
        out.append((PP['pnl'].sum(), d.mean()/d.std() if d.std()>0 else np.nan,
                    float((cum.cummax()-cum).max()) if len(PP) else 0))
    (n_, s_, dd_), (n7, s7, dd7) = out
    print(f'{label:46s} ${n_:+8,.0f} {s_:5.2f} ${dd_:7,.0f} | ${n7:+7,.0f} {s7:5.2f}')

K = lambda g: g['pnl_k']; KH = lambda g: g['pnl_k']*g['hmul']; F = lambda g: g['pnl_flat']
MO = lambda g: g['pnl']
print(f'{"combo":46s} {"net":>9s} {"S":>5s} {"DD":>8s} | {"net7":>8s} {"S7":>5s}')
score([(old, old['m_stack'], K)], 'old SW-stack kelly FLAT (revert cand)')
score([(old, old['m_stack'], KH)], 'old SW-stack xHdamp *PAPER')
score([(old, old['m_stackR'], KH)], 'old dipRESC-retro xHdamp')
score([(slp, slp['m_stack'], K)], 'slope SW-stack kelly flat')
score([(slp, slp['m_stack'], KH)], 'slope SW-stack xHdamp')
score([(old, None, F)], 'old flat$100 (no gates)')
score([(slp, None, F)], 'slope flat$100 (no gates)')
score([(mo, None, MO)], 'mktfavOUtau solo (transfer)')
score([(old, old['m_stack'], KH), (slp, slp['m_stack'], KH)], 'DUAL old-stack + slope-stack (both xH)')
score([(old, old['m_stack'], KH), (slp, slp['m_stack'], K)], 'DUAL old-xH + slope-flatK')
score([(old, old['m_stack'], KH), (mo, None, MO)], 'DUAL old-xH + mfOUtau')
score([(old, old['m_stack'], K), (mo, None, MO)], 'DUAL old-flatK + mfOUtau')
score([(old, old['m_stack'], KH), (slp, slp['m_stack'], KH), (mo, None, MO)], 'TRIPLE old-xH + slope-xH + mfOUtau')
score([(old, old['m_stack'], K), (slp, slp['m_stack'], K), (mo, None, MO)], 'TRIPLE old-K + slope-K + mfOUtau')
score([(old, old['m_stackR'], KH), (slp, slp['m_stack'], KH), (mo, None, MO)], 'TRIPLE oldDipR-xH + slope-xH + mfOUtau')
print()
to = set(old[old['m_stack']]['contract_ticker']); tsl = set(slp[slp['m_stack']]['contract_ticker']); tm = set(mo['contract_ticker'])
print(f'overlap: old n={len(to)} slope n={len(tsl)} mf n={len(tm)} | old∩slope={len(to&tsl)} old∩mf={len(to&tm)} slope∩mf={len(tsl&tm)} all3={len(to&tsl&tm)}')
os_ = old[old['m_stack']].set_index('contract_ticker')['side']; ss_ = slp[slp['m_stack']].set_index('contract_ticker')['side']
both = list(to&tsl); opp = sum(1 for t in both if os_.get(t) != ss_.get(t))
print(f'old-vs-slope overlap: same-side {len(both)-opp}, opposite {opp}')
