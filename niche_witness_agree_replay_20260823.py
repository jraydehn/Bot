"""[2026-08-23 pre-registered split — see revisit_tracker] Agreement-witness rescue test: within the refresh's mkv-Sideways-blocked
population (Aug, all models OOS), does v1/v2 co-fire partition winners?
Causal form: witness fire time <= refresh decision time."""
import pandas as pd, numpy as np, pickle, warnings
warnings.simplefilter('ignore')
BASE = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc'

# ---- load archive window (ALL rows for v2 slope joins) ----
keep = []
for chunk in pd.read_csv(f'{BASE}/results/btc_scan_archive.csv', low_memory=False,
                         on_bad_lines='skip', chunksize=200_000):
    chunk['dt'] = pd.to_datetime(chunk['logged_at'].astype(str).str.replace(r'\+00:00$', '', regex=True),
                                 errors='coerce', utc=True, format='mixed')
    c = chunk[chunk['dt'] >= pd.Timestamp('2026-08-20', tz='UTC')]
    if len(c):
        keep.append(c)
arch = pd.concat(keep, ignore_index=True).sort_values('dt').reset_index(drop=True)
print('archive rows 08-20+:', len(arch))

# ---- v1 fires (static model, band 0.32-0.45, edge [0.06,0.20)) ----
art1 = pickle.load(open(f'{BASE}/models/btc_hourly_lgbm_niche_20260728.pkl', 'rb'))
m1, F1 = art1['model'], art1['features']
a1 = arch.copy()
for c in set(F1 + ['p_market', 'tau_minutes', 'strike', 'spot']) - {'z_moneyness'}:
    a1[c] = pd.to_numeric(a1[c], errors='coerce') if c in a1.columns else np.nan
with np.errstate(divide='ignore', invalid='ignore'):
    a1['z_moneyness'] = np.log(a1['strike'] / a1['spot']) / np.sqrt(a1['tau_minutes'].clip(lower=1))
b1 = a1[a1['p_market'].between(0.32, 0.45)].copy()
p1 = m1.predict_proba(b1[F1])[:, 1]
fee1 = 0.07 * b1['p_market'] * (1 - b1['p_market'])
e1 = p1 - b1['p_market'] - fee1
f1 = b1[(e1 >= 0.06) & (e1 < 0.20)].sort_values('dt').drop_duplicates('contract_ticker', keep='first')
v1_time = f1.set_index('contract_ticker')['dt'].to_dict()
print('v1 qualifying fires:', len(f1))

# ---- v2 fires (slope model, band 0.35-0.65, edge>=0.06) ----
art2 = pickle.load(open(f'{BASE}/models/btc_hourly_lgbm_niche_v2_20260728.pkl', 'rb'))
m2, F2, SB = art2['model'], art2['features'], art2['slope_bases']
a2 = arch.copy()
static_needed = (set(F2) - {'z_moneyness'}
                 - {c for c in F2 if c.startswith(('D15_', 'D45_', 'D120_', 'S15_', 'S45_', 'S120_', 'dprice_'))})
for c in set(list(static_needed) + SB + ['p_market', 'tau_minutes', 'strike', 'spot']):
    a2[c] = pd.to_numeric(a2[c], errors='coerce') if c in a2.columns else np.nan
ts = a2['dt'].astype('int64') / 1e9
nc = {}
for tag, sec in [('15', 900), ('45', 2700), ('120', 7200)]:
    idx = np.searchsorted(ts, ts - sec, side='right') - 1
    valid = idx >= 0
    pv = np.where(valid, a2['spot'].values[np.clip(idx, 0, None)], np.nan)
    dp = pd.Series((a2['spot'].values / pv - 1) * 100, index=a2.index)
    nc[f'dprice_{tag}'] = dp
    for c in SB:
        pr = np.where(valid, a2[c].values[np.clip(idx, 0, None)], np.nan)
        d = a2[c].values - pr
        nc[f'D{tag}_{c}'] = d
        nc[f'S{tag}_{c}'] = np.clip(d / dp.replace(0, np.nan), -50, 50)
a2 = pd.concat([a2, pd.DataFrame(nc, index=a2.index)], axis=1)
with np.errstate(divide='ignore', invalid='ignore'):
    a2['z_moneyness'] = np.log(a2['strike'] / a2['spot']) / np.sqrt(a2['tau_minutes'].clip(lower=1))
b2 = a2[a2['p_market'].between(0.35, 0.65)].copy()
p2 = m2.predict_proba(b2[F2])[:, 1]
fee2 = 0.07 * b2['p_market'] * (1 - b2['p_market'])
e2 = p2 - b2['p_market'] - fee2
f2 = b2[e2 >= 0.06].sort_values('dt').drop_duplicates('contract_ticker', keep='first')
v2_time = f2.set_index('contract_ticker')['dt'].to_dict()
print('v2 qualifying fires:', len(f2))

# ---- refresh mkv-blocked population (Aug Sideways, OOS) ----
sw = pd.read_csv('PLACEHOLDER_REBUILD_VIA_nr_full_replay', low_memory=False)
sw['dt'] = pd.to_datetime(sw['dt'], utc=True)
sw = sw[sw['dt'] >= pd.Timestamp('2026-08-20', tz='UTC')].copy()
sw['day'] = sw['dt'].dt.date
cost2 = (pd.to_numeric(sw['p_market'], errors='coerce') + 0.015).clip(0.01, 0.99)
sw['pnl_slip'] = np.where(sw['win'], 100 * (1 - cost2) / cost2, -100.0) - 100 * 0.07 * (1 - cost2)
print('refresh blocked population (Aug Sideways):', len(sw))

def causal_agree(row, tmap):
    t = tmap.get(row['contract_ticker'])
    return t is not None and t <= row['dt']

sw['v1_any'] = sw['contract_ticker'].isin(v1_time)
sw['v2_any'] = sw['contract_ticker'].isin(v2_time)
sw['v1_causal'] = sw.apply(lambda r: causal_agree(r, v1_time), axis=1)
sw['v2_causal'] = sw.apply(lambda r: causal_agree(r, v2_time), axis=1)
sw['v2_applicable'] = pd.to_numeric(sw['p_market'], errors='coerce') >= 0.35

def seg(g, lab):
    if not len(g):
        print(f'  {lab:40s} (empty)'); return
    days = g.groupby('day')['pnl'].sum()
    print(f'  {lab:40s} n={len(g):3d} WR={g.win.mean():5.1%} mid=${g.pnl.sum():+7,.0f} '
          f'slip=${g.pnl_slip.sum():+7,.0f} days[{" ".join(f"{v:+5.0f}" for v in days)}]')

print()
print('== v2 agreement (within pm>=0.35 overlap only) ==')
ov = sw[sw['v2_applicable']]
print(f'  overlap population n={len(ov)}')
seg(ov[ov['v2_causal']], 'v2 CAUSAL agree (fired before refresh)')
seg(ov[ov['v2_any'] & ~ov['v2_causal']], 'v2 late agree (fired after)')
seg(ov[~ov['v2_any']], 'v2 disagree (never fired)')
seg(sw[~sw['v2_applicable']], 'v2 n/a (pm<0.35, cheap side)')
print()
print('== v1 agreement ==')
seg(sw[sw['v1_causal']], 'v1 CAUSAL agree')
seg(sw[sw['v1_any'] & ~sw['v1_causal']], 'v1 late agree')
seg(sw[~sw['v1_any']], 'v1 disagree')
print()
print('== ensemble forms ==')
seg(sw[sw['v1_causal'] | sw['v2_causal']], 'EITHER causal agree')
seg(sw[sw['v1_causal'] & sw['v2_causal']], 'BOTH causal agree')
seg(sw[~(sw['v1_any'] | sw['v2_any'])], 'NEITHER ever agrees')
