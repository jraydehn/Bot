"""
Deep rescue search for gate: zd_state==2 AND pm_drift_5m<-0.001 AND hmm_vol_state==0
Finds sub-conditions where YES bets should be rescued (WR >= BE).
"""
import pandas as pd, numpy as np, pickle

with open('models/hmm_zdrift_btc.pkl', 'rb') as f:
    bundle = pickle.load(f)
model  = bundle['model']
scaler = bundle['scaler']

# ── 1. Build gate rows from CSV (correct resolved_yes) ──────────────────────
arc = pd.read_csv('results/btc_scan_archive.csv', low_memory=False)
for c in ['pm_drift_5m','rvol_1h','resolved_yes','p_market',
          'vol_score','ema_stack_bias','composite_trend','composite_rev',
          'confirmation_score','funding_bias','offset_pct']:
    arc[c] = pd.to_numeric(arc[c], errors='coerce')
arc['logged_at'] = pd.to_datetime(arc['logged_at'], format='mixed', utc=True, errors='coerce')

ok = arc['pm_drift_5m'].notna() & arc['rvol_1h'].notna()
Xsc = scaler.transform(arc.loc[ok, ['pm_drift_5m','rvol_1h']].values)
arc.loc[ok, 'zd_state'] = model.predict(Xsc).astype(float)

pq_vs = pd.read_parquet('results/btc_scan_archive_hmm.parquet',
                        columns=['logged_at','contract_ticker','hmm_vol_state'])
pq_vs['logged_at'] = pd.to_datetime(pq_vs['logged_at'], format='mixed', utc=True, errors='coerce')
pq_vs['hmm_vol_state'] = pd.to_numeric(pq_vs['hmm_vol_state'], errors='coerce')
pq_vs = pq_vs.dropna(subset=['logged_at','hmm_vol_state'])

arc_s = arc.dropna(subset=['logged_at']).sort_values('logged_at')
merged = pd.merge_asof(arc_s,
                       pq_vs[['logged_at','contract_ticker','hmm_vol_state']].sort_values('logged_at'),
                       on='logged_at', by='contract_ticker',
                       direction='nearest', tolerance=pd.Timedelta('5min'))

# ── 2. Enrich with parquet indicators ───────────────────────────────────────
extra_cols = ['logged_at','contract_ticker',
              'stoch_k_15m','stoch_k_1h','stoch_k_5m','stoch_k',
              'rsi_1h','rsi_4h','macd_hist_1h','macd_hist_4h',
              'bp_1h','chg_1h','chg_10m','chg_30m',
              'adx_4h','adx_1h','p_gbdt','p_up_v2','ls_long_pct',
              'liq_score','tau_minutes','vpin_score','obi_score',
              'liq_bias','oi_chg_pct','squeeze_1h']
pq_extra = pd.read_parquet('results/btc_scan_archive_hmm.parquet', columns=extra_cols)
pq_extra['logged_at'] = pd.to_datetime(pq_extra['logged_at'], format='mixed', utc=True, errors='coerce')
pq_extra = pq_extra.dropna(subset=['logged_at'])
for c in extra_cols[2:]:
    pq_extra[c] = pd.to_numeric(pq_extra[c], errors='coerce')

merged2 = pd.merge_asof(merged.sort_values('logged_at'),
                        pq_extra.sort_values('logged_at'),
                        on='logged_at', by='contract_ticker',
                        direction='nearest', tolerance=pd.Timedelta('2min'),
                        suffixes=('','_pq'))

gate = merged2[
    (merged2['zd_state'] == 2) &
    (merged2['pm_drift_5m'] < -0.001) &
    (merged2['hmm_vol_state'] == 0) &
    merged2['resolved_yes'].notna() &
    merged2['p_market'].notna()
].copy()

gate['hour_utc'] = gate['logged_at'].dt.hour
print(f'Gate: n={len(gate)}  WR={gate["resolved_yes"].mean():.1%}  '
      f'BE={gate["p_market"].mean():.1%}  '
      f'edge={gate["resolved_yes"].mean()-gate["p_market"].mean():+.1%}')
print(f'Indicator coverage: sk1h={gate["stoch_k_1h"].notna().mean():.0%}  '
      f'rsi1h={gate["rsi_1h"].notna().mean():.0%}  '
      f'macd1h={gate["macd_hist_1h"].notna().mean():.0%}  '
      f'bp1h={gate["bp_1h"].notna().mean():.0%}')
print()


def stats(df, label='', min_n=50):
    n = len(df)
    if n < min_n:
        return None
    wr = float(df['resolved_yes'].mean())
    be = float(df['p_market'].mean())
    e  = wr - be
    z  = e / (be*(1-be)/n)**0.5
    return dict(label=label, n=n, wr=wr, be=be, e=e, z=z)


def show(r):
    if r is None:
        return
    if r['e'] > -0.005:
        flag = ' ◄◄ RESCUE'
    elif r['e'] > -0.015:
        flag = ' ◄ WEAKER'
    else:
        flag = ''
    print(f"  {r['label']:<52} n={r['n']:5,}  WR={r['wr']:.1%}  "
          f"BE={r['be']:.1%}  e={r['e']:+.1%}  z={r['z']:+.2f}{flag}")


# ── SECTION 1: Individual signals ───────────────────────────────────────────
print('=== 1. rsi_1h ===')
for lo, hi in [(0,30),(30,40),(40,50),(50,60),(60,70),(70,100)]:
    show(stats(gate[gate['rsi_1h'].between(lo,hi,inclusive='left')], f'rsi1h[{lo},{hi})'))

print()
print('=== 2. rsi_4h ===')
for lo, hi in [(0,30),(30,40),(40,50),(50,60),(60,70),(70,100)]:
    show(stats(gate[gate['rsi_4h'].between(lo,hi,inclusive='left')], f'rsi4h[{lo},{hi})'))

print()
print('=== 3. stoch_k_1h ===')
for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
    show(stats(gate[gate['stoch_k_1h'].between(lo,hi,inclusive='left')], f'sk1h[{lo},{hi})'))

print()
print('=== 4. stoch_k_15m ===')
for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
    show(stats(gate[gate['stoch_k_15m'].between(lo,hi,inclusive='left')], f'sk15m[{lo},{hi})'))

print()
print('=== 5. stoch_k_5m ===')
for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
    show(stats(gate[gate['stoch_k_5m'].between(lo,hi,inclusive='left')], f'sk5m[{lo},{hi})'))

print()
print('=== 6. chg_1h ===')
for lo, hi in [(-5,-1),(-1,-0.5),(-0.5,-0.2),(-0.2,0),(0,0.2),(0.2,0.5),(0.5,5)]:
    show(stats(gate[gate['chg_1h'].between(lo,hi,inclusive='left')], f'chg1h[{lo},{hi})'))

print()
print('=== 7. chg_30m ===')
for lo, hi in [(-3,-0.5),(-0.5,-0.2),(-0.2,0),(0,0.2),(0.2,0.5),(0.5,3)]:
    show(stats(gate[gate['chg_30m'].between(lo,hi,inclusive='left')], f'chg30m[{lo},{hi})'))

print()
print('=== 8. chg_10m ===')
for lo, hi in [(-1,-0.2),(-0.2,-0.1),(-0.1,0),(0,0.1),(0.1,0.2),(0.2,1)]:
    show(stats(gate[gate['chg_10m'].between(lo,hi,inclusive='left')], f'chg10m[{lo},{hi})'))

print()
print('=== 9. bp_1h ===')
for lo, hi in [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]:
    show(stats(gate[gate['bp_1h'].between(lo,hi,inclusive='left')], f'bp1h[{lo},{hi})'))

print()
print('=== 10. macd_hist_1h ===')
show(stats(gate[gate['macd_hist_1h'] > 0],   'macd1h>0'))
show(stats(gate[gate['macd_hist_1h'] < 0],   'macd1h<0'))
show(stats(gate[gate['macd_hist_1h'] > 50],  'macd1h>50'))
show(stats(gate[gate['macd_hist_1h'] > 100], 'macd1h>100'))
show(stats(gate[gate['macd_hist_1h'] < -100],'macd1h<-100'))

print()
print('=== 11. macd_hist_4h ===')
show(stats(gate[gate['macd_hist_4h'] > 0],   'macd4h>0'))
show(stats(gate[gate['macd_hist_4h'] < 0],   'macd4h<0'))
show(stats(gate[gate['macd_hist_4h'] > 100], 'macd4h>100'))

print()
print('=== 12. ema_stack_bias ===')
for v, l in [(-1,'bear'),(0,'neutral'),(1,'bull')]:
    show(stats(gate[gate['ema_stack_bias'] == v], f'ema={l}'))

print()
print('=== 13. composite_trend ===')
for v in sorted(gate['composite_trend'].dropna().unique()):
    show(stats(gate[gate['composite_trend'] == v], f'c_trend={int(v)}', min_n=30))

print()
print('=== 14. composite_rev ===')
for lo, hi in [(0,2),(2,4),(4,6),(6,8),(8,10)]:
    show(stats(gate[gate['composite_rev'].between(lo,hi,inclusive='left')], f'c_rev[{lo},{hi})'))

print()
print('=== 15. adx_1h ===')
for lo, hi in [(0,15),(15,20),(20,25),(25,30),(30,40),(40,100)]:
    show(stats(gate[gate['adx_1h'].between(lo,hi,inclusive='left')], f'adx1h[{lo},{hi})'))

print()
print('=== 16. adx_4h ===')
for lo, hi in [(0,20),(20,25),(25,30),(30,40),(40,100)]:
    show(stats(gate[gate['adx_4h'].between(lo,hi,inclusive='left')], f'adx4h[{lo},{hi})'))

print()
print('=== 17. p_up_v2 ===')
for lo, hi in [(0,0.40),(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,1.0)]:
    show(stats(gate[gate['p_up_v2'].between(lo,hi,inclusive='left')], f'p_up_v2[{lo},{hi})'))

print()
print('=== 18. ls_long_pct ===')
for lo, hi in [(40,55),(55,60),(60,65),(65,70),(70,100)]:
    show(stats(gate[gate['ls_long_pct'].between(lo,hi,inclusive='left')], f'ls_long[{lo},{hi})'))

print()
print('=== 19. vpin_score ===')
for lo, hi in [(-3,-1),(-1,0),(0,1),(1,3)]:
    show(stats(gate[gate['vpin_score'].between(lo,hi,inclusive='left')], f'vpin[{lo},{hi})'))

print()
print('=== 20. obi_score ===')
for lo, hi in [(-3,-1),(-1,0),(0,1),(1,3)]:
    show(stats(gate[gate['obi_score'].between(lo,hi,inclusive='left')], f'obi[{lo},{hi})'))

print()
print('=== 21. liq_score ===')
for lo, hi in [(-3,-1),(-1,0),(0,1),(1,3)]:
    show(stats(gate[gate['liq_score'].between(lo,hi,inclusive='left')], f'liq[{lo},{hi})'))

print()
print('=== 22. oi_chg_pct ===')
for lo, hi in [(-20,-2),(-2,-0.5),(-0.5,0),(0,0.5),(0.5,2),(2,20)]:
    show(stats(gate[gate['oi_chg_pct'].between(lo,hi,inclusive='left')], f'oi_chg[{lo},{hi})'))

print()
print('=== 23. p_market depth ===')
for lo, hi in [(0.1,0.3),(0.3,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    show(stats(gate[gate['p_market'].between(lo,hi,inclusive='left')], f'pm[{lo},{hi})'))

print()
print('=== 24. offset_pct ===')
for lo, hi in [(-10,-2),(-2,-0.5),(-0.5,0),(0,0.5),(0.5,2),(2,10)]:
    show(stats(gate[gate['offset_pct'].between(lo,hi,inclusive='left')], f'offset[{lo},{hi})'))

print()
print('=== 25. pm_drift_5m magnitude ===')
for lo, hi in [(-0.003,-0.001),(-0.01,-0.003),(-0.05,-0.01),(-1,-0.05)]:
    show(stats(gate[gate['pm_drift_5m'].between(lo,hi,inclusive='left')], f'drift[{lo},{hi})'))

print()
print('=== 26. hour_utc ===')
for lo, hi in [(0,6),(6,12),(12,18),(18,24)]:
    show(stats(gate[gate['hour_utc'].between(lo,hi,inclusive='left')], f'hour[{lo},{hi})UTC'))

print()
print('=== 27. rvol_1h ===')
for lo, hi in [(0,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.25),(0.25,0.30),(0.30,0.40)]:
    show(stats(gate[gate['rvol_1h'].between(lo,hi,inclusive='left')], f'rvol[{lo},{hi})'))

print()
print('=== 28. tau_minutes ===')
for lo, hi in [(0,30),(30,60),(60,120),(120,240),(240,600)]:
    show(stats(gate[gate['tau_minutes'].between(lo,hi,inclusive='left')], f'tau[{lo},{hi})min'))

print()
print('=== 29. squeeze_1h ===')
for lo, hi in [(-5,-1),(-1,0),(0,1),(1,5)]:
    show(stats(gate[gate['squeeze_1h'].between(lo,hi,inclusive='left')], f'squeeze1h[{lo},{hi})'))

print()
print('=== 30. p_gbdt ===')
for lo, hi in [(0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    show(stats(gate[gate['p_gbdt'].between(lo,hi,inclusive='left')], f'p_gbdt[{lo},{hi})'))

print()
print('=== 31. funding_bias ===')
for lo, hi in [(-1,-0.3),(-0.3,0),(0,0.3),(0.3,1)]:
    show(stats(gate[gate['funding_bias'].between(lo,hi,inclusive='left')], f'fund[{lo},{hi})'))

print()
print('=== 32. vol_score ===')
for lo, hi in [(-3,-1),(-1,0),(0,1),(1,3)]:
    show(stats(gate[gate['vol_score'].between(lo,hi,inclusive='left')], f'vol_score[{lo},{hi})'))

print()
print('=== 33. confirmation_score ===')
for lo, hi in [(-3,-1),(-1,0),(0,1),(1,3)]:
    show(stats(gate[gate['confirmation_score'].between(lo,hi,inclusive='left')], f'conf[{lo},{hi})'))

# ── SECTION 2: Key single-feature rescues ────────────────────────────────────
print()
print('=== 34. KEY SINGLE-FEATURE RESCUES ===')
singles = [
    gate['rsi_1h'] > 50,
    gate['rsi_1h'] > 55,
    gate['rsi_1h'] > 60,
    gate['rsi_4h'] > 50,
    gate['chg_1h'] > 0,
    gate['chg_1h'] > 0.2,
    gate['bp_1h'] > 0.5,
    gate['bp_1h'] > 0.6,
    gate['macd_hist_1h'] > 0,
    gate['macd_hist_4h'] > 0,
    gate['ema_stack_bias'] == 1,
    gate['stoch_k_1h'] > 50,
    gate['stoch_k_1h'] > 70,
    gate['stoch_k_15m'] > 60,
    gate['composite_trend'] >= 2,
    gate['adx_4h'] < 25,
    gate['p_up_v2'] > 0.52,
]
labels = [
    'rsi1h>50','rsi1h>55','rsi1h>60','rsi4h>50','chg1h>0','chg1h>0.2',
    'bp1h>0.5','bp1h>0.6','macd1h>0','macd4h>0','ema_bull',
    'sk1h>50','sk1h>70','sk15m>60','c_trend>=2','adx4h<25','p_up_v2>0.52',
]
for cond, lbl in zip(singles, labels):
    show(stats(gate[cond], lbl))

# ── SECTION 3: Two-factor combos ────────────────────────────────────────────
print()
print('=== 35. TWO-FACTOR COMBOS ===')
two_combos = [
    ((gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),        'rsi1h>50 + chg1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['bp_1h'] > 0.5),       'rsi1h>50 + bp1h>0.5'),
    ((gate['rsi_1h'] > 50) & (gate['macd_hist_1h'] > 0),  'rsi1h>50 + macd1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['stoch_k_1h'] > 50),   'rsi1h>50 + sk1h>50'),
    ((gate['rsi_1h'] > 50) & (gate['ema_stack_bias'] == 1),'rsi1h>50 + ema_bull'),
    ((gate['rsi_1h'] > 50) & (gate['rsi_4h'] > 50),       'rsi1h>50 + rsi4h>50'),
    ((gate['rsi_1h'] > 50) & (gate['macd_hist_4h'] > 0),  'rsi1h>50 + macd4h>0'),
    ((gate['rsi_1h'] > 50) & (gate['chg_30m'] > 0),       'rsi1h>50 + chg30m>0'),
    ((gate['rsi_1h'] > 55) & (gate['chg_1h'] > 0),        'rsi1h>55 + chg1h>0'),
    ((gate['rsi_1h'] > 55) & (gate['bp_1h'] > 0.5),       'rsi1h>55 + bp1h>0.5'),
    ((gate['chg_1h'] > 0) & (gate['bp_1h'] > 0.5),        'chg1h>0 + bp1h>0.5'),
    ((gate['chg_1h'] > 0) & (gate['macd_hist_1h'] > 0),   'chg1h>0 + macd1h>0'),
    ((gate['chg_1h'] > 0) & (gate['stoch_k_1h'] > 50),    'chg1h>0 + sk1h>50'),
    ((gate['chg_1h'] > 0) & (gate['macd_hist_4h'] > 0),   'chg1h>0 + macd4h>0'),
    ((gate['chg_1h'] > 0) & (gate['ema_stack_bias'] == 1), 'chg1h>0 + ema_bull'),
    ((gate['bp_1h'] > 0.5) & (gate['macd_hist_1h'] > 0),  'bp1h>0.5 + macd1h>0'),
    ((gate['bp_1h'] > 0.5) & (gate['stoch_k_1h'] > 50),   'bp1h>0.5 + sk1h>50'),
    ((gate['bp_1h'] > 0.5) & (gate['macd_hist_4h'] > 0),  'bp1h>0.5 + macd4h>0'),
    ((gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0),'macd1h>0 + macd4h>0'),
    ((gate['macd_hist_1h'] > 0) & (gate['ema_stack_bias'] == 1),'macd1h>0 + ema_bull'),
    ((gate['stoch_k_1h'] > 70) & (gate['rsi_1h'] > 50),   'sk1h>70 + rsi1h>50'),
    ((gate['stoch_k_1h'] > 70) & (gate['chg_1h'] > 0),    'sk1h>70 + chg1h>0'),
    ((gate['stoch_k_1h'] > 70) & (gate['bp_1h'] > 0.5),   'sk1h>70 + bp1h>0.5'),
    ((gate['stoch_k_15m'] > 60) & (gate['rsi_1h'] > 50),  'sk15m>60 + rsi1h>50'),
    ((gate['stoch_k_15m'] > 60) & (gate['chg_1h'] > 0),   'sk15m>60 + chg1h>0'),
    ((gate['rsi_4h'] > 50) & (gate['rsi_1h'] > 50),       'rsi4h>50 + rsi1h>50'),
    ((gate['rsi_4h'] > 50) & (gate['chg_1h'] > 0),        'rsi4h>50 + chg1h>0'),
    ((gate['rsi_4h'] > 50) & (gate['bp_1h'] > 0.5),       'rsi4h>50 + bp1h>0.5'),
    ((gate['rsi_4h'] > 50) & (gate['macd_hist_1h'] > 0),  'rsi4h>50 + macd1h>0'),
    ((gate['composite_trend'] >= 2) & (gate['rsi_1h'] > 50),'c_trend>=2 + rsi1h>50'),
    ((gate['composite_trend'] >= 2) & (gate['chg_1h'] > 0),'c_trend>=2 + chg1h>0'),
    ((gate['p_up_v2'] > 0.52) & (gate['rsi_1h'] > 50),   'p_up_v2>0.52 + rsi1h>50'),
    ((gate['p_up_v2'] > 0.52) & (gate['chg_1h'] > 0),    'p_up_v2>0.52 + chg1h>0'),
]
for cond, lbl in two_combos:
    show(stats(gate[cond], lbl))

# ── SECTION 4: Three-factor combos ──────────────────────────────────────────
print()
print('=== 36. THREE-FACTOR COMBOS ===')
three_combos = [
    ((gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0) & (gate['bp_1h'] > 0.5),
     'rsi1h>50 + chg1h>0 + bp1h>0.5'),
    ((gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0) & (gate['macd_hist_1h'] > 0),
     'rsi1h>50 + chg1h>0 + macd1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0) & (gate['stoch_k_1h'] > 50),
     'rsi1h>50 + chg1h>0 + sk1h>50'),
    ((gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0) & (gate['rsi_4h'] > 50),
     'rsi1h>50 + chg1h>0 + rsi4h>50'),
    ((gate['rsi_1h'] > 50) & (gate['bp_1h'] > 0.5) & (gate['macd_hist_1h'] > 0),
     'rsi1h>50 + bp1h>0.5 + macd1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0),
     'rsi1h>50 + macd1h>0 + macd4h>0'),
    ((gate['rsi_1h'] > 50) & (gate['macd_hist_1h'] > 0) & (gate['ema_stack_bias'] == 1),
     'rsi1h>50 + macd1h>0 + ema_bull'),
    ((gate['rsi_1h'] > 50) & (gate['rsi_4h'] > 50) & (gate['chg_1h'] > 0),
     'rsi1h>50 + rsi4h>50 + chg1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['rsi_4h'] > 50) & (gate['macd_hist_1h'] > 0),
     'rsi1h>50 + rsi4h>50 + macd1h>0'),
    ((gate['rsi_1h'] > 50) & (gate['rsi_4h'] > 50) & (gate['bp_1h'] > 0.5),
     'rsi1h>50 + rsi4h>50 + bp1h>0.5'),
    ((gate['chg_1h'] > 0) & (gate['bp_1h'] > 0.5) & (gate['macd_hist_1h'] > 0),
     'chg1h>0 + bp1h>0.5 + macd1h>0'),
    ((gate['chg_1h'] > 0) & (gate['bp_1h'] > 0.5) & (gate['rsi_4h'] > 50),
     'chg1h>0 + bp1h>0.5 + rsi4h>50'),
    ((gate['chg_1h'] > 0) & (gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0),
     'chg1h>0 + macd1h>0 + macd4h>0'),
    ((gate['chg_1h'] > 0) & (gate['macd_hist_1h'] > 0) & (gate['ema_stack_bias'] == 1),
     'chg1h>0 + macd1h>0 + ema_bull'),
    ((gate['chg_1h'] > 0) & (gate['stoch_k_1h'] > 50) & (gate['rsi_4h'] > 50),
     'chg1h>0 + sk1h>50 + rsi4h>50'),
    ((gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0) & (gate['rsi_1h'] > 50),
     'macd1h>0 + macd4h>0 + rsi1h>50'),
    ((gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0) & (gate['chg_1h'] > 0),
     'macd1h>0 + macd4h>0 + chg1h>0'),
    ((gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0) & (gate['bp_1h'] > 0.5),
     'macd1h>0 + macd4h>0 + bp1h>0.5'),
    ((gate['ema_stack_bias'] == 1) & (gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),
     'ema_bull + rsi1h>50 + chg1h>0'),
    ((gate['ema_stack_bias'] == 1) & (gate['bp_1h'] > 0.5) & (gate['rsi_1h'] > 50),
     'ema_bull + bp1h>0.5 + rsi1h>50'),
    ((gate['stoch_k_1h'] > 70) & (gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),
     'sk1h>70 + rsi1h>50 + chg1h>0'),
    ((gate['stoch_k_1h'] > 70) & (gate['bp_1h'] > 0.5) & (gate['macd_hist_1h'] > 0),
     'sk1h>70 + bp1h>0.5 + macd1h>0'),
    ((gate['rsi_4h'] > 50) & (gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),
     'rsi4h>50 + rsi1h>50 + chg1h>0'),
    ((gate['rsi_4h'] > 50) & (gate['macd_hist_1h'] > 0) & (gate['macd_hist_4h'] > 0),
     'rsi4h>50 + macd1h>0 + macd4h>0'),
    ((gate['p_up_v2'] > 0.52) & (gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),
     'p_up_v2>0.52 + rsi1h>50 + chg1h>0'),
    ((gate['composite_trend'] >= 2) & (gate['rsi_1h'] > 50) & (gate['chg_1h'] > 0),
     'c_trend>=2 + rsi1h>50 + chg1h>0'),
]
for cond, lbl in three_combos:
    show(stats(gate[cond], lbl, min_n=30))
