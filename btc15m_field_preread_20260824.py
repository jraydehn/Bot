"""Pre-08-25-read BTC 15m field sweep: every arm x sizer, every DUAL/TRIPLE combo.
Descriptive decision support — the read's pre-registered conditions still govern."""
import pandas as pd, numpy as np
from scipy.stats import norm

PT = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_btc15m.csv'
ab = pd.read_csv(PT, low_memory=False)
ab['dt'] = pd.to_datetime(ab['logged_at'], errors='coerce', utc=True, format='mixed')
num = ['p_market','p_model_15m','p_gbdt','resolved_yes','raw_edge','offset_pct','body_15m','dir_15m',
       'stoch_k_5m','stoch_k_15m','stoch_k_1h','chg_5m','chg_15m','chg_1h','composite_p_up','liq_score',
       'vol_ratio','vwap_hmm_state','z_drift_6h','kalman_velocity_15m','d15_realized_vol_annual',
       'ou_theta','tau_minutes','realized_vol_annual']
for c in num: ab[c] = pd.to_numeric(ab.get(c), errors='coerce')
ab = ab[(ab['dt'] >= pd.Timestamp('2026-08-05 22:00', tz='UTC')) & ab['resolved_yes'].notna()
        & ab['p_market'].between(0.03, 0.97)].copy()
cut12 = pd.Timestamp('2026-08-12 22:00', tz='UTC')
vv = ab['p_model_15m']; rev = ab['raw_edge']
dyes = (rev - (vv - ab['p_market'])).abs(); dno = (rev - (vv - (1 - ab['p_market']))).abs()
flip = (ab['dt'] < cut12) & (dno < dyes)
ab.loc[flip, 'p_model_15m'] = 1 - vv[flip]
fee = 0.07 * ab['p_market'] * (1 - ab['p_market'])

# ---- ARM 1: prod g+k (12 gates, kelly) ----
ey = ab['p_model_15m'] - ab['p_market'] - fee
en = ab['p_market'] - ab['p_model_15m'] - fee
ab['side'] = np.where(ey >= en, 'yes', 'no'); ab['edge'] = np.maximum(ey, en)
q = ab[ab['edge'] >= 0.04].sort_values('dt').drop_duplicates('contract_ticker', keep='first').copy()
m1 = q['markov_regime_1h'].astype(str); m15 = q['markov_regime_15m'].astype(str)
sk1 = q['stoch_k_1h'].fillna(50); sk15 = q['stoch_k_15m'].fillna(50); sk5 = q['stoch_k_5m']
cpu = q['composite_p_up']; body = q['body_15m'].fillna(1.0); d15 = q['dir_15m']
chg1h = q['chg_1h'].fillna(0); chg15 = q['chg_15m'].fillna(0); chg5 = q['chg_5m']
vr = q['vol_ratio'].fillna(1.0); vst = q['vwap_hmm_state']; liq = q['liq_score']
pm = q['p_market']; yes = q['side'] == 'yes'
hit_yes = ((q['offset_pct'] < 0.025) | (m1 == 'Bear')
           | ((m15 == 'Bear') & ~((cpu <= 0.488).fillna(False)))
           | ((d15 == 1) & (pm >= 0.50) & (pm < 0.65)
              & ~((m1 == 'Bull') | ((m1 == 'Bear') & (m15 == 'Bear') & (sk1 < 35))))
           | ((m1 == 'Sideways') & (body < 0.30)
              & ~(((cpu < 0.40).fillna(False)) | ((sk15 >= 20) & (sk15 < 40))))
           | ((sk1 >= 95) & (liq == -1).fillna(False)))
hit_no = (((pm >= 0.50) & (pm < 0.80)) | ((chg1h > 0) & (sk1 >= 30) & (sk1 < 70))
          | ((m1 == 'Sideways') & (pm >= 0.70) & (sk1 >= 70))
          | ((m1 == 'Sideways') & (m15 == 'Sideways') & (pm >= 0.55))
          | ((m1 == 'Bear') & (m15 == 'Bull'))
          | ((sk5 > 76).fillna(False) & (chg5 > 0).fillna(False))
          | ((vst == 4).fillna(False) | ((vst == 2).fillna(False) & (vr < 0.216))
             | ((vst == 5).fillna(False) & (sk1 < 85))
             | ((vst == 7).fillna(False) & (chg15 >= -0.112))))
zext = (q['z_drift_6h'] > 2.5).fillna(False) & ~yes
gk = q[~(np.where(yes, hit_yes, hit_no) | zext)].copy()
fee_g = 0.07 * gk['p_market'] * (1 - gk['p_market'])
kel = np.where(gk['side'] == 'yes',
               (gk['p_model_15m'] - gk['p_market'] - fee_g) / (1 - gk['p_market']),
               (gk['p_market'] - gk['p_model_15m'] - fee_g) / gk['p_market'])
gk['stake'] = 2500.0 * np.clip(kel, 0, 0.10)
gk = gk[gk['stake'] > 0].copy()
gc = np.where(gk['side'] == 'yes', gk['p_market'], 1 - gk['p_market'])
gw = np.where(gk['side'] == 'yes', gk['resolved_yes'] == 1, gk['resolved_yes'] == 0)
gk['pnl'] = np.where(gw, gk['stake'] * (1 - gc) / gc, -gk['stake']) - (gk['stake'] / gc) * fee_g
# MFconv tiers (raw + OU-conditioned)
pmf = norm.cdf(1.8 * norm.ppf(gk['p_market'].clip(0.01, 0.99)))
feem = 0.07 * gk['p_market'] * (1 - gk['p_market'])
eym = pmf - gk['p_market'] - feem; enm = gk['p_market'] - pmf - feem
mf_side = np.where(eym >= enm, 'yes', 'no'); mf_edge = np.maximum(eym, enm)
agm = mf_side == gk['side']
tier = np.where(agm & (mf_edge >= 0.04), 1.00, np.where(agm, 0.75, np.where(mf_edge < 0.04, 0.50, 0.25)))
ou_ok_g = ~(((gk['ou_theta'] < 2.2243).fillna(False)) | ((gk['tau_minutes'] < 8.64).fillna(False)))
gk['w_mf'] = tier; gk['w_mfou'] = np.where(ou_ok_g, tier, 0.50)

# ---- ARM 2: shadow +KV (flat $100) ----
sh = ab[ab['p_gbdt'].notna()].copy()
fs = 0.07 * sh['p_market'] * (1 - sh['p_market'])
eys = sh['p_gbdt'] - sh['p_market'] - fs; ens = sh['p_market'] - sh['p_gbdt'] - fs
sh['side'] = np.where(eys >= ens, 'yes', 'no'); sh['edge'] = np.maximum(eys, ens)
sq = sh[sh['edge'] >= 0.04].sort_values('dt').drop_duplicates('contract_ticker', keep='first').copy()
kv_blk = ((sq['kalman_velocity_15m'] < 1e-4).fillna(False)
          | ((sq['side'] == 'yes') & (sq['d15_realized_vol_annual'] < -0.012).fillna(False)))
skv = sq[~kv_blk].copy()
sc = np.where(skv['side'] == 'yes', skv['p_market'], 1 - skv['p_market'])
sw_ = np.where(skv['side'] == 'yes', skv['resolved_yes'] == 1, skv['resolved_yes'] == 0)
sfee = 0.07 * skv['p_market'] * (1 - skv['p_market'])
skv['pnl'] = np.where(sw_, 100 * (1 - sc) / sc, -100.0) - (100 / sc) * sfee
skv['w_edge'] = np.clip(skv['edge'] / 0.08, 0.5, 2.0)
skv['w_vold'] = np.where((skv['realized_vol_annual'] >= 0.30).fillna(False), 0.5, 1.0)

# ---- ARM 3: mkt-fav +OUtau (flat $100) ----
mfp = norm.cdf(1.8 * norm.ppf(ab['p_market'].clip(0.01, 0.99)))
eyf = mfp - ab['p_market'] - fee; enf = ab['p_market'] - mfp - fee
mf = ab.copy(); mf['side'] = np.where(eyf >= enf, 'yes', 'no'); mf['edge'] = np.maximum(eyf, enf)
mq = mf[mf['edge'] >= 0.04].sort_values('dt').drop_duplicates('contract_ticker', keep='first').copy()
zdm = (mq['z_drift_6h'] > 2.5).fillna(False) & (mq['side'] == 'no')
mq = mq[~zdm]
ou_blk = ((mq['ou_theta'] < 2.2243).fillna(False)) | ((mq['tau_minutes'] < 8.64).fillna(False))
mo = mq[~ou_blk].copy()
mc = np.where(mo['side'] == 'yes', mo['p_market'], 1 - mo['p_market'])
mw = np.where(mo['side'] == 'yes', mo['resolved_yes'] == 1, mo['resolved_yes'] == 0)
mfee = 0.07 * mo['p_market'] * (1 - mo['p_market'])
mo['pnl'] = np.where(mw, 100 * (1 - mc) / mc, -100.0) - (100 / mc) * mfee

def compose(parts, label, recent=None):
    rows = []
    for df, wcol in parts:
        w = df[wcol] if wcol else 1.0
        rows.append(pd.DataFrame({'dt': df['dt'], 'pnl': df['pnl'] * w}))
    P = pd.concat(rows).sort_values('dt')
    if recent is not None:
        P = P[P['dt'] >= recent]
    d = P.set_index('dt')['pnl'].resample('D').sum()
    cum = P['pnl'].cumsum()
    return (P['pnl'].sum(), d.mean() / d.std() if d.std() > 0 else np.nan,
            float((cum.cummax() - cum).max()) if len(P) else 0, len(P))

GK, KV, MO = (gk, None), (skv, None), (mo, None)
combos = [
  ('gk flat (solo)',              [(gk, None)]),
  ('KV flat (solo)',              [(skv, None)]),
  ('mktfav+OUtau (solo)',         [(mo, None)]),
  ('DUAL v2+KV  [gk + KV]  *PAPER',            [(gk, None), (skv, None)]),
  ('DUAL gk + KVxEDGE',           [(gk, None), (skv, 'w_edge')]),
  ('DUAL gk + KVxVOLd',           [(gk, None), (skv, 'w_vold')]),
  ('DUAL gk + KVxEDGExVOLd',      [(gk, None), (skv, 'w_ev')]),
  ('DUAL v1-OU [gk + mfOUtau]',   [(gk, None), (mo, None)]),
  ('DUAL v3d [gkxMFou + KVxEDGE]',[(gk, 'w_mfou'), (skv, 'w_edge')]),
  ('DUAL gkxMFou + KVxVOLd',      [(gk, 'w_mfou'), (skv, 'w_vold')]),
  ('TRIPLE flat [gk+KV+mfOUtau]', [(gk, None), (skv, None), (mo, None)]),
  ('TRIPLE conv [gkxMFou+KVxEDGE+mfOUtau]', [(gk, 'w_mfou'), (skv, 'w_edge'), (mo, None)]),
  ('TRIPLE gk+KVxVOLd+mfOUtau',   [(gk, None), (skv, 'w_vold'), (mo, None)]),
  ('TRIPLE gk+KVxEDGE+mfOUtau',   [(gk, None), (skv, 'w_edge'), (mo, None)]),
]
skv['w_ev'] = skv['w_edge'] * skv['w_vold']
r7 = pd.Timestamp('2026-08-17', tz='UTC')
print(f'{"combo":42s} {"net":>9s} {"S":>5s} {"DD":>8s} {"n":>4s} | {"net7":>8s} {"S7":>5s} {"DD7":>7s}')
res = []
for name, parts in combos:
    net, s, dd, n = compose(parts, name)
    net7, s7, dd7, _ = compose(parts, name, recent=r7)
    res.append((name, net, s, dd, n, net7, s7, dd7))
    print(f'{name:42s} ${net:+8,.0f} {s:5.2f} ${dd:7,.0f} {n:4d} | ${net7:+7,.0f} {s7:5.2f} ${dd7:6,.0f}')
print()
print('== TRIPLE overlap audit (same-contract exposure) ==')
tg, tk, tm = set(gk['contract_ticker']), set(skv['contract_ticker']), set(mo['contract_ticker'])
print(f'gk n={len(tg)}  KV n={len(tk)}  mfOUtau n={len(tm)}')
print(f'gk∩KV={len(tg&tk)}  gk∩mf={len(tg&tm)}  KV∩mf={len(tk&tm)}  all3={len(tg&tk&tm)}')
gs_ = gk.set_index('contract_ticker')['side']
ms_ = mo.set_index('contract_ticker')['side']
both = list(tg & tm)
opp = sum(1 for t in both if gs_.get(t) != ms_.get(t))
print(f'gk-vs-mf overlap: same-side {len(both)-opp}, opposite-side {opp} (fee-burn pairs)')
