"""ETH 15m retrain — 2026-08-25 (the reserved ~08-30 slot, pulled forward on
payload maturity; user call).

NEW INPUTS vs the retired same-features refresh (the same-features-same-
blindness lesson): causally recomputed d/slope features over FULL history
(the hourly-v2 lever that fixed the ETH/SOL hourly nulls), pm-path features
(candle-archive backfill + live logging), market anchoring (p_market/
z_moneyness/spread — the BTC mktanchor direction), and the archive's native
15m/5m-frame kalman/KC/donchian families. composite_p_up excluded (dead
06-26; its _live revival is next-cycle payload).

Protocol (six-null-SOL lessons): 6 seeds, 3 walk-forward origins, UNTOUCHED
08-10+ holdout scored once, economics-first evaluation (flat-$100 fee-net
book vs the logged production stream on identical scans), calibration slope
reported (kelly-informativeness goal).
"""
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier

ARCHIVE = 'results/eth_scan_archive_15m.csv'
BACKFILL = 'results/eth15m_pmpath_backfill.csv'

STATIC = ['p_market', 'tau_minutes', 'spread', 'offset_pct',
          'bp_5m', 'vol_ratio', 'vol_ratio_5m', 'body_15m', 'bp_15m',
          'dir_15m', 'upper_wick_15m', 'lower_wick_15m', 'atr_ratio_15m',
          'range_ratio_15m', 'consec_dir_15m', 'stoch_k_5m', 'stoch_k_15m',
          'chg_1m', 'chg_5m', 'chg_15m', 'vwap_dist', 'ema_bias',
          'ema_bias_1h', 'realized_vol_annual', 'vol_ratio_1h', 'bp_1h',
          'chg_1h', 'dir_1h', 'consec_dir_1h', 'stoch_k_1h',
          'stoch_cross_1h', 'rsi_1h', 'macd_hist_1h',
          'donchian_breakout_1h', 'engulfing_1h', 'liq_score', 'liq_bias',
          'oi_chg_pct', 'ls_long_pct', 'z_drift_6h', 'rv_ratio_15m',
          'kc_pct_5m', 'kc_bo_5m', 'kc_pct_15m', 'kc_bo_15m',
          'donch_breakout_5m', 'donch_pos_5m', 'donch_breakout_15m',
          'donch_pos_15m', 'stoch_cross_5m', 'stoch_cross_15m',
          'kalman_velocity_5m', 'kalman_residual_5m', 'hurst_exponent_5m',
          'ou_theta_5m', 'kalman_velocity_15m', 'kalman_residual_15m',
          'hurst_exponent_15m', 'ou_theta_15m', 'arima_forecast_15m']
SLOPE_BASES = ['stoch_k_5m', 'stoch_k_15m', 'rsi_1h', 'vwap_dist', 'bp_5m',
               'vol_ratio', 'realized_vol_annual', 'kc_pct_5m', 'oi_chg_pct']


def build_frame():
    df = pd.read_csv(ARCHIVE, low_memory=False)
    df['dt'] = pd.to_datetime(df['logged_at'].astype(str)
                              .str.replace(r'\+00:00$', '', regex=True),
                              errors='coerce', utc=True, format='mixed')
    df = df.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)
    for c in set(STATIC + SLOPE_BASES + ['strike', 'spot', 'resolved_yes']):
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        df['z_moneyness'] = (np.log(df['strike'] / df['spot'])
                             / np.sqrt(df['tau_minutes'].clip(lower=1)))
    # causal d/slope features from the archive's own scan history
    ts = df['dt'].astype('int64') / 1e9
    nc = {}
    for tag, sec in [('15', 900), ('45', 2700), ('120', 7200)]:
        idx = np.searchsorted(ts, ts - sec, side='right') - 1
        valid = idx >= 0
        pv = np.where(valid, df['spot'].values[np.clip(idx, 0, None)], np.nan)
        dp = pd.Series((df['spot'].values / pv - 1) * 100, index=df.index)
        nc[f'dprice_{tag}'] = dp
        for c in SLOPE_BASES:
            pr = np.where(valid, df[c].values[np.clip(idx, 0, None)], np.nan)
            d = df[c].values - pr
            nc[f'D{tag}_{c}'] = d
            nc[f'S{tag}_{c}'] = np.clip(d / dp.replace(0, np.nan), -50, 50)
    df = pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)
    # pm-path (backfill; causal candle-derived)
    try:
        bf = pd.read_csv(BACKFILL)
        df = df.merge(bf, on=['logged_at', 'contract_ticker'], how='left')
        df['pm_path_drift'] = pd.to_numeric(df['pm_path_drift_bf'],
                                            errors='coerce')
        df['pm_path_vr3'] = pd.to_numeric(df['pm_path_vr3_bf'],
                                          errors='coerce')
    except Exception:
        df['pm_path_drift'] = np.nan
        df['pm_path_vr3'] = np.nan
    df = df[df['resolved_yes'].notna()
            & df['p_market'].between(0.03, 0.97)].copy()
    df['y'] = (df['resolved_yes'] == 1).astype(int)
    feats = (STATIC + ['z_moneyness', 'pm_path_drift', 'pm_path_vr3']
             + list(nc.keys()))
    return df, feats


def book_pnl(sub, pcol):
    """flat-$100 fee-net book: edge>=0.04 both sides, keep-first."""
    fee = 0.07 * sub['p_market'] * (1 - sub['p_market'])
    ey = sub[pcol] - sub['p_market'] - fee
    en = sub['p_market'] - sub[pcol] - fee
    side = np.where(ey >= en, 'yes', 'no')
    edge = np.maximum(ey, en)
    q = sub[edge >= 0.04].copy()
    q['side'] = side[edge >= 0.04]
    q = q.sort_values('dt').drop_duplicates('contract_ticker', keep='first')
    c = np.where(q['side'] == 'yes', q['p_market'], 1 - q['p_market'])
    w = np.where(q['side'] == 'yes', q['y'] == 1, q['y'] == 0)
    f = 0.07 * q['p_market'] * (1 - q['p_market'])
    pnl = np.where(w, 100 * (1 - c) / c, -100.0) - (100 / c) * f
    return len(q), float(pnl.sum()), (float(np.mean(w)) if len(q) else np.nan,
                                      float(np.mean(c)) if len(q) else np.nan)


def fit(train, val, feats, seed):
    m = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0,
                      random_state=seed, verbose=-1)
    m.fit(train[feats], train['y'],
          eval_set=[(val[feats], val['y'])],
          eval_metric='binary_logloss',
          callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)])
    return m


def main():
    df, feats = build_frame()
    print(f'frame: {len(df)} rows, {len(feats)} features, '
          f'{df["dt"].min().date()} -> {df["dt"].max().date()}')
    T = lambda s: pd.Timestamp(s, tz='UTC')
    # walk-forward origins (val tail of train for early stop)
    origins = [('O1', T('2026-07-01'), T('2026-07-15')),
               ('O2', T('2026-07-15'), T('2026-08-01')),
               ('O3', T('2026-08-01'), T('2026-08-10'))]
    print('\n== walk-forward (6 seeds each; test-book flat-$100 fee-net) ==')
    wf_ok = 0
    for name, t0, t1 in origins:
        tr = df[df['dt'] < t0]
        va = tr[tr['dt'] >= t0 - pd.Timedelta('7D')]
        trc = tr[tr['dt'] < t0 - pd.Timedelta('7D')]
        te = df[(df['dt'] >= t0) & (df['dt'] < t1)]
        nets = []
        for seed in range(6):
            m = fit(trc, va, feats, seed)
            te2 = te.copy()
            te2['p'] = m.predict_proba(te[feats])[:, 1]
            n, net, (wr, cost) = book_pnl(te2, 'p')
            nets.append(net)
        med = float(np.median(nets))
        pos = sum(1 for x in nets if x > 0)
        print(f'  {name} test {t0.date()}..{t1.date()}: seeds median '
              f'${med:+,.0f}  positive {pos}/6  [{", ".join(f"{x:+,.0f}" for x in nets)}]')
        if pos >= 4:
            wf_ok += 1
    print(f'walk-forward origins passing (>=4/6 seeds positive): {wf_ok}/3')
    # final: train < 08-10, HOLDOUT 08-10+ untouched
    t0 = T('2026-08-10')
    tr = df[df['dt'] < t0]
    va = tr[tr['dt'] >= t0 - pd.Timedelta('7D')]
    trc = tr[tr['dt'] < t0 - pd.Timedelta('7D')]
    ho = df[df['dt'] >= t0]
    print(f'\n== FINAL (train<08-10 n={len(trc)}, holdout 08-10+ n={len(ho)}) ==')
    results = []
    for seed in range(6):
        m = fit(trc, va, feats, seed)
        hv = ho.copy()
        hv['p'] = m.predict_proba(ho[feats])[:, 1]
        n, net, (wr, cost) = book_pnl(hv, 'p')
        # calibration slope on holdout (want ~1.0)
        x = hv['p'] - 0.5
        slope = float(np.polyfit(x, hv['y'] - 0.5, 1)[0]) if len(hv) else np.nan
        vp = m.predict_proba(va[feats])[:, 1]
        vll = float(-(va['y'] * np.log(np.clip(vp, 1e-9, 1))
                      + (1 - va['y']) * np.log(np.clip(1 - vp, 1e-9, 1))).mean())
        results.append((seed, m, net, n, wr, cost, slope, vll))
        print(f'  seed {seed}: holdout book n={n} ${net:+,.0f} WR={wr:.1%} '
              f'avg-cost={cost:.2f} calib-slope={slope:.2f} val-ll={vll:.4f}')
    # production reference on the SAME holdout scans
    ho2 = ho.copy()
    ho2['p_prod'] = pd.to_numeric(ho2['p_model_yes'], errors='coerce')
    ho2 = ho2[ho2['p_prod'].notna()]
    n, net, (wr, cost) = book_pnl(ho2, 'p_prod')
    print(f'  PRODUCTION ref same holdout: n={n} ${net:+,.0f} WR={wr:.1%}')
    # median-val-loss seed selection (niche precedent)
    results.sort(key=lambda r: r[7])
    pick = results[len(results) // 2]
    print(f'\nselected seed {pick[0]} (median val-loss): holdout '
          f'${pick[2]:+,.0f} / n={pick[3]}')
    # [2026-08-25 TRAINING-ENVELOPE CONVENTION (user call): every saved
    # model ships its training-data feature quantiles so OOD/input-drift
    # checking is automatic forever (the T0 staleness tell needs these;
    # reconstructing envelopes after the fact is archaeology).]
    env = {c: trc[c].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                              ).round(6).to_dict()
           for c in feats if pd.api.types.is_numeric_dtype(trc[c])}
    payload = {'model': pick[1], 'features': feats,
               'trained': '2026-08-25', 'train_end': '2026-08-10',
               'protocol': '6seed-medianval, wf O1-O3, holdout 08-10+',
               'train_envelope': env}
    out = 'models/lgbm_15m_eth_retrain_20260825.pkl'
    with open(out, 'wb') as f:
        pickle.dump(payload, f)
    print('saved', out)


if __name__ == '__main__':
    main()
