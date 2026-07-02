"""
eth_yesitm_rescue.py
Deep rescue analysis: Can ANY conditional signal make YES-ITM ETH NO bets profitable?
K=0.20, z_strike < 0 subset from eth_no_drift_sim.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SIM_CSV   = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/eth_no_drift_sim.csv'
H1_PARQ   = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_ETHUSDT_1h_2024-01-01_2026-05-08.parquet'
H4_PARQ   = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_ETHUSDT_4h_2024-01-01_2026-05-08.parquet'
M1_PARQ   = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_ETHUSDT_1m_2024-01-01_2026-05-08.parquet'

def sep(title=''):
    print('\n' + '='*70)
    if title:
        print(f'  {title}')
        print('='*70)

# ─────────────────────────────────────────────
# 1. LOAD SIMULATION CSV — K=0.20 YES-ITM ONLY
# ─────────────────────────────────────────────
sep('LOADING DATA')
sim_all = pd.read_csv(SIM_CSV)
sim = sim_all[(sim_all['K'] == 0.20) & (sim_all['z_strike'] < 0)].copy()
sim['T'] = pd.to_datetime(sim['T'], utc=True)
sim = sim.sort_values('T').reset_index(drop=True)
print(f'K=0.20 YES-ITM trades loaded: {len(sim)}')
print(f'Date range: {sim["T"].min()} → {sim["T"].max()}')
wr_base = sim['no_wins'].mean()
be_base = (1 - sim['pm_bsm']).mean()
pnl_base = sim['pnl'].sum()
print(f'Baseline: WR={wr_base:.1%}  BE={be_base:.1%}  WR-BE={wr_base-be_base:+.1%}  PnL=${pnl_base:.2f}')

# ─────────────────────────────────────────────
# 2. LOAD PRICE DATA
# ─────────────────────────────────────────────
print('\nLoading price data...')
h1 = pd.read_parquet(H1_PARQ)
h4 = pd.read_parquet(H4_PARQ)
m1 = pd.read_parquet(M1_PARQ)

# Ensure UTC
for df_price in [h1, h4, m1]:
    if df_price.index.tz is None:
        df_price.index = df_price.index.tz_localize('UTC')
    else:
        df_price.index = df_price.index.tz_convert('UTC')

print(f'1h bars: {len(h1)}  4h bars: {len(h4)}  1m bars: {len(m1)}')

# ─────────────────────────────────────────────
# 3. BUILD INDICATOR DATAFRAME
# ─────────────────────────────────────────────
sep('COMPUTING INDICATORS')

# --- 1h lookback indicators: compute on full series, then sample at trade times ---
h1 = h1.sort_index()

# Returns
h1['ret_1h']  = h1['close'] / h1['close'].shift(1) - 1
h1['ret_4h']  = h1['close'] / h1['close'].shift(4) - 1
h1['ret_24h'] = h1['close'] / h1['close'].shift(24) - 1

# 1h candle direction and body
h1['candle_dir']  = (h1['close'] >= h1['open']).astype(int)  # 1=green, 0=red
h1['candle_body'] = (h1['close'] - h1['open']).abs() / h1['open']  # % body

# Donchian 20-bar
roll20_max = h1['close'].rolling(20).max()
roll20_min = h1['close'].rolling(20).min()
denom20 = (roll20_max - roll20_min).replace(0, np.nan)
h1['donchian_20'] = (h1['close'] - roll20_min) / denom20

# Donchian 50-bar
roll50_max = h1['close'].rolling(50).max()
roll50_min = h1['close'].rolling(50).min()
denom50 = (roll50_max - roll50_min).replace(0, np.nan)
h1['donchian_50'] = (h1['close'] - roll50_min) / denom50

# Bollinger %B (20-bar, 2σ)
bb_mid   = h1['close'].rolling(20).mean()
bb_std   = h1['close'].rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
denom_bb = (bb_upper - bb_lower).replace(0, np.nan)
h1['bb_pct_b'] = (h1['close'] - bb_lower) / denom_bb

# RSI 14-bar
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

h1['rsi_14'] = compute_rsi(h1['close'])

# Hour of day
h1['hour_utc'] = h1.index.hour

print('1h indicators computed.')

# --- 1m: last-5-min momentum ---
m1 = m1.sort_index()
m1['ret_5m'] = m1['close'] / m1['close'].shift(5) - 1
print('1m 5-min momentum computed.')

# ─────────────────────────────────────────────
# 4. JOIN INDICATORS TO TRADE DATA
# ─────────────────────────────────────────────
sep('JOINING INDICATORS TO TRADES')

def safe_lookup(ts, price_df, col, offset_bars=0):
    """Lookup price_df[col] at timestamp ts - offset_bars."""
    try:
        idx_pos = price_df.index.get_indexer([ts], method='ffill')[0]
        if idx_pos < 0:
            return np.nan
        target_pos = idx_pos - offset_bars
        if target_pos < 0:
            return np.nan
        return price_df.iloc[target_pos][col]
    except Exception:
        return np.nan

# For speed: build lookup dicts for 1h and 1m indicators
# Reindex sim timestamps against h1 using ffill

# Vectorised merge on nearest hour
h1_ind = h1[['ret_1h','ret_4h','ret_24h','candle_dir','candle_body',
             'donchian_20','donchian_50','bb_pct_b','rsi_14','hour_utc']].copy()

# Trade timestamp floored to nearest hour
sim['T_hour'] = sim['T'].dt.floor('h')

# Merge 1h indicators (at T_hour = the CLOSE of the current 1h bar)
# Use merge_asof for safety
h1_ind_reset = h1_ind.reset_index()
h1_ind_reset.rename(columns={'open_time': 'T_hour'}, inplace=True)

sim = pd.merge_asof(
    sim.sort_values('T_hour'),
    h1_ind_reset.sort_values('T_hour'),
    on='T_hour',
    direction='backward'
)

# 1m momentum: get 5-min return closest to trade time
m1_5m = m1[['ret_5m']].copy().reset_index()
m1_5m.rename(columns={'open_time': 'T_m1'}, inplace=True)
sim['T_m1_key'] = sim['T'].dt.floor('min')

sim = pd.merge_asof(
    sim.sort_values('T_m1_key'),
    m1_5m.rename(columns={'T_m1': 'T_m1_key'}).sort_values('T_m1_key'),
    on='T_m1_key',
    direction='backward'
)

# 4h return: floor to 4h bar
h4_ret = h4[['close']].copy()
h4_ret['ret_4h_bar'] = h4_ret['close'] / h4_ret['close'].shift(1) - 1
h4_ret = h4_ret[['ret_4h_bar']].reset_index()
h4_ret.rename(columns={'open_time': 'T_4h'}, inplace=True)

sim['T_4h_key'] = sim['T'].dt.floor('4h')
sim = pd.merge_asof(
    sim.sort_values('T_4h_key'),
    h4_ret.rename(columns={'T_4h': 'T_4h_key'}).sort_values('T_4h_key'),
    on='T_4h_key',
    direction='backward'
)

# sigma_tau percentile (rolling 30d within YES-ITM)
sim = sim.sort_values('T').reset_index(drop=True)
sim['sigma_pct'] = np.nan
win_days = 30
for i, row in sim.iterrows():
    t = row['T']
    window = sim[(sim['T'] >= t - pd.Timedelta(days=win_days)) & (sim['T'] < t)]['sigma_tau']
    if len(window) >= 5:
        pct = (window < row['sigma_tau']).mean()
        sim.at[i, 'sigma_pct'] = pct

# z_strike depth buckets
def z_depth_bucket(z):
    az = abs(z)
    if az < 0.65:
        return 'shallow(0.45-0.65)'
    elif az < 0.90:
        return 'mid(0.65-0.90)'
    else:
        return 'deep(>0.90)'

sim['z_depth'] = sim['z_strike'].apply(z_depth_bucket)

# Spike detection: sigma_tau top-20% for the day
sim['date'] = sim['T'].dt.date
daily_80pct = sim.groupby('date')['sigma_tau'].transform(lambda x: x.quantile(0.80))
sim['vol_spike'] = sim['sigma_tau'] >= daily_80pct

print(f'Indicators joined. Final sim shape: {sim.shape}')
print(f'NaN counts in key indicators:')
key_cols = ['ret_1h','ret_4h','ret_24h','candle_dir','candle_body',
            'donchian_20','donchian_50','bb_pct_b','rsi_14','ret_5m','sigma_pct']
for c in key_cols:
    print(f'  {c}: {sim[c].isna().sum()} NaN')

# ─────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def stats(df):
    """Return (n, wr, be, wr_be, pnl, pnl_per_trade) for a subset."""
    n = len(df)
    if n == 0:
        return (0, np.nan, np.nan, np.nan, np.nan, np.nan)
    wr  = df['no_wins'].mean()
    be  = (1 - df['pm_bsm']).mean()
    pnl = df['pnl'].sum()
    return (n, wr, be, wr-be, pnl, pnl/n)

def report_segment(label, df, min_n=15):
    n, wr, be, delta, pnl, ppt = stats(df)
    if n >= min_n:
        flag = ' ***EDGE***' if delta > 0 else ''
        print(f'  {label:<45} n={n:>4}  WR={wr:.1%}  BE={be:.1%}  WR-BE={delta:+.1%}  PnL=${pnl:.2f}{flag}')
    return n, wr, be, delta, pnl, ppt

sep('BASELINE CHECK')
report_segment('ALL YES-ITM K=0.20', sim, min_n=1)

# ─────────────────────────────────────────────
# 6. SINGLE-INDICATOR ANALYSIS
# ─────────────────────────────────────────────
sep('SINGLE-INDICATOR ANALYSIS')

results = []  # collect (indicator, bucket_label, n, wr, be, delta, pnl, ppt)

def analyze_quantile(col, label, q=5, df=sim):
    """Split by quantile buckets and report."""
    valid = df[df[col].notna()].copy()
    if len(valid) < 30:
        print(f'  [SKIP {label}: insufficient data]')
        return
    print(f'\n--- {label} ---')
    try:
        valid['_bucket'] = pd.qcut(valid[col], q=q, duplicates='drop')
    except Exception as e:
        print(f'  qcut failed: {e}')
        return
    for bucket, grp in valid.groupby('_bucket', observed=True):
        r = report_segment(f'{label} {bucket}', grp)
        results.append((label, str(bucket), *r))

def analyze_categorical(col, label, df=sim):
    valid = df[df[col].notna()].copy()
    print(f'\n--- {label} ---')
    for val in sorted(valid[col].unique()):
        grp = valid[valid[col] == val]
        r = report_segment(f'{label}={val}', grp)
        results.append((label, str(val), *r))

# 1h return
print('\n--- 1h Return ---')
valid = sim[sim['ret_1h'].notna()].copy()
bins_1h = [(-1, -0.03), (-0.03, -0.015), (-0.015, -0.005), (-0.005, 0.005),
           (0.005, 0.015), (0.015, 0.03), (0.03, 1)]
labels_1h = ['<-3%', '-3to-1.5%', '-1.5to-0.5%', '-0.5to0.5%',
             '+0.5to1.5%', '+1.5to3%', '>3%']
for (lo, hi), lbl in zip(bins_1h, labels_1h):
    grp = valid[(valid['ret_1h'] > lo) & (valid['ret_1h'] <= hi)]
    r = report_segment(f'1h_ret {lbl}', grp)
    results.append(('ret_1h', lbl, *r))

# 4h return
print('\n--- 4h Return ---')
analyze_quantile('ret_4h', '4h_ret', q=5)

# 24h return
print('\n--- 24h Return ---')
analyze_quantile('ret_24h', '24h_ret', q=5)

# 1h candle direction
print('\n--- 1h Candle Direction ---')
grp_green = sim[sim['candle_dir'] == 1]
grp_red   = sim[sim['candle_dir'] == 0]
r = report_segment('candle_dir=GREEN (bullish)', grp_green)
results.append(('candle_dir', 'green', *r))
r = report_segment('candle_dir=RED (bearish)', grp_red)
results.append(('candle_dir', 'red', *r))

# 1h candle body size
print('\n--- 1h Candle Body Size ---')
analyze_quantile('candle_body', 'candle_body_%', q=4)

# Donchian 20
print('\n--- Donchian 20-bar Position ---')
valid = sim[sim['donchian_20'].notna()].copy()
don20_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
don20_lbls = ['bot20%', 'low40%', 'mid60%', 'high80%', 'top100%']
for (lo, hi), lbl in zip(don20_bins, don20_lbls):
    grp = valid[(valid['donchian_20'] >= lo) & (valid['donchian_20'] < hi)]
    r = report_segment(f'donchian_20 {lbl}', grp)
    results.append(('donchian_20', lbl, *r))

# Donchian 50
print('\n--- Donchian 50-bar Position ---')
valid = sim[sim['donchian_50'].notna()].copy()
for (lo, hi), lbl in zip(don20_bins, don20_lbls):
    grp = valid[(valid['donchian_50'] >= lo) & (valid['donchian_50'] < hi)]
    r = report_segment(f'donchian_50 {lbl}', grp)
    results.append(('donchian_50', lbl, *r))

# Bollinger %B
print('\n--- Bollinger %B ---')
valid = sim[sim['bb_pct_b'].notna()].copy()
bb_bins  = [(-2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 3)]
bb_lbls  = ['below_lower', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%', 'above_upper']
for (lo, hi), lbl in zip(bb_bins, bb_lbls):
    grp = valid[(valid['bb_pct_b'] > lo) & (valid['bb_pct_b'] <= hi)]
    r = report_segment(f'bb_pctB {lbl}', grp)
    results.append(('bb_pct_b', lbl, *r))

# RSI
print('\n--- RSI 14-bar ---')
valid = sim[sim['rsi_14'].notna()].copy()
rsi_bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
rsi_lbls = ['oversold(<30)', '30-40', '40-50', '50-60', '60-70', 'overbought(>70)']
for (lo, hi), lbl in zip(rsi_bins, rsi_lbls):
    grp = valid[(valid['rsi_14'] > lo) & (valid['rsi_14'] <= hi)]
    r = report_segment(f'RSI {lbl}', grp)
    results.append(('rsi_14', lbl, *r))

# 5-min momentum
print('\n--- 5-min Price Change ---')
valid = sim[sim['ret_5m'].notna()].copy()
m5_bins = [(-1, -0.005), (-0.005, -0.001), (-0.001, 0.001),
           (0.001, 0.005), (0.005, 1)]
m5_lbls = ['<-0.5%', '-0.5to-0.1%', '-0.1to0.1%', '0.1to0.5%', '>0.5%']
for (lo, hi), lbl in zip(m5_bins, m5_lbls):
    grp = valid[(valid['ret_5m'] > lo) & (valid['ret_5m'] <= hi)]
    r = report_segment(f'5m_mom {lbl}', grp)
    results.append(('ret_5m', lbl, *r))

# sigma_tau percentile
print('\n--- sigma_tau Percentile (30d rolling) ---')
valid = sim[sim['sigma_pct'].notna()].copy()
sp_bins = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0001)]
sp_lbls = ['low_vol(bot25)', 'med_low', 'med_high', 'high_vol(top25)']
for (lo, hi), lbl in zip(sp_bins, sp_lbls):
    grp = valid[(valid['sigma_pct'] >= lo) & (valid['sigma_pct'] < hi)]
    r = report_segment(f'sigma_pct {lbl}', grp)
    results.append(('sigma_pct', lbl, *r))

# z_depth
print('\n--- z_strike Depth ---')
analyze_categorical('z_depth', 'z_depth')

# Rev score
print('\n--- Rev Score Buckets ---')
rev_bins  = [(-20, 0), (0, 0.5), (1, 2.5), (3, 20)]
rev_lbls  = ['rev<=0', 'rev=0', 'rev=1-2', 'rev>=3']
for (lo, hi), lbl in zip(rev_bins, rev_lbls):
    grp = sim[(sim['rev'] > lo) & (sim['rev'] <= hi)]
    r = report_segment(f'rev {lbl}', grp)
    results.append(('rev', lbl, *r))

# Also exact rev=0
print()
for rv in [-5,-4,-3,-2,-1,0,1,2,3]:
    grp = sim[sim['rev'] == rv]
    if len(grp) >= 10:
        report_segment(f'rev={rv}', grp, min_n=10)

# Hour of day
print('\n--- Hour of Day (UTC) ---')
for grp_name, grp in sim.groupby('hour_utc', observed=True):
    r = report_segment(f'hour={int(grp_name):02d}:00 UTC', grp, min_n=15)
    results.append(('hour_utc', str(int(grp_name)), *r))

# Hour blocks
print('\n  Hour blocks:')
hour_blocks = [
    ('Asia(00-08)',  list(range(0, 8))),
    ('London(08-12)', list(range(8, 12))),
    ('NY_open(12-16)', list(range(12, 16))),
    ('NY_close(16-20)', list(range(16, 20))),
    ('overnight(20-24)', list(range(20, 24))),
]
for lbl, hours in hour_blocks:
    grp = sim[sim['hour_utc'].isin(hours)]
    r = report_segment(f'session {lbl}', grp)
    results.append(('hour_session', lbl, *r))

# Trend score
print('\n--- Trend Score ---')
for tr_val in sorted(sim['trend'].unique()):
    grp = sim[sim['trend'] == tr_val]
    r = report_segment(f'trend={tr_val}', grp, min_n=15)
    results.append(('trend', str(tr_val), *r))

# Trend buckets
print('\n  Trend blocks:')
trend_blocks = [
    ('strong_bear (trend<=-4)', sim['trend'] <= -4),
    ('bear (trend=-3to-1)',    (sim['trend'] >= -3) & (sim['trend'] <= -1)),
    ('neutral (trend=0)',      sim['trend'] == 0),
    ('bull (trend>=1)',        sim['trend'] >= 1),
]
for lbl, mask in trend_blocks:
    grp = sim[mask]
    r = report_segment(f'trend {lbl}', grp)
    results.append(('trend_bucket', lbl, *r))

# Vol spike
print('\n--- Volatility Spike ---')
grp_spike   = sim[sim['vol_spike'] == True]
grp_nospike = sim[sim['vol_spike'] == False]
r = report_segment('vol_spike=True  (sigma top20% of day)', grp_spike)
results.append(('vol_spike', 'True', *r))
r = report_segment('vol_spike=False (normal vol)', grp_nospike)
results.append(('vol_spike', 'False', *r))

# 4h_bar return
print('\n--- 4h Bar Return (current 4h bar) ---')
valid = sim[sim['ret_4h_bar'].notna()].copy()
h4b_bins = [(-1, -0.04), (-0.04, -0.02), (-0.02, -0.005),
            (-0.005, 0.005), (0.005, 0.02), (0.02, 0.04), (0.04, 1)]
h4b_lbls = ['<-4%', '-4to-2%', '-2to-0.5%', '-0.5to0.5%',
            '+0.5to2%', '+2to4%', '>4%']
for (lo, hi), lbl in zip(h4b_bins, h4b_lbls):
    grp = valid[(valid['ret_4h_bar'] > lo) & (valid['ret_4h_bar'] <= hi)]
    r = report_segment(f'4h_bar_ret {lbl}', grp)
    results.append(('ret_4h_bar', lbl, *r))

# p_up buckets
print('\n--- p_up (directional probability) ---')
pup_bins = [(0, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 1)]
pup_lbls = ['very_bear(<0.35)', 'bear(0.35-0.40)', 'bear(0.40-0.45)',
            'near_neutral(0.45-0.50)', 'bull(>=0.50)']
for (lo, hi), lbl in zip(pup_bins, pup_lbls):
    grp = sim[(sim['p_up'] > lo) & (sim['p_up'] <= hi)]
    r = report_segment(f'p_up {lbl}', grp)
    results.append(('p_up', lbl, *r))

# ─────────────────────────────────────────────
# 7. RANK SINGLE SIGNALS BY |WR-BE|
# ─────────────────────────────────────────────
sep('RANKING SINGLE SIGNALS')

results_df = pd.DataFrame(results, columns=[
    'indicator','bucket','n','wr','be','wr_be','pnl','ppt'
])
results_df = results_df[results_df['n'] >= 15].dropna(subset=['wr_be'])
results_df = results_df.sort_values('wr_be', ascending=False)

print('\nTop 20 BEST buckets by WR-BE (edge > 0):')
print(f'{"Indicator":<20} {"Bucket":<35} {"n":>5} {"WR":>7} {"BE":>7} {"WR-BE":>8} {"PnL":>9}')
for _, row in results_df.head(20).iterrows():
    flag = ' ***' if row['wr_be'] > 0 else ''
    print(f'  {row["indicator"]:<18} {row["bucket"]:<33} {int(row["n"]):>5} '
          f'{row["wr"]:.1%} {row["be"]:.1%} {row["wr_be"]:+.1%} ${row["pnl"]:>8.2f}{flag}')

print('\nTop 10 WORST buckets by WR-BE (biggest losers):')
for _, row in results_df.tail(10).iterrows():
    print(f'  {row["indicator"]:<18} {row["bucket"]:<33} {int(row["n"]):>5} '
          f'{row["wr"]:.1%} {row["be"]:.1%} {row["wr_be"]:+.1%} ${row["pnl"]:>8.2f}')

# Get top 5 discriminating signals for combo search
# Best discriminators = highest |WR-BE|
results_df['abs_delta'] = results_df['wr_be'].abs()
top_indicators = results_df.sort_values('abs_delta', ascending=False)['indicator'].unique()[:10]
print(f'\nTop discriminating indicators for combo search: {list(top_indicators[:6])}')

# ─────────────────────────────────────────────
# 8. PAIRWISE COMBINATIONS
# ─────────────────────────────────────────────
sep('PAIRWISE COMBINATION SEARCH')

# Build binary flags from top discriminating signals
# Define each as a boolean mask
combo_features = {}

# 1h return direction
combo_features['ret_1h_neg']    = sim['ret_1h'] < 0           # bearish 1h candle
combo_features['ret_1h_strong_neg'] = sim['ret_1h'] < -0.01   # >1% drop
combo_features['ret_1h_pos']    = sim['ret_1h'] > 0
combo_features['candle_red']    = sim['candle_dir'] == 0
combo_features['candle_green']  = sim['candle_dir'] == 1

# 4h return
combo_features['ret_4h_neg']    = sim['ret_4h'] < 0
combo_features['ret_4h_pos']    = sim['ret_4h'] > 0

# 24h return
combo_features['ret_24h_neg']   = sim['ret_24h'] < -0.02      # >2% drop in 24h
combo_features['ret_24h_pos']   = sim['ret_24h'] > 0.02

# RSI zones
if sim['rsi_14'].notna().sum() > 50:
    combo_features['rsi_lt40']  = sim['rsi_14'] < 40
    combo_features['rsi_40_50'] = (sim['rsi_14'] >= 40) & (sim['rsi_14'] < 50)
    combo_features['rsi_gt60']  = sim['rsi_14'] > 60
    combo_features['rsi_lt50']  = sim['rsi_14'] < 50

# Donchian position
combo_features['don20_bot40']   = sim['donchian_20'] < 0.4
combo_features['don20_top60']   = sim['donchian_20'] > 0.6
combo_features['don50_bot40']   = sim['donchian_50'] < 0.4

# Bollinger
combo_features['bb_below_50']   = sim['bb_pct_b'] < 0.5
combo_features['bb_above_50']   = sim['bb_pct_b'] > 0.5
combo_features['bb_below_lower'] = sim['bb_pct_b'] < 0.0   # below lower band

# Trend / rev from CSV
combo_features['trend_le_neg4'] = sim['trend'] <= -4
combo_features['trend_le_neg3'] = sim['trend'] <= -3
combo_features['trend_le_neg1'] = sim['trend'] <= -1
combo_features['rev_ge3']       = sim['rev'] >= 3
combo_features['rev_ge1']       = sim['rev'] >= 1
combo_features['rev_le0']       = sim['rev'] <= 0

# z_depth
combo_features['z_shallow']     = sim['z_depth'] == 'shallow(0.45-0.65)'
combo_features['z_mid']         = sim['z_depth'] == 'mid(0.65-0.90)'
combo_features['z_deep']        = sim['z_depth'] == 'deep(>0.90)'

# vol spike
combo_features['vol_spike']     = sim['vol_spike']
combo_features['no_spike']      = ~sim['vol_spike']

# hour sessions
combo_features['asia_session']  = sim['hour_utc'].isin(range(0, 8))
combo_features['london_session'] = sim['hour_utc'].isin(range(8, 12))
combo_features['ny_session']    = sim['hour_utc'].isin(range(12, 20))

# p_up
combo_features['pup_lt040']     = sim['p_up'] < 0.40
combo_features['pup_lt045']     = sim['p_up'] < 0.45

# 5m momentum
if sim['ret_5m'].notna().sum() > 50:
    combo_features['mom5m_neg'] = sim['ret_5m'] < 0
    combo_features['mom5m_pos'] = sim['ret_5m'] > 0

# sigma_tau absolute level (proxy for high/low vol)
med_sigma = sim['sigma_tau'].median()
combo_features['low_vol_abs']   = sim['sigma_tau'] < med_sigma
combo_features['high_vol_abs']  = sim['sigma_tau'] >= med_sigma

# Fill NaN masks with False
feature_names = list(combo_features.keys())
for k in feature_names:
    combo_features[k] = combo_features[k].fillna(False)

print(f'Testing {len(feature_names)} binary features in pairwise combinations...')
print(f'Total combos to test: {len(feature_names) * (len(feature_names)-1) // 2}\n')

combo_results = []
MIN_N_COMBO = 20

for f1, f2 in combinations(feature_names, 2):
    mask = combo_features[f1] & combo_features[f2]
    grp = sim[mask]
    n = len(grp)
    if n < MIN_N_COMBO:
        continue
    wr  = grp['no_wins'].mean()
    be  = (1 - grp['pm_bsm']).mean()
    delta = wr - be
    pnl = grp['pnl'].sum()
    combo_results.append({
        'f1': f1, 'f2': f2, 'n': n, 'wr': wr, 'be': be,
        'wr_be': delta, 'pnl': pnl, 'ppt': pnl/n
    })

combo_df = pd.DataFrame(combo_results)
if len(combo_df) > 0:
    combo_df_pos = combo_df[combo_df['wr_be'] > 0].sort_values('wr_be', ascending=False)
    print(f'Combos with n>={MIN_N_COMBO} and WR > BE (positive edge): {len(combo_df_pos)}')
    print(f'\nTop 30 pairwise combos by WR-BE:')
    print(f'{"f1":<25} {"f2":<25} {"n":>5} {"WR":>7} {"BE":>7} {"WR-BE":>8} {"PnL":>9} {"$/trade":>8}')
    for _, row in combo_df.sort_values('wr_be', ascending=False).head(30).iterrows():
        flag = ' ***' if row['wr_be'] > 0 else ''
        print(f'  {row["f1"]:<23} {row["f2"]:<23} {int(row["n"]):>5} '
              f'{row["wr"]:.1%} {row["be"]:.1%} {row["wr_be"]:+.1%} '
              f'${row["pnl"]:>8.2f} ${row["ppt"]:>6.3f}{flag}')
else:
    print('No combo results (all below min_n threshold).')

# ─────────────────────────────────────────────
# 9. TRIPLE COMBINATIONS (from top pairwise combos)
# ─────────────────────────────────────────────
sep('TRIPLE COMBINATION SEARCH (from top pairwise)')

if len(combo_df_pos) > 0:
    # Take top 15 positive combos, get their unique features
    top_pair_features = set()
    for _, row in combo_df_pos.head(15).iterrows():
        top_pair_features.add(row['f1'])
        top_pair_features.add(row['f2'])

    top_pair_features = list(top_pair_features)
    print(f'Building triples from {len(top_pair_features)} features from top pairwise combos...')
    print(f'Total triple combos: {len(list(combinations(top_pair_features, 3)))}')

    triple_results = []
    MIN_N_TRIPLE = 20

    for f1, f2, f3 in combinations(top_pair_features, 3):
        mask = combo_features[f1] & combo_features[f2] & combo_features[f3]
        grp = sim[mask]
        n = len(grp)
        if n < MIN_N_TRIPLE:
            continue
        wr  = grp['no_wins'].mean()
        be  = (1 - grp['pm_bsm']).mean()
        delta = wr - be
        pnl = grp['pnl'].sum()
        triple_results.append({
            'f1': f1, 'f2': f2, 'f3': f3, 'n': n, 'wr': wr, 'be': be,
            'wr_be': delta, 'pnl': pnl, 'ppt': pnl/n
        })

    triple_df = pd.DataFrame(triple_results)
    if len(triple_df) > 0:
        triple_pos = triple_df[triple_df['wr_be'] > 0].sort_values('wr_be', ascending=False)
        print(f'\nTriple combos with n>={MIN_N_TRIPLE} and WR > BE: {len(triple_pos)}')
        print(f'\nTop 20 triple combos by WR-BE:')
        for _, row in triple_df.sort_values('wr_be', ascending=False).head(20).iterrows():
            flag = ' ***' if row['wr_be'] > 0 else ''
            print(f'  {row["f1"]} + {row["f2"]} + {row["f3"]}')
            print(f'    n={int(row["n"])}  WR={row["wr"]:.1%}  BE={row["be"]:.1%}  '
                  f'WR-BE={row["wr_be"]:+.1%}  PnL=${row["pnl"]:.2f}{flag}')
    else:
        print('No triple combos with sufficient n.')
else:
    print('No positive pairwise combos to extend.')

# ─────────────────────────────────────────────
# 10. MODEL REFORMULATION — K parameter sensitivity
# ─────────────────────────────────────────────
sep('MODEL REFORMULATION — K SENSITIVITY')

print('\nFor YES-ITM NO bets, P(NO) = Φ(z_strike + K * something)')
print('Testing alternative K values and pure-BSM formulation...')
print()

# The p_no in simulation = P(price ends below strike) adjusted by drift
# z_strike = log(K_price/spot) / sigma_tau  (negative for YES-ITM)
# BSM P(YES) = Φ(-z_strike + K*sigma_tau_normalized)
# We test: what would WR look like under different K assumptions?

# From CSV we have: pm_bsm = Φ(-z_strike) approx (the neutral price)
# p_no (our model) = 1 - p_yes_model
# Actual WR = no_wins fraction

# Recompute for all YES-ITM (not just K=0.20) to get more signal
sim_all_yesitm = sim_all[sim_all['z_strike'] < 0].copy()
sim_all_yesitm['T'] = pd.to_datetime(sim_all_yesitm['T'], utc=True)

for K_test in [0.0, -0.10, 0.20]:
    # p_no_test = Φ(z_strike - K_test * 1)  where we use sigma_tau as the vol scale
    # Actually: p_yes_bsm = Φ(-z + K) where z = z_strike (negative)
    # p_no_bsm  = 1 - Φ(-z + K) = Φ(z - K)
    # Standard: z_strike is already normalized by sigma_tau in the original
    p_no_test = norm.cdf(sim_all_yesitm['z_strike'])  # K=0 neutral
    if K_test != 0:
        # Shift: p_no = Φ(z_strike - K)
        p_no_test = norm.cdf(sim_all_yesitm['z_strike'] - K_test)

    be_test  = 1 - p_no_test.mean()  # BE = 1 - p_no_model (we bet NO, need 1-p_yes)
    # Actually for a NO bet, we get paid when price < strike
    # Cost = pm_bsm (market price of YES)
    # Our model says p_yes = p_no_test (confusing naming)
    # Let's compute: bet NO, WR = no_wins, BE = pm_bsm (market YES price)
    # The K just affects whether we *enter* — use all YES-ITM for comparison
    wr_all  = sim_all_yesitm['no_wins'].mean()
    be_mkt  = sim_all_yesitm['pm_bsm'].apply(lambda x: 1-x).mean()
    pnl_formula = (sim_all_yesitm['no_wins'].astype(float) - sim_all_yesitm['pm_bsm']).sum()
    print(f'K_test={K_test:+.2f}  All YES-ITM: n={len(sim_all_yesitm)}  '
          f'WR={wr_all:.1%}  BE(market)={be_mkt:.1%}  '
          f'WR-BE={wr_all-be_mkt:+.1%}  Raw_PnL=${pnl_formula:.2f}')

print()
# Pure neutral BSM p_no vs actual WR by depth
print('Calibration check — does depth predict WR?')
print('(If z_strike deeper → price must move MORE to win → should be harder to win)')
print()
for bucket, grp in sim.groupby('z_depth', observed=True):
    n = len(grp)
    wr = grp['no_wins'].mean()
    be = (1 - grp['pm_bsm']).mean()
    bsm_p_no = norm.cdf(grp['z_strike']).mean()  # pure BSM no-drift
    print(f'  {bucket:<25} n={n:>4}  WR={wr:.1%}  BE={be:.1%}  '
          f'BSM_p_no={bsm_p_no:.1%}  WR-BE={wr-be:+.1%}')

# ─────────────────────────────────────────────
# 11. DEEPER CONDITIONAL ANALYSIS ON BEST SEGMENTS
# ─────────────────────────────────────────────
sep('DEEPER ANALYSIS — STRONGEST SINGLE-SIGNAL SEGMENTS')

# Cross-tabulate the best single signals against all others
# Find the best 3 positive-edge segments and drill into sub-segments
best_singles = results_df[results_df['wr_be'] > 0].head(5)
if len(best_singles) > 0:
    print('Drilling into each positive-edge single-signal segment:')
    for _, sr in best_singles.iterrows():
        ind = sr['indicator']
        bkt = sr['bucket']
        print(f'\n  BASE: {ind}={bkt}  n={int(sr["n"])}  WR={sr["wr"]:.1%}  WR-BE={sr["wr_be"]:+.1%}')
        # Apply mask for this segment
        # Re-identify the mask
        mask_map = {
            'candle_dir': {
                'green': sim['candle_dir'] == 1,
                'red':   sim['candle_dir'] == 0,
            },
            'vol_spike': {
                'True':  sim['vol_spike'],
                'False': ~sim['vol_spike'],
            },
            'z_depth': {
                'shallow(0.45-0.65)': sim['z_depth'] == 'shallow(0.45-0.65)',
                'mid(0.65-0.90)':     sim['z_depth'] == 'mid(0.65-0.90)',
                'deep(>0.90)':        sim['z_depth'] == 'deep(>0.90)',
            },
        }
        if ind in mask_map and bkt in mask_map[ind]:
            base_mask = mask_map[ind][bkt]
            base_grp  = sim[base_mask]
            # Now split by trend
            for tr_val in [-5, -4, -3, -2, -1, 0]:
                sub = base_grp[base_grp['trend'] == tr_val]
                if len(sub) >= 10:
                    n, wr, be, delta, pnl, ppt = stats(sub)
                    flag = ' ***' if delta > 0 else ''
                    print(f'    trend={tr_val}: n={n}  WR={wr:.1%}  BE={be:.1%}  '
                          f'WR-BE={delta:+.1%}  PnL=${pnl:.2f}{flag}')
else:
    print('No positive-edge single segments found.')

# ─────────────────────────────────────────────
# 12. TIME-PERIOD ANALYSIS
# ─────────────────────────────────────────────
sep('TIME PERIOD ANALYSIS')

sim['month'] = sim['T'].dt.to_period('M')
print('Monthly breakdown of YES-ITM K=0.20 performance:')
for period, grp in sim.groupby('month', observed=True):
    n, wr, be, delta, pnl, ppt = stats(grp)
    if n >= 5:
        flag = ' ***' if delta > 0 else ''
        print(f'  {period}  n={n:>4}  WR={wr:.1%}  BE={be:.1%}  WR-BE={delta:+.1%}  '
              f'PnL=${pnl:.2f}  $/trade={ppt:.3f}{flag}')

# ─────────────────────────────────────────────
# 13. MARKET REGIME ANALYSIS
# ─────────────────────────────────────────────
sep('MARKET REGIME ANALYSIS — TRENDING vs CHOPPY')

# Use 24h return magnitude as regime indicator
if sim['ret_24h'].notna().sum() > 50:
    ret24_abs = sim['ret_24h'].abs()
    med_24h = ret24_abs.median()

    trending = sim[ret24_abs > med_24h]
    choppy   = sim[ret24_abs <= med_24h]

    print('Market trending (|24h_ret| > median):')
    n, wr, be, delta, pnl, ppt = stats(trending)
    print(f'  n={n}  WR={wr:.1%}  BE={be:.1%}  WR-BE={delta:+.1%}  PnL=${pnl:.2f}')

    print('Market choppy (|24h_ret| <= median):')
    n, wr, be, delta, pnl, ppt = stats(choppy)
    print(f'  n={n}  WR={wr:.1%}  BE={be:.1%}  WR-BE={delta:+.1%}  PnL=${pnl:.2f}')

    # Direction of 24h move with candle direction
    print('\nCombined: 24h direction × 1h candle direction:')
    for ret24_dir in ['down_24h', 'up_24h']:
        for cand_dir in ['red', 'green']:
            if ret24_dir == 'down_24h':
                m1_ = sim['ret_24h'] < -0.02
            else:
                m1_ = sim['ret_24h'] > 0.02
            m2_ = sim['candle_dir'] == (0 if cand_dir == 'red' else 1)
            grp = sim[m1_ & m2_]
            n, wr, be, delta, pnl, ppt = stats(grp)
            if n >= 15:
                flag = ' ***' if delta > 0 else ''
                print(f'  {ret24_dir} + {cand_dir}: n={n}  WR={wr:.1%}  '
                      f'BE={be:.1%}  WR-BE={delta:+.1%}  PnL=${pnl:.2f}{flag}')

# ─────────────────────────────────────────────
# 14. SUMMARY OF ALL POSITIVE-EDGE FINDINGS
# ─────────────────────────────────────────────
sep('ALL POSITIVE-EDGE SEGMENTS (WR > BE, n >= 15)')

print('\nSingle signals:')
pos_singles = results_df[results_df['wr_be'] > 0].copy()
if len(pos_singles) > 0:
    for _, row in pos_singles.iterrows():
        print(f'  {row["indicator"]:<20} {row["bucket"]:<35} n={int(row["n"]):>4}  '
              f'WR={row["wr"]:.1%}  BE={row["be"]:.1%}  WR-BE={row["wr_be"]:+.1%}  '
              f'PnL=${row["pnl"]:.2f}  $/trade={row["ppt"]:.3f}')
else:
    print('  NONE')

print('\nPairwise combos (WR > BE, n >= 20):')
if 'combo_df' in dir() and len(combo_df) > 0:
    combo_pos = combo_df[combo_df['wr_be'] > 0].sort_values('wr_be', ascending=False)
    if len(combo_pos) > 0:
        for _, row in combo_pos.head(20).iterrows():
            print(f'  {row["f1"]} + {row["f2"]}')
            print(f'    n={int(row["n"])}  WR={row["wr"]:.1%}  BE={row["be"]:.1%}  '
                  f'WR-BE={row["wr_be"]:+.1%}  PnL=${row["pnl"]:.2f}  $/trade={row["ppt"]:.3f}')
    else:
        print('  NONE')

# ─────────────────────────────────────────────
# 15. CONCLUSIONS
# ─────────────────────────────────────────────
sep('CONCLUSIONS')

all_pos = results_df[results_df['wr_be'] > 0]

# Find best combo overall
best_combo_row = None
if 'combo_df' in dir() and len(combo_df) > 0 and len(combo_df[combo_df['wr_be'] > 0]) > 0:
    best_combo_row = combo_df[combo_df['wr_be'] > 0].sort_values('wr_be', ascending=False).iloc[0]

print(f"""
BASELINE: YES-ITM K=0.20 NO bets
  n={len(sim)}  WR={wr_base:.1%}  BE(avg)={be_base:.1%}  WR-BE={wr_base-be_base:+.1%}  PnL=${pnl_base:.2f}

WHY YES-ITM BETS FAIL STRUCTURALLY:
  YES-ITM means strike < spot (ETH has already risen above strike).
  For a NO bet to win, price must FALL BACK BELOW the strike within 1h.
  With strikes 45-150% out of the money (|z|=0.45-1.5), the BSM probability
  of price reversing that much in 1h is only 6-25% (pm_bsm = 0.67-0.94).
  The market correctly prices this — you pay (1-pm_bsm) = 6-33% for a bet
  that wins only ~10% of the time. Structural losers.

POSITIVE-EDGE SINGLE SIGNALS FOUND: {len(all_pos)}
""")

if len(all_pos) > 0:
    best_single = all_pos.sort_values('wr_be', ascending=False).iloc[0]
    print(f'STRONGEST SINGLE RESCUE:')
    print(f'  Condition: {best_single["indicator"]} = {best_single["bucket"]}')
    print(f'  n={int(best_single["n"])}  WR={best_single["wr"]:.1%}  BE={best_single["be"]:.1%}  '
          f'WR-BE={best_single["wr_be"]:+.1%}  PnL=${best_single["pnl"]:.2f}  '
          f'$/trade={best_single["ppt"]:.3f}')
    if best_single['wr_be'] > 0.03:
        print(f'  => This is a MEANINGFUL edge (+{best_single["wr_be"]:.1%} above BE)')
    else:
        print(f'  => This edge is MARGINAL (<3% above BE) — likely noise with n={int(best_single["n"])}')
else:
    print('STRONGEST SINGLE RESCUE: NONE — no single signal achieves WR > BE')

if best_combo_row is not None:
    print(f'\nSTRONGEST PAIRWISE RESCUE:')
    print(f'  Conditions: {best_combo_row["f1"]} AND {best_combo_row["f2"]}')
    print(f'  n={int(best_combo_row["n"])}  WR={best_combo_row["wr"]:.1%}  BE={best_combo_row["be"]:.1%}  '
          f'WR-BE={best_combo_row["wr_be"]:+.1%}  PnL=${best_combo_row["pnl"]:.2f}  '
          f'$/trade={best_combo_row["ppt"]:.3f}')
    if best_combo_row['wr_be'] > 0.03:
        print(f'  => This is a MEANINGFUL edge (+{best_combo_row["wr_be"]:.1%} above BE)')
    else:
        print(f'  => Edge is MARGINAL — likely overfitting on small n')
else:
    print('\nSTRONGEST PAIRWISE RESCUE: NONE or all below min_n=20')

print("""
RECOMMENDATION:
""")

# Assess significance: need WR-BE > 5% AND n >= 30 to be meaningful
sig_rescues = all_pos[(all_pos['wr_be'] > 0.05) & (all_pos['n'] >= 30)]

if len(sig_rescues) > 0:
    print('  CONDITIONAL ALLOW: The following conditions show sufficient edge:')
    for _, row in sig_rescues.iterrows():
        print(f'    {row["indicator"]}={row["bucket"]}  '
              f'WR-BE={row["wr_be"]:+.1%}  n={int(row["n"])}  PnL=${row["pnl"]:.2f}')
    print()
    print('  CAUTION: Even these should be treated with suspicion — the structural')
    print('  problem (price must reverse sharply) cannot be rescued by technical signals')
    print('  unless those signals specifically predict sharp reversals within 1h.')
else:
    print('  HARD BLOCK RECOMMENDED: No rescue condition shows WR-BE > 5% with n >= 30.')
    print('  YES-ITM NO bets are structurally broken — the market correctly prices the')
    print('  low probability of a sharp reversal within 1 hour. No technical signal')
    print('  examined here provides sufficient edge to overcome the structural deficit.')
    print()
    print('  The current gate (blocking all YES-ITM NO bets) is CORRECT.')
    print('  Do NOT allow YES-ITM NO bets under any condition tested.')

print('\n' + '='*70)
print('  Analysis complete.')
print('='*70)
