"""
Backfill multi-timeframe HMM features for all BTC paper trades using Binance OHLCV data.
Computes: 5m, 15m, 1h candle indicators at each scan timestamp.
Output: results/btc_hmm_training_data.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

BASE = Path(__file__).parent

# ── Load Binance data ──────────────────────────────────────────────────────────
print("Loading Binance 1m data...")
f1m = sorted(glob.glob(str(BASE / 'data/binanceus_BTCUSDT_1m_1970-01-01_*.parquet')))[-1]
import pyarrow.parquet as _pq
df1m = _pq.read_table(f1m).to_pandas()
df1m.index = pd.to_datetime(df1m.index, utc=True)
df1m = df1m.sort_index()
df1m.index = df1m.index.floor('T')
print(f"  1m bars: {len(df1m):,}  ({df1m.index.min().date()} → {df1m.index.max().date()})")

# Resample to 5m, 15m, 1h
def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open':  'first', 'high': 'max', 'low': 'min',
        'close': 'last',  'volume': 'sum'
    }).dropna()

print("Resampling...")
df5m  = resample_ohlcv(df1m, '5T')
df15m = resample_ohlcv(df1m, '15T')
df1h  = resample_ohlcv(df1m, '1H')
print(f"  5m: {len(df5m):,}  15m: {len(df15m):,}  1h: {len(df1h):,}")

# ── Technical indicator functions ─────────────────────────────────────────────
def buying_pressure(df, n=14):
    """Buying pressure: (close - low) / (high - low), smoothed."""
    hl = df['high'] - df['low']
    hl = hl.replace(0, np.nan)
    bp = (df['close'] - df['low']) / hl
    return bp.rolling(n, min_periods=1).mean()

def stoch_k(df, k=14, smooth=3):
    """Stochastic %K."""
    lowest  = df['low'].rolling(k, min_periods=1).min()
    highest = df['high'].rolling(k, min_periods=1).max()
    rng = highest - lowest
    rng = rng.replace(0, np.nan)
    raw_k = 100 * (df['close'] - lowest) / rng
    return raw_k.rolling(smooth, min_periods=1).mean()

def rsi(df, n=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def body_size(df):
    """Candle body as fraction of high-low range."""
    hl = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['open']).abs() / hl)

def direction(df):
    return np.sign(df['close'] - df['open'])

def upper_wick(df):
    hl = (df['high'] - df['low']).replace(0, np.nan)
    body_top = df[['open','close']].max(axis=1)
    return (df['high'] - body_top) / hl

def lower_wick(df):
    hl = (df['high'] - df['low']).replace(0, np.nan)
    body_bot = df[['open','close']].min(axis=1)
    return (body_bot - df['low']) / hl

def consec_dir(df, window=4):
    d = np.sign(df['close'] - df['open'])
    result = []
    for i in range(len(d)):
        if i < 1:
            result.append(0)
            continue
        w = d.iloc[max(0,i-window+1):i+1]
        if (w == 1).all():  result.append(int((w==1).sum()))
        elif (w == -1).all(): result.append(-int((w==-1).sum()))
        else: result.append(0)
    return pd.Series(result, index=df.index, dtype=float)

def vwap_dist(df):
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return (df['close'] - vwap) / vwap

def ema_dist(df, span=20):
    ema = df['close'].ewm(span=span, adjust=False).mean()
    return (df['close'] - ema) / ema

# ── Pre-compute indicators on all data ────────────────────────────────────────
print("Computing indicators...")
for df, label in [(df5m,'5m'), (df15m,'15m'), (df1h,'1h')]:
    df['bp']         = buying_pressure(df)
    df['stoch_k']    = stoch_k(df)
    df['body']       = body_size(df)
    df['dir']        = direction(df)
    df['upper_wick'] = upper_wick(df)
    df['lower_wick'] = lower_wick(df)
    df['chg']        = df['close'].pct_change() * 100
    df['consec_dir'] = consec_dir(df)
    if label == '1h':
        df['rsi']      = rsi(df)
        df['vwap_dist']= vwap_dist(df)
        df['ema20_dist']= ema_dist(df, 20)
    print(f"  {label}: done")

# 1m chg for 1m and 5m sub-minute
df1m['chg_1m'] = df1m['close'].pct_change() * 100

# ── Load paper trades ──────────────────────────────────────────────────────────
sources = {
    'btc15m': BASE / 'results' / 'paper_trades_btc15m.csv',
    'btc1h':  BASE / 'results' / 'paper_trades.csv',
}
dfs = []
for src, path in sources.items():
    df = pd.read_csv(path, low_memory=False)
    df['logged_at'] = pd.to_datetime(df['logged_at'], errors='coerce', utc=True)
    df = df.dropna(subset=['logged_at']).sort_values('logged_at')
    df['source'] = src
    # Normalize outcome columns
    for c in ['resolved_yes','p_market','bet_amount','side','decision']:
        if c in df.columns:
            df[c] = df[c] if c in ['side','decision'] else pd.to_numeric(df[c], errors='coerce')
    dfs.append(df)
    print(f"Loaded {src}: {len(df)} rows")

all_trades = pd.concat(dfs, ignore_index=True).sort_values('logged_at')
print(f"Combined: {len(all_trades)} rows")

# ── Merge indicators by timestamp lookup ──────────────────────────────────────
_FREQ_MAP = {'T': '1min', '5T': '5min', '15T': '15min', '1H': '1h', 'min': '1min'}
def lookup(ts, df_ind, col, floor_freq):
    """Get indicator value at the closed bar before ts."""
    pd_freq = floor_freq.replace('T','min').replace('H','h') if floor_freq != 'T' else '1min'
    bar_ts = ts.floor(pd_freq)
    td_str = _FREQ_MAP.get(floor_freq, floor_freq.replace('T','min').replace('H','h'))
    bar_ts = bar_ts - pd.Timedelta(td_str)
    try:
        return float(df_ind.loc[bar_ts, col])
    except (KeyError, TypeError):
        # Try nearest previous
        idx = df_ind.index.get_indexer([bar_ts], method='pad')[0]
        if idx >= 0:
            return float(df_ind.iloc[idx][col])
        return np.nan

print("\nBackfilling features (this may take ~1 min)...")
rows = []
for i, row in enumerate(all_trades.itertuples()):
    ts = row.logged_at
    r = {'logged_at': ts, 'source': row.source}
    # Decision/outcome
    r['decision']     = getattr(row, 'decision', None)
    r['side']         = getattr(row, 'side', None)
    r['resolved_yes'] = getattr(row, 'resolved_yes', np.nan)
    r['p_market']     = getattr(row, 'p_market', np.nan)
    r['bet_amount']   = getattr(row, 'bet_amount', np.nan)

    # 5m features
    r['bp_5m']         = lookup(ts, df5m,  'bp',         '5T')
    r['chg_5m']        = lookup(ts, df5m,  'chg',        '5T')
    r['stoch_k_5m']    = lookup(ts, df5m,  'stoch_k',    '5T')
    r['body_5m']       = lookup(ts, df5m,  'body',       '5T')
    r['dir_5m']        = lookup(ts, df5m,  'dir',        '5T')
    r['upper_wick_5m'] = lookup(ts, df5m,  'upper_wick', '5T')
    r['lower_wick_5m'] = lookup(ts, df5m,  'lower_wick', '5T')
    r['chg_1m']        = lookup(ts, df1m,  'chg_1m',     'T')

    # 15m features
    r['bp_15m']         = lookup(ts, df15m, 'bp',         '15T')
    r['chg_15m']        = lookup(ts, df15m, 'chg',        '15T')
    r['stoch_k_15m']    = lookup(ts, df15m, 'stoch_k',    '15T')
    r['body_15m']       = lookup(ts, df15m, 'body',       '15T')
    r['dir_15m']        = lookup(ts, df15m, 'dir',        '15T')
    r['upper_wick_15m'] = lookup(ts, df15m, 'upper_wick', '15T')
    r['lower_wick_15m'] = lookup(ts, df15m, 'lower_wick', '15T')
    r['consec_dir_15m'] = lookup(ts, df15m, 'consec_dir', '15T')

    # 1h features
    r['bp_1h']          = lookup(ts, df1h,  'bp',         '1H')
    r['chg_1h']         = lookup(ts, df1h,  'chg',        '1H')
    r['stoch_k_1h']     = lookup(ts, df1h,  'stoch_k',    '1H')
    r['body_1h']        = lookup(ts, df1h,  'body',       '1H')
    r['dir_1h']         = lookup(ts, df1h,  'dir',        '1H')
    r['rsi_1h']         = lookup(ts, df1h,  'rsi',        '1H')
    r['vwap_dist']      = lookup(ts, df1h,  'vwap_dist',  '1H')
    r['ema20_dist_1h']  = lookup(ts, df1h,  'ema20_dist', '1H')
    r['consec_dir_1h']  = lookup(ts, df1h,  'consec_dir', '1H')

    rows.append(r)
    if i % 500 == 0:
        print(f"  {i}/{len(all_trades)}...")

out = pd.DataFrame(rows)
out_path = BASE / 'results' / 'btc_hmm_training_data.csv'
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} rows → {out_path}")

# Quick fill check
feat_cols = [c for c in out.columns if c not in
             ['logged_at','source','decision','side','resolved_yes','p_market','bet_amount']]
fills = out[feat_cols].notna().mean().sort_values()
print("\nFill rates:")
for c, v in fills.items():
    print(f"  {c:<25}: {v*100:.1f}%")
