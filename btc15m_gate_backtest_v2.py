"""
BTC 15m Gate Backtest v2 — Correct deduplication (latest logged_at per contract)
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os

# ── 1. Load & concatenate CSVs ──────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Load CSVs")
print("=" * 70)

ARCHIVE = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_btc15m_archive_20260525_1432_pre_branched_drift.csv"
CURRENT = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_btc15m.csv"

df_a = pd.read_csv(ARCHIVE, low_memory=False)
df_b = pd.read_csv(CURRENT, low_memory=False)

print(f"Archive raw rows: {len(df_a):,}")
print(f"Current raw rows: {len(df_b):,}")

# Align columns — fill missing with NaN
all_cols = list(dict.fromkeys(df_a.columns.tolist() + df_b.columns.tolist()))
df_a = df_a.reindex(columns=all_cols)
df_b = df_b.reindex(columns=all_cols)

df = pd.concat([df_a, df_b], ignore_index=True)
print(f"Combined raw rows: {len(df):,}")

# ── 2. Filter: decision=="trade" AND resolved_yes not null/empty ─────────────
print("\n" + "=" * 70)
print("STEP 2: Filter trade + resolved")
print("=" * 70)

df['resolved_yes'] = pd.to_numeric(df['resolved_yes'], errors='coerce')
df_filt = df[(df['decision'] == 'trade') & df['resolved_yes'].notna()].copy()
print(f"After filter (trade + resolved): {len(df_filt):,} rows")
print(f"Unique contracts before dedup: {df_filt['contract_ticker'].nunique():,}")

# ── 3. Dedup: keep latest logged_at per contract_ticker ─────────────────────
print("\n" + "=" * 70)
print("STEP 3: Dedup — keep latest logged_at per contract")
print("=" * 70)

df_filt['logged_at'] = pd.to_datetime(df_filt['logged_at'], utc=True, errors='coerce')
df_filt = df_filt.sort_values('logged_at')
df_dedup = df_filt.drop_duplicates(subset=['contract_ticker'], keep='last').copy()
df_dedup = df_dedup.reset_index(drop=True)

n_contracts = len(df_dedup)
date_min = df_dedup['logged_at'].min()
date_max = df_dedup['logged_at'].max()
yes_count = int((df_dedup['side'] == 'yes').sum())
no_count  = int((df_dedup['side'] == 'no').sum())

print(f"n_contracts (after dedup): {n_contracts}")
print(f"Date range: {date_min.date()} → {date_max.date()}")
print(f"YES trades: {yes_count}  |  NO trades: {no_count}")
print(f"YES %: {yes_count/n_contracts*100:.1f}%  |  NO %: {no_count/n_contracts*100:.1f}%")

# ── 4. HMM Backfill ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: HMM Backfill")
print("=" * 70)

HMM_1H  = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_1h.pkl"
HMM_15M = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_15m.pkl"
BINANCE_1H = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_BTCUSDT_1h_1970-01-01_2026-05-28.parquet"
BINANCE_1M = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_BTCUSDT_1m_1970-01-01_2026-05-28.parquet"

def load_hmm(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # obj is a dict with keys: model, state_to_name, feature_cols, n_states, ...
    return obj

def compute_features(ohlcv, feature_cols):
    """Compute features matching the HMM's feature_cols"""
    close = ohlcv['close'].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(20).std()
    ret_5 = np.log(close / close.shift(5))
    feat = pd.DataFrame({
        'log_ret': log_ret,
        'realized_vol': realized_vol,
        'ret_5bar': ret_5
    })
    # Only keep the cols the model was trained on (in order)
    feat = feat[feature_cols]
    return feat

def run_hmm_on_ohlcv(ohlcv, hmm_bundle):
    """Returns a labelled Series of regimes indexed by ohlcv.index"""
    model = hmm_bundle['model']
    state_to_name = hmm_bundle['state_to_name']
    feature_cols = hmm_bundle['feature_cols']
    feat = compute_features(ohlcv, feature_cols)
    feat = feat.dropna()
    X = feat.values
    states = model.predict(X)
    state_series = pd.Series(states, index=feat.index)
    label_series = state_series.map(state_to_name)
    return label_series

# Load 1h data
print("Loading 1h Binance data...")
df_1h = pd.read_parquet(BINANCE_1H)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)
df_1h = df_1h.sort_index()
print(f"  1h rows: {len(df_1h):,}, range: {df_1h.index[0].date()} → {df_1h.index[-1].date()}")

# Load HMM 1h
print("Loading HMM 1h model...")
hmm_1h = load_hmm(HMM_1H)
print(f"  HMM 1h n_states: {hmm_1h['n_states']}")
print(f"  HMM 1h state_to_name: {hmm_1h['state_to_name']}")
print(f"  HMM 1h feature_cols: {hmm_1h['feature_cols']}")

# Predict 1h regimes
print("Predicting 1h regimes...")
labels_1h = run_hmm_on_ohlcv(df_1h, hmm_1h)
labels_1h.name = 'hmm_1h'
print(f"  1h label distribution:\n{labels_1h.value_counts().to_string()}")

# Load 1m data and resample to 15m
print("\nLoading 1m Binance data and resampling to 15m...")
df_1m = pd.read_parquet(BINANCE_1M)
df_1m.index = pd.to_datetime(df_1m.index, utc=True)
df_1m = df_1m.sort_index()
print(f"  1m rows: {len(df_1m):,}")

# Resample 1m → 15m OHLCV
df_15m = df_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna(subset=['close'])
print(f"  15m rows after resample: {len(df_15m):,}")

# Load HMM 15m
print("Loading HMM 15m model...")
hmm_15m = load_hmm(HMM_15M)
print(f"  HMM 15m n_states: {hmm_15m['n_states']}")
print(f"  HMM 15m state_to_name: {hmm_15m['state_to_name']}")
print(f"  HMM 15m feature_cols: {hmm_15m['feature_cols']}")

# Predict 15m regimes
print("Predicting 15m regimes...")
labels_15m = run_hmm_on_ohlcv(df_15m, hmm_15m)
labels_15m.name = 'hmm_15m'
print(f"  15m label distribution:\n{labels_15m.value_counts().to_string()}")

# ── 5. Merge HMM labels into df_dedup ───────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Merge HMM labels into trade data")
print("=" * 70)

# Prepare merge keys
df_dedup_sorted = df_dedup.sort_values('logged_at').copy()
# Drop rows where logged_at is NaT (can't merge on null key)
n_before = len(df_dedup_sorted)
df_dedup_sorted = df_dedup_sorted[df_dedup_sorted['logged_at'].notna()]
print(f"Dropped {n_before - len(df_dedup_sorted)} rows with null logged_at")

# 1h labels: reset index to get timestamp column
labels_1h_df = labels_1h.reset_index()
labels_1h_df.columns = ['ts_1h', 'hmm_1h']
labels_1h_df = labels_1h_df.sort_values('ts_1h')

# 15m labels
labels_15m_df = labels_15m.reset_index()
labels_15m_df.columns = ['ts_15m', 'hmm_15m']
labels_15m_df = labels_15m_df.sort_values('ts_15m')

# merge_asof backward on logged_at
df_dedup_sorted = pd.merge_asof(
    df_dedup_sorted,
    labels_1h_df,
    left_on='logged_at', right_on='ts_1h',
    direction='backward'
)
df_dedup_sorted = pd.merge_asof(
    df_dedup_sorted,
    labels_15m_df,
    left_on='logged_at', right_on='ts_15m',
    direction='backward'
)

df = df_dedup_sorted.copy()

print(f"HMM 1h label dist after merge:\n{df['hmm_1h'].value_counts().to_string()}")
print(f"\nHMM 15m label dist after merge:\n{df['hmm_15m'].value_counts().to_string()}")
print(f"\nHMM 1h NaN: {df['hmm_1h'].isna().sum()}")
print(f"HMM 15m NaN: {df['hmm_15m'].isna().sum()}")

# ── 6. Regime label comparison ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Regime Label Distribution Comparison (old stored vs HMM new)")
print("=" * 70)

print("\nOld stored markov_regime_1h:")
print(df['markov_regime_1h'].value_counts(dropna=False).to_string())

print("\nNew HMM hmm_1h:")
print(df['hmm_1h'].value_counts(dropna=False).to_string())

print("\nOld stored markov_regime_15m:")
print(df['markov_regime_15m'].value_counts(dropna=False).to_string())

print("\nNew HMM hmm_15m:")
print(df['hmm_15m'].value_counts(dropna=False).to_string())

# ── 7. Core metrics helpers ──────────────────────────────────────────────────

def metrics(subset, label=""):
    n = len(subset)
    if n == 0:
        print(f"  {label}: 0 trades")
        return
    wins = int(subset['would_win'].sum()) if 'would_win' in subset.columns else int((subset['would_pnl'] > 0).sum())
    pnl  = subset['would_pnl'].sum()
    wr   = wins / n * 100
    yes_n = int((subset['side'] == 'yes').sum())
    no_n  = int((subset['side'] == 'no').sum())
    print(f"  {label}: n={n}, WR={wr:.1f}% ({wins}W/{n-wins}L), PnL=${pnl:+.2f}, YES={yes_n}, NO={no_n}")

def gate_report(kept, removed, label=""):
    """Report kept and removed with wins/losses breakdown"""
    n_kept = len(kept)
    n_rem  = len(removed)
    wins_rem  = int((removed['would_pnl'] > 0).sum())
    losses_rem = n_rem - wins_rem
    pnl_kept  = kept['would_pnl'].sum()
    pnl_rem   = removed['would_pnl'].sum()
    wr_kept   = (kept['would_pnl'] > 0).mean() * 100 if n_kept > 0 else 0
    print(f"  Kept: {n_kept} trades, WR={wr_kept:.1f}%, PnL=${pnl_kept:+.2f}")
    print(f"  Removed: {n_rem} trades ({wins_rem}W / {losses_rem}L), PnL of removed=${pnl_rem:+.2f}")
    print(f"  Net PnL delta (removed from baseline): ${-pnl_rem:+.2f}")

# ── 8. S0 Baseline ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S0 BASELINE")
print("=" * 70)
metrics(df, "S0 All trades")

# Breakdown by side
yes_trades = df[df['side'] == 'yes']
no_trades  = df[df['side'] == 'no']
metrics(yes_trades, "  YES trades")
metrics(no_trades,  "  NO trades")

# ── 9. S1 Gate A only ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S1: Gate A only — block NO where stoch_k_15m>=80 AND ema_bias==1")
print("=" * 70)

mask_a = (df['side'] == 'no') & (df['stoch_k_15m'] >= 80) & (df['ema_bias'] == 1)
s1_removed = df[mask_a]
s1_kept    = df[~mask_a]
print(f"Gate A fired on: {mask_a.sum()} trades")
gate_report(s1_kept, s1_removed, "S1")
metrics(s1_kept, "S1 final")

# ── 10. S2 Gate A + rescue ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S2: Gate A + rescue — keep if bp_15m<0.35 AND dir_15m==-1")
print("=" * 70)

rescue_mask = (df['bp_15m'] < 0.35) & (df['dir_15m'] == -1)
# Remove = gate fires AND NOT rescued
mask_a_no_rescue = mask_a & ~rescue_mask
s2_removed = df[mask_a_no_rescue]
s2_kept    = df[~mask_a_no_rescue]

print(f"Gate A fires: {mask_a.sum()}")
print(f"  Rescued by rescue condition: {(mask_a & rescue_mask).sum()}")
print(f"  Final removed (gate A, not rescued): {mask_a_no_rescue.sum()}")
gate_report(s2_kept, s2_removed, "S2")
metrics(s2_kept, "S2 final")

# ── 11. S3 HMM new labels ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S3: HMM new labels — block YES where hmm_1h==Bear OR (hmm_15m==Bear AND composite_p_up>0.488)")
print("=" * 70)

mask_s3_1h = (df['side'] == 'yes') & (df['hmm_1h'] == 'Bear')
mask_s3_15m = (df['side'] == 'yes') & (df['hmm_15m'] == 'Bear') & (df['composite_p_up'] > 0.488)
mask_s3 = mask_s3_1h | mask_s3_15m

s3_removed = df[mask_s3]
s3_kept    = df[~mask_s3]

print(f"Removed by hmm_1h==Bear (YES only): {mask_s3_1h.sum()}")
print(f"  Wins blocked: {int((df[mask_s3_1h]['would_pnl'] > 0).sum())}")
print(f"  Losses blocked: {int((df[mask_s3_1h]['would_pnl'] <= 0).sum())}")
print(f"Removed by hmm_15m==Bear & p_up>0.488 (YES only): {mask_s3_15m.sum()}")
print(f"  Wins blocked: {int((df[mask_s3_15m]['would_pnl'] > 0).sum())}")
print(f"  Losses blocked: {int((df[mask_s3_15m]['would_pnl'] <= 0).sum())}")
print(f"Total removed (union): {mask_s3.sum()}")
gate_report(s3_kept, s3_removed, "S3")
metrics(s3_kept, "S3 final")

# ── 12. S3b Old Markov labels ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S3b: OLD Markov labels — same gates with markov_regime_1h / markov_regime_15m")
print("=" * 70)

mask_s3b_1h  = (df['side'] == 'yes') & (df['markov_regime_1h'] == 'Bear')
mask_s3b_15m = (df['side'] == 'yes') & (df['markov_regime_15m'] == 'Bear') & (df['composite_p_up'] > 0.488)
mask_s3b = mask_s3b_1h | mask_s3b_15m

s3b_removed = df[mask_s3b]
s3b_kept    = df[~mask_s3b]

print(f"Removed by markov_regime_1h==Bear (YES only): {mask_s3b_1h.sum()}")
print(f"  Wins blocked: {int((df[mask_s3b_1h]['would_pnl'] > 0).sum())}")
print(f"  Losses blocked: {int((df[mask_s3b_1h]['would_pnl'] <= 0).sum())}")
print(f"Removed by markov_regime_15m==Bear & p_up>0.488 (YES only): {mask_s3b_15m.sum()}")
print(f"  Wins blocked: {int((df[mask_s3b_15m]['would_pnl'] > 0).sum())}")
print(f"  Losses blocked: {int((df[mask_s3b_15m]['would_pnl'] <= 0).sum())}")
print(f"Total removed (union): {mask_s3b.sum()}")
gate_report(s3b_kept, s3b_removed, "S3b")
metrics(s3b_kept, "S3b final")

# ── 13. S4 Gate A rescue + HMM new combined ─────────────────────────────────
print("\n" + "=" * 70)
print("S4: Gate A rescue (S2) + HMM new (S3) combined")
print("=" * 70)

mask_s4 = mask_a_no_rescue | mask_s3
s4_removed = df[mask_s4]
s4_kept    = df[~mask_s4]

print(f"Total removed: {mask_s4.sum()} (Gate A: {mask_a_no_rescue.sum()}, HMM: {mask_s3.sum()}, overlap: {(mask_a_no_rescue & mask_s3).sum()})")
gate_report(s4_kept, s4_removed, "S4")
metrics(s4_kept, "S4 final")

# ── 14. Weekly breakdown ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WEEKLY BREAKDOWN — S0 Baseline and S2 (Gate A + rescue)")
print("=" * 70)

df['week'] = df['logged_at'].dt.to_period('W')
s2_df = df[~mask_a_no_rescue].copy()

weeks_s0 = df.groupby('week')
weeks_s2 = s2_df.groupby('week')

print("\n  Week              | n  | WR%  |  PnL     | [S2] n  | WR%  |  PnL")
print("  " + "-" * 70)
for week in sorted(df['week'].unique()):
    g0 = df[df['week'] == week]
    g2 = s2_df[s2_df['week'] == week]
    n0 = len(g0); wr0 = (g0['would_pnl'] > 0).mean() * 100 if n0 else 0; pnl0 = g0['would_pnl'].sum()
    n2 = len(g2); wr2 = (g2['would_pnl'] > 0).mean() * 100 if n2 else 0; pnl2 = g2['would_pnl'].sum()
    print(f"  {str(week):<17} | {n0:3d} | {wr0:4.1f} | {pnl0:+8.2f} | {n2:5d} | {wr2:4.1f} | {pnl2:+8.2f}")

print("\n  Totals:")
n0t = len(df); wr0t = (df['would_pnl'] > 0).mean()*100; pnl0t = df['would_pnl'].sum()
n2t = len(s2_df); wr2t = (s2_df['would_pnl'] > 0).mean()*100; pnl2t = s2_df['would_pnl'].sum()
print(f"  S0: n={n0t}, WR={wr0t:.1f}%, PnL=${pnl0t:+.2f}")
print(f"  S2: n={n2t}, WR={wr2t:.1f}%, PnL=${pnl2t:+.2f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
