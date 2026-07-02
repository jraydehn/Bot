"""
Multi-timeframe HMM for BTC candle regime detection.
Observation vector: 5m / 15m / 1h features at each scan tick.
Trains Gaussian HMM, selects n_states via BIC, labels states by WR/PnL.
Saves: hmm_btc_multitf.pkl
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Feature columns ────────────────────────────────────────────────────────────
FEATURES_5M  = ['bp_5m', 'chg_5m', 'chg_1m', 'stoch_k_5m']
FEATURES_15M = ['bp_15m', 'chg_15m', 'body_15m', 'dir_15m',
                'stoch_k_15m', 'consec_dir_15m',
                'upper_wick_15m', 'lower_wick_15m']
FEATURES_1H  = ['bp_1h', 'chg_1h', 'stoch_k_1h', 'dir_1h',
                'rsi_1h', 'consec_dir_1h', 'vwap_dist', 'ema20_dist_1h']

ALL_FEATURES = FEATURES_5M + FEATURES_15M + FEATURES_1H
N_FEATURES   = len(ALL_FEATURES)

print(f"Feature vector: {N_FEATURES} dims")
print(f"  5m  ({len(FEATURES_5M)}): {FEATURES_5M}")
print(f"  15m ({len(FEATURES_15M)}): {FEATURES_15M}")
print(f"  1h  ({len(FEATURES_1H)}): {FEATURES_1H}")

# ── Load data ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
csv15 = BASE / 'results' / 'paper_trades_btc15m.csv'
csv1h = BASE / 'results' / 'paper_trades.csv'

def load_and_prep(path, all_features):
    df = pd.read_csv(path, low_memory=False)
    df['logged_at'] = pd.to_datetime(df['logged_at'], errors='coerce', utc=True)
    df = df.dropna(subset=['logged_at']).sort_values('logged_at').reset_index(drop=True)
    # Keep rows that have most features populated
    available = [c for c in all_features if c in df.columns]
    for c in available:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, available

df15, avail15 = load_and_prep(csv15, ALL_FEATURES)
df1h, avail1h = load_and_prep(csv1h, ALL_FEATURES)

print(f"\n15m CSV: {len(df15)} rows, {len(avail15)}/{N_FEATURES} features available")
print(f"1h  CSV: {len(df1h)} rows, {len(avail1h)}/{N_FEATURES} features available")
print(f"Missing in 15m: {set(ALL_FEATURES) - set(avail15)}")
print(f"Missing in 1h:  {set(ALL_FEATURES) - set(avail1h)}")

# ── Use 15m runner as primary (richest features) ───────────────────────────────
df = df15.copy()
feat_cols = [c for c in ALL_FEATURES if c in df.columns]
print(f"\nUsing {len(feat_cols)} features from 15m runner:")

# Drop rows with too many NaNs
X_raw = df[feat_cols].copy()
row_fill = X_raw.notna().mean(axis=1)
df = df[row_fill >= 0.75].reset_index(drop=True)
X_raw = df[feat_cols].copy()

# Fill remaining NaNs with column medians
for c in feat_cols:
    X_raw[c] = X_raw[c].fillna(X_raw[c].median())

print(f"Training rows after NaN filter: {len(df)}")

# ── Build time series (all scan rows, including pass/trade) ────────────────────
# Sort by time, use ALL rows — HMM needs continuous sequence
X = X_raw.values.astype(float)

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build sequence lengths (split on gaps > 10 min to avoid crossing restart gaps)
timestamps = df['logged_at'].values
gaps_sec = pd.Series(timestamps).diff().dt.total_seconds().fillna(0).values
split_mask = gaps_sec > 600  # 10-minute gap = new sequence
seq_starts = np.where(split_mask)[0].tolist()
seq_starts = [0] + seq_starts
seq_ends   = seq_starts[1:] + [len(X_scaled)]
lengths    = [e - s for s, e in zip(seq_starts, seq_ends) if e - s >= 3]
# Rebuild X with only valid sequences
valid_idx = []
for s, e in zip(seq_starts, seq_ends):
    if e - s >= 3:
        valid_idx.extend(range(s, e))
X_seq = X_scaled[valid_idx]
print(f"Sequences: {len(lengths)}, total obs: {len(X_seq)}, avg len: {np.mean(lengths):.1f}")

# ── BIC model selection ────────────────────────────────────────────────────────
print("\n=== BIC model selection ===")
bic_scores = {}
models = {}
for n in range(2, 9):
    try:
        m = GaussianHMM(n_components=n, covariance_type='diag',
                        n_iter=200, random_state=42, tol=1e-4)
        m.fit(X_seq, lengths=lengths)
        log_l = m.score(X_seq, lengths=lengths)
        n_params = n**2 + 2 * n * len(feat_cols) - 1
        bic = -2 * log_l * len(X_seq) + n_params * np.log(len(X_seq))
        bic_scores[n] = bic
        models[n] = m
        print(f"  n={n}: logL={log_l:.2f}  BIC={bic:.0f}")
    except Exception as e:
        print(f"  n={n}: FAILED — {e}")

best_n = min(bic_scores, key=bic_scores.get)
print(f"\nBest n_states = {best_n} (BIC={bic_scores[best_n]:.0f})")
model = models[best_n]

# ── Assign states to all rows ──────────────────────────────────────────────────
states = model.predict(X_seq, lengths=lengths)
df_valid = df.iloc[valid_idx].copy().reset_index(drop=True)
df_valid['hmm_state'] = states

# ── Label states by WR / PnL ──────────────────────────────────────────────────
traded = df_valid[df_valid['decision'] == 'trade'].copy()
traded['resolved_yes'] = pd.to_numeric(traded['resolved_yes'], errors='coerce')
traded['p_market']     = pd.to_numeric(traded['p_market'],     errors='coerce')
traded['bet_amount']   = pd.to_numeric(traded['bet_amount'],   errors='coerce')
traded = traded.dropna(subset=['resolved_yes', 'p_market', 'bet_amount'])

def did_win(row):
    s = str(row['side']).lower(); res = int(row['resolved_yes'])
    return (res==1 and s=='yes') or (res==0 and s=='no')
def calc_pnl(row):
    pm,amt,won,s = row['p_market'],row['bet_amount'],int(row['resolved_yes']),str(row['side']).lower()
    if s=='yes': return amt*(1-pm)/pm if won else -amt
    else: return amt*pm/(1-pm) if won==0 else -amt

traded['won'] = traded.apply(did_win, axis=1)
traded['pnl'] = traded.apply(calc_pnl, axis=1)

print(f"\n=== State labels (based on {len(traded)} resolved trades) ===")
state_stats = {}
for st in sorted(df_valid['hmm_state'].unique()):
    n_obs  = (df_valid['hmm_state'] == st).sum()
    t_sub  = traded[traded['hmm_state'] == st]
    wr  = t_sub['won'].mean() * 100 if len(t_sub) else float('nan')
    pnl = t_sub['pnl'].sum()
    ppt = t_sub['pnl'].mean() if len(t_sub) else float('nan')
    pct_obs = n_obs / len(df_valid) * 100
    state_stats[st] = {'wr': wr, 'pnl': pnl, 'ppt': ppt, 'n_obs': n_obs}
    print(f"  State {st}: obs={n_obs:>4} ({pct_obs:.0f}%)  trades={len(t_sub):>3}  "
          f"WR={wr:.1f}%  PnL=${pnl:+.2f}  $/trade=${ppt:+.2f}")

# Top feature means per state
print(f"\n=== Top features per state (standardised means) ===")
key_feats = ['bp_5m','chg_5m','stoch_k_5m','bp_15m','chg_15m',
             'stoch_k_15m','dir_15m','bp_1h','chg_1h','stoch_k_1h']
key_feats = [f for f in key_feats if f in feat_cols]
feat_idx  = [feat_cols.index(f) for f in key_feats]

means_raw = np.array([scaler.inverse_transform(model.means_)[i] for i in range(best_n)])
print(f"{'State':<8}", end='')
for f in key_feats: print(f"{f[:12]:>13}", end='')
print()
for st in range(best_n):
    print(f"  St{st}   ", end='')
    for fi in feat_idx:
        print(f"{model.means_[st][fi]:>13.3f}", end='')
    print()

# Transition matrix
print(f"\n=== Transition matrix ===")
print(pd.DataFrame(model.transmat_,
      columns=[f'→St{i}' for i in range(best_n)],
      index=[f'St{i}' for i in range(best_n)]).round(3).to_string())

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    'model':        model,
    'scaler':       scaler,
    'feat_cols':    feat_cols,
    'n_states':     best_n,
    'state_stats':  state_stats,
    'bic_scores':   bic_scores,
}
pkl_path = BASE / 'hmm_btc_multitf.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"\nSaved → {pkl_path}")
