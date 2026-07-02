"""
Train 3-state GaussianHMM regime models for BTC at 1h and 15m resolution.
Matches the format of hmm_3state_btc.pkl (daily model).
"""

import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_ret, realized_vol (20-bar std), ret_5bar."""
    out = pd.DataFrame(index=df.index)
    out['log_ret']      = np.log(df['close'] / df['close'].shift(1))
    out['realized_vol'] = out['log_ret'].rolling(20, min_periods=10).std()
    out['ret_5bar']     = np.log(df['close'] / df['close'].shift(5))
    return out

def assign_state_names(model: GaussianHMM) -> dict:
    """Map state IDs to Bull/Bear/Sideways based on mean log_ret."""
    mean_log_ret = model.means_[:, 0]   # first feature is log_ret
    sorted_ids = np.argsort(mean_log_ret)  # ascending
    bear_id     = sorted_ids[0]
    sideways_id = sorted_ids[1]
    bull_id     = sorted_ids[2]
    return {bear_id: 'Bear', sideways_id: 'Sideways', bull_id: 'Bull'}

def bic_score(model: GaussianHMM, X: np.ndarray) -> float:
    """BIC = -2 * log_likelihood + k * log(n)."""
    n = len(X)
    n_features = X.shape[1]
    n_states = model.n_components
    # free params: startprob (n-1), transmat (n*(n-1)), means (n*f), covars (n*f*f for full)
    k = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features * n_features
    ll = model.score(X)
    return -2 * ll * n + k * np.log(n)

def print_validation(label: str, model: GaussianHMM, feats: pd.DataFrame,
                     state_to_name: dict, X: np.ndarray, states: np.ndarray):
    print(f"\n{'='*60}")
    print(f"  MODEL: {label}")
    print(f"{'='*60}")

    n_states = model.n_components
    feature_cols = list(feats.columns)

    # 1. State means
    print("\n[1] State means (log_ret, realized_vol, ret_5bar):")
    print(f"  {'State':<10} {'Name':<10} {'log_ret':>12} {'realized_vol':>14} {'ret_5bar':>12}")
    for sid in range(n_states):
        name = state_to_name[sid]
        means = model.means_[sid]
        print(f"  {sid:<10} {name:<10} {means[0]:>12.6f} {means[1]:>14.6f} {means[2]:>12.6f}")

    # 2. State label mapping
    print("\n[2] State label mapping:")
    for sid, name in state_to_name.items():
        print(f"  State {sid} → {name}")

    # 3. Occupancy %
    print("\n[3] State occupancy %:")
    total = len(states)
    for sid in range(n_states):
        count = np.sum(states == sid)
        name = state_to_name[sid]
        print(f"  {name:<10} (state {sid}): {count:>6} bars  {100*count/total:.1f}%")

    # 4. Transition matrix
    print("\n[4] Transition matrix:")
    header = "         " + "".join(f"  →{state_to_name[s]:<9}" for s in range(n_states))
    print(f"  {header}")
    for i in range(n_states):
        row_name = state_to_name[i]
        row_str = "".join(f"  {model.transmat_[i, j]:.6f}  " for j in range(n_states))
        print(f"  {row_name:<10} {row_str}")

    # 5. BIC
    ll = model.score(X)
    bic = bic_score(model, X)
    print(f"\n[5] Log-likelihood: {ll * len(X):.2f}   BIC: {bic:.2f}")

    # 6. Last 10 state predictions
    print("\n[6] Last 10 state predictions:")
    last_idx   = feats.index[-10:]
    last_states = states[-10:]
    for ts, sid in zip(last_idx, last_states):
        print(f"  {ts}  state={sid}  ({state_to_name[sid]})")

    # 7. Current regime
    current_state = states[-1]
    print(f"\n[7] Current regime (most recent bar): {state_to_name[current_state]}  (state {current_state})")


def train_and_save(label: str, df: pd.DataFrame, save_path: str):
    # Build features
    feats = build_features(df)
    feats = feats.rename(columns={'ret_5bar': 'ret_5bar'})  # explicit
    feats = feats.dropna()
    print(f"\n{label}: {len(feats)} bars after dropna (from {len(df)} raw bars)")

    X = feats.values  # shape (N, 3)

    # Train
    model = GaussianHMM(n_components=3, covariance_type='full', n_iter=500, random_state=42)
    model.fit(X)

    # Predict states
    states = model.predict(X)

    # Assign names
    state_to_name = assign_state_names(model)

    # BIC
    ll = model.score(X)
    bic = bic_score(model, X)

    # Print validation
    print_validation(label, model, feats, state_to_name, X, states)

    # Save payload (match daily pkl exactly, with ret_5bar as feature name)
    feature_cols = ['log_ret', 'realized_vol', 'ret_5bar']
    payload = {
        'model':         model,
        'state_to_name': state_to_name,
        'feature_cols':  feature_cols,
        'n_states':      3,
        'train_end':     str(feats.index[-1].date()),
        'log_likelihood': ll * len(X),    # total log-likelihood (matches daily convention)
        'bic':           bic,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\nSaved → {save_path}")
    return model, feats, state_to_name, states, X


def runtime_feasibility_check(label: str, feats_full: pd.DataFrame,
                               model: GaussianHMM, state_to_name: dict,
                               n_bars: int):
    """Simulate what the runner does: feed last n_bars to model.predict."""
    print(f"\n{'─'*60}")
    print(f"  RUNTIME FEASIBILITY CHECK: {label} (last {n_bars} bars)")
    print(f"{'─'*60}")

    recent = feats_full.tail(n_bars)
    feature_cols = ['log_ret', 'realized_vol', 'ret_5bar']
    X_live = recent[feature_cols].dropna().values

    if len(X_live) == 0:
        print("  ERROR: no valid bars after dropna")
        return

    states_live = model.predict(X_live)
    current = states_live[-1]
    regime  = state_to_name[current]

    print(f"  Bars fed to model:  {len(X_live)}")
    print(f"  Current regime:     {regime}  (state {current})")
    print(f"  Last 5 regimes:     {[state_to_name[s] for s in states_live[-5:]]}")
    print(f"  Result type:        {type(regime).__name__}")
    assert isinstance(regime, str), "regime must be a string"
    print(f"  PASS — returned valid regime string '{regime}'")


# ─────────────────────────────────────────────
# 1h MODEL
# ─────────────────────────────────────────────

print("\nLoading 1h parquet...")
df_1h = pd.read_parquet(
    '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_BTCUSDT_1h_1970-01-01_2026-05-28.parquet'
)
# Drop the placeholder 1970 row
df_1h = df_1h[df_1h.index > pd.Timestamp('2020-01-01', tz='UTC')]
print(f"1h rows after date filter: {len(df_1h)}")

model_1h, feats_1h, s2n_1h, states_1h, X_1h = train_and_save(
    label     = '3-state GaussianHMM @ 1h',
    df        = df_1h,
    save_path = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_1h.pkl',
)

# ─────────────────────────────────────────────
# 15m MODEL  (resample 1m → 15m)
# ─────────────────────────────────────────────

print("\nLoading 1m parquet and resampling to 15m...")
df_1m = pd.read_parquet(
    '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data/binanceus_BTCUSDT_1m_1970-01-01_2026-05-28.parquet'
)
# Drop placeholder row and filter
df_1m = df_1m[df_1m.index > pd.Timestamp('2020-01-01', tz='UTC')]

df_15m = df_1m.resample('15min').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum',
}).dropna(subset=['close'])

print(f"1m rows after filter: {len(df_1m)}")
print(f"15m rows after resample: {len(df_15m)}")

model_15m, feats_15m, s2n_15m, states_15m, X_15m = train_and_save(
    label     = '3-state GaussianHMM @ 15m',
    df        = df_15m,
    save_path = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_15m.pkl',
)

# ─────────────────────────────────────────────
# Runtime feasibility checks
# ─────────────────────────────────────────────

runtime_feasibility_check('1h model',  feats_1h,  model_1h,  s2n_1h,  n_bars=30)
runtime_feasibility_check('15m model', feats_15m, model_15m, s2n_15m, n_bars=100)

# ─────────────────────────────────────────────
# Verify saved pkls can be reloaded
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("  RELOAD VERIFICATION")
print(f"{'='*60}")
for path in [
    '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_1h.pkl',
    '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc_15m.pkl',
]:
    with open(path, 'rb') as f:
        p = pickle.load(f)
    print(f"\n{path.split('/')[-1]}:")
    print(f"  keys:          {list(p.keys())}")
    print(f"  n_states:      {p['n_states']}")
    print(f"  feature_cols:  {p['feature_cols']}")
    print(f"  train_end:     {p['train_end']}")
    print(f"  log_likelihood:{p['log_likelihood']:.2f}")
    print(f"  bic:           {p['bic']:.2f}")
    print(f"  state_to_name: {p['state_to_name']}")

print("\nDone.")
