"""
HMM 3-state BTC Daily Regime Backfill + Gate B Simulation
==========================================================
Steps:
  1. Load pre-trained HMM from hmm_3state_btc.pkl
  2. Fetch BTC daily price via yfinance, compute features, predict regimes
  3. Join regimes to btc_scan_archive_15m.csv
  4. Run Gate A, Gate B, and combined simulations
  5. Print full results
"""

import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load HMM ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load HMM Model")
print("=" * 60)

payload = pickle.load(open(
    '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/hmm_3state_btc.pkl', 'rb'))
model         = payload['model']
state_to_name = payload['state_to_name']   # {0: 'Bear', 2: 'Sideways', 1: 'Bull'}
feature_cols  = payload['feature_cols']    # ['log_ret', 'realized_vol', 'ret_5d']

print(f"  n_states      : {payload['n_states']}")
print(f"  train_end     : {payload['train_end']}")
print(f"  log_likelihood: {payload['log_likelihood']:.4f}")
print(f"  bic           : {payload['bic']:.4f}")
print(f"  feature_cols  : {feature_cols}")
print(f"  state_to_name : {state_to_name}")

# ── 2. Fetch BTC Daily Price & Compute Features ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Fetch BTC Daily Price & Compute Features")
print("=" * 60)

df_daily = yf.download(
    'BTC-USD',
    start='2025-10-01',
    end='2026-05-28',
    interval='1d',
    auto_adjust=True,
    progress=False
)

# Flatten multi-index if present
if isinstance(df_daily.columns, pd.MultiIndex):
    df_daily.columns = df_daily.columns.get_level_values(0)

close = df_daily['Close'].squeeze()
print(f"  Downloaded {len(close)} daily bars ({close.index[0].date()} to {close.index[-1].date()})")

# Compute features
df_daily = df_daily.copy()
df_daily['log_ret']      = np.log(close / close.shift(1))
df_daily['realized_vol'] = df_daily['log_ret'].rolling(20, min_periods=10).std()
df_daily['ret_5d']       = np.log(close / close.shift(5))

feat_df = df_daily[feature_cols].dropna()
print(f"  Rows after dropna: {len(feat_df)}")
print(f"  Feature date range: {feat_df.index[0].date()} to {feat_df.index[-1].date()}")

# ── 3. Predict Regimes ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Compute HMM Regime for Each Date")
print("=" * 60)

X = feat_df.values
states = model.predict(X)

regime_by_date = {
    ts.date(): state_to_name[s]
    for ts, s in zip(feat_df.index, states)
}

# Print date → regime mapping for the archive window (May 21–27)
print("\n  BTC daily regime assignments (full window):")
sorted_dates = sorted(regime_by_date.keys())
for d in sorted_dates:
    print(f"    {d}  →  {regime_by_date[d]}")

# ── 4. Load Scan Archive & Join Regime ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Join Regime to Scan Archive")
print("=" * 60)

archive_path = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/btc_scan_archive_15m.csv'
df = pd.read_csv(archive_path)
df['logged_at'] = pd.to_datetime(df['logged_at'], utc=True)

# Assign regime
df['markov_regime_1h'] = df['logged_at'].dt.date.map(regime_by_date)

print(f"\n  Archive total rows : {len(df)}")
print(f"  Unique dates in archive:")
for d in sorted(df['logged_at'].dt.date.unique()):
    n = (df['logged_at'].dt.date == d).sum()
    r = regime_by_date.get(d, 'MISSING')
    print(f"    {d}  n_rows={n:4d}  regime={r}")

print(f"\n  Regime coverage (rows):")
print(df['markov_regime_1h'].value_counts().to_string())

# ── 5. Save updated archive ──────────────────────────────────────────────────
df.to_csv(archive_path, index=False)
print(f"\n  Saved updated archive → {archive_path}")

# ── 6. Simulation Setup ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Gate Simulations")
print("=" * 60)

MIN_EDGE    = 0.04
KELLY_MULT  = 0.30
KELLY_CAP   = 0.06
BANKROLL    = 1000

# Deduplicate: one row per contract_ticker (latest logged_at)
df_dedup = (
    df.sort_values('logged_at')
      .groupby('contract_ticker', as_index=False)
      .last()
)

# Filter to resolved rows
df_resolved = df_dedup[df_dedup['resolved_yes'].notna()].copy()
df_resolved['resolved_yes'] = pd.to_numeric(df_resolved['resolved_yes'], errors='coerce')
df_resolved = df_resolved[df_resolved['resolved_yes'].notna()].copy()

print(f"\n  Deduplicated contracts  : {len(df_dedup)}")
print(f"  Resolved contracts      : {len(df_resolved)}")

# Compute edges
df_resolved['edge_yes'] = df_resolved['p_model_yes'] - df_resolved['p_market']
df_resolved['edge_no']  = df_resolved['p_market']    - df_resolved['p_model_no']

def kelly_bet(edge, p_mkt, side):
    """Compute Kelly-sized bet amount."""
    if side == 'YES':
        frac = min(edge / p_mkt * KELLY_MULT, KELLY_CAP)
    else:
        frac = min(edge / (1 - p_mkt) * KELLY_MULT, KELLY_CAP)
    return frac * BANKROLL

def compute_pnl(row, side):
    pm  = row['p_market']
    fee = 0.07 * min(pm, 1 - pm)
    won = int(row['resolved_yes'])
    if side == 'YES':
        bet = kelly_bet(row['edge_yes'], pm, 'YES')
        return bet * (1 - pm - fee) if won == 1 else -bet * (pm + fee)
    else:
        bet = kelly_bet(row['edge_no'], pm, 'NO')
        return bet * (pm - fee) if won == 0 else -bet * (1 - pm + fee)

def select_side(row):
    """Baseline: pick YES or NO based on edge, else None."""
    ey = row['edge_yes']
    en = row['edge_no']
    if ey > en and ey > MIN_EDGE:
        return 'YES'
    elif en > ey and en > MIN_EDGE:
        return 'NO'
    return None

# Build base trade list
base_rows = []
for _, row in df_resolved.iterrows():
    side = select_side(row)
    if side is None:
        continue
    pnl = compute_pnl(row, side)
    won = (side == 'YES' and row['resolved_yes'] == 1) or (side == 'NO' and row['resolved_yes'] == 0)
    base_rows.append({
        'contract_ticker': row['contract_ticker'],
        'side': side,
        'pnl': pnl,
        'won': won,
        'p_market': row['p_market'],
        'stoch_k_15m': row['stoch_k_15m'],
        'ema_bias': row['ema_bias'],
        'markov_regime_1h': row['markov_regime_1h'],
        'resolved_yes': row['resolved_yes'],
    })

df_base = pd.DataFrame(base_rows)

def report_scenario(name, df_trades, df_base_trades):
    """Print scenario stats and gate impact."""
    if len(df_trades) == 0:
        print(f"\n  [{name}] NO TRADES")
        return

    n_total = len(df_trades)
    n_yes   = (df_trades['side'] == 'YES').sum()
    n_no    = (df_trades['side'] == 'NO').sum()
    n_wins  = df_trades['won'].sum()
    wr      = n_wins / n_total * 100
    total_pnl = df_trades['pnl'].sum()
    delta_pnl = total_pnl - df_base_trades['pnl'].sum()

    print(f"\n  ── {name} ──")
    print(f"     n_trades : {n_total}  (YES={n_yes}, NO={n_no})")
    print(f"     Win rate : {wr:.1f}%  ({int(n_wins)} wins)")
    print(f"     Total PnL: ${total_pnl:.2f}")
    print(f"     Δ vs base: ${delta_pnl:+.2f}")

    # Blocked trades (in base, not in this scenario)
    base_tickers = set(df_base_trades['contract_ticker'])
    scen_tickers = set(df_trades['contract_ticker'])
    blocked_tickers = base_tickers - scen_tickers
    if blocked_tickers:
        blocked = df_base_trades[df_base_trades['contract_ticker'].isin(blocked_tickers)]
        wins_blocked   = blocked['won'].sum()
        losses_blocked = (~blocked['won']).sum()
        pnl_blocked    = blocked['pnl'].sum()
        print(f"     Blocked   : {len(blocked_tickers)} contracts | wins_blocked={wins_blocked} | losses_blocked={losses_blocked} | PnL_blocked=${pnl_blocked:.2f}")

# ─── SCENARIO 1: Baseline ────────────────────────────────────────────────────
report_scenario("BASELINE", df_base, df_base)

# ─── SCENARIO 2: Gate A only ─────────────────────────────────────────────────
# Block NO when stoch_k_15m >= 80 AND ema_bias == 1
def gate_a(row):
    if row['side'] == 'NO':
        stoch = row['stoch_k_15m']
        ema   = row['ema_bias']
        if pd.notna(stoch) and pd.notna(ema):
            if stoch >= 80 and ema == 1:
                return True  # blocked
    return False

df_gate_a = df_base[~df_base.apply(gate_a, axis=1)].copy()
report_scenario("GATE A ONLY (block NO: stoch_k_15m>=80 AND ema_bias==1)", df_gate_a, df_base)

# ─── SCENARIO 3: Gate B full ─────────────────────────────────────────────────
# Block NO when 0.70 <= p_market < 0.80 AND markov_regime_1h == 'Sideways'
def gate_b_full(row):
    if row['side'] == 'NO':
        pm  = row['p_market']
        reg = row['markov_regime_1h']
        if 0.70 <= pm < 0.80 and reg == 'Sideways':
            return True
    return False

df_gate_b_full = df_base[~df_base.apply(gate_b_full, axis=1)].copy()
report_scenario("GATE B FULL (block NO: 0.70<=pm<0.80 AND Sideways)", df_gate_b_full, df_base)

# ─── SCENARIO 4: Gate B broad ────────────────────────────────────────────────
# Block NO when p_market >= 0.70 AND markov_regime_1h == 'Sideways'
def gate_b_broad(row):
    if row['side'] == 'NO':
        pm  = row['p_market']
        reg = row['markov_regime_1h']
        if pm >= 0.70 and reg == 'Sideways':
            return True
    return False

df_gate_b_broad = df_base[~df_base.apply(gate_b_broad, axis=1)].copy()
report_scenario("GATE B BROAD (block NO: pm>=0.70 AND Sideways)", df_gate_b_broad, df_base)

# ─── SCENARIO 5: Gate A + Gate B full ────────────────────────────────────────
def gate_ab(row):
    return gate_a(row) or gate_b_full(row)

df_gate_ab = df_base[~df_base.apply(gate_ab, axis=1)].copy()
report_scenario("GATE A + GATE B FULL", df_gate_ab, df_base)

# ─── BONUS: Regime breakdown of NO trades ────────────────────────────────────
print("\n" + "=" * 60)
print("BONUS: NO Trade Analysis by Regime & p_market Band")
print("=" * 60)

df_no = df_base[df_base['side'] == 'NO'].copy()
df_no['pm_band'] = pd.cut(
    df_no['p_market'],
    bins=[0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
    labels=['<0.50','0.50-0.60','0.60-0.70','0.70-0.80','0.80-0.90','>=0.90']
)

print("\n  NO trades by regime:")
print(df_no.groupby('markov_regime_1h', observed=True).agg(
    n=('won', 'count'),
    wins=('won', 'sum'),
    pnl=('pnl', 'sum')
).assign(wr=lambda x: x['wins']/x['n']*100).round(2).to_string())

print("\n  NO trades by regime × pm_band:")
grp = df_no.groupby(['markov_regime_1h','pm_band'], observed=True).agg(
    n=('won','count'),
    wins=('won','sum'),
    pnl=('pnl','sum')
).assign(wr=lambda x: x['wins']/x['n']*100).round(2)
print(grp.to_string())

print("\n  Contracts per day per regime (from resolved set):")
day_reg = (
    df_resolved.assign(date=df_resolved['logged_at'].dt.date)
    .groupby(['date','markov_regime_1h'], observed=True)
    .size()
    .reset_index(name='n_contracts')
)
print(day_reg.to_string(index=False))

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
