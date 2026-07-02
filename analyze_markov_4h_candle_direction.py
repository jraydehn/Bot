"""
analyze_markov_4h_candle_direction.py

Tests whether the 4h Markov regime label can predict the direction of the
NEXT 1h candle (bullish = close > open). No trades involved — pure price
prediction accuracy on the full BTC-USD 1h history.

Also tests:
  - Transition probability signal: P(Bull|current) - P(Bear|current) as a
    graded predictor of next-bar direction.
  - Walk-forward version: matrix re-estimated at each bar using only past data.
  - Rolling accuracy windows to see if the signal is time-stable.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

WINDOW    = 20       # 20 × 4h bars lookback for regime label
THRESHOLD = 0.010    # ±1% on 20-bar rolling return

SEP  = "=" * 68
SEP2 = "-" * 52

# ── 1. Fetch 1h data ─────────────────────────────────────────────────────────
print("\nFetching BTC-USD 1h data...")
df_1h = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)
df_1h = df_1h[["Open", "High", "Low", "Close"]].dropna()
print(f"  {len(df_1h)} 1h bars: {df_1h.index.min()} → {df_1h.index.max()}")

# ── 2. Build 4h close series, compute regime ─────────────────────────────────
close_4h   = df_1h["Close"].resample("4h").last().dropna()
roll_ret   = close_4h.pct_change(WINDOW)
regime_4h  = pd.Series("Sideways", index=close_4h.index)
regime_4h[roll_ret >  THRESHOLD] = "Bull"
regime_4h[roll_ret < -THRESHOLD] = "Bear"
regime_4h  = regime_4h[roll_ret.notna()]
regime_4h.index = pd.to_datetime(regime_4h.index, utc=True)

states     = ["Bear", "Sideways", "Bull"]
state_idx  = {s: i for i, s in enumerate(states)}

# ── 3. Build full transition matrix (for signal computation) ─────────────────
arr    = regime_4h.to_numpy()
counts = np.zeros((3, 3), dtype=float)
for i in range(len(arr) - 1):
    counts[state_idx[arr[i]], state_idx[arr[i + 1]]] += 1
rs = counts.sum(axis=1, keepdims=True)
rs[rs == 0] = 1.0
P_full = counts / rs

print(f"\n4h transition matrix (±{THRESHOLD*100:.1f}%, window={WINDOW}):")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states):
    row = "  ".join(f"{P_full[i,j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")

# ── 4. Join 4h regime to every 1h bar ────────────────────────────────────────
# For each 1h bar, find the last completed 4h bar whose timestamp ≤ bar time.
regime_df = regime_4h.reset_index()
regime_df.columns = ["bar_4h_ts", "regime_4h"]
regime_df["bar_4h_ts"] = regime_df["bar_4h_ts"].dt.as_unit("us")

df_1h_r = df_1h.copy()
df_1h_r.index = df_1h_r.index.as_unit("us")
df_1h_r = df_1h_r.reset_index().rename(columns={"Datetime": "ts"})

df_joined = pd.merge_asof(
    df_1h_r.sort_values("ts"),
    regime_df.sort_values("bar_4h_ts"),
    left_on="ts",
    right_on="bar_4h_ts",
    direction="backward",
)
df_joined = df_joined.dropna(subset=["regime_4h"])

# 1h candle direction: next bar
df_joined = df_joined.sort_values("ts").reset_index(drop=True)
df_joined["next_close"]    = df_joined["Close"].shift(-1)
df_joined["next_open"]     = df_joined["Open"].shift(-1)
df_joined["next_bull"]     = (df_joined["next_close"] > df_joined["next_open"]).astype(float)
df_joined["next_ret"]      = (df_joined["next_close"] - df_joined["next_open"]) / df_joined["next_open"]
df_joined = df_joined.dropna(subset=["next_bull"])

# Transition-probability signal: P(Bull|cur) - P(Bear|cur)
def tp_signal(regime):
    idx = state_idx.get(regime, 1)
    return float(P_full[idx, 2] - P_full[idx, 0])

df_joined["tp_signal"] = df_joined["regime_4h"].map(tp_signal)

print(f"\n  Total 1h bars with regime label: {len(df_joined)}")

# ── 5. Directional accuracy by regime ────────────────────────────────────────
print(f"\n{SEP}")
print("  NEXT-1h-CANDLE DIRECTION vs 4h MARKOV REGIME")
print(f"  (bullish = next 1h close > open)")
print(SEP)

overall_bull = df_joined["next_bull"].mean()
print(f"\n  Base rate (all bars bullish): {overall_bull*100:.2f}%")

for reg in ("Bull", "Sideways", "Bear"):
    sub = df_joined[df_joined["regime_4h"] == reg]
    n   = len(sub)
    if n == 0:
        continue
    acc   = sub["next_bull"].mean()
    avg_r = sub["next_ret"].mean() * 100
    std_r = sub["next_ret"].std()  * 100
    lift  = acc - overall_bull
    # Expected direction given signal (Bull regime → predict bull, Bear → predict bear)
    if reg == "Bull":
        pred_correct = acc
    elif reg == "Bear":
        pred_correct = 1 - acc   # we'd predict bearish
    else:
        pred_correct = max(acc, 1 - acc)

    print(f"\n  Regime = {reg:<10s}  (n={n:,})")
    print(f"    Next-bar bullish rate:  {acc*100:.2f}%  (base={overall_bull*100:.2f}%,  lift={lift*100:+.2f}pp)")
    print(f"    Avg next-bar return:   {avg_r:+.4f}%  (std={std_r:.4f}%)")
    print(f"    Correct if predict {('Bull' if reg=='Bull' else 'Bear' if reg=='Bear' else 'either'):<5s}: {pred_correct*100:.2f}%")

# ── 6. Statistical significance ──────────────────────────────────────────────
from scipy import stats

print(f"\n{SEP2}")
print("  STATISTICAL TESTS (chi-square on bullish counts):")
bull_counts = []
bear_counts = []
for reg in ("Bull", "Sideways", "Bear"):
    sub = df_joined[df_joined["regime_4h"] == reg]
    bull_counts.append((sub["next_bull"] == 1).sum())
    bear_counts.append((sub["next_bull"] == 0).sum())

contingency = np.array([bull_counts, bear_counts])
chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
print(f"  chi2={chi2:.3f}  p={p_val:.4f}  dof={dof}")
print(f"  {'Significant' if p_val < 0.05 else 'NOT significant'} at p<0.05")

# Pointwise z-tests (Bull vs Bear)
sub_bull = df_joined[df_joined["regime_4h"] == "Bull"]["next_bull"]
sub_bear = df_joined[df_joined["regime_4h"] == "Bear"]["next_bull"]
z, pz = stats.ttest_ind(sub_bull, sub_bear)
print(f"\n  Bull vs Bear next-bar direction t-test:  t={z:.3f}  p={pz:.4f}")

# ── 7. Return magnitude by regime ────────────────────────────────────────────
print(f"\n{SEP}")
print("  NEXT-1h-CANDLE RETURN MAGNITUDE vs REGIME")
print(SEP)

for reg in ("Bull", "Sideways", "Bear"):
    sub = df_joined[df_joined["regime_4h"] == reg]
    if len(sub) == 0:
        continue
    pos = sub[sub["next_bull"] == 1]["next_ret"].mean() * 100
    neg = sub[sub["next_bull"] == 0]["next_ret"].mean() * 100
    print(f"\n  Regime = {reg}")
    print(f"    Avg return on UP   bars: {pos:+.4f}%")
    print(f"    Avg return on DOWN bars: {neg:+.4f}%")
    print(f"    Avg abs return:          {sub['next_ret'].abs().mean()*100:.4f}%")

# ── 8. Transition-probability signal as graded predictor ─────────────────────
print(f"\n{SEP}")
print("  TRANSITION-PROBABILITY SIGNAL AS GRADED PREDICTOR")
print(f"  signal = P(next_regime=Bull|cur) - P(next_regime=Bear|cur)")
print(SEP)

df_joined["tp_bin"] = pd.cut(df_joined["tp_signal"], bins=5)
tp_summary = df_joined.groupby("tp_bin", observed=True).agg(
    n=("next_bull","count"),
    bull_rate=("next_bull","mean"),
    avg_ret=("next_ret","mean"),
).reset_index()
tp_summary["avg_ret_pct"] = tp_summary["avg_ret"] * 100
print(f"\n  {'tp_signal range':<28s}  {'n':>6s}  {'bull_rate':>9s}  {'avg_ret':>9s}")
print("  " + "-" * 58)
for _, row in tp_summary.iterrows():
    print(f"  {str(row['tp_bin']):<28s}  {row['n']:>6.0f}  {row['bull_rate']*100:>8.2f}%  {row['avg_ret_pct']:>+8.4f}%")

# Pearson correlation: tp_signal vs next_ret
_tp_sub = df_joined[["tp_signal", "next_ret"]].dropna()
r, p = stats.pearsonr(_tp_sub["tp_signal"], _tp_sub["next_ret"])
print(f"\n  Pearson r (tp_signal vs next_ret): {r:.4f}  p={p:.4f}")

# ── 9. Walk-forward accuracy ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("  WALK-FORWARD ACCURACY (matrix re-estimated at every 4h bar)")
print(f"  (no lookahead — only data before each bar used)")
print(SEP)

MIN_TRAIN = 200   # need at least this many 4h bars before trusting the matrix
correct   = []
signals   = []
actuals   = []

arr_full = regime_4h.to_numpy()
ts_full  = regime_4h.index

for t in range(MIN_TRAIN, len(arr_full) - 1):
    # Build matrix from data up to t-1
    cnt = np.zeros((3, 3), dtype=float)
    for i in range(t - 1):
        cnt[state_idx[arr_full[i]], state_idx[arr_full[i + 1]]] += 1
    rs_t = cnt.sum(axis=1, keepdims=True)
    rs_t[rs_t == 0] = 1.0
    P_t = cnt / rs_t

    cur_state = state_idx[arr_full[t]]
    next_state = state_idx[arr_full[t + 1]]

    # Signal: predict Bull if P(Bull|cur) > P(Bear|cur), else Bear
    sig = P_t[cur_state, 2] - P_t[cur_state, 0]
    predicted_bull_regime = sig > 0
    actual_bull_regime    = next_state == 2  # next 4h regime = Bull

    correct.append(int(predicted_bull_regime == actual_bull_regime))
    signals.append(sig)
    actuals.append(float(actual_bull_regime))

wf_acc = np.mean(correct)
print(f"\n  Walk-forward regime prediction accuracy (n={len(correct):,}): {wf_acc*100:.2f}%")
print(f"  (predicting whether next 4h regime bar = Bull vs not-Bull)")

r_wf, p_wf = stats.pearsonr(signals, actuals)
print(f"  Pearson r (signal vs actual Bull regime): {r_wf:.4f}  p={p_wf:.4f}")

# ── 10. Rolling 30-day accuracy window ──────────────────────────────────────
print(f"\n{SEP}")
print("  ROLLING MONTHLY ACCURACY (Bull-rate of next 1h bar, by calendar month)")
print(SEP)

df_joined["month"] = df_joined["ts"].dt.to_period("M")
monthly = df_joined.groupby(["month","regime_4h"])["next_bull"].mean().unstack(fill_value=np.nan)
monthly_n = df_joined.groupby(["month","regime_4h"])["next_bull"].count().unstack(fill_value=0)
print(f"\n  {'Month':<10s}  {'Bear_bull%':>10s}  {'Side_bull%':>10s}  {'Bull_bull%':>10s}  "
      f"{'Bear_n':>7s}  {'Side_n':>7s}  {'Bull_n':>7s}")
print("  " + "-" * 70)
for m in monthly.index:
    def f(col):
        v = monthly.loc[m, col] if col in monthly.columns else np.nan
        n = int(monthly_n.loc[m, col]) if col in monthly_n.columns else 0
        return v, n
    bv, bn = f("Bear")
    sv, sn = f("Sideways")
    uv, un = f("Bull")
    bs = f"{bv*100:.1f}%" if pd.notna(bv) else "  —  "
    ss = f"{sv*100:.1f}%" if pd.notna(sv) else "  —  "
    us = f"{uv*100:.1f}%" if pd.notna(uv) else "  —  "
    print(f"  {str(m):<10s}  {bs:>10s}  {ss:>10s}  {us:>10s}  "
          f"{bn:>7d}  {sn:>7d}  {un:>7d}")
