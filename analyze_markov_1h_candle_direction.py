"""
analyze_markov_1h_candle_direction.py

Builds Markov regime labels directly on 1h BTC bars (20-bar rolling return)
and tests whether the current regime predicts the direction of the NEXT 1h candle.

Includes:
  - Threshold sweep to find the best Bull/Bear cutoff
  - Directional accuracy + statistical significance
  - Walk-forward (no-lookahead) regime prediction accuracy
  - Monthly stability check
  - Transition-probability signal as a graded predictor
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

WINDOW    = 20      # 20 × 1h bars = 20 hours lookback
SEP       = "=" * 68
SEP2      = "-" * 52

# ── 1. Fetch 1h data ─────────────────────────────────────────────────────────
print("\nFetching BTC-USD 1h data (max 730-day yfinance window)...")
df = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                 interval="1h", progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index, utc=True)
df = df[["Open", "High", "Low", "Close"]].dropna()
print(f"  {len(df)} 1h bars: {df.index.min()} → {df.index.max()}")

close = df["Close"]

# ── 2. Threshold sweep ───────────────────────────────────────────────────────
print(f"\nRegime label counts at various thresholds (window={WINDOW} × 1h bars = {WINDOW}h):")
roll_ret = close.pct_change(WINDOW)
print(f"  {'Threshold':>10s}  {'Bear':>6s}  {'Side':>6s}  {'Bull':>6s}  {'B/S/Bu%'}")
for thr in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
    reg = pd.Series("Sideways", index=close.index)
    reg[roll_ret >  thr] = "Bull"
    reg[roll_ret < -thr] = "Bear"
    reg = reg[roll_ret.notna()]
    bc, sc, uc, n = (reg=="Bear").sum(), (reg=="Sideways").sum(), (reg=="Bull").sum(), len(reg)
    print(f"  {thr*100:>8.2f}%   {bc:6d}  {sc:6d}  {uc:6d}  "
          f"({bc/n*100:.0f}% / {sc/n*100:.0f}% / {uc/n*100:.0f}%)")

# Working threshold
THRESHOLD = 0.008
print(f"\nWorking threshold: ±{THRESHOLD*100:.1f}%  (20h rolling return)")

regime = pd.Series("Sideways", index=close.index)
regime[roll_ret >  THRESHOLD] = "Bull"
regime[roll_ret < -THRESHOLD] = "Bear"
regime = regime[roll_ret.notna()]

states    = ["Bear", "Sideways", "Bull"]
state_idx = {s: i for i, s in enumerate(states)}

vc = regime.value_counts()
for s in states:
    n = vc.get(s, 0)
    print(f"  {s:<10s}: {n:5d} bars ({n/len(regime)*100:.1f}%)")

# ── 3. Transition matrix ─────────────────────────────────────────────────────
arr    = regime.to_numpy()
counts = np.zeros((3, 3), dtype=float)
for i in range(len(arr) - 1):
    counts[state_idx[arr[i]], state_idx[arr[i+1]]] += 1
rs = counts.sum(axis=1, keepdims=True)
rs[rs == 0] = 1.0
P = counts / rs

print(f"\n1h Transition matrix (rows=from, cols=to):")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states):
    row = "  ".join(f"{P[i,j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")
print("\nPersistence diagonal:")
for i, s in enumerate(states):
    print(f"  {s} → {s}: {P[i,i]*100:.2f}%")

# ── 4. Label every bar and compute next-bar targets ──────────────────────────
df2 = df.copy()
df2["regime"]    = regime.reindex(df2.index)
df2             = df2.dropna(subset=["regime"])
df2             = df2.sort_index()

df2["next_close"] = df2["Close"].shift(-1)
df2["next_open"]  = df2["Open"].shift(-1)
df2["next_ret"]   = (df2["next_close"] - df2["next_open"]) / df2["next_open"]
df2["next_bull"]  = (df2["next_close"] > df2["next_open"]).astype(float)
df2 = df2.dropna(subset=["next_bull", "next_ret"])

# Transition-probability signal
def tp_sig(r):
    idx = state_idx.get(r, 1)
    return float(P[idx, 2] - P[idx, 0])
df2["tp_signal"] = df2["regime"].map(tp_sig)

base_rate = df2["next_bull"].mean()
print(f"\nTotal labeled bars: {len(df2):,}   Base bullish rate: {base_rate*100:.2f}%")

# ── 5. Directional accuracy by regime ────────────────────────────────────────
print(f"\n{SEP}")
print("  NEXT-1h CANDLE DIRECTION  vs  1h MARKOV REGIME")
print(f"  (bullish = next close > next open)")
print(SEP)

for reg in states:
    sub = df2[df2["regime"] == reg]
    n   = len(sub)
    if n < 20:
        print(f"\n  {reg}: n={n} (too small)")
        continue
    acc   = sub["next_bull"].mean()
    avg_r = sub["next_ret"].mean() * 100
    std_r = sub["next_ret"].std()  * 100
    lift  = acc - base_rate
    if reg == "Bull":
        pred_acc = acc
        pred_dir = "Bull"
    elif reg == "Bear":
        pred_acc = 1 - acc
        pred_dir = "Bear"
    else:
        pred_acc = max(acc, 1 - acc)
        pred_dir = "either"

    print(f"\n  Regime = {reg:<10s}  (n={n:,})")
    print(f"    Next-bar bullish rate:       {acc*100:.2f}%")
    print(f"    Lift vs base ({base_rate*100:.2f}%):      {lift*100:+.2f} pp")
    print(f"    Avg next-bar return:         {avg_r:+.5f}%")
    print(f"    Std next-bar return:         ±{std_r:.4f}%")
    print(f"    Correct if predict {pred_dir:<5s}:    {pred_acc*100:.2f}%")

# ── 6. Statistical tests ─────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  STATISTICAL SIGNIFICANCE")

bull_c = [(df2[df2["regime"]==s]["next_bull"]==1).sum() for s in states]
bear_c = [(df2[df2["regime"]==s]["next_bull"]==0).sum() for s in states]
chi2, p_chi, dof, _ = stats.chi2_contingency(np.array([bull_c, bear_c]))
print(f"\n  Chi-square (3 regimes vs up/down): chi2={chi2:.3f}  p={p_chi:.5f}  dof={dof}")
print(f"  {'Significant' if p_chi < 0.05 else 'NOT significant'} at p<0.05")

sub_b = df2[df2["regime"]=="Bull"]["next_bull"]
sub_r = df2[df2["regime"]=="Bear"]["next_bull"]
t, pt = stats.ttest_ind(sub_b, sub_r)
print(f"\n  Bull vs Bear t-test: t={t:.3f}  p={pt:.5f}")

_sub = df2[["tp_signal","next_ret"]].dropna()
r_p, p_p = stats.pearsonr(_sub["tp_signal"], _sub["next_ret"])
print(f"\n  Pearson r (tp_signal → next_ret): r={r_p:.4f}  p={p_p:.5f}")
print(f"  Variance explained (r²):           {r_p**2*100:.3f}%")

# ── 7. Return magnitude & volatility by regime ───────────────────────────────
print(f"\n{SEP}")
print("  RETURN MAGNITUDE BY REGIME")
print(SEP)
for reg in states:
    sub = df2[df2["regime"] == reg]
    if len(sub) < 20:
        continue
    up   = sub[sub["next_bull"]==1]["next_ret"].mean()*100
    dn   = sub[sub["next_bull"]==0]["next_ret"].mean()*100
    ab   = sub["next_ret"].abs().mean()*100
    print(f"\n  Regime = {reg}")
    print(f"    Avg UP   bar return: {up:+.4f}%")
    print(f"    Avg DOWN bar return: {dn:+.4f}%")
    print(f"    Avg abs  return:     {ab:.4f}%")

# ── 8. Graded signal (transition probability bins) ───────────────────────────
print(f"\n{SEP}")
print("  TRANSITION-PROBABILITY SIGNAL AS GRADED PREDICTOR")
print(f"  signal = P(next_regime=Bull) - P(next_regime=Bear)  given current regime")
print(SEP)

bins = pd.cut(df2["tp_signal"], bins=5)
tp_tbl = df2.groupby(bins, observed=True).agg(
    n=("next_bull","count"),
    bull_rate=("next_bull","mean"),
    avg_ret=("next_ret","mean"),
).reset_index()
print(f"\n  {'Signal range':<28s}  {'n':>6s}  {'Bull%':>7s}  {'Avg ret':>9s}")
print("  " + "-" * 56)
for _, row in tp_tbl.iterrows():
    print(f"  {str(row['tp_signal']):<28s}  {row['n']:>6.0f}  "
          f"{row['bull_rate']*100:>6.2f}%  {row['avg_ret']*100:>+8.4f}%")

# ── 9. Walk-forward no-lookahead test ────────────────────────────────────────
print(f"\n{SEP}")
print("  WALK-FORWARD (no lookahead): predict next 1h candle direction")
print(f"  At each bar t, matrix estimated from bars 1..t-1 only")
print(SEP)

MIN_TRAIN = 500
wf_correct = []
wf_sigs    = []
wf_rets    = []
arr_full   = df2["regime"].to_numpy()
rets_full  = df2["next_ret"].to_numpy()
bull_full  = df2["next_bull"].to_numpy()

for t in range(MIN_TRAIN, len(arr_full) - 1):
    cnt = np.zeros((3, 3), dtype=float)
    for i in range(t - 1):
        cnt[state_idx[arr_full[i]], state_idx[arr_full[i+1]]] += 1
    rs_t = cnt.sum(axis=1, keepdims=True)
    rs_t[rs_t == 0] = 1.0
    P_t = cnt / rs_t

    cur = state_idx[arr_full[t]]
    sig = float(P_t[cur, 2] - P_t[cur, 0])
    pred_up = sig > 0
    actual_up = bool(bull_full[t])
    wf_correct.append(int(pred_up == actual_up))
    wf_sigs.append(sig)
    wf_rets.append(rets_full[t])

wf_acc = np.mean(wf_correct)
r_wf, p_wf = stats.pearsonr(wf_sigs, wf_rets)
print(f"\n  Walk-forward directional accuracy: {wf_acc*100:.2f}%  (n={len(wf_correct):,})")
print(f"  Pearson r (signal → next return):  {r_wf:.4f}  p={p_wf:.5f}")
print(f"  Variance explained (r²):            {r_wf**2*100:.3f}%")

# Also show by signal strength quartile
wf_df = pd.DataFrame({"sig": wf_sigs, "ret": wf_rets, "correct": wf_correct})
wf_df["sig_abs"] = wf_df["sig"].abs()
wf_df["quartile"] = pd.qcut(wf_df["sig_abs"], q=4, labels=["Q1 (weak)","Q2","Q3","Q4 (strong)"])
print(f"\n  Accuracy by signal strength quartile:")
print(f"  {'Quartile':<14s}  {'n':>6s}  {'Accuracy':>9s}  {'Avg|sig|':>9s}")
print("  " + "-" * 44)
for q, grp in wf_df.groupby("quartile", observed=True):
    print(f"  {str(q):<14s}  {len(grp):>6d}  {grp['correct'].mean()*100:>8.2f}%  {grp['sig_abs'].mean():>+9.4f}")

# ── 10. Monthly stability ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MONTHLY STABILITY: Bull-rate of next 1h bar by regime")
print(SEP)

df2["month"] = df2.index.to_period("M")
mo  = df2.groupby(["month","regime"])["next_bull"].mean().unstack(fill_value=np.nan)
mo_n = df2.groupby(["month","regime"])["next_bull"].count().unstack(fill_value=0)

print(f"\n  {'Month':<10s}  {'Bear_bull%':>10s}  {'Side_bull%':>10s}  {'Bull_bull%':>10s}  "
      f"{'Bear_n':>7s}  {'Side_n':>7s}  {'Bull_n':>7s}")
print("  " + "-" * 72)
for m in mo.index:
    def g(col):
        v = mo.loc[m, col] if col in mo.columns else np.nan
        n = int(mo_n.loc[m, col]) if col in mo_n.columns else 0
        return v, n
    bv,bn = g("Bear"); sv,sn = g("Sideways"); uv,un = g("Bull")
    print(f"  {str(m):<10s}  "
          f"{bv*100:>9.1f}%  {sv*100:>9.1f}%  {uv*100:>9.1f}%  "
          f"{bn:>7d}  {sn:>7d}  {un:>7d}")

# ── 11. Current state ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  CURRENT 1h STATE")
print(SEP)
cur_reg = regime.iloc[-1]
cur_ts  = regime.index[-1]
cur_idx = state_idx[cur_reg]
print(f"\n  Current 1h regime: {cur_reg}  ({cur_ts})")
print(f"  Given {cur_reg}, next-bar distribution:")
for j, s in enumerate(states):
    print(f"    → {s:<10s}: {P[cur_idx,j]*100:.1f}%")
bull_p  = P[cur_idx, 2]
bear_p  = P[cur_idx, 0]
print(f"\n  Directional bias signal: {bull_p - bear_p:+.4f}  "
      f"({'bullish' if bull_p > bear_p else 'bearish' if bear_p > bull_p else 'neutral'})")
