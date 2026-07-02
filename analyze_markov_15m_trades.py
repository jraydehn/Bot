"""
analyze_markov_15m_trades.py

Tests four Markov timescales against BTC 15m paper trades:
  1. Daily   (20d  rolling return, ±2.0%)  — same gate live on 1h model
  2. 4h      (20×4h rolling return, ±1.0%) — previously analysed
  3. 1h      (20×1h rolling return, ±0.8%) — previously analysed
  4. 15m     (20×15m rolling return, ±?%)  — new, native to contract cadence

For each: WR + flat P&L split by regime (all / YES / NO).
Also tests next-15m-candle direction prediction for the 15m Markov.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found — pip install yfinance")

TRADES_CSV  = "results/paper_trades_btc15m.csv"
FLAT_BET    = 25.0
KALSHI_TAKE = 0.10
MIN_N       = 10

SEP  = "=" * 68
SEP2 = "-" * 52

# ── helpers ──────────────────────────────────────────────────────────────────
def flat_pnl(row):
    p = float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = max(1 - p, 1e-6)
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET

def report(sub, label):
    n = len(sub)
    if n < MIN_N:
        return
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    sides = sub["side"].value_counts().to_dict()
    print(f"  {label:<44s}  n={n:4d}  WR={wr*100:5.1f}%  P&L=${pnl:+7.0f}")

# ── 1. Fetch price data ───────────────────────────────────────────────────────
print("Fetching BTC-USD price data...")
# Daily
df_d = yf.download("BTC-USD", start="2025-01-01", end="2026-05-23",
                   progress=False, auto_adjust=True)
if isinstance(df_d.columns, pd.MultiIndex):
    df_d.columns = df_d.columns.get_level_values(0)
close_d = df_d["Close"].dropna()
close_d.index = pd.to_datetime(close_d.index, utc=True)

# 1h (also used for 4h resample)
df_1h = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)
close_1h = df_1h["Close"].dropna()

# 15m
df_15m = yf.download("BTC-USD", start="2026-04-01", end="2026-05-23",
                     interval="15m", progress=False, auto_adjust=True)
if isinstance(df_15m.columns, pd.MultiIndex):
    df_15m.columns = df_15m.columns.get_level_values(0)
df_15m.index = pd.to_datetime(df_15m.index, utc=True)
close_15m = df_15m["Close"].dropna()
print(f"  Daily: {len(close_d)} bars | 1h: {len(close_1h)} | 15m: {len(close_15m)}")

# ── 2. Build regime series for each timescale ─────────────────────────────────
def make_regime(close, window, threshold):
    rr  = close.pct_change(window)
    reg = pd.Series("Sideways", index=close.index)
    reg[rr >  threshold] = "Bull"
    reg[rr < -threshold] = "Bear"
    reg = reg[rr.notna()]
    reg.index = pd.to_datetime(reg.index, utc=True).as_unit("us")
    return reg

close_4h = close_1h.resample("4h").last().dropna()

reg_daily = make_regime(close_d,   window=20, threshold=0.020)
reg_4h    = make_regime(close_4h,  window=20, threshold=0.010)
reg_1h    = make_regime(close_1h,  window=20, threshold=0.008)
reg_15m   = make_regime(close_15m, window=20, threshold=0.004)

# Also sweep threshold for 15m
print(f"\n15m regime threshold sweep (window=20 × 15m = 5h):")
print(f"  {'Thresh':>7s}  {'Bear':>6s}  {'Side':>6s}  {'Bull':>6s}")
for thr in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008]:
    rr = close_15m.pct_change(20)
    r = pd.Series("Sideways", index=close_15m.index)
    r[rr >  thr] = "Bull"; r[rr < -thr] = "Bear"; r = r[rr.notna()]
    bc, sc, uc = (r=="Bear").sum(), (r=="Sideways").sum(), (r=="Bull").sum()
    print(f"  ±{thr*100:.2f}%   {bc:6d}  {sc:6d}  {uc:6d}")

# ── 3. Load 15m trades ────────────────────────────────────────────────────────
print(f"\nLoading 15m trades...")
df = pd.read_csv(TRADES_CSV, low_memory=False)
df = df[df["decision"] == "trade"].copy()
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df["p_market"]     = pd.to_numeric(df["p_market"],   errors="coerce")
df["trade_ts"]     = pd.to_datetime(df["logged_at"], format="ISO8601", utc=True).dt.as_unit("us")
df["flat_pnl"]     = df.apply(flat_pnl, axis=1)
df["won"]          = (
    ((df["side"]=="yes") & (df["resolved_yes"]==1)) |
    ((df["side"]=="no")  & (df["resolved_yes"]==0))
).astype(int)
df = df.sort_values("trade_ts")
print(f"  {len(df)} resolved trades  YES={len(df[df['side']=='yes'])}  NO={len(df[df['side']=='no'])}")
print(f"  WR={df['won'].mean()*100:.1f}%  P&L=${df['flat_pnl'].sum():+,.0f}")

# ── 4. Join each regime scale ─────────────────────────────────────────────────
def join_regime(df_trades, regime_series, col_name):
    reg_df = regime_series.reset_index()
    reg_df.columns = ["_bar_ts_join", col_name]
    reg_df["_bar_ts_join"] = pd.to_datetime(reg_df["_bar_ts_join"]).dt.as_unit("us")
    merged = pd.merge_asof(
        df_trades.sort_values("trade_ts"),
        reg_df.sort_values("_bar_ts_join"),
        left_on="trade_ts", right_on="_bar_ts_join",
        direction="backward",
    )
    return merged.drop(columns=["_bar_ts_join"], errors="ignore")

df = join_regime(df, reg_daily, "reg_daily")
df = join_regime(df, reg_4h,    "reg_4h")
df = join_regime(df, reg_1h,    "reg_1h")
df = join_regime(df, reg_15m,   "reg_15m")

for col in ["reg_daily","reg_4h","reg_1h","reg_15m"]:
    df[col] = df[col].fillna("Unknown")

# ── 5. Report by scale ────────────────────────────────────────────────────────
for scale, col in [("Daily (20d ±2%)", "reg_daily"),
                   ("4h    (20×4h ±1%)", "reg_4h"),
                   ("1h    (20×1h ±0.8%)", "reg_1h"),
                   ("15m   (20×15m ±0.4%)", "reg_15m")]:
    print(f"\n{SEP}")
    print(f"  BTC 15m TRADES — {scale}")
    print(SEP)
    report(df, "ALL trades")
    for reg in ("Bull","Sideways","Bear"):
        report(df[df[col]==reg], f"  ALL | {reg}")
    print()
    for side in ("yes","no"):
        ds = df[df["side"]==side]
        for reg in ("Bull","Sideways","Bear"):
            report(ds[ds[col]==reg], f"  {side.upper()} | {reg}")

# ── 6. Gate simulations ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  GATE SIMULATIONS — flat $25/trade")
print(SEP)
baseline = df["flat_pnl"].sum()
print(f"\n  Baseline P&L: ${baseline:+,.2f}")

gates = [
    ("Daily Sideways — block ALL",   (df["reg_daily"]=="Sideways")),
    ("Daily Sideways — block YES",   (df["reg_daily"]=="Sideways") & (df["side"]=="yes")),
    ("Daily Sideways — block NO pm>0.39",
     (df["reg_daily"]=="Sideways") & (df["side"]=="no") & (df["p_market"]>0.39)),
    ("4h Sideways — block ALL",      (df["reg_4h"]=="Sideways")),
    ("4h Bear — block ALL",          (df["reg_4h"]=="Bear")),
    ("1h Sideways — block ALL",      (df["reg_1h"]=="Sideways")),
    ("1h Bear — block ALL",          (df["reg_1h"]=="Bear")),
    ("15m Sideways — block ALL",     (df["reg_15m"]=="Sideways")),
    ("15m Bear — block ALL",         (df["reg_15m"]=="Bear")),
    ("15m Bear — block YES",         (df["reg_15m"]=="Bear") & (df["side"]=="yes")),
    ("15m Bear — block NO",          (df["reg_15m"]=="Bear") & (df["side"]=="no")),
    ("15m Bull — block NO",          (df["reg_15m"]=="Bull") & (df["side"]=="no")),
    ("15m Bull — block YES",         (df["reg_15m"]=="Bull") & (df["side"]=="yes")),
]
for label, mask in gates:
    blocked = df[mask]
    if len(blocked) < MIN_N:
        continue
    delta   = -blocked["flat_pnl"].sum()
    new_pnl = baseline + delta
    print(f"  {label:<42s}  n={len(blocked):4d}  WR={blocked['won'].mean()*100:5.1f}%  "
          f"block_pnl=${blocked['flat_pnl'].sum():+6.0f}  delta=${delta:+6.0f}")

# ── 7. 15m Markov next-candle direction test ──────────────────────────────────
print(f"\n{SEP}")
print("  15m MARKOV: DOES IT PREDICT THE NEXT 15m CANDLE DIRECTION?")
print(f"  (bullish = next 15m close > open, on raw price data)")
print(SEP)

df_15m_full = df_15m[["Open","Close"]].dropna().copy()
df_15m_full.index = pd.to_datetime(df_15m_full.index, utc=True)

rr_15m = df_15m_full["Close"].pct_change(20)
THR = 0.004
reg_f = pd.Series("Sideways", index=df_15m_full.index)
reg_f[rr_15m >  THR] = "Bull"
reg_f[rr_15m < -THR] = "Bear"
reg_f = reg_f[rr_15m.notna()]

df_15m_full["regime"] = reg_f.reindex(df_15m_full.index)
df_15m_full = df_15m_full.dropna(subset=["regime"])
df_15m_full["next_close"] = df_15m_full["Close"].shift(-1)
df_15m_full["next_open"]  = df_15m_full["Open"].shift(-1)
df_15m_full["next_bull"]  = (df_15m_full["next_close"] > df_15m_full["next_open"]).astype(float)
df_15m_full["next_ret"]   = (df_15m_full["next_close"] - df_15m_full["next_open"]) / df_15m_full["next_open"]
df_15m_full = df_15m_full.dropna(subset=["next_bull"])

base = df_15m_full["next_bull"].mean()
print(f"\n  Base bullish rate: {base*100:.2f}%  (n={len(df_15m_full):,} bars)")

for reg in ("Bull","Sideways","Bear"):
    sub = df_15m_full[df_15m_full["regime"]==reg]
    n   = len(sub)
    if n < 50:
        continue
    acc  = sub["next_bull"].mean()
    avgr = sub["next_ret"].mean()*100
    print(f"\n  Regime = {reg:<10s}  n={n:,}")
    print(f"    Next-bar bullish:  {acc*100:.2f}%  (lift {(acc-base)*100:+.2f}pp)")
    print(f"    Avg next-bar ret:  {avgr:+.5f}%")
    print(f"    Avg abs return:    {sub['next_ret'].abs().mean()*100:.4f}%")

# Stats
states_15m = ["Bear","Sideways","Bull"]
bull_c = [(df_15m_full[df_15m_full["regime"]==s]["next_bull"]==1).sum() for s in states_15m]
bear_c = [(df_15m_full[df_15m_full["regime"]==s]["next_bull"]==0).sum() for s in states_15m]
chi2, pval, dof, _ = stats.chi2_contingency(np.array([bull_c, bear_c]))
print(f"\n  Chi-square: chi2={chi2:.3f}  p={pval:.5f}  ({'significant' if pval<0.05 else 'NOT significant'})")

sb = df_15m_full[df_15m_full["regime"]=="Bull"]["next_bull"]
sr = df_15m_full[df_15m_full["regime"]=="Bear"]["next_bull"]
t, pt = stats.ttest_ind(sb, sr)
print(f"  Bull vs Bear t-test: t={t:.3f}  p={pt:.5f}")

# Walk-forward
states_idx = {s: i for i, s in enumerate(states_15m)}
arr = reg_f.to_numpy()
counts = np.zeros((3,3), dtype=float)
for i in range(len(arr)-1):
    counts[states_idx[arr[i]], states_idx[arr[i+1]]] += 1
rs = counts.sum(axis=1, keepdims=True); rs[rs==0]=1.0
P = counts / rs

print(f"\n  15m Transition matrix (rows=from, cols=to):")
print(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
for i, s in enumerate(states_15m):
    row = "  ".join(f"{P[i,j]*100:7.2f}%" for j in range(3))
    print(f"  {s:>9s}  {row}")

cur_reg = reg_f.iloc[-1]
cur_ts  = reg_f.index[-1]
print(f"\n  Current 15m regime: {cur_reg}  ({cur_ts})")
cur_idx = states_idx[cur_reg]
for j, s in enumerate(states_15m):
    print(f"    → {s:<10s}: {P[cur_idx,j]*100:.1f}%")

# ── 8. Rescue within 15m Sideways (if it has value) ──────────────────────────
print(f"\n{SEP2}")
print("  15m SIDEWAYS REGIME — rescue search (same features as 1h analysis)")
sw15 = df[df["reg_15m"]=="Sideways"]
if len(sw15) >= MIN_N:
    print(f"\n  Sideways trades: {len(sw15)}  WR={sw15['won'].mean()*100:.1f}%  P&L=${sw15['flat_pnl'].sum():+,.0f}")
    for feat in ["p_market","composite_p_up","offset_pct","stoch_k_15m","stoch_k_1h",
                 "ema_bias","bp_15m","vol_ratio","chg_15m","chg_1h"]:
        if feat not in sw15.columns:
            continue
        sw15[feat] = pd.to_numeric(sw15[feat], errors="coerce")
        col = sw15[feat].dropna()
        if len(col) < MIN_N:
            continue
        q25, q75 = col.quantile(0.25), col.quantile(0.75)
        for mask_val, label in [(sw15[feat]<=q25, f"≤Q1({q25:.3g})"),
                                 (sw15[feat]>=q75, f"≥Q3({q75:.3g})")]:
            sub = sw15[mask_val.fillna(False)]
            if len(sub) >= MIN_N:
                report(sub, f"  {feat} {label}")
