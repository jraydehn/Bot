"""
analyze_6h_sideways_no.py

Deep dive into the 6h Sideways NO cell from the 15m BTC model.
  6h Sideways NO: n=162, WR=56.8%, P&L=+$318 — the only profitable regime×side cell.

Questions:
  1. Is the +$318 robust or driven by a few outlier wins?
  2. What subsets within 6h Sideways NO are the strongest?
  3. Does 6h Sideways add conviction vs baseline NO (i.e., is it a positive filter)?
  4. Breakeven WR for each p_market bin, and which bins clear it?
  5. Overlap with existing gates — what fraction would have been blocked already?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

TRADES_CSV  = "results/paper_trades_btc15m.csv"
FLAT_BET    = 25.0
KALSHI_TAKE = 0.10
MIN_N       = 8
SEP  = "=" * 68
SEP2 = "-" * 52

# ── helpers ───────────────────────────────────────────────────────────────────
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

def breakeven_wr(p_market, side):
    """Minimum WR needed to be profitable at given p_market."""
    if side == "no":
        p_no = max(1 - p_market, 1e-6)
        payout_net = FLAT_BET / p_no * (1 - KALSHI_TAKE)
        return FLAT_BET / payout_net
    else:
        payout_net = FLAT_BET / p_market * (1 - KALSHI_TAKE)
        return FLAT_BET / payout_net

def show(sub, label, be_wr=None):
    n = len(sub)
    if n < MIN_N:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    r = {"label": label, "n": n, "wr": wr, "pnl": pnl}
    if be_wr is not None:
        r["be_wr"] = be_wr
        r["edge_vs_be"] = wr - be_wr
    return r

def print_row(r):
    if r is None: return
    be_str = f"  BE={r['be_wr']*100:4.1f}%  edge={r['edge_vs_be']*100:+4.1f}pp" if "be_wr" in r else ""
    print(f"  {r['label']:<52s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}{be_str}")

# ── 1. Build 6h regime ────────────────────────────────────────────────────────
print("Fetching BTC-USD 1h data...")
df_1h = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)
df_6h = df_1h.resample("6h").agg({"Close": "last"}).dropna()
close_6h = df_6h["Close"]

roll_ret = close_6h.pct_change(20)
reg_6h = pd.Series("Sideways", index=close_6h.index)
reg_6h[roll_ret >  0.015] = "Bull"
reg_6h[roll_ret < -0.015] = "Bear"
reg_6h = reg_6h[roll_ret.notna()]
reg_6h.index = pd.to_datetime(reg_6h.index, utc=True)

# ── 2. Load trades ─────────────────────────────────────────────────────────────
print("Loading trades...")
df = pd.read_csv(TRADES_CSV, low_memory=False)
df = df[df["decision"] == "trade"].copy()
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df["p_market"]  = pd.to_numeric(df["p_market"],  errors="coerce")
df["trade_ts"]  = pd.to_datetime(df["logged_at"], format="ISO8601", utc=True)
df["flat_pnl"]  = df.apply(flat_pnl, axis=1)
df["won"] = (
    ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
    ((df["side"] == "no")  & (df["resolved_yes"] == 0))
).astype(int)

# Numeric features
FEATURES = [
    "p_market", "offset_pct", "tau_minutes", "composite_p_up",
    "bp_15m", "bp_1h", "vol_ratio", "vol_ratio_5m", "vol_ratio_1h",
    "body_15m", "upper_wick_15m", "lower_wick_15m", "atr_ratio_15m",
    "consec_dir_15m", "consec_dir_1h", "dir_15m", "dir_1h",
    "stoch_k_15m", "stoch_k_1h", "stoch_k_5m", "stoch_cross_1h",
    "chg_1m", "chg_5m", "chg_15m", "chg_1h",
    "vwap_dist", "ema_bias", "ema_bias_1h",
    "realized_vol_annual", "rsi_1h", "macd_hist_1h",
    "donchian_breakout_1h", "engulfing_1h",
    "liq_score", "liq_bias", "oi_chg_pct", "ls_long_pct",
    "fear_greed", "cg_composite", "spread",
]
for f in FEATURES:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors="coerce")

# Join 6h regime
reg_df = reg_6h.reset_index()
reg_df.columns = ["_jts", "reg_6h"]
reg_df["_jts"] = pd.to_datetime(reg_df["_jts"], utc=True).dt.as_unit("us")
df["_ts_us"] = df["trade_ts"].dt.as_unit("us")
df = pd.merge_asof(
    df.sort_values("_ts_us"),
    reg_df.sort_values("_jts"),
    left_on="_ts_us", right_on="_jts",
    direction="backward",
).drop(columns=["_jts", "_ts_us"], errors="ignore")
df["reg_6h"] = df["reg_6h"].fillna("Unknown")

# ── 3. Baselines ──────────────────────────────────────────────────────────────
all_no   = df[df["side"] == "no"]
sw_no    = df[(df["reg_6h"] == "Sideways") & (df["side"] == "no")]
all_yes  = df[df["side"] == "yes"]
sw_yes   = df[(df["reg_6h"] == "Sideways") & (df["side"] == "yes")]

print(f"\n{SEP}")
print("  BASELINES")
print(SEP)
print(f"\n  All 15m BTC trades:    n={len(df)}   WR={df['won'].mean()*100:.1f}%   P&L=${df['flat_pnl'].sum():+,.0f}")
print(f"  All NO trades:         n={len(all_no)}  WR={all_no['won'].mean()*100:.1f}%   P&L=${all_no['flat_pnl'].sum():+,.0f}")
print(f"  6h Sideways all:       n={len(df[df['reg_6h']=='Sideways'])}  WR={df[df['reg_6h']=='Sideways']['won'].mean()*100:.1f}%   P&L=${df[df['reg_6h']=='Sideways']['flat_pnl'].sum():+,.0f}")
print(f"  6h Sideways NO:        n={len(sw_no)}  WR={sw_no['won'].mean()*100:.1f}%   P&L=${sw_no['flat_pnl'].sum():+,.0f}  ← target")
print(f"  6h Sideways YES:       n={len(sw_yes)}  WR={sw_yes['won'].mean()*100:.1f}%   P&L=${sw_yes['flat_pnl'].sum():+,.0f}")

# Lift: 6h Sideways NO vs all NO
lift_wr  = sw_no["won"].mean() - all_no["won"].mean()
lift_pnl = sw_no["flat_pnl"].mean() - all_no["flat_pnl"].mean()
print(f"\n  Lift of 6h Sideways NO vs all NO:  WR {lift_wr*100:+.1f}pp   avg P&L/trade ${lift_pnl:+.2f}")

# ── 4. p_market bins with breakeven WR ────────────────────────────────────────
print(f"\n{SEP}")
print("  p_market BINS — breakeven WR vs observed WR  (6h Sideways NO)")
print(SEP)
pm_bins = [
    (0.00, 0.30, "pm < 0.30"),
    (0.30, 0.40, "pm [0.30, 0.40)"),
    (0.40, 0.50, "pm [0.40, 0.50)"),
    (0.50, 0.60, "pm [0.50, 0.60)"),
    (0.60, 0.70, "pm [0.60, 0.70)"),
    (0.70, 1.00, "pm ≥ 0.70"),
]
print(f"\n  {'Range':<20s}  {'n':>4s}  {'WR':>6s}  {'BE_WR':>6s}  {'Edge':>7s}  {'P&L':>8s}")
print("  " + "-" * 62)
for lo, hi, label in pm_bins:
    sub = sw_no[(sw_no["p_market"] >= lo) & (sw_no["p_market"] < hi)]
    if len(sub) < MIN_N:
        continue
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    avg_pm = sub["p_market"].mean()
    be = breakeven_wr(avg_pm, "no")
    print(f"  {label:<20s}  {len(sub):>4d}  {wr*100:5.1f}%  {be*100:5.1f}%  {(wr-be)*100:+6.1f}pp  ${pnl:+7.0f}")

# Same for all NO (comparison)
print(f"\n  {'Range':<20s}  {'n':>4s}  {'WR':>6s}  {'BE_WR':>6s}  {'Edge':>7s}  {'P&L':>8s}  (ALL NO, for reference)")
print("  " + "-" * 76)
for lo, hi, label in pm_bins:
    sub = all_no[(all_no["p_market"] >= lo) & (all_no["p_market"] < hi)]
    if len(sub) < MIN_N:
        continue
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    avg_pm = sub["p_market"].mean()
    be = breakeven_wr(avg_pm, "no")
    print(f"  {label:<20s}  {len(sub):>4d}  {wr*100:5.1f}%  {be*100:5.1f}%  {(wr-be)*100:+6.1f}pp  ${pnl:+7.0f}")

# ── 5. P&L stability — cumulative by trade date ────────────────────────────────
print(f"\n{SEP}")
print("  P&L STABILITY — 6h Sideways NO trades by date")
print(SEP)
sw_no_sorted = sw_no.sort_values("trade_ts").copy()
sw_no_sorted["cum_pnl"] = sw_no_sorted["flat_pnl"].cumsum()
sw_no_sorted["date"]    = sw_no_sorted["trade_ts"].dt.date

daily = sw_no_sorted.groupby("date").agg(
    n=("flat_pnl", "count"),
    pnl=("flat_pnl", "sum"),
    wr=("won", "mean"),
    cum_pnl=("cum_pnl", "last"),
).reset_index()
print(f"\n  {'Date':<12s}  {'n':>3s}  {'WR':>6s}  {'DailyP&L':>9s}  {'CumP&L':>9s}")
print("  " + "-" * 48)
for _, row in daily.iterrows():
    print(f"  {str(row['date']):<12s}  {row['n']:>3.0f}  {row['wr']*100:5.1f}%  "
          f"${row['pnl']:>+8.0f}  ${row['cum_pnl']:>+8.0f}")
print(f"\n  Total P&L: ${sw_no_sorted['flat_pnl'].sum():+,.0f}  over {len(daily)} trading days")
print(f"  Win days:  {(daily['pnl'] > 0).sum()}  /  {len(daily)}")

# ── 6. Feature sweep — find best subsets ─────────────────────────────────────
print(f"\n{SEP}")
print("  FEATURE SWEEP — best subsets within 6h Sideways NO (WR≥60%)")
print(SEP)

rescues = []

# Continuous threshold sweeps
for feat in FEATURES:
    if feat not in sw_no.columns:
        continue
    col = sw_no[feat].dropna()
    if len(col) < MIN_N:
        continue
    q25, q50, q75 = col.quantile([0.25, 0.50, 0.75])
    for op, cut, label in [
        (">=", q25, f"≥Q1({q25:.3g})"),
        (">=", q50, f"≥Med({q50:.3g})"),
        (">=", q75, f"≥Q3({q75:.3g})"),
        ("<=", q25, f"≤Q1({q25:.3g})"),
        ("<=", q50, f"≤Med({q50:.3g})"),
        ("<=", q75, f"≤Q3({q75:.3g})"),
    ]:
        mask = (sw_no[feat] >= cut) if op == ">=" else (sw_no[feat] <= cut)
        sub = sw_no[mask.fillna(False)]
        r = show(sub, f"{feat} {label}")
        if r and r["wr"] >= 0.60:
            rescues.append(r)

# Binary splits
for feat in ["dir_15m", "dir_1h", "consec_dir_15m", "consec_dir_1h",
             "donchian_breakout_1h", "engulfing_1h", "stoch_cross_1h",
             "liq_bias", "ema_bias", "ema_bias_1h"]:
    if feat not in sw_no.columns:
        continue
    for val in sw_no[feat].dropna().unique():
        mask = sw_no[feat] == val
        r = show(sw_no[mask], f"{feat}={val}")
        if r and r["wr"] >= 0.60:
            rescues.append(r)

rescues.sort(key=lambda x: x["pnl"], reverse=True)

if rescues:
    print(f"\n  Subsets with WR≥60% (sorted by P&L):")
    print(f"  {'Condition':<52s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 74)
    for r in rescues[:30]:
        print_row(r)
else:
    print(f"\n  No subsets found at WR≥60% with n≥{MIN_N}.")

# ── 7. Targeted combo sweep ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  COMBO SWEEP — p_market × trend/stoch × composite_p_up × liq")
print(SEP)

combos = []

for pm_label, pm_mask in [
    ("pm≤0.40", sw_no["p_market"] <= 0.40),
    ("pm≤0.45", sw_no["p_market"] <= 0.45),
    ("pm≤0.50", sw_no["p_market"] <= 0.50),
    ("pm[0.35,0.50]", (sw_no["p_market"] >= 0.35) & (sw_no["p_market"] <= 0.50)),
    ("pm≥0.50", sw_no["p_market"] >= 0.50),
    ("pm≥0.55", sw_no["p_market"] >= 0.55),
]:
    sub_pm = sw_no[pm_mask.fillna(False)]
    r = show(sub_pm, f"NO {pm_label}")
    if r: combos.append(r)

    for sk_label, sk_mask in [
        ("sk15≤30", sub_pm["stoch_k_15m"] <= 30),
        ("sk15≥70", sub_pm["stoch_k_15m"] >= 70),
        ("sk1h≤30", sub_pm["stoch_k_1h"]  <= 30),
        ("sk1h≥70", sub_pm["stoch_k_1h"]  >= 70),
    ]:
        r2 = show(sub_pm[sk_mask.fillna(False)], f"NO {pm_label} + {sk_label}")
        if r2: combos.append(r2)

    for cpu_label, cpu_mask in [
        ("cpu≤0.45", sub_pm["composite_p_up"] <= 0.45),
        ("cpu≤0.50", sub_pm["composite_p_up"] <= 0.50),
        ("cpu≥0.50", sub_pm["composite_p_up"] >= 0.50),
        ("cpu≥0.55", sub_pm["composite_p_up"] >= 0.55),
    ]:
        r2 = show(sub_pm[cpu_mask.fillna(False)], f"NO {pm_label} + {cpu_label}")
        if r2: combos.append(r2)

    for chg_label, chg_mask in [
        ("chg15m≤0%",   sub_pm["chg_15m"] <= 0.0),
        ("chg15m≥0%",   sub_pm["chg_15m"] >= 0.0),
        ("chg1h≤0%",    sub_pm["chg_1h"]  <= 0.0),
        ("chg1h≥0%",    sub_pm["chg_1h"]  >= 0.0),
    ]:
        r2 = show(sub_pm[chg_mask.fillna(False)], f"NO {pm_label} + {chg_label}")
        if r2: combos.append(r2)

    for liq_label, liq_mask in [
        ("liq≥1",  sub_pm["liq_score"] >= 1),
        ("liq≤-1", sub_pm["liq_score"] <= -1),
    ]:
        r2 = show(sub_pm[liq_mask.fillna(False)], f"NO {pm_label} + {liq_label}")
        if r2: combos.append(r2)

combos.sort(key=lambda x: x["pnl"], reverse=True)

print(f"\n  {'Condition':<52s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
print("  " + "-" * 74)
for r in combos[:25]:
    print_row(r)

# ── 8. Comparison: 6h Sideways NO vs other regime NO ─────────────────────────
print(f"\n{SEP}")
print("  REGIME COMPARISON — all NO trades by 6h regime")
print(SEP)
print(f"\n  {'Regime NO':<22s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}  avg_pm")
print("  " + "-" * 56)
for reg in ["Bull", "Sideways", "Bear", "Unknown"]:
    sub = df[(df["reg_6h"] == reg) & (df["side"] == "no")]
    if len(sub) < MIN_N:
        continue
    avg_pm = sub["p_market"].mean()
    r = show(sub, f"6h={reg} NO")
    if r:
        print(f"  {r['label']:<22s}  {r['n']:>4d}  {r['wr']*100:5.1f}%  ${r['pnl']:>+7.0f}  {avg_pm:.3f}")

# ── 9. Statistical test: is 6h Sideways NO WR significantly above 50%? ────────
print(f"\n{SEP}")
print("  STATISTICAL SIGNIFICANCE")
print(SEP)

n_sw_no = len(sw_no)
wins    = sw_no["won"].sum()
wr_obs  = sw_no["won"].mean()

# Binomial test: H0 = WR ≤ 50%
from scipy.stats import binomtest as _binomtest
binom_p = _binomtest(int(wins), n_sw_no, 0.5, alternative="greater").pvalue
print(f"\n  6h Sideways NO: n={n_sw_no}  wins={wins}  WR={wr_obs*100:.1f}%")
print(f"  Binomial test (H0: WR≤50%): p={binom_p:.4f}  "
      f"({'significant' if binom_p < 0.05 else 'NOT significant'} at p<0.05)")

# vs all NO
t, pt = stats.ttest_ind(sw_no["won"], all_no["won"])
print(f"  6h Sideways NO vs all NO (t-test): t={t:.3f}  p={pt:.4f}  "
      f"({'significant' if pt < 0.05 else 'NOT significant'})")

# ── 10. Overlap with existing gates ───────────────────────────────────────────
print(f"\n{SEP}")
print("  OVERLAP WITH EXISTING GATES")
print(SEP)

# Check which sw_no trades would already be blocked by existing gates in the runner
# Existing BTC gates affect YES side only — so all NO trades pass through.
# The question is: are there any existing NO filters that overlap?

# ema_bias, consec_dir, stoch
for feat, label in [
    ("ema_bias",    "ema_bias distribution"),
    ("consec_dir_15m", "consec_dir_15m distribution"),
    ("stoch_k_15m", "stoch_k_15m distribution"),
    ("composite_p_up", "composite_p_up distribution"),
]:
    if feat not in sw_no.columns:
        continue
    col = sw_no[feat].dropna()
    if len(col) < 5:
        continue
    print(f"\n  {label}: mean={col.mean():.3f}  median={col.median():.3f}  "
          f"q25={col.quantile(0.25):.3f}  q75={col.quantile(0.75):.3f}")

# ETH-style gate check: consec_dir_15m ≤ -1 (bearish streak for NO)
if "consec_dir_15m" in sw_no.columns:
    cd_neg = sw_no[sw_no["consec_dir_15m"] <= -1]
    cd_pos = sw_no[sw_no["consec_dir_15m"] >= 1]
    r1 = show(cd_neg, "consec_dir_15m ≤ -1  (bearish streak)")
    r2 = show(cd_pos, "consec_dir_15m ≥ +1  (bullish streak)")
    print(f"\n  Effect of 15m streak on Sideways NO:")
    print_row(r1)
    print_row(r2)

# ── 11. Summary ───────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)

best_combos = [r for r in (rescues + combos) if r["wr"] >= 0.60 and r["pnl"] > 0]
best_combos.sort(key=lambda x: x["pnl"], reverse=True)

print(f"\n  6h Sideways NO baseline:  n={n_sw_no}  WR={wr_obs*100:.1f}%  P&L=${sw_no['flat_pnl'].sum():+,.0f}")
print(f"  All NO baseline:          n={len(all_no)}  WR={all_no['won'].mean()*100:.1f}%  P&L=${all_no['flat_pnl'].sum():+,.0f}")
print(f"  Lift: {lift_wr*100:+.1f}pp WR,  ${lift_pnl:+.2f}/trade")
print(f"\n  Profitable subsets (WR≥60%, P&L>0):")
if best_combos:
    seen = set()
    for r in best_combos[:8]:
        if r["label"] in seen: continue
        seen.add(r["label"])
        print(f"    {r['label']:<52s}  n={r['n']}  WR={r['wr']*100:.1f}%  P&L=${r['pnl']:+,.0f}")
else:
    print("    None found.")
