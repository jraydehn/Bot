"""
analyze_markov_15m_rescues.py

Rescue analysis for two 15m gate candidates:
  Gate A: YES when 1h Markov regime = Bear  (n=105, WR=48.6%, -$758)
  Gate B: YES when 15m Markov regime = Bear (n=68,  WR=47.1%, -$529)

Sweeps every available feature for subsets that rescue WR to ≥ 55%.
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

def show(sub, label):
    n = len(sub)
    if n < MIN_N:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    return {"label": label, "n": n, "wr": wr, "pnl": pnl}

def print_row(r):
    if r is None: return
    print(f"  {r['label']:<50s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}")

# ── 1. Build regimes ──────────────────────────────────────────────────────────
print("Fetching price data...")
df_1h = yf.download("BTC-USD", start="2024-11-01", end="2026-05-23",
                    interval="1h", progress=False, auto_adjust=True)
if isinstance(df_1h.columns, pd.MultiIndex):
    df_1h.columns = df_1h.columns.get_level_values(0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)

df_15m_prices = yf.download("BTC-USD", start="2026-04-01", end="2026-05-23",
                             interval="15m", progress=False, auto_adjust=True)
if isinstance(df_15m_prices.columns, pd.MultiIndex):
    df_15m_prices.columns = df_15m_prices.columns.get_level_values(0)
df_15m_prices.index = pd.to_datetime(df_15m_prices.index, utc=True)

def make_regime(close, window, threshold):
    rr  = close.pct_change(window)
    reg = pd.Series("Sideways", index=close.index)
    reg[rr >  threshold] = "Bull"
    reg[rr < -threshold] = "Bear"
    reg = reg[rr.notna()]
    reg.index = pd.to_datetime(reg.index, utc=True).as_unit("us")
    return reg

def join_regime(df_trades, regime_series, col_name):
    reg_df = regime_series.reset_index()
    reg_df.columns = ["_jts", col_name]
    reg_df["_jts"] = pd.to_datetime(reg_df["_jts"]).dt.as_unit("us")
    merged = pd.merge_asof(
        df_trades.sort_values("trade_ts"),
        reg_df.sort_values("_jts"),
        left_on="trade_ts", right_on="_jts",
        direction="backward",
    )
    return merged.drop(columns=["_jts"], errors="ignore")

reg_1h  = make_regime(df_1h["Close"].dropna(),            window=20, threshold=0.008)
reg_15m = make_regime(df_15m_prices["Close"].dropna(),    window=20, threshold=0.004)

# ── 2. Load trades ────────────────────────────────────────────────────────────
print("Loading trades...")
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

df = join_regime(df, reg_1h,  "reg_1h")
df = join_regime(df, reg_15m, "reg_15m")
for col in ["reg_1h","reg_15m"]:
    df[col] = df[col].fillna("Unknown")

# Numeric coerce all feature columns
FEATURES = [
    "p_market","offset_pct","tau_minutes","net_edge",
    "composite_p_up","bp_15m","bp_1h","vol_ratio","vol_ratio_5m","vol_ratio_1h",
    "body_15m","upper_wick_15m","lower_wick_15m","atr_ratio_15m","range_ratio_15m",
    "consec_dir_15m","consec_dir_1h","dir_15m","dir_1h",
    "stoch_k_15m","stoch_k_1h","stoch_k_5m","stoch_cross_1h",
    "chg_1m","chg_5m","chg_15m","chg_1h",
    "vwap_dist","ema_bias","ema_bias_1h",
    "realized_vol_annual","rsi_1h","macd_hist_1h",
    "donchian_breakout_1h","engulfing_1h",
    "liq_score","liq_bias","oi_chg_pct","ls_long_pct",
    "fear_greed","cg_composite","spread",
    "nearest_res_dist_pct",
]
for f in FEATURES:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors="coerce")

# ── 3. Define gate sets ───────────────────────────────────────────────────────
gate_a = df[(df["reg_1h"]  == "Bear") & (df["side"] == "yes")]   # 1h Bear YES
gate_b = df[(df["reg_15m"] == "Bear") & (df["side"] == "yes")]   # 15m Bear YES

print(f"\nGate A — 1h Bear YES:  n={len(gate_a)}, WR={gate_a['won'].mean()*100:.1f}%, P&L=${gate_a['flat_pnl'].sum():+,.0f}")
print(f"Gate B — 15m Bear YES: n={len(gate_b)}, WR={gate_b['won'].mean()*100:.1f}%, P&L=${gate_b['flat_pnl'].sum():+,.0f}")

# ── 4. Sweep rescues ──────────────────────────────────────────────────────────
def sweep(gate_df, gate_label):
    print(f"\n{SEP}")
    print(f"  RESCUE SWEEP: {gate_label}")
    print(SEP)

    rescues = []

    # Continuous threshold sweeps
    for feat in FEATURES:
        if feat not in gate_df.columns:
            continue
        col = gate_df[feat].dropna()
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
            mask = (gate_df[feat] >= cut) if op == ">=" else (gate_df[feat] <= cut)
            r = show(gate_df[mask.fillna(False)], f"{feat} {label}")
            if r and r["wr"] >= 0.58:
                rescues.append(r)

    # Binary splits
    for feat in ["dir_15m","dir_1h","consec_dir_15m","consec_dir_1h",
                 "donchian_breakout_1h","engulfing_1h","stoch_cross_1h","liq_bias"]:
        if feat not in gate_df.columns:
            continue
        for val in gate_df[feat].dropna().unique():
            mask = gate_df[feat] == val
            r = show(gate_df[mask], f"{feat}={val}")
            if r and r["wr"] >= 0.58:
                rescues.append(r)

    rescues.sort(key=lambda x: x["pnl"], reverse=True)

    if rescues:
        print(f"\n  RESCUE CANDIDATES (WR≥58%, n≥{MIN_N}) sorted by P&L:")
        print(f"  {'Condition':<50s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
        print("  " + "-" * 74)
        for r in rescues[:25]:
            print_row(r)
    else:
        print(f"\n  No rescue candidates at WR≥58%.")

    # Targeted combos
    print(f"\n  COMBO SWEEP (p_market × stoch × z/chg × liq):")
    combos = []

    for pm_label, pm_mask in [
        ("pm≤0.35", gate_df["p_market"] <= 0.35),
        ("pm≤0.45", gate_df["p_market"] <= 0.45),
        ("pm≤0.55", gate_df["p_market"] <= 0.55),
        ("pm≥0.55", gate_df["p_market"] >= 0.55),
        ("pm≥0.65", gate_df["p_market"] >= 0.65),
    ]:
        sub_pm = gate_df[pm_mask.fillna(False)]
        r = show(sub_pm, f"pm: {pm_label}")
        if r: combos.append(r)

        for sk_label, sk_mask in [
            ("sk15≤30", sub_pm["stoch_k_15m"] <= 30 if "stoch_k_15m" in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("sk15≥70", sub_pm["stoch_k_15m"] >= 70 if "stoch_k_15m" in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("sk1h≤30", sub_pm["stoch_k_1h"]  <= 30 if "stoch_k_1h"  in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("sk1h≥70", sub_pm["stoch_k_1h"]  >= 70 if "stoch_k_1h"  in sub_pm else pd.Series(False, index=sub_pm.index)),
        ]:
            sub_sk = sub_pm[sk_mask.fillna(False)]
            r = show(sub_sk, f"pm: {pm_label} + {sk_label}")
            if r: combos.append(r)

        for chg_label, chg_mask in [
            ("chg15m≤-0.3%", sub_pm["chg_15m"] <= -0.003 if "chg_15m" in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("chg15m≥0.3%",  sub_pm["chg_15m"] >= 0.003  if "chg_15m" in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("chg1h≤-0.5%",  sub_pm["chg_1h"]  <= -0.005 if "chg_1h"  in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("chg1h≥0.5%",   sub_pm["chg_1h"]  >= 0.005  if "chg_1h"  in sub_pm else pd.Series(False, index=sub_pm.index)),
        ]:
            sub_chg = sub_pm[chg_mask.fillna(False)]
            r = show(sub_chg, f"pm: {pm_label} + {chg_label}")
            if r: combos.append(r)

        for liq_label, liq_mask in [
            ("liq≥1", sub_pm["liq_score"] >= 1 if "liq_score" in sub_pm else pd.Series(False, index=sub_pm.index)),
            ("liq≤-1", sub_pm["liq_score"] <= -1 if "liq_score" in sub_pm else pd.Series(False, index=sub_pm.index)),
        ]:
            sub_liq = sub_pm[liq_mask.fillna(False)]
            r = show(sub_liq, f"pm: {pm_label} + {liq_label}")
            if r: combos.append(r)

    combos.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n  {'Condition':<50s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 74)
    for r in combos[:20]:
        print_row(r)

    # Summary
    print(f"\n  HARD BLOCK all {len(gate_df)} trades: saves ${-gate_df['flat_pnl'].sum():+,.0f}")
    best_rescues = [r for r in (rescues + combos) if r["wr"] >= 0.58]
    best_rescues.sort(key=lambda x: x["pnl"], reverse=True)
    if best_rescues:
        print(f"\n  Best rescue (keep these, block the rest):")
        seen = set()
        for r in best_rescues[:5]:
            if r["label"] in seen: continue
            seen.add(r["label"])
            rest = gate_df[~gate_df.index.isin(
                gate_df.index  # approximate — just show the math
            )]
            saved = -gate_df["flat_pnl"].sum() + r["pnl"]
            print(f"    Keep {r['n']} trades ({r['label']}) WR={r['wr']*100:.1f}% P&L=${r['pnl']:+,.0f}")
            print(f"    Block remaining {len(gate_df)-r['n']} → net saved vs baseline: ${saved:+,.0f}")

sweep(gate_a, "Gate A: 1h Bear YES (n=105, WR=48.6%)")
sweep(gate_b, "Gate B: 15m Bear YES (n=68, WR=47.1%)")

# ── 5. Overlap between A and B ────────────────────────────────────────────────
print(f"\n{SEP}")
print("  OVERLAP: trades in BOTH 1h Bear AND 15m Bear YES")
print(SEP)
both = df[(df["reg_1h"]=="Bear") & (df["reg_15m"]=="Bear") & (df["side"]=="yes")]
print(f"\n  n={len(both)}  WR={both['won'].mean()*100:.1f}%  P&L=${both['flat_pnl'].sum():+,.0f}")
only_1h = df[(df["reg_1h"]=="Bear") & (df["reg_15m"]!="Bear") & (df["side"]=="yes")]
only_15m = df[(df["reg_1h"]!="Bear") & (df["reg_15m"]=="Bear") & (df["side"]=="yes")]
print(f"  Only 1h Bear (not 15m Bear): n={len(only_1h)}, WR={only_1h['won'].mean()*100:.1f}% P&L=${only_1h['flat_pnl'].sum():+,.0f}")
print(f"  Only 15m Bear (not 1h Bear): n={len(only_15m)}, WR={only_15m['won'].mean()*100:.1f}% P&L=${only_15m['flat_pnl'].sum():+,.0f}")
