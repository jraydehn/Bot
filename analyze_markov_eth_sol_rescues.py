"""
analyze_markov_eth_sol_rescues.py

Full rescue sweep for every flagged regime×side cell from the ETH/SOL Markov sweep.

Flagged cells (15m model):
  ETH daily Sideways ALL   (n=114, WR=37.7%, -$793)
  ETH 4h Bull ALL          (n=66,  WR=28.8%, -$745)
  SOL 6h Bull ALL          (n=177, WR=45.2%, -$549)
  SOL 4h Sideways ALL      (n=255, WR=50.2%, -$1,240)
  SOL 1h Sideways YES      (n=183, WR=48.6%, -$852)

For each cell:
  - Full feature threshold sweep (every logged signal, YES+NO separately)
  - Binary/categorical splits
  - Top combos
  - Hard block vs best rescue comparison
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

FLAT_BET    = 25.0
KALSHI_TAKE = 0.10
MIN_N       = 8
SEP  = "=" * 72
SEP2 = "-" * 56

def flat_pnl_row(row):
    try:
        p = float(row["p_market"])
        if not (0 < p < 1):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = max(1 - p, 1e-6)
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE)
        return (payout - FLAT_BET) if won else -FLAT_BET

def prep_trades_15m(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["decision"] == "trade"].copy()
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df["p_market"]  = pd.to_numeric(df["p_market"],  errors="coerce")
    df["trade_ts"]  = pd.to_datetime(df["logged_at"], format="ISO8601", utc=True)
    df = df.dropna(subset=["trade_ts", "p_market", "side"]).copy()
    df["flat_pnl"]  = df.apply(flat_pnl_row, axis=1)
    df["won"] = (
        ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
        ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    ).astype(int)
    return df

def build_regime(close, window, threshold):
    rr  = close.pct_change(window)
    reg = pd.Series("Sideways", index=close.index)
    reg[rr >  threshold] = "Bull"
    reg[rr < -threshold] = "Bear"
    reg = reg[rr.notna()]
    reg.index = pd.to_datetime(reg.index, utc=True)
    return reg

def join_regime(df_trades, regime_series, col):
    reg_df = regime_series.reset_index()
    reg_df.columns = ["_jts", col]
    reg_df["_jts"] = pd.to_datetime(reg_df["_jts"], utc=True).dt.as_unit("us")
    df = df_trades.copy()
    df["_ts"] = df["trade_ts"].dt.as_unit("us")
    merged = pd.merge_asof(
        df.sort_values("_ts"),
        reg_df.sort_values("_jts"),
        left_on="_ts", right_on="_jts",
        direction="backward",
    ).drop(columns=["_jts", "_ts"], errors="ignore")
    merged[col] = merged[col].fillna("Unknown")
    return merged

def show(sub, label):
    n = len(sub)
    if n < MIN_N:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    return {"label": label, "n": n, "wr": wr, "pnl": pnl}

def print_row(r, marker=""):
    if r is None: return
    flag = f"  ◄ {marker}" if marker else ""
    print(f"  {r['label']:<52s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}{flag}")

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
    "net_edge",
]
BINARY_FEATURES = [
    "dir_15m", "dir_1h", "consec_dir_15m", "consec_dir_1h",
    "donchian_breakout_1h", "engulfing_1h", "stoch_cross_1h",
    "liq_bias", "ema_bias", "ema_bias_1h",
]

def full_rescue_sweep(gate_df, gate_label, wr_threshold=0.58):
    """Run full feature sweep on a gate subset, for both YES and NO separately."""
    print(f"\n{SEP}")
    print(f"  RESCUE SWEEP: {gate_label}")
    print(f"  Baseline: n={len(gate_df)}  WR={gate_df['won'].mean()*100:.1f}%  "
          f"P&L=${gate_df['flat_pnl'].sum():+,.0f}")
    print(SEP)

    for side in ["yes", "no"]:
        sub_side = gate_df[gate_df["side"] == side]
        if len(sub_side) < MIN_N:
            continue

        print(f"\n  [{side.upper()}  n={len(sub_side)}  WR={sub_side['won'].mean()*100:.1f}%  "
              f"P&L=${sub_side['flat_pnl'].sum():+,.0f}]")

        rescues = []

        # Numeric feature threshold sweeps
        for feat in FEATURES:
            if feat not in sub_side.columns:
                continue
            col = pd.to_numeric(sub_side[feat], errors="coerce").dropna()
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
                vals = pd.to_numeric(sub_side[feat], errors="coerce")
                mask = (vals >= cut) if op == ">=" else (vals <= cut)
                r = show(sub_side[mask.fillna(False)], f"{feat} {label}")
                if r and r["wr"] >= wr_threshold:
                    rescues.append(r)

        # Binary splits
        for feat in BINARY_FEATURES:
            if feat not in sub_side.columns:
                continue
            for val in pd.to_numeric(sub_side[feat], errors="coerce").dropna().unique():
                mask = pd.to_numeric(sub_side[feat], errors="coerce") == val
                r = show(sub_side[mask.fillna(False)], f"{feat}={val}")
                if r and r["wr"] >= wr_threshold:
                    rescues.append(r)

        rescues.sort(key=lambda x: x["pnl"], reverse=True)

        if rescues:
            print(f"\n  Rescue candidates (WR≥{wr_threshold*100:.0f}%, n≥{MIN_N}), sorted by P&L:")
            print(f"  {'Condition':<52s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
            print("  " + "─" * 74)
            for r in rescues[:20]:
                marker = "PROFITABLE" if r["pnl"] > 0 else ""
                print_row(r, marker)
        else:
            print(f"  No rescue candidates at WR≥{wr_threshold*100:.0f}%.")

    # Hard block summary
    total_pnl = gate_df["flat_pnl"].sum()
    print(f"\n  Hard block all {len(gate_df)} trades saves: ${-total_pnl:+,.0f}")

    # Best rescue vs hard block
    all_rescues = []
    for side in ["yes", "no"]:
        sub_side = gate_df[gate_df["side"] == side]
        if len(sub_side) < MIN_N:
            continue
        for feat in FEATURES:
            if feat not in sub_side.columns:
                continue
            col = pd.to_numeric(sub_side[feat], errors="coerce").dropna()
            if len(col) < MIN_N:
                continue
            q25, q50, q75 = col.quantile([0.25, 0.50, 0.75])
            for op, cut in [(">=", q25), (">=", q50), (">=", q75),
                             ("<=", q25), ("<=", q50), ("<=", q75)]:
                vals = pd.to_numeric(sub_side[feat], errors="coerce")
                mask = (vals >= cut) if op == ">=" else (vals <= cut)
                kept = sub_side[mask.fillna(False)]
                r = show(kept, f"{side} {feat}{op}{cut:.3g}")
                if r and r["wr"] >= wr_threshold and r["pnl"] > 0:
                    all_rescues.append(r)
        for feat in BINARY_FEATURES:
            if feat not in sub_side.columns:
                continue
            for val in pd.to_numeric(sub_side[feat], errors="coerce").dropna().unique():
                mask = pd.to_numeric(sub_side[feat], errors="coerce") == val
                kept = sub_side[mask.fillna(False)]
                r = show(kept, f"{side} {feat}={val}")
                if r and r["wr"] >= wr_threshold and r["pnl"] > 0:
                    all_rescues.append(r)

    all_rescues.sort(key=lambda x: x["pnl"], reverse=True)

    if all_rescues:
        print(f"\n  PROFITABLE rescues (keep these, block the rest):")
        seen = set()
        for r in all_rescues[:6]:
            if r["label"] in seen: continue
            seen.add(r["label"])
            blocked_pnl = total_pnl - r["pnl"]
            saved_vs_block = -blocked_pnl - (-total_pnl) if total_pnl < 0 else -blocked_pnl
            print(f"    Keep: {r['label']:<52s}  n={r['n']}  WR={r['wr']*100:.1f}%  P&L=${r['pnl']:+,.0f}")
            print(f"    Block rest: saves ${-blocked_pnl:+,.0f} vs hard-block-all saves ${-total_pnl:+,.0f}")
    else:
        print(f"\n  No profitable rescues found — hard block is optimal.")

# ── Fetch price data ──────────────────────────────────────────────────────────
print("Fetching price data...")
price_1h = {}
for asset, ticker in [("ETH", "ETH-USD"), ("SOL", "SOL-USD")]:
    df = yf.download(ticker, start="2024-11-01", end="2026-05-23",
                     interval="1h", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    price_1h[asset] = df["Close"].dropna()
    print(f"  {asset}: {len(price_1h[asset])} 1h bars")

print("Fetching SOL 15m prices for 15m regime...")
sol_15m_p = yf.download("SOL-USD", start="2026-04-01", end="2026-05-23",
                         interval="15m", progress=False, auto_adjust=True)
if isinstance(sol_15m_p.columns, pd.MultiIndex):
    sol_15m_p.columns = sol_15m_p.columns.get_level_values(0)
sol_15m_p.index = pd.to_datetime(sol_15m_p.index, utc=True)
sol_close_15m = sol_15m_p["Close"].dropna()

# ── Load trades ───────────────────────────────────────────────────────────────
print("Loading trades...")
eth_15m = prep_trades_15m("results/paper_trades_eth15m.csv")
sol_15m = prep_trades_15m("results/paper_trades_sol15m.csv")
print(f"  ETH 15m: {len(eth_15m)} trades  SOL 15m: {len(sol_15m)} trades")

# Numeric-coerce all features
for df in [eth_15m, sol_15m]:
    for f in FEATURES:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

# ── Build regimes ─────────────────────────────────────────────────────────────
reg_eth_1d  = build_regime(price_1h["ETH"].resample("1D").last().dropna(), 20, 0.030)
reg_eth_4h  = build_regime(price_1h["ETH"].resample("4h").last().dropna(), 20, 0.015)
reg_sol_6h  = build_regime(price_1h["SOL"].resample("6h").last().dropna(), 20, 0.030)
reg_sol_4h  = build_regime(price_1h["SOL"].resample("4h").last().dropna(), 20, 0.025)
reg_sol_1h  = build_regime(price_1h["SOL"], 20, 0.015)

# Join regimes to trade frames
eth_15m = join_regime(eth_15m, reg_eth_1d, "reg_1d")
eth_15m = join_regime(eth_15m, reg_eth_4h, "reg_4h")
sol_15m = join_regime(sol_15m, reg_sol_6h, "reg_6h")
sol_15m = join_regime(sol_15m, reg_sol_4h, "reg_4h")
sol_15m = join_regime(sol_15m, reg_sol_1h, "reg_1h")

# ── Gate subsets ──────────────────────────────────────────────────────────────
eth_1d_sw   = eth_15m[eth_15m["reg_1d"] == "Sideways"]
eth_4h_bull = eth_15m[eth_15m["reg_4h"] == "Bull"]
sol_6h_bull = sol_15m[sol_15m["reg_6h"] == "Bull"]
sol_4h_sw   = sol_15m[sol_15m["reg_4h"] == "Sideways"]
sol_1h_sw_y = sol_15m[(sol_15m["reg_1h"] == "Sideways") & (sol_15m["side"] == "yes")]

# ── Quick overlap check ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  OVERLAP CHECK")
print(SEP)
overlap_eth = eth_15m[(eth_15m["reg_1d"] == "Sideways") & (eth_15m["reg_4h"] == "Bull")]
print(f"\n  ETH: daily Sideways ∩ 4h Bull = {len(overlap_eth)} trades "
      f"(of {len(eth_1d_sw)} daily-SW and {len(eth_4h_bull)} 4h-Bull)")
print(f"  ETH: union (daily SW + 4h Bull, any overlap counted once) = "
      f"{len(eth_15m[(eth_15m['reg_1d']=='Sideways') | (eth_15m['reg_4h']=='Bull')])} trades")

# ── Run rescue sweeps ─────────────────────────────────────────────────────────
full_rescue_sweep(eth_1d_sw,
    "ETH 15m — daily Sideways (WR=37.7%, -$793, n=114)")

full_rescue_sweep(eth_4h_bull,
    "ETH 15m — 4h Bull (WR=28.8%, -$745, n=66)")

full_rescue_sweep(sol_6h_bull,
    "SOL 15m — 6h Bull (WR=45.2%, -$549, n=177)")

full_rescue_sweep(sol_4h_sw,
    "SOL 15m — 4h Sideways (WR=50.2%, -$1,240, n=255)")

# SOL 1h Sideways YES — run as single-side (only YES)
print(f"\n{SEP}")
print("  RESCUE SWEEP: SOL 15m — 1h Sideways YES (WR=48.6%, -$852, n=183)")
print(f"  Baseline: n={len(sol_1h_sw_y)}  WR={sol_1h_sw_y['won'].mean()*100:.1f}%  "
      f"P&L=${sol_1h_sw_y['flat_pnl'].sum():+,.0f}")
print(SEP)

rescues_sol_sw_y = []
for feat in FEATURES:
    if feat not in sol_1h_sw_y.columns:
        continue
    col = pd.to_numeric(sol_1h_sw_y[feat], errors="coerce").dropna()
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
        vals = pd.to_numeric(sol_1h_sw_y[feat], errors="coerce")
        mask = (vals >= cut) if op == ">=" else (vals <= cut)
        r = show(sol_1h_sw_y[mask.fillna(False)], f"{feat} {label}")
        if r and r["wr"] >= 0.58:
            rescues_sol_sw_y.append(r)

for feat in BINARY_FEATURES:
    if feat not in sol_1h_sw_y.columns:
        continue
    for val in pd.to_numeric(sol_1h_sw_y[feat], errors="coerce").dropna().unique():
        mask = pd.to_numeric(sol_1h_sw_y[feat], errors="coerce") == val
        r = show(sol_1h_sw_y[mask.fillna(False)], f"{feat}={val}")
        if r and r["wr"] >= 0.58:
            rescues_sol_sw_y.append(r)

rescues_sol_sw_y.sort(key=lambda x: x["pnl"], reverse=True)
if rescues_sol_sw_y:
    print(f"\n  Rescue candidates (WR≥58%):")
    print(f"  {'Condition':<52s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "─" * 74)
    for r in rescues_sol_sw_y[:20]:
        print_row(r, "PROFITABLE" if r["pnl"] > 0 else "")
else:
    print("  No rescue candidates at WR≥58%.")
total_pnl_sol_sw_y = sol_1h_sw_y["flat_pnl"].sum()
print(f"\n  Hard block all {len(sol_1h_sw_y)} trades saves: ${-total_pnl_sol_sw_y:+,.0f}")
