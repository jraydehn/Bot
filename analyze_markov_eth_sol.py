"""
analyze_markov_eth_sol.py

Multi-timeframe Markov regime sweep for ETH and SOL.
Timeframes tested: daily, 6h, 4h, 1h, 15m.

For each asset × timeframe:
  1. Regime distribution and transition matrix persistence
  2. Next-candle directional accuracy (chi-square + t-test)
  3. Trade P&L by regime × side (1h model + 15m model)
  4. Gate candidates: cells with WR < 50% or P&L < -$50 flagged
"""

import warnings
warnings.filterwarnings("ignore")

import math
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
WINDOW      = 20

SEP  = "=" * 72
SEP2 = "-" * 56

# ── Thresholds per timeframe (scale with asset vol) ──────────────────────────
# Picked to give roughly equal thirds at the given window.
THRESHOLDS = {
    # (ETH_thresh, SOL_thresh)
    "1d":  (0.030, 0.045),
    "6h":  (0.020, 0.030),
    "4h":  (0.015, 0.025),
    "1h":  (0.010, 0.015),
    "15m": (0.005, 0.008),
}

ASSET_YFTICKER = {"ETH": "ETH-USD", "SOL": "SOL-USD"}

TRADES_1H_CSV  = {"ETH": "results/paper_trades_eth.csv",
                  "SOL": "results/paper_trades_sol.csv"}
TRADES_15M_CSV = {"ETH": "results/paper_trades_eth15m.csv",
                  "SOL": "results/paper_trades_sol15m.csv"}

# ── helpers ───────────────────────────────────────────────────────────────────
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

def prep_trades_1h(csv_path):
    """Load resolved 1h-model trades (ETH/SOL hourly runner)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
    ts_col = "logged_at" if "logged_at" in df.columns else df.columns[0]
    df["trade_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["trade_ts", "p_market", "side"]).copy()
    df["flat_pnl"] = df.apply(flat_pnl_row, axis=1)
    df["won"] = (
        ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
        ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    ).astype(int)
    return df

def prep_trades_15m(csv_path):
    """Load resolved 15m-model trades."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["decision"] == "trade"].copy()
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
    df["trade_ts"] = pd.to_datetime(df["logged_at"], format="ISO8601", utc=True)
    df = df.dropna(subset=["trade_ts", "p_market", "side"]).copy()
    df["flat_pnl"] = df.apply(flat_pnl_row, axis=1)
    df["won"] = (
        ((df["side"] == "yes") & (df["resolved_yes"] == 1)) |
        ((df["side"] == "no")  & (df["resolved_yes"] == 0))
    ).astype(int)
    return df

def build_regime(close: pd.Series, window: int, threshold: float) -> pd.Series:
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

def print_row(r, flag=""):
    if r is None: return
    marker = f"  ◄ {flag}" if flag else ""
    print(f"  {r['label']:<42s}  n={r['n']:3d}  WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}{marker}")

def gate_flag(r):
    if r is None:
        return ""
    if r["wr"] < 0.50 and r["n"] >= 12:
        return "GATE?"
    if r["pnl"] < -100 and r["n"] >= 12:
        return "P&L loss"
    return ""

# ── Analyze one asset × timeframe ────────────────────────────────────────────
def analyze(asset, tf_label, close_prices, regime_series,
            trades_1h, trades_15m):
    states    = ["Bear", "Sideways", "Bull"]
    state_idx = {s: i for i, s in enumerate(states)}

    vc = regime_series.value_counts()
    total_bars = len(regime_series)
    counts_str = "  ".join(f"{s}={vc.get(s,0)}({vc.get(s,0)/total_bars*100:.0f}%)" for s in states)

    print(f"\n{'─'*72}")
    print(f"  {asset} / {tf_label}   {counts_str}")
    print(f"{'─'*72}")

    # Transition matrix persistence
    arr    = regime_series.to_numpy()
    counts = np.zeros((3, 3), dtype=float)
    for i in range(len(arr) - 1):
        counts[state_idx[arr[i]], state_idx[arr[i + 1]]] += 1
    rs = counts.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = counts / rs
    persist_str = "  ".join(f"{s}→{s}:{P[i,i]*100:.0f}%" for i, s in enumerate(states))
    print(f"  Persistence: {persist_str}")

    # Next-candle direction by regime
    df2 = pd.DataFrame({"regime": regime_series, "close": close_prices.reindex(regime_series.index)})
    df2 = df2.dropna(subset=["regime"]).sort_index()
    df2["next_close"] = close_prices.reindex(df2.index).shift(-1)
    df2["next_bull"]  = (df2["next_close"] > df2["close"]).astype(float)
    df2 = df2.dropna(subset=["next_bull"])

    base_rate = df2["next_bull"].mean()
    dir_parts = []
    for s in states:
        sub = df2[df2["regime"] == s]
        if len(sub) >= 20:
            acc  = sub["next_bull"].mean()
            lift = acc - base_rate
            dir_parts.append(f"{s}:{acc*100:.1f}%({lift*100:+.1f}pp)")
    print(f"  Next-bar bullish (base={base_rate*100:.1f}%): {' | '.join(dir_parts)}")

    # Chi-square significance
    bull_c = [(df2[df2["regime"] == s]["next_bull"] == 1).sum() for s in states]
    bear_c = [(df2[df2["regime"] == s]["next_bull"] == 0).sum() for s in states]
    try:
        _, p_chi, _, _ = stats.chi2_contingency(np.array([bull_c, bear_c]))
        sig = "p<0.05 ✓" if p_chi < 0.05 else f"p={p_chi:.3f} ns"
        print(f"  Chi-square: {sig}")
    except Exception:
        pass

    # Trade impact — 1h model
    if trades_1h is not None and len(trades_1h) > 0:
        df_j = join_regime(trades_1h, regime_series, "reg")
        print(f"\n  [1h model] n={len(df_j)} resolved  "
              f"WR={df_j['won'].mean()*100:.1f}%  P&L=${df_j['flat_pnl'].sum():+,.0f}")
        print(f"  {'Regime×Side':<42s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
        print("  " + "─" * 64)
        for reg in states:
            sub_all = df_j[df_j["reg"] == reg]
            r_all = show(sub_all, f"{reg} (all)")
            print_row(r_all, gate_flag(r_all))
            for side in ["yes", "no"]:
                r = show(sub_all[sub_all["side"] == side], f"  {reg}  {side}")
                print_row(r, gate_flag(r))

    # Trade impact — 15m model
    if trades_15m is not None and len(trades_15m) > 0:
        df_j = join_regime(trades_15m, regime_series, "reg")
        print(f"\n  [15m model] n={len(df_j)} resolved  "
              f"WR={df_j['won'].mean()*100:.1f}%  P&L=${df_j['flat_pnl'].sum():+,.0f}")
        print(f"  {'Regime×Side':<42s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
        print("  " + "─" * 64)
        for reg in states:
            sub_all = df_j[df_j["reg"] == reg]
            r_all = show(sub_all, f"{reg} (all)")
            print_row(r_all, gate_flag(r_all))
            for side in ["yes", "no"]:
                r = show(sub_all[sub_all["side"] == side], f"  {reg}  {side}")
                print_row(r, gate_flag(r))

            # p_market splits for flagged YES cells
            sub_yes = sub_all[sub_all["side"] == "yes"]
            r_yes = show(sub_yes, "")
            if r_yes and (r_yes["wr"] < 0.52 or r_yes["pnl"] < -100) and len(sub_yes) >= MIN_N:
                print(f"    [rescue sweep: {reg} YES]")
                for pm_cut, pm_label, pm_op in [
                    (0.40, "pm≤0.40", "le"), (0.50, "pm≤0.50", "le"),
                    (0.50, "pm≥0.50", "ge"), (0.55, "pm≥0.55", "ge"),
                    (0.60, "pm≥0.60", "ge"),
                ]:
                    if pm_op == "le":
                        mask = sub_yes["p_market"] <= pm_cut
                    else:
                        mask = sub_yes["p_market"] >= pm_cut
                    r2 = show(sub_yes[mask.fillna(False)], f"    {reg} YES {pm_label}")
                    if r2:
                        print_row(r2, "RESCUE" if r2["wr"] >= 0.58 else "")
                # composite_p_up if available
                if "composite_p_up" in sub_yes.columns:
                    sub_yes["composite_p_up"] = pd.to_numeric(sub_yes["composite_p_up"], errors="coerce")
                    for cpu_cut, cpu_label, cpu_op in [
                        (0.50, "cpu≤0.50", "le"), (0.50, "cpu≥0.50", "ge"),
                        (0.48, "cpu≤0.48", "le"), (0.52, "cpu≥0.52", "ge"),
                    ]:
                        if cpu_op == "le":
                            mask = sub_yes["composite_p_up"] <= cpu_cut
                        else:
                            mask = sub_yes["composite_p_up"] >= cpu_cut
                        r2 = show(sub_yes[mask.fillna(False)], f"    {reg} YES {cpu_label}")
                        if r2:
                            print_row(r2, "RESCUE" if r2["wr"] >= 0.58 else "")

# ── Fetch price data ──────────────────────────────────────────────────────────
print("Fetching ETH-USD and SOL-USD price data (1h)...")
price_data = {}
for asset in ["ETH", "SOL"]:
    ticker = ASSET_YFTICKER[asset]
    df = yf.download(ticker, start="2024-11-01", end="2026-05-23",
                     interval="1h", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    price_data[asset] = df
    print(f"  {asset}: {len(df)} 1h bars ({df.index.min().date()} → {df.index.max().date()})")

# ── Load trades ───────────────────────────────────────────────────────────────
print("\nLoading trades...")
trades_1h, trades_15m = {}, {}
for asset in ["ETH", "SOL"]:
    try:
        trades_1h[asset]  = prep_trades_1h(TRADES_1H_CSV[asset])
        print(f"  {asset} 1h:  {len(trades_1h[asset])} resolved trades")
    except Exception as e:
        print(f"  {asset} 1h:  load error: {e}")
        trades_1h[asset] = None
    try:
        trades_15m[asset] = prep_trades_15m(TRADES_15M_CSV[asset])
        print(f"  {asset} 15m: {len(trades_15m[asset])} resolved trades")
    except Exception as e:
        print(f"  {asset} 15m: load error: {e}")
        trades_15m[asset] = None

# ── Main sweep ────────────────────────────────────────────────────────────────
for asset in ["ETH", "SOL"]:
    df_1h = price_data[asset]
    close_1h = df_1h["Close"]

    print(f"\n\n{SEP}")
    print(f"  {asset} — MULTI-TIMEFRAME MARKOV SWEEP")
    print(SEP)

    # Print threshold distribution summary
    print(f"\n  Regime label balance (window={WINDOW} bars) at working thresholds:")
    print(f"  {'Timeframe':>8s}  {'Threshold':>10s}  {'Bear':>6s}  {'Side':>6s}  {'Bull':>6s}  {'B/S/Bu%'}")
    print("  " + "─" * 58)

    for tf_label, (eth_thr, sol_thr) in THRESHOLDS.items():
        thr = eth_thr if asset == "ETH" else sol_thr
        if tf_label == "1d":
            df_d = close_1h.resample("1D").last().dropna()
            close_tf = df_d
        elif tf_label == "6h":
            close_tf = close_1h.resample("6h").last().dropna()
        elif tf_label == "4h":
            close_tf = close_1h.resample("4h").last().dropna()
        elif tf_label == "1h":
            close_tf = close_1h
        else:  # 15m — need higher-res data but we only have 1h; skip distribution
            print(f"  {'15m':>8s}  (skipped in distribution — uses yfinance 15m below)")
            continue

        rr  = close_tf.pct_change(WINDOW)
        reg = pd.Series("Sideways", index=close_tf.index)
        reg[rr >  thr] = "Bull"
        reg[rr < -thr] = "Bear"
        reg = reg[rr.notna()]
        bc, sc, uc, n = (reg=="Bear").sum(), (reg=="Sideways").sum(), (reg=="Bull").sum(), len(reg)
        print(f"  {tf_label:>8s}  {thr*100:>8.1f}%   {bc:6d}  {sc:6d}  {uc:6d}  "
              f"({bc/n*100:.0f}% / {sc/n*100:.0f}% / {uc/n*100:.0f}%)")

    # ── Per-timeframe deep analysis ───────────────────────────────────────────
    for tf_label, (eth_thr, sol_thr) in THRESHOLDS.items():
        thr = eth_thr if asset == "ETH" else sol_thr

        if tf_label == "1d":
            close_tf = close_1h.resample("1D").last().dropna()
        elif tf_label == "6h":
            close_tf = close_1h.resample("6h").last().dropna()
        elif tf_label == "4h":
            close_tf = close_1h.resample("4h").last().dropna()
        elif tf_label == "1h":
            close_tf = close_1h
        else:  # 15m: fetch separately
            print(f"\n  Fetching {asset} 15m data for regime...")
            try:
                df_15m_p = yf.download(ASSET_YFTICKER[asset],
                                        start="2026-04-01", end="2026-05-23",
                                        interval="15m", progress=False, auto_adjust=True)
                if isinstance(df_15m_p.columns, pd.MultiIndex):
                    df_15m_p.columns = df_15m_p.columns.get_level_values(0)
                df_15m_p.index = pd.to_datetime(df_15m_p.index, utc=True)
                close_tf = df_15m_p["Close"].dropna()
            except Exception as e:
                print(f"  15m fetch failed: {e}")
                continue

        reg = build_regime(close_tf, WINDOW, thr)
        analyze(asset, tf_label, close_tf, reg, trades_1h[asset], trades_15m[asset])

# ── Current state snapshot ────────────────────────────────────────────────────
print(f"\n\n{SEP}")
print("  CURRENT REGIME SNAPSHOT")
print(SEP)
for asset in ["ETH", "SOL"]:
    df_1h = price_data[asset]
    close_1h = df_1h["Close"]
    print(f"\n  {asset}:")
    for tf_label, (eth_thr, sol_thr) in THRESHOLDS.items():
        thr = eth_thr if asset == "ETH" else sol_thr
        if tf_label == "1d":
            close_tf = close_1h.resample("1D").last().dropna()
        elif tf_label == "6h":
            close_tf = close_1h.resample("6h").last().dropna()
        elif tf_label == "4h":
            close_tf = close_1h.resample("4h").last().dropna()
        elif tf_label == "1h":
            close_tf = close_1h
        else:
            continue  # skip 15m for snapshot (already fetched above)
        reg = build_regime(close_tf, WINDOW, thr)
        cur = reg.iloc[-1]
        rr  = float(close_tf.pct_change(WINDOW).iloc[-1])
        print(f"    {tf_label:>4s}: {cur:<10s}  (20-bar ret={rr*100:+.2f}%, thresh=±{thr*100:.1f}%)")
