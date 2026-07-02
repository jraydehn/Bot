"""
Short-horizon momentum indicator backtest against 985 resolved BTC paper trades.
Computes RSI-14 (1m), price momentum (5m/15m/30m), rolling 1h VWAP deviation,
and a composite short_momentum_score. Compares against existing long-horizon
indicators (EMA alignment, stoch_bias) and runs gate simulations.
"""

import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")

BASE = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc"

# ─────────────────────────────────────────────
# 1.  LOAD & COMBINE RESOLVED TRADE ARCHIVES
# ─────────────────────────────────────────────
ARCHIVE_FILES = [
    f"{BASE}/results/paper_trades_archive_20260407_122844.csv",
    f"{BASE}/results/paper_trades_archive_20260407_152310.csv",
    f"{BASE}/results/paper_trades_archive_20260405_1633pdt.csv",
    f"{BASE}/results/paper_trades_archive_20260330_103000.csv",
    f"{BASE}/results/paper_trades_archive_20260330_2230pdt.csv",
    f"{BASE}/results/paper_trades_archive_20260325_090712.csv",
    f"{BASE}/results/paper_trades_archive_20260323_003206.csv",
    f"{BASE}/results/paper_trades_all.csv",
]

dfs = []
for f in ARCHIVE_FILES:
    df = pd.read_csv(f)
    df["source_file"] = f.split("/")[-1]
    dfs.append(df)

trades = pd.concat(dfs, ignore_index=True, sort=False)
print(f"Total rows across all files: {len(trades)}")

# Keep only rows with a resolved outcome
trades = trades[trades["resolved_yes"].notna()].copy()
print(f"Rows with resolved_yes: {len(trades)}")

# Parse timestamps
trades["logged_at"] = pd.to_datetime(trades["logged_at"], utc=True, errors="coerce")
trades["decision_time"] = pd.to_datetime(trades["decision_time"], utc=True, errors="coerce")

# Use decision_time if available, else logged_at
trades["eval_time"] = trades["decision_time"].where(
    trades["decision_time"].notna(), trades["logged_at"]
)

# Deduplicate on contract_ticker + logged_at
before_dedup = len(trades)
trades = trades.drop_duplicates(subset=["contract_ticker", "logged_at"])
print(f"After dedup (contract_ticker + logged_at): {len(trades)} (dropped {before_dedup - len(trades)})")

trades["resolved_yes"] = trades["resolved_yes"].astype(float)
trades = trades.reset_index(drop=True)
print(f"\nTotal unique resolved trades: {len(trades)}")
print(f"eval_time range: {trades['eval_time'].min()} to {trades['eval_time'].max()}")
print(f"Overall YES rate: {trades['resolved_yes'].mean():.3f}")

# ─────────────────────────────────────────────
# 2.  LOAD 1-MINUTE OHLCV DATA
# ─────────────────────────────────────────────
files_1m = sorted(glob.glob(f"{BASE}/data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet"))
parquet_path = files_1m[-1]
print(f"\nLoading 1m parquet: {parquet_path}")
ohlcv = pd.read_parquet(parquet_path)
ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
ohlcv = ohlcv.sort_index()
print(f"1m data: {ohlcv.index.min()} to {ohlcv.index.max()} ({len(ohlcv):,} bars)")

# ─────────────────────────────────────────────
# 3.  PRE-COMPUTE INDICATORS ON FULL 1m SERIES
# ─────────────────────────────────────────────
print("\nPre-computing indicators on 1m OHLCV series...")

close = ohlcv["close"].values
volume = ohlcv["volume"].values
n = len(close)

# RSI-14
def compute_rsi(closes, period=14):
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Use Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    # seed first value
    avg_gain[period] = gain[1:period+1].mean()
    avg_loss[period] = loss[1:period+1].mean()
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_arr = compute_rsi(close, period=14)

# Momentum
def mom_pct(closes, lag):
    result = np.full(n, np.nan)
    for i in range(lag, n):
        if closes[i-lag] != 0:
            result[i] = (closes[i] - closes[i-lag]) / closes[i-lag]
    return result

mom_5m_arr  = mom_pct(close, 5)
mom_15m_arr = mom_pct(close, 15)
mom_30m_arr = mom_pct(close, 30)

# Rolling 1h VWAP (60 1m bars)
vwap_window = 60
cv = close * volume
cv_roll = np.full(n, np.nan)
v_roll  = np.full(n, np.nan)
for i in range(vwap_window-1, n):
    s = i - vwap_window + 1
    v_sum = volume[s:i+1].sum()
    if v_sum > 0:
        cv_roll[i] = cv[s:i+1].sum() / v_sum
    else:
        cv_roll[i] = close[i]

vwap_arr = cv_roll
vwap_dev_arr = np.where(
    ~np.isnan(vwap_arr) & (vwap_arr != 0),
    (close - vwap_arr) / vwap_arr,
    np.nan
)

# Build indicator DataFrame aligned with OHLCV index
ind = pd.DataFrame({
    "rsi":      rsi_arr,
    "mom_5m":   mom_5m_arr,
    "mom_15m":  mom_15m_arr,
    "mom_30m":  mom_30m_arr,
    "vwap":     vwap_arr,
    "vwap_dev": vwap_dev_arr,
}, index=ohlcv.index)

print("Indicator pre-computation done.")

# ─────────────────────────────────────────────
# 4.  MATCH EACH TRADE TO NEAREST PRIOR 1m BAR
# ─────────────────────────────────────────────
print("\nMatching trades to 1m bars...")

ohlcv_idx = ohlcv.index  # DatetimeIndex UTC

matched = 0
unmatched = 0
records = []

for _, row in trades.iterrows():
    et = row["eval_time"]
    if pd.isna(et):
        unmatched += 1
        records.append({})
        continue

    # Find last bar at or before eval_time
    pos = ohlcv_idx.searchsorted(et, side="right") - 1
    if pos < 60:   # need at least 60 bars for VWAP
        unmatched += 1
        records.append({})
        continue

    bar_time = ohlcv_idx[pos]

    # RSI signal
    rsi_val = ind["rsi"].iloc[pos]
    if np.isnan(rsi_val):
        rsi_signal = np.nan
    elif rsi_val > 60:
        rsi_signal = 1
    elif rsi_val < 40:
        rsi_signal = -1
    else:
        rsi_signal = 0

    # Mom bins
    def mom_bin(v):
        if np.isnan(v):
            return np.nan
        if v > 0.001:
            return 1
        elif v < -0.001:
            return -1
        else:
            return 0

    m5  = ind["mom_5m"].iloc[pos]
    m15 = ind["mom_15m"].iloc[pos]
    m30 = ind["mom_30m"].iloc[pos]

    mom_5m_bin  = mom_bin(m5)
    mom_15m_bin = mom_bin(m15)
    mom_30m_bin = mom_bin(m30)

    # VWAP dev bin
    vd = ind["vwap_dev"].iloc[pos]
    if np.isnan(vd):
        vwap_dev_bin = np.nan
    elif vd > 0.0005:
        vwap_dev_bin = 1
    elif vd < -0.0005:
        vwap_dev_bin = -1
    else:
        vwap_dev_bin = 0

    # Composite short_momentum_score
    components = [rsi_signal, mom_15m_bin, vwap_dev_bin]
    if any(np.isnan(c) for c in components):
        short_mom_score = np.nan
    else:
        short_mom_score = sum(components)

    matched += 1
    records.append({
        "bar_time":        bar_time,
        "rsi_val":         rsi_val,
        "rsi_signal":      rsi_signal,
        "mom_5m_raw":      m5,
        "mom_15m_raw":     m15,
        "mom_30m_raw":     m30,
        "mom_5m_bin":      mom_5m_bin,
        "mom_15m_bin":     mom_15m_bin,
        "mom_30m_bin":     mom_30m_bin,
        "vwap_dev_raw":    vd,
        "vwap_dev_bin":    vwap_dev_bin,
        "short_mom_score": short_mom_score,
    })

ind_df = pd.DataFrame(records, index=trades.index)
full = pd.concat([trades, ind_df], axis=1)

print(f"Matched to 1m bar: {matched} / {len(trades)}")
print(f"Could not match:   {unmatched} / {len(trades)}")

# Work with matched subset
m = full[full["bar_time"].notna()].copy()
print(f"\nWorking dataset: {len(m)} trades with 1m indicators")

# ─────────────────────────────────────────────
# 5.  HELPER: CALIBRATION TABLE
# ─────────────────────────────────────────────
def calibration_table(df, signal_col, label):
    rows = []
    for val in sorted(df[signal_col].dropna().unique()):
        sub = df[df[signal_col] == val]
        n = len(sub)
        win_rate = sub["resolved_yes"].mean()
        p_model_mean = sub["p_yes_model"].mean() if "p_yes_model" in sub.columns else np.nan
        ratio = win_rate / p_model_mean if p_model_mean and p_model_mean > 0 else np.nan
        residual = win_rate - p_model_mean if not np.isnan(p_model_mean) else np.nan
        rows.append({
            "signal": val,
            "n": n,
            "win_rate": round(win_rate, 4),
            "p_model": round(p_model_mean, 4),
            "ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
            "residual": round(residual, 4) if not np.isnan(residual) else np.nan,
        })
    tbl = pd.DataFrame(rows)
    if len(tbl) > 0 and tbl["ratio"].notna().sum() > 1:
        spread = round(tbl["ratio"].max() - tbl["ratio"].min(), 4)
    else:
        spread = np.nan
    return tbl, spread

# ─────────────────────────────────────────────
# 6.  ANALYSIS A: SINGLE-INDICATOR CALIBRATION
# ─────────────────────────────────────────────
sep = "=" * 70

print(f"\n{sep}")
print("ANALYSIS A — SINGLE-INDICATOR CALIBRATION TABLES")
print("NOTE: win_rate = p(resolved_yes=1). Calibration tests whether the")
print("indicator predicts the BTC binary outcome (price above/below strike).")
print("A higher ratio when RSI=-1 means bearish RSI correlates with YES settling,")
print("which can happen because the model preferentially bets NO in bearish conditions.")
print(sep)

indicators_to_test = [
    ("rsi_signal",      "RSI-14 (1m) signal"),
    ("mom_5m_bin",      "Momentum 5m bin"),
    ("mom_15m_bin",     "Momentum 15m bin"),
    ("mom_30m_bin",     "Momentum 30m bin"),
    ("vwap_dev_bin",    "VWAP deviation bin (rolling 1h)"),
    ("short_mom_score", "Composite short_momentum_score"),
]

spreads_new = {}

for col, label in indicators_to_test:
    sub = m[m[col].notna()]
    tbl, spread = calibration_table(sub, col, label)
    print(f"\n  [{label}]  n={len(sub)}  ratio_spread={spread}")
    print(tbl.to_string(index=False))
    spreads_new[col] = spread

# ─────────────────────────────────────────────
# 7.  ANALYSIS B: COMPARE TO CURRENT INDICATORS
# ─────────────────────────────────────────────
print(f"\n{sep}")
print("ANALYSIS B — RATIO SPREAD COMPARISON: NEW vs CURRENT INDICATORS")
print(sep)

# Current long-horizon indicators already in the CSV
current_indicators = []

# EMA alignment (where present)
if "ema_alignment" in m.columns:
    sub = m[m["ema_alignment"].notna()]
    _, spr = calibration_table(sub, "ema_alignment", "EMA alignment")
    current_indicators.append(("ema_alignment", "EMA alignment (current)", spr))

# Stoch bias
if "stoch_bias" in m.columns:
    sub = m[m["stoch_bias"].notna()]
    _, spr = calibration_table(sub, "stoch_bias", "stoch_bias")
    current_indicators.append(("stoch_bias", "Stoch bias (current)", spr))

# Funding bias
if "funding_bias" in m.columns:
    sub = m[m["funding_bias"].notna()]
    _, spr = calibration_table(sub, "funding_bias", "funding_bias")
    current_indicators.append(("funding_bias", "Funding bias (current)", spr))

print(f"\n  {'Indicator':<40} {'n_valid':>8}  {'ratio_spread':>12}")
print("  " + "-" * 64)
for col, label, spr in current_indicators:
    n_valid = m[col].notna().sum()
    print(f"  {label:<40} {n_valid:>8}  {spr:>12.4f}")

print("  " + "-" * 64)
for col, label in indicators_to_test:
    n_valid = m[col].notna().sum()
    spr = spreads_new[col]
    print(f"  {label:<40} {n_valid:>8}  {spr:>12.4f}")

# ─────────────────────────────────────────────
# 8.  GATE SIMULATION HELPER
# ─────────────────────────────────────────────
def trade_pnl(row, bet=5.0):
    """
    Compute flat-$5-bet PnL for a single trade row.
    Kalshi binary: bet $bet at price p_market.
    YES win: +bet*(1-p)/p; YES lose: -bet
    NO  win: +bet*p/(1-p); NO  lose: -bet
    Side column is lowercase ('yes'/'no').
    """
    side = str(row.get("side", "")).lower()
    resolved = row["resolved_yes"]
    p_mkt = row.get("p_market", 0.5)
    if pd.isna(p_mkt) or p_mkt <= 0 or p_mkt >= 1:
        p_mkt = 0.5
    if side == "yes":
        if resolved == 1:
            return bet * (1 - p_mkt) / p_mkt, True
        else:
            return -bet, False
    elif side == "no":
        if resolved == 0:
            return bet * p_mkt / (1 - p_mkt), True
        else:
            return -bet, False
    else:
        return 0.0, False


def simulate_gate(df, gate_fn, label, bet=5.0):
    """gate_fn(row) -> True to take the trade, False to skip"""
    trades_taken = df[df.apply(gate_fn, axis=1)]
    n = len(trades_taken)
    if n == 0:
        return {"label": label, "n": 0, "win_pct": np.nan, "total_pnl": 0.0, "avg_pnl": np.nan}

    wins = 0
    total_pnl = 0.0
    for _, row in trades_taken.iterrows():
        pnl, win = trade_pnl(row, bet)
        wins += int(win)
        total_pnl += pnl

    win_pct = wins / n if n > 0 else np.nan
    avg_pnl = total_pnl / n if n > 0 else np.nan
    return {"label": label, "n": n, "win_pct": round(win_pct, 4),
            "total_pnl": round(total_pnl, 2), "avg_pnl": round(avg_pnl, 4)}

# ─────────────────────────────────────────────
# 9.  ANALYSIS C: GATE SIMULATION CONFIGS
# ─────────────────────────────────────────────
print(f"\n{sep}")
print("ANALYSIS C — GATE SIMULATION (flat $5 bet)")
print(sep)

# Prepare columns with defaults
m["stoch_bias_val"]   = pd.to_numeric(m.get("stoch_bias", np.nan), errors="coerce")
m["funding_bias_val"] = pd.to_numeric(m.get("funding_bias", np.nan), errors="coerce")

configs = []

# 1. Baseline — all trades
configs.append(simulate_gate(
    m,
    lambda r: True,
    "1. Baseline (all trades)"
))

# 2. Stoch + funding aligned
configs.append(simulate_gate(
    m,
    lambda r: (
        not pd.isna(r["stoch_bias_val"]) and not pd.isna(r["funding_bias_val"]) and
        r["stoch_bias_val"] != 0 and r["funding_bias_val"] != 0 and
        r["stoch_bias_val"] == r["funding_bias_val"]
    ),
    "2. Stoch + funding aligned"
))

# 3. RSI + mom_15m aligned (both same direction, not zero)
configs.append(simulate_gate(
    m,
    lambda r: (
        not pd.isna(r["rsi_signal"]) and not pd.isna(r["mom_15m_bin"]) and
        r["rsi_signal"] != 0 and r["mom_15m_bin"] != 0 and
        r["rsi_signal"] == r["mom_15m_bin"]
    ),
    "3. RSI + mom_15m aligned"
))

# 4. short_momentum_score + funding aligned
configs.append(simulate_gate(
    m,
    lambda r: (
        not pd.isna(r["short_mom_score"]) and not pd.isna(r["funding_bias_val"]) and
        r["short_mom_score"] != 0 and r["funding_bias_val"] != 0 and
        np.sign(r["short_mom_score"]) == r["funding_bias_val"]
    ),
    "4. short_mom_score + funding aligned"
))

# 5. RSI + mom_15m + funding all aligned
configs.append(simulate_gate(
    m,
    lambda r: (
        not pd.isna(r["rsi_signal"]) and not pd.isna(r["mom_15m_bin"]) and
        not pd.isna(r["funding_bias_val"]) and
        r["rsi_signal"] != 0 and r["mom_15m_bin"] != 0 and r["funding_bias_val"] != 0 and
        r["rsi_signal"] == r["mom_15m_bin"] and
        r["mom_15m_bin"] == r["funding_bias_val"]
    ),
    "5. RSI + mom_15m + funding all aligned"
))

# 6. Best combination scan — try all combos of available signals
best_pnl = -np.inf
best_label = ""
best_result = None

signal_cols = ["rsi_signal", "mom_15m_bin", "vwap_dev_bin", "stoch_bias_val", "funding_bias_val"]
# Try pairs
from itertools import combinations

for a, b in combinations(signal_cols, 2):
    for dir_combo in ["same", "any_nonzero"]:
        if dir_combo == "same":
            def gate(r, a=a, b=b):
                va, vb = r.get(a, np.nan), r.get(b, np.nan)
                if pd.isna(va) or pd.isna(vb):
                    return False
                return va != 0 and vb != 0 and va == vb
            lbl = f"[pair] {a}=={b} (same nonzero)"
        else:
            def gate(r, a=a, b=b):
                va, vb = r.get(a, np.nan), r.get(b, np.nan)
                if pd.isna(va) or pd.isna(vb):
                    return False
                return va != 0 and vb != 0
            lbl = f"[pair] {a} & {b} (both nonzero)"

        res = simulate_gate(m, gate, lbl)
        if res["n"] >= 20 and res["total_pnl"] > best_pnl:
            best_pnl = res["total_pnl"]
            best_label = lbl
            best_result = res

# Try triples
for a, b, c in combinations(signal_cols, 3):
    def gate(r, a=a, b=b, c=c):
        va, vb, vc = r.get(a, np.nan), r.get(b, np.nan), r.get(c, np.nan)
        if pd.isna(va) or pd.isna(vb) or pd.isna(vc):
            return False
        return va != 0 and vb != 0 and vc != 0 and va == vb and vb == vc
    lbl = f"[triple] {a}=={b}=={c} (all same nonzero)"
    res = simulate_gate(m, gate, lbl)
    if res["n"] >= 15 and res["total_pnl"] > best_pnl:
        best_pnl = res["total_pnl"]
        best_label = lbl
        best_result = res

if best_result:
    best_result["label"] = f"6. Best combo: {best_label}"
    configs.append(best_result)

# Print results
print(f"\n  {'Config':<45} {'n':>6}  {'win%':>7}  {'total_PnL':>10}  {'avg_PnL':>9}")
print("  " + "-" * 82)
for r in configs:
    win_pct_str = f"{r['win_pct']*100:.1f}%" if not pd.isna(r.get("win_pct", np.nan)) else "  n/a"
    avg_str     = f"${r['avg_pnl']:.3f}" if not pd.isna(r.get("avg_pnl", np.nan)) else "  n/a"
    print(f"  {r['label']:<45} {r['n']:>6}  {win_pct_str:>7}  ${r['total_pnl']:>9.2f}  {avg_str:>9}")

# ─────────────────────────────────────────────
# 10. ANALYSIS D: 3-WAY COMBO (stoch + RSI + funding)
# ─────────────────────────────────────────────
print(f"\n{sep}")
print("ANALYSIS D — 3-WAY COMBO: (stoch_bias, rsi_signal, funding_bias)")
print("Showing combos with n >= 5")
print(sep)

d = m[
    m["stoch_bias_val"].notna() &
    m["rsi_signal"].notna() &
    m["funding_bias_val"].notna()
].copy()

d["stoch_bin"]   = d["stoch_bias_val"].astype(int)
d["rsi_bin"]     = d["rsi_signal"].astype(int)
d["funding_bin"] = d["funding_bias_val"].astype(int)

# Compute trade_win: whether the TRADE (not just YES) won
def _trade_won(row):
    side = str(row.get("side", "")).lower()
    res = row["resolved_yes"]
    if side == "yes":
        return float(res == 1)
    elif side == "no":
        return float(res == 0)
    return np.nan

d["trade_win"] = d.apply(_trade_won, axis=1)

group = d.groupby(["stoch_bin", "rsi_bin", "funding_bin"]).agg(
    n=("resolved_yes", "count"),
    yes_rate=("resolved_yes", "mean"),    # p(resolved_yes=1)
    trade_win_rate=("trade_win", "mean"),  # p(trade won)
    p_model=("p_yes_model", "mean"),
).reset_index()

# Compute PnL for each combo
def group_pnl(sub, bet=5.0):
    total = 0.0
    for _, row in sub.iterrows():
        pnl, _ = trade_pnl(row, bet)
        total += pnl
    return total

pnl_list = []
for _, grow in group.iterrows():
    sub = d[
        (d["stoch_bin"] == grow["stoch_bin"]) &
        (d["rsi_bin"] == grow["rsi_bin"]) &
        (d["funding_bin"] == grow["funding_bin"])
    ]
    pnl_list.append(group_pnl(sub))

group["pnl"] = pnl_list
group["yes_ratio"] = group["yes_rate"] / group["p_model"]
group["yes_residual"] = group["yes_rate"] - group["p_model"]

# Filter n >= 5
group5 = group[group["n"] >= 5].sort_values("pnl", ascending=False)

print(f"\n  NOTE: 'yes%'=p(resolved_yes=1), 'trade_win%'=p(bet won), sorted by PnL")
print(f"\n  {'stoch':>6}  {'rsi':>5}  {'fund':>5}  {'n':>5}  {'yes%':>7}  {'trade_win%':>11}  {'p_model':>8}  {'pnl':>9}")
print("  " + "-" * 72)
for _, r in group5.iterrows():
    print(f"  {int(r['stoch_bin']):>6}  {int(r['rsi_bin']):>5}  {int(r['funding_bin']):>5}  "
          f"{int(r['n']):>5}  {r['yes_rate']*100:>6.1f}%  "
          f"{r['trade_win_rate']*100:>10.1f}%  {r['p_model']:>8.4f}  "
          f"${r['pnl']:>8.2f}")

# Also show ALL combos summary sorted by trade win rate (for interpretation)
print(f"\n  Top combos by trade_win_rate (n>=5):")
print(f"  {'stoch':>6}  {'rsi':>5}  {'fund':>5}  {'n':>5}  {'trade_win%':>11}  {'pnl':>9}")
print("  " + "-" * 50)
for _, r in group5.sort_values("trade_win_rate", ascending=False).head(10).iterrows():
    print(f"  {int(r['stoch_bin']):>6}  {int(r['rsi_bin']):>5}  {int(r['funding_bin']):>5}  "
          f"{int(r['n']):>5}  {r['trade_win_rate']*100:>10.1f}%  ${r['pnl']:>8.2f}")

# ─────────────────────────────────────────────
# 11. SAVE OUTPUT CSV
# ─────────────────────────────────────────────
# Add trade_win to main frame
def _tw(row):
    side = str(row.get("side", "")).lower()
    res = row["resolved_yes"]
    if side == "yes":
        return float(res == 1)
    elif side == "no":
        return float(res == 0)
    return np.nan
m["trade_win"] = m.apply(_tw, axis=1)

save_cols = [
    "logged_at", "decision_time", "eval_time", "contract_ticker",
    "side", "strike", "spot", "p_market", "p_yes_model",
    "resolved_yes", "trade_win", "would_pnl",
    "stoch_bias", "funding_bias", "ema_alignment",
    "bar_time",
    "rsi_val", "rsi_signal",
    "mom_5m_raw", "mom_15m_raw", "mom_30m_raw",
    "mom_5m_bin", "mom_15m_bin", "mom_30m_bin",
    "vwap_dev_raw", "vwap_dev_bin",
    "short_mom_score",
]

out_cols = [c for c in save_cols if c in m.columns]
out_path = f"{BASE}/results/indicator_backtest.csv"
m[out_cols].to_csv(out_path, index=False)
print(f"\n{sep}")
print(f"Results saved to: {out_path}")
print(f"Rows: {len(m)}")
print(sep)
