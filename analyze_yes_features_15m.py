#!/usr/bin/env python3
"""
analyze_yes_features_15m.py

For YES trades on the 15m BTC model:
  - Compute point-biserial correlation of each feature vs resolved_yes
  - Show win rate by feature tercile (low / mid / high)
  - Flag which features give false bullish signals
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("results")

df = pd.read_csv(RESULTS_DIR / "paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)

# YES trades only
yes = df[df["side"].str.upper() == "YES"].copy()
all_trades = df.copy()

print(f"All trades: {len(df)}  YES: {len(yes)}  YES res rate: {yes['resolved_yes'].mean():.1%}")
print()

FEATURES = [
    # 15m candle signals
    "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "upper_wick_15m", "lower_wick_15m", "atr_ratio_15m", "range_ratio_15m", "consec_dir_15m",
    # 5m signals
    "bp_5m", "chg_5m", "stoch_k_5m", "vol_ratio_5m",
    # 1h signals
    "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
    "consec_dir_1h", "dir_1h", "donchian_breakout_1h", "engulfing_1h", "stoch_cross_1h",
    "rsi_1h", "macd_hist_1h",
    # Composite / market-wide
    "composite_p_up", "ema_bias", "vwap_dist",
    "liq_score", "liq_bias", "oi_chg_pct",
    "fear_greed", "cg_composite",
    # Volatility / pricing
    "realized_vol_annual", "vol_ratio", "vol_ratio_1h",
    "offset_pct", "tau_minutes",
]

print("=" * 80)
print(f"  Feature vs resolved_yes correlation — YES trades (n={len(yes)})")
print(f"  Breakeven WR ≈ {yes['p_market'].mean():.1%}  (avg p_market)")
print("=" * 80)
print(f"  {'Feature':<28} {'r':>6}  {'p':>8}  {'Low WR':>7}  {'Mid WR':>7}  {'Hi WR':>7}  {'Direction'}")
print("  " + "-" * 76)

rows = []
for feat in FEATURES:
    if feat not in yes.columns:
        continue
    col = pd.to_numeric(yes[feat], errors="coerce")
    valid = yes[col.notna()].copy()
    valid["_f"] = col[col.notna()]
    if len(valid) < 30 or valid["_f"].std() < 1e-9:
        continue

    r, p = stats.pointbiserialr(valid["_f"], valid["resolved_yes"])

    # Win rate by tercile
    try:
        valid["_q"] = pd.qcut(valid["_f"], 3, labels=["lo", "mid", "hi"], duplicates="drop")
        wr = valid.groupby("_q")["resolved_yes"].mean()
        lo_wr  = wr.get("lo",  np.nan)
        mid_wr = wr.get("mid", np.nan)
        hi_wr  = wr.get("hi",  np.nan)
    except Exception:
        lo_wr = mid_wr = hi_wr = np.nan

    direction = ""
    if abs(r) >= 0.05:
        direction = "YES↑" if r > 0 else "NO↑"
    if abs(r) >= 0.08:
        direction = ("✓YES" if r > 0 else "✗FALSE") + f"  |r|={abs(r):.3f}"

    sig = "*" if p < 0.05 else (" " if p < 0.10 else " ")
    rows.append((abs(r), feat, r, p, lo_wr, mid_wr, hi_wr, direction, sig))

rows.sort(reverse=True)
for _, feat, r, p, lo_wr, mid_wr, hi_wr, direction, sig in rows:
    lo  = f"{lo_wr:.1%}"  if not np.isnan(lo_wr)  else "  n/a"
    mid = f"{mid_wr:.1%}" if not np.isnan(mid_wr) else "  n/a"
    hi  = f"{hi_wr:.1%}"  if not np.isnan(hi_wr)  else "  n/a"
    p_str = f"{p:.4f}{sig}"
    print(f"  {feat:<28} {r:>+6.3f}  {p_str:>8}  {lo:>7}  {mid:>7}  {hi:>7}  {direction}")

# ── Conditional WR for key features ─────────────────────────────────────────

print()
print("=" * 80)
print("  Conditional YES win rates — key features")
print("=" * 80)

KEY_CUTS = {
    "ema_bias":         [(None, -0.005, "bearish"), (-0.005, 0.005, "neutral"), (0.005, None, "bullish")],
    "ema_bias_1h":      [(None, -0.005, "bearish"), (-0.005, 0.005, "neutral"), (0.005, None, "bullish")],
    "stoch_k_15m":      [(None, 30, "oversold"), (30, 70, "mid"), (70, None, "overbought")],
    "stoch_k_1h":       [(None, 30, "oversold"), (30, 70, "mid"), (70, None, "overbought")],
    "composite_p_up":   [(None, 0.45, "bearish"), (0.45, 0.55, "neutral"), (0.55, None, "bullish")],
    "chg_1h":           [(None, -0.002, "neg"), (-0.002, 0.002, "flat"), (0.002, None, "pos")],
    "bp_5m":            [(None, 0.3, "bearish"), (0.3, 0.7, "neutral"), (0.7, None, "bullish")],
    "vwap_dist":        [(None, -0.002, "below"), (-0.002, 0.002, "near"), (0.002, None, "above")],
    "dir_15m":          [(-1.5, -0.5, "down"), (-0.5, 0.5, "flat"), (0.5, 1.5, "up")],
    "rsi_1h":           [(None, 40, "OS"), (40, 60, "mid"), (60, None, "OB")],
    "macd_hist_1h":     [(None, -0.5, "neg"), (-0.5, 0.5, "flat"), (0.5, None, "pos")],
}

be_wr = yes["p_market"].mean()
for feat, cuts in KEY_CUTS.items():
    if feat not in yes.columns:
        continue
    col = pd.to_numeric(yes[feat], errors="coerce")
    print(f"\n  {feat}  (overall YES WR={yes['resolved_yes'].mean():.1%}  BE={be_wr:.1%})")
    for lo, hi, label in cuts:
        if lo is None:
            mask = col < hi
        elif hi is None:
            mask = col >= lo
        else:
            mask = (col >= lo) & (col < hi)
        sub = yes[mask & col.notna()]
        if len(sub) < 5:
            continue
        wr = sub["resolved_yes"].mean()
        flag = "  ✗" if wr < be_wr - 0.03 else ("  ✓" if wr > be_wr + 0.03 else "")
        print(f"    {label:12s}  n={len(sub):4d}  WR={wr:.1%}{flag}")

# ── Feature agreement analysis ───────────────────────────────────────────────

print()
print("=" * 80)
print("  Feature agreement score vs YES WR")
print("  (bullish = ema_bias>0, stoch_k_15m<70, bp_5m>0.5, chg_1h>0, composite_p_up>0.5)")
print("=" * 80)

bull_signals = {
    "ema_bias":       lambda s: s > 0,
    "stoch_k_15m":    lambda s: s < 70,
    "bp_5m":          lambda s: s > 0.5,
    "chg_1h":         lambda s: s > 0,
    "composite_p_up": lambda s: s > 0.5,
    "stoch_k_1h":     lambda s: s < 70,
    "vwap_dist":      lambda s: s > 0,
}

yes2 = yes.copy()
yes2["bull_agree"] = 0
for feat, fn in bull_signals.items():
    if feat in yes2.columns:
        col = pd.to_numeric(yes2[feat], errors="coerce").fillna(0)
        yes2["bull_agree"] += fn(col).astype(int)

print(f"  {'Score':>6}  {'n':>5}  {'WR':>7}  {'BE':>7}")
print("  " + "-" * 35)
be_wr = yes2["p_market"].mean()
for score in range(8):
    sub = yes2[yes2["bull_agree"] == score]
    if len(sub) < 5:
        continue
    wr = sub["resolved_yes"].mean()
    flag = "  ✗" if wr < be_wr - 0.03 else ("  ✓" if wr > be_wr + 0.03 else "")
    print(f"  {score:>6}  {len(sub):>5}  {wr:>7.1%}  {be_wr:>7.1%}{flag}")
