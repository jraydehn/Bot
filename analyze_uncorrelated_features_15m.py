#!/usr/bin/env python3
"""
Identify which indicators have no correlation with YES resolution on YES trades.
Also check ALL trades for comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)

yes  = df[df["side"].str.upper() == "YES"].copy()
no   = df[df["side"].str.upper() == "NO"].copy()

FEATURES = [
    "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "upper_wick_15m", "lower_wick_15m", "consec_dir_15m",
    "bp_5m", "chg_5m", "stoch_k_5m", "vol_ratio_5m",
    "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
    "consec_dir_1h", "dir_1h", "donchian_breakout_1h", "engulfing_1h",
    "stoch_cross_1h", "rsi_1h", "macd_hist_1h",
    "composite_p_up", "ema_bias", "vwap_dist",
    "liq_score", "liq_bias", "oi_chg_pct",
    "fear_greed", "cg_composite",
    "realized_vol_annual", "vol_ratio", "vol_ratio_1h",
    "tau_minutes", "offset_pct",
]

def correlate(subset, label):
    rows = []
    for feat in FEATURES:
        if feat not in subset.columns:
            continue
        col = pd.to_numeric(subset[feat], errors="coerce")
        valid_mask = col.notna()
        if valid_mask.sum() < 30 or col[valid_mask].std() < 1e-9:
            continue
        r, p = stats.pointbiserialr(col[valid_mask], subset.loc[valid_mask, "resolved_yes"])
        rows.append((abs(r), feat, r, p))
    rows.sort(reverse=True)
    return rows

yes_corr = correlate(yes,  "YES trades")
no_corr  = correlate(no,   "NO trades")
no_dict  = {feat: (r, p) for _, feat, r, p in no_corr}

print("=" * 80)
print(f"  Indicator correlation with resolution  (YES n={len(yes)}, NO n={len(no)})")
print("=" * 80)
print(f"  {'Feature':<28} {'YES r':>7}  {'YES p':>8}  {'NO r':>7}  {'NO p':>8}  Signal?")
print("  " + "-" * 72)

NOISE_THRESH = 0.07

print("\n  --- USEFUL FOR YES (|r| >= 0.07, p < 0.10) ---")
for absr, feat, r, p in yes_corr:
    if absr < NOISE_THRESH:
        continue
    nr, np_ = no_dict.get(feat, (np.nan, np.nan))
    sig_yes = "✓" if p < 0.05 else "~"
    direction = "YES↑" if r > 0 else "NO↑"
    print(f"  {feat:<28} {r:>+7.3f}  {p:>8.4f}{sig_yes}  {nr:>+7.3f}  {np_:>8.4f}  {direction}")

print("\n  --- NOISE FOR YES (|r| < 0.07) ---")
for absr, feat, r, p in yes_corr:
    if absr >= NOISE_THRESH:
        continue
    nr, np_ = no_dict.get(feat, (np.nan, np.nan))
    # Mark if it IS useful for NO side
    no_note = "  [works for NO]" if abs(nr) >= NOISE_THRESH and np_ < 0.10 else ""
    print(f"  {feat:<28} {r:>+7.3f}  {p:>8.4f}   {nr:>+7.3f}  {np_:>8.4f}{no_note}")

# ── Summary: features useful for NO but not YES ─────────────────────────────
print()
print("=" * 80)
print("  Features that work for NO but NOT for YES")
print("=" * 80)
yes_dict = {feat: (r, p) for _, feat, r, p in yes_corr}
for absr, feat, r, p in no_corr:
    if absr < NOISE_THRESH:
        continue
    yr, yp = yes_dict.get(feat, (np.nan, np.nan))
    if abs(yr) < NOISE_THRESH:
        print(f"  {feat:<28} YES r={yr:>+.3f}  NO r={r:>+.3f}  → NO-only signal")

# ── Check if indicators have any price movement correlation at all ────────────
print()
print("=" * 80)
print("  Directional check: feature vs 15m close direction (did price go up?)")
print("  (independent of strike — pure price direction signal)")
print("=" * 80)

# Compute actual price direction from chg_15m sign
df["price_went_up"] = (pd.to_numeric(df["chg_15m"], errors="coerce") > 0).astype(float)
price_rows = []
for feat in FEATURES:
    if feat not in df.columns or feat == "chg_15m":
        continue
    col = pd.to_numeric(df[feat], errors="coerce")
    valid = df[col.notna() & df["price_went_up"].notna()]
    if len(valid) < 50 or col[col.notna()].std() < 1e-9:
        continue
    r, p = stats.pointbiserialr(col[valid.index], valid["price_went_up"])
    price_rows.append((abs(r), feat, r, p))

price_rows.sort(reverse=True)
print(f"  {'Feature':<28} {'r vs chg_15m':>13}  {'p':>8}  Signal?")
print("  " + "-" * 55)
for absr, feat, r, p in price_rows:
    if absr < 0.05:
        break
    sig = "✓" if p < 0.05 else "~"
    direction = "predicts UP" if r > 0 else "predicts DN"
    print(f"  {feat:<28} {r:>+13.3f}  {p:>8.4f}{sig}  {direction}")

print()
print("  (Features below |r|=0.05 with price direction — shown for completeness)")
shown = 0
for absr, feat, r, p in price_rows:
    if absr >= 0.05:
        continue
    sig = "✓" if p < 0.05 else " "
    print(f"  {feat:<28} {r:>+13.3f}  {p:>8.4f}{sig}")
    shown += 1
    if shown >= 10:
        break
