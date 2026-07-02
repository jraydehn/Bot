#!/usr/bin/env python3
"""
Which indicators correlate with offset_pct on YES trades?
offset_pct captures strike proximity (how far ITM/OTM the contract already is).
Indicators correlated with offset_pct track WHERE price is relative to strike,
not just which direction it's moving.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].isin([0.0, 1.0])].copy()
df["resolved_yes"] = df["resolved_yes"].astype(int)
yes = df[df["side"].str.upper() == "YES"].copy()

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce") if c not in ["side", "asset", "contract_ticker",
        "logged_at", "decision_time", "close_time"] else df[c]

yes_num = yes.copy()
for c in yes_num.columns:
    yes_num[c] = pd.to_numeric(yes_num[c], errors="coerce") if c not in [
        "side", "asset", "contract_ticker", "logged_at", "decision_time", "close_time"
    ] else yes_num[c]

offset = pd.to_numeric(yes_num["offset_pct"], errors="coerce")

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
    "tau_minutes",
]

print(f"YES trades: {len(yes_num)}  mean offset_pct: {offset.mean():.4f}  std: {offset.std():.4f}")
print()
print("=" * 80)
print("  Indicator vs offset_pct correlation (YES trades)")
print("  High |r| = indicator tracks strike proximity, not just momentum")
print("=" * 80)
print(f"  {'Feature':<28} {'r_offset':>9}  {'p':>8}  {'r_resol':>9}  {'Interpretation'}")
print("  " + "-" * 74)

rows = []
resol_dict = {}
for feat in FEATURES:
    if feat not in yes_num.columns:
        continue
    col = pd.to_numeric(yes_num[feat], errors="coerce")
    both = offset.notna() & col.notna()
    if both.sum() < 30 or col[both].std() < 1e-9:
        continue
    r_off, p_off = stats.pearsonr(col[both], offset[both])
    r_res, p_res = stats.pointbiserialr(col[both], yes_num.loc[both, "resolved_yes"])
    resol_dict[feat] = r_res
    rows.append((abs(r_off), feat, r_off, p_off, r_res, p_res))

rows.sort(reverse=True)

for absr, feat, r_off, p_off, r_res, p_res in rows:
    sig = "✓" if p_off < 0.05 else "~"
    # Classify relationship
    if absr >= 0.15:
        if r_off > 0 and r_res > 0.05:
            interp = "strike-proximity: bullish = more ITM → YES edge"
        elif r_off < 0 and r_res < -0.05:
            interp = "strike-proximity: bearish = more ITM → YES edge"
        elif r_off > 0 and r_res < -0.05:
            interp = "paradox: corr w/ offset but anti-YES"
        elif absr >= 0.30:
            interp = "strong offset link"
        else:
            interp = "moderate offset link"
    elif absr >= 0.07:
        interp = "weak offset link"
    else:
        interp = "independent of offset"
    print(f"  {feat:<28} {r_off:>+9.3f}  {p_off:>8.4f}{sig}  {r_res:>+9.3f}  {interp}")

# ── Partial correlation: r(feature, resolved_yes) controlling for offset_pct ──
print()
print("=" * 80)
print("  Partial correlation: feature vs YES resolution, controlling for offset_pct")
print("  This shows which features add signal BEYOND what offset_pct already captures")
print("=" * 80)
print(f"  {'Feature':<28} {'r_partial':>10}  {'p_partial':>10}  {'r_raw':>8}  Adds value?")
print("  " + "-" * 70)

partial_rows = []
for feat in FEATURES:
    if feat not in yes_num.columns:
        continue
    col = pd.to_numeric(yes_num[feat], errors="coerce")
    both = offset.notna() & col.notna() & yes_num["resolved_yes"].notna()
    if both.sum() < 30 or col[both].std() < 1e-9:
        continue

    X = col[both].values
    Y = yes_num.loc[both, "resolved_yes"].values.astype(float)
    O = offset[both].values

    # Residualize X and Y on offset_pct
    def residualize(a, b):
        slope = np.cov(a, b)[0, 1] / np.var(b)
        return a - slope * b

    X_res = residualize(X, O)
    Y_res = residualize(Y, O)

    if X_res.std() < 1e-9 or Y_res.std() < 1e-9:
        continue

    r_part, p_part = stats.pearsonr(X_res, Y_res)
    r_raw = resol_dict.get(feat, np.nan)
    partial_rows.append((abs(r_part), feat, r_part, p_part, r_raw))

partial_rows.sort(reverse=True)
for absr, feat, r_part, p_part, r_raw in partial_rows:
    sig = "✓" if p_part < 0.05 else ("~" if p_part < 0.10 else " ")
    adds = "YES" if p_part < 0.05 and absr >= 0.08 else ("maybe" if p_part < 0.10 else "no")
    print(f"  {feat:<28} {r_part:>+10.3f}  {p_part:>10.4f}{sig}  {r_raw:>+8.3f}  {adds}")
