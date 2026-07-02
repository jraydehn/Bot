"""
analyze_markov_sideways_rescue.py

Within the 114 daily-Markov Sideways trades (WR=34.2%, -$735),
find subsets that perform above breakeven — rescue candidates.

Tests every available numeric and categorical feature from paper_trades.csv.
Runs threshold sweeps and binary splits, reports WR + flat P&L for each.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found")

TRADES_CSV  = "results/paper_trades.csv"
WINDOW      = 20
THRESHOLD   = 0.02       # daily ±2%
FLAT_BET    = 25.0
KALSHI_TAKE = 0.10
MIN_N       = 8          # minimum trades in a cell to report

SEP  = "=" * 68
SEP2 = "-" * 52

# ── 1. Build daily Markov regime ─────────────────────────────────────────────
print("Fetching BTC-USD daily data...")
df_d = yf.download("BTC-USD", start="2025-01-01", end="2026-05-23",
                   progress=False, auto_adjust=True)
if isinstance(df_d.columns, pd.MultiIndex):
    df_d.columns = df_d.columns.get_level_values(0)
close_d = df_d["Close"].dropna()
roll_d  = close_d.pct_change(WINDOW)
reg_d   = pd.Series("Sideways", index=close_d.index)
reg_d[roll_d >  THRESHOLD] = "Bull"
reg_d[roll_d < -THRESHOLD] = "Bear"
reg_d   = reg_d[roll_d.notna()]
reg_d.index = pd.to_datetime(reg_d.index, utc=True).normalize()

regime_df = reg_d.reset_index()
regime_df.columns = ["day_ts", "regime_daily"]

# ── 2. Load trades, filter to Sideways only ──────────────────────────────────
print(f"Loading {TRADES_CSV}...")
df = pd.read_csv(TRADES_CSV, low_memory=False)
df = df[df["decision"] == "trade"].copy()
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
df["p_market"]     = pd.to_numeric(df["p_market"],   errors="coerce")
df["trade_ts"]     = pd.to_datetime(df["logged_at"],  utc=True)
df["day_ts"]       = df["trade_ts"].dt.normalize()
df = df.merge(regime_df, on="day_ts", how="left")
df["regime_daily"] = df["regime_daily"].fillna("Unknown")

# Flat P&L
def flat_pnl(row):
    p = float(row["p_market"]) if pd.notna(row["p_market"]) else 0.5
    if row["side"] == "yes":
        won    = row["resolved_yes"] == 1
        payout = FLAT_BET / p * (1 - KALSHI_TAKE) if p > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET
    else:
        p_no   = 1 - p
        won    = row["resolved_yes"] == 0
        payout = FLAT_BET / p_no * (1 - KALSHI_TAKE) if p_no > 0 else 0
        return (payout - FLAT_BET) if won else -FLAT_BET

df["flat_pnl"] = df.apply(flat_pnl, axis=1)
df["won"]      = (
    ((df["side"]=="yes") & (df["resolved_yes"]==1)) |
    ((df["side"]=="no")  & (df["resolved_yes"]==0))
).astype(int)

sw = df[df["regime_daily"] == "Sideways"].copy()
print(f"\nSideways regime trades: {len(sw)}  "
      f"(yes={len(sw[sw['side']=='yes'])}, no={len(sw[sw['side']=='no'])})")
print(f"Overall WR: {sw['won'].mean()*100:.1f}%  P&L: ${sw['flat_pnl'].sum():+,.2f}")

# ── 3. Feature list ──────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "offset_pct", "p_market", "z_score", "net_edge", "tau_minutes",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score",
    "stoch_k", "stoch_d",
    "vwap_stretch_score", "vwap_distance_pct",
    "chg_30m", "chg_10m", "chg_5m", "bp_5m", "body_15m", "dir_15m",
    "vol_score", "vol_eff", "vpin_score", "obi_score",
    "funding_bias", "liq_score", "liq_bias",
    "ls_long_pct", "oi_chg_pct",
    "adx_1h", "rvol_1h", "squeeze_1h",
    "pm_drift_5m", "confirmation_score", "no_score",
]

for f in NUMERIC_FEATURES:
    if f in sw.columns:
        sw[f] = pd.to_numeric(sw[f], errors="coerce")

# ── 4. Helper ────────────────────────────────────────────────────────────────
def report(sub, label, baseline_n=len(sw)):
    n = len(sub)
    if n < MIN_N:
        return None
    wr  = sub["won"].mean()
    pnl = sub["flat_pnl"].sum()
    return {"label": label, "n": n, "pct_of_sw": n/baseline_n*100,
            "wr": wr, "pnl": pnl}

def print_row(r):
    if r is None:
        return
    print(f"  {r['label']:<42s}  n={r['n']:3d} ({r['pct_of_sw']:4.0f}%)  "
          f"WR={r['wr']*100:5.1f}%  P&L=${r['pnl']:+7.0f}")

# ── 5. By side ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SIDEWAYS GATE RESCUE ANALYSIS")
print(SEP)

print(f"\n--- BY SIDE ---")
for side in ("yes","no"):
    r = report(sw[sw["side"]==side], f"side={side}")
    print_row(r)

# ── 6. Threshold sweeps on continuous features ───────────────────────────────
print(f"\n--- FEATURE THRESHOLD SWEEPS (each side separately) ---")
print(f"  Looking for cells with WR≥55% and n≥{MIN_N}")
print()

rescues = []

for side in ("yes","no"):
    sub_side = sw[sw["side"]==side]
    if len(sub_side) < MIN_N:
        continue
    print(f"  [{side.upper()} trades — n={len(sub_side)}]")

    for feat in NUMERIC_FEATURES:
        if feat not in sw.columns:
            continue
        col = sub_side[feat].dropna()
        if len(col) < MIN_N:
            continue

        # quartile cutpoints
        q25, q50, q75 = col.quantile([0.25, 0.50, 0.75])

        for op, cutval, op_label in [
            (">=", q25, f"≥Q1({q25:.3g})"),
            (">=", q50, f"≥Q2({q50:.3g})"),
            (">=", q75, f"≥Q3({q75:.3g})"),
            ("<=", q25, f"≤Q1({q25:.3g})"),
            ("<=", q50, f"≤Q2({q50:.3g})"),
            ("<=", q75, f"≤Q3({q75:.3g})"),
        ]:
            if op == ">=":
                mask = sub_side[feat] >= cutval
            else:
                mask = sub_side[feat] <= cutval
            sub_cut = sub_side[mask.fillna(False)]
            r = report(sub_cut, f"{side} | {feat} {op_label}")
            if r and r["wr"] >= 0.55:
                rescues.append(r)

# Sort by P&L descending
rescues.sort(key=lambda x: x["pnl"], reverse=True)
if rescues:
    print(f"\n  CANDIDATES (WR≥55%, n≥{MIN_N}) sorted by P&L:")
    print(f"  {'Condition':<44s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 68)
    for r in rescues[:30]:
        print_row(r)
else:
    print("\n  No rescue candidates found at WR≥55%.")

# ── 7. Binary / categorical splits ───────────────────────────────────────────
print(f"\n--- BINARY / CATEGORICAL SPLITS ---")
binary_features = [
    "ema_stack_bias", "funding_bias", "liq_bias", "dir_15m",
    "squeeze_1h", "stoch_crossover_active", "sharp_move_active",
    "choch_1h", "choch_4h",
]

binary_rescues = []
for side in ("yes","no"):
    sub_side = sw[sw["side"]==side]
    if len(sub_side) < MIN_N:
        continue
    for feat in binary_features:
        if feat not in sw.columns:
            continue
        for val in sub_side[feat].dropna().unique():
            mask = sub_side[feat] == val
            sub_cut = sub_side[mask]
            r = report(sub_cut, f"{side} | {feat}={val}")
            if r and r["wr"] >= 0.55:
                binary_rescues.append(r)

binary_rescues.sort(key=lambda x: x["pnl"], reverse=True)
if binary_rescues:
    print(f"\n  CANDIDATES (WR≥55%, n≥{MIN_N}):")
    print(f"  {'Condition':<44s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
    print("  " + "-" * 68)
    for r in binary_rescues[:20]:
        print_row(r)
else:
    print("  No binary rescue candidates found at WR≥55%.")

# ── 8. Combined 2-feature interactions ───────────────────────────────────────
print(f"\n--- TOP COMBINATIONS (pairs from best rescues above) ---")

# Take top single-feature rescues and test pairwise combos
top_single = (rescues + binary_rescues)[:15]
combo_rescues = []
for i, r1 in enumerate(top_single):
    for r2 in top_single[i+1:]:
        # Parse label back is fragile; just re-test from the conditions dict
        pass  # We'll do this manually below via direct feature pairs

# Manual targeted combos: composite_trend + stoch + z_score + funding
combo_candidates = []
for side in ("yes","no"):
    sub_side = sw[sw["side"]==side]
    if len(sub_side) < MIN_N:
        continue

    # composite_trend direction match
    for ct_val, ct_label in [(-1,"trend≤-1"), (0,"trend=0"), (1,"trend≥1")]:
        if ct_val == -1:
            mask_ct = sub_side["composite_trend"] <= -1
        elif ct_val == 0:
            mask_ct = sub_side["composite_trend"] == 0
        else:
            mask_ct = sub_side["composite_trend"] >= 1

        sub_ct = sub_side[mask_ct.fillna(False)]
        r = report(sub_ct, f"{side} | {ct_label}")
        if r:
            combo_candidates.append(r)

        # + stoch splits
        for sk_thresh, sk_label in [(30,"stoch≤30"), (50,"stoch≤50"), (70,"stoch≤70"),
                                     (50,"stoch≥50"), (70,"stoch≥70")]:
            if "≤" in sk_label:
                mask_sk = sub_ct["stoch_k"] <= sk_thresh
            else:
                mask_sk = sub_ct["stoch_k"] >= sk_thresh
            r2 = report(sub_ct[mask_sk.fillna(False)],
                        f"{side} | {ct_label} + {sk_label}")
            if r2:
                combo_candidates.append(r2)

        # + z_score splits
        for z_thresh, z_label in [(-0.5,"z≤-0.5"), (-0.2,"z≤-0.2"),
                                    (0.2,"z≥0.2"),  (0.5,"z≥0.5")]:
            if "≤" in z_label:
                mask_z = sub_ct["z_score"] <= z_thresh
            else:
                mask_z = sub_ct["z_score"] >= z_thresh
            r2 = report(sub_ct[mask_z.fillna(False)],
                        f"{side} | {ct_label} + {z_label}")
            if r2:
                combo_candidates.append(r2)

        # + funding_bias
        for fb_val in [-1, 0, 1]:
            mask_fb = sub_ct["funding_bias"] == fb_val
            r2 = report(sub_ct[mask_fb.fillna(False)],
                        f"{side} | {ct_label} + fund={fb_val}")
            if r2:
                combo_candidates.append(r2)

        # + ema_stack_bias
        for em_val in [-1, 0, 1]:
            mask_em = sub_ct["ema_stack_bias"] == em_val
            r2 = report(sub_ct[mask_em.fillna(False)],
                        f"{side} | {ct_label} + ema={em_val}")
            if r2:
                combo_candidates.append(r2)

combo_candidates.sort(key=lambda x: x["pnl"], reverse=True)
print(f"\n  All combo cells (n≥{MIN_N}), sorted by P&L:")
print(f"  {'Condition':<50s}  {'n':>4s}  {'WR':>6s}  {'P&L':>8s}")
print("  " + "-" * 74)
for r in combo_candidates[:35]:
    print_row(r)

# ── 9. Summary of best rescue vs hard block ──────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY: block vs rescue options")
print(SEP)
print(f"\n  HARD BLOCK all Sideways: save ${sw['flat_pnl'].sum():+,.2f} on {len(sw)} trades")
print(f"\n  Best rescue candidates (WR≥55%, n≥{MIN_N}):")
all_candidates = sorted(rescues + binary_rescues + combo_candidates,
                        key=lambda x: x["pnl"], reverse=True)
seen = set()
for r in all_candidates:
    if r["label"] in seen or r["wr"] < 0.55:
        continue
    seen.add(r["label"])
    blocked_pnl = sw["flat_pnl"].sum() - r["pnl"]
    print(f"  RESCUE: {r['label']}")
    print(f"    Keep {r['n']} trades (WR={r['wr']*100:.1f}%, P&L=${r['pnl']:+,.0f})")
    print(f"    Block remaining {len(sw)-r['n']} trades")
    print(f"    P&L saved by blocking non-rescue: ${-blocked_pnl:+,.0f}")
    print()
    if len(seen) >= 8:
        break
