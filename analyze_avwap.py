"""
analyze_avwap.py

Data sweep: does anchored VWAP (from last 4h swing high) add edge
beyond daily VWAP for BTC hourly contract prediction?

Approach:
  1. Load btc_scan_archive.csv (all resolved rows)
  2. Load BTC 4h OHLCV data
  3. Detect 4h swing highs (local max with n=4 bars each side, min prominence 0.8%)
  4. For each scan row: find the most recent swing high before scan_ts
  5. Compute AVWAP from that swing high using 4h bars
  6. Tag rows with: avwap_dist_pct, above_avwap, sh_price, sh_dist_pct
  7. WR analysis stratified by:
      - above/below AVWAP × YES/NO side
      - Near swing high (sh_dist_pct in [-3%, 0%]) × above/below AVWAP
      - AVWAP vs daily VWAP comparison (already in archive as vwap_stretch_score)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

# ── Load scan archive ────────────────────────────────────────────────────────
print("Loading scan archive...")
arc = pd.read_csv(RESULTS_DIR / "btc_scan_archive.csv", low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce")
arc = arc[arc["logged_at"].notna() & arc["resolved_yes"].notna()].copy()
arc["resolved_yes"] = arc["resolved_yes"].astype(int)
print(f"  {len(arc):,} resolved rows  |  {arc['logged_at'].min().date()} → {arc['logged_at'].max().date()}")

# ── Load 4h BTC data ─────────────────────────────────────────────────────────
print("Loading 4h data...")
df4h = pd.read_parquet(DATA_DIR / "binanceus_BTCUSDT_4h_1970-01-01_2026-06-14.parquet")
df4h.index = pd.to_datetime(df4h.index, utc=True)
# Keep only bars that cover our scan window + 30 days buffer for swing high search
cutoff = arc["logged_at"].min() - pd.Timedelta(days=30)
df4h   = df4h[df4h.index >= cutoff].copy()
print(f"  {len(df4h)} 4h bars  |  {df4h.index[0].date()} → {df4h.index[-1].date()}")

# ── Detect swing highs ───────────────────────────────────────────────────────
# A bar is a swing high if it is the highest high in ±N bars AND has prominence >= min_prom_pct
N_SIDE     = 4    # bars each side (= 16h each side)
MIN_PROM   = 0.8  # minimum prominence as % of price

highs = df4h["high"].values
peaks, props = find_peaks(
    highs,
    distance=N_SIDE,
    prominence=highs * (MIN_PROM / 100),  # relative prominence
)

sh_times  = df4h.index[peaks]
sh_prices = highs[peaks]
print(f"  {len(peaks)} swing highs detected (N_side={N_SIDE}, min_prom={MIN_PROM}%)")

# ── Compute AVWAP from last swing high for each scan row ─────────────────────
# AVWAP = cumsum(typical_price * volume) / cumsum(volume)  from swing high bar onward
# We precompute a lookup: for each 4h bar, what is the AVWAP from the most recent swing high?

# Build a mapping: bar_time → (avwap, sh_price, sh_time)
# For efficiency, process in order.

df4h["tp"] = (df4h["high"] + df4h["low"] + df4h["close"]) / 3
df4h["tv"] = df4h["tp"] * df4h["volume"]

avwap_arr    = np.full(len(df4h), np.nan)
sh_price_arr = np.full(len(df4h), np.nan)
sh_time_list = [pd.NaT] * len(df4h)

cum_tv = 0.0
cum_v  = 0.0
peak_set = set(peaks)
last_sh_price = np.nan
last_sh_idx   = -1

for i in range(len(df4h)):
    if i in peak_set:
        cum_tv = 0.0
        cum_v  = 0.0
        last_sh_price = df4h["high"].iloc[i]
        last_sh_idx   = i
    cum_tv += df4h["tv"].iloc[i]
    cum_v  += df4h["volume"].iloc[i]
    if cum_v > 0 and not np.isnan(last_sh_price):
        avwap_arr[i]    = cum_tv / cum_v
        sh_price_arr[i] = last_sh_price
        sh_time_list[i] = df4h.index[last_sh_idx]

df4h["avwap_sh"]  = avwap_arr
df4h["sh_price"]  = sh_price_arr
df4h["sh_time"]   = pd.to_datetime(sh_time_list, utc=True)

# ── Merge AVWAP into scan archive ─────────────────────────────────────────────
# For each scan row, find the 4h bar that was completed just before logged_at
# (i.e., the most recent 4h bar whose open_time <= logged_at)
print("Merging AVWAP into scan archive...")

df4h_reset = df4h[["avwap_sh", "sh_price", "sh_time"]].reset_index()
df4h_reset.columns = ["bar_time", "avwap_sh", "sh_price", "sh_time"]
df4h_reset = df4h_reset.sort_values("bar_time")

arc_sorted = arc.sort_values("logged_at")
merged = pd.merge_asof(
    arc_sorted,
    df4h_reset,
    left_on="logged_at",
    right_on="bar_time",
    direction="backward",
)

merged["avwap_dist_pct"] = (merged["spot"] - merged["avwap_sh"]) / merged["avwap_sh"] * 100
merged["sh_dist_pct"]    = (merged["spot"] - merged["sh_price"]) / merged["sh_price"] * 100
merged["above_avwap"]    = merged["spot"] > merged["avwap_sh"]

print(f"  AVWAP coverage: {merged['avwap_sh'].notna().sum():,} / {len(merged):,} rows")
print(f"  above_avwap: {merged['above_avwap'].sum():,}  below: {(~merged['above_avwap']).sum():,}")
print(f"  avg avwap_dist_pct: {merged['avwap_dist_pct'].mean():.2f}%  std: {merged['avwap_dist_pct'].std():.2f}%")
print(f"  avg sh_dist_pct:    {merged['sh_dist_pct'].mean():.2f}%  std: {merged['sh_dist_pct'].std():.2f}%")

# ── Helper ───────────────────────────────────────────────────────────────────
def wr_stats(df, label=""):
    n = len(df)
    if n == 0:
        return
    wr  = df["resolved_yes"].mean()
    pnl = (df["resolved_yes"] * (1 - df["p_market"]) - (1 - df["resolved_yes"]) * df["p_market"]).sum()
    bkev = df["p_market"].mean()
    print(f"  {label:<45s}  n={n:>5d}  WR={wr:.1%}  bkev={bkev:.1%}  est_pnl=${pnl:+.0f}")

df = merged.dropna(subset=["avwap_sh", "resolved_yes"])

# ── Analysis 1: Overall AVWAP position vs resolved_yes ───────────────────────
print("\n═══ 1. AVWAP position (all contracts) ═══")
wr_stats(df[df["above_avwap"]],  "Above AVWAP")
wr_stats(df[~df["above_avwap"]], "Below AVWAP")

# ── Analysis 2: YES-side contracts (offset_pct > 0 → strike above spot, YES=OTM) ──
# offset_pct sign: from btc_scan_archive offset_pct = (strike - spot) / spot
# YES OTM = strike above spot = offset_pct > 0
# YES ITM = strike below spot = offset_pct < 0
print("\n═══ 2. YES-side contracts (all offset) ═══")
yes_side = df[df["p_market"] <= 0.5]   # low p_market = YES OTM (market thinks unlikely to be YES)
no_side  = df[df["p_market"] > 0.5]    # high p_market = YES ITM / NO OTM

wr_stats(yes_side[yes_side["above_avwap"]],   "YES-side  + above AVWAP")
wr_stats(yes_side[~yes_side["above_avwap"]],  "YES-side  + below AVWAP")
wr_stats(no_side[no_side["above_avwap"]],     "NO-side   + above AVWAP")
wr_stats(no_side[~no_side["above_avwap"]],    "NO-side   + below AVWAP")

# ── Analysis 3: Near swing high (spot within 3% below the sh_price) ──────────
print("\n═══ 3. Near swing high (sh_dist_pct in [-3%, 0%]) ═══")
near_sh = df[(df["sh_dist_pct"] >= -3.0) & (df["sh_dist_pct"] <= 0.5)]
far_sh  = df[df["sh_dist_pct"] < -3.0]
print(f"  Near swing high: {len(near_sh):,}  Far from sh: {len(far_sh):,}")
wr_stats(near_sh[near_sh["above_avwap"]],   "Near SH + above AVWAP (double block)")
wr_stats(near_sh[~near_sh["above_avwap"]],  "Near SH + below AVWAP (cost basis below)")
wr_stats(far_sh[far_sh["above_avwap"]],     "Far SH  + above AVWAP")
wr_stats(far_sh[~far_sh["above_avwap"]],    "Far SH  + below AVWAP")

# ── Analysis 4: YES-side near swing high ─────────────────────────────────────
print("\n═══ 4. YES-side × near swing high × AVWAP ═══")
y_near = yes_side[(yes_side["sh_dist_pct"] >= -3.0) & (yes_side["sh_dist_pct"] <= 0.5)]
wr_stats(y_near[y_near["above_avwap"]],   "YES + near SH + above AVWAP (worst)")
wr_stats(y_near[~y_near["above_avwap"]],  "YES + near SH + below AVWAP")
wr_stats(yes_side[yes_side["sh_dist_pct"] < -3.0], "YES + far from SH (baseline)")

# ── Analysis 5: AVWAP distance quartiles ─────────────────────────────────────
print("\n═══ 5. AVWAP distance quartiles (YES-side) ═══")
yes_side = yes_side.copy()
yes_side["avwap_q"] = pd.qcut(yes_side["avwap_dist_pct"], q=4, labels=["Q1(far below)", "Q2(below)", "Q3(above)", "Q4(far above)"])
for q in ["Q1(far below)", "Q2(below)", "Q3(above)", "Q4(far above)"]:
    bucket = yes_side[yes_side["avwap_q"] == q]
    rng = (bucket["avwap_dist_pct"].min(), bucket["avwap_dist_pct"].max())
    wr_stats(bucket, f"YES {q} [{rng[0]:.2f}% to {rng[1]:.2f}%]")

# ── Analysis 6: AVWAP vs daily VWAP comparison ────────────────────────────────
print("\n═══ 6. AVWAP vs daily VWAP (vwap_stretch_score) — YES-side ═══")
# vwap_stretch_score: positive = above daily VWAP, negative = below
yes_vwap = yes_side.dropna(subset=["vwap_stretch_score"])
# Above daily VWAP AND above AVWAP
aa = yes_vwap[(yes_vwap["vwap_stretch_score"] > 0) & yes_vwap["above_avwap"]]
ab = yes_vwap[(yes_vwap["vwap_stretch_score"] > 0) & ~yes_vwap["above_avwap"]]
ba = yes_vwap[(yes_vwap["vwap_stretch_score"] <= 0) & yes_vwap["above_avwap"]]
bb = yes_vwap[(yes_vwap["vwap_stretch_score"] <= 0) & ~yes_vwap["above_avwap"]]
wr_stats(aa, "Daily VWAP+ & AVWAP+  (both bullish)")
wr_stats(ab, "Daily VWAP+ & AVWAP-  (divergence)")
wr_stats(ba, "Daily VWAP- & AVWAP+  (divergence)")
wr_stats(bb, "Daily VWAP- & AVWAP-  (both bearish)")

# ── Analysis 7: swing high gate synergy ──────────────────────────────────────
# Existing gate blocks YES when sigma_swing_high_1pct is True or when
# swing high is close. Use sh_dist_pct as a proxy: sh_dist_pct in [-1%, 0%]
print("\n═══ 7. AVWAP as refinement of swing_high_gate (sh_dist_pct in [-1%, 0%]) ═══")
tight_sh = yes_side[(yes_side["sh_dist_pct"] >= -1.0) & (yes_side["sh_dist_pct"] <= 0.5)]
print(f"  Tight swing high population (YES, within 1%): {len(tight_sh):,}")
wr_stats(tight_sh[tight_sh["above_avwap"]],    "Tight SH + above AVWAP  → block YES")
wr_stats(tight_sh[~tight_sh["above_avwap"]],   "Tight SH + below AVWAP  → possible rescue")

# ── Analysis 8: NO-side specific ─────────────────────────────────────────────
print("\n═══ 8. NO-side (p_market > 0.5) × AVWAP ═══")
no_side_avwap = no_side.dropna(subset=["avwap_dist_pct"])
no_side_avwap = no_side_avwap.copy()
no_side_avwap["avwap_q"] = pd.qcut(no_side_avwap["avwap_dist_pct"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
for q in ["Q1", "Q2", "Q3", "Q4"]:
    bucket = no_side_avwap[no_side_avwap["avwap_q"] == q]
    rng = (bucket["avwap_dist_pct"].min(), bucket["avwap_dist_pct"].max())
    wr_stats(bucket, f"NO {q} [{rng[0]:.2f}% to {rng[1]:.2f}%]  (resolved_yes=1 = NO loses)")

print("\nDone.")
