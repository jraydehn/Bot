"""
backfill_he_signals.py

Backfills HierarchicalExtremes Level-1 (and Level-2) swing signals into the
BTC scan archive, extending btc_scan_archive_dc.parquet.

For each level n ∈ {1, 2}:
  he_l{n}_high         : last confirmed Level-n swing high price
  he_l{n}_low          : last confirmed Level-n swing low price
  he_l{n}_direction    : +1 if last L{n} extreme was a high, -1 if a low
  he_l{n}_dist_high    : (spot - he_l{n}_high) / spot * 100
  he_l{n}_dist_low     : (spot - he_l{n}_low)  / spot * 100
  he_l{n}_higher_high  : bool — last L{n} high > previous L{n} high (structural HH)
  he_l{n}_lower_low    : bool — last L{n} low  < previous L{n} low  (structural LL)

Level 0 = same as ATR-DC extremes (already in btc_scan_archive_dc.parquet as swing_high_240).
Level 1 = structural highs/lows: confirmed only when a lower high or higher low validates
           the prior extreme as a significant pivot — the main shadow signal of interest.
Level 2 = major structural pivots (monthly-scale); coverage will be thin early on.

Methodology: same merge_asof approach as backfill_dc_signals.py.
  conf_timestamp is used (not timestamp) to prevent lookahead.

Usage:
  python3 backfill_he_signals.py [--atr-lookback 240] [--dry-run]

Output: results/btc_scan_archive_he.parquet
  (reads btc_scan_archive_dc.parquet as base; adds HE columns)
"""
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_extremes import HierarchicalExtremes

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--atr-lookback", type=int, default=240,
                    help="ATR lookback in 1m bars (default 240 = 4h)")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

ATR_LB = args.atr_lookback
LEVELS = 3   # Level 0 (DC), Level 1 (structural), Level 2 (major)


# ── Load 1m price data ────────────────────────────────────────────────────────

print("Loading 1m parquet …", end=" ", flush=True)
pq_1m = max(DATA.glob("binanceus_BTCUSDT_1m_*.parquet"), key=os.path.getmtime)
df1m  = pd.read_parquet(pq_1m, columns=["high", "low", "close"])
if df1m.index.tz is None:
    df1m.index = df1m.index.tz_localize("UTC")
df1m = df1m.sort_index()
print(f"{len(df1m):,} bars  ({df1m.index[0].date()} → {df1m.index[-1].date()})")

h = df1m["high"].to_numpy()
l = df1m["low"].to_numpy()
c = df1m["close"].to_numpy()
t = df1m.index


# ── Load scan archive (DC parquet as base) ────────────────────────────────────

print("Loading scan archive …", end=" ", flush=True)
dc_path = RESULTS / "btc_scan_archive_dc.parquet"
sa = pd.read_parquet(dc_path)
sa = sa[sa["resolved_yes"].notna()].copy()
sa["logged_at"] = pd.to_datetime(sa["logged_at"], errors="coerce", utc=True)
sa["spot"]      = pd.to_numeric(sa["spot"],       errors="coerce")
print(f"{len(sa):,} resolved rows  ({sa['logged_at'].min().date()} → {sa['logged_at'].max().date()})")


# ── Run HierarchicalExtremes ──────────────────────────────────────────────────

print(f"\nRunning HierarchicalExtremes (atr_lookback={ATR_LB}, levels={LEVELS}) "
      f"on {len(h):,} bars …", end=" ", flush=True)

he = HierarchicalExtremes(levels=LEVELS, atr_lookback=ATR_LB)
for i in range(len(h)):
    he.update(i, t, h, l, c)

print("done.")
for lvl in range(LEVELS):
    n_h = sum(1 for e in he.extremes[lvl] if e.ext_type ==  1)
    n_l = sum(1 for e in he.extremes[lvl] if e.ext_type == -1)
    print(f"  Level {lvl}: {len(he.extremes[lvl])} extremes  (highs={n_h}  lows={n_l})")


# ── merge_asof helper (same as backfill_dc_signals.py) ───────────────────────

sa = sa.sort_values("logged_at").reset_index(drop=True)
sa_clean = sa.dropna(subset=["logged_at"]).copy()
sa_clean["_orig_idx"] = sa_clean.index


def _asof(right_ts: pd.Series, right_val: pd.Series, col_name: str) -> pd.Series:
    right = right_ts.rename("logged_at").to_frame()
    right[col_name] = right_val.values
    merged = pd.merge_asof(
        sa_clean[["logged_at", "_orig_idx"]].sort_values("logged_at"),
        right.sort_values("logged_at"),
        on="logged_at", direction="backward",
    )
    return merged.set_index("_orig_idx")[col_name]


# ── Build columns for Level 1 and Level 2 ────────────────────────────────────

for lvl in range(1, LEVELS):
    exts = he.extremes[lvl]
    if not exts:
        print(f"\nLevel {lvl}: no extremes — skipping.")
        continue

    print(f"\n── Level {lvl} ──────────────────────────────────────────────────────")

    ext_df = pd.DataFrame([{
        "ext_type":       e.ext_type,
        "price":          e.price,
        "conf_timestamp": e.conf_timestamp,
    } for e in exts])
    ext_df["conf_timestamp"] = pd.to_datetime(ext_df["conf_timestamp"], utc=True)

    highs = ext_df[ext_df["ext_type"] ==  1].copy().sort_values("conf_timestamp").reset_index(drop=True)
    lows  = ext_df[ext_df["ext_type"] == -1].copy().sort_values("conf_timestamp").reset_index(drop=True)
    all_e = ext_df.sort_values("conf_timestamp").reset_index(drop=True)

    print(f"  Highs: {len(highs)}  Lows: {len(lows)}")

    # Last confirmed high/low price
    prefix = f"he_l{lvl}"
    sa[f"{prefix}_high"]      = _asof(highs["conf_timestamp"], highs["price"],         f"{prefix}_high")
    sa[f"{prefix}_low"]       = _asof(lows["conf_timestamp"],  lows["price"],          f"{prefix}_low")
    sa[f"{prefix}_direction"] = _asof(all_e["conf_timestamp"], all_e["ext_type"],      f"{prefix}_direction")

    # Distance from spot (pct)
    sa[f"{prefix}_dist_high"] = (sa["spot"] - sa[f"{prefix}_high"]) / sa["spot"] * 100
    sa[f"{prefix}_dist_low"]  = (sa["spot"] - sa[f"{prefix}_low"])  / sa["spot"] * 100

    # Higher-high: at each confirmed high, was it > the previous confirmed high?
    # Build a time series keyed on conf_timestamp of each high.
    if len(highs) >= 2:
        highs["higher_high"] = highs["price"] > highs["price"].shift(1)
        highs["higher_high"] = highs["higher_high"].fillna(False)
        sa[f"{prefix}_higher_high"] = _asof(
            highs["conf_timestamp"], highs["higher_high"].astype(int), f"{prefix}_higher_high"
        ).astype("boolean")
    else:
        sa[f"{prefix}_higher_high"] = pd.NA

    # Lower-low: at each confirmed low, was it < the previous confirmed low?
    if len(lows) >= 2:
        lows["lower_low"] = lows["price"] < lows["price"].shift(1)
        lows["lower_low"] = lows["lower_low"].fillna(False)
        sa[f"{prefix}_lower_low"] = _asof(
            lows["conf_timestamp"], lows["lower_low"].astype(int), f"{prefix}_lower_low"
        ).astype("boolean")
    else:
        sa[f"{prefix}_lower_low"] = pd.NA

    # Coverage
    cov_h = sa[f"{prefix}_high"].notna().mean()
    cov_l = sa[f"{prefix}_low"].notna().mean()
    print(f"  Coverage: high={cov_h:.1%}  low={cov_l:.1%}")
    if sa[f"{prefix}_higher_high"].notna().any():
        hh_rate = sa[f"{prefix}_higher_high"].dropna().astype(float).mean()
        ll_rate = sa[f"{prefix}_lower_low"].dropna().astype(float).mean()
        print(f"  HH rate={hh_rate:.1%}  LL rate={ll_rate:.1%}")


# ── Save ──────────────────────────────────────────────────────────────────────

out = RESULTS / "btc_scan_archive_he.parquet"
new_cols = [c for c in sa.columns if c.startswith("he_")]
print(f"\nNew columns: {new_cols}")
print(f"Output: {out.name}  ({len(sa):,} rows, {len(sa.columns)} columns)")

if args.dry_run:
    print("[dry-run] Not saving.")
else:
    sa.to_parquet(out, index=False)
    print("Saved.")

print("Done.")
