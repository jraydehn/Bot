#!/usr/bin/env python3
"""
backfill_smc_paper_trades.py

Compute SMC signals (bos_4h, choch_4h, bos_1h, choch_1h, supply_pct, demand_pct,
in_supply_zone, in_demand_zone, swing_high/low 4h/1h) for BTC paper-trade rows that
predate the live SMC logging (Mar 23 – Apr 21 2026).

Writes results/paper_trades_smc_backfill.csv — same columns as the live paper_trades
SMC columns, keyed on (contract_ticker, logged_at) for merging.
"""

import sys
from pathlib import Path
from datetime import timezone

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import smc_signals  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
ARCHIVES = [
    "results/paper_trades_archive_20260323_003206.csv",
    "results/paper_trades_archive_20260325_090712.csv",
    "results/paper_trades_archive_20260330_103000.csv",
    "results/paper_trades_archive_20260330_1030pdt.csv",
    "results/paper_trades_archive_20260330_2230pdt.csv",
    "results/paper_trades_archive_pre_v2.csv",
    "results/paper_trades_archive_pre_vwap_invert.csv",
    "results/paper_trades_archive_20260405_1633pdt.csv",
    "results/paper_trades_archive_20260407_122844.csv",
    "results/paper_trades_archive_20260407_152310.csv",
    "results/paper_trades_archive_20260415_1342_precal.csv",
    "results/paper_trades_archive_20260419_1407_predrift.csv",
    "results/paper_trades_archive_20260420_0431_pre_counter_tape.csv",
    "results/paper_trades_archive_20260420_2328_pre_direct_model.csv",
    "results/paper_trades_archive_20260421_btc_drift14_bad.csv",
]

# Most recent parquet covers all history
PARQUET_1M = ROOT / "data" / "binanceus_BTCUSDT_1m_1970-01-01_2026-06-01.parquet"
OUT_CSV    = ROOT / "results" / "paper_trades_smc_backfill.csv"

MIN_1H_BARS = 30   # need at least 30 hours for BOS/ChoCH detection
MIN_4H_BARS = 12   # need at least 48 hours for 4h detection

# ── Load and deduplicate source trades ────────────────────────────────────────
print("Loading archive files...")
dfs = []
for f in ARCHIVES:
    p = ROOT / f
    if p.exists():
        dfs.append(pd.read_csv(p, low_memory=False))
    else:
        print(f"  MISSING: {f}")

combined = pd.concat(dfs, ignore_index=True)
combined["logged_at"] = pd.to_datetime(combined["logged_at"], utc=True, errors="coerce")
combined = combined.dropna(subset=["logged_at"])
combined = combined[combined["contract_ticker"].str.contains("KXBTCD", na=False)]
combined = combined.drop_duplicates(subset=["contract_ticker", "logged_at"])

print(f"  {len(combined)} unique BTC rows  "
      f"({combined['logged_at'].min().date()} → {combined['logged_at'].max().date()})")

# ── Load price data ───────────────────────────────────────────────────────────
print(f"\nLoading 1m parquet: {PARQUET_1M.name} ...")
df1m = pd.read_parquet(PARQUET_1M)
df1m.index = pd.to_datetime(df1m.index, utc=True)
df1m = df1m.sort_index()
print(f"  {len(df1m):,} 1m bars  ({df1m.index[0].date()} → {df1m.index[-1].date()})")

# Resample to 1h and 4h (closed='left', label='left')
print("  Resampling to 1h and 4h...")
agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df1h = df1m.resample("1h",  closed="left", label="left").agg(agg).dropna()
df4h = df1m.resample("4h",  closed="left", label="left").agg(agg).dropna()
print(f"  1h: {len(df1h):,} bars   4h: {len(df4h):,} bars")

# ── Compute SMC per unique trade hour (cache by floor-hour) ──────────────────
print("\nComputing SMC signals...")

# Group by floor-hour to avoid recomputing for every row
combined["_hour"] = combined["logged_at"].dt.floor("1h")
unique_hours = sorted(combined["_hour"].unique())
print(f"  {len(unique_hours)} unique hours to process")

smc_cache: dict = {}   # hour → SMCResult

for i, hour_ts in enumerate(unique_hours):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{len(unique_hours)}] {hour_ts.date()} {hour_ts.time()}")

    # Slice price data up to (but not including) the current bar
    # Use bars strictly before this hour so we don't look forward
    df1h_slice = df1h[df1h.index < hour_ts]
    df4h_slice = df4h[df4h.index < hour_ts]

    if len(df1h_slice) < MIN_1H_BARS or len(df4h_slice) < MIN_4H_BARS:
        smc_cache[hour_ts] = None
        continue

    # Spot: use close of the 1m bar just before this hour
    df1m_slice = df1m[df1m.index < hour_ts]
    spot = float(df1m_slice["close"].iloc[-1]) if len(df1m_slice) else float(df1h_slice["close"].iloc[-1])

    try:
        result = smc_signals.get_smc_signals(df1h_slice, df4h_slice, spot)
        smc_cache[hour_ts] = result
    except Exception as e:
        print(f"    ERROR at {hour_ts}: {e}")
        smc_cache[hour_ts] = None

# ── Attach SMC to rows ────────────────────────────────────────────────────────
print("\nAttaching SMC signals to rows...")

rows = []
for _, row in combined.iterrows():
    smc = smc_cache.get(row["_hour"])
    if smc is None:
        rows.append({
            "contract_ticker": row["contract_ticker"],
            "logged_at":       row["logged_at"].isoformat(),
            "smc_4h":          None, "smc_1h":          None,
            "choch_4h":        None, "choch_1h":        None,
            "swing_high_4h":   None, "swing_low_4h":    None,
            "swing_high_1h":   None, "swing_low_1h":    None,
            "supply_pct":      None, "demand_pct":      None,
            "in_supply_zone":  None, "in_demand_zone":  None,
            "n_supply_zones":  None, "n_demand_zones":  None,
        })
    else:
        rows.append({
            "contract_ticker": row["contract_ticker"],
            "logged_at":       row["logged_at"].isoformat(),
            "smc_4h":          smc.bos_4h,
            "smc_1h":          smc.bos_1h,
            "choch_4h":        smc.choch_4h,
            "choch_1h":        smc.choch_1h,
            "swing_high_4h":   smc.swing_high_4h,
            "swing_low_4h":    smc.swing_low_4h,
            "swing_high_1h":   smc.swing_high_1h,
            "swing_low_1h":    smc.swing_low_1h,
            "supply_pct":      smc.nearest_supply_pct,
            "demand_pct":      smc.nearest_demand_pct,
            "in_supply_zone":  smc.in_supply_zone,
            "in_demand_zone":  smc.in_demand_zone,
            "n_supply_zones":  smc.n_supply_zones,
            "n_demand_zones":  smc.n_demand_zones,
        })

out = pd.DataFrame(rows)
filled = out["smc_4h"].notna().sum()
print(f"  {filled}/{len(out)} rows filled ({filled/len(out)*100:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(out)} rows)")

# Quick validation
bets_with_smc = out[out["smc_4h"].notna()]
print(f"\nSample distribution (smc_4h):")
print(bets_with_smc["smc_4h"].value_counts().to_string())
