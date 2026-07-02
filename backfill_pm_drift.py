"""
Backfill pm_drift_5m in btc_scan_archive.csv.

For each row missing pm_drift_5m, find the same ticker's p_market closest to
(logged_at - 5min) within a 2-8 minute lookback window, compute the difference,
and write it back.

Uses merge_asof per ticker for vectorized performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ARCHIVE_PATH = Path(__file__).parent / "results" / "btc_scan_archive.csv"
# Look for a historical reading between 2 and 8 minutes before logged_at
LOOKBACK_TARGET = pd.Timedelta("5min")
LOOKBACK_TOLERANCE = pd.Timedelta("3min")   # ±3min around the 5min target


def _backfill_ticker(sub: pd.DataFrame) -> pd.Series:
    """Return a pm_drift_5m series for all rows in sub (one ticker, sorted by logged_at)."""
    if len(sub) < 2:
        return pd.Series(np.nan, index=sub.index)

    # Build reference: shift logged_at forward by 5min so merge_asof maps each row
    # to the historical reading ~5min prior.
    ref = sub[["logged_at", "p_market"]].dropna(subset=["logged_at"]).copy()
    ref = ref.rename(columns={"p_market": "_pm_past"})
    # merge_asof key for target rows: logged_at - 5min
    target = sub[["logged_at", "p_market"]].dropna(subset=["logged_at"]).copy()
    target["_key"] = target["logged_at"] - LOOKBACK_TARGET

    # Sort both by their merge key
    ref_sorted = ref.sort_values("logged_at")
    target_sorted = target.sort_values("_key")

    merged = pd.merge_asof(
        target_sorted,
        ref_sorted,
        left_on="_key",
        right_on="logged_at",
        direction="nearest",
        tolerance=LOOKBACK_TOLERANCE,
        suffixes=("", "_ref"),
    )
    # Compute drift where we found a historical match
    drift = (merged["p_market"] - merged["_pm_past"]).round(6)
    drift.index = target_sorted.index
    # Restore original order
    drift = drift.reindex(sub.index)
    return drift


def main():
    print("Loading archive...")
    df = pd.read_csv(ARCHIVE_PATH, low_memory=False)
    orig_filled = df["pm_drift_5m"].notna().sum()
    print(f"  Rows: {len(df):,}  pm_drift_5m already filled: {orig_filled:,}")

    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")

    # Only process tickers that have at least one missing row
    tickers_missing = df.loc[df["pm_drift_5m"].isna(), "contract_ticker"].dropna().unique()
    print(f"  Tickers with gaps: {len(tickers_missing):,}")

    results = []
    for i, tk in enumerate(tickers_missing):
        if i % 500 == 0:
            print(f"  ... {i}/{len(tickers_missing)} tickers processed")
        mask = df["contract_ticker"] == tk
        sub = df.loc[mask].sort_values("logged_at")
        drift_series = _backfill_ticker(sub)
        # Only fill rows that were NaN
        nan_mask = df.loc[mask, "pm_drift_5m"].isna()
        fill_idx = nan_mask.index[nan_mask]
        results.append(drift_series.loc[fill_idx])

    if results:
        all_fills = pd.concat(results)
        df.loc[all_fills.index, "pm_drift_5m"] = all_fills

    final_filled = df["pm_drift_5m"].notna().sum()
    newly_filled = final_filled - orig_filled
    print(f"  Newly filled: {newly_filled:,}  Total: {final_filled:,} / {len(df):,} "
          f"({100*final_filled/len(df):.1f}%)")

    print("Writing archive...")
    df.to_csv(ARCHIVE_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
