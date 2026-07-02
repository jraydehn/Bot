"""
backfill_keltner.py — Backfill kc_pct_1h and kc_bo_1h to all paper_trades
and scan_archive CSVs.

Keltner Channel definition:
  EMA10 on 1h close
  ATR14 on 1h OHLC
  KC_upper = EMA10 + 1.5 * ATR14
  KC_lower = EMA10 - 1.5 * ATR14
  kc_pct_1h  = (close - KC_lower) / (KC_upper - KC_lower)  # channel position 0-1
  kc_bo_1h   = +1 if close > KC_upper, -1 if close < KC_lower, 0 otherwise

Timestamp alignment:
  Hourly archives (paper_trades / scan_archive): completed_bar_ts = logged_at.floor('1H') - 1H
  15m paper_trades: completed_bar_ts = close_time.floor('1H') - 1H

All parquet files span 2024-01-01 → 2026-06-25 (verified).
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"

# ── KC computation ─────────────────────────────────────────────────────────────

def compute_kc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute kc_pct and kc_bo on a 1h OHLC frame with DatetimeIndex."""
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    ema10  = close.ewm(span=10, adjust=False).mean()

    # True range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()

    kc_upper = ema10 + 1.5 * atr14
    kc_lower = ema10 - 1.5 * atr14
    width    = (kc_upper - kc_lower).replace(0, np.nan)

    kc_pct = ((close - kc_lower) / width).round(4)
    kc_bo  = np.where(close > kc_upper, 1,
             np.where(close < kc_lower, -1, 0)).astype(int)

    out = df[[]].copy()
    out["kc_pct_1h"] = kc_pct.values
    out["kc_bo_1h"]  = kc_bo
    return out


def load_kc_series(asset: str) -> pd.DataFrame:
    """Return kc_pct_1h / kc_bo_1h indexed by bar open time."""
    pattern = str(DATA_DIR / f"binanceus_{asset.upper()}USDT_1h_*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No 1h parquet found for {asset}: {pattern}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)

    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    kc = compute_kc(combined)
    kc.index = combined.index  # bar open time
    kc["bar_open"] = kc.index
    return kc.reset_index(drop=True)


# ── CSV join helper ────────────────────────────────────────────────────────────

def backfill_csv(csv_path: Path, ts_col: str, asset: str, kc: pd.DataFrame) -> int:
    """Add kc_pct_1h and kc_bo_1h to csv_path. Returns rows updated."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop existing KC columns so re-runs don't create _x/_y conflicts
    df = df.drop(columns=["kc_pct_1h", "kc_bo_1h"], errors="ignore")

    # Parse timestamp — use format='mixed' to handle both tz-aware strings
    # ("2026-04-15 20:43:49+00:00") and naive strings ("2026-06-25 10:30:00")
    # that the runner produces via strftime("%Y-%m-%d %H:%M:%S").
    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", utc=True, errors="coerce")

    # Compute completed bar timestamp: floor to hour then subtract 1h
    df["_bar_ts"] = df[ts_col].dt.floor("1h") - pd.Timedelta(hours=1)

    # Prepare KC lookup (keyed by bar_open = bar start time)
    kc_sorted = kc.sort_values("bar_open").copy()
    kc_sorted["bar_open"] = pd.to_datetime(kc_sorted["bar_open"], utc=True)

    # Sort by _bar_ts for merge_asof, tracking original row order
    df["_orig_order"] = range(len(df))

    # Separate NaT rows (can't join) from joinable rows
    nat_mask = df["_bar_ts"].isna()
    df_valid  = df[~nat_mask].sort_values("_bar_ts").copy()
    df_nat    = df[nat_mask].copy()

    merged_valid = pd.merge_asof(
        df_valid,
        kc_sorted[["bar_open", "kc_pct_1h", "kc_bo_1h"]],
        left_on="_bar_ts",
        right_on="bar_open",
        direction="backward",
    )

    # Re-combine NaT rows (kc columns will be NaN, which is correct)
    merged = pd.concat([merged_valid, df_nat], ignore_index=True)

    # Restore original order and drop helper columns
    merged = merged.sort_values("_orig_order").reset_index(drop=True)
    merged = merged.drop(columns=["_bar_ts", "_orig_order", "bar_open"], errors="ignore")

    merged.to_csv(csv_path, index=False)

    filled = merged["kc_pct_1h"].notna().sum()
    return int(filled)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    TARGETS = [
        # (csv_path,                              ts_col,       asset)
        (RES_DIR / "paper_trades.csv",            "logged_at",  "BTC"),
        (RES_DIR / "paper_trades_eth.csv",        "logged_at",  "ETH"),
        (RES_DIR / "paper_trades_sol.csv",        "logged_at",  "SOL"),
        (RES_DIR / "btc_scan_archive.csv",        "logged_at",  "BTC"),
        (RES_DIR / "eth_scan_archive.csv",        "logged_at",  "ETH"),
        (RES_DIR / "sol_scan_archive.csv",        "logged_at",  "SOL"),
        (RES_DIR / "paper_trades_btc15m.csv",     "close_time", "BTC"),
        (RES_DIR / "paper_trades_eth15m.csv",     "close_time", "ETH"),
        (RES_DIR / "paper_trades_sol15m.csv",     "close_time", "SOL"),
    ]

    # Cache KC series per asset
    kc_cache: dict[str, pd.DataFrame] = {}
    for asset in ("BTC", "ETH", "SOL"):
        print(f"Loading {asset} 1h KC series ...", end=" ", flush=True)
        kc_cache[asset] = load_kc_series(asset)
        print(f"{len(kc_cache[asset])} bars")

    for csv_path, ts_col, asset in TARGETS:
        if not csv_path.exists():
            print(f"  SKIP (not found): {csv_path.name}")
            continue
        rows_before = len(pd.read_csv(csv_path, usecols=[ts_col]))
        print(f"  {csv_path.name} ({rows_before:,} rows) ...", end=" ", flush=True)
        filled = backfill_csv(csv_path, ts_col, asset, kc_cache[asset])
        print(f"filled {filled:,} / {rows_before:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
