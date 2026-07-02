"""
backfill_stochastic_signals.py

Backfills 8 shadow stochastic signals into scan archives and paper-trade CSVs:

  autocorr1_15   lag-1 autocorrelation of 1h log-returns (30-bar window)
  autocorr1_30   lag-1 autocorrelation of 1h log-returns (60-bar window)
  hurst_exponent R/S Hurst exponent (64-bar 1h window); H>0.5=trend, H<0.5=MR
  ou_theta       OU mean-reversion speed per hour (AR(1) on 48-bar 1h returns)
  ou_halflife    ln(2)/ou_theta in hours
  ou_mu_distance (current_return - OU mean) / OU vol  — z-score of current drift
  kalman_velocity Kalman-filtered 1h return trend (constant-velocity model)
  kalman_residual last observation minus Kalman-filtered value

All computations use only data available AT OR BEFORE each row's timestamp
(no lookahead). Uses the 1m parquet files already in data/ and resamples to 1h.

Usage:
  python3 backfill_stochastic_signals.py --asset BTC [--dry-run]
  python3 backfill_stochastic_signals.py --asset ETH [--dry-run]
  python3 backfill_stochastic_signals.py --asset SOL [--dry-run]
  python3 backfill_stochastic_signals.py --asset ALL [--dry-run]

Outputs (in-place unless --dry-run):
  results/{asset}_scan_archive_15m.csv   — adds 8 columns
  results/paper_trades_{asset}15m.csv    — adds 8 columns
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

NEW_COLS = [
    "autocorr1_15", "autocorr1_30",
    "hurst_exponent",
    "ou_theta", "ou_halflife", "ou_mu_distance",
    "kalman_velocity", "kalman_residual",
]

ASSET_MAP = {
    "BTC": {"ticker": "BTCUSDT", "archive": "btc_scan_archive_15m.csv",  "paper": "paper_trades_btc15m.csv"},
    "ETH": {"ticker": "ETHUSDT", "archive": "eth_scan_archive_15m.csv",  "paper": "paper_trades_eth15m.csv"},
    "SOL": {"ticker": "SOLUSDT", "archive": "sol_scan_archive_15m.csv",  "paper": "paper_trades_sol15m.csv"},
}

# Hourly runner CSVs (paper_trade_runner.py) — note: ou_halflife and ou_mu_distance are
# NOT added here; the hourly runner already has ou_halflife_min and ou_z_score equivalents.
HOURLY_COLS = ["autocorr1_15", "autocorr1_30", "hurst_exponent",
               "ou_theta", "kalman_velocity", "kalman_residual"]

HOURLY_MAP = {
    "BTC": {"ticker": "BTCUSDT",
            "files": ["btc_scan_archive.csv", "paper_trades.csv"]},
    "ETH": {"ticker": "ETHUSDT",
            "files": ["eth_scan_archive.csv", "paper_trades_eth.csv"]},
    "SOL": {"ticker": "SOLUSDT",
            "files": ["sol_scan_archive.csv", "paper_trades_sol.csv"]},
}

# ── Signal computation ────────────────────────────────────────────────────────

def _lag1_ac(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    x = arr[:-1] - arr[:-1].mean()
    y = arr[1:]  - arr[1:].mean()
    denom = float(np.sqrt((x**2).sum() * (y**2).sum()))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def compute_signals(lr: np.ndarray) -> dict:
    """
    Compute all 8 signals from an array of 1h log-returns available AT this moment.
    Returns a dict with NaN for any signal that can't be computed.
    """
    out = {c: float("nan") for c in NEW_COLS}
    if len(lr) < 10:
        return out

    # autocorr1
    if len(lr) >= 4:
        out["autocorr1_15"] = _lag1_ac(lr[-30:] if len(lr) >= 30 else lr)
        out["autocorr1_30"] = _lag1_ac(lr[-60:] if len(lr) >= 60 else lr)

    # Hurst exponent (R/S) on up to last 64 bars
    _wins = [8, 16, 32, 64]
    _h_lr = lr[-64:] if len(lr) >= 64 else lr
    _rs_pts = []
    for _w in _wins:
        if len(_h_lr) < _w:
            continue
        _seg = _h_lr[-_w:]
        _mean = _seg.mean()
        _dev  = np.cumsum(_seg - _mean)
        _r    = _dev.max() - _dev.min()
        _s    = _seg.std(ddof=1)
        if _s > 0 and _r > 0:
            _rs_pts.append((np.log(_w), np.log(_r / _s)))
    if len(_rs_pts) >= 2:
        _xs = np.array([p[0] for p in _rs_pts])
        _ys = np.array([p[1] for p in _rs_pts])
        _h  = float(np.polyfit(_xs, _ys, 1)[0])
        out["hurst_exponent"] = round(float(np.clip(_h, 0.0, 1.0)), 4)

    # OU parameters (AR(1) on last 48 bars)
    _ou_lr = lr[-48:] if len(lr) >= 48 else lr
    if len(_ou_lr) >= 10:
        _mu_ou = _ou_lr.mean()
        _y_c   = _ou_lr - _mu_ou
        _phi   = float(np.dot(_y_c[:-1], _y_c[1:]) /
                       (np.dot(_y_c[:-1], _y_c[:-1]) + 1e-12))
        _phi   = float(np.clip(_phi, -0.9999, 0.9999))
        _theta = float(-np.log(abs(_phi)))
        _theta = float(np.clip(_theta, 0.0, 10.0))
        out["ou_theta"]    = round(_theta, 6)
        out["ou_halflife"] = round(float(np.log(2) / _theta), 4) if _theta > 1e-6 else 999.0
        _ou_std = float(_y_c.std(ddof=1)) + 1e-10
        out["ou_mu_distance"] = round(float((lr[-1] - _mu_ou) / _ou_std), 4)

    # Kalman filter (constant-velocity model on last 48 bars)
    _kl = lr[-48:] if len(lr) >= 48 else lr
    if len(_kl) >= 5:
        _Q = np.array([[1e-5, 0.0], [0.0, 1e-5]])
        _R = float(np.var(_kl)) + 1e-10
        _x = np.array([_kl[0], 0.0])
        _P = np.eye(2) * 0.1
        _F = np.array([[1.0, 1.0], [0.0, 1.0]])
        _H = np.array([[1.0, 0.0]])
        for _obs in _kl:
            _x = _F @ _x
            _P = _F @ _P @ _F.T + _Q
            _K = _P @ _H.T / (float(_H @ _P @ _H.T) + _R)
            _innov = _obs - float(_H @ _x)
            _x = _x + _K.flatten() * _innov
            _P = (np.eye(2) - np.outer(_K.flatten(), _H)) @ _P
        out["kalman_velocity"] = round(float(_x[1]), 6)
        out["kalman_residual"] = round(float(_kl[-1] - float(_H @ _x)), 6)

    return out


# ── Price data loader ─────────────────────────────────────────────────────────

def load_price_data(ticker: str) -> pd.DataFrame:
    """
    Load the best available price data for a ticker.
    BTC: 1m parquet ends May 16 — fall back to 1h parquet (1970-01-01 series).
    ETH/SOL: 1m parquet is current; resample to 1h inside build_1h_lr.
    Returns a DataFrame with a 'close' column at 1h resolution, UTC-indexed.
    """
    # Try 1m first (ETH/SOL are current; BTC is stale)
    files_1m = sorted(DATA.glob(f"binanceus_{ticker}_1m_2024-01-01_*.parquet"))
    if files_1m:
        path = files_1m[-1]
        df = pd.read_parquet(path, columns=["close"])
        df.index = pd.to_datetime(df.index, utc=True)
        latest_1m = df.index.max()
        print(f"  1m: {path.name}  max={latest_1m.date()}", end="")

        # If 1m is more than 10 days stale, supplement with 1h parquet
        files_1h = sorted(DATA.glob(f"binanceus_{ticker}_1h_1970-01-01_*.parquet"))
        if files_1h:
            df_1h_raw = pd.read_parquet(files_1h[-1], columns=["close"])
            df_1h_raw.index = pd.to_datetime(df_1h_raw.index, utc=True)
            latest_1h = df_1h_raw.index.max()
            if (latest_1h - latest_1m).days > 10:
                # Resample 1m → 1h then append the 1h extension
                df_1h_from_1m = df["close"].resample("1h").last().dropna()
                extension = df_1h_raw.loc[df_1h_raw.index > latest_1m, "close"]
                combined = pd.concat([df_1h_from_1m, extension]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                print(f"  + 1h extension to {latest_1h.date()}")
                print(f"  → {len(combined):,} 1h bars total")
                return combined.to_frame(name="close")
            else:
                # Resample 1m → 1h in-place
                df_1h_from_1m = df["close"].resample("1h").last().dropna()
                print(f"  → {len(df_1h_from_1m):,} 1h bars (from 1m)")
                return df_1h_from_1m.to_frame(name="close")

        # No 1h supplement — just resample 1m
        df_1h_from_1m = df["close"].resample("1h").last().dropna()
        print(f"  → {len(df_1h_from_1m):,} 1h bars (from 1m)")
        return df_1h_from_1m.to_frame(name="close")

    # No 1m at all — use 1h directly
    files_1h = sorted(DATA.glob(f"binanceus_{ticker}_1h_1970-01-01_*.parquet"))
    if not files_1h:
        raise FileNotFoundError(f"No price data found for {ticker}")
    path = files_1h[-1]
    df = pd.read_parquet(path, columns=["close"])
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  1h only: {path.name}  → {len(df):,} bars")
    return df


def build_1h_lr(df_1h: pd.DataFrame, t_min, t_max) -> pd.Series:
    """Slice 1h close data to the archive window, compute log-returns."""
    mask = (df_1h.index >= t_min - pd.Timedelta(hours=72)) & (df_1h.index <= t_max)
    df_slice = df_1h.loc[mask, "close"].dropna()
    lr = np.log(df_slice / df_slice.shift(1)).dropna()
    return lr


# ── Backfill one CSV ──────────────────────────────────────────────────────────

def backfill_csv(csv_path: Path, df_1m: pd.DataFrame, dry_run: bool = False) -> None:
    if not csv_path.exists():
        print(f"  SKIP (not found): {csv_path.name}")
        return

    print(f"\n  Processing {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    rows={len(df):,}")

    # Determine timestamp column
    ts_col = "logged_at" if "logged_at" in df.columns else "close_ts"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    valid_mask = df[ts_col].notna()
    print(f"    valid timestamps: {valid_mask.sum():,}")

    t_min = df.loc[valid_mask, ts_col].min()
    t_max = df.loc[valid_mask, ts_col].max()

    # Build 1h log-return series covering the full archive range
    lr_series = build_1h_lr(df_1m, t_min, t_max)
    lr_index  = lr_series.index  # UTC-aware hourly timestamps

    # Add new columns if missing
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    # Count already-filled rows
    already = df[NEW_COLS[0]].notna().sum()
    print(f"    already filled: {already:,} rows")

    filled = 0
    for idx in df.index[valid_mask]:
        row_ts = df.at[idx, ts_col]
        # Use 1h bars completed BEFORE this row's timestamp
        bar_end = row_ts.floor("1h")
        avail = lr_series[lr_index < bar_end].values
        if len(avail) < 10:
            continue
        signals = compute_signals(avail)
        for c, v in signals.items():
            if pd.isna(df.at[idx, c]):  # don't overwrite existing values
                df.at[idx, c] = v
        filled += 1

    print(f"    filled: {filled:,} rows")

    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"    saved → {csv_path.name}")
    else:
        print(f"    dry-run: no write")


def backfill_hourly_csv(csv_path: Path, df_1h: pd.DataFrame, dry_run: bool = False) -> None:
    """Backfill HOURLY_COLS into an hourly-runner CSV using completed 1h bars only."""
    if not csv_path.exists():
        print(f"  SKIP (not found): {csv_path.name}")
        return

    print(f"\n  Processing {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    rows={len(df):,}")

    # Determine timestamp column (hourly runner uses 'logged_at' or 'close_ts')
    ts_col = "logged_at" if "logged_at" in df.columns else "close_ts"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    valid_mask = df[ts_col].notna()
    print(f"    valid timestamps: {valid_mask.sum():,}")

    if valid_mask.sum() == 0:
        print("    no valid timestamps — skipping")
        return

    t_min = df.loc[valid_mask, ts_col].min()
    t_max = df.loc[valid_mask, ts_col].max()
    lr_series = build_1h_lr(df_1h, t_min, t_max)
    lr_index  = lr_series.index

    for c in HOURLY_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    already = df[HOURLY_COLS[0]].notna().sum()
    print(f"    already filled: {already:,} rows")

    filled = 0
    for idx in df.index[valid_mask]:
        row_ts  = df.at[idx, ts_col]
        bar_end = row_ts.floor("1h")
        avail   = lr_series[lr_index < bar_end].values
        if len(avail) < 10:
            continue
        signals = compute_signals(avail)
        # Only write the subset of signals relevant for hourly CSVs
        for c in HOURLY_COLS:
            if pd.isna(df.at[idx, c]):
                df.at[idx, c] = signals.get(c, float("nan"))
        filled += 1

    print(f"    filled: {filled:,} rows")

    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"    saved → {csv_path.name}")
    else:
        print(f"    dry-run: no write")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["BTC", "ETH", "SOL", "ALL"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hourly", action="store_true", help="Backfill hourly runner CSVs instead of 15m")
    args = parser.parse_args()

    assets = list(ASSET_MAP.keys()) if args.asset == "ALL" else [args.asset]

    for asset in assets:
        print(f"\n{'='*60}")
        print(f"  {asset}{'  [HOURLY]' if args.hourly else '  [15m]'}")
        print(f"{'='*60}")

        if args.hourly:
            cfg_h = HOURLY_MAP[asset]
            df_1h = load_price_data(cfg_h["ticker"])
            for fname in cfg_h["files"]:
                backfill_hourly_csv(RESULTS / fname, df_1h, dry_run=args.dry_run)
        else:
            cfg = ASSET_MAP[asset]
            df_1m = load_price_data(cfg["ticker"])
            for fname in [cfg["archive"], cfg["paper"]]:
                backfill_csv(RESULTS / fname, df_1m, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
