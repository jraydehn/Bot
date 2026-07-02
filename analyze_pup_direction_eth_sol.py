#!/usr/bin/env python3
"""
analyze_pup_direction_eth_sol.py

Tests whether composite_p_up predicts actual price direction for ETH and SOL contracts.
Analogous to the BTC analysis that found rho(p_up, actual_z) = 0.022 (useless).

For each resolved trade row:
  - sigma_tau = vol_eff * sqrt(tau_minutes)
  - actual_z = log(price_at_expiry / spot) / sigma_tau
  - actual_direction = 1 if resolved_yes else 0
  - p_up = composite_p_up

Computes:
  1. Spearman rho between p_up and actual_z
  2. Point-biserial correlation between p_up and resolved_yes
  3. Per-week rho (last 5 weeks)
  4. Calibration: p_up decile -> mean(actual_direction)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, '/Users/justindehn/Documents/ClaudeCode/kalshi_btc')

from evaluate_point import load_data

# ── config ─────────────────────────────────────────────────────────────────────

ETH_CSV = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_eth.csv'
SOL_CSV = '/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_sol.csv'

SEP  = "=" * 70
SEP2 = "-" * 70

# ── helpers ────────────────────────────────────────────────────────────────────

def lookup_price_at_expiry(close_ts_str: str, df_confirm: pd.DataFrame) -> float:
    """
    Look up the 1h candle open at close_ts (first price of the new hour ≈ settlement).
    If exact timestamp not found, use searchsorted.
    """
    # Parse the ISO timestamp
    ts = pd.Timestamp(close_ts_str).tz_localize('UTC') if pd.Timestamp(close_ts_str).tzinfo is None \
         else pd.Timestamp(close_ts_str).tz_convert('UTC')

    if ts in df_confirm.index:
        return df_confirm.loc[ts, 'open']

    # Use searchsorted to find nearest
    idx_arr = df_confirm.index
    pos = idx_arr.searchsorted(ts)
    if pos >= len(idx_arr):
        pos = len(idx_arr) - 1
    return df_confirm.iloc[pos]['open']


def deduplicate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    For duplicate contract_ticker rows in resolved trades,
    keep the one with the smallest tau_minutes (most representative single signal).
    """
    return df.sort_values('tau_minutes').drop_duplicates('contract_ticker', keep='first')


def prepare_resolved(csv_path: str, df_confirm: pd.DataFrame) -> pd.DataFrame:
    """
    Load paper trades CSV, filter to resolved rows, compute actual_z, return analysis frame.
    """
    raw = pd.read_csv(csv_path, low_memory=False)

    # Keep only resolved rows
    resolved = raw[raw['resolved_yes'].notna()].copy()
    resolved['resolved_yes'] = resolved['resolved_yes'].astype(bool)

    # Deduplicate by contract_ticker (keep first/smallest tau)
    resolved = deduplicate_trades(resolved)

    # Drop rows missing key fields
    needed = ['vol_eff', 'tau_minutes', 'composite_p_up', 'spot', 'close_ts']
    resolved = resolved.dropna(subset=needed)
    resolved = resolved[resolved['vol_eff'] > 0]
    resolved = resolved[resolved['tau_minutes'] > 0]

    print(f"  Resolved trades after dedup & dropna: {len(resolved)}")

    # Compute sigma_tau
    resolved['sigma_tau'] = resolved['vol_eff'] * np.sqrt(resolved['tau_minutes'])

    # Look up price at expiry from df_confirm
    prices = []
    missing = 0
    for _, row in resolved.iterrows():
        try:
            p = lookup_price_at_expiry(row['close_ts'], df_confirm)
            prices.append(p)
        except Exception:
            prices.append(np.nan)
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing} rows couldn't find expiry price")

    resolved = resolved.copy()
    resolved['price_at_expiry'] = prices

    # Drop rows where expiry price lookup failed or spot is 0
    resolved = resolved[resolved['price_at_expiry'].notna()]
    resolved = resolved[resolved['spot'] > 0]

    # Compute actual_z = log(price_at_expiry / spot) / sigma_tau
    resolved['log_ret'] = np.log(resolved['price_at_expiry'] / resolved['spot'])
    resolved['actual_z'] = resolved['log_ret'] / resolved['sigma_tau']

    # Binary direction
    resolved['actual_direction'] = resolved['resolved_yes'].astype(int)

    # Parse close_ts for weekly grouping
    resolved['close_ts_parsed'] = pd.to_datetime(resolved['close_ts'], utc=True)
    resolved['week'] = resolved['close_ts_parsed'].dt.isocalendar().week.astype(int)
    resolved['year'] = resolved['close_ts_parsed'].dt.isocalendar().year.astype(int)
    resolved['year_week'] = resolved['year'].astype(str) + '-W' + resolved['week'].astype(str).str.zfill(2)

    print(f"  Final analysis rows: {len(resolved)}")
    print(f"  Date range: {resolved['close_ts_parsed'].min().date()} to {resolved['close_ts_parsed'].max().date()}")
    print(f"  p_up range: [{resolved['composite_p_up'].min():.4f}, {resolved['composite_p_up'].max():.4f}]")
    print(f"  actual_z range: [{resolved['actual_z'].min():.3f}, {resolved['actual_z'].max():.3f}]")
    print(f"  resolved_yes rate: {resolved['actual_direction'].mean():.3f}")

    return resolved


def spearman(x: np.ndarray, y: np.ndarray):
    """Return (rho, p-value) for Spearman correlation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return stats.spearmanr(x[mask], y[mask])


def point_biserial(continuous: np.ndarray, binary: np.ndarray):
    """Return (r, p-value) for point-biserial correlation."""
    mask = np.isfinite(continuous) & np.isfinite(binary)
    if mask.sum() < 10:
        return np.nan, np.nan
    return stats.pointbiserialr(binary[mask], continuous[mask])


def per_week_rho(df: pd.DataFrame, n_weeks: int = 5) -> pd.DataFrame:
    """
    Compute Spearman rho(p_up, actual_z) per week for the last n_weeks weeks.
    """
    weeks = sorted(df['year_week'].unique())[-n_weeks:]
    rows = []
    for yw in weeks:
        sub = df[df['year_week'] == yw]
        n = len(sub)
        if n < 5:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearman(
                sub['composite_p_up'].values,
                sub['actual_z'].values
            )
        rows.append({'week': yw, 'n': n, 'rho': rho, 'p': pval})
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    For each p_up decile, show mean(actual_direction) and mean(p_up).
    If p_up is well calibrated, mean(p_up) ≈ mean(actual_direction) per bucket.
    """
    df2 = df[['composite_p_up', 'actual_direction']].dropna().copy()
    try:
        df2['decile'] = pd.qcut(df2['composite_p_up'], n_deciles, labels=False, duplicates='drop')
    except Exception:
        # Fewer unique values — use fewer bins
        n_bins = min(n_deciles, df2['composite_p_up'].nunique())
        df2['decile'] = pd.qcut(df2['composite_p_up'], n_bins, labels=False, duplicates='drop')

    grp = df2.groupby('decile').agg(
        n=('actual_direction', 'count'),
        mean_p_up=('composite_p_up', 'mean'),
        mean_resolved_yes=('actual_direction', 'mean'),
    ).reset_index()
    grp['calibration_error'] = grp['mean_p_up'] - grp['mean_resolved_yes']
    return grp


def run_asset(asset: str, csv_path: str):
    print(f"\n{SEP}")
    print(f"  ASSET: {asset}")
    print(SEP)

    # Load OHLCV data
    print(f"\n  Loading OHLCV for {asset}...")
    _, df_confirm, _ = load_data(asset)

    # Prepare resolved trade data
    print(f"\n  Preparing resolved trades...")
    df = prepare_resolved(csv_path, df_confirm)

    if len(df) < 20:
        print(f"  ERROR: insufficient resolved data ({len(df)} rows). Skipping.")
        return

    pup = df['composite_p_up'].values.astype(float)
    az  = df['actual_z'].values.astype(float)
    ad  = df['actual_direction'].values.astype(float)

    # ── 1. Spearman rho(p_up, actual_z) ────────────────────────────────────────
    rho_z, p_z = spearman(pup, az)
    n_valid = (np.isfinite(pup) & np.isfinite(az)).sum()

    print(f"\n  [1] Spearman rho(p_up, actual_z)")
    print(f"      rho = {rho_z:+.4f}  p = {p_z:.4f}  n = {n_valid}")
    print(f"      BTC baseline: rho = +0.022 (useless)")

    # ── 2. Point-biserial rho(p_up, resolved_yes) ──────────────────────────────
    r_dir, p_dir = point_biserial(pup, ad)
    n_dir = (np.isfinite(pup) & np.isfinite(ad)).sum()

    print(f"\n  [2] Point-biserial r(p_up, resolved_yes)")
    print(f"      r = {r_dir:+.4f}  p = {p_dir:.4f}  n = {n_dir}")

    # ── 3. Per-week rho ─────────────────────────────────────────────────────────
    week_df = per_week_rho(df, n_weeks=5)

    print(f"\n  [3] Per-week Spearman rho(p_up, actual_z) — last 5 weeks")
    print(f"      {'Week':<12} {'N':>5} {'rho':>8} {'p':>8}")
    print(f"      {'-'*12} {'-'*5} {'-'*8} {'-'*8}")
    for _, row in week_df.iterrows():
        rho_str = f"{row['rho']:+.4f}" if np.isfinite(row['rho']) else "    n/a"
        p_str   = f"{row['p']:.4f}"    if np.isfinite(row['rho']) else "    n/a"
        print(f"      {row['week']:<12} {int(row['n']):>5} {rho_str:>8} {p_str:>8}")

    print(f"\n      BTC reference: week 16→20 went from +0.22 to -0.252 (deteriorating)")

    # ── 4. Calibration table ────────────────────────────────────────────────────
    cal = calibration_table(df, n_deciles=10)

    print(f"\n  [4] Calibration: p_up decile → mean resolved_yes")
    print(f"      If well calibrated: mean_p_up ≈ mean_resolved_yes")
    print(f"      {'Decile':>7} {'N':>5} {'mean_p_up':>10} {'mean_yes':>9} {'cal_err':>8}")
    print(f"      {'-'*7} {'-'*5} {'-'*10} {'-'*9} {'-'*8}")
    for _, row in cal.iterrows():
        print(f"      {int(row['decile'])+1:>7} {int(row['n']):>5} "
              f"{row['mean_p_up']:>10.4f} {row['mean_resolved_yes']:>9.4f} "
              f"{row['calibration_error']:>+8.4f}")

    # Overall calibration error (MAE)
    cal_mae = cal['calibration_error'].abs().mean()
    print(f"\n      Mean absolute calibration error: {cal_mae:.4f}")

    # ── 5. Interpretation ───────────────────────────────────────────────────────
    print(f"\n  [5] Interpretation for {asset}")
    print(f"      rho(p_up, actual_z) = {rho_z:+.4f}")

    if abs(rho_z) < 0.05:
        quality = "USELESS (essentially zero correlation, like BTC)"
    elif abs(rho_z) < 0.10:
        quality = "WEAK (marginal signal, likely not actionable)"
    elif abs(rho_z) < 0.20:
        quality = "MODERATE (some directional content)"
    else:
        quality = "STRONG (meaningful directional signal)"

    print(f"      Signal quality: {quality}")

    trend = ""
    if len(week_df.dropna()) >= 3:
        rhos = week_df.dropna()['rho'].values
        if rhos[-1] < rhos[0]:
            trend = "DETERIORATING (most recent week weaker than earliest)"
        elif rhos[-1] > rhos[0]:
            trend = "IMPROVING (most recent week stronger than earliest)"
        else:
            trend = "STABLE"
        print(f"      Weekly trend: {trend}")

    if cal_mae > 0.15:
        print(f"      Calibration: POOR (MAE={cal_mae:.4f}) — p_up values are miscalibrated")
    elif cal_mae > 0.08:
        print(f"      Calibration: FAIR (MAE={cal_mae:.4f}) — moderate miscalibration")
    else:
        print(f"      Calibration: GOOD (MAE={cal_mae:.4f})")

    print(f"\n{SEP2}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  composite_p_up directional predictiveness: ETH & SOL")
    print("  BTC baseline: rho(p_up, actual_z) = +0.022, week 16→20: +0.22 → -0.252")
    print(SEP)

    run_asset('ETH', ETH_CSV)
    run_asset('SOL', SOL_CSV)

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)


if __name__ == '__main__':
    main()
