#!/usr/bin/env python3
"""
calibrate_mu_drift_k.py — Find the optimal k multiplier for the rolling mu drift terms.

The current live formula applies mu6h/mu12h/mu24h raw (k=1.0 implicit).
This script calibrates a single k that multiplies the mu+regime_z component,
separately for the YES and NO formulas, using log-loss on the btc_scan_archive.

YES formula:
  z_drift_yes = k × [(mu6h + mu24h) × (tau/60) / sigma_tau
                      + regime_z × sqrt(tau/60)]
                + (composite_trend/5) × 0.15 × sqrt(tau/60)   [fixed, not scaled]

NO formula:
  z_drift_no  = k × [(mu6h + mu12h + mu24h) × (tau/60) / sigma_tau
                      + regime_z × sqrt(tau/60)]
                + norm.ppf(p_up_v2) × 1.14 × sqrt(tau/60)     [fixed, not scaled]

Also tests k=0 (no drift) as a baseline.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
ARCHIVE_PATH = ROOT / "results" / "btc_scan_archive.csv"
DATA_DIR     = ROOT / "data"

P_BAND   = (0.05, 0.95)   # exclude deep OTM/ITM noise
TAU_MIN  = 20.0
TAU_MAX  = 120.0
EPS      = 1e-7


def load_1h_data() -> pd.DataFrame:
    f = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
               key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def compute_mu_series(df1h: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(df1h["close"] / df1h["close"].shift(1))
    mu6  = lr.rolling(6,  min_periods=1).mean()
    mu12 = lr.rolling(12, min_periods=1).mean()
    mu24 = lr.rolling(24, min_periods=1).mean()
    ewm_mean = lr.ewm(span=12).mean()
    ewm_std  = lr.ewm(span=24).std()
    rz = np.clip(ewm_mean / ewm_std.replace(0, np.nan), -3.0, 3.0).fillna(0.0)
    return pd.DataFrame({"mu6": mu6, "mu12": mu12, "mu24": mu24, "regime_z": rz},
                        index=df1h.index)


def log_loss_yes(k, records):
    ll = 0.0
    for z_strike, sigma_tau, tau, mu6, mu24, rz, comp_trend, y in records:
        sq = math.sqrt(tau / 60.0)
        z_mu = (mu6 + mu24) * (tau / 60.0) / sigma_tau + rz * sq
        z_drift = k * z_mu + (comp_trend / 5.0) * 0.15 * sq
        p = float(np.clip(1.0 - norm.cdf(z_strike - z_drift), EPS, 1 - EPS))
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -ll / len(records)


def log_loss_no(k, records):
    ll = 0.0
    for z_strike, sigma_tau, tau, mu6, mu12, mu24, rz, p_up_v2, y in records:
        sq = math.sqrt(tau / 60.0)
        z_mu = (mu6 + mu12 + mu24) * (tau / 60.0) / sigma_tau + rz * sq
        pup_z = norm.ppf(float(np.clip(p_up_v2, 0.01, 0.99))) * 1.14 * sq if not math.isnan(p_up_v2) else 0.0
        z_drift = k * z_mu + pup_z
        p_yes = float(np.clip(1.0 - norm.cdf(z_strike - z_drift), EPS, 1 - EPS))
        p_no  = 1.0 - p_yes
        p_no  = float(np.clip(p_no, EPS, 1 - EPS))
        ll += y * math.log(p_no) + (1 - y) * math.log(1 - p_no)
    return -ll / len(records)


def main():
    print("Loading scan archive...")
    arc = pd.read_csv(ARCHIVE_PATH, low_memory=False)
    arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True)
    arc = arc[arc["resolved_yes"].notna()].copy()
    arc["resolved_yes"] = arc["resolved_yes"].astype(float)
    print(f"  Resolved rows: {len(arc):,}  ({arc['logged_at'].min().date()} → {arc['logged_at'].max().date()})")

    print("Loading 1h BTC data and computing mu series...")
    df1h = load_1h_data()
    mu_df = compute_mu_series(df1h)

    # Floor each scan row to the nearest completed 1h bar (use bar that CLOSED before decision)
    arc["bar_ts"] = arc["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
    arc = arc.join(mu_df.rename(columns={"mu6": "mu6h", "mu12": "mu12h",
                                          "mu24": "mu24h", "regime_z": "regime_z"}),
                   on="bar_ts", how="left")

    # Filter
    mask = (
        arc["vol_eff"].notna() & (arc["vol_eff"] > 0) &
        arc["mu6h"].notna() &
        arc["tau_minutes"].between(TAU_MIN, TAU_MAX) &
        arc["p_market"].between(*P_BAND)
    )
    arc = arc[mask].copy()
    print(f"  After filters: {len(arc):,} rows")

    arc["sigma_tau"] = arc["vol_eff"] * np.sqrt(arc["tau_minutes"])
    arc["z_strike"]  = np.log(arc["strike"] / arc["spot"]) / arc["sigma_tau"]
    arc["p_up_v2"]   = pd.to_numeric(arc["p_up_v2"], errors="coerce")
    arc["composite_trend"] = pd.to_numeric(arc["composite_trend"], errors="coerce").fillna(0)

    # YES records
    yes_recs = []
    for _, r in arc.iterrows():
        if math.isnan(r["sigma_tau"]) or r["sigma_tau"] <= 0:
            continue
        yes_recs.append((
            float(r["z_strike"]),
            float(r["sigma_tau"]),
            float(r["tau_minutes"]),
            float(r["mu6h"]),
            float(r["mu24h"]),
            float(r["regime_z"]),
            float(r["composite_trend"]),
            float(r["resolved_yes"]),
        ))

    # NO records (rows where p_up_v2 available)
    no_recs = []
    for _, r in arc.iterrows():
        if math.isnan(r["sigma_tau"]) or r["sigma_tau"] <= 0:
            continue
        pup = float(r["p_up_v2"]) if not math.isnan(r["p_up_v2"]) else float("nan")
        no_recs.append((
            float(r["z_strike"]),
            float(r["sigma_tau"]),
            float(r["tau_minutes"]),
            float(r["mu6h"]),
            float(r["mu12h"]),
            float(r["mu24h"]),
            float(r["regime_z"]),
            pup,
            float(r["resolved_yes"]),
        ))

    print(f"\n  YES records: {len(yes_recs):,}   NO records: {len(no_recs):,}")
    print(f"  YES-rate: {np.mean([r[7] for r in yes_recs]):.3f}")

    # ── YES calibration ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("YES DRIFT — calibrating k_yes")
    print("="*60)
    ll_yes_nodrift = log_loss_yes(0.0, yes_recs)
    ll_yes_current = log_loss_yes(1.0, yes_recs)
    res_yes = minimize_scalar(log_loss_yes, bounds=(-1.0, 3.0), method="bounded",
                              args=(yes_recs,))
    k_yes_opt = res_yes.x
    ll_yes_opt = res_yes.fun

    print(f"  k=0.0  (no drift)   ll={ll_yes_nodrift:.5f}")
    print(f"  k=1.0  (current)    ll={ll_yes_current:.5f}  Δ={ll_yes_current-ll_yes_nodrift:+.5f}")
    print(f"  k={k_yes_opt:.4f} (optimal)   ll={ll_yes_opt:.5f}  Δ={ll_yes_opt-ll_yes_nodrift:+.5f}")

    # Grid for visibility
    print("\n  k sweep (YES):")
    for k_t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        ll = log_loss_yes(k_t, yes_recs)
        marker = " ← current" if k_t == 1.0 else (" ← optimal" if abs(k_t - k_yes_opt) < 0.06 else "")
        print(f"    k={k_t:.1f}  ll={ll:.5f}{marker}")

    # ── NO calibration ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("NO DRIFT — calibrating k_no")
    print("="*60)
    ll_no_nodrift = log_loss_no(0.0, no_recs)
    ll_no_current = log_loss_no(1.0, no_recs)
    res_no = minimize_scalar(log_loss_no, bounds=(-1.0, 3.0), method="bounded",
                             args=(no_recs,))
    k_no_opt = res_no.x
    ll_no_opt = res_no.fun

    print(f"  k=0.0  (no drift)   ll={ll_no_nodrift:.5f}")
    print(f"  k=1.0  (current)    ll={ll_no_current:.5f}  Δ={ll_no_current-ll_no_nodrift:+.5f}")
    print(f"  k={k_no_opt:.4f} (optimal)   ll={ll_no_opt:.5f}  Δ={ll_no_opt-ll_no_nodrift:+.5f}")

    print("\n  k sweep (NO):")
    for k_t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        ll = log_loss_no(k_t, no_recs)
        marker = " ← current" if k_t == 1.0 else (" ← optimal" if abs(k_t - k_no_opt) < 0.06 else "")
        print(f"    k={k_t:.1f}  ll={ll:.5f}{marker}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  YES:  current k=1.0 → optimal k={k_yes_opt:.3f}  "
          f"(ll {ll_yes_current:.5f} → {ll_yes_opt:.5f}, "
          f"Δ={ll_yes_opt-ll_yes_current:+.5f} vs current)")
    print(f"  NO:   current k=1.0 → optimal k={k_no_opt:.3f}  "
          f"(ll {ll_no_current:.5f} → {ll_no_opt:.5f}, "
          f"Δ={ll_no_opt-ll_no_current:+.5f} vs current)")

    if ll_yes_nodrift <= ll_yes_opt + 0.0001:
        print("\n  NOTE: YES — no-drift baseline beats or ties optimal k. Mu drift may not help YES side.")
    if ll_no_nodrift <= ll_no_opt + 0.0001:
        print("\n  NOTE: NO  — no-drift baseline beats or ties optimal k. Mu drift may not help NO side.")


if __name__ == "__main__":
    main()
