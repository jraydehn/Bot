#!/usr/bin/env python3
"""
calibrate_k_no_pnl.py

Sweeps K_DRIFT_NO_BTC over a grid and evaluates actual dollar PnL against
live paper_trades.csv data (real Kalshi prices, real outcomes).

This is a PnL-optimized calibration — the correct objective for a betting
system — as opposed to log-loss on synthetic contracts.

Uses:
  - paper_trades.csv (Apr 15 – May 20, all scanned rows with resolved outcomes)
  - Backfilled p_up_v2 from /tmp/calib_feat_dataset.pkl
  - Flat $1000 bankroll, EDGE_MIN=0.04, KELLY_CAP=0.25

Output: table of (k, bets_taken, win_rate, NO_pnl, total_pnl) + optimal k.
"""

import csv
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
TRADES_CSV = ROOT / "results" / "paper_trades.csv"
FEAT_PKL   = Path("/tmp/calib_feat_dataset.pkl")
MODEL_PATH = ROOT / "reform_results" / "btc_p_up_v2.pkl"

BANKROLL  = 1_000.0
EDGE_MIN  = 0.04
KELLY_CAP = 0.25

FEATURES = [
    "stoch_k_4h","ema50_dist","rsi_4h","rsi_14","macd_hist_1h",
    "stoch_k","vwap_distance_pct","chg_4h_atr","bb_pct",
    "composite_trend","composite_rev","composite_p_up",
    "ema_stack_bias","ema_stretch_score","vwap_stretch_score",
    "confirmation_bias","stoch_bias","vpin_score","pm_drift_5m","rvol_1h",
]

K_GRID = [round(x * 0.05, 2) for x in range(0, 41)]  # 0.00 → 2.00 step 0.05


# ── helpers ───────────────────────────────────────────────────────────────────

def p_no_model(z_strike: float, p_up: float, k: float) -> float:
    p_up_c  = float(np.clip(p_up, 0.01, 0.99))
    z_drift = norm.ppf(p_up_c) * k
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.01, 0.99))


def kelly_no(p_model_no: float, p_market_no: float) -> float:
    if p_market_no <= 0 or p_market_no >= 1:
        return 0.0
    b = (1.0 - p_market_no) / p_market_no
    f = (b * p_model_no - (1.0 - p_model_no)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))


def pnl_no(kelly: float, p_market_no: float, win: bool) -> float:
    stake = kelly * BANKROLL
    if win:
        return stake * (1.0 - p_market_no) / p_market_no
    return -stake


# ── load p_up_v2 backfill ─────────────────────────────────────────────────────

def load_pup_series() -> pd.Series:
    print("Loading p_up_v2 backfill...")
    df = pd.read_pickle(FEAT_PKL)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)["clf"]
    X   = df[FEATURES].values.astype(float)
    pup = clf.predict_proba(X)[:, 1]
    s   = pd.Series(pup, index=df.index, name="p_up_v2")
    print(f"  {len(s):,} bars  {s.index[0].date()} → {s.index[-1].date()}")
    return s


def lookup_pup(ts_utc: pd.Timestamp, series: pd.Series) -> float:
    idx = series.index.searchsorted(ts_utc, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(series.iloc[idx])


# ── build resolved NO records from paper_trades ───────────────────────────────

def load_no_records(pup_series: pd.Series) -> tuple:
    """Return (no_records, yes_pnl) where no_records is list of dicts."""
    print("Loading paper_trades.csv...")
    with open(TRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows):,} rows")

    yes_pnl    = 0.0
    no_records = []
    skipped    = 0

    for r in rows:
        try:
            side      = r.get("side", "").strip().lower()
            p_mkt     = float(r["p_market"]) if r.get("p_market","").strip() else None
            spot      = float(r["spot"])      if r.get("spot","").strip()     else None
            strike    = float(r["strike"])    if r.get("strike","").strip()   else None
            tau       = float(r["tau_minutes"]) if r.get("tau_minutes","").strip() else None
            vol       = float(r["vol_60m_model"]) if r.get("vol_60m_model","").strip() else None
            would_win = r.get("would_win","").strip()
            would_pnl = r.get("would_pnl","").strip()

            if None in (p_mkt, spot, strike, tau, vol):
                skipped += 1; continue
            if tau <= 0 or vol <= 0:
                skipped += 1; continue

            sigma_tau = vol * math.sqrt(tau)
            if sigma_tau <= 0:
                skipped += 1; continue

            z_str = math.log(strike / spot) / sigma_tau

            if side == "yes" and would_pnl:
                yes_pnl += float(would_pnl)
                continue

            if side != "no" or not would_win:
                continue

            stored_pup = r.get("p_up_v2","").strip()
            if stored_pup:
                p_up = float(stored_pup)
            else:
                ts   = pd.Timestamp(r["logged_at"], tz="UTC")
                p_up = lookup_pup(ts, pup_series)
            if math.isnan(p_up):
                p_up = float(r.get("composite_p_up","0.504") or "0.504")

            no_records.append({
                "z_str"     : z_str,
                "p_up"      : p_up,
                "p_mkt_no"  : 1.0 - p_mkt,
                "win"       : (would_win.lower() == "true"),
            })

        except Exception:
            skipped += 1
            continue

    print(f"  YES bets: {sum(1 for r in rows if r.get('side','').strip().lower()=='yes' and r.get('would_pnl','').strip()):,}  "
          f"YES PnL: ${yes_pnl:+.2f}")
    print(f"  Resolved NO records: {len(no_records):,}   Skipped: {skipped:,}")
    return no_records, yes_pnl


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(no_records: list, yes_pnl: float) -> None:
    print(f"\nSweeping k_no from {K_GRID[0]} to {K_GRID[-1]} (step 0.05)...")
    print()
    print(f"{'k':>6}  {'bets':>6}  {'WR%':>6}  {'NO PnL':>10}  {'Total PnL':>11}  {'Δ vs k=0':>10}")
    print("-" * 65)

    results = []
    base_no_pnl = None

    for k in K_GRID:
        taken = 0
        wins  = 0
        no_pnl = 0.0

        for rec in no_records:
            p_no  = p_no_model(rec["z_str"], rec["p_up"], k)
            edge  = p_no - rec["p_mkt_no"]
            if edge < EDGE_MIN:
                continue
            kf = kelly_no(p_no, rec["p_mkt_no"])
            if kf <= 0:
                continue
            taken  += 1
            if rec["win"]:
                wins += 1
            no_pnl += pnl_no(kf, rec["p_mkt_no"], rec["win"])

        wr        = wins / taken * 100 if taken else 0.0
        total_pnl = yes_pnl + no_pnl
        if k == 0.0:
            base_no_pnl = no_pnl
        delta = no_pnl - (base_no_pnl or 0.0)

        results.append((k, taken, wr, no_pnl, total_pnl))
        print(f"{k:6.2f}  {taken:6d}  {wr:6.1f}%  {no_pnl:+10.2f}  {total_pnl:+11.2f}  {delta:+10.2f}")

    # ── find optimum ──────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x[4])   # maximise total PnL
    best_no = max(results, key=lambda x: x[3])

    print()
    print("=" * 65)
    print(f"★ Best total PnL  : k={best[0]:.2f}   "
          f"bets={best[1]}  WR={best[2]:.1f}%  "
          f"NO PnL={best[3]:+.2f}  Total={best[4]:+.2f}")
    print(f"★ Best NO PnL only: k={best_no[0]:.2f}   "
          f"bets={best_no[1]}  WR={best_no[2]:.1f}%  "
          f"NO PnL={best_no[3]:+.2f}  Total={best_no[4]:+.2f}")
    print(f"\n  Current K_DRIFT_NO_BTC = 1.00  "
          f"(log-loss-calibrated; PnL says k={best[0]:.2f})")
    print(f"  YES PnL (fixed): ${yes_pnl:+.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pup_series         = load_pup_series()
    no_records, yes_pnl = load_no_records(pup_series)
    sweep(no_records, yes_pnl)


if __name__ == "__main__":
    main()
