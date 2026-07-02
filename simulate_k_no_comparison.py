#!/usr/bin/env python3
"""
simulate_k_no_comparison.py

Simulates PnL on historical paper_trades.csv under two k_no scenarios:
  - Baseline : K_DRIFT_NO_BTC = 0.15  (old)
  - New      : K_DRIFT_NO_BTC = 1.00  (calibrated 2026-05-20)

Uses:
  - paper_trades.csv (Apr 15 – May 20, all scanned rows)
  - Backfilled p_up_v2 from /tmp/calib_feat_dataset.pkl (Jan 2024 – May 2026)
  - Flat $1000 bankroll (non-compounding)

For each scanned row the script re-derives whether a NO bet would be
taken and at what Kelly, then records PnL using the stored would_win
outcome (already computed from actual BTC price movements).

YES bets are reported as-is (k_yes unchanged between scenarios).

Output: side-by-side YES / NO PnL table + net delta.
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

BANKROLL   = 1_000.0   # flat bankroll for every bet
EDGE_MIN   = 0.04      # minimum net edge to take a bet (mirrors live threshold)
KELLY_CAP  = 0.25      # max bet fraction of bankroll

FEATURES = [
    "stoch_k_4h","ema50_dist","rsi_4h","rsi_14","macd_hist_1h",
    "stoch_k","vwap_distance_pct","chg_4h_atr","bb_pct",
    "composite_trend","composite_rev","composite_p_up",
    "ema_stack_bias","ema_stretch_score","vwap_stretch_score",
    "confirmation_bias","stoch_bias","vpin_score","pm_drift_5m","rvol_1h",
]

K_OLD = 0.15
K_NEW = 1.00


# ── helpers ───────────────────────────────────────────────────────────────────

def kelly_no(p_model_no: float, p_market_no: float) -> float:
    """Standard Kelly fraction for a NO binary bet."""
    if p_market_no <= 0 or p_market_no >= 1:
        return 0.0
    b   = (1.0 - p_market_no) / p_market_no   # payout per unit staked
    f   = (b * p_model_no - (1.0 - p_model_no)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))


def pnl_no(kelly: float, p_market_no: float, win: bool) -> float:
    """Dollar PnL for a NO bet."""
    stake = kelly * BANKROLL
    if win:
        return stake * (1.0 - p_market_no) / p_market_no
    return -stake


def p_no_model(z_strike: float, p_up: float, k: float) -> float:
    p_up_c = float(np.clip(p_up, 0.01, 0.99))
    z_drift = norm.ppf(p_up_c) * k
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.01, 0.99))


# ── load p_up_v2 backfill ─────────────────────────────────────────────────────

def load_pup_v2_series() -> pd.Series:
    """Return Series of p_up_v2 indexed by 1h bar UTC timestamp."""
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


def lookup_pup_v2(ts_utc: pd.Timestamp, series: pd.Series) -> float:
    """Get p_up_v2 for the most recent 1h bar at or before ts_utc."""
    idx = series.index.searchsorted(ts_utc, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(series.iloc[idx])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pup_series = load_pup_v2_series()

    print("Loading paper_trades.csv...")
    with open(TRADES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows):,} rows  "
          f"{rows[0]['logged_at'][:10]} → {rows[-1]['logged_at'][:10]}")

    # ── accumulators ─────────────────────────────────────────────────────────
    # YES side (unchanged between scenarios)
    yes_bets_taken   = 0
    yes_pnl          = 0.0

    # NO side — old k
    no_old_taken     = 0
    no_old_wins      = 0
    no_old_pnl       = 0.0

    # NO side — new k
    no_new_taken     = 0
    no_new_wins      = 0
    no_new_pnl       = 0.0

    # tracking
    no_dropped       = 0   # old bet, new k drops it (edge gone)
    no_added         = 0   # new bet found by new k only
    no_resized_up    = 0   # edge increased → larger Kelly
    no_resized_down  = 0   # edge decreased → smaller Kelly

    skipped          = 0

    for r in rows:
        try:
            side       = r.get("side", "").strip().lower()
            p_mkt      = float(r["p_market"]) if r.get("p_market","").strip() else None
            spot       = float(r["spot"])      if r.get("spot","").strip()     else None
            strike     = float(r["strike"])    if r.get("strike","").strip()   else None
            tau        = float(r["tau_minutes"]) if r.get("tau_minutes","").strip() else None
            vol        = float(r["vol_60m_model"]) if r.get("vol_60m_model","").strip() else None
            would_win  = r.get("would_win","").strip()
            would_pnl  = r.get("would_pnl","").strip()

            if None in (p_mkt, spot, strike, tau, vol):
                skipped += 1
                continue
            if tau <= 0 or vol <= 0:
                skipped += 1
                continue

            sigma_tau  = vol * math.sqrt(tau)
            if sigma_tau <= 0:
                skipped += 1
                continue

            z_str = math.log(strike / spot) / sigma_tau

            # ── YES side ─────────────────────────────────────────────────────
            if side == "yes" and would_pnl:
                yes_bets_taken += 1
                yes_pnl        += float(would_pnl)
                continue

            # ── NO side ──────────────────────────────────────────────────────
            if side != "no":
                continue

            # Get p_up_v2 (stored or backfill)
            stored_pup = r.get("p_up_v2","").strip()
            if stored_pup:
                p_up = float(stored_pup)
            else:
                ts   = pd.Timestamp(r["logged_at"], tz="UTC")
                p_up = lookup_pup_v2(ts, pup_series)
            if math.isnan(p_up):
                p_up = float(r.get("composite_p_up","0.504") or "0.504")

            p_mkt_no  = 1.0 - p_mkt          # market NO probability
            p_no_old  = p_no_model(z_str, p_up, K_OLD)
            p_no_new  = p_no_model(z_str, p_up, K_NEW)

            edge_old  = p_no_old - p_mkt_no
            edge_new  = p_no_new - p_mkt_no

            k_old_frac = kelly_no(p_no_old, p_mkt_no) if edge_old >= EDGE_MIN else 0.0
            k_new_frac = kelly_no(p_no_new, p_mkt_no) if edge_new >= EDGE_MIN else 0.0

            # Only evaluate outcome rows (would_win populated)
            if not would_win:
                # count new bets found even without known outcome
                if k_old_frac == 0.0 and k_new_frac > 0.0:
                    no_added += 1
                elif k_old_frac > 0.0 and k_new_frac == 0.0:
                    no_dropped += 1
                continue

            win = (would_win.lower() == "true")

            # old scenario
            if k_old_frac > 0.0:
                no_old_taken += 1
                if win: no_old_wins += 1
                no_old_pnl += pnl_no(k_old_frac, p_mkt_no, win)

            # new scenario
            if k_new_frac > 0.0:
                no_new_taken += 1
                if win: no_new_wins += 1
                no_new_pnl += pnl_no(k_new_frac, p_mkt_no, win)

            # resize tracking
            if k_old_frac > 0.0 and k_new_frac > 0.0:
                if k_new_frac > k_old_frac + 0.001:
                    no_resized_up += 1
                elif k_new_frac < k_old_frac - 0.001:
                    no_resized_down += 1
            elif k_old_frac > 0.0 and k_new_frac == 0.0:
                no_dropped += 1
            elif k_old_frac == 0.0 and k_new_frac > 0.0:
                no_added += 1

        except Exception:
            skipped += 1
            continue

    # ── breakeven rates ───────────────────────────────────────────────────────
    def bkr(taken, pnl):
        if taken == 0:
            return float("nan")
        avg_stake = abs(pnl) / taken if pnl != 0 else 1.0
        return float("nan")   # simplified — not enough info without per-trade stakes

    # ── print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PnL SIMULATION: k_no = 0.15 vs 1.00")
    print(f"Flat bankroll ${BANKROLL:.0f}  |  edge threshold {EDGE_MIN:.2f}")
    print("=" * 60)

    print(f"\nYES SIDE  (k_yes unchanged — adaptive formula)")
    print(f"  Bets: {yes_bets_taken}   PnL: ${yes_pnl:+.2f}")

    print(f"\nNO SIDE  — k_no = {K_OLD} (baseline)")
    wr_old = no_old_wins/no_old_taken*100 if no_old_taken else 0
    print(f"  Bets taken : {no_old_taken}")
    print(f"  Win rate   : {wr_old:.1f}%")
    print(f"  PnL        : ${no_old_pnl:+.2f}")

    print(f"\nNO SIDE  — k_no = {K_NEW} (new calibrated)")
    wr_new = no_new_wins/no_new_taken*100 if no_new_taken else 0
    print(f"  Bets taken : {no_new_taken}")
    print(f"  Win rate   : {wr_new:.1f}%")
    print(f"  PnL        : ${no_new_pnl:+.2f}")

    print(f"\nNO SIDE CHANGES  (k_no = 0.15 → 1.00)")
    print(f"  Bets resized UP   : {no_resized_up}")
    print(f"  Bets resized DOWN : {no_resized_down}")
    print(f"  Bets DROPPED      : {no_dropped}  (edge lost under new k)")
    print(f"  Bets ADDED        : {no_added}   (new edge found under new k)")

    print(f"\n{'=' * 60}")
    print(f"NET DELTA  (new k vs old k, NO side only)")
    delta = no_new_pnl - no_old_pnl
    print(f"  NO PnL delta  : ${delta:+.2f}")
    print(f"  Total new PnL : ${yes_pnl + no_new_pnl:+.2f}  "
          f"(vs ${yes_pnl + no_old_pnl:+.2f} baseline)")
    print(f"\n  Skipped rows  : {skipped}")


if __name__ == "__main__":
    main()
