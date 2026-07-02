#!/usr/bin/env python3
"""
simulate_tau_drift.py

Compares three p_model approaches on paper_trades.csv (Apr 15 – May 20):

  CURRENT  YES: z_drift = Φ⁻¹(p_up_v2) × 1.40 × exp(-2.0 × max(0, z_strike))
           NO : z_drift = Φ⁻¹(p_up_v2) × 1.00

  PROPOSED YES: z_drift = Φ⁻¹(p_up_v2) × sqrt(tau / 60)
           NO : z_drift = Φ⁻¹(p_up_v2) × sqrt(tau / 60)

  BASELINE     z_drift = 0  (pure lognormal, no directional adjustment)

Flat $1000 bankroll, EDGE_MIN=0.04, KELLY_CAP=0.25.
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


# ── probability models ────────────────────────────────────────────────────────

def z_drift_current_yes(p_up: float, z_strike: float) -> float:
    k = 1.40 * math.exp(-2.0 * max(0.0, z_strike))
    return norm.ppf(float(np.clip(p_up, 0.01, 0.99))) * k

def z_drift_current_no(p_up: float) -> float:
    return norm.ppf(float(np.clip(p_up, 0.01, 0.99))) * 1.00

def z_drift_proposed(p_up: float, tau: float) -> float:
    return norm.ppf(float(np.clip(p_up, 0.01, 0.99))) * math.sqrt(tau / 60.0)

def z_drift_baseline() -> float:
    return 0.0

def p_yes(z_strike: float, z_drift: float) -> float:
    return float(np.clip(1.0 - norm.cdf(z_strike - z_drift), 0.01, 0.99))

def p_no(z_strike: float, z_drift: float) -> float:
    return float(np.clip(norm.cdf(z_strike - z_drift), 0.01, 0.99))

def kelly_yes(p_model: float, p_market: float) -> float:
    if p_market <= 0 or p_market >= 1:
        return 0.0
    b = (1.0 - p_market) / p_market
    f = (b * p_model - (1.0 - p_model)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))

def kelly_no_f(p_model_no: float, p_market_no: float) -> float:
    if p_market_no <= 0 or p_market_no >= 1:
        return 0.0
    b = (1.0 - p_market_no) / p_market_no
    f = (b * p_model_no - (1.0 - p_model_no)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))

def pnl_yes_f(kf: float, p_market: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_market) / p_market if win else -stake

def pnl_no_f(kf: float, p_market_no: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_market_no) / p_market_no if win else -stake


# ── load p_up_v2 ──────────────────────────────────────────────────────────────

def load_pup_series() -> pd.Series:
    print("Loading p_up_v2 backfill...")
    df = pd.read_pickle(FEAT_PKL)
    for c in FEATURES:
        if c not in df.columns: df[c] = np.nan
    clf = pickle.load(open(MODEL_PATH, "rb"))["clf"]
    pup = clf.predict_proba(df[FEATURES].values.astype(float))[:, 1]
    s = pd.Series(pup, index=df.index, name="p_up_v2")
    print(f"  {len(s):,} bars  {s.index[0].date()} → {s.index[-1].date()}")
    return s

def lookup_pup(ts: pd.Timestamp, series: pd.Series) -> float:
    idx = series.index.searchsorted(ts, side="right") - 1
    return float(series.iloc[idx]) if idx >= 0 else float("nan")


# ── accumulator ───────────────────────────────────────────────────────────────

class Acc:
    def __init__(self, name):
        self.name   = name
        self.taken  = 0
        self.wins   = 0
        self.pnl    = 0.0

    def record(self, kf, p_pay, win, side):
        if kf <= 0:
            return
        self.taken += 1
        if win:
            self.wins += 1
        if side == "yes":
            self.pnl += pnl_yes_f(kf, p_pay, win)
        else:
            self.pnl += pnl_no_f(kf, p_pay, win)

    def wr(self):
        return self.wins / self.taken * 100 if self.taken else 0.0

    def report(self, label):
        print(f"  {label:<8}  bets={self.taken:4d}  WR={self.wr():5.1f}%  PnL={self.pnl:+8.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pup_series = load_pup_series()

    print("Loading paper_trades.csv...")
    rows = list(csv.DictReader(open(TRADES_CSV)))
    print(f"  {len(rows):,} rows")

    # accumulators: [current, proposed, baseline] × [yes, no]
    cur_yes  = Acc("cur_yes");  cur_no  = Acc("cur_no")
    prop_yes = Acc("prop_yes"); prop_no = Acc("prop_no")
    base_yes = Acc("base_yes"); base_no = Acc("base_no")

    skipped = 0

    for r in rows:
        try:
            side      = r.get("side","").strip().lower()
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

            # p_up_v2
            stored = r.get("p_up_v2","").strip()
            if stored:
                p_up = float(stored)
            else:
                ts   = pd.Timestamp(r["logged_at"], tz="UTC")
                p_up = lookup_pup(ts, pup_series)
            if math.isnan(p_up):
                p_up = float(r.get("composite_p_up","0.504") or "0.504")

            # ── YES side ──────────────────────────────────────────────────────
            if side == "yes" and would_pnl and would_win:
                win = would_win.lower() == "true"
                p_mkt_yes = p_mkt

                # current
                zd = z_drift_current_yes(p_up, z_str)
                pm = p_yes(z_str, zd)
                cur_yes.record(kelly_yes(pm, p_mkt_yes) if pm - p_mkt_yes >= EDGE_MIN else 0.0, p_mkt_yes, win, "yes")

                # proposed
                zd = z_drift_proposed(p_up, tau)
                pm = p_yes(z_str, zd)
                prop_yes.record(kelly_yes(pm, p_mkt_yes) if pm - p_mkt_yes >= EDGE_MIN else 0.0, p_mkt_yes, win, "yes")

                # baseline
                pm = p_yes(z_str, 0.0)
                base_yes.record(kelly_yes(pm, p_mkt_yes) if pm - p_mkt_yes >= EDGE_MIN else 0.0, p_mkt_yes, win, "yes")

                continue

            # ── NO side ───────────────────────────────────────────────────────
            if side != "no" or not would_win:
                continue

            win       = would_win.lower() == "true"
            p_mkt_no  = 1.0 - p_mkt

            # current
            zd = z_drift_current_no(p_up)
            pm = p_no(z_str, zd)
            cur_no.record(kelly_no_f(pm, p_mkt_no) if pm - p_mkt_no >= EDGE_MIN else 0.0, p_mkt_no, win, "no")

            # proposed
            zd = z_drift_proposed(p_up, tau)
            pm = p_no(z_str, zd)
            prop_no.record(kelly_no_f(pm, p_mkt_no) if pm - p_mkt_no >= EDGE_MIN else 0.0, p_mkt_no, win, "no")

            # baseline
            pm = p_no(z_str, 0.0)
            base_no.record(kelly_no_f(pm, p_mkt_no) if pm - p_mkt_no >= EDGE_MIN else 0.0, p_mkt_no, win, "no")

        except Exception:
            skipped += 1
            continue

    # ── results ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("SIMULATION: current k vs tau-scaled vs baseline (no drift)")
    print(f"Flat ${BANKROLL:.0f}  |  edge ≥ {EDGE_MIN:.2f}  |  Kelly cap {KELLY_CAP:.0%}")
    print("=" * 62)

    for label, y_acc, n_acc in [
        ("CURRENT  (k_yes=1.40×exp, k_no=1.00)", cur_yes,  cur_no),
        ("PROPOSED (k = sqrt(tau/60) both sides)",  prop_yes, prop_no),
        ("BASELINE (k = 0, pure lognormal)",        base_yes, base_no),
    ]:
        total = y_acc.pnl + n_acc.pnl
        print(f"\n{label}")
        y_acc.report("YES")
        n_acc.report("NO")
        print(f"  {'TOTAL':<8}  bets={y_acc.taken+n_acc.taken:4d}               Total={total:+8.2f}")

    print(f"\n  Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
