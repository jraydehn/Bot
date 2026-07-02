#!/usr/bin/env python3
"""
simulate_otm_pup_gate.py

Tests p_up_v2 as an OTM YES gate:
  - ITM YES (z_strike <= z_otm_thresh): take based on lognormal edge alone
  - OTM YES (z_strike > z_otm_thresh): only take if p_up_v2 >= pup_thresh

Sweeps:
  z_otm_thresh : [0.0, 0.1, 0.2, 0.3]
  pup_thresh   : [0.46, 0.48, 0.50, 0.52, 0.54, 0.56]

Flat $1000 bankroll, EDGE_MIN=0.04, KELLY_CAP=0.25, k=0 (pure lognormal).
NO bets are included unmodified (gate does not touch NO side).
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

Z_OTM_THRESHOLDS = [0.0, 0.1, 0.2, 0.3]
PUP_THRESHOLDS   = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56]


def load_pup_series() -> pd.Series:
    df = pd.read_pickle(FEAT_PKL)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    clf = pickle.load(open(MODEL_PATH, "rb"))["clf"]
    pup = clf.predict_proba(df[FEATURES].values.astype(float))[:, 1]
    return pd.Series(pup, index=df.index, name="p_up_v2")


def lookup_pup(ts: pd.Timestamp, series: pd.Series) -> float:
    idx = series.index.searchsorted(ts, side="right") - 1
    return float(series.iloc[idx]) if idx >= 0 else float("nan")


def p_yes_lognormal(z_strike: float) -> float:
    return float(np.clip(1.0 - norm.cdf(z_strike), 0.01, 0.99))


def p_no_lognormal(z_strike: float) -> float:
    return float(np.clip(norm.cdf(z_strike), 0.01, 0.99))


def kelly_f(p_model: float, p_market: float) -> float:
    if p_market <= 0 or p_market >= 1:
        return 0.0
    b = (1.0 - p_market) / p_market
    f = (b * p_model - (1.0 - p_model)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))


def pnl_yes(kf: float, p_market: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_market) / p_market if win else -stake


def pnl_no(kf: float, p_market_no: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_market_no) / p_market_no if win else -stake


def main():
    print("Loading p_up_v2 series...")
    pup_series = load_pup_series()
    print(f"  {len(pup_series):,} bars  {pup_series.index[0].date()} → {pup_series.index[-1].date()}")

    print("Loading paper_trades.csv...")
    rows = list(csv.DictReader(open(TRADES_CSV)))
    print(f"  {len(rows):,} rows")

    # ── parse all rows once ───────────────────────────────────────────────────
    records = []
    skipped = 0
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

            stored = r.get("p_up_v2","").strip()
            if stored:
                p_up = float(stored)
            else:
                ts   = pd.Timestamp(r["logged_at"], tz="UTC")
                p_up = lookup_pup(ts, pup_series)
            if math.isnan(p_up):
                p_up = float(r.get("composite_p_up","0.504") or "0.504")

            if side == "yes" and would_pnl and would_win:
                records.append({
                    "side": "yes", "z_str": z_str, "p_up": p_up,
                    "p_mkt": p_mkt, "win": would_win.lower() == "true",
                })
            elif side == "no" and would_win:
                records.append({
                    "side": "no", "z_str": z_str, "p_up": p_up,
                    "p_mkt": p_mkt, "win": would_win.lower() == "true",
                })
        except Exception:
            skipped += 1

    yes_recs = [r for r in records if r["side"] == "yes"]
    no_recs  = [r for r in records if r["side"] == "no"]
    print(f"  YES records: {len(yes_recs):,}   NO records: {len(no_recs):,}   Skipped: {skipped}")

    # ── baseline: k=0, no p_up gate ──────────────────────────────────────────
    def run_yes(recs, z_otm_thresh, pup_thresh):
        taken = wins = 0
        blocked_wins = blocked_losses = 0
        pnl = 0.0
        for rec in recs:
            pm    = p_yes_lognormal(rec["z_str"])
            edge  = pm - rec["p_mkt"]
            if edge < EDGE_MIN:
                continue
            kf = kelly_f(pm, rec["p_mkt"])
            if kf <= 0:
                continue
            # OTM gate
            is_otm = rec["z_str"] > z_otm_thresh
            if is_otm and rec["p_up"] < pup_thresh:
                if rec["win"]:
                    blocked_wins += 1
                else:
                    blocked_losses += 1
                continue
            taken += 1
            if rec["win"]: wins += 1
            pnl += pnl_yes(kf, rec["p_mkt"], rec["win"])
        return taken, wins, blocked_wins, blocked_losses, pnl

    def run_no(recs):
        taken = wins = 0
        pnl = 0.0
        for rec in recs:
            p_mkt_no = 1.0 - rec["p_mkt"]
            pm       = p_no_lognormal(rec["z_str"])
            edge     = pm - p_mkt_no
            if edge < EDGE_MIN:
                continue
            kf = kelly_f(pm, p_mkt_no)
            if kf <= 0:
                continue
            taken += 1
            if rec["win"]: wins += 1
            pnl += pnl_no(kf, p_mkt_no, rec["win"])
        return taken, wins, pnl

    # baseline (no gate)
    base_yes_taken, base_yes_wins, _, _, base_yes_pnl = run_yes(yes_recs, z_otm_thresh=-999, pup_thresh=0.0)
    no_taken, no_wins, no_pnl = run_no(no_recs)
    base_total = base_yes_pnl + no_pnl
    base_yes_wr = base_yes_wins / base_yes_taken * 100 if base_yes_taken else 0

    print()
    print("=" * 80)
    print("OTM YES GATE SWEEP  (k=0 pure lognormal, NO side unmodified)")
    print(f"Flat ${BANKROLL:.0f}  |  edge >= {EDGE_MIN:.2f}  |  Kelly cap {KELLY_CAP:.0%}")
    print("=" * 80)
    print(f"\nBASELINE (no gate):  YES bets={base_yes_taken}  WR={base_yes_wr:.1f}%  YES PnL={base_yes_pnl:+.2f}")
    print(f"                     NO  bets={no_taken}  WR={no_wins/no_taken*100:.1f}%  NO PnL={no_pnl:+.2f}  "
          f"TOTAL={base_total:+.2f}")

    print()
    print(f"{'z_otm':>6}  {'pup_min':>7}  {'YES bets':>9}  {'YES WR':>7}  {'blk_W':>6}  {'blk_L':>6}  "
          f"{'YES PnL':>9}  {'TOTAL':>9}  {'delta':>8}")
    print("-" * 85)

    best = None
    for z_thresh in Z_OTM_THRESHOLDS:
        for pup_thresh in PUP_THRESHOLDS:
            taken, wins, bw, bl, ypnl = run_yes(yes_recs, z_thresh, pup_thresh)
            wr = wins / taken * 100 if taken else 0.0
            total = ypnl + no_pnl
            delta = total - base_total
            flag = " ★" if delta > 0 else ""
            print(f"{z_thresh:>6.1f}  {pup_thresh:>7.2f}  {taken:>9d}  {wr:>6.1f}%  {bw:>6d}  {bl:>6d}  "
                  f"{ypnl:>+9.2f}  {total:>+9.2f}  {delta:>+8.2f}{flag}")
            if best is None or total > best[0]:
                best = (total, z_thresh, pup_thresh, taken, wr, ypnl, bw, bl)
        print()

    print("=" * 85)
    print(f"  Best: z_otm={best[1]}  pup>={best[2]}  "
          f"bets={best[3]}  WR={best[4]:.1f}%  YES PnL={best[5]:+.2f}  "
          f"TOTAL={best[0]:+.2f}  delta={best[0]-base_total:+.2f}")
    print(f"  Blocked: {best[6]} wins  {best[7]} losses")
    print(f"\n  Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
