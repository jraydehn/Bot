#!/usr/bin/env python3
"""
simulate_pup_regime.py

Tests p_up_v2 as a rolling REGIME indicator for NO bets.

Regime logic:
  - Compute rolling mean of p_up_v2 over N prior 1h bars at trade time
  - If rolling_pup >= bull_thresh  → BULL regime → block NO bets
  - If rolling_pup <= bear_thresh  → BEAR regime → block YES bets (optional)
  - Otherwise neutral → trade normally

Sweeps:
  roll_window  : [2, 4, 6, 8, 12]
  bull_thresh  : [0.52, 0.53, 0.54, 0.55, 0.56]

YES bets: unmodified (k=0 lognormal, no gate)
NO bets : blocked when bull regime fires

Flat $1000, EDGE_MIN=0.04, KELLY_CAP=0.25, k=0 pure lognormal.
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

ROLL_WINDOWS  = [2, 4, 6, 8, 12]
BULL_THRESHOLDS = [0.52, 0.53, 0.54, 0.55, 0.56]


def load_pup_series() -> pd.Series:
    df = pd.read_pickle(FEAT_PKL)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    clf = pickle.load(open(MODEL_PATH, "rb"))["clf"]
    pup = clf.predict_proba(df[FEATURES].values.astype(float))[:, 1]
    return pd.Series(pup, index=df.index, name="p_up_v2")


def build_rolling_pup(series: pd.Series) -> pd.DataFrame:
    """Pre-compute rolling means for all window sizes."""
    df = pd.DataFrame({"pup": series})
    for w in ROLL_WINDOWS:
        df[f"roll_{w}"] = series.rolling(w).mean()
    return df


def lookup_rolling(ts: pd.Timestamp, roll_df: pd.DataFrame, window: int) -> float:
    col = f"roll_{window}"
    idx = roll_df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(roll_df[col].iloc[idx])


def p_yes_lognormal(z: float) -> float:
    return float(np.clip(1.0 - norm.cdf(z), 0.01, 0.99))


def p_no_lognormal(z: float) -> float:
    return float(np.clip(norm.cdf(z), 0.01, 0.99))


def kelly_f(p_model: float, p_market: float) -> float:
    if p_market <= 0 or p_market >= 1:
        return 0.0
    b = (1.0 - p_market) / p_market
    f = (b * p_model - (1.0 - p_model)) / b
    return float(np.clip(f, 0.0, KELLY_CAP))


def pnl_yes(kf: float, p_mkt: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_mkt) / p_mkt if win else -stake


def pnl_no(kf: float, p_mkt_no: float, win: bool) -> float:
    stake = kf * BANKROLL
    return stake * (1.0 - p_mkt_no) / p_mkt_no if win else -stake


def main():
    print("Loading p_up_v2 series and precomputing rolling means...")
    pup_series = load_pup_series()
    roll_df    = build_rolling_pup(pup_series)
    print(f"  {len(pup_series):,} bars  {pup_series.index[0].date()} → {pup_series.index[-1].date()}")

    print("Loading paper_trades.csv...")
    rows = list(csv.DictReader(open(TRADES_CSV)))
    print(f"  {len(rows):,} rows")

    # ── parse once ────────────────────────────────────────────────────────────
    records = []
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
            ts    = pd.Timestamp(r["logged_at"], tz="UTC")

            # instantaneous p_up_v2 (stored or backfilled)
            stored = r.get("p_up_v2","").strip()
            pup_inst = float(stored) if stored else float("nan")

            if side == "yes" and would_pnl and would_win:
                records.append({
                    "side": "yes", "z_str": z_str, "p_mkt": p_mkt,
                    "win": would_win.lower() == "true", "ts": ts,
                    "pup_inst": pup_inst,
                })
            elif side == "no" and would_win:
                records.append({
                    "side": "no", "z_str": z_str, "p_mkt": p_mkt,
                    "win": would_win.lower() == "true", "ts": ts,
                    "pup_inst": pup_inst,
                })
        except Exception:
            skipped += 1

    yes_recs = [r for r in records if r["side"] == "yes"]
    no_recs  = [r for r in records if r["side"] == "no"]
    print(f"  YES: {len(yes_recs):,}   NO: {len(no_recs):,}   Skipped: {skipped}")

    # ── precompute rolling pup for each NO record (all windows) ──────────────
    print("  Pre-fetching rolling p_up_v2 for each trade...")
    for rec in no_recs:
        rec["rolling"] = {w: lookup_rolling(rec["ts"], roll_df, w) for w in ROLL_WINDOWS}

    # ── fixed YES PnL (no gate on YES) ────────────────────────────────────────
    yes_taken = yes_wins = 0
    yes_pnl = 0.0
    for rec in yes_recs:
        pm   = p_yes_lognormal(rec["z_str"])
        edge = pm - rec["p_mkt"]
        if edge < EDGE_MIN: continue
        kf = kelly_f(pm, rec["p_mkt"])
        if kf <= 0: continue
        yes_taken += 1
        if rec["win"]: yes_wins += 1
        yes_pnl += pnl_yes(kf, rec["p_mkt"], rec["win"])

    # ── baseline NO (no regime gate) ──────────────────────────────────────────
    no_base_taken = no_base_wins = 0
    no_base_pnl = 0.0
    for rec in no_recs:
        p_mkt_no = 1.0 - rec["p_mkt"]
        pm   = p_no_lognormal(rec["z_str"])
        edge = pm - p_mkt_no
        if edge < EDGE_MIN: continue
        kf = kelly_f(pm, p_mkt_no)
        if kf <= 0: continue
        no_base_taken += 1
        if rec["win"]: no_base_wins += 1
        no_base_pnl += pnl_no(kf, p_mkt_no, rec["win"])

    base_total = yes_pnl + no_base_pnl

    print()
    print("=" * 82)
    print("ROLLING p_up_v2 REGIME GATE: block NO bets when bull regime fires")
    print(f"Flat ${BANKROLL:.0f}  |  edge >= {EDGE_MIN:.2f}  |  Kelly cap {KELLY_CAP:.0%}  |  k=0 lognormal")
    print("=" * 82)
    print(f"\nYES side (no gate):  bets={yes_taken}  WR={yes_wins/yes_taken*100:.1f}%  PnL={yes_pnl:+.2f}")
    print(f"NO  base (no gate):  bets={no_base_taken}  WR={no_base_wins/no_base_taken*100:.1f}%  "
          f"PnL={no_base_pnl:+.2f}  TOTAL={base_total:+.2f}")

    print()
    print(f"{'win':>5}  {'bull_t':>7}  {'NO bets':>8}  {'NO WR':>7}  "
          f"{'blk_W':>6}  {'blk_L':>6}  {'NO PnL':>9}  {'TOTAL':>9}  {'delta':>8}")
    print("-" * 82)

    results = []
    for w in ROLL_WINDOWS:
        for bt in BULL_THRESHOLDS:
            taken = wins = blocked_w = blocked_l = 0
            pnl = 0.0
            for rec in no_recs:
                p_mkt_no = 1.0 - rec["p_mkt"]
                pm   = p_no_lognormal(rec["z_str"])
                edge = pm - p_mkt_no
                if edge < EDGE_MIN: continue
                kf = kelly_f(pm, p_mkt_no)
                if kf <= 0: continue

                roll_pup = rec["rolling"][w]
                if not math.isnan(roll_pup) and roll_pup >= bt:
                    if rec["win"]: blocked_w += 1
                    else:          blocked_l += 1
                    continue

                taken += 1
                if rec["win"]: wins += 1
                pnl += pnl_no(kf, p_mkt_no, rec["win"])

            wr    = wins / taken * 100 if taken else 0.0
            total = yes_pnl + pnl
            delta = total - base_total
            flag  = " ★" if delta > 0 else ""
            results.append((total, w, bt, taken, wr, blocked_w, blocked_l, pnl))
            print(f"{w:>5}  {bt:>7.2f}  {taken:>8d}  {wr:>6.1f}%  "
                  f"{blocked_w:>6d}  {blocked_l:>6d}  {pnl:>+9.2f}  {total:>+9.2f}  {delta:>+8.2f}{flag}")
        print()

    best = max(results, key=lambda x: x[0])
    print("=" * 82)
    print(f"  Best: window={best[1]}h  bull_thresh={best[2]}  "
          f"NO bets={best[3]}  WR={best[4]:.1f}%")
    print(f"  Blocked: {best[5]} wins  {best[6]} losses")
    print(f"  NO PnL={best[7]:+.2f}  TOTAL={best[0]:+.2f}  delta={best[0]-base_total:+.2f}")
    print(f"\n  Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
