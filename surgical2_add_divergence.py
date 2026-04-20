#!/usr/bin/env python3
"""
surgical2_add_divergence.py — Experiment 2: ADDITIVE divergence vote.

Keeps all 14 existing votes (no removals — Experiment 1 showed removal hurts).
Adds a cross-timeframe stochastic divergence vote to the REVERSION score:

  Let div = stoch_1h - stoch_4h (range -100 to +100)
    div > +40   → rev -= 2   (short-term strongly overbought vs 4h → revert down)
    div > +20   → rev -= 1
    div < -40   → rev += 2   (short-term strongly oversold vs 4h → revert up)
    div < -20   → rev += 1

Rebuilds per-asset calibration on train (2025), backtests on test (2026-03-16 on).
"""

import math, sys, glob, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import (
    _trend_votes, _reversion_votes, _stoch_k,
    ASSET_BASELINES, SMOOTHING_N
)
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from surgical1_drop_dead import (
    load_asset, load_archive, p_model_from_p_up, evaluate_row, kelly_bet, trade_pnl,
    ASSET_PARAMS, DRIFT_MULT, TEST_START, TRAIN_START, TRAIN_END, BANKROLL_0,
)

OUT_DIR = Path(__file__).parent / "reform_results"


def compute_divergence_vote(d_1h, d_4h):
    """Added rev vote from (stoch_1h - stoch_4h)."""
    stoch_1h = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    stoch_4h = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(d_1h.index, method="ffill")
    div = stoch_1h - stoch_4h
    # Sign: positive div → overbought short-term → mean revert down → rev -= N
    vote = pd.Series(0, index=d_1h.index)
    vote -= 2 * (div > 40).astype(int)
    vote -= 1 * ((div > 20) & (div <= 40)).astype(int)
    vote += 1 * ((div < -20) & (div >= -40)).astype(int)
    vote += 2 * (div < -40).astype(int)
    return vote.fillna(0).astype(int)


def compute_scores_additive(d_1m, d_15m, d_1h, d_4h):
    """Existing trend + existing rev + divergence rev vote."""
    ts_1h = d_1h.index
    trend_4h = _trend_votes(d_4h["close"], d_4h["high"], d_4h["low"], d_4h["volume"])
    trend_1h = trend_4h.reindex(ts_1h, method="ffill").fillna(0).astype(int)
    rev = _reversion_votes(d_1h["close"], d_1h["high"], d_1h["low"],
                            d_15m["close"], d_15m["high"], d_15m["low"],
                            d_1m["close"], d_1m["volume"], ts_1h).fillna(0).astype(int)
    div_vote = compute_divergence_vote(d_1h, d_4h)
    rev_additive = (rev + div_vote).astype(int)
    return trend_1h, rev_additive


def build_calibration(trend, rev, close_1h, asset, train_start, train_end):
    baseline = ASSET_BASELINES.get(asset, 0.504)
    next_up = (close_1h.shift(-1) > close_1h).astype(int)
    df = pd.DataFrame({"trend": trend, "rev": rev, "next_up": next_up}, index=close_1h.index)
    df = df[(df.index >= train_start) & (df.index < train_end)].dropna()
    df["tb"] = df["trend"].clip(-3, 3).astype(int)
    # Additive rev has wider range — allow bucket ±10 to accommodate
    df["rb"] = df["rev"].clip(-10, 10).astype(int)
    cal = {}
    for tb in range(-3, 4):
        for rb in range(-10, 11):
            cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
            n = len(cell)
            if n >= 10:
                up = cell["next_up"].mean()
                w = min(1.0, n / SMOOTHING_N)
                p_cal = w * up + (1 - w) * baseline
                cal[(tb, rb)] = round(float(p_cal), 4)
    return cal, len(df), baseline


def lookup_p_up_additive(trend_score, rev_score, calibration, baseline):
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score, -10, 10))
    if (tb, rb) in calibration:
        return calibration[(tb, rb)]
    return float(np.clip(baseline + 0.006 * rb + 0.003 * tb, 0.25, 0.80))


def run_bt(asset, scans_df, p_up_override=None):
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None
        for _, row in group.iterrows():
            if p_up_override is not None:
                dt_floor = pd.Timestamp(dt).floor("1h")
                p_up_v = p_up_override.get(dt_floor, np.nan)
                if not np.isfinite(p_up_v): continue
            else:
                p_up_v = row["composite_p_up"]
            p_mv = p_model_from_p_up(row["spot"], row["strike"], row["vol_eff"],
                                      row["tau_minutes"], p_up_v, asset)
            cand = evaluate_row(row, p_mv, p_up_v, params)
            if cand is None: continue
            if best is None or cand["net"] > best["net"]:
                best = cand; best_row = row
        if best is None: continue
        bet = kelly_bet(best["pm_use"], best["pm"], best["side"], bankroll)
        if bet <= 0: continue
        actual_yes = int(best_row["resolved_yes"])
        won = (actual_yes == 1 and best["side"] == "yes") or (actual_yes == 0 and best["side"] == "no")
        pnl = trade_pnl(bet, best["side"], best["pm"], won)
        bankroll = max(1.0, bankroll + pnl)
        pnls.append(pnl); wins.append(won); sides.append(best["side"])
    if not pnls:
        return dict(n=0, wr=0, pnl=0, max_streak=0, n_yes=0, n_no=0)
    n = len(pnls); wins_n = sum(wins)
    streak = 0; ms = 0
    for w in wins:
        if not w: streak += 1; ms = max(ms, streak)
        else: streak = 0
    return dict(n=n, wr=wins_n/n, pnl=sum(pnls), max_streak=ms,
                n_yes=sum(1 for s in sides if s=="yes"),
                n_no=sum(1 for s in sides if s=="no"))


def main():
    print(f"\n{'='*78}\n  SURGICAL EXPERIMENT 2 — ADDITIVE stoch 1h-vs-4h divergence rev vote\n{'='*78}", flush=True)

    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        print(f"\n  [{asset}] loading data...", flush=True)
        d_1m, d_15m, d_1h, d_4h = load_asset(sym)
        trend_s, rev_s = compute_scores_additive(d_1m, d_15m, d_1h, d_4h)
        cal, n_train, baseline = build_calibration(trend_s, rev_s, d_1h["close"], asset, TRAIN_START, TRAIN_END)
        print(f"  Built additive calibration: {len(cal)} cells from {n_train:,} train hours (baseline {baseline:.3f})", flush=True)

        # Also show rev distribution to sanity check
        rev_test = rev_s[(rev_s.index >= TEST_START)]
        print(f"  Rev distribution on test: min={rev_test.min()} median={rev_test.median()} max={rev_test.max()}", flush=True)

        test_mask = d_1h.index >= TEST_START
        p_up_series = pd.Series(
            [lookup_p_up_additive(int(t), int(r), cal, baseline) for t, r in zip(trend_s[test_mask].values, rev_s[test_mask].values)],
            index=d_1h.index[test_mask]
        )
        p_up_map = dict(zip(p_up_series.index, p_up_series.values))

        scans = load_archive(asset)
        if scans.empty: print("  no scans"); continue
        print(f"  Test: {len(scans):,} scans across {scans['decision_time'].nunique():,} hours", flush=True)

        r_base = run_bt(asset, scans, p_up_override=None)
        r_surg = run_bt(asset, scans, p_up_override=p_up_map)

        def fmt(r):
            return f"n={r['n']:4d} WR={r['wr']:.1%} PnL=${r['pnl']:+.2f} streak={r['max_streak']:2d} ({r['n_yes']}y/{r['n_no']}n)"
        print(f"\n  BASELINE : {fmt(r_base)}", flush=True)
        print(f"  ADDITIVE : {fmt(r_surg)}", flush=True)
        dpnl = r_surg["pnl"] - r_base["pnl"]
        dwr = (r_surg["wr"] - r_base["wr"]) if r_surg["n"] and r_base["n"] else 0
        print(f"  Δ PnL=${dpnl:+.2f}  Δ WR={dwr:+.1%}", flush=True)

        cal_out = OUT_DIR / f"surgical2_calibration_{asset}.json"
        with open(cal_out, "w") as f:
            json.dump({f"{k[0]},{k[1]}": v for k, v in cal.items()}, f, indent=2)
        print(f"  Saved → {cal_out}", flush=True)


if __name__ == "__main__":
    main()
