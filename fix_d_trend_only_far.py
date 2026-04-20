#!/usr/bin/env python3
"""
fix_d_trend_only_far.py — Fix D: drop `rev` from p_up calculation at large offsets.

At |offset| ≤ 0.5% : use current lookup_p_up(trend, rev)  — rev helps for near-ATM
At |offset| > 0.5% : use lookup_trend_only(trend)          — rev is inverse-predictive here

lookup_trend_only: 1D table built on train data (2025), mapping trend bucket
(-3 to +3) → P(close_next > close_now).

Validates on:
  1. Strike-hit AUC at various offsets (target: beat current composite_p_up)
  2. PnL backtest on test window (Mar 16, 2026 → present) through full gate stack
"""

import math, sys, glob, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import compute_scores, lookup_p_up, ASSET_BASELINES, SMOOTHING_N
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from surgical1_drop_dead import (
    load_asset, load_archive, p_model_from_p_up, evaluate_row, kelly_bet, trade_pnl,
    ASSET_PARAMS, DRIFT_MULT, TEST_START, TRAIN_START, TRAIN_END, BANKROLL_0,
)

OUT_DIR = Path(__file__).parent / "reform_results"

FAR_OFFSET_THRESHOLD = 0.005  # |offset_pct| > 0.5% → use trend-only lookup


def build_trend_only_table(trend, close_1h, asset, train_start, train_end):
    """1D table: trend bucket → P(next-hour close > this-hour close). Smoothed toward baseline."""
    baseline = ASSET_BASELINES.get(asset, 0.504)
    next_up = (close_1h.shift(-1) > close_1h).astype(int)
    df = pd.DataFrame({"trend": trend, "next_up": next_up}, index=close_1h.index)
    df = df[(df.index >= train_start) & (df.index < train_end)].dropna()
    df["tb"] = df["trend"].clip(-3, 3).astype(int)
    table = {}
    for tb in range(-3, 4):
        cell = df[df["tb"] == tb]
        n = len(cell)
        if n >= 10:
            up = cell["next_up"].mean()
            w = min(1.0, n / SMOOTHING_N)
            p_cal = w * up + (1 - w) * baseline
            table[tb] = round(float(p_cal), 4)
        else:
            table[tb] = baseline
    return table, baseline


def p_up_fixD(trend_val, rev_val, offset_pct, trend_table, asset):
    """Fix D: use trend-only at far offsets, current lookup at near-ATM."""
    if abs(offset_pct) > FAR_OFFSET_THRESHOLD:
        tb = int(np.clip(trend_val, -3, 3))
        return trend_table.get(tb, ASSET_BASELINES.get(asset, 0.504))
    else:
        return lookup_p_up(int(trend_val), int(rev_val), asset=asset)


# ── Strike-hit AUC validation ─────────────────────────────────────────────────

def auc_binary(feature, target):
    m = feature.notna() & target.notna()
    if m.sum() < 200: return np.nan
    f = feature[m].values; t = target[m].values
    n_pos = int(t.sum()); n_neg = len(t) - n_pos
    if n_pos < 20 or n_neg < 20: return np.nan
    ranks = rankdata(f)
    return (ranks[t == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def validate_auc(asset, sym, trend_table):
    print(f"\n  [{asset}] Strike-hit AUC comparison: current p_up vs Fix D p_up", flush=True)
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    idx = d_1h.index
    trend_s, rev_s = compute_scores(
        d_1h["close"], d_1h["high"], d_1h["low"], d_1h["volume"],
        d_4h["close"], d_4h["high"], d_4h["low"], d_4h["volume"],
        d_15m["close"], d_15m["high"], d_15m["low"],
        d_1m["close"], d_1m["volume"], idx,
    )
    mask = idx >= pd.Timestamp("2025-01-01", tz="UTC")

    # Current p_up (from lookup)
    cur_pup = pd.Series(
        [lookup_p_up(int(t), int(r), asset=asset) for t, r in zip(trend_s.values, rev_s.values)],
        index=idx,
    )

    close = d_1h["close"]
    next_close = close.shift(-1)

    # Offsets to compare
    ASSET_OFFSETS = {
        "BTC": [-2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0],
        "ETH": [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
        "SOL": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
    }
    offsets = ASSET_OFFSETS[asset]

    print(f"    {'offset':>8}  {'current':>9}  {'fix_D':>9}  {'Δ':>7}", flush=True)
    print(f"    {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}", flush=True)
    for off_pct in offsets:
        off_frac = off_pct / 100.0
        # Fix D p_up — uses trend-only if |off_frac| > threshold
        fixD_pup = pd.Series(
            [p_up_fixD(int(t), int(r), off_frac, trend_table, asset) for t, r in zip(trend_s.values, rev_s.values)],
            index=idx,
        )
        if off_pct > 0:
            y = (next_close > close * (1 + off_frac)).astype(int)
        else:
            y = (next_close < close * (1 + off_frac)).astype(int)
        y_m = y[mask]
        cur_auc = auc_binary(cur_pup[mask] * (1 if off_pct > 0 else -1), y_m)
        fix_auc = auc_binary(fixD_pup[mask] * (1 if off_pct > 0 else -1), y_m)
        d = fix_auc - cur_auc
        marker = "+" if d > 0.005 else ("-" if d < -0.005 else " ")
        print(f"    {off_pct:+7.2f}%  {cur_auc:>9.3f}  {fix_auc:>9.3f}  {d:>+7.3f} {marker}", flush=True)


# ── PnL backtest ──────────────────────────────────────────────────────────────

def run_bt(asset, scans_df, p_up_override_map=None):
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None
        for _, row in group.iterrows():
            if p_up_override_map is not None:
                dt_floor = pd.Timestamp(dt).floor("1h")
                off_frac = (row["strike"] - row["spot"]) / row["spot"] if row["spot"] > 0 else 0
                # Fix D: use trend-only when |offset| > 0.5%, else current
                tr = int(row["composite_trend"])
                rv = int(row["composite_rev"])
                p_up_v = p_up_override_map["trend_table"].get(int(np.clip(tr, -3, 3)), None) \
                         if abs(off_frac) > FAR_OFFSET_THRESHOLD else \
                         lookup_p_up(tr, rv, asset=asset)
                if p_up_v is None: continue
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
    print(f"\n{'='*78}\n  FIX D — trend-only p_up at |offset| > 0.5%\n{'='*78}", flush=True)

    tables = {}
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        print(f"\n  [{asset}] building trend-only lookup from train data...", flush=True)
        d_1m, d_15m, d_1h, d_4h = load_asset(sym)
        trend_s, _ = compute_scores(
            d_1h["close"], d_1h["high"], d_1h["low"], d_1h["volume"],
            d_4h["close"], d_4h["high"], d_4h["low"], d_4h["volume"],
            d_15m["close"], d_15m["high"], d_15m["low"],
            d_1m["close"], d_1m["volume"], d_1h.index,
        )
        table, baseline = build_trend_only_table(trend_s, d_1h["close"], asset, TRAIN_START, TRAIN_END)
        tables[asset] = table
        print(f"    Trend-only table (baseline {baseline:.3f}):", flush=True)
        for tb in sorted(table.keys()):
            print(f"      trend={tb:+d}:  P(up)={table[tb]:.3f}", flush=True)

        # Save
        with open(OUT_DIR / f"fixD_trend_only_{asset}.json", "w") as f:
            json.dump({str(k): v for k, v in table.items()}, f, indent=2)

    # Validate strike-hit AUC
    print(f"\n{'='*78}\n  STRIKE-HIT AUC VALIDATION (2025-01-01 → 2026-present)\n{'='*78}", flush=True)
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        validate_auc(asset, sym, tables[asset])

    # PnL backtest on test window
    print(f"\n{'='*78}\n  PnL BACKTEST on TEST window (2026-03-16 → present)\n{'='*78}", flush=True)
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        scans = load_archive(asset)
        if scans.empty: print(f"  [{asset}] no scans"); continue
        r_base = run_bt(asset, scans, p_up_override_map=None)
        r_fix = run_bt(asset, scans, p_up_override_map={"trend_table": tables[asset]})
        def fmt(r):
            return f"n={r['n']:4d} WR={r['wr']:.1%} PnL=${r['pnl']:+.2f} streak={r['max_streak']:2d} ({r['n_yes']}y/{r['n_no']}n)"
        print(f"\n  [{asset}]", flush=True)
        print(f"    BASELINE: {fmt(r_base)}", flush=True)
        print(f"    FIX D   : {fmt(r_fix)}", flush=True)
        dpnl = r_fix["pnl"] - r_base["pnl"]
        dwr = (r_fix["wr"] - r_base["wr"]) if r_fix["n"] and r_base["n"] else 0
        print(f"    Δ PnL=${dpnl:+.2f}  Δ WR={dwr:+.1%}", flush=True)


if __name__ == "__main__":
    main()
