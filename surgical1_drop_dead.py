#!/usr/bin/env python3
"""
surgical1_drop_dead.py — Experiment 1 of surgical composite improvements.

Drops two dead/duplicate indicators from the current composite:
  - Volume 4h (IC ≈ 0 across all 3 assets in Phase 1 audit)
  - Williams %R 4h (mathematically identical to Stoch 4h, r=1.00)

Keeps the remaining 12 indicators + same vote-based architecture + same
lookup-table calibration structure. Rebuilds per-asset calibration on train
data (2025), then backtests on test set (2026-03-16 → present) against current
production baseline.

Preserves everything else: drift multipliers (BTC=1.4, ETH=0.8, SOL=0.2),
full gate stack, Kelly sizing. This is a surgical swap, not a rebuild.
"""

import math, sys, glob, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import (
    _stoch_k, _macd_cross, _bb_pct, _keltner_pct, _wpr, _reversion_votes,
    ASSET_BASELINES, SMOOTHING_N
)
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "reform_results"

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")
TEST_START  = pd.Timestamp("2026-03-16", tz="UTC")

BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD

DRIFT_MULT = {"BTC": 1.40, "ETH": 0.80, "SOL": 0.20}
ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max_otm_no": 0.40, "gate3": 0.01, "strike_step": 100.0},
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.005, "strike_step": 10.0},
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.01, "strike_step": 1.0},
}
GATE_CS_MIN_YES_OTM = 0.55
GATE_CI_MIN_BEARISH = 0.45
RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = 4.0, 0.33, 3.0, 0.08


# ── Surgical trend votes: drop Volume 4h + WPR 4h ─────────────────────────────

def _trend_votes_surgical(close_4h, high_4h, low_4h):
    """Same as _trend_votes but with Volume 4h and WPR 4h votes REMOVED.
    Range: -4 to +4 (down from -6 to +6)."""
    score = pd.Series(0, index=close_4h.index)

    # 1. Stochastic 4h
    stk4 = _stoch_k(high_4h, low_4h, close_4h, 14)
    score += (stk4 > 80).astype(int)
    score -= (stk4 < 20).astype(int)

    # 2. (REMOVED: Volume 4h — IC ≈ 0 across all assets per Phase 1 audit)

    # 3. MACD 4h crossover
    macd_st = _macd_cross(close_4h)
    score += macd_st.isin(["crossed_up", "up_lag"]).astype(int)
    score -= macd_st.isin(["crossed_down", "down_lag"]).astype(int)

    # 4. BB 4h
    bb4 = _bb_pct(high_4h, low_4h, close_4h, 20)
    score += (bb4 > 0.80).astype(int)
    score -= (bb4 < 0.20).astype(int)

    # 5. Keltner 4h
    kc4_pct, kc4_dn, kc4_up = _keltner_pct(high_4h, low_4h, close_4h, 20, 2)
    score += ((kc4_pct > 0.85) | (close_4h > kc4_up)).astype(int)
    score -= ((kc4_pct < 0.15) | (close_4h < kc4_dn)).astype(int)

    # 6. (REMOVED: Williams %R 4h — r=1.000 with Stoch 4h, pure duplicate)

    return score.clip(-4, 4)


def compute_scores_surgical(close_1h, high_1h, low_1h, volume_1h,
                             close_4h, high_4h, low_4h,
                             close_15m, high_15m, low_15m,
                             close_1m, volume_1m, ts_1h):
    """Surgical version: trend uses 4 indicators, rev unchanged."""
    trend_4h = _trend_votes_surgical(close_4h, high_4h, low_4h)
    trend_1h = trend_4h.reindex(ts_1h, method="ffill").fillna(0).astype(int)
    reversion = _reversion_votes(close_1h, high_1h, low_1h,
                                  close_15m, high_15m, low_15m,
                                  close_1m, volume_1m, ts_1h).fillna(0).astype(int)
    return trend_1h, reversion


# ── Load OHLCV ────────────────────────────────────────────────────────────────

def load_asset(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h


# ── Build calibration table (same methodology as composite_scorer.py main) ────

def build_calibration(trend, rev, close_1h, asset, train_start, train_end):
    """Build (trend_bucket, rev_bucket) → p_up lookup table from train data."""
    baseline = ASSET_BASELINES.get(asset, 0.504)
    next_up = (close_1h.shift(-1) > close_1h).astype(int)
    df = pd.DataFrame({"trend": trend, "rev": rev, "next_up": next_up}, index=close_1h.index)
    mask = (df.index >= train_start) & (df.index < train_end)
    df = df[mask].dropna()
    # Bucket: trend clipped to [-3,3], rev clipped to [-8,8] (matches current system)
    df["tb"] = df["trend"].clip(-3, 3).astype(int)
    df["rb"] = df["rev"].clip(-8, 8).astype(int)

    cal = {}
    for tb in range(-3, 4):
        for rb in range(-8, 9):
            cell = df[(df["tb"] == tb) & (df["rb"] == rb)]
            n = len(cell)
            if n >= 10:
                up = cell["next_up"].mean()
                w = min(1.0, n / SMOOTHING_N)
                p_cal = w * up + (1 - w) * baseline
                cal[(tb, rb)] = round(float(p_cal), 4)
    # Return total coverage
    return cal, len(df), baseline


def lookup_p_up_surgical(trend_score, rev_score, calibration, baseline):
    tb = int(np.clip(trend_score, -3, 3))
    rb = int(np.clip(rev_score, -8, 8))
    if (tb, rb) in calibration:
        return calibration[(tb, rb)]
    # Fallback: linear estimate (same as current system)
    return float(np.clip(baseline + 0.006 * rb + 0.003 * tb, 0.25, 0.80))


# ── Backtest helpers (same as reform_phase5_backtest.py) ──────────────────────

def load_archive(asset):
    if asset == "BTC":
        patterns = ["paper_trades_archive_2026*.csv", "paper_trades_archive_pre_*.csv", "paper_trades.csv"]
    elif asset == "ETH":
        patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    else:
        patterns = ["paper_trades_sol_archive_*.csv", "paper_trades_sol.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))
    if asset == "BTC":
        files = [f for f in files if "_eth" not in f.name and "_sol" not in f.name]
    dfs = []
    for f in files:
        try: dfs.append(pd.read_csv(f, low_memory=False))
        except Exception: pass
    if not dfs: return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    needed = ["decision_time","contract_ticker","spot","strike","p_market","vol_eff",
              "tau_minutes","composite_trend","composite_rev","composite_p_up","resolved_yes"]
    for c in needed:
        if c not in raw.columns: return pd.DataFrame()
    raw = raw.dropna(subset=needed)
    for c in ["spot","strike","p_market","vol_eff","tau_minutes","composite_trend",
              "composite_rev","composite_p_up","resolved_yes"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=needed)
    raw = raw.drop_duplicates(subset=["decision_time","contract_ticker"], keep="last")
    raw["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True)
    raw = raw[raw["decision_time"] >= TEST_START]
    return raw.sort_values("decision_time").reset_index(drop=True)


def p_model_from_p_up(spot, strike, vol_eff, tau, p_up, asset):
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0: return 0.5
    z_strike = math.log(strike/spot) / sigma_tau
    k = DRIFT_MULT.get(asset, 1.0)
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def evaluate_row(row, p_model_v, p_up_v, params):
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    if pm <= 0 or pm >= 1: return None
    offset = (strike - spot) / spot if spot > 0 else 0
    best = None
    for side in ("yes", "no"):
        pm_use = p_model_v
        if not (params["pm_min"] <= pm_use <= params["pm_max"]): continue
        if not (0.04 <= pm <= 0.96): continue
        if side == "yes":
            if offset > 0 and p_up_v < GATE_CS_MIN_YES_OTM: continue
            if offset <= 0 and p_up_v < GATE_CI_MIN_BEARISH: continue
        if side == "no":
            if offset < 0 and p_up_v > params["ns_max_otm_no"]: continue
        fee = kalshi_fee(pm)
        if side == "yes":
            raw = pm_use - pm; net = raw - fee - SLIPPAGE - SPREAD
            rr = pm/(1-pm) if pm < 1 else 999
            if rr > RR_MAX_YES: continue
            if pm < 0.15: tier_min = 0.04
            elif pm < 0.25: tier_min = 0.03
            elif pm < 0.35: tier_min = 0.02
            else: tier_min = 0.0
        else:
            raw = pm - pm_use; net = raw - fee - SLIPPAGE - SPREAD
            rr = (1-pm)/pm if pm > 0 else 999
            if (rr < RR_MIN_NO or rr > RR_MAX_NO) and net < RR_EDGE_EXC: continue
            tier_min = 0.0
        if net < max(params["gate3"], tier_min): continue
        if best is None or net > best["net"]:
            best = {"side": side, "pm_use": pm_use, "pm": pm, "net": net, "strike": strike, "offset": offset}
    return best


def kelly_bet(pm_use, pm, side, bankroll):
    if side == "yes":
        b = (1-pm)/pm if pm > 0 else 0
        p, q = pm_use, 1 - pm_use
    else:
        b = pm/(1-pm) if pm < 1 else 0
        p_no = 1 - pm_use; p, q = p_no, 1 - p_no
    if b <= 0: return 0
    kf = max(0.0, (b*p - q)/b)
    bf = min(kf * KELLY_MULT, KELLY_CAP)
    return round(bankroll * bf, 2)


def trade_pnl(bet, side, pm, won):
    fee_rate = kalshi_fee(pm)
    if bet <= 0: return 0
    if side == "yes":
        if won:
            n_ct = bet/pm if pm > 0 else 0
            return bet*(1-pm)/pm - fee_rate*n_ct
        return -bet
    else:
        if won:
            n_ct = bet/(1-pm) if pm < 1 else 0
            return bet*pm/(1-pm) - fee_rate*n_ct
        return -bet


def run_bt(asset, scans_df, p_up_override=None):
    """p_up_override: dict from (decision_time_floor_1h) → new p_up. None = use stored p_up."""
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None; best_p_up = None
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
                best = cand; best_row = row; best_p_up = p_up_v
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
    print(f"\n{'='*78}\n  SURGICAL EXPERIMENT 1 — drop Volume 4h + WPR 4h votes\n{'='*78}", flush=True)

    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        print(f"\n  [{asset}] loading data...", flush=True)
        d_1m, d_15m, d_1h, d_4h = load_asset(sym)
        # Compute surgical scores
        trend_s, rev_s = compute_scores_surgical(
            d_1h["close"], d_1h["high"], d_1h["low"], d_1h["volume"],
            d_4h["close"], d_4h["high"], d_4h["low"],
            d_15m["close"], d_15m["high"], d_15m["low"],
            d_1m["close"], d_1m["volume"], d_1h.index)
        # Build calibration on train set
        cal, n_train, baseline = build_calibration(trend_s, rev_s, d_1h["close"], asset, TRAIN_START, TRAIN_END)
        print(f"  Built surgical calibration: {len(cal)} cells from {n_train:,} train hours (baseline {baseline:.3f})", flush=True)

        # Generate p_up override for test window (aligned to 1h bars)
        test_mask = d_1h.index >= TEST_START
        p_up_series = pd.Series(
            [lookup_p_up_surgical(int(t), int(r), cal, baseline) for t, r in zip(trend_s[test_mask].values, rev_s[test_mask].values)],
            index=d_1h.index[test_mask]
        )
        p_up_map = dict(zip(p_up_series.index, p_up_series.values))

        scans = load_archive(asset)
        if scans.empty:
            print("  [!] no archive scans in test window"); continue
        print(f"  Test: {len(scans):,} scans across {scans['decision_time'].nunique():,} hours", flush=True)

        # Baseline
        r_base = run_bt(asset, scans, p_up_override=None)
        # Surgical
        r_surg = run_bt(asset, scans, p_up_override=p_up_map)

        def fmt(r):
            if r["n"] == 0: return "no trades"
            return f"n={r['n']:4d} WR={r['wr']:.1%} PnL=${r['pnl']:+.2f} streak={r['max_streak']:2d} ({r['n_yes']}y/{r['n_no']}n)"
        print(f"\n  BASELINE : {fmt(r_base)}", flush=True)
        print(f"  SURGICAL : {fmt(r_surg)}", flush=True)
        dpnl = r_surg["pnl"] - r_base["pnl"]
        dwr = (r_surg["wr"] - r_base["wr"]) if r_surg["n"] and r_base["n"] else 0
        print(f"  Δ PnL=${dpnl:+.2f}  Δ WR={dwr:+.1%}", flush=True)

        # Save the surgical calibration
        cal_out = OUT_DIR / f"surgical1_calibration_{asset}.json"
        with open(cal_out, "w") as f:
            json.dump({f"{k[0]},{k[1]}": v for k, v in cal.items()}, f, indent=2)
        print(f"  Saved surgical calibration → {cal_out}", flush=True)


if __name__ == "__main__":
    main()
