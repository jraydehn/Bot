#!/usr/bin/env python3
"""
backtest_calibration_sweep.py — P&L-based calibration sweep.

For each asset (BTC/ETH/SOL), sweeps (k_drift, no_mult, yes_mult) and reports
realized PnL, WR, max consecutive losses, and max drawdown over the full
15-month historical window. Replaces bin-bias optimization with profit-based.
"""

import math, sys, glob, warnings, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import compute_scores, lookup_p_up
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

DATA_DIR = Path(__file__).parent / "data"
BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
MIN_NET_EDGE = 0.01      # match decision.py Gate 3 BTC/SOL=1%, ETH=0.5%
MIN_NET_EDGE_ETH = 0.005
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD
TAU = 60.0

GATE_CS_MIN_YES_OTM = 0.55  # composite_p_up for OTM YES
GATE_CI_MIN_BEARISH = 0.45  # ITM YES bearish block
GATE_NS_MAX_NO_OTM_BTC = 0.40
GATE_NS_MAX_NO_OTM_ALT = 0.45
P_MODEL_MIN_BTC = 0.04
P_MODEL_MAX_BTC = 0.96
P_MODEL_MIN_ALT = 0.02
P_MODEL_MAX_ALT = 0.98
RR_MAX_NO = 4.0
RR_MIN_NO = 0.33
RR_MAX_YES = 3.0
RR_EDGE_EXC = 0.08

OFFSETS = [-0.020, -0.015, -0.010, -0.0075, -0.005, -0.0025,
            0.0025,  0.005,  0.0075,  0.010,  0.015,  0.020]

ASSETS = [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]


def load_data(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    df_1m = pd.read_parquet(f_1m); df_1m.index = pd.to_datetime(df_1m.index, utc=True); df_1m.sort_index(inplace=True)
    df_1h = pd.read_parquet(f_1h); df_1h.index = pd.to_datetime(df_1h.index, utc=True); df_1h.sort_index(inplace=True)
    df_4h = pd.read_parquet(f_4h); df_4h.index = pd.to_datetime(df_4h.index, utc=True); df_4h.sort_index(inplace=True)
    return df_1m, df_1h, df_4h


def precompute(asset, sym, start_ts):
    print(f"[{asset}] loading data...", flush=True)
    df_1m, df_1h, df_4h = load_data(sym)
    df_1m = df_1m[df_1m.index >= start_ts]
    df_1h = df_1h[df_1h.index >= start_ts]
    df_4h = df_4h[df_4h.index >= start_ts]
    df_15m = df_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    print(f"[{asset}] computing composite scores ({len(df_1h):,} bars)...", flush=True)
    trend_s, rev_s = compute_scores(
        df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"],
        df_4h["close"], df_4h["high"], df_4h["low"], df_4h["volume"],
        df_15m["close"], df_15m["high"], df_15m["low"],
        df_1m["close"], df_1m["volume"], df_1h.index,
    )
    # vol per min from 1m
    lr = np.log(df_1m["close"]/df_1m["close"].shift(1))
    vol_1m = lr.rolling(60).std()
    vol_pm = vol_1m.resample("1h", origin="start_day").last().reindex(df_1h.index, method="ffill").fillna(3e-4).clip(lower=1e-6)

    # Pre-compute p_up per bar
    p_up_arr = np.array([lookup_p_up(int(t), int(r), asset=asset) for t, r in zip(trend_s.values, rev_s.values)])
    z_drift_unit = np.array([norm.ppf(np.clip(p, 0.01, 0.99)) for p in p_up_arr])
    return {
        "ts": df_1h.index,
        "spot": df_1h["close"].values,
        "next_close": df_1h["close"].shift(-1).values,  # next bar close
        "p_up": p_up_arr,
        "z_drift_unit": z_drift_unit,
        "vol_pm": vol_pm.values,
    }


def run_backtest(asset, data, k_drift, no_mult, yes_mult, gate3_min=None):
    """Run a single backtest with given params; return summary dict."""
    if gate3_min is None:
        gate3_min = MIN_NET_EDGE_ETH if asset == "ETH" else MIN_NET_EDGE
    pmin = P_MODEL_MIN_BTC if asset == "BTC" else P_MODEL_MIN_ALT
    pmax = P_MODEL_MAX_BTC if asset == "BTC" else P_MODEL_MAX_ALT
    ns_max = GATE_NS_MAX_NO_OTM_BTC if asset == "BTC" else GATE_NS_MAX_NO_OTM_ALT

    spot = data["spot"]; next_close = data["next_close"]
    p_up = data["p_up"]; z_du = data["z_drift_unit"]; vol_pm = data["vol_pm"]

    sigma_tau = vol_pm * math.sqrt(TAU)
    bankroll = BANKROLL_0
    pnl_seq = []
    win_seq = []
    side_seq = []
    bankroll_history = [BANKROLL_0]
    n_eval = len(spot) - 1

    for i in range(n_eval):
        if np.isnan(next_close[i]) or sigma_tau[i] <= 0: continue
        s = spot[i]; nc = next_close[i]
        zd_drift = z_du[i] * k_drift
        # Find best candidate among offsets/sides
        best = None
        for off in OFFSETS:
            strike = s * (1 + off)
            # log-normal p_market simulation (use vol_pm unsmoothed)
            z_strike = math.log(strike / s) / sigma_tau[i]
            pm = float(np.clip(1 - norm.cdf(z_strike), 0.04, 0.96))
            # raw p_model with k drift
            pm_raw = float(np.clip(1 - norm.cdf(z_strike - zd_drift), 0.01, 0.99))

            for side in ("yes", "no"):
                # Apply side multiplier
                if side == "yes":
                    pm_use = float(np.clip(pm_raw * yes_mult, 0.01, 0.99))
                else:
                    pm_use = float(np.clip(pm_raw * no_mult, 0.01, 0.99))
                # Gate 0
                if not (pmin <= pm_use <= pmax): continue
                if not (0.04 <= pm <= 0.96): continue
                # Gate CS / NS (composite path)
                if side == "yes":
                    if off > 0 and p_up[i] < GATE_CS_MIN_YES_OTM: continue
                    if off <= 0 and p_up[i] < GATE_CI_MIN_BEARISH: continue
                if side == "no":
                    if off < 0 and p_up[i] > ns_max: continue
                # Direction edge
                fee = kalshi_fee(pm)
                if side == "yes":
                    raw = pm_use - pm
                    rr = pm / (1 - pm) if pm < 1 else 999
                    if rr > RR_MAX_YES: continue   # YES R:R unconditional block
                    # OTM tier
                    if pm < 0.15: tier_min = 0.04
                    elif pm < 0.25: tier_min = 0.03
                    elif pm < 0.35: tier_min = 0.02
                    else: tier_min = 0.0
                else:
                    raw = pm - pm_use
                    rr = (1 - pm) / pm if pm > 0 else 999
                    if (rr < RR_MIN_NO or rr > RR_MAX_NO) and (raw - fee - SLIPPAGE - SPREAD) < RR_EDGE_EXC: continue
                    tier_min = 0.0
                net = raw - fee - SLIPPAGE - SPREAD
                # Gate 3
                if net < max(gate3_min, tier_min): continue
                if best is None or net > best["net"]:
                    best = dict(off=off, strike=strike, side=side, pm=pm, pm_use=pm_use, net=net)
        if best is None: continue
        # Kelly
        if best["side"] == "yes":
            b = (1 - best["pm"]) / best["pm"]; p, q = best["pm_use"], 1 - best["pm_use"]
        else:
            p_no = 1 - best["pm_use"]; b = best["pm"] / (1 - best["pm"]); p, q = p_no, 1 - p_no
        kf = max(0.0, (b*p - q)/b)
        bf = min(kf * KELLY_MULT, KELLY_CAP)
        bet = round(bankroll * bf, 2)
        if bet <= 0: continue
        # Outcome
        actual_yes = 1 if nc > best["strike"] else 0
        won = (actual_yes == 1 and best["side"] == "yes") or (actual_yes == 0 and best["side"] == "no")
        # PnL
        fee_rate = kalshi_fee(best["pm"])
        if best["side"] == "yes":
            if won:
                n_ct = bet / best["pm"]; gross = bet * (1 - best["pm"])/best["pm"]; pnl = gross - fee_rate * n_ct
            else: pnl = -bet
        else:
            if won:
                n_ct = bet / (1 - best["pm"]); gross = bet * best["pm"]/(1-best["pm"]); pnl = gross - fee_rate * n_ct
            else: pnl = -bet
        bankroll = max(1.0, bankroll + pnl)
        pnl_seq.append(pnl)
        win_seq.append(won)
        side_seq.append(best["side"])
        bankroll_history.append(bankroll)

    if not pnl_seq:
        return dict(n=0, wr=0, pnl=0, final=BANKROLL_0, max_loss_streak=0, max_dd=0, n_yes=0, n_no=0)
    wins = sum(win_seq)
    # Max consecutive losses
    streak = 0; max_streak = 0
    for w in win_seq:
        if not w:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0
    # Max drawdown from running peak
    peak = BANKROLL_0; max_dd = 0
    for b in bankroll_history:
        peak = max(peak, b)
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)
    return dict(
        n=len(pnl_seq), wr=wins/len(pnl_seq), pnl=sum(pnl_seq),
        final=bankroll_history[-1], max_loss_streak=max_streak, max_dd=max_dd,
        n_yes=sum(1 for s in side_seq if s == "yes"),
        n_no=sum(1 for s in side_seq if s == "no"),
    )


def main():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    print(f"\n{'='*78}\n  CALIBRATION SWEEP — P&L based, full gate stack, 15mo data\n{'='*78}\n", flush=True)

    # Per-asset: pre-compute once, then sweep params
    for asset, sym in ASSETS:
        t0 = time.time()
        data = precompute(asset, sym, start)
        print(f"[{asset}] precomputed in {time.time()-t0:.1f}s, evaluating sweep...", flush=True)

        # Phase 1: sweep k only with mult=1.0
        results = []
        for k in [0.50, 0.80, 1.00, 1.20, 1.40, 1.70, 2.00]:
            r = run_backtest(asset, data, k, 1.0, 1.0)
            r["k"], r["no_m"], r["yes_m"] = k, 1.0, 1.0
            results.append(r)
            print(f"  {asset} k={k:.2f} no_m=1.00 yes_m=1.00: n={r['n']:5d} WR={r['wr']:.1%} pnl=${r['pnl']:+8.0f} streak={r['max_loss_streak']:2d} maxDD={r['max_dd']:.1%}", flush=True)

        # Phase 2: at best-k, sweep multipliers (BTC only, since the discussion is BTC-specific)
        if asset == "BTC":
            best_k = max(results, key=lambda x: x["pnl"])["k"]
            print(f"\n[BTC] Best-k from phase 1 = {best_k}; sweeping multipliers around it:", flush=True)
            for k in [max(0.5, best_k-0.2), best_k, min(2.0, best_k+0.2)]:
                for nm in [0.65, 0.80, 1.00]:
                    for ym in [0.80, 0.90, 1.00]:
                        r = run_backtest(asset, data, k, nm, ym)
                        r["k"], r["no_m"], r["yes_m"] = k, nm, ym
                        results.append(r)
                        print(f"  BTC k={k:.2f} no_m={nm:.2f} yes_m={ym:.2f}: n={r['n']:5d} WR={r['wr']:.1%} pnl=${r['pnl']:+8.0f} streak={r['max_loss_streak']:2d} maxDD={r['max_dd']:.1%}", flush=True)

        # Top 5 by pnl
        results.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"\n[{asset}] TOP 5 by P&L:", flush=True)
        for r in results[:5]:
            print(f"  k={r['k']:.2f} no_m={r['no_m']:.2f} yes_m={r['yes_m']:.2f}  n={r['n']:5d} WR={r['wr']:.1%} pnl=${r['pnl']:+8.0f} streak={r['max_loss_streak']:2d} maxDD={r['max_dd']:.1%} (yes={r['n_yes']}, no={r['n_no']})", flush=True)
        print()


if __name__ == "__main__":
    main()
