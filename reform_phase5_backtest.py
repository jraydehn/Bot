#!/usr/bin/env python3
"""
reform_phase5_backtest.py — Phase 5: strategy backtest on TEST set.

Tests both:
  - REFORMED: reformed p_model pipeline (from Phase 4) plugged into full gate stack
  - BASELINE: current production (composite calibration table + k=1.4/0.8/0.2 drift)

On identical scans: paper_trade archives filtered to TEST window
(2026-03-16 onward). Uses real Kalshi p_market values from stored scans.

Reports per asset:
  n, WR, breakeven_WR, $PnL, max consecutive losses, max drawdown
  Side breakdown (yes/no)
  Strike-distance bucket profile
"""

import math, sys, glob, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import lookup_p_up
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from reform_phase3_score import (
    load_asset, extract_features, compute_targets, build_variant_D,
)

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent / "reform_results"

TEST_START = pd.Timestamp("2026-03-16", tz="UTC")

BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD

DRIFT_MULT_BASELINE = {"BTC": 1.40, "ETH": 0.80, "SOL": 0.20}

ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max_otm_no": 0.40, "gate3": 0.01, "strike_step": 100.0},
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.005, "strike_step": 10.0},
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.01, "strike_step": 1.0},
}
GATE_CS_MIN_YES_OTM = 0.55
GATE_CI_MIN_BEARISH = 0.45
RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = 4.0, 0.33, 3.0, 0.08


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
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs: return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    needed = ["decision_time", "contract_ticker", "spot", "strike", "p_market",
              "vol_eff", "tau_minutes", "composite_trend", "composite_rev",
              "composite_p_up", "resolved_yes"]
    for c in needed:
        if c not in raw.columns: return pd.DataFrame()
    raw = raw.dropna(subset=needed)
    for c in ["spot","strike","p_market","vol_eff","tau_minutes","composite_trend","composite_rev","composite_p_up","resolved_yes"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=needed)
    raw = raw.drop_duplicates(subset=["decision_time","contract_ticker"], keep="last")
    raw["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True)
    raw = raw[raw["decision_time"] >= TEST_START]
    raw = raw.sort_values("decision_time").reset_index(drop=True)
    return raw


def p_model_baseline(spot, strike, vol_eff, tau, p_up, asset):
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0: return 0.5
    z_strike = math.log(strike/spot) / sigma_tau
    k = DRIFT_MULT_BASELINE.get(asset, 1.0)
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def p_model_reformed(spot, strike, vol_eff, tau, p_up_reformed):
    """Reformed p_up becomes the drift directly; use log-normal + drift."""
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0: return 0.5
    z_strike = math.log(strike/spot) / sigma_tau
    # p_up_reformed is the calibrated predicted prob of next-hour UP.
    # Treat same as baseline: Φ⁻¹(p_up) as drift in z-units (k=1.0, since model is already calibrated).
    z_drift = norm.ppf(np.clip(p_up_reformed, 0.01, 0.99))
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def evaluate_row(row, p_model_value, p_up_value, params):
    """Apply gate stack. p_model_value already computed (from baseline or reformed)."""
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    if pm <= 0 or pm >= 1: return None
    offset = (strike - spot) / spot if spot > 0 else 0
    best = None
    for side in ("yes", "no"):
        pm_use = p_model_value
        if not (params["pm_min"] <= pm_use <= params["pm_max"]): continue
        if not (0.04 <= pm <= 0.96): continue
        if side == "yes":
            if offset > 0 and p_up_value < GATE_CS_MIN_YES_OTM: continue
            if offset <= 0 and p_up_value < GATE_CI_MIN_BEARISH: continue
        if side == "no":
            if offset < 0 and p_up_value > params["ns_max_otm_no"]: continue
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


def run_backtest(asset, scans_df, mode, reform_p_up_lookup=None):
    """mode = 'baseline' or 'reformed'. reform_p_up_lookup maps (decision_time) → p_up."""
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []; offsets_strike = []; bankrolls = [BANKROLL_0]
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None; best_p_up = None
        for _, row in group.iterrows():
            # Select p_up + p_model source
            if mode == "baseline":
                p_up_v = row["composite_p_up"]
                p_mv = p_model_baseline(row["spot"], row["strike"], row["vol_eff"],
                                         row["tau_minutes"], p_up_v, asset)
            else:
                p_up_v = reform_p_up_lookup.get(dt, np.nan) if reform_p_up_lookup is not None else np.nan
                if not np.isfinite(p_up_v): continue
                p_mv = p_model_reformed(row["spot"], row["strike"], row["vol_eff"],
                                         row["tau_minutes"], p_up_v)
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
        sd = round(best["offset"] * best_row["spot"] / params["strike_step"])
        offsets_strike.append(sd)
        bankrolls.append(bankroll)
    if not pnls:
        return dict(n=0, wr=0, pnl=0, final=BANKROLL_0, max_loss_streak=0, max_dd=0,
                    n_yes=0, n_no=0, breakeven_wr=0, sd_buckets={})
    n = len(pnls); wins_n = sum(wins); wr = wins_n/n
    streak = 0; max_streak = 0
    for w in wins:
        if not w: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    peak = BANKROLL_0; max_dd = 0
    for b in bankrolls:
        peak = max(peak, b); max_dd = max(max_dd, (peak-b)/peak)
    avg_win = sum(p for p,w in zip(pnls,wins) if w) / max(1, wins_n)
    avg_loss = -sum(p for p,w in zip(pnls,wins) if not w) / max(1, n-wins_n)
    be_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0
    sd_buckets = {}
    for sd, p, w in zip(offsets_strike, pnls, wins):
        if sd not in sd_buckets: sd_buckets[sd] = {"n":0, "wins":0, "pnl":0.0}
        sd_buckets[sd]["n"] += 1
        sd_buckets[sd]["wins"] += int(w)
        sd_buckets[sd]["pnl"] += p
    return dict(n=n, wr=wr, pnl=sum(pnls), final=bankrolls[-1],
                max_loss_streak=max_streak, max_dd=max_dd,
                n_yes=sum(1 for s in sides if s=="yes"),
                n_no=sum(1 for s in sides if s=="no"),
                breakeven_wr=be_wr, sd_buckets=sd_buckets)


def compute_reformed_p_up(asset, sym, btc_close_1h, scans_df):
    """Build p_up lookup: decision_time → reformed p_up. Uses Phase 4 pipeline."""
    with open(OUT_DIR / f"phase4_{asset}.pkl", "rb") as f:
        pipe = pickle.load(f)
    d_1m, d_15m, d_1h, d_4h, d_1d = load_asset(sym)
    btc = btc_close_1h if asset != "BTC" else None
    X = extract_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=btc)
    X_D = build_variant_D(X)
    # Only predict on unique decision_times from scans
    want_times = pd.to_datetime(scans_df["decision_time"].unique(), utc=True)
    # Align to 1h boundaries
    want_times = pd.DatetimeIndex(want_times).floor("1h")
    mask = X_D.index.isin(want_times)
    Xs = X_D[mask].dropna()
    if Xs.empty:
        return {}
    Xs = Xs[pipe["feature_columns"]]
    Xn = pipe["scaler"].transform(Xs)
    p_raw = pipe["clf"].predict_proba(Xn)[:, 1]
    p_cal = pipe["isotonic"].predict(p_raw)
    return dict(zip(Xs.index, p_cal))


def report(asset, mode, r):
    print(f"\n  [{asset} — {mode}]", flush=True)
    print(f"    n={r['n']}  WR={r['wr']:.1%}  breakeven={r['breakeven_wr']:.1%}  PnL=${r['pnl']:+.2f}  streak={r['max_loss_streak']}  maxDD={r['max_dd']:.1%}  ({r['n_yes']}y / {r['n_no']}n)", flush=True)
    if r["n"] == 0: return
    print(f"    strike-distance buckets:", flush=True)
    step = ASSET_PARAMS[asset]["strike_step"]
    for sd in sorted(r["sd_buckets"].keys()):
        b = r["sd_buckets"][sd]
        print(f"      sd={sd:+3d} (${sd*step:+.0f})  n={b['n']:3d}  WR={b['wins']/b['n']:.1%}  PnL=${b['pnl']:+7.2f}", flush=True)


def main():
    print(f"\n{'='*78}\n  PHASE 5 — strategy backtest on TEST set (2026-03-16 → present)\n{'='*78}", flush=True)
    _, _, btc_1h, _, _ = load_asset("BTCUSDT")
    btc_close_1h = btc_1h["close"]

    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        print(f"\n{'─'*78}\n  {asset}\n{'─'*78}", flush=True)
        scans = load_archive(asset)
        if scans.empty:
            print("  no scans in test window"); continue
        print(f"  {len(scans):,} scans across {scans['decision_time'].nunique():,} unique hours", flush=True)

        # Baseline (current production)
        r_base = run_backtest(asset, scans, "baseline")
        report(asset, "BASELINE (current)", r_base)

        # Reformed: compute p_up lookup first, then backtest
        p_up_lookup = compute_reformed_p_up(asset, sym, btc_close_1h, scans)
        # lookup is indexed by 1h-floor timestamps; normalize scan decision_times to 1h
        scans_floor = scans.copy()
        scans_floor["decision_time"] = pd.to_datetime(scans_floor["decision_time"], utc=True).dt.floor("1h")
        r_ref = run_backtest(asset, scans_floor, "reformed", p_up_lookup)
        report(asset, "REFORMED", r_ref)

        d_pnl = r_ref["pnl"] - r_base["pnl"]
        d_wr = r_ref["wr"] - r_base["wr"]
        d_dd = r_ref["max_dd"] - r_base["max_dd"]
        print(f"\n  [{asset} — DELTA]  ΔPnL=${d_pnl:+.2f}  ΔWR={d_wr:+.1%}  ΔmaxDD={d_dd:+.1%}", flush=True)


if __name__ == "__main__":
    main()
