#!/usr/bin/env python3
"""
backtest_counter_tape_gate.py — Simulate the hybrid counter-tape gate on real Kalshi archive data.

Uses existing stored chg_5m, chg_10m, chg_30m values per scan row. For each decision-hour's
selected trade under current calibration (BTC k=1.4, ETH k=0.8, SOL k=0.2), computes a severity
score and applies:
  severity < 0.5  → full Kelly (no change)
  0.5 ≤ severity < 1.5 → Kelly scale = max(0.25, 1 − (severity − 0.5) × 0.75)
  severity ≥ 1.5 → hard block

Reports per-asset: trades blocked, trades dampened, wins blocked, losses blocked,
Kelly-deltas on dampened bets, and net PnL delta vs baseline.
"""

import math, sys, glob, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import lookup_p_up
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

RESULTS_DIR = Path(__file__).parent / "results"
BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD

# Current production calibration
DRIFT_MULT = {"BTC": 1.40, "ETH": 0.80, "SOL": 0.20}

# Gate 0/CS/CI/NS params
ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max_otm_no": 0.40, "gate3": 0.01, "strike_step": 100.0},
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.005, "strike_step": 10.0},
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.01, "strike_step": 1.0},
}
GATE_CS_MIN_YES_OTM = 0.55
GATE_CI_MIN_BEARISH = 0.45
RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = 4.0, 0.33, 3.0, 0.08

# Counter-tape thresholds (where severity == 1.0). Scale per asset by baseline hourly vol.
TAPE_THRESHOLDS = {
    "BTC": {"chg_5m": 0.0016, "chg_10m": 0.0024, "chg_30m": 0.0040},
    "ETH": {"chg_5m": 0.0015, "chg_10m": 0.0025, "chg_30m": 0.0040},
    "SOL": {"chg_5m": 0.0025, "chg_10m": 0.0040, "chg_30m": 0.0065},
}


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
              "composite_p_up", "resolved_yes", "chg_30m", "chg_10m", "chg_5m"]
    for c in needed:
        if c not in raw.columns: return pd.DataFrame()
    raw = raw.dropna(subset=["decision_time","contract_ticker","spot","strike","p_market",
                              "vol_eff","tau_minutes","composite_trend","composite_rev",
                              "composite_p_up","resolved_yes"])
    for c in ["spot","strike","p_market","vol_eff","tau_minutes","composite_trend","composite_rev",
              "composite_p_up","resolved_yes","chg_30m","chg_10m","chg_5m"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    # chg fields are stored as percentages (e.g. -0.8686 = -0.87%) — convert to fractions
    for c in ["chg_30m","chg_10m","chg_5m"]:
        raw[c] = raw[c] / 100.0
    raw = raw.dropna(subset=["spot","strike","p_market","vol_eff","tau_minutes","composite_trend",
                              "composite_rev","composite_p_up","resolved_yes"])
    raw = raw.drop_duplicates(subset=["decision_time","contract_ticker"], keep="last")
    raw = raw.sort_values("decision_time").reset_index(drop=True)
    return raw


def compute_pmodel(spot, strike, vol_eff, tau, p_up, k_drift):
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0: return 0.5
    z_strike = math.log(strike/spot) / sigma_tau
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k_drift
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def evaluate_row(row, k_drift, params):
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    vol_eff = row["vol_eff"]; tau = row["tau_minutes"]
    p_up = row["composite_p_up"]
    if vol_eff <= 0 or tau <= 0 or pm <= 0 or pm >= 1: return None
    offset = (strike - spot) / spot if spot > 0 else 0
    pm_raw = compute_pmodel(spot, strike, vol_eff, tau, p_up, k_drift)
    best = None
    for side in ("yes", "no"):
        pm_use = pm_raw  # mult=1.0 for all assets now
        if not (params["pm_min"] <= pm_use <= params["pm_max"]): continue
        if not (0.04 <= pm <= 0.96): continue
        if side == "yes":
            if offset > 0 and p_up < GATE_CS_MIN_YES_OTM: continue
            if offset <= 0 and p_up < GATE_CI_MIN_BEARISH: continue
        if side == "no":
            if offset < 0 and p_up > params["ns_max_otm_no"]: continue
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


def severity_score(side, chg_5m, chg_10m, chg_30m, thr):
    """Counter-tape severity. Positive = fighting tape."""
    if pd.isna(chg_5m) or pd.isna(chg_10m) or pd.isna(chg_30m): return 0.0
    # YES loses when price falls (negative chg fights YES). NO loses when price rises.
    if side == "yes":
        c5, c10, c30 = -chg_5m, -chg_10m, -chg_30m
    else:
        c5, c10, c30 = chg_5m, chg_10m, chg_30m
    sev = max(0.0, c5/thr["chg_5m"], c10/thr["chg_10m"], c30/thr["chg_30m"])
    return sev


def kelly_bet(pm_use, pm, side, bankroll, kelly_scale=1.0):
    if side == "yes":
        b = (1-pm)/pm if pm > 0 else 0
        p, q = pm_use, 1 - pm_use
    else:
        b = pm/(1-pm) if pm < 1 else 0
        p_no = 1 - pm_use
        p, q = p_no, 1 - p_no
    if b <= 0: return 0
    kf = max(0.0, (b*p - q)/b)
    bf = min(kf * KELLY_MULT * kelly_scale, KELLY_CAP)
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


def run_bt(asset, scans_df, use_gate=True):
    """Run backtest with or without counter-tape gate."""
    params = ASSET_PARAMS[asset]
    thr = TAPE_THRESHOLDS[asset]
    k = DRIFT_MULT[asset]
    bankroll = BANKROLL_0
    results = {"pnls": [], "wins": [], "sides": [], "bankrolls": [BANKROLL_0],
               "blocked": 0, "dampened": 0, "wins_blocked": 0, "losses_blocked": 0,
               "pnl_blocked_saved": 0.0, "pnl_dampened_delta": 0.0}
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best_cand = None; best_row = None
        for _, row in group.iterrows():
            c = evaluate_row(row, k, params)
            if c is None: continue
            if best_cand is None or c["net"] > best_cand["net"]:
                best_cand = c; best_row = row
        if best_cand is None: continue
        # Severity on best trade
        sev = severity_score(best_cand["side"],
                             best_row.get("chg_5m", 0), best_row.get("chg_10m", 0), best_row.get("chg_30m", 0),
                             thr) if use_gate else 0.0
        # Scale decision
        kelly_scale = 1.0
        actual_yes = int(best_row["resolved_yes"])
        won = (actual_yes == 1 and best_cand["side"] == "yes") or (actual_yes == 0 and best_cand["side"] == "no")
        baseline_bet = kelly_bet(best_cand["pm_use"], best_cand["pm"], best_cand["side"], bankroll, 1.0)
        baseline_pnl = trade_pnl(baseline_bet, best_cand["side"], best_cand["pm"], won)
        if use_gate and sev >= 1.5:
            # Hard block
            results["blocked"] += 1
            if won: results["wins_blocked"] += 1
            else: results["losses_blocked"] += 1
            results["pnl_blocked_saved"] -= baseline_pnl  # negated since blocking avoids the realized pnl
            continue
        elif use_gate and sev >= 0.5:
            kelly_scale = max(0.25, 1.0 - (sev - 0.5) * 0.75)
            results["dampened"] += 1
        bet = kelly_bet(best_cand["pm_use"], best_cand["pm"], best_cand["side"], bankroll, kelly_scale)
        if bet <= 0: continue
        pnl = trade_pnl(bet, best_cand["side"], best_cand["pm"], won)
        if use_gate and sev >= 0.5:
            results["pnl_dampened_delta"] += pnl - baseline_pnl
        bankroll = max(1.0, bankroll + pnl)
        results["pnls"].append(pnl); results["wins"].append(won); results["sides"].append(best_cand["side"])
        results["bankrolls"].append(bankroll)
    if not results["pnls"]:
        return {**results, "n": 0, "wr": 0, "pnl": 0, "final": BANKROLL_0,
                "max_streak": 0, "max_dd": 0}
    n = len(results["pnls"]); wins = sum(results["wins"])
    streak = 0; max_streak = 0
    for w in results["wins"]:
        if not w: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    peak = BANKROLL_0; max_dd = 0
    for b in results["bankrolls"]:
        peak = max(peak, b); max_dd = max(max_dd, (peak-b)/peak)
    results["n"] = n; results["wr"] = wins/n; results["pnl"] = sum(results["pnls"])
    results["final"] = results["bankrolls"][-1]
    results["max_streak"] = max_streak; results["max_dd"] = max_dd
    return results


def main():
    print(f"\n{'='*78}\n  COUNTER-TAPE GATE SIM — real Kalshi archive, current calibration\n{'='*78}\n", flush=True)
    for asset in ("BTC","ETH","SOL"):
        t0 = time.time()
        print(f"[{asset}] loading archives...", flush=True)
        df = load_archive(asset)
        if df.empty:
            print(f"  no data"); continue
        print(f"[{asset}] {len(df):,} scans, {df['decision_time'].nunique():,} unique hours", flush=True)
        baseline = run_bt(asset, df, use_gate=False)
        with_gate = run_bt(asset, df, use_gate=True)
        delta = with_gate["pnl"] - baseline["pnl"]
        print(f"\n[{asset}] BASELINE (no gate):  n={baseline['n']:4d} WR={baseline['wr']:.1%} pnl=${baseline['pnl']:+8.2f} streak={baseline['max_streak']:2d} maxDD={baseline['max_dd']:.1%}", flush=True)
        print(f"[{asset}] WITH GATE:            n={with_gate['n']:4d} WR={with_gate['wr']:.1%} pnl=${with_gate['pnl']:+8.2f} streak={with_gate['max_streak']:2d} maxDD={with_gate['max_dd']:.1%}", flush=True)
        print(f"[{asset}] NET DELTA: ${delta:+8.2f}", flush=True)
        print(f"  Hard blocks: {with_gate['blocked']:3d}  (wins_blocked={with_gate['wins_blocked']}, losses_blocked={with_gate['losses_blocked']}, saved=${with_gate['pnl_blocked_saved']:+.2f})", flush=True)
        print(f"  Dampened:    {with_gate['dampened']:3d}  (delta vs full size ${with_gate['pnl_dampened_delta']:+.2f})", flush=True)
        print(f"  [{asset}] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
