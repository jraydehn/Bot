#!/usr/bin/env python3
"""
backtest_no_kdrift.py — Sweep k_drift_no independently from k_drift_yes.

Current production:
  p_yes_model = Phi(-z_strike + norm.ppf(p_up) * 0.8)   [k_drift_yes = 0.8]
  p_no_model  = 1 - p_yes_model                          [dependent, overestimates]

Reform:
  p_yes_model unchanged (k_drift_yes = 0.8)
  p_no_model  = Phi(z_strike - norm.ppf(p_up) * k_drift_no)   [independent]

At k_drift_no=0.0: pure vol pricing for NO (no directional bias).
At k_drift_no=0.8: same as current (p_no = 1 - p_yes).

Sweeps k_drift_no in {0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8} and reports
trade count, WR, P&L, and calibration for NO trades at each setting.
"""

import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

# ── constants matching production ──────────────────────────────────────────
KALSHI_FEE_RATE = 0.07
SLIPPAGE        = 0.003
SPREAD          = 0.005
BANKROLL_0      = 1000.0
KELLY_MULT      = 0.30
KELLY_CAP       = 0.06

K_DRIFT_YES     = 0.80      # fixed — current reform value
BASE_Z_YES      = 2.0       # vol gate for OTM YES
Z_ABS_NO_MIN    = 0.30      # structural NO gate (near-ATM block)

G3_MIN          = 0.02      # minimum net edge (Gate 3)
PM_MIN, PM_MAX  = 0.05, 0.95
RR_MIN_NO       = 0.10      # min reward:risk for NO (pm/(1-pm) ratio)
RR_MAX_NO       = 10.0
RR_EDGE_EXC     = 0.06      # edge exception bypass for R:R gate


def vol_factor_from_score(vol_score):
    vs = pd.to_numeric(vol_score, errors='coerce')
    if pd.isna(vs):
        return 1.0
    return float(np.clip(1.0 + vs * 0.08, 0.60, 1.40))


def kalshi_fee(pm):
    return KALSHI_FEE_RATE * min(pm, 1 - pm)


def compute_p_yes(spot, strike, vol_eff, tau_min, p_up, k_drift_yes=K_DRIFT_YES):
    """Drift-adjusted YES probability (current reform formula).
    vol_eff is stored per-sqrt(minute); sigma_tau = vol_eff * sqrt(tau_min).
    """
    if vol_eff <= 0 or tau_min <= 0 or spot <= 0:
        return None
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return None
    z = math.log(strike / spot) / sigma_tau
    z_adj = z - norm.ppf(p_up) * k_drift_yes
    return float(np.clip(1 - norm.cdf(z_adj), 0.01, 0.99))


def compute_p_no(spot, strike, vol_eff, tau_min, p_up, k_drift_no):
    """Independent NO probability — NOT forced to be 1 - p_yes.
    vol_eff is stored per-sqrt(minute); sigma_tau = vol_eff * sqrt(tau_min).
    """
    if vol_eff <= 0 or tau_min <= 0 or spot <= 0:
        return None
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return None
    z = math.log(strike / spot) / sigma_tau
    # For NO: Phi(z - drift) where drift shifts toward YES when bullish,
    # reducing p_no; and toward NO when bearish, increasing p_no.
    z_adj = z - norm.ppf(p_up) * k_drift_no
    return float(np.clip(norm.cdf(z_adj), 0.01, 0.99))


def kelly_bet(p_model, pm, side):
    if side == 'yes':
        b = (1 - pm) / pm if pm > 0 else 0
        p, q = p_model, 1 - p_model
    else:
        b = pm / (1 - pm) if pm < 1 else 0
        p, q = p_model, 1 - p_model
    if b <= 0:
        return 0.0
    kf = max(0.0, (b * p - q) / b)
    bf = min(kf * KELLY_MULT, KELLY_CAP)
    return round(BANKROLL_0 * bf, 2)


def trade_pnl(bet, side, pm, won):
    if bet <= 0:
        return 0.0
    fee = kalshi_fee(pm)
    if side == 'yes':
        if won:
            n_ct = bet / pm if pm > 0 else 0
            return bet * (1 - pm) / pm - fee * n_ct
        return -bet
    else:
        if won:
            n_ct = bet / (1 - pm) if pm < 1 else 0
            return bet * pm / (1 - pm) - fee * n_ct
        return -bet


def evaluate_row(row, k_drift_no):
    spot   = float(row['spot'])
    strike = float(row['strike'])
    pm     = float(row['p_market'])
    vol_eff = float(row['vol_eff'])
    tau    = float(row['tau_minutes'])
    p_up   = float(row['composite_p_up'])
    vs     = row['vol_score']
    resolved = int(row['resolved_yes'])

    if not (PM_MIN <= pm <= PM_MAX):
        return None

    vf = vol_factor_from_score(vs)
    sigma_tau_base = vol_eff * math.sqrt(tau)   # vol_eff is per-sqrt(min)
    if sigma_tau_base <= 0:
        return None

    z_abs = abs(math.log(strike / spot) / sigma_tau_base) if spot > 0 else 0
    offset = (strike - spot) / spot if spot > 0 else 0

    candidates = []

    # ── YES candidate ──────────────────────────────────────────────────────
    p_yes = compute_p_yes(spot, strike, vol_eff, tau, p_up, K_DRIFT_YES)
    if p_yes is not None:
        # Vol gate: OTM YES only
        if offset > 0 and z_abs > BASE_Z_YES * vf:
            pass  # blocked
        else:
            fee = kalshi_fee(pm)
            net = (p_yes - pm) - fee - SLIPPAGE - SPREAD
            if net >= G3_MIN:
                won = (resolved == 1)
                candidates.append({'side': 'yes', 'p_model': p_yes, 'pm': pm,
                                    'net': net, 'won': won, 'offset': offset})

    # ── NO candidate ───────────────────────────────────────────────────────
    p_no = compute_p_no(spot, strike, vol_eff, tau, p_up, k_drift_no)
    if p_no is not None:
        # Structural gate: must be ≥0.3σ from spot
        if z_abs < Z_ABS_NO_MIN:
            pass  # blocked
        else:
            pm_no = 1 - pm
            fee = kalshi_fee(pm)  # fee based on YES price
            net = (p_no - pm_no) - fee - SLIPPAGE - SPREAD
            # R:R gate for NO
            rr = pm / (1 - pm) if pm < 1 else 999
            if rr < RR_MIN_NO or rr > RR_MAX_NO:
                if net < RR_EDGE_EXC:
                    pass  # blocked by R:R
                else:
                    won = (resolved == 0)
                    candidates.append({'side': 'no', 'p_model': p_no, 'pm': pm,
                                        'net': net, 'won': won, 'offset': offset})
            else:
                if net >= G3_MIN:
                    won = (resolved == 0)
                    candidates.append({'side': 'no', 'p_model': p_no, 'pm': pm,
                                        'net': net, 'won': won, 'offset': offset})

    if not candidates:
        return None
    return max(candidates, key=lambda c: c['net'])


def run_sweep(df, k_drift_no):
    pnls = []
    no_trades = []
    yes_trades = []

    for dt, group in df.groupby('decision_time', sort=True):
        cands = []
        for _, row in group.iterrows():
            c = evaluate_row(row, k_drift_no)
            if c is not None:
                cands.append(c)
        if not cands:
            continue
        best = max(cands, key=lambda c: c['net'])
        bet = kelly_bet(best['p_model'], best['pm'], best['side'])
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, best['side'], best['pm'], best['won'])
        pnls.append((best['won'], pnl, best['side'], best['p_model'], best['pm']))
        if best['side'] == 'no':
            no_trades.append((best['won'], pnl, best['p_model'], best['pm']))
        else:
            yes_trades.append((best['won'], pnl, best['p_model'], best['pm']))

    if not pnls:
        return None

    n = len(pnls)
    wins = sum(1 for w, *_ in pnls if w)
    total_pnl = sum(p for _, p, *_ in pnls)
    n_no = len(no_trades)
    no_wins = sum(1 for w, *_ in no_trades if w)
    no_pnl = sum(p for _, p, *_ in no_trades)
    n_yes = len(yes_trades)
    yes_wins = sum(1 for w, *_ in yes_trades if w)
    yes_pnl = sum(p for _, p, *_ in yes_trades)

    return {
        'n': n, 'wr': wins / n, 'pnl': total_pnl,
        'n_no': n_no,
        'no_wr': no_wins / n_no if n_no else 0,
        'no_pnl': no_pnl,
        'n_yes': n_yes,
        'yes_wr': yes_wins / n_yes if n_yes else 0,
        'yes_pnl': yes_pnl,
        'no_trades': no_trades,
    }


def calibration_table(no_trades):
    """Bin p_no_model vs actual WR for NO trades."""
    if not no_trades:
        return
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
    print(f"    p_no_model bucket   n    actual_WR   model_avg   bias")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        sub = [(w, pnl, pm, mkt) for (w, pnl, pm, mkt) in no_trades
               if lo <= pm < hi]
        if not sub:
            continue
        n = len(sub)
        wr = sum(1 for w, *_ in sub if w) / n
        avg_model = sum(pm for _, _, pm, _ in sub) / n
        bias = avg_model - wr
        print(f"    [{lo:.2f}, {hi:.2f})          {n:3d}   {wr:.1%}       {avg_model:.2f}       {bias:+.2f}")


def main():
    csv_path = Path(__file__).parent / 'results' / 'paper_trades.csv'
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['resolved_yes'].notna()].copy()
    df['resolved_yes'] = pd.to_numeric(df['resolved_yes'], errors='coerce')
    df['p_market'] = pd.to_numeric(df['p_market'], errors='coerce')
    df['vol_eff'] = pd.to_numeric(df['vol_eff'], errors='coerce')
    df['tau_minutes'] = pd.to_numeric(df['tau_minutes'], errors='coerce')
    df['composite_p_up'] = pd.to_numeric(df['composite_p_up'], errors='coerce')
    df['vol_score'] = pd.to_numeric(df['vol_score'], errors='coerce')
    df['spot'] = pd.to_numeric(df['spot'], errors='coerce')
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df = df.dropna(subset=['resolved_yes', 'p_market', 'vol_eff', 'tau_minutes',
                            'composite_p_up', 'spot', 'strike'])
    df = df[df['decision'] == 'trade']

    print(f"\nbacktest_no_kdrift.py — independent NO probability model")
    print(f"Archive: {len(df)} resolved trade-decisions\n")
    print(f"k_drift_yes = {K_DRIFT_YES} (fixed)")
    print(f"z_abs_no_min = {Z_ABS_NO_MIN}, base_z_yes = {BASE_Z_YES}\n")

    sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

    print(f"{'k_no':>5}  {'n':>4}  {'WR':>6}  {'PnL':>8}  |  {'n_no':>4}  {'NO_WR':>6}  {'NO_PnL':>8}  |  {'n_yes':>4}  {'YES_WR':>6}  {'YES_PnL':>8}")
    print("-" * 95)

    results = {}
    for k in sweep:
        r = run_sweep(df, k)
        if r is None:
            print(f"  {k:.1f}   no trades")
            continue
        results[k] = r
        print(f"  {k:.1f}  {r['n']:4d}  {r['wr']:6.1%}  {r['pnl']:+8.2f}  |  "
              f"{r['n_no']:4d}  {r['no_wr']:6.1%}  {r['no_pnl']:+8.2f}  |  "
              f"{r['n_yes']:4d}  {r['yes_wr']:6.1%}  {r['yes_pnl']:+8.2f}")

    # Calibration tables for key values
    for k in [0.0, 0.4, 0.8]:
        if k not in results:
            continue
        print(f"\n  Calibration — k_drift_no={k:.1f}:")
        calibration_table(results[k]['no_trades'])


if __name__ == '__main__':
    main()
