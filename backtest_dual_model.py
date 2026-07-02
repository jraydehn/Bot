#!/usr/bin/env python3
"""
backtest_dual_model.py — Independently calibrate k_drift_yes and k_drift_no.

YES sweep: evaluate only YES candidates, find k_drift_yes that maximizes YES P&L.
NO sweep:  evaluate only NO candidates, find k_drift_no  that maximizes NO P&L.

Each model is optimized for its own directional P&L — no cross-contamination.
Then combines the two optimal values for a final combined run.
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

BASE_Z_YES      = 2.0       # vol gate for OTM YES
Z_ABS_NO_MIN    = 0.30      # structural NO gate
G3_MIN          = 0.02
PM_MIN, PM_MAX  = 0.05, 0.95
RR_MIN_NO       = 0.10
RR_MAX_NO       = 10.0
RR_EDGE_EXC     = 0.06


def vol_factor(vol_score):
    vs = pd.to_numeric(vol_score, errors='coerce')
    if pd.isna(vs):
        return 1.0
    return float(np.clip(1.0 + vs * 0.08, 0.60, 1.40))


def kalshi_fee(pm):
    return KALSHI_FEE_RATE * min(pm, 1 - pm)


def p_yes(spot, strike, vol_eff, tau_min, p_up, k_drift):
    """Log-normal YES probability with drift. vol_eff is per-sqrt(min)."""
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return None
    z = math.log(strike / spot) / sigma_tau
    return float(np.clip(1 - norm.cdf(z - norm.ppf(p_up) * k_drift), 0.01, 0.99))


def p_no(spot, strike, vol_eff, tau_min, p_up, k_drift):
    """Independent NO probability with drift. vol_eff is per-sqrt(min)."""
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return None
    z = math.log(strike / spot) / sigma_tau
    return float(np.clip(norm.cdf(z - norm.ppf(p_up) * k_drift), 0.01, 0.99))


def kelly_bet(p_model, pm, side):
    if side == 'yes':
        b = (1 - pm) / pm if pm > 0 else 0
    else:
        b = pm / (1 - pm) if pm < 1 else 0
    if b <= 0:
        return 0.0
    kf = max(0.0, (b * p_model - (1 - p_model)) / b)
    bf = min(kf * KELLY_MULT, KELLY_CAP)
    return round(BANKROLL_0 * bf, 2)


def trade_pnl(bet, side, pm, won):
    if bet <= 0:
        return 0.0
    fee = kalshi_fee(pm)
    if side == 'yes':
        return (bet * (1 - pm) / pm - fee * (bet / pm)) if won else -bet
    else:
        return (bet * pm / (1 - pm) - fee * (bet / (1 - pm))) if won else -bet


def yes_candidate(row, k_drift_yes):
    spot   = float(row['spot']); strike = float(row['strike'])
    pm     = float(row['p_market']); vol_eff = float(row['vol_eff'])
    tau    = float(row['tau_minutes']); p_up = float(row['composite_p_up'])
    vs     = row['vol_score']; resolved = int(row['resolved_yes'])
    if not (PM_MIN <= pm <= PM_MAX):
        return None
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0:
        return None
    z_abs = abs(math.log(strike / spot) / sigma_tau)
    offset = (strike - spot) / spot
    vf = vol_factor(vs)
    # Vol gate: OTM YES only
    if offset > 0 and z_abs > BASE_Z_YES * vf:
        return None
    pm_yes = p_yes(spot, strike, vol_eff, tau, p_up, k_drift_yes)
    if pm_yes is None:
        return None
    fee = kalshi_fee(pm)
    net = (pm_yes - pm) - fee - SLIPPAGE - SPREAD
    if net < G3_MIN:
        return None
    won = (resolved == 1)
    return {'side': 'yes', 'p_model': pm_yes, 'pm': pm, 'net': net,
            'won': won, 'offset': offset, 'z_abs': z_abs}


def no_candidate(row, k_drift_no):
    spot   = float(row['spot']); strike = float(row['strike'])
    pm     = float(row['p_market']); vol_eff = float(row['vol_eff'])
    tau    = float(row['tau_minutes']); p_up = float(row['composite_p_up'])
    resolved = int(row['resolved_yes'])
    if not (PM_MIN <= pm <= PM_MAX):
        return None
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0:
        return None
    z_abs = abs(math.log(strike / spot) / sigma_tau)
    if z_abs < Z_ABS_NO_MIN:
        return None
    # R:R gate
    rr = pm / (1 - pm) if pm < 1 else 999
    pm_no = p_no(spot, strike, vol_eff, tau, p_up, k_drift_no)
    if pm_no is None:
        return None
    fee = kalshi_fee(pm)
    net = (pm_no - (1 - pm)) - fee - SLIPPAGE - SPREAD
    if rr < RR_MIN_NO or rr > RR_MAX_NO:
        if net < RR_EDGE_EXC:
            return None
    if net < G3_MIN:
        return None
    won = (resolved == 0)
    return {'side': 'no', 'p_model': pm_no, 'pm': pm, 'net': net,
            'won': won, 'z_abs': z_abs}


def run_yes_only(df, k_drift_yes):
    """Per decision_time: pick best YES candidate only."""
    trades = []
    for _, group in df.groupby('decision_time', sort=True):
        cands = [c for _, row in group.iterrows()
                 for c in [yes_candidate(row, k_drift_yes)] if c]
        if not cands:
            continue
        best = max(cands, key=lambda c: c['net'])
        bet = kelly_bet(best['p_model'], best['pm'], 'yes')
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, 'yes', best['pm'], best['won'])
        trades.append((best['won'], pnl, best['p_model'], best['pm']))
    return trades


def run_no_only(df, k_drift_no):
    """Per decision_time: pick best NO candidate only."""
    trades = []
    for _, group in df.groupby('decision_time', sort=True):
        cands = [c for _, row in group.iterrows()
                 for c in [no_candidate(row, k_drift_no)] if c]
        if not cands:
            continue
        best = max(cands, key=lambda c: c['net'])
        bet = kelly_bet(best['p_model'], best['pm'], 'no')
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, 'no', best['pm'], best['won'])
        trades.append((best['won'], pnl, best['p_model'], best['pm']))
    return trades


def run_combined(df, k_drift_yes, k_drift_no):
    """Per decision_time: evaluate YES and NO independently, pick best."""
    trades = {'yes': [], 'no': []}
    for _, group in df.groupby('decision_time', sort=True):
        yes_cands = [c for _, row in group.iterrows()
                     for c in [yes_candidate(row, k_drift_yes)] if c]
        no_cands  = [c for _, row in group.iterrows()
                     for c in [no_candidate(row, k_drift_no)] if c]
        all_cands = yes_cands + no_cands
        if not all_cands:
            continue
        best = max(all_cands, key=lambda c: c['net'])
        bet = kelly_bet(best['p_model'], best['pm'], best['side'])
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, best['side'], best['pm'], best['won'])
        trades[best['side']].append((best['won'], pnl))
    return trades


def summarize(trades, label):
    if not trades:
        print(f"  {label}: no trades")
        return 0.0
    n = len(trades)
    wins = sum(1 for w, *_ in trades if w)
    pnl = sum(p for _, p, *_ in trades)
    print(f"  {label}: n={n:3d}  WR={wins/n:.1%}  PnL={pnl:+.2f}")
    return pnl


def calibration(trades, label):
    if not trades:
        return
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
    print(f"  Calibration [{label}]:")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        sub = [(w, pnl, pm, mkt) for (w, pnl, pm, mkt) in trades
               if lo <= pm < hi]
        if len(sub) < 3:
            continue
        n = len(sub)
        wr = sum(1 for w, *_ in sub if w) / n
        avg_pm = sum(pm for _, _, pm, _ in sub) / n
        bias = avg_pm - wr
        flag = " <--" if abs(bias) > 0.07 else ""
        print(f"    [{lo:.2f},{hi:.2f})  n={n:3d}  WR={wr:.1%}  model={avg_pm:.2f}  bias={bias:+.2f}{flag}")


def load_archive():
    csv = Path(__file__).parent / 'results' / 'paper_trades.csv'
    df = pd.read_csv(csv, low_memory=False)
    df = df[df['resolved_yes'].notna()].copy()
    for col in ['resolved_yes','p_market','vol_eff','tau_minutes',
                'composite_p_up','vol_score','spot','strike']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['resolved_yes','p_market','vol_eff','tau_minutes',
                            'composite_p_up','spot','strike'])
    df = df[df['decision'] == 'trade']
    return df


def main():
    df = load_archive()
    print(f"\nbacktest_dual_model.py — independent YES/NO calibration")
    print(f"Archive: {len(df)} resolved trade-decisions\n")

    # ── YES sweep ────────────────────────────────────────────────────────
    print("=" * 60)
    print("YES MODEL SWEEP  (YES bets only, k_drift_no irrelevant)")
    print("=" * 60)
    yes_sweep = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    best_yes_pnl = -9e9
    best_k_yes = None
    yes_results = {}
    print(f"{'k_yes':>6}  {'n':>4}  {'WR':>6}  {'PnL':>9}")
    print("-" * 35)
    for k in yes_sweep:
        trades = run_yes_only(df, k)
        if not trades:
            print(f"  {k:.1f}   no trades")
            continue
        n = len(trades); wins = sum(1 for w,*_ in trades if w)
        pnl = sum(p for _,p,*_ in trades)
        yes_results[k] = trades
        marker = " <-- best" if pnl > best_yes_pnl else ""
        if pnl > best_yes_pnl:
            best_yes_pnl = pnl; best_k_yes = k
        print(f"  {k:.1f}  {n:4d}  {wins/n:6.1%}  {pnl:+9.2f}{marker}")

    print(f"\n  => Best k_drift_yes = {best_k_yes} (YES PnL = {best_yes_pnl:+.2f})\n")

    # Calibration for best and current
    for k in [best_k_yes, 0.8]:
        if k in yes_results:
            calibration(yes_results[k], f"k_drift_yes={k}")

    # ── NO sweep ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NO MODEL SWEEP  (NO bets only, k_drift_yes irrelevant)")
    print("=" * 60)
    no_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    best_no_pnl = -9e9
    best_k_no = None
    no_results = {}
    print(f"{'k_no':>5}  {'n':>4}  {'WR':>6}  {'PnL':>9}")
    print("-" * 35)
    for k in no_sweep:
        trades = run_no_only(df, k)
        if not trades:
            print(f"  {k:.1f}   no trades")
            continue
        n = len(trades); wins = sum(1 for w,*_ in trades if w)
        pnl = sum(p for _,p,*_ in trades)
        no_results[k] = trades
        marker = " <-- best" if pnl > best_no_pnl else ""
        if pnl > best_no_pnl:
            best_no_pnl = pnl; best_k_no = k
        print(f"  {k:.1f}  {n:4d}  {wins/n:6.1%}  {pnl:+9.2f}{marker}")

    print(f"\n  => Best k_drift_no = {best_k_no} (NO PnL = {best_no_pnl:+.2f})\n")

    for k in [best_k_no, 0.8]:
        if k in no_results:
            calibration(no_results[k], f"k_drift_no={k}")

    # ── Combined run with optimal params ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"COMBINED: k_drift_yes={best_k_yes}  k_drift_no={best_k_no}")
    print("=" * 60)
    combined = run_combined(df, best_k_yes, best_k_no)
    y = combined['yes']; no = combined['no']
    all_t = y + no
    n_all = len(all_t)
    wins_all = sum(1 for w,_ in all_t if w)
    pnl_all = sum(p for _,p in all_t)
    ny = len(y); wy = sum(1 for w,_ in y if w); py = sum(p for _,p in y)
    nn = len(no); wn = sum(1 for w,_ in no if w); pn = sum(p for _,p in no)
    print(f"  Total:  n={n_all:3d}  WR={wins_all/n_all:.1%}  PnL={pnl_all:+.2f}")
    print(f"  YES:    n={ny:3d}  WR={wy/ny:.1%}  PnL={py:+.2f}" if ny else "  YES: none")
    print(f"  NO:     n={nn:3d}  WR={wn/nn:.1%}  PnL={pn:+.2f}" if nn else "  NO: none")

    # Baseline: current model (k_yes=0.8, k_no=0.8 dependent)
    print(f"\n  Baseline (k_yes=0.8, k_no=0.8 dependent):")
    base = run_combined(df, 0.8, 0.8)
    by = base['yes']; bn = base['no']
    ba = by + bn
    if ba:
        bw = sum(1 for w,_ in ba if w)
        bp = sum(p for _,p in ba)
        bny = len(by); bwy = sum(1 for w,_ in by if w); bpy = sum(p for _,p in by)
        bnn = len(bn); bwn = sum(1 for w,_ in bn if w); bpn = sum(p for _,p in bn)
        print(f"  Total:  n={len(ba):3d}  WR={bw/len(ba):.1%}  PnL={bp:+.2f}")
        print(f"  YES:    n={bny:3d}  WR={bwy/bny:.1%}  PnL={bpy:+.2f}" if bny else "  YES: none")
        print(f"  NO:     n={bnn:3d}  WR={bwn/bnn:.1%}  PnL={bpn:+.2f}" if bnn else "  NO: none")
        print(f"\n  Delta vs baseline:  PnL {pnl_all - bp:+.2f}")


if __name__ == '__main__':
    main()
