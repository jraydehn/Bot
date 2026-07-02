"""
simulate_zdrift_eth_sol.py — Walk-forward simulation of empirical z_drift for ETH and SOL.

Tests whether replacing the composite p_up/k_drift path with a rolling empirical z_drift
(computed from realized trade outcomes) improves model performance for ETH and SOL.

For each weekly test fold:
  - z_drift computed from all resolved trades BEFORE that week (no lookahead)
  - p_yes_new / p_no_new recomputed via score_to_p_model/score_to_p_no_model with z_drift_override
  - Compare baseline PnL vs new model (trades blocked vs kept)

Flat $1000 bankroll throughout.
"""

import sys
import math

sys.path.insert(0, '/Users/justindehn/Documents/ClaudeCode/kalshi_btc')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm
from evaluate_point import load_data
from composite_scorer import score_to_p_model, score_to_p_no_model
from paper_trade_runner import compute_zdrift_empirical

MIN_EDGE = 0.04   # standard trade threshold

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def week_label(period):
    return f"{period.start_time.strftime('%b %d')}-{period.end_time.strftime('%b %d')}"


def run_asset(asset: str, df_confirm):
    fname = {'ETH': 'paper_trades_eth.csv', 'SOL': 'paper_trades_sol.csv'}[asset]
    df = pd.read_csv(
        f'/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/{fname}',
        low_memory=False
    )
    df['logged_at'] = pd.to_datetime(df['logged_at'], utc=True, errors='coerce')

    # All resolved rows for building z_drift history
    df_resolved_all = df[df['resolved_yes'].notna()].copy()

    # Actual traded rows with resolution
    df_trades = df[(df['decision'] == 'trade') & df['resolved_yes'].notna()].copy()
    df_trades['week'] = df_trades['logged_at'].dt.to_period('W')

    weeks = sorted(df_trades['week'].unique())
    # Use all 5 weeks (walk-forward: first week may have sparse history, still valid)
    test_weeks = weeks  # all available

    print(f"\n{'='*80}")
    print(f"  {asset} — Walk-Forward z_drift Simulation (MIN_EDGE={MIN_EDGE})")
    print(f"{'='*80}")
    print(f"  Total resolved trades: {len(df_resolved_all)}")
    print(f"  Total traded+resolved: {len(df_trades)}")
    print(f"  Test weeks: {[week_label(w) for w in test_weeks]}")
    print()

    # Header
    hdr = (
        f"{'Week':<20} {'z_drift':>8} {'n':>5} "
        f"{'base_$':>10} {'new_$':>10} {'delta_$':>10} "
        f"{'blocked':>8} {'W_blk':>7} {'L_blk':>7}"
    )
    print(hdr)
    print('-' * len(hdr))

    total_baseline = 0.0
    total_new = 0.0
    total_blocked = total_wins_blk = total_losses_blk = 0
    fold_results = []

    for week in test_weeks:
        week_trades = df_trades[df_trades['week'] == week].copy()
        week_start  = week.start_time.tz_localize('UTC')

        # z_drift from all resolved trades BEFORE this week (walk-forward, no lookahead)
        df_prior = df_resolved_all[df_resolved_all['logged_at'] < week_start]
        zdrift = compute_zdrift_empirical(
            df_prior, df_confirm,
            w_short=10, w_long=30, alpha=0.6, cap=0.5
        )

        baseline_pnl = 0.0
        new_pnl      = 0.0
        wins_blocked = losses_blocked = trades_blocked = 0

        # YES / NO breakdown
        yes_base = yes_new = no_base = no_new = 0.0
        yes_blk_w = yes_blk_l = no_blk_w = no_blk_l = 0

        for _, row in week_trades.iterrows():
            try:
                spot     = float(row['spot'])
                strike   = float(row['strike'])
                vol_eff  = float(row['vol_eff'])
                tau_min  = float(row['tau_minutes'])
                sigma_tau = vol_eff * math.sqrt(tau_min) if vol_eff > 0 and tau_min > 0 else 0.0
                trend    = int(row.get('composite_trend', 0) or 0)
                rev      = int(row.get('composite_rev',   0) or 0)
                p_up     = float(row.get('composite_p_up', 0.5) or 0.5)
                p_mkt    = float(row['p_market'])
                side     = str(row['side']).lower()
                pnl      = float(row['would_pnl'])
                won      = bool(row['would_win'])
            except Exception:
                baseline_pnl += float(row.get('would_pnl', 0))
                new_pnl      += float(row.get('would_pnl', 0))
                continue

            baseline_pnl += pnl
            if side == 'yes':
                yes_base += pnl
            else:
                no_base += pnl

            if sigma_tau <= 0:
                new_pnl += pnl
                if side == 'yes':
                    yes_new += pnl
                else:
                    no_new += pnl
                continue

            # New model probabilities with z_drift_override
            p_yes_new = score_to_p_model(
                trend, rev, spot, strike, sigma_tau,
                asset=asset, p_up_override=p_up, z_drift_override=zdrift
            )
            p_no_new = score_to_p_no_model(
                trend, rev, spot, strike, sigma_tau,
                asset=asset, p_up_override=p_up, z_drift_override=zdrift
            )

            if side == 'yes':
                new_edge = p_yes_new - p_mkt
            else:
                new_edge = p_no_new - (1.0 - p_mkt)

            if new_edge < MIN_EDGE:
                trades_blocked += 1
                if won:
                    wins_blocked   += 1
                    if side == 'yes':
                        yes_blk_w += 1
                    else:
                        no_blk_w  += 1
                else:
                    losses_blocked += 1
                    if side == 'yes':
                        yes_blk_l += 1
                    else:
                        no_blk_l  += 1
                # P&L: blocked trade contributes 0
            else:
                new_pnl += pnl
                if side == 'yes':
                    yes_new += pnl
                else:
                    no_new += pnl

        delta = new_pnl - baseline_pnl
        total_baseline   += baseline_pnl
        total_new        += new_pnl
        total_blocked    += trades_blocked
        total_wins_blk   += wins_blocked
        total_losses_blk += losses_blocked

        n = len(week_trades)
        label = week_label(week)
        prior_n = len(df_prior)

        print(
            f"{label:<20} {zdrift:>+8.4f} {n:>5} "
            f"{baseline_pnl:>+10.2f} {new_pnl:>+10.2f} {delta:>+10.2f} "
            f"{trades_blocked:>8} {wins_blocked:>7} {losses_blocked:>7}"
        )

        fold_results.append(dict(
            week=label, zdrift=zdrift, n=n, prior_n=prior_n,
            baseline_pnl=baseline_pnl, new_pnl=new_pnl, delta=delta,
            trades_blocked=trades_blocked, wins_blocked=wins_blocked, losses_blocked=losses_blocked,
            yes_base=yes_base, yes_new=yes_new,
            no_base=no_base,  no_new=no_new,
            yes_blk_w=yes_blk_w, yes_blk_l=yes_blk_l,
            no_blk_w=no_blk_w,  no_blk_l=no_blk_l,
        ))

    # Totals row
    total_delta = total_new - total_baseline
    total_n = sum(r['n'] for r in fold_results)
    print('-' * len(hdr))
    print(
        f"{'TOTAL':<20} {'':>8} {total_n:>5} "
        f"{total_baseline:>+10.2f} {total_new:>+10.2f} {total_delta:>+10.2f} "
        f"{total_blocked:>8} {total_wins_blk:>7} {total_losses_blk:>7}"
    )

    # ── YES vs NO breakdown ──
    print()
    print(f"  {asset} — YES vs NO Breakdown by Week")
    hdr2 = (
        f"{'Week':<20} {'z_drift':>8} "
        f"{'YES_base':>10} {'YES_new':>10} {'YES_delta':>10} {'Y_blk(W/L)':>12} "
        f"{'NO_base':>10}  {'NO_new':>10}  {'NO_delta':>10}  {'N_blk(W/L)':>12}"
    )
    print(hdr2)
    print('-' * len(hdr2))

    tot_yes_base = tot_yes_new = tot_no_base = tot_no_new = 0.0
    for r in fold_results:
        yes_delta = r['yes_new'] - r['yes_base']
        no_delta  = r['no_new']  - r['no_base']
        print(
            f"{r['week']:<20} {r['zdrift']:>+8.4f} "
            f"{r['yes_base']:>+10.2f} {r['yes_new']:>+10.2f} {yes_delta:>+10.2f} {r['yes_blk_w']:>5}/{r['yes_blk_l']:<5} "
            f"{r['no_base']:>+10.2f}  {r['no_new']:>+10.2f}  {no_delta:>+10.2f}  {r['no_blk_w']:>5}/{r['no_blk_l']:<5}"
        )
        tot_yes_base += r['yes_base']
        tot_yes_new  += r['yes_new']
        tot_no_base  += r['no_base']
        tot_no_new   += r['no_new']

    print('-' * len(hdr2))
    print(
        f"{'TOTAL':<20} {'':>8} "
        f"{tot_yes_base:>+10.2f} {tot_yes_new:>+10.2f} {tot_yes_new-tot_yes_base:>+10.2f} {'':>12} "
        f"{tot_no_base:>+10.2f}  {tot_no_new:>+10.2f}  {tot_no_new-tot_no_base:>+10.2f}  {'':>12}"
    )

    # ── Summary ──
    print()
    print(f"  {asset} SUMMARY")
    print(f"    Baseline total P&L : ${total_baseline:>+,.2f}")
    print(f"    New model total P&L : ${total_new:>+,.2f}")
    print(f"    Net delta           : ${total_delta:>+,.2f}  ({total_delta/max(abs(total_baseline),1)*100:+.1f}%)")
    print(f"    Total trades        : {total_n}")
    print(f"    Trades blocked      : {total_blocked} ({total_blocked/max(total_n,1)*100:.1f}%)")
    print(f"    Wins blocked        : {total_wins_blk}")
    print(f"    Losses blocked      : {total_losses_blk}")
    if total_wins_blk + total_losses_blk > 0:
        print(f"    Blocked WR          : {total_wins_blk/(total_wins_blk+total_losses_blk)*100:.1f}%  (ideal: <50%)")

    # z_drift progression
    print()
    print(f"  {asset} — z_drift progression (prior data size):")
    for r in fold_results:
        print(f"    {r['week']:<20}  z_drift={r['zdrift']:>+.4f}  (n_prior={r['prior_n']})")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")

    print("\n[ETH] Loading 1h confirmation data...")
    _, df_confirm_eth, _ = load_data(asset='ETH')

    print("\n[SOL] Loading 1h confirmation data...")
    _, df_confirm_sol, _ = load_data(asset='SOL')

    eth_results = run_asset('ETH', df_confirm_eth)
    sol_results = run_asset('SOL', df_confirm_sol)

    # Cross-asset comparison
    print(f"\n{'='*80}")
    print("  CROSS-ASSET SUMMARY")
    print(f"{'='*80}")
    for asset, results in [('ETH', eth_results), ('SOL', sol_results)]:
        base = sum(r['baseline_pnl'] for r in results)
        new  = sum(r['new_pnl']      for r in results)
        delta = new - base
        blk = sum(r['trades_blocked'] for r in results)
        n   = sum(r['n']              for r in results)
        print(f"  {asset:4s}  baseline={base:>+10,.2f}  new={new:>+10,.2f}  delta={delta:>+10,.2f}  blocked={blk}/{n}")

    print()
    print("Done.")
