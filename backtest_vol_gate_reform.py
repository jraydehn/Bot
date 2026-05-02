#!/usr/bin/env python3
"""
backtest_vol_gate_reform.py — Simulate vol_factor-as-gate + k_drift sweep for BTC.

Two reforms tested together:
  1. vol_factor-as-gate: remove vol_factor from sigma, use it as a reachability
     gate multiplier instead (block if |z_strike| > BASE_Z_MAX * vol_factor).
  2. k_drift reduction: current k_drift=1.0 is over-weighted relative to the
     actual directional signal strength (p_up vs OTM resolution corr ~0.05-0.06).
     Sweep k_drift in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0}.

Core formula:
  BEFORE: p_model = 1 - Phi(log(K/S)/(sigma*vol_factor*sqrt(tau)) - norm.ppf(p_up)*1.0)
  AFTER:  p_model = 1 - Phi(log(K/S)/(sigma*sqrt(tau)) - norm.ppf(p_up)*k_drift)
  GATE:   block if |log(K/S)/(sigma*sqrt(tau))| > BASE_Z_MAX * vol_factor

vol_factor reconstructed from logged vol_score:
  vol_factor = clip(1.0 + vol_score * 0.08, 0.60, 1.40)

Sweep: k_drift × BASE_Z_MAX (best directional gates fixed at OFF/0.50/0.48/None
from prior run, since directional gate effect was minimal).
"""

import math, sys, time, warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from gate_attribution import (
    load_archive, ASSET_PARAMS, RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC,
    OTM_TIERS, TAPE_THRESHOLDS, severity, kalshi_fee, SLIPPAGE, SPREAD,
    BANKROLL_0, KELLY_MULT, KELLY_CAP,
)
from gate_attribution_v2 import kelly_bet_flat, trade_pnl, evaluate_row_v2, run_v2

# Production baseline gates
ALL_GATES_BASELINE = {
    "G0_pm", "GCS", "GCI", "GNS", "GOTM", "G3", "GRR",
    "Gpm15_btc", "Gtape", "Gpup_btc",
}

# Vol_factor reconstruction constants (from vol_layer.py)
VOL_VOTE_STEP = 0.08
VOL_FACTOR_MIN = 0.60
VOL_FACTOR_MAX = 1.40


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def reconstruct_vol_factor(vol_score) -> float:
    """Reconstruct vol_factor from logged vol_score integer."""
    try:
        s = int(vol_score)
    except (TypeError, ValueError):
        return 1.0
    return float(np.clip(1.0 + s * VOL_VOTE_STEP, VOL_FACTOR_MIN, VOL_FACTOR_MAX))


def compute_p_model(
    spot: float, strike: float, vol_eff: float, tau: float, p_up: float,
    k_drift: float = 1.0,
) -> float:
    """
    Drift-adjusted p_model WITHOUT vol_factor in sigma.
    sigma_tau = vol_eff * sqrt(tau)   (no vol_factor multiplier)
    z_adj     = log(K/S)/sigma_tau - norm.ppf(p_up) * k_drift
    k_drift=0.0 → pure log-normal; k_drift=1.0 → full production drift.
    """
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0:
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    z_drift  = norm.ppf(p_up) * k_drift
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


# ------------------------------------------------------------------
# Evaluate one row under the reformed model
# ------------------------------------------------------------------

def evaluate_row_reformed(
    row,
    base_z_max,   # None = OFF; gate threshold = base_z_max * vol_factor
    k_drift,      # drift weight: 0.0 = pure log-normal, 1.0 = full drift
):
    p = ASSET_PARAMS["BTC"]
    spot      = row["spot"]
    strike    = row["strike"]
    pm        = row["p_market"]
    vol_eff   = row["vol_eff"]
    tau       = row["tau_minutes"]
    p_up      = row["composite_p_up"]
    vol_score = row.get("vol_score", 0)
    resolved_yes = int(row["resolved_yes"])

    if vol_eff <= 0 or tau <= 0 or pm <= 0 or pm >= 1:
        return None

    offset = (strike - spot) / spot if spot > 0 else 0.0
    vf     = reconstruct_vol_factor(vol_score)

    # Vol gate: block if |z_strike| > base_z_max * vol_factor
    if base_z_max is not None:
        sigma_tau = vol_eff * math.sqrt(tau)
        if sigma_tau > 0:
            z_abs = abs(math.log(strike / spot) / sigma_tau)
            if z_abs > base_z_max * vf:
                return None

    # Compute p_model with swept k_drift, no vol_factor in sigma
    p_model = compute_p_model(spot, strike, vol_eff, tau, p_up, k_drift)

    best = None
    for side in ("yes", "no"):
        # Gate 0: p_market bounds
        if not (p["pm_min"] <= pm <= p["pm_max"]):
            continue

        # Edge math
        fee = kalshi_fee(pm)
        if side == "yes":
            net = (p_model - pm) - fee - SLIPPAGE - SPREAD
            rr  = pm / (1 - pm) if pm < 1 else 999
            if rr > RR_MAX_YES:
                continue
            tier_min = 0.0
            for thr, mn in OTM_TIERS:
                if pm < thr:
                    tier_min = mn; break
            if net < tier_min:
                continue
        else:
            net = (pm - p_model) - fee - SLIPPAGE - SPREAD
            rr  = (1 - pm) / pm if pm > 0 else 999
            if (rr < RR_MIN_NO or rr > RR_MAX_NO) and net < RR_EDGE_EXC:
                continue

        # Gate 3: minimum net edge
        if net < p["g3_min"]:
            continue

        won = (resolved_yes == 1 and side == "yes") or (resolved_yes == 0 and side == "no")
        if best is None or net > best["net"]:
            best = {
                "side": side, "pm": pm, "p_model": p_model, "net": net,
                "offset": offset, "won": won, "p_up": p_up, "vf": vf,
            }
    return best


# ------------------------------------------------------------------
# Run one configuration
# ------------------------------------------------------------------

def run_reformed(df, base_z_max, k_drift):
    pnls = []
    for _dt, group in df.groupby("decision_time", sort=True):
        cands = []
        for _, row in group.iterrows():
            c = evaluate_row_reformed(row, base_z_max, k_drift)
            if c is not None:
                cands.append(c)
        if not cands:
            continue
        best = max(cands, key=lambda c: c["net"])
        bet  = kelly_bet_flat(best["p_model"], best["pm"], best["side"])
        if bet <= 0:
            continue
        pnl = trade_pnl(bet, best["side"], best["pm"], best["won"])
        pnls.append({
            "won": best["won"], "pnl": pnl, "side": best["side"],
            "pm": best["pm"], "p_model": best["p_model"],
            "offset": best["offset"], "p_up": best["p_up"],
            "vf": best["vf"],
        })
    if not pnls:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "trades": []}
    df_t = pd.DataFrame(pnls)
    return {
        "n": len(df_t),
        "wr": df_t["won"].mean(),
        "pnl": df_t["pnl"].sum(),
        "trades": pnls,
    }


# ------------------------------------------------------------------
# Calibration curve
# ------------------------------------------------------------------

def calibration_curve(trades, label=""):
    if not trades:
        return
    df = pd.DataFrame(trades)
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    df["p_bin"] = pd.cut(df["p_model"], bins=bins, right=False)
    print(f"\n  Calibration curve{' — ' + label if label else ''}:")
    print(f"  {'p_model bin':<22} {'n':>5}  {'WR':>6}  {'diff':>7}  PnL")
    for b, g in df.groupby("p_bin", observed=True):
        mid  = (b.left + b.right) / 2
        wr   = g["won"].mean()
        diff = wr - mid
        pnl  = g["pnl"].sum()
        marker = " <-- overestimates" if diff < -0.08 else (" <-- underestimates" if diff > 0.08 else "")
        print(f"  {str(b):<22} {len(g):>5}  {wr:>6.1%}  {diff:>+6.1%}  {pnl:>+8.2f}{marker}")


# ------------------------------------------------------------------
# Breakdown by side, offset, pm, vol_factor
# ------------------------------------------------------------------

def breakdown(trades, label=""):
    if not trades:
        return
    df = pd.DataFrame(trades)
    print(f"\n  Breakdown{' — ' + label if label else ''}:")

    print("  [by side]")
    for side, g in df.groupby("side"):
        print(f"    {side.upper()}: n={len(g):3d}  WR={g.won.mean():.0%}  PnL={g.pnl.sum():+.2f}")

    print("  [YES by offset]")
    yes = df[df["side"] == "yes"].copy()
    if len(yes):
        yes["obin"] = pd.cut(yes["offset"], bins=[-1, -0.02, -0.005, 0, 0.005, 0.01, 0.02, 1],
                             labels=["ITM>2%", "ITM<2%", "ATM-", "ATM+", "<+1%", "1-2%", ">2%"])
        for b, g in yes.groupby("obin", observed=True):
            print(f"    {str(b):<10}: n={len(g):3d}  WR={g.won.mean():.0%}  PnL={g.pnl.sum():+.2f}")

    print("  [NO by pm]")
    no = df[df["side"] == "no"].copy()
    if len(no):
        no["pmbin"] = pd.cut(no["pm"], bins=[0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1])
        for b, g in no.groupby("pmbin", observed=True):
            be = 1 - g["pm"].mean()
            print(f"    pm={str(b):<18}: n={len(g):3d}  WR={g.won.mean():.0%}  BE={be:.0%}  PnL={g.pnl.sum():+.2f}")

    print("  [by vol_factor bucket]")
    df["vf_bin"] = pd.cut(df["vf"],
                           bins=[0.55, 0.75, 0.90, 1.05, 1.20, 1.45],
                           labels=["0.60-0.74", "0.75-0.89", "0.90-1.04", "1.05-1.19", "1.20-1.40"])
    for b, g in df.groupby("vf_bin", observed=True):
        print(f"    vf={str(b):<12}: n={len(g):3d}  WR={g.won.mean():.0%}  PnL={g.pnl.sum():+.2f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=== BTC Reform: vol_factor-as-gate + k_drift sweep ===\n")
    print("  Formula: p_model = 1 - Phi(log(K/S)/(vol_eff*sqrt(tau)) - norm.ppf(p_up)*k_drift)")
    print("  Gate:    block if |z_strike| > BASE_Z_MAX * vol_factor\n")

    btc = load_archive("BTC")
    if btc.empty:
        print("ERROR: BTC archive is empty."); return
    print(f"  Archive: {len(btc):,} scans, {btc['decision_time'].nunique():,} hours\n")

    # ---- Production baseline ----
    print("=== Production baseline (k_drift=1.0, vol_factor in sigma, full gates) ===")
    base = run_v2("BTC", btc, ALL_GATES_BASELINE)
    print(f"  n={base['n']:4d}  WR={base['wr']:.1%}  PnL={base['pnl']:+.2f}\n")

    # ---- k_drift × BASE_Z_MAX sweep ----
    k_drifts   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    base_z_maxes = [None, 0.5, 1.0, 1.5, 2.0]

    print("=== k_drift × BASE_Z_MAX sweep ===")
    print(f"  {'k_drift':<9} {'BASE_Z':<8} {'n':>5} {'WR':>6} {'YES_n':>6} {'YES_WR':>7} "
          f"{'YES_PnL':>9} {'NO_n':>5} {'NO_WR':>6} {'NO_PnL':>9} {'Total_PnL':>11}")

    results = []
    for kd, bz in product(k_drifts, base_z_maxes):
        r = run_reformed(btc, bz, kd)
        df_t = pd.DataFrame(r["trades"]) if r["trades"] else pd.DataFrame()
        yes_n = no_n = yes_wr = no_wr = yes_pnl = no_pnl = 0
        if not df_t.empty:
            yes = df_t[df_t.side == "yes"]
            no  = df_t[df_t.side == "no"]
            yes_n   = len(yes); yes_wr  = yes.won.mean() if yes_n else 0; yes_pnl = yes.pnl.sum() if yes_n else 0
            no_n    = len(no);  no_wr   = no.won.mean()  if no_n  else 0; no_pnl  = no.pnl.sum()  if no_n  else 0
        results.append({"kd": kd, "bz": bz, "n": r["n"], "wr": r["wr"], "pnl": r["pnl"],
                        "yes_n": yes_n, "yes_wr": yes_wr, "yes_pnl": yes_pnl,
                        "no_n": no_n, "no_wr": no_wr, "no_pnl": no_pnl,
                        "trades": r["trades"]})
        z_str = f"{bz:.1f}x" if bz is not None else "OFF"
        print(f"  {kd:<9.2f} {z_str:<8} {r['n']:>5} {r['wr']:>6.1%} "
              f"{yes_n:>6} {yes_wr:>7.1%} {yes_pnl:>+9.2f} "
              f"{no_n:>5} {no_wr:>6.1%} {no_pnl:>+9.2f} {r['pnl']:>+11.2f}")

    # ---- Best overall ----
    best = max(results, key=lambda x: x["pnl"])
    z_str = f"{best['bz']:.1f}x" if best["bz"] is not None else "OFF"
    print(f"\n=== Best: k_drift={best['kd']}  BASE_Z={z_str} ===")
    print(f"  n={best['n']:4d}  WR={best['wr']:.1%}  PnL={best['pnl']:+.2f}")
    calibration_curve(best["trades"], label=f"k_drift={best['kd']} BASE_Z={z_str}")
    breakdown(best["trades"], label=f"k_drift={best['kd']} BASE_Z={z_str}")

    # ---- k_drift sensitivity at best BASE_Z ----
    best_bz = best["bz"]
    same_bz = [r for r in results if r["bz"] == best_bz]
    print(f"\n=== k_drift sensitivity at BASE_Z={z_str} ===")
    print(f"  {'k_drift':<9} {'n':>5} {'WR':>6} {'YES_PnL':>9} {'NO_PnL':>9} {'Total':>10}")
    for r in sorted(same_bz, key=lambda x: x["kd"]):
        print(f"  {r['kd']:<9.2f} {r['n']:>5} {r['wr']:>6.1%} "
              f"{r['yes_pnl']:>+9.2f} {r['no_pnl']:>+9.2f} {r['pnl']:>+10.2f}")

    # ---- Summary ----
    print(f"\n=== Reform vs Baseline ===")
    print(f"  Baseline (k_drift=1.0, vol_factor in sigma):  "
          f"n={base['n']:4d}  WR={base['wr']:.1%}  PnL={base['pnl']:+.2f}")
    print(f"  Best reform (k_drift={best['kd']}, BASE_Z={z_str}):  "
          f"n={best['n']:4d}  WR={best['wr']:.1%}  PnL={best['pnl']:+.2f}")
    delta = best["pnl"] - base["pnl"]
    print(f"  Delta: {delta:+.2f}  ({'improvement' if delta > 0 else 'regression'})")

    print(f"\n  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
