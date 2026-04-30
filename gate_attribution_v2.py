#!/usr/bin/env python3
"""
gate_attribution_v2.py — Re-verify the four committed gate decisions with:

  1. Logged p_yes_model from archive (not recomputed log-normal+drift).
     This is critical for ETH/SOL because production uses HistGradientBoosting
     direct models there, not log-normal. v1 harness silently used log-normal
     for all three assets — invalidating ETH/SOL deltas.

  2. Flat $1,000 bankroll per trade (no compounding). Compounding Kelly
     inflated v1 dollar magnitudes 3–9× without changing rankings.

  3. Corrected BTC Gpup rescue: (funding==0 AND struct>=0) OR comp_rev>0
     (v1 had net_edge>=0.04 — wrong).

What's still NOT modeled (deferred to v3 if BTC counter-tape number shifts):
  - BTC vol_score=1 block + ema_stack rescue
  - BTC NO edge<0.02 + comp_rev/vol_ratio rescue
  - BTC spread>=0.04 + chg_10m rescue
  - BTC tau<30 + p_up/kelly rescue
  - Streak v2 directional rescues
  - Pure-edge override (8% bypass)

These are all BTC-only gates; their absence affects the BTC counter-tape
result baseline only. ETH/SOL are unaffected.
"""

import math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gate_attribution import (
    load_archive, ASSET_PARAMS, GCS_MIN, GCI_MIN_BEARISH,
    RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC,
    OTM_TIERS, TAPE_THRESHOLDS, severity, kalshi_fee, SLIPPAGE, SPREAD,
    BANKROLL_0, KELLY_MULT, KELLY_CAP,
)

ALL_GATES = {
    "G0_pm", "GCS", "GCI", "GNS", "GOTM", "G3", "GRR",
    "Gpm15_btc", "Gpm45_eth", "Gtape", "Gpup_btc",
}


def evaluate_row_v2(row, asset, gates, ns_max_override=None):
    """Same shape as v1 evaluate_row but uses logged p_yes_model and the
    corrected Gpup_btc rescue."""
    p = ASSET_PARAMS[asset]
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    try:
        p_model = float(row["p_yes_model"])   # ← logged, not recomputed (coerce str→float)
    except (TypeError, ValueError):
        return None
    p_up = row["composite_p_up"]
    resolved_yes = int(row["resolved_yes"])
    if pd.isna(p_model) or pm <= 0 or pm >= 1:
        return None
    offset = (strike - spot) / spot if spot > 0 else 0.0
    ns_max = ns_max_override if ns_max_override is not None else p["ns_max"]

    # Gpup_btc rescue features
    funding = row.get("funding_bias", 0)
    struct = row.get("structure_bias", 0)
    comp_rev = row.get("composite_rev", 0)
    if pd.isna(funding): funding = 0
    if pd.isna(struct): struct = 0
    if pd.isna(comp_rev): comp_rev = 0
    btc_pup_rescue = (
        (int(funding) == 0 and int(struct) >= 0) or float(comp_rev) > 0
    )

    best = None
    for side in ("yes", "no"):
        if "G0_pm" in gates and not (p["pm_min"] <= pm <= p["pm_max"]):
            continue
        if side == "yes" and asset == "BTC" and "Gpm15_btc" in gates and pm < 0.15:
            continue
        if side == "yes" and asset == "ETH" and "Gpm45_eth" in gates \
                and offset > 0 and pm < 0.45:
            continue
        if side == "yes":
            if "GCS" in gates and offset > 0 and p_up < GCS_MIN: continue
            if "GCI" in gates and offset <= 0 and p_up < GCI_MIN_BEARISH: continue
        if side == "no":
            if "GNS" in gates and offset < 0 and p_up > ns_max: continue
        # Gpup_btc with CORRECTED rescue
        if side == "yes" and asset == "BTC" and "Gpup_btc" in gates \
                and p_up < 0.52 and not btc_pup_rescue:
            continue

        fee = kalshi_fee(pm)
        if side == "yes":
            net = (p_model - pm) - fee - SLIPPAGE - SPREAD
            rr = pm / (1 - pm) if pm < 1 else 999
            if "GRR" in gates and rr > RR_MAX_YES: continue
            if "GOTM" in gates:
                tier_min = 0.0
                for thr, mn in OTM_TIERS:
                    if pm < thr:
                        tier_min = mn; break
                if net < tier_min: continue
        else:
            net = (pm - p_model) - fee - SLIPPAGE - SPREAD
            rr = (1 - pm) / pm if pm > 0 else 999
            if "GRR" in gates and (rr < RR_MIN_NO or rr > RR_MAX_NO) \
                    and net < RR_EDGE_EXC:
                continue
        if "G3" in gates and net < p["g3_min"]:
            continue

        won = (resolved_yes == 1 and side == "yes") or (resolved_yes == 0 and side == "no")
        if best is None or net > best["net"]:
            best = {"side": side, "pm": pm, "p_model": p_model, "net": net,
                    "offset": offset, "won": won, "p_up": p_up,
                    "chg_5m": row.get("chg_5m", 0), "chg_10m": row.get("chg_10m", 0),
                    "chg_30m": row.get("chg_30m", 0)}
    return best


def kelly_bet_flat(p_model, pm, side, kelly_scale=1.0):
    """Flat bankroll — bet sized as fraction of fixed BANKROLL_0."""
    if side == "yes":
        b = (1 - pm) / pm if pm > 0 else 0
        p, q = p_model, 1 - p_model
    else:
        b = pm / (1 - pm) if pm < 1 else 0
        p, q = 1 - p_model, p_model
    if b <= 0:
        return 0.0
    kf = max(0.0, (b * p - q) / b)
    bf = min(kf * KELLY_MULT * kelly_scale, KELLY_CAP)
    return round(BANKROLL_0 * bf, 2)


def trade_pnl(bet, side, pm, won):
    if bet <= 0: return 0.0
    fee = kalshi_fee(pm)
    if side == "yes":
        if won:
            n_ct = bet / pm if pm > 0 else 0
            return bet * (1 - pm) / pm - fee * n_ct
        return -bet
    else:
        if won:
            n_ct = bet / (1 - pm) if pm < 1 else 0
            return bet * pm / (1 - pm) - fee * n_ct
        return -bet


def run_v2(asset, df, gates, ns_max_override=None, tape_thr=None):
    thr = tape_thr if tape_thr is not None else TAPE_THRESHOLDS[asset]
    pnls = []
    for dt_, group in df.groupby("decision_time", sort=True):
        cands = []
        for _, row in group.iterrows():
            c = evaluate_row_v2(row, asset, gates, ns_max_override=ns_max_override)
            if c is not None: cands.append(c)
        if not cands: continue
        best = max(cands, key=lambda c: c["net"])
        sev = severity(best["side"], best["chg_5m"], best["chg_10m"],
                       best["chg_30m"], thr) if "Gtape" in gates else 0.0
        kelly_scale = 1.0
        if "Gtape" in gates and sev >= 1.5: continue
        if "Gtape" in gates and sev >= 0.5:
            kelly_scale = max(0.25, 1.0 - (sev - 0.5) * 0.75)
        bet = kelly_bet_flat(best["p_model"], best["pm"], best["side"], kelly_scale)
        if bet <= 0: continue
        pnl = trade_pnl(bet, best["side"], best["pm"], best["won"])
        pnls.append((best["won"], pnl))
    if not pnls:
        return {"n": 0, "wr": 0.0, "pnl": 0.0}
    n = len(pnls)
    wins = sum(1 for w, _ in pnls if w)
    pnl = sum(p for _, p in pnls)
    return {"n": n, "wr": wins / n, "pnl": pnl}


def main():
    print(f"\ngate_attribution_v2.py — logged p_model + flat bankroll + corrected Gpup rescue\n")

    btc = load_archive("BTC")
    eth = load_archive("ETH")
    sol = load_archive("SOL")

    # ETH GNS sweep
    print(f"=== ETH GNS sweep (logged p_model, flat $1k) ===")
    print(f"  thr     n     WR      pnl")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        r = run_v2("ETH", eth, ALL_GATES, ns_max_override=thr)
        print(f"  {thr:.2f}   {r['n']:3d}   {r['wr']:.1%}   ${r['pnl']:+8.2f}")

    print(f"\n=== SOL GNS sweep (logged p_model, flat $1k) ===")
    print(f"  thr     n     WR      pnl")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        r = run_v2("SOL", sol, ALL_GATES, ns_max_override=thr)
        print(f"  {thr:.2f}   {r['n']:3d}   {r['wr']:.1%}   ${r['pnl']:+8.2f}")

    # Counter-tape sweep, all three assets
    print(f"\n=== Counter-tape sweep (logged p_model, flat $1k) ===")
    for asset, df in (("BTC", btc), ("ETH", eth), ("SOL", sol)):
        print(f" [{asset}]")
        print(f"   mult       n     WR      pnl")
        orig = TAPE_THRESHOLDS[asset]
        for label, k in (("OFF", None), ("×0.5", 0.5), ("×0.75", 0.75),
                          ("×1.0", 1.0), ("×1.5", 1.5), ("×2.0", 2.0)):
            if k is None:
                gates = ALL_GATES - {"Gtape"}
                r = run_v2(asset, df, gates)
            else:
                tape = {kk: v * k for kk, v in orig.items()}
                r = run_v2(asset, df, ALL_GATES, tape_thr=tape)
            print(f"   {label:<8s}   {r['n']:3d}   {r['wr']:.1%}   ${r['pnl']:+8.2f}")


if __name__ == "__main__":
    main()
