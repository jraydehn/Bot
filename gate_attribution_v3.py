#!/usr/bin/env python3
"""
gate_attribution_v3.py — v2 + missing BTC gates with rescue conditions.

Adds to v2 (gate_attribution_v2.py):
  • BTC vol_score=1 YES block, rescue: ema_stack==1 AND (conf==0 OR funding==0)
  • BTC NO edge<2% block, rescue: comp_rev≤−1 AND vol_ratio≥1.0
  • BTC spread≥0.04 block (any side), rescue: chg_10m direction-aligned AND net_edge≥0.07
  • BTC tau<30 block (any side), rescue: p_up conviction (>0.55 YES / <0.45 NO)
                                          OR (kelly≥0.15 AND spread≤0.02)

Confirmed by Sonnet's overview:
  • Pure-edge override / Gate P is dead code — not wired up. GRR has no bypass
    beyond the existing NO-side 0.08 edge exception.
  • Streak v2 deferred — requires price-data joins; affects BTC slightly
    (more passes, lower WR) but not the directional signal of GRR sweeps.
  • BTC ISO rescue not modeled — pre-iso p_yes_model not in archive; would
    need to approximate.

Inherits from v2: logged p_yes_model, flat $1k bankroll, corrected Gpup_btc
rescue: (funding==0 AND struct≥0) OR comp_rev>0.
"""

import math, sys
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
from gate_attribution_v2 import kelly_bet_flat, trade_pnl

ALL_GATES = {
    "G0_pm", "GCS", "GCI", "GNS", "GOTM", "G3", "GRR",
    "Gpm15_btc", "Gpm45_eth", "Gtape", "Gpup_btc",
    # New BTC-only gates
    "Gvol_btc",      # vol_score=1 YES block
    "GnoEdge_btc",   # BTC NO edge<2% block
    "Gspread_btc",   # BTC spread>=0.04 block
    "Gtau_btc",      # BTC tau<30 block
}


def _kelly_fraction(p_model, pm, side):
    """Approx Kelly fraction at full mult — used for tau<30 rescue."""
    if side == "yes":
        b = (1 - pm) / pm if pm > 0 else 0
        p, q = p_model, 1 - p_model
    else:
        b = pm / (1 - pm) if pm < 1 else 0
        p, q = 1 - p_model, p_model
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b) * KELLY_MULT


def evaluate_row_v3(row, asset, gates, ns_max_override=None):
    p = ASSET_PARAMS[asset]
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    try:
        p_model = float(row["p_yes_model"])
    except (TypeError, ValueError):
        return None
    p_up = row["composite_p_up"]
    resolved_yes = int(row["resolved_yes"])
    if pd.isna(p_model) or pm <= 0 or pm >= 1:
        return None
    offset = (strike - spot) / spot if spot > 0 else 0.0
    ns_max = ns_max_override if ns_max_override is not None else p["ns_max"]

    # rescue features
    funding = row.get("funding_bias", 0)
    struct = row.get("structure_bias", 0)
    comp_rev = row.get("composite_rev", 0)
    ema_stack = row.get("ema_stack_bias", 0)
    conf_score = row.get("confirmation_score", 0)
    vol_score = row.get("vol_score", 0)
    vol_ratio = row.get("vol_ratio", 1.0)
    spread_v = row.get("spread", 0.0)
    tau = row.get("tau_minutes", 60.0)
    chg_10m = row.get("chg_10m", 0.0)
    for v_name in ("funding","struct","comp_rev","ema_stack","conf_score","vol_score","vol_ratio","spread_v","tau","chg_10m"):
        v_val = locals()[v_name]
        if pd.isna(v_val):
            locals()[v_name] = 0
    funding = 0 if pd.isna(funding) else funding
    struct = 0 if pd.isna(struct) else struct
    comp_rev = 0 if pd.isna(comp_rev) else comp_rev
    ema_stack = 0 if pd.isna(ema_stack) else ema_stack
    conf_score = 0 if pd.isna(conf_score) else conf_score
    vol_score = 0 if pd.isna(vol_score) else vol_score
    vol_ratio = 1.0 if pd.isna(vol_ratio) else vol_ratio
    spread_v = 0.0 if pd.isna(spread_v) else spread_v
    tau = 60.0 if pd.isna(tau) else tau
    chg_10m = 0.0 if pd.isna(chg_10m) else chg_10m

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
        if side == "yes" and asset == "BTC" and "Gpup_btc" in gates \
                and p_up < 0.52 and not btc_pup_rescue:
            continue

        # --- BTC vol_score=1 YES block ---
        if asset == "BTC" and side == "yes" and "Gvol_btc" in gates and int(vol_score) == 1:
            vol_rescue = (int(ema_stack) == 1 and (int(conf_score) == 0 or int(funding) == 0))
            if not vol_rescue:
                continue

        # --- BTC spread>=0.04 block (computed before edge math; rescue uses post-edge) ---
        # We defer the rescue decision until after net is computed below.

        # --- Edge math (side-aware bid/ask, 2026-05-01 patch) ---
        # Production (decision.py:283-284) uses p_market_ask for YES bets and
        # p_market_bid for NO bets — using mid inflates edge for wide-spread
        # contracts. The CSV logs only mid `p_market` and `spread`; we approximate
        # ask = mid + spread/2 and bid = mid - spread/2.
        # DEFAULT_SPREAD constant (still in net_edge) is a separate friction cost
        # representing slippage execution loss, independent of market spread.
        s_half = float(spread_v) / 2.0
        pm_yes_ask = pm + s_half     # what you pay to buy YES
        pm_no_bid  = pm - s_half     # reference for NO (no_cost = 1 - bid)
        fee = kalshi_fee(pm)
        if side == "yes":
            net = (p_model - pm_yes_ask) - fee - SLIPPAGE - SPREAD
            rr = pm / (1 - pm) if pm < 1 else 999
            if "GRR" in gates and rr > RR_MAX_YES: continue
            if "GOTM" in gates:
                tier_min = 0.0
                for thr, mn in OTM_TIERS:
                    if pm < thr:
                        tier_min = mn; break
                if net < tier_min: continue
        else:
            net = (pm_no_bid - p_model) - fee - SLIPPAGE - SPREAD
            rr = (1 - pm) / pm if pm > 0 else 999
            if "GRR" in gates and (rr < RR_MIN_NO or rr > RR_MAX_NO) \
                    and net < RR_EDGE_EXC:
                continue
        if "G3" in gates and net < p["g3_min"]:
            continue

        # --- BTC NO edge<2% block (after edge computed) ---
        if asset == "BTC" and side == "no" and "GnoEdge_btc" in gates and net < 0.02:
            no_edge_rescue = (float(comp_rev) <= -1 and float(vol_ratio) >= 1.0)
            if not no_edge_rescue:
                continue

        # --- BTC spread>=0.04 block (after edge computed; rescue uses chg_10m alignment + net edge) ---
        if asset == "BTC" and "Gspread_btc" in gates and float(spread_v) >= 0.04:
            chg_aligned = (side == "yes" and float(chg_10m) > 0) or (side == "no" and float(chg_10m) < 0)
            spread_rescue = (chg_aligned and net >= 0.07)
            if not spread_rescue:
                continue

        # --- BTC tau<30 block (after edge computed; rescue: p_up conviction OR kelly>=0.15+spread<=0.02) ---
        if asset == "BTC" and "Gtau_btc" in gates and float(tau) < 30:
            pup_conv = (side == "yes" and p_up > 0.55) or (side == "no" and p_up < 0.45)
            kf = _kelly_fraction(p_model, pm, side)
            kelly_rescue = (kf >= 0.15 and float(spread_v) <= 0.02)
            if not (pup_conv or kelly_rescue):
                continue

        won = (resolved_yes == 1 and side == "yes") or (resolved_yes == 0 and side == "no")
        # pm_eff = side-specific p_market reference (yes_ask for YES, yes_bid for NO).
        # Used downstream by kelly_bet_flat and trade_pnl so bet sizing and PnL
        # reflect the actual cost paid, not the mid quote.
        pm_eff = pm_yes_ask if side == "yes" else pm_no_bid
        if best is None or net > best["net"]:
            best = {"side": side, "pm": pm, "pm_eff": pm_eff,
                    "p_model": p_model, "net": net,
                    "offset": offset, "won": won, "p_up": p_up,
                    "chg_5m": row.get("chg_5m", 0), "chg_10m": chg_10m,
                    "chg_30m": row.get("chg_30m", 0)}
    return best


def run_v3(asset, df, gates, ns_max_override=None, tape_thr=None,
           rr_max_no=None, rr_min_no=None, rr_max_yes=None, rr_edge_exc=None):
    """Like run_v2 but uses evaluate_row_v3 and accepts GRR overrides."""
    import gate_attribution as ga
    thr = tape_thr if tape_thr is not None else TAPE_THRESHOLDS[asset]
    # GRR overrides
    o_max_no, o_min_no, o_max_yes, o_edge = ga.RR_MAX_NO, ga.RR_MIN_NO, ga.RR_MAX_YES, ga.RR_EDGE_EXC
    if rr_max_no  is not None: ga.RR_MAX_NO  = rr_max_no
    if rr_min_no  is not None: ga.RR_MIN_NO  = rr_min_no
    if rr_max_yes is not None: ga.RR_MAX_YES = rr_max_yes
    if rr_edge_exc is not None: ga.RR_EDGE_EXC = rr_edge_exc
    # Need to also patch the module-level constants v3 imported
    global RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC
    RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = ga.RR_MAX_NO, ga.RR_MIN_NO, ga.RR_MAX_YES, ga.RR_EDGE_EXC
    try:
        pnls = []
        for dt_, group in df.groupby("decision_time", sort=True):
            cands = []
            for _, row in group.iterrows():
                c = evaluate_row_v3(row, asset, gates, ns_max_override=ns_max_override)
                if c is not None: cands.append(c)
            if not cands: continue
            best = max(cands, key=lambda c: c["net"])
            sev = severity(best["side"], best["chg_5m"], best["chg_10m"],
                           best["chg_30m"], thr) if "Gtape" in gates else 0.0
            kelly_scale = 1.0
            if "Gtape" in gates and sev >= 1.5: continue
            if "Gtape" in gates and sev >= 0.5:
                kelly_scale = max(0.25, 1.0 - (sev - 0.5) * 0.75)
            # Use pm_eff (side-specific) for Kelly and PnL math, not mid pm.
            bet = kelly_bet_flat(best["p_model"], best["pm_eff"], best["side"], kelly_scale)
            if bet <= 0: continue
            pnl = trade_pnl(bet, best["side"], best["pm_eff"], best["won"])
            pnls.append((best["won"], pnl))
        if not pnls:
            return {"n": 0, "wr": 0.0, "pnl": 0.0}
        n = len(pnls)
        wins = sum(1 for w, _ in pnls if w)
        return {"n": n, "wr": wins / n, "pnl": sum(p for _, p in pnls)}
    finally:
        ga.RR_MAX_NO, ga.RR_MIN_NO, ga.RR_MAX_YES, ga.RR_EDGE_EXC = o_max_no, o_min_no, o_max_yes, o_edge
        RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = o_max_no, o_min_no, o_max_yes, o_edge


def main():
    print(f"\ngate_attribution_v3.py — v2 + BTC vol/spread/tau/no-edge gates with rescues\n")

    # Sanity: verify ETH/SOL outputs match v2 (same code path for non-BTC assets)
    print("=== Sanity: ETH/SOL baseline should match v2 ===")
    for asset in ("ETH", "SOL"):
        df = load_archive(asset)
        r = run_v3(asset, df, ALL_GATES)
        print(f"  [{asset}] n={r['n']:3d}  WR={r['wr']:.1%}  pnl=${r['pnl']:+8.2f}")

    # BTC baseline with new gates
    print("\n=== BTC baseline: v2 vs v3 (with new gates) ===")
    btc = load_archive("BTC")
    import gate_attribution_v2 as v2
    r_v2 = v2.run_v2("BTC", btc, v2.ALL_GATES)
    r_v3 = run_v3("BTC", btc, ALL_GATES)
    print(f"  v2: n={r_v2['n']:3d}  WR={r_v2['wr']:.1%}  pnl=${r_v2['pnl']:+8.2f}")
    print(f"  v3: n={r_v3['n']:3d}  WR={r_v3['wr']:.1%}  pnl=${r_v3['pnl']:+8.2f}")

    # Per-new-gate LOO attribution for BTC
    print("\n=== BTC new-gate attribution (v3 baseline LOO) ===")
    print("  gate           Δpnl     baseline_n  loo_n   loo_pnl")
    base = run_v3("BTC", btc, ALL_GATES)
    for g in ("Gvol_btc", "GnoEdge_btc", "Gspread_btc", "Gtau_btc", "Gpup_btc"):
        loo = run_v3("BTC", btc, ALL_GATES - {g})
        delta = base["pnl"] - loo["pnl"]
        print(f"  {g:<14s} ${delta:+7.2f}    {base['n']:4d}      {loo['n']:4d}    ${loo['pnl']:+8.2f}")

    # GRR sweep, all three assets, v3
    print(f"\n=== GRR sweep (v3 — logged p_model, flat $1k, BTC rescues active) ===")
    SCENARIOS = [
        ("current (3,4,.33,.08)", {}),
        ("YES → 4.0",             {"rr_max_yes": 4.0}),
        ("YES → 5.0",             {"rr_max_yes": 5.0}),
        ("YES → 8.0",             {"rr_max_yes": 8.0}),
        ("YES OFF",               {"rr_max_yes": 999}),
        ("NO max → 5.0",          {"rr_max_no": 5.0}),
        ("NO max → 8.0",          {"rr_max_no": 8.0}),
        ("NO bounds OFF",         {"rr_max_no": 999, "rr_min_no": 0}),
        ("edge exc → 0.05",       {"rr_edge_exc": 0.05}),
        ("edge exc → 0.03",       {"rr_edge_exc": 0.03}),
        ("all bounds OFF",        {"rr_max_no": 999, "rr_min_no": 0, "rr_max_yes": 999}),
        ("GRR fully OFF",         "off"),
    ]
    for asset in ("BTC", "ETH", "SOL"):
        df = load_archive(asset)
        print(f"\n [{asset}]   scans={len(df):,}  hours={df['decision_time'].nunique():,}")
        print(f"   scenario                       n     WR      pnl")
        for label, kwargs in SCENARIOS:
            if kwargs == "off":
                r = run_v3(asset, df, ALL_GATES - {"GRR"})
            else:
                r = run_v3(asset, df, ALL_GATES, **kwargs)
            print(f"   {label:<30s}  {r['n']:3d}   {r['wr']:.1%}   ${r['pnl']:+8.2f}")


if __name__ == "__main__":
    main()
