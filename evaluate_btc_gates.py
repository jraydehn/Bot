"""
evaluate_btc_gates.py
Evaluates each BTC gate's relevance using paper_trades.csv.

For each gate:
  - If we can simulate who WOULD have been blocked: split blocked vs passed, report WR+PnL for blocked
  - Note: trades in CSV mostly PASSED all gates (blocks aren't in CSV), so we analyze the condition
    directly on trades that went through — showing whether the conditions correlate with wins/losses
    within the trades we can observe.
  - For gates with rescue conditions: evaluate blocked-but-rescued subsets specifically.

Flat $10 per trade (for comparability), actual resolved_yes outcomes.
"""

import pandas as pd
import numpy as np

CSV = "results/paper_trades.csv"
FLAT = 10.0

df = pd.read_csv(CSV, low_memory=False)
df = df[(df["decision"] == "trade") & (df["resolved_yes"].notna())].copy()
df["resolved_yes"] = df["resolved_yes"].astype(float)

# BTC only (tickers start with KXBTCD)
btc = df[df["contract_ticker"].str.startswith("KXBTCD")].copy()
yes = btc[btc["side"] == "yes"].copy()
no  = btc[btc["side"] == "no"].copy()

def stats(mask, side_df, label):
    sub = side_df[mask]
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr = sub["resolved_yes"].mean() if side_df is yes else (1 - sub["resolved_yes"]).mean()
    be_arr = sub["p_market"].values
    be = (1 - be_arr).mean() if side_df is no else be_arr.mean()
    pnl = sum((1 - p) * FLAT if r == 1 else -p * FLAT
              for r, p in zip(sub["resolved_yes"], sub["p_market"]))
    delta = wr - be
    verdict = "KEEP" if delta < 0 else "QUESTIONABLE"
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={delta:+.1%}, PnL=${pnl:+.0f}  [{verdict}]"

def stats_both(mask, label, flip_pnl=False):
    """For gates that apply to both sides."""
    sub_y = yes[mask[yes.index] if hasattr(mask, 'index') and len(mask)==len(btc) else mask.reindex(yes.index, fill_value=False)]
    sub_n = no[mask[no.index] if hasattr(mask, 'index') and len(mask)==len(btc) else mask.reindex(no.index, fill_value=False)]
    parts = []
    for sub, side_label, is_no in [(sub_y, "YES", False), (sub_n, "NO", True)]:
        if len(sub) < 3:
            parts.append(f"    {side_label}: n={len(sub)} (too small)")
            continue
        wr = (1 - sub["resolved_yes"]).mean() if is_no else sub["resolved_yes"].mean()
        be_arr = sub["p_market"].values
        be = (1 - be_arr).mean() if is_no else be_arr.mean()
        pnl = sum((1 - p) * FLAT if (r == 1 and not is_no) or (r == 0 and is_no) else -p * FLAT * (1 if is_no else 1)
                  for r, p in zip(sub["resolved_yes"], sub["p_market"]))
        if is_no:
            pnl = sum((r == 0) * (1 - p) * FLAT - (r == 1) * p * FLAT
                      for r, p in zip(sub["resolved_yes"], sub["p_market"]))
        delta = wr - be
        verdict = "KEEP" if delta < 0 else "QUESTIONABLE"
        parts.append(f"    {side_label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={delta:+.1%}, PnL=${pnl:+.0f}  [{verdict}]")
    return f"  {label}:\n" + "\n".join(parts)

# Helper: YES PnL
def yes_pnl(sub):
    return sum((1 - p) * FLAT if r == 1 else -p * FLAT
               for r, p in zip(sub["resolved_yes"], sub["p_market"]))

# Helper: NO PnL
def no_pnl(sub):
    return sum((r == 0) * (1 - p) * FLAT - (r == 1) * p * FLAT
               for r, p in zip(sub["resolved_yes"], sub["p_market"]))

def report_yes(mask, label):
    sub = yes[mask.reindex(yes.index, fill_value=False)]
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr = sub["resolved_yes"].mean()
    be = sub["p_market"].mean()
    pnl = yes_pnl(sub)
    delta = wr - be
    verdict = "GATE VALID" if delta < -0.02 else ("BORDERLINE" if delta < 0.02 else "GATE HARMFUL?")
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={delta:+.1%}, PnL=${pnl:+.0f}  [{verdict}]"

def report_no(mask, label):
    sub = no[mask.reindex(no.index, fill_value=False)]
    if len(sub) < 3:
        return f"  {label}: n={len(sub)} (too small)"
    wr = (1 - sub["resolved_yes"]).mean()
    be = (1 - sub["p_market"]).mean()
    pnl = no_pnl(sub)
    delta = wr - be
    verdict = "GATE VALID" if delta < -0.02 else ("BORDERLINE" if delta < 0.02 else "GATE HARMFUL?")
    return f"  {label}: n={len(sub)}, WR={wr:.1%}, BE={be:.1%}, Δ={delta:+.1%}, PnL=${pnl:+.0f}  [{verdict}]"

# ── Convert columns to numeric ────────────────────────────────────────────────
for col in ["stoch_k", "composite_rev", "composite_trend", "composite_p_up",
            "ema_stack_bias", "structure_bias", "vpin_score", "funding_bias",
            "vwap_score", "stretch_score", "vol_score", "offset_pct",
            "p_market", "body_15m", "bp_5m", "dir_15m", "vwap_stretch_score",
            "ema_stretch_score", "chg_30m", "chg_5m", "spread", "tau_minutes",
            "stoch_bias", "obi_score", "supply_pct", "demand_pct"]:
    if col in btc.columns:
        btc[col] = pd.to_numeric(btc[col], errors="coerce")
        yes[col] = pd.to_numeric(yes[col], errors="coerce")
        no[col]  = pd.to_numeric(no[col], errors="coerce")

# Use vwap_stretch_score as stretch_score if stretch_score not present
if "vwap_stretch_score" in yes.columns:
    yes["stretch_score"] = yes["vwap_stretch_score"]
    no["stretch_score"]  = no["vwap_stretch_score"]
    btc["stretch_score"] = btc["vwap_stretch_score"]

print(f"\n{'='*70}")
print(f"BTC GATE EVALUATION (flat ${FLAT:.0f}/trade)")
print(f"Total BTC resolved trades: {len(btc)} (YES={len(yes)}, NO={len(no)})")
print(f"Overall YES WR: {yes['resolved_yes'].mean():.1%}, PnL=${yes_pnl(yes):+.0f}")
print(f"Overall NO WR:  {(1-no['resolved_yes']).mean():.1%}, PnL=${no_pnl(no):+.0f}")
print(f"{'='*70}")

print("\n=== BTC YES GATES ===")

# 1. near_itm_gate: pm>0.50 AND (4h RSI>62 OR 4h MACD hist>80)
# We can't replicate 4h RSI from the CSV, but we can look at pm>0.50 overall
m_nig_pm = yes["p_market"] > 0.50
print("\n[near_itm_gate] pm>0.50 context (gate blocks when 4h overbought)")
print(report_yes(m_nig_pm, "pm>0.50 YES (gate context)"))
print(report_yes(~m_nig_pm, "pm<=0.50 YES (gate not active)"))

# 2. cg_fr_gate: fr_vol_1d>0 AND pm<0.60 AND vpin<1 → proxy: funding_bias=1 AND pm<0.60 AND vpin=0
m_fr = (yes["funding_bias"] == 1) & (yes["p_market"] < 0.60) & (yes["vpin_score"] == 0)
m_fr_valid = m_fr & yes["funding_bias"].notna()
print("\n[cg_fr_gate] funding_bias=+1 AND pm<0.60 AND vpin=0 (would-be-blocked)")
print(report_yes(m_fr_valid, "GATE CONDITION"))
print(report_yes((yes["funding_bias"]==1) & (yes["p_market"]<0.60) & (yes["vpin_score"]>=1), "RESCUED (vpin>=1)"))
# Also check funding_bias=-1 (bearish) gate we discussed earlier
m_fr_neg = (yes["funding_bias"] == -1) & (yes["p_market"] < 0.60) & (yes["vpin_score"] == 0)
print(report_yes(m_fr_neg, "funding_bias=-1 AND pm<0.60 AND vpin=0 (cg_fr_gate fires on this!)"))

# 3. rev_div_gate: ema_stack=1, rev<=-4, stoch_k>55
m_rev = (yes["ema_stack_bias"]==1) & (yes["composite_rev"]<=-4) & (yes["stoch_k"]>55)
print("\n[rev_div_gate] ema=1, rev<=-4, stoch>55 (pm>0.65 rescued)")
print(report_yes(m_rev & (yes["p_market"]<=0.65), "BLOCKED (pm<=0.65)"))
print(report_yes(m_rev & (yes["p_market"]>0.65), "RESCUED (pm>0.65)"))

# 4. cg_oi_stable_yes_gate: oi_stable_4h>2%, pm<0.50 → can't test directly from CSV
print("\n[cg_oi_stable_yes_gate] Cannot evaluate — oi_stable not in CSV")

# 5. neutral_ema_g2: ema=0, vwap=-1, pm<0.60
m_g2 = (yes["ema_stack_bias"]==0) & (yes["vwap_score"]==-1) & (yes["p_market"]<0.60)
print("\n[neutral_ema_g2] ema=0, vwap=-1, pm<0.60")
print(report_yes(m_g2, "GATE CONDITION (trades through = rescued or pre-gate)"))
print(report_yes((yes["ema_stack_bias"]==0) & (yes["vwap_score"]==0), "ema=0 vwap=0 (comparison)"))

# 6. neutral_ema_g3: ema=0, comp_trend=-1
m_g3 = (yes["ema_stack_bias"]==0) & (yes["composite_trend"]==-1)
print("\n[neutral_ema_g3] ema=0, comp_trend=-1")
print(report_yes(m_g3, "GATE CONDITION"))
print(report_yes((yes["ema_stack_bias"]==0) & (yes["composite_trend"]>=0), "ema=0, trend>=0 (comparison)"))

# 7. bear_drift (arm 1): ema=-1, rev<=3, stoch>=35
m_bd1 = (yes["ema_stack_bias"]==-1) & (yes["composite_rev"]<=3) & (yes["stoch_k"]>=35)
m_bd1_rescued = m_bd1 & ((yes["vpin_score"]==1) | (yes["ema_stretch_score"]==1))
m_bd1_blocked = m_bd1 & ~m_bd1_rescued
print("\n[bear_drift arm1] ema=-1, rev<=3, stoch>=35")
print(report_yes(m_bd1_blocked, "BLOCKED (no rescue)"))
print(report_yes(m_bd1_rescued, "RESCUED (vpin=1 OR ema_stretch=1)"))

# 8. btc_otmlow_gate: pm<0.20, vpin=0
m_otmlow = (yes["p_market"]<0.20) & (yes["vpin_score"]==0)
print("\n[btc_otmlow_gate] pm<0.20, vpin=0")
print(report_yes(m_otmlow, "GATE CONDITION"))
print(report_yes((yes["p_market"]<0.20) & (yes["vpin_score"]>=1), "pm<0.20 RESCUED (vpin>=1)"))

# 9. btc_struct_gate: structure_bias=-1
m_sg_block = (yes["structure_bias"]==-1)
m_sg_rescue = m_sg_block & (
    (yes["chg_5m"]>=0.0005) | (yes["vwap_score"]==-1) | (yes["chg_30m"]<-0.002)
)
m_sg_hard = m_sg_block & ~m_sg_rescue
print("\n[btc_struct_gate] structure_bias=-1")
print(report_yes(m_sg_hard, "BLOCKED (no rescue)"))
print(report_yes(m_sg_rescue, "RESCUED"))
print(report_yes(yes["structure_bias"]==1, "struct=+1 (comparison)"))

# 10. liq_cascade_gate: can't test — liq_score not in CSV
print("\n[liq_cascade_gate] Cannot evaluate — liq_score not in CSV")

# 11. btc_exhaustion_gate: ema=1, rev<=-4, stretch<=-1, stoch>=75
m_exh = (yes["ema_stack_bias"]==1) & (yes["composite_rev"]<=-4) & (yes["stoch_k"]>=75)
if "stretch_score" in yes.columns:
    m_exh = m_exh & (yes["stretch_score"]<=-1)
print("\n[btc_exhaustion_gate] ema=1, rev<=-4, stoch>=75, stretch<=-1")
print(report_yes(m_exh, "GATE CONDITION (would be blocked)"))
print(report_yes((yes["ema_stack_bias"]==1) & (yes["composite_rev"]>=-3), "ema=1, rev>=-3 (comparison)"))

# 12. btc_adx5_gate: OTM, pm<0.27, bearish lower_HL/ADX — proxy: OTM YES pm<0.27 performance
m_adx5 = (yes["p_market"]<0.27) & (yes["offset_pct"]>0)
print("\n[btc_adx5_gate] OTM YES pm<0.27 (gate fires when bearish ADX/lower_HL)")
print(report_yes(m_adx5, "pm<0.27 OTM (gate context)"))
print(report_yes((yes["p_market"]>=0.27) & (yes["offset_pct"]>0), "pm>=0.27 OTM (comparison)"))

# 13. btc_falling_knife_gate: rev>=4, chg_30m<-0.20%
m_fk = (yes["composite_rev"]>=4) & (yes["chg_30m"]<-0.002)
m_fk_rescue = m_fk & ((yes["chg_5m"]>0.001) | (yes["offset_pct"]<-0.001))
m_fk_hard = m_fk & ~m_fk_rescue
print("\n[btc_falling_knife_gate] rev>=4, chg_30m<-0.20%")
print(report_yes(m_fk_hard, "BLOCKED (no rescue)"))
print(report_yes(m_fk_rescue, "RESCUED"))

# 14. btc_body_bp_gate: body in [0.50, 0.60), bp<0.55
m_bbp_block = (yes["body_15m"]>=0.50) & (yes["body_15m"]<0.60) & (yes["bp_5m"]<0.55)
m_bbp_rescue = (yes["body_15m"]>=0.50) & (yes["body_15m"]<0.60) & (yes["bp_5m"]>=0.55)
print("\n[btc_body_bp_gate] body in [0.50,0.60)")
print(report_yes(m_bbp_block, "BLOCKED (bp<0.55)"))
print(report_yes(m_bbp_rescue, "RESCUED (bp>=0.55)"))

# 15. btc_contra_bar_gate YES: body>=0.70, dir=-1
m_contra_y = (yes["body_15m"]>=0.70) & (yes["dir_15m"]==-1)
print("\n[btc_contra_bar_gate YES] body>=0.70, dir=-1 (bearish bar)")
print(report_yes(m_contra_y, "GATE CONDITION"))
print(report_yes((yes["body_15m"]>=0.70) & (yes["dir_15m"]==1), "body>=0.70 dir=+1 (comparison)"))

# 16. btc_ema0_stretch2_gate: ema=0, stretch=+2
m_es2 = (yes["ema_stack_bias"]==0) & (yes["stretch_score"]==2) if "stretch_score" in yes.columns else pd.Series(False, index=yes.index)
print("\n[btc_ema0_stretch2_gate] ema=0, stretch=+2")
print(report_yes(m_es2, "GATE CONDITION"))

# 17. btc_otm_neutral_gate: ema=0, p_up>=0.60, OTM
m_otn = (yes["ema_stack_bias"]==0) & (yes["composite_p_up"]>=0.60) & (yes["offset_pct"]>0)
print("\n[btc_otm_neutral_gate] ema=0, p_up>=0.60, OTM")
print(report_yes(m_otn, "GATE CONDITION"))
print(report_yes((yes["ema_stack_bias"]==0) & (yes["composite_p_up"]>=0.60) & (yes["offset_pct"]<=0), "ema=0, p_up>=0.60, ITM (pass-through)"))

# 18. btc_ema0_itm_gate: ema=0, offset<=0, trend>=3, rev=0
m_eitm = (yes["ema_stack_bias"]==0) & (yes["offset_pct"]<=0) & (yes["composite_trend"]>=3) & (yes["composite_rev"]==0)
print("\n[btc_ema0_itm_gate] ema=0, ITM, trend>=3, rev=0")
print(report_yes(m_eitm, "GATE CONDITION"))

# 19. ema_stack3_gate: ema_stack_liq=3 — proxy: ema_stack_bias=0
print("\n[ema_stack3_gate] ema_stack_liq=3 (cannot exactly simulate — not in CSV)")

# 20. btc_tau_gate YES: tau<30, pm<0.40
m_tau_y = (yes["tau_minutes"]<30) & (yes["p_market"]<0.40)
print("\n[btc_tau_gate YES] tau<30, pm<0.40 OTM")
print(report_yes(m_tau_y, "GATE CONDITION"))
print(report_yes((yes["tau_minutes"]<30) & (yes["p_market"]>=0.40), "tau<30 pm>=0.40 (comparison)"))

# 21. streak_gate YES: streak30=bearish, stoch<=70 — proxy: chg_30m<0, stoch<=70
m_streak_y = (yes["chg_30m"]<0) & (yes["stoch_k"]<=70)
m_streak_y_rescue = m_streak_y & (yes["structure_bias"]==1)
m_streak_y_block = m_streak_y & (yes["structure_bias"]!=1) & (yes["chg_30m"]<=0)
print("\n[streak_gate YES] chg_30m<0 (bearish), stoch<=70 (proxy for streak30=bearish)")
print(report_yes(m_streak_y_block, "WOULD BLOCK (struct!=1, no bounce)"))
print(report_yes(m_streak_y_rescue, "WOULD RESCUE (struct=1)"))

print("\n\n=== BTC NO GATES ===")

# 1. btc_no_z_gate: |z| < 0.45 — proxy: offset_pct near 0
m_noz = (no["offset_pct"].abs()<0.0045)  # ~0.45% as rough proxy
print("\n[btc_no_z_gate] |offset_pct|<0.45% (near-ATM NO proxy)")
print(report_no(m_noz, "NEAR-ATM NO"))
print(report_no(no["offset_pct"].abs()>=0.0045, "OTM NO (comparison)"))

# 2. btc_no_wrongdir_gate: pm>=0.65, ema=1, stretch<=-2
m_wrongdir = (no["p_market"]>=0.65) & (no["ema_stack_bias"]==1)
if "stretch_score" in no.columns:
    m_wrongdir = m_wrongdir & (no["stretch_score"]<=-2)
print("\n[btc_no_wrongdir_gate] pm>=0.65, ema=1, stretch<=-2")
print(report_no(m_wrongdir, "GATE CONDITION"))

# 3. btc_no_smc_demand_gate: bearish SMC 1h, demand_pct<1.2%
m_smc_no = (no["smc_1h"]=="bearish") & (no["demand_pct"]<1.2)
m_smc_no_rescue = m_smc_no & (no["supply_pct"]<1.0)
m_smc_no_block = m_smc_no & ~m_smc_no_rescue
print("\n[btc_no_smc_demand_gate] smc_1h=bearish, demand_pct<1.2%")
print(report_no(m_smc_no_block, "BLOCKED (no supply rescue)"))
print(report_no(m_smc_no_rescue, "RESCUED (supply_pct<1.0%)"))

# 4. btc_spread_gate: spread>=0.04
m_spread = no["spread"]>=0.04
print("\n[btc_spread_gate] spread>=0.04")
print(report_no(m_spread, "GATE CONDITION (NO side)"))
print(report_yes(yes["spread"]>=0.04, "GATE CONDITION (YES side)"))

# 5. btc_contra_bar_gate NO: body>=0.70, dir=+1
m_contra_n = (no["body_15m"]>=0.70) & (no["dir_15m"]==1)
print("\n[btc_contra_bar_gate NO] body>=0.70, dir=+1 (bullish bar vs NO bet)")
print(report_no(m_contra_n, "GATE CONDITION"))
print(report_no((no["body_15m"]>=0.70) & (no["dir_15m"]==-1), "body>=0.70 dir=-1 (comparison)"))

# 6. btc_no_highpm_bearema_gate: pm>0.70, ema=-1
m_hpbe = (no["p_market"]>0.70) & (no["ema_stack_bias"]==-1)
print("\n[btc_no_highpm_bearema_gate] pm>0.70, ema=-1")
print(report_no(m_hpbe, "GATE CONDITION"))
print(report_no((no["p_market"]>0.70) & (no["ema_stack_bias"]==1), "pm>0.70, ema=+1 (comparison)"))

# 7. btc_nopup_gate: p_up<=0.36 OR p_up>=0.50, pm>=0.20
m_nopup_cond = ((no["composite_p_up"]<=0.36) | (no["composite_p_up"]>=0.50)) & (no["p_market"]>=0.20)
m_nopup_rescue = m_nopup_cond & ((no["stretch_score"]==1) | (no["vol_score"]==1))
m_nopup_block = m_nopup_cond & ~m_nopup_rescue
print("\n[btc_nopup_gate] p_up<=0.36 OR p_up>=0.50, pm>=0.20")
print(report_no(m_nopup_block, "BLOCKED (no rescue)"))
print(report_no(m_nopup_rescue, "RESCUED (stretch=1 OR vol=1)"))
print(report_no(((no["composite_p_up"]>0.36) & (no["composite_p_up"]<0.50)) & (no["p_market"]>=0.20), "p_up in (0.36,0.50) — GATE PASSES"))

# 8. btc_tau_gate NO: tau<30, p_up>0.48
m_tau_n = (no["tau_minutes"]<30) & (no["composite_p_up"]>0.48)
print("\n[btc_tau_gate NO] tau<30, p_up>0.48")
print(report_no(m_tau_n, "GATE CONDITION"))
print(report_no((no["tau_minutes"]<30) & (no["composite_p_up"]<=0.48), "tau<30 p_up<=0.48 (low-tau conviction)"))

# 9. cg_oi_stable_no_gate: oi_stable_4h>1% — can't test
print("\n[cg_oi_stable_no_gate] Cannot evaluate — oi_stable not in CSV")

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Gate':<30} {'Side':<5} {'n':>5} {'WR':>6} {'BE':>6} {'Δ':>6} {'PnL':>8}  Verdict")
print("-"*75)

gates_summary = []

def add_gate(name, side, mask, side_df):
    sub = side_df[mask.reindex(side_df.index, fill_value=False)]
    if len(sub) < 3:
        return
    is_no = (side == "NO")
    wr = (1 - sub["resolved_yes"]).mean() if is_no else sub["resolved_yes"].mean()
    be = (1 - sub["p_market"]).mean() if is_no else sub["p_market"].mean()
    if is_no:
        pnl = no_pnl(sub)
    else:
        pnl = yes_pnl(sub)
    delta = wr - be
    verdict = "KEEP" if delta < -0.02 else ("BORDERLINE" if abs(delta) <= 0.02 else "REVIEW")
    gates_summary.append((name, side, len(sub), wr, be, delta, pnl, verdict))

add_gate("near_itm_gate", "YES", m_nig_pm, yes)
add_gate("cg_fr_gate (+)", "YES", m_fr_valid, yes)
add_gate("rev_div_gate", "YES", m_rev & (yes["p_market"]<=0.65), yes)
add_gate("neutral_ema_g2", "YES", m_g2, yes)
add_gate("neutral_ema_g3", "YES", m_g3, yes)
add_gate("bear_drift arm1", "YES", m_bd1_blocked, yes)
add_gate("btc_otmlow_gate", "YES", m_otmlow, yes)
add_gate("btc_struct_gate", "YES", m_sg_hard, yes)
add_gate("btc_exhaustion", "YES", m_exh, yes)
add_gate("btc_adx5 ctx", "YES", m_adx5, yes)
add_gate("btc_falling_knife", "YES", m_fk_hard, yes)
add_gate("btc_body_bp", "YES", m_bbp_block, yes)
add_gate("btc_contra_bar", "YES", m_contra_y, yes)
add_gate("btc_ema0_stretch2", "YES", m_es2, yes)
add_gate("btc_otm_neutral", "YES", m_otn, yes)
add_gate("btc_ema0_itm", "YES", m_eitm, yes)
add_gate("btc_tau YES", "YES", m_tau_y, yes)
add_gate("streak YES", "YES", m_streak_y_block, yes)
add_gate("btc_no_wrongdir", "NO", m_wrongdir, no)
add_gate("btc_no_smc_demand", "NO", m_smc_no_block, no)
add_gate("btc_spread_n", "NO", m_spread, no)
add_gate("btc_contra_bar NO", "NO", m_contra_n, no)
add_gate("btc_no_highpm_bear", "NO", m_hpbe, no)
add_gate("btc_nopup blocked", "NO", m_nopup_block, no)
add_gate("btc_nopup rescued", "NO", m_nopup_rescue, no)
add_gate("btc_tau NO", "NO", m_tau_n, no)
add_gate("btc_nopup PASS", "NO", ((no["composite_p_up"]>0.36) & (no["composite_p_up"]<0.50)) & (no["p_market"]>=0.20), no)

for name, side, n, wr, be, delta, pnl, verdict in sorted(gates_summary, key=lambda x: x[5]):
    print(f"{name:<30} {side:<5} {n:>5} {wr:>6.1%} {be:>6.1%} {delta:>+6.1%} {pnl:>+8.0f}  {verdict}")
