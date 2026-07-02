#!/usr/bin/env python3
"""
simulate_april_vs_may_model.py

Direct comparison: "April model on May data" vs "current model on May data"

April model = Era 3, paper_trade_runner_pre_stoch_gate.py (Apr 28)
  Gates it had:  streak_gate, btc_vol1_gate, counter_tape
  Gates it lacked (added after Apr 28):
    smc_gate, btc_no_smc_demand_gate, btc_no_z_gate, no_pm_floor,
    btc_adx5_gate, liq_cascade_gate, near_atm_ema_gate,
    strong_trend_nearatm_gate, btc_deepno_neutral_gate, btc_gbdt_gate,
    + misc gates added May 1-18

"April model on May data" simulation:
  = current May taken trades
  + blocked_trades.csv rows where gate was NOT in April stack AND kelly_fraction > 0

"Current model on May data" = actual May paper_trades decision=='trade'
"""

import csv
import math
from collections import defaultdict, Counter

PAPER_TRADES = "results/paper_trades.csv"
BLOCKED      = "results/blocked_trades.csv"

# ─── helpers ────────────────────────────────────────────────────────────────

def norm_res(v):
    v = str(v).strip().lower()
    if v in ("true", "1"):  return 1
    if v in ("false", "0"): return 0
    return None

def calc_pnl(rows, ba_key="bet_amount"):
    pnl = 0.0
    for r in rows:
        res = norm_res(r.get("resolved_yes", ""))
        if res is None:
            continue
        side = r.get("side", "").strip().lower()
        try:
            pm = float(r.get("pm", r.get("p_market", 0.5)) or 0.5)
            if ba_key == "bet_amount":
                ba = float(r.get("bet_amount", 0) or 0)
            else:
                kf = float(r.get("kelly_fraction", 0) or 0)
                bk = float(r.get("bankroll", 0) or 0)
                ba = kf * bk
        except (ValueError, TypeError):
            continue
        if side == "yes":
            pnl += ba * (1 / pm - 1) if res == 1 else -ba
        else:
            cost = max(1.0 - pm, 0.01)
            pnl += ba * (1 / cost - 1) if res == 0 else -ba
    return pnl

def stats(rows, label, ba_key="bet_amount"):
    resolved = [r for r in rows if norm_res(r.get("resolved_yes", "")) is not None]
    if not resolved:
        print(f"  {label}: n=0 resolved")
        return
    wins = sum(
        1 for r in resolved
        if (r.get("side","")=="yes" and norm_res(r["resolved_yes"])==1)
        or (r.get("side","")=="no"  and norm_res(r["resolved_yes"])==0)
    )
    wr  = wins / len(resolved)
    pnl = calc_pnl(resolved, ba_key=ba_key)

    yes_r = [r for r in resolved if r.get("side","") == "yes"]
    no_r  = [r for r in resolved if r.get("side","") == "no"]
    be_yes = sum(float(r.get("pm", r.get("p_market", 0.5))) for r in yes_r)/len(yes_r) if yes_r else 0
    be_no  = 1-sum(float(r.get("pm", r.get("p_market", 0.5))) for r in no_r)/len(no_r)  if no_r  else 0

    print(f"  {label}")
    print(f"    trades={len(resolved):5d}  WR={wr:.1%}  PnL=${pnl:+,.0f}")
    if yes_r:
        wy = sum(1 for r in yes_r if norm_res(r["resolved_yes"])==1)/len(yes_r)
        print(f"    YES n={len(yes_r):4d}  WR={wy:.1%}  BE={be_yes:.1%}  edge={wy-be_yes:+.1%}")
    if no_r:
        wn = sum(1 for r in no_r if norm_res(r["resolved_yes"])==0)/len(no_r)
        print(f"    NO  n={len(no_r):4d}  WR={wn:.1%}  BE={be_no:.1%}  edge={wn-be_no:+.1%}")

# ─── load ────────────────────────────────────────────────────────────────────

with open(PAPER_TRADES, newline="") as f:
    all_pt = list(csv.DictReader(f))

with open(BLOCKED, newline="") as f:
    all_bl = list(csv.DictReader(f))

# May BTC trades actually taken by current model
may_taken = [
    r for r in all_pt
    if r.get("logged_at","")[:10] >= "2026-05-01"
    and "BTC" in r.get("contract_ticker","")
    and r.get("decision","").strip() == "trade"
]

# May BTC blocked trades — gates NOT in April model, kelly_fraction > 0
APRIL_GATES = {"streak_gate", "btc_vol1_gate", "counter_tape"}

# All gates that show up in May blocked_trades
may_btc_blocked = [
    r for r in all_bl
    if r.get("logged_at","")[:10] >= "2026-05-01"
    and r.get("asset","") == "BTC"
]

# April model would have taken these (gate added post-Apr-28, non-zero kelly)
april_would_also_take = [
    r for r in may_btc_blocked
    if r.get("gate_name","") not in APRIL_GATES
    and r.get("gate_name","") != ""
    and float(r.get("kelly_fraction","0") or 0) > 0
]

# ─── April reference period (actual, for comparison) ─────────────────────

apr_taken = [
    r for r in all_pt
    if "2026-04-20" <= r.get("logged_at","")[:10] <= "2026-04-28"
    and "BTC" in r.get("contract_ticker","")
    and r.get("decision","").strip() == "trade"
]

SEP  = "=" * 72
SEP2 = "-" * 72

print(SEP)
print("  BTC MODEL COMPARISON:  April model  vs  Current model on May data")
print(SEP)
print()

# 1. April reference (actual live performance)
print("1. APRIL REFERENCE — actual Apr 20-28 (the winning era, pre-gate-additions):")
stats(apr_taken, "Apr 20-28 actual trades", ba_key="bet_amount")
apr_by_exp = defaultdict(set)
for r in apr_taken:
    apr_by_exp[r.get("close_ts","")].add(r.get("side",""))
avg_per_exp = len(apr_taken) / max(len(apr_by_exp), 1)
yesno_exp   = sum(1 for s in apr_by_exp.values() if len(s) > 1)
print(f"    expiries={len(apr_by_exp)}  avg trades/expiry={avg_per_exp:.1f}  expiries w/ YES+NO={yesno_exp}")
print()

# 2. Current model on May
print("2. CURRENT MODEL on May 1-18 (all post-Apr-28 gates active):")
stats(may_taken, "May actual trades", ba_key="bet_amount")
may_by_exp = defaultdict(set)
for r in may_taken:
    may_by_exp[r.get("close_ts","")].add(r.get("side",""))
avg_per_exp_may = len(may_taken) / max(len(may_by_exp), 1)
yesno_exp_may   = sum(1 for s in may_by_exp.values() if len(s) > 1)
print(f"    expiries={len(may_by_exp)}  avg trades/expiry={avg_per_exp_may:.1f}  expiries w/ YES+NO={yesno_exp_may}")
print()

# 3. April model on May data
print("3. APRIL MODEL SIMULATION on May 1-18 data:")
print(f"   (current taken trades + {len(april_would_also_take)} re-enabled by removing post-Apr-28 gates)")
print()

# What gates are being re-enabled?
gate_counts = Counter(r.get("gate_name","") for r in april_would_also_take)
print("   Gates removed (trades re-enabled):")
for g, n in gate_counts.most_common():
    print(f"     {g}: {n}")
print()

april_on_may = may_taken + april_would_also_take

# Build expiry map for the combined set
combined_by_exp = defaultdict(set)
for r in april_on_may:
    combined_by_exp[r.get("close_ts", r.get("close_time",""))].add(r.get("side",""))
avg_per_exp_sim = len(april_on_may) / max(len(combined_by_exp), 1)
yesno_exp_sim   = sum(1 for s in combined_by_exp.values() if len(s) > 1)

# Combined PnL
resolved_taken   = [r for r in may_taken            if norm_res(r.get("resolved_yes","")) is not None]
resolved_reenabl = [r for r in april_would_also_take if norm_res(r.get("resolved_yes","")) is not None]

pnl_taken   = calc_pnl(resolved_taken,   ba_key="bet_amount")
pnl_reenabl = calc_pnl(resolved_reenabl, ba_key="kelly_fraction")
pnl_combined = pnl_taken + pnl_reenabl

all_resolved = resolved_taken + resolved_reenabl
wins_all = sum(
    1 for r in all_resolved
    if (r.get("side","")=="yes" and norm_res(r["resolved_yes"])==1)
    or (r.get("side","")=="no"  and norm_res(r["resolved_yes"])==0)
)
wr_combined = wins_all / len(all_resolved) if all_resolved else 0

print(f"  April model on May:")
print(f"    total trades={len(all_resolved):5d}  WR={wr_combined:.1%}  PnL=${pnl_combined:+,.0f}")
print(f"    from taken  ={len(resolved_taken):5d}  PnL=${pnl_taken:+,.0f}")
print(f"    re-enabled  ={len(resolved_reenabl):5d}  PnL=${pnl_reenabl:+,.0f}")
print(f"    expiries={len(combined_by_exp)}  avg trades/expiry={avg_per_exp_sim:.1f}  expiries w/ YES+NO={yesno_exp_sim}")

# Breakdown by side
yes_sim = [r for r in all_resolved if r.get("side","") == "yes"]
no_sim  = [r for r in all_resolved if r.get("side","") == "no"]
if yes_sim:
    wy = sum(1 for r in yes_sim if norm_res(r["resolved_yes"])==1)/len(yes_sim)
    be_y = sum(float(r.get("pm",r.get("p_market",0.5))) for r in yes_sim)/len(yes_sim)
    print(f"    YES n={len(yes_sim):4d}  WR={wy:.1%}  BE={be_y:.1%}  edge={wy-be_y:+.1%}")
if no_sim:
    wn = sum(1 for r in no_sim if norm_res(r["resolved_yes"])==0)/len(no_sim)
    be_n = 1-sum(float(r.get("pm",r.get("p_market",0.5))) for r in no_sim)/len(no_sim)
    print(f"    NO  n={len(no_sim):4d}  WR={wn:.1%}  BE={be_n:.1%}  edge={wn-be_n:+.1%}")

print()
print(SEP2)
print("SUMMARY:")
print(f"  April model actual  (Apr 20-28):        WR={0:.0%}  PnL=$+1,922  (from prior analysis)")
print(f"  Current model       (May 1-18 actual):  WR=45.8%  PnL=$-670")
pnl_diff = pnl_combined - pnl_taken
print(f"  April model sim     (May 1-18 data):    WR={wr_combined:.1%}  PnL=${pnl_combined:+,.0f}  (delta vs current: ${pnl_diff:+,.0f})")
print()
print("NOTE: Re-enabled trades sized by kelly_fraction × bankroll at block time.")
print("      Kelly fractions reflect the gate-blocked evaluation, not recomputed.")
print(SEP)
