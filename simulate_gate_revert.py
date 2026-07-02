#!/usr/bin/env python3
"""
simulate_gate_revert.py

Simulates May BTC performance if the post-Apr-28 gate additions were reverted.

Pre-Apr-28 gate stack (paper_trade_runner_pre_stoch_gate.py, the best BTC era):
  - streak_gate        (present Apr 28, keep)
  - btc_vol1_gate      (present Apr 28, keep)
  - counter_tape       (present Apr 28, keep)

Gates added AFTER Apr 28 that we test removing:
  Apr 30 (SMC reform): smc_gate, btc_no_smc_demand_gate
  May 1  (vol reform):  btc_no_z_gate, no_pm_floor (may predate this)
  May 6  (ADX5):        btc_adx5_gate
  May 17 (liq/near-ITM): liq_cascade_gate, near_atm_ema_gate,
                          strong_trend_nearatm_gate, btc_deepno_neutral_gate
  May 18 (LGBM gate):   btc_gbdt_gate

For each scenario, blocked trades that would be re-enabled are pulled from
blocked_trades.csv and priced using kelly_fraction × bankroll.
Current taken trades come from paper_trades.csv (decision='trade').
"""

import csv
import math
from collections import defaultdict

PAPER_TRADES = "results/paper_trades.csv"
BLOCKED      = "results/blocked_trades.csv"

# ─── helpers ────────────────────────────────────────────────────────────────

def norm_res(v):
    v = str(v).strip().lower()
    if v in ("true", "1"):  return 1
    if v in ("false", "0"): return 0
    return None

def calc_pnl_rows(rows, ba_key="bet_amount"):
    """Dollar P&L for a list of rows using bet_amount or kelly × bankroll."""
    pnl = 0.0
    for r in rows:
        res = norm_res(r.get("resolved_yes", ""))
        if res is None:
            continue
        side = r.get("side", r.get("side", "")).strip().lower()
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

def summarize(rows, label, ba_key="bet_amount"):
    resolved = [r for r in rows if norm_res(r.get("resolved_yes", "")) is not None]
    if not resolved:
        print(f"  {label}: n={len(rows)}  (no resolved outcomes)")
        return
    wins = sum(
        1 for r in resolved
        if (r.get("side","") == "yes" and norm_res(r["resolved_yes"]) == 1)
        or (r.get("side","") == "no"  and norm_res(r["resolved_yes"]) == 0)
    )
    wr = wins / len(resolved)
    pnl = calc_pnl_rows(resolved, ba_key=ba_key)
    yes_r = [r for r in resolved if r.get("side","") == "yes"]
    no_r  = [r for r in resolved if r.get("side","") == "no"]
    be_yes = sum(float(r.get("pm", r.get("p_market", 0.5))) for r in yes_r) / len(yes_r) if yes_r else 0
    be_no  = 1 - sum(float(r.get("pm", r.get("p_market", 0.5))) for r in no_r)  / len(no_r)  if no_r  else 0
    print(f"  {label}")
    print(f"    n={len(resolved):5d}  WR={wr:.1%}  PnL=${pnl:+,.0f}")
    if yes_r:
        wy = sum(1 for r in yes_r if norm_res(r["resolved_yes"])==1) / len(yes_r)
        print(f"    YES n={len(yes_r):5d}  WR={wy:.1%}  BE={be_yes:.1%}  edge={wy-be_yes:+.1%}")
    if no_r:
        wn = sum(1 for r in no_r  if norm_res(r["resolved_yes"])==0) / len(no_r)
        print(f"    NO  n={len(no_r):5d}  WR={wn:.1%}  BE={be_no:.1%}  edge={wn-be_no:+.1%}")

# ─── load data ───────────────────────────────────────────────────────────────

with open(PAPER_TRADES, newline="") as f:
    all_pt = list(csv.DictReader(f))

with open(BLOCKED, newline="") as f:
    all_bl = list(csv.DictReader(f))

# May BTC only
may_trades  = [r for r in all_pt if r.get("logged_at", "")[:10] >= "2026-05-01"
               and "BTC" in r.get("contract_ticker", "")
               and r.get("decision", "").strip() == "trade"]
may_blocked = [r for r in all_bl if r.get("logged_at", "")[:10] >= "2026-05-01"
               and r.get("asset", "") == "BTC"]

# ─── gate sets ───────────────────────────────────────────────────────────────

# Gates present in the pre-stoch-gate (Apr 28) model — KEEP these
KEEP_GATES = {"streak_gate", "btc_vol1_gate", "counter_tape"}

# Gates added after Apr 28 (simulate removing in each scenario)
SMC_GATES   = {"smc_gate", "btc_no_smc_demand_gate"}
Z_GATE      = {"btc_no_z_gate"}
PM_FLOOR    = {"no_pm_floor"}
ADX5        = {"btc_adx5_gate"}
LIQ_NEAR    = {"liq_cascade_gate", "near_atm_ema_gate",
               "strong_trend_nearatm_gate", "btc_deepno_neutral_gate"}
LGBM        = {"btc_gbdt_gate"}
MISC        = {"btc_stoch_no_gate", "rvol_gate", "btc_adx_gate",
               "btc_contra_bar_gate", "btc_falling_knife_gate",
               "btc_nopup_gate", "btc_body_bp_gate", "btc_otmlow_gate",
               "btc_otm_neutral_gate", "bear_drift", "btc_exhaustion_gate",
               "btc_spread_gate", "btc_tau_gate", "btc_struct_gate",
               "btc_no_highpm_bearema_gate", "btc_no_wrongdir_gate"}

ALL_POST_APR28 = SMC_GATES | Z_GATE | PM_FLOOR | ADX5 | LIQ_NEAR | LGBM | MISC

def blocked_by(gates):
    return [r for r in may_blocked if r.get("gate_name", "") in gates]

# ─── scenarios ───────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72

print(SEP)
print("  BTC May 2026 Gate Revert Simulation")
print(f"  Baseline: May 1-18 actual trades")
print(SEP)
print()

# 0. Baseline
print("0. BASELINE — current May trades (all gates active):")
summarize(may_trades, "May trades taken", ba_key="bet_amount")
print()

# 1. Per-gate blocked outcome breakdown
print(SEP2)
print("1. BLOCKED TRADE OUTCOMES by gate (if gate were removed):")
print()
gate_groups = [
    ("smc_gate (YES only)",          {"smc_gate"}),
    ("btc_no_smc_demand_gate (NO)",  {"btc_no_smc_demand_gate"}),
    ("btc_no_z_gate (NO)",           {"btc_no_z_gate"}),
    ("no_pm_floor",                  {"no_pm_floor"}),
    ("streak_gate (keep in old stack)", {"streak_gate"}),
    ("liq_cascade_gate",             {"liq_cascade_gate"}),
    ("btc_adx5_gate",                {"btc_adx5_gate"}),
    ("All post-Apr-28 gates",        ALL_POST_APR28),
]
for label, gates in gate_groups:
    b = blocked_by(gates)
    if not b:
        print(f"  {label}: no blocks")
        continue
    summarize(b, label, ba_key="kelly_fraction")
    print()

# 2. Simulated scenarios
print(SEP2)
print("2. SIMULATED P&L SCENARIOS:")
print()

scenarios = [
    ("A: Remove smc_gate only",
     SMC_GATES),
    ("B: Remove SMC gates + NO-suppressing (btc_no_z + btc_no_smc_demand)",
     SMC_GATES | Z_GATE),
    ("C: Remove all post-Apr-28 except streak+vol1+counter_tape",
     ALL_POST_APR28),
]

for scenario_name, remove_gates in scenarios:
    extra = blocked_by(remove_gates)
    combined = may_trades + extra
    # Use kelly×bankroll for blocked rows, bet_amount for taken trades
    resolved_taken  = [r for r in may_trades if norm_res(r.get("resolved_yes","")) is not None]
    resolved_extra  = [r for r in extra       if norm_res(r.get("resolved_yes","")) is not None]

    pnl_taken = calc_pnl_rows(resolved_taken, ba_key="bet_amount")
    pnl_extra = calc_pnl_rows(resolved_extra, ba_key="kelly_fraction")
    pnl_total = pnl_taken + pnl_extra

    wins_taken = sum(1 for r in resolved_taken
                     if (r["side"]=="yes" and norm_res(r["resolved_yes"])==1)
                     or (r["side"]=="no"  and norm_res(r["resolved_yes"])==0))
    wins_extra = sum(1 for r in resolved_extra
                     if (r.get("side","")=="yes" and norm_res(r["resolved_yes"])==1)
                     or (r.get("side","")=="no"  and norm_res(r["resolved_yes"])==0))

    total_res = len(resolved_taken) + len(resolved_extra)
    total_wins = wins_taken + wins_extra
    wr_blended = total_wins / total_res if total_res else 0

    print(f"  {scenario_name}")
    print(f"    Trades: {len(may_trades)} taken + {len(extra)} re-enabled = {len(combined)} total")
    print(f"    WR: {wr_blended:.1%}  |  PnL from taken: ${pnl_taken:+,.0f}  |  Added from re-enabled: ${pnl_extra:+,.0f}")
    print(f"    NET PnL: ${pnl_total:+,.0f}  (vs baseline ${pnl_taken:+,.0f})")
    print()

print(SEP)
print("NOTES:")
print("  - 'kelly_fraction' sizing uses the stored kelly_frac × bankroll at block time.")
print("  - Blended WR mixes fixed-bet trades and kelly-bet blocked rows — directional only.")
print("  - smc_gate blocks YES-only; btc_no_z + btc_no_smc_demand block NO-only.")
print("  - streak_gate is in BOTH old and new stacks — kept in all scenarios.")
print(SEP)
