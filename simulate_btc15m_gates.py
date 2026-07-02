"""
BTC 15m scan archive gate simulation.
Simulates proposed Gate A and Gate B (and sub-scenarios) against the baseline model.
"""

import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
ARCHIVE_PATH = "/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/btc_scan_archive_15m.csv"
MIN_EDGE     = 0.04
KELLY_MULT   = 0.30
KELLY_CAP    = 0.06
BANKROLL     = 1000.0
FEE_RATE     = 0.07   # applied to min(pm, 1-pm)

# ── Load & deduplicate ───────────────────────────────────────────────────────
df_raw = pd.read_csv(ARCHIVE_PATH, low_memory=False)
print(f"Raw rows: {len(df_raw)}")

# Latest logged_at per contract_ticker
df_raw["logged_at"] = pd.to_datetime(df_raw["logged_at"], utc=True, errors="coerce")
df = (df_raw.sort_values("logged_at")
            .groupby("contract_ticker", as_index=False)
            .last())
print(f"After dedup (one row per contract): {len(df)}")

# ── Filter to resolved ───────────────────────────────────────────────────────
df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
df = df[df["resolved_yes"].notna()].copy()
print(f"Resolved contracts: {len(df)}")

# ── Data validation ──────────────────────────────────────────────────────────
print("\n--- Data validation ---")
print(f"ema_bias unique values: {sorted(df['ema_bias'].dropna().unique())}")
print(f"stoch_k_15m range: min={df['stoch_k_15m'].min():.1f}, max={df['stoch_k_15m'].max():.1f}, "
      f"mean={df['stoch_k_15m'].mean():.1f}")
print(f"p_market range: min={df['p_market'].min():.3f}, max={df['p_market'].max():.3f}")
print(f"body_15m range: min={df['body_15m'].min():.3f}, max={df['body_15m'].max():.3f}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def fee(pm):
    return FEE_RATE * np.minimum(pm, 1 - pm)

def kelly_frac(edge, pm_risk):
    return np.minimum(edge / pm_risk * KELLY_MULT, KELLY_CAP)

def compute_pnl(row, side):
    """Compute PnL for a bet on 'YES' or 'NO' given the row."""
    pm  = row["p_market"]
    f   = fee(pm)
    won = bool(row["resolved_yes"])

    if side == "YES":
        edge     = row["p_model_yes"] - pm
        pm_risk  = pm
        frac     = kelly_frac(edge, pm_risk)
        bet      = frac * BANKROLL
        return bet * (1 - pm - f) if won else -bet * (pm + f)
    else:  # NO
        edge     = pm - row["p_model_no"]
        pm_risk  = 1 - pm
        frac     = kelly_frac(edge, pm_risk)
        bet      = frac * BANKROLL
        return bet * (pm - f) if not won else -bet * (1 - pm + f)

# ── Build baseline decision ──────────────────────────────────────────────────
results = []
for _, row in df.iterrows():
    pm           = row["p_market"]
    edge_yes     = row["p_model_yes"] - pm
    edge_no      = pm - row["p_model_no"]

    if edge_yes > edge_no and edge_yes > MIN_EDGE:
        side = "YES"
    elif edge_no > edge_yes and edge_no > MIN_EDGE:
        side = "NO"
    else:
        side = None

    pnl = compute_pnl(row, side) if side else 0.0
    results.append({
        "contract_ticker": row["contract_ticker"],
        "side"           : side,
        "resolved_yes"   : row["resolved_yes"],
        "p_market"       : pm,
        "edge_yes"       : edge_yes,
        "edge_no"        : edge_no,
        "stoch_k_15m"    : row["stoch_k_15m"],
        "ema_bias"       : row["ema_bias"],
        "body_15m"       : row["body_15m"],
        "baseline_pnl"   : pnl,
    })

sim = pd.DataFrame(results)

# ── Stats helper ─────────────────────────────────────────────────────────────
def stats(df_s, label, pnl_col="baseline_pnl", side_filter=None):
    """Print statistics for a scenario subset."""
    if side_filter:
        subset = df_s[df_s["side"] == side_filter]
    else:
        subset = df_s[df_s["side"].notna()]
    n      = len(subset)
    pnl    = subset[pnl_col].sum()
    if n == 0:
        print(f"  {label}: 0 trades")
        return pnl, 0, 0.0
    wins   = (subset[pnl_col] > 0).sum()
    wr     = wins / n * 100
    print(f"  {label}: n={n}, WR={wr:.1f}%, PnL=${pnl:.2f}")
    return pnl, n, wr

# ── Baseline ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE")
print("="*60)
baseline_trades = sim[sim["side"].notna()]
n_base   = len(baseline_trades)
pnl_base = baseline_trades["baseline_pnl"].sum()
wins_base = (baseline_trades["baseline_pnl"] > 0).sum()
wr_base  = wins_base / n_base * 100 if n_base > 0 else 0
print(f"Total contracts evaluated (resolved): {len(sim)}")
print(f"Baseline trades taken: {n_base}")
print(f"  YES bets: {(baseline_trades['side']=='YES').sum()}")
print(f"  NO bets:  {(baseline_trades['side']=='NO').sum()}")
print(f"Baseline WR: {wr_base:.1f}%")
print(f"Baseline PnL: ${pnl_base:.2f}")

# ── Scenario runner ──────────────────────────────────────────────────────────
def run_gate_scenario(sim_df, name, gate_condition_fn, gate_desc,
                      apply_to="NO", baseline_pnl=pnl_base,
                      baseline_n=n_base, baseline_wr=wr_base):
    """
    Apply a gate that blocks trades matching `apply_to` side when gate_condition_fn(row) is True.
    Reports wins blocked, losses blocked, PnL delta.
    """
    print("\n" + "="*60)
    print(f"Scenario: {name}")
    print(f"Gate condition: {gate_desc}")
    print("="*60)

    gated_pnl = 0.0
    gated_n   = 0
    gated_wins = 0
    wins_blocked  = 0
    losses_blocked = 0

    for _, row in sim_df.iterrows():
        side = row["side"]
        bl_pnl = row["baseline_pnl"]

        if side is None:
            continue  # not a baseline trade, skip

        # Check if this trade is blocked by the gate
        blocked = (side == apply_to) and gate_condition_fn(row)

        if blocked:
            if bl_pnl > 0:
                wins_blocked += 1
            else:
                losses_blocked += 1
            # Trade is skipped entirely (pnl = 0)
        else:
            gated_pnl  += bl_pnl
            gated_n    += 1
            if bl_pnl > 0:
                gated_wins += 1

    gated_wr = gated_wins / gated_n * 100 if gated_n > 0 else 0.0
    delta    = gated_pnl - baseline_pnl

    print(f"Contracts affected: {wins_blocked + losses_blocked} (of {baseline_n} baseline trades)")
    print(f"  - wins blocked:   {wins_blocked}")
    print(f"  - losses blocked: {losses_blocked}")
    print(f"Baseline PnL: ${baseline_pnl:.2f}")
    print(f"Gated PnL:    ${gated_pnl:.2f}")
    print(f"Delta:        ${delta:+.2f}")
    print(f"WR change:    {baseline_wr:.1f}% → {gated_wr:.1f}%")

    return gated_pnl, delta

# ── Gate A: Block NO when stoch_k_15m >= 80 AND ema_bias == 1 ───────────────
def gate_a(row):
    stoch = row["stoch_k_15m"]
    ema   = row["ema_bias"]
    try:
        stoch_ok = float(stoch) >= 80
    except (TypeError, ValueError):
        stoch_ok = False
    try:
        ema_ok = float(ema) == 1
    except (TypeError, ValueError):
        ema_ok = str(ema) == "1"
    return stoch_ok and ema_ok

run_gate_scenario(sim, "Gate A", gate_a,
                  "Block NO bets when stoch_k_15m >= 80 AND ema_bias == 1")

# ── Gate B: Block NO when p_market in [0.70, 0.80) ──────────────────────────
def gate_b_range(row):
    pm = row["p_market"]
    return 0.70 <= pm < 0.80

run_gate_scenario(sim, "Gate B — pm in [0.70, 0.80)", gate_b_range,
                  "Block NO bets when 0.70 <= p_market < 0.80")

# ── Gate B extra: pm >= 0.80 (sanity check) ─────────────────────────────────
def gate_b_high(row):
    return row["p_market"] >= 0.80

run_gate_scenario(sim, "Gate B sanity — pm >= 0.80", gate_b_high,
                  "Block NO bets when p_market >= 0.80")

# ── Gate A+B combined ────────────────────────────────────────────────────────
def gate_ab(row):
    return gate_a(row) or gate_b_range(row)

run_gate_scenario(sim, "Gate A + Gate B combined", gate_ab,
                  "Block NO when (stoch_k_15m>=80 & ema_bias==1) OR pm in [0.70,0.80)")

# ── Positive signal: stoch_k_15m in [20, 40) ────────────────────────────────
print("\n" + "="*60)
print("Positive signal: stoch_k_15m in [20, 40) — all baseline trades")
print("="*60)
in_range   = baseline_trades[baseline_trades["stoch_k_15m"].between(20, 40, inclusive="left")]
out_range  = baseline_trades[~baseline_trades["stoch_k_15m"].between(20, 40, inclusive="left")]
n_in       = len(in_range)
pnl_in     = in_range["baseline_pnl"].sum()
wr_in      = (in_range["baseline_pnl"] > 0).sum() / n_in * 100 if n_in else 0
n_out      = len(out_range)
pnl_out    = out_range["baseline_pnl"].sum()
wr_out     = (out_range["baseline_pnl"] > 0).sum() / n_out * 100 if n_out else 0

print(f"Trades with stoch_k_15m in [20,40): n={n_in} ({n_in/n_base*100:.1f}% of baseline)")
print(f"  WR={wr_in:.1f}%, PnL=${pnl_in:.2f}")
print(f"Trades with stoch_k_15m outside [20,40): n={n_out}")
print(f"  WR={wr_out:.1f}%, PnL=${pnl_out:.2f}")

# Breakdown by side within stoch range
print("\n  Breakdown within stoch_k_15m [20,40):")
for s in ["YES","NO"]:
    sub = in_range[in_range["side"]==s]
    if len(sub) > 0:
        wr_s = (sub["baseline_pnl"]>0).sum()/len(sub)*100
        print(f"    {s}: n={len(sub)}, WR={wr_s:.1f}%, PnL=${sub['baseline_pnl'].sum():.2f}")

# ── Positive signal: body_15m < 0.30 ────────────────────────────────────────
print("\n" + "="*60)
print("Positive signal: body_15m < 0.30 — all baseline trades")
print("="*60)
body_low   = baseline_trades[baseline_trades["body_15m"] < 0.30]
body_high  = baseline_trades[baseline_trades["body_15m"] >= 0.30]
n_bl       = len(body_low)
pnl_bl     = body_low["baseline_pnl"].sum()
wr_bl      = (body_low["baseline_pnl"]>0).sum()/n_bl*100 if n_bl else 0
n_bh       = len(body_high)
pnl_bh     = body_high["baseline_pnl"].sum()
wr_bh      = (body_high["baseline_pnl"]>0).sum()/n_bh*100 if n_bh else 0

print(f"Trades with body_15m < 0.30: n={n_bl} ({n_bl/n_base*100:.1f}% of baseline)")
print(f"  WR={wr_bl:.1f}%, PnL=${pnl_bl:.2f}")
print(f"Trades with body_15m >= 0.30: n={n_bh}")
print(f"  WR={wr_bh:.1f}%, PnL=${pnl_bh:.2f}")

print("\n  Breakdown within body_15m < 0.30:")
for s in ["YES","NO"]:
    sub = body_low[body_low["side"]==s]
    if len(sub) > 0:
        wr_s = (sub["baseline_pnl"]>0).sum()/len(sub)*100
        print(f"    {s}: n={len(sub)}, WR={wr_s:.1f}%, PnL=${sub['baseline_pnl'].sum():.2f}")

# ── Cross-tabulation: stoch_k_15m bins vs side ───────────────────────────────
print("\n" + "="*60)
print("stoch_k_15m distribution across NO trades (baseline)")
print("="*60)
no_trades = baseline_trades[baseline_trades["side"]=="NO"].copy()
bins = [0, 20, 40, 60, 80, 101]
labels = ["<20","20-40","40-60","60-80",">=80"]
no_trades["stoch_bin"] = pd.cut(no_trades["stoch_k_15m"], bins=bins, labels=labels, right=False)
grp = no_trades.groupby("stoch_bin", observed=True)["baseline_pnl"].agg(["count","sum",lambda x:(x>0).sum()])
grp.columns = ["count","pnl","wins"]
grp["wr"] = grp["wins"]/grp["count"]*100
print(grp.to_string())

# Gate A breakdown: how many NO trades have stoch>=80 AND ema_bias==1?
print("\n" + "="*60)
print("Gate A detailed breakdown")
print("="*60)
no_t = baseline_trades[baseline_trades["side"]=="NO"].copy()
no_t["stoch_k_15m_num"] = pd.to_numeric(no_t["stoch_k_15m"], errors="coerce")
no_t["ema_bias_num"]    = pd.to_numeric(no_t["ema_bias"], errors="coerce")
no_stoch_hi  = no_t[no_t["stoch_k_15m_num"] >= 80]
no_stoch_and_ema = no_t[(no_t["stoch_k_15m_num"] >= 80) & (no_t["ema_bias_num"] == 1)]
print(f"NO trades with stoch_k_15m >= 80: {len(no_stoch_hi)}")
if len(no_stoch_hi):
    wr_h = (no_stoch_hi["baseline_pnl"]>0).sum()/len(no_stoch_hi)*100
    print(f"  WR={wr_h:.1f}%, PnL=${no_stoch_hi['baseline_pnl'].sum():.2f}")
print(f"NO trades with stoch_k_15m >= 80 AND ema_bias==1: {len(no_stoch_and_ema)}")
if len(no_stoch_and_ema):
    wr_e = (no_stoch_and_ema["baseline_pnl"]>0).sum()/len(no_stoch_and_ema)*100
    print(f"  WR={wr_e:.1f}%, PnL=${no_stoch_and_ema['baseline_pnl'].sum():.2f}")
    print(f"  ema_bias values in this group: {sorted(no_stoch_and_ema['ema_bias'].dropna().unique())}")

# Gate B breakdown: p_market distribution for NO bets
print("\n" + "="*60)
print("Gate B detailed breakdown — p_market bins for NO trades")
print("="*60)
pm_bins = [0, 0.50, 0.60, 0.70, 0.80, 1.01]
pm_labels = ["<0.50","0.50-0.60","0.60-0.70","0.70-0.80",">=0.80"]
no_t["pm_bin"] = pd.cut(no_t["p_market"], bins=pm_bins, labels=pm_labels, right=False)
grp2 = no_t.groupby("pm_bin", observed=True)["baseline_pnl"].agg(["count","sum",lambda x:(x>0).sum()])
grp2.columns = ["count","pnl","wins"]
grp2["wr"] = grp2["wins"]/grp2["count"]*100
print(grp2.to_string())

print("\nDone.")
