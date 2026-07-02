"""
simulate_model_distribution.py

Compare three probability model variants on resolved BTC paper trades:

  Baseline : current model — p_yes_model from CSV (uses composite k_drift internally;
             stored as 1 - norm.cdf(z_adj) for both YES and NO sides)

  Scenario A: z_adj = z_score (strip k_drift entirely; z_drift_override=0 path)
              p_yes = 1 - norm.cdf(z_score)

  Scenario B: keep z_adj from current model, swap distribution to Student-t
              p_yes = _p_yes_t(z_adj)  where z_adj is recovered from p_yes_model

Key insight: for both YES and NO trades, p_yes_model = 1 - norm.cdf(z_adj), so
z_adj = -norm.ppf(p_yes_model). Scenario B applies Student-t to this z_adj,
which is the correct apples-to-apples comparison for the proposed distribution change.

Edge thresholds:
  YES: edge = p_yes_model - pm  → taken if >= EDGE_THRESH
  NO : edge = pm - p_yes_model  → taken if >= EDGE_THRESH

Flat $10 per trade.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

FLAT        = 10.0
EDGE_THRESH = 0.04

# ── Student-t helper ──────────────────────────────────────────────────────────
def _p_yes_t(z_adj: float, offset_frac: float) -> float:
    """Student-t YES probability; offset_frac = (strike-spot)/spot fraction."""
    a  = abs(offset_frac)
    nu = 4.0 if a <= 0.001 else (3.0 if a <= 0.003 else (2.5 if a <= 0.010 else 2.1))
    sc = math.sqrt((nu - 2) / nu)
    return float(np.clip(1 - student_t.cdf(z_adj / sc, df=nu), 0.01, 0.99))

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("results/paper_trades.csv", low_memory=False)
btc = df[
    df["contract_ticker"].str.contains("BTC", na=False)
    & (df["decision"] == "trade")
    & df["resolved_yes"].notna()
].copy()

for col in ["z_score", "offset_pct", "p_market", "p_yes_model", "resolved_yes"]:
    btc[col] = pd.to_numeric(btc[col], errors="coerce")
btc = btc.dropna(subset=["z_score", "offset_pct", "p_market", "p_yes_model", "resolved_yes"])
btc["offset_frac"] = btc["offset_pct"] / 100.0

print(f"BTC resolved executed trades: {len(btc)}")
print(f"YES: {(btc['side']=='yes').sum()}  NO: {(btc['side']=='no').sum()}")
print()

# ── Recover z_adj from p_yes_model ────────────────────────────────────────────
# p_yes_model = 1 - norm.cdf(z_adj)  for both sides
# → z_adj = -norm.ppf(p_yes_model)
btc["z_adj"] = -norm.ppf(btc["p_yes_model"].clip(0.01, 0.99))

# ── Scenario A: strip drift (z_adj = z_score) ─────────────────────────────────
btc["p_A"] = 1 - norm.cdf(btc["z_score"])

# ── Scenario B: same z_adj as baseline, swap to Student-t ────────────────────
btc["p_B"] = btc.apply(lambda r: _p_yes_t(r["z_adj"], r["offset_frac"]), axis=1)

pm    = btc["p_market"]
pbase = btc["p_yes_model"]
pA    = btc["p_A"]
pB    = btc["p_B"]

# ── Edge under each scenario ──────────────────────────────────────────────────
is_yes = btc["side"] == "yes"
is_no  = btc["side"] == "no"

# Baseline edge (YES: p_yes_model - pm;  NO: pm - p_yes_model)
btc["edge_base"] = np.where(is_yes, pbase - pm, pm - pbase)

# Scenario A edge
btc["edge_A"] = np.where(is_yes, pA - pm, pm - pA)

# Scenario B edge
btc["edge_B"] = np.where(is_yes, pB - pm, pm - pB)

# ── PnL helpers ───────────────────────────────────────────────────────────────
def pnl_row(r):
    pm_val = r["p_market"]
    if r["side"] == "yes":
        return (1 - pm_val) * FLAT if r["resolved_yes"] == 1 else -pm_val * FLAT
    else:
        return pm_val * FLAT if r["resolved_yes"] == 0 else -(1 - pm_val) * FLAT

btc["pnl"] = btc.apply(pnl_row, axis=1)

def report(mask, label):
    sub = btc[mask]
    if len(sub) == 0:
        print(f"  {label}: n=0")
        return
    yes = sub[is_yes & mask]
    no  = sub[is_no  & mask]
    wr_y = yes["resolved_yes"].mean()       if len(yes) else float("nan")
    be_y = yes["p_market"].mean()           if len(yes) else float("nan")
    wr_n = (1-no["resolved_yes"]).mean()    if len(no)  else float("nan")
    be_n = (1-no["p_market"]).mean()        if len(no)  else float("nan")
    pnl  = sub["pnl"].sum()
    print(f"  {label}: n={len(sub)} ({len(yes)}Y/{len(no)}N)  "
          f"YES WR={wr_y:.1%}/BE={be_y:.1%}  NO WR={wr_n:.1%}/BE={be_n:.1%}  "
          f"PnL=${pnl:+.0f}")

mask_base = btc["edge_base"] >= EDGE_THRESH   # should be all (they were taken)
mask_A    = btc["edge_A"]    >= EDGE_THRESH
mask_B    = btc["edge_B"]    >= EDGE_THRESH

print("=" * 72)
print("Overall (flat $10/trade)")
print("=" * 72)
report(mask_base, "Baseline                      ")
report(mask_A,    "Scenario A (lognormal no drift)")
report(mask_B,    "Scenario B (Student-t, same z_adj)")

print()
print(f"Scenario A drops {(~mask_A).sum()} trades; Scenario B drops {(~mask_B).sum()} trades")
dropped_A = btc[mask_base & ~mask_A]
dropped_B = btc[mask_base & ~mask_B]
if len(dropped_A):
    report(mask_base & ~mask_A, "Dropped by A")
    print(f"    → Dropping them saves: ${-dropped_A['pnl'].sum():+.0f}")
if len(dropped_B):
    report(mask_base & ~mask_B, "Dropped by B")
    print(f"    → Dropping them saves: ${-dropped_B['pnl'].sum():+.0f}")

# ── Calibration: actual WR vs predicted p_model ───────────────────────────────
print()
print("=" * 72)
print("Calibration — YES trades: actual WR vs predicted p_yes (all 3 models)")
print("=" * 72)
yes_t = btc[is_yes].copy()

def cal_table(col, label):
    print(f"\n[{label}]")
    yes_t["bucket"] = pd.cut(yes_t[col], bins=[0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0])
    g = yes_t.groupby("bucket", observed=True).agg(
        n=("resolved_yes", "count"),
        actual_wr=("resolved_yes", "mean"),
        pred_p=(col, "mean")
    )
    g["delta"] = g["actual_wr"] - g["pred_p"]
    print(g.to_string())
    bs = np.mean((yes_t[col].values - yes_t["resolved_yes"].values) ** 2)
    print(f"  Brier score: {bs:.5f}  (mean_abs_delta: {g['delta'].abs().mean():.4f})")

cal_table("p_yes_model", "Baseline")
cal_table("p_A",         "Scenario A (lognormal no drift)")
cal_table("p_B",         "Scenario B (Student-t same z_adj)")

# ── Calibration: NO trades ────────────────────────────────────────────────────
print()
print("=" * 72)
print("Calibration — NO trades: actual NO WR vs predicted NO prob (1-p_yes_model)")
print("=" * 72)
no_t = btc[is_no].copy()
no_t["actual_no_wr"] = 1 - no_t["resolved_yes"]
no_t["pred_no_base"] = 1 - no_t["p_yes_model"]
no_t["pred_no_A"]    = 1 - no_t["p_A"]
no_t["pred_no_B"]    = 1 - no_t["p_B"]
for col, label in [("pred_no_base","Baseline"), ("pred_no_A","Scen A"), ("pred_no_B","Scen B")]:
    bs = np.mean((no_t[col].values - no_t["actual_no_wr"].values) ** 2)
    mae = (no_t[col] - no_t["actual_no_wr"]).abs().mean()
    mean_pred = no_t[col].mean(); mean_act = no_t["actual_no_wr"].mean()
    print(f"  {label}: Brier={bs:.5f}  MAE={mae:.4f}  mean_pred={mean_pred:.3f} actual={mean_act:.3f}")

# ── Nu distribution for Scenario B ────────────────────────────────────────────
print()
print("=" * 72)
print("Scenario B — nu selected per offset bucket")
print("=" * 72)
def get_nu(off_frac):
    a = abs(off_frac)
    return 4.0 if a <= 0.001 else (3.0 if a <= 0.003 else (2.5 if a <= 0.010 else 2.1))
yes_t["nu"] = yes_t["offset_frac"].apply(get_nu)
print(yes_t["nu"].value_counts().sort_index())

# ── p_model shift: baseline → Scenario B ─────────────────────────────────────
print()
print("=" * 72)
print("p_model shift: Baseline → Scenario B  (YES trades)")
print("=" * 72)
yes_t["shift"] = yes_t["p_B"] - yes_t["p_yes_model"]
print(yes_t["shift"].describe())
print(f"  Baseline mean p_yes : {yes_t['p_yes_model'].mean():.4f}")
print(f"  Scenario B mean p_yes: {yes_t['p_B'].mean():.4f}")
print(f"  OTM YES (offset>0): "
      f"Baseline {yes_t.loc[yes_t['offset_pct']>0,'p_yes_model'].mean():.4f} → "
      f"B {yes_t.loc[yes_t['offset_pct']>0,'p_B'].mean():.4f}")
print(f"  ITM YES (offset<0): "
      f"Baseline {yes_t.loc[yes_t['offset_pct']<0,'p_yes_model'].mean():.4f} → "
      f"B {yes_t.loc[yes_t['offset_pct']<0,'p_B'].mean():.4f}")
