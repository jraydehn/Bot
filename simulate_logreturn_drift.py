"""
simulate_logreturn_drift.py

PnL simulation: add 6h rolling log-return drift to the current BTC model.

Methodology:
  - Recover current z_adj from p_yes_model: z_adj = -norm.ppf(p_yes_model)
    (works for both YES and NO since both store 1 - norm.cdf(z_adj))
  - Compute 6h drift: mu = rolling_mean(log_ret_1h, 6)
  - z_drift = mu * (tau_min/60) / sigma_tau  (scale mu to tau, divide by sigma_tau)
  - z_adj_new = z_adj - z_drift
  - p_yes_new = 1 - norm.cdf(z_adj_new)
  - YES edge_new = p_yes_new - pm;  NO edge_new = pm - p_yes_new

Flat $10/trade. Edge threshold = 0.04.
"""

import math, glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

FLAT        = 10.0
EDGE_THRESH = 0.04
WINDOW      = 6      # hours

# ── Load 1h BTC prices ────────────────────────────────────────────────────────
BASE = Path(".")
f1h  = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
ohlcv = pd.read_parquet(f1h[-1])
ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
close  = ohlcv["close"].astype(float).sort_index()
log_ret = np.log(close / close.shift(1))
mu_6h   = log_ret.rolling(WINDOW).mean()

# ── Load live trades ──────────────────────────────────────────────────────────
pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt = pt[
    pt["contract_ticker"].str.contains("BTC", na=False)
    & (pt["decision"] == "trade")
    & pt["resolved_yes"].notna()
].copy()
for col in ["z_score","offset_pct","p_market","p_yes_model","resolved_yes",
            "vol_eff","tau_minutes"]:
    pt[col] = pd.to_numeric(pt[col], errors="coerce")
pt["logged_at"] = pd.to_datetime(pt["logged_at"], utc=True)
pt = pt.dropna(subset=["z_score","offset_pct","p_market","p_yes_model",
                        "resolved_yes","vol_eff","tau_minutes"])
pt["sigma_tau"]  = pt["vol_eff"] * np.sqrt(pt["tau_minutes"])
pt["offset_frac"] = pt["offset_pct"] / 100.0

# ── Align 6h mu to each trade (look-back only, ffill) ────────────────────────
pt["mu_6h"] = mu_6h.reindex(pt["logged_at"], method="ffill").values

# ── Compute z_drift and new p_model ──────────────────────────────────────────
# mu_6h is log-return per 1h bar; scale to tau_minutes
pt["mu_tau"]   = pt["mu_6h"] * (pt["tau_minutes"] / 60.0)
pt["z_drift"]  = pt["mu_tau"] / pt["sigma_tau"]

# Recover current z_adj from p_yes_model
pt["z_adj_cur"] = -norm.ppf(pt["p_yes_model"].clip(0.01, 0.99))
pt["z_adj_new"] = pt["z_adj_cur"] - pt["z_drift"]
pt["p_new"]     = np.clip(1 - norm.cdf(pt["z_adj_new"]), 0.01, 0.99)

pm     = pt["p_market"]
is_yes = pt["side"] == "yes"
is_no  = pt["side"] == "no"

pt["edge_cur"] = np.where(is_yes, pt["p_yes_model"] - pm, pm - pt["p_yes_model"])
pt["edge_new"] = np.where(is_yes, pt["p_new"] - pm,       pm - pt["p_new"])

# ── PnL per row ───────────────────────────────────────────────────────────────
def row_pnl(r):
    p = r["p_market"]
    return ((1-p)*FLAT if r["resolved_yes"]==1 else -p*FLAT) if r["side"]=="yes" \
        else (p*FLAT if r["resolved_yes"]==0 else -(1-p)*FLAT)

pt["pnl"] = pt.apply(row_pnl, axis=1)

mask_cur = pt["edge_cur"] >= EDGE_THRESH
mask_new = pt["edge_new"] >= EDGE_THRESH

SEP = "=" * 68

def report(mask, label):
    sub = pt[mask]
    yes = sub[is_yes & mask]; no = sub[is_no & mask]
    wr_y = yes["resolved_yes"].mean()      if len(yes) else float("nan")
    be_y = yes["p_market"].mean()          if len(yes) else float("nan")
    wr_n = (1-no["resolved_yes"]).mean()   if len(no)  else float("nan")
    be_n = (1-no["p_market"]).mean()       if len(no)  else float("nan")
    pnl  = sub["pnl"].sum()
    print(f"  {label}: n={len(sub)} ({len(yes)}Y/{len(no)}N)  "
          f"YES WR={wr_y:.1%}/BE={be_y:.1%}  "
          f"NO WR={wr_n:.1%}/BE={be_n:.1%}  PnL=${pnl:+.0f}")

print(SEP)
print(f"BTC resolved executed trades: {len(pt)}  "
      f"(YES:{is_yes.sum()}  NO:{is_no.sum()})")
print(SEP)
report(mask_cur, "Current model              ")
report(mask_new, "6h drift model             ")

# ── Dropped vs added ─────────────────────────────────────────────────────────
dropped = pt[mask_cur & ~mask_new]
added   = pt[mask_new & ~mask_cur]
print()
print(f"Trades dropped by drift model (edge falls below {EDGE_THRESH}): n={len(dropped)}")
if len(dropped):
    report(mask_cur & ~mask_new, "  Dropped")
    print(f"    → PnL saved by dropping: ${-dropped['pnl'].sum():+.0f}")

print(f"Trades added by drift model (edge rises above {EDGE_THRESH}): n={len(added)}")
if len(added):
    report(mask_new & ~mask_cur, "  Added  ")
    print(f"    → PnL from new trades:  ${added['pnl'].sum():+.0f}")

# ── Net impact ────────────────────────────────────────────────────────────────
pnl_cur = pt[mask_cur]["pnl"].sum()
pnl_new = pt[mask_new]["pnl"].sum()
print()
print(f"Net PnL change: ${pnl_new - pnl_cur:+.0f}")

# ── How much does drift shift p_yes? ─────────────────────────────────────────
print()
print(SEP)
print("Drift magnitude on YES trades")
print(SEP)
yes_df = pt[is_yes].copy()
yes_df["p_shift"] = yes_df["p_new"] - yes_df["p_yes_model"]
print(yes_df["p_shift"].describe().to_string())
print(f"  % of YES trades where drift increases p_yes: "
      f"{(yes_df['p_shift']>0).mean():.1%}  (mu>0 → BTC trending up → higher YES)")
print(f"  % of YES trades where drift decreases p_yes: "
      f"{(yes_df['p_shift']<0).mean():.1%}  (mu<0 → BTC trending down → lower YES)")

# ── Calibration comparison ───────────────────────────────────────────────────
print()
print(SEP)
print(f"YES calibration: current model vs 6h drift (buckets by lognormal z_score)")
print(SEP)
yes_df["p_base_lognorm"] = 1 - norm.cdf(yes_df["z_score"])

print(f"  {'Bucket':>15}  {'n':>5}  {'ActWR':>7}  {'CurModel':>9}  {'6hDrift':>8}  "
      f"{'ΔCur':>7}  {'ΔDrift':>7}")
print("  " + "-" * 72)
for lo, hi in [(0,.35),(.35,.45),(.45,.55),(.55,.65),(.65,.75),(.75,.85),(.85,1)]:
    m = (yes_df["p_base_lognorm"] >= lo) & (yes_df["p_base_lognorm"] < hi)
    sub = yes_df[m]
    if len(sub) < 5:
        continue
    act  = sub["resolved_yes"].mean()
    pcur = sub["p_yes_model"].mean()
    pnew = sub["p_new"].mean()
    print(f"  [{lo:.2f},{hi:.2f}): n={len(sub):>5}  {act:>7.1%}  "
          f"{pcur:>9.3f}  {pnew:>8.3f}  "
          f"{act-pcur:>+7.3f}  {act-pnew:>+7.3f}")

bs_cur = np.mean((yes_df["p_yes_model"] - yes_df["resolved_yes"])**2)
bs_new = np.mean((yes_df["p_new"]       - yes_df["resolved_yes"])**2)
print(f"\n  Brier: current={bs_cur:.5f}  6h-drift={bs_new:.5f}  "
      f"Δ={bs_new-bs_cur:+.5f}")

# ── NO calibration ────────────────────────────────────────────────────────────
print()
print(SEP)
print("NO calibration: current vs 6h drift")
print(SEP)
no_df = pt[is_no].copy()
no_df["pred_no_cur"]  = 1 - no_df["p_yes_model"]
no_df["pred_no_new"]  = 1 - no_df["p_new"]
no_df["actual_no_wr"] = 1 - no_df["resolved_yes"]
bs_no_cur = np.mean((no_df["pred_no_cur"] - no_df["actual_no_wr"])**2)
bs_no_new = np.mean((no_df["pred_no_new"] - no_df["actual_no_wr"])**2)
print(f"  Current: Brier={bs_no_cur:.5f}  "
      f"mean_pred={no_df['pred_no_cur'].mean():.3f}  "
      f"actual={no_df['actual_no_wr'].mean():.3f}")
print(f"  6h drift: Brier={bs_no_new:.5f}  "
      f"mean_pred={no_df['pred_no_new'].mean():.3f}  "
      f"actual={no_df['actual_no_wr'].mean():.3f}")

# ── Mu distribution at trade time ────────────────────────────────────────────
print()
print(SEP)
print("6h mu distribution at YES trade time")
print(SEP)
print(yes_df["mu_6h"].describe().to_string())
print(f"\n  z_drift at trade time (z-units):")
print(yes_df["z_drift"].describe().to_string())
