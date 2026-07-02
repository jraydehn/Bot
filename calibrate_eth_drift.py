"""
calibrate_eth_drift.py — Sweep k_drift_yes and k_drift_no for ETH.

Finds the optimal per-branch drift multipliers for the score_to_p_model YES
branch and score_to_p_no_model NO branch to replace ETH's direct_p_model.

Method
------
For each 1h bar in the simulation window, for each strike offset in a grid:
  - p_market   = risk-neutral lognormal (no drift) — baseline proxy for what
                 the Kalshi market would price the contract at
  - outcome    = 1 if next 1h close > (1+offset)*spot (YES resolves)
  - p_yes(k)   = score_to_p_model formula with k_drift = k
  - p_no(k)    = score_to_p_no_model formula with k_drift_no = k
  Bet YES if p_yes - p_market > MIN_EDGE; NO if p_no - (1-p_market) > MIN_EDGE
  P&L at flat $5 notional (comparable to live bet sizing).

Train: 2025-07-01 → 2026-01-01   (in-sample, for intuition only)
Val:   2026-01-01 → 2026-03-16   (used to select optimal k)
Test:  2026-03-16 → 2026-04-06   (held-out final validation)
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from composite_scorer import compute_scores, lookup_p_up

BASE = Path(__file__).parent
DATA = BASE / "data"
SYM  = "ETHUSDT"

TRAIN_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_START   = pd.Timestamp("2026-01-01", tz="UTC")
TEST_START  = pd.Timestamp("2026-03-16", tz="UTC")
END         = pd.Timestamp("2026-04-07", tz="UTC")   # last date with ETH 15m data

TAU         = 60.0            # minutes — simulate 1h-forward contracts
MIN_EDGE    = 0.02            # minimum net edge to place a bet
BET_SIZE    = 5.0             # flat $ notional per trade
ASSET       = "ETH"

# Strike offset grid (fraction of spot): negative = below spot (ITM YES / OTM NO)
OFFSETS = [-0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020]

# Drift grids to sweep
K_YES_GRID = [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
K_NO_GRID  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]

# ── load OHLCV ───────────────────────────────────────────────────────────────
print("Loading ETH data …")

def latest(pattern):
    files = sorted(DATA.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]

df_1h  = pd.read_parquet(latest(f"binanceus_{SYM}_1h_2024-01-01_*.parquet"))
df_4h  = pd.read_parquet(latest(f"binanceus_{SYM}_4h_2024-01-01_*.parquet"))
df_15m = pd.read_parquet(latest(f"binanceus_{SYM}_15m_2024-01-01_*.parquet"))
df_1m  = pd.read_parquet(latest(f"binanceus_{SYM}_1m_2024-01-01_*.parquet"))

for df in (df_1h, df_4h, df_15m, df_1m):
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

# Resample 1m → 15m to extend beyond the last 15m parquet date
df_15m_ext = df_1m.resample("15min").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()
df_15m_ext.index = df_15m_ext.index.tz_localize("UTC") if df_15m_ext.index.tz is None else df_15m_ext.index
df_15m_combined = pd.concat([df_15m, df_15m_ext]).sort_index()
df_15m_combined = df_15m_combined[~df_15m_combined.index.duplicated(keep="first")]

# ── compute composite scores ──────────────────────────────────────────────────
print("Computing composite scores …")
# Use data from 2025-01-01 for warmup (indicators need ~200 bars)
warmup = pd.Timestamp("2025-01-01", tz="UTC")
h1  = df_1h[df_1h.index >= warmup]
h4  = df_4h[df_4h.index >= warmup]
m15 = df_15m_combined[df_15m_combined.index >= warmup]
m1  = df_1m[df_1m.index >= warmup]

trend_s, rev_s = compute_scores(
    h1["close"], h1["high"], h1["low"], h1["volume"],
    h4["close"], h4["high"], h4["low"], h4["volume"],
    m15["close"], m15["high"], m15["low"],
    m1["close"], m1["volume"],
    ts_1h=h1.index,
)

# ── compute vol_pm per 1h bar ─────────────────────────────────────────────────
# vol_pm = 60-bar rolling std of 1m log returns (same as live system)
log_ret_1m = np.log(m1["close"] / m1["close"].shift(1)).dropna()
vol_1m_rolling = log_ret_1m.rolling(60).std()
# Resample to 1h: take the last vol_pm value in each hour
vol_pm_1h = vol_1m_rolling.resample("1h").last().reindex(h1.index, method="ffill").fillna(0.0005)

# ── build simulation dataframe ────────────────────────────────────────────────
# Restrict to bars where scores are valid and we're within our sim window
sim = pd.DataFrame({
    "trend":  trend_s,
    "rev":    rev_s,
    "close":  h1["close"],
    "vol_pm": vol_pm_1h,
}, index=h1.index).dropna()

# p_up per bar (from composite calibration table)
sim["p_up"] = sim.apply(
    lambda r: lookup_p_up(int(r["trend"]), int(r["rev"]), asset=ASSET), axis=1
)

# Next-bar close (1h forward = outcome for YES)
sim["close_next"] = sim["close"].shift(-1)
sim = sim.dropna(subset=["close_next"])

# sigma_tau for 1h contract
sim["sigma_tau"] = sim["vol_pm"] * math.sqrt(TAU)
sim = sim[sim["sigma_tau"] > 0]

# Window masks
mask_train = (sim.index >= TRAIN_START) & (sim.index < VAL_START)
mask_val   = (sim.index >= VAL_START)   & (sim.index < TEST_START)
mask_test  = (sim.index >= TEST_START)  & (sim.index < END)

print(f"  Bars — train={mask_train.sum()}  val={mask_val.sum()}  test={mask_test.sum()}")

# ── simulation core ───────────────────────────────────────────────────────────
def simulate(mask, k_yes=None, k_no=None):
    """
    Run YES and/or NO simulation for bars in `mask`.
    Returns (yes_pnl, yes_n, yes_wr, no_pnl, no_n, no_wr).
    Pass k_yes=None to skip YES simulation, k_no=None to skip NO.
    """
    sub = sim[mask]
    yes_pnl_total = 0.0; yes_n = 0; yes_wins = 0
    no_pnl_total  = 0.0; no_n  = 0; no_wins  = 0

    for _, r in sub.iterrows():
        spot      = float(r["close"])
        close_nxt = float(r["close_next"])
        sigma_tau = float(r["sigma_tau"])
        p_up      = float(r["p_up"])
        z_up      = norm.ppf(p_up)

        for offset in OFFSETS:
            strike = spot * (1.0 + offset)
            z_raw  = math.log(strike / spot) / sigma_tau      # lognormal z (no drift)
            p_mkt  = float(1.0 - norm.cdf(z_raw))             # risk-neutral YES probability
            outcome = int(close_nxt > strike)                  # 1 = YES resolves

            # YES branch
            if k_yes is not None:
                z_yes  = z_raw - z_up * k_yes
                p_yes  = float(np.clip(1.0 - norm.cdf(z_yes), 0.01, 0.99))
                edge_y = p_yes - p_mkt
                if edge_y > MIN_EDGE and 0.03 < p_mkt < 0.97:
                    yes_n += 1
                    # P&L: cost = p_mkt * BET_SIZE; win = (1-p_mkt)*BET_SIZE
                    yes_pnl_total += (1.0 - p_mkt) * BET_SIZE if outcome else -p_mkt * BET_SIZE
                    yes_wins += outcome

            # NO branch
            if k_no is not None:
                z_no   = z_raw - z_up * k_no
                p_no   = float(np.clip(norm.cdf(z_no), 0.01, 0.99))
                edge_n = p_no - (1.0 - p_mkt)
                if edge_n > MIN_EDGE and 0.03 < p_mkt < 0.97:
                    no_n += 1
                    # P&L: cost = (1-p_mkt)*BET_SIZE; win = p_mkt*BET_SIZE
                    no_pnl_total += p_mkt * BET_SIZE if (1 - outcome) else -(1.0 - p_mkt) * BET_SIZE
                    no_wins += (1 - outcome)

    yes_wr = yes_wins / yes_n if yes_n else 0.0
    no_wr  = no_wins  / no_n  if no_n  else 0.0
    return yes_pnl_total, yes_n, yes_wr, no_pnl_total, no_n, no_wr

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — k_drift_yes sweep (YES bets only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("YES DRIFT SWEEP — k_drift_yes  (val window selects optimal)")
print(f"{'k_yes':>6}  {'train_n':>7}  {'train_$':>8}  {'val_n':>6}  {'val_$':>8}  {'val_WR':>7}  {'test_n':>6}  {'test_$':>8}")
print("=" * 72)

best_yes_val = -1e9
best_k_yes   = K_YES_GRID[0]

for k in K_YES_GRID:
    tr_pnl, tr_n, _, _, _, _ = simulate(mask_train, k_yes=k)
    va_pnl, va_n, va_wr, _, _, _ = simulate(mask_val, k_yes=k)
    te_pnl, te_n, _, _, _, _ = simulate(mask_test, k_yes=k)
    print(f"  {k:4.2f}  {tr_n:7d}  {tr_pnl:+8.1f}  {va_n:6d}  {va_pnl:+8.1f}  {va_wr:6.1%}  {te_n:6d}  {te_pnl:+8.1f}")
    if va_pnl > best_yes_val:
        best_yes_val = va_pnl
        best_k_yes   = k

print(f"\n  ► Optimal k_drift_yes = {best_k_yes}  (val P&L = {best_yes_val:+.1f})")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — k_drift_no sweep (NO bets only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("NO DRIFT SWEEP — k_drift_no  (val window selects optimal)")
print(f"{'k_no':>6}  {'train_n':>7}  {'train_$':>8}  {'val_n':>6}  {'val_$':>8}  {'val_WR':>7}  {'test_n':>6}  {'test_$':>8}")
print("=" * 72)

best_no_val = -1e9
best_k_no   = K_NO_GRID[0]

for k in K_NO_GRID:
    tr_pnl, _, _, tr_pnl_n, tr_n_no, _ = simulate(mask_train, k_no=k)
    va_pnl, _, _, va_pnl_n, va_n_no, va_wr_no = simulate(mask_val, k_no=k)
    te_pnl, _, _, te_pnl_n, te_n_no, _ = simulate(mask_test, k_no=k)
    print(f"  {k:4.2f}  {tr_n_no:7d}  {tr_pnl_n:+8.1f}  {va_n_no:6d}  {va_pnl_n:+8.1f}  {va_wr_no:6.1%}  {te_n_no:6d}  {te_pnl_n:+8.1f}")
    if va_pnl_n > best_no_val:
        best_no_val = va_pnl_n
        best_k_no   = k

print(f"\n  ► Optimal k_drift_no = {best_k_no}  (val P&L = {best_no_val:+.1f})")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Combined validation with optimal params
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"COMBINED RESULT — k_yes={best_k_yes}  k_no={best_k_no}")
print("=" * 72)

for label, mask in [("train", mask_train), ("val", mask_val), ("test", mask_test)]:
    y_pnl, y_n, y_wr, n_pnl, n_no, n_wr = simulate(mask, k_yes=best_k_yes, k_no=best_k_no)
    total = y_pnl + n_pnl
    print(f"  {label:5s}:  YES n={y_n:5d} P&L={y_pnl:+8.1f} WR={y_wr:.1%}  |  "
          f"NO n={n_no:5d} P&L={n_pnl:+8.1f} WR={n_wr:.1%}  |  total={total:+8.1f}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — 2D grid search (k_yes x k_no) on val set for interaction check
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("2D GRID — val P&L for (k_yes, k_no) combinations  [top 10]")
print("=" * 72)

k_yes_fine = [k for k in K_YES_GRID if abs(k - best_k_yes) <= 0.4]
k_no_fine  = [k for k in K_NO_GRID  if abs(k - best_k_no)  <= 0.2]

results_2d = []
for ky in k_yes_fine:
    for kn in k_no_fine:
        y_pnl, y_n, _, n_pnl, n_no, _ = simulate(mask_val, k_yes=ky, k_no=kn)
        results_2d.append((ky, kn, y_pnl + n_pnl, y_n, n_no))

results_2d.sort(key=lambda x: -x[2])
print(f"  {'k_yes':>6}  {'k_no':>6}  {'val_total':>10}  {'yes_n':>6}  {'no_n':>6}")
for ky, kn, tot, yn, nn in results_2d[:10]:
    print(f"  {ky:6.2f}  {kn:6.2f}  {tot:+10.1f}  {yn:6d}  {nn:6d}")

print(f"\nDone. Recommended: k_drift_yes={best_k_yes}  k_drift_no={best_k_no}")
print("Next: update DRIFT_MULTIPLIER and DRIFT_MULTIPLIER_NO in composite_scorer.py,")
print("then route ETH through the dual YES/NO branch in paper_trade_runner.py.")
