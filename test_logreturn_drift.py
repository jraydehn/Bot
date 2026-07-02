"""
test_logreturn_drift.py

Tests whether rolling log-return drift has genuine out-of-sample predictive power
for BTC price direction, and whether plugging it into z_adj improves model calibration
on live paper trades.

Math:
    log_returns_1h[t] = log(close[t] / close[t-1])
    mu[t]             = rolling_mean(log_returns, window=W) at time t (look-back only)
    z_adj             = (log(K/S) - mu * tau_min) / (sigma * sqrt(tau_min))
                      = z_score - mu * sqrt(tau_min) / sigma
                      = z_score - z_drift

Test 1 — Predictive power (BTC history, independent hourly observations):
    Bucket hours by mu quintile.
    Check whether next-hour direction aligns with mu sign.
    Report IC (rank correlation of mu with next-hour return), p-value, and hit rate.

Test 2 — Brier score on synthetic contracts (7 offsets × all hours):
    For each hour, compute z_adj with and without drift.
    Measure Brier score vs actual outcomes.

Test 3 — Calibration on live paper_trades.csv (executed BTC trades):
    Use tau_minutes and vol_eff from CSV.
    Compare actual YES WR vs predicted per p-bucket for each window.
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")

FLAT         = 10.0
OFFSETS      = [-0.004, -0.002, -0.001, 0.001, 0.002, 0.004, 0.006]
WINDOWS      = [6, 12, 24, 48, 96, 168]   # hours: 6h, 12h, 1d, 2d, 4d, 1wk
TEST_START   = pd.Timestamp("2025-01-01", tz="UTC")
SEP          = "=" * 68

# ── Load 1h BTC data ──────────────────────────────────────────────────────────
import glob
from pathlib import Path
BASE = Path(".")
f1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
print(f"Loading {f1h[-1]}")
ohlcv = pd.read_parquet(f1h[-1])
ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
ohlcv = ohlcv.sort_index()
close = ohlcv["close"].astype(float)
print(f"  {len(close):,} 1h bars  ({close.index[0].date()} → {close.index[-1].date()})")

log_ret = np.log(close / close.shift(1))

# Pre-compute rolling sigma (60-bar) and drift for each window
sigma_60 = log_ret.rolling(60).std()   # sigma per 1h bar (used as proxy for vol_eff)

print()
print(SEP)
print("TEST 1 — Predictive power: does mu predict next-hour direction?")
print(SEP)
print(f"  {'Window':>8}  {'IC':>7}  {'p-val':>8}  {'Hit↑ when mu>0':>17}  {'Hit↓ when mu<0':>17}  {'n':>7}")
print("  " + "-" * 75)

test_mask = close.index >= TEST_START
next_ret  = log_ret.shift(-1)   # the return we're trying to predict

for W in WINDOWS:
    mu = log_ret.rolling(W).mean()
    sub = pd.DataFrame({"mu": mu, "next": next_ret}).dropna()
    sub = sub[sub.index >= TEST_START]
    if len(sub) < 200:
        continue

    ic, pval = spearmanr(sub["mu"], sub["next"])

    pos = sub[sub["mu"] > 0]
    neg = sub[sub["mu"] < 0]
    hit_pos = (pos["next"] > 0).mean() if len(pos) > 10 else float("nan")
    hit_neg = (neg["next"] < 0).mean() if len(neg) > 10 else float("nan")

    sig = "★" if pval < 0.05 else "○" if pval < 0.10 else ""
    print(f"  {W:>5}h  {ic:>+7.4f}  {pval:>8.4f}  "
          f"{hit_pos:>14.1%} (n={len(pos):,})  {hit_neg:>14.1%} (n={len(neg):,})  {sig}")

print()
print("  Baseline (no drift): hit rate for next-hour up = "
      f"{(next_ret[next_ret.index >= TEST_START] > 0).mean():.1%}")


# ── Test 2: Brier score on synthetic contracts ────────────────────────────────
print()
print(SEP)
print("TEST 2 — Brier score: synthetic contracts (all test hours × 7 offsets)")
print(SEP)

# vol: use 60-bar rolling std of log_ret (per-1h); tau=45 min → sigma_tau = vol * sqrt(45)
TAU = 45.0
rows = []
for ts, s in close[close.index >= TEST_START].items():
    i = close.index.get_loc(ts)
    if i < 200:
        continue
    sig = float(sigma_60.iat[i])
    if not (sig > 0):
        continue
    sigma_tau = sig * math.sqrt(TAU)
    next_close_ts = close.index[i + 1] if i + 1 < len(close) else None
    if next_close_ts is None:
        continue
    actual_close = float(close.iat[i + 1])

    for off in OFFSETS:
        K = s * (1.0 + off)
        resolved = int(actual_close > K)
        z_score  = math.log(K / s) / sigma_tau
        p_base   = float(np.clip(1 - norm.cdf(z_score), 0.01, 0.99))
        rows.append({
            "ts": ts, "offset": off, "z_score": z_score, "sigma_tau": sigma_tau,
            "resolved": resolved, "p_base": p_base,
            "log_ret_prev": float(log_ret.iat[i]),
        })

df_syn = pd.DataFrame(rows)
print(f"  Synthetic contracts: {len(df_syn):,}  ({len(df_syn)//len(OFFSETS):,} hours × {len(OFFSETS)} offsets)")

bs_base = np.mean((df_syn["p_base"] - df_syn["resolved"]) ** 2)
print(f"\n  {'Model':>25}  {'Brier':>9}  {'vs base':>9}")
print("  " + "-" * 50)
print(f"  {'Zero-drift (baseline)':>25}  {bs_base:.6f}   (ref)")

best_w = None; best_bs = bs_base
for W in WINDOWS:
    mu_series = log_ret.rolling(W).mean()
    mu_at_ts  = mu_series.reindex(df_syn["ts"]).values  # mu at trade time

    # z_drift = mu * sqrt(tau) / sigma  (per-bar mu; tau in same bar units)
    # mu is log-return per 1h bar. tau=45min=0.75h. So drift over tau = mu * 0.75
    # z_drift = (mu * 0.75) / (sigma * sqrt(0.75 * 60)) — no, let's keep it dimensional:
    # sigma_tau = sigma_1h_bar * sqrt(45/60) — sigma per sqrt(1h), scaled to 45min
    # Actually sigma_tau is already sigma * sqrt(tau_min), where sigma is per-minute.
    # Here sigma_60 is std of 1h log-returns. Per-minute sigma = sigma_60 / sqrt(60).
    # sigma_tau = sigma_60 / sqrt(60) * sqrt(45) = sigma_60 * sqrt(45/60)
    # mu is per-1h. Per-minute mu = mu / 60. Drift over tau: mu/60 * 45 = mu * 0.75
    # z_drift = mu_tau / sigma_tau = (mu * 0.75) / (sigma_60 * sqrt(0.75))

    sigma_tau_arr = df_syn["sigma_tau"].values
    sigma_1h_arr  = sigma_tau_arr / math.sqrt(TAU / 60.0)   # back to per-sqrt-hour units
    mu_tau        = mu_at_ts * (TAU / 60.0)                 # scale mu from per-hour to per-tau
    z_drift       = mu_tau / (sigma_1h_arr * math.sqrt(TAU / 60.0))
    z_adj         = df_syn["z_score"].values - z_drift

    p_drift = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    bs_drift = np.mean((p_drift - df_syn["resolved"].values) ** 2)
    delta    = bs_drift - bs_base
    marker   = " ← best" if bs_drift < best_bs else ""
    if bs_drift < best_bs:
        best_bs = bs_drift; best_w = W
    print(f"  {'Drift W='+str(W)+'h':>25}  {bs_drift:.6f}  {delta:>+9.6f}{marker}")

if best_w:
    print(f"\n  Best window: {best_w}h  (Brier improvement: {best_bs - bs_base:+.6f})")
else:
    print(f"\n  No drift window improves on baseline.")


# ── Test 3: Calibration on live paper trades ──────────────────────────────────
print()
print(SEP)
print("TEST 3 — Calibration on live executed BTC paper trades")
print(SEP)

pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt = pt[
    pt["contract_ticker"].str.contains("BTC", na=False)
    & (pt["decision"] == "trade")
    & pt["resolved_yes"].notna()
].copy()
for col in ["z_score", "offset_pct", "p_market", "p_yes_model", "resolved_yes",
            "vol_eff", "tau_minutes"]:
    pt[col] = pd.to_numeric(pt[col], errors="coerce")
pt["logged_at"] = pd.to_datetime(pt["logged_at"], utc=True)
pt = pt.dropna(subset=["z_score", "offset_pct", "p_yes_model", "resolved_yes",
                        "vol_eff", "tau_minutes"])
pt["sigma_tau"] = pt["vol_eff"] * np.sqrt(pt["tau_minutes"])
pt["offset_frac"] = pt["offset_pct"] / 100.0
print(f"  Live trades: {len(pt)}  YES:{(pt['side']=='yes').sum()}  NO:{(pt['side']=='no').sum()}")

# Align mu to each trade's logged_at (look-back only)
# Use 1h close series; reindex to trade time
close_reindexed = close.reindex(pt["logged_at"], method="ffill")

yes_t = pt[pt["side"] == "yes"].copy()

print(f"\n  {'Model':>30}  {'Brier':>9}  {'vs base':>9}")
print("  " + "-" * 52)

bs_base_live = np.mean((yes_t["p_yes_model"] - yes_t["resolved_yes"]) ** 2)
p_base_live  = 1 - norm.cdf(yes_t["z_score"])
bs_lognorm   = np.mean((p_base_live - yes_t["resolved_yes"]) ** 2)
print(f"  {'Current model (p_yes_model)':>30}  {bs_base_live:.5f}   (ref)")
print(f"  {'Lognormal (z_score, no drift)':>30}  {bs_lognorm:.5f}  {bs_lognorm-bs_base_live:>+9.5f}")

best_w_live = None; best_bs_live = bs_lognorm
for W in WINDOWS:
    mu_series = log_ret.rolling(W).mean()
    # Match mu to each trade's timestamp (last available 1h bar at or before trade time)
    mu_at_trade = mu_series.reindex(yes_t["logged_at"], method="ffill").values

    sigma_tau_v = yes_t["sigma_tau"].values
    tau_v       = yes_t["tau_minutes"].values
    # vol_eff is per-minute; sigma_tau = vol_eff * sqrt(tau_min)
    # mu_series is log-return per 1h bar. Per-minute: mu / 60
    # drift over tau_min minutes: mu / 60 * tau_min
    mu_tau = mu_at_trade * (tau_v / 60.0)
    z_drift = mu_tau / sigma_tau_v   # drift in z-units
    z_adj   = yes_t["z_score"].values - z_drift
    p_drift = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    bs_d    = np.mean((p_drift - yes_t["resolved_yes"].values) ** 2)
    delta   = bs_d - bs_lognorm
    marker  = " ← best" if bs_d < best_bs_live else ""
    if bs_d < best_bs_live:
        best_bs_live = bs_d; best_w_live = W
    print(f"  {'Drift W='+str(W)+'h':>30}  {bs_d:.5f}  {delta:>+9.5f}{marker}")

if best_w_live:
    best_mu = log_ret.rolling(best_w_live).mean()
    best_mu_at_trade = best_mu.reindex(yes_t["logged_at"], method="ffill").values
    tau_v = yes_t["tau_minutes"].values
    mu_tau = best_mu_at_trade * (tau_v / 60.0)
    z_adj  = yes_t["z_score"].values - mu_tau / yes_t["sigma_tau"].values
    p_best = np.clip(1 - norm.cdf(z_adj), 0.01, 0.99)
    yes_t = yes_t.copy()
    yes_t["p_best"] = p_best
    print(f"\n  Best window: {best_w_live}h  (Brier improvement over lognormal: {best_bs_live-bs_lognorm:+.5f})")
    print()
    print(f"  Calibration detail for W={best_w_live}h vs lognormal (YES trades):")
    print(f"  {'Bucket':>15}  {'n':>5}  {'ActualWR':>9}  {'PredBase':>9}  {'PredDrift':>10}  {'ΔBase':>7}  {'ΔDrift':>7}")
    print("  " + "-" * 72)
    for lo, hi in [(0,.35),(.35,.45),(.45,.55),(.55,.65),(.65,.75),(.75,.85),(.85,1)]:
        m = (p_base_live >= lo) & (p_base_live < hi)
        sub = yes_t[m]
        if len(sub) < 5:
            continue
        act = sub["resolved_yes"].mean()
        pb  = p_base_live[m].mean()
        pd_ = sub["p_best"].mean()
        print(f"  [{lo:.2f},{hi:.2f}): n={len(sub):>5}  {act:>8.1%}  {pb:>9.3f}  "
              f"{pd_:>10.3f}  {act-pb:>+7.3f}  {act-pd_:>+7.3f}")
else:
    print(f"\n  No drift window improves on lognormal baseline.")
