"""
S21 -- backtest the p_up_v2 parse-bug fix before deploying it live.

Reconstructs what the K_YES=0.50/K_NO=0.30 non-coherent model (the 06-30
reform, never actually live-exercised because fetch_p_up_v2 broke on 06-26)
would have decided, using the REAL logged hourly p_up_v2 history (joined
zero-lookahead) against every candidate in btc_scan_archive_15m.csv --
not just taken trades, so this replicates the live selection process.

Formula mirrors compute_p_yes_pup_v2_15m / compute_p_no_pup_v2_15m exactly
(log-normal + Φ⁻¹(p_up_v2)×K×√(τ/60) drift), using a fixed vol_eff proxy
from the archive's own implied vol (back out from p_market) blended with
a realized-vol estimate from the 1m parquet, matching REALIZED_VOL_WEIGHT=0.35.
Edge threshold 0.04 (matches the live scan's qualifying bar).
"""
import warnings
import pathlib
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
K_YES, K_NO = 0.50, 0.30
EDGE_THRESH = 0.04

# ---- corrected p_up_v2 history (the actual fix) ----
hourly = pd.read_csv("results/paper_trades.csv", usecols=["logged_at", "p_up_v2"], low_memory=False)
hourly["p_up_v2"] = pd.to_numeric(hourly["p_up_v2"], errors="coerce")
hourly["logged_at"] = pd.to_datetime(hourly["logged_at"], utc=True, errors="coerce", format="mixed")
hourly = hourly.dropna(subset=["p_up_v2", "logged_at"]).sort_values("logged_at")
hourly = hourly.drop_duplicates(subset="logged_at", keep="last")
print(f"corrected p_up_v2 series: {len(hourly)} rows  {hourly['logged_at'].min()} -> {hourly['logged_at'].max()}")

# ---- candidate population: every scanned BTC 15m contract, both eras ----
arc = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False,
                  usecols=["logged_at", "contract_ticker", "spot", "strike", "p_market",
                           "tau_minutes", "resolved_yes"])
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes", "tau_minutes", "spot", "strike"])
arc = arc[arc["logged_at"] >= "2026-06-01"].sort_values("logged_at")
print(f"candidate rows: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")

# 2-hour staleness rule matches fetch_p_up_v2's own guard
arc = pd.merge_asof(arc, hourly.rename(columns={"logged_at": "pu_ts"}),
                    left_on="logged_at", right_on="pu_ts", direction="backward",
                    tolerance=pd.Timedelta("120min"))
arc = arc.dropna(subset=["p_up_v2"])
print(f"candidate rows with valid (fresh) p_up_v2: {len(arc)}")

# realized vol proxy (5m bars, 20-bar rolling ann.) joined causally
p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
c5 = px["close"].resample("5min").last().dropna()
rv_ann = (c5.pct_change().rolling(20).std() * math.sqrt(288 * 365)).rename("rv_ann")
rv_ann.index = rv_ann.index + pd.Timedelta("5min")
rv_ann.index.name = "rv_ts"
arc = pd.merge_asof(arc.sort_values("logged_at"), rv_ann.reset_index().sort_values("rv_ts"),
                    left_on="logged_at", right_on="rv_ts", direction="backward")
arc = arc.dropna(subset=["rv_ann"])

def implied_vol_from_pm(pm, spot, strike, tau_min):
    z = norm.ppf(np.clip(pm, 0.02, 0.98))
    denom = math.log(strike / spot) if strike != spot else 1e-9
    # Solve sigma_tau s.t. norm.cdf(-log(strike/spot)/sigma_tau) = pm  =>  sigma_tau = -log(strike/spot)/z
    if abs(z) < 1e-6 or denom == 0:
        return np.nan
    sigma_tau = -denom / z
    if sigma_tau <= 0:
        return np.nan
    return sigma_tau / math.sqrt(tau_min)

arc["vol_imp"] = [implied_vol_from_pm(pm, s, k, t) for pm, s, k, t in
                  zip(arc["p_market"], arc["spot"], arc["strike"], arc["tau_minutes"])]
arc["vol_real"] = arc["rv_ann"] / math.sqrt(525600)
arc["vol_eff"] = 0.35 * arc["vol_real"] + 0.65 * arc["vol_imp"].fillna(arc["vol_real"])
arc = arc[(arc["vol_eff"] > 0) & arc["vol_eff"].notna()]

arc["sigma_tau"] = (arc["vol_eff"] * np.sqrt(arc["tau_minutes"])).clip(lower=1e-6)
arc["z_strike"] = np.log(arc["strike"] / arc["spot"]) / arc["sigma_tau"]
arc["tau_scale"] = np.sqrt(arc["tau_minutes"].clip(upper=60) / 60.0)
pu_clip = arc["p_up_v2"].clip(0.02, 0.98)
z_drift_yes = norm.ppf(pu_clip) * K_YES * arc["tau_scale"]
z_drift_no = norm.ppf(pu_clip) * K_NO * arc["tau_scale"]
arc["p_model_yes"] = np.clip(norm.cdf(z_drift_yes - arc["z_strike"]), 0.03, 0.97)
arc["p_model_no"] = np.clip(1 - norm.cdf(z_drift_no - arc["z_strike"]), 0.03, 0.97)
arc["edge_yes"] = arc["p_model_yes"] - arc["p_market"]
arc["edge_no"] = arc["p_model_no"] - (1 - arc["p_market"])
arc["best_side"] = np.where(arc["edge_yes"] >= arc["edge_no"], "yes", "no")
arc["best_edge"] = np.where(arc["edge_yes"] >= arc["edge_no"], arc["edge_yes"], arc["edge_no"])

qual = arc[arc["best_edge"] >= EDGE_THRESH].copy()
print(f"\nqualifying candidates (edge>={EDGE_THRESH}): {len(qual)} of {len(arc)}")
print("side mix:", qual["best_side"].value_counts().to_dict())

# one bet per ticker (best edge among its qualifying scan cycles) -- matches
# the runner's "first qualifying cycle wins" cadence closely enough for a
# comparative backtest (exact cooldown/session-traded logic not replicated).
per_ticker = qual.sort_values("best_edge", ascending=False).drop_duplicates("contract_ticker")
per_ticker["win"] = np.where(per_ticker["best_side"] == "yes", per_ticker["resolved_yes"],
                             1 - per_ticker["resolved_yes"])
per_ticker["cost"] = np.where(per_ticker["best_side"] == "yes", per_ticker["p_market"],
                              1 - per_ticker["p_market"])
per_ticker["day"] = per_ticker["logged_at"].dt.date
print(f"\nsimulated trades (one per ticker): {len(per_ticker)}")
print(per_ticker.groupby("best_side").agg(n=("win", "size"), wr=("win", "mean"),
     be=("cost", "mean")).round(3).to_string())

print("\nby day (K_YES/K_NO model, corrected p_up_v2):")
d = per_ticker.groupby("day").agg(n=("win", "size"), n_yes=("best_side", lambda s: (s == "yes").sum()),
     n_no=("best_side", lambda s: (s == "no").sum()), wr=("win", "mean")).round(3)
print(d.to_string())

# compare same-day actual book (real, taken, would_pnl) for the drawdown window
real = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
real["decision_time"] = pd.to_datetime(real["decision_time"], utc=True, errors="coerce", format="mixed")
rt = real[real["side"].isin(["yes", "no"]) & (pd.to_numeric(real["bet_amount"], errors="coerce") > 0)]
rt = rt.dropna(subset=["would_pnl"])
rt["day"] = rt["decision_time"].dt.date
real_daily = rt.groupby("day").agg(real_n=("would_pnl", "size"), real_pnl=("would_pnl", "sum"))
print("\nJuly comparison: simulated K_YES/K_NO side-mix+WR vs actual book PnL:")
jul = d[d.index >= pd.Timestamp("2026-07-01").date()].join(real_daily, how="left")
print(jul.to_string())

per_ticker.to_csv(f"{OUT}/pupv2_fix_backtest.csv", index=False)
print("DONE_S21")
