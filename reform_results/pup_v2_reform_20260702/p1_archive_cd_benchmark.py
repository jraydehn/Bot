#!/usr/bin/env python3
"""PHASE 1 (C/D) — archive-era benchmark: flow/positioning features and the
Kalshi market-implied benchmark + residual test.

Universe: one ATM contract per hourly expiry from results/btc_scan_archive.csv,
rows with tau in [50,70] min (snapshot just after the hour boundary), strike
nearest spot (|strike-spot|/spot <= 0.05%). Label = resolved_yes (did price end
the hour above the ~ATM strike). market_p_up = p_market (Kalshi YES price).

Group C (flow/positioning), all values logged/backfilled at or before decision:
  from archive row itself: funding_bias, ls_long_pct, oi_chg_pct, liq_score, liq_bias
  from paper_trades (hourly last-obs-carried): avg_funding_rate
  from paper_trades_btc15m (close_time <= decision): cvd_4h, cg_futures_delta_4h,
    cg_futures_ratio_4h, cg_futures_cvd_12h  [CAVEAT: partially backfilled from
    CoinGlass history by backfill_cg_futures_cvd.py — treated as point-in-time]

Group D tests:
  (i)  pm's own accuracy (AUC/Brier/calibration) vs every model group
  (ii) residual y - pm: rank-IC of each A/B/C feature and of the A-model WF score
  (iii) WF combos on archive era + unit-$ backtest vs pm

Outputs: archive_hourly_frame.parquet, d_benchmark.txt (log), residual_ic.csv,
archive_wf_summary.csv, archive_weekly.csv
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
SCRATCH = Path("/private/tmp/claude-501/-Users-justindehn-Documents-ClaudeCode/600a1ce9-48de-420d-94bc-84e3bd4f9871/scratchpad")
OUT = PROJ / "reform_results" / "pup_v2_reform_20260702"
RES = PROJ / "results"

# ── 1. ATM hourly frame from scan archive ─────────────────────────────────
usecols = ["logged_at", "close_ts", "spot", "strike", "p_market", "tau_minutes",
           "resolved_yes", "p_up_v2", "funding_bias", "ls_long_pct", "oi_chg_pct",
           "liq_score", "liq_bias"]
arch = pd.read_csv(RES / "btc_scan_archive.csv", usecols=usecols, low_memory=False)
for c in ["spot", "strike", "p_market", "tau_minutes", "p_up_v2", "funding_bias",
          "ls_long_pct", "oi_chg_pct", "liq_score", "liq_bias"]:
    arch[c] = pd.to_numeric(arch[c], errors="coerce")
arch["logged_at"] = pd.to_datetime(arch["logged_at"], format="mixed", utc=True)
arch["close_ts"] = pd.to_datetime(arch["close_ts"], format="mixed", utc=True)
arch["y"] = arch["resolved_yes"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
arch = arch[arch.y.notna() & arch.p_market.notna()]
sub = arch[(arch.tau_minutes >= 50) & (arch.tau_minutes <= 70)].copy()
sub["od"] = (sub.strike - sub.spot).abs() / sub.spot
sub = sub[sub.od <= 0.0005]
sub = sub.sort_values(["close_ts", "od", "logged_at"])
atm = sub.groupby("close_ts", as_index=False).first()   # nearest strike per expiry
atm["dec_hour"] = atm["close_ts"] - pd.Timedelta(hours=1)  # decision ~ hour open
print(f"ATM hours: {len(atm)}  {atm.close_ts.min()} -> {atm.close_ts.max()}  "
      f"median |strike-spot|/spot = {atm.od.median():.5%}")
print(f"base up-rate = {atm.y.mean():.3f}")

# ── 2. group C features ────────────────────────────────────────────────────
def hour_lut(df, tcol, cols):
    """last observation logged strictly before each decision time, per hour"""
    d = df[[tcol] + cols].dropna(subset=[tcol]).sort_values(tcol)
    return d, cols

# paper_trades (hourly runner): avg_funding_rate + redundancy check cols
pt_frames = []
for f in ["paper_trades_pre_regime_pup_20260616.csv", "paper_trades.csv"]:
    d = pd.read_csv(RES / f, low_memory=False,
                    usecols=lambda c: c in ("logged_at", "avg_funding_rate"))
    pt_frames.append(d)
pt = pd.concat(pt_frames, ignore_index=True)
pt["logged_at"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True)
pt["avg_funding_rate"] = pd.to_numeric(pt["avg_funding_rate"], errors="coerce")
pt = pt.dropna().sort_values("logged_at")

m15 = pd.read_csv(RES / "paper_trades_btc15m.csv", low_memory=False,
                  usecols=["close_time", "cvd_4h", "cg_futures_delta_4h",
                           "cg_futures_ratio_4h", "cg_futures_cvd_12h"])
m15["close_time"] = pd.to_datetime(m15["close_time"], format="mixed", utc=True)
CG_COLS = ["cvd_4h", "cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h"]
for c in CG_COLS:
    m15[c] = pd.to_numeric(m15[c], errors="coerce")
m15 = m15.dropna(subset=["close_time"]).sort_values("close_time").drop_duplicates("close_time", keep="last")

atm = atm.sort_values("logged_at")
atm = pd.merge_asof(atm, pt.rename(columns={"logged_at": "t_pt"}),
                    left_on="logged_at", right_on="t_pt", direction="backward",
                    tolerance=pd.Timedelta("2h"))
atm = pd.merge_asof(atm, m15.rename(columns={"close_time": "t_15"}),
                    left_on="logged_at", right_on="t_15", direction="backward",
                    tolerance=pd.Timedelta("45min"))
C_FEATURES = ["funding_bias", "avg_funding_rate", "ls_long_pct", "oi_chg_pct",
              "liq_score", "liq_bias"] + CG_COLS
print("\nC feature fill rates:")
print(atm[C_FEATURES].notna().mean().round(3).to_string())

# ── 3. join A/B/E WF predictions + features from long-history run ─────────
wf = pd.read_parquet(OUT / "wf_preds_groups.parquet")
bb = pd.read_parquet(SCRATCH / "pup_v2_dataset_fixed_20260702.parquet")
# decision at close_ts-1h == backbone bar that OPENS at close_ts-2h (its close
# is the spot at ~decision time; its label bar is the contract hour)
atm["bb_ts"] = atm["close_ts"] - pd.Timedelta(hours=2)
atm = atm.merge(wf.add_prefix("wf_"), left_on="bb_ts", right_index=True, how="left")
A_FEATURES = ["stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
              "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
              "composite_trend", "composite_rev", "composite_p_up",
              "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score", "rvol_1h"]
atm = atm.merge(bb[A_FEATURES], left_on="bb_ts", right_index=True, how="left",
                suffixes=("", "_bb"))
print(f"\nWF pred coverage on ATM hours: {atm['wf_p_A'].notna().mean():.1%}")

atm.to_parquet(OUT / "archive_hourly_frame.parquet")

ev = atm.dropna(subset=["wf_p_A"]).copy()
ev["week"] = ev.close_ts.dt.to_period("W-WED").astype(str)

# ── 4. D(i): market benchmark ──────────────────────────────────────────────
print("\n=== D(i) MARKET BENCHMARK (same rows, n=%d, weeks=%d) ===" % (len(ev), ev.week.nunique()))
def bench(name, p, y):
    p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    print(f"  {name:<22} AUC={roc_auc_score(y, p):.4f}  Brier={brier_score_loss(y, p):.4f}  "
          f"acc@0.5={((p > .5).astype(int) == y).mean():.3f}")
y_ev = ev.y.values.astype(int)
bench("market pm (ATM)", ev.p_market.values, y_ev)
bench("A-model (honest WF)", ev.wf_p_A.values, y_ev)
if ev["wf_p_A+B"].notna().any():
    bench("A+B (honest WF)", ev["wf_p_A+B"].values, y_ev)
    bench("A+B+E (honest WF)", ev["wf_p_A+B+E"].values, y_ev)
if ev.p_up_v2.notna().mean() > 0.5:
    bench("p_up_v2 LIVE (leaky)", ev.p_up_v2.values, y_ev)
# pm calibration deciles
ev["pm_bin"] = pd.cut(ev.p_market, np.arange(0, 1.01, 0.1))
cal = ev.groupby("pm_bin").agg(n=("y", "size"), pm=("p_market", "mean"), up=("y", "mean"))
print("\npm calibration (ATM contracts):")
print(cal.round(3).to_string())

# ── 5. D(ii): residual rank-IC ─────────────────────────────────────────────
ev["resid"] = ev.y - ev.p_market
rows = []
cand = [("A_model_wf", "wf_p_A"), ("AB_model_wf", "wf_p_A+B"), ("ABE_model_wf", "wf_p_A+B+E")]
cand += [(f, f) for f in A_FEATURES + C_FEATURES]
B_FEATURES = ["eth_ret_1h", "sol_ret_1h", "eth_ret_4h", "sol_ret_4h",
              "spread_eth_1h", "spread_sol_1h", "spread_eth_24h", "spread_sol_24h"]
# B features live in wf preds parquet? no — recompute from backbone merge if present
for f in B_FEATURES:
    if f in bb.columns:
        cand.append((f, f))
for name, col in cand:
    if col not in ev.columns:
        continue
    g = ev[[col, "resid", "week"]].dropna()
    if len(g) < 60 or g[col].nunique() < 3:
        continue
    ic = spearmanr(g[col], g.resid)
    wk_ics = g.groupby("week").apply(
        lambda x: spearmanr(x[col], x.resid).statistic if len(x) > 15 and x[col].nunique() > 2 else np.nan)
    wk_ics = wk_ics.dropna()
    rows.append({"feature": name, "n": len(g), "ic_vs_resid": ic.statistic, "p": ic.pvalue,
                 "n_weeks": len(wk_ics), "wk_ic_mean": wk_ics.mean(),
                 "pct_weeks_pos": (wk_ics > 0).mean() if len(wk_ics) else np.nan})
ric = pd.DataFrame(rows).sort_values("ic_vs_resid", key=abs, ascending=False)
ric.to_csv(OUT / "residual_ic.csv", index=False)
print("\n=== D(ii) RANK-IC vs RESIDUAL (y - pm) ===")
print(ric.round(4).to_string(index=False))

# ── 6. D(iii): archive-era WF combos + unit-$ vs pm ───────────────────────
print("\n=== D(iii) ARCHIVE-ERA WF (weekly refits, expanding, SMALL-N: flag) ===")
ev = ev.sort_values("close_ts").reset_index(drop=True)
ts = ev.close_ts
CONFS = {
    "pm_only":  ["p_market"],
    "C_only":   C_FEATURES,
    "A_score":  ["wf_p_A"],
    "A+D":      ["wf_p_A", "p_market"],
    "C+D":      C_FEATURES + ["p_market"],
    "A+C":      ["wf_p_A"] + C_FEATURES,
    "A+C+D":    ["wf_p_A", "p_market"] + C_FEATURES,
}
week_starts = pd.date_range(ts.min().normalize() + pd.Timedelta(days=14), ts.max(), freq="7D")
res_rows, wk_rows = [], []
y_all = ev.y.values.astype(int)
for name, feats in CONFS.items():
    X = ev[feats].values.astype(float)
    p = np.full(len(ev), np.nan)
    for ws in week_starts:
        te = np.where((ts >= ws) & (ts < ws + pd.Timedelta(days=7)))[0]
        tr = np.where(ts < ws - pd.Timedelta(hours=1))[0]
        if len(te) == 0 or len(tr) < 150:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                               num_leaves=7, min_child_samples=40, reg_lambda=5.0,
                               subsample=0.8, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y_all[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(p)
    if mask.sum() < 100:
        continue
    auc = roc_auc_score(y_all[mask], p[mask])
    # unit-$ vs pm
    pm = ev.p_market.values
    edge = p - pm
    pnl = np.where(edge > 0.02, y_all - pm, np.where(edge < -0.02, pm - y_all, np.nan))
    pnl_m = pnl[mask & ~np.isnan(pnl)]
    res_rows.append({"config": name, "n_oos": int(mask.sum()), "auc_oos": auc,
                     "n_bets": len(pnl_m), "unit_pnl_mean": np.nanmean(pnl_m) if len(pnl_m) else np.nan,
                     "unit_pnl_total": np.nansum(pnl_m) if len(pnl_m) else np.nan})
    dd = ev.loc[mask, ["week", "y"]].copy(); dd["p"] = p[mask]
    for wk, g in dd.groupby("week"):
        if g.y.nunique() > 1 and len(g) > 25:
            wk_rows.append({"config": name, "week": wk, "n": len(g),
                            "auc": roc_auc_score(g.y, g.p)})
# pm itself as predictor (no model)
mask = np.ones(len(ev), bool)
res_rows.insert(0, {"config": "raw_pm_benchmark", "n_oos": len(ev),
                    "auc_oos": roc_auc_score(y_all, ev.p_market.values),
                    "n_bets": 0, "unit_pnl_mean": np.nan, "unit_pnl_total": np.nan})
summ = pd.DataFrame(res_rows)
summ.to_csv(OUT / "archive_wf_summary.csv", index=False)
pd.DataFrame(wk_rows).to_csv(OUT / "archive_weekly.csv", index=False)
print(summ.round(4).to_string(index=False))
print("\nper-week AUC:")
if wk_rows:
    print(pd.DataFrame(wk_rows).pivot_table(index="week", columns="config", values="auc").round(3).to_string())
