#!/usr/bin/env python3
"""PHASE 1 (A/B/E) — leak-free long-history benchmark for the p_up_v2 redevelopment.

Backbone: scratchpad/pup_v2_dataset_fixed_20260702.parquet (2024-01 -> 2026-07-02,
lag-corrected price/tech features, label = next-1h close-to-close direction).

Groups:
  A = fixed price/tech features (16 available offline; the 4 live-only cols are all-NaN)
  B = cross-asset lead/lag: ETH & SOL 1h/4h returns + BTC-minus-alt return spreads (1h/24h)
  E = temporal: hour-of-day, day-of-week

Walk-forward: expanding window, weekly refits, embargo 1 bar (train ts <= week_start-2h
so no train label bar overlaps test). Metrics: AUC + weekly rank-IC vs next-1h direction.

Also: hour-of-day / day-of-week seasonality on FULL history (2019->2026, split by year).

Outputs (this dir): longhist_weekly_metrics.csv, longhist_summary.csv,
wf_preds_groups.parquet, seasonality_hour_by_year.csv, seasonality_dow_by_year.csv
"""
import warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
SCRATCH = Path("/private/tmp/claude-501/-Users-justindehn-Documents-ClaudeCode/600a1ce9-48de-420d-94bc-84e3bd4f9871/scratchpad")
OUT = PROJ / "reform_results" / "pup_v2_reform_20260702"

A_FEATURES = ["stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
              "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
              "composite_trend", "composite_rev", "composite_p_up",
              "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
              "rvol_1h"]
B_FEATURES = ["eth_ret_1h", "sol_ret_1h", "eth_ret_4h", "sol_ret_4h",
              "spread_eth_1h", "spread_sol_1h", "spread_eth_24h", "spread_sol_24h"]
E_FEATURES = ["hour_utc", "dow"]

# ── load backbone ──────────────────────────────────────────────────────────
df = pd.read_parquet(SCRATCH / "pup_v2_dataset_fixed_20260702.parquet")
df = df.sort_index()

# ── group B: cross-asset (bar-t closes of ETH/SOL are known at the same
#    decision time t+1h as BTC bar-t close — leak-consistent with backbone) ──
def latest_pq(sym):
    files = sorted((PROJ / "data").glob(f"binanceus_{sym}_1h_*.parquet"))
    best, best_end = None, None
    for f in files:
        d = pd.read_parquet(f)
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        # ignore corrupt tiny files
        if len(d) < 1000:
            continue
        if best_end is None or d.index[-1] > best_end:
            best, best_end = d, d.index[-1]
    return best

btc_c = df["close"]
btc_r1 = btc_c.pct_change()
btc_r24 = btc_c.pct_change(24)
for sym, tag in [("ETHUSDT", "eth"), ("SOLUSDT", "sol")]:
    alt = latest_pq(sym)["close"].reindex(df.index)
    df[f"{tag}_ret_1h"] = alt.pct_change()
    df[f"{tag}_ret_4h"] = alt.pct_change(4)
    df[f"spread_{tag}_1h"] = btc_r1 - alt.pct_change()
    df[f"spread_{tag}_24h"] = btc_r24 - alt.pct_change(24)

# ── group E ────────────────────────────────────────────────────────────────
# decision time is bar-open + 1h; hour feature = hour of the LABEL bar open
dec_time = df.index + pd.Timedelta(hours=1)
df["hour_utc"] = dec_time.hour
df["dow"] = dec_time.dayofweek

# ── WF harness ─────────────────────────────────────────────────────────────
CONFIGS = {
    "A":     A_FEATURES,
    "B":     B_FEATURES,
    "E":     E_FEATURES,
    "A+B":   A_FEATURES + B_FEATURES,
    "A+E":   A_FEATURES + E_FEATURES,
    "A+B+E": A_FEATURES + B_FEATURES + E_FEATURES,
}
y = df["label"].values.astype(int)
ts = df.index
WF_START = pd.Timestamp("2024-07-03", tz="UTC")
week_starts = pd.date_range(WF_START, ts[-1], freq="7D")

def fit_lgbm(X, tr_idx, va_idx, cat_idx):
    t = np.arange(len(tr_idx), dtype=float)
    w = np.exp(1.5 * t / max(t[-1], 1))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                           num_leaves=15, min_child_samples=60, reg_lambda=5.0,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbose=-1, n_jobs=4)
    m.fit(X[tr_idx], y[tr_idx], sample_weight=w,
          eval_set=[(X[va_idx], y[va_idx])],
          categorical_feature=cat_idx,
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return m

preds = {}
weekly_rows = []
for name, feats in CONFIGS.items():
    X = df[feats].values.astype(float)
    cat_idx = [i for i, f in enumerate(feats) if f in ("hour_utc", "dow")]
    p = np.full(len(df), np.nan)
    for ws in week_starts:
        te = np.where((ts >= ws) & (ts < ws + pd.Timedelta(days=7)))[0]
        if len(te) == 0:
            continue
        tr = np.where(ts <= ws - pd.Timedelta(hours=2))[0]   # embargo 1 bar
        nv = max(int(len(tr) * 0.05), 200)
        m = fit_lgbm(X, tr[:-nv], tr[-nv:], cat_idx)
        p[te] = m.predict_proba(X[te])[:, 1]
    preds[name] = p
    ev = pd.DataFrame({"y": y, "p": p}, index=ts).dropna()
    for wk, g in ev.groupby(ev.index.to_period("W-WED")):
        if g.y.nunique() < 2 or len(g) < 20:
            continue
        ic = spearmanr(g.p, g.y).statistic
        weekly_rows.append({"config": name, "week": str(wk), "n": len(g),
                            "auc": roc_auc_score(g.y, g.p), "rank_ic": ic})
    print(f"{name:<7} overall AUC={roc_auc_score(ev.y, ev.p):.4f}  n={len(ev)}", flush=True)

wk_df = pd.DataFrame(weekly_rows)
wk_df.to_csv(OUT / "longhist_weekly_metrics.csv", index=False)
pred_df = pd.DataFrame({f"p_{k}": v for k, v in preds.items()}, index=ts)
pred_df["label"] = y
pred_df.to_parquet(OUT / "wf_preds_groups.parquet")

rows = []
for name in CONFIGS:
    ev = pd.DataFrame({"y": y, "p": preds[name]}, index=ts).dropna()
    w = wk_df[wk_df.config == name]
    row = {"config": name, "n": len(ev), "n_weeks": len(w),
           "auc_overall": roc_auc_score(ev.y, ev.p),
           "auc_weekly_mean": w.auc.mean(), "auc_weekly_med": w.auc.median(),
           "pct_weeks_auc_gt_05": (w.auc > 0.5).mean(),
           "rank_ic_weekly_mean": w.rank_ic.mean(),
           "ic_tstat": w.rank_ic.mean() / (w.rank_ic.std() / np.sqrt(len(w)))}
    for yr, g in ev.groupby(ev.index.year):
        if g.y.nunique() > 1:
            row[f"auc_{yr}"] = roc_auc_score(g.y, g.p)
    rows.append(row)
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "longhist_summary.csv", index=False)
print("\n=== SUMMARY ===")
print(summ.round(4).to_string(index=False))

# ── E standalone: full-history seasonality (per feedback_test_on_long_history) ─
full = latest_pq("BTCUSDT") if False else None
files = sorted((PROJ / "data").glob("binanceus_BTCUSDT_1h_1970*.parquet"))
fh = pd.read_parquet(files[-1])
if fh.index.tz is None:
    fh.index = fh.index.tz_localize("UTC")
lab = (fh["close"].shift(-1) > fh["close"]).astype(float)[:-1]
fr = pd.DataFrame({"y": lab})
dec = fr.index + pd.Timedelta(hours=1)
fr["hour"] = dec.hour
fr["dow"] = dec.dayofweek
fr["year"] = fr.index.year
hr = fr.pivot_table(index="hour", columns="year", values="y", aggfunc="mean")
hr["n_per_year"] = fr.groupby("hour").size() / fr.year.nunique()
hr.to_csv(OUT / "seasonality_hour_by_year.csv")
dw = fr.pivot_table(index="dow", columns="year", values="y", aggfunc="mean")
dw.to_csv(OUT / "seasonality_dow_by_year.csv")
print(f"\nfull-history seasonality rows={len(fr)}  years {fr.year.min()}-{fr.year.max()}")
print("hour-of-day up-rate by year:")
print(hr.round(3).to_string())
print("dow up-rate by year:")
print(dw.round(3).to_string())
