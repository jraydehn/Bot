#!/usr/bin/env python3
"""
train_eth_sol_p_up_v2.py — Train directional p_up v2 model for ETH or SOL.

Label: did close[t+1] > close[t] on the 1h bar?

Identical 20-feature set as BTC p_up_v2:
  OHLCV (full history):
    stoch_k_4h, ema50_dist, rsi_4h, rsi_14, macd_hist_1h,
    vwap_distance_pct, chg_4h_atr, bb_pct, adx_1h, rvol_1h
  15m-derived (from parquet, 2024-2026):
    stoch_k, ema_stack_bias, ema_stretch_score, vwap_stretch_score
  Composite signals (recomputed from OHLCV + asset calibration JSON):
    composite_trend, composite_rev, composite_p_up
  Live-only (always NaN in training, present at inference):
    confirmation_bias, stoch_bias, vpin_score, pm_drift_5m

Outputs:
  reform_results/eth_p_up_v2.pkl
  reform_results/sol_p_up_v2.pkl

Run:
  python3 train_eth_sol_p_up_v2.py --asset ETH
  python3 train_eth_sol_p_up_v2.py --asset SOL
  python3 train_eth_sol_p_up_v2.py --asset ETH SOL
"""

import argparse
import json
import math
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("LightGBM required: pip install lightgbm")

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "reform_results"

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]

NAN = float("nan")


# ── indicator helpers ──────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def stoch_k_series(h, lo, c, k=14):
    ll = lo.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100

def atr_series(h, lo, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()

def macd_hist_series(c, f=12, s=26, sig=9):
    macd = _ema(c, f) - _ema(c, s)
    return macd - macd.ewm(span=sig, adjust=False).mean()

def bb_pct_series(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    lo  = mid - 2 * std
    hi  = mid + 2 * std
    return (c - lo) / (hi - lo).replace(0, np.nan)

def ema50_dist_series(c):
    return (c - _ema(c, 50)) / _ema(c, 50).replace(0, np.nan) * 100

def chg_4h_atr_series(df4):
    a = atr_series(df4["high"], df4["low"], df4["close"], 14)
    return (df4["close"] - df4["close"].shift(5)) / a.replace(0, np.nan)

def adx_series(h, lo, c, p=14):
    cp  = c.shift(1)
    tr  = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    pdm = (h - h.shift(1)).clip(lower=0).where((h - h.shift(1)) > (lo.shift(1) - lo), 0)
    ndm = (lo.shift(1) - lo).clip(lower=0).where((lo.shift(1) - lo) > (h - h.shift(1)), 0)
    atr_w = tr.ewm(com=p - 1, adjust=False).mean()
    pdm_w = pdm.ewm(com=p - 1, adjust=False).mean()
    ndm_w = ndm.ewm(com=p - 1, adjust=False).mean()
    pdi   = 100 * pdm_w / atr_w.replace(0, np.nan)
    ndi   = 100 * ndm_w / atr_w.replace(0, np.nan)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=p - 1, adjust=False).mean()

def daily_vwap_dist_series(df1h):
    tp  = (df1h["high"] + df1h["low"] + df1h["close"]) / 3
    vol = df1h["volume"]
    day = df1h.index.date
    df_tmp = pd.DataFrame({"tp": tp, "vol": vol, "day": day}, index=df1h.index)
    df_tmp["cum_tpv"] = df_tmp.groupby("day")["tp"].transform(
        lambda x: (x * df_tmp.loc[x.index, "vol"]).cumsum()
    )
    df_tmp["cum_vol"] = df_tmp.groupby("day")["vol"].transform("cumsum")
    vwap  = df_tmp["cum_tpv"] / df_tmp["cum_vol"].replace(0, np.nan)
    dist  = (df1h["close"] - vwap) / vwap.replace(0, np.nan)
    df_tmp["day_std"] = df_tmp.groupby("day")["tp"].transform("std")
    stretch = pd.cut(
        dist / (df_tmp["day_std"] / vwap.replace(0, np.nan)).replace(0, np.nan),
        bins=[-np.inf, -2, -1, 1, 2, np.inf],
        labels=[2, 1, 0, -1, -2],
    ).astype(float)
    return dist * 100, stretch


def compute_composite_signals(df1h, df4h, cal):
    c1h = df1h["close"]
    c4h = df4h["close"]

    rsi4    = rsi_series(c4h, 14)
    macd4   = _ema(c4h, 12) - _ema(c4h, 26)
    sig4    = macd4.ewm(span=9, adjust=False).mean()
    bb_mid4 = c4h.rolling(20).mean()
    bb_std4 = c4h.rolling(20).std()
    bb_lo4  = bb_mid4 - 2 * bb_std4
    bb_hi4  = bb_mid4 + 2 * bb_std4
    bb_pct4 = (c4h - bb_lo4) / (bb_hi4 - bb_lo4).replace(0, np.nan)
    sk4     = stoch_k_series(df4h["high"], df4h["low"], c4h, 14)
    wr4     = -100 * (df4h["high"].rolling(14).max() - c4h) / \
              (df4h["high"].rolling(14).max() - df4h["low"].rolling(14).min()).replace(0, np.nan)
    vol_ma4 = df4h["volume"].rolling(20).mean()
    vol_r4  = df4h["volume"] / vol_ma4.replace(0, np.nan)

    trend_4h = pd.Series(0.0, index=df4h.index)
    trend_4h += (rsi4 > 55).astype(float) - (rsi4 < 45).astype(float)
    trend_4h += (macd4 > sig4).astype(float) - (macd4 <= sig4).astype(float)
    trend_4h += (bb_pct4 > 0.80).astype(float) - (bb_pct4 < 0.20).astype(float)
    trend_4h += (sk4 > 80).astype(float) - (sk4 < 20).astype(float)
    trend_4h += (wr4 > -20).astype(float) - (wr4 < -80).astype(float)
    trend_4h += ((vol_r4 > 1.5) & (c4h > c4h.shift(1))).astype(float) - \
                ((vol_r4 > 1.5) & (c4h < c4h.shift(1))).astype(float)
    trend_4h = trend_4h.clip(-6, 6)

    rsi1   = rsi_series(c1h, 14)
    sk1    = stoch_k_series(df1h["high"], df1h["low"], c1h, 14)
    vd, _  = daily_vwap_dist_series(df1h)
    log_r  = np.log(c1h / c1h.shift(1))
    z_std  = log_r.rolling(24).std()
    z_sc   = log_r / z_std.replace(0, np.nan)

    rev_1h = pd.Series(0.0, index=df1h.index)
    rev_1h += 2 * (rsi1 < 30).astype(float) + (rsi1 < 40).astype(float)
    rev_1h -= 2 * (rsi1 > 70).astype(float) + (rsi1 > 60).astype(float)
    rev_1h += 2 * (sk1 < 10).astype(float)  + (sk1 < 20).astype(float)
    rev_1h -= 2 * (sk1 > 90).astype(float)  + (sk1 > 80).astype(float)
    rev_1h += 2 * (vd < -1.5).astype(float) + (vd < -0.5).astype(float)
    rev_1h -= 2 * (vd >  1.5).astype(float) + (vd >  0.5).astype(float)
    rev_1h += 2 * (z_sc < -2.0).astype(float) + (z_sc < -1.5).astype(float)
    rev_1h -= 2 * (z_sc >  2.0).astype(float) + (z_sc >  1.5).astype(float)
    rev_1h  = rev_1h.clip(-8, 8)

    trend_1h = trend_4h.reindex(df1h.index, method="ffill")

    if cal:
        def lookup(t, r):
            k = f"{int(round(t))}_{int(round(r))}"
            e = cal.get(k)
            return e["p_yes"] if e and e.get("n", 0) >= 5 else 0.504
        p_up_ser = pd.Series(
            [lookup(t, r) for t, r in zip(trend_1h, rev_1h)],
            index=df1h.index,
        )
    else:
        p_up_ser = pd.Series(0.504, index=df1h.index)

    return trend_1h.rename("composite_trend"), rev_1h.rename("composite_rev"), p_up_ser.rename("composite_p_up")


def compute_15m_features(df15: pd.DataFrame) -> pd.DataFrame:
    c  = df15["close"]
    sk = stoch_k_series(df15["high"], df15["low"], c, 14).rename("stoch_k")

    e9  = _ema(c, 9)
    e21 = _ema(c, 21)
    e50 = _ema(c, 50)
    bull = (e9 > e21) & (e21 > e50) & (c > e9)
    bear = (e9 < e21) & (e21 < e50) & (c < e9)
    ema_stack = pd.Series(0.0, index=c.index)
    ema_stack[bull] =  1
    ema_stack[bear] = -1
    ema_stack = ema_stack.rename("ema_stack_bias")

    e20    = _ema(c, 20)
    stretch = (c - e20) / e20.replace(0, np.nan)
    ema_ex  = pd.cut(stretch, bins=[-np.inf, -0.001, 0.001, np.inf],
                     labels=[1, 0, -1]).astype(float).rename("ema_stretch_score")

    return pd.concat([sk, ema_stack, ema_ex], axis=1)


# ── dataset builder ────────────────────────────────────────────────────────────

def build_dataset(asset: str) -> pd.DataFrame:
    sym = f"{asset}USDT"
    print(f"\nLoading {asset} OHLCV parquet files...")

    f1h = sorted(DATA.glob(f"binanceus_{sym}_1h_*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    f4h = sorted(DATA.glob(f"binanceus_{sym}_4h_*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    f15m_list = sorted(DATA.glob(f"binanceus_{sym}_15m_*.parquet"),
                       key=lambda p: p.stat().st_mtime)

    print(f"  1h : {f1h.name}")
    print(f"  4h : {f4h.name}")

    df1h = pd.read_parquet(f1h)
    df4h = pd.read_parquet(f4h)
    for d in (df1h, df4h):
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")

    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    df1h = df1h[df1h.index >= cutoff].copy()
    df4h = df4h[df4h.index >= cutoff].copy()
    print(f"  1h : {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
    print(f"  4h : {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

    # 1h indicators
    print("Computing 1h/4h indicators...")
    c1h = df1h["close"]
    c4h = df4h["close"]
    vd_pct, vwap_stretch_1h = daily_vwap_dist_series(df1h)

    ind1h = pd.DataFrame({
        "rsi_14":             rsi_series(c1h, 14),
        "macd_hist_1h":       macd_hist_series(c1h),
        "bb_pct":             bb_pct_series(c1h),
        "ema50_dist":         ema50_dist_series(c1h),
        "adx_1h":             adx_series(df1h["high"], df1h["low"], c1h, 14),
        "rvol_1h":            df1h["volume"] / df1h["volume"].rolling(24).mean().replace(0, np.nan),
        "vwap_distance_pct":  vd_pct / 100.0,
        "vwap_stretch_score_1h": vwap_stretch_1h,
    }, index=df1h.index)

    ind4h = pd.DataFrame({
        "stoch_k_4h": stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
        "rsi_4h":     rsi_series(c4h, 14),
        "chg_4h_atr": chg_4h_atr_series(df4h),
    }, index=df4h.index)

    # composite signals
    print("Computing composite signals...")
    cal_path = ROOT / f"composite_calibration_{asset.lower()}.json"
    cal = None
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        print(f"  Loaded calibration: {cal_path.name} ({len(cal)} entries)")
    else:
        print(f"  WARNING: {cal_path.name} not found — using p_up=0.504")
    trend_s, rev_s, pup_s = compute_composite_signals(df1h, df4h, cal)

    # 15m indicators
    print("Computing 15m features...")
    df15m_feats = pd.DataFrame()
    for f15m in reversed(f15m_list):
        try:
            df15 = pd.read_parquet(f15m)
            if df15.index.tz is None:
                df15.index = df15.index.tz_localize("UTC")
            df15 = df15[df15.index >= cutoff].copy()
            # Validate: must have 15m-ish frequency (check median gap < 1h)
            if len(df15) >= 100:
                gaps = df15.index.to_series().diff().dt.total_seconds().dropna()
                med_gap = gaps.median()
                if med_gap <= 3600:  # 15m or 1h bars are OK
                    df15m_feats = compute_15m_features(df15)
                    print(f"  15m: {f15m.name}  {len(df15m_feats):,} bars  "
                          f"({df15m_feats.index[0].date()} → {df15m_feats.index[-1].date()})")
                    break
        except Exception:
            pass
    if df15m_feats.empty:
        print("  WARNING: no valid 15m data found — 15m features NaN in training (will be populated at inference)")
        df15m_feats = pd.DataFrame(columns=["stoch_k", "ema_stack_bias", "ema_stretch_score"])
        df15m_feats.index = pd.DatetimeIndex([], tz="UTC")

    # merge onto 1h index
    print("Merging onto 1h index...")
    df = df1h[["close"]].copy()
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["label"])

    df = df.join(ind1h, how="left")

    # 4h via merge_asof
    ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})
    df_r    = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    df_r    = pd.merge_asof(df_r, ind4h_r, on="ts", direction="backward")
    df      = df_r.set_index("ts")

    df["composite_trend"] = trend_s.reindex(df.index)
    df["composite_rev"]   = rev_s.reindex(df.index)
    df["composite_p_up"]  = pup_s.reindex(df.index)

    if len(df15m_feats) > 0:
        # Normalize index name to "ts" for merge_asof
        df15m_feats.index.name = "ts"
        df15_r = df15m_feats.reset_index()
        df15_r.columns = df15_r.columns.str.strip()
        df.index.name = "ts"
        df_r2 = df.reset_index()
        df_r2 = pd.merge_asof(
            df_r2.sort_values("ts"),
            df15_r.sort_values("ts"),
            on="ts", direction="backward", tolerance=pd.Timedelta("60min"),
        )
        df = df_r2.set_index("ts")

    if "vwap_stretch_score" not in df.columns:
        df["vwap_stretch_score"] = df.get("vwap_stretch_score_1h", np.nan)
    else:
        df["vwap_stretch_score"] = df["vwap_stretch_score"].fillna(
            df.get("vwap_stretch_score_1h", np.nan)
        )

    for col in ("confirmation_bias", "stoch_bias", "vpin_score", "pm_drift_5m"):
        df[col] = np.nan

    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_index()
    print(f"Dataset: {len(df):,} bars")
    return df


# ── training ───────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, asset: str) -> dict:
    n      = len(df)
    n_test = max(int(n * 0.15), 100)
    n_val  = max(int(n * 0.15), 100)
    n_tr   = n - n_val - n_test

    tr = df.iloc[:n_tr]
    va = df.iloc[n_tr:n_tr + n_val]
    te = df.iloc[n_tr + n_val:]

    print(f"\nTime split:")
    print(f"  Train: {len(tr):,}  ({tr.index[0].date()} → {tr.index[-1].date()})")
    print(f"  Val:   {len(va):,}  ({va.index[0].date()} → {va.index[-1].date()})")
    print(f"  Test:  {len(te):,}  ({te.index[0].date()} → {te.index[-1].date()})")

    label_rate = tr["label"].mean()
    print(f"\n  Label rate (train): {label_rate:.3f}  (% hours price went up)")

    print(f"\nFeature coverage (train set):")
    ohlcv_feats = {"stoch_k_4h","ema50_dist","rsi_4h","rsi_14","macd_hist_1h",
                   "chg_4h_atr","bb_pct","adx_1h","rvol_1h","vwap_distance_pct"}
    m15_feats   = {"stoch_k","ema_stack_bias","ema_stretch_score","vwap_stretch_score"}
    live_feats  = {"confirmation_bias","stoch_bias","vpin_score","pm_drift_5m"}
    for f in FEATURES:
        cov = tr[f].notna().mean() * 100
        tag = "[OHLCV]" if f in ohlcv_feats else \
              "[15m]"   if f in m15_feats  else \
              "[live-NaN]" if f in live_feats else "[composite]"
        print(f"  {f:<28} {cov:5.1f}%  {tag}")

    X_tr = tr[FEATURES].values.astype(float)
    y_tr = tr["label"].values.astype(int)
    X_va = va[FEATURES].values.astype(float)
    y_va = va["label"].values.astype(int)
    X_te = te[FEATURES].values.astype(float)
    y_te = te["label"].values.astype(int)

    t_vals  = np.arange(len(tr), dtype=float)
    weights = np.exp(1.5 * t_vals / t_vals[-1])

    print(f"\nTraining LightGBM ({asset})...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        min_child_samples=60,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=2,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    auc_val  = roc_auc_score(y_va, p_va)
    auc_test = roc_auc_score(y_te, p_te)
    print(f"\n  Val AUC:  {auc_val:.4f}  (best iter: {model.best_iteration_})")
    print(f"  Test AUC: {auc_test:.4f}")

    print(f"\nFeature importance (gain) — {asset}:")
    gain  = model.booster_.feature_importance("gain")
    total = gain.sum() or 1
    ranked = sorted(zip(FEATURES, gain), key=lambda x: -x[1])
    for nm, g in ranked:
        bar = "█" * int(g / total * 40)
        print(f"  {nm:<28} {g/total*100:5.1f}%  {bar}")

    # p_up calibration check: does higher p_up actually predict higher win rate?
    print(f"\nCalibration check (p_up_v2 vs actual label, test set):")
    bins = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_te >= lo) & (p_te < hi)
        n_b  = mask.sum()
        if n_b == 0:
            continue
        wr = y_te[mask].mean()
        print(f"  p_up [{lo:.2f},{hi:.2f})  n={n_b:4d}  actual_WR={wr:.3f}")

    return {
        "clf":       model,
        "features":  FEATURES,
        "auc_val":   auc_val,
        "auc_test":  auc_test,
        "asset":     asset,
        "label_rate": float(label_rate),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.chdir(ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", nargs="+", default=["ETH", "SOL"],
                        choices=["ETH", "SOL", "BTC"],
                        help="Asset(s) to train (default: ETH SOL)")
    args = parser.parse_args()

    SEP = "=" * 68
    for asset in args.asset:
        print(f"\n{SEP}")
        print(f"  {asset} p_up v2 — directional next-1h-bar classifier")
        print(SEP)

        out_path = OUT_DIR / f"{asset.lower()}_p_up_v2.pkl"
        if out_path.exists():
            bak = out_path.with_suffix(".pkl.bak")
            shutil.copy2(out_path, bak)
            print(f"Backed up existing model → {bak.name}")

        df   = build_dataset(asset)
        pipe = train(df, asset)

        with open(out_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"\nSaved → {out_path}")
        print(f"Val AUC={pipe['auc_val']:.4f}  Test AUC={pipe['auc_test']:.4f}")

    print(f"\n{SEP}")
    print("Done. Next: wire inference into paper_trade_runner.py when results look good.")
    print(SEP)


if __name__ == "__main__":
    main()
