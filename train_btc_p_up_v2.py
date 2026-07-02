#!/usr/bin/env python3
"""
train_btc_p_up_v2.py — Retrain the BTC p_up v2 directional model.

Label: did BTC close higher on the next 1h bar? (close[t+1] > close[t])

Features (20 total):
  OHLCV-derived (from parquet, full history):
    stoch_k_4h, ema50_dist, rsi_4h, rsi_14, macd_hist_1h,
    vwap_distance_pct, chg_4h_atr, bb_pct
  15m-derived (from 15m API fetch, April 2025+; NaN before):
    stoch_k, ema_stack_bias, ema_stretch_score, vwap_stretch_score  ← 3 new
  Composite signals (from parquet/recomputed, full history):
    composite_trend, composite_rev, composite_p_up, adx_1h, rvol_1h
  Live-only (always NaN in training, present at inference):
    confirmation_bias, stoch_bias, vpin_score, pm_drift_5m

Run: python3 train_btc_p_up_v2.py
"""

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    _USE_LGBM = True
except ImportError:
    raise SystemExit("LightGBM required: pip install lightgbm")

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
OUT_PATH   = Path("reform_results/btc_p_up_v2.pkl")
CAL_FILE   = Path("composite_calibration.json")
API_PKL    = Path("/tmp/btc_ohlcv.pkl")   # fresh 15m data from Binance API

NAN = float("nan")

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── OHLCV indicator helpers ────────────────────────────────────────────────────

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
    cp = c.shift(1)
    tr  = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    pdm = (h - h.shift(1)).clip(lower=0).where((h - h.shift(1)) > (lo.shift(1) - lo), 0)
    ndm = (lo.shift(1) - lo).clip(lower=0).where((lo.shift(1) - lo) > (h - h.shift(1)), 0)
    atr_w  = tr.ewm(com=p - 1, adjust=False).mean()
    pdm_w  = pdm.ewm(com=p - 1, adjust=False).mean()
    ndm_w  = ndm.ewm(com=p - 1, adjust=False).mean()
    pdi    = 100 * pdm_w / atr_w.replace(0, np.nan)
    ndi    = 100 * ndm_w / atr_w.replace(0, np.nan)
    dx     = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=p - 1, adjust=False).mean()

def daily_vwap_dist_series(df_1h):
    tp    = (df_1h["high"] + df_1h["low"] + df_1h["close"]) / 3
    vol   = df_1h["volume"]
    day   = df_1h.index.date
    cum_tpv = tp * vol
    cum_vol = vol.copy()
    # Reset at each new day
    for col_s, target in [(cum_tpv, "cum_tpv"), (cum_vol, "cum_vol")]:
        pass  # vectorised below
    df_tmp = pd.DataFrame({"tp": tp, "vol": vol, "day": day}, index=df_1h.index)
    df_tmp["cum_tpv"] = df_tmp.groupby("day")["tp"].transform(lambda x: (x * df_tmp.loc[x.index, "vol"]).cumsum())
    df_tmp["cum_vol"] = df_tmp.groupby("day")["vol"].transform("cumsum")
    vwap  = df_tmp["cum_tpv"] / df_tmp["cum_vol"].replace(0, np.nan)
    dist  = (df_1h["close"] - vwap) / vwap.replace(0, np.nan)
    # Sigma stretch
    df_tmp["day_std"] = df_tmp.groupby("day")["tp"].transform("std")
    stretch = pd.cut(
        dist / (df_tmp["day_std"] / vwap.replace(0, np.nan)).replace(0, np.nan),
        bins=[-np.inf, -2, -1, 1, 2, np.inf],
        labels=[2, 1, 0, -1, -2],
    ).astype(float)
    return dist * 100, stretch


# ── Composite signals from calibration table ──────────────────────────────────

def load_calibration():
    if not CAL_FILE.exists():
        return None
    import json
    with open(CAL_FILE) as f:
        return json.load(f)

def compute_composite_signals(df1h, df4h, cal):
    """Compute composite_trend, composite_rev, composite_p_up for each 1h bar."""
    c1h = df1h["close"]
    c4h = df4h["close"]

    # ── composite_trend (4h-based score −6..+6) ──
    rsi4     = rsi_series(c4h, 14)
    macd4    = _ema(c4h, 12) - _ema(c4h, 26)
    sig4     = macd4.ewm(span=9, adjust=False).mean()
    bb_mid4  = c4h.rolling(20).mean()
    bb_std4  = c4h.rolling(20).std()
    bb_lo4   = bb_mid4 - 2 * bb_std4
    bb_hi4   = bb_mid4 + 2 * bb_std4
    bb_pct4  = (c4h - bb_lo4) / (bb_hi4 - bb_lo4).replace(0, np.nan)
    sk4      = stoch_k_series(df4h["high"], df4h["low"], c4h, 14)
    wr4      = -100 * (df4h["high"].rolling(14).max() - c4h) / \
               (df4h["high"].rolling(14).max() - df4h["low"].rolling(14).min()).replace(0, np.nan)
    vol_ma4  = df4h["volume"].rolling(20).mean()
    vol_rat4 = df4h["volume"] / vol_ma4.replace(0, np.nan)

    trend_4h = pd.Series(0.0, index=df4h.index)
    trend_4h += (rsi4 > 55).astype(float) - (rsi4 < 45).astype(float)
    trend_4h += (macd4 > sig4).astype(float) - (macd4 <= sig4).astype(float)
    trend_4h += (bb_pct4 > 0.80).astype(float) - (bb_pct4 < 0.20).astype(float)
    trend_4h += (sk4 > 80).astype(float) - (sk4 < 20).astype(float)
    trend_4h += (wr4 > -20).astype(float) - (wr4 < -80).astype(float)
    trend_4h += ((vol_rat4 > 1.5) & (c4h > c4h.shift(1))).astype(float) - \
                ((vol_rat4 > 1.5) & (c4h < c4h.shift(1))).astype(float)
    trend_4h = trend_4h.clip(-6, 6)

    # ── composite_rev (1h-based score −8..+8) ──
    rsi1     = rsi_series(c1h, 14)
    sk1      = stoch_k_series(df1h["high"], df1h["low"], c1h, 14)
    vd, _    = daily_vwap_dist_series(df1h)
    log_ret  = np.log(c1h / c1h.shift(1))
    z_std    = log_ret.rolling(24).std()
    z_score  = log_ret / z_std.replace(0, np.nan)

    rev_1h = pd.Series(0.0, index=df1h.index)
    rev_1h += 2 * (rsi1 < 30).astype(float) + (rsi1 < 40).astype(float)
    rev_1h -= 2 * (rsi1 > 70).astype(float) + (rsi1 > 60).astype(float)
    rev_1h += 2 * (sk1 < 10).astype(float)  + (sk1 < 20).astype(float)
    rev_1h -= 2 * (sk1 > 90).astype(float)  + (sk1 > 80).astype(float)
    rev_1h += 2 * (vd < -1.5).astype(float) + (vd < -0.5).astype(float)
    rev_1h -= 2 * (vd >  1.5).astype(float) + (vd >  0.5).astype(float)
    rev_1h += 2 * (z_score < -2.0).astype(float) + (z_score < -1.5).astype(float)
    rev_1h -= 2 * (z_score >  2.0).astype(float) + (z_score >  1.5).astype(float)
    rev_1h  = rev_1h.clip(-8, 8)

    # ── Map 4h trend to 1h index ──
    trend_1h = trend_4h.reindex(df1h.index, method="ffill")

    # ── composite_p_up lookup ──
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


# ── 15m derived features ──────────────────────────────────────────────────────

def compute_15m_features(raw_15m: list) -> pd.DataFrame:
    """
    raw_15m: list of raw Binance kline lists [open_ms, o, h, l, c, v, ...]
             OR list of dicts {ts, o, h, l, c, v}.
    Returns DataFrame indexed by UTC timestamp with columns:
      stoch_k, ema_stack_bias, ema_stretch_score
    """
    if isinstance(raw_15m[0], (list, tuple)):
        # Raw Binance format: [open_time_ms, open, high, low, close, volume, ...]
        records = []
        for k in raw_15m:
            records.append({
                "ts": int(k[0]) // 1000,
                "open": float(k[1]), "high": float(k[2]),
                "low":  float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    else:
        df = pd.DataFrame(raw_15m)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)

    c = df["close"]
    sk = stoch_k_series(df["high"], df["low"], c, 14).rename("stoch_k")

    e9  = _ema(c, 9)
    e21 = _ema(c, 21)
    e50 = _ema(c, 50)
    bull = (e9 > e21) & (e21 > e50) & (c > e9)
    bear = (e9 < e21) & (e21 < e50) & (c < e9)
    ema_stack = pd.Series(0, index=c.index, dtype=float)
    ema_stack[bull] =  1
    ema_stack[bear] = -1
    ema_stack = ema_stack.rename("ema_stack_bias")

    e20     = _ema(c, 20)
    stretch = (c - e20) / e20.replace(0, np.nan)
    ema_ex  = pd.cut(stretch, bins=[-np.inf, -0.001, 0.001, np.inf],
                     labels=[1, 0, -1]).astype(float).rename("ema_stretch_score")

    return pd.concat([sk, ema_stack, ema_ex], axis=1)


# ── Build training dataset ────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    print("Loading OHLCV parquet files...")
    f1h = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    f4h = sorted(DATA_DIR.glob("binanceus_BTCUSDT_4h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    print(f"  1h: {f1h.name}")
    print(f"  4h: {f4h.name}")

    df1h = pd.read_parquet(f1h)
    df4h = pd.read_parquet(f4h)
    for d in (df1h, df4h):
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
    # Trim to 2024-01-01 onward for reasonable training window
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    df1h = df1h[df1h.index >= cutoff].copy()
    df4h = df4h[df4h.index >= cutoff].copy()
    print(f"  1h: {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
    print(f"  4h: {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

    # ── OHLCV features ────────────────────────────────────────────────────────
    print("Computing OHLCV indicators...")
    c1h = df1h["close"]
    c4h = df4h["close"]

    ind1h = pd.DataFrame({
        "rsi_14":       rsi_series(c1h, 14),
        "macd_hist_1h": macd_hist_series(c1h),
        "bb_pct":       bb_pct_series(c1h),
        "ema50_dist":   ema50_dist_series(c1h),
        "adx_1h":       adx_series(df1h["high"], df1h["low"], c1h, 14),
        "rvol_1h":      (c1h / c1h.shift(1) - 1).rolling(1).std() /
                        (c1h / c1h.shift(1) - 1).rolling(24).std().replace(0, np.nan),
    }, index=df1h.index)
    # Use volume-based rvol instead
    ind1h["rvol_1h"] = df1h["volume"] / df1h["volume"].rolling(24).mean().replace(0, np.nan)
    vd_pct, vwap_stretch = daily_vwap_dist_series(df1h)
    ind1h["vwap_distance_pct"] = vd_pct / 100.0   # match live signal scale
    ind1h["vwap_stretch_score_1h"] = vwap_stretch  # 1h-bar session VWAP stretch

    ind4h = pd.DataFrame({
        "stoch_k_4h": stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
        "rsi_4h":     rsi_series(c4h, 14),
        "chg_4h_atr": chg_4h_atr_series(df4h),
    }, index=df4h.index)

    # ── Composite signals ─────────────────────────────────────────────────────
    print("Computing composite signals...")
    cal = load_calibration()
    if cal is None:
        print("  WARNING: composite_calibration.json not found; p_up=0.504 everywhere")
    trend_s, rev_s, pup_s = compute_composite_signals(df1h, df4h, cal)

    # ── 15m features ─────────────────────────────────────────────────────────
    print("Computing 15m features...")
    if API_PKL.exists():
        with open(API_PKL, "rb") as f:
            api_data = pickle.load(f)
        df_15m = compute_15m_features(api_data["15m"])
        print(f"  15m: {len(df_15m):,} bars  ({df_15m.index[0].date()} → {df_15m.index[-1].date()})")
    else:
        print("  WARNING: /tmp/btc_ohlcv.pkl not found — 15m features will be NaN")
        df_15m = pd.DataFrame(columns=["stoch_k", "ema_stack_bias", "ema_stretch_score"])
        df_15m.index = pd.DatetimeIndex([], tz="UTC")

    # ── Merge everything onto 1h index ────────────────────────────────────────
    print("Merging features onto 1h index...")
    df = df1h[["close"]].copy()
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["label"])

    # 1h indicators
    df = df.join(ind1h, how="left")

    # 4h indicators via merge_asof
    ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})
    df_r    = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    df_r    = pd.merge_asof(df_r, ind4h_r, on="ts", direction="backward")
    df      = df_r.set_index("ts")

    # composite signals
    df["composite_trend"] = trend_s.reindex(df.index)
    df["composite_rev"]   = rev_s.reindex(df.index)
    df["composite_p_up"]  = pup_s.reindex(df.index)

    # 15m features: merge_asof (latest 15m bar <= 1h bar open time)
    if len(df_15m) > 0:
        df_15m_r = df_15m.reset_index().rename(columns={"ts": "ts"})
        df_15m_r.columns = df_15m_r.columns.str.strip()
        df_r2 = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        df_r2 = pd.merge_asof(
            df_r2.sort_values("ts"),
            df_15m_r.sort_values("ts"),
            on="ts", direction="backward", tolerance=pd.Timedelta("60min"),
        )
        df = df_r2.set_index("ts")

    # Use 1h session VWAP stretch as fallback for vwap_stretch_score if 15m not available
    if "vwap_stretch_score" not in df.columns:
        df["vwap_stretch_score"] = df.get("vwap_stretch_score_1h", np.nan)
    else:
        # Fill NaN periods with 1h approximation
        df["vwap_stretch_score"] = df["vwap_stretch_score"].fillna(df.get("vwap_stretch_score_1h", np.nan))

    # Live-only signals: always NaN in training
    for col in ("confirmation_bias", "stoch_bias", "vpin_score", "pm_drift_5m"):
        df[col] = np.nan

    # Ensure all feature columns exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_index()
    print(f"Dataset: {len(df):,} bars")
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    n = len(df)
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

    print(f"\nFeature coverage (train set):")
    for f in FEATURES:
        cov = tr[f].notna().mean() * 100
        tag = "  [OHLCV]" if f in ("stoch_k_4h","ema50_dist","rsi_4h","rsi_14",
                                    "macd_hist_1h","chg_4h_atr","bb_pct") else \
              "  [15m]"   if f in ("stoch_k","ema_stack_bias","ema_stretch_score",
                                    "vwap_stretch_score") else \
              "  [live-NaN]" if f in ("confirmation_bias","stoch_bias","vpin_score","pm_drift_5m") else \
              "  [composite]"
        print(f"  {f:<25} {cov:5.1f}%{tag}")

    X_tr = tr[FEATURES].values.astype(float)
    y_tr = tr["label"].values.astype(int)
    X_va = va[FEATURES].values.astype(float)
    y_va = va["label"].values.astype(int)
    X_te = te[FEATURES].values.astype(float)
    y_te = te["label"].values.astype(int)

    # Recency weighting
    t_vals = np.arange(len(tr), dtype=float)
    weights = np.exp(1.5 * t_vals / t_vals[-1])

    print("\nTraining LightGBM (LGBMClassifier)...")
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
    print(f"  Val AUC:  {auc_val:.4f}  (best iter: {model.best_iteration_})")
    print(f"  Test AUC: {auc_test:.4f}")

    print(f"\nFeature importance (gain):")
    gain  = model.booster_.feature_importance("gain")
    total = gain.sum() or 1
    for nm, g in sorted(zip(FEATURES, gain), key=lambda x: -x[1]):
        print(f"  {nm:<25} {g/total*100:5.1f}%")

    return {
        "clf":        model,
        "features":   FEATURES,
        "price_feats": [f for f in FEATURES if f not in
                        ("confirmation_bias","stoch_bias","vpin_score","pm_drift_5m")],
        "live_feats":  ["confirmation_bias","stoch_bias","vpin_score","pm_drift_5m"],
        "auc_val":    auc_val,
        "auc_test":   auc_test,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import shutil
    print("=" * 65)
    print("  train_btc_p_up_v2.py — Retraining BTC directional p_up model")
    print("=" * 65)

    # Backup existing model
    if OUT_PATH.exists():
        bak = OUT_PATH.with_suffix(".pkl.bak")
        shutil.copy2(OUT_PATH, bak)
        print(f"Backed up existing model → {bak.name}")

    df = build_dataset()

    print("\n" + "=" * 65)
    pipe = train(df)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Val AUC: {pipe['auc_val']:.4f}  Test AUC: {pipe['auc_test']:.4f}")
    print("=" * 65)
    print("Next: restart paper_trade_runner.py to load the new model.")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
