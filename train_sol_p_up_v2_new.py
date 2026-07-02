#!/usr/bin/env python3
"""
train_sol_p_up_v2_new.py

Build SOL p_up v2 model with expanded feature set including new signals:
  - All BTC model features: ema50_dist, stoch_k_4h, rsi_14, chg_4h_atr, rsi_4h,
    macd_hist_1h, vwap_distance_pct, bb_pct, vwap_stretch_score, rvol_1h,
    composite_rev, composite_trend
  - New signals: kalman_velocity, kalman_residual, ou_theta, hurst_exponent,
    autocorr_1h, pc1_rsi, donchian_pos, chg_4h

Steps:
  1. Fetch 2yr SOL/USDT 1h + 4h from Binance US (or use existing parquet)
  2. Compute all features
  3. IC analysis per year (2024, 2025, 2026)
  4. Feature selection: |IC| > 0.02 stable across years
  5. Train LightGBM with 60/20/20 split + Platt calibration
  6. Save to reform_results/sol_p_up_v2_new.pkl
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("pip install lightgbm")

warnings.filterwarnings("ignore")

ROOT    = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
DATA    = ROOT / "data"
OUT_DIR = ROOT / "reform_results"

# ─────────────────────────── Binance fetch ────────────────────────────────────

def fetch_binance_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Fetch klines from Binance US in batches."""
    url = "https://api.binance.us/api/v3/klines"
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = dict(symbol=symbol, interval=interval,
                      startTime=cur, endTime=end_ms, limit=limit)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        last_ts = int(data[-1][0])
        if last_ts <= cur:
            break
        cur = last_ts + 1
    return all_rows

def klines_to_df(rows):
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["ts"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df.set_index("ts")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open","high","low","close","volume"]].sort_index()

def load_or_fetch(symbol, interval, start_str, end_str):
    tag = f"binanceus_{symbol}_{interval}_{start_str}_{end_str}"
    cached = list(DATA.glob(f"binanceus_{symbol}_{interval}_*.parquet"))
    # Use the most recent existing file that covers the range
    # (existing files go to 2026-06-24, which covers our target)
    if cached:
        cached_sorted = sorted(cached, key=lambda p: p.stat().st_mtime, reverse=True)
        df = pd.read_parquet(cached_sorted[0])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        start_ts = pd.Timestamp(start_str, tz="UTC")
        end_ts   = pd.Timestamp(end_str,   tz="UTC")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(df) > 100:
            print(f"  Using cached parquet for {symbol} {interval}: {len(df):,} bars "
                  f"({df.index[0].date()} → {df.index[-1].date()})")
            return df
    # Fetch from API
    print(f"  Fetching {symbol} {interval} from Binance US API...")
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end_str,   tz="UTC").timestamp() * 1000)
    rows = fetch_binance_klines(symbol, interval, start_ms, end_ms)
    df = klines_to_df(rows)
    out_path = DATA / f"{tag}.parquet"
    df.to_parquet(out_path)
    print(f"  Saved {len(df):,} bars → {out_path.name}")
    return df

# ─────────────────────────── Indicator helpers ────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def stoch_k_series(h, lo, c, k=14):
    ll = lo.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100

def atr_series(h, lo, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

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
    e50 = _ema(c, 50)
    return (c - e50) / e50.replace(0, np.nan) * 100

def chg_4h_atr_series(df4):
    a = atr_series(df4["high"], df4["low"], df4["close"], 14)
    return (df4["close"] - df4["close"].shift(5)) / a.replace(0, np.nan)

def daily_vwap_dist_series(df1h):
    tp  = (df1h["high"] + df1h["low"] + df1h["close"]) / 3
    vol = df1h["volume"]
    day = df1h.index.date
    df_tmp = pd.DataFrame({"tp": tp, "vol": vol, "day": day}, index=df1h.index)
    df_tmp["cum_tpv"] = df_tmp.groupby("day")["tp"].transform(
        lambda x: (x * df_tmp.loc[x.index, "vol"]).cumsum()
    )
    df_tmp["cum_vol"] = df_tmp.groupby("day")["vol"].transform("cumsum")
    vwap = df_tmp["cum_tpv"] / df_tmp["cum_vol"].replace(0, np.nan)
    dist = (df1h["close"] - vwap) / vwap.replace(0, np.nan)
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

    rsi1  = rsi_series(c1h, 14)
    sk1   = stoch_k_series(df1h["high"], df1h["low"], c1h, 14)
    vd, _ = daily_vwap_dist_series(df1h)
    log_r = np.log(c1h / c1h.shift(1))
    z_std = log_r.rolling(24).std()
    z_sc  = log_r / z_std.replace(0, np.nan)

    rev_1h = pd.Series(0.0, index=df1h.index)
    rev_1h += 2 * (rsi1 < 30).astype(float) + (rsi1 < 40).astype(float)
    rev_1h -= 2 * (rsi1 > 70).astype(float) + (rsi1 > 60).astype(float)
    rev_1h += 2 * (sk1 < 10).astype(float) + (sk1 < 20).astype(float)
    rev_1h -= 2 * (sk1 > 90).astype(float) + (sk1 > 80).astype(float)
    rev_1h += 2 * (vd < -1.5).astype(float) + (vd < -0.5).astype(float)
    rev_1h -= 2 * (vd > 1.5).astype(float) + (vd > 0.5).astype(float)
    rev_1h += 2 * (z_sc < -2.0).astype(float) + (z_sc < -1.5).astype(float)
    rev_1h -= 2 * (z_sc > 2.0).astype(float) + (z_sc > 1.5).astype(float)
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

    return (
        trend_1h.rename("composite_trend"),
        rev_1h.rename("composite_rev"),
        p_up_ser.rename("composite_p_up"),
    )

# ─────────────────────────── New signals ─────────────────────────────────────

def kalman_filter(prices):
    """
    Simple Kalman filter: state = [level, velocity]
    F = [[1,1],[0,1]], Q = 1e-5 * I, R = 0.01
    Returns (levels, velocities, residuals)
    """
    n = len(prices)
    levels    = np.full(n, np.nan)
    velocities = np.full(n, np.nan)
    residuals  = np.full(n, np.nan)

    if n < 2:
        return levels, velocities, residuals

    # Init with first two prices
    x = np.array([prices.iloc[0], prices.iloc[1] - prices.iloc[0]])
    P = np.eye(2) * 1.0

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-5 * np.eye(2)
    R = np.array([[0.01]])

    for i in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z = prices.iloc[i]
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T + R)[0, 0]
        K = P_pred @ H.T / S
        x = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred

        levels[i]     = x[0]
        velocities[i] = x[1]
        residuals[i]  = z - x[0]

    return pd.Series(levels, index=prices.index), \
           pd.Series(velocities, index=prices.index), \
           pd.Series(residuals, index=prices.index)


def ou_theta_rolling(log_returns, window=30):
    """
    Fit AR(1) on rolling 30 log-returns: r[t] = theta * r[t-1] + eps
    Return theta (AR coefficient). Rolling, no lookahead.
    Uses closed-form OLS: theta = cov(x,y)/var(x) to avoid SVD issues.
    """
    r = log_returns.values
    n = len(r)
    thetas = np.full(n, np.nan)
    for i in range(window, n):
        y = r[i-window+1:i+1]
        x = r[i-window:i]
        vx = np.var(x)
        if vx < 1e-20:
            continue
        # Closed-form OLS slope = cov(x,y)/var(x)
        thetas[i] = np.cov(x, y, ddof=0)[0, 1] / vx
    return pd.Series(thetas, index=log_returns.index)


def hurst_rs_rolling(log_returns, window=30):
    """
    R/S Hurst exponent on rolling 30 log-returns.
    """
    r = log_returns.values
    n = len(r)
    hursts = np.full(n, np.nan)
    for i in range(window, n):
        seg = r[i-window:i]
        # R/S
        mean_seg = np.mean(seg)
        devs = np.cumsum(seg - mean_seg)
        R = devs.max() - devs.min()
        S = np.std(seg, ddof=1)
        if S < 1e-12:
            continue
        hursts[i] = np.log(R / S) / np.log(window)
    return pd.Series(hursts, index=log_returns.index)


def autocorr_rolling(log_returns, lag=1, window=30):
    """Rolling lag-1 autocorrelation of log returns."""
    return log_returns.rolling(window).apply(
        lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else np.nan,
        raw=True
    )


def pc1_rsi_series(rsi_1h, rsi_4h_on_1h, rsi_daily_on_1h):
    """
    First PC of [RSI_1h, RSI_4h, RSI_daily] approximated as average standardized RSI.
    Each is z-scored rolling 30 bars, then averaged.
    """
    def roll_z(s, w=30):
        mu = s.rolling(w).mean()
        sd = s.rolling(w).std()
        return (s - mu) / sd.replace(0, np.nan)

    z1h    = roll_z(rsi_1h)
    z4h    = roll_z(rsi_4h_on_1h)
    zdaily = roll_z(rsi_daily_on_1h)
    return ((z1h + z4h + zdaily) / 3.0).rename("pc1_rsi")


def donchian_pos_series(c, n=20):
    """(close - 20h low) / (20h high - 20h low)"""
    lo = c.rolling(n).min()
    hi = c.rolling(n).max()
    rng = (hi - lo).replace(0, np.nan)
    return (c - lo) / rng

# ─────────────────────────── Build dataset ────────────────────────────────────

ALL_FEATURES_BASE = [
    "ema50_dist", "stoch_k_4h", "rsi_14", "chg_4h_atr", "rsi_4h",
    "macd_hist_1h", "vwap_distance_pct", "bb_pct", "vwap_stretch_score",
    "rvol_1h", "composite_rev", "composite_trend",
]
NEW_FEATURES = [
    "kalman_velocity", "kalman_residual", "ou_theta", "hurst_exponent",
    "autocorr_1h", "pc1_rsi", "donchian_pos", "chg_4h",
]
ALL_FEATURE_CANDIDATES = ALL_FEATURES_BASE + NEW_FEATURES


def build_dataset(df1h, df4h):
    print("Computing features...")
    c1h = df1h["close"]
    c4h = df4h["close"]
    log_r = np.log(c1h / c1h.shift(1))

    # ── BTC-style base features ────────────────────────────────────────────────
    rsi_1h_s   = rsi_series(c1h, 14)
    rsi_4h_s   = rsi_series(c4h, 14)
    rsi_daily_s = rsi_series(c1h.resample("1D").last().ffill(), 14).reindex(c1h.index, method="ffill")

    vd_pct, vwap_stretch = daily_vwap_dist_series(df1h)

    ind1h = pd.DataFrame({
        "rsi_14":            rsi_1h_s,
        "macd_hist_1h":      macd_hist_series(c1h),
        "bb_pct":            bb_pct_series(c1h),
        "ema50_dist":        ema50_dist_series(c1h),
        "rvol_1h":           df1h["volume"] / df1h["volume"].rolling(24).mean().replace(0, np.nan),
        "vwap_distance_pct": vd_pct / 100.0,
        "vwap_stretch_score": vwap_stretch,
    }, index=df1h.index)

    ind4h = pd.DataFrame({
        "stoch_k_4h": stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
        "rsi_4h":     rsi_4h_s,
        "chg_4h_atr": chg_4h_atr_series(df4h),
        "chg_4h":     (c4h - c4h.shift(5)) / c4h.shift(5).replace(0, np.nan) * 100,
    }, index=df4h.index)

    # ── Composite signals ──────────────────────────────────────────────────────
    cal_path = ROOT / "composite_calibration_sol.json"
    cal = None
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        print(f"  Loaded SOL calibration ({len(cal)} entries)")
    trend_s, rev_s, pup_s = compute_composite_signals(df1h, df4h, cal)

    # ── Kalman filter ──────────────────────────────────────────────────────────
    print("  Computing Kalman filter...")
    _, kv_s, kr_s = kalman_filter(c1h)

    # ── OU theta (AR(1) rolling 30) ────────────────────────────────────────────
    print("  Computing OU theta (rolling AR1)...")
    ou_s = ou_theta_rolling(log_r, 30)

    # ── Hurst exponent (R/S rolling 30) ───────────────────────────────────────
    print("  Computing Hurst exponent...")
    hurst_s = hurst_rs_rolling(log_r, 30)

    # ── Autocorrelation lag-1 ─────────────────────────────────────────────────
    print("  Computing autocorrelation...")
    ac_s = autocorr_rolling(log_r, lag=1, window=30)

    # ── PC1 RSI ───────────────────────────────────────────────────────────────
    # Reindex rsi_4h onto 1h grid using ffill
    rsi_4h_on_1h = rsi_4h_s.reindex(c1h.index, method="ffill")
    pc1_s = pc1_rsi_series(rsi_1h_s, rsi_4h_on_1h, rsi_daily_s)

    # ── Donchian position ─────────────────────────────────────────────────────
    donch_s = donchian_pos_series(c1h, 20).rename("donchian_pos")

    # ── Label: close[t+1] > close[t] ──────────────────────────────────────────
    df = df1h[["close"]].copy()
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["label"])

    # ── Merge 1h indicators ────────────────────────────────────────────────────
    df = df.join(ind1h, how="left")
    df["composite_trend"] = trend_s.reindex(df.index)
    df["composite_rev"]   = rev_s.reindex(df.index)
    df["composite_p_up"]  = pup_s.reindex(df.index)

    # Kalman / new signals on 1h index
    df["kalman_velocity"] = kv_s.reindex(df.index)
    df["kalman_residual"] = kr_s.reindex(df.index)
    df["ou_theta"]        = ou_s.reindex(df.index)
    df["hurst_exponent"]  = hurst_s.reindex(df.index)
    df["autocorr_1h"]     = ac_s.reindex(df.index)
    df["pc1_rsi"]         = pc1_s.reindex(df.index)
    df["donchian_pos"]    = donch_s.reindex(df.index)

    # ── Merge 4h indicators via merge_asof ────────────────────────────────────
    ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})
    df_r    = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    df_r    = pd.merge_asof(df_r.sort_values("ts"), ind4h_r.sort_values("ts"),
                             on="ts", direction="backward")
    df = df_r.set_index("ts")

    df = df.sort_index()
    print(f"Dataset: {len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────── IC Analysis ──────────────────────────────────────

def ic_analysis(df, feature_candidates):
    """Compute IC (Spearman) per feature per year. Returns DataFrame."""
    df = df.copy()
    df["year"] = df.index.year

    years = sorted(df["year"].unique())
    results = []

    for feat in feature_candidates:
        if feat not in df.columns:
            continue
        row = {"feature": feat}
        signs = []
        for yr in years:
            sub = df[df["year"] == yr][[feat, "label"]].dropna()
            n = len(sub)
            if n < 50:
                row[f"IC_{yr}"] = np.nan
                row[f"n_{yr}"]  = n
                continue
            ic, pval = stats.spearmanr(sub[feat], sub["label"])
            row[f"IC_{yr}"] = round(ic, 4)
            row[f"n_{yr}"]  = n
            signs.append(np.sign(ic))
        results.append(row)

    ic_df = pd.DataFrame(results).set_index("feature")
    return ic_df, years


# ─────────────────────────── Feature selection ────────────────────────────────

def select_features(ic_df, years, ic_thresh=0.02):
    """
    Select features where |IC| > ic_thresh AND same sign in >= 2 years.
    """
    selected = []
    for feat in ic_df.index:
        ics = [ic_df.loc[feat, f"IC_{yr}"] for yr in years if f"IC_{yr}" in ic_df.columns]
        valid_ics = [ic for ic in ics if not np.isnan(ic)]
        if len(valid_ics) < 2:
            continue
        pos = sum(1 for ic in valid_ics if ic > ic_thresh)
        neg = sum(1 for ic in valid_ics if ic < -ic_thresh)
        # Same direction in at least 2 years with |IC| > threshold
        if pos >= 2 or neg >= 2:
            selected.append(feat)
    return selected


# ─────────────────────────── Training ─────────────────────────────────────────

def train_model(df, features):
    n      = len(df)
    n_test = max(int(n * 0.20), 200)
    n_val  = max(int(n * 0.20), 200)
    n_tr   = n - n_val - n_test

    tr = df.iloc[:n_tr]
    va = df.iloc[n_tr:n_tr + n_val]
    te = df.iloc[n_tr + n_val:]

    print(f"\nTime split:")
    print(f"  Train: {len(tr):,}  ({tr.index[0].date()} → {tr.index[-1].date()})")
    print(f"  Val:   {len(va):,}  ({va.index[0].date()} → {va.index[-1].date()})")
    print(f"  Test:  {len(te):,}  ({te.index[0].date()} → {te.index[-1].date()})")

    label_rate = tr["label"].mean()
    print(f"  Label rate (train): {label_rate:.3f}")

    # Coverage check
    print(f"\nFeature coverage on train set:")
    for f in features:
        cov = tr[f].notna().mean() * 100
        print(f"  {f:<28} {cov:5.1f}%")

    X_tr = tr[features].values.astype(float)
    y_tr = tr["label"].values.astype(int)
    X_va = va[features].values.astype(float)
    y_va = va["label"].values.astype(int)
    X_te = te[features].values.astype(float)
    y_te = te["label"].values.astype(int)

    # Recency weighting
    t_vals  = np.arange(len(tr), dtype=float)
    weights = np.exp(1.5 * t_vals / t_vals[-1])

    # Base LightGBM
    base_clf = lgb.LGBMClassifier(
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
    base_clf.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    # Uncalibrated AUC
    p_va_raw = base_clf.predict_proba(X_va)[:, 1]
    p_te_raw = base_clf.predict_proba(X_te)[:, 1]
    auc_val_raw  = roc_auc_score(y_va, p_va_raw)
    auc_test_raw = roc_auc_score(y_te, p_te_raw)
    print(f"\nBase LightGBM (uncalibrated):")
    print(f"  Val AUC:  {auc_val_raw:.4f}  (best iter: {base_clf.best_iteration_})")
    print(f"  Test AUC: {auc_test_raw:.4f}")

    # Platt calibration on val set
    print("\nApplying Platt calibration (CalibratedClassifierCV, cv=5, sigmoid)...")
    # Re-train base clf without early stopping for calibration wrapper
    base_for_cal = lgb.LGBMClassifier(
        n_estimators=base_clf.best_iteration_ or 300,
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
    cal_clf = CalibratedClassifierCV(base_for_cal, cv=5, method="sigmoid")
    cal_clf.fit(X_tr, y_tr, **{"sample_weight": weights})

    p_va_cal = cal_clf.predict_proba(X_va)[:, 1]
    p_te_cal = cal_clf.predict_proba(X_te)[:, 1]
    auc_val_cal  = roc_auc_score(y_va, p_va_cal)
    auc_test_cal = roc_auc_score(y_te, p_te_cal)
    print(f"Calibrated model:")
    print(f"  Val AUC:  {auc_val_cal:.4f}")
    print(f"  Test AUC: {auc_test_cal:.4f}")

    # Use calibrated if AUC not worse than raw; else use raw base
    if auc_val_cal >= auc_val_raw - 0.005:
        final_clf = cal_clf
        auc_val   = auc_val_cal
        auc_test  = auc_test_cal
        print("  -> Using calibrated model")
    else:
        final_clf = base_clf
        auc_val   = auc_val_raw
        auc_test  = auc_test_raw
        print("  -> Using uncalibrated base model (calibration degraded AUC)")

    # Feature importance from base model
    print(f"\nFeature importance (gain):")
    gain  = base_clf.booster_.feature_importance("gain")
    total = gain.sum() or 1
    for nm, g in sorted(zip(features, gain), key=lambda x: -x[1]):
        bar = "█" * int(g / total * 40)
        print(f"  {nm:<30} {g/total*100:5.1f}%  {bar}")

    # Calibration check on test
    print(f"\nCalibration check (test set):")
    if final_clf is cal_clf:
        p_check = p_te_cal
    else:
        p_check = p_te_raw
    bins = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_check >= lo) & (p_check < hi)
        n_b  = mask.sum()
        if n_b == 0:
            continue
        wr = y_te[mask].mean()
        print(f"  p_up [{lo:.2f},{hi:.2f})  n={n_b:4d}  actual_WR={wr:.3f}")

    return final_clf, auc_val, auc_test


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  SOL p_up v2 (new) — expanded feature set + IC selection")
    print("=" * 68)

    START = "2024-06-01"
    END   = "2026-06-23"

    print(f"\nLoading SOL data ({START} → {END})...")
    df1h = load_or_fetch("SOLUSDT", "1h", START, END)
    df4h = load_or_fetch("SOLUSDT", "4h", START, END)

    df = build_dataset(df1h, df4h)

    # ─── IC Analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("IC Analysis (Spearman correlation with next-1h direction)")
    print("=" * 68)

    ic_df, years = ic_analysis(df, ALL_FEATURE_CANDIDATES)

    # Print IC table
    yr_cols = [f"IC_{y}" for y in years] + [f"n_{y}" for y in years]
    print(f"\n{'Feature':<30} " + " ".join(f"{c:<12}" for c in yr_cols))
    print("-" * (30 + 14 * len(yr_cols)))
    for feat in ic_df.index:
        row_str = f"{feat:<30} "
        for c in yr_cols:
            val = ic_df.loc[feat, c] if c in ic_df.columns else np.nan
            if "IC_" in c:
                row_str += f"{val:>+.4f}     " if not np.isnan(val) else "   NaN       "
            else:
                row_str += f"{int(val) if not np.isnan(val) else 0:>6d}       "
        print(row_str)

    # ─── Feature selection ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Feature Selection: |IC| > 0.02, stable sign >= 2 years")
    print("=" * 68)

    selected = select_features(ic_df, years, ic_thresh=0.02)
    print(f"\nSelected {len(selected)} features:")
    for f in selected:
        ics = {yr: ic_df.loc[f, f"IC_{yr}"] for yr in years if f"IC_{yr}" in ic_df.columns}
        print(f"  {f:<30} " + " ".join(f"IC_{yr}={v:+.4f}" for yr, v in ics.items()))

    # Always include key base features that may just barely miss threshold
    must_include = ["ema50_dist", "rsi_14", "composite_rev", "composite_trend",
                    "stoch_k_4h", "rsi_4h", "macd_hist_1h"]
    for f in must_include:
        if f not in selected and f in df.columns:
            # Check if it has at least some IC signal (any year > 0.01)
            ics = [ic_df.loc[f, f"IC_{yr}"] for yr in years
                   if f"IC_{yr}" in ic_df.columns and not np.isnan(ic_df.loc[f, f"IC_{yr}"])]
            if ics and max(abs(ic) for ic in ics) > 0.01:
                selected.append(f)
                print(f"  [added back] {f}")

    # Remove features with very low coverage
    final_features = []
    for f in selected:
        if f in df.columns:
            cov = df[f].notna().mean()
            if cov > 0.50:
                final_features.append(f)
            else:
                print(f"  [dropped low coverage {cov:.1%}] {f}")

    print(f"\nFinal feature set ({len(final_features)} features):")
    for i, f in enumerate(final_features):
        print(f"  {i+1:2d}. {f}")

    # ─── Train model ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Training LightGBM + Platt Calibration")
    print("=" * 68)

    clf, auc_val, auc_test = train_model(df, final_features)

    # ─── Save model ───────────────────────────────────────────────────────
    out_path = OUT_DIR / "sol_p_up_v2_new.pkl"
    payload = {
        "clf":      clf,
        "features": final_features,
        "auc_val":  auc_val,
        "auc_test": auc_test,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n{'=' * 68}")
    print(f"Saved → {out_path}")
    print(f"Val AUC:  {auc_val:.4f}")
    print(f"Test AUC: {auc_test:.4f}")

    # Compare with existing model
    existing_path = OUT_DIR / "sol_p_up_v2.pkl"
    if existing_path.exists():
        with open(existing_path, "rb") as f:
            existing = pickle.load(f)
        print(f"\nComparison with existing sol_p_up_v2.pkl:")
        print(f"  Existing Val AUC:  {existing.get('auc_val', 'N/A'):.4f}")
        print(f"  Existing Test AUC: {existing.get('auc_test', 'N/A'):.4f}")
        print(f"  New Val AUC:       {auc_val:.4f}")
        print(f"  New Test AUC:      {auc_test:.4f}")

    print(f"\nIC Table Summary:")
    print(ic_df.to_string())

    return {
        "ic_df": ic_df,
        "selected_features": final_features,
        "auc_val": auc_val,
        "auc_test": auc_test,
    }


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
