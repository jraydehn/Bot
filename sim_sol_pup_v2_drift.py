#!/usr/bin/env python3
"""
sim_sol_pup_v2_drift.py

Part 1: IC of sol_p_up_v2 vs SOL Kalshi trade outcomes
Part 2: k_no sweep (0, 0.05, 0.10, 0.20)

Usage: python3 sim_sol_pup_v2_drift.py
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT     = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
REFORM   = ROOT / "reform_results"

# ─── Load data ────────────────────────────────────────────────────────────────

print("Loading data...")

# Scan archive
arc = pd.read_csv(RESULTS / "sol_scan_archive.csv")
arc["close_ts"] = pd.to_datetime(arc["close_ts"], utc=True)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True)
print(f"  Scan archive: {len(arc):,} rows  |  unique close_ts: {arc['close_ts'].nunique():,}")

# 1h and 4h price data
df1h = pd.read_parquet(DATA / "binanceus_SOLUSDT_1h_2024-01-01_2026-06-24.parquet")
df4h = pd.read_parquet(DATA / "binanceus_SOLUSDT_4h_2024-01-01_2026-06-24.parquet")
if df1h.index.tz is None:
    df1h.index = df1h.index.tz_localize("UTC")
if df4h.index.tz is None:
    df4h.index = df4h.index.tz_localize("UTC")
print(f"  df1h: {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
print(f"  df4h: {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

# Model
with open(REFORM / "sol_p_up_v2_new.pkl", "rb") as f:
    payload = pickle.load(f)
clf      = payload["clf"]
features = payload["features"]
print(f"  Model features ({len(features)}): {features}")

# ─── Feature computation helpers (same as train_sol_p_up_v2_new.py) ───────────

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

def daily_vwap_dist_series(df1h_in):
    tp  = (df1h_in["high"] + df1h_in["low"] + df1h_in["close"]) / 3
    vol = df1h_in["volume"]
    day = df1h_in.index.date
    df_tmp = pd.DataFrame({"tp": tp, "vol": vol, "day": day}, index=df1h_in.index)
    df_tmp["cum_tpv"] = df_tmp.groupby("day")["tp"].transform(
        lambda x: (x * df_tmp.loc[x.index, "vol"]).cumsum()
    )
    df_tmp["cum_vol"] = df_tmp.groupby("day")["vol"].transform("cumsum")
    vwap = df_tmp["cum_tpv"] / df_tmp["cum_vol"].replace(0, np.nan)
    dist = (df1h_in["close"] - vwap) / vwap.replace(0, np.nan)
    df_tmp["day_std"] = df_tmp.groupby("day")["tp"].transform("std")
    stretch = pd.cut(
        dist / (df_tmp["day_std"] / vwap.replace(0, np.nan)).replace(0, np.nan),
        bins=[-np.inf, -2, -1, 1, 2, np.inf],
        labels=[2, 1, 0, -1, -2],
    ).astype(float)
    return dist * 100, stretch

def kalman_filter(prices):
    n = len(prices)
    levels     = np.full(n, np.nan)
    velocities = np.full(n, np.nan)
    residuals  = np.full(n, np.nan)
    if n < 2:
        return (pd.Series(levels, index=prices.index),
                pd.Series(velocities, index=prices.index),
                pd.Series(residuals, index=prices.index))
    x = np.array([prices.iloc[0], prices.iloc[1] - prices.iloc[0]])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-5 * np.eye(2)
    R = np.array([[0.01]])
    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = prices.iloc[i]
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T + R)[0, 0]
        K = P_pred @ H.T / S
        x = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred
        levels[i]     = x[0]
        velocities[i] = x[1]
        residuals[i]  = z - x[0]
    return (pd.Series(levels, index=prices.index),
            pd.Series(velocities, index=prices.index),
            pd.Series(residuals, index=prices.index))

def ou_theta_rolling(log_returns, window=30):
    r = log_returns.values
    n = len(r)
    thetas = np.full(n, np.nan)
    for i in range(window, n):
        y = r[i-window+1:i+1]
        x = r[i-window:i]
        vx = np.var(x)
        if vx < 1e-20:
            continue
        thetas[i] = np.cov(x, y, ddof=0)[0, 1] / vx
    return pd.Series(thetas, index=log_returns.index)

def hurst_rs_rolling(log_returns, window=30):
    r = log_returns.values
    n = len(r)
    hursts = np.full(n, np.nan)
    for i in range(window, n):
        seg = r[i-window:i]
        mean_seg = np.mean(seg)
        devs = np.cumsum(seg - mean_seg)
        R = devs.max() - devs.min()
        S = np.std(seg, ddof=1)
        if S < 1e-12:
            continue
        hursts[i] = np.log(R / S) / np.log(window)
    return pd.Series(hursts, index=log_returns.index)

def autocorr_rolling(log_returns, lag=1, window=30):
    return log_returns.rolling(window).apply(
        lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else np.nan,
        raw=True
    )

def pc1_rsi_series(rsi_1h, rsi_4h_on_1h, rsi_daily_on_1h):
    def roll_z(s, w=30):
        mu = s.rolling(w).mean()
        sd = s.rolling(w).std()
        return (s - mu) / sd.replace(0, np.nan)
    z1h    = roll_z(rsi_1h)
    z4h    = roll_z(rsi_4h_on_1h)
    zdaily = roll_z(rsi_daily_on_1h)
    return ((z1h + z4h + zdaily) / 3.0).rename("pc1_rsi")

def donchian_pos_series(c, n=20):
    lo = c.rolling(n).min()
    hi = c.rolling(n).max()
    rng = (hi - lo).replace(0, np.nan)
    return (c - lo) / rng

def compute_composite_signals(df1h_in, df4h_in, cal):
    c1h = df1h_in["close"]
    c4h = df4h_in["close"]
    rsi4    = rsi_series(c4h, 14)
    macd4   = _ema(c4h, 12) - _ema(c4h, 26)
    sig4    = macd4.ewm(span=9, adjust=False).mean()
    bb_mid4 = c4h.rolling(20).mean()
    bb_std4 = c4h.rolling(20).std()
    bb_lo4  = bb_mid4 - 2 * bb_std4
    bb_hi4  = bb_mid4 + 2 * bb_std4
    bb_pct4 = (c4h - bb_lo4) / (bb_hi4 - bb_lo4).replace(0, np.nan)
    sk4     = stoch_k_series(df4h_in["high"], df4h_in["low"], c4h, 14)
    wr4     = -100 * (df4h_in["high"].rolling(14).max() - c4h) / \
              (df4h_in["high"].rolling(14).max() - df4h_in["low"].rolling(14).min()).replace(0, np.nan)
    vol_ma4 = df4h_in["volume"].rolling(20).mean()
    vol_r4  = df4h_in["volume"] / vol_ma4.replace(0, np.nan)
    trend_4h = pd.Series(0.0, index=df4h_in.index)
    trend_4h += (rsi4 > 55).astype(float) - (rsi4 < 45).astype(float)
    trend_4h += (macd4 > sig4).astype(float) - (macd4 <= sig4).astype(float)
    trend_4h += (bb_pct4 > 0.80).astype(float) - (bb_pct4 < 0.20).astype(float)
    trend_4h += (sk4 > 80).astype(float) - (sk4 < 20).astype(float)
    trend_4h += (wr4 > -20).astype(float) - (wr4 < -80).astype(float)
    trend_4h += ((vol_r4 > 1.5) & (c4h > c4h.shift(1))).astype(float) - \
                ((vol_r4 > 1.5) & (c4h < c4h.shift(1))).astype(float)
    trend_4h = trend_4h.clip(-6, 6)
    rsi1  = rsi_series(c1h, 14)
    sk1   = stoch_k_series(df1h_in["high"], df1h_in["low"], c1h, 14)
    vd, _ = daily_vwap_dist_series(df1h_in)
    log_r = np.log(c1h / c1h.shift(1))
    z_std = log_r.rolling(24).std()
    z_sc  = log_r / z_std.replace(0, np.nan)
    rev_1h = pd.Series(0.0, index=df1h_in.index)
    rev_1h += 2 * (rsi1 < 30).astype(float) + (rsi1 < 40).astype(float)
    rev_1h -= 2 * (rsi1 > 70).astype(float) + (rsi1 > 60).astype(float)
    rev_1h += 2 * (sk1 < 10).astype(float) + (sk1 < 20).astype(float)
    rev_1h -= 2 * (sk1 > 90).astype(float) + (sk1 > 80).astype(float)
    rev_1h += 2 * (vd < -1.5).astype(float) + (vd < -0.5).astype(float)
    rev_1h -= 2 * (vd > 1.5).astype(float) + (vd > 0.5).astype(float)
    rev_1h += 2 * (z_sc < -2.0).astype(float) + (z_sc < -1.5).astype(float)
    rev_1h -= 2 * (z_sc > 2.0).astype(float) + (z_sc > 1.5).astype(float)
    rev_1h  = rev_1h.clip(-8, 8)
    trend_1h = trend_4h.reindex(df1h_in.index, method="ffill")
    if cal:
        def lookup(t, r):
            k = f"{int(round(t))}_{int(round(r))}"
            e = cal.get(k)
            return e["p_yes"] if e and e.get("n", 0) >= 5 else 0.504
        p_up_ser = pd.Series(
            [lookup(t, r) for t, r in zip(trend_1h, rev_1h)],
            index=df1h_in.index,
        )
    else:
        p_up_ser = pd.Series(0.504, index=df1h_in.index)
    return (
        trend_1h.rename("composite_trend"),
        rev_1h.rename("composite_rev"),
        p_up_ser.rename("composite_p_up"),
    )

# ─── Build full feature matrix on 1h grid ──────────────────────────────────────

print("\nBuilding full feature matrix...")
c1h = df1h["close"]
c4h = df4h["close"]
log_r = np.log(c1h / c1h.shift(1))

rsi_1h_s    = rsi_series(c1h, 14)
rsi_4h_s    = rsi_series(c4h, 14)
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

# Load SOL calibration if present
cal_path = ROOT / "composite_calibration_sol.json"
cal = None
if cal_path.exists():
    with open(cal_path) as f:
        cal = json.load(f)
    print(f"  Loaded SOL calibration ({len(cal)} entries)")

trend_s, rev_s, pup_s = compute_composite_signals(df1h, df4h, cal)

print("  Computing Kalman filter...")
_, kv_s, kr_s = kalman_filter(c1h)

print("  Computing OU theta...")
ou_s = ou_theta_rolling(log_r, 30)

print("  Computing Hurst exponent...")
hurst_s = hurst_rs_rolling(log_r, 30)

print("  Computing autocorrelation...")
ac_s = autocorr_rolling(log_r, lag=1, window=30)

rsi_4h_on_1h = rsi_4h_s.reindex(c1h.index, method="ffill")
pc1_s = pc1_rsi_series(rsi_1h_s, rsi_4h_on_1h, rsi_daily_s)
donch_s = donchian_pos_series(c1h, 20)

# Assemble on 1h index
feat_df = ind1h.copy()
feat_df["composite_trend"] = trend_s.reindex(feat_df.index)
feat_df["composite_rev"]   = rev_s.reindex(feat_df.index)
feat_df["kalman_velocity"] = kv_s.reindex(feat_df.index)
feat_df["kalman_residual"] = kr_s.reindex(feat_df.index)
feat_df["ou_theta"]        = ou_s.reindex(feat_df.index)
feat_df["hurst_exponent"]  = hurst_s.reindex(feat_df.index)
feat_df["autocorr_1h"]     = ac_s.reindex(feat_df.index)
feat_df["pc1_rsi"]         = pc1_s.reindex(feat_df.index)
feat_df["donchian_pos"]    = donch_s.reindex(feat_df.index)

# Merge 4h via asof
ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})
feat_r  = feat_df.reset_index().rename(columns={feat_df.index.name or "index": "ts"})
feat_r  = pd.merge_asof(feat_r.sort_values("ts"), ind4h_r.sort_values("ts"),
                         on="ts", direction="backward")
feat_df = feat_r.set_index("ts").sort_index()

print(f"  Feature matrix: {len(feat_df):,} rows")

# ─── Part 1: Compute p_up_v2 for each unique close_ts ─────────────────────────

print("\n" + "="*68)
print("PART 1: p_up_v2 IC vs SOL Kalshi outcomes")
print("="*68)

# For each unique close_ts, find the last completed 1h bar before it
# (bar whose close_time = close_ts - 1h → bar that OPENED at close_ts - 1h,
#  i.e., the 1h bar whose index (open_time) is close_ts - 2h on Binance
#  because Binance bar index = open_time, close_time = open_time + 1h)
#
# Actually: Binance 1h bar index = open_time.  close_time = open_time + 1h.
# The "last completed bar before close_ts" has close_time = close_ts - 1h
# → open_time = close_ts - 2h.
# But we can also simply do merge_asof with direction="backward" on close_ts - 1h.

unique_ts = arc["close_ts"].unique()
print(f"  Unique close_ts: {len(unique_ts):,}")

# We'll use merge_asof: for each close_ts, find the most recent 1h bar
# whose open_time <= close_ts - 1h (i.e., bar has already closed by close_ts)
lookup_df = pd.DataFrame({"close_ts": pd.to_datetime(unique_ts, utc=True)})
lookup_df["bar_ref"] = lookup_df["close_ts"] - pd.Timedelta(hours=1)

feat_reset = feat_df.reset_index().rename(columns={"ts": "bar_ref"})
lookup_merged = pd.merge_asof(
    lookup_df.sort_values("bar_ref"),
    feat_reset[["bar_ref"] + features].sort_values("bar_ref"),
    on="bar_ref",
    direction="backward"
)

# Score with model
X = lookup_merged[features].values.astype(float)
preds = clf.predict_proba(X)[:, 1]
lookup_merged["p_up_v2_new"] = preds
lookup_merged = lookup_merged.set_index("close_ts")

print(f"  p_up_v2_new: min={preds.min():.3f}  max={preds.max():.3f}  mean={preds.mean():.3f}  std={preds.std():.3f}")
nan_count = np.isnan(preds).sum()
print(f"  NaN predictions: {nan_count}")

# Merge back to arc
arc = arc.merge(
    lookup_merged[["p_up_v2_new"]].reset_index(),
    on="close_ts", how="left"
)
print(f"  Arc rows with p_up_v2_new: {arc['p_up_v2_new'].notna().sum():,} / {len(arc):,}")

# ─── IC Calculations ──────────────────────────────────────────────────────────

# Use one row per unique close_ts (expiry-level analysis)
# Take best NO contract per expiry (deepest OTM = highest offset_pct)
# For outcome: resolved_yes and price_move_pct

# First: expiry-level IC using all rows (multiple contracts per expiry)
arc_valid = arc.dropna(subset=["p_up_v2_new", "resolved_yes", "price_move_pct"])
print(f"\n  Valid rows for IC: {len(arc_valid):,}")

# NO_win = 1 when resolved_yes == 0
arc_valid = arc_valid.copy()
arc_valid["no_win"] = (arc_valid["resolved_yes"] == 0).astype(int)

# IC vs price_move_pct (higher p_up_v2 → price goes up → positive price_move_pct)
ic_price, p_price = stats.spearmanr(arc_valid["p_up_v2_new"], arc_valid["price_move_pct"])
# IC vs NO win (higher p_up_v2 → price up → NO loses → negative IC expected)
ic_no, p_no = stats.spearmanr(arc_valid["p_up_v2_new"], arc_valid["no_win"])

print("\n  ── IC Table (all archive rows with resolution) ──")
print(f"  IC(p_up_v2_new, price_move_pct)  = {ic_price:+.4f}  (p={p_price:.4f})  n={len(arc_valid):,}")
print(f"  IC(p_up_v2_new, NO_win)          = {ic_no:+.4f}  (p={p_no:.4f})  n={len(arc_valid):,}")

# Expiry-level (one row per expiry using deepest OTM contract)
# "deepest OTM NO" = highest strike above spot = highest offset_pct (positive = above spot)
# But archive likely has both ITM and OTM. Let's take the contract with highest offset_pct per expiry.
arc_otm = arc_valid.sort_values("offset_pct", ascending=False).drop_duplicates("close_ts")
print(f"\n  Expiry-level (deepest-OTM per expiry): {len(arc_otm):,} expiries")
ic_price_exp, _ = stats.spearmanr(arc_otm["p_up_v2_new"], arc_otm["price_move_pct"])
ic_no_exp, _    = stats.spearmanr(arc_otm["p_up_v2_new"], arc_otm["no_win"])
print(f"  IC(p_up_v2_new, price_move_pct)  = {ic_price_exp:+.4f}")
print(f"  IC(p_up_v2_new, NO_win)          = {ic_no_exp:+.4f}")

# ─── WR by quintile ───────────────────────────────────────────────────────────

print("\n  ── NO Win Rate by p_up_v2_new Quintile ──")
print("  (Using deepest-OTM contract per expiry)")
arc_otm2 = arc_otm.copy()
arc_otm2["quintile"] = pd.qcut(arc_otm2["p_up_v2_new"], 5, labels=["Q1\n(most bearish)","Q2","Q3","Q4","Q5\n(most bullish)"])

q_stats = arc_otm2.groupby("quintile", observed=True).agg(
    n=("no_win", "count"),
    no_wr=("no_win", "mean"),
    p_up_mean=("p_up_v2_new", "mean"),
    price_move_mean=("price_move_pct", "mean"),
).reset_index()

print(f"\n  {'Quintile':<22} {'n':>5}  {'NO_WR':>7}  {'p_up_mean':>10}  {'price_move_mean':>16}")
print("  " + "-"*65)
for _, row in q_stats.iterrows():
    q_label = str(row["quintile"]).replace("\n", " ")
    print(f"  {q_label:<22} {int(row['n']):>5}  {row['no_wr']:>6.1%}  {row['p_up_mean']:>10.3f}  {row['price_move_mean']:>15.4f}")

# ─── Gate analysis: p_up_v2 >= 0.60 block ──────────────────────────────────────

print("\n  ── Gate Analysis: Block NO when p_up_v2_new >= threshold ──")
print("  (Using deepest-OTM contract per expiry)")

for thresh in [0.55, 0.58, 0.60, 0.62, 0.65]:
    blocked   = arc_otm2[arc_otm2["p_up_v2_new"] >= thresh]
    taken     = arc_otm2[arc_otm2["p_up_v2_new"] < thresh]
    if len(blocked) == 0:
        continue
    w_blocked = (blocked["no_win"] == 1).sum()  # NO wins we'd block (BAD)
    l_blocked = (blocked["no_win"] == 0).sum()  # NO losses we'd block (GOOD)
    wr_blocked = blocked["no_win"].mean()
    wr_taken   = taken["no_win"].mean() if len(taken) > 0 else np.nan
    print(f"  thresh={thresh:.2f}  blocked={len(blocked):3d} (WR={wr_blocked:.1%}  wins_blocked={w_blocked}  losses_blocked={l_blocked})  taken WR={wr_taken:.1%}  n_taken={len(taken)}")

# More detail on >= 0.60
thresh = 0.60
above = arc_otm2[arc_otm2["p_up_v2_new"] >= thresh]
below = arc_otm2[arc_otm2["p_up_v2_new"] < thresh]
print(f"\n  Detail for thresh=0.60:")
print(f"    p_up_v2 >= 0.60: n={len(above)}  NO WR={above['no_win'].mean():.1%}  price_move_mean={above['price_move_pct'].mean():.4f}")
print(f"    p_up_v2 <  0.60: n={len(below)}  NO WR={below['no_win'].mean():.1%}  price_move_mean={below['price_move_pct'].mean():.4f}")

# Below 0.50 (bearish signal → NO should win)
below50 = arc_otm2[arc_otm2["p_up_v2_new"] < 0.50]
print(f"    p_up_v2 <  0.50: n={len(below50)}  NO WR={below50['no_win'].mean():.1%}")

# ─── Part 2: k_no sweep ───────────────────────────────────────────────────────

print("\n" + "="*68)
print("PART 2: k_no sweep (k=0, 0.05, 0.10, 0.20)")
print("="*68)

# Assumptions:
# - Lognormal pricing: z_strike = log(strike/spot) / sigma_tau
# - sigma_tau = sigma_1h * sqrt(tau_min / 60)  (tau_min capped at 60 for NO)
# - z_drift = norm.ppf(p_up_v2) * k_no * sqrt(min(tau_min, 60) / 60)
# - p_model_no = norm.cdf(z_strike - z_drift)
# - edge_no = p_model_no - (1 - p_market)   [i.e., p_model_yes_price side]
#   Actually: NO pays out when price ends below strike.
#   edge_no = p_model_no - (1 - p_market)
#
# We need sigma_1h for SOL. Use rolling 24h annualized vol → hourly sigma.
# sigma_1h = std(log returns) over rolling 24 bars.

# Compute hourly sigma from 1h data
c1h_s = df1h["close"]
log_r_1h = np.log(c1h_s / c1h_s.shift(1))
sigma_1h_ser = log_r_1h.rolling(24).std()  # hourly vol (as fraction)

# Get sigma at each close_ts (use the bar BEFORE close_ts, same as p_up_v2)
sigma_lookup = sigma_1h_ser.reset_index().rename(columns={"ts": "bar_ref", "close": "sigma_1h"})
sigma_lookup.columns = ["bar_ref", "sigma_1h"]

# For the sweep, use deepest-OTM NO contract per expiry
# (which the runner would most likely select given positive offset_pct)
# We'll take the NO contracts: those with positive offset_pct (strike above spot = OTM for NO)
# Actually in Kalshi NO = "price ends below strike" → OTM NO = strike well above current spot
# so high offset_pct = deeper OTM NO (safer for NO)

arc_valid2 = arc.dropna(subset=["p_up_v2_new", "resolved_yes"]).copy()
arc_valid2["no_win"] = (arc_valid2["resolved_yes"] == 0).astype(int)

# Take deepest OTM NO contract per expiry (highest offset_pct)
arc_sweep = arc_valid2.sort_values("offset_pct", ascending=False).drop_duplicates("close_ts").copy()
print(f"\n  Expiries for k_no sweep: {len(arc_sweep):,}")
print(f"  Overall NO WR: {arc_sweep['no_win'].mean():.1%}")
print(f"  Mean offset_pct: {arc_sweep['offset_pct'].mean():.4f}")
print(f"  Mean p_market: {arc_sweep['p_market'].mean():.3f}")

# Merge sigma
arc_sweep["bar_ref"] = arc_sweep["close_ts"] - pd.Timedelta(hours=1)
arc_sweep = pd.merge_asof(
    arc_sweep.sort_values("bar_ref"),
    sigma_lookup.sort_values("bar_ref"),
    on="bar_ref",
    direction="backward"
)

# Fill missing sigma with median
sigma_med = arc_sweep["sigma_1h"].median()
arc_sweep["sigma_1h"] = arc_sweep["sigma_1h"].fillna(sigma_med)
print(f"  Median hourly sigma: {sigma_med:.5f} ({sigma_med*100:.4f}%)")

EDGE_THRESHOLD = 0.04
K_NO_VALUES = [0.0, 0.05, 0.10, 0.20]

# Simulate for each k_no
results = {}

for k_no in K_NO_VALUES:
    row_results = []
    for _, row in arc_sweep.iterrows():
        spot   = row["spot"]
        strike = row["strike"]
        p_mkt  = row["p_market"]
        tau    = row["tau_minutes"]
        sigma  = row["sigma_1h"]
        p_up   = row["p_up_v2_new"]
        no_win = row["no_win"]

        # sigma_tau: sigma per bar scaled to tau
        tau_eff = min(tau, 60.0)  # cap at 60 min for NO as in BTC model
        sigma_tau = sigma * np.sqrt(tau_eff / 60.0)
        if sigma_tau < 1e-8:
            sigma_tau = 0.005  # fallback

        # z_strike: log(strike/spot) / sigma_tau
        if spot <= 0 or strike <= 0:
            continue
        z_strike = np.log(strike / spot) / sigma_tau

        # z_drift
        if k_no == 0.0 or np.isnan(p_up):
            z_drift = 0.0
        else:
            tau_scale = np.sqrt(min(tau, 60.0) / 60.0)
            z_drift = norm.ppf(np.clip(p_up, 0.001, 0.999)) * k_no * tau_scale

        # p_model_no = P(price ends below strike) = norm.cdf(z_strike - z_drift)
        p_model_no = norm.cdf(z_strike - z_drift)

        # edge_no = p_model_no - (1 - p_market)
        edge_no = p_model_no - (1.0 - p_mkt)

        # Would we take this bet?
        take = edge_no >= EDGE_THRESHOLD
        row_results.append({
            "close_ts": row["close_ts"],
            "no_win": no_win,
            "p_up": p_up,
            "p_market": p_mkt,
            "offset_pct": row["offset_pct"],
            "p_model_no": p_model_no,
            "edge_no": edge_no,
            "take": take,
        })
    results[k_no] = pd.DataFrame(row_results)

print("\n  ── k_no Sweep Results ──")
print(f"  {'k_no':<8} {'n_taken':>8} {'NO_WR_taken':>12} {'n_blocked':>10} {'WR_blocked':>11} {'WR_pass':>9}  notes")
print("  " + "-"*75)

k0_taken = results[0.0][results[0.0]["take"]]

for k_no in K_NO_VALUES:
    df_k = results[k_no]
    taken   = df_k[df_k["take"]]
    blocked = df_k[~df_k["take"]]

    n_taken    = len(taken)
    wr_taken   = taken["no_win"].mean() if n_taken > 0 else np.nan
    n_blocked  = len(blocked)
    wr_blocked = blocked["no_win"].mean() if n_blocked > 0 else np.nan

    # How many are NEWLY blocked vs k=0 baseline?
    if k_no == 0.0:
        newly_blocked = 0
    else:
        # rows taken at k=0 but not at k_no
        taken_k0_ids = set(k0_taken["close_ts"].astype(str))
        taken_kn_ids = set(taken["close_ts"].astype(str))
        newly_blocked_set = taken_k0_ids - taken_kn_ids
        newly_blocked = len(newly_blocked_set)

    note = f"  newly_blocked={newly_blocked}" if k_no > 0 else ""
    print(f"  {k_no:<8} {n_taken:>8} {wr_taken:>11.1%} {n_blocked:>10} {wr_blocked:>10.1%} {wr_taken:>9.1%} {note}")

# ─── Detailed impact: which bets does drift filter? ──────────────────────────

print("\n  ── Impact of k_no=0.10: Newly-Blocked Bets ──")
df_k0  = results[0.0]
df_k10 = results[0.10]

# Bets taken at k=0 but blocked at k=0.10
taken_k0  = df_k0[df_k0["take"]].set_index("close_ts")
taken_k10 = df_k10[df_k10["take"]].set_index("close_ts")
blocked_by_drift = taken_k0[~taken_k0.index.isin(taken_k10.index)]
unaffected       = taken_k0[ taken_k0.index.isin(taken_k10.index)]

print(f"  Taken at k=0: {len(taken_k0)}")
print(f"  Taken at k=0.10: {len(taken_k10)}")
print(f"  Newly blocked by drift: {len(blocked_by_drift)}")
if len(blocked_by_drift) > 0:
    print(f"    NO WR of blocked: {blocked_by_drift['no_win'].mean():.1%}  (wins lost / losses saved)")
    wins_lost   = (blocked_by_drift["no_win"] == 1).sum()
    losses_saved = (blocked_by_drift["no_win"] == 0).sum()
    print(f"    Wins lost: {wins_lost}  Losses saved: {losses_saved}")
    print(f"    p_up_mean of blocked: {blocked_by_drift['p_up'].mean():.3f}")
print(f"  Unaffected (still taken): {len(unaffected)}")
if len(unaffected) > 0:
    print(f"    NO WR of unaffected: {unaffected['no_win'].mean():.1%}")

# ─── PnL delta simulation ─────────────────────────────────────────────────────

print("\n  ── PnL Delta Simulation ──")
print("  (Assumes $100 flat bet, NO win = +$100*(1/pm - 1), loss = -$100)")
print("  Using deepest-OTM contract per expiry.\n")

BASE_BET = 100.0

def sim_pnl(df_taken):
    """Rough PnL: win → collect (1-pm)/pm * bet; lose → -bet"""
    pnl = 0.0
    for _, r in df_taken.iterrows():
        pm  = r["p_market"]
        # NO cost = (1 - pm) * 100 cents → pay (1-pm) per contract
        # NO win  = 1.0 - (1-pm) per contract at $1 face → gross_win = pm
        # Simplified: bet $100 at NO, win = +100*(pm/(1-pm)), lose = -100
        # Or more simply: win = +100 * pm / (1 - pm)  loss = -100
        # Actually Kalshi: buy NO at price (1-pm)*100¢, pays $1 if win
        # So per $100 risked: win = $100 / (1-pm) - $100 = $100 * pm/(1-pm)
        # But simpler: win = 100 * (1/(1-pm) - 1), loss = -100
        if (1 - pm) < 1e-6:
            continue
        if r["no_win"] == 1:
            pnl += BASE_BET * pm / (1.0 - pm)
        else:
            pnl -= BASE_BET
    return pnl

print(f"  {'k_no':<8} {'n_taken':>8}  {'WR':>7}  {'PnL ($)':>10}  {'PnL_delta vs k=0':>18}")
print("  " + "-"*60)

pnl_k0 = None
for k_no in K_NO_VALUES:
    taken = results[k_no][results[k_no]["take"]]
    pnl   = sim_pnl(taken)
    wr    = taken["no_win"].mean() if len(taken) > 0 else np.nan
    delta = (pnl - pnl_k0) if pnl_k0 is not None else 0.0
    print(f"  {k_no:<8} {len(taken):>8}  {wr:>6.1%}  {pnl:>10.1f}  {delta:>+18.1f}")
    if pnl_k0 is None:
        pnl_k0 = pnl

# ─── Summary stats by p_up bucket ──────────────────────────────────────────────

print("\n  ── NO WR by p_up_v2_new bucket (all expiries) ──")
arc_sweep2 = arc_sweep.copy()
arc_sweep2["p_up_bucket"] = pd.cut(arc_sweep2["p_up_v2_new"],
    bins=[0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0],
    labels=["<0.40","[0.40,0.45)","[0.45,0.50)","[0.50,0.55)","[0.55,0.60)","[0.60,0.65)","≥0.65"])

bkt = arc_sweep2.groupby("p_up_bucket", observed=True).agg(
    n=("no_win","count"),
    no_wr=("no_win","mean"),
    price_move_mean=("price_move_pct","mean"),
).reset_index()
print(f"\n  {'p_up bucket':<15} {'n':>5}  {'NO_WR':>7}  {'price_move_mean':>16}")
print("  " + "-"*48)
for _, row in bkt.iterrows():
    print(f"  {str(row['p_up_bucket']):<15} {int(row['n']):>5}  {row['no_wr']:>6.1%}  {row['price_move_mean']:>15.4f}")

# ─── Final recommendation ─────────────────────────────────────────────────────

print("\n" + "="*68)
print("RECOMMENDATION SUMMARY")
print("="*68)

# Compute key stats for recommendation
k0_n    = len(results[0.0][results[0.0]["take"]])
k10_n   = len(results[0.10][results[0.10]["take"]])
k0_wr   = results[0.0][results[0.0]["take"]]["no_win"].mean()
k10_wr  = results[0.10][results[0.10]["take"]]["no_win"].mean()

drift_blocked = results[0.0][results[0.0]["take"]].set_index("close_ts")
drift_taken10 = results[0.10][results[0.10]["take"]].set_index("close_ts")
newly = drift_blocked[~drift_blocked.index.isin(drift_taken10.index)]
newly_wr = newly["no_win"].mean() if len(newly) > 0 else np.nan

print(f"\n  Baseline (k=0):   n={k0_n}  WR={k0_wr:.1%}")
print(f"  Drift (k=0.10):   n={k10_n}  WR={k10_wr:.1%}")
print(f"  Newly blocked:    n={len(newly)}  WR={newly_wr:.1%}")

print(f"\n  Gate (p_up>=0.60):")
g_above = arc_otm2[arc_otm2["p_up_v2_new"] >= 0.60]
g_below = arc_otm2[arc_otm2["p_up_v2_new"] < 0.60]
print(f"    Block (p_up>=0.60): n={len(g_above)}  NO WR={g_above['no_win'].mean():.1%}")
print(f"    Pass  (p_up< 0.60): n={len(g_below)}  NO WR={g_below['no_win'].mean():.1%}")

print(f"\n  IC summary:")
print(f"    IC(p_up_v2, price_move_pct) = {ic_price:+.4f}")
print(f"    IC(p_up_v2, NO_win)         = {ic_no:+.4f}")
print(f"\n  All done.")
