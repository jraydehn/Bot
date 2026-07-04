#!/usr/bin/env python3
"""S3 — Build the extended (2020-01 -> 2026-07) leak-safe dataset for the
p_up_v2 rebuild. Reuses the EXACT lag-corrected A-feature spec from
scratchpad/fixed_ds.py (the honest 0.535 benchmark), applied to the extended
hist parquets, plus candidate feature groups:

  A  : 16 price/tech features (fixed spec, 4h shifted +3h, expanding vwap std)
  B  : ETH/SOL 1h/4h returns + BTC-alt spreads (now back to 2020/2020-09)
  C  : wider cross-asset (XRP/DOGE/ADA rets, alt-basket, BTC dominance) +
       session-return INTERACTIONS (seasonality alone is dead — dummies never
       enter alone)
  Ft : full-history flow proxy from BinanceUS klines taker-buy volume
  Fcg: CoinGlass point-in-time flow (2026-01-05+; NaN before) — funding, OI,
       liquidations, futures/spot taker CVD
  M  : intra-hour microstructure from 1m bars of hour [T, T+1h) — complete at
       decision time T+1h (requires hist_BTCUSDT_1m.parquet; skipped if absent)
  R  : derived regime features (efficiency ratio, vol-of-vol, donchian,
       30d high/low distance) — rolling stats only, no full-period stats

LEAK RULES: label(T) = direction of bar T+1 close vs bar T close; decision
time = T+1h. Every feature uses only data with timestamp+period <= T+1h.
4h bars shifted +3h before backward merge. All z-scores are rolling (240h).

Output: extended_dataset.parquet + feature_groups.json
"""
import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJ = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(PROJ)); os.chdir(PROJ)
import train_btc_p_up_v2 as T

df1h = pd.read_parquet(HERE / "hist_BTCUSDT_1h.parquet")
df4h = pd.read_parquet(HERE / "hist_BTCUSDT_4h.parquet")
d15 = pd.read_parquet(HERE / "hist_BTCUSDT_15m.parquet")
c1h, c4h = df1h["close"], df4h["close"]

# ── A: 1h indicators (fixed spec) ─────────────────────────────────────────
ind1h = pd.DataFrame({
    "rsi_14": T.rsi_series(c1h, 14),
    "macd_hist_1h": T.macd_hist_series(c1h),
    "bb_pct": T.bb_pct_series(c1h),
    "ema50_dist": T.ema50_dist_series(c1h),
    "rvol_1h": df1h["volume"] / df1h["volume"].rolling(24).mean().replace(0, np.nan),
}, index=df1h.index)
tp = (df1h["high"] + df1h["low"] + df1h["close"]) / 3
day = pd.Series(df1h.index.date, index=df1h.index)
cum_tpv = (tp * df1h["volume"]).groupby(day.values).cumsum()
cum_vol = df1h["volume"].groupby(day.values).cumsum()
vwap = cum_tpv / cum_vol.replace(0, np.nan)
dist = (df1h["close"] - vwap) / vwap.replace(0, np.nan)
ind1h["vwap_distance_pct"] = dist
std_exp = tp.groupby(day.values).expanding().std().reset_index(level=0, drop=True)
std_exp.index = df1h.index
z = dist / (std_exp / vwap.replace(0, np.nan)).replace(0, np.nan)
ind1h["vwap_stretch_score"] = pd.cut(
    z, bins=[-np.inf, -2, -1, 1, 2, np.inf], labels=[2, 1, 0, -1, -2]).astype(float)

# A: 4h indicators, lag-corrected +3h
ind4h = pd.DataFrame({
    "stoch_k_4h": T.stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
    "rsi_4h": T.rsi_series(c4h, 14),
    "chg_4h_atr": T.chg_4h_atr_series(df4h),
}, index=df4h.index)
ind4h.index = ind4h.index + pd.Timedelta(hours=3)

# A: composite trend (lag-corrected) / rev (causal) / p_up LUT
cal = T.load_calibration()
_, rev_s, _ = T.compute_composite_signals(df1h, df4h, cal)
rsi4 = T.rsi_series(c4h, 14)
macd4 = T._ema(c4h, 12) - T._ema(c4h, 26)
sig4 = macd4.ewm(span=9, adjust=False).mean()
bbm = c4h.rolling(20).mean(); bbs = c4h.rolling(20).std()
bbp4 = (c4h - (bbm - 2 * bbs)) / (4 * bbs).replace(0, np.nan)
sk4 = T.stoch_k_series(df4h["high"], df4h["low"], c4h, 14)
wr4 = -100 * (df4h["high"].rolling(14).max() - c4h) / \
      (df4h["high"].rolling(14).max() - df4h["low"].rolling(14).min()).replace(0, np.nan)
vr4 = df4h["volume"] / df4h["volume"].rolling(20).mean().replace(0, np.nan)
trend4 = ((rsi4 > 55).astype(float) - (rsi4 < 45).astype(float)
          + np.where(macd4 > sig4, 1.0, -1.0)
          + (bbp4 > 0.80).astype(float) - (bbp4 < 0.20).astype(float)
          + (sk4 > 80).astype(float) - (sk4 < 20).astype(float)
          + (wr4 > -20).astype(float) - (wr4 < -80).astype(float)
          + ((vr4 > 1.5) & (c4h > c4h.shift(1))).astype(float)
          - ((vr4 > 1.5) & (c4h < c4h.shift(1))).astype(float)).clip(-6, 6)
trend4_lag = pd.Series(trend4.values, index=df4h.index + pd.Timedelta(hours=3))
trend_1h = trend4_lag.reindex(df1h.index.union(trend4_lag.index)).ffill().reindex(df1h.index)

def lut(t, r):
    if t != t or r != r:
        return 0.504
    e = (cal or {}).get(f"{int(round(t))}_{int(round(r))}")
    return float(e["p_yes"]) if e and e.get("n", 0) >= 5 else 0.504
pup_1h = pd.Series([lut(t, r) for t, r in zip(trend_1h, rev_s)], index=df1h.index)

# A: 15m features (bar open <= T)
sk15 = T.stoch_k_series(d15["high"], d15["low"], d15["close"], 14).rename("stoch_k")
e9, e21, e50 = T._ema(d15["close"], 9), T._ema(d15["close"], 21), T._ema(d15["close"], 50)
ema_stack = pd.Series(0.0, index=d15.index)
ema_stack[(e9 > e21) & (e21 > e50) & (d15["close"] > e9)] = 1
ema_stack[(e9 < e21) & (e21 < e50) & (d15["close"] < e9)] = -1
e20 = T._ema(d15["close"], 20)
stretch15 = (d15["close"] - e20) / e20.replace(0, np.nan)
ema_ex = pd.cut(stretch15, bins=[-np.inf, -0.001, 0.001, np.inf], labels=[1, 0, -1]).astype(float)
m15 = pd.DataFrame({"stoch_k": sk15, "ema_stack_bias": ema_stack, "ema_stretch_score": ema_ex})

# ── assemble backbone ─────────────────────────────────────────────────────
df = df1h[["close"]].copy()
df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.iloc[:-1]
df = df.join(ind1h, how="left")
r = df.reset_index().rename(columns={"open_time": "ts"})
i4 = ind4h.reset_index().rename(columns={"open_time": "ts"})
r = pd.merge_asof(r.sort_values("ts"), i4.sort_values("ts"), on="ts", direction="backward")
m15r = m15.reset_index().rename(columns={"open_time": "ts"})
r = pd.merge_asof(r, m15r.sort_values("ts"), on="ts", direction="backward",
                  tolerance=pd.Timedelta("60min"))
df = r.set_index("ts")
df["composite_trend"] = trend_1h.reindex(df.index)
df["composite_rev"] = rev_s.reindex(df.index)
df["composite_p_up"] = pup_1h.reindex(df.index)

A = ["stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h", "stoch_k",
     "vwap_distance_pct", "chg_4h_atr", "bb_pct", "composite_trend",
     "composite_rev", "composite_p_up", "ema_stack_bias", "ema_stretch_score",
     "vwap_stretch_score", "rvol_1h"]

# ── B: ETH/SOL lead-lag ───────────────────────────────────────────────────
btc_r1, btc_r4, btc_r24 = c1h.pct_change(fill_method=None), c1h.pct_change(4, fill_method=None), c1h.pct_change(24, fill_method=None)
alt_r1 = {}
for sym, tag in [("ETHUSDT", "eth"), ("SOLUSDT", "sol")]:
    ac = pd.read_parquet(HERE / f"hist_{sym}_1h.parquet")["close"].reindex(df.index)
    df[f"{tag}_ret_1h"] = ac.pct_change(fill_method=None)
    df[f"{tag}_ret_4h"] = ac.pct_change(4, fill_method=None)
    df[f"spread_{tag}_1h"] = btc_r1.reindex(df.index) - ac.pct_change(fill_method=None)
    df[f"spread_{tag}_24h"] = btc_r24.reindex(df.index) - ac.pct_change(24, fill_method=None)
    alt_r1[tag] = ac.pct_change(fill_method=None)
B = ["eth_ret_1h", "sol_ret_1h", "eth_ret_4h", "sol_ret_4h",
     "spread_eth_1h", "spread_sol_1h", "spread_eth_24h", "spread_sol_24h"]

# ── C: wider cross-asset + session interactions ───────────────────────────
alt24 = {}
for sym, tag in [("XRPUSDT", "xrp"), ("DOGEUSDT", "doge"), ("ADAUSDT", "ada")]:
    ac = pd.read_parquet(HERE / f"hist_{sym}_1h.parquet")["close"].reindex(df.index)
    df[f"{tag}_ret_1h"] = ac.pct_change(fill_method=None)
    df[f"{tag}_ret_4h"] = ac.pct_change(4, fill_method=None)
    alt_r1[tag] = ac.pct_change(fill_method=None)
    alt24[tag] = ac.pct_change(24, fill_method=None)
basket1 = pd.concat(alt_r1.values(), axis=1).mean(axis=1)
df["alt_basket_ret_1h"] = basket1
df["btc_dom_1h"] = btc_r1.reindex(df.index) - basket1
b24 = pd.concat([pd.read_parquet(HERE / f"hist_{s}_1h.parquet")["close"]
                 .reindex(df.index).pct_change(24, fill_method=None)
                 for s in ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")],
                axis=1).mean(axis=1)
df["btc_dom_24h"] = btc_r24.reindex(df.index) - b24
dec = df.index + pd.Timedelta(hours=1)
us = ((dec.hour >= 13) & (dec.hour < 21)).astype(float)
asia = (dec.hour < 8).astype(float)
wknd = (dec.dayofweek >= 5).astype(float)
df["us_x_ret1h"] = us * btc_r1.reindex(df.index)
df["asia_x_ret1h"] = asia * btc_r1.reindex(df.index)
df["wknd_x_ret1h"] = wknd * btc_r1.reindex(df.index)
df["wknd_x_ret24h"] = wknd * btc_r24.reindex(df.index)
C = ["xrp_ret_1h", "xrp_ret_4h", "doge_ret_1h", "doge_ret_4h", "ada_ret_1h",
     "ada_ret_4h", "alt_basket_ret_1h", "btc_dom_1h", "btc_dom_24h",
     "us_x_ret1h", "asia_x_ret1h", "wknd_x_ret1h", "wknd_x_ret24h"]

# ── Ft: full-history taker-buy flow proxy (BinanceUS klines) ──────────────
tb, vol = df1h["taker_buy_vol"], df1h["volume"]
df["tbr_1h"] = (tb / vol.replace(0, np.nan)).reindex(df.index)
df["tbr_4h"] = (tb.rolling(4).sum() / vol.rolling(4).sum().replace(0, np.nan)).reindex(df.index)
df["tbr_24h"] = (tb.rolling(24).sum() / vol.rolling(24).sum().replace(0, np.nan)).reindex(df.index)
t1 = df["tbr_1h"]
df["tbr_z_10d"] = (t1 - t1.rolling(240).mean()) / t1.rolling(240).std().replace(0, np.nan)
lv = np.log1p(vol)
df["dvol_z_10d"] = ((lv - lv.rolling(240).mean()) / lv.rolling(240).std().replace(0, np.nan)).reindex(df.index)
Ft = ["tbr_1h", "tbr_4h", "tbr_24h", "tbr_z_10d", "dvol_z_10d"]

# ── Fcg: CoinGlass point-in-time flow (2026-01-05+) ───────────────────────
cg = pd.read_parquet(HERE / "cg_flow_btc_1h.parquet").reindex(df.index)
fb, fs = cg["fut_buy_usd"], cg["fut_sell_usd"]
sb, ss = cg["spot_buy_usd"], cg["spot_sell_usd"]
df["fut_ratio_1h"] = fb / (fb + fs).replace(0, np.nan)
for w in (4, 12, 24):
    df[f"fut_cvd_{w}h"] = (fb - fs).rolling(w).sum() / (fb + fs).rolling(w).sum().replace(0, np.nan)
df["spot_ratio_1h"] = sb / (sb + ss).replace(0, np.nan)
df["spot_cvd_24h"] = (sb - ss).rolling(24).sum() / (sb + ss).rolling(24).sum().replace(0, np.nan)
oi = cg["oi_close"]
df["oi_chg_1h"] = oi.pct_change(fill_method=None)
df["oi_chg_4h"] = oi.pct_change(4, fill_method=None)
df["oi_chg_24h"] = oi.pct_change(24, fill_method=None)
o1 = df["oi_chg_1h"]
df["oi_z_10d"] = (o1 - o1.rolling(240).mean()) / o1.rolling(240).std().replace(0, np.nan)
fu = cg["funding_close"]
df["funding_close"] = fu
df["funding_z_10d"] = (fu - fu.rolling(240).mean()) / fu.rolling(240).std().replace(0, np.nan)
df["funding_chg_24h"] = fu.diff(24)
ll, ls = cg["liq_long_usd"], cg["liq_short_usd"]
df["liq_imb_1h"] = (ls - ll) / (ls + ll + 1.0)
df["liq_imb_4h"] = (ls.rolling(4).sum() - ll.rolling(4).sum()) / (ls.rolling(4).sum() + ll.rolling(4).sum() + 1.0)
lt = np.log1p(ll + ls)
df["liq_tot_z_10d"] = (lt - lt.rolling(240).mean()) / lt.rolling(240).std().replace(0, np.nan)
Fcg = ["fut_ratio_1h", "fut_cvd_4h", "fut_cvd_12h", "fut_cvd_24h", "spot_ratio_1h",
       "spot_cvd_24h", "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_10d",
       "funding_close", "funding_z_10d", "funding_chg_24h",
       "liq_imb_1h", "liq_imb_4h", "liq_tot_z_10d"]

# ── R: regime features ────────────────────────────────────────────────────
lr = np.log(c1h / c1h.shift(1))
df["er_24h"] = ((c1h - c1h.shift(24)).abs() / c1h.diff().abs().rolling(24).sum().replace(0, np.nan)).reindex(df.index)
df["er_72h"] = ((c1h - c1h.shift(72)).abs() / c1h.diff().abs().rolling(72).sum().replace(0, np.nan)).reindex(df.index)
rv24 = lr.rolling(24).std()
df["vov_7d"] = (rv24.rolling(168).std() / rv24.rolling(168).mean().replace(0, np.nan)).reindex(df.index)
df["rv_ratio_24_168"] = (rv24 / lr.rolling(168).std().replace(0, np.nan)).reindex(df.index)
lo480, hi480 = c1h.rolling(480).min(), c1h.rolling(480).max()
df["donch_pos_20d"] = ((c1h - lo480) / (hi480 - lo480).replace(0, np.nan)).reindex(df.index)
df["dist_30d_high"] = (c1h / c1h.rolling(720).max() - 1).reindex(df.index)
df["dist_30d_low"] = (c1h / c1h.rolling(720).min() - 1).reindex(df.index)
R = ["er_24h", "er_72h", "vov_7d", "rv_ratio_24_168", "donch_pos_20d",
     "dist_30d_high", "dist_30d_low"]

# ── M: intra-hour microstructure from 1m (hour [T, T+1h)) ─────────────────
M = []
f1m = HERE / "hist_BTCUSDT_1m.parquet"
if f1m.exists():
    m = pd.read_parquet(f1m)
    m["hr"] = m.index.floor("h")
    r1m = np.log(m["close"] / m["close"].shift(1))
    r1m[(m["hr"] != m["hr"].shift(1)).values] = np.nan  # no cross-hour returns
    m["_r2"] = r1m ** 2
    m["_up"] = (m["close"] > m["open"]).astype(float)
    mn = m.index.minute
    m["_c45"] = m["close"].where(mn == 44)
    m["_vl20"] = m["volume"].where(mn >= 40, 0.0)
    g = m.groupby("hr")
    dd = (m["close"] / g["close"].cummax() - 1)
    m["_dd"] = dd
    agg = g.agg(rv2=("_r2", "sum"), upmin_frac=("_up", "mean"),
                first_o=("open", "first"), last_c=("close", "last"),
                c45=("_c45", "max"), maxdd_60m=("_dd", "min"),
                v_tot=("volume", "sum"), v_last20=("_vl20", "sum"))
    micro = pd.DataFrame(index=agg.index)
    micro["rv_60m"] = agg["rv2"] ** 0.5
    micro["upmin_frac"] = agg["upmin_frac"]
    micro["ret_first45"] = np.log(agg["c45"] / agg["first_o"])
    micro["ret_last15"] = np.log(agg["last_c"] / agg["c45"])
    micro["maxdd_60m"] = agg["maxdd_60m"]
    micro["volskew_last20"] = agg["v_last20"] / agg["v_tot"].replace(0, np.nan)
    lrv = np.log(micro["rv_60m"].replace(0, np.nan))
    micro["rv60_z_10d"] = (lrv - lrv.rolling(240).mean()) / lrv.rolling(240).std().replace(0, np.nan)
    if micro.index.tz is None:
        micro.index = micro.index.tz_localize("UTC")
    for col in micro.columns:
        df[col] = micro[col].reindex(df.index)
    M = ["rv_60m", "rv60_z_10d", "upmin_frac", "ret_first45", "ret_last15",
         "maxdd_60m", "volskew_last20"]
else:
    print("WARNING: hist_BTCUSDT_1m.parquet missing — M group skipped")

groups = {"A": A, "B": B, "C": C, "Ft": Ft, "Fcg": Fcg, "R": R, "M": M}
with open(HERE / "feature_groups.json", "w") as f:
    json.dump(groups, f, indent=1)
keep = ["close", "label"] + sum(groups.values(), [])
df = df[keep]
df.to_parquet(HERE / "extended_dataset.parquet")
print(f"extended dataset: {len(df):,} rows {df.index[0]} -> {df.index[-1]}")
for gname, cols in groups.items():
    if cols:
        print(f"  {gname}: {len(cols)} feats, non-null frac "
              f"{df[cols].notna().mean().mean():.3f}")
print("S3 DONE")
