"""
S1 -- Build ETH candidate feature dataset, mirroring the BTC p_up_v3
rebuild's taxonomy (A/B/C/Ft/Fcg/R/M groups) but computed FOR ETH as the
prediction target, reusing already-fetched raw price parquets (no
refetch needed -- same underlying history the BTC rebuild pulled).

Group semantics, re-derived for ETH (NOT copy-pasted from BTC):
  A: ETH's own lag-corrected price/technicals
  B: BTC + SOL lead-lag (predicting ETH, the reverse of BTC's B group)
  C: alt-coin (XRP/DOGE/ADA) returns + BTC dominance + session interactions
  M: ETH's own intra-hour 1m microstructure
  Ft/Fcg/R: same categories as BTC (taker flow, CoinGlass, regime) --
    included for completeness, expected to fail again for the same
    reasons (thin CoinGlass history, regime redundant with A) but tested
    not assumed.
Label: direction of ETH bar T+1 close vs bar T close (next 1h).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REBUILD = "reform_results/pup_v2_rebuild_20260704"
OUT = "reform_results/eth_pup_rebuild_20260706"

c_eth = pd.read_parquet(f"{REBUILD}/hist_ETHUSDT_1h.parquet")
c_btc = pd.read_parquet(f"{REBUILD}/hist_BTCUSDT_1h.parquet")["close"].astype(float)
c_sol = pd.read_parquet(f"{REBUILD}/hist_SOLUSDT_1h.parquet")["close"].astype(float)
c_xrp = pd.read_parquet(f"{REBUILD}/hist_XRPUSDT_1h.parquet")["close"].astype(float)
c_doge = pd.read_parquet(f"{REBUILD}/hist_DOGEUSDT_1h.parquet")["close"].astype(float)
c_ada = pd.read_parquet(f"{REBUILD}/hist_ADAUSDT_1h.parquet")["close"].astype(float)

c1h = c_eth["close"].astype(float)
h1h = c_eth["high"].astype(float)
l1h = c_eth["low"].astype(float)
v1h = c_eth["volume"].astype(float)

df = pd.DataFrame(index=c1h.index)
df["close"] = c1h
label = (c1h.shift(-1) > c1h).astype(float)
label[c1h.shift(-1).isna()] = np.nan
df["label"] = label

# ── A: ETH's own lag-corrected price/technicals ──────────────────────────
lr = np.log(c1h / c1h.shift(1))
ema12 = c1h.ewm(span=12, adjust=False).mean(); ema26 = c1h.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
df["macd_hist_1h"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
delta = c1h.diff()
gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
c4h = c1h.resample("4h").last().shift(3)  # shifted +3h before backward merge (no lookahead)
delta4 = c4h.diff()
gain4 = delta4.clip(lower=0).ewm(span=14, adjust=False).mean()
loss4 = (-delta4.clip(upper=0)).ewm(span=14, adjust=False).mean()
rsi_4h = (100 - 100 / (1 + gain4 / loss4.replace(0, np.nan))).reindex(df.index, method="ffill")
df["rsi_4h"] = rsi_4h
ll14 = l1h.rolling(14).min(); hh14 = h1h.rolling(14).max()
df["stoch_k"] = ((c1h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100
c4h_full = c1h.resample("4h").last()
ll14_4h = c4h_full.rolling(14).min(); hh14_4h = c4h_full.rolling(14).max()
stoch_k_4h = (((c4h_full - ll14_4h) / (hh14_4h - ll14_4h).replace(0, np.nan)) * 100).shift(3).reindex(df.index, method="ffill")
df["stoch_k_4h"] = stoch_k_4h
ema50 = c1h.ewm(span=50, adjust=False).mean()
df["ema50_dist"] = (c1h - ema50) / ema50
sma20 = c1h.rolling(20).mean(); std20 = c1h.rolling(20).std()
bb_hi = sma20 + 2 * std20; bb_lo = sma20 - 2 * std20
df["bb_pct"] = (c1h - bb_lo) / (bb_hi - bb_lo).replace(0, np.nan)
vwap_num = (c1h * v1h).rolling(24).sum(); vwap_den = v1h.rolling(24).sum().replace(0, np.nan)
vwap = vwap_num / vwap_den
df["vwap_distance_pct"] = (c1h - vwap) / vwap
atr = (h1h - l1h).rolling(14).mean()
df["chg_4h_atr"] = (c1h - c1h.shift(4)) / atr.replace(0, np.nan)
ema9 = c1h.ewm(span=9, adjust=False).mean(); ema21 = c1h.ewm(span=21, adjust=False).mean(); ema55 = c1h.ewm(span=55, adjust=False).mean()
df["ema_stack_bias"] = np.sign(ema9 - ema21) + np.sign(ema21 - ema55)
df["ema_stretch_score"] = ((c1h - ema9) / ema9).clip(-0.05, 0.05) * 20
vwap_std = ((c1h - vwap) ** 2).rolling(24).mean() ** 0.5
df["vwap_stretch_score"] = ((c1h - vwap) / vwap_std.replace(0, np.nan)).clip(-3, 3)
df["rvol_1h"] = lr.rolling(24).std() / lr.rolling(168).std().replace(0, np.nan)
# composite trend/rev/p_up proxies (simple, causal, ETH-specific -- NOT reusing BTC's
# composite_scorer.py which is BTC-calibrated; a crude but honest stand-in for group A)
trend_votes = (np.sign(ema9 - ema21) + np.sign(ema21 - ema55) + np.sign(c1h - c1h.shift(4)) +
              np.sign(c1h - c1h.shift(24)))
df["composite_trend"] = trend_votes
df["composite_rev"] = (df["rsi_14"] < 30).astype(int) - (df["rsi_14"] > 70).astype(int) + \
                      (df["stoch_k"] < 20).astype(int) - (df["stoch_k"] > 80).astype(int)
df["composite_p_up"] = 0.5 + 0.02 * trend_votes.clip(-4, 4)

A = ["stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h", "stoch_k",
    "vwap_distance_pct", "chg_4h_atr", "bb_pct", "composite_trend", "composite_rev",
    "composite_p_up", "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score", "rvol_1h"]

# ── B: BTC + SOL lead-lag (predicting ETH) ───────────────────────────────
for name, series in [("btc", c_btc), ("sol", c_sol)]:
    r1h = np.log(series / series.shift(1)).reindex(df.index, method="ffill")
    r4h = np.log(series / series.shift(4)).reindex(df.index, method="ffill")
    df[f"{name}_ret_1h"] = r1h
    df[f"{name}_ret_4h"] = r4h
    eth_r1h = np.log(c1h / c1h.shift(1))
    eth_r4h = np.log(c1h / c1h.shift(4))
    df[f"spread_{name}_1h"] = eth_r1h - r1h
    df[f"spread_{name}_24h"] = (np.log(c1h / c1h.shift(24))) - np.log(series / series.shift(24)).reindex(df.index, method="ffill")
B = ["btc_ret_1h", "sol_ret_1h", "btc_ret_4h", "sol_ret_4h", "spread_btc_1h", "spread_sol_1h",
    "spread_btc_24h", "spread_sol_24h"]

# ── C: alt-coin returns + dominance + session interactions ───────────────
for name, series in [("xrp", c_xrp), ("doge", c_doge), ("ada", c_ada)]:
    r1h = np.log(series / series.shift(1)).reindex(df.index, method="ffill")
    r4h = np.log(series / series.shift(4)).reindex(df.index, method="ffill")
    df[f"{name}_ret_1h"] = r1h; df[f"{name}_ret_4h"] = r4h
alt_basket = pd.concat([df["xrp_ret_1h"], df["doge_ret_1h"], df["ada_ret_1h"]], axis=1).mean(axis=1)
df["alt_basket_ret_1h"] = alt_basket
btc_vol_1h = c_btc.reindex(df.index, method="ffill")
df["eth_dom_1h"] = np.log(c1h / c1h.shift(1)) - np.log(btc_vol_1h / btc_vol_1h.shift(1))
df["eth_dom_24h"] = np.log(c1h / c1h.shift(24)) - np.log(btc_vol_1h / btc_vol_1h.shift(24))
hour_utc = df.index.hour
is_us = ((hour_utc >= 13) & (hour_utc < 21)).astype(float)
is_asia = ((hour_utc >= 0) & (hour_utc < 8)).astype(float)
is_wknd = (df.index.dayofweek >= 5).astype(float)
eth_r1h_all = np.log(c1h / c1h.shift(1)); eth_r24h_all = np.log(c1h / c1h.shift(24))
df["us_x_ret1h"] = is_us * eth_r1h_all
df["asia_x_ret1h"] = is_asia * eth_r1h_all
df["wknd_x_ret1h"] = is_wknd * eth_r1h_all
df["wknd_x_ret24h"] = is_wknd * eth_r24h_all
C = ["xrp_ret_1h", "xrp_ret_4h", "doge_ret_1h", "doge_ret_4h", "ada_ret_1h", "ada_ret_4h",
    "alt_basket_ret_1h", "eth_dom_1h", "eth_dom_24h",
    "us_x_ret1h", "asia_x_ret1h", "wknd_x_ret1h", "wknd_x_ret24h"]
Cs = ["us_x_ret1h", "asia_x_ret1h", "wknd_x_ret1h", "wknd_x_ret24h"]

# ── R: regime/donchian features ──────────────────────────────────────────
df["er_24h"] = ((c1h - c1h.shift(24)).abs() / c1h.diff().abs().rolling(24).sum().replace(0, np.nan))
df["er_72h"] = ((c1h - c1h.shift(72)).abs() / c1h.diff().abs().rolling(72).sum().replace(0, np.nan))
rv24 = lr.rolling(24).std()
df["vov_7d"] = rv24.rolling(168).std() / rv24.rolling(168).mean().replace(0, np.nan)
df["rv_ratio_24_168"] = rv24 / lr.rolling(168).std().replace(0, np.nan)
lo480, hi480 = c1h.rolling(480).min(), c1h.rolling(480).max()
df["donch_pos_20d"] = (c1h - lo480) / (hi480 - lo480).replace(0, np.nan)
df["dist_30d_high"] = c1h / c1h.rolling(720).max() - 1
df["dist_30d_low"] = c1h / c1h.rolling(720).min() - 1
R = ["er_24h", "er_72h", "vov_7d", "rv_ratio_24_168", "donch_pos_20d", "dist_30d_high", "dist_30d_low"]

# ── M: intra-hour microstructure from ETH 1m ─────────────────────────────
M = []
f1m_candidates = ["data/binanceus_ETHUSDT_1m_2024-01-01_2026-07-06.parquet"]
import glob
avail = sorted(glob.glob("data/binanceus_ETHUSDT_1m_2024-01-01_*.parquet"))
f1m = avail[-1] if avail else None
if f1m:
    m = pd.read_parquet(f1m)
    m["hr"] = m.index.floor("h")
    r1m = np.log(m["close"] / m["close"].shift(1))
    r1m[(m["hr"] != m["hr"].shift(1)).values] = np.nan
    m["_r2"] = r1m ** 2
    m["_up"] = (m["close"] > m["open"]).astype(float)
    mn = m.index.minute
    m["_c45"] = m["close"].where(mn == 44)
    m["_vl20"] = m["volume"].where(mn >= 40, 0.0)
    g = m.groupby("hr")
    dd = (m["close"] / g["close"].cummax() - 1)
    m["_dd"] = dd
    agg = g.agg(rv2=("_r2", "sum"), upmin_frac=("_up", "mean"), first_o=("open", "first"),
                last_c=("close", "last"), c45=("_c45", "max"), maxdd_60m=("_dd", "min"),
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
    M = ["rv_60m", "rv60_z_10d", "upmin_frac", "ret_first45", "ret_last15", "maxdd_60m", "volskew_last20"]
else:
    print("WARNING: ETH 1m data missing -- M group skipped")

feature_groups = {"A": A, "B": B, "C": C, "Cs": Cs, "R": R, "M": M}
import json
with open(f"{OUT}/feature_groups.json", "w") as f:
    json.dump(feature_groups, f, indent=1)

# NOTE: M group only has real coverage from 2024-01-01 onward (1m data limit) --
# NOT truncating the whole dataset for it; A/B/C/R keep full 2020-2026 coverage,
# M-inclusive configs will naturally have fewer valid rows pre-2024 (same
# handling as BTC's Fcg group being CoinGlass-data-limited to 2026-01-12+).
df = df.dropna(subset=["label"])
df.to_parquet(f"{OUT}/eth_dataset.parquet")
print(f"saved {OUT}/eth_dataset.parquet: {df.shape}")
print(f"range: {df.index.min()} -> {df.index.max()}")
print(f"label balance: up={df['label'].mean():.3f}")
for g, feats in feature_groups.items():
    cov = df[feats].notna().all(axis=1).mean() if feats else 0
    print(f"  group {g}: {len(feats)} feats, {cov:.1%} full-row coverage")
