"""
S15 -- pup15m calibrated PER pup_v3 intraday regime state (user directive).
Hard conditioning on the causal 3-label state (neutral/rising/crashing),
fallback chain per state: state cell (n>=150) -> state trend-marginal
(n>=150) -> pooled cell -> pooled marginal -> global.
Walk-forward: train <=Y-1, test Y (2025, 2026). Compare pooled vs
state-conditioned IC/quintile spread + per-state detail + how the tables
actually differ at the trend extremes (interpretability).
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

# ---- vote matrix (same construction as s1/s12) ----
p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2023-10-01"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
d15 = df1m.resample("15min").agg(AGG).dropna()
d5 = df1m.resample("5min").agg(AGG).dropna()
c, h, l, v = d15["close"], d15["high"], d15["low"], d15["volume"]
lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
rng14 = (hi14 - lo14).replace(0, np.nan)
stoch = ((c - lo14) / rng14) * 100
wpr = -100 * (hi14 - c) / rng14
ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
bbp = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
e10 = c.ewm(span=10, adjust=False).mean()
tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
a14 = tr.ewm(span=14, adjust=False).mean()
kcp = (c - (e10 - 1.5 * a14)) / (3 * a14).replace(0, np.nan)
macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
hist = macd - macd.ewm(span=9, adjust=False).mean()
kd = stoch - stoch.rolling(3).mean()
vol_med = v.rolling(20).median()
hv_up = (v > 1.5 * vol_med) & (c > d15["open"])
hv_dn = (v > 1.5 * vol_med) & (c < d15["open"])
trend = pd.Series(0, index=d15.index, dtype=float)
trend += (stoch > 80).astype(int) - (stoch < 20).astype(int)
trend += (wpr > -20).astype(int) - (wpr < -80).astype(int)
trend += (bbp > 0.80).astype(int) - (bbp < 0.20).astype(int)
trend += (kcp > 0.85).astype(int) - (kcp < 0.15).astype(int)
trend += ((hist > 0) & (hist > hist.shift(1))).astype(int) - ((hist < 0) & (hist < hist.shift(1))).astype(int)
trend += 2 * (kd > 10).astype(int) + ((kd > 5) & (kd <= 10)).astype(int)
trend -= 2 * (kd < -10).astype(int) + ((kd < -5) & (kd >= -10)).astype(int)
trend += hv_up.astype(int) - hv_dn.astype(int)
trend = trend.clip(-6, 6)
c5, h5, l5 = d5["close"], d5["high"], d5["low"]
lo5, hi5 = l5.rolling(14).min(), h5.rolling(14).max()
st5 = ((c5 - lo5) / (hi5 - lo5).replace(0, np.nan)) * 100
ma5, sd5 = c5.rolling(20).mean(), c5.rolling(20).std()
bbp5 = (c5 - (ma5 - 2 * sd5)) / (4 * sd5).replace(0, np.nan)
chg5 = c5.pct_change() * 100
chg5_z = (chg5 - chg5.rolling(96).mean()) / chg5.rolling(96).std().replace(0, np.nan)
rng5 = (h5 - l5).replace(0, np.nan)
upper_w = (h5 - pd.concat([d5["open"], c5], axis=1).max(axis=1)) / rng5
lower_w = (pd.concat([d5["open"], c5], axis=1).min(axis=1) - l5) / rng5
rev5 = pd.Series(0, index=d5.index, dtype=float)
rev5 += (st5 < 20).astype(int) - (st5 > 80).astype(int)
rev5 += (bbp5 < 0.10).astype(int) - (bbp5 > 0.90).astype(int)
rev5 += (chg5_z < -2).astype(int) - (chg5_z > 2).astype(int)
rev5 += (lower_w > 0.5).astype(int) - (upper_w > 0.5).astype(int)
rev = rev5.resample("15min").last().reindex(d15.index).fillna(0).clip(-4, 4)

feat = pd.DataFrame({"trend": trend, "rev": rev}).dropna()
feat["up"] = (c.shift(-1) > c).astype(float).reindex(feat.index)
feat = feat.dropna()
feat["effective"] = feat.index + pd.Timedelta("15min")

stt = pd.read_csv(f"{OUT}/pupv3_state_causal.csv", parse_dates=["hour_ts"]).sort_values("hour_ts")
feat = pd.merge_asof(feat.sort_values("effective"), stt[["hour_ts", "pv3_state"]],
                     left_on="effective", right_on="hour_ts", direction="backward")
feat = feat.dropna(subset=["pv3_state"])
feat["stale_h"] = (feat["effective"] - feat["hour_ts"]).dt.total_seconds() / 3600
feat = feat[feat["stale_h"] <= 2.0]     # drop bars where the state reading is stale (gaps)
feat["year"] = feat["effective"].dt.year
print(f"bars with state: {len(feat)}  state mix: "
      f"{(feat['pv3_state'].value_counts(normalize=True)*100).round(1).to_dict()}")

STATES = ["neutral", "rising", "crashing"]

def build_pooled(df, min_n=200):
    tbl, marg = {}, {}
    for t in range(-6, 7):
        s_t = df[df["trend"] == t]
        marg[t] = s_t["up"].mean() if len(s_t) >= min_n else None
        for r in range(-4, 5):
            s = s_t[s_t["rev"] == r]
            if len(s) >= min_n:
                tbl[(t, r)] = s["up"].mean()
    return tbl, marg, df["up"].mean()

def build_state(df, state, min_n=150):
    sub = df[df["pv3_state"] == state]
    tbl, marg = {}, {}
    for t in range(-6, 7):
        s_t = sub[sub["trend"] == t]
        marg[t] = s_t["up"].mean() if len(s_t) >= min_n else None
        for r in range(-4, 5):
            s = s_t[s_t["rev"] == r]
            if len(s) >= min_n:
                tbl[(t, r)] = s["up"].mean()
    return tbl, marg

def lookup(tbl, marg, ptbl, pmarg, g, t, r):
    if (t, r) in tbl:
        return tbl[(t, r)]
    if marg.get(t) is not None:
        return marg[t]
    if (t, r) in ptbl:
        return ptbl[(t, r)]
    if pmarg.get(t) is not None:
        return pmarg[t]
    return g

print("\n=== walk-forward: pooled vs pv3-state-conditioned ===")
keep = {}
for y in [2025, 2026]:
    train, test = feat[feat["year"] < y], feat[feat["year"] == y].copy()
    ptbl, pmarg, g = build_pooled(train)
    stabs = {s: build_state(train, s) for s in STATES}
    test["p_pooled"] = [lookup(ptbl, pmarg, ptbl, pmarg, g, t, r)
                        for t, r in zip(test["trend"], test["rev"])]
    test["p_sc"] = [lookup(*stabs[s], ptbl, pmarg, g, t, r)
                    for s, t, r in zip(test["pv3_state"], test["trend"], test["rev"])]
    for nm in ["p_pooled", "p_sc"]:
        ic = test[nm].corr(test["up"])
        q = test[nm].quantile([0.2, 0.8])
        spread = test[test[nm] >= q[0.8]]["up"].mean() - test[test[nm] <= q[0.2]]["up"].mean()
        print(f"  {y} {nm:9s}: IC={ic:+.4f}  quintile spread={spread:+.4f}")
    for s in STATES:
        sub = test[test["pv3_state"] == s]
        print(f"      {y} {s:9s} bars={len(sub):5d}: IC pooled={sub['p_pooled'].corr(sub['up']):+.4f}"
              f"  sc={sub['p_sc'].corr(sub['up']):+.4f}  base P(up)={sub['up'].mean():.3f}")
    if y == 2026:
        keep = {"pooled": (ptbl, pmarg, g), "state": stabs}
        test.reset_index()[["effective", "trend", "rev", "pv3_state", "p_pooled", "p_sc"]].to_csv(
            f"{OUT}/pup15m_sc_series_2026.csv", index=False)

print("\n=== how the tables differ (train<=2025): P(up) at trend extremes per state ===")
ptbl, pmarg, g = keep["pooled"]
for t in [-6, -3, 0, 3, 6]:
    row = f"  trend={t:+d}: pooled={pmarg.get(t) if pmarg.get(t) else float('nan'):.3f}"
    for s in STATES:
        m = keep["state"][s][1].get(t)
        row += f"  {s}={m:.3f}" if m is not None else f"  {s}=--"
    print(row)

with open(f"{OUT}/pup15m_sc_tables.pkl", "wb") as f:
    pickle.dump({"pooled": keep["pooled"], "state": keep["state"], "states": STATES,
                 "conditioner": "pup_v3 regime HMM causal 3-label state",
                 "built_on": "2024-01..2025-12"}, f)
print("DONE_S15")
