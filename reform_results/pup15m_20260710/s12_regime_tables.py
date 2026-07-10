"""
S12 -- regime-conditioned pup15m tables (Phase 2 of the pinned plan).
Per-regime (trend15, rev5) -> P(up next 15m bar) tables, built with SOFT
posterior weights (each bar contributes to all 3 regime tables weighted by
its causal posterior), blended at lookup by the same posteriors -- the exact
structure of the hourly composite's lookup_p_up_regime.

Fallback chain per regime: weighted cell (eff_n>=150) -> weighted trend
marginal (eff_n>=150) -> pooled cell -> pooled marginal -> global.
Walk-forward: train <=Y-1, test Y (2025, 2026). Metrics: IC + quintile
spread, pooled vs regime-conditioned, plus per-regime IC in the test year.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

# ---- rebuild the vote matrix (same as s1) ----
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

# ---- join causal regime posteriors (regime effective <= bar effective) ----
reg = pd.read_csv(f"{OUT}/macro_regime_posteriors_1h.csv", parse_dates=["effective"]).sort_values("effective")
feat = pd.merge_asof(feat.sort_values("effective"), reg[["effective", "p_bull", "p_sdwy", "p_bear"]],
                     on="effective", direction="backward").dropna(subset=["p_bull"])
feat["year"] = feat["effective"].dt.year
feat = feat.set_index("effective")
print(f"bars with regime: {len(feat)}  years {sorted(feat['year'].unique())}")

REGS = ["p_bull", "p_sdwy", "p_bear"]

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

def build_regime(df, w_col, min_eff=150):
    tbl, marg = {}, {}
    w = df[w_col]
    for t in range(-6, 7):
        m_t = df["trend"] == t
        wt = w[m_t]
        if wt.sum() >= min_eff:
            marg[t] = float(np.average(df.loc[m_t, "up"], weights=wt))
        else:
            marg[t] = None
        for r in range(-4, 5):
            m = m_t & (df["rev"] == r)
            wm = w[m]
            if wm.sum() >= min_eff:
                tbl[(t, r)] = float(np.average(df.loc[m, "up"], weights=wm))
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

print("\n=== walk-forward: pooled vs regime-conditioned ===")
artifacts = {}
for y in [2025, 2026]:
    train, test = feat[feat["year"] < y], feat[feat["year"] == y].copy()
    ptbl, pmarg, g = build_pooled(train)
    rtabs = {rc: build_regime(train, rc) for rc in REGS}
    test["p_pooled"] = [lookup(ptbl, pmarg, ptbl, pmarg, g, t, r)
                        for t, r in zip(test["trend"], test["rev"])]
    vals = np.zeros(len(test))
    for rc in REGS:
        tbl, marg = rtabs[rc]
        pv = np.array([lookup(tbl, marg, ptbl, pmarg, g, t, r)
                       for t, r in zip(test["trend"], test["rev"])])
        vals += test[rc].values * pv
    test["p_rc"] = vals
    for nm in ["p_pooled", "p_rc"]:
        ic = test[nm].corr(test["up"])
        q = test[nm].quantile([0.2, 0.8])
        spread = test[test[nm] >= q[0.8]]["up"].mean() - test[test[nm] <= q[0.2]]["up"].mean()
        print(f"  {y} {nm:9s}: IC={ic:+.4f}  quintile spread={spread:+.4f}")
    # per-regime IC in test year (hard argmax split)
    top = test[REGS].idxmax(axis=1)
    for rc in REGS:
        s = test[top == rc]
        print(f"      {y} {rc} bars={len(s):5d}: IC pooled={s['p_pooled'].corr(s['up']):+.4f} "
              f" rc={s['p_rc'].corr(s['up']):+.4f}")
    if y == 2026:
        artifacts = {"pooled": (ptbl, pmarg, g), "regime": rtabs}
        test.reset_index()[["effective", "trend", "rev", "p_pooled", "p_rc",
                            "p_bull", "p_sdwy", "p_bear"]].to_csv(f"{OUT}/pup15m_rc_series_2026.csv", index=False)

with open(f"{OUT}/pup15m_rc_tables.pkl", "wb") as f:
    pickle.dump({"pooled": artifacts["pooled"], "regime": artifacts["regime"],
                 "regs": REGS, "built_on": "2024-01..2025-12 (walk-forward artifact for 2026)"}, f)
print("\nsaved pup15m_rc_tables.pkl + pup15m_rc_series_2026.csv")
print("DONE_S12")
