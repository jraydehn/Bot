"""
S1 -- 15-minute directional p_up (the pinned plan, Phase 1).
Structural analog of the hourly composite: indicator votes -> empirical
calibration tables -> P(price up over the next 15m bar).

Conventions:
- Votes at 15m bar T (open-time indexed) use bars <= T; the bar's own close
  is knowable at T+15m, so the signal's EFFECTIVE time is T+15m and its
  target is the NEXT bar's close vs bar T's close. Zero lookahead.
- Trend votes (15m): stoch14 bands, Williams %R, Bollinger %B, Keltner
  position, MACD hist sign + slope, K-D diff (double-weight, hourly analog),
  high-volume direction.  Range ~[-8, +8], clipped +/-6.
- Reversion votes (5m, contra-directional): oversold/overbought stoch + BB,
  extreme 5m change z, wick imbalance.  Range ~[-4, +4].
- Calibration: P(up) per (trend_bin, rev_bin) cell with min-count fallback
  to the trend-marginal, then global. Built on TRAIN years, validated on
  holdout years (walk-forward by year), per feedback_test_on_long_history.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2023-10-01"]      # warmup margin before 2024
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
d15 = df1m.resample("15min").agg(AGG).dropna()
d5 = df1m.resample("5min").agg(AGG).dropna()
print(f"15m bars: {len(d15)}  {d15.index.min()} -> {d15.index.max()}")

c, h, l, v = d15["close"], d15["high"], d15["low"], d15["volume"]

# ── trend votes on 15m ─────────────────────────────────────────────────────
lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
stoch = ((c - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100
wpr = -100 * (hi14 - c) / (hi14 - lo14).replace(0, np.nan)
ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
bbp = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
e10 = c.ewm(span=10, adjust=False).mean()
tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
a14 = tr.ewm(span=14, adjust=False).mean()
kcp = (c - (e10 - 1.5 * a14)) / (3 * a14).replace(0, np.nan)
macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
sig9 = macd.ewm(span=9, adjust=False).mean()
hist = macd - sig9
stoch_d = stoch.rolling(3).mean()
kd = stoch - stoch_d
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

# ── reversion votes on 5m (contra: oversold -> up pressure) ───────────────
c5, h5, l5 = d5["close"], d5["high"], d5["low"]
lo14_5, hi14_5 = l5.rolling(14).min(), h5.rolling(14).max()
st5 = ((c5 - lo14_5) / (hi14_5 - lo14_5).replace(0, np.nan)) * 100
ma20_5, sd20_5 = c5.rolling(20).mean(), c5.rolling(20).std()
bbp5 = (c5 - (ma20_5 - 2 * sd20_5)) / (4 * sd20_5).replace(0, np.nan)
chg5 = c5.pct_change() * 100
chg5_z = (chg5 - chg5.rolling(96).mean()) / chg5.rolling(96).std().replace(0, np.nan)
rng5 = (h5 - l5).replace(0, np.nan)
upper_w = (h5 - pd.concat([d5["open"], c5], axis=1).max(axis=1)) / rng5
lower_w = (pd.concat([d5["open"], c5], axis=1).min(axis=1) - l5) / rng5

rev5 = pd.Series(0, index=d5.index, dtype=float)
rev5 += (st5 < 20).astype(int) - (st5 > 80).astype(int)
rev5 += (bbp5 < 0.10).astype(int) - (bbp5 > 0.90).astype(int)
rev5 += (chg5_z < -2).astype(int) - (chg5_z > 2).astype(int)
rev5 += ((lower_w > 0.5)).astype(int) - ((upper_w > 0.5)).astype(int)
rev = rev5.resample("15min").last().reindex(d15.index).fillna(0).clip(-4, 4)

feat = pd.DataFrame({"trend": trend, "rev": rev}).dropna()
target = (c.shift(-1) > c).astype(float).reindex(feat.index)   # next 15m bar up?
feat["up"] = target
feat = feat.dropna()
feat["year"] = feat.index.year
print(f"vote matrix: {len(feat)} bars, years {sorted(feat['year'].unique())}")
print("trend distribution:", feat["trend"].value_counts().sort_index().to_dict())

# ── walk-forward year-split calibration + validation ─────────────────────
def build_table(df, min_n=200):
    tbl, marg = {}, {}
    for t in range(-6, 7):
        sub_t = df[df["trend"] == t]
        marg[t] = sub_t["up"].mean() if len(sub_t) >= min_n else None
        for r in range(-4, 5):
            s = sub_t[sub_t["rev"] == r]
            if len(s) >= min_n:
                tbl[(t, r)] = s["up"].mean()
    g = df["up"].mean()
    return tbl, marg, g

def predict(tbl, marg, g, t, r):
    if (t, r) in tbl:
        return tbl[(t, r)]
    if marg.get(t) is not None:
        return marg[t]
    return g

print("\n=== walk-forward year validation (train on prior years, test on year Y) ===")
years = sorted(feat["year"].unique())
all_preds = []
for y in years[1:]:
    train = feat[feat["year"] < y]
    test = feat[feat["year"] == y].copy()
    if len(train) < 5000 or len(test) < 1000:
        continue
    tbl, marg, g = build_table(train)
    test["p15"] = [predict(tbl, marg, g, t, r) for t, r in zip(test["trend"], test["rev"])]
    # spread check: top-quintile p15 vs bottom-quintile realized up-rate
    q = test["p15"].quantile([0.2, 0.8])
    lo_r = test[test["p15"] <= q[0.2]]["up"].mean()
    hi_r = test[test["p15"] >= q[0.8]]["up"].mean()
    # simple IC
    ic = test["p15"].corr(test["up"])
    print(f"  {y}: n={len(test)}  bottom-q up-rate={lo_r:.4f}  top-q up-rate={hi_r:.4f}  "
          f"spread={hi_r-lo_r:+.4f}  IC={ic:+.4f}")
    all_preds.append(test)

# final table on all data through 2025, applied to 2026 -- and save everything
train_final = feat[feat["year"] <= 2025]
tbl, marg, g = build_table(train_final)
with open(f"{OUT}/pup15m_tables.pkl", "wb") as f:
    pickle.dump({"table": tbl, "marginal": marg, "global": g,
                 "built_on": "2023-10..2025-12", "target": "P(next 15m bar close up)"}, f)

# full 2026 signal series for the entry-timing sim (effective = bar_open + 15m)
f26 = feat[feat["year"] == 2026].copy()
f26["p15"] = [predict(tbl, marg, g, t, r) for t, r in zip(f26["trend"], f26["rev"])]
sigser = f26[["p15", "trend", "rev"]].reset_index()
sigser = sigser.rename(columns={sigser.columns[0]: "bar_open"})
sigser["effective"] = sigser["bar_open"] + pd.Timedelta("15min")
sigser.to_csv(f"{OUT}/pup15m_series_2026.csv", index=False)
print(f"\n2026 signal series saved: {len(sigser)} bars  p15 range "
      f"[{f26['p15'].min():.3f}, {f26['p15'].max():.3f}]  mean={f26['p15'].mean():.3f}")
print("DONE_S1")
