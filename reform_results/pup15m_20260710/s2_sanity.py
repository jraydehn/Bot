"""
S2 -- sanity battery for the suspiciously-strong 15m p_up.
1. Per-trend-bin realized up-rate (train vs each holdout year): monotone?
2. Baselines: (a) 1-bar momentum sign, (b) 1-bar reversion sign, (c) stoch alone.
   If the composite barely beats the naive baseline it's just autocorrelation.
3. Shuffle test: signal vs time-shuffled target IC (should be ~0).
4. Timing stress: recompute target with 1-bar delay (predict bar T+2 from
   info at T+15) -- if IC survives a full extra bar of delay it's regime/vol
   artifact risk; if it drops toward 0 the edge is genuinely short-horizon.
"""
import warnings, pathlib, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2023-10-01"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
d15 = df1m.resample("15min").agg(AGG).dropna()
c, h, l, v = d15["close"], d15["high"], d15["low"], d15["volume"]

# rebuild the exact trend/rev from s1 (import-free copy)
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
hist = macd - macd.ewm(span=9, adjust=False).mean()
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

up1 = (c.shift(-1) > c).astype(float)     # target
mom = np.sign(c.pct_change())             # naive 1-bar momentum
df = pd.DataFrame({"trend": trend, "up": up1, "mom": mom, "stoch": stoch}).dropna()
df["year"] = df.index.year

print("=== 1. per-trend-bin realized up-rate by year (monotone check) ===")
piv = df.pivot_table(index="trend", columns="year", values="up", aggfunc=["mean", "count"])
print(piv.round(4).to_string())

print("\n=== 2. baselines (IC vs up1, per year) ===")
for y in [2024, 2025, 2026]:
    s = df[df["year"] == y]
    print(f"  {y}: composite trend IC={s['trend'].corr(s['up']):+.4f}   "
          f"naive momentum IC={s['mom'].corr(s['up']):+.4f}   "
          f"stoch-alone IC={s['stoch'].corr(s['up']):+.4f}")

print("\n=== 3. shuffle test (3 shuffles of target) ===")
rng = np.random.default_rng(42)
for i in range(3):
    sh = df["up"].sample(frac=1, random_state=i).values
    print(f"  shuffle {i}: IC={pd.Series(sh, index=df.index).corr(df['trend']):+.4f}")

print("\n=== 4. delayed-target stress: predict bar T+2 (skip one full bar) ===")
up2 = (c.shift(-2) > c.shift(-1)).astype(float)
df["up2"] = up2.reindex(df.index)
for y in [2024, 2025, 2026]:
    s = df[df["year"] == y].dropna(subset=["up2"])
    print(f"  {y}: IC(trend, next-next bar)={s['trend'].corr(s['up2']):+.4f}")

print("\n=== 5. direction of the signal (is trend=+6 momentum or fade?) ===")
for t in [-6, -3, 0, 3, 6]:
    s = df[df["trend"] == t]
    print(f"  trend={t:+d}: n={len(s):6d}  P(next bar up)={s['up'].mean():.4f}")
print("DONE_S2")
