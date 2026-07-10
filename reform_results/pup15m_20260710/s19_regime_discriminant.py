"""
S19 -- what differentiates the BTC 15m YES book's good days from bad days?
Method (per feedback_test_on_long_history): replicate the book's exact bet
synthetically over 2.5 YEARS, then find causal market-state features that
separate winning from losing periods, requiring the SAME direction in all
three years. No fitting to July.

Synthetic bet (the book's signature, measured from the real 309-trade book):
at every 15m expiry T, decide at T-10.5min: strike K = S x (1-0.000633)
(spot 0.0633% above strike); win = close(T) >= K; breakeven = 0.730.

Causal features at decision time (all trailing):
  whipsaw_24h    : fraction of adjacent 15m closes with opposite sign (96 bars)
  late_rev_24h   : fraction of 15m bars whose final 5m move opposed the bar's
                   first 10m move (the book's exact failure mode), 24h trailing
  rv_ratio       : realized vol 6h / realized vol 72h (1m rets)
  ret_24h        : trailing 24h return (%)
  abs_ret_24h    : |ret_24h|
  atr_ratio      : 1h-bar ATR14 vs its own 240h mean
  range_pos_7d   : (S - 7d low) / (7d high - 7d low)
  dd_24h         : drawdown from trailing 24h high (%)
  bar_flip_1h    : sign flips among the last four 15m bars (0-3)
"""
import warnings
import pathlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
px = px[px.index >= "2023-12-01"]
c1 = px["close"]

# 15m bars for features
d15 = px.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
r15 = d15["close"].pct_change()
sign15 = np.sign(r15)
whip = (sign15 != sign15.shift(1)).rolling(96).mean()
# late reversal: last-5m move sign vs first-10m move sign within each 15m bar
c5 = px["close"].resample("5min").last().dropna()
c5f = c5.reindex(pd.date_range(c5.index.min(), c5.index.max(), freq="5min")).ffill()
bar_open = d15["open"]
first10 = c5f.reindex(d15.index + pd.Timedelta("10min")).values - bar_open.values
last5 = d15["close"].values - c5f.reindex(d15.index + pd.Timedelta("10min")).values
late_rev_bar = pd.Series((np.sign(first10) * np.sign(last5)) < 0, index=d15.index).astype(float)
late_rev = late_rev_bar.rolling(96).mean()
flips1h = (sign15 != sign15.shift(1)).rolling(4).sum()

r1m = c1.pct_change()
rv6h = r1m.rolling(360).std()
rv72h = r1m.rolling(4320).std()
rv_ratio_s = rv6h / rv72h.replace(0, np.nan)
ret24 = (c1 / c1.shift(1440) - 1) * 100
hi7d, lo7d = c1.rolling(10080).max(), c1.rolling(10080).min()
range_pos = (c1 - lo7d) / (hi7d - lo7d).replace(0, np.nan)
dd24 = (c1 / c1.rolling(1440).max() - 1) * 100
h1 = px.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
tr1 = pd.concat([h1["high"] - h1["low"], (h1["high"] - h1["close"].shift(1)).abs(),
                 (h1["low"] - h1["close"].shift(1)).abs()], axis=1).max(axis=1)
atr14 = tr1.ewm(span=14, adjust=False).mean()
atr_ratio_s = atr14 / atr14.rolling(240).mean()

# synthetic bets: decision at T-10.5min for every 15m expiry T
expiries = d15.index + pd.Timedelta("15min")          # bar close times = expiry grid
dec_times = expiries - pd.Timedelta("10.5min")
S = c1.reindex(dec_times, method="ffill")
K = S * (1 - 0.000633)
settle = c1.reindex(expiries, method="ffill")
win = (settle.values >= K.values).astype(float)

bets = pd.DataFrame({"dec": dec_times, "win": win}).dropna()
def at_dec(series):
    return series.reindex(bets["dec"], method="ffill").values
bets["whipsaw_24h"] = pd.Series(whip.values, index=d15.index + pd.Timedelta("15min")).reindex(bets["dec"], method="ffill").values
bets["late_rev_24h"] = pd.Series(late_rev.values, index=d15.index + pd.Timedelta("15min")).reindex(bets["dec"], method="ffill").values
bets["bar_flip_1h"] = pd.Series(flips1h.values, index=d15.index + pd.Timedelta("15min")).reindex(bets["dec"], method="ffill").values
bets["rv_ratio"] = at_dec(rv_ratio_s)
bets["ret_24h"] = at_dec(ret24)
bets["abs_ret_24h"] = np.abs(bets["ret_24h"])
bets["atr_ratio"] = pd.Series(atr_ratio_s.values, index=h1.index + pd.Timedelta("1h")).reindex(bets["dec"], method="ffill").values
bets["range_pos_7d"] = at_dec(range_pos)
bets["dd_24h"] = at_dec(dd24)
bets = bets.dropna()
bets["year"] = bets["dec"].dt.year
bets["day"] = bets["dec"].dt.date
bets = bets[bets["year"] >= 2024]
BE = 0.730
print(f"synthetic bets: {len(bets)}  years {sorted(bets['year'].unique())}  "
      f"overall WR={bets['win'].mean():.4f} vs BE {BE}")
print(bets.groupby("year")["win"].mean().round(4).to_string())

FEATS = ["whipsaw_24h", "late_rev_24h", "bar_flip_1h", "rv_ratio", "ret_24h",
         "abs_ret_24h", "atr_ratio", "range_pos_7d", "dd_24h"]
print("\n=== per-feature quintile WR spread, per year (robust = same sign all 3 years) ===")
rows = []
for f in FEATS:
    line = {"feat": f}
    for y in [2024, 2025, 2026]:
        s = bets[bets["year"] == y]
        q = s[f].quantile([0.2, 0.8])
        lo = s[s[f] <= q[0.2]]["win"].mean()
        hi = s[s[f] >= q[0.8]]["win"].mean()
        line[y] = round(hi - lo, 4)
    line["consistent"] = (np.sign(line[2024]) == np.sign(line[2025]) == np.sign(line[2026]))
    rows.append(line)
res = pd.DataFrame(rows).sort_values(2026)
print(res.to_string(index=False))

# day-clustered check for the strongest consistent features
print("\n=== day-clustered top-vs-bottom quintile (bootstrap over days), consistent feats ===")
rng = np.random.default_rng(5)
for f in [r["feat"] for r in rows if r["consistent"]]:
    for y in [2024, 2025, 2026]:
        s = bets[bets["year"] == y]
        q = s[f].quantile([0.2, 0.8])
        d_lo = s[s[f] <= q[0.2]].groupby("day")["win"].mean()
        d_hi = s[s[f] >= q[0.8]].groupby("day")["win"].mean()
        boots = [d_hi.sample(frac=1, replace=True, random_state=i).mean()
                 - d_lo.sample(frac=1, replace=True, random_state=i).mean() for i in range(1000)]
        arr = np.array(boots)
        p = min(np.mean(arr <= 0), np.mean(arr >= 0))
        print(f"  {f:14s} {y}: spread={d_hi.mean()-d_lo.mean():+.4f}  P(two-sided)={2*p:.4f}")
bets.to_csv(f"{OUT}/synthetic_yes_bets.csv", index=False)
print("DONE_S19")
