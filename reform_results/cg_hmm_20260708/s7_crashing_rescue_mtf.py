"""
S7 -- Comprehensive rescue search on hmm_pup_v3_crashing_no_gate's blocked
population (92 unique real contracts, 07-06->07-09), specifically over:
  offset, price-change (multi-TF), Bollinger %B (multi-TF), RSI (multi-TF),
  stochastic (multi-TF), Donchian position (multi-TF), Keltner %B (multi-TF).
Timeframes: 15m, 1h, 4h, 1d (this is an HOURLY-contract gate).

Zero-lookahead discipline throughout (per feedback_zero_lookahead_reconstruction):
cutoff = ts - bar_duration; last bar with open <= cutoff is used. Ticker-level
(not row-level) bootstrap given the known ~13x pseudo-replication in
blocked_trades.csv. Any rescue candidate is checked for leakage into the two
KNOWN real bad events (07-07 09:00-10:00, 07-08 07:00-08:00) -- a real rescue
must not let those back in.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(311)
OUT = "reform_results/cg_hmm_20260708"

base = pd.read_csv(f"{OUT}/crashing_no_full_features.csv", low_memory=False)
base["logged_at"] = pd.to_datetime(base["logged_at"], utc=True, errors="coerce")
print(f"base population: n={len(base)} unique tickers")

df1m = pd.read_parquet(sorted(__import__("pathlib").Path(".").glob(
    "data/binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]).sort_index()
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
FRAMES = {
    "15m": df1m.resample("15min").agg(AGG).dropna(),
    "1h":  df1m.resample("1h").agg(AGG).dropna(),
    "4h":  df1m.resample("4h").agg(AGG).dropna(),
    "1d":  df1m.resample("1D").agg(AGG).dropna(),
}
DUR = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def bars_before(df, ts, frame_min, n_needed):
    cutoff = ts - pd.Timedelta(minutes=frame_min)
    idx = df.index.searchsorted(cutoff, side="right") - 1
    if idx < 20:
        return None
    return df.iloc[max(0, idx - n_needed):idx + 1]


def chg_at(df, ts, fm):
    b = bars_before(df, ts, fm, 2)
    if b is None or len(b) < 2:
        return np.nan
    return float((b["close"].iloc[-1] / b["close"].iloc[-2] - 1) * 100)


def bollinger_at(df, ts, fm, n=20, k=2.0):
    b = bars_before(df, ts, fm, n + 5)
    if b is None or len(b) < n:
        return np.nan, np.nan
    c = b["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    width = float((upper - lower).iloc[-1])
    if width <= 0:
        return np.nan, np.nan
    last = float(c.iloc[-1])
    pct_b = (last - float(lower.iloc[-1])) / width
    bandwidth = width / float(ma.iloc[-1]) if ma.iloc[-1] else np.nan
    return pct_b, bandwidth


def rsi_at(df, ts, fm, n=14):
    b = bars_before(df, ts, fm, n * 4)
    if b is None or len(b) < n + 2:
        return np.nan
    d = b["close"].diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def stoch_at(df, ts, fm, n=14):
    b = bars_before(df, ts, fm, n + 5)
    if b is None or len(b) < n:
        return np.nan
    lo, hi = b["low"].rolling(n).min(), b["high"].rolling(n).max()
    sk = ((b["close"] - lo) / (hi - lo).replace(0, np.nan)) * 100
    return float(sk.iloc[-1])


def donchian_at(df, ts, fm, n=20):
    b = bars_before(df, ts, fm, n + 5)
    if b is None or len(b) < n:
        return np.nan
    hi, lo = b["high"].rolling(n).max().iloc[-1], b["low"].rolling(n).min().iloc[-1]
    last = float(b["close"].iloc[-1])
    return (last - lo) / (hi - lo) if hi > lo else np.nan


def keltner_at(df, ts, fm, n=10, atr_n=14, k=1.5):
    b = bars_before(df, ts, fm, max(n, atr_n) + 10)
    if b is None or len(b) < max(n, atr_n) + 2:
        return np.nan
    c, h, l = b["close"], b["high"], b["low"]
    ema = c.ewm(span=n, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_n, adjust=False).mean()
    upper, lower = ema + k * atr, ema - k * atr
    width = float((upper - lower).iloc[-1])
    if width <= 0:
        return np.nan
    return (float(c.iloc[-1]) - float(lower.iloc[-1])) / width


print("reconstructing all indicator families x timeframes (zero-lookahead)...")
for tf, df in FRAMES.items():
    fm = DUR[tf]
    base[f"chg_{tf}"] = base["logged_at"].apply(lambda ts: chg_at(df, ts, fm))
    bb = base["logged_at"].apply(lambda ts: bollinger_at(df, ts, fm))
    base[f"bb_pctb_{tf}"] = bb.apply(lambda x: x[0])
    base[f"bb_width_{tf}"] = bb.apply(lambda x: x[1])
    base[f"rsi_{tf}"] = base["logged_at"].apply(lambda ts: rsi_at(df, ts, fm))
    base[f"stoch_{tf}"] = base["logged_at"].apply(lambda ts: stoch_at(df, ts, fm))
    base[f"donch_{tf}"] = base["logged_at"].apply(lambda ts: donchian_at(df, ts, fm))
    base[f"kc_pctb_{tf}"] = base["logged_at"].apply(lambda ts: keltner_at(df, ts, fm))
    print(f"  {tf} done")

RECON_COLS = [c for c in base.columns if any(c.startswith(p) for p in
              ["chg_", "bb_pctb_", "bb_width_", "rsi_", "stoch_", "donch_", "kc_pctb_"])]
print(f"\n{len(RECON_COLS)} reconstructed columns:")
for c in RECON_COLS:
    print(f"  {c}: {base[c].notna().sum()}/{len(base)} non-null")

base.to_csv(f"{OUT}/crashing_no_mtf_indicators.csv", index=False)
print("\nsaved crashing_no_mtf_indicators.csv")
print("DONE_S7")
