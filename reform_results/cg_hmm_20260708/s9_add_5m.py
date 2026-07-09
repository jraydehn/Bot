"""
S9 -- Add 5m timeframe (missed in s7/s8) to the crashing_no_gate rescue search.
Same zero-lookahead discipline, same 92 unique tickers, same 7 indicator
families. Then re-sweep including 5m alongside the existing 15m/1h/4h/1d
columns, checking overlap with the already-found stoch_15m-based rescue.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(401)
OUT = "reform_results/cg_hmm_20260708"

base = pd.read_csv(f"{OUT}/crashing_no_mtf_indicators.csv", low_memory=False)
base["logged_at"] = pd.to_datetime(base["logged_at"], utc=True, errors="coerce")
print(f"base population: n={len(base)}")

df1m = pd.read_parquet(sorted(__import__("pathlib").Path(".").glob(
    "data/binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]).sort_index()
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df5m = df1m.resample("5min").agg(AGG).dropna()


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


print("reconstructing 5m indicators (zero-lookahead)...")
base["chg_5m"] = base["logged_at"].apply(lambda ts: chg_at(df5m, ts, 5))
bb = base["logged_at"].apply(lambda ts: bollinger_at(df5m, ts, 5))
base["bb_pctb_5m"] = bb.apply(lambda x: x[0])
base["bb_width_5m"] = bb.apply(lambda x: x[1])
base["rsi_5m"] = base["logged_at"].apply(lambda ts: rsi_at(df5m, ts, 5))
base["stoch_5m"] = base["logged_at"].apply(lambda ts: stoch_at(df5m, ts, 5))
base["donch_5m"] = base["logged_at"].apply(lambda ts: donchian_at(df5m, ts, 5))
base["kc_pctb_5m"] = base["logged_at"].apply(lambda ts: keltner_at(df5m, ts, 5))

M5_COLS = ["chg_5m", "bb_pctb_5m", "bb_width_5m", "rsi_5m", "stoch_5m", "donch_5m", "kc_pctb_5m"]
for c in M5_COLS:
    print(f"  {c}: {base[c].notna().sum()}/{len(base)}")

base.to_csv(f"{OUT}/crashing_no_mtf_indicators_with5m.csv", index=False)

# ── sweep 5m alone against the same rigor ──────────────────────────────────
b = pd.read_csv("results/blocked_trades.csv", low_memory=False)
b["logged_at"] = pd.to_datetime(b["logged_at"], format="mixed", utc=True, errors="coerce")
sub = b[b["gate_name"] == "hmm_pup_v3_crashing_no_gate"].copy()
sub["resolved_yes"] = pd.to_numeric(sub["resolved_yes"], errors="coerce")
sub["pm"] = pd.to_numeric(sub["pm"], errors="coerce")
sub["would_pnl"] = pd.to_numeric(sub["would_pnl"], errors="coerce")
sub = sub.dropna(subset=["resolved_yes", "pm", "would_pnl"]).copy()
sub["won"] = sub["resolved_yes"] == 0
sub["be"] = 1 - sub["pm"]
CLUSTER1 = (sub["logged_at"] >= pd.Timestamp("2026-07-07 09:00", tz="UTC")) & (sub["logged_at"] < pd.Timestamp("2026-07-07 10:00", tz="UTC"))
CLUSTER2 = (sub["logged_at"] >= pd.Timestamp("2026-07-08 07:00", tz="UTC")) & (sub["logged_at"] < pd.Timestamp("2026-07-08 08:00", tz="UTC"))
sub["bad_cluster"] = CLUSTER1 | CLUSTER2

for col in M5_COLS:
    m = base.set_index("contract_ticker")[col].to_dict()
    sub[col] = sub["contract_ticker"].map(m)

# also pull in stoch_15m so we can check overlap with the previously found rescue
m2 = base.set_index("contract_ticker")["stoch_15m"].to_dict()
sub["stoch_15m"] = sub["contract_ticker"].map(m2)


def ticker_boot(df, n_boot=5000):
    nt = df["contract_ticker"].nunique()
    if nt < 5:
        return nt, np.nan, np.nan
    pt = df.groupby("contract_ticker").apply(
        lambda g: g["won"].astype(float).mean() - g["be"].mean(), include_groups=False)
    e = pt.values
    means = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)])
    return nt, means.mean(), (means <= 0).mean()


found = []
for feat in M5_COLS:
    col = sub[feat]
    if col.notna().sum() < 200:
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for d, mask in [(">=", col >= th), ("<", col < th)]:
            s = sub[mask.fillna(False)]
            n_t = s["contract_ticker"].nunique()
            if n_t < 15 or (sub["contract_ticker"].nunique() - n_t) < 15:
                continue
            leak = s["bad_cluster"].sum()
            wr, be = s["won"].mean(), s["be"].mean()
            found.append({"feature": feat, "split": f"{d}{th:.4g}(q{q:.1f})", "rows": len(s),
                         "tickers": n_t, "edge": wr - be, "leak": leak, "pnl": s["would_pnl"].sum()})

fd = pd.DataFrame(found)
print(f"\n5m candidates with >=15 tickers, zero leak, positive edge:")
resc5 = fd[(fd["edge"] > 0) & (fd["leak"] == 0) & (fd["tickers"] >= 15)].sort_values("edge", ascending=False)
print(resc5.head(15).round(4).to_string(index=False))

print(f"\nbootstrap on top 5m candidates:")
for _, r in resc5.head(5).iterrows():
    feat, split = r["feature"], r["split"]
    col = sub[feat]
    if split.startswith(">="):
        thv = float(split.split("(")[0][2:]); mask = col >= thv
    else:
        thv = float(split.split("(")[0][1:]); mask = col < thv
    s = sub[mask.fillna(False)]
    n_t, edge, p = ticker_boot(s)
    print(f"  {feat} {split}: rows={len(s)} tickers={n_t} edge={edge:+.4f} P(<=0)={p:.4f} pnl=${s['would_pnl'].sum():+.2f}")

# overlap with the previously found stoch_15m rescue
prev = set(sub[sub["stoch_15m"] >= 85]["contract_ticker"])
if len(resc5):
    top5m = resc5.iloc[0]
    feat, split = top5m["feature"], top5m["split"]
    col = sub[feat]
    if split.startswith(">="):
        thv = float(split.split("(")[0][2:]); mask = col >= thv
    else:
        thv = float(split.split("(")[0][1:]); mask = col < thv
    new_set = set(sub[mask.fillna(False)]["contract_ticker"])
    print(f"\ntop 5m candidate ({feat} {split}) tickers: {len(new_set)}")
    print(f"overlap with stoch_15m>=85 rescue ({len(prev)} tickers): {len(new_set & prev)}")
    print(f"NEW tickers not in the 15m rescue: {new_set - prev}")

print("\nDONE_S9")
