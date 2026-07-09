"""
S5 -- User-directed extension of the within-bucket rescue battery:
EMA stack + individual MA distances/slopes, ADX (+DI spread), MACD
(hist, hist-slope, line sign), and rolling VWAP (distance, velocity,
above/below flag) -- each at 5m / 15m / 1h / 4h. Zero-lookahead
(completed bars only), episode-clustered bootstrap, zero streak
leakage required, same bar as s4.
"""
import warnings
import pathlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1039)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

bucket = pd.read_csv(f"{OUT}/bucket_exhaustive.csv", low_memory=False)
bucket["logged_at_p"] = pd.to_datetime(bucket["logged_at_p"], utc=True)
bucket["week"] = bucket["logged_at_p"].dt.to_period("W-FRI").astype(str)
print(f"bucket: n={len(bucket)} edge={bucket['tedge'].mean():+.4f}")

p1m = sorted(pathlib.Path("data").glob("binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2026-03-15"]     # extra history for 4h EMA50/ADX warmup
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
FR = {"5m": (df1m.resample("5min").agg(AGG).dropna(), 5),
      "15m": (df1m.resample("15min").agg(AGG).dropna(), 15),
      "1h": (df1m.resample("1h").agg(AGG).dropna(), 60),
      "4h": (df1m.resample("4h").agg(AGG).dropna(), 240)}


def bars_before(df, ts, fm, n):
    cutoff = ts - pd.Timedelta(minutes=fm)
    i = df.index.searchsorted(cutoff, side="right") - 1
    if i < 60:
        return None
    return df.iloc[max(0, i - n):i + 1]


def feats_at(df, ts, fm):
    out = {k: np.nan for k in ["ema9d", "ema20d", "ema50d", "stack", "ema20slope",
                               "adx", "didiff", "macdh", "macdslope", "macdsign",
                               "vwapd", "vwapvel", "abovevwap"]}
    b = bars_before(df, ts, fm, 240)
    if b is None or len(b) < 80:
        return out
    c, h, l, v = b["close"], b["high"], b["low"], b["volume"]
    px = float(c.iloc[-1])
    e9 = c.ewm(span=9, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    out["ema9d"] = (px / float(e9.iloc[-1]) - 1) * 100
    out["ema20d"] = (px / float(e20.iloc[-1]) - 1) * 100
    out["ema50d"] = (px / float(e50.iloc[-1]) - 1) * 100
    if e9.iloc[-1] > e20.iloc[-1] > e50.iloc[-1]:
        out["stack"] = 1
    elif e9.iloc[-1] < e20.iloc[-1] < e50.iloc[-1]:
        out["stack"] = -1
    else:
        out["stack"] = 0
    out["ema20slope"] = (float(e20.iloc[-1]) / float(e20.iloc[-6]) - 1) * 100 if len(e20) > 6 else np.nan
    # ADX(14), Wilder smoothing
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=b.index).ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=b.index).ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["adx"] = float(adx.iloc[-1])
    out["didiff"] = float((pdi - mdi).iloc[-1])
    # MACD(12,26,9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    out["macdh"] = float(hist.iloc[-1]) / px * 100          # normalized to price pct
    out["macdslope"] = float(hist.iloc[-1] - hist.iloc[-4]) / px * 100 if len(hist) > 4 else np.nan
    out["macdsign"] = 1 if macd.iloc[-1] > 0 else -1
    # rolling VWAP(20)
    tp = (h + l + c) / 3
    ctv = (tp * v).rolling(20, min_periods=20).sum()
    cv = v.rolling(20, min_periods=20).sum()
    vw = ctv / cv.replace(0, np.nan)
    vd = (c - vw) / vw.replace(0, np.nan) * 100
    out["vwapd"] = float(vd.iloc[-1])
    out["vwapvel"] = float(vd.iloc[-1] - vd.iloc[-4]) if len(vd) > 4 and not pd.isna(vd.iloc[-4]) else np.nan
    out["abovevwap"] = 1 if vd.iloc[-1] > 0 else -1
    return out


print("reconstructing EMA/ADX/MACD/VWAP at 4 TFs (zero-lookahead)...")
for tf, (df, fm) in FR.items():
    res = bucket["logged_at_p"].apply(lambda ts: feats_at(df, ts, fm))
    for k in ["ema9d", "ema20d", "ema50d", "stack", "ema20slope", "adx", "didiff",
              "macdh", "macdslope", "macdsign", "vwapd", "vwapvel", "abovevwap"]:
        bucket[f"m_{k}_{tf}"] = res.apply(lambda d: d[k])
    print(f"  {tf} done ({bucket[f'm_adx_{tf}'].notna().sum()}/{len(bucket)})")

bucket.to_csv(f"{OUT}/bucket_ma_battery.csv", index=False)

# ── sweep ─────────────────────────────────────────────────────────────────
streak_mask = bucket["contract_ticker"].isin(STREAK)


def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()


NUMS = [f"m_{k}_{tf}" for k in ["ema9d", "ema20d", "ema50d", "ema20slope", "adx", "didiff",
                                "macdh", "macdslope", "vwapd", "vwapvel"]
        for tf in ["5m", "15m", "1h", "4h"]]
CATS = [f"m_{k}_{tf}" for k in ["stack", "macdsign", "abovevwap"] for tf in ["5m", "15m", "1h", "4h"]]

results, n_tests = [], 0
for feat in NUMS:
    col = pd.to_numeric(bucket[feat], errors="coerce")
    if col.notna().sum() < 200 or col.dropna().nunique() < 6:
        print(f"  SKIP {feat} coverage={col.notna().sum()}")
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for lab, mk in [(f">= {th:.4g}", col >= th), (f"< {th:.4g}", col < th)]:
            n_tests += 1
            d = bucket[mk.fillna(False)]
            if len(d) < 80 or len(bucket) - len(d) < 80 or d["tedge"].mean() < 0.005:
                continue
            ne, ee, p_neg = ep_stats(d)
            wk = d.groupby("week")["tedge"].mean()
            results.append({"feature": feat, "split": lab, "n": len(d), "eps": ne,
                            "edge": d["tedge"].mean(), "ep_edge": ee, "P_neg": p_neg,
                            "wk_pos": float((wk > 0).mean()),
                            "streak_leak": int((mk.fillna(False) & streak_mask).sum()),
                            "pnl": d["would_pnl"].sum()})
for feat in CATS:
    col = bucket[feat]
    if col.notna().sum() < 200:
        continue
    for val in col.dropna().unique():
        n_tests += 1
        mk = (col == val).fillna(False)
        d = bucket[mk]
        if len(d) < 60 or d["tedge"].mean() < 0.005:
            continue
        ne, ee, p_neg = ep_stats(d)
        wk = d.groupby("week")["tedge"].mean()
        results.append({"feature": feat, "split": f"== {val}", "n": len(d), "eps": ne,
                        "edge": d["tedge"].mean(), "ep_edge": ee, "P_neg": p_neg,
                        "wk_pos": float((wk > 0).mean()),
                        "streak_leak": int((mk & streak_mask).sum()),
                        "pnl": d["would_pnl"].sum()})

print(f"\nTOTAL tests this pass: {n_tests}")
rf = pd.DataFrame(results)
print(f"positive-edge subsets: {len(rf)}")
if len(rf):
    surv = rf[(rf["P_neg"] <= 0.05) & (rf["streak_leak"] == 0) & (rf["wk_pos"] >= 0.6) & (rf["n"] >= 80)]
    print(f"\nSURVIVORS: {len(surv)}")
    if len(surv):
        print(surv.sort_values("ep_edge", ascending=False).round(4).to_string(index=False))
    print("\nnear-misses (P<=0.15), top 12:")
    print(rf[rf["P_neg"] <= 0.15].sort_values("ep_edge", ascending=False).head(12).round(4).to_string(index=False))
rf.to_csv(f"{OUT}/ma_battery_results.csv", index=False)
print("DONE_S5")
