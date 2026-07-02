"""
validate_vol_signals.py — Test whether signals correlate with realized
next-hour volatility rather than (or in addition to) direction.

Realized vol metrics:
  abs_return   : |log(close_t+1 / close_t)|   — magnitude of next-hour move
  hl_range     : (high_t+1 - low_t+1) / close_t — high-low range of next bar

For OTM Kalshi contracts, vol magnitude matters more than direction.
A strike 0.3% above spot is only reachable if next-hour vol is large enough.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    h1_files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_2024-01-01_*.parquet"))
    m1_files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1m_2024-01-01_*.parquet"))
    df_1h = pd.read_parquet(h1_files[-1])
    df_1m = pd.read_parquet(m1_files[-1])
    for df in (df_1h, df_1m):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df_1h.columns = df_1h.columns.str.lower()
    df_1m.columns = df_1m.columns.str.lower()
    return df_1h, df_1m


# ── indicators ───────────────────────────────────────────────────────────────

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _stoch_k(h, l, c, k=14):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100


def _atr(h, l, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()


def _keltner_pct(h, l, c, span=20, mult=2):
    ema = c.ewm(span=span, adjust=False).mean()
    atr = _atr(h, l, c, span)
    up = ema + mult * atr
    dn = ema - mult * atr
    return (c - dn) / (up - dn).replace(0, np.nan), dn, up


def _bb_pct(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    up = mid + 2 * std
    dn = mid - 2 * std
    return (c - dn) / (up - dn).replace(0, np.nan)


def _wpr(h, l, c, p=14):
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def _macd_cross(c, f=12, s=26, sig=9):
    ema_f = c.ewm(span=f, adjust=False).mean()
    ema_s = c.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    xup = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    xdn = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    state = pd.Series(0, index=c.index)
    state[xup] = 1
    state[xdn] = -1
    for sh in [1, 2]:
        state[(xup.shift(sh).fillna(False)) & (state == 0)] = 1
        state[(xdn.shift(sh).fillna(False)) & (state == 0)] = -1
    return state


def _vwap_dev(df_1m, ts_1h):
    date_1m = df_1m.index.normalize()
    tpv = df_1m["close"] * df_1m["volume"]
    cum_tpv = tpv.groupby(date_1m).cumsum()
    cum_vol = df_1m["volume"].groupby(date_1m).cumsum()
    vwap_1m = cum_tpv / cum_vol.replace(0, np.nan)
    vwap_1h = vwap_1m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    close_1h = df_1m["close"].resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    return (close_1h - vwap_1h) / vwap_1h.replace(0, np.nan)


# ── reporting ────────────────────────────────────────────────────────────────

def report_vol(name, signal, abs_ret, hl_range, median_ret, median_hl, min_n=100):
    """
    For each discrete signal value, show:
      - median |return| next hour vs overall median
      - median HL range next hour vs overall median
      - % of bars where |return| > 0.3% (rough OTM threshold)
    """
    df = pd.DataFrame({"sig": signal, "ar": abs_ret, "hl": hl_range}).dropna()
    vals = sorted(df["sig"].unique())
    print(f"\n  {name}:")
    print(f"  {'Value':>8}  {'N':>6}  {'Med|ret|':>9}  {'vs base':>7}  {'Med HL%':>8}  {'vs base':>7}  {'>0.3%':>6}  {'>0.5%':>6}")
    for v in vals:
        sub = df[df["sig"] == v]
        if len(sub) < min_n:
            continue
        med_r = sub["ar"].median() * 100
        med_h = sub["hl"].median() * 100
        pct_03 = (sub["ar"] > 0.003).mean() * 100
        pct_05 = (sub["ar"] > 0.005).mean() * 100
        dr = med_r - median_ret * 100
        dh = med_h - median_hl * 100
        flag = "  <-- HIGH VOL" if dr > 0.02 else ("  <-- LOW VOL" if dr < -0.02 else "")
        print(f"  {v:>8}  {len(sub):>6}  {med_r:>8.4f}%  {dr:>+6.3f}%  {med_h:>8.4f}%  {dh:>+6.3f}%  {pct_03:>5.1f}%  {pct_05:>5.1f}%{flag}")


def report_vol_bins(name, signal, abs_ret, hl_range, edges, median_ret, median_hl, min_n=100):
    df = pd.DataFrame({"sig": signal, "ar": abs_ret, "hl": hl_range}).dropna()
    print(f"\n  {name}:")
    print(f"  {'Bucket':>18}  {'N':>6}  {'Med|ret|':>9}  {'vs base':>7}  {'Med HL%':>8}  {'vs base':>7}  {'>0.3%':>6}  {'>0.5%':>6}")
    all_edges = list(edges) + [np.inf]
    for i in range(len(all_edges) - 1):
        lo, hi = all_edges[i], all_edges[i+1]
        sub = df[(df["sig"] >= lo) & (df["sig"] < hi)]
        if len(sub) < min_n:
            continue
        med_r = sub["ar"].median() * 100
        med_h = sub["hl"].median() * 100
        pct_03 = (sub["ar"] > 0.003).mean() * 100
        pct_05 = (sub["ar"] > 0.005).mean() * 100
        dr = med_r - median_ret * 100
        dh = med_h - median_hl * 100
        flag = "  <-- HIGH VOL" if dr > 0.02 else ("  <-- LOW VOL" if dr < -0.02 else "")
        bucket = f"[{lo:.2f},{hi:.2f})" if hi != np.inf else f"[{lo:.2f},+inf)"
        print(f"  {bucket:>18}  {len(sub):>6}  {med_r:>8.4f}%  {dr:>+6.3f}%  {med_h:>8.4f}%  {dh:>+6.3f}%  {pct_03:>5.1f}%  {pct_05:>5.1f}%{flag}")


def main():
    print("Loading data...")
    df_1h, df_1m = load_data()
    ts = df_1h.index
    c = df_1h["close"]
    h = df_1h["high"]
    l = df_1h["low"]
    v = df_1h["volume"]

    # Next-hour realized vol metrics
    log_ret     = np.log(c / c.shift(1))
    abs_ret     = log_ret.shift(-1).abs()          # |next-hour log return|
    hl_range    = ((h.shift(-1) - l.shift(-1)) / c).shift(0)  # next-hour HL range / current close

    # Remove last bar (no next bar)
    abs_ret.iloc[-1] = np.nan
    hl_range.iloc[-1] = np.nan

    median_ret = abs_ret.dropna().median()
    median_hl  = hl_range.dropna().median()
    pct_03_base = (abs_ret.dropna() > 0.003).mean() * 100
    pct_05_base = (abs_ret.dropna() > 0.005).mean() * 100

    print(f"\n  Bars: {abs_ret.dropna().shape[0]}")
    print(f"  Median |return| next hour : {median_ret*100:.4f}%")
    print(f"  Median HL range next hour : {median_hl*100:.4f}%")
    print(f"  % bars with |ret| > 0.3%  : {pct_03_base:.1f}%")
    print(f"  % bars with |ret| > 0.5%  : {pct_05_base:.1f}%")

    # ── derive signals ───────────────────────────────────────────────────────
    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    df_15m = df_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    # z-score of current 1h move
    roll_vol_24 = log_ret.rolling(24).std()
    z_score = log_ret / roll_vol_24.replace(0, np.nan)

    # Rolling realized vol (lookback periods)
    rv_1h  = log_ret.abs()                             # current bar's |return|
    rv_6h  = log_ret.rolling(6).std()                  # 6h realized vol
    rv_24h = log_ret.rolling(24).std()                 # 24h realized vol
    rv_48h = log_ret.rolling(48).std()                 # 48h realized vol

    # RSI
    rsi_1h = _rsi(c, 14)
    rsi_4h = _rsi(df_4h["close"], 14).reindex(ts, method="ffill")

    # Stochastic
    stk_1h  = _stoch_k(h, l, c, 14)
    stk_4h  = _stoch_k(df_4h["high"], df_4h["low"], df_4h["close"], 14).reindex(ts, method="ffill")
    stk_15m = _stoch_k(df_15m["high"], df_15m["low"], df_15m["close"], 14)
    stk_15m = stk_15m.resample("1h", origin="start_day").last().reindex(ts, method="ffill")

    # BB
    bb_1h = _bb_pct(c, 20)
    bb_4h = _bb_pct(df_4h["close"], 20).reindex(ts, method="ffill")

    # Keltner
    kc_1h_pct, _, _ = _keltner_pct(h, l, c, 20, 2)
    kc_4h_pct, _, _ = _keltner_pct(df_4h["high"], df_4h["low"], df_4h["close"], 20, 2)
    kc_4h_pct = kc_4h_pct.reindex(ts, method="ffill")

    # W%R
    wpr_1h = _wpr(h, l, c, 14)
    wpr_4h = _wpr(df_4h["high"], df_4h["low"], df_4h["close"], 14).reindex(ts, method="ffill")

    # MACD 4h
    macd_4h = _macd_cross(df_4h["close"]).reindex(ts, method="ffill").fillna(0)

    # VWAP
    print("Computing VWAP...")
    vwap_dev = _vwap_dev(df_1m, ts)

    # EMA alignment
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema_align = pd.Series(0, index=ts)
    ema_align[(ema9 > ema21) & (ema21 > ema50)] = 1
    ema_align[(ema9 < ema21) & (ema21 < ema50)] = -1

    # Volume ratio (current vol vs 20-bar MA)
    vol_ratio = v / v.rolling(20).mean()

    # ATR (current vs 24h MA of ATR — measures vol regime)
    atr_1h = _atr(h, l, c, 14)
    atr_ratio = atr_1h / atr_1h.rolling(24).mean()

    # Composite trend score
    stk4_vote = ((stk_4h > 80).astype(int) - (stk_4h < 20).astype(int))
    vol_up = (v > v.rolling(20).mean() * 1.5) & (c > c.shift(1))
    vol_dn = (v > v.rolling(20).mean() * 1.5) & (c < c.shift(1))
    vol_vote = vol_up.astype(int) - vol_dn.astype(int)
    macd_vote = macd_4h
    bb4_vote = (bb_4h > 0.80).astype(int) - (bb_4h < 0.20).astype(int)
    kc4_vote = (kc_4h_pct > 0.85).astype(int) - (kc_4h_pct < 0.15).astype(int)
    wpr4_vote = (wpr_4h > -20).astype(int) - (wpr_4h < -80).astype(int)
    trend_score = (stk4_vote + vol_vote + macd_vote + bb4_vote + kc4_vote + wpr4_vote).clip(-6, 6)

    # ── reports ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("REALIZED VOLATILITY CORRELATIONS")
    print("Med|ret| = median |next-hour log return|   vs base = vs overall median")
    print(">0.3% / >0.5% = % of bars where |next-hour move| exceeds that threshold")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("CURRENT BAR VOLATILITY (does recent vol predict next-hour vol?)")
    print("-" * 70)
    report_vol_bins("Current |return| (rv_1h)", rv_1h * 100, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("6h realized vol (annualized proxy)", rv_6h * 100, abs_ret, hl_range,
                    edges=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("24h realized vol", rv_24h * 100, abs_ret, hl_range,
                    edges=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("ATR ratio (current ATR / 24h mean ATR)", atr_ratio, abs_ret, hl_range,
                    edges=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
                    median_ret=median_ret, median_hl=median_hl)

    print("\n" + "-" * 70)
    print("Z-SCORE (magnitude of current move vs rolling vol)")
    print("-" * 70)
    report_vol_bins("z_score", z_score, abs_ret, hl_range,
                    edges=[-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("|z_score| (magnitude only)", z_score.abs(), abs_ret, hl_range,
                    edges=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                    median_ret=median_ret, median_hl=median_hl)

    print("\n" + "-" * 70)
    print("VOLUME SIGNALS")
    print("-" * 70)
    report_vol_bins("Volume ratio (vol / 20-bar MA)", vol_ratio, abs_ret, hl_range,
                    edges=[0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    median_ret=median_ret, median_hl=median_hl)

    print("\n" + "-" * 70)
    print("MOMENTUM / TREND SIGNALS (do they predict vol magnitude?)")
    print("-" * 70)
    report_vol("Trend score (-6 to +6)", trend_score, abs_ret, hl_range,
               median_ret=median_ret, median_hl=median_hl)
    report_vol("EMA alignment", ema_align, abs_ret, hl_range,
               median_ret=median_ret, median_hl=median_hl)
    report_vol("MACD 4h", macd_4h, abs_ret, hl_range,
               median_ret=median_ret, median_hl=median_hl)

    print("\n" + "-" * 70)
    print("OSCILLATORS (overbought/oversold — do extremes predict vol spikes?)")
    print("-" * 70)
    report_vol_bins("RSI 1h", rsi_1h, abs_ret, hl_range,
                    edges=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("RSI 4h", rsi_4h, abs_ret, hl_range,
                    edges=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("Stoch %K 1h", stk_1h, abs_ret, hl_range,
                    edges=[0, 10, 20, 30, 50, 70, 80, 90, 100],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("Stoch %K 4h", stk_4h, abs_ret, hl_range,
                    edges=[0, 10, 20, 30, 50, 70, 80, 90, 100],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("BB %B 1h", bb_1h, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("BB %B 4h", bb_4h, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("Keltner 1h", kc_1h_pct, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("W%R 1h", wpr_1h, abs_ret, hl_range,
                    edges=[-100, -80, -60, -40, -20, -10, 0],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("W%R 4h", wpr_4h, abs_ret, hl_range,
                    edges=[-100, -80, -60, -40, -20, -10, 0],
                    median_ret=median_ret, median_hl=median_hl)

    print("\n" + "-" * 70)
    print("VWAP DEVIATION (distance from VWAP — does it predict vol?)")
    print("-" * 70)
    report_vol_bins("VWAP dev %", vwap_dev * 100, abs_ret, hl_range,
                    edges=[-5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 5],
                    median_ret=median_ret, median_hl=median_hl)
    report_vol_bins("|VWAP dev| (magnitude)", vwap_dev.abs() * 100, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0],
                    median_ret=median_ret, median_hl=median_hl)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Vol predictors ranked by impact on median |next-hour return|")
    print("=" * 70)

    rows = []
    def chk(name, sig, lo, hi):
        df2 = pd.DataFrame({"s": sig, "ar": abs_ret}).dropna()
        sub = df2[(df2["s"] >= lo) & (df2["s"] < hi)]
        if len(sub) < 150:
            return
        delta = sub["ar"].median() - median_ret
        pct_05 = (sub["ar"] > 0.005).mean() * 100
        rows.append((name, f"[{lo},{hi})", len(sub), sub["ar"].median()*100, delta*100, pct_05))

    chk("rv_1h",        rv_1h*100,     0.5,  999)
    chk("rv_1h",        rv_1h*100,     0,    0.1)
    chk("rv_6h",        rv_6h*100,     0.3,  999)
    chk("rv_6h",        rv_6h*100,     0,    0.08)
    chk("rv_24h",       rv_24h*100,    0.25, 999)
    chk("rv_24h",       rv_24h*100,    0,    0.08)
    chk("|z_score|",    z_score.abs(), 2.0,  999)
    chk("|z_score|",    z_score.abs(), 0,    0.5)
    chk("vol_ratio",    vol_ratio,     3.0,  999)
    chk("vol_ratio",    vol_ratio,     0,    0.3)
    chk("atr_ratio",    atr_ratio,     1.5,  999)
    chk("atr_ratio",    atr_ratio,     0,    0.75)
    chk("|VWAP dev|",   vwap_dev.abs()*100, 1.0, 999)
    chk("|VWAP dev|",   vwap_dev.abs()*100, 0,   0.2)
    chk("bb_1h",        bb_1h,         1.0,  999)
    chk("bb_1h",        bb_1h,         0,    0.1)
    chk("stk_1h",       stk_1h,        90,   101)
    chk("stk_1h",       stk_1h,        0,    10)
    chk("rsi_1h",       rsi_1h,        80,   101)
    chk("rsi_1h",       rsi_1h,        0,    20)

    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  {'Signal':<15} {'Bucket':<12} {'N':>6}  {'Med|ret|%':>10}  {'Delta':>8}  {'>0.5%':>6}")
    for name, bucket, n, med, delta, p05 in rows[:20]:
        flag = "  VOL SIGNAL" if abs(delta) > 0.02 else ""
        print(f"  {name:<15} {bucket:<12} {n:>6}  {med:>9.4f}%  {delta:>+7.4f}%  {p05:>5.1f}%{flag}")


if __name__ == "__main__":
    main()
