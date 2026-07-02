"""
validate_signals.py — Historical validation of individual signals against
next-hour BTC price direction.

For each 1h bar in the full historical dataset, computes every OHLCV-derived
signal independently and measures whether price closed higher in the next hour.

Usage:
    python3 validate_signals.py
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"
BASELINE = 0.504  # BTC 1h upward drift rate


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper indicators
# ─────────────────────────────────────────────────────────────────────────────

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
    w = (up - dn).replace(0, np.nan)
    return (c - dn) / w, dn, up


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


def _dc_pct(h, l, n=20):
    dc_h = h.rolling(n).max()
    dc_l = l.rolling(n).min()
    return (l - dc_l) / (dc_h - dc_l).replace(0, np.nan)  # 0=near low, 1=near high


def _vwap_dev(df_1m, ts_1h):
    """Daily-reset VWAP deviation at 1h resolution."""
    date_1m = df_1m.index.normalize()
    tpv = df_1m["close"] * df_1m["volume"]
    cum_tpv = tpv.groupby(date_1m).cumsum()
    cum_vol = df_1m["volume"].groupby(date_1m).cumsum()
    vwap_1m = cum_tpv / cum_vol.replace(0, np.nan)
    vwap_1h = vwap_1m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    close_1h = df_1m["close"].resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    return (close_1h - vwap_1h) / vwap_1h.replace(0, np.nan)


def _ema_alignment(c, spans=(9, 21, 50)):
    emas = [c.ewm(span=s, adjust=False).mean() for s in spans]
    bullish = (emas[0] > emas[1]) & (emas[1] > emas[2])
    bearish = (emas[0] < emas[1]) & (emas[1] < emas[2])
    result = pd.Series(0, index=c.index)
    result[bullish] = 1
    result[bearish] = -1
    return result


def _structure_bias_rolling(df_15m, window=90, pivot_lb=3):
    """
    Compute structure_bias at each 15m bar using the same HH/HL, LH/LL logic
    as market_structure.py, then resample to 1h.
    Uses a rolling window for efficiency.
    """
    highs = df_15m["high"].values
    lows  = df_15m["low"].values
    n     = len(highs)
    bias  = np.zeros(n)

    for i in range(window, n):
        seg_h = highs[max(0, i - window): i + 1]
        seg_l = lows[max(0, i - window):  i + 1]
        m = len(seg_h)
        sh, sl = [], []
        for j in range(pivot_lb, m - pivot_lb):
            if seg_h[j] > max(seg_h[j - pivot_lb:j]) and seg_h[j] > max(seg_h[j + 1:j + pivot_lb + 1]):
                sh.append(seg_h[j])
            if seg_l[j] < min(seg_l[j - pivot_lb:j]) and seg_l[j] < min(seg_l[j + 1:j + pivot_lb + 1]):
                sl.append(seg_l[j])
        if len(sh) >= 2 and len(sl) >= 2:
            h_asc = sh[-1] > sh[-2]
            l_asc = sl[-1] > sl[-2]
            if h_asc and l_asc:
                bias[i] = 1
            elif not h_asc and not l_asc:
                bias[i] = -1

    s = pd.Series(bias, index=df_15m.index)
    return s.resample("1h", origin="start_day").last()


# ─────────────────────────────────────────────────────────────────────────────
# Report helper
# ─────────────────────────────────────────────────────────────────────────────

def report(name, signal, label, min_n=50):
    """Print win rate by signal bucket. signal and label must be aligned Series."""
    combined = pd.DataFrame({"sig": signal, "up": label}).dropna()
    vals = sorted(combined["sig"].unique())
    print(f"\n  {name}:")
    print(f"  {'Value':>8}  {'N':>6}  {'Up%':>6}  {'Edge':>7}  bar")
    for v in vals:
        subset = combined[combined["sig"] == v]
        n = len(subset)
        if n < min_n:
            continue
        up_pct = subset["up"].mean() * 100
        edge = up_pct - BASELINE * 100
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        print(f"  {v:>8}  {n:>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")


def report_bins(name, signal, label, edges, min_n=50):
    """Print win rate for continuous signal bucketed by edges."""
    combined = pd.DataFrame({"sig": signal, "up": label}).dropna()
    print(f"\n  {name}:")
    print(f"  {'Bucket':>18}  {'N':>6}  {'Up%':>6}  {'Edge':>7}  bar")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        subset = combined[(combined["sig"] >= lo) & (combined["sig"] < hi)]
        n = len(subset)
        if n < min_n:
            continue
        up_pct = subset["up"].mean() * 100
        edge = up_pct - BASELINE * 100
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        print(f"  [{lo:>7.3f},{hi:>7.3f})  {n:>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")
    # tail bucket
    lo = edges[-1]
    subset = combined[combined["sig"] >= lo]
    if len(subset) >= min_n:
        up_pct = subset["up"].mean() * 100
        edge = up_pct - BASELINE * 100
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        print(f"  [{lo:>7.3f},   +inf)  {len(subset):>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading parquet data...")
    df_1h, df_1m = load_data()
    print(f"  1h bars: {len(df_1h)}  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
    print(f"  1m bars: {len(df_1m)}")

    ts = df_1h.index
    c1h = df_1h["close"]
    h1h = df_1h["high"]
    l1h = df_1h["low"]
    v1h = df_1h["volume"]

    # Next-hour label: 1 if close goes up, 0 if down
    label = (c1h.shift(-1) > c1h).astype(float)
    label.iloc[-1] = np.nan  # last bar has no next bar

    n_total = label.dropna().shape[0]
    baseline_up = label.dropna().mean() * 100
    print(f"\n  Total bars for analysis: {n_total}")
    print(f"  Baseline up%: {baseline_up:.2f}%  (target: {BASELINE*100:.1f}%)")

    # ── 4h resampled data ────────────────────────────────────────────────────
    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    # ── 15m resampled data ───────────────────────────────────────────────────
    df_15m = df_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    print("\nComputing signals...")

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 1: Z-SCORE (1h move relative to recent volatility)
    # ════════════════════════════════════════════════════════════════════════
    log_ret = np.log(c1h / c1h.shift(1))
    roll_vol = log_ret.rolling(24).std()
    z_score = log_ret / roll_vol.replace(0, np.nan)
    z_binned = pd.cut(z_score, bins=[-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf],
                      labels=[-3, -2, -1, -0.5, 0.5, 1, 2, 3]).astype(float)

    print("\n" + "=" * 60)
    print("GROUP 1: Z-SCORE (last 1h log return vs 24h rolling vol)")
    print("=" * 60)
    report_bins("z_score continuous", z_score, label,
                edges=[-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 2: RSI
    # ════════════════════════════════════════════════════════════════════════
    rsi_1h = _rsi(c1h, 14)
    rsi_4h_raw = _rsi(df_4h["close"], 14)
    rsi_4h = rsi_4h_raw.reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 2: RSI")
    print("=" * 60)
    report_bins("RSI 1h", rsi_1h, label, edges=[0, 20, 30, 40, 50, 60, 70, 80, 100])
    report_bins("RSI 4h", rsi_4h, label, edges=[0, 20, 30, 40, 50, 60, 70, 80, 100])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 3: STOCHASTIC
    # ════════════════════════════════════════════════════════════════════════
    stk_1h = _stoch_k(h1h, l1h, c1h, 14)
    stk_4h_raw = _stoch_k(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    stk_4h = stk_4h_raw.reindex(ts, method="ffill")
    stk_15m_raw = _stoch_k(df_15m["high"], df_15m["low"], df_15m["close"], 14)
    stk_15m = stk_15m_raw.resample("1h", origin="start_day").last().reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 3: STOCHASTIC")
    print("=" * 60)
    report_bins("Stoch %K 1h", stk_1h, label, edges=[0, 10, 20, 30, 50, 70, 80, 90, 100])
    report_bins("Stoch %K 4h", stk_4h, label, edges=[0, 10, 20, 30, 50, 70, 80, 90, 100])
    report_bins("Stoch %K 15m", stk_15m, label, edges=[0, 10, 20, 30, 50, 70, 80, 90, 100])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 4: MACD CROSSOVER (4h)
    # ════════════════════════════════════════════════════════════════════════
    macd_4h_raw = _macd_cross(df_4h["close"])
    macd_4h = macd_4h_raw.reindex(ts, method="ffill").fillna(0)

    print("\n" + "=" * 60)
    print("GROUP 4: MACD CROSSOVER (4h) — +1=bullish cross/lag, -1=bearish, 0=none")
    print("=" * 60)
    report("MACD 4h state", macd_4h, label)

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 5: BOLLINGER BANDS
    # ════════════════════════════════════════════════════════════════════════
    bb_1h = _bb_pct(c1h, 20)
    bb_4h_raw = _bb_pct(df_4h["close"], 20)
    bb_4h = bb_4h_raw.reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 5: BOLLINGER BAND POSITION (0=lower band, 1=upper band)")
    print("=" * 60)
    report_bins("BB %B 1h", bb_1h, label, edges=[0, 0.1, 0.2, 0.35, 0.65, 0.8, 0.9, 1.0, 1.5])
    report_bins("BB %B 4h", bb_4h, label, edges=[0, 0.1, 0.2, 0.35, 0.65, 0.8, 0.9, 1.0, 1.5])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 6: KELTNER CHANNEL
    # ════════════════════════════════════════════════════════════════════════
    kc_1h_pct, _, _ = _keltner_pct(h1h, l1h, c1h, 20, 2)
    kc_4h_pct_raw, _, _ = _keltner_pct(df_4h["high"], df_4h["low"], df_4h["close"], 20, 2)
    kc_4h_pct = kc_4h_pct_raw.reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 6: KELTNER CHANNEL POSITION (0=lower, 1=upper, >1=above)")
    print("=" * 60)
    report_bins("Keltner 1h", kc_1h_pct, label, edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5])
    report_bins("Keltner 4h", kc_4h_pct, label, edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 7: WILLIAMS %R
    # ════════════════════════════════════════════════════════════════════════
    wpr_1h = _wpr(h1h, l1h, c1h, 14)
    wpr_4h_raw = _wpr(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    wpr_4h = wpr_4h_raw.reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 7: WILLIAMS %R (0=at high, -100=at low)")
    print("=" * 60)
    report_bins("W%R 1h", wpr_1h, label, edges=[-100, -80, -60, -40, -20, -10, 0])
    report_bins("W%R 4h", wpr_4h, label, edges=[-100, -80, -60, -40, -20, -10, 0])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 8: DONCHIAN CHANNEL (15m)
    # ════════════════════════════════════════════════════════════════════════
    dc_15m_raw = _dc_pct(df_15m["high"], df_15m["low"], 20)
    dc_15m = dc_15m_raw.resample("1h", origin="start_day").last().reindex(ts, method="ffill")

    print("\n" + "=" * 60)
    print("GROUP 8: DONCHIAN CHANNEL 15m (0=at 20-bar low, 1=at 20-bar high)")
    print("=" * 60)
    report_bins("Donchian 15m", dc_15m, label, edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 9: VWAP DEVIATION
    # ════════════════════════════════════════════════════════════════════════
    print("\nComputing VWAP (slow)...")
    vwap_dev = _vwap_dev(df_1m, ts)

    print("\n" + "=" * 60)
    print("GROUP 9: VWAP DEVIATION (negative = below VWAP)")
    print("=" * 60)
    report_bins("VWAP dev %", vwap_dev * 100, label,
                edges=[-5, -2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2, 5])

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 10: EMA ALIGNMENT (9/21/50 on 1h)
    # ════════════════════════════════════════════════════════════════════════
    ema_align = _ema_alignment(c1h, spans=(9, 21, 50))

    print("\n" + "=" * 60)
    print("GROUP 10: EMA ALIGNMENT 1h (9/21/50) — +1=bullish stack, -1=bearish, 0=mixed")
    print("=" * 60)
    report("EMA alignment 1h", ema_align, label)

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 11: COMPOSITE TREND SCORE (4h)
    # ════════════════════════════════════════════════════════════════════════
    stk4_vote = ((stk_4h > 80).astype(int) - (stk_4h < 20).astype(int))
    vol_up = (v1h > v1h.rolling(20).mean() * 1.5) & (c1h > c1h.shift(1))
    vol_dn = (v1h > v1h.rolling(20).mean() * 1.5) & (c1h < c1h.shift(1))
    vol_vote = vol_up.astype(int) - vol_dn.astype(int)
    macd_vote = macd_4h
    bb4_vote = (bb_4h > 0.80).astype(int) - (bb_4h < 0.20).astype(int)
    kc4_vote = (kc_4h_pct > 0.85).astype(int) - (kc_4h_pct < 0.15).astype(int)
    wpr4_vote = (wpr_4h > -20).astype(int) - (wpr_4h < -80).astype(int)
    trend_score = (stk4_vote + vol_vote + macd_vote + bb4_vote + kc4_vote + wpr4_vote).clip(-6, 6)

    print("\n" + "=" * 60)
    print("GROUP 11: COMPOSITE TREND SCORE (-6 to +6)")
    print("=" * 60)
    report("Trend score", trend_score, label)

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 12: STRUCTURE BIAS (15m swing HH/HL)
    # ════════════════════════════════════════════════════════════════════════
    print("\nComputing structure bias (slow)...")
    struct_1h = _structure_bias_rolling(df_15m).reindex(ts, method="ffill").fillna(0)

    print("\n" + "=" * 60)
    print("GROUP 12: MARKET STRUCTURE BIAS (15m HH/HL) — +1=bullish, -1=bearish, 0=neutral")
    print("=" * 60)
    report("Structure bias", struct_1h, label)

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY: Strongest edge signals (vs baseline {:.1f}%)".format(BASELINE * 100))
    print("=" * 60)

    summary = []

    def check(name, signal, lo, hi):
        s = pd.DataFrame({"sig": signal, "up": label}).dropna()
        s = s[(s["sig"] >= lo) & (s["sig"] < hi)]
        if len(s) < 100:
            return
        up = s["up"].mean() * 100
        edge = up - BASELINE * 100
        summary.append((name, f"[{lo},{hi})", len(s), up, edge))

    check("z_score",     z_score, -2,    -1)
    check("z_score",     z_score, -1,     0)
    check("z_score",     z_score,  0,     1)
    check("z_score",     z_score,  1,     2)
    check("RSI 1h",      rsi_1h,   0,    30)
    check("RSI 1h",      rsi_1h,  70,   100)
    check("RSI 4h",      rsi_4h,   0,    30)
    check("RSI 4h",      rsi_4h,  70,   100)
    check("Stoch 1h",    stk_1h,   0,    20)
    check("Stoch 1h",    stk_1h,  80,   100)
    check("Stoch 4h",    stk_4h,   0,    20)
    check("Stoch 4h",    stk_4h,  80,   100)
    check("BB 1h",       bb_1h,    0,   0.2)
    check("BB 1h",       bb_1h,  0.8,   1.5)
    check("BB 4h",       bb_4h,    0,   0.2)
    check("BB 4h",       bb_4h,  0.8,   1.5)
    check("VWAP dev",    vwap_dev * 100, -2, -0.5)
    check("VWAP dev",    vwap_dev * 100,  0.5,  2)
    check("Keltner 1h",  kc_1h_pct, 0,  0.2)
    check("Keltner 1h",  kc_1h_pct, 0.8, 1.5)
    check("W%R 1h",      wpr_1h,  -100, -80)
    check("W%R 1h",      wpr_1h,   -20,   0)
    check("W%R 4h",      wpr_4h,  -100, -80)
    check("W%R 4h",      wpr_4h,   -20,   0)
    check("Donchian 15m", dc_15m,   0,  0.1)
    check("Donchian 15m", dc_15m, 0.9,  1.0)

    summary.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  {'Signal':<20} {'Bucket':<12} {'N':>6}  {'Up%':>6}  {'Edge':>7}")
    for name, bucket, n, up, edge in summary[:20]:
        flag = "  <-- USEFUL" if abs(edge) >= 2.0 else ""
        print(f"  {name:<20} {bucket:<12} {n:>6}  {up:>5.1f}%  {edge:>+6.1f}%{flag}")


if __name__ == "__main__":
    main()
