"""
validate_asset.py — Unified vol + direction signal validation for any asset.

Runs two analyses on the full historical 1h dataset for the given asset:
  1. VOL SIGNALS: which signals predict next-hour |return| magnitude
     (informs vol_layer thresholds)
  2. DIRECTION SIGNALS: which signals predict next-hour price direction (up/down)
     (informs direction_layer thresholds)

Usage:
    python3 validate_asset.py --asset ETH
    python3 validate_asset.py --asset SOL
    python3 validate_asset.py --asset BTC   # re-validate BTC as reference
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(asset: str):
    sym = f"{asset.upper()}USDT"
    h1_files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1h_2024-01-01_*.parquet"))
    m1_files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
    if not h1_files:
        raise FileNotFoundError(f"No 1h parquet files found for {sym}")
    if not m1_files:
        raise FileNotFoundError(f"No 1m parquet files found for {sym}")
    # exclude .ckpt files
    m1_files = [f for f in m1_files if ".ckpt" not in f.name]
    df_1h = pd.read_parquet(h1_files[-1])
    df_1m = pd.read_parquet(m1_files[-1])
    for df in (df_1h, df_1m):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df_1h.columns = df_1h.columns.str.lower()
    df_1m.columns = df_1m.columns.str.lower()
    return df_1h, df_1m


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
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


def _keltner_pct(h, l, c, span=20, mult=2.0):
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


def _dc_pct(h, l, n=20):
    dc_h = h.rolling(n).max()
    dc_l = l.rolling(n).min()
    return (l - dc_l) / (dc_h - dc_l).replace(0, np.nan)


def _vwap_dev(df_1m, ts_1h):
    date_1m = df_1m.index.normalize()
    tpv = df_1m["close"] * df_1m["volume"]
    cum_tpv = tpv.groupby(date_1m).cumsum()
    cum_vol = df_1m["volume"].groupby(date_1m).cumsum()
    vwap_1m = cum_tpv / cum_vol.replace(0, np.nan)
    vwap_1h = vwap_1m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    close_1h = df_1m["close"].resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    return (close_1h - vwap_1h) / vwap_1h.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers — VOL
# ─────────────────────────────────────────────────────────────────────────────

def report_vol_bins(name, signal, abs_ret, hl_range, edges, median_ret, median_hl,
                    threshold_lo, threshold_hi, min_n=100):
    """Print next-hour vol stats by signal bucket. threshold_lo/hi mark OTM reach levels."""
    df = pd.DataFrame({"sig": signal, "ar": abs_ret, "hl": hl_range}).dropna()
    print(f"\n  {name}:")
    print(f"  {'Bucket':>18}  {'N':>6}  {'Med|ret|':>9}  {'vs base':>7}  "
          f"{'>{:.1f}%'.format(threshold_lo*100):>7}  {'>{:.1f}%'.format(threshold_hi*100):>7}")
    all_edges = list(edges) + [np.inf]
    for i in range(len(all_edges) - 1):
        lo, hi = all_edges[i], all_edges[i+1]
        sub = df[(df["sig"] >= lo) & (df["sig"] < hi)]
        if len(sub) < min_n:
            continue
        med_r = sub["ar"].median() * 100
        pct_lo = (sub["ar"] > threshold_lo).mean() * 100
        pct_hi = (sub["ar"] > threshold_hi).mean() * 100
        dr = med_r - median_ret * 100
        flag = "  <-- HIGH VOL" if dr > 0.03 else ("  <-- LOW VOL" if dr < -0.02 else "")
        bucket = f"[{lo:.2f},{hi:.2f})" if hi != np.inf else f"[{lo:.2f},+inf)"
        print(f"  {bucket:>18}  {len(sub):>6}  {med_r:>8.4f}%  {dr:>+6.3f}%  "
              f"{pct_lo:>6.1f}%  {pct_hi:>6.1f}%{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers — DIRECTION
# ─────────────────────────────────────────────────────────────────────────────

def report_bins(name, signal, label, edges, baseline, min_n=50):
    combined = pd.DataFrame({"sig": signal, "up": label}).dropna()
    print(f"\n  {name}:")
    print(f"  {'Bucket':>18}  {'N':>6}  {'Up%':>6}  {'Edge':>7}  bar")
    all_edges = list(edges) + [np.inf]
    for i in range(len(all_edges) - 1):
        lo, hi = all_edges[i], all_edges[i+1]
        sub = combined[(combined["sig"] >= lo) & (combined["sig"] < hi)]
        if len(sub) < min_n:
            continue
        up_pct = sub["up"].mean() * 100
        edge = up_pct - baseline * 100
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        bucket = f"[{lo:.2f},{hi:.2f})" if hi != np.inf else f"[{lo:.2f},+inf)"
        print(f"  {bucket:>18}  {len(sub):>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")


def report_discrete(name, signal, label, baseline, min_n=50):
    combined = pd.DataFrame({"sig": signal, "up": label}).dropna()
    vals = sorted(combined["sig"].unique())
    print(f"\n  {name}:")
    print(f"  {'Value':>8}  {'N':>6}  {'Up%':>6}  {'Edge':>7}  bar")
    for v in vals:
        sub = combined[combined["sig"] == v]
        if len(sub) < min_n:
            continue
        up_pct = sub["up"].mean() * 100
        edge = up_pct - baseline * 100
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        print(f"  {v:>8}  {len(sub):>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=["BTC", "ETH", "SOL"],
                        type=str.upper, help="Asset to validate")
    args = parser.parse_args()
    asset = args.asset

    print(f"\n{'='*70}")
    print(f"  SIGNAL VALIDATION — {asset}USDT")
    print(f"{'='*70}")

    print(f"\nLoading {asset} data...")
    df_1h, df_1m = load_data(asset)
    print(f"  1h bars: {len(df_1h)}  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
    print(f"  1m bars: {len(df_1m)}")

    ts  = df_1h.index
    c   = df_1h["close"]
    h   = df_1h["high"]
    l   = df_1h["low"]
    v   = df_1h["volume"]

    # Next-hour labels
    log_ret  = np.log(c / c.shift(1))
    abs_ret  = log_ret.shift(-1).abs()
    hl_range = ((h.shift(-1) - l.shift(-1)) / c)
    label    = (c.shift(-1) > c).astype(float)
    abs_ret.iloc[-1]  = np.nan
    hl_range.iloc[-1] = np.nan
    label.iloc[-1]    = np.nan

    baseline  = float(label.dropna().mean())
    median_ret = abs_ret.dropna().median()
    median_hl  = hl_range.dropna().median()

    # OTM thresholds: ETH/SOL contracts are typically at 0.3-0.5% OTM
    # Use 0.3% and 0.5% as "can strike be reached" thresholds
    threshold_lo = 0.003   # 0.3% — typical OTM for ETH/SOL
    threshold_hi = 0.005   # 0.5% — more OTM

    pct_lo_base = (abs_ret.dropna() > threshold_lo).mean() * 100
    pct_hi_base = (abs_ret.dropna() > threshold_hi).mean() * 100

    print(f"\n  Baseline up%              : {baseline*100:.2f}%")
    print(f"  Median |return| next hour : {median_ret*100:.4f}%")
    print(f"  Median HL range next hour : {median_hl*100:.4f}%")
    print(f"  % bars with |ret| > {threshold_lo*100:.1f}%  : {pct_lo_base:.1f}%")
    print(f"  % bars with |ret| > {threshold_hi*100:.1f}%  : {pct_hi_base:.1f}%")

    # ── resample ─────────────────────────────────────────────────────────────
    df_4h = df_1h.resample("4h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    df_15m = df_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])

    # ── compute indicators ───────────────────────────────────────────────────
    print("\nComputing indicators...")

    roll_vol_24 = log_ret.rolling(24).std()
    z_score     = log_ret / roll_vol_24.replace(0, np.nan)

    # Vol-specific
    rv_6h_1m = (np.log(df_1m["close"] / df_1m["close"].shift(1))
                .rolling(360).std() * 100)  # % per 1m bar
    rv_6h_1m_1h = rv_6h_1m.resample("1h", origin="start_day").last().reindex(ts, method="ffill")

    atr_1h    = _atr(h, l, c, 14)
    atr_ratio = atr_1h / atr_1h.rolling(24).mean()
    vol_ratio = v / v.rolling(20).mean()

    print("  Computing VWAP...")
    vwap_dev  = _vwap_dev(df_1m, ts)

    # Oscillators
    rsi_1h    = _rsi(c, 14)
    rsi_4h    = _rsi(df_4h["close"], 14).reindex(ts, method="ffill")
    stk_1h    = _stoch_k(h, l, c, 14)
    stk_4h    = _stoch_k(df_4h["high"], df_4h["low"], df_4h["close"], 14).reindex(ts, method="ffill")
    stk_15m   = (_stoch_k(df_15m["high"], df_15m["low"], df_15m["close"], 14)
                 .resample("1h", origin="start_day").last().reindex(ts, method="ffill"))
    bb_4h     = _bb_pct(df_4h["close"], 20).reindex(ts, method="ffill")
    kc_4h_pct, _, _ = _keltner_pct(df_4h["high"], df_4h["low"], df_4h["close"], 20, 2.0)
    kc_4h_pct = kc_4h_pct.reindex(ts, method="ffill")
    kc_1h_pct, _, _ = _keltner_pct(h, l, c, 20, 2.0)
    wpr_1h    = _wpr(h, l, c, 14)
    wpr_4h    = _wpr(df_4h["high"], df_4h["low"], df_4h["close"], 14).reindex(ts, method="ffill")
    macd_4h   = _macd_cross(df_4h["close"]).reindex(ts, method="ffill").fillna(0)
    dc_15m    = (_dc_pct(df_15m["high"], df_15m["low"], 20)
                 .resample("1h", origin="start_day").last().reindex(ts, method="ffill"))

    # Volume directional (inverted per BTC validation: selling climax → bounce)
    vol_up    = (v > v.rolling(20).mean() * 1.5) & (c > c.shift(1))
    vol_dn    = (v > v.rolling(20).mean() * 1.5) & (c < c.shift(1))
    vol_dir   = vol_up.astype(int) - vol_dn.astype(int)

    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PART 1 — VOL SIGNALS (what predicts next-hour |move| magnitude?)")
    print(f"{'='*70}")
    print(f"  Baseline: median |ret|={median_ret*100:.4f}%  "
          f">{threshold_lo*100:.1f}%={pct_lo_base:.1f}%  >{threshold_hi*100:.1f}%={pct_hi_base:.1f}%")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  --- ATR RATIO (current ATR / 24h mean ATR) ---")
    report_vol_bins("ATR ratio", atr_ratio, abs_ret, hl_range,
                    edges=[0, 0.5, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0],
                    median_ret=median_ret, median_hl=median_hl,
                    threshold_lo=threshold_lo, threshold_hi=threshold_hi)

    print(f"\n  --- |Z-SCORE| of current 1h move ---")
    report_vol_bins("|z_score|", z_score.abs(), abs_ret, hl_range,
                    edges=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                    median_ret=median_ret, median_hl=median_hl,
                    threshold_lo=threshold_lo, threshold_hi=threshold_hi)

    print(f"\n  --- VOLUME RATIO (current bar vol / 20-bar MA) ---")
    report_vol_bins("Volume ratio", vol_ratio, abs_ret, hl_range,
                    edges=[0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    median_ret=median_ret, median_hl=median_hl,
                    threshold_lo=threshold_lo, threshold_hi=threshold_hi)

    print(f"\n  --- |VWAP DEVIATION| ---")
    report_vol_bins("|VWAP dev| %", vwap_dev.abs() * 100, abs_ret, hl_range,
                    edges=[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                    median_ret=median_ret, median_hl=median_hl,
                    threshold_lo=threshold_lo, threshold_hi=threshold_hi)

    print(f"\n  --- 6H REALIZED VOL (1m bars, % per 1m bar) ---")
    report_vol_bins("rv_6h (1m std, %/bar)", rv_6h_1m_1h, abs_ret, hl_range,
                    edges=[0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                    median_ret=median_ret, median_hl=median_hl,
                    threshold_lo=threshold_lo, threshold_hi=threshold_hi)

    # Summary: find optimal high/low thresholds for each vol signal
    print(f"\n{'='*70}")
    print(f"  VOL SIGNAL THRESHOLD RECOMMENDATIONS FOR {asset}")
    print(f"{'='*70}")
    print(f"  (Find the bucket where >0.5% moves are highest/lowest vs baseline {pct_hi_base:.1f}%)")

    def find_thresholds(name, sig, edges):
        """For each candidate threshold bucket, show >0.5% move rate."""
        df2 = pd.DataFrame({"s": sig, "ar": abs_ret}).dropna()
        rows = []
        all_e = list(edges) + [np.inf]
        for lo, hi in zip(all_e[:-1], all_e[1:]):
            sub = df2[(df2["s"] >= lo) & (df2["s"] < hi)]
            if len(sub) < 100:
                continue
            p05 = (sub["ar"] > threshold_hi).mean() * 100
            delta = p05 - pct_hi_base
            bucket = f"[{lo:.2f},{hi:.2f})" if hi != np.inf else f"[{lo:.2f},+inf)"
            rows.append((bucket, len(sub), p05, delta))
        print(f"\n  {name}:")
        print(f"  {'Bucket':>18}  {'N':>6}  {'>{:.1f}%'.format(threshold_hi*100):>7}  {'delta':>7}")
        for bucket, n, p05, delta in rows:
            flag = "  *** HIGH VOL" if delta > 5 else ("  *** LOW VOL" if delta < -5 else "")
            print(f"  {bucket:>18}  {n:>6}  {p05:>6.1f}%  {delta:>+6.1f}%{flag}")

    find_thresholds("ATR ratio", atr_ratio,
                    [0, 0.5, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0])
    find_thresholds("|z_score|", z_score.abs(),
                    [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    find_thresholds("Volume ratio", vol_ratio,
                    [0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    find_thresholds("|VWAP dev| %", vwap_dev.abs() * 100,
                    [0, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    find_thresholds("rv_6h (%/bar)", rv_6h_1m_1h,
                    [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PART 2 — DIRECTION SIGNALS (what predicts next-hour up/down?)")
    print(f"{'='*70}")
    print(f"  Baseline up%: {baseline*100:.2f}%   Edge = Up% - baseline")

    print(f"\n  --- TREND SIGNALS (4h — continuation) ---")

    report_bins("Stoch %K 4h", stk_4h, label,
                edges=[0, 10, 20, 30, 50, 70, 80, 90, 100],
                baseline=baseline)
    report_bins("RSI 4h", rsi_4h, label,
                edges=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                baseline=baseline)
    report_bins("BB %B 4h", bb_4h, label,
                edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5],
                baseline=baseline)
    report_bins("Keltner 4h", kc_4h_pct, label,
                edges=[0, 0.1, 0.15, 0.20, 0.40, 0.60, 0.80, 0.85, 1.0, 1.5],
                baseline=baseline)
    report_bins("W%R 4h", wpr_4h, label,
                edges=[-100, -80, -60, -40, -20, -10, 0],
                baseline=baseline)
    report_discrete("MACD 4h crossover", macd_4h, label, baseline=baseline)
    report_discrete("Volume directional (inverted)", vol_dir, label, baseline=baseline)

    print(f"\n  --- MEAN-REVERSION SIGNALS (1h/15m) ---")

    report_bins("RSI 1h", rsi_1h, label,
                edges=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                baseline=baseline)
    report_bins("Stoch %K 1h", stk_1h, label,
                edges=[0, 10, 20, 30, 50, 70, 80, 90, 100],
                baseline=baseline)
    report_bins("Stoch %K 15m", stk_15m, label,
                edges=[0, 10, 20, 30, 50, 70, 80, 90, 100],
                baseline=baseline)
    report_bins("Keltner 1h", kc_1h_pct, label,
                edges=[-0.5, 0, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 0.85, 1.0, 1.5],
                baseline=baseline)
    report_bins("W%R 1h", wpr_1h, label,
                edges=[-100, -80, -60, -40, -20, -10, 0],
                baseline=baseline)
    report_bins("z_score (signed)", z_score, label,
                edges=[-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4],
                baseline=baseline)
    report_bins("VWAP dev %", vwap_dev * 100, label,
                edges=[-5, -2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2, 5],
                baseline=baseline)
    report_bins("Donchian 15m (0=near low, 1=near high)", dc_15m, label,
                edges=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                baseline=baseline)

    # RSI multi-TF interaction
    print(f"\n  --- RSI MULTI-TF INTERACTION ---")
    rsi_1h_os = rsi_1h < 30
    rsi_1h_ob = rsi_1h > 70
    rsi_4h_os = rsi_4h < 30
    rsi_4h_ob = rsi_4h > 70
    rsi_4h_neutral = (rsi_4h >= 40) & (rsi_4h <= 60)

    cases = {
        "1h OS + 4h neutral (bounce setup)": rsi_1h_os & rsi_4h_neutral,
        "1h OB + 4h neutral (fade setup)":   rsi_1h_ob & rsi_4h_neutral,
        "both OS (downtrend continuation)":   rsi_1h_os & rsi_4h_os,
        "both OB (uptrend continuation)":     rsi_1h_ob & rsi_4h_ob,
    }
    for case_name, mask in cases.items():
        sub = pd.DataFrame({"up": label}).loc[mask].dropna()
        if len(sub) < 30:
            print(f"  {case_name}: n={len(sub)} (too few)")
            continue
        up_pct = sub["up"].mean() * 100
        edge = up_pct - baseline * 100
        print(f"  {case_name}: n={len(sub)}  up={up_pct:.1f}%  edge={edge:+.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DIRECTION SIGNAL SUMMARY — {asset} (sorted by |edge|)")
    print(f"{'='*70}")

    rows = []

    def chk_dir(name, sig, lo, hi):
        df2 = pd.DataFrame({"s": sig, "up": label}).dropna()
        sub = df2[(df2["s"] >= lo) & (df2["s"] < hi)]
        if len(sub) < 100:
            return
        up_pct = sub["up"].mean() * 100
        edge = up_pct - baseline * 100
        rows.append((name, f"[{lo},{hi})", len(sub), up_pct, edge))

    # 4h trend (strong extremes only)
    chk_dir("stk_4h",  stk_4h,   80,  101)
    chk_dir("stk_4h",  stk_4h,    0,   20)
    chk_dir("rsi_4h",  rsi_4h,   70,  101)
    chk_dir("rsi_4h",  rsi_4h,    0,   30)
    chk_dir("bb_4h",   bb_4h,    0.8,  9.9)
    chk_dir("bb_4h",   bb_4h,    0,    0.2)
    chk_dir("kc_4h",   kc_4h_pct, 0.85, 9.9)
    chk_dir("kc_4h",   kc_4h_pct, 0,   0.15)
    chk_dir("wpr_4h",  wpr_4h,  -20,   1)
    chk_dir("wpr_4h",  wpr_4h, -100, -80)
    # 1h/15m reversion
    chk_dir("rsi_1h",  rsi_1h,    0,   20)
    chk_dir("rsi_1h",  rsi_1h,   80,  101)
    chk_dir("stk_1h",  stk_1h,    0,   10)
    chk_dir("stk_1h",  stk_1h,   90,  101)
    chk_dir("stk_15m", stk_15m,   0,   10)
    chk_dir("stk_15m", stk_15m,  90,  101)
    chk_dir("kc_1h",   kc_1h_pct, 0,  0.10)
    chk_dir("kc_1h",   kc_1h_pct, 0.90, 9.9)
    chk_dir("wpr_1h",  wpr_1h,    0,   -80+1e-6)
    chk_dir("wpr_1h",  wpr_1h,  -80,  -60)
    chk_dir("z_score", z_score,  -2,  -1.5)
    chk_dir("z_score", z_score,   1.5,  2.5)
    chk_dir("vwap_dev%", vwap_dev * 100, -1.5, -0.5)
    chk_dir("vwap_dev%", vwap_dev * 100,  0.5,  1.5)
    chk_dir("dc_15m",  dc_15m,   0,   0.10)
    chk_dir("dc_15m",  dc_15m,   0.90, 9.9)

    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  {'Signal':<12} {'Bucket':<14} {'N':>6}  {'Up%':>6}  {'Edge':>7}  bar")
    for name, bucket, n, up_pct, edge in rows[:30]:
        bar = "#" * int(abs(edge) * 2) if abs(edge) > 0.5 else "·"
        direction = "+" if edge > 0 else "-"
        print(f"  {name:<12} {bucket:<14} {n:>6}  {up_pct:>5.1f}%  {edge:>+6.1f}%  {direction}{bar}")

    print(f"\n  Done. Use these results to calibrate vol_layer and direction_layer thresholds for {asset}.")


if __name__ == "__main__":
    main()
