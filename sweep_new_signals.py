"""
sweep_new_signals.py — IC sweep of candidate signals vs next-hour BTC price direction.

Measures Pearson IC of each signal vs next_up (1h forward price direction) to
identify which new signals carry directional information beyond the current
trend/rev composite.

Baseline shown for: current trend, current rev, existing component signals.
Candidates: Ichimoku, ADX/DI, CCI, MFI, MACD 1h, Kalman residual, OBV slope,
            CMF, Aroon, rolling returns, EMA distance z-score.

All computed from OHLCV only → swept on full 2024-01-01 to present history.
Test period restricted to 2025-01-01+ to match composite calibration.
"""

import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
SYM      = "BTCUSDT"

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores, _rsi, _stoch_k, _atr, _keltner_pct, _wpr,
    _macd_cross, _bb_pct, _dc_pct, _vwap_1h,
)

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    def pick(pattern):
        files = sorted(glob.glob(str(DATA_DIR / pattern)))
        if not files:
            raise FileNotFoundError(f"No files for {pattern}")
        return files[-1]

    def load(p):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = df.columns.str.lower()
        return df.sort_index()

    f1h  = pick(f"binanceus_{SYM}_1h_2024-01-01_*.parquet")
    f4h  = pick(f"binanceus_{SYM}_4h_2024-01-01_*.parquet")
    f15m = pick(f"binanceus_{SYM}_15m_2024-01-01_*.parquet")
    f1m  = pick(f"binanceus_{SYM}_1m_2024-01-01_*.parquet")

    o1h  = load(f1h)
    o4h  = load(f4h)
    o15m = load(f15m)
    o1m  = load(f1m)
    print(f"  1h: {len(o1h):,}  4h: {len(o4h):,}  15m: {len(o15m):,}  1m: {len(o1m):,}")
    print(f"  Range: {o1h.index[0].date()} → {o1h.index[-1].date()}")
    return o1h, o4h, o15m, o1m


# ─────────────────────────────────────────────────────────────────────────────
def ic(signal: pd.Series, target: pd.Series) -> tuple:
    """Pearson IC of signal vs target on overlapping non-NaN rows."""
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    if len(df) < 50:
        return float("nan"), float("nan"), 0
    r, p = pearsonr(df["s"], df["t"])
    return r, p, len(df)


def report(label: str, signal: pd.Series, target: pd.Series):
    r, p, n = ic(signal, target)
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {label:<35}  IC={r:+.4f}  p={p:.4f}{stars:3}  n={n:,}")
    return r, p, n


# ─────────────────────────────────────────────────────────────────────────────
# Ichimoku helpers
def _ichi_mp(s, p):
    return (s.rolling(p).max() + s.rolling(p).min()) / 2


# ADX / DI helper
def _adx_di(h, l, c, p=14):
    hp  = h.shift(1)
    lp  = l.shift(1)
    dm_p = np.where((h - hp) > (lp - l), np.maximum(h - hp, 0.0), 0.0)
    dm_m = np.where((lp - l) > (h - hp), np.maximum(lp - l, 0.0), 0.0)
    atr_ = _atr(h, l, c, p)
    di_p = pd.Series(dm_p, index=c.index).ewm(com=p-1, adjust=False).mean() / atr_ * 100
    di_m = pd.Series(dm_m, index=c.index).ewm(com=p-1, adjust=False).mean() / atr_ * 100
    dx   = (di_p - di_m).abs() / (di_p + di_m).replace(0, float("nan")) * 100
    adx  = dx.ewm(com=p-1, adjust=False).mean()
    return adx, di_p, di_m


# CCI helper
def _cci(h, l, c, p=20):
    tp   = (h + l + c) / 3
    ma   = tp.rolling(p).mean()
    md   = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, float("nan")))


# MFI helper
def _mfi(h, l, c, v, p=14):
    tp   = (h + l + c) / 3
    mf   = tp * v
    pos  = mf.where(tp > tp.shift(1), 0.0).rolling(p).sum()
    neg  = mf.where(tp < tp.shift(1), 0.0).rolling(p).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, float("nan")))


# OBV slope helper
def _obv_slope(c, v, p=20):
    direction = np.sign(c.diff())
    obv = (v * direction).fillna(0).cumsum()
    return obv.rolling(p).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]), raw=True
    )


# CMF helper
def _cmf(h, l, c, v, p=20):
    clv = ((c - l) - (h - c)) / (h - l).replace(0, float("nan"))
    return (clv * v).rolling(p).sum() / v.rolling(p).sum().replace(0, float("nan"))


# Aroon oscillator helper
def _aroon(h, l, p=25):
    aroon_up   = h.rolling(p + 1).apply(lambda x: (np.argmax(x) / p) * 100, raw=True)
    aroon_down = l.rolling(p + 1).apply(lambda x: (np.argmin(x) / p) * 100, raw=True)
    return aroon_up - aroon_down


# Kalman filter (constant-velocity) residual
def _kalman_residual(c, q=1e-5, r=0.01):
    prices = c.values.astype(float)
    n      = len(prices)
    x      = np.array([prices[0], 0.0])
    P      = np.eye(2) * 1.0
    F      = np.array([[1, 1], [0, 1]])
    H      = np.array([[1, 0]])
    Q      = np.array([[q, 0], [0, q]])
    R      = np.array([[r]])
    resids = np.full(n, float("nan"))
    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y      = prices[i] - (H @ x_pred)[0]
        S      = (H @ P_pred @ H.T + R)[0, 0]
        K      = (P_pred @ H.T / S).flatten()
        x      = x_pred + K * y
        P      = (np.eye(2) - np.outer(K, H)) @ P_pred
        resids[i] = y / (prices[i] + 1e-10)   # normalized residual
    return pd.Series(resids, index=c.index)


# EMA z-score distance
def _ema_dist_z(c, span=20, roll=48):
    ema   = c.ewm(span=span, adjust=False).mean()
    dist  = (c - ema) / ema
    mu    = dist.rolling(roll).mean()
    sigma = dist.rolling(roll).std()
    return (dist - mu) / sigma.replace(0, float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading BTC data...")
    o1h, o4h, o15m, o1m = load_data()

    c1h  = o1h["close"].astype(float)
    h1h  = o1h["high"].astype(float)
    l1h  = o1h["low"].astype(float)
    v1h  = o1h["volume"].astype(float)
    c4h  = o4h["close"].astype(float)
    h4h  = o4h["high"].astype(float)
    l4h  = o4h["low"].astype(float)
    v4h  = o4h["volume"].astype(float)
    c15m = o15m["close"].astype(float)
    h15m = o15m["high"].astype(float)
    l15m = o15m["low"].astype(float)
    c1m  = o1m["close"].astype(float)
    v1m  = o1m["volume"].astype(float)

    ts_1h = c1h.index

    # ── Target ────────────────────────────────────────────────────────────────
    next_ret = np.log(c1h / c1h.shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(float)
    next_ret_cont = next_ret   # continuous return target

    # Restrict to test period
    mask = c1h.index >= TEST_START
    print(f"\nTest period: {TEST_START.date()} → {c1h.index[mask][-1].date()}"
          f"  ({mask.sum():,} 1h bars)\n")

    # ── Existing composite baseline ────────────────────────────────────────────
    print("=" * 70)
    print("  BASELINE — EXISTING COMPOSITE SIGNALS")
    print("=" * 70)

    print("\nComputing existing composite scores (may take 60-90s)...")
    trend_ser, rev_ser = compute_scores(
        c1h, h1h, l1h, v1h,
        c4h, h4h, l4h, v4h,
        c15m, h15m, l15m,
        c1m, v1m, ts_1h,
    )
    report("trend (current)",          trend_ser[mask], next_up[mask])
    report("rev (current)",            rev_ser[mask],   next_up[mask])
    report("trend (vs continuous ret)",trend_ser[mask], next_ret_cont[mask])
    report("rev (vs continuous ret)",  rev_ser[mask],   next_ret_cont[mask])

    # ── Component baselines ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPONENT BASELINES — EXISTING INDICATORS")
    print("=" * 70)

    stk4h_raw = _stoch_k(h4h, l4h, c4h, 14)
    stk4h_1h  = stk4h_raw.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    report("stoch_k_4h (raw)",         stk4h_1h[mask],  next_up[mask])

    rsi4h     = _rsi(c4h, 14)
    rsi4h_1h  = rsi4h.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    report("rsi_14_4h (raw)",          rsi4h_1h[mask],  next_up[mask])

    rsi1h     = _rsi(c1h, 14)
    report("rsi_14_1h (raw)",          rsi1h[mask],     next_up[mask])

    stk1h     = _stoch_k(h1h, l1h, c1h, 14)
    report("stoch_k_1h (raw)",         stk1h[mask],     next_up[mask])

    macd4h_f  = c4h.ewm(span=12, adjust=False).mean()
    macd4h_s  = c4h.ewm(span=26, adjust=False).mean()
    macd4h_sig= (macd4h_f - macd4h_s).ewm(span=9, adjust=False).mean()
    macd4h_h  = (macd4h_f - macd4h_s - macd4h_sig)
    macd4h_1h = macd4h_h.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    # Normalize by ATR
    atr4h_1h  = _atr(h4h, l4h, c4h, 14).resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
    macd4h_norm = macd4h_1h / atr4h_1h
    report("macd_hist_4h / atr (norm)",macd4h_norm[mask], next_up[mask])

    # ── CANDIDATE SIGNALS ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CANDIDATES — NEW SIGNALS")
    print("=" * 70)

    # ── 1. Ichimoku 1h ────────────────────────────────────────────────────────
    print("\n  [Ichimoku 1h]")
    tenkan_1h  = _ichi_mp(h1h, 9)
    kijun_1h   = _ichi_mp(h1h, 26)
    span_a_1h  = (tenkan_1h + kijun_1h) / 2
    span_b_1h  = _ichi_mp(h1h, 52)
    chikou_1h  = c1h.shift(26)

    tk_diff_1h     = tenkan_1h - kijun_1h
    tk_diff_norm   = tk_diff_1h / c1h                         # TK distance % price
    cloud_diff_1h  = span_a_1h - span_b_1h                    # +ve = bullish cloud
    cloud_norm_1h  = cloud_diff_1h / c1h
    close_vs_cloud = (c1h - (span_a_1h + span_b_1h) / 2) / c1h  # close vs cloud midpoint
    chikou_vs_cl   = (c1h.shift(26) - c1h.shift(52)) / c1h.shift(52)   # chikou momentum

    report("ichi_tk_diff_norm_1h",     tk_diff_norm[mask],    next_up[mask])
    report("ichi_cloud_diff_norm_1h",  cloud_norm_1h[mask],   next_up[mask])
    report("ichi_close_vs_cloud_1h",   close_vs_cloud[mask],  next_up[mask])
    report("ichi_chikou_mom_1h",       chikou_vs_cl[mask],    next_up[mask])

    # ── 2. Ichimoku 4h ────────────────────────────────────────────────────────
    print("\n  [Ichimoku 4h → resampled to 1h]")
    tenkan_4h  = _ichi_mp(h4h, 9)
    kijun_4h   = _ichi_mp(h4h, 26)
    span_a_4h  = (tenkan_4h + kijun_4h) / 2
    span_b_4h  = _ichi_mp(h4h, 52)

    def to_1h(s):
        return s.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    tk_diff_4h_1h    = to_1h(tenkan_4h - kijun_4h) / c1h
    cloud_diff_4h_1h = to_1h(span_a_4h - span_b_4h) / c1h
    close_vs_cloud_4h = (c1h - to_1h((span_a_4h + span_b_4h) / 2)) / c1h

    report("ichi_tk_diff_norm_4h",     tk_diff_4h_1h[mask],    next_up[mask])
    report("ichi_cloud_diff_norm_4h",  cloud_diff_4h_1h[mask], next_up[mask])
    report("ichi_close_vs_cloud_4h",   close_vs_cloud_4h[mask],next_up[mask])

    # ── 3. ADX / DI (1h and 4h) ───────────────────────────────────────────────
    print("\n  [ADX / DI]")
    adx1h, di_p1h, di_m1h = _adx_di(h1h, l1h, c1h, 14)
    di_diff1h = di_p1h - di_m1h

    adx4h, di_p4h, di_m4h = _adx_di(h4h, l4h, c4h, 14)
    di_diff4h = di_p4h - di_m4h
    di_diff4h_1h = to_1h(di_diff4h)
    adx4h_1h     = to_1h(adx4h)

    report("adx_14_1h",                adx1h[mask],       next_up[mask])
    report("di_diff_1h (DI+ - DI-)",   di_diff1h[mask],   next_up[mask])
    report("di_diff_4h (DI+ - DI-)",   di_diff4h_1h[mask],next_up[mask])
    report("adx_14_4h",                adx4h_1h[mask],    next_up[mask])

    # ── 4. CCI ────────────────────────────────────────────────────────────────
    print("\n  [CCI]")
    cci1h = _cci(h1h, l1h, c1h, 20)
    cci4h = to_1h(_cci(h4h, l4h, c4h, 20))
    report("cci_20_1h",                cci1h[mask],  next_up[mask])
    report("cci_20_4h",                cci4h[mask],  next_up[mask])

    # ── 5. MFI ────────────────────────────────────────────────────────────────
    print("\n  [MFI]")
    mfi1h = _mfi(h1h, l1h, c1h, v1h, 14)
    mfi4h = to_1h(_mfi(h4h, l4h, c4h, v4h, 14))
    report("mfi_14_1h",                mfi1h[mask],  next_up[mask])
    report("mfi_14_4h",                mfi4h[mask],  next_up[mask])

    # ── 6. MACD histogram 1h ─────────────────────────────────────────────────
    print("\n  [MACD histogram 1h]")
    macd1h_f   = c1h.ewm(span=12, adjust=False).mean()
    macd1h_s   = c1h.ewm(span=26, adjust=False).mean()
    macd1h_sig = (macd1h_f - macd1h_s).ewm(span=9, adjust=False).mean()
    macd1h_h   = macd1h_f - macd1h_s - macd1h_sig
    atr1h      = _atr(h1h, l1h, c1h, 14)
    macd1h_norm= macd1h_h / atr1h
    macd1h_chg = macd1h_h.diff()   # rate of change of histogram
    report("macd_hist_1h / atr (norm)",macd1h_norm[mask], next_up[mask])
    report("macd_hist_1h chg (slope)", macd1h_chg[mask],  next_up[mask])

    # ── 7. Kalman residual 1h ─────────────────────────────────────────────────
    print("\n  [Kalman filter residual 1h]")
    kalman_r = _kalman_residual(c1h)
    report("kalman_residual_1h",       kalman_r[mask],   next_up[mask])
    report("kalman_residual_1h (vs ret)", kalman_r[mask], next_ret_cont[mask])

    # ── 8. EMA distance z-score ───────────────────────────────────────────────
    print("\n  [EMA distance z-score]")
    ema_z_20_1h  = _ema_dist_z(c1h, span=20, roll=48)
    ema_z_50_1h  = _ema_dist_z(c1h, span=50, roll=72)
    ema_z_200_1h = _ema_dist_z(c1h, span=200, roll=200)
    ema_z_20_4h  = to_1h(_ema_dist_z(c4h, span=20, roll=48))
    report("ema_dist_z_20_1h",         ema_z_20_1h[mask],  next_up[mask])
    report("ema_dist_z_50_1h",         ema_z_50_1h[mask],  next_up[mask])
    report("ema_dist_z_200_1h",        ema_z_200_1h[mask], next_up[mask])
    report("ema_dist_z_20_4h",         ema_z_20_4h[mask],  next_up[mask])

    # ── 9. OBV slope ──────────────────────────────────────────────────────────
    print("\n  [OBV slope 1h]")
    obv_slope_1h = _obv_slope(c1h, v1h, p=20)
    obv_slope_norm = obv_slope_1h / c1h / v1h.rolling(20).mean()
    report("obv_slope_20_1h (norm)",   obv_slope_norm[mask], next_up[mask])

    # ── 10. CMF ───────────────────────────────────────────────────────────────
    print("\n  [CMF 1h]")
    cmf1h = _cmf(h1h, l1h, c1h, v1h, 20)
    report("cmf_20_1h",                cmf1h[mask],  next_up[mask])

    # ── 11. Aroon oscillator ─────────────────────────────────────────────────
    print("\n  [Aroon oscillator 1h]")
    aroon1h = _aroon(h1h, l1h, p=25)
    aroon4h = to_1h(_aroon(h4h, l4h, p=25))
    report("aroon_osc_25_1h",          aroon1h[mask], next_up[mask])
    report("aroon_osc_25_4h",          aroon4h[mask], next_up[mask])

    # ── 12. Rolling returns (momentum) ───────────────────────────────────────
    print("\n  [Rolling momentum returns]")
    log_ret = np.log(c1h / c1h.shift(1))
    ret6h   = log_ret.rolling(6).sum()
    ret12h  = log_ret.rolling(12).sum()
    ret24h  = log_ret.rolling(24).sum()
    ret72h  = log_ret.rolling(72).sum()
    report("ret_6h",                   ret6h[mask],  next_up[mask])
    report("ret_12h",                  ret12h[mask], next_up[mask])
    report("ret_24h",                  ret24h[mask], next_up[mask])
    report("ret_72h",                  ret72h[mask], next_up[mask])

    # ── 13. Rolling Sharpe ────────────────────────────────────────────────────
    print("\n  [Rolling Sharpe]")
    sharpe_24h = ret24h / log_ret.rolling(24).std().replace(0, float("nan"))
    sharpe_72h = ret72h / log_ret.rolling(72).std().replace(0, float("nan"))
    report("sharpe_24h",               sharpe_24h[mask], next_up[mask])
    report("sharpe_72h",               sharpe_72h[mask], next_up[mask])

    # ── 14. Stoch slope (momentum within stoch) ───────────────────────────────
    print("\n  [Stochastic slope / cross]")
    stk4h_1h_ser = stk4h_1h   # already computed above
    stk_d4h  = stk4h_raw.ewm(span=3, adjust=False).mean()
    stk_d4h_1h = to_1h(stk_d4h)
    stk_diff4h = stk4h_1h_ser - stk_d4h_1h   # K - D (cross signal)
    stk_chg4h  = stk4h_1h_ser.diff()          # slope
    report("stoch_k_4h_raw",           stk4h_1h_ser[mask], next_up[mask])
    report("stoch_kd_diff_4h (K-D)",   stk_diff4h[mask],   next_up[mask])
    report("stoch_k_4h_chg (slope)",   stk_chg4h[mask],    next_up[mask])

    stk1h_d   = stk1h.ewm(span=3, adjust=False).mean()
    stk_diff1h = stk1h - stk1h_d
    stk_chg1h  = stk1h.diff()
    report("stoch_k_1h_raw",           stk1h[mask],       next_up[mask])
    report("stoch_kd_diff_1h (K-D)",   stk_diff1h[mask],  next_up[mask])
    report("stoch_k_1h_chg (slope)",   stk_chg1h[mask],   next_up[mask])

    # ── 15. BB position cross-TF ──────────────────────────────────────────────
    print("\n  [BB position cross-TF]")
    bb4h_pct    = _bb_pct(h4h, l4h, c4h, 20)
    bb4h_pct_1h = to_1h(bb4h_pct)
    bb1h_pct    = _bb_pct(h1h, l1h, c1h, 20)
    bb_diff     = bb1h_pct - bb4h_pct_1h   # 1h vs 4h BB divergence
    report("bb_pct_4h",                bb4h_pct_1h[mask], next_up[mask])
    report("bb_pct_1h",                bb1h_pct[mask],    next_up[mask])
    report("bb_diff_1h_vs_4h",         bb_diff[mask],     next_up[mask])

    # ── 16. Volatility ratio (realized vs implied) ────────────────────────────
    print("\n  [Volatility regime]")
    rv1h  = log_ret.rolling(24).std() * np.sqrt(24)   # 24h realized vol
    rv4h  = log_ret.rolling(6).std() * np.sqrt(6)     # 6h short-term vol
    rvol_ratio = rv4h / rv1h.replace(0, float("nan"))
    report("rv_24h",                   rv1h[mask],       next_up[mask])
    report("rvol_short_long_ratio",     rvol_ratio[mask], next_up[mask])

    print("\n" + "=" * 70)
    print("  Done. Signals with |IC| > 0.02 and p<0.05 are worth investigating.")
    print("=" * 70)


if __name__ == "__main__":
    main()
