#!/usr/bin/env python3
"""
analyze_signal_correlation.py

Comprehensive directional correlation sweep for BTC, ETH, SOL.
Uses full Binance 1m parquet history (~1.2M bars each, Jan 2024–present).

Signals tested (65+):
  Price action : bp, body, dir, chg, z_score, vwap_dist
  Momentum     : stoch_k/d/cross, RSI, MACD hist/cross, consecutive_dir
  Trend        : ema_bias (5>20), ema_stack (5>20>50)
  Volatility   : range_ratio, ATR_ratio, vol_ratio, BB_width, BB_pct
  Structure    : upper/lower wick, wick_imbalance, engulfing, hammer, star, doji
  Donchian     : donchian_pos_20, donchian_width_20, donchian_breakout
  Mean rev     : BB_pct, donchian_pos, vwap_dist, RSI, z_score
  Multi-TF     : above repeated at 5m and 1h timeframes
  Combined     : ema_stoch_agree, bp_body_agree, multi_tf_agree
  External     : liq_bias, liq_score, ls_long_pct, ls_short_pct, oi_chg_pct (90-day)

Forward horizons: 15m, 30m, 1h, 2h  (from bar close, no look-ahead)

Run:
  python3 analyze_signal_correlation.py
  python3 analyze_signal_correlation.py --asset ETH
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
SEP  = "=" * 80
SEP2 = "-" * 80

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}
COINALYZE_SYMBOLS = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}
COINALYZE_KEY  = "d5841821-3f45-4e5f-9ee7-d2779d2fb01b"
COINALYZE_BASE = "https://api.coinalyze.net/v1"

FWD_HORIZONS = {          # label → number of 15m bars ahead
    "fwd_15m":  1,
    "fwd_30m":  2,
    "fwd_1h":   4,
    "fwd_2h":   8,
}
MIN_N        = 200        # minimum observations for a signal to be reported
DECILES      = 10
TOP_N        = 25         # rows in main ranked table

# ── parquet helpers ───────────────────────────────────────────────────────────

def latest_parquet(sym: str) -> Path:
    files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 1m parquet for {sym}")
    return files[-1]


def load_1m(sym: str) -> pd.DataFrame:
    p = latest_parquet(sym)
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


# ── indicator primitives ──────────────────────────────────────────────────────

def _stoch_k(high, low, close, k=14):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    rng = hi - lo
    return np.where(rng > 0, 100.0 * (close - lo) / rng, 50.0)


def _stoch_d(stoch_k_series, d=3):
    return pd.Series(stoch_k_series).rolling(d).mean()


def _rsi(close, p=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=p - 1, min_periods=p).mean()
    avg_l = loss.ewm(com=p - 1, min_periods=p).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close, f=12, s=26, sig=9):
    ema_f  = close.ewm(span=f, adjust=False).mean()
    ema_s  = close.ewm(span=s, adjust=False).mean()
    macd   = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def _atr(high, low, close, p=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, min_periods=p).mean()


def _bb(close, n=20, nstd=2):
    sma  = close.rolling(n).mean()
    std  = close.rolling(n).std()
    upper = sma + nstd * std
    lower = sma - nstd * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pct   = (close - lower) / (upper - lower).replace(0, np.nan)
    return width, pct


def _donchian(high, low, close, n=20):
    dc_high = high.rolling(n).max()
    dc_low  = low.rolling(n).min()
    dc_rng  = dc_high - dc_low
    pos     = (close - dc_low) / dc_rng.replace(0, np.nan)
    width   = dc_rng / close.replace(0, np.nan)
    breakout = np.where(close >= dc_high.shift(), 1,
               np.where(close <= dc_low.shift(),  -1, 0))
    return pos, width, pd.Series(breakout, index=close.index)


def _bp(close, low, high):
    rng = high - low
    return np.where(rng > 0, (close - low) / rng, 0.5)


def _body(open_, close, high, low):
    rng = high - low
    return np.where(rng > 0, (close - open_).abs() / rng, 0.0)


def _upper_wick(open_, close, high, low):
    rng = high - low
    top = np.maximum(open_, close)
    return np.where(rng > 0, (high - top) / rng, 0.0)


def _lower_wick(open_, close, high, low):
    rng = high - low
    bot = np.minimum(open_, close)
    return np.where(rng > 0, (bot - low) / rng, 0.0)


def _engulfing(open_, close):
    body     = close - open_
    body_abs = body.abs()
    bullish  = (body > 0) & (body_abs > body_abs.shift().abs()) & (body.shift() < 0)
    bearish  = (body < 0) & (body_abs > body_abs.shift().abs()) & (body.shift() > 0)
    return pd.Series(
        np.where(bullish, 1, np.where(bearish, -1, 0)),
        index=open_.index
    )


def _consecutive_dir(close):
    dir_series = np.sign(close.diff())
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(dir_series)):
        d = dir_series.iloc[i]
        if d == 0:
            streak.iloc[i] = 0
        elif d == streak.iloc[i - 1] / max(abs(streak.iloc[i - 1]), 1) if streak.iloc[i - 1] != 0 else True:
            streak.iloc[i] = streak.iloc[i - 1] + d
        else:
            streak.iloc[i] = d
    return streak.clip(-5, 5)


def _vwap_daily(close, volume, index):
    dates   = pd.Series(index.date, index=index)
    tpv     = close * volume
    cum_vol = volume.groupby(dates.values).cumsum()
    cum_tpv = tpv.groupby(dates.values).cumsum()
    vwap    = cum_tpv / cum_vol.replace(0, np.nan)
    return (close - vwap) / vwap.replace(0, np.nan) * 100.0


def _z_score(close, n=20):
    mu  = close.rolling(n).mean()
    std = close.rolling(n).std()
    return (close - mu) / std.replace(0, np.nan)


def _hammer(open_, close, high, low, body_thresh=0.3, wick_thresh=2.0):
    rng  = high - low
    body = (close - open_).abs()
    lw   = np.minimum(open_, close) - low
    uw   = high - np.maximum(open_, close)
    body_frac = np.where(rng > 0, body / rng, 0.0)
    return (body_frac <= body_thresh) & (np.where(body > 0, lw / body, 0) >= wick_thresh) & (uw < body)


def _shooting_star(open_, close, high, low, body_thresh=0.3, wick_thresh=2.0):
    rng  = high - low
    body = (close - open_).abs()
    uw   = high - np.maximum(open_, close)
    lw   = np.minimum(open_, close) - low
    body_frac = np.where(rng > 0, body / rng, 0.0)
    return (body_frac <= body_thresh) & (np.where(body > 0, uw / body, 0) >= wick_thresh) & (lw < body)


def _doji(open_, close, high, low, thresh=0.05):
    rng  = high - low
    body = (close - open_).abs()
    return np.where(rng > 0, body / rng, 0.0) <= thresh


# ── signal computation ────────────────────────────────────────────────────────

def compute_signals_for_tf(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute all signals for a given OHLCV DataFrame (any timeframe)."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    sig = pd.DataFrame(index=df.index)

    # ── price action ──────────────────────────────────────────────────────────
    sig[f"bp_{label}"]          = _bp(c, l, h)
    sig[f"body_{label}"]        = _body(o, c, h, l)
    sig[f"dir_{label}"]         = np.sign(c - o)
    sig[f"chg_{label}"]         = c.pct_change() * 100.0
    sig[f"z_score_{label}"]     = _z_score(c, 20)
    if label == "15m":
        sig[f"vwap_dist_{label}"] = _vwap_daily(c, v, df.index)

    # ── candle structure ──────────────────────────────────────────────────────
    uw = _upper_wick(o, c, h, l)
    lw = _lower_wick(o, c, h, l)
    sig[f"upper_wick_{label}"]     = uw
    sig[f"lower_wick_{label}"]     = lw
    sig[f"wick_imbalance_{label}"] = lw - uw          # + = bullish rejection
    sig[f"engulfing_{label}"]      = _engulfing(o, c)
    sig[f"hammer_{label}"]         = _hammer(o, c, h, l).astype(float)
    sig[f"shooting_star_{label}"]  = _shooting_star(o, c, h, l).astype(float)
    sig[f"doji_{label}"]           = _doji(o, c, h, l).astype(float)

    # ── momentum ──────────────────────────────────────────────────────────────
    sk  = pd.Series(_stoch_k(h, l, c, 14), index=df.index)
    sd  = _stoch_d(sk, 3)
    sig[f"stoch_k_{label}"]     = sk
    sig[f"stoch_d_{label}"]     = sd
    cross_up   = (sk > sd) & (sk.shift() <= sd.shift())
    cross_dn   = (sk < sd) & (sk.shift() >= sd.shift())
    sig[f"stoch_cross_{label}"] = np.where(cross_up, 1, np.where(cross_dn, -1, 0)).astype(float)
    sig[f"stoch_ob_{label}"]    = (sk >= 80).astype(float)
    sig[f"stoch_os_{label}"]    = (sk <= 20).astype(float)

    rsi = _rsi(c, 14)
    sig[f"rsi_{label}"]    = rsi
    sig[f"rsi_ob_{label}"] = (rsi >= 70).astype(float)
    sig[f"rsi_os_{label}"] = (rsi <= 30).astype(float)

    macd_line, macd_sig, macd_hist = _macd(c)
    sig[f"macd_hist_{label}"]  = macd_hist
    cross_macd_up = (macd_hist > 0) & (macd_hist.shift() <= 0)
    cross_macd_dn = (macd_hist < 0) & (macd_hist.shift() >= 0)
    sig[f"macd_cross_{label}"] = np.where(cross_macd_up, 1, np.where(cross_macd_dn, -1, 0)).astype(float)

    sig[f"consec_dir_{label}"] = _consecutive_dir(c)

    # ── trend ─────────────────────────────────────────────────────────────────
    ema5  = c.ewm(span=5,  adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    sig[f"ema_bias_{label}"]  = np.sign(ema5 - ema20).astype(float)
    sig[f"ema_stack_{label}"] = np.where(
        (ema5 > ema20) & (ema20 > ema50),  1,
        np.where((ema5 < ema20) & (ema20 < ema50), -1, 0)
    ).astype(float)
    # ema5 vs ema20 gap (normalized)
    sig[f"ema_gap_{label}"] = (ema5 - ema20) / c.replace(0, np.nan) * 100.0

    # ── volatility / range ────────────────────────────────────────────────────
    bar_range   = h - l
    avg_range   = bar_range.rolling(20).mean()
    sig[f"range_ratio_{label}"] = bar_range / avg_range.replace(0, np.nan)
    atr_val     = _atr(h, l, c, 14)
    sig[f"atr_ratio_{label}"]   = atr_val / c.replace(0, np.nan) * 100.0
    vol_ma = v.rolling(20).mean()
    sig[f"vol_ratio_{label}"]   = v / vol_ma.replace(0, np.nan)
    bb_w, bb_p  = _bb(c, 20, 2)
    sig[f"bb_width_{label}"]    = bb_w
    sig[f"bb_pct_{label}"]      = bb_p

    # ── Donchian ──────────────────────────────────────────────────────────────
    dc_pos, dc_wid, dc_brk = _donchian(h, l, c, 20)
    sig[f"donchian_pos_{label}"]      = dc_pos
    sig[f"donchian_width_{label}"]    = dc_wid
    sig[f"donchian_breakout_{label}"] = dc_brk.astype(float)

    return sig


def build_signals(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5m, 15m, 1h and compute all signals. Return 15m-aligned DataFrame."""

    def resample(tf):
        return df_1m.resample(tf).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"),     close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna()

    df_5m  = resample("5min")
    df_15m = resample("15min")
    df_1h  = resample("1h")

    # Compute signals at each TF
    sig_5m  = compute_signals_for_tf(df_5m,  "5m")
    sig_15m = compute_signals_for_tf(df_15m, "15m")
    sig_1h  = compute_signals_for_tf(df_1h,  "1h")

    # Forward returns at 15m resolution (from bar close)
    fwd = pd.DataFrame(index=df_15m.index)
    c15 = df_15m["close"]
    for lbl, n_bars in FWD_HORIZONS.items():
        fwd[lbl] = c15.shift(-n_bars) / c15 - 1.0
    fwd["fwd_dir_15m"] = np.sign(fwd["fwd_15m"])  # binary +1/-1

    # Join: last completed 5m bar before each 15m bar close
    sig_5m_lagged  = sig_5m.shift(1)    # last completed 5m bar
    sig_1h_lagged  = sig_1h.reindex(df_15m.index, method="ffill")

    # Align 5m to 15m index (forward-fill, then take lag)
    sig_5m_on_15m  = sig_5m_lagged.reindex(df_15m.index, method="ffill")

    # Combine: 15m signals use last completed bar (shift 1)
    sig_15m_lagged = sig_15m.shift(1)

    result = pd.concat([sig_15m_lagged, sig_5m_on_15m, sig_1h_lagged, fwd], axis=1)

    # ── Combined / agreement signals ──────────────────────────────────────────
    # EMA + stoch both agree on direction at 15m
    result["ema_stoch_agree_15m"] = np.where(
        (result["ema_bias_15m"] == 1)  & (result["stoch_k_15m"] > 50), 1,
        np.where((result["ema_bias_15m"] == -1) & (result["stoch_k_15m"] < 50), -1, 0)
    ).astype(float)

    # BP and body both agree on direction at 15m
    result["bp_body_agree_15m"] = np.where(
        (result["bp_15m"] > 0.5) & (result["dir_15m"] == 1),  1,
        np.where((result["bp_15m"] < 0.5) & (result["dir_15m"] == -1), -1, 0)
    ).astype(float)

    # Multi-TF agreement: 5m ema + 15m ema + 1h ema all same direction
    result["multi_tf_ema_agree"] = np.where(
        (result["ema_bias_5m"] == 1) & (result["ema_bias_15m"] == 1) & (result["ema_bias_1h"] == 1),  1,
        np.where((result["ema_bias_5m"] == -1) & (result["ema_bias_15m"] == -1) & (result["ema_bias_1h"] == -1), -1, 0)
    ).astype(float)

    # BP agreement: 5m + 15m + 1h all bullish/bearish
    result["multi_tf_bp_agree"] = np.where(
        (result["bp_5m"] > 0.5) & (result["bp_15m"] > 0.5) & (result["bp_1h"] > 0.5),  1,
        np.where((result["bp_5m"] < 0.5) & (result["bp_15m"] < 0.5) & (result["bp_1h"] < 0.5), -1, 0)
    ).astype(float)

    # Stoch crossover at 5m AND 15m same direction
    result["stoch_cross_agree"] = np.where(
        (result["stoch_cross_5m"] == 1) & (result["stoch_cross_15m"] == 1),  1,
        np.where((result["stoch_cross_5m"] == -1) & (result["stoch_cross_15m"] == -1), -1, 0)
    ).astype(float)

    # Squeeze setup: BB tight + volume dropping
    result["bb_squeeze_15m"] = np.where(
        result["bb_width_15m"] < result["bb_width_15m"].rolling(50).quantile(0.20), 1, 0
    ).astype(float)

    # RSI mean reversion: oversold on 15m + 1h bullish ema
    result["rsi_mean_rev_long"] = np.where(
        (result["rsi_15m"] < 35) & (result["ema_bias_1h"] == 1), 1, 0
    ).astype(float)
    result["rsi_mean_rev_short"] = np.where(
        (result["rsi_15m"] > 65) & (result["ema_bias_1h"] == -1), -1, 0
    ).astype(float)

    return result


# ── Coinalyze fetch ───────────────────────────────────────────────────────────

def fetch_coinalyze(symbol: str) -> pd.DataFrame:
    """Fetch 90-day liq + L/S + OI at 15min resolution. Returns indexed DataFrame."""
    now_unix = int(time.time())
    from_unix = now_unix - 90 * 24 * 3600

    params = {"symbols": symbol, "interval": "15min",
              "from": from_unix, "to": now_unix, "api_key": COINALYZE_KEY}

    try:
        r_liq = requests.get(f"{COINALYZE_BASE}/liquidation-history",      params=params, timeout=15)
        time.sleep(0.3)
        r_ls  = requests.get(f"{COINALYZE_BASE}/long-short-ratio-history", params=params, timeout=15)
        time.sleep(0.3)
        r_oi  = requests.get(f"{COINALYZE_BASE}/open-interest-history",    params=params, timeout=15)

        r_liq.raise_for_status(); r_ls.raise_for_status(); r_oi.raise_for_status()

        def rows_to_df(resp, cols):
            rows = resp.json()[0]["history"]
            df = pd.DataFrame(rows)
            df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
            return df.set_index("t").rename(columns=cols)

        liq_df = rows_to_df(r_liq, {"l": "long_liq",  "s": "short_liq"})
        ls_df  = rows_to_df(r_ls,  {"l": "ls_long",   "s": "ls_short"})
        oi_df  = rows_to_df(r_oi,  {"o": "oi"})

        df = liq_df.join(ls_df, how="outer").join(oi_df, how="outer")
        df = df.astype(float)

        total_liq       = df["long_liq"] + df["short_liq"]
        df["liq_bias"]  = np.where(total_liq > 0.001,
                                   (df["short_liq"] - df["long_liq"]) / total_liq, 0.0)
        score = pd.Series(0, index=df.index)
        score += (df["liq_bias"] >= _LIQ_BIAS_STRONG).astype(int)
        score -= (df["liq_bias"] <= -_LIQ_BIAS_STRONG).astype(int)
        score += (df["ls_short"] >= _LS_CROWD_THRESH).astype(int)
        score -= (df["ls_long"]  >= _LS_CROWD_THRESH).astype(int)
        df["liq_score"]  = score.clip(-2, 2)
        df["oi_chg_pct"] = df["oi"].pct_change() * 100.0

        return df[["liq_bias", "liq_score", "ls_long", "ls_short", "oi_chg_pct"]]

    except Exception as e:
        print(f"  [coinalyze] {symbol}: {e}")
        return pd.DataFrame()


# ── correlation engine ────────────────────────────────────────────────────────

def correlate_signals(data: pd.DataFrame, signal_cols: list[str],
                      target_cols: list[str], min_n: int = MIN_N) -> pd.DataFrame:
    """
    For each signal and each target, compute Pearson r, Spearman r, p-value, N.
    Returns a tidy DataFrame.
    """
    rows = []
    for sig in signal_cols:
        if sig not in data.columns:
            continue
        for tgt in target_cols:
            if tgt not in data.columns:
                continue
            sub = data[[sig, tgt]].dropna()
            n   = len(sub)
            if n < min_n:
                continue
            s  = sub[sig].values.astype(float)
            t  = sub[tgt].values.astype(float)
            if np.std(s) < 1e-9 or np.std(t) < 1e-9:
                continue
            pr, pp = stats.pearsonr(s, t)
            sr, sp = stats.spearmanr(s, t)
            rows.append({
                "signal": sig, "target": tgt,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
                "n": n,
            })
    return pd.DataFrame(rows)


def decile_analysis(data: pd.DataFrame, signal: str, target: str,
                    n_deciles: int = DECILES) -> str:
    """Return a text table of mean forward return per signal decile."""
    sub = data[[signal, target]].dropna()
    if len(sub) < n_deciles * 10:
        return ""
    try:
        sub = sub.copy()
        sub["decile"] = pd.qcut(sub[signal], n_deciles, labels=False, duplicates="drop")
        grp = sub.groupby("decile")[target].agg(["mean", "count"])
        lines = [f"    {'Decile':<8} {'N':>6} {'Mean fwd%':>10}"]
        for dec, row in grp.iterrows():
            bar_len = int(abs(row["mean"]) * 2000)
            bar     = ("+" if row["mean"] > 0 else "-") * min(bar_len, 30)
            lines.append(f"    {dec+1:<8} {int(row['count']):>6} {row['mean']*100:>+9.3f}%  {bar}")
        return "\n".join(lines)
    except Exception:
        return ""


# ── reporting ─────────────────────────────────────────────────────────────────

def _sig_category(name: str) -> str:
    if any(x in name for x in ("liq", "ls_", "oi_chg")):
        return "External"
    if any(x in name for x in ("multi_tf", "agree", "mean_rev", "squeeze")):
        return "Combined"
    if any(x in name for x in ("donchian",)):
        return "Donchian"
    if any(x in name for x in ("bb_", "range_ratio", "atr_ratio", "vol_ratio")):
        return "Volatility"
    if any(x in name for x in ("ema_", "ema_gap")):
        return "Trend"
    if any(x in name for x in ("stoch", "rsi", "macd", "consec")):
        return "Momentum"
    if any(x in name for x in ("hammer", "shooting", "doji", "engulf", "wick", "body", "dir")):
        return "Structure"
    if any(x in name for x in ("bp_", "chg_", "z_score", "vwap")):
        return "Price Action"
    return "Other"


def report_asset(asset: str, data: pd.DataFrame, corr_df: pd.DataFrame):
    primary_target = "fwd_15m"    # sort table by this horizon

    print(f"\n{SEP}")
    print(f"  {asset} — {len(data):,} 15m bars  |  signal → fwd_15m / fwd_30m / fwd_1h / fwd_2h")
    print(SEP)

    sub = corr_df[corr_df["target"] == primary_target].copy()
    sub["abs_r"] = sub["pearson_r"].abs()
    sub = sub.sort_values("abs_r", ascending=False)

    # Main table
    print(f"\n  {'Signal':<38} {'Cat':<12} {'r_15m':>7} {'r_30m':>7} {'r_1h':>7} {'r_2h':>7} {'p':>7} {'N':>6}")
    print(f"  {'-'*38} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    printed = 0
    for _, row in sub.iterrows():
        if printed >= TOP_N:
            break
        sig = row["signal"]
        cat = _sig_category(sig)

        def _r(tgt):
            m = corr_df[(corr_df["signal"] == sig) & (corr_df["target"] == tgt)]
            return f"{m['pearson_r'].iloc[0]:+.3f}" if len(m) else "   n/a"

        p_str = f"{row['pearson_p']:.2e}"
        print(f"  {sig:<38} {cat:<12} {row['pearson_r']:+7.3f} {_r('fwd_30m'):>7} {_r('fwd_1h'):>7} {_r('fwd_2h'):>7} {p_str:>7} {int(row['n']):>6}")
        printed += 1

    # Category summary
    print(f"\n  {'Category summary (mean |r| for fwd_15m)':}")
    print(f"  {'-'*50}")
    cats = sub.copy()
    cats["cat"] = cats["signal"].map(_sig_category)
    cat_summary = cats.groupby("cat")["abs_r"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for cat, row in cat_summary.iterrows():
        bar = "█" * int(row["mean"] * 200)
        print(f"  {cat:<14} mean|r|={row['mean']:.4f}  n_signals={int(row['count']):>3}  {bar}")

    # Decile analysis for top 5 signals
    print(f"\n  {'Top-5 decile analysis vs fwd_15m (% forward return per decile)':}")
    top5 = sub.head(5)["signal"].tolist()
    for sig in top5:
        if sig not in data.columns or "fwd_15m" not in data.columns:
            continue
        txt = decile_analysis(data, sig, "fwd_15m")
        if txt:
            r_val = sub[sub["signal"] == sig]["pearson_r"].iloc[0]
            print(f"\n  {sig}  (r={r_val:+.4f})")
            print(txt)

    # Positive vs negative direction breakdown for key binary signals
    print(f"\n  {'Binary signal breakdown (mean fwd_15m when signal = +1 vs −1)':}")
    binary_sigs = [s for s in corr_df[corr_df["target"] == "fwd_15m"]["signal"]
                   if any(x in s for x in ("dir_", "ema_bias", "ema_stack",
                                           "stoch_cross", "macd_cross", "engulf",
                                           "breakout", "liq_score"))]
    print(f"  {'Signal':<38} {'Up mean%':>10} {'Dn mean%':>10} {'Spread':>8} {'N+':>6} {'N-':>6}")
    print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
    for sig in binary_sigs[:20]:
        if sig not in data.columns:
            continue
        sub2 = data[[sig, "fwd_15m"]].dropna()
        pos  = sub2[sub2[sig] > 0]["fwd_15m"]
        neg  = sub2[sub2[sig] < 0]["fwd_15m"]
        if len(pos) < 30 or len(neg) < 30:
            continue
        spread = pos.mean() - neg.mean()
        print(f"  {sig:<38} {pos.mean()*100:>+9.3f}% {neg.mean()*100:>+9.3f}% {spread*100:>+7.3f}% {len(pos):>6} {len(neg):>6}")


def report_coinalyze(asset: str, data: pd.DataFrame, corr_df: pd.DataFrame):
    ext_sigs = [s for s in corr_df["signal"].unique()
                if any(x in s for x in ("liq_bias", "liq_score", "ls_long", "ls_short", "oi_chg"))]
    if not ext_sigs:
        return

    print(f"\n  {'-'*60}")
    print(f"  {asset} COINALYZE signals (90-day window)")
    print(f"  {'Signal':<28} {'r_15m':>7} {'r_30m':>7} {'r_1h':>7} {'r_2h':>7} {'p':>8} {'N':>6}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")

    sub = corr_df[corr_df["target"] == "fwd_15m"].set_index("signal")
    for sig in ext_sigs:
        if sig not in sub.index:
            continue
        row = sub.loc[sig]
        def _r(tgt):
            m = corr_df[(corr_df["signal"] == sig) & (corr_df["target"] == tgt)]
            return f"{m['pearson_r'].iloc[0]:+.3f}" if len(m) else "   n/a"
        p_str = f"{row['pearson_p']:.2e}"
        print(f"  {sig:<28} {row['pearson_r']:+7.3f} {_r('fwd_30m'):>7} {_r('fwd_1h'):>7} {_r('fwd_2h'):>7} {p_str:>8} {int(row['n']):>6}")

        # Decile for continuous, binary breakdown for liq_score
        if "liq_score" in sig:
            sub2 = data[[sig, "fwd_15m"]].dropna()
            for val in sorted(sub2[sig].unique()):
                grp = sub2[sub2[sig] == val]["fwd_15m"]
                if len(grp) < 10:
                    continue
                print(f"    score={int(val):+d}  N={len(grp):>4}  mean={grp.mean()*100:>+7.3f}%  "
                      f"std={grp.std()*100:.3f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def run_asset(asset: str, sym: str):
    print(f"\n{SEP}")
    print(f"  Loading {asset} ({sym}) …")
    df_1m = load_1m(sym)
    print(f"  {len(df_1m):,} 1m bars  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")

    print(f"  Computing signals …", end=" ", flush=True)
    data = build_signals(df_1m)
    print(f"{len(data):,} 15m bars, {len(data.columns)} columns")

    # Fetch Coinalyze (90-day)
    print(f"  Fetching Coinalyze …", end=" ", flush=True)
    cz_sym = COINALYZE_SYMBOLS.get(asset)
    if cz_sym:
        cz = fetch_coinalyze(cz_sym)
        if not cz.empty:
            cz_lagged = cz.shift(1)   # use last completed 15m bar
            data = data.join(cz_lagged.reindex(data.index, method="ffill"), how="left")
            print(f"{len(cz)} Coinalyze bars joined")
        else:
            print("unavailable")
    else:
        print("no symbol")

    # Build signal list (everything except forward targets)
    target_cols = list(FWD_HORIZONS.keys()) + ["fwd_dir_15m"]
    signal_cols = [c for c in data.columns if c not in target_cols]

    print(f"  Running correlations ({len(signal_cols)} signals × {len(FWD_HORIZONS)} horizons) …",
          end=" ", flush=True)
    corr_df = correlate_signals(data, signal_cols, list(FWD_HORIZONS.keys()))
    print(f"{len(corr_df)} valid pairs")

    report_asset(asset, data, corr_df)
    report_coinalyze(asset, data, corr_df)

    return data, corr_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ALL",
                        choices=list(ASSETS.keys()) + ["ALL"])
    args = parser.parse_args()

    print(SEP)
    print("  Comprehensive signal → price direction correlation sweep")
    print("  Horizons: 15m / 30m / 1h / 2h  |  Min N:", MIN_N)
    print(SEP)

    targets = {args.asset: ASSETS[args.asset]} if args.asset != "ALL" else ASSETS
    results = {}
    for asset, sym in targets.items():
        data, corr = run_asset(asset, sym)
        results[asset] = (data, corr)

    # Cross-asset summary: signals that rank in top-15 for ALL assets
    if len(results) == 3:
        print(f"\n{SEP}")
        print("  CROSS-ASSET: signals ranking top-15 by |r_15m| in ALL three assets")
        print(SEP)
        top_sets = []
        for asset, (_, corr) in results.items():
            sub = corr[corr["target"] == "fwd_15m"].copy()
            sub["abs_r"] = sub["pearson_r"].abs()
            top = set(sub.nlargest(15, "abs_r")["signal"])
            top_sets.append(top)
        universal = top_sets[0] & top_sets[1] & top_sets[2]
        if universal:
            print(f"  Found {len(universal)} universal top-15 signals:")
            for sig in sorted(universal):
                vals = []
                for asset, (_, corr) in results.items():
                    m = corr[(corr["signal"] == sig) & (corr["target"] == "fwd_15m")]
                    vals.append(f"{asset}:{m['pearson_r'].iloc[0]:+.3f}" if len(m) else f"{asset}:n/a")
                print(f"  {sig:<40} {' | '.join(vals)}")
        else:
            # Show top-10 overlap (any 2 of 3)
            print("  No signal in top-15 for all three assets. Top-10 appearing in ≥2 assets:")
            from collections import Counter
            all_top = []
            for asset, (_, corr) in results.items():
                sub = corr[corr["target"] == "fwd_15m"].copy()
                sub["abs_r"] = sub["pearson_r"].abs()
                all_top.extend(sub.nlargest(10, "abs_r")["signal"].tolist())
            cnt = Counter(all_top)
            for sig, c in sorted(cnt.items(), key=lambda x: -x[1]):
                if c < 2:
                    continue
                vals = []
                for asset, (_, corr) in results.items():
                    m = corr[(corr["signal"] == sig) & (corr["target"] == "fwd_15m")]
                    vals.append(f"{asset}:{m['pearson_r'].iloc[0]:+.3f}" if len(m) else f"{asset}:n/a")
                print(f"  {sig:<40} (in {c}/3)  {' | '.join(vals)}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
