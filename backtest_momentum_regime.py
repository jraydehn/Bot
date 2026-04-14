"""
backtest_momentum_regime.py

Tests whether short-term indicators act as CONTINUATION vs MEAN-REVERSION signals
during sharp price moves (≥0.8% in 30 minutes).

For each candidate indicator, computes:
  - Baseline win rate (no sharp move filter)
  - Win rate during sharp UP moves when indicator is overbought (classically bearish)
  - Win rate during sharp DOWN moves when indicator is oversold  (classically bullish)

If the indicator is a CONTINUATION signal during sharp moves, then:
  - Overbought during rally  → price continues UP   (classical reversion fails)
  - Oversold during drop     → price continues DOWN  (classical reversion fails)

Lookforward windows: 15m, 30m, 60m

Usage:
    python3 backtest_momentum_regime.py
    python3 backtest_momentum_regime.py --thresh 0.006   # looser threshold
    python3 backtest_momentum_regime.py --fwd 60         # 60m lookforward only
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
SHARP_THRESH = 0.008       # 0.8% in 30 min → sharp move
LOOKFORWARD_WINDOWS = [15, 30, 60]   # minutes
MIN_SAMPLES = 30           # skip any cell with fewer samples


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS  (applied to 1m data)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _stoch_k(h, l, c, k=14):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _atr(h, l, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()


def _keltner_pct(h, l, c, span=20, mult=2):
    ema = _ema(c, span)
    atr = _atr(h, l, c, span)
    dn = ema - mult * atr
    up = ema + mult * atr
    w = (up - dn).replace(0, np.nan)
    return (c - dn) / w


def _bb_pct(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    dn = mid - 2 * std
    up = mid + 2 * std
    rng = (up - dn).replace(0, np.nan)
    return (c - dn) / rng


def _dc_pct(h, l, c, n=20):
    dc_h = h.rolling(n).max()
    dc_l = l.rolling(n).min()
    rng = (dc_h - dc_l).replace(0, np.nan)
    return (c - dc_l) / rng


def _wpr(h, l, c, p=14):
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def _macd_hist(c, f=12, s=26, sig=9):
    ema_f = _ema(c, f)
    ema_s = _ema(c, s)
    macd = ema_f - ema_s
    signal = _ema(macd, sig)
    return macd - signal


def _vwap_deviation(c, v):
    """Daily-reset VWAP deviation as fraction of price."""
    date_idx = c.index.normalize()
    tpv = c * v
    cum_tpv = tpv.groupby(date_idx).cumsum()
    cum_vol = v.groupby(date_idx).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return (c - vwap) / vwap


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_1m() -> pd.DataFrame:
    """Load the most recent BTC 1m parquet file."""
    files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1m_2024-01-01_*.parquet"))
    if not files:
        raise FileNotFoundError("No BTC 1m parquet found in data/")
    path = files[-1]
    print(f"Loading {path.name} …")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    print(f"  {len(df):,} 1m bars  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INDICATOR SIGNALS  (on 1m data, evaluated at each bar)
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all candidate indicators on 1m data.
    Returns a DataFrame with one row per 1m bar.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    sig = pd.DataFrame(index=df.index)

    # ── 30m price change (sharp move detection) ────────────────────────────
    sig["chg_30m"] = c / c.shift(30) - 1

    # ── 30m forward return (outcome) ──────────────────────────────────────
    for fwd in LOOKFORWARD_WINDOWS:
        sig[f"fwd_{fwd}m"] = c.shift(-fwd) / c - 1

    # ── Stochastic K — 15-bar (fast, ~15m equivalent on 1m data) ──────────
    # We use 15-bar and 60-bar windows to approximate 15m and 1h signals
    stk15 = _stoch_k(h, l, c, 15)
    stk60 = _stoch_k(h, l, c, 60)

    sig["stoch_15_oversold"]   = stk15 < 20
    sig["stoch_15_overbought"] = stk15 > 80
    sig["stoch_15_extreme_os"] = stk15 < 10
    sig["stoch_15_extreme_ob"] = stk15 > 90
    sig["stoch_15_val"]        = stk15

    sig["stoch_60_oversold"]   = stk60 < 20
    sig["stoch_60_overbought"] = stk60 > 80
    sig["stoch_60_extreme_os"] = stk60 < 10
    sig["stoch_60_extreme_ob"] = stk60 > 90
    sig["stoch_60_val"]        = stk60

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14)
    rsi60 = _rsi(c, 60)

    sig["rsi_14_oversold"]   = rsi14 < 30
    sig["rsi_14_overbought"] = rsi14 > 70
    sig["rsi_60_oversold"]   = rsi60 < 30
    sig["rsi_60_overbought"] = rsi60 > 70
    sig["rsi_14_val"]        = rsi14

    # ── EMA slope ─────────────────────────────────────────────────────────
    ema9  = _ema(c, 9)
    ema21 = _ema(c, 21)
    ema9_slope  = ema9.diff(5)    # 5-bar slope
    ema21_slope = ema21.diff(5)

    sig["ema9_rising"]   = ema9_slope > 0
    sig["ema9_falling"]  = ema9_slope < 0
    sig["ema21_rising"]  = ema21_slope > 0
    sig["ema21_falling"] = ema21_slope < 0
    sig["ema_cross_bull"] = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
    sig["ema_cross_bear"] = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))

    # ── MACD histogram ────────────────────────────────────────────────────
    mh = _macd_hist(c, 12, 26, 9)
    sig["macd_hist_positive"] = mh > 0
    sig["macd_hist_negative"] = mh < 0
    sig["macd_hist_rising"]   = mh > mh.shift(1)
    sig["macd_hist_falling"]  = mh < mh.shift(1)

    # ── Keltner channel (20-bar, 2×ATR) ───────────────────────────────────
    kc_pct = _keltner_pct(h, l, c, 20, 2)
    sig["kc_below"]      = kc_pct < 0       # below lower band
    sig["kc_lower_zone"] = kc_pct < 0.20
    sig["kc_upper_zone"] = kc_pct > 0.80
    sig["kc_above"]      = kc_pct > 1.0

    # ── Bollinger Band position ────────────────────────────────────────────
    bb_pct = _bb_pct(c, 20)
    sig["bb_lower_zone"] = bb_pct < 0.20
    sig["bb_upper_zone"] = bb_pct > 0.80
    sig["bb_below"]      = bb_pct < 0
    sig["bb_above"]      = bb_pct > 1

    # ── Donchian channel (20-bar) ──────────────────────────────────────────
    dc_pct = _dc_pct(h, l, c, 20)
    sig["dc_lower_zone"] = dc_pct < 0.20
    sig["dc_upper_zone"] = dc_pct > 0.80
    sig["dc_near_low"]   = dc_pct < 0.10
    sig["dc_near_high"]  = dc_pct > 0.90

    # ── Williams %R (14-bar) ───────────────────────────────────────────────
    wpr = _wpr(h, l, c, 14)
    sig["wpr_oversold"]   = wpr < -80
    sig["wpr_overbought"] = wpr > -20

    # ── Volume surge ──────────────────────────────────────────────────────
    vol_ma = v.rolling(20).mean()
    vol_ratio = v / vol_ma.replace(0, np.nan)
    sig["vol_surge"]  = vol_ratio > 1.5
    sig["vol_quiet"]  = vol_ratio < 0.5
    sig["vol_ratio"]  = vol_ratio

    # ── VWAP deviation ────────────────────────────────────────────────────
    vwap_dev = _vwap_deviation(c, v)
    sig["vwap_far_above"] = vwap_dev > 0.005
    sig["vwap_above"]     = vwap_dev > 0.001
    sig["vwap_below"]     = vwap_dev < -0.001
    sig["vwap_far_below"] = vwap_dev < -0.005

    # ── Price momentum (short ROC) ─────────────────────────────────────────
    roc5  = c / c.shift(5)  - 1   # 5m ROC
    roc15 = c / c.shift(15) - 1   # 15m ROC
    sig["roc5_positive"]  = roc5 > 0
    sig["roc5_negative"]  = roc5 < 0
    sig["roc15_positive"] = roc15 > 0
    sig["roc15_negative"] = roc15 < 0

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_indicator(sig: pd.DataFrame,
                      signal_col: str,
                      fwd_col: str,
                      direction: str,    # "up" or "down"
                      sharp_thresh: float) -> dict:
    """
    For a binary signal column (True/False), compute:
      - baseline: win rate (continuation in `direction`) when signal is True, all periods
      - sharp_move: same but only during sharp moves in `direction`
      - sharp_no_signal: sharp move but signal is False (control group)

    `direction` = "up"  → sharp move = chg_30m > +thresh; continuation = fwd > 0
    `direction` = "down" → sharp move = chg_30m < -thresh; continuation = fwd < 0
    """
    if direction == "up":
        sharp_mask = sig["chg_30m"] > sharp_thresh
        win = sig[fwd_col] > 0       # continuation up
    else:
        sharp_mask = sig["chg_30m"] < -sharp_thresh
        win = sig[fwd_col] < 0       # continuation down

    signal_on  = sig[signal_col].fillna(False).astype(bool)
    valid       = sig[fwd_col].notna() & sig["chg_30m"].notna() & signal_on

    # Baseline (all periods where signal is True)
    base_mask   = valid
    base_n      = base_mask.sum()
    base_win    = win[base_mask].mean() if base_n >= MIN_SAMPLES else np.nan

    # Sharp move + signal True
    sharp_mask2 = valid & sharp_mask
    sharp_n     = sharp_mask2.sum()
    sharp_win   = win[sharp_mask2].mean() if sharp_n >= MIN_SAMPLES else np.nan

    # Sharp move + signal False (control)
    ctrl_mask   = sig[fwd_col].notna() & sig["chg_30m"].notna() & sharp_mask & ~signal_on
    ctrl_n      = ctrl_mask.sum()
    ctrl_win    = win[ctrl_mask].mean() if ctrl_n >= MIN_SAMPLES else np.nan

    return {
        "signal":      signal_col,
        "direction":   direction,
        "fwd":         fwd_col,
        "base_n":      int(base_n),
        "base_win%":   round(base_win * 100, 1) if base_win == base_win else None,
        "sharp_n":     int(sharp_n),
        "sharp_win%":  round(sharp_win * 100, 1) if sharp_win == sharp_win else None,
        "ctrl_n":      int(ctrl_n),
        "ctrl_win%":   round(ctrl_win * 100, 1) if ctrl_win == ctrl_win else None,
        "delta":       round((sharp_win - base_win) * 100, 1) if (sharp_win == sharp_win and base_win == base_win) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE SIGNAL DEFINITIONS
# Each tuple: (signal_col, classical_interpretation, test_direction)
# test_direction = "up"  → during sharp rally, does this signal (classically bearish reversion)
#                          predict continuation up?  (flip = good for momentum regime)
# test_direction = "down" → during sharp drop, does this signal (classically bullish reversion)
#                          predict continuation down?
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = [
    # Stochastic — oversold = classically bullish reversion → test if continuation during drops
    ("stoch_15_oversold",   "oversold → expect bounce (classically)",    "down"),
    ("stoch_15_overbought", "overbought → expect fade (classically)",    "up"),
    ("stoch_15_extreme_os", "extreme oversold <10",                       "down"),
    ("stoch_15_extreme_ob", "extreme overbought >90",                     "up"),
    ("stoch_60_oversold",   "60-bar stoch oversold",                     "down"),
    ("stoch_60_overbought", "60-bar stoch overbought",                   "up"),
    ("stoch_60_extreme_os", "60-bar stoch extreme oversold",             "down"),
    ("stoch_60_extreme_ob", "60-bar stoch extreme overbought",           "up"),

    # RSI
    ("rsi_14_oversold",   "RSI<30 → expect bounce (classically)",       "down"),
    ("rsi_14_overbought", "RSI>70 → expect fade (classically)",         "up"),
    ("rsi_60_oversold",   "RSI-60bar <30",                               "down"),
    ("rsi_60_overbought", "RSI-60bar >70",                               "up"),

    # EMA slope — trending direction → test as continuation signal
    ("ema9_rising",   "EMA rising → continuation up",  "up"),
    ("ema9_falling",  "EMA falling → continuation dn", "down"),
    ("ema21_rising",  "EMA21 rising",                  "up"),
    ("ema21_falling", "EMA21 falling",                 "down"),
    ("ema_cross_bull", "EMA bullish cross",            "up"),
    ("ema_cross_bear", "EMA bearish cross",            "down"),

    # MACD
    ("macd_hist_positive", "MACD hist positive → bullish",  "up"),
    ("macd_hist_negative", "MACD hist negative → bearish",  "down"),
    ("macd_hist_rising",   "MACD hist rising → momentum",   "up"),
    ("macd_hist_falling",  "MACD hist falling → weakness",  "down"),

    # Keltner
    ("kc_below",      "below KC → classically bullish reversion", "down"),
    ("kc_lower_zone", "KC lower zone",                             "down"),
    ("kc_upper_zone", "KC upper zone → classically bearish",      "up"),
    ("kc_above",      "above KC → classically bearish",           "up"),

    # Bollinger
    ("bb_below",      "below lower BB",                           "down"),
    ("bb_lower_zone", "BB lower zone",                            "down"),
    ("bb_upper_zone", "BB upper zone",                            "up"),
    ("bb_above",      "above upper BB",                           "up"),

    # Donchian
    ("dc_near_low",   "near DC low → classically bullish",  "down"),
    ("dc_lower_zone", "DC lower zone",                       "down"),
    ("dc_upper_zone", "DC upper zone",                       "up"),
    ("dc_near_high",  "near DC high",                        "up"),

    # Williams %R
    ("wpr_oversold",   "WPR oversold (<-80) → classically bullish", "down"),
    ("wpr_overbought", "WPR overbought (>-20) → classically bearish","up"),

    # Volume
    ("vol_surge", "Volume surge → amplifier",  "up"),
    ("vol_surge", "Volume surge → amplifier",  "down"),
    ("vol_quiet", "Low volume → weak move",    "up"),
    ("vol_quiet", "Low volume → weak move",    "down"),

    # VWAP
    ("vwap_far_above", "far above VWAP → classically bearish", "up"),
    ("vwap_above",     "above VWAP → classically bearish",     "up"),
    ("vwap_below",     "below VWAP → classically bullish",     "down"),
    ("vwap_far_below", "far below VWAP",                       "down"),

    # ROC
    ("roc5_positive",  "5m ROC positive",  "up"),
    ("roc5_negative",  "5m ROC negative",  "down"),
    ("roc15_positive", "15m ROC positive", "up"),
    ("roc15_negative", "15m ROC negative", "down"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SHARP MOVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def sharp_move_base_rates(sig: pd.DataFrame, thresh: float) -> None:
    """Print how often sharp moves continue vs revert, with no signal filter."""
    print("\n" + "=" * 70)
    print("SHARP MOVE BASE CONTINUATION RATES (no indicator filter)")
    print("=" * 70)

    for direction in ("up", "down"):
        if direction == "up":
            sharp = sig["chg_30m"] > thresh
            label = f"Sharp UP  (30m chg > +{thresh*100:.1f}%)"
        else:
            sharp = sig["chg_30m"] < -thresh
            label = f"Sharp DOWN (30m chg < -{thresh*100:.1f}%)"

        n_sharp = sharp.sum()
        print(f"\n{label}  n={n_sharp:,}")

        for fwd in LOOKFORWARD_WINDOWS:
            fwd_col = f"fwd_{fwd}m"
            valid   = sharp & sig[fwd_col].notna() & sig["chg_30m"].notna()
            if valid.sum() < MIN_SAMPLES:
                continue
            if direction == "up":
                cont_pct = (sig.loc[valid, fwd_col] > 0).mean() * 100
            else:
                cont_pct = (sig.loc[valid, fwd_col] < 0).mean() * 100
            n = valid.sum()
            print(f"  fwd {fwd:2d}m  continuation={cont_pct:.1f}%  n={n:,}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(sharp_thresh: float, fwd_filter: int = None) -> None:
    df   = load_1m()
    sig  = build_signals(df)

    # Drop rows where 30m change can't be computed
    sig  = sig.dropna(subset=["chg_30m"])
    print(f"  {len(sig):,} bars with valid 30m change")

    # Sharp move distribution
    sharp_up   = (sig["chg_30m"] > sharp_thresh).sum()
    sharp_down = (sig["chg_30m"] < -sharp_thresh).sum()
    total      = len(sig)
    print(f"\n  Sharp UP   (>{sharp_thresh*100:.1f}%): {sharp_up:,}  ({sharp_up/total*100:.1f}% of all bars)")
    print(f"  Sharp DOWN (<-{sharp_thresh*100:.1f}%): {sharp_down:,}  ({sharp_down/total*100:.1f}% of all bars)")

    sharp_move_base_rates(sig, sharp_thresh)

    # Per-indicator analysis
    fwd_windows = [fwd_filter] if fwd_filter else LOOKFORWARD_WINDOWS
    rows = []
    for (signal_col, description, direction) in CANDIDATES:
        for fwd in fwd_windows:
            fwd_col = f"fwd_{fwd}m"
            r = analyse_indicator(sig, signal_col, fwd_col, direction, sharp_thresh)
            r["description"] = description
            rows.append(r)

    results = pd.DataFrame(rows)
    results = results.dropna(subset=["sharp_win%", "base_win%"])

    # Flag signals where sharp_win% strongly supports CONTINUATION
    # i.e. continuation rate during sharp moves > 60%
    results["continuation_signal"] = results["sharp_win%"] >= 60

    print("\n" + "=" * 70)
    print("INDICATOR RESULTS DURING SHARP MOVES")
    print("Columns: signal | direction | fwd | base_win% | sharp_win% | ctrl_win% | delta | continuation?")
    print("=" * 70)
    print()

    for direction in ("up", "down"):
        for fwd in fwd_windows:
            subset = results[(results["direction"] == direction) & (results["fwd"] == f"fwd_{fwd}m")].copy()
            subset = subset.sort_values("sharp_win%", ascending=False)

            dir_label = "SHARP UP  → does signal predict continuation UP?" if direction == "up" else "SHARP DOWN → does signal predict continuation DOWN?"
            print(f"\n── {dir_label}  (lookforward={fwd}m) ──")
            print(f"{'Signal':<30} {'base_n':>8} {'base%':>7} {'sharp_n':>8} {'sharp%':>8} {'ctrl%':>7} {'Δ':>6}  {'Cont?'}")
            print("-" * 95)

            for _, r in subset.iterrows():
                marker = "  ✓ YES" if r["continuation_signal"] else "       "
                sn = f"{r['sharp_n']:,}" if r["sharp_n"] else "-"
                print(
                    f"{r['signal']:<30} {r['base_n']:>8,} {r['base_win%']:>6.1f}%"
                    f" {sn:>8} {r['sharp_win%']:>7.1f}%"
                    f" {(r['ctrl_win%'] if r['ctrl_win%'] else 0):>6.1f}%"
                    f" {(r['delta'] if r['delta'] else 0):>+6.1f}%"
                    f"  {marker}"
                )

    # Summary: strong continuation signals across all fwd windows
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNALS THAT SUPPORT CONTINUATION IN MAJORITY OF FWD WINDOWS")
    print("(sharp_win% ≥ 60% in ≥ 2 of 3 lookforward windows)")
    print("=" * 70)

    summary = (
        results[results["continuation_signal"]]
        .groupby(["signal", "direction"])
        .agg(
            windows_passing=("continuation_signal", "sum"),
            avg_sharp_win=("sharp_win%", "mean"),
            avg_base_win=("base_win%", "mean"),
            avg_sharp_n=("sharp_n", "mean"),
        )
        .reset_index()
        .query("windows_passing >= 2")
        .sort_values(["direction", "avg_sharp_win"], ascending=[True, False])
    )

    if summary.empty:
        print("  No signals passed the ≥2/3 windows filter at 60% continuation threshold.")
        print("  Try looser threshold with --thresh 0.006 or lower continuation bar.")
    else:
        print(f"\n{'Signal':<30} {'Dir':>5} {'Windows':>8} {'avg_sharp%':>11} {'avg_base%':>10} {'avg_sharp_n':>12}")
        print("-" * 80)
        for _, r in summary.iterrows():
            print(
                f"{r['signal']:<30} {r['direction']:>5} {int(r['windows_passing']):>8}"
                f" {r['avg_sharp_win']:>10.1f}%  {r['avg_base_win']:>9.1f}%  {int(r['avg_sharp_n']):>11,}"
            )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest momentum regime indicators")
    parser.add_argument("--thresh", type=float, default=SHARP_THRESH,
                        help="Sharp move threshold (default 0.008 = 0.8%%)")
    parser.add_argument("--fwd",   type=int,   default=None,
                        help="Single lookforward window in minutes (default: test 15, 30, 60)")
    args = parser.parse_args()

    main(sharp_thresh=args.thresh, fwd_filter=args.fwd)
