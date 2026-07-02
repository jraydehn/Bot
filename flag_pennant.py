"""
flag_pennant.py

Self-contained flag/pennant pattern detector.  All dependencies
(rw_top/rw_bottom, fit_trendlines_single) are implemented inline.
Uses the trendline confirmation method only — more robust than PIP method.

Key columns produced at each bar (log-price space internally, output in real price):
  flag_bull_bars_ago   : bars since last confirmed bull flag  (-1 = none in lookback)
  flag_bear_bars_ago   : bars since last confirmed bear flag  (-1 = none in lookback)
  flag_bull_tip_y      : pole top price of most recent bull flag
  flag_bear_tip_y      : pole bottom price of most recent bear flag
  flag_bull_pole_pct   : pole height as % of base price
  flag_bear_pole_pct   : pole depth as % of base price
  flag_signal          : +1 recent bull, -1 recent bear, 0 none
                         (bull takes priority when both active)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ── Rolling-window top / bottom ───────────────────────────────────────────────

def rw_top(data: np.ndarray, i: int, order: int) -> bool:
    """True if data[i-order] is the high of the window [i-2*order, i]."""
    if i < 2 * order:
        return False
    pivot = i - order
    window = data[i - 2 * order: i + 1]
    return float(data[pivot]) == float(window.max())


def rw_bottom(data: np.ndarray, i: int, order: int) -> bool:
    """True if data[i-order] is the low of the window [i-2*order, i]."""
    if i < 2 * order:
        return False
    pivot = i - order
    window = data[i - 2 * order: i + 1]
    return float(data[pivot]) == float(window.min())


# ── Trendline fitting ─────────────────────────────────────────────────────────

def fit_trendlines_single(data: np.ndarray):
    """
    Fit a parallel price channel to data.

    Returns ((support_slope, support_intercept), (resist_slope, resist_intercept)).
    Intercepts are at index 0.  Both lines share the OLS slope; intercepts
    are shifted to be tangent to the price min/max (parallel channel).
    """
    n = len(data)
    if n < 2:
        return (0.0, float(data[0])), (0.0, float(data[0]))

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = data.mean()
    denom = np.sum((x - x_mean) ** 2)
    slope = float(np.sum((x - x_mean) * (data - y_mean)) / denom) if denom > 0 else 0.0

    # Detrend → find intercept offsets
    detrended = data - slope * x
    support_intercept = float(detrended.min())
    resist_intercept  = float(detrended.max())

    return (slope, support_intercept), (slope, resist_intercept)


# ── Pattern dataclass ─────────────────────────────────────────────────────────

@dataclass
class FlagPattern:
    base_x: int
    base_y: float
    tip_x:  int   = -1
    tip_y:  float = -1.
    conf_x: int   = -1
    conf_y: float = -1.
    pennant: bool = False
    flag_width:    int   = -1
    flag_height:   float = -1.
    pole_width:    int   = -1
    pole_height:   float = -1.
    support_intercept: float = -1.
    support_slope:     float = -1.
    resist_intercept:  float = -1.
    resist_slope:      float = -1.


# ── Bull / bear pattern checks (trendline method) ────────────────────────────

def _check_bull(pending: FlagPattern, data: np.ndarray, i: int, order: int) -> bool:
    if data[pending.tip_x + 1: i].size and data[pending.tip_x + 1: i].max() > pending.tip_y:
        return False

    flag_min   = data[pending.tip_x: i].min()
    pole_height = pending.tip_y - pending.base_y
    pole_width  = pending.tip_x - pending.base_x
    flag_height = pending.tip_y - flag_min
    flag_width  = i - pending.tip_x

    if pole_height <= 0 or pole_width <= 0:
        return False
    if flag_width > pole_width * 0.5:
        return False
    if flag_height > pole_height * 0.75:
        return False

    flag_data = data[pending.tip_x: i]
    if len(flag_data) < 3:
        return False

    (s_slope, s_intercept), (r_slope, r_intercept) = fit_trendlines_single(flag_data)

    # Confirm breakout above resistance
    current_resist = r_intercept + r_slope * (flag_width + 1)
    if data[i] <= current_resist:
        return False

    pending.pennant         = s_slope > 0
    pending.conf_x          = i
    pending.conf_y          = float(data[i])
    pending.flag_width      = flag_width
    pending.flag_height     = flag_height
    pending.pole_width      = pole_width
    pending.pole_height     = pole_height
    pending.support_slope   = s_slope
    pending.support_intercept = s_intercept
    pending.resist_slope    = r_slope
    pending.resist_intercept = r_intercept
    return True


def _check_bear(pending: FlagPattern, data: np.ndarray, i: int, order: int) -> bool:
    if data[pending.tip_x + 1: i].size and data[pending.tip_x + 1: i].min() < pending.tip_y:
        return False

    flag_max    = data[pending.tip_x: i].max()
    pole_height = pending.base_y - pending.tip_y
    pole_width  = pending.tip_x - pending.base_x
    flag_height = flag_max - pending.tip_y
    flag_width  = i - pending.tip_x

    if pole_height <= 0 or pole_width <= 0:
        return False
    if flag_width > pole_width * 0.5:
        return False
    if flag_height > pole_height * 0.75:
        return False

    flag_data = data[pending.tip_x: i]
    if len(flag_data) < 3:
        return False

    (s_slope, s_intercept), (r_slope, r_intercept) = fit_trendlines_single(flag_data)

    # Confirm breakdown below support
    current_support = s_intercept + s_slope * (flag_width + 1)
    if data[i] >= current_support:
        return False

    pending.pennant         = r_slope < 0
    pending.conf_x          = i
    pending.conf_y          = float(data[i])
    pending.flag_width      = flag_width
    pending.flag_height     = flag_height
    pending.pole_width      = pole_width
    pending.pole_height     = pole_height
    pending.support_slope   = s_slope
    pending.support_intercept = s_intercept
    pending.resist_slope    = r_slope
    pending.resist_intercept = r_intercept
    return True


# ── Main detection loop ───────────────────────────────────────────────────────

def find_flags_pennants(
    data: np.ndarray, order: int = 10
) -> tuple[list, list, list, list]:
    """
    Run flag/pennant detection on a 1D close-price array (log prices recommended).

    Returns (bull_flags, bear_flags, bull_pennants, bear_pennants).
    Each item is a FlagPattern with conf_x = confirmation bar index.
    """
    assert order >= 3
    pending_bull = None
    pending_bear = None
    last_top = last_bottom = -1

    bull_flags = []; bear_flags = []
    bull_pennants = []; bear_pennants = []

    for i in range(len(data)):
        if rw_top(data, i, order):
            last_top = i - order
            if last_bottom != -1:
                p = FlagPattern(last_bottom, float(data[last_bottom]))
                p.tip_x = last_top
                p.tip_y = float(data[last_top])
                pending_bull = p

        if rw_bottom(data, i, order):
            last_bottom = i - order
            if last_top != -1:
                p = FlagPattern(last_top, float(data[last_top]))
                p.tip_x = last_bottom
                p.tip_y = float(data[last_bottom])
                pending_bear = p

        if pending_bear is not None:
            if _check_bear(pending_bear, data, i, order):
                (bear_pennants if pending_bear.pennant else bear_flags).append(pending_bear)
                pending_bear = None

        if pending_bull is not None:
            if _check_bull(pending_bull, data, i, order):
                (bull_pennants if pending_bull.pennant else bull_flags).append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags, bull_pennants, bear_pennants


# ── Signal series builder ─────────────────────────────────────────────────────

def build_signal_series(
    close: pd.Series,
    order: int = 10,
    lookback_bars: int = 48,
) -> pd.DataFrame:
    """
    Run flag/pennant detection on a log-price close series and return a
    DataFrame indexed by bar timestamp with one row per bar.

    Columns:
      flag_bull_bars_ago  : bars since last bull flag/pennant confirmed (-1 = none)
      flag_bear_bars_ago  : bars since last bear flag/pennant confirmed (-1 = none)
      flag_bull_tip_y     : actual price at top of bull pole (np.nan if none)
      flag_bear_tip_y     : actual price at bottom of bear pole (np.nan if none)
      flag_bull_pole_pct  : pole height as % of base price (np.nan if none)
      flag_bear_pole_pct  : pole depth as % of base price (np.nan if none)
      flag_signal         : +1 bull active, -1 bear active, 0 none
    """
    log_c = np.log(close.values.astype(float))
    idx   = close.index
    n     = len(log_c)

    bull_f, bear_f, bull_p, bear_p = find_flags_pennants(log_c, order=order)

    # Merge flags + pennants; sort by conf_x
    bull_all = sorted(bull_f + bull_p, key=lambda p: p.conf_x)
    bear_all = sorted(bear_f + bear_p, key=lambda p: p.conf_x)

    # Build bar-level arrays
    bull_bars_ago = np.full(n, -1, dtype=float)
    bear_bars_ago = np.full(n, -1, dtype=float)
    bull_tip_y    = np.full(n, np.nan)
    bear_tip_y    = np.full(n, np.nan)
    bull_pole_pct = np.full(n, np.nan)
    bear_pole_pct = np.full(n, np.nan)

    # For each bar, look back to find the most recent confirmed pattern
    bull_ptr = 0; bear_ptr = 0

    for i in range(n):
        # Advance pointers to most recent confirmed pattern visible at bar i
        while bull_ptr < len(bull_all) - 1 and bull_all[bull_ptr + 1].conf_x <= i:
            bull_ptr += 1
        while bear_ptr < len(bear_all) - 1 and bear_all[bear_ptr + 1].conf_x <= i:
            bear_ptr += 1

        # Bull
        if bull_ptr < len(bull_all) and bull_all[bull_ptr].conf_x <= i:
            pat = bull_all[bull_ptr]
            ago = i - pat.conf_x
            if ago <= lookback_bars:
                bull_bars_ago[i] = ago
                bull_tip_y[i]    = float(np.exp(pat.tip_y))   # back to real price
                base_price       = float(np.exp(pat.base_y))
                bull_pole_pct[i] = (float(np.exp(pat.tip_y)) - base_price) / base_price * 100

        # Bear
        if bear_ptr < len(bear_all) and bear_all[bear_ptr].conf_x <= i:
            pat = bear_all[bear_ptr]
            ago = i - pat.conf_x
            if ago <= lookback_bars:
                bear_bars_ago[i] = ago
                bear_tip_y[i]    = float(np.exp(pat.tip_y))
                base_price       = float(np.exp(pat.base_y))
                bear_pole_pct[i] = (base_price - float(np.exp(pat.tip_y))) / base_price * 100

    # Combined signal: bull takes priority when both active
    flag_signal = np.zeros(n, dtype=float)
    flag_signal[bull_bars_ago >= 0] =  1.0
    flag_signal[bear_bars_ago >= 0] = -1.0
    flag_signal[(bull_bars_ago >= 0) & (bear_bars_ago >= 0)] = 1.0  # bull priority

    return pd.DataFrame({
        "flag_bull_bars_ago": bull_bars_ago,
        "flag_bear_bars_ago": bear_bars_ago,
        "flag_bull_tip_y":    bull_tip_y,
        "flag_bear_tip_y":    bear_tip_y,
        "flag_bull_pole_pct": bull_pole_pct,
        "flag_bear_pole_pct": bear_pole_pct,
        "flag_signal":        flag_signal,
    }, index=idx)
