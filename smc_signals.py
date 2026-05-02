"""
smc_signals.py — Smart Money Concepts (SMC) structural signals.

Computes three families of signals from OHLCV data:

  1. Break of Structure (BOS): direction of the most recent confirmed
     structural break. Bullish BOS = price closed above a prior swing high.
     Bearish BOS = price closed below a prior swing low.

  2. Change of Character (ChoCH): the first BOS in the opposite direction of
     the prevailing structure — signals a potential regime flip before EMA
     crosses or composite indicators confirm it.

  3. Supply & Demand Zones: price areas where a large impulse candle originated.
     Demand zone = base of a bullish impulse (price expected to bounce from).
     Supply zone  = base of a bearish impulse (price expected to reject from).
     Zones are removed ("mitigated") once price re-enters them.

Timeframe design:
  - 4h BOS/ChoCH  → structural regime (changes slowly, infrequently)
  - 1h BOS/ChoCH  → tactical signal (changes within a trading session)
  - 4h zones       → key price levels (strong institutional interest)

All signals are currently diagnostic only. No gating is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SMCResult:
    # 4h structural regime
    bos_4h: str                      # 'bullish' | 'bearish' | 'neutral'
    choch_4h: bool                   # True if last 4h BOS reversed prior structure
    swing_high_4h: Optional[float]   # most recent confirmed 4h pivot high price
    swing_low_4h: Optional[float]    # most recent confirmed 4h pivot low price

    # 1h tactical signal
    bos_1h: str                      # 'bullish' | 'bearish' | 'neutral'
    choch_1h: bool                   # True if last 1h BOS reversed prior structure
    swing_high_1h: Optional[float]
    swing_low_1h: Optional[float]

    # Supply / demand zones (4h based)
    nearest_supply_pct: Optional[float]  # % above spot to nearest unmitigated supply zone
    nearest_demand_pct: Optional[float]  # % below spot to nearest unmitigated demand zone
    in_supply_zone: bool                 # spot is currently inside a supply zone
    in_demand_zone: bool                 # spot is currently inside a demand zone
    n_supply_zones: int                  # total active supply zones found
    n_demand_zones: int                  # total active demand zones found


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------

def _find_pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[list[int], list[int]]:
    """
    Find confirmed pivot highs and lows using a symmetric n-bar window.

    A pivot high at position i: high[i] is the strict maximum of the
    2n+1 bar window [i-n, i+n]. Requires n confirmed bars on each side,
    so the most recent possible pivot is at index len-n-1 (no lookahead).

    Args:
        high, low : raw numpy arrays
        n         : bars required on each side for confirmation

    Returns:
        pivot_highs, pivot_lows : lists of integer bar indices
    """
    N = len(high)
    pivot_highs: list[int] = []
    pivot_lows:  list[int] = []

    for i in range(n, N - n):
        window_h = high[i - n : i + n + 1]
        window_l = low[i - n : i + n + 1]
        if high[i] == window_h.max() and (window_h == high[i]).sum() == 1:
            pivot_highs.append(i)
        if low[i] == window_l.min() and (window_l == low[i]).sum() == 1:
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


# ---------------------------------------------------------------------------
# BOS / ChoCH detection
# ---------------------------------------------------------------------------

def _detect_bos_choch(
    df: pd.DataFrame,
    n: int = 5,
) -> tuple[str, bool, Optional[float], Optional[float]]:
    """
    Detect Break of Structure (BOS) and Change of Character (ChoCH).

    Algorithm:
      1. Find all confirmed pivot highs and lows.
      2. For each pivot, locate the first subsequent close that breaks it:
           - Close above pivot high  → bullish BOS at that bar
           - Close below pivot low   → bearish BOS at that bar
      3. Sort all BOS events chronologically. The last event = current structure.
      4. ChoCH = last event direction ≠ second-to-last event direction.

    Returns:
        bos_direction : 'bullish' | 'bearish' | 'neutral'
        choch         : True if the last BOS reversed the prior one
        last_sh_price : price of most recent confirmed pivot high (None if unavailable)
        last_sl_price : price of most recent confirmed pivot low  (None if unavailable)
    """
    if len(df) < n * 2 + 10:
        return "neutral", False, None, None

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values

    ph_idx, pl_idx = _find_pivots(high, low, n)

    last_sh = float(high[ph_idx[-1]]) if ph_idx else None
    last_sl = float(low[pl_idx[-1]])  if pl_idx else None

    # Build BOS event list: (bar_index_of_break, direction)
    bos_events: list[tuple[int, str]] = []

    for idx in ph_idx:
        price = high[idx]
        for j in range(idx + 1, len(close)):
            if close[j] > price:
                bos_events.append((j, "bullish"))
                break

    for idx in pl_idx:
        price = low[idx]
        for j in range(idx + 1, len(close)):
            if close[j] < price:
                bos_events.append((j, "bearish"))
                break

    if not bos_events:
        return "neutral", False, last_sh, last_sl

    bos_events.sort(key=lambda x: x[0])

    last_bos = bos_events[-1][1]
    choch    = len(bos_events) >= 2 and bos_events[-1][1] != bos_events[-2][1]

    return last_bos, choch, last_sh, last_sl


# ---------------------------------------------------------------------------
# Supply & demand zone detection
# ---------------------------------------------------------------------------

def _find_zones(
    df: pd.DataFrame,
    atr_mult: float = 1.0,
    lookback: int = 200,
    max_age_bars: int = 30,
) -> tuple[list[dict], list[dict]]:
    """
    Identify active (unmitigated) supply and demand zones from impulse candles.

    A demand zone is formed at the base of a large bullish impulse candle:
        zone = [candle_low, candle_body_bot]  (lower wick + body base area)
    A supply zone is formed at the base of a large bearish impulse candle:
        zone = [candle_body_top, candle_high] (upper wick + body top area)

    A zone is "mitigated" (discarded) once a subsequent bar's range re-enters
    the zone, implying institutions have already filled their orders there.

    Fixes vs original:
    - atr_mult lowered 1.5 → 1.0 (original was too restrictive, ~2 zones per 120 bars)
    - lookback extended 100 → 200 (capture more structural context)
    - max_age_bars: zones older than this many bars are discarded (prevent stale levels)
    - Supply zone bug fixed: z_bot now uses max(open, close) not just open, preventing
      zero-width zones when open ≈ high on a bearish candle
    - Minimum width check: skip zones narrower than 0.05× ATR (avoids degenerate zones)

    Args:
        atr_mult     : body must exceed atr_mult × ATR(14) to qualify as impulse
        lookback     : number of recent bars to scan
        max_age_bars : discard zones older than this many bars from end of data

    Returns:
        supply_zones, demand_zones : lists of {'top': float, 'bot': float, 'age': int}
    """
    df_s = df.iloc[-lookback:].reset_index(drop=True)
    N = len(df_s)
    if N < 15:
        return [], []

    high  = df_s["high"].values
    low   = df_s["low"].values
    close = df_s["close"].values
    open_ = df_s["open"].values

    # Wilder ATR (14)
    hl = high - low
    hc = np.abs(high - np.concatenate([[hl[0]], close[:-1]]))
    lc = np.abs(low  - np.concatenate([[hl[0]], close[:-1]]))
    tr = np.maximum(hl, np.maximum(hc, lc))

    atr = np.full(N, np.nan)
    if N >= 14:
        atr[13] = tr[:14].mean()
        for i in range(14, N):
            atr[i] = (atr[i - 1] * 13.0 + tr[i]) / 14.0

    supply_zones: list[dict] = []
    demand_zones: list[dict] = []

    for i in range(1, N - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        body = abs(close[i] - open_[i])
        if body < atr_mult * atr[i]:
            continue

        age = N - 1 - i  # bars since this candle (0 = most recent)
        if age > max_age_bars:
            continue

        sub_low  = low[i + 1:]
        sub_high = high[i + 1:]
        min_width = atr[i] * 0.05  # skip degenerate near-zero-width zones

        if close[i] > open_[i]:                  # bullish impulse → demand zone
            z_top = float(min(open_[i], close[i]))   # body bottom
            z_bot = float(low[i])
            if z_top - z_bot < min_width:
                continue
            if len(sub_low) == 0:
                continue
            mitigated = bool(((sub_low <= z_top) & (sub_high >= z_bot)).any())
            if not mitigated:
                demand_zones.append({"top": z_top, "bot": z_bot, "age": age})

        elif open_[i] > close[i]:                # bearish impulse → supply zone
            z_top = float(high[i])
            z_bot = float(max(open_[i], close[i]))   # body top (fix: was just open_[i])
            if z_top - z_bot < min_width:
                continue
            if len(sub_low) == 0:
                continue
            mitigated = bool(((sub_low <= z_top) & (sub_high >= z_bot)).any())
            if not mitigated:
                supply_zones.append({"top": z_top, "bot": z_bot, "age": age})

    return supply_zones, demand_zones


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_smc_signals(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    spot: float,
) -> SMCResult:
    """
    Compute all SMC signals for the current bar.

    Args:
        df_1h : 1-hour OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume)
        df_4h : 4-hour OHLCV DataFrame (same format)
        spot  : current spot price (used for zone proximity calculations)

    Returns:
        SMCResult with all structural and zone signals.
    """
    # 4h BOS / ChoCH  (n=3: requires 3 confirmed bars each side = 12h window)
    bos_4h, choch_4h, sh_4h, sl_4h = _detect_bos_choch(df_4h, n=3)

    # 1h BOS / ChoCH  (n=5: requires 5 confirmed bars each side = 5h window)
    bos_1h, choch_1h, sh_1h, sl_1h = _detect_bos_choch(df_1h, n=5)

    # Supply / demand zones from 4h data (stronger, more institutionally relevant)
    supply_zones, demand_zones = _find_zones(df_4h, atr_mult=1.0, lookback=200, max_age_bars=30)

    # Nearest supply zone above spot
    above = [z for z in supply_zones if z["bot"] > spot]
    nearest_supply_pct = None
    if above:
        nearest = min(above, key=lambda z: z["bot"])
        nearest_supply_pct = round((nearest["bot"] / spot - 1.0) * 100, 3)

    # Nearest demand zone below spot
    below = [z for z in demand_zones if z["top"] < spot]
    nearest_demand_pct = None
    if below:
        nearest = max(below, key=lambda z: z["top"])
        nearest_demand_pct = round((1.0 - nearest["top"] / spot) * 100, 3)

    # Is spot currently inside a zone?
    in_supply = any(z["bot"] <= spot <= z["top"] for z in supply_zones)
    in_demand = any(z["bot"] <= spot <= z["top"] for z in demand_zones)

    return SMCResult(
        bos_4h=bos_4h, choch_4h=choch_4h,
        swing_high_4h=round(sh_4h, 2) if sh_4h else None,
        swing_low_4h=round(sl_4h, 2)  if sl_4h else None,
        bos_1h=bos_1h, choch_1h=choch_1h,
        swing_high_1h=round(sh_1h, 2) if sh_1h else None,
        swing_low_1h=round(sl_1h, 2)  if sl_1h else None,
        nearest_supply_pct=nearest_supply_pct,
        nearest_demand_pct=nearest_demand_pct,
        in_supply_zone=in_supply,
        in_demand_zone=in_demand,
        n_supply_zones=len(supply_zones),
        n_demand_zones=len(demand_zones),
    )
