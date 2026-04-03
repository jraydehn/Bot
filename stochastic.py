"""
Stochastic oscillator module using 15-minute bars resampled from 1-minute history.

The stochastic oscillator measures where the current close sits relative to the
high-low range over a lookback period. At extremes it signals mean reversion:
overbought (>80) means price is near the top of its recent range and likely to
pull back; oversold (<20) means price is near the bottom and likely to bounce.

Crossover signals (K crossing D) are higher conviction than position-only signals
because they show momentum is actively turning, not just extended.
"""

from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

STOCH_K_PERIOD   = 14   # lookback bars for highest high / lowest low
STOCH_D_PERIOD   = 3    # smoothing period for signal line (%D)
STOCH_OVERBOUGHT = 80   # above this level price is in overbought territory
STOCH_OVERSOLD   = 20   # below this level price is in oversold territory
STOCH_MIN_CANDLES = 20  # minimum 15m candles required before emitting a signal


@dataclass
class StochasticResult:
    """Output of the stochastic oscillator module."""

    stoch_bias: int               # +1 oversold/bullish, -1 overbought/bearish, 0 neutral
    stoch_k: float                # current %K value (0–100)
    stoch_d: float                # current %D value (smoothed %K, 0–100)
    bearish_crossover: bool       # %K crossed below %D this candle while overbought
    bullish_crossover: bool       # %K crossed above %D this candle while oversold
    stoch_crossover_active: bool  # crossover fired on current or previous 15m candle
    in_overbought: bool           # current %K > 80
    in_oversold: bool             # current %K < 20
    reason: str                   # plain-English explanation


# Neutral fallback — returned when hist_1m is unavailable or insufficient.
_FALLBACK = StochasticResult(
    stoch_bias=0,
    stoch_k=float("nan"),
    stoch_d=float("nan"),
    bearish_crossover=False,
    bullish_crossover=False,
    stoch_crossover_active=False,
    in_overbought=False,
    in_oversold=False,
    reason="Stochastic unavailable — insufficient 1m data. stoch_bias=0 (neutral).",
)


def compute_stochastic(hist_1m: pd.DataFrame) -> StochasticResult:
    """
    Compute stochastic oscillator from 1-minute OHLCV bars resampled to 15-minute bars.

    Resamples hist_1m to 15-minute candles anchored at 00:00 UTC, then computes
    %K and %D. Requires at least STOCH_MIN_CANDLES (20) 15-minute candles before
    returning a signal — returns neutral fallback if insufficient data.

    Args:
        hist_1m: 1-minute OHLCV DataFrame with DatetimeIndex (or positional index).
                 Must have columns: open, high, low, close, volume (case-insensitive).

    Returns:
        StochasticResult with stoch_bias, raw values, crossover flags, and reason.
    """
    if hist_1m is None or len(hist_1m) == 0:
        return _FALLBACK

    try:
        h = hist_1m.copy()
        h.columns = h.columns.str.lower()

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(h.columns)):
            return _FALLBACK

        # --- Resample 1m → 15m bars ---
        # Anchor at 00:00 UTC so session boundaries align at midnight.
        # Standard OHLCV aggregation: open=first, high=max, low=min,
        # close=last, volume=sum. Drop bars with no data (gaps in feed).
        if isinstance(h.index, pd.DatetimeIndex):
            df_15m = h.resample("15min", origin="start_day").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["close"])
        else:
            # No datetime index — group consecutive 15-bar windows as a fallback.
            n = len(h) // 15
            if n < STOCH_MIN_CANDLES:
                return _FALLBACK
            records = []
            for i in range(n):
                sl = h.iloc[i * 15 : (i + 1) * 15]
                records.append({
                    "open":   float(sl["open"].iloc[0]),
                    "high":   float(sl["high"].max()),
                    "low":    float(sl["low"].min()),
                    "close":  float(sl["close"].iloc[-1]),
                    "volume": float(sl["volume"].sum()),
                })
            df_15m = pd.DataFrame(records)

        if len(df_15m) < STOCH_MIN_CANDLES:
            return StochasticResult(
                stoch_bias=0,
                stoch_k=float("nan"),
                stoch_d=float("nan"),
                bearish_crossover=False,
                bullish_crossover=False,
                stoch_crossover_active=False,
                in_overbought=False,
                in_oversold=False,
                reason=(
                    f"Stochastic unavailable — only {len(df_15m)} 15m candles "
                    f"(need {STOCH_MIN_CANDLES}). stoch_bias=0."
                ),
            )

        # --- Compute %K ---
        # %K: where did the current close land relative to the highest high
        # and lowest low over the lookback period? 100 = at the top of the range,
        # 0 = at the bottom. Values above 80 mean price is near recent highs
        # (overbought); values below 20 mean price is near recent lows (oversold).
        lowest_low   = df_15m["low"].rolling(STOCH_K_PERIOD).min()
        highest_high = df_15m["high"].rolling(STOCH_K_PERIOD).max()
        hl_range     = highest_high - lowest_low

        # Avoid division by zero when range collapses (flat market)
        stoch_k = ((df_15m["close"] - lowest_low) / hl_range.replace(0, float("nan"))) * 100

        # --- Compute %D ---
        # %D: smoothed version of %K over STOCH_D_PERIOD candles.
        # Acts as a signal line — crossovers between %K and %D indicate
        # momentum shifts that often precede price reversals.
        stoch_d = stoch_k.rolling(STOCH_D_PERIOD).mean()

        # Need at least 3 valid candles to detect crossovers
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()
        if len(valid_k) < 2 or len(valid_d) < 2:
            return _FALLBACK

        k_curr = float(stoch_k.iloc[-1])
        d_curr = float(stoch_d.iloc[-1])
        k_prev = float(stoch_k.iloc[-2])
        d_prev = float(stoch_d.iloc[-2])

        if any(v != v for v in (k_curr, d_curr, k_prev, d_prev)):
            # NaN present — insufficient history for crossover detection
            return _FALLBACK

        # --- Detect crossover signals (current candle) ---
        # A crossover is high-conviction: %K has just crossed %D, showing
        # that momentum is actively reversing at an extreme level.

        # Bearish crossover: %K was above %D (momentum up) but just dropped below
        # (momentum turning down), while still in overbought territory. This is the
        # classic stochastic sell signal — price has been stretched high and is
        # beginning to roll over.
        bearish_crossover = (
            k_prev > d_prev              # was above signal line
            and k_curr < d_curr          # now below signal line — crossover
            and k_prev > STOCH_OVERBOUGHT  # crossover originated from overbought zone
        )

        # Bullish crossover: %K was below %D (momentum down) but just crossed above
        # (momentum turning up), while originating from oversold territory. Classic buy
        # signal — price has been stretched low and is beginning to recover.
        bullish_crossover = (
            k_prev < d_prev              # was below signal line
            and k_curr > d_curr          # now above signal line — crossover
            and k_prev < STOCH_OVERSOLD    # crossover originated from oversold zone
        )

        # --- Check if crossover fired on previous 15m candle ---
        # A crossover from the prior candle is still actionable — it takes time
        # for a momentum shift to fully manifest in price.
        prev_bearish_xover = False
        prev_bullish_xover = False
        if len(stoch_k) >= 3 and len(stoch_d) >= 3:
            k_prev2 = float(stoch_k.iloc[-3])
            d_prev2 = float(stoch_d.iloc[-3])
            if k_prev2 == k_prev2 and d_prev2 == d_prev2:
                prev_bearish_xover = (
                    k_prev2 > d_prev2
                    and k_prev  < d_prev
                    and k_prev2 > STOCH_OVERBOUGHT
                )
                prev_bullish_xover = (
                    k_prev2 < d_prev2
                    and k_prev  > d_prev
                    and k_prev2 < STOCH_OVERSOLD
                )

        stoch_crossover_active = (
            bearish_crossover or bullish_crossover
            or prev_bearish_xover or prev_bullish_xover
        )

        # --- Position signals (lower conviction than crossover) ---
        # Price already in extreme territory even without an active crossover.
        # Momentum may still extend further before reversing, so these carry
        # less conviction than a confirmed crossover.
        in_overbought = k_curr > STOCH_OVERBOUGHT
        in_oversold   = k_curr < STOCH_OVERSOLD

        # --- Classify stoch_bias ---
        # Crossover signals take priority; position-only signals are secondary.
        if bearish_crossover:
            stoch_bias = -1
            signal_label = f"bearish crossover (%K={k_curr:.1f} crossed below %D={d_curr:.1f} in overbought)"
        elif bullish_crossover:
            stoch_bias = +1
            signal_label = f"bullish crossover (%K={k_curr:.1f} crossed above %D={d_curr:.1f} in oversold)"
        elif in_overbought:
            stoch_bias = -1
            signal_label = f"overbought: %K={k_curr:.1f} > {STOCH_OVERBOUGHT} (no crossover yet)"
        elif in_oversold:
            stoch_bias = +1
            signal_label = f"oversold: %K={k_curr:.1f} < {STOCH_OVERSOLD} (no crossover yet)"
        else:
            stoch_bias = 0
            signal_label = f"neutral: %K={k_curr:.1f}, %D={d_curr:.1f} within [{STOCH_OVERSOLD}, {STOCH_OVERBOUGHT}]"

        reason = (
            f"Stochastic ({len(df_15m)} 15m bars): {signal_label}. "
            f"crossover_active={stoch_crossover_active}. "
            f"stoch_bias={stoch_bias:+d}."
        )

        return StochasticResult(
            stoch_bias=stoch_bias,
            stoch_k=round(k_curr, 2),
            stoch_d=round(d_curr, 2),
            bearish_crossover=bearish_crossover,
            bullish_crossover=bullish_crossover,
            stoch_crossover_active=stoch_crossover_active,
            in_overbought=in_overbought,
            in_oversold=in_oversold,
            reason=reason,
        )

    except Exception as exc:
        return StochasticResult(
            stoch_bias=0,
            stoch_k=float("nan"),
            stoch_d=float("nan"),
            bearish_crossover=False,
            bullish_crossover=False,
            stoch_crossover_active=False,
            in_overbought=False,
            in_oversold=False,
            reason=f"Stochastic computation failed: {exc}. stoch_bias=0.",
        )
