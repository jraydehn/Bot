"""
EMA stack indicator using 15-minute bars resampled from 1-minute history.

A "stack" means all three EMAs are aligned in the same direction AND price
is on the correct side of the fast EMA. This distinguishes genuine trending
conditions from choppy crossover noise:

  Bullish stack: EMA9 > EMA21 > EMA50 AND price > EMA9
    All three moving averages are in rising order and price is above the
    fastest one — trend is aligned across short, medium, and intermediate
    timeframes. Momentum is sustained, not just a momentary spike.

  Bearish stack: EMA9 < EMA21 < EMA50 AND price < EMA9
    All three are in falling order and price is below the fastest — sellers
    are in control across all timeframes. Sustained bearish momentum.

  Neutral: partial alignment or price between EMAs.
    Trend is either transitioning or contested — no high-conviction signal.

15-minute bars are used because they balance responsiveness (8x faster than
1-hour EMA) with noise reduction (less reactive than 1m or 5m bars).
"""

from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

EMA_STACK_FAST   = 9    # fast EMA — most sensitive to recent price action
EMA_STACK_MID    = 21   # medium EMA — intermediate trend
EMA_STACK_SLOW   = 50   # slow EMA — broader trend anchor
EMA_STACK_MIN_CANDLES = 50  # need at least 50 15m bars to compute EMA50 reliably


@dataclass
class EMAStackResult:
    """Output of the EMA stack indicator."""

    ema_stack_bias: int     # +1 bullish stack, -1 bearish stack, 0 neutral
    ema_9: float            # current 9-period EMA on 15m bars
    ema_21: float           # current 21-period EMA on 15m bars
    ema_50: float           # current 50-period EMA on 15m bars
    price: float            # current close (last 15m bar)
    bullish_stack: bool     # EMA9 > EMA21 > EMA50 and price > EMA9
    bearish_stack: bool     # EMA9 < EMA21 < EMA50 and price < EMA9
    reason: str             # plain-English explanation


# Neutral fallback — returned when hist_1m is unavailable or insufficient.
_FALLBACK = EMAStackResult(
    ema_stack_bias=0,
    ema_9=float("nan"),
    ema_21=float("nan"),
    ema_50=float("nan"),
    price=float("nan"),
    bullish_stack=False,
    bearish_stack=False,
    reason="EMA stack unavailable — insufficient 1m data. ema_stack_bias=0.",
)


def compute_ema_stack(hist_1m: pd.DataFrame) -> EMAStackResult:
    """
    Compute 9/21/50 EMA stack on 15-minute bars resampled from hist_1m.

    Resamples to 15-minute candles anchored at 00:00 UTC, computes three EMAs
    on closing prices, and checks whether they are fully aligned in one direction
    with price on the correct side of the fast EMA.

    Requires at least EMA_STACK_MIN_CANDLES (50) 15m candles — returns neutral
    fallback if insufficient data.

    Args:
        hist_1m: 1-minute OHLCV DataFrame. Columns: open, high, low, close, volume
                 (case-insensitive). DatetimeIndex preferred for UTC alignment.

    Returns:
        EMAStackResult with ema_stack_bias (+1/0/-1), raw EMA values, and reason.
    """
    if hist_1m is None or len(hist_1m) == 0:
        return _FALLBACK

    try:
        h = hist_1m.copy()
        h.columns = h.columns.str.lower()

        if "close" not in h.columns:
            return _FALLBACK

        # --- Resample 1m → 15m bars anchored at 00:00 UTC ---
        # Using close=last for the EMA calculation. Same resampling logic as
        # stochastic.py to ensure consistency across indicators.
        if isinstance(h.index, pd.DatetimeIndex):
            closes_15m = h["close"].resample("15min", origin="start_day").last().dropna()
        else:
            # No datetime index — group consecutive 15-bar windows.
            n = len(h) // 15
            closes_15m = pd.Series([
                float(h["close"].iloc[(i + 1) * 15 - 1]) for i in range(n)
            ])

        if len(closes_15m) < EMA_STACK_MIN_CANDLES:
            return EMAStackResult(
                ema_stack_bias=0,
                ema_9=float("nan"),
                ema_21=float("nan"),
                ema_50=float("nan"),
                price=float("nan"),
                bullish_stack=False,
                bearish_stack=False,
                reason=(
                    f"EMA stack unavailable — only {len(closes_15m)} 15m bars "
                    f"(need {EMA_STACK_MIN_CANDLES}). ema_stack_bias=0."
                ),
            )

        # --- Compute the three EMAs ---
        # Exponential moving averages on 15m closes. The fast EMA (9) reacts
        # quickly to recent price, the mid (21) smooths out noise, and the slow
        # (50) represents the broader intermediate trend. When all three line up
        # in the same order, the trend is confirmed across timeframes.
        ema_9_series  = closes_15m.ewm(span=EMA_STACK_FAST, adjust=False).mean()
        ema_21_series = closes_15m.ewm(span=EMA_STACK_MID,  adjust=False).mean()
        ema_50_series = closes_15m.ewm(span=EMA_STACK_SLOW, adjust=False).mean()

        ema_9  = float(ema_9_series.iloc[-1])
        ema_21 = float(ema_21_series.iloc[-1])
        ema_50 = float(ema_50_series.iloc[-1])
        price  = float(closes_15m.iloc[-1])

        # --- Detect stack alignment ---
        # Bullish stack: all EMAs in ascending order (fast > mid > slow) AND
        # price is above the fast EMA — trend is aligned and price is riding it.
        bullish_stack = (ema_9 > ema_21 > ema_50) and (price > ema_9)

        # Bearish stack: all EMAs in descending order (fast < mid < slow) AND
        # price is below the fast EMA — trend is aligned down and price is below it.
        bearish_stack = (ema_9 < ema_21 < ema_50) and (price < ema_9)

        if bullish_stack:
            ema_stack_bias = +1
            label = f"bullish stack: EMA9({ema_9:.0f}) > EMA21({ema_21:.0f}) > EMA50({ema_50:.0f}), price({price:.0f}) > EMA9"
        elif bearish_stack:
            ema_stack_bias = -1
            label = f"bearish stack: EMA9({ema_9:.0f}) < EMA21({ema_21:.0f}) < EMA50({ema_50:.0f}), price({price:.0f}) < EMA9"
        else:
            ema_stack_bias = 0
            label = f"neutral: EMA9={ema_9:.0f}, EMA21={ema_21:.0f}, EMA50={ema_50:.0f}, price={price:.0f} (not fully stacked)"

        reason = (
            f"EMA stack ({len(closes_15m)} 15m bars): {label}. "
            f"ema_stack_bias={ema_stack_bias:+d}."
        )

        return EMAStackResult(
            ema_stack_bias=ema_stack_bias,
            ema_9=round(ema_9, 2),
            ema_21=round(ema_21, 2),
            ema_50=round(ema_50, 2),
            price=round(price, 2),
            bullish_stack=bullish_stack,
            bearish_stack=bearish_stack,
            reason=reason,
        )

    except Exception as exc:
        return EMAStackResult(
            ema_stack_bias=0,
            ema_9=float("nan"),
            ema_21=float("nan"),
            ema_50=float("nan"),
            price=float("nan"),
            bullish_stack=False,
            bearish_stack=False,
            reason=f"EMA stack computation failed: {exc}. ema_stack_bias=0.",
        )
