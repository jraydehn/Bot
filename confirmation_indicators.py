"""
Confirmation indicators module: EMA alignment, stochastic oscillator, volume,
VWAP, OBI, and funding rate on 1-hour / 1-minute candles.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from stochastic import compute_stochastic, StochasticResult
from ema_stack import compute_ema_stack, EMAStackResult


MIN_CANDLES = 60
EMA_FAST = 20            # fast EMA period for trend alignment
EMA_SLOW = 50            # slow EMA period for trend alignment
VOLUME_MA_PERIOD = 20    # simple moving average period for volume baseline
CMF_PERIOD = 30          # Chaikin Money Flow lookback (1m bars = 30-minute window)
EMA_CONFIRM_BARS = 3     # consecutive bars the EMA spread must hold to confirm
EMA_STRETCH_PERIOD = 20  # EMA period on 5m bars (covers ~100 minutes)
EMA_STRETCH_THRESHOLD = 0.001  # ±0.1% from 5m EMA triggers overbought/oversold signal

# VPIN parameters
VPIN_BUCKET_BARS = 5     # target number of 1m bars per volume bucket
VPIN_N_BUCKETS   = 10    # number of recent buckets to average
VPIN_THRESHOLD   = 0.80  # minimum unsigned imbalance to assign a score

# Session-anchored VWAP parameters (session resets at 00:00 UTC)
VWAP_NEUTRAL_BAND     = 0.002  # ±0.2% — price within this distance of VWAP = no signal
VWAP_REJECTION_WEIGHT = 2      # rejection score counts double in vwap_total
VWAP_STRETCH_WEIGHT   = 1      # position and stretch scores count single
VWAP_MIN_SESSION_BARS = 60     # minimum candles in session to produce a signal


@dataclass
class ConfirmationResult:
    """Output of the confirmation indicators module."""

    confirmation_bias: int    # +1 bullish, -1 bearish, 0 neutral — from 4-indicator YES score
    no_bias: int              # -1 bearish (good for NO), +1 bullish (bad for NO), 0 mixed
    confirmation_score: int   # count of bullish signals for YES (0–4): OBI, funding, stoch, vwap
    no_score: int             # count of bearish signals for NO (0–4): OBI, funding, stoch, vwap
    obi_score: int            # +1 bullish, -1 bearish, 0 neutral — from order book imbalance
    vpin_score: int           # +1 bullish, -1 bearish, 0 neutral — from VPIN informed flow
    vpin_raw: float           # unsigned VPIN magnitude 0–1 (NaN if insufficient data)
    ema_alignment: str        # "bullish", "bearish", or "neutral" — logged for analysis
    volume_confirmed: bool    # True if latest volume exceeds its 20-period average
    ema_20_current: float     # current value of the 20-period EMA
    ema_50_current: float     # current value of the 50-period EMA
    mom_15m_score: int        # +1 if 15-min price change > 0, -1 if < 0, 0 if unavailable
    mom_60m_score: int        # +1 if 60-min price change > 0, -1 if < 0, 0 if unavailable
    ema_stretch_score: int    # +1 oversold (below 5m EMA), -1 overbought (above), 0 neutral
    stoch_bias: int           # +1 oversold/bullish crossover, -1 overbought/bearish crossover, 0 neutral
    stoch_k: float            # current %K value (0–100)
    stoch_d: float            # current %D value (smoothed %K)
    stoch_crossover_active: bool  # True if crossover fired on current or previous 15m candle
    ema_stack_bias: int        # +1 bullish EMA9>21>50 stack, -1 bearish, 0 neutral/insufficient
    vol_score: int            # +1 high vol + up candle, -1 high vol + down candle, 0 low vol
    cmf_raw: float            # Chaikin Money Flow (14-period): positive = buying pressure, negative = selling
    cmf_score: int            # +1 if cmf > 0.05, -1 if cmf < -0.05, 0 neutral
    vwap_score: int           # composite VWAP signal inverted for contrarian edge (= -vwap_signal)
    vwap_signal: int          # raw VWAP composite signal: +1 bullish, -1 bearish, 0 neutral
    vwap_total: int           # sum: position_score + stretch_score + rejection_score*2
    stretch_score: int        # -2 above 2σ, -1 above 1σ, +1 below 1σ, +2 below 2σ, 0 within bands
    bearish_rejection: bool   # prev candle high > VWAP but close < VWAP and spot < VWAP
    bullish_rejection: bool   # prev candle low < VWAP but close > VWAP and spot > VWAP
    vwap_current: float       # session-anchored VWAP value at current candle
    vwap_upper_1: float       # VWAP + 1σ upper band
    vwap_upper_2: float       # VWAP + 2σ upper band
    vwap_lower_1: float       # VWAP - 1σ lower band
    vwap_lower_2: float       # VWAP - 2σ lower band
    distance_pct: float       # (spot - vwap) / vwap — how far price is from session anchor
    funding_bias: int         # +1 bullish (overcrowded shorts), -1 bearish (overcrowded longs), 0 neutral
    avg_funding_rate: float   # averaged funding rate across exchanges (0.0 if unavailable)
    squeeze_1h: bool          # True if BB width < KC width (volatility compression before breakout)
    adx_1h: float             # 14-period Average Directional Index on 1h bars (trend strength, not direction)
    reason: str               # plain-English explanation of the classification



def compute_vpin(hist_1m: pd.DataFrame) -> tuple:
    """
    Compute VPIN (Volume-synchronized Probability of Informed Trading) from 1m OHLCV data
    using bulk volume classification (BVC).

    For each bar: V_buy = volume * (close - low) / (high - low), V_sell = volume - V_buy.
    Bars are accumulated into equal-volume buckets; VPIN is the mean unsigned imbalance
    over the last VPIN_N_BUCKETS buckets. Direction comes from net flow in the most recent bucket.

    Returns (vpin_raw, vpin_score):
        vpin_raw:  unsigned VPIN magnitude 0–1, or NaN if insufficient data
        vpin_score: +1 informed buying, -1 informed selling, 0 neutral/insufficient
    """
    try:
        h = hist_1m.copy()
        h.columns = h.columns.str.lower()
        if len(h) < VPIN_BUCKET_BARS * VPIN_N_BUCKETS:
            return float("nan"), 0

        high   = h["high"].values.astype(float)
        low    = h["low"].values.astype(float)
        close  = h["close"].values.astype(float)
        volume = h["volume"].values.astype(float)

        # Bulk volume classification
        hl_range     = high - low
        safe_hl      = np.where(hl_range > 0, hl_range, 1.0)  # avoid divide-by-zero warning; np.where evals both branches
        buy_vol      = np.where(hl_range > 0, volume * (close - low) / safe_hl, volume / 2.0)
        sell_vol = volume - buy_vol

        # Dynamic bucket size: average bar volume × VPIN_BUCKET_BARS
        bucket_size = float(np.mean(volume)) * VPIN_BUCKET_BARS
        if bucket_size <= 0:
            return float("nan"), 0

        # Accumulate bars into equal-volume buckets
        buckets = []
        cur_buy = cur_sell = cur_total = 0.0
        for bv, sv in zip(buy_vol, sell_vol):
            remaining_b, remaining_s, remaining_v = bv, sv, bv + sv
            while remaining_v > 1e-9:
                space = bucket_size - cur_total
                if remaining_v <= space:
                    cur_buy   += remaining_b
                    cur_sell  += remaining_s
                    cur_total += remaining_v
                    remaining_b = remaining_s = remaining_v = 0.0
                else:
                    frac = space / remaining_v
                    cur_buy   += remaining_b * frac
                    cur_sell  += remaining_s * frac
                    cur_total  = bucket_size
                    remaining_b *= (1 - frac)
                    remaining_s *= (1 - frac)
                    remaining_v -= space
                    buckets.append((cur_buy, cur_sell))
                    cur_buy = cur_sell = cur_total = 0.0

        if len(buckets) < 2:
            return float("nan"), 0

        recent = buckets[-VPIN_N_BUCKETS:]
        imbalances = [abs(b - s) / (b + s) for b, s in recent if (b + s) > 0]
        if not imbalances:
            return float("nan"), 0

        vpin_raw = float(np.mean(imbalances))

        last_b, last_s = recent[-1]
        last_total = last_b + last_s
        net_flow = (last_b - last_s) / last_total if last_total > 0 else 0.0

        if vpin_raw >= VPIN_THRESHOLD and net_flow > 0.05:
            vpin_score = +1
        elif vpin_raw >= VPIN_THRESHOLD and net_flow < -0.05:
            vpin_score = -1
        else:
            vpin_score = 0

        return vpin_raw, vpin_score

    except Exception:
        return float("nan"), 0


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """14-period ADX using Wilder's smoothing. Returns NaN if insufficient data (< 2*period+2 bars)."""
    h  = high.values.astype(float)
    lo = low.values.astype(float)
    c  = close.values.astype(float)
    m  = len(c) - 1  # number of bar-to-bar pairs
    if m < period * 2:
        return float("nan")

    tr  = np.maximum(h[1:] - lo[1:],
          np.maximum(np.abs(h[1:] - c[:-1]),
                     np.abs(lo[1:] - c[:-1])))
    up  = h[1:] - h[:-1]
    dn  = lo[:-1] - lo[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)

    def _ws(arr):
        s = np.zeros(m)
        s[period - 1] = arr[:period].sum()
        for i in range(period, m):
            s[i] = s[i - 1] * (period - 1) / period + arr[i]
        return s

    atr_s = _ws(tr)
    pdm_s = _ws(pdm)
    ndm_s = _ws(ndm)

    safe_atr = np.where(atr_s > 0, atr_s, 1.0)
    di_p = 100.0 * pdm_s / safe_atr
    di_n = 100.0 * ndm_s / safe_atr
    di_t     = di_p + di_n
    safe_dit = np.where(di_t > 0, di_t, 1.0)
    dx       = np.where(di_t > 0, 100.0 * np.abs(di_p - di_n) / safe_dit, 0.0)

    start = period - 1
    if m < start + period:
        return float("nan")
    adx = np.zeros(m)
    adx[start + period - 1] = dx[start:start + period].mean()
    for i in range(start + period, m):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    result = adx[-1]
    return float(result) if result == result else float("nan")


def compute_confirmation(df: pd.DataFrame, hist_1m: pd.DataFrame = None, obi_score: int = 0, momentum_enabled: bool = True, funding_bias: int = 0, avg_funding_rate: float = 0.0, sharp_move_pct: float = 0.0, asset: str = "BTC") -> ConfirmationResult:
    """
    Compute EMA alignment, RSI regime, and volume confirmation from 1-hour bars.

    EMA Alignment: 20-EMA must have been strictly above (or below) 50-EMA
    for at least the last 3 consecutive candles.

    RSI Regime: RSI(21) >= 55 is bullish, <= 45 is bearish, 45–55 is neutral.

    Volume Confirmation: latest candle volume must strictly exceed its
    20-period simple moving average.

    Confirmation bias is +1 when EMA is bullish, RSI is not bearish, and volume
    confirms. It is -1 when EMA is bearish and RSI is not bullish — volume is not
    required for bearish confirmation, as low volume in a bearish structure reflects
    absence of buying pressure and supports NO bets. Neutral RSI (45–55) does not
    block either direction.

    Args:
        df: 1-hour OHLCV DataFrame. Must have at least 60 candles.
            Required columns: open, high, low, close, volume (case-insensitive).
        hist_1m: Optional 1-minute OHLCV DataFrame with at least 62 candles.
            If provided, adds 15-minute and 60-minute price momentum scores.

    Returns:
        ConfirmationResult dataclass with all indicator values and combined bias.

    Raises:
        ValueError: If fewer than 60 candles are provided.
    """
    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"DataFrame must have at least {MIN_CANDLES} candles, got {len(df)}."
        )

    df = df.copy()
    df.columns = df.columns.str.lower()
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    # --- EMA Alignment ---
    # Exponential moving averages on closing prices
    ema_20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_50 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    ema_20_current = float(ema_20.iloc[-1])
    ema_50_current = float(ema_50.iloc[-1])

    # Check the last EMA_CONFIRM_BARS candles for a consistent spread.
    # Three conditions determine alignment:
    #   Bullish: EMA-20 > EMA-50 (golden cross) AND price > EMA-20 (above both)
    #   Bearish: EMA-20 < EMA-50 (death cross) OR price < EMA-50 (below both EMAs)
    #   Neutral: golden cross but price trapped between the two EMAs
    # Price below both EMAs is bearish regardless of crossover status — both EMAs
    # act as overhead resistance when price is underneath them.
    last_ema20  = ema_20.iloc[-EMA_CONFIRM_BARS:].values
    last_ema50  = ema_50.iloc[-EMA_CONFIRM_BARS:].values
    last_close  = close.iloc[-EMA_CONFIRM_BARS:].values

    golden_cross  = all(last_ema20 > last_ema50)
    death_cross   = all(last_ema20 < last_ema50)
    price_above   = all(last_close > last_ema20)   # price above faster EMA (above both)
    price_below   = all(last_close < last_ema50)   # price below slower EMA (below both)

    if golden_cross and price_above:
        ema_alignment = "bullish"
    elif death_cross or price_below:
        ema_alignment = "bearish"
    else:
        ema_alignment = "neutral"

    # --- Stochastic Oscillator (15m bars resampled from hist_1m) ---
    # Replaces RSI regime and RSI mean reversion. Measures where price closed
    # relative to its recent high-low range on 15-minute bars — more responsive
    # than hourly RSI for 1-hour contract decisions.
    stoch = compute_stochastic(hist_1m)

    # --- EMA Stack (9/21/50 on 15m bars resampled from hist_1m) ---
    # Bullish: EMA9 > EMA21 > EMA50 AND price > EMA9. All timeframes aligned up.
    # Bearish: EMA9 < EMA21 < EMA50 AND price < EMA9. All timeframes aligned down.
    ema_stack = compute_ema_stack(hist_1m)

    # --- Volume Confirmation ---
    # Directional volume: high volume on an up candle = bullish, high volume on a down
    # candle = bearish, low volume either way = neutral (0). This correctly handles
    # selling pressure (high volume + price down) as bearish rather than penalizing
    # any high-volume candle regardless of direction.
    vol_sma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
    # Use the last COMPLETED bar (-2) not the current partial bar (-1) for volume.
    # The live 1h feed always includes the in-progress candle whose accumulated
    # volume is a fraction of a full bar, making high_vol=False nearly always.
    high_vol = bool(float(volume.iloc[-2]) > float(vol_sma.iloc[-2]))
    price_up = bool(float(close.iloc[-2]) > float(close.iloc[-3]))
    volume_confirmed = high_vol  # kept for display: True if volume above average

    macd_score = 0  # MACD removed — too lagging for 1-hour contracts (12-26h lookback)

    # --- Session-Anchored VWAP ---
    # VWAP resets at 00:00 UTC each day and accumulates through the session.
    # It is the volume-weighted average price for the current trading day —
    # a key institutional reference level. Price above session VWAP means
    # buyers have been in control all day; below means sellers.
    vwap_score        = 0
    vwap_signal       = 0
    vwap_total        = 0
    stretch_score     = 0
    bearish_rejection = False
    bullish_rejection = False
    vwap_current      = float("nan")
    vwap_upper_1      = float("nan")
    vwap_upper_2      = float("nan")
    vwap_lower_1      = float("nan")
    vwap_lower_2      = float("nan")
    distance_pct      = float("nan")

    # Use 1-minute bars for session VWAP: 60 candles = 1 hour of data, matching
    # the "60 session candles" minimum threshold. Falls back to no signal if
    # hist_1m is unavailable.
    _vwap_src = None
    if hist_1m is not None and len(hist_1m) > 0:
        _h1m = hist_1m.copy()
        _h1m.columns = _h1m.columns.str.lower()
        if all(c in _h1m.columns for c in ["high", "low", "close", "volume"]):
            _vwap_src = _h1m

    if _vwap_src is not None:
        try:
            # Slice to current session: 1m bars from 00:00 UTC today onward.
            if isinstance(_vwap_src.index, pd.DatetimeIndex):
                last_ts       = _vwap_src.index[-1]
                session_start = last_ts.normalize().tz_localize(None) if last_ts.tzinfo is None else last_ts.normalize()
                session_df    = _vwap_src[_vwap_src.index >= session_start]
            else:
                session_df = _vwap_src

            if len(session_df) < VWAP_MIN_SESSION_BARS:
                # Too early in the session — fewer than 60 candles, signal unreliable
                vwap_score = 0
            else:
                tp  = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
                vol = session_df["volume"]

                # Cumulative session VWAP: sum(typical_price × volume) / sum(volume)
                # anchored to session open — unlike rolling VWAP this never forgets
                # early-session volume that set the institutional reference.
                vwap_series  = (tp * vol).cumsum() / vol.cumsum()
                vwap_current = float(vwap_series.iloc[-1])

                # Standard deviation of (typical_price − VWAP) across session candles.
                # Measures how widely price has oscillated around the session anchor —
                # used to define statistically significant deviation bands.
                vwap_std  = float((tp - vwap_series).std())
                spot      = float(session_df["close"].iloc[-1])
                distance_pct = (spot - vwap_current) / vwap_current

                # σ-bands: 1σ and 2σ above/below VWAP.
                # Price beyond these levels is statistically stretched and
                # historically reverts toward the session anchor.
                vwap_upper_1 = vwap_current + vwap_std
                vwap_upper_2 = vwap_current + 2 * vwap_std
                vwap_lower_1 = vwap_current - vwap_std
                vwap_lower_2 = vwap_current - 2 * vwap_std

                # --- Position score: price location relative to VWAP and neutral band ---
                # Price well above VWAP: institutions buying all session, bullish.
                # Price well below VWAP: selling pressure dominant, bearish.
                # Within ±0.2% neutral band: noise — no directional signal.
                if distance_pct > VWAP_NEUTRAL_BAND or spot > vwap_upper_1:
                    position_score = +1   # above VWAP or upper band — bullish
                elif distance_pct < -VWAP_NEUTRAL_BAND or spot < vwap_lower_1:
                    position_score = -1   # below VWAP or lower band — bearish
                else:
                    position_score = 0    # within neutral band — no signal

                # --- Band stretch score: mean reversion from σ-bands ---
                # Price beyond 1σ/2σ is overextended and tends to revert.
                # The farther price is from VWAP, the stronger the reversion pressure.
                if spot > vwap_upper_2:
                    stretch_score = -2   # >2σ above VWAP — strong bearish mean reversion
                elif spot > vwap_upper_1:
                    stretch_score = -1   # >1σ above VWAP — mild bearish reversion pressure
                elif spot < vwap_lower_2:
                    stretch_score = +2   # >2σ below VWAP — strong bullish mean reversion
                elif spot < vwap_lower_1:
                    stretch_score = +1   # >1σ below VWAP — mild bullish reversion pressure
                else:
                    stretch_score = 0    # within 1σ bands — no stretch signal

                # --- VWAP rejection: previous candle tested VWAP but failed to hold ---
                # A rejection is a high-conviction mean reversion signal: price tagged
                # VWAP (confirming it as a reference level) but was pushed back,
                # indicating the level acted as resistance/support.
                rejection_score = 0
                if len(session_df) >= 2:
                    prev       = session_df.iloc[-2]
                    prev_vwap  = float(vwap_series.iloc[-2])

                    # Bearish rejection: prev candle high exceeded VWAP (tested resistance)
                    # but close fell back below — sellers defended VWAP as resistance.
                    # Current spot also below VWAP confirms the rejection is holding.
                    bearish_rejection = (
                        float(prev["high"])  > prev_vwap and
                        float(prev["close"]) < prev_vwap and
                        spot < vwap_current
                    )

                    # Bullish rejection: prev candle low dipped below VWAP (tested support)
                    # but close recovered above — buyers defended VWAP as support.
                    # Current spot above VWAP confirms the bounce is holding.
                    bullish_rejection = (
                        float(prev["low"])   < prev_vwap and
                        float(prev["close"]) > prev_vwap and
                        spot > vwap_current
                    )

                    if bearish_rejection:
                        rejection_score = -1   # VWAP acted as resistance — bearish
                    elif bullish_rejection:
                        rejection_score = +1   # VWAP acted as support — bullish

                # --- Combine into vwap_total ---
                # stretch_score alone reaches ±2 at >2σ overextension.
                # rejection counts double (high-conviction reversal signal).
                # position_score removed: it always opposes stretch_score (same condition,
                # opposite sign convention), causing systematic cancellation that prevents
                # the signal from ever firing when price is in the 1σ–2σ band.
                vwap_total = (
                    stretch_score * VWAP_STRETCH_WEIGHT
                    + rejection_score * VWAP_REJECTION_WEIGHT
                )

                # Require composite score ≥ ±2 to avoid single-indicator noise.
                if vwap_total >= 2:
                    vwap_signal = +1
                elif vwap_total <= -2:
                    vwap_signal = -1
                else:
                    vwap_signal = 0

                # vwap_score is the INVERTED composite signal — historical data shows
                # the old rolling VWAP was a reliable contrarian indicator: aligned
                # trades had 19% win rate vs 37% for misaligned. Inverting preserves
                # that edge while the session-anchored model accumulates new data.
                vwap_score = -vwap_signal

        except Exception:
            vwap_score = 0
            vwap_signal = 0

    # --- VPIN ---
    vpin_raw, vpin_score = compute_vpin(hist_1m) if hist_1m is not None and len(hist_1m) >= VPIN_BUCKET_BARS * VPIN_N_BUCKETS else (float("nan"), 0)

    # --- Supporting indicators (logged for analysis, not in primary scoring) ---
    ema_score = 1 if ema_alignment == "bullish" else (-1 if ema_alignment == "bearish" else 0)
    if high_vol and price_up:
        vol_score = 1
    elif high_vol and not price_up:
        vol_score = -1
    else:
        vol_score = 0

    # --- Chaikin Money Flow (CMF) on 1m bars ---
    # Uses 1m OHLCV (hist_1m) with a 30-bar lookback = 30-minute intraday window.
    # This gives CMF a unique input: short-term accumulation/distribution pressure
    # independent of the multi-hour EMA trend (ema_stack_bias) and VPIN's equal-volume
    # bucket approach. Money Flow Multiplier = [(close-low)-(high-close)] / (high-low).
    cmf_raw = float("nan")
    cmf_score = 0
    if hist_1m is not None and len(hist_1m) >= CMF_PERIOD + 1:
        try:
            _1m = hist_1m.copy()
            _1m.columns = [c.lower() for c in _1m.columns]
            _c1 = _1m["close"].astype(float)
            _h1 = _1m["high"].astype(float)
            _l1 = _1m["low"].astype(float)
            _v1 = _1m["volume"].astype(float)
            _hl1 = (_h1 - _l1).replace(0, float("nan"))
            _mfm1 = ((_c1 - _l1) - (_h1 - _c1)) / _hl1
            _mfv1 = _mfm1 * _v1
            _vol_sum = _v1.rolling(CMF_PERIOD).sum()
            _cmf_s = _mfv1.rolling(CMF_PERIOD).sum() / _vol_sum.replace(0, float("nan"))
            _cmf_val = _cmf_s.iloc[-2]  # last completed bar
            cmf_raw = float(_cmf_val) if _cmf_val == _cmf_val else float("nan")
            if cmf_raw != cmf_raw:
                cmf_score = 0
            elif cmf_raw > 0.05:
                cmf_score = 1
            elif cmf_raw < -0.05:
                cmf_score = -1
            else:
                cmf_score = 0
        except Exception:
            cmf_raw = float("nan")
            cmf_score = 0

    # --- Volatility Squeeze (BB inside KC) ---
    # Fires when Bollinger Band width (4 * rolling std, 20 bars) < Keltner Channel width (4 * ATR, 20 bars).
    # Compression phase: price coiling before a breakout. No directional signal — only flags that a
    # sustained move (YES strike reach) is more likely when the squeeze resolves.
    try:
        _close_prev = close.shift(1)
        _tr_s = pd.concat([(high - low), (high - _close_prev).abs(), (low - _close_prev).abs()], axis=1).max(axis=1)
        _atr20    = _tr_s.rolling(20).mean()
        _bb_std20 = close.rolling(20).std()
        squeeze_1h = bool(float(_bb_std20.iloc[-1]) < float(_atr20.iloc[-1]))
    except Exception:
        squeeze_1h = False

    # --- ADX (Average Directional Index) ---
    # Measures trend STRENGTH regardless of direction. ADX > 25 = trending; < 20 = ranging/choppy.
    # Trending markets sustain directional moves and are more likely to reach strikes;
    # ranging markets reject them. No existing trend signal captures this — EMA stack captures
    # direction only, not magnitude of trend conviction.
    adx_1h = _compute_adx(high, low, close, period=14)

    # --- Short-term momentum (logged only, not in primary scoring) ---
    mom_15m_score = 0
    mom_60m_score = 0
    if momentum_enabled and hist_1m is not None and len(hist_1m) >= 62:
        hist_1m_c = hist_1m.copy()
        hist_1m_c.columns = hist_1m_c.columns.str.lower()
        c1m = hist_1m_c["close"]
        last_close_1m = float(c1m.iloc[-1])
        close_15_ago  = float(c1m.iloc[-16])
        close_60_ago  = float(c1m.iloc[-61])
        mom_15m = (last_close_1m - close_15_ago) / close_15_ago
        mom_60m = (last_close_1m - close_60_ago) / close_60_ago
        mom_15m_score = 2 if mom_15m > 0 else -2
        mom_60m_score = 2 if mom_60m > 0 else -2

    # --- EMA Stretch score (5m candles resampled from 1m, 20-period EMA) ---
    # Measures mean reversion: if price is extended above the 5m EMA it is overbought
    # and likely to revert down (-1); if below, oversold and likely to revert up (+1).
    # Groups consecutive 1m bars into 5-bar buckets (no datetime index required).
    ema_stretch_score = 0
    if hist_1m is not None and len(hist_1m) >= 100:
        try:
            h1m = hist_1m.copy()
            h1m.columns = h1m.columns.str.lower()
            closes_1m = h1m["close"].values
            n_5m = len(closes_1m) // 5
            closes_5m = pd.Series([closes_1m[(i + 1) * 5 - 1] for i in range(n_5m)])
            if len(closes_5m) >= EMA_STRETCH_PERIOD:
                ema_5m = closes_5m.ewm(span=EMA_STRETCH_PERIOD, adjust=False).mean()
                current_5m_close = float(closes_5m.iloc[-1])
                current_5m_ema   = float(ema_5m.iloc[-1])
                stretch = (current_5m_close - current_5m_ema) / current_5m_ema
                if stretch > EMA_STRETCH_THRESHOLD:
                    ema_stretch_score = -1   # overbought — expect reversion down
                elif stretch < -EMA_STRETCH_THRESHOLD:
                    ema_stretch_score = +1   # oversold — expect reversion up
        except Exception:
            ema_stretch_score = 0

    # --- Stoch bias effective value: flip during sharp directional moves ---
    # In trending markets stochastic extremes signal continuation, not reversion.
    # Sharp drop + oversold (stoch_bias=+1): flip to -1 (bearish continuation)
    # Sharp rally + overbought (stoch_bias=-1): flip to +1 (bullish continuation)
    _sharp_thresholds = {"BTC": 0.008, "ETH": 0.015, "SOL": 0.020}
    _sharp_thresh = _sharp_thresholds.get(asset.upper(), 0.008)
    _eff_stoch_bias = stoch.stoch_bias
    if abs(sharp_move_pct) > _sharp_thresh:
        if sharp_move_pct < 0 and stoch.stoch_bias > 0:   # oversold + sharp drop → bearish
            _eff_stoch_bias = -stoch.stoch_bias
        elif sharp_move_pct > 0 and stoch.stoch_bias < 0:  # overbought + sharp rally → bullish
            _eff_stoch_bias = -stoch.stoch_bias

    # --- 4-indicator NO score (0–4, where higher = more bearish = better for NO) ---
    # Threshold: 2 of 4 bearish signals required for NO confirmation.
    no_score = 0
    if obi_score         == -1: no_score += 1   # bearish order book pressure
    if funding_bias      == -1: no_score += 1   # overcrowded longs → mean reversion down
    if _eff_stoch_bias   == -1: no_score += 1   # overbought stoch or bearish continuation
    if vwap_signal       == -1: no_score += 1   # price at premium above VWAP

    if no_score >= 2:
        no_bias = -1   # majority bearish → supports NO bet
    elif no_score == 0:
        no_bias = +1   # no bearish signals → bullish conditions, bad for NO
    else:
        no_bias = 0    # mixed signals

    # --- 4-indicator YES score (0–4, where higher = more bullish = better for YES) ---
    # Threshold: 2 of 4 bullish signals required for YES confirmation.
    confirmation_score = 0
    if obi_score         == +1: confirmation_score += 1   # bullish order book pressure
    if funding_bias      == +1: confirmation_score += 1   # overcrowded shorts → squeeze up
    if _eff_stoch_bias   == +1: confirmation_score += 1   # oversold stoch or bullish continuation
    if vwap_signal       == +1: confirmation_score += 1   # price at discount below VWAP

    if confirmation_score >= 2:
        confirmation_bias = +1
        label = "Bullish"
    elif confirmation_score == 0:
        confirmation_bias = -1
        label = "Bearish"
    else:
        confirmation_bias = 0
        label = "Neutral"

    max_score = 4
    vwap_detail = (
        f"VWAP={vwap_current:.2f}(dist={distance_pct*100:+.3f}%,total={vwap_total:+d}"
        f",stretch={stretch_score:+d},rej={'B' if bearish_rejection else ('U' if bullish_rejection else '-')})"
        if not (vwap_current != vwap_current) else "VWAP=unavail"
    )
    reason = (
        "{}: YES={}/4 (OBI={:+d}, Funding={:+d}, Stoch={:+d}, VWAP={:+d}). "
        "NO={}/4 (no_bias={:+d}). "
        "Stoch: K={:.1f} D={:.1f} xover={}. "
        "EMA={} ({:.0f}/{:.0f}). {}. "
        "Aux: EMA_stack={:+d}, EMA_str={:+d}, VPIN={:+d}, Vol={:+d}, CMF={:.3f}({:+d}), Funding={:+.4f}%/8h.".format(
            label, confirmation_score,
            obi_score, funding_bias, stoch.stoch_bias, vwap_signal,
            no_score, no_bias,
            stoch.stoch_k, stoch.stoch_d, stoch.stoch_crossover_active,
            ema_alignment, ema_20_current, ema_50_current,
            vwap_detail,
            ema_stack.ema_stack_bias, ema_stretch_score, vpin_score, vol_score,
            cmf_raw if cmf_raw == cmf_raw else 0.0, cmf_score, avg_funding_rate * 100,
        )
    )

    return ConfirmationResult(
        confirmation_bias=confirmation_bias,
        no_bias=no_bias,
        confirmation_score=confirmation_score,
        no_score=no_score,
        obi_score=obi_score,
        vpin_score=vpin_score,
        vpin_raw=vpin_raw,
        ema_alignment=ema_alignment,
        volume_confirmed=volume_confirmed,
        ema_20_current=ema_20_current,
        ema_50_current=ema_50_current,
        mom_15m_score=mom_15m_score,
        mom_60m_score=mom_60m_score,
        ema_stretch_score=ema_stretch_score,
        stoch_bias=stoch.stoch_bias,
        stoch_k=stoch.stoch_k,
        stoch_d=stoch.stoch_d,
        stoch_crossover_active=stoch.stoch_crossover_active,
        ema_stack_bias=ema_stack.ema_stack_bias,
        vol_score=vol_score,
        cmf_raw=cmf_raw,
        cmf_score=cmf_score,
        vwap_score=vwap_score,
        vwap_signal=vwap_signal,
        vwap_total=vwap_total,
        stretch_score=stretch_score,
        bearish_rejection=bearish_rejection,
        bullish_rejection=bullish_rejection,
        vwap_current=vwap_current,
        vwap_upper_1=vwap_upper_1,
        vwap_upper_2=vwap_upper_2,
        vwap_lower_1=vwap_lower_1,
        vwap_lower_2=vwap_lower_2,
        distance_pct=distance_pct,
        funding_bias=funding_bias,
        avg_funding_rate=avg_funding_rate,
        squeeze_1h=squeeze_1h,
        adx_1h=adx_1h,
        reason=reason,
    )
