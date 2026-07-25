"""
drawdown_risk.py
-----------------
Two causal, self-normalizing risk overlays shared by all six paper/live
runners (BTC/ETH/SOL x hourly/15m), added 2026-07-23 after a deep-dive into
ETH 15m's 07-13/18 streak followed by a 07-19/22 giveback.

Both are informed by the SAME underlying observation, validated across all
three assets and both timeframes before deploy: strong trailing performance
is followed by weaker forward performance (weekly PnL autocorrelation
-0.21 to -0.72 across assets) -- ordinary regression-to-the-mean in a real
but noisy edge, not a predictable direction flip. Neither mechanism tries to
predict WHEN an edge will invert (every such attempt this session -- EMA
bias, Hurst, OU theta, BOS regime -- failed split-half/weekly robustness
checks). Instead they react to realized drawdown-from-peak, which is honest
and causal.

1. kelly_dampener_multiplier(): a SOFT, continuous size reducer. Tracks a
   rolling 10-day P&L high-water mark; when today's drawdown-from-peak,
   z-scored against that asset's own historical drawdown volatility,
   exceeds 2.0, returns a 0.5x multiplier for Kelly bet sizing. Self-resets
   as the account recovers OR as the stale peak ages out of the 10-day
   window (~10 days with no further losses). Backtested net-positive on
   all three assets' full paper-trading history at this config (BTC +$307,
   ETH +$175, SOL +$126) with 0 impact on the flagged assets' genuine
   winning streaks (drawdown-from-peak stays ~0 while a new peak keeps
   being made). Safe to apply in BOTH paper and live -- it only shrinks
   bet size, never stops trading, so it doesn't compromise paper's
   always-on data-collection role.

2. cascading_daily_loss_limit(): a HARD, ratchet-style circuit breaker.
   Starts at a configured base (whatever --daily-loss-limit already is for
   that runner). Each day the EFFECTIVE limit is breached, tomorrow's limit
   drops one rung (default 80% of the prior rung), down to a floor (default
   60% of base -- e.g. base=250 -> 200 -> 150, matching the ratio the user
   specified). Any day that does NOT breach its effective limit resets back
   to the base the following day. Backtested far more conservative than the
   Kelly dampener: it only ever chains below the base rung on genuinely
   consecutive large-loss days (BTC: 1 such episode in 60 days; ETH: 1 in
   75 days, exactly the 07-20/22 stretch; SOL: 0 in 74 days -- SOL's losses
   have never yet been large/consecutive enough to chain). This mirrors the
   existing check_daily_loss_limit() architecture in live_trading.py
   (live-only hard stop) -- it supplies the LIMIT value; it does not itself
   block trades.
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _daily_pnl_before_today(csv_path: Path, pnl_col: str = "would_pnl",
                             time_col: str = "logged_at") -> pd.Series:
    """Resolved-trade daily PnL, STRICTLY before today (local calendar date),
    indexed by date, gap-filled to a continuous daily series. Excluding
    today entirely (rather than shifting within a series that includes it)
    keeps this correct regardless of how far into today's session we are.
    """
    if not Path(csv_path).exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or pnl_col not in df.columns or time_col not in df.columns:
        return pd.Series(dtype=float)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col, pnl_col])
    if df.empty:
        return pd.Series(dtype=float)
    df["_date"] = pd.to_datetime(df[time_col].dt.date)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    df = df[df["_date"] < today]
    if df.empty:
        return pd.Series(dtype=float)
    daily = df.groupby("_date")[pnl_col].sum()
    daily = daily.asfreq("D").fillna(0.0)
    return daily


def kelly_dampener_multiplier(
    csv_path: Path,
    lookback_days: int = 10,
    z_threshold: float = 2.0,
    dampen_factor: float = 0.5,
    min_periods: int = 15,
) -> Tuple[float, str]:
    """Return (multiplier, reason). multiplier is 1.0 (no-op) unless the
    causal drawdown-from-peak z-score exceeds z_threshold, in which case
    it's dampen_factor. Requires min_periods days of history to form a
    stable std estimate; fails open (1.0) before that / on any error.
    """
    try:
        daily = _daily_pnl_before_today(csv_path)
        if len(daily) < min_periods:
            return 1.0, f"insufficient history ({len(daily)}d < {min_periods}d) — no dampening"
        cum = daily.cumsum()
        hwm = cum.rolling(lookback_days, min_periods=1).max()
        dd = (hwm - cum).clip(lower=0)
        dd_std = dd.expanding(min_periods=min_periods).std()
        if dd_std.iloc[-1] in (0.0, None) or pd.isna(dd_std.iloc[-1]):
            return 1.0, "drawdown std not yet stable — no dampening"
        z = dd.iloc[-1] / dd_std.iloc[-1]
        if z > z_threshold:
            return dampen_factor, (
                f"drawdown-from-{lookback_days}d-peak z={z:.2f} > {z_threshold} "
                f"(dd=${dd.iloc[-1]:.2f}) → Kelly x{dampen_factor}"
            )
        return 1.0, f"drawdown z={z:.2f} <= {z_threshold} — no dampening"
    except Exception as e:
        return 1.0, f"dampener error ({e}) — fail-open, no dampening"


def cascading_daily_loss_limit(
    csv_path: Path,
    base_limit: float,
    step_ratio: float = 0.8,
    floor_ratio: float = 0.6,
) -> Tuple[float, str]:
    """Return (effective_limit, reason) for TODAY. Walks the full causal
    daily-PnL history forward from the first day on record, ratcheting the
    limit down step_ratio per consecutive breach (floor floor_ratio*base),
    resetting to base_limit the day after any non-breach. base_limit is
    whatever --daily-loss-limit already resolves to for this runner, so
    existing per-runner risk tuning (e.g. hourly BTC $250 vs hourly ETH/SOL
    $120) is preserved -- only the ratchet shape is added.
    """
    try:
        daily = _daily_pnl_before_today(csv_path)
        floor_limit = round(base_limit * floor_ratio, 2)
        if daily.empty:
            return base_limit, "no history yet — base limit"
        limit = base_limit
        for pnl in daily.values:
            hit = pnl <= -abs(limit)
            limit = max(floor_limit, round(limit * step_ratio, 2)) if hit else base_limit
        if limit < base_limit:
            return limit, (
                f"cascaded to ${limit:.2f} after a consecutive-breach streak "
                f"(base=${base_limit:.2f}, floor=${floor_limit:.2f})"
            )
        return limit, f"base limit ${base_limit:.2f} (no active cascade)"
    except Exception as e:
        return base_limit, f"cascade error ({e}) — fail-open, base limit"
