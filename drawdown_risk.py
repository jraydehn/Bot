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

3. realized_edge_dampener_multiplier(): a FAST, trade-count-windowed (not
   calendar-day-windowed) sibling to the Kelly dampener, added 2026-07-25
   after SOL 15m round-tripped +$1,190 -> +$42 -> +$399 within a single
   calendar day -- a swing large enough and fast enough that neither the
   10-day Kelly dampener nor the 96h price-extension dampener reacted (SOL
   stayed near its 4-day low throughout, never triggering the latter).
   Tracks the trailing `window_trades` resolved trades (any side) and
   computes realized edge = WR - avg breakeven WR; dampens new bets 0.5x
   when that rolling edge is running meaningfully negative.

   CRITICAL ASSET-SPECIFIC FINDING: this only backtests as a net positive
   for BTC (+$382 at window=20/thresh=-0.08) and ETH (+$601, same config)
   -- SOL's realized edge shows genuine NEGATIVE autocorrelation at the
   trade-count level (a 25-point parameter sweep found EVERY combination
   net-negative-or-negligible for SOL, with flagged periods showing large
   POSITIVE would-be PnL, up to +$1,327). SOL's bad stretches predict
   recoveries, not continuations -- the opposite of BTC/ETH, and exactly
   the pattern observed live on 2026-07-25. Deploying this for SOL would
   systematically dampen right before its bounces. The function refuses to
   dampen SOL by design (hard-coded, not just "don't call it there") --
   see the asset allowlist below.
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Assets this dampener is validated for. SOL is deliberately excluded --
# see module docstring point 3. Revisit only after a fresh backtest shows
# SOL's autocorrelation has genuinely changed, not just a different
# parameter search.
_REALIZED_EDGE_DAMPENER_ASSETS = {"BTC", "ETH"}


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


def realized_edge_dampener_multiplier(
    csv_path: Path,
    asset: str,
    window_trades: int = 20,
    edge_threshold: float = -0.08,
    dampen_factor: float = 0.5,
    min_trades: int = 10,
) -> Tuple[float, str]:
    """Return (multiplier, reason). Dampens new bet sizing 0.5x when the
    trailing `window_trades` resolved trades' realized edge (win rate minus
    average breakeven win rate, both sides combined) has fallen below
    `edge_threshold`. Fully causal: only ever looks at trades strictly
    before the current decision, no lookahead.

    BTC/ETH only -- see module docstring point 3 for why SOL is excluded
    by design, not by caller convention. Calling this for SOL returns a
    no-op with an explicit reason, so a caller can't silently misuse it.
    """
    if asset.upper() not in _REALIZED_EDGE_DAMPENER_ASSETS:
        return 1.0, f"realized-edge dampener not validated for {asset.upper()} — no-op by design"
    try:
        if not Path(csv_path).exists():
            return 1.0, "no trade history yet — no dampening"
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty or "decision" not in df.columns:
            return 1.0, "no trade history yet — no dampening"
        trades = df[df["decision"] == "trade"].dropna(subset=["would_pnl", "would_win", "logged_at", "side", "p_market"])
        if trades.empty:
            return 1.0, "no resolved trades yet — no dampening"
        trades = trades.copy()
        trades["logged_at"] = pd.to_datetime(trades["logged_at"], errors="coerce", utc=True)
        trades = trades.dropna(subset=["logged_at"]).sort_values("logged_at")
        if len(trades) < min_trades:
            return 1.0, f"insufficient history ({len(trades)} < {min_trades} trades) — no dampening"
        window = trades.tail(window_trades)
        win = pd.to_numeric(window["would_win"], errors="coerce")
        pm = pd.to_numeric(window["p_market"], errors="coerce")
        be = pd.Series(
            [pm.iloc[i] if window["side"].iloc[i] == "yes" else 1 - pm.iloc[i] for i in range(len(window))],
            index=window.index,
        )
        realized_edge = win.mean() - be.mean()
        if realized_edge < edge_threshold:
            return dampen_factor, (
                f"trailing {len(window)}-trade realized edge={realized_edge:+.3f} < {edge_threshold:+.2f} "
                f"(WR={win.mean():.3f} vs avg BE={be.mean():.3f}) → Kelly x{dampen_factor}"
            )
        return 1.0, f"trailing {len(window)}-trade realized edge={realized_edge:+.3f} >= {edge_threshold:+.2f} — no dampening"
    except Exception as e:
        return 1.0, f"realized-edge dampener error ({e}) — fail-open, no dampening"


def losing_streak_active(csv_path: Path, min_streak: int = 2) -> Tuple[bool, int, str]:
    """Causal: walks ALL resolved 'trade' rows (both sides pooled), strictly
    before the current decision, in chronological order. streak_in > 0 =
    current winning-streak length, < 0 = losing-streak length. Returns
    True when streak_in <= -min_streak (an active losing streak).

    Asset-agnostic (takes whatever csv_path is passed) -- added 2026-07-26
    for SOL 15m (originally named sol_15m_losing_streak_active), then reused
    2026-07-26 for BTC/ETH 15m after the same streak-conditioned trend sweep
    found analogous but asset-specific interaction effects: within an active
    losing streak, a signal's recent trend differentiates a genuinely
    better- (or for BTC's kalman_velocity case, worse-) than-average forward
    outcome. The effect does NOT exist outside an active losing streak --
    this gate is the precondition, not a standalone signal, for all three
    assets' conditional boosts/dampeners.
    """
    try:
        if not Path(csv_path).exists():
            return False, 0, "no trade history yet"
        df = pd.read_csv(csv_path, low_memory=False)
        trades = df[df["decision"] == "trade"].dropna(subset=["would_win", "logged_at"])
        if trades.empty:
            return False, 0, "no resolved trades yet"
        trades = trades.copy()
        trades["logged_at"] = pd.to_datetime(trades["logged_at"], errors="coerce", utc=True)
        trades = trades.dropna(subset=["logged_at"]).sort_values("logged_at")
        cur = 0
        for w in pd.to_numeric(trades["would_win"], errors="coerce").dropna().values:
            cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
        is_ls = cur <= -min_streak
        return is_ls, cur, f"current streak={cur:+d}" + (" (active losing streak)" if is_ls else "")
    except Exception as e:
        return False, 0, f"streak calc error ({e}) -- fail-open, no boost"
