"""
price_extension_risk.py
------------------------
A third, independent risk overlay (2026-07-23) alongside drawdown_risk.py's
two P&L-lagging mechanisms. This one reacts to REAL-TIME PRICE STRUCTURE
instead of trailing PnL, so it doesn't share their "penalty box" lag and
engages/disengages daily as price moves, not over ~10 days.

Origin: investigating ETH 15m's 07-13/18 streak -> 07-19/22 giveback found
that NO's edge (which carries most of the capital via the Kelly sizing
asymmetry) systematically weakens near a fresh multi-day high, and YES's
edge weakens near a fresh multi-day low -- confirmed on the real settled-
market archive (n=400-2000+ per bucket), split-half robust, and independently
confirmed in raw short-horizon (15m/60m) price continuation across all three
assets (P(up 15m) ~57-58% near a 96h high vs ~50% baseline, all three
assets). Initially found weak/inverted for BTC/SOL when tested against the
actual taken paper trades -- traced to those runners' 15m history spanning
three since-superseded model versions plus small samples, not a real asset
difference; re-tested against the current model + full archive and it
replicates cleanly on all three. Also replicates directionally at the
HOURLY timeframe on all three assets' real scan archives.

Relationship to btc_donch_high_no_gate (existing, paper_trade_runner.py):
that gate hard-blocks BTC/ETH hourly NO on a SHORTER 20h Donchian window.
This module uses a longer 96h (4-day) window and applies a soft dampener
(not a block) everywhere, including where the 20h gate doesn't reach.
Checked for overlap: when the 20h gate fires it already `continue`s before
sizing runs, so this module's code never executes for those candidates --
no double-dampening. The incremental population (96h-extended but not
20h-extended) still shows a real negative edge for NO (BTC -9.16pp, ETH
-2.97pp on the real archive), so this adds coverage rather than sitting
idle behind the existing gate.
"""
from typing import Tuple

import pandas as pd


def donchian_pos(df_1h: pd.DataFrame, window: int = 96,
                  drop_forming_bar: bool = True) -> float:
    """Position of the current close within the trailing `window`-hour
    Donchian channel: 0.0 = at the low, 1.0 = at the high. Uses only
    COMPLETED bars for the high/low range (drops the forming last bar
    when drop_forming_bar, matching this codebase's established
    completed-bar convention -- see _kc_pct_1h_done precedent). Returns
    None if there isn't enough history.
    """
    if df_1h is None or len(df_1h) < window + (1 if drop_forming_bar else 0):
        return None
    d = df_1h.iloc[:-1] if drop_forming_bar else df_1h
    if len(d) < window:
        return None
    hi = float(d["high"].iloc[-window:].max())
    lo = float(d["low"].iloc[-window:].min())
    if hi <= lo:
        return None
    close = float(d["close"].iloc[-1])
    return (close - lo) / (hi - lo)


def donchian_dampener_multiplier(
    df_1h: pd.DataFrame,
    side: str,
    window: int = 96,
    threshold: float = 0.80,
    dampen_factor: float = 0.5,
    drop_forming_bar: bool = True,
) -> Tuple[float, str]:
    """Return (multiplier, reason). NO gets dampened near a fresh high,
    YES gets dampened near a fresh low -- both edges measurably weaken
    there (see module docstring). 1.0 (no-op) everywhere else, or on
    insufficient history / any error (fail-open).

    drop_forming_bar: True for live-fetched series that include an
    in-progress last candle (e.g. the 15m runner's Binance-fetched
    live_1h). False for parquet-backed native series whose last row is
    already a completed bar (e.g. the hourly runner's df_confirm -- see
    its existing _dc_hi/_dc_lo computation, which uses .iloc[-1] directly
    with no drop).
    """
    try:
        pos = donchian_pos(df_1h, window=window, drop_forming_bar=drop_forming_bar)
        if pos is None:
            return 1.0, f"insufficient {window}h history — no dampening"
        if side == "no" and pos >= threshold:
            return dampen_factor, (
                f"donch_pos_{window}h={pos:.3f}>={threshold} (near {window}h high, "
                f"NO edge measurably weaker there) → NO x{dampen_factor}"
            )
        if side == "yes" and pos <= (1.0 - threshold):
            return dampen_factor, (
                f"donch_pos_{window}h={pos:.3f}<={1.0-threshold:.2f} (near {window}h low, "
                f"YES edge measurably weaker there) → YES x{dampen_factor}"
            )
        return 1.0, f"donch_pos_{window}h={pos:.3f} — no dampening"
    except Exception as e:
        return 1.0, f"donchian dampener error ({e}) — fail-open, no dampening"
