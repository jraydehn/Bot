"""
End-to-end demo for the Kalshi BTC event trading model.

Generates realistic mock 1-minute, 1-hour, and 4-hour OHLCV data for BTC,
runs every module in sequence, and prints a labeled summary of all intermediate
values and the final trade decision.
"""

import numpy as np
import pandas as pd

from market_data import compute_realized_volatility
from probability_engine import estimate_probability
from pricing_comparison import evaluate_edge
from market_structure import detect_market_structure, resample_to_15min
from confirmation_indicators import compute_confirmation
from kelly_sizing import compute_kelly_size
from decision import evaluate_trade


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

def make_ohlcv_1min(
    n: int = 2000,
    start_price: float = 84_000.0,
    drift: float = 0.00003,
    vol: float = 0.0006,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate n rows of synthetic 1-minute BTC OHLCV data with a DatetimeIndex.

    Uses a geometric random walk plus a deterministic zigzag component so that
    resampled 15-minute candles produce clear ascending swing highs and lows.
    2000 minutes → ~133 15-minute bars (well above the 120-candle minimum).

    Args:
        n: Number of 1-minute bars to generate (default 2000, ≥1800 needed for
           at least 120 15-minute bars after resampling).
        start_price: Starting BTC price in USD.
        drift: Per-minute log drift.
        vol: Per-minute log volatility.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume and a UTC DatetimeIndex.
        The DatetimeIndex is required for resample_to_15min().
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    closes_rw = start_price * np.exp(np.cumsum(log_returns))
    closes_rw = np.insert(closes_rw, 0, start_price)[:-1]  # shift so index 0 = start_price

    # Superimpose a deterministic triangle-wave zigzag (±1.5%, 600-min cycle =
    # 40 15-min bars per cycle) so swing pivots survive resampling.
    t = np.arange(n)
    zigzag_period = 600
    triangle = closes_rw * 0.015 * (
        2 * np.abs((t % zigzag_period) / zigzag_period - 0.5) - 0.5
    )
    closes = closes_rw + triangle

    noise = rng.uniform(0.0003, 0.001, n)
    opens = closes * np.exp(rng.normal(0, vol * 0.3, n))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows  = np.minimum(opens, closes) * (1 - noise)
    volume = rng.uniform(5, 50, n) * 1e6

    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volume}, index=idx)


def make_ohlcv_1h(
    n: int = 100,
    start_price: float = 82_000.0,
    drift: float = 0.0008,
    vol: float = 0.012,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate n rows of synthetic 1-hour BTC OHLCV data.

    A stronger drift produces a bullish EMA alignment and RSI above 50.

    Args:
        n: Number of 1-hour bars (minimum 60 required by confirmation_indicators).
        start_price: Starting BTC price.
        drift: Per-hour log drift.
        vol: Per-hour log volatility.
        seed: Random seed.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    closes = start_price * np.exp(np.cumsum(log_returns))
    closes = np.insert(closes, 0, start_price)[:-1]

    noise = rng.uniform(0.002, 0.008, n)
    opens = closes * np.exp(rng.normal(0, vol * 0.3, n))
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    # Inject above-average volume on the last candle to trigger volume_confirmed
    volume = rng.uniform(50, 200, n) * 1e6
    volume[-1] = volume[-1] * 2.5

    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volume})



# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def divider(title: str) -> None:
    """Print a section divider for readability."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main() -> None:
    """Run the full end-to-end demo and print all intermediate values."""

    print("\n" + "=" * 60)
    print("  KALSHI BTC EVENT TRADING MODEL — END-TO-END DEMO")
    print("  Scope: 1-hour expiry | 'Will BTC be above K at T?'")
    print("=" * 60)

    # --- Generate mock data ---
    df_1min = make_ohlcv_1min()          # 2000 1-min bars with DatetimeIndex
    df_1h = make_ohlcv_1h()
    df_15m = resample_to_15min(df_1min)  # ~133 15-min bars for market structure

    current_price = float(df_1min["close"].iloc[-1])
    strike = current_price * 0.997           # strike is 0.3% BELOW spot (slight ITM)
    tau = 60.0                               # 60 minutes to expiry
    p_market = 0.52                          # Kalshi market is pricing 52% YES
    bankroll = 10_000.0                      # $10,000 bankroll

    divider("TRADE SETUP")
    print(f"  Current BTC price  : ${current_price:,.2f}")
    print(f"  Strike (K)         : ${strike:,.2f}  (-0.3% below spot, slight ITM)")
    print(f"  Minutes to expiry  : {tau:.0f}")
    print(f"  Kalshi market price: {p_market:.2%} (YES probability)")
    print(f"  Bankroll           : ${bankroll:,.2f}")

    # -------------------------------------------------------------------
    # Module 1: Realized Volatility
    # -------------------------------------------------------------------
    divider("MODULE 1 — REALIZED VOLATILITY (1-min data)")
    vol_result = compute_realized_volatility(df_1min)
    print(f"  vol_30m  (per min) : {vol_result.vol_30m:.6f}")
    print(f"  vol_60m  (per min) : {vol_result.vol_60m:.6f}")
    print(f"  vol_120m (per min) : {vol_result.vol_120m:.6f}")

    # Use the 60-minute window as the primary sigma for a 1-hour expiry
    sigma_min = vol_result.vol_60m

    # -------------------------------------------------------------------
    # Module 2: Probability Engine
    # -------------------------------------------------------------------
    divider("MODULE 2 — PROBABILITY ENGINE")
    prob_result = estimate_probability(
        S=current_price, K=strike, tau=tau, sigma_min=sigma_min
    )
    print(f"  sigma_min          : {sigma_min:.6f}  (per-minute vol, 60m window)")
    print(f"  sigma_tau          : {prob_result.sigma_to_expiry:.6f}  (scaled to {tau:.0f}-min expiry)")
    print(f"  log_distance       : {prob_result.log_distance:.6f}  (ln(K/S))")
    print(f"  z_score            : {prob_result.z_score:.4f}  (std devs strike is above spot)")
    print(f"  expected_move_pct  : {prob_result.expected_move_pct:.4f}%  (1-sigma move)")
    print(f"  p_yes (model)      : {prob_result.p_yes:.4f}  ({prob_result.p_yes:.2%})")

    # -------------------------------------------------------------------
    # Module 3: Pricing Comparison
    # -------------------------------------------------------------------
    divider("MODULE 3 — PRICING COMPARISON")
    pricing_result = evaluate_edge(p_model=prob_result.p_yes, p_market=p_market)
    print(f"  p_model            : {prob_result.p_yes:.4f}")
    print(f"  p_market           : {p_market:.4f}")
    print(f"  raw_edge           : {pricing_result.raw_edge:.4f}")
    print(f"  net_edge           : {pricing_result.net_edge:.4f}")
    print(f"  qualifies          : {pricing_result.qualifies}")
    print(f"  reason             : {pricing_result.reason}")

    # -------------------------------------------------------------------
    # Module 4: Market Structure
    # -------------------------------------------------------------------
    divider("MODULE 4 — MARKET STRUCTURE (15-minute data)")
    structure_result = detect_market_structure(df_15m)
    print(f"  structure_bias     : {structure_result.structure_bias:+d}")
    print(f"  swing_highs        : {[f'${h:,.2f}' for h in structure_result.swing_highs]}")
    print(f"  swing_lows         : {[f'${l:,.2f}' for l in structure_result.swing_lows]}")
    print(f"  reason             : {structure_result.reason}")

    # -------------------------------------------------------------------
    # Module 5: Confirmation Indicators
    # -------------------------------------------------------------------
    divider("MODULE 5 — CONFIRMATION INDICATORS (1-hour data)")
    confirm_result = compute_confirmation(df_1h)
    print(f"  ema_20_current     : {confirm_result.ema_20_current:,.2f}")
    print(f"  ema_50_current     : {confirm_result.ema_50_current:,.2f}")
    print(f"  ema_alignment      : {confirm_result.ema_alignment}")
    print(f"  rsi_value          : {confirm_result.rsi_value:.2f}")
    print(f"  rsi_regime         : {confirm_result.rsi_regime}")
    print(f"  volume_confirmed   : {confirm_result.volume_confirmed}")
    print(f"  confirmation_bias  : {confirm_result.confirmation_bias:+d}")
    print(f"  reason             : {confirm_result.reason}")

    # -------------------------------------------------------------------
    # Module 6: Kelly Sizing (standalone view)
    # -------------------------------------------------------------------
    divider("MODULE 6 — KELLY SIZING")
    kelly_result = compute_kelly_size(
        p_model=prob_result.p_yes, p_market=p_market, bankroll=bankroll
    )
    print(f"  kelly_fraction     : {kelly_result.kelly_fraction:.4f}  ({kelly_result.kelly_fraction:.2%})")
    print(f"  bet_fraction       : {kelly_result.bet_fraction:.4f}  ({kelly_result.bet_fraction:.2%})")
    print(f"  was_capped         : {kelly_result.was_capped}")
    print(f"  bet_amount         : ${kelly_result.bet_amount:,.2f}")
    print(f"  reason             : {kelly_result.reason}")

    # -------------------------------------------------------------------
    # Module 7: Final Decision
    # -------------------------------------------------------------------
    divider("MODULE 7 — FINAL DECISION")
    decision = evaluate_trade(
        structure_bias=structure_result.structure_bias,
        confirmation_bias=confirm_result.confirmation_bias,
        p_model=prob_result.p_yes,
        p_market=p_market,
        bankroll=bankroll,
    )
    print(f"  decision           : {decision.decision.upper()}")
    print(f"  side               : {decision.side.upper()}")
    print(f"  p_model            : {decision.p_model:.4f}")
    print(f"  p_market           : {decision.p_market:.4f}")
    print(f"  raw_edge           : {decision.raw_edge:.4f}")
    print(f"  net_edge           : {decision.net_edge:.4f}")
    print(f"  structure_bias     : {decision.structure_bias:+d}")
    print(f"  confirmation_bias  : {decision.confirmation_bias:+d}")
    print(f"  kelly_fraction     : {decision.kelly_fraction:.4f}")
    print(f"  bet_fraction       : {decision.bet_fraction:.4f}  ({decision.bet_fraction:.2%})")
    print(f"  bet_amount         : ${decision.bet_amount:,.2f}")
    print(f"  was_capped         : {decision.was_capped}")
    print(f"\n  Gate outcomes:")
    for i, r in enumerate(decision.reasons, 1):
        print(f"    {i}. {r}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
