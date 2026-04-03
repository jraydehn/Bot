"""
Probability engine for estimating P(BTC finishes above strike K at expiry).

Uses a log-normal price model with an optional directional drift term derived
from market structure and confirmation signals. No machine learning; pure
closed-form math based on realized volatility.
"""

import math
from dataclasses import dataclass

from scipy.stats import norm




# Maximum z-shift as a fraction of sigma_tau.
# At alpha=0.20, a perfectly bullish signal (confirmation_score=max_score) shifts z
# by 0.20 * sigma_tau — meaningful but bounded; can never dominate log(K/S).
DIRECTION_ALPHA = 0.20


@dataclass
class ProbabilityResult:
    """All outputs from the probability engine."""

    p_yes: float            # probability BTC closes above strike at expiry
    z_score: float          # adjusted z-score (after directional shift)
    log_distance: float     # ln(K / S) — log-space gap between spot and strike
    sigma_to_expiry: float  # total volatility scaled to the full expiry window
    expected_move_pct: float  # 1-sigma expected move as a % of current price
    z_raw: float = 0.0          # unadjusted z-score (before directional shift)
    z_shift: float = 0.0        # directional shift applied to z
    direction_strength: float = 0.0  # normalized [-1, +1] confirmation strength


def estimate_probability(
    S: float,
    K: float,
    tau: float,
    sigma_min: float,
    structure_bias: int = 0,    # kept for API compatibility, unused in calculation
    confirmation_score: int = 0,
    max_score: int = 5,
) -> ProbabilityResult:
    """
    Estimate the probability that BTC finishes above strike K at expiration.

    Uses a log-normal model with a capped directional z-shift derived from
    confirmation indicators. The shift is bounded at DIRECTION_ALPHA * sigma_tau
    so it can never dominate the log-distance term (unlike accumulated drift).

    Shift formula: z_adjusted = z - alpha * D * sigma_tau
      D = confirmation_score / max_score, clamped to [-1, +1]
      Bullish (D > 0) → lower z → higher p_yes
      Bearish (D < 0) → higher z → lower p_yes
      Neutral (D = 0) → no shift (default behavior)

    Args:
        S: Current BTC spot price in USD.
        K: Strike price of the event contract in USD.
        tau: Minutes remaining until expiry. Must be > 0.
        sigma_min: Realized per-minute volatility (std of log returns). Must be > 0.
        structure_bias: Unused (kept for API compatibility).
        confirmation_score: Net confirmation indicator score (e.g. -5 to +5).
        max_score: Maximum possible absolute value of confirmation_score.

    Returns:
        ProbabilityResult with p_yes and supporting diagnostic metrics.

    Raises:
        ValueError: If tau <= 0 or sigma_min <= 0.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive (minutes to expiry), got {tau}")
    if sigma_min <= 0:
        raise ValueError(f"sigma_min must be positive, got {sigma_min}")

    # Scale per-minute volatility up to the full expiry window using sqrt(time)
    sigma_tau = sigma_min * math.sqrt(tau)

    # Log-distance: how far the strike sits above current price in log space
    log_distance = math.log(K / S)

    # z: unadjusted standard deviations from spot to strike (pure log-normal)
    z = log_distance / sigma_tau

    # Directional z-shift: shift z toward the direction of confirmation signals.
    # Bounded by DIRECTION_ALPHA * sigma_tau — proportional to volatility, never linear.
    if max_score > 0 and confirmation_score != 0:
        direction_strength = max(-1.0, min(1.0, confirmation_score / max_score))
        z_shift = DIRECTION_ALPHA * direction_strength * sigma_tau
    else:
        direction_strength = 0.0
        z_shift = 0.0

    z_adjusted = z - z_shift  # bullish shift reduces z, increasing p_yes

    # p_yes: probability that BTC finishes above the strike (right-tail area)
    p_yes = 1 - norm.cdf(z_adjusted)

    # Expected move: convert 1-sigma log move to a percentage of current price
    expected_move_pct = (math.exp(sigma_tau) - 1) * 100

    return ProbabilityResult(
        p_yes=p_yes,
        z_score=z_adjusted,
        log_distance=log_distance,
        sigma_to_expiry=sigma_tau,
        expected_move_pct=expected_move_pct,
        z_raw=z,
        z_shift=z_shift,
        direction_strength=direction_strength,
    )


# Fraction of realized vol in the blend (0 = all implied, 1 = all realized).
# At 0.3, the model uses 30% realized vol + 70% market-implied vol.
REALIZED_VOL_WEIGHT = 0.6


def blend_vol(
    vol_realized: float,
    vol_implied: float,
    weight: float = REALIZED_VOL_WEIGHT,
) -> float:
    """
    Linear blend of realized and market-implied per-minute volatility.
    Falls back to realized vol if implied is unavailable (NaN or non-positive).

    Args:
        vol_realized: Realized per-minute vol from compute_realized_volatility().
        vol_implied:  Implied per-minute vol from implied_vol_from_price().
        weight:       Weight on realized vol (1-weight goes to implied).
    """
    if not (vol_implied > 0):   # handles NaN, inf, and non-positive
        return vol_realized
    return weight * vol_realized + (1.0 - weight) * vol_implied


def implied_vol_from_price(
    p_market: float,
    S: float,
    K: float,
    tau: float,
) -> float:
    """
    Back-calculate per-minute volatility implied by the Kalshi market price,
    using the inverse of the log-normal model in estimate_probability().

    p_market = 1 - norm.cdf(log(K/S) / (sigma_min * sqrt(tau)))
    => sigma_min = log(K/S) / (norm.ppf(1 - p_market) * sqrt(tau))

    Valid for both OTM YES (K > S, p_market < 0.50) and OTM NO / ITM YES
    (K < S, p_market > 0.50) — in both cases log(K/S) and z_implied have
    matching signs, producing a positive sigma_min.

    Returns float('nan') if inputs are degenerate or signs are inconsistent
    (e.g. K > S but p_market > 0.50, or K = S, or p_market = 0.50).
    """
    try:
        if not (0 < p_market < 1) or tau <= 0 or K == S:
            return float("nan")
        log_dist  = math.log(K / S)
        z_implied = norm.ppf(1.0 - p_market)
        if z_implied == 0:
            return float("nan")
        sigma_min = log_dist / (z_implied * math.sqrt(tau))
        if sigma_min <= 0:
            return float("nan")  # inconsistent: market price contradicts strike direction
        return sigma_min
    except Exception:
        return float("nan")
