"""
Probability engine for estimating P(BTC finishes above strike K at expiry).

Uses a zero-drift log-normal price model. No machine learning; pure closed-form
math based on realized volatility.
"""

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class ProbabilityResult:
    """All outputs from the probability engine."""

    p_yes: float            # probability BTC closes above strike at expiry
    z_score: float          # standard deviations the strike is above current price
    log_distance: float     # ln(K / S) — log-space gap between spot and strike
    sigma_to_expiry: float  # total volatility scaled to the full expiry window
    expected_move_pct: float  # 1-sigma expected move as a % of current price


def estimate_probability(
    S: float,
    K: float,
    tau: float,
    sigma_min: float,
) -> ProbabilityResult:
    """
    Estimate the probability that BTC finishes above strike K at expiration.

    Assumes a log-normal model with no drift. Volatility is scaled from
    per-minute to the full expiry window using the square-root-of-time rule.

    Args:
        S: Current BTC spot price in USD.
        K: Strike price of the event contract in USD.
        tau: Minutes remaining until expiry. Must be > 0.
        sigma_min: Realized per-minute volatility (std of log returns). Must be > 0.

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

    # z: how many standard deviations the strike is away from current price
    z = log_distance / sigma_tau

    # p_yes: probability that BTC finishes above the strike (right-tail area under normal curve)
    p_yes = 1 - norm.cdf(z)

    # Expected move: convert 1-sigma log move to a percentage of current price
    expected_move_pct = (math.exp(sigma_tau) - 1) * 100

    return ProbabilityResult(
        p_yes=p_yes,
        z_score=z,
        log_distance=log_distance,
        sigma_to_expiry=sigma_tau,
        expected_move_pct=expected_move_pct,
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

    Returns float('nan') if the inputs are degenerate (e.g. ITM strike,
    extreme probabilities, or non-positive tau).
    """
    try:
        if not (0 < p_market < 1) or K <= S or tau <= 0:
            return float("nan")
        z_implied = norm.ppf(1.0 - p_market)
        if z_implied <= 0:
            return float("nan")
        return math.log(K / S) / (z_implied * math.sqrt(tau))
    except Exception:
        return float("nan")
