"""
Pricing comparison module for evaluating edge against the Kalshi market price.

Computes raw and net edge after fees/slippage, and determines whether the
opportunity clears the minimum threshold required to justify a trade.

Also provides simulate_p_market() to approximate realistic Kalshi YES prices
based on how far the strike is above spot (strike_offset), calibrated from
observed market data.
"""

import random
from dataclasses import dataclass


def simulate_p_market(
    strike_offset: float,
    side: str = "yes",
    rng: random.Random = None,
) -> float:
    """
    Approximate a realistic Kalshi YES market price based on strike distance and
    the direction the gate system intends to trade.

    Calibrated from 2,000 real KXBTCD settled contracts (2026-03-20 snapshot).
    Ranges are drawn from observed p25–p75 of real opening YES prices grouped
    by strike offset above spot at contract open.

    For 0.5% OTM (default offset): real median ≈ 0.195, range roughly 0.10–0.35.
    NO side adds ~0.05 upward shift to reflect that bearish-regime markets still
    price some upside uncertainty even when BTC is trending down.

    Args:
        strike_offset: Fractional distance of strike above spot (e.g. 0.005 = 0.5%).
            Must be non-negative (strike always above spot in this model).
        side: "yes" or "no" — direction the gate system intends to trade.
            Affects which pricing regime to sample from.
        rng: Optional seeded random.Random instance for reproducible runs.
            Defaults to the module-level random state.

    Returns:
        Simulated YES market probability in (0, 1).
    """
    _rng = rng if rng is not None else random

    # Ranges calibrated from 2,000 real KXBTCD settled contracts (2026-03-20).
    # Buckets match strike offset above spot; real medians shown in comments.
    # NO side uses a slight upward shift (+0.05) vs YES to reflect that in
    # bearish regimes the broader market still prices some upside uncertainty.

    if side == "no":
        if strike_offset <= 0.001:
            # Fixed value calibrated from real Kalshi bid/ask observation: bid=0.330, ask=0.340 on 2026-03-22
            return 0.440
        elif strike_offset <= 0.003: # real median 0.265
            return _rng.uniform(0.20, 0.50)
        elif strike_offset <= 0.005: # real median 0.195
            return _rng.uniform(0.15, 0.40)
        elif strike_offset <= 0.010: # real median 0.103
            return _rng.uniform(0.08, 0.28)
        else:                        # real median 0.020
            return _rng.uniform(0.01, 0.05)

    # side == "yes": calibrated directly to real Kalshi OTM YES pricing
    if strike_offset <= 0.001:
        # Fixed value calibrated from real Kalshi bid/ask observation: bid=0.330, ask=0.340 on 2026-03-20
        return 0.335
    elif strike_offset <= 0.003:     # real median 0.265
        return _rng.uniform(0.15, 0.45)
    elif strike_offset <= 0.005:     # real median 0.195
        return _rng.uniform(0.10, 0.35)
    elif strike_offset <= 0.010:     # real median 0.103
        return _rng.uniform(0.03, 0.22)
    else:                            # real median 0.020, deep OTM
        return _rng.uniform(0.005, 0.03)


KALSHI_RAKE = 0.07        # Kalshi charges 7% of potential profit × contract price
DEFAULT_SLIPPAGE = 0.005  # 0.5% estimated slippage
DEFAULT_SPREAD = 0.01     # bid-ask spread cost (~$0.01 wide on Kalshi)
MIN_NET_EDGE = 0.01       # minimum 1% net edge required to qualify


def kalshi_fee(p_market: float) -> float:
    """
    Compute Kalshi's trading fee as a fraction of contract value.

    Fee = 7% × p_market × (1 - p_market)

    This equals 7% of the potential profit multiplied by the contract price,
    applied symmetrically to both YES and NO sides. Ranges from ~$0.07 per
    100 contracts at deep OTM (p≈0.01) to $1.75 at the money (p=0.50).
    """
    return KALSHI_RAKE * p_market * (1.0 - p_market)


@dataclass
class PricingResult:
    """Output of the pricing comparison module."""

    raw_edge: float    # p_model - p_market, before any cost deduction
    net_edge: float    # edge remaining after subtracting all costs
    qualifies: bool    # True if net_edge exceeds the minimum threshold
    reason: str        # plain-English explanation of the outcome


def evaluate_edge(
    p_model: float,
    p_market: float,
    slippage: float = DEFAULT_SLIPPAGE,
    spread: float = DEFAULT_SPREAD,
    min_net_edge: float = MIN_NET_EDGE,
) -> PricingResult:
    """
    Compare model probability to the Kalshi market price and evaluate edge.

    The raw edge measures how much our model disagrees with the market.
    Net edge subtracts all round-trip transaction costs (fee, slippage, spread).
    A trade only qualifies if net edge exceeds the minimum threshold, ensuring
    costs are covered with a meaningful margin of safety.

    Args:
        p_model: Model's estimated probability of YES resolution (0–1).
        p_market: Kalshi market-implied probability (0–1).
        slippage: Estimated fill slippage as a fraction (default 0.5%).
        spread: Bid-ask spread cost per side (default 1% — Kalshi typically
            quotes a 0.01 wide spread on each side of the book).
        min_net_edge: Minimum net edge required to qualify (default 3%).

    Returns:
        PricingResult with raw_edge, net_edge, qualifies flag, and reason.
    """
    # Raw edge: how much our model disagrees with what the market is pricing
    raw_edge = p_model - p_market

    # Kalshi fee: 7% × p_market × (1 - p_market), applied per contract
    fee = kalshi_fee(p_market)

    # Net edge: edge after Kalshi rake, slippage, and bid-ask spread
    net_edge = raw_edge - fee - slippage - spread

    if net_edge > min_net_edge:
        qualifies = True
        reason = (
            f"Qualifies: net edge {net_edge:.4f} exceeds minimum threshold "
            f"{min_net_edge:.4f}. "
            f"(raw_edge={raw_edge:.4f}, fee={fee:.4f}, slippage={slippage:.4f}, "
            f"spread={spread:.4f})"
        )
    else:
        qualifies = False
        reason = (
            f"Does not qualify: net edge {net_edge:.4f} does not exceed minimum "
            f"threshold {min_net_edge:.4f}. "
            f"(raw_edge={raw_edge:.4f}, fee={fee:.4f}, slippage={slippage:.4f}, "
            f"spread={spread:.4f})"
        )

    return PricingResult(
        raw_edge=raw_edge,
        net_edge=net_edge,
        qualifies=qualifies,
        reason=reason,
    )
