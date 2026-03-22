"""
Decision module: evaluates structure, confirmation, and edge gates in sequence,
then calls Kelly sizing to produce a final trade decision with full audit trail.
"""

from dataclasses import dataclass, field
from typing import List

from pricing_comparison import evaluate_edge, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from kelly_sizing import compute_kelly_size


@dataclass
class DecisionResult:
    """Fully structured output of the decision module."""

    decision: str             # "trade" or "no_trade"
    side: str                 # "yes" or "no" — the direction of the proposed trade
    p_model: float            # model's estimated probability
    p_market: float           # Kalshi market-implied probability
    raw_edge: float           # p_model - p_market before costs
    net_edge: float           # edge after deducting fee and slippage
    structure_bias: int       # +1, -1, or 0 from market structure module
    confirmation_bias: int    # +1, -1, or 0 from confirmation indicators module
    kelly_fraction: float     # raw Kelly fraction (0 if no trade)
    bet_fraction: float       # capped fraction to bet (0 if no trade)
    bet_amount: float         # dollar bet size (0 if no trade)
    was_capped: bool          # True if 5% cap was applied
    reasons: List[str]        # ordered list explaining each gate outcome


def evaluate_trade(
    structure_bias: int,
    confirmation_bias: int,
    p_model: float,
    p_market: float,
    bankroll: float,
    slippage: float = DEFAULT_SLIPPAGE,
    spread: float = DEFAULT_SPREAD,
    min_net_edge: float = MIN_NET_EDGE,
) -> DecisionResult:
    """
    Evaluate all gates in sequence and produce a final trade decision.

    Gates are evaluated in this order — if any gate fails, the function
    returns immediately with decision="no_trade":
        1. Market structure bias must align with the proposed trade direction.
        2. Confirmation indicators bias must align with the proposed trade direction.
        3. Net edge (after fees and slippage) must exceed the minimum threshold.

    If all three gates pass, Kelly sizing is called to determine bet amount.

    Trade side is determined by the gate signals, not by comparing p_model to
    p_market. structure_bias = +1 and confirmation_bias = +1 → YES trade;
    both = -1 → NO trade. If the biases are neutral or conflicting, Gate 1 or
    Gate 2 blocks the trade before side matters.

    For NO bets the edge is direction-adjusted: since a NO bet profits when the
    market overprices YES, the relevant raw_edge is p_market - p_model (how much
    the market overestimates the probability of YES).

    Args:
        structure_bias: +1 / -1 / 0 from detect_market_structure().
        confirmation_bias: +1 / -1 / 0 from compute_confirmation().
        p_model: Model's probability estimate for YES resolution.
        p_market: Kalshi market-implied probability.
        bankroll: Total capital available for sizing, in USD.
        slippage: Slippage fraction (default from pricing_comparison).
        spread: Bid-ask spread cost per side (default from pricing_comparison).
        min_net_edge: Minimum net edge required (default from pricing_comparison).

    Returns:
        DecisionResult with all gate outcomes, sizing details, and reasons list.
    """
    reasons: List[str] = []

    # Side and required bias come from the gate signals, not from p_model vs p_market.
    # structure_bias = +1 → YES trade; -1 → NO trade; 0 → blocked at Gate 1.
    # We derive side from structure_bias here so the gate checks are self-consistent.
    side = "yes" if structure_bias == 1 else "no"
    required_bias = +1 if side == "yes" else -1

    # Direction-aware edge:
    #   YES bet: we profit when market underprices YES → edge = p_model - p_market
    #   NO  bet: we profit when market overprices YES  → edge = p_market - p_model
    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # --- Gate 1: Market structure ---
    if structure_bias != required_bias:
        reasons.append(
            f"Gate 1 FAILED: structure_bias={structure_bias} does not align with "
            f"required bias {required_bias} for a {side.upper()} trade. "
            f"The 4-hour trend does not support this direction."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(
        f"Gate 1 PASSED: structure_bias={structure_bias} aligns with {side.upper()} trade."
    )

    # --- Gate 2: Confirmation indicators ---
    if confirmation_bias != required_bias:
        reasons.append(
            f"Gate 2 FAILED: confirmation_bias={confirmation_bias} does not align with "
            f"required bias {required_bias} for a {side.upper()} trade. "
            f"EMA/RSI/volume signals do not confirm this direction."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(
        f"Gate 2 PASSED: confirmation_bias={confirmation_bias} aligns with {side.upper()} trade."
    )

    # --- Gate 3: Net edge threshold ---
    # Pass direction-adjusted probabilities so evaluate_edge always sees a positive edge
    # for trades that clear Gates 1 and 2: YES → (p_model, p_market); NO → (p_market, p_model)
    p_edge, p_ref = (p_model, p_market) if side == "yes" else (p_market, p_model)
    pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
    if not pricing.qualifies:
        reasons.append(f"Gate 3 FAILED: {pricing.reason}")
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate 3 PASSED: {pricing.reason}")

    # --- All gates passed: compute Kelly sizing ---
    # p_model and p_market are always passed as YES-space probabilities.
    # compute_kelly_size applies the correct payout formula internally based on side.
    # Multipliers differ by side to reflect their different variance profiles:
    #   YES: quarter Kelly (0.25) — ~26% win rate, high per-trade variance
    #   NO:  half Kelly    (0.50) — ~96% win rate, low per-trade variance
    kelly_multiplier = 0.25 if side == "yes" else 0.50
    kelly = compute_kelly_size(p_model, p_market, bankroll, kelly_multiplier, side=side)
    reasons.append(f"Sizing: {kelly.reason}")

    return DecisionResult(
        decision="trade", side=side,
        p_model=p_model, p_market=p_market,
        raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
        structure_bias=structure_bias, confirmation_bias=confirmation_bias,
        kelly_fraction=kelly.kelly_fraction, bet_fraction=kelly.bet_fraction,
        bet_amount=kelly.bet_amount, was_capped=kelly.was_capped,
        reasons=reasons,
    )
