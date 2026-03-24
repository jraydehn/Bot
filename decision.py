"""
Decision module: evaluates structure, confirmation, and edge gates in sequence,
then calls Kelly sizing to produce a final trade decision with full audit trail.
"""

from dataclasses import dataclass, field
from typing import List

from pricing_comparison import evaluate_edge, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from kelly_sizing import compute_kelly_size

# Extra net-edge required when structure is neutral (0) but confirmation is bearish.
# Compensates for the missing directional structural confirmation — only trades with
# a larger probability gap justify entering during a ranging/transitioning market.
NEUTRAL_STRUCTURE_EDGE_PREMIUM = 0.01

# Gate P: pure-edge override threshold and Kelly multiplier.
# When the market mispricing is large enough, take the trade regardless of
# structure/confirmation. 1/8 Kelly is used (vs 1/4 or 1/2) because there is
# no technical confirmation backing the direction.
PURE_EDGE_MIN_NET_EDGE     = 0.08
PURE_EDGE_KELLY_MULTIPLIER = 0.125


def _pure_edge_override(
    p_model: float,
    p_market: float,
    bankroll: float,
    structure_bias: int,
    confirmation_bias: int,
    confirmation_score: int,
    no_score: int,
    prior_reasons: List[str],
    slippage: float,
    spread: float,
) -> "DecisionResult | None":
    """
    Gate P: fire when either direction has a net edge >= threshold,
    bypassing structure and confirmation gates. Returns a DecisionResult trade if
    the threshold is met, else None (caller should return no_trade).

    Tiered edge requirements — 3-indicator model, score range -3 to +3:

    YES (confirmation_score -3 to +3):
      score >= 3 (all three bullish)       -> edge >= 6%
      score >= 1 (net positive)            -> edge >= 8%
      score <= 0 (neutral or bearish)      -> blocked

    NO (no_score -3 to +3):
      no_score <= -2 (EMA+RSI+Vol bearish) -> edge >= 6%
      no_score <=  0 (neutral/mixed)       -> edge >= 8%
      no_score <=  1 (slightly bullish)    -> edge >= 10%
      no_score >=  2 (clearly bullish)     -> blocked
    """
    fee = kalshi_fee(p_market)
    yes_net = (p_model - p_market) - fee - slippage - spread
    no_net  = (p_market - p_model) - fee - slippage - spread

    if confirmation_score >= 3:
        yes_min = 0.06
    elif confirmation_score >= 1:
        yes_min = PURE_EDGE_MIN_NET_EDGE  # 8%
    else:
        yes_min = float("inf")            # blocked

    if no_score <= -2:
        no_min = 0.06
    elif no_score <= 0:
        no_min = PURE_EDGE_MIN_NET_EDGE   # 8%
    elif no_score <= 1:
        no_min = 0.10
    else:
        no_min = float("inf")             # blocked

    yes_ok = yes_net >= yes_min
    no_ok  = no_net  >= no_min

    if yes_ok:
        pure_side, pure_net, pure_raw = "yes", yes_net, p_model - p_market
    elif no_ok:
        pure_side, pure_net, pure_raw = "no",  no_net,  p_market - p_model
    else:
        return None

    yes_min_str = f"{yes_min:.0%}" if yes_min != float("inf") else "blocked"
    no_min_str  = f"{no_min:.0%}"  if no_min  != float("inf") else "blocked"
    reasons = list(prior_reasons) + [
        f"Gate P PASSED: pure-edge override — {pure_side.upper()} net_edge={pure_net:+.4f} "
        f"exceeds threshold. "
        f"yes_score={confirmation_score:+d} (min={yes_min_str}), no_score={no_score:+d} (min={no_min_str}). "
        f"(1/8 Kelly applied)."
    ]
    kelly = compute_kelly_size(p_model, p_market, bankroll, PURE_EDGE_KELLY_MULTIPLIER, side=pure_side)
    reasons.append(f"Sizing: {kelly.reason}")
    return DecisionResult(
        decision="trade", side=pure_side,
        p_model=p_model, p_market=p_market,
        raw_edge=pure_raw, net_edge=pure_net,
        structure_bias=structure_bias, confirmation_bias=confirmation_bias,
        kelly_fraction=kelly.kelly_fraction, bet_fraction=kelly.bet_fraction,
        bet_amount=kelly.bet_amount, was_capped=kelly.was_capped,
        reasons=reasons,
    )


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
    confirmation_score: int = 0,
    no_score: int = 0,
) -> DecisionResult:
    """
    Evaluate all gates in sequence and produce a final trade decision.

    Gates are evaluated in this order:
        1. Market structure bias must support the proposed trade direction.
           - YES trades: structure_bias must be +1 (bullish).
           - NO trades:  structure_bias must be -1 (bearish, standard threshold)
                         OR 0 (neutral, higher edge threshold applied at Gate 3).
                         structure_bias = +1 always blocks a NO trade.
        2. Confirmation indicators bias must align with the proposed trade direction.
        3. Net edge (after fees and slippage) must exceed the minimum threshold.
           Neutral-structure NO trades use MIN_NET_EDGE + NEUTRAL_STRUCTURE_EDGE_PREMIUM.

    If Gates 1 or 2 fail, Gate P is attempted before returning no_trade.
    If all three gates pass, Kelly sizing is called to determine bet amount.

    Args:
        structure_bias: +1 / -1 / 0 from detect_market_structure().
        confirmation_bias: +1 / -1 / 0 from compute_confirmation().
        p_model: Model's probability estimate for YES resolution.
        p_market: Kalshi market-implied probability.
        bankroll: Total capital available for sizing, in USD.
        slippage: Slippage fraction (default from pricing_comparison).
        spread: Bid-ask spread cost per side (default from pricing_comparison).
        min_net_edge: Minimum net edge required (default from pricing_comparison).
        confirmation_score: 3-indicator score for Gate P YES filter (-3 to +3).
        no_score: 3-indicator score for Gate P NO filter (-3 to +3).

    Returns:
        DecisionResult with all gate outcomes, sizing details, and reasons list.
    """
    reasons: List[str] = []

    if structure_bias == 1:
        side = "yes"
    elif structure_bias == -1:
        side = "no"
    else:  # neutral (0): let confirmation pick direction
        side = "yes" if confirmation_bias == 1 else "no"
    required_bias = +1 if side == "yes" else -1

    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # --- Gate 1: Market structure ---
    neutral_trade = (structure_bias == 0)
    gate1_passes  = (structure_bias == required_bias) or neutral_trade

    if not gate1_passes:
        reasons.append(
            f"Gate 1 FAILED: structure_bias={structure_bias} does not support "
            f"a {side.upper()} trade."
        )
        override = _pure_edge_override(p_model, p_market, bankroll, structure_bias,
                                       confirmation_bias, confirmation_score, no_score, reasons, slippage, spread)
        if override:
            return override
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )

    if neutral_trade:
        reasons.append(
            f"Gate 1 PASSED (neutral): structure_bias=0 — market is ranging. "
            f"{side.upper()} direction from confirmation_bias={confirmation_bias}. "
            f"Higher edge threshold (+{NEUTRAL_STRUCTURE_EDGE_PREMIUM:.2f}) applied at Gate 3."
        )
    else:
        reasons.append(
            f"Gate 1 PASSED: structure_bias={structure_bias} confirms "
            f"{side.upper()} trade direction."
        )

    # --- Gate 2: Confirmation indicators ---
    if confirmation_bias != required_bias:
        reasons.append(
            f"Gate 2 FAILED: confirmation_bias={confirmation_bias} does not align with "
            f"required bias {required_bias} for a {side.upper()} trade. "
            f"EMA/RSI/volume signals do not confirm this direction."
        )
        override = _pure_edge_override(p_model, p_market, bankroll, structure_bias,
                                       confirmation_bias, confirmation_score, no_score, reasons, slippage, spread)
        if override:
            return override
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
    effective_min_edge = min_net_edge + (NEUTRAL_STRUCTURE_EDGE_PREMIUM if neutral_trade else 0.0)
    p_edge, p_ref = (p_model, p_market) if side == "yes" else (p_market, p_model)
    pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=effective_min_edge)
    if not pricing.qualifies:
        reasons.append(f"Gate 3 FAILED: {pricing.reason}")
        override = _pure_edge_override(p_model, p_market, bankroll, structure_bias,
                                       confirmation_bias, confirmation_score, no_score, reasons, slippage, spread)
        if override:
            return override
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
