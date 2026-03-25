"""
Decision module: evaluates structure, confirmation, and edge gates in sequence,
then calls Kelly sizing to produce a final trade decision with full audit trail.
"""

from dataclasses import dataclass, field
from typing import List

from pricing_comparison import evaluate_edge, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from kelly_sizing import compute_kelly_size

# Gate 3 tiered minimum net edge thresholds.
# Both YES and NO use the same 3-indicator score (-3 to +3).
# Neutral structure adds a premium to each tier.
#
# YES tiers (confirmation_score, range 2–3 when Gate 2 passes):
#   Confirmed structure (+1):   score>=3 → 2%,  score>=2 → 4%
#   Neutral structure (0):      score>=3 → 4%,  score>=2 → 6%
YES_GATE3_TIERS = [(3, 0.02), (2, 0.04)]
YES_NEUTRAL_GATE3_TIERS = [(3, 0.04), (2, 0.06)]
#
# NO tiers (no_score, range -3 to 0 when Gate 2 passes — Gate 2 requires no_score<=0):
#   Confirmed structure (-1):   no_score<=-3 → 2%,  no_score<=-2 → 4%,  no_score<=0 → 6%
#   Neutral structure (0):      no_score<=-3 → 4%,  no_score<=-2 → 6%,  no_score<=0 → 10%
NO_GATE3_TIERS = [(-3, 0.02), (-2, 0.04), (0, 0.06)]
NO_NEUTRAL_GATE3_TIERS = [(-3, 0.04), (-2, 0.06), (0, 0.10)]


def _yes_gate3_threshold(score: int, neutral: bool) -> float:
    tiers = YES_NEUTRAL_GATE3_TIERS if neutral else YES_GATE3_TIERS
    for min_score, threshold in tiers:
        if score >= min_score:
            return threshold
    return float("inf")


def _no_gate3_threshold(score: int, neutral: bool) -> float:
    tiers = NO_NEUTRAL_GATE3_TIERS if neutral else NO_GATE3_TIERS
    for max_score, threshold in tiers:
        if score <= max_score:
            return threshold
    return float("inf")

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
    Gate P: fire when either direction has a net edge >= PURE_EDGE_MIN_NET_EDGE,
    bypassing structure and confirmation gates. Returns a DecisionResult trade if
    the threshold is met, else None (caller should return no_trade).
    """
    fee = kalshi_fee(p_market)
    yes_net = (p_model - p_market) - fee - slippage - spread
    no_net  = (p_market - p_model) - fee - slippage - spread

    # Tiered edge requirements — both directions use the same 3-indicator score (-3 to +3).
    #
    # YES (confirmation_score -3 to +3):
    #   score >= 3 (all 3 bullish)       → edge >= 6%
    #   score >= 2 (2 of 3 bullish)      → edge >= 8%
    #   score >= 1 (1 of 3 bullish)      → edge >= 10%
    #   score <= 0 (neutral/bearish)     → edge >= 25%  (universal floor)
    #
    # NO (no_score -3 to +3):
    #   no_score <= -2 (majority bearish) → edge >= 6%
    #   no_score <=  0 (neutral/mixed)   → edge >= 8%
    #   no_score <=  1 (slightly bullish) → edge >= 10%
    #   no_score <=  2 (mostly bullish)  → edge >= 15%
    #   no_score  =  3 (strongly bullish) → edge >= 25%  (universal floor)
    if confirmation_score >= 3:
        yes_min = 0.06
    elif confirmation_score >= 2:
        yes_min = PURE_EDGE_MIN_NET_EDGE  # 8%
    elif confirmation_score >= 1:
        yes_min = 0.10
    else:
        yes_min = 0.25  # universal floor

    if no_score <= -2:
        no_min = 0.06
    elif no_score <= 0:
        no_min = PURE_EDGE_MIN_NET_EDGE   # 8%
    elif no_score <= 1:
        no_min = 0.10
    elif no_score <= 2:
        no_min = 0.15
    else:
        no_min = 0.25  # universal floor — any score qualifies at 25%+ net edge

    yes_ok = yes_net >= yes_min
    no_ok  = no_net  >= no_min

    if yes_ok and no_ok:
        # Both qualify — take whichever has the higher net edge (NO wins ties)
        if no_net >= yes_net:
            pure_side, pure_net, pure_raw = "no",  no_net,  p_market - p_model
        else:
            pure_side, pure_net, pure_raw = "yes", yes_net, p_model - p_market
    elif yes_ok:
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
    no_bias: int = 0,
    force_side: str = None,
) -> DecisionResult:
    """
    Evaluate all gates in sequence and produce a final trade decision.

    Gates are evaluated in this order — if any gate fails, the function
    returns immediately with decision="no_trade":
        1. Market structure bias must support the proposed trade direction.
           - YES trades: structure_bias must be +1 (bullish). Neutral (0) blocks YES
                         via the normal gate path — confirmed structure is required
                         because lagging indicators (EMA/RSI) reflect prior momentum,
                         not immediate direction in a ranging market.
                         Neutral-structure YES can still fire through Gate P if the
                         edge is large enough relative to confirmation_score.
           - NO trades:  structure_bias must be -1 (bearish, standard threshold)
                         OR 0 (neutral, higher edge threshold applied at Gate 3).
                         structure_bias = +1 always blocks a NO trade.
        2. Confirmation indicators bias must align with the proposed trade direction.
        3. Net edge (after fees and slippage) must exceed the tiered minimum threshold.
           Thresholds scale with confirmation strength (confirmation_score for YES,
           no_score for NO). Neutral structure adds a premium to each tier.

    If all three gates pass, Kelly sizing is called to determine bet amount.

    Trade side is determined by the gate signals, not by comparing p_model to
    p_market. structure_bias = +1 → YES trade; -1 or 0 → NO trade considered.
    Neutral structure (0) allows both directions but applies higher edge premiums
    at Gate 3, especially for YES (higher neutral tier premium than NO).

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

    # Side determination:
    #   force_side overrides structure when provided (used to evaluate both directions).
    #   Otherwise structure_bias drives direction; neutral defers to confirmation_bias.
    if force_side is not None:
        side = force_side
    elif structure_bias == 1:
        side = "yes"
    elif structure_bias == -1:
        side = "no"
    else:
        side = "yes" if confirmation_bias == 1 else "no"
    required_bias = +1 if side == "yes" else -1

    # Direction-aware edge:
    #   YES bet: we profit when market underprices YES → edge = p_model - p_market
    #   NO  bet: we profit when market overprices YES  → edge = p_market - p_model
    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # --- Gate 1: Market structure ---
    # Structure confirming direction → confirmed thresholds at Gate 3.
    # Structure neutral or opposing → neutral (higher) thresholds at Gate 3.
    # Structure never hard-blocks a direction — it only affects required edge.
    structure_confirms = (structure_bias == required_bias)
    structure_opposes  = (structure_bias != 0 and structure_bias != required_bias)
    neutral_trade      = not structure_confirms  # neutral OR opposing → higher thresholds

    if structure_confirms:
        reasons.append(
            f"Gate 1 PASSED: structure_bias={structure_bias} confirms {side.upper()} trade."
        )
    elif structure_opposes:
        reasons.append(
            f"Gate 1 PASSED (against structure): structure_bias={structure_bias} opposes "
            f"{side.upper()} trade — neutral/higher thresholds applied at Gate 3."
        )
    else:
        reasons.append(
            f"Gate 1 PASSED (neutral): structure_bias=0 — market is ranging. "
            f"{side.upper()} direction from {'force_side' if force_side else 'confirmation_bias'}. "
            f"Tiered edge threshold applied at Gate 3 (neutral premium)."
        )

    # --- Gate 2: Confirmation indicators ---
    # Both YES and NO use the same 3-indicator score (EMA + RSI + directional vol).
    # YES: confirmation_bias == +1 required (score >= 2).
    # NO:  no_score <= 0 passes (neutral or bearish); > 0 blocked (net bullish signals).
    if side == "no":
        gate2_passes = no_score <= 0
        bias_label = f"no_score={no_score:+d}"
    else:
        gate2_passes = confirmation_bias == required_bias
        bias_label = f"confirmation_bias={confirmation_bias}"

    if not gate2_passes:
        reasons.append(
            f"Gate 2 FAILED: {bias_label} does not align with "
            f"required bias for a {side.upper()} trade. "
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
        f"Gate 2 PASSED: {bias_label} aligns with {side.upper()} trade."
    )

    # --- Gate 3: Tiered net edge threshold ---
    # Minimum edge required scales with confirmation strength:
    #   YES: tiered by confirmation_score (7-indicator, range 2–9 at this point)
    #   NO:  tiered by no_score (3-indicator, range -3 to -2 at this point)
    # Neutral structure adds a premium to each tier.
    if side == "yes":
        effective_min_edge = _yes_gate3_threshold(confirmation_score, neutral_trade)
        tier_label = f"YES score={confirmation_score:+d} ({'neutral' if neutral_trade else 'confirmed'} structure)"
    else:
        effective_min_edge = _no_gate3_threshold(no_score, neutral_trade)
        tier_label = f"NO no_score={no_score:+d} ({'neutral' if neutral_trade else 'confirmed'} structure)"

    p_edge, p_ref = (p_model, p_market) if side == "yes" else (p_market, p_model)
    pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=effective_min_edge)
    if not pricing.qualifies:
        reasons.append(f"Gate 3 FAILED [{tier_label}, min={effective_min_edge:.0%}]: {pricing.reason}")
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
    reasons.append(f"Gate 3 PASSED [{tier_label}, min={effective_min_edge:.0%}]: {pricing.reason}")

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
