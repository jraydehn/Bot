"""
Decision module: evaluates edge gates in sequence, then calls Kelly sizing
to produce a final trade decision with full audit trail.

Active gates (in order):
    0. Gate 0  — model saturation + market liquidity filter
    1. Gate PM — p_market range filter (BTC/ETH asset-specific)
    2. Gate 3  — minimum net edge (1% floor)
    3. Gate R:R — risk-to-reward ratio bounds
"""

from dataclasses import dataclass, field
from typing import List

from pricing_comparison import evaluate_edge, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from kelly_sizing import compute_kelly_size

# Gate P: pure-edge override threshold and Kelly multiplier.
PURE_EDGE_KELLY_MULTIPLIER = 0.125


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
    force_side: str = None,
    obi_score: int = 0,
    vol_score: int = 0,
    asset: str = "BTC",
    offset_pct: float = 0.0,
) -> DecisionResult:
    """
    Evaluate all gates in sequence and produce a final trade decision.

    Both YES and NO directions are always evaluated; whichever passes with the
    higher net edge is returned.

    Args:
        structure_bias    : +1 / -1 / 0 from detect_market_structure() (logged, not gating)
        confirmation_bias : +1 / -1 / 0 from compute_confirmation() (logged, not gating)
        p_model           : Vol-adjusted probability estimate for YES resolution.
        p_market          : Kalshi market-implied probability.
        bankroll          : Total capital available for sizing, in USD.
        force_side        : "yes" or "no" to evaluate only one direction; None evaluates both.
        asset             : "BTC", "ETH", or "SOL" — used for asset-specific gate thresholds.

    Returns:
        DecisionResult with all gate outcomes, sizing details, and reasons list.
    """
    reasons: List[str] = []

    # --- Gate 0: Model saturation + market liquidity filter ---
    P_MODEL_MIN  = 0.04 if asset == "BTC" else 0.02
    P_MODEL_MAX  = 0.96 if asset == "BTC" else 0.98
    P_MARKET_MIN = 0.04
    P_MARKET_MAX = 0.96
    fee      = kalshi_fee(p_market)
    raw_edge = p_model - p_market if (force_side or "yes") == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread
    if not (P_MODEL_MIN <= p_model <= P_MODEL_MAX):
        reasons.append(
            f"Gate 0 FAILED (saturation): p_model={p_model:.4f} outside [{P_MODEL_MIN}, {P_MODEL_MAX}] ({asset}) — "
            f"model saturated at this strike."
        )
        return DecisionResult(
            decision="no_trade", side=force_side or ("yes" if p_model > 0.5 else "no"),
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    if not (P_MARKET_MIN <= p_market <= P_MARKET_MAX):
        reasons.append(
            f"Gate 0 FAILED (liquidity): p_market={p_market:.4f} outside [{P_MARKET_MIN}, {P_MARKET_MAX}] — "
            f"Kalshi contract is illiquid or stale-quoted."
        )
        return DecisionResult(
            decision="no_trade", side=force_side or ("yes" if p_model > 0.5 else "no"),
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )

    # Evaluate both directions; return whichever qualifies with higher net edge.
    if force_side is None:
        dec_yes = evaluate_trade(structure_bias, confirmation_bias, p_model, p_market,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, force_side="yes",
                                 obi_score=obi_score, vol_score=vol_score,
                                 asset=asset, offset_pct=offset_pct)
        dec_no  = evaluate_trade(structure_bias, confirmation_bias, p_model, p_market,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, force_side="no",
                                 obi_score=obi_score, vol_score=vol_score,
                                 asset=asset, offset_pct=offset_pct)
        if dec_yes.decision == "trade" and dec_no.decision == "trade":
            return dec_yes if dec_yes.net_edge >= dec_no.net_edge else dec_no
        elif dec_yes.decision == "trade":
            return dec_yes
        elif dec_no.decision == "trade":
            return dec_no
        else:
            return dec_yes if dec_yes.net_edge >= dec_no.net_edge else dec_no

    # force_side path: evaluate a specific direction.
    side = force_side

    # Direction-aware edge:
    #   YES bet: profit when market underprices YES → edge = p_model - p_market
    #   NO  bet: profit when market overprices YES  → edge = p_market - p_model
    fee      = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # Gate PM: TEMPORARILY REMOVED — thresholds were calibrated on old composite scorer
    # model data and have not been validated against the current vol+direction model.
    # Will be re-evaluated and recalibrated once 50-100+ trades are collected under
    # the new model. Gate 3 (net edge) and Gate R:R provide the primary quality filters.

    # --- Gate 3: Minimum net edge (1% floor) ---
    effective_min_edge = 0.01
    p_edge, p_ref = (p_model, p_market) if side == "yes" else (p_market, p_model)
    pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=effective_min_edge)
    if not pricing.qualifies:
        reasons.append(f"Gate 3 FAILED [min={effective_min_edge:.0%}]: {pricing.reason}")
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate 3 PASSED [min={effective_min_edge:.0%}]: {pricing.reason}")

    # --- Gate R:R: risk-to-reward ratio bounds ---
    RR_MAX_NO  = 4.0
    RR_MIN_NO  = 0.33
    RR_MAX_YES = 3.0
    RR_EDGE_EXCEPTION = 0.08
    rr = p_market / (1 - p_market) if side == "yes" else (1 - p_market) / p_market
    if side == "yes":
        rr_fail = rr > RR_MAX_YES and pricing.net_edge < RR_EDGE_EXCEPTION
    else:
        rr_fail = (rr < RR_MIN_NO or rr > RR_MAX_NO) and pricing.net_edge < RR_EDGE_EXCEPTION
    if rr_fail:
        if side == "yes":
            bound = f"> {RR_MAX_YES} (high p_market YES)"
        else:
            bound = f"< {RR_MIN_NO} (near-ATM NO)" if rr < RR_MIN_NO else f"> {RR_MAX_NO} (cheap NO)"
        reasons.append(
            f"Gate R:R FAILED: R:R={rr:.2f} {bound} for {side.upper()} at p_market={p_market:.3f} "
            f"and net_edge={pricing.net_edge:+.4f} < {RR_EDGE_EXCEPTION:.0%} exception threshold."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate R:R PASSED: R:R={rr:.2f}  net_edge={pricing.net_edge:+.4f} >= 1% required  {side.upper()}.")

    # --- All gates passed: compute Kelly sizing ---
    kelly_multiplier = 0.50
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
