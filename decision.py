"""
Decision module: evaluates structure, confirmation, and edge gates in sequence,
then calls Kelly sizing to produce a final trade decision with full audit trail.
"""

from dataclasses import dataclass, field
from typing import List

from pricing_comparison import evaluate_edge, kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, MIN_NET_EDGE
from kelly_sizing import compute_kelly_size

# Gate 3 tiered minimum net edge thresholds.
# Separate tiers for YES (confirmation_score) and NO (no_score).
# Gate P thresholds are intentionally higher — Gate P fires without confirmation.
#
# YES tiers (confirmation_score):
#   score>=4 → 1%,  score>=3 → 2%,  score>=2 → 4%
YES_NEUTRAL_GATE3_TIERS  = [(4, 0.01), (3, 0.02), (2, 0.04)]
#
# NO tiers (no_score, Gate 2 requires no_score<=0):
#   no_score<=-3 → 1%,  no_score<=-2 → 2%,  no_score<=0 → 4%
NO_NEUTRAL_GATE3_TIERS  = [(-3, 0.01), (-2, 0.02), (0, 0.04)]

# When structure is neutral and the trade direction opposes confirmation, a flat
# minimum net edge is required.
NEUTRAL_CONTRARIAN_MIN_EDGE = 0.10

# Gate 2S: strong-opposition block threshold.
# When confirmation_score is this strongly opposed to the trade direction,
# block the trade even in dual-direction (conflict) evaluation mode.
# YES trade blocked if confirmation_score <= -STRONG_OPPOSITION_THRESHOLD
# NO  trade blocked if confirmation_score >= +STRONG_OPPOSITION_THRESHOLD
STRONG_OPPOSITION_THRESHOLD = 3


def _rr_min_edge(p_market: float, side: str) -> float:
    """
    Minimum net edge required based on risk-to-reward ratio.

    YES bets: R:R = p_market / (1 - p_market)  — risk per dollar of potential reward
    NO  bets: R:R = (1 - p_market) / p_market  — NO bets on high-p_market contracts
                                                   are naturally favorable (low R:R)

    Tiers:
        R:R <= 1  → 3%    (even or better payout)
        R:R <= 2  → 6%
        R:R <= 4  → 9%
        R:R <= 6  → 15%
        R:R <= 8  → 20%
        R:R  > 8  → 25%   (universal floor)
    """
    if side == "yes":
        rr = p_market / (1 - p_market) if (1 - p_market) > 0 else float("inf")
    else:
        rr = (1 - p_market) / p_market if p_market > 0 else float("inf")

    if rr <= 1: return 0.03
    if rr <= 2: return 0.06
    if rr <= 4: return 0.09
    if rr <= 6: return 0.15
    if rr <= 8: return 0.20
    return 0.25


def _yes_gate3_threshold(score: int) -> float:
    for min_score, threshold in YES_NEUTRAL_GATE3_TIERS:
        if score >= min_score:
            return threshold
    return float("inf")


def _no_gate3_threshold(score: int) -> float:
    for max_score, threshold in NO_NEUTRAL_GATE3_TIERS:
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

    # Tiered edge requirements — each direction has its own model and sliding threshold.
    #
    # YES (5-indicator model, momentum paused, confirmation_score -5 to +5):
    #   score >= 4 (4 of 5 indicators bullish) → edge >= 6%
    #   score >= 3 (majority bullish)          → edge >= 8%
    #   score <  3 (weak/negative confirmation) → edge >= 25%  (universal floor)
    #
    # NO (6-indicator model, no_score -5 to +5):
    #   no_score <= -3 (majority bearish)           → edge >= 6%
    #   no_score <=  0 (neutral/mixed)              → edge >= 10%
    #   no_score <=  2 (slightly bullish)           → edge >= 12%
    #   no_score <=  4 (mostly bullish)             → edge >= 15%
    #   no_score  =  5 (all bullish)                → edge >= 25%  (universal floor)
    if confirmation_score >= 4:
        yes_min = 0.05
    elif confirmation_score >= 3:
        yes_min = 0.06
    else:
        yes_min = 0.15  # universal floor — any score qualifies at 15%+ net edge

    if no_score <= -3:
        no_min = 0.05
    elif no_score <= 0:
        no_min = 0.08
    elif no_score <= 1:
        no_min = 0.10
    elif no_score <= 2:
        no_min = 0.10
    elif no_score <= 4:
        no_min = 0.12
    else:
        no_min = 0.20  # universal floor — any score qualifies at 20%+ net edge

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
    obi_score: int = 0,
    vol_score: int = 0,
    ema_alignment: str = "neutral",
) -> DecisionResult:
    """
    Evaluate all gates in sequence and produce a final trade decision.

    Gates are evaluated in this order — if any gate fails, the function
    returns immediately with decision="no_trade":
        1. Confirmation indicators bias must align with the proposed trade direction.
        2. Net edge (after fees and slippage) must exceed the tiered minimum threshold.
           Thresholds scale with confirmation strength (confirmation_score for YES,
           no_score for NO).

    Both YES and NO directions are always evaluated; whichever passes with the
    higher net edge is returned.  structure_bias is recorded but does not gate trades.

    If all gates pass, Kelly sizing is called to determine bet amount.

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

    # --- Gate 0: Model saturation filter ---
    # p_model outside [0.05, 0.95] means the strike is so far ITM or OTM that the
    # log-normal model hits numerical saturation. Any apparent edge vs p_market at
    # these extremes is noise from stale or illiquid market quotes, not real mispricing.
    P_MODEL_MIN, P_MODEL_MAX = 0.04, 0.96
    if not (P_MODEL_MIN <= p_model <= P_MODEL_MAX):
        fee = kalshi_fee(p_market)
        raw_edge = p_model - p_market if (force_side or "yes") == "yes" else p_market - p_model
        net_edge = raw_edge - fee - slippage - spread
        reasons.append(
            f"Gate 0 FAILED: p_model={p_model:.4f} outside [{P_MODEL_MIN}, {P_MODEL_MAX}] — "
            f"strike is too deep ITM or OTM, model is saturated."
        )
        return DecisionResult(
            decision="no_trade", side=force_side or ("yes" if p_model > 0.5 else "no"),
            p_model=p_model, p_market=p_market,
            raw_edge=raw_edge, net_edge=net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )

    # Always evaluate both directions; return whichever qualifies with higher net edge.
    if force_side is None:
        dec_yes = evaluate_trade(structure_bias, confirmation_bias, p_model, p_market,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, no_bias, force_side="yes",
                                 obi_score=obi_score, vol_score=vol_score,
                                 ema_alignment=ema_alignment)
        dec_no  = evaluate_trade(structure_bias, confirmation_bias, p_model, p_market,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, no_bias, force_side="no",
                                 obi_score=obi_score, vol_score=vol_score,
                                 ema_alignment=ema_alignment)
        if dec_yes.decision == "trade" and dec_no.decision == "trade":
            return dec_yes if dec_yes.net_edge >= dec_no.net_edge else dec_no
        elif dec_yes.decision == "trade":
            return dec_yes
        elif dec_no.decision == "trade":
            return dec_no
        else:
            return dec_yes if dec_yes.net_edge >= dec_no.net_edge else dec_no

    # force_side path: evaluate a specific direction (called from the block above).
    side = force_side
    required_bias = +1 if side == "yes" else -1

    # Direction-aware edge:
    #   YES bet: we profit when market underprices YES → edge = p_model - p_market
    #   NO  bet: we profit when market overprices YES  → edge = p_market - p_model
    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # --- Gate EMA: block neutral EMA alignment ---
    # Data analysis: ema=neutral produces 16.7% win rate (-$1,079 on 72 trades).
    # Structure=+1 + ema=neutral wins 0% across 21 trades. Neutral EMA means price
    # is trapped between the 20 and 50 EMA — no directional conviction, no edge.
    if ema_alignment == "neutral":
        reasons.append(
            f"Gate EMA FAILED: ema_alignment=neutral — price between EMAs, "
            f"no directional conviction. 16.7% historical win rate."
        )
        p_edge, p_ref = (p_model, p_market) if side == "yes" else (p_market, p_model)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate EMA PASSED: ema_alignment={ema_alignment}.")

    # --- Gate EMA-Dir: EMA alignment must oppose the trade direction ---
    # Paper trade analysis (n=235): ema=bullish+YES wins 0% (n=19); ema=bearish+YES wins 63.5% (n=52).
    # ema=bullish+NO wins 63% (n=46); ema=bearish+NO wins only 37% (n=46, below break-even).
    #
    # Mechanism: 1-hour Kalshi contracts reward contrarian positioning. After bearish EMA
    # (price below both MAs, downtrend established), a bounce of 0.5% to reach the strike
    # is statistically likely — the market underprices the reversal. After bullish EMA,
    # the strike 0.5% above already-elevated spot requires continued acceleration, which
    # rarely sustains for a full hour — the market correctly prices continuation risk.
    #
    # Rule: ema=bearish → only YES (fade the downtrend, catch the bounce).
    #       ema=bullish → only NO (fade the uptrend, market already priced in continuation).
    if ema_alignment == "bullish" and side == "yes":
        reasons.append(
            f"Gate EMA-Dir FAILED: ema=bullish+YES — continuation bet, 0% historical win "
            f"rate (n=19). 1h contracts do not sustain momentum; YES only valid after "
            f"bearish EMA (contrarian bounce)."
        )
        p_edge, p_ref = (p_model, p_market)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    if ema_alignment == "bearish" and side == "no":
        reasons.append(
            f"Gate EMA-Dir FAILED: ema=bearish+NO — downtrend continuation bet, 37% win "
            f"rate (n=46, below break-even). Market already prices further decline; "
            f"NO only valid after bullish EMA (contrarian fade)."
        )
        p_edge, p_ref = (p_market, p_model)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(
        f"Gate EMA-Dir PASSED: ema={ema_alignment} opposes {side.upper()} direction (contrarian)."
    )

    # --- Gate PM: p_market must be in the directionally valid range ---
    # Paper trade analysis cross-tabbed by EMA regime and side:
    #
    # YES (ema=bearish):
    #   p_market ≥ 0.55 → 96.4% win rate (n=28)  ← the only YES regime with real edge
    #   p_market 0.45–0.55 → 33.3% win rate (n=3, coin flip, no edge)
    #   p_market < 0.45 → 23.5% win rate (n=21)   ← log-normal systematically overestimates OTM
    #
    # NO (ema=bullish):
    #   p_market ≤ 0.45 → 78–100% win rate (n=32) ← market overprices YES in a declining trend
    #   p_market 0.45–0.55 → 33.3% win rate (n=9, coin flip, no edge)
    #   p_market > 0.55 → 20% win rate (n=5)       ← NO on near-ITM contracts is dangerous
    #
    # Mechanism: When YES wins at high p_market, the strike is near/in-the-money — the contract
    # is fairly priced near certainty and a small model edge is real. When YES is cheap
    # (p_market < 0.55), the model's log-normal estimate inflates OTM probabilities; the
    # market is correct that the 0.5% move rarely materialises in an uptrend.
    P_MARKET_YES_MIN = 0.55
    P_MARKET_NO_MAX  = 0.45
    if side == "yes" and p_market < P_MARKET_YES_MIN:
        p_edge, p_ref = (p_model, p_market)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        reasons.append(
            f"Gate PM FAILED: p_market={p_market:.3f} < {P_MARKET_YES_MIN} for YES — "
            f"OTM YES bets in bearish EMA win only 24% (n=21). Log-normal "
            f"overestimates OTM probability; YES valid only near/in-the-money."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    if side == "no" and p_market > P_MARKET_NO_MAX:
        p_edge, p_ref = (p_market, p_model)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        reasons.append(
            f"Gate PM FAILED: p_market={p_market:.3f} > {P_MARKET_NO_MAX} for NO — "
            f"near-ATM NO in bullish EMA wins only 33% (n=9) and ITM NO wins 20% (n=5). "
            f"NO valid only when YES contract is cheap (p_market ≤ {P_MARKET_NO_MAX})."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(
        f"Gate PM PASSED: p_market={p_market:.3f} in valid range for {side.upper()} "
        f"({'≥' if side == 'yes' else '≤'}{P_MARKET_YES_MIN if side == 'yes' else P_MARKET_NO_MAX})."
    )

    # --- Gate NO quality: NO trades require no_score >= 1 ---
    # Data analysis: NO trades with no_score=0 win 39.4% and are consistently
    # unprofitable. no_score=1 (at least one bearish indicator firing) produces
    # 46.9% win rate with +$219 PnL. Requires at least one signal supporting NO.
    if side == "no" and no_score < 1:
        reasons.append(
            f"Gate NO FAILED: no_score={no_score} < 1 — insufficient bearish signal "
            f"confirmation for NO trade. no_score=0 historically 39.4% win rate."
        )
        p_edge, p_ref = (p_market, p_model)
        pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    if side == "no":
        reasons.append(f"Gate NO PASSED: no_score={no_score} >= 1.")

    # --- Gate 3: Minimum net edge threshold = 5% ---
    # Data analysis: trades with net_edge < 0.05 collectively lose -$2,136 at ~30% win.
    # net_edge > 0.05 produces 71.4% win rate (+$435 on 49 trades). The model's
    # edge calculation is accurate when it fires clearly — trust it.
    effective_min_edge = 0.05
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

    # --- Gate R:R: filter by risk-to-reward ratio ---
    # Formula:
    #   YES: rr = p_market / (1 - p_market)  — increases with p_market
    #   NO:  rr = (1 - p_market) / p_market  — decreases with p_market
    #
    # Lower bound (both sides): rr < 0.33 blocks deep-underdog YES (pm < 0.25,
    #   historically 12% win) and deep-favorite NO (pm > 0.75, already mostly
    #   filtered by offset gate).
    #
    # Upper bound (NO only): rr > 4.0 blocks cheap NO bets (pm < 0.20) where
    #   payout is tiny relative to risk — historically negative PnL despite 79%
    #   win rate due to outsized losses. Upper bound NOT applied to YES because
    #   high-p_market YES bets (rr > 4 at pm > 0.80) win 96% historically.
    RR_MIN = 0.33
    RR_MAX_NO = 4.0
    rr = p_market / (1 - p_market) if side == "yes" else (1 - p_market) / p_market
    rr_fail = rr < RR_MIN or (side == "no" and rr > RR_MAX_NO)
    if rr_fail:
        bound = f"< {RR_MIN}" if rr < RR_MIN else f"> {RR_MAX_NO} (NO only)"
        reasons.append(
            f"Gate R:R FAILED: R:R={rr:.2f} {bound} for {side.upper()} at p_market={p_market:.3f}."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate R:R PASSED: R:R={rr:.2f} for {side.upper()}.")

    # --- All gates passed: compute Kelly sizing ---
    # Multipliers calibrated from paper trade analysis with EMA-Dir + PM gates applied:
    #   YES: half Kelly (0.50) — 96.4% win rate (ema=bearish + p_market≥0.55 + edge≥5%)
    #        Full Kelly ~0.91 at these win rates; capped at half for model uncertainty.
    #   NO:  half Kelly (0.50) — 78% win rate (ema=bullish + p_market≤0.45 + no_score≥1)
    #        Raised from 0.25 → 0.50 as the qualifying NO regime improved from 50% to 78%.
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
