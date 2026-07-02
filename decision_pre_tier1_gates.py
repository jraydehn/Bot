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
    Flat 1% — matches Gate 3 floor. The composite p_model is calibrated so the
    Kelly criterion already accounts for R:R; escalating edge requirements were
    redundant and blocked profitable mid-R:R NO bets.
    """
    return 0.01


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
    asset: str = "BTC",
    composite_active: bool = False,
    composite_p_up: float = 0.504,
    offset_pct: float = 0.0,
    p_market_bid: float = None,
    p_market_ask: float = None,
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

    # --- Gate 0: Model saturation + market liquidity filter ---
    #
    # Two independent checks:
    #   (a) Model saturation: p_model outside bounds means the composite drift-adjusted
    #       log-normal is at a numerical extreme where a tiny vol or spot change creates
    #       large p_model swings — apparent edge is noise. BTC [0.04, 0.96], ETH/SOL wider.
    #
    #   (b) Market liquidity: p_market outside [0.04, 0.96] means Kalshi has no reliable
    #       quoted price — stale market makers, near-zero bid/ask, illiquid contract.
    #       Independent of what our model says; an illiquid contract can't be traded cleanly.
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
            f"composite model saturated at this strike."
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

    # Always evaluate both directions; return whichever qualifies with higher net edge.
    # Pass raw p_model so calibration is applied exactly once in the force_side path.
    # Side-aware p_market: YES bet cost = ask, NO bet cost = 1 - bid (so bid is the
    # reference YES price). Using mid (average) inflates edge for wide-spread contracts.
    if force_side is None:
        _pm_yes = p_market_ask if p_market_ask is not None else p_market
        _pm_no  = p_market_bid if p_market_bid is not None else p_market
        dec_yes = evaluate_trade(structure_bias, confirmation_bias, p_model, _pm_yes,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, no_bias, force_side="yes",
                                 obi_score=obi_score, vol_score=vol_score,
                                 ema_alignment=ema_alignment, asset=asset,
                                 composite_active=composite_active,
                                 composite_p_up=composite_p_up,
                                 offset_pct=offset_pct)
        dec_no  = evaluate_trade(structure_bias, confirmation_bias, p_model, _pm_no,
                                 bankroll, slippage, spread, min_net_edge,
                                 confirmation_score, no_score, no_bias, force_side="no",
                                 obi_score=obi_score, vol_score=vol_score,
                                 ema_alignment=ema_alignment, asset=asset,
                                 composite_active=composite_active,
                                 composite_p_up=composite_p_up,
                                 offset_pct=offset_pct)
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

    # --- BTC p_model calibration correction (applied once, in force_side path) ---
    # Calibration is side-specific:
    #
    # NO bets (0.65×): validated against 15-month backtest and live paper trade simulation.
    #   Blocked trades (72% win) vs pass trades (59% win) confirmed 0.65 is correct for NO.
    #
    # YES bets (0.90×): 0.65 was systematically blocking ALL YES trades because calibrated
    #   p_model always fell below p_market even for ITM contracts. Walk-forward backtest
    #   calib factor across all bets was 0.9831 (~1.0). 0.90 is a conservative correction
    #   for YES — allows ITM YES edge to surface while still discounting OTM YES slightly.
    #
    # ETH/SOL: NOT applied — their model is validated as-is.
    if asset == "BTC" and not composite_active:
        if side == "no":
            p_model = p_model * 0.65
        else:  # yes
            p_model = p_model * 0.90

    # Direction-aware edge:
    #   YES bet: we profit when market underprices YES → edge = p_model - p_market
    #   NO  bet: we profit when market overprices YES  → edge = p_market - p_model
    fee = kalshi_fee(p_market)
    raw_edge = p_model - p_market if side == "yes" else p_market - p_model
    net_edge = raw_edge - fee - slippage - spread

    # Gate EMA (neutral block) removed: Gate EMA-Dir + Gate 3 provide sufficient
    # quality filtering without hard-blocking neutral EMA regimes.
    reasons.append(f"Gate EMA PASSED: ema_alignment={ema_alignment}.")

    # --- Gate EMA-Dir: block bullish+YES only (BTC only) ---
    # 15-month backtest (11,000 bars): bullish+YES wins 8.1% (n=2,911) — continuation
    # bets at 0.5% OTM almost never succeed; BTC rarely accelerates enough to hit the
    # strike in 1 hour after an already-bullish EMA. Correctly blocked.
    #
    # bearish+NO block REMOVED: backtest of 5,930 bearish+NO bars shows 89.2% win rate.
    # The prior 37% estimate came from only 46 live paper trades — insufficient sample.
    # Removing this block roughly doubles BTC NO trade volume.
    #
    # ETH/SOL: gate not applied — different volatility regimes, models profitable as-is.
    # Gate EMA-Dir bullish+YES block removed: composite Gate CS now provides directional
    # filtering for OTM YES bets; EMA-Dir block was redundant and limited data collection.
    if asset == "BTC":
        reasons.append(
            f"Gate EMA-Dir PASSED: ema={ema_alignment}, side={side.upper()} — "
            f"EMA-Dir block removed; composite Gate CS handles OTM YES directional filtering."
        )

    # --- Gate PM: p_market range filter (BTC + ETH) ---
    #
    # BTC YES: p_market ≥ 0.55 (bearish+YES near/in-the-money only; OTM wins 23-33%)
    # BTC NO:  p_market ≤ 0.35 (lowered from 0.45; live analysis 2026-04-07 n=61 trades)
    #
    # ETH YES: p_market ≥ 0.35 (archive n=82: <0.35 → 1/8 wins -$92; ≥0.35 → 18/19 wins +$266)
    # ETH NO:  p_market ≤ 0.35 (archive: >0.35 NO trades net negative; <0.25 → 12/12 wins)
    #
    # SOL: Gate PM not applied — different volatility/liquidity regime; ITM NO filter handles
    #      the main failure case. Revisit when more live data available.
    #
    # Revert: copy decision_v4.py → decision.py
    P_MARKET_BTC_YES_MIN = 0.55
    P_MARKET_NO_MAX      = 0.35   # shared by BTC and ETH
    P_MARKET_ETH_YES_MIN = 0.35

    # YES filter
    yes_min = None
    if asset == "BTC":
        yes_min = P_MARKET_BTC_YES_MIN
    elif asset == "ETH":
        yes_min = P_MARKET_ETH_YES_MIN

    if not composite_active:
        if yes_min is not None and side == "yes" and p_market < yes_min:
            p_edge, p_ref = (p_model, p_market)
            pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
            reasons.append(
                f"Gate PM FAILED: p_market={p_market:.3f} < {yes_min} for {asset} YES — "
                f"OTM YES win rate is too low at this p_market level."
            )
            return DecisionResult(
                decision="no_trade", side=side,
                p_model=p_model, p_market=p_market,
                raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
                structure_bias=structure_bias, confirmation_bias=confirmation_bias,
                kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
                was_capped=False, reasons=reasons,
            )

        if asset in ("BTC", "ETH") and side == "no" and p_market > P_MARKET_NO_MAX:
            p_edge, p_ref = (p_market, p_model)
            pricing = evaluate_edge(p_edge, p_ref, slippage=slippage, spread=spread, min_net_edge=min_net_edge)
            reasons.append(
                f"Gate PM FAILED: p_market={p_market:.3f} > {P_MARKET_NO_MAX} for {asset} NO — "
                f"near-ATM/high p_market NO is net negative (live+archive analysis 2026-04-07)."
            )
            return DecisionResult(
                decision="no_trade", side=side,
                p_model=p_model, p_market=p_market,
                raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
                structure_bias=structure_bias, confirmation_bias=confirmation_bias,
                kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
                was_capped=False, reasons=reasons,
            )

        if asset in ("BTC", "ETH"):
            reasons.append(
                f"Gate PM PASSED: p_market={p_market:.3f} in valid range for {asset} {side.upper()}."
            )
    else:
        reasons.append(
            f"Gate PM SKIPPED: composite_active=True — p_market={p_market:.3f} range filter "
            f"replaced by composite signal + edge gate."
        )

    # --- Gate CS: composite_p_up ≥ 0.55 for YES bets when composite is active ---
    # When composite_active=True, Gate PM is skipped and the composite signal + edge
    # gate are the primary quality filters. However, a p_model > p_market gap can exist
    # even when the composite signal is nearly neutral (p_up barely above 0.504 baseline).
    # Without a directional conviction check, the system will take YES bets on far-OTM
    # contracts where the calibration says nothing meaningful.
    #
    # Threshold 0.55: calibration must show ≥55% historical up-move rate for this
    # (trend, rev) cell — a real directional lean, not noise around baseline.
    # Applies to YES only; NO bets are covered by Gate NS.
    # Gate CS applies only to OTM YES (offset_pct > 0, strike above spot): ITM YES
    # contracts (strike below spot) already have a natural buffer and pricing edge alone
    # is sufficient. OTM YES requires a move UP to win, so directional conviction is needed.
    COMPOSITE_YES_P_UP_MIN = 0.55
    if composite_active and side == "yes" and offset_pct > 0 and composite_p_up < COMPOSITE_YES_P_UP_MIN:
        fee = kalshi_fee(p_market)
        cs_raw_edge = p_model - p_market
        cs_net_edge = cs_raw_edge - fee - slippage - spread
        reasons.append(
            f"Gate CS FAILED: composite_p_up={composite_p_up:.3f} < {COMPOSITE_YES_P_UP_MIN} — "
            f"calibration shows insufficient directional lean for OTM YES (offset={offset_pct:+.3f}, strike above spot). "
            f"(p_model={p_model:.3f}, p_market={p_market:.3f}, net_edge={cs_net_edge:+.3f}). "
            f"Requires composite_p_up ≥ {COMPOSITE_YES_P_UP_MIN} for OTM YES bets."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=cs_raw_edge, net_edge=cs_net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    if composite_active and side == "yes":
        if offset_pct > 0:
            reasons.append(
                f"Gate CS PASSED: composite_p_up={composite_p_up:.3f} ≥ {COMPOSITE_YES_P_UP_MIN} — "
                f"calibration confirms directional lean for OTM YES (offset={offset_pct:+.3f})."
            )
        else:
            reasons.append(
                f"Gate CS SKIPPED: offset={offset_pct:+.3f} ≤ 0 — ITM YES (strike below spot), directional gate not required."
            )

    # --- Gate CI: ITM YES composite directional check ---
    # Hard block only: composite_p_up < 0.45 means composite is genuinely bearish —
    # do not bet ITM YES against a bearish signal.
    #
    # The former neutral zone (0.45–0.55 requiring 10% edge) was calibrated for the
    # old model where p_up was a standalone directional indicator separate from p_model.
    # In the composite drift model, p_up is already embedded in p_model via
    # score_to_p_model() — a neutral composite already produces near-zero edge
    # naturally, which Gate R:R and Gate 3 catch without needing this extra layer.
    # Applies to all assets (BTC, ETH, SOL).
    COMPOSITE_ITM_BEARISH_MAX = 0.45
    if composite_active and side == "yes" and offset_pct <= 0:
        if composite_p_up < COMPOSITE_ITM_BEARISH_MAX:
            reasons.append(
                f"Gate CI FAILED: ITM YES with bearish composite "
                f"(p_up={composite_p_up:.3f} < {COMPOSITE_ITM_BEARISH_MAX}) — "
                f"composite opposes YES direction. Hard block."
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
            f"Gate CI PASSED: ITM YES — composite_p_up={composite_p_up:.3f} ≥ {COMPOSITE_ITM_BEARISH_MAX} "
            f"(non-bearish). Gate R:R and Gate 3 apply."
        )

    # --- Gate NS: directional confirmation for NO bets ---
    # When composite_active=True: requires composite_p_up ≤ threshold — calibration
    # must show a genuine bearish lean. Applies to OTM NO only (offset_pct < 0).
    #
    # Thresholds (calibrated on 11k-hour test sets, Jan 2025–Apr 2026):
    #   BTC: 0.40 — tighter; calibrated from live trade data
    #   ETH: 0.45 — 0.40–0.45 bucket: 57.0% down, +7.9% edge ★★★ (n=816)
    #   SOL: 0.45 — 0.40–0.45 bucket: 58.4% down, +8.4% edge ★★★ (n=1,304)
    #
    # When composite_active=False (legacy path, BTC only): requires no_score ≥ 1.
    #   no_score=0 NO: 55.3% win — below break-even at p_market 0.20–0.35
    #   no_score=1 NO: 73.4% win — profitable at every p_market bin tested
    # BTC raised 0.40→0.50: p_up 0.40–0.50 = neutral/slightly bearish = NO bets directionally
    # aligned. Historical sim: 273 blocked trades at 77.7% WR (+$803). Threshold 0.40 was
    # cutting into aligned NO bets; neutral baseline is ~0.504 so 0.40–0.50 is bearish lean.
    # ETH raised 0.45→0.55: p_up 0.45–0.55 NO bets win at 79–82% WR (+$364). ETH NO bets
    # profitable regardless of composite direction; threshold 0.45 was over-blocking.
    # ETH lowered 0.55→0.50 (2026-04-28): full-stack joint replay (gate_attribution.py)
    # showed 0.50 peak at +$3,436 vs 0.55 +$3,142 (Δ=+$294, 2 fewer trades, drawdown
    # 20.0% vs 24.1%). Bucket-level +$364 was masking opportunity cost — joint PnL is
    # the test.
    # SOL reverted to 0.45 (2026-04-28): silently moved 0.45→0.55 on 04-27 when the
    # ETH-side else branch was widened. Joint replay strictly monotonic — every
    # loosening added losing trades: 0.45 +$10,696, 0.50 +$9,167, 0.55 +$7,749
    # (cost ~$2,947 on archive). Restored to original documented threshold.
    if asset == "SOL":
        COMPOSITE_NO_P_UP_MAX = 0.45
    else:  # BTC or ETH
        COMPOSITE_NO_P_UP_MAX = 0.50
    if asset in ("BTC", "ETH", "SOL") and side == "no":
        if composite_active:
            if offset_pct < 0 and composite_p_up > COMPOSITE_NO_P_UP_MAX:
                fee = kalshi_fee(p_market)
                ns_raw_edge = p_market - p_model
                ns_net_edge = ns_raw_edge - fee - slippage - spread
                reasons.append(
                    f"Gate NS FAILED: composite_p_up={composite_p_up:.3f} > {COMPOSITE_NO_P_UP_MAX} — "
                    f"calibration shows insufficient bearish lean for OTM NO (offset={offset_pct:+.3f}, strike below spot). "
                    f"(p_model={p_model:.3f}, p_market={p_market:.3f}, net_edge={ns_net_edge:+.3f}). "
                    f"Requires composite_p_up ≤ {COMPOSITE_NO_P_UP_MAX} for OTM NO bets."
                )
                return DecisionResult(
                    decision="no_trade", side=side,
                    p_model=p_model, p_market=p_market,
                    raw_edge=ns_raw_edge, net_edge=ns_net_edge,
                    structure_bias=structure_bias, confirmation_bias=confirmation_bias,
                    kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
                    was_capped=False, reasons=reasons,
                )
            if offset_pct < 0:
                reasons.append(
                    f"Gate NS PASSED: composite_p_up={composite_p_up:.3f} ≤ {COMPOSITE_NO_P_UP_MAX} — "
                    f"calibration confirms bearish lean for OTM NO (offset={offset_pct:+.3f})."
                )
            else:
                reasons.append(
                    f"Gate NS SKIPPED: offset={offset_pct:+.3f} ≥ 0 — ITM NO (strike above spot), directional gate not required."
                )
        elif asset == "BTC":
            if no_score < 1:
                p_edge_ns, p_ref_ns = (p_market, p_model)
                pricing_ns = evaluate_edge(p_edge_ns, p_ref_ns, slippage=slippage, spread=spread, min_net_edge=0.03)
                reasons.append(
                    f"Gate NS FAILED: no_score={no_score} < 1 for BTC NO — "
                    f"no_score=0 NO bets win only 55.3% (below break-even). "
                    f"Requires no_score ≥ 1 for directional confirmation."
                )
                return DecisionResult(
                    decision="no_trade", side=side,
                    p_model=p_model, p_market=p_market,
                    raw_edge=pricing_ns.raw_edge, net_edge=pricing_ns.net_edge,
                    structure_bias=structure_bias, confirmation_bias=confirmation_bias,
                    kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
                    was_capped=False, reasons=reasons,
                )
            reasons.append(f"Gate NS PASSED: no_score={no_score} ≥ 1 for BTC NO.")

    # --- Gate OTM: minimum edge scales with p_market for YES bets ---
    # Deep OTM YES contracts have large variance — a small edge is more likely to be
    # noise than genuine mispricing. Standard Gate 3 (1%) is too permissive here.
    # Tiers apply to all assets (composite or legacy path):
    #   p_market < 0.15 → net_edge ≥ 4%   (extreme OTM, payout ~12x)
    #   p_market < 0.25 → net_edge ≥ 3%   (very OTM, payout ~4x)
    #   p_market < 0.35 → net_edge ≥ 2%   (deep OTM, payout ~2x)
    #   p_market ≥ 0.35 → no extra floor  (Gate 3 handles it)
    if side == "yes":
        if p_market < 0.15:
            _otm_min = 0.04
        elif p_market < 0.25:
            _otm_min = 0.03
        elif p_market < 0.35:
            _otm_min = 0.02
        else:
            _otm_min = 0.0
        if _otm_min > 0:
            _otm_pricing = evaluate_edge(p_model, p_market, slippage=slippage, spread=spread, min_net_edge=_otm_min)
            if not _otm_pricing.qualifies:
                reasons.append(
                    f"Gate OTM FAILED: YES at p_market={p_market:.3f} requires net_edge ≥ {_otm_min:.0%} "
                    f"(deep-OTM variance tier), got {_otm_pricing.net_edge:+.4f}."
                )
                return DecisionResult(
                    decision="no_trade", side=side,
                    p_model=p_model, p_market=p_market,
                    raw_edge=_otm_pricing.raw_edge, net_edge=_otm_pricing.net_edge,
                    structure_bias=structure_bias, confirmation_bias=confirmation_bias,
                    kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
                    was_capped=False, reasons=reasons,
                )
            reasons.append(
                f"Gate OTM PASSED: YES at p_market={p_market:.3f} — "
                f"net_edge={_otm_pricing.net_edge:+.4f} ≥ {_otm_min:.0%} required."
            )

    # --- Gate 3: Minimum net edge threshold ---
    # ETH: 0.5% — lowered for data collection (0.5-1% bucket is net profitable).
    # BTC/SOL: 1% — 0.5-1% bucket is net negative for SOL; BTC unchanged.
    effective_min_edge = 0.005 if asset == "ETH" else 0.01
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
    #   YES: rr = p_market / (1 - p_market)
    #        Upper bound: rr > 3.0 blocks YES at p_market > 0.75 unconditionally.
    #        Archive data (75%+ YES): 8 trades, 62.5% WR, -$89 net even with avg 9% edge.
    #        R:R math: at pm=0.76, breakeven WR = 76%. Getting 62.5%. No edge exception —
    #        the data shows the exception was losing money.
    #
    #   NO:  rr = (1 - p_market) / p_market
    #        Lower bound: rr < 0.33 blocks near-ATM NO (pm > 0.75).
    #        Upper bound: rr > 4.0 blocks cheap NO (pm < 0.20).
    #        Exception: net_edge >= 0.08 overrides both bounds.
    RR_MAX_NO  = 4.0
    RR_MIN_NO  = 0.33
    RR_MAX_YES = 3.0
    RR_EDGE_EXCEPTION = 0.08
    rr = p_market / (1 - p_market) if side == "yes" else (1 - p_market) / p_market
    if side == "yes":
        rr_fail = rr > RR_MAX_YES  # no edge exception — archive shows it loses regardless
    else:
        rr_fail = (rr < RR_MIN_NO or rr > RR_MAX_NO) and pricing.net_edge < RR_EDGE_EXCEPTION
    if rr_fail:
        if side == "yes":
            bound = f"> {RR_MAX_YES} (p_market > 0.75 — poor R:R, blocked unconditionally)"
        else:
            bound = f"< {RR_MIN_NO} (near-ATM NO)" if rr < RR_MIN_NO else f"> {RR_MAX_NO} (cheap NO)"
        edge_clause = "" if side == "yes" else f" and net_edge={pricing.net_edge:+.4f} < {RR_EDGE_EXCEPTION:.0%} exception threshold"
        reasons.append(
            f"Gate R:R FAILED: R:R={rr:.2f} {bound} for {side.upper()} at p_market={p_market:.3f}{edge_clause}."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    # Apply minimum edge requirement scaled to R:R — high R:R bets need more edge
    rr_min_edge = _rr_min_edge(p_market, side)
    if pricing.net_edge < rr_min_edge:
        reasons.append(
            f"Gate R:R FAILED: net_edge={pricing.net_edge:+.4f} < {rr_min_edge:.0%} required"
            f" for R:R={rr:.2f} {side.upper()} at p_market={p_market:.3f}."
        )
        return DecisionResult(
            decision="no_trade", side=side,
            p_model=p_model, p_market=p_market,
            raw_edge=pricing.raw_edge, net_edge=pricing.net_edge,
            structure_bias=structure_bias, confirmation_bias=confirmation_bias,
            kelly_fraction=0.0, bet_fraction=0.0, bet_amount=0.0,
            was_capped=False, reasons=reasons,
        )
    reasons.append(f"Gate R:R PASSED: R:R={rr:.2f}  net_edge={pricing.net_edge:+.4f} >= {rr_min_edge:.0%} required  {side.upper()}.")

    # --- All gates passed: compute Kelly sizing ---
    # Multipliers calibrated from paper trade analysis with EMA-Dir gate applied:
    #   YES: half Kelly (0.50) — 63.5% win rate (ema=bearish + edge≥5%)
    #   NO:  half Kelly (0.50) — 63% win rate (ema=bullish + no_score≥1 + edge≥5%)
    #
    # ITM YES (p_market > 0.75): cap at 2% of bankroll.
    # These bets are profitable (84% WR, +$47 archive) but the payout ratio is poor
    # — at pm=0.84 you risk $16.80 to win $3.20. Capping at 2% keeps the trade but
    # limits single-trade exposure to ~$8 at a $400 bankroll.
    kelly_multiplier = 0.50
    _max_bet = 0.02 if (side == "yes" and p_market > 0.75) else 0.05
    kelly = compute_kelly_size(p_model, p_market, bankroll, kelly_multiplier, side=side,
                               max_bet_fraction=_max_bet)
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
