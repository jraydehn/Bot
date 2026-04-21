"""
Kelly sizing module for calculating the optimal bet size on a Kalshi binary
event contract using the Kelly criterion.

All inputs are in YES-space (p_model = model's p_yes, p_market = Kalshi YES price).
The side parameter controls which payout structure and formula are applied:

    YES bet: you pay p_market, win (1 - p_market) if YES resolves.
        b = (1 - p_market) / p_market
        f = (p_yes_model * b - p_no_model) / b

    NO  bet: you pay (1 - p_market), win p_market if NO resolves.
        b = p_market / (1 - p_market)
        f = (p_no_model * b - p_yes_model) / b

A kelly_multiplier scales the raw full-Kelly fraction before the hard cap:
    YES bets: 0.25 (quarter Kelly) — low win rate, high variance
    NO  bets: 0.50 (half Kelly)   — high win rate, low variance

kelly_fraction always reflects the raw full-Kelly output. bet_fraction
reflects what is actually bet after the multiplier and cap are applied.
"""

from dataclasses import dataclass


MAX_BET_FRACTION = 0.05  # hard cap: never risk more than 5% of bankroll on one bet


@dataclass
class KellyResult:
    """Output of the Kelly sizing module."""

    kelly_fraction: float   # raw Kelly output before the 5% cap
    bet_fraction: float     # fraction to bet after applying the cap
    bet_amount: float       # dollar amount to bet, rounded to 2 decimal places
    bankroll: float         # total capital provided
    was_capped: bool        # True if the raw Kelly fraction exceeded the 5% cap
    reason: str             # plain-English explanation of the sizing decision


def compute_kelly_size(
    p_model: float,
    p_market: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    side: str = "yes",
    max_bet_fraction: float = MAX_BET_FRACTION,
) -> KellyResult:
    """
    Compute optimal bet size using the Kelly criterion for a binary event contract.

    Both p_model and p_market are always expressed as YES probabilities, regardless
    of trade direction. The side parameter selects the correct payout structure:

        YES bet: pay p_market per contract, win (1 - p_market) if YES resolves.
        NO  bet: pay (1 - p_market) per contract, win p_market if NO resolves.

    Args:
        p_model: Model's estimated probability of YES resolution (0–1).
        p_market: Kalshi YES market price (0–1). Must be strictly between 0 and 1.
        bankroll: Total capital available for betting, in USD. Must be > 0.
        kelly_multiplier: Fraction of full Kelly to apply before the cap.
            Use 0.25 for YES bets (low win rate, high variance) and 0.50
            for NO bets (high win rate, low variance). Default is 0.25.
        side: "yes" or "no" — which contract is being purchased. Determines
            the payout ratio b and the win/loss probabilities in the Kelly formula.

    Returns:
        KellyResult with bet sizing details and a plain-English reason.

    Raises:
        ValueError: If p_market is not strictly between 0 and 1, or bankroll <= 0.
    """
    if p_market <= 0 or p_market >= 1:
        raise ValueError(
            f"p_market must be strictly between 0 and 1, got {p_market}"
        )
    if bankroll <= 0:
        raise ValueError(f"bankroll must be positive, got {bankroll}")

    if side == "no":
        # NO contract: pay (1 - p_market), win p_market if YES does not resolve.
        #   b = p_market / (1 - p_market)
        #   f = (p_no_model * b - p_yes_model) / b
        p_no_model  = 1 - p_model
        p_yes_model = p_model
        b = p_market / (1 - p_market)
        f = (p_no_model * b - p_yes_model) / b
    else:
        # YES contract: pay p_market, win (1 - p_market) if YES resolves.
        #   b = (1 - p_market) / p_market
        #   f = (p_yes_model * b - p_no_model) / b
        b = (1 - p_market) / p_market
        f = (p_model * b - (1 - p_model)) / b

    if f <= 0:
        return KellyResult(
            kelly_fraction=f,
            bet_fraction=0.0,
            bet_amount=0.0,
            bankroll=bankroll,
            was_capped=False,
            reason="Kelly fraction is zero or negative — no edge, do not bet.",
        )

    # Scale by kelly_multiplier, then hard cap at max_bet_fraction (default 5%)
    scaled_f = f * kelly_multiplier
    was_capped = scaled_f > max_bet_fraction
    bet_fraction = min(scaled_f, max_bet_fraction)
    bet_amount = round(bet_fraction * bankroll, 2)

    multiplier_pct = int(kelly_multiplier * 100)
    if was_capped:
        reason = (
            f"{multiplier_pct}% Kelly {scaled_f:.4f} ({scaled_f:.2%}) exceeds the 5% cap. "
            f"Bet capped at {bet_fraction:.2%} of bankroll = ${bet_amount:.2f}."
        )
    else:
        reason = (
            f"{multiplier_pct}% Kelly {scaled_f:.4f} ({scaled_f:.2%}) is within the 5% cap. "
            f"Betting {bet_fraction:.2%} of bankroll = ${bet_amount:.2f}."
        )

    return KellyResult(
        kelly_fraction=f,
        bet_fraction=bet_fraction,
        bet_amount=bet_amount,
        bankroll=bankroll,
        was_capped=was_capped,
        reason=reason,
    )
