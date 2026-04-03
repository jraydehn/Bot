"""
Funding rate module: fetches perpetual futures funding rate from Binance,
Bybit, and OKX concurrently and classifies the averaged result as a directional
bias signal.

Background on funding mechanics
---------------------------------
Perpetual futures have no expiry, so exchanges use a periodic funding payment
to keep the perpetual price anchored to spot. Every 8 hours, one side pays the
other based on the current funding rate:

  Positive funding rate:
    Longs pay shorts. This happens when the perpetual trades above spot —
    meaning more demand to be long than short. The market is overcrowded on
    the long side. Mean-reversion pressure is downward: leveraged longs
    eventually exit or get liquidated, pulling price down.
    → funding_bias = -1 (bearish signal)

  Negative funding rate:
    Shorts pay longs. This happens when the perpetual trades below spot —
    meaning more demand to be short than long. The market is overcrowded on
    the short side. Squeeze pressure is upward: shorts are forced to cover,
    pushing price up.
    → funding_bias = +1 (bullish signal)

  Neutral (near-zero) funding:
    Neither side is meaningfully overcrowded. No strong positioning signal.
    → funding_bias = 0

Typical funding ranges -0.05% to +0.10% per 8-hour period.
Anything above +0.05% is aggressively overcrowded long and a strong
bearish mean-reversion signal for NO trades.
"""

import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional

import requests

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Positive funding threshold: when funding exceeds this, longs are paying shorts
# at a rate that signals overcrowded positioning. Mean-reversion pressure favors
# a price decline — bearish for YES, supportive for NO trades.
FUNDING_BEARISH_THRESHOLD = +0.00005  # +0.005% per 8 hours (~5.5% annualized)

# Negative funding threshold: when funding falls below this, shorts are paying
# longs at a rate that signals overcrowded short positioning. Squeeze pressure
# favors a price rise — bullish for YES, negative for NO trades.
FUNDING_BULLISH_THRESHOLD = -0.00005  # -0.005% per 8 hours

# Per-request timeout in seconds.
_TIMEOUT = 2.0

# Funding periods per day (8-hour intervals) × days per year
_ANNUALIZE_FACTOR = 3 * 365

# ---------------------------------------------------------------------------
# Asset symbol maps per exchange
# ---------------------------------------------------------------------------

_GATEIO_SYMBOLS = {"BTC": "BTC_USDT",       "ETH": "ETH_USDT",       "SOL": "SOL_USDT"}
_BITMEX_SYMBOLS = {"BTC": "XBTUSD",          "ETH": "ETHUSD",          "SOL": "SOLUSD"}
_OKX_SYMBOLS    = {"BTC": "BTC-USDT-SWAP",   "ETH": "ETH-USDT-SWAP",   "SOL": "SOL-USDT-SWAP"}


@dataclass
class FundingRateResult:
    """Output of the funding rate module."""

    funding_bias: int              # +1 bullish, 0 neutral, -1 bearish
    avg_funding_rate: float        # equal-weighted average across responding exchanges
    gateio_rate: Optional[float]   # individual exchange rates; None if fetch failed
    bitmex_rate: Optional[float]
    okx_rate: Optional[float]
    exchanges_used: List[str]      # names of exchanges that contributed to the average
    data_available: bool           # False only if all three exchanges failed
    annualized_rate: float         # avg_funding_rate * 3 * 365 (context only)
    reason: str                    # plain-English explanation of the classification


def _fetch_gateio(asset: str) -> float:
    """
    Fetch perpetual funding rate from Gate.io.
    Endpoint: GET https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}
    """
    symbol = _GATEIO_SYMBOLS.get(asset.upper(), "BTC_USDT")
    resp = requests.get(
        f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}",
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return float(resp.json()["funding_rate"])


def _fetch_bitmex(asset: str) -> float:
    """
    Fetch perpetual funding rate from BitMEX.
    Endpoint: GET https://www.bitmex.com/api/v1/instrument?symbol={symbol}&columns=fundingRate
    """
    symbol = _BITMEX_SYMBOLS.get(asset.upper(), "XBTUSD")
    resp = requests.get(
        "https://www.bitmex.com/api/v1/instrument",
        params={"symbol": symbol, "columns": "fundingRate"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return float(resp.json()[0]["fundingRate"])


def _fetch_okx(asset: str) -> float:
    """
    Fetch perpetual funding rate from OKX.
    Endpoint: GET https://www.okx.com/api/v5/public/funding-rate?instId={symbol}
    """
    symbol = _OKX_SYMBOLS.get(asset.upper(), "BTC-USDT-SWAP")
    resp = requests.get(
        "https://www.okx.com/api/v5/public/funding-rate",
        params={"instId": symbol},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return float(resp.json()["data"][0]["fundingRate"])


# Neutral fallback returned when all three exchanges fail — never crashes the
# main loop.
_FALLBACK = FundingRateResult(
    funding_bias=0,
    avg_funding_rate=0.0,
    gateio_rate=None,
    bitmex_rate=None,
    okx_rate=None,
    exchanges_used=[],
    data_available=False,
    annualized_rate=0.0,
    reason="All exchanges failed — funding_bias=0 (neutral fallback).",
)


def fetch_funding_rate(asset: str = "BTC") -> FundingRateResult:
    """
    Fetch perpetual funding rate from Gate.io, BitMEX, and OKX concurrently
    for the given asset.

    Uses a thread pool with a 2-second per-request timeout. All three requests
    fire simultaneously; partial results (1 or 2 exchanges) are used if some fail.

    Args:
        asset: Asset to fetch funding rate for: "BTC", "ETH", or "SOL".

    Returns:
        FundingRateResult with averaged rate, per-exchange breakdown, and bias.
        On total failure returns a neutral fallback (data_available=False).
    """
    asset = asset.upper()
    fetchers = {
        "gateio": lambda: _fetch_gateio(asset),
        "bitmex": lambda: _fetch_bitmex(asset),
        "okx":    lambda: _fetch_okx(asset),
    }

    raw: dict = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_name = {executor.submit(fn): name for name, fn in fetchers.items()}
            done, _ = concurrent.futures.wait(
                future_to_name, timeout=_TIMEOUT + 1,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for future in future_to_name:
                name = future_to_name[future]
                if future in done:
                    try:
                        raw[name] = future.result()
                    except Exception as exc:
                        print(f"  [funding/{asset}] {name} failed: {exc}")
                        raw[name] = None
                else:
                    print(f"  [funding/{asset}] {name} timed out")
                    raw[name] = None
    except Exception as exc:
        print(f"  [funding/{asset}] ThreadPoolExecutor error: {exc}")
        return _FALLBACK

    gateio_rate = raw.get("gateio")
    bitmex_rate = raw.get("bitmex")
    okx_rate    = raw.get("okx")

    available = {k: v for k, v in raw.items() if v is not None}
    exchanges_used = list(available.keys())

    if not available:
        return _FALLBACK

    if len(available) < 3:
        missing = [k for k in fetchers if k not in available]
        print(f"  [funding/{asset}] {len(available)}/3 exchanges responded ({', '.join(missing)} unavailable)")

    avg_funding_rate = sum(available.values()) / len(available)
    annualized_rate  = avg_funding_rate * _ANNUALIZE_FACTOR

    if avg_funding_rate > FUNDING_BEARISH_THRESHOLD:
        funding_bias = -1
        direction    = "bearish"
        detail = (
            f"overcrowded longs — funding={avg_funding_rate*100:+.4f}%/8h "
            f"exceeds bearish threshold of {FUNDING_BEARISH_THRESHOLD*100:+.4f}%"
        )
    elif avg_funding_rate < FUNDING_BULLISH_THRESHOLD:
        funding_bias = +1
        direction    = "bullish"
        detail = (
            f"overcrowded shorts — funding={avg_funding_rate*100:+.4f}%/8h "
            f"below bullish threshold of {FUNDING_BULLISH_THRESHOLD*100:+.4f}%"
        )
    else:
        funding_bias = 0
        direction    = "neutral"
        detail = (
            f"neutral positioning — funding={avg_funding_rate*100:+.4f}%/8h "
            f"within [{FUNDING_BULLISH_THRESHOLD*100:+.4f}%, {FUNDING_BEARISH_THRESHOLD*100:+.4f}%]"
        )

    reason = (
        f"Funding {direction} ({asset}): {detail}. "
        f"Exchanges used: {', '.join(exchanges_used)}. "
        f"Annualized: {annualized_rate*100:.1f}%/yr."
    )

    return FundingRateResult(
        funding_bias=funding_bias,
        avg_funding_rate=avg_funding_rate,
        gateio_rate=gateio_rate,
        bitmex_rate=bitmex_rate,
        okx_rate=okx_rate,
        exchanges_used=exchanges_used,
        data_available=True,
        annualized_rate=annualized_rate,
        reason=reason,
    )
