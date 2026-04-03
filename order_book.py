"""
Order book imbalance module.

Fetches real-time order book data from Binance, Coinbase, and Kraken
concurrently and computes a strike-targeted imbalance score for use as a
real-time directional confirmation indicator.

OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

A positive OBI indicates more buying pressure (bullish); negative indicates
more selling pressure (bearish). Only orders within a depth window around the
mid-price are included to filter far-from-market noise.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

import requests

# Score threshold: |OBI| must exceed this to produce a non-zero score.
OBI_THRESHOLD = 0.10

# Only include orders within this fraction of mid-price (±0.5%).
OBI_DEPTH_WINDOW = 0.005

TIMEOUT = 5  # seconds per exchange request

_ASSET_URLS = {
    "BTC": {
        "binance":  "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100",
        "coinbase": "https://api.coinbase.com/api/v3/brokerage/market/product_book?product_id=BTC-USD&limit=100",
        "kraken":   "https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=100",
    },
    "ETH": {
        "binance":  "https://api.binance.com/api/v3/depth?symbol=ETHUSDT&limit=100",
        "coinbase": "https://api.coinbase.com/api/v3/brokerage/market/product_book?product_id=ETH-USD&limit=100",
        "kraken":   "https://api.kraken.com/0/public/Depth?pair=XETHZUSD&count=100",
    },
    "SOL": {
        "binance":  "https://api.binance.com/api/v3/depth?symbol=SOLUSDT&limit=100",
        "coinbase": "https://api.coinbase.com/api/v3/brokerage/market/product_book?product_id=SOL-USD&limit=100",
        "kraken":   "https://api.kraken.com/0/public/Depth?pair=SOLUSD&count=100",
    },
}


@dataclass
class OrderBookResult:
    """Output of the order book imbalance module."""

    obi: float           # raw OBI value, -1.0 to +1.0 (NaN if unavailable)
    obi_score: int       # +1 bullish, -1 bearish, 0 neutral
    bid_volume: float    # total bid volume within depth window, averaged across exchanges
    ask_volume: float    # total ask volume within depth window, averaged across exchanges
    exchanges_used: int  # number of exchanges that responded successfully
    reason: str          # plain-English explanation


def _parse_binance(data: dict) -> Tuple[List, List]:
    """Parse Binance depth response into (bids, asks) lists of (price, qty) tuples."""
    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]
    return bids, asks


def _parse_coinbase(data: dict) -> Tuple[List, List]:
    """Parse Coinbase v3 product_book response into (bids, asks) lists."""
    pb = data["pricebook"]
    bids = [(float(e["price"]), float(e["size"])) for e in pb["bids"]]
    asks = [(float(e["price"]), float(e["size"])) for e in pb["asks"]]
    return bids, asks


def _parse_kraken(data: dict) -> Tuple[List, List]:
    """Parse Kraken depth response into (bids, asks) lists."""
    book = list(data["result"].values())[0]
    bids = [(float(e[0]), float(e[1])) for e in book["bids"]]
    asks = [(float(e[0]), float(e[1])) for e in book["asks"]]
    return bids, asks


def _fetch_exchange(url: str, parser) -> Tuple[List, List]:
    """Fetch and parse one exchange's order book. Raises on any failure."""
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return parser(r.json())


def _compute_obi(bids: List, asks: List, mid: float, window: float) -> Tuple[float, float, float]:
    """
    Compute OBI for orders within ±window of mid price.

    Returns (obi, bid_vol, ask_vol). Returns (NaN, 0, 0) if no orders in range.
    """
    lo = mid * (1 - window)
    hi = mid * (1 + window)
    bid_vol = sum(q for p, q in bids if lo <= p <= hi)
    ask_vol = sum(q for p, q in asks if lo <= p <= hi)
    total = bid_vol + ask_vol
    if total == 0:
        return float("nan"), 0.0, 0.0
    return (bid_vol - ask_vol) / total, bid_vol, ask_vol


def fetch_order_book_imbalance(asset: str = "BTC", window: float = OBI_DEPTH_WINDOW) -> OrderBookResult:
    """
    Fetch order books from Binance, Coinbase, and Kraken for the given asset
    concurrently, compute OBI within a depth window around the current mid-price,
    and return the average score across exchanges.

    Args:
        asset: Asset to fetch order book for: "BTC", "ETH", or "SOL".
        window: Fractional price range around mid-price to include (default ±0.5%).

    Returns:
        OrderBookResult with obi_score (+1/0/-1), raw OBI, and diagnostics.
        Returns obi_score=0 (neutral) if all exchanges fail.
    """
    urls = _ASSET_URLS.get(asset.upper(), _ASSET_URLS["BTC"])
    sources = {
        "binance":  (urls["binance"],  _parse_binance),
        "coinbase": (urls["coinbase"], _parse_coinbase),
        "kraken":   (urls["kraken"],   _parse_kraken),
    }

    books = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_fetch_exchange, url, parser): name
            for name, (url, parser) in sources.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                books[name] = fut.result()
            except Exception:
                pass  # exchange failed — skip silently, handle below

    if not books:
        return OrderBookResult(
            obi=float("nan"), obi_score=0,
            bid_volume=0.0, ask_volume=0.0,
            exchanges_used=0,
            reason="OBI unavailable — all exchange requests failed. Score=0 (neutral).",
        )

    # Derive mid-price from best bid/ask across all available books
    all_best_bids = [max(p for p, _ in bids) for bids, _ in books.values() if bids]
    all_best_asks = [min(p for p, _ in asks) for _, asks in books.values() if asks]
    if not all_best_bids or not all_best_asks:
        return OrderBookResult(
            obi=float("nan"), obi_score=0,
            bid_volume=0.0, ask_volume=0.0,
            exchanges_used=len(books),
            reason=f"OBI unavailable — could not determine mid-price from {len(books)} exchange(s). Score=0.",
        )
    mid = (sum(all_best_bids) / len(all_best_bids) + sum(all_best_asks) / len(all_best_asks)) / 2

    # Compute OBI per exchange, average the ratios (not raw volumes)
    obis, bid_vols, ask_vols = [], [], []
    for name, (bids, asks) in books.items():
        obi_val, bv, av = _compute_obi(bids, asks, mid, window)
        if not math.isnan(obi_val):
            obis.append(obi_val)
            bid_vols.append(bv)
            ask_vols.append(av)

    if not obis:
        return OrderBookResult(
            obi=float("nan"), obi_score=0,
            bid_volume=0.0, ask_volume=0.0,
            exchanges_used=len(books),
            reason=(
                f"OBI unavailable — no orders within ±{window:.1%} of mid ${mid:,.0f} "
                f"across {len(books)} exchange(s). Score=0."
            ),
        )

    avg_obi  = sum(obis) / len(obis)
    avg_bids = sum(bid_vols) / len(bid_vols)
    avg_asks = sum(ask_vols) / len(ask_vols)

    if avg_obi > OBI_THRESHOLD:
        score, label = +1, "bullish"
    elif avg_obi < -OBI_THRESHOLD:
        score, label = -1, "bearish"
    else:
        score, label = 0, "neutral"

    reason = (
        f"OBI={avg_obi:+.3f} ({label}) from {len(books)} exchange(s) "
        f"[{', '.join(books.keys())}] within ±{window:.1%} of mid ${mid:,.0f}. "
        f"bid={avg_bids:.3f}, ask={avg_asks:.3f}. "
        f"score={score:+d} (threshold=±{OBI_THRESHOLD})."
    )

    return OrderBookResult(
        obi=avg_obi, obi_score=score,
        bid_volume=avg_bids, ask_volume=avg_asks,
        exchanges_used=len(books),
        reason=reason,
    )
