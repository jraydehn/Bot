"""
Order book depth signal for Kalshi BTC binary options.

Fetches Coinbase spot order book (BTC-USD) and computes bid/ask mass
in a configurable window around a given strike price.

Signal rationale:
  - Heavy bids near/below strike  → natural floor; YES structurally supported
  - Heavy asks near/above strike  → ceiling resistance; NO structurally supported
  - Kalshi's lognormal p_market doesn't incorporate this information

Usage:
    from orderbook_depth import fetch_ob_signal
    sig = fetch_ob_signal(strike=97500, spot=97800)
    print(sig.imbalance, sig.bid_mass_usd, sig.ask_mass_usd)
"""

import time
import warnings
from dataclasses import dataclass
from typing import Optional

import requests

# Coinbase Exchange spot order books — free, US-accessible, full depth
_CB_DEPTH_URL = "https://api.exchange.coinbase.com/products/{pair}/book?level=2"
_ASSET_PAIRS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}

# Per-asset cache
_CACHE_TTL = 30.0  # seconds
_cache: dict = {}   # asset → {"ts": float, "bids": list, "asks": list}

# Window around strike for cluster measurement (fraction of strike price)
DEPTH_PCT = 0.005   # 0.5%  — ~$480 at BTC=96k
WALL_THRESHOLD_USD = 500_000   # minimum notional to count as a "wall" ($500k)


@dataclass
class OBSignal:
    bid_mass_usd: float    # USD bid notional within DEPTH_PCT below strike
    ask_mass_usd: float    # USD ask notional within DEPTH_PCT above strike
    imbalance: float       # (bid - ask) / (bid + ask), range [-1, +1]
    bid_wall_pct: float    # distance to nearest large bid wall below spot (as % of spot, negative)
    ask_wall_pct: float    # distance to nearest large ask wall above spot (as % of spot, positive)
    total_bid_usd: float   # total book bid notional (for context / normalization)
    total_ask_usd: float   # total book ask notional
    # Path-to-strike signals (the genuinely new information vs offset_pct)
    path_ask_usd: float    # total USD asks between spot and strike (OTM YES: resistance to clear)
    path_bid_usd: float    # total USD bids between strike and spot (OTM NO: support below)
    ask_frac: float        # ask_mass_usd / total_ask_usd — normalized resistance fraction
    bid_frac: float        # bid_mass_usd / total_bid_usd — normalized support fraction


def _fetch_raw_book(asset: str = "BTC") -> tuple[list, list]:
    """Return (bids, asks) for the given asset, with 30s per-asset cache."""
    asset_u = asset.upper()
    now = time.monotonic()
    cached = _cache.get(asset_u, {})
    if cached.get("ts", 0) and now - cached["ts"] < _CACHE_TTL and cached.get("bids"):
        return cached["bids"], cached["asks"]
    pair = _ASSET_PAIRS.get(asset_u, f"{asset_u}-USD")
    url = _CB_DEPTH_URL.format(pair=pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    _cache[asset_u] = {"ts": now, "bids": data["bids"], "asks": data["asks"]}
    return data["bids"], data["asks"]


def _notional(levels: list, lo: float, hi: float) -> float:
    """Sum price×qty for all levels where lo <= price <= hi."""
    total = 0.0
    for lvl in levels:
        p = float(lvl[0])
        if lo <= p <= hi:
            total += p * float(lvl[1])
    return total


def _total_notional(levels: list) -> float:
    return sum(float(lvl[0]) * float(lvl[1]) for lvl in levels)


def _nearest_wall_pct(levels: list, spot: float, side: str) -> float:
    """
    Find the nearest price level with notional >= WALL_THRESHOLD_USD.
    Returns distance as signed fraction of spot:
      side='bid': negative (wall is below spot)
      side='ask': positive (wall is above spot)
    Returns 0.0 if none found.
    """
    best_dist = 0.0
    for lvl in levels:
        p = float(lvl[0])
        notional = p * float(lvl[1])
        if notional >= WALL_THRESHOLD_USD:
            dist = (p - spot) / spot   # negative for bids below spot, positive for asks above
            if side == 'bid' and dist < 0:
                if best_dist == 0.0 or dist > best_dist:  # closest to spot = least negative
                    best_dist = dist
            elif side == 'ask' and dist > 0:
                if best_dist == 0.0 or dist < best_dist:  # closest to spot = least positive
                    best_dist = dist
    return best_dist


def fetch_ob_signal(strike: float, spot: float,
                    asset: str = "BTC",
                    depth_pct: float = DEPTH_PCT) -> Optional[OBSignal]:
    """
    Compute order book imbalance signal around a given strike price.

    Args:
        strike:    Kalshi contract strike price
        spot:      Current asset spot price
        asset:     "BTC", "ETH", or "SOL"
        depth_pct: Window half-width as fraction of strike (default 0.5%)

    Returns OBSignal or None if fetch fails.
    """
    try:
        bids, asks = _fetch_raw_book(asset)
    except Exception as e:
        print(f"  [ob_depth] fetch failed: {e}")
        return None

    lo_bid = strike * (1.0 - depth_pct)
    hi_bid = strike                          # bids up to strike
    lo_ask = strike                          # asks from strike up
    hi_ask = strike * (1.0 + depth_pct)

    bid_mass = _notional(bids, lo_bid, hi_bid)
    ask_mass = _notional(asks, lo_ask, hi_ask)
    denom = bid_mass + ask_mass
    imbalance = (bid_mass - ask_mass) / denom if denom > 0 else 0.0

    total_bid = _total_notional(bids)
    total_ask = _total_notional(asks)

    # Path-to-strike: asks between spot and strike (OTM YES resistance to clear)
    if strike > spot:
        path_ask = _notional(asks, spot, strike)
        path_bid = 0.0
    elif strike < spot:
        path_bid = _notional(bids, strike, spot)
        path_ask = 0.0
    else:
        path_ask = path_bid = 0.0

    bid_wall = _nearest_wall_pct(bids, spot, 'bid')
    ask_wall = _nearest_wall_pct(asks, spot, 'ask')

    return OBSignal(
        bid_mass_usd=bid_mass,
        ask_mass_usd=ask_mass,
        imbalance=round(imbalance, 4),
        bid_wall_pct=round(bid_wall, 5),
        ask_wall_pct=round(ask_wall, 5),
        total_bid_usd=total_bid,
        total_ask_usd=total_ask,
        path_ask_usd=round(path_ask, 2),
        path_bid_usd=round(path_bid, 2),
        ask_frac=round(ask_mass / total_ask, 6) if total_ask > 0 else 0.0,
        bid_frac=round(bid_mass / total_bid, 6) if total_bid > 0 else 0.0,
    )


if __name__ == "__main__":
    for asset in ["BTC", "ETH", "SOL"]:
        try:
            bids, asks = _fetch_raw_book(asset)
        except Exception as e:
            print(f"{asset}: fetch failed — {e}")
            continue
        spot = float(bids[0][0])
        print(f"\n{asset} spot ~${spot:,.2f}  |  {len(bids)} bid levels, {len(asks)} ask levels")
        print(f"Total book: bids=${_total_notional(bids)/1e6:.1f}M  asks=${_total_notional(asks)/1e6:.1f}M")

        offsets = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
        print(f"{'Strike':>12}  {'Offset':>7}  {'Imbalance':>10}  {'PathAsk$M':>10}  {'PathBid$M':>10}  {'AskFrac%':>9}")
        print("-" * 72)
        for off in offsets:
            k = spot * (1 + off)
            sig = fetch_ob_signal(k, spot, asset=asset)
            if sig:
                print(f"${k:>11,.2f}  {off:>+7.1%}  "
                      f"{sig.imbalance:>+10.3f}  "
                      f"${sig.path_ask_usd/1e6:>9.2f}M  "
                      f"${sig.path_bid_usd/1e6:>9.2f}M  "
                      f"{sig.ask_frac:>9.4%}")
