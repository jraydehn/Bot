"""
coinglass_data.py — CoinGlass Hobbyist-tier signals for BTC, ETH, SOL.

Available on the Hobbyist plan ($29/month):
  1. Exchange balance flows  — net BTC/ETH/SOL moving to/from exchanges
     Inflows (positive)  = selling pressure → bearish
     Outflows (negative) = accumulation/HODLing → bullish
     Not available from Coinalyze — this is unique to CoinGlass.

  2. Options OI snapshot    — total open interest + 24h change across Deribit/OKX/Binance/Bybit/CME
     Expanding OI = market pricing in a large move → elevated vol expected
     BTC and ETH only (options markets).

  3. Spot taker buy/sell    — aggregated buy vs sell volume across Binance/OKX/Bybit
     ratio > 1 = buyers dominant → bullish momentum
     Daily resolution only at Hobbyist tier.

  4. Fear & Greed index     — daily macro sentiment (0=extreme fear, 100=extreme greed)
     Contrarian: extreme fear (<25) → bullish; extreme greed (>75) → bearish

NOT available at Hobbyist:
  - Liquidation heatmap (requires Professional, $699/month)
  - Futures L/S ratio history, OI history, funding rate history (require Startup+)

Usage:
    sig = fetch_coinglass_signals("BTC")
    if sig:
        print(sig.exchange_flow_label, sig.options_oi_change_24h, sig.fg_score)
"""

import os
import time
from typing import Optional, NamedTuple

import requests

_BASE    = "https://open-api-v4.coinglass.com"
_API_KEY = os.environ.get("COINGLASS_API_KEY", "8f0a30c29a5e424ba2641f649051786b")
_HEADERS = {"CG-API-KEY": _API_KEY}

# Exchange flow: % of total exchange holdings flowing in/out that counts as a signal
_FLOW_BEARISH_PCT = +0.10   # > +0.10% of total balance flowing IN per day → bearish
_FLOW_BULLISH_PCT = -0.10   # < -0.10% of total balance flowing OUT per day → bullish

# Options OI growth: 24h change that's significant
_OI_EXPANSION_THRESH = +3.0  # > +3% OI growth → vol expansion expected
_OI_CONTRACTION_THRESH = -3.0

# Fear & Greed
_EXTREME_FEAR  = 25
_EXTREME_GREED = 75

# Cache TTLs
_FLOW_CACHE_TTL = 1800    # exchange flows update slowly; 30-min cache
_OPTIONS_CACHE_TTL = 600  # options OI updates more often; 10-min cache
_FG_CACHE_TTL = 3600      # fear & greed is daily; 1-hour cache
_TAKER_CACHE_TTL    = 3600  # spot taker daily (legacy); 1-hour cache
_TAKER_4H_CACHE_TTL    = 3600  # spot taker 4h bar; 1-hour cache (bar changes every 4h)
_FR_VOL_CACHE_TTL      = 3600  # vol-weighted funding rate is daily; 1-hour cache
_OI_STABLE_CACHE_TTL   = 3600  # stablecoin OI 4h bar; 1-hour cache
_FUTURES_TAKER_TTL     = 3600  # futures aggregated taker 4h bar; 1-hour cache
_SPOT_CB_CACHE_TTL     = 3600  # cross-exchange spot CVD (Binance+Coinbase+OKX) 4h bar; 1-hour cache
_LIQ_CACHE_TTL         = 3600  # aggregated liquidation 4h bar; 1-hour cache
_HL_SHADOW_TTL         =  900  # Hyperliquid whale snapshot; 15-min cache (real-time endpoint)
_EXCH_LIQ_TTL          =  900  # exchange-list liquidation rolling windows; 15-min cache

_CACHE: dict = {}

_SPOT_EXCHANGES    = "Binance,OKX,Bybit"
_SPOT_CB_EXCHANGES = "Binance,Coinbase,OKX"   # Coinbase-inclusive for spot CVD gate
_FUTURES_EXCHANGES = "Binance,OKX,Bybit"


class CoinGlassSignals(NamedTuple):
    # Exchange flows
    exchange_flow_1d:       float   # net BTC/ETH/SOL change on exchanges today (units of asset)
    exchange_flow_1d_pct:   float   # as % of total exchange holdings
    exchange_flow_7d_pct:   float   # 7-day % change (trend confirmation)
    exchange_flow_score:    int     # +1 (bullish outflow) | 0 | -1 (bearish inflow)
    exchange_flow_label:    str

    # Options OI (BTC/ETH only; SOL has no meaningful options market)
    options_oi_usd:         float   # total options open interest in USD
    options_oi_change_24h:  float   # % change in options OI over 24h
    options_vol_usd_24h:    float   # 24h options volume in USD
    options_score:          int     # +1 (contraction=caution) | 0 | -1 (expansion=volatile)

    # Spot taker (daily only)
    spot_buy_usd:           float   # total spot buy volume USD (latest daily bar)
    spot_sell_usd:          float   # total spot sell volume USD
    spot_taker_ratio:       float   # buy/sell ratio (>1 = buyers dominant)

    # Fear & Greed
    fg_value:               float   # 0–100
    fg_trend:               str     # "rising" | "falling" | "flat"
    fg_regime:              str     # "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed"
    fg_score:               int     # +1 (extreme fear) | 0 | -1 (extreme greed)

    # Live funding rate (equal-weighted avg across Binance/OKX/Bybit, 30-min cache)
    fr_vol_1d:              float   # current aggregate funding rate
                                    # >0 = longs paying (crowded long, bearish for YES OTM)
                                    # <0 = shorts paying (crowded short, contrarian bullish)

    # Stablecoin-margin OI % change (most recent completed 4h bar)
    oi_stable_pct_4h:       float   # 4h % change in stablecoin-margined open interest
                                    # >2% = leveraged longs crowding in → OTM YES risky
                                    # >1% on NO side = longs crowding → NO unlikely

    # Spot taker buy/sell ratio (most recent completed 4h bar)
    taker_ratio_4h:         float   # buy/sell ratio (>1 = buyers dominant, <1 = sellers)

    # Futures aggregated taker (Binance+OKX+Bybit perps, most recent completed 4h bar)
    futures_delta_4h:       float   # buy_usd - sell_usd; +ve = net buying pressure
    futures_ratio_4h:       float   # buy/sell ratio; >1 = buyers dominant
    futures_cvd_12h:        float   # rolling 12h futures delta (3 × 4h bars); trend window

    # Cross-exchange spot CVD (Binance+Coinbase+OKX, most recent completed 4h bar)
    spot_cb_ratio_4h:       float   # buy/sell ratio; >1.05 = buyers dominant → NO loses
                                    # Block BTC/ETH NO when >1.05 (stoch_k<=80 for BTC)
                                    # Boost BTC NO ×1.25 when <0.90 + futures_ratio>=0.90

    # Aggregated liquidation data (Binance+OKX+Bybit, most recent completed 4h bar)
    liq_ratio_4h:           float   # long_liq_usd / short_liq_usd; >1 = more longs liquidated (bearish)
                                    # Block NO when <0.70 (short squeeze; BTC rescue stoch_k>80)
                                    # Boost BTC NO ×1.25 when >5.0; SOL ×1.25 when >1.5
    liq_total_4h:           float   # total liquidation USD (long + short) in last 4h bar
                                    # Boost BTC NO ×1.5 when >$60M (extreme liquidation event)

    # Hyperliquid whale signals (real-time snapshot; shadow-logged only)
    hl_ls_ratio:            float   # HL whale long_value / short_value; >1 = net long
    hl_squeeze_idx:         float   # (short_liq_near - long_liq_near) / total; +ve = squeeze risk
    hl_liq_ratio_4h:        float   # HL-only rolling 4h long_liq / short_liq [SHADOW]
    all_liq_ratio_1h:       float   # All-exchange rolling 1h long_liq / short_liq [SHADOW]
    all_liq_ratio_4h:       float   # All-exchange rolling 4h long_liq / short_liq [SHADOW]

    # Composite
    composite_score:        int     # exchange_flow_score + fg_score, clamped [-2, +2]
    label:                  str


def _get(path: str, params: dict, timeout: float = 6.0) -> Optional[dict]:
    try:
        r = requests.get(f"{_BASE}{path}", headers=_HEADERS, params=params, timeout=timeout)
        body = r.json()
        if body.get("code") == "0":
            return body.get("data")
        return None
    except Exception:
        return None


def _fetch_exchange_flows(asset: str) -> tuple:
    """Returns (flow_1d, flow_1d_pct, flow_7d_pct, score, label)."""
    cache_key = f"flows_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _FLOW_CACHE_TTL:
        return _CACHE[cache_key][0]

    data = _get("/api/exchange/balance/list", {"symbol": asset.upper()})
    if not data:
        return (0.0, 0.0, 0.0, 0, "exchange flow unavailable")

    total_balance = sum(x.get("total_balance", 0) for x in data)
    flow_1d = sum(x.get("balance_change_1d", 0) for x in data)
    flow_7d = sum(x.get("balance_change_7d", 0) for x in data)

    flow_1d_pct = (flow_1d / total_balance * 100) if total_balance > 0 else 0.0
    flow_7d_pct = (flow_7d / total_balance * 100) if total_balance > 0 else 0.0

    if flow_1d_pct <= _FLOW_BULLISH_PCT:
        score = +1
        label = f"exchange outflow {flow_1d:+.0f} ({flow_1d_pct:+.3f}%) → accumulation bullish"
    elif flow_1d_pct >= _FLOW_BEARISH_PCT:
        score = -1
        label = f"exchange inflow {flow_1d:+.0f} ({flow_1d_pct:+.3f}%) → selling pressure bearish"
    else:
        score = 0
        label = f"exchange flow {flow_1d:+.0f} ({flow_1d_pct:+.3f}%) neutral"

    result = (flow_1d, flow_1d_pct, flow_7d_pct, score, label)
    _CACHE[cache_key] = (result, now)
    return result


def _fetch_options(asset: str) -> tuple:
    """Returns (oi_usd, oi_change_24h, vol_usd_24h, score)."""
    cache_key = f"options_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _OPTIONS_CACHE_TTL:
        return _CACHE[cache_key][0]

    data = _get("/api/option/info", {"symbol": asset.upper()})
    if not data:
        return (0.0, 0.0, 0.0, 0)

    # Find the "All" aggregate row
    agg = next((x for x in data if x.get("exchange_name") == "All"), data[0] if data else None)
    if agg is None:
        return (0.0, 0.0, 0.0, 0)

    oi_usd        = float(agg.get("open_interest_usd", 0))
    oi_change_24h = float(agg.get("open_interest_change_24h", 0))
    vol_usd_24h   = float(agg.get("volume_usd_24h", 0))

    # Expanding OI = market is adding positions = anticipating a move → elevated vol
    if oi_change_24h >= _OI_EXPANSION_THRESH:
        score = -1   # high vol expected → widen uncertainty, don't over-bet
    elif oi_change_24h <= _OI_CONTRACTION_THRESH:
        score = +1   # OI shrinking = positioning unwinding = calmer → model more reliable
    else:
        score = 0

    result = (oi_usd, oi_change_24h, vol_usd_24h, score)
    _CACHE[cache_key] = (result, now)
    return result


def _fetch_spot_taker(asset: str) -> tuple:
    """Returns (buy_usd, sell_usd, ratio) — latest daily bar."""
    cache_key = f"taker_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _TAKER_CACHE_TTL:
        return _CACHE[cache_key][0]

    data = _get("/api/spot/aggregated-taker-buy-sell-volume/history", {
        "symbol": asset.upper(), "interval": "1d", "limit": 2, "exchange_list": _SPOT_EXCHANGES
    })
    if not data or not isinstance(data, list) or not data:
        return (0.0, 0.0, 1.0)

    latest = data[-1]
    buy  = float(latest.get("aggregated_buy_volume_usd",  0))
    sell = float(latest.get("aggregated_sell_volume_usd", 0))
    ratio = buy / sell if sell > 0 else 1.0

    result = (buy, sell, ratio)
    _CACHE[cache_key] = (result, now)
    return result


def _fetch_fear_greed() -> tuple:
    """Returns (value, trend, regime, score)."""
    cache_key = "fear_greed"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _FG_CACHE_TTL:
        return _CACHE[cache_key][0]

    # Fear & Greed is on v3 base URL
    try:
        r = requests.get(
            "https://open-api-v3.coinglass.com/api/index/fear-greed-history",
            headers=_HEADERS, timeout=6,
        )
        body = r.json()
        if body.get("code") != "0":
            return (50.0, "flat", "neutral", 0)
        values = body["data"]["values"]
        if len(values) < 8:
            return (50.0, "flat", "neutral", 0)
    except Exception:
        return (50.0, "flat", "neutral", 0)

    current  = float(values[-1])
    week_ago = float(values[-8])
    delta    = current - week_ago

    trend = "rising" if delta > 3 else "falling" if delta < -3 else "flat"

    if current < _EXTREME_FEAR:
        regime, score = "extreme_fear",  +1
    elif current < 40:
        regime, score = "fear",           0
    elif current < 60:
        regime, score = "neutral",        0
    elif current < _EXTREME_GREED:
        regime, score = "greed",         -1
    else:
        regime, score = "extreme_greed", -1

    result = (current, trend, regime, score)
    _CACHE[cache_key] = (result, now)
    return result


_FR_LIVE_EXCHANGES = {"Binance", "OKX", "Bybit"}

def _fetch_vol_weighted_funding(asset: str) -> float:
    """Returns live equal-weighted average funding rate across Binance/OKX/Bybit
    (stablecoin-margin). Refreshed every 30 minutes.
    Positive = longs paying shorts (crowded long). Negative = shorts paying longs."""
    cache_key = f"fr_vol_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _FR_VOL_CACHE_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/futures/funding-rate/exchange-list",
            headers=_HEADERS,
            params={"symbol": asset.upper()},
            timeout=6,
        )
        data = r.json().get("data") or []
        # Find the entry for this asset
        entry = next((x for x in data if x.get("symbol", "").upper() == asset.upper()), None)
        if entry is None:
            val = 0.0
        else:
            rates = [
                float(x["funding_rate"])
                for x in entry.get("stablecoin_margin_list", [])
                if x.get("exchange") in _FR_LIVE_EXCHANGES and x.get("funding_rate") is not None
            ]
            val = sum(rates) / len(rates) if rates else 0.0
    except Exception:
        val = 0.0

    _CACHE[cache_key] = (val, now)
    return val


def _fetch_stablecoin_oi_pct_change_4h(asset: str) -> float:
    """Returns most recent completed 4h bar's stablecoin-margin OI % change (open→close).
    >2% = leveraged longs crowding in → OTM YES risky; >1% on NO side blocks NO."""
    cache_key = f"oi_stable4h_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _OI_STABLE_CACHE_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/futures/open-interest/aggregated-stablecoin-history",
            headers=_HEADERS,
            params={"symbol": asset.upper(), "interval": "4h", "limit": 3,
                    "exchange_list": "Binance,OKX,Bybit"},
            timeout=6,
        )
        data = r.json().get("data") or []
        # data[-1] = current incomplete bar, data[-2] = last completed bar
        if len(data) >= 2:
            bar = data[-2]
            o = float(bar.get("open",  0) or 0)
            c = float(bar.get("close", 0) or 0)
            pct = (c - o) / o * 100 if o != 0 else 0.0
        else:
            pct = 0.0
    except Exception:
        pct = 0.0

    _CACHE[cache_key] = (pct, now)
    return pct


def _fetch_taker_ratio_4h(asset: str) -> float:
    """Returns most recent completed 4h bar's spot taker buy/sell ratio
    across Binance/OKX/Bybit. <1.0 = sellers dominant."""
    cache_key = f"taker4h_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _TAKER_4H_CACHE_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/spot/aggregated-taker-buy-sell-volume/history",
            headers=_HEADERS,
            params={"symbol": asset.upper(), "interval": "4h", "limit": 3,
                    "exchange_list": _SPOT_EXCHANGES},
            timeout=6,
        )
        data = r.json().get("data") or []
        # data[-1] = current incomplete bar, data[-2] = last completed bar
        if len(data) >= 2:
            bar  = data[-2]
            buy  = float(bar.get("aggregated_buy_volume_usd",  0))
            sell = float(bar.get("aggregated_sell_volume_usd", 0))
            val  = buy / sell if sell > 0 else 1.0
        else:
            val = 1.0
    except Exception:
        val = 1.0

    _CACHE[cache_key] = (val, now)
    return val


def _fetch_futures_taker_4h(asset: str) -> tuple:
    """Returns (delta_4h, ratio_4h, cvd_12h) from aggregated futures taker volume
    (Binance+OKX+Bybit perps). Uses the last completed 4h bar; 12h = 3-bar rolling sum.
    delta = buy_usd - sell_usd; ratio = buy/sell; cvd_12h = sum of 3 bar deltas."""
    cache_key = f"fut_taker4h_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _FUTURES_TAKER_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/futures/aggregated-taker-buy-sell-volume/history",
            headers=_HEADERS,
            params={"symbol": asset.upper(), "interval": "4h", "limit": 5,
                    "exchange_list": _FUTURES_EXCHANGES},
            timeout=8,
        )
        data = r.json().get("data") or []
        # data[-1] = current (possibly incomplete) bar; data[-2] = last completed bar
        if len(data) < 2:
            result = (0.0, 1.0, 0.0)
        else:
            completed = data[:-1]   # exclude current incomplete bar
            last  = completed[-1]
            buy1  = float(last.get("aggregated_buy_volume_usd",  0))
            sell1 = float(last.get("aggregated_sell_volume_usd", 1))
            delta_4h  = buy1 - sell1
            ratio_4h  = buy1 / sell1 if sell1 > 0 else 1.0

            # 12h CVD = sum of last 3 completed bars' deltas
            last3 = completed[-3:] if len(completed) >= 3 else completed
            cvd_12h = sum(
                float(b.get("aggregated_buy_volume_usd", 0)) -
                float(b.get("aggregated_sell_volume_usd", 0))
                for b in last3
            )
            result = (round(delta_4h, 2), round(ratio_4h, 6), round(cvd_12h, 2))
    except Exception:
        result = (0.0, 1.0, 0.0)

    _CACHE[cache_key] = (result, now)
    return result


def _fetch_liquidation_4h(asset: str) -> tuple:
    """Returns (liq_ratio_4h, liq_total_4h) from aggregated futures liquidations
    (Binance+OKX+Bybit). Uses the last completed 4h bar.
    liq_ratio = long_usd / short_usd; >1 = more longs liquidated (price fell = bearish cascade).
    liq_total = total USD liquidated (long + short); spike >$60M = extreme event."""
    cache_key = f"liq4h_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _LIQ_CACHE_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/futures/liquidation/aggregated-history",
            headers=_HEADERS,
            params={"symbol": asset.upper(), "interval": "4h", "limit": 3,
                    "exchange_list": "Binance,OKX,Bybit"},
            timeout=8,
        )
        data = r.json().get("data") or []
        if len(data) >= 2:
            bar      = data[-2]   # last completed bar
            long_liq = float(bar.get("aggregated_long_liquidation_usd",  0))
            shrt_liq = float(bar.get("aggregated_short_liquidation_usd", 0))
            total    = long_liq + shrt_liq
            ratio    = long_liq / shrt_liq if shrt_liq > 0 else 1.0
        else:
            ratio, total = 1.0, 0.0
    except Exception:
        ratio, total = 1.0, 0.0

    result = (round(ratio, 4), round(total, 2))
    _CACHE[cache_key] = (result, now)
    return result


def _fetch_hl_whale_signals(asset: str) -> tuple:
    """Returns (hl_ls_ratio, hl_squeeze_idx) from Hyperliquid whale-position snapshot.
    hl_ls_ratio = total_long_value / total_short_value for positions >$1M on HL.
    hl_squeeze_idx = (short_liq_near - long_liq_near) / total_value; positive = short squeeze risk."""
    cache_key = "hl_whale_all"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _HL_SHADOW_TTL:
        all_results = _CACHE[cache_key][0]
        return all_results.get(asset.upper(), (1.0, 0.0))

    try:
        r = requests.get(f"{_BASE}/api/hyperliquid/whale-position",
                         headers=_HEADERS, timeout=8)
        rows = r.json().get("data") or []
    except Exception:
        rows = []

    all_results: dict = {}
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        buckets[sym].append(row)

    for sym, positions in buckets.items():
        total_long = total_short = 0.0
        short_liq_near = long_liq_near = 0.0
        for p in positions:
            try:
                size  = float(p.get("position_size", 0))
                val   = float(p.get("position_value_usd", 0))
                mark  = float(p.get("mark_price", 0))
                liq   = float(p.get("liq_price", 0))
            except (TypeError, ValueError):
                continue
            if mark <= 0:
                continue
            dist = (liq - mark) / mark  # positive = liq above mark (short side)
            if size > 0:
                total_long += val
                if -0.10 < dist < 0:   # long near liq (liq below mark)
                    long_liq_near += val
            else:
                total_short += val
                if 0 < dist < 0.10:    # short near liq (liq above mark)
                    short_liq_near += val
        total_val = total_long + total_short
        ls_ratio     = total_long / total_short if total_short > 0 else 1.0
        squeeze_idx  = (short_liq_near - long_liq_near) / total_val if total_val > 0 else 0.0
        all_results[sym] = (round(ls_ratio, 4), round(squeeze_idx, 4))

    _CACHE[cache_key] = (all_results, now)
    return all_results.get(asset.upper(), (1.0, 0.0))


def _fetch_exchange_liq_signals(asset: str) -> tuple:
    """Returns (hl_liq_ratio_4h, all_liq_ratio_1h, all_liq_ratio_4h) from
    /api/futures/liquidation/exchange-list. Rolling windows (not bar-aligned)."""
    cache_key = f"exch_liq_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _EXCH_LIQ_TTL:
        return _CACHE[cache_key][0]

    def _liq_ratio(rows: list, exchange: str) -> float:
        row = next((r for r in rows if r.get("exchange") == exchange), None)
        if not row:
            return 1.0
        long_l = float(row.get("longLiquidation_usd", 0))
        shrt_l = float(row.get("shortLiquidation_usd", 0))
        return round(long_l / shrt_l if shrt_l > 0 else 1.0, 4)

    try:
        r1h = requests.get(f"{_BASE}/api/futures/liquidation/exchange-list",
                           headers=_HEADERS,
                           params={"symbol": asset.upper(), "range": "1h"},
                           timeout=6)
        data_1h = r1h.json().get("data") or []
        r4h = requests.get(f"{_BASE}/api/futures/liquidation/exchange-list",
                           headers=_HEADERS,
                           params={"symbol": asset.upper(), "range": "4h"},
                           timeout=6)
        data_4h = r4h.json().get("data") or []
    except Exception:
        data_1h, data_4h = [], []

    hl_liq_ratio_4h  = _liq_ratio(data_4h, "Hyperliquid")
    all_liq_ratio_1h = _liq_ratio(data_1h, "All")
    all_liq_ratio_4h = _liq_ratio(data_4h, "All")

    result = (hl_liq_ratio_4h, all_liq_ratio_1h, all_liq_ratio_4h)
    _CACHE[cache_key] = (result, now)
    return result


def _fetch_spot_cb_ratio_4h(asset: str) -> float:
    """Returns most recent completed 4h bar's spot taker buy/sell ratio across
    Binance+Coinbase+OKX. >1.05 = buyers dominant (blocks BTC/ETH NO); <0.90 = sellers dominant."""
    cache_key = f"spot_cb4h_{asset}"
    now = time.monotonic()
    if cache_key in _CACHE and now - _CACHE[cache_key][1] < _SPOT_CB_CACHE_TTL:
        return _CACHE[cache_key][0]

    try:
        r = requests.get(
            f"{_BASE}/api/spot/aggregated-taker-buy-sell-volume/history",
            headers=_HEADERS,
            params={"symbol": asset.upper(), "interval": "4h", "limit": 3,
                    "exchange_list": _SPOT_CB_EXCHANGES},
            timeout=6,
        )
        data = r.json().get("data") or []
        if len(data) >= 2:
            bar  = data[-2]  # last completed bar (data[-1] = current incomplete)
            buy  = float(bar.get("aggregated_buy_volume_usd",  0))
            sell = float(bar.get("aggregated_sell_volume_usd", 0))
            val  = buy / sell if sell > 0 else 1.0
        else:
            val = 1.0
    except Exception:
        val = 1.0

    _CACHE[cache_key] = (val, now)
    return val


def fetch_coinglass_signals(asset: str) -> Optional[CoinGlassSignals]:
    """
    Fetch all available CoinGlass Hobbyist signals for the given asset.
    Returns None only if all fetches fail. Individual components default to
    neutral if their specific endpoint fails.
    """
    asset = asset.upper()

    flow_1d, flow_1d_pct, flow_7d_pct, flow_score, flow_label = _fetch_exchange_flows(asset)
    oi_usd, oi_chg_24h, vol_24h, opts_score = _fetch_options(asset)
    buy_usd, sell_usd, taker_ratio = _fetch_spot_taker(asset)
    fg_val, fg_trend, fg_regime, fg_score = _fetch_fear_greed()
    fr_vol               = _fetch_vol_weighted_funding(asset)
    oi_stable_pct4h      = _fetch_stablecoin_oi_pct_change_4h(asset)
    taker_ratio_4h       = _fetch_taker_ratio_4h(asset)
    fut_delta, fut_ratio, fut_cvd12h = _fetch_futures_taker_4h(asset)
    spot_cb_ratio_4h     = _fetch_spot_cb_ratio_4h(asset)
    liq_ratio_4h, liq_total_4h = _fetch_liquidation_4h(asset)
    hl_ls_ratio, hl_squeeze_idx = _fetch_hl_whale_signals(asset)
    hl_liq_ratio_4h, all_liq_ratio_1h, all_liq_ratio_4h = _fetch_exchange_liq_signals(asset)

    composite = max(-2, min(2, flow_score + fg_score))

    if composite >= 2:
        lbl = f"BULLISH++ | {flow_label} | F&G={fg_val:.0f} ({fg_regime})"
    elif composite == 1:
        lbl = f"bullish+ | {flow_label} | F&G={fg_val:.0f} ({fg_regime})"
    elif composite == -1:
        lbl = f"bearish- | {flow_label} | F&G={fg_val:.0f} ({fg_regime})"
    elif composite <= -2:
        lbl = f"BEARISH-- | {flow_label} | F&G={fg_val:.0f} ({fg_regime})"
    else:
        lbl = f"neutral | {flow_label} | F&G={fg_val:.0f} ({fg_regime})"

    return CoinGlassSignals(
        exchange_flow_1d=flow_1d,
        exchange_flow_1d_pct=flow_1d_pct,
        exchange_flow_7d_pct=flow_7d_pct,
        exchange_flow_score=flow_score,
        exchange_flow_label=flow_label,
        options_oi_usd=oi_usd,
        options_oi_change_24h=oi_chg_24h,
        options_vol_usd_24h=vol_24h,
        options_score=opts_score,
        spot_buy_usd=buy_usd,
        spot_sell_usd=sell_usd,
        spot_taker_ratio=taker_ratio,
        fg_value=fg_val,
        fg_trend=fg_trend,
        fg_regime=fg_regime,
        fg_score=fg_score,
        fr_vol_1d=fr_vol,
        oi_stable_pct_4h=oi_stable_pct4h,
        taker_ratio_4h=taker_ratio_4h,
        futures_delta_4h=fut_delta,
        futures_ratio_4h=fut_ratio,
        futures_cvd_12h=fut_cvd12h,
        spot_cb_ratio_4h=spot_cb_ratio_4h,
        liq_ratio_4h=liq_ratio_4h,
        liq_total_4h=liq_total_4h,
        hl_ls_ratio=hl_ls_ratio,
        hl_squeeze_idx=hl_squeeze_idx,
        hl_liq_ratio_4h=hl_liq_ratio_4h,
        all_liq_ratio_1h=all_liq_ratio_1h,
        all_liq_ratio_4h=all_liq_ratio_4h,
        composite_score=composite,
        label=lbl,
    )


if __name__ == "__main__":
    for _asset in ("BTC", "ETH", "SOL"):
        print(f"\n── {_asset} ──")
        s = fetch_coinglass_signals(_asset)
        if s:
            print(f"  Exchange flow: {s.exchange_flow_1d:+.1f} ({s.exchange_flow_1d_pct:+.3f}%/day, {s.exchange_flow_7d_pct:+.3f}%/7d)  score={s.exchange_flow_score:+d}")
            print(f"  Options OI:   ${s.options_oi_usd/1e9:.1f}B  24h={s.options_oi_change_24h:+.2f}%  vol=${s.options_vol_usd_24h/1e9:.1f}B  score={s.options_score:+d}")
            print(f"  Spot taker:   buy=${s.spot_buy_usd/1e9:.2f}B  sell=${s.spot_sell_usd/1e9:.2f}B  ratio={s.spot_taker_ratio:.3f}")
            print(f"  Fear & Greed: {s.fg_value:.0f} ({s.fg_regime})  trend={s.fg_trend}  score={s.fg_score:+d}")
            print(f"  Fr live:      {s.fr_vol_1d:+.6f} (Binance/OKX/Bybit avg)  oi_stable_chg_4h={s.oi_stable_pct_4h:+.2f}%  taker_4h={s.taker_ratio_4h:.3f}")
            print(f"  Fut CVD:      delta_4h={s.futures_delta_4h/1e6:+.0f}M  ratio_4h={s.futures_ratio_4h:.4f}  cvd_12h={s.futures_cvd_12h/1e6:+.0f}M")
            print(f"  Spot CVD(CB): ratio_4h={s.spot_cb_ratio_4h:.4f}  (Binance+Coinbase+OKX; >1.05 blocks BTC/ETH NO)")
            print(f"  Liq 4h:       ratio={s.liq_ratio_4h:.3f}  total=${s.liq_total_4h/1e6:.1f}M  (>5 boosts BTC NO; <0.7 blocks BTC/SOL NO)")
            print(f"  HL whale:     ls_ratio={s.hl_ls_ratio:.3f}  squeeze_idx={s.hl_squeeze_idx:+.4f}  [shadow]")
            print(f"  HL liq 4h:    hl_ratio={s.hl_liq_ratio_4h:.3f}  all_1h={s.all_liq_ratio_1h:.3f}  all_4h={s.all_liq_ratio_4h:.3f}  [shadow]")
            print(f"  Composite:    {s.composite_score:+d}  → {s.label}")
