"""
coinalyze_liq.py — Liquidation cascade + positioning signal from Coinalyze API.

Fetches the most recent 15-minute liquidation history, long/short ratio, and
open interest for BTC, ETH, and SOL from Binance perpetual futures.

Signal: liq_score (int, -2 to +2)
  +2: strong short squeeze (short liqs dominant + crowd heavily short)
  +1: mild bullish liquidation pressure
   0: neutral
  -1: mild bearish liquidation pressure
  -2: strong long cascade (long liqs dominant + crowd heavily long)

liq_bias convention (float, -1 to +1):
  +1.0 = all recent liquidations are shorts (pure short squeeze, bullish)
  -1.0 = all recent liquidations are longs (pure long cascade, bearish)

oi_chg_pct: % change in open interest over the last completed bar.
  Positive = OI expanding (new positions opening); negative = OI shrinking (closing).

SOL: attempted via SOLUSDT_PERP.A; returns None if Coinalyze has no data.

API key: set COINALYZE_API_KEY env var, or falls back to module default.
"""

import os
import time
from typing import Optional, NamedTuple

import requests

_BASE = "https://api.coinalyze.net/v1"
_API_KEY = os.environ.get("COINALYZE_API_KEY", "d5841821-3f45-4e5f-9ee7-d2779d2fb01b")

_SYMBOLS = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}

_CACHE: dict[str, tuple["LiqSignal", float]] = {}
_CACHE_TTL = 300  # 5 minutes

# Thresholds for scoring
_LIQ_BIAS_STRONG = 0.60   # |liq_bias| >= this → ±1 from liq direction
_LS_CROWD_THRESH = 65.0   # ls_long_pct or ls_short_pct >= this → ±1 from positioning


class LiqSignal(NamedTuple):
    liq_bias: float      # -1 to +1; positive = bullish (short squeeze), negative = bearish (long cascade)
    ls_long_pct: float   # % of open positions that are long (0–100)
    ls_short_pct: float  # % of open positions that are short (0–100)
    liq_score: int       # -2 to +2 summary score (positive = bullish)
    label: str           # human-readable summary for logging
    oi_chg_pct: float    # open interest % change over last completed bar (+= expanding, -= shrinking)


def fetch_liq_signal(asset: str, timeout: float = 6.0) -> Optional[LiqSignal]:
    """
    Fetch liquidation cascade + positioning signal. Cached for 5 minutes.
    Returns None on API error or for unsupported assets (SOL).
    """
    asset = asset.upper()
    sym = _SYMBOLS.get(asset)
    if sym is None:
        return None

    now = time.monotonic()
    if asset in _CACHE:
        sig, ts = _CACHE[asset]
        if now - ts < _CACHE_TTL:
            return sig

    now_unix = int(time.time())
    start_unix = now_unix - 4 * 900  # 4 x 15-min bars back (enough for 2 completed bars)

    try:
        params_base = {
            "symbols": sym,
            "interval": "15min",
            "from": start_unix,
            "to": now_unix,
            "api_key": _API_KEY,
        }

        r_liq = requests.get(f"{_BASE}/liquidation-history",    params=params_base, timeout=timeout)
        r_ls  = requests.get(f"{_BASE}/long-short-ratio-history", params=params_base, timeout=timeout)
        r_oi  = requests.get(f"{_BASE}/open-interest-history",  params=params_base, timeout=timeout)

        r_liq.raise_for_status()
        r_ls.raise_for_status()
        r_oi.raise_for_status()

        liq_hist = r_liq.json()
        ls_hist  = r_ls.json()
        oi_hist  = r_oi.json()

        liq_rows = liq_hist[0]["history"] if liq_hist else []
        ls_rows  = ls_hist[0]["history"]  if ls_hist  else []
        oi_rows  = oi_hist[0]["history"]  if oi_hist  else []

        if not liq_rows or not ls_rows:
            return None

        # Use the most recent completed 15-min bar for both signals
        latest_liq = liq_rows[-1]
        long_liq  = float(latest_liq["l"])   # long positions liquidated (bearish if high)
        short_liq = float(latest_liq["s"])   # short positions liquidated (bullish if high)
        total_liq = long_liq + short_liq

        # liq_bias: +1 = all short liquidations (squeeze, bullish), -1 = all long liqs (cascade, bearish)
        liq_bias = (short_liq - long_liq) / total_liq if total_liq > 0.001 else 0.0

        latest_ls = ls_rows[-1]
        ls_long_pct  = float(latest_ls["l"])   # % of positions that are long
        ls_short_pct = float(latest_ls["s"])   # % of positions that are short

        # OI delta: % change from previous completed bar to latest completed bar
        oi_chg_pct = 0.0
        if len(oi_rows) >= 2:
            oi_prev = float(oi_rows[-2]["o"])
            oi_last = float(oi_rows[-1]["o"])
            if oi_prev > 0:
                oi_chg_pct = (oi_last - oi_prev) / oi_prev * 100.0

        # Score: each component contributes ±1, clamped to [-2, +2]
        score = 0

        # Liquidation direction signal
        if liq_bias >= _LIQ_BIAS_STRONG:
            score += 1   # short squeeze active → bullish
        elif liq_bias <= -_LIQ_BIAS_STRONG:
            score -= 1   # long cascade active → bearish

        # Positioning vulnerability signal
        if ls_short_pct >= _LS_CROWD_THRESH:
            score += 1   # crowd heavily short → fragile to squeeze → bullish
        elif ls_long_pct >= _LS_CROWD_THRESH:
            score -= 1   # crowd heavily long → fragile to cascade → bearish

        score = max(-2, min(2, score))

        if score >= 2:
            label = "SQUEEZE++"
        elif score == 1:
            label = "squeeze+"
        elif score == -1:
            label = "cascade-"
        elif score <= -2:
            label = "CASCADE--"
        else:
            label = "neutral"

        sig = LiqSignal(
            liq_bias=liq_bias,
            ls_long_pct=ls_long_pct,
            ls_short_pct=ls_short_pct,
            liq_score=score,
            label=label,
            oi_chg_pct=oi_chg_pct,
        )
        _CACHE[asset] = (sig, now)
        return sig

    except Exception:
        return None
