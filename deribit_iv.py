"""
deribit_iv.py — Fetch Deribit DVOL (30-day implied volatility index) for BTC and ETH.

Deribit DVOL is the dominant liquid-market IV benchmark for crypto options.
It is more reliable than back-computing IV from Kalshi prices, which are small
markets with OTM/ITM pricing distortions.

Usage:
    from deribit_iv import fetch_dvol, dvol_to_sigma_per_min, dvol_to_sigma_tau

    dvol = fetch_dvol("BTC")         # e.g. 0.65 for 65% annualized IV
    sig_per_min = dvol_to_sigma_per_min(dvol)
    sig_tau = dvol_to_sigma_tau(dvol, tau_minutes=45)

SOL: Deribit does not publish a DVOL index for SOL — returns None (caller falls back).
"""

import math
import time
from typing import Optional

import requests

_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_MINUTES_PER_YEAR = 365 * 24 * 60

_CACHE: dict[str, tuple[float, float]] = {}  # asset → (dvol_decimal, fetch_time)
_CACHE_TTL = 300  # seconds (5 minutes)

_SUPPORTED_ASSETS = {"BTC", "ETH"}


def fetch_dvol(asset: str, timeout: float = 5.0) -> Optional[float]:
    """
    Return current Deribit DVOL as an annualized IV decimal (e.g. 0.65 = 65%).

    Caches for 5 minutes. Returns None on any error or for unsupported assets (SOL).
    """
    asset = asset.upper()
    if asset not in _SUPPORTED_ASSETS:
        return None

    now = time.monotonic()
    if asset in _CACHE:
        cached_val, cached_time = _CACHE[asset]
        if now - cached_time < _CACHE_TTL:
            return cached_val

    try:
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - 60_000  # 1 minute window — we just want the latest bar

        resp = requests.get(
            _DVOL_URL,
            params={
                "currency": asset,
                "resolution": "60",       # 1-minute resolution
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        result = data.get("result", {})
        rows = result.get("data", [])  # each row: [timestamp_ms, open, high, low, close]
        if not rows:
            return None

        # Use the close of the most recent bar
        latest_close = float(rows[-1][4])  # index 4 = close
        if latest_close <= 0:
            return None

        dvol_decimal = latest_close / 100.0
        _CACHE[asset] = (dvol_decimal, now)
        return dvol_decimal

    except Exception:
        return None


def dvol_to_sigma_per_min(dvol_decimal: float) -> float:
    """Convert annualized IV decimal to per-minute sigma (for use in sigma_tau = σ·√τ)."""
    return dvol_decimal / math.sqrt(_MINUTES_PER_YEAR)


def dvol_to_sigma_tau(dvol_decimal: float, tau_minutes: float) -> float:
    """Convert annualized IV decimal to sigma_tau for a given horizon in minutes."""
    return dvol_to_sigma_per_min(dvol_decimal) * math.sqrt(tau_minutes)
