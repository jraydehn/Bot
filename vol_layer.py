"""
vol_layer.py — Volatility regime prediction layer.

Computes a vol_regime_factor in [0.60, 1.40] from five independently validated
signals. This factor multiplies the blended sigma input to estimate_probability(),
inflating it in high-vol regimes (making OTM strikes more reachable) and deflating
it in low-vol regimes (restricting viable strikes to ITM/near-ATM).

Validated against 19,947 hours of BTCUSDT 1h data (Jan 2024 – Apr 2026).
Each signal vote threshold was chosen based on the '>0.5% next-hour move' frequency
vs 20.3% baseline.

Usage:
    from vol_layer import compute_vol_regime_factor
    factor = compute_vol_regime_factor(df_1h_recent, df_1m_recent)
    vol_adj = vol_blended * factor
"""

import numpy as np
import pandas as pd


# Each signal contributes ±1 vote (or 0 for neutral).
# Combined score maps linearly to a factor: each vote = 8% adjustment.
# Score range: -5 to +5 → factor range: 0.60 to 1.40
VOL_VOTE_STEP = 0.08
VOL_FACTOR_MIN = 0.60
VOL_FACTOR_MAX = 1.40

# Asset-specific signal thresholds.
# Validated against 19,947 hours of each asset's 1h OHLCV data.
# ETH/SOL are more volatile than BTC — their rv_6h, ATR, and vol thresholds are lower.
_VOL_CONFIGS = {
    "BTC": {
        "atr_ratio_hi": 1.50, "atr_ratio_lo": 0.75,
        "abs_z_hi":     2.00, "abs_z_lo":     0.50,
        "vol_ratio_hi": 3.00, "vol_ratio_lo":  0.30,
        "vwap_dev_hi":  0.010,"vwap_dev_lo":   0.002,
        "rv_6h_hi":     0.30, "rv_6h_lo":      0.10,
    },
    "ETH": {
        # ATR: >1.25 → +15% HIGH (delta +10.2%); <0.75 → LOW (delta -15.5%)
        # rv_6h: >0.10 → HIGH (delta +8.6%); <0.05 → LOW (delta -20.0%)
        # vol_ratio: >2.0 → HIGH (delta +8.1%); <0.30 → LOW (delta -5.5%)
        # VWAP dev: >1.0% → HIGH; <0.2% → LOW (same as BTC)
        "atr_ratio_hi": 1.25, "atr_ratio_lo": 0.75,
        "abs_z_hi":     2.00, "abs_z_lo":     0.50,
        "vol_ratio_hi": 2.00, "vol_ratio_lo":  0.30,
        "vwap_dev_hi":  0.010,"vwap_dev_lo":   0.002,
        "rv_6h_hi":     0.10, "rv_6h_lo":      0.05,
    },
    "SOL": {
        # ATR: >1.25 → HIGH (delta +10.2%); <0.75 → LOW (delta -17.7%)
        # rv_6h: >0.15 → HIGH (delta +12.6%); <0.08 → LOW (delta -8.3%)
        # vol_ratio: >2.0 → HIGH (delta +6.5%); <0.30 → LOW (delta -5.1%)
        # VWAP dev: >2.0% → HIGH (delta +8.7%); <0.50% → LOW (delta -6.4%)
        # |z_score|: >1.50 → HIGH (delta +6.9%); LOW threshold weak — kept at 0.5
        "atr_ratio_hi": 1.25, "atr_ratio_lo": 0.75,
        "abs_z_hi":     1.50, "abs_z_lo":     0.50,
        "vol_ratio_hi": 2.00, "vol_ratio_lo":  0.30,
        "vwap_dev_hi":  0.020,"vwap_dev_lo":   0.005,
        "rv_6h_hi":     0.15, "rv_6h_lo":      0.08,
    },
}


def compute_vol_regime_factor(df_1h: pd.DataFrame, df_1m: pd.DataFrame,
                               asset: str = "BTC") -> tuple:
    """
    Compute volatility regime factor from five signals.

    Args:
        df_1h : Recent 1h OHLCV bars (need at least 48 bars for ATR/6h vol).
                Must have columns: open, high, low, close, volume.
        df_1m : Recent 1m bars (need at least 60 bars for 6h vol estimate).
                Must have columns: close, volume.

    Returns:
        (factor, score, details) where:
          factor  : float in [0.60, 1.40] — multiply blended sigma by this
          score   : int in [-5, +5] — raw vote sum
          details : dict of individual signal values and votes for diagnostics
    """
    df_1h = df_1h.copy()
    df_1h.columns = df_1h.columns.str.lower()
    df_1m = df_1m.copy()
    df_1m.columns = df_1m.columns.str.lower()

    cfg = _VOL_CONFIGS.get(asset.upper(), _VOL_CONFIGS["BTC"])
    votes = {}
    details = {}

    # ── Signal 1: ATR ratio (current ATR / 24h mean ATR) ─────────────────────
    try:
        c = df_1h["close"]
        h = df_1h["high"]
        l = df_1h["low"]
        cp = c.shift(1)
        tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
        atr = tr.ewm(com=13, adjust=False).mean()
        atr_24h_mean = atr.rolling(24).mean()
        atr_ratio = float(atr.iloc[-1] / atr_24h_mean.iloc[-1]) if atr_24h_mean.iloc[-1] > 0 else 1.0
        if atr_ratio > cfg["atr_ratio_hi"]:
            v = +1
        elif atr_ratio < cfg["atr_ratio_lo"]:
            v = -1
        else:
            v = 0
        votes["atr_ratio"] = v
        details["atr_ratio"] = round(atr_ratio, 3)
    except Exception:
        votes["atr_ratio"] = 0
        details["atr_ratio"] = None

    # ── Signal 2: |z_score| of current 1h move ───────────────────────────────
    try:
        log_ret = np.log(c / c.shift(1))
        roll_std = log_ret.rolling(24).std()
        z = float(log_ret.iloc[-1] / roll_std.iloc[-1]) if roll_std.iloc[-1] > 0 else 0.0
        abs_z = abs(z)
        if abs_z > cfg["abs_z_hi"]:
            v = +1
        elif abs_z < cfg["abs_z_lo"]:
            v = -1
        else:
            v = 0
        votes["abs_z"] = v
        details["abs_z"] = round(abs_z, 3)
    except Exception:
        votes["abs_z"] = 0
        details["abs_z"] = None

    # ── Signal 3: Volume ratio (current bar vol / 20-bar MA) ─────────────────
    try:
        vol = df_1h["volume"]
        vol_ma = vol.rolling(20).mean()
        vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
        if vol_ratio > cfg["vol_ratio_hi"]:
            v = +1
        elif vol_ratio < cfg["vol_ratio_lo"]:
            v = -1
        else:
            v = 0
        votes["vol_ratio"] = v
        details["vol_ratio"] = round(vol_ratio, 3)
    except Exception:
        votes["vol_ratio"] = 0
        details["vol_ratio"] = None

    # ── Signal 4: |VWAP deviation| ───────────────────────────────────────────
    try:
        date_1m = df_1m.index.normalize()
        tpv = df_1m["close"] * df_1m["volume"]
        cum_tpv = tpv.groupby(date_1m).cumsum()
        cum_vol = df_1m["volume"].groupby(date_1m).cumsum()
        vwap = (cum_tpv / cum_vol.replace(0, np.nan)).iloc[-1]
        spot_1m = float(df_1m["close"].iloc[-1])
        vwap_dev = abs((spot_1m - vwap) / vwap) if vwap > 0 else 0.0
        if vwap_dev > cfg["vwap_dev_hi"]:
            v = +1
        elif vwap_dev < cfg["vwap_dev_lo"]:
            v = -1
        else:
            v = 0
        votes["vwap_dev"] = v
        details["vwap_dev_pct"] = round(vwap_dev * 100, 4)
    except Exception:
        votes["vwap_dev"] = 0
        details["vwap_dev_pct"] = None

    # ── Signal 5: 6h realized vol vs threshold ───────────────────────────────
    try:
        lr_1m = np.log(df_1m["close"] / df_1m["close"].shift(1))
        rv_6h = float(lr_1m.rolling(360).std().iloc[-1]) * 100  # as % per 1m bar
        if rv_6h > cfg["rv_6h_hi"]:
            v = +1
        elif rv_6h < cfg["rv_6h_lo"]:
            v = -1
        else:
            v = 0
        votes["rv_6h"] = v
        details["rv_6h_pct"] = round(rv_6h, 4)
    except Exception:
        votes["rv_6h"] = 0
        details["rv_6h_pct"] = None

    # ── Combine ───────────────────────────────────────────────────────────────
    score = sum(votes.values())
    factor = 1.0 + score * VOL_VOTE_STEP
    factor = max(VOL_FACTOR_MIN, min(VOL_FACTOR_MAX, factor))

    details["votes"] = votes
    details["score"] = score
    details["factor"] = round(factor, 4)

    return factor, score, details
