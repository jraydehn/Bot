"""
eth_p_up_v1_model.py — ETH honest p_up model, SHADOW-ONLY inference.

Model: reform_results/eth_pup_rebuild_20260706/eth_p_up_v1_20260706.pkl
(29 features = A16 lag-corrected ETH price/tech + C13 alt-coin returns/
dominance/session interactions). Cross-asset BTC/SOL (B) and intra-hour
1m microstructure (M) were tested and REJECTED for ETH -- a genuinely
different winning shape than BTC's A+B+M+Cs, NOT a template copy.
Honest walk-forward OOS: AUC 0.5476, weekly-mean 0.5561, IC t=22.6,
positive every year 2021-2026. Output range ~[0.447, 0.577] -- narrow,
same caveat as BTC v3: do not feed into wide legacy fire-zone thresholds.

Feature construction mirrors reform_results/eth_pup_rebuild_20260706/
s1_build_dataset.py exactly (same crude standalone composite_trend/rev/
p_up proxy used there -- NOT btc_p_up_v3_model.py's BTC-calibrated
composite_scorer lookup, which would not be valid for ETH):
  * every feature from COMPLETED bars only; row T = last completed 1h
    bar, decision time T+1h
  * 4h bars = plain resample of 1h (no lookahead: only uses bars <= T)
  * label convention: direction of bar T+1 close vs bar T close

Data sources:
  ETH 1h      : BinanceUS klines via live_signal.fetch_recent_candles
                (1,000 bars)
  BTC 1h      : same fetch (needed for eth_dom_1h/24h) -- BTC IS in
                ASSET_CONFIG so fetch_recent_candles(asset="BTC") is safe
  XRP/DOGE/ADA 1h : NOT in ASSET_CONFIG -- fetch_recent_candles would
                silently fall back to BTC's symbol if called with these
                asset names. Fetched via a DIRECT Binance API call by
                explicit symbol string instead (bypasses that lookup
                entirely). Disk parquet fallback if the API call fails.

Graceful degradation: any feature-group failure -> NaN for that group
(LGBM handles NaN natively); load/inference failure -> None. Never
raises into the scan loop; never read by any decision path.
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import train_btc_p_up_v2 as _T  # generic indicator helpers, asset-agnostic

_MODEL_PATH = _PROJ / "reform_results" / "eth_pup_rebuild_20260706" / "eth_p_up_v1_20260706.pkl"
_CACHE: "dict | None | str" = "unloaded"
NAN = float("nan")

_last = {"bar_ts": None, "p": None}

_ALT_SYMBOLS = {"xrp": "XRPUSDT", "doge": "DOGEUSDT", "ada": "ADAUSDT"}


def _load() -> "dict | None":
    global _CACHE
    if _CACHE != "unloaded":
        return _CACHE
    try:
        with open(_MODEL_PATH, "rb") as f:
            _CACHE = pickle.load(f)
    except Exception:
        _CACHE = None
    return _CACHE


# ── data acquisition helpers ──────────────────────────────────────────────

def _fetch_asset_configured(interval: str, bars: int, asset: str) -> "pd.DataFrame | None":
    """Safe ONLY for assets actually in live_signal.ASSET_CONFIG (BTC/ETH/SOL)."""
    try:
        from live_signal import fetch_recent_candles
        return fetch_recent_candles(interval, lookback_bars=bars, asset=asset)
    except Exception:
        return None


def _fetch_by_symbol(symbol: str, interval: str, bars: int) -> "pd.DataFrame | None":
    """Direct Binance US fetch by explicit symbol string -- for assets NOT in
    ASSET_CONFIG (XRP/DOGE/ADA). Does NOT go through fetch_recent_candles,
    which would silently substitute BTC's symbol for an unrecognized asset name."""
    try:
        r = requests.get("https://api.binance.us/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": bars}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def _disk_alt_1h(sym: str) -> "pd.DataFrame | None":
    """Freshest data/binanceus_{sym}_1h_*.parquet fallback."""
    try:
        best, best_end = None, None
        for f in sorted((_PROJ / "data").glob(f"binanceus_{sym}_1h_*.parquet")):
            d = pd.read_parquet(f)
            if len(d) < 200:
                continue
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            if best_end is None or d.index[-1] > best_end:
                best, best_end = d, d.index[-1]
        return best
    except Exception:
        return None


def _alt_close(tag: str, T: pd.Timestamp) -> "pd.Series | None":
    """XRP/DOGE/ADA 1h closes through bar T. Direct-symbol API primary, disk
    parquet fallback. None if freshest completed bar is staler than 2h vs T."""
    symbol = _ALT_SYMBOLS[tag]
    d = _fetch_by_symbol(symbol, "1h", 30)
    if d is None or len(d) < 26:
        d = _disk_alt_1h(symbol)
    if d is None or len(d) < 26:
        return None
    c = d["close"][d.index <= T]
    if len(c) < 26 or (T - c.index[-1]) > pd.Timedelta(hours=2):
        return None
    return c


# ── feature assembly (exact port of eth_pup_rebuild_20260706/s1_build_dataset.py) ──

def _assemble(T: pd.Timestamp, b1: pd.DataFrame, btc_c: "pd.Series | None",
              xrp_c: "pd.Series | None", doge_c: "pd.Series | None",
              ada_c: "pd.Series | None") -> dict:
    """Feature dict for bar T. b1 = completed ETH 1h bars ending at T."""
    feats = {}
    c1h = b1["close"]
    h1h = b1["high"]
    l1h = b1["low"]
    v1h = b1["volume"]

    # -- A: ETH 1h indicators (exact port of s1_build_dataset.py) ------------
    lr = np.log(c1h / c1h.shift(1))
    try:
        # manual span=14 EWM RSI (training: s1_build_dataset.py lines 49-52) --
        # NOT _T.rsi_series, which uses com=13 (a different, faster-decaying
        # smoothing factor) and was designed for BTC's model, not this one.
        delta1 = c1h.diff()
        gain1 = delta1.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss1 = (-delta1.clip(upper=0)).ewm(span=14, adjust=False).mean()
        feats["rsi_14"] = float((100 - 100 / (1 + gain1 / loss1.replace(0, np.nan))).iloc[-1])
        feats["macd_hist_1h"] = float(_T.macd_hist_series(c1h).iloc[-1])
        feats["bb_pct"] = float(_T.bb_pct_series(c1h).iloc[-1])
        feats["ema50_dist"] = float(_T.ema50_dist_series(c1h).iloc[-1])
        # rvol_1h = ratio of 24h to 168h (7d) rolling log-return volatility --
        # NOT a volume ratio (training: s1_build_dataset.py line 80).
        feats["rvol_1h"] = float((lr.rolling(24).std() / lr.rolling(168).std().replace(0, np.nan)).iloc[-1])
        ll14 = l1h.rolling(14).min(); hh14 = h1h.rolling(14).max()
        feats["stoch_k"] = float((((c1h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100).iloc[-1])
        ema9, ema21, ema55 = c1h.ewm(span=9, adjust=False).mean(), c1h.ewm(span=21, adjust=False).mean(), c1h.ewm(span=55, adjust=False).mean()
        ema_stack = np.sign(ema9 - ema21) + np.sign(ema21 - ema55)
        feats["ema_stack_bias"] = float(ema_stack.iloc[-1])
        stretch = ((c1h - ema9) / ema9).clip(-0.05, 0.05) * 20
        feats["ema_stretch_score"] = float(stretch.iloc[-1])
        vwap_num = (c1h * v1h).rolling(24).sum(); vwap_den = v1h.rolling(24).sum().replace(0, np.nan)
        vwap = vwap_num / vwap_den
        feats["vwap_distance_pct"] = float(((c1h - vwap) / vwap).iloc[-1])
        vwap_std = ((c1h - vwap) ** 2).rolling(24).mean() ** 0.5
        feats["vwap_stretch_score"] = float(((c1h - vwap) / vwap_std.replace(0, np.nan)).clip(-3, 3).iloc[-1])
        # chg_4h_atr is a PURE 1h-bar feature despite the name (training:
        # s1_build_dataset.py line 73-74) -- 4h-ago change over a 14h ATR,
        # NOT computed from actual 4h-resampled bars.
        atr = (h1h - l1h).rolling(14).mean()
        feats["chg_4h_atr"] = float(((c1h - c1h.shift(4)) / atr.replace(0, np.nan)).iloc[-1])
    except Exception:
        for k in ("rsi_14", "macd_hist_1h", "bb_pct", "ema50_dist", "rvol_1h", "stoch_k",
                  "ema_stack_bias", "ema_stretch_score", "vwap_distance_pct", "vwap_stretch_score",
                  "chg_4h_atr"):
            feats.setdefault(k, NAN)

    # -- A: ETH 4h indicators -- EXACT port of training's resample+shift(3)
    # quirk: shift(3) on an already-4h-spaced series moves by 3 PERIODS
    # (12 CALENDAR HOURS), not 3 hours despite the training comment's wording.
    # The model was trained on this exact transform; replicating "what the
    # comment meant" instead of what the code does would break parity.
    try:
        b4 = b1[b1.index <= T]
        c4h_full = b4["close"].resample("4h").last()
        c4h_shifted = c4h_full.shift(3)  # rsi_4h uses the PRICE-shifted series
        delta4 = c4h_shifted.diff()
        gain4 = delta4.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss4 = (-delta4.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi_4h_series = (100 - 100 / (1 + gain4 / loss4.replace(0, np.nan))).reindex(
            pd.date_range(c4h_full.index[0], T, freq="h", tz="UTC"), method="ffill")
        feats["rsi_4h"] = float(rsi_4h_series.iloc[-1])

        ll14_4h = c4h_full.rolling(14).min(); hh14_4h = c4h_full.rolling(14).max()
        stoch_k_4h_series = (((c4h_full - ll14_4h) / (hh14_4h - ll14_4h).replace(0, np.nan)) * 100).shift(3)
        stoch_k_4h_hourly = stoch_k_4h_series.reindex(
            pd.date_range(c4h_full.index[0], T, freq="h", tz="UTC"), method="ffill")
        feats["stoch_k_4h"] = float(stoch_k_4h_hourly.iloc[-1])
    except Exception:
        for k in ("stoch_k_4h", "rsi_4h"):
            feats[k] = NAN

    # -- A: crude standalone composite trend/rev/p_up (ETH-specific, NOT the
    #    BTC-calibrated composite_scorer -- matches training s1 exactly) -----
    try:
        ema9, ema21, ema55 = c1h.ewm(span=9, adjust=False).mean(), c1h.ewm(span=21, adjust=False).mean(), c1h.ewm(span=55, adjust=False).mean()
        trend_votes = (np.sign(ema9 - ema21) + np.sign(ema21 - ema55) +
                      np.sign(c1h - c1h.shift(4)) + np.sign(c1h - c1h.shift(24)))
        feats["composite_trend"] = float(trend_votes.iloc[-1])
        rsi14 = _T.rsi_series(c1h, 14)
        ll14 = l1h.rolling(14).min(); hh14 = h1h.rolling(14).max()
        stoch_k = ((c1h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100
        rev = ((rsi14 < 30).astype(int) - (rsi14 > 70).astype(int) +
              (stoch_k < 20).astype(int) - (stoch_k > 80).astype(int))
        feats["composite_rev"] = float(rev.iloc[-1])
        feats["composite_p_up"] = float(0.5 + 0.02 * np.clip(trend_votes.iloc[-1], -4, 4))
    except Exception:
        feats["composite_trend"] = NAN
        feats["composite_rev"] = NAN
        feats["composite_p_up"] = 0.5

    # -- C: alt-coin returns + ETH-relative dominance + session interactions -
    eth_r1 = c1h.pct_change(fill_method=None)
    eth_r24 = c1h.pct_change(24, fill_method=None)
    for tag, ac in (("xrp", xrp_c), ("doge", doge_c), ("ada", ada_c)):
        try:
            if ac is None:
                raise ValueError("alt missing")
            a = ac.reindex(b1.index).ffill()
            feats[f"{tag}_ret_1h"] = float(a.pct_change(fill_method=None).iloc[-1])
            feats[f"{tag}_ret_4h"] = float(a.pct_change(4, fill_method=None).iloc[-1])
        except Exception:
            feats[f"{tag}_ret_1h"] = NAN
            feats[f"{tag}_ret_4h"] = NAN
    try:
        alt_basket = np.mean([feats["xrp_ret_1h"], feats["doge_ret_1h"], feats["ada_ret_1h"]])
        feats["alt_basket_ret_1h"] = float(alt_basket) if alt_basket == alt_basket else NAN
    except Exception:
        feats["alt_basket_ret_1h"] = NAN
    try:
        if btc_c is None:
            raise ValueError("btc missing")
        b = btc_c.reindex(b1.index).ffill()
        btc_r1 = b.pct_change(fill_method=None)
        btc_r24 = b.pct_change(24, fill_method=None)
        feats["eth_dom_1h"] = float((eth_r1 - btc_r1).iloc[-1])
        feats["eth_dom_24h"] = float((eth_r24 - btc_r24).iloc[-1])
    except Exception:
        feats["eth_dom_1h"] = NAN
        feats["eth_dom_24h"] = NAN
    try:
        dec = T + pd.Timedelta(hours=1)
        us = 1.0 if 13 <= dec.hour < 21 else 0.0
        asia = 1.0 if 0 <= dec.hour < 8 else 0.0
        wknd = 1.0 if dec.dayofweek >= 5 else 0.0
        r1 = float(eth_r1.iloc[-1]); r24 = float(eth_r24.iloc[-1])
        feats["us_x_ret1h"] = us * r1
        feats["asia_x_ret1h"] = asia * r1
        feats["wknd_x_ret1h"] = wknd * r1
        feats["wknd_x_ret24h"] = wknd * r24
    except Exception:
        for k in ("us_x_ret1h", "asia_x_ret1h", "wknd_x_ret1h", "wknd_x_ret24h"):
            feats[k] = NAN

    return feats


# ── public API ─────────────────────────────────────────────────────────────

def compute_eth_p_up(df_1h: "pd.DataFrame | None" = None,
                     now: "datetime | None" = None) -> "float | None":
    """SHADOW-ONLY ETH p_up score. Cached per completed 1h bar.

    df_1h : runner's ETH 1h frame -- fallback if this module's own fetch fails.
    Returns float or None. Never raises.
    """
    try:
        pipe = _load()
        if pipe is None:
            return None
        now = now or datetime.now(timezone.utc)
        now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo \
            else pd.Timestamp(now, tz="UTC")
        T = now_ts.floor("h") - pd.Timedelta(hours=1)
        if _last["bar_ts"] == T and _last["p"] is not None:
            return _last["p"]

        b1 = _fetch_asset_configured("1h", 1000, "ETH")
        if b1 is None or len(b1) < 200:
            b1 = df_1h
        if b1 is None or len(b1) < 200:
            return None
        if b1.index.tz is None:
            b1 = b1.copy(); b1.index = b1.index.tz_localize("UTC")
        b1 = b1[b1.index <= T]
        if not len(b1) or (T - b1.index[-1]) > pd.Timedelta(hours=2):
            return None
        T = b1.index[-1]

        btc1h = _fetch_asset_configured("1h", 60, "BTC")
        if btc1h is not None and btc1h.index.tz is None:
            btc1h = btc1h.copy(); btc1h.index = btc1h.index.tz_localize("UTC")
        btc_c = btc1h["close"][btc1h.index <= T] if btc1h is not None else None
        xrp_c = _alt_close("xrp", T)
        doge_c = _alt_close("doge", T)
        ada_c = _alt_close("ada", T)

        feats = _assemble(T, b1, btc_c, xrp_c, doge_c, ada_c)
        vec = np.array([[feats.get(f, NAN) for f in pipe["features"]]], dtype=float)
        if np.isnan(vec).all():
            return None
        p = float(pipe["clf"].predict_proba(vec)[0, 1])
        _last["bar_ts"], _last["p"] = T, p
        return p
    except Exception:
        return None
