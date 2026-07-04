"""
btc_p_up_v3_model.py — BTC honest p_up rebuild (v3), SHADOW-ONLY inference.

Model: reform_results/pup_v2_rebuild_20260704/btc_p_up_v3_20260704.pkl
(35 features = A16 lag-corrected price/tech + B8 ETH/SOL lead-lag +
M7 intra-hour 1m microstructure + Cs4 session-return interactions).
Honest walk-forward OOS: AUC 0.5570, weekly-mean 0.5637, IC t=26.9,
positive every year 2021-2026. Output range is NARROW (~p05-p95
[0.43, 0.59]) — do NOT feed into the old ≤0.20/≥0.70 fire zones.

Feature construction EXACTLY mirrors the training builder
(reform_results/pup_v2_rebuild_20260704/s3_build_dataset.py):
  * every feature from COMPLETED bars only; row T = last completed 1h bar,
    decision time T+1h (i.e., "now" is inside bar T+1)
  * 4h bars = UTC-aligned resample of 1h, shifted +3h before backward merge
  * session VWAP with EXPANDING intraday std (no full-day lookahead)
  * rolling z-scores only (240h), pct_change(fill_method=None)
  * label convention at training: direction of bar T+1 close vs bar T close

Data sources:
  BTC 1h / 15m : BinanceUS klines via live_signal.fetch_recent_candles
                 (1,000 bars — enough for EMA/RSI convergence)
  ETH/SOL 1h   : BinanceUS klines (30 bars); FALLBACK freshest
                 data/binanceus_{ETH,SOL}USDT_1h_*.parquet; bars staler
                 than 2h vs bar T → the B features are NaN
  BTC 1m       : the live_1m frame passed by the runner (last completed
                 hour) merged over an internal incrementally-cached API
                 backfill (needed for the 240h rv60_z_10d baseline)

Graceful degradation: any feature-group failure → NaN for that group
(LGBM handles NaN natively); load/inference failure → None. This module
must NEVER raise into the scan loop and is NEVER read by any decision
path — logging only (deployment gate: shadow data must confirm the
2026-07-04 replay before any live use).
"""

import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import train_btc_p_up_v2 as _T
_T.CAL_FILE = _PROJ / "composite_calibration.json"   # absolute (CWD-proof)

_MODEL_PATH = _PROJ / "reform_results" / "pup_v2_rebuild_20260704" / "btc_p_up_v3_20260704.pkl"
_CACHE: "dict | None | str" = "unloaded"

NAN = float("nan")

# per-completed-hour result cache: recompute only when a new hour closes
_last = {"bar_ts": None, "p": None}
# incremental 1m history cache for rv60_z_10d (240h rolling baseline)
_m1_cache: "pd.DataFrame | None" = None

M_FEATURES = ["rv_60m", "rv60_z_10d", "upmin_frac", "ret_first45", "ret_last15",
              "maxdd_60m", "volskew_last20"]
B_FEATURES = ["eth_ret_1h", "sol_ret_1h", "eth_ret_4h", "sol_ret_4h",
              "spread_eth_1h", "spread_sol_1h", "spread_eth_24h", "spread_sol_24h"]


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

def _fetch(interval: str, bars: int, asset: str = "BTC") -> "pd.DataFrame | None":
    try:
        from live_signal import fetch_recent_candles
        return fetch_recent_candles(interval, lookback_bars=bars, asset=asset)
    except Exception:
        return None


def _disk_alt_1h(sym: str) -> "pd.DataFrame | None":
    """Freshest data/binanceus_{sym}_1h_*.parquet (fallback source for B)."""
    try:
        best, best_end = None, None
        for f in sorted((_PROJ / "data").glob(f"binanceus_{sym}_1h_*.parquet")):
            d = pd.read_parquet(f)
            if len(d) < 1000:
                continue
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            d = d[d.index >= pd.Timestamp("2000-01-01", tz="UTC")]
            if best_end is None or d.index[-1] > best_end:
                best, best_end = d, d.index[-1]
        return best
    except Exception:
        return None


def _alt_close(sym: str, T: pd.Timestamp) -> "pd.Series | None":
    """ETH/SOL 1h closes through bar T. API primary, disk parquet fallback.
    Returns None if freshest completed bar is staler than 2h vs T."""
    d = _fetch("1h", 40, asset=sym[:3])
    if d is None or len(d) < 26:
        d = _disk_alt_1h(sym)
    if d is None or len(d) < 26:
        return None
    c = d["close"][d.index <= T]          # completed-at-decision bars only
    if len(c) < 26 or (T - c.index[-1]) > pd.Timedelta(hours=2):
        return None
    return c


def _get_1m_frame(T: pd.Timestamp, live_1m: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """1m closes/volume covering [T-240h, T+1h): cached API backfill merged
    with the runner-passed live_1m frame (which supplies the newest bars)."""
    global _m1_cache
    lo = T - pd.Timedelta(hours=240)
    hi = T + pd.Timedelta(hours=1)
    parts = []
    if _m1_cache is not None:
        parts.append(_m1_cache)
    if live_1m is not None and len(live_1m):
        lm = live_1m[["open", "high", "low", "close", "volume"]].copy()
        if lm.index.tz is None:
            lm.index = lm.index.tz_localize("UTC")
        parts.append(lm[lm.index < hi])
    cur = pd.concat(parts).sort_index() if parts else None
    if cur is not None:
        cur = cur[~cur.index.duplicated(keep="last")]
        cur = cur[cur.index >= lo - pd.Timedelta(hours=2)]
    # backfill any missing span from the API (cold start: ~15 calls; then 0-1)
    try:
        import requests
        need_from = lo
        if cur is not None and len(cur):
            covered = cur.index[cur.index >= lo]
            # find earliest gap: expect one bar per minute from lo
            expect = pd.date_range(lo, hi - pd.Timedelta(minutes=1), freq="min", tz="UTC")
            missing = expect.difference(covered)
            if len(missing) == 0:
                return cur[(cur.index >= lo) & (cur.index < hi)]
            need_from = missing[0]
        rows, curms = [], int(need_from.timestamp() * 1000)
        hims = int(hi.timestamp() * 1000)
        calls = 0
        while curms < hims and calls < 16:
            r = requests.get("https://api.binance.us/api/v3/klines",
                             params={"symbol": "BTCUSDT", "interval": "1m",
                                     "startTime": curms, "limit": 1000}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            calls += 1
            if not batch:
                break
            rows.extend(batch)
            nxt = batch[-1][0] + 60_000
            if nxt <= curms:
                break
            curms = nxt
        if rows:
            fb = pd.DataFrame(rows).iloc[:, :6]
            fb.columns = ["open_time", "open", "high", "low", "close", "volume"]
            for c in ("open", "high", "low", "close", "volume"):
                fb[c] = fb[c].astype(float)
            fb.index = pd.to_datetime(fb.pop("open_time"), unit="ms", utc=True)
            cur = pd.concat([cur, fb]) if cur is not None else fb
            cur = cur[~cur.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass
    if cur is None or not len(cur):
        return None
    cur = cur[cur.index >= lo - pd.Timedelta(hours=2)]
    _m1_cache = cur
    return cur[(cur.index >= lo) & (cur.index < hi)]


# ── feature assembly (exact port of s3_build_dataset.py, trailing window) ──

def _assemble(T: pd.Timestamp, b1: pd.DataFrame, m15: "pd.DataFrame | None",
              eth_c: "pd.Series | None", sol_c: "pd.Series | None",
              m1: "pd.DataFrame | None") -> dict:
    """Feature dict for bar T. b1 = completed BTC 1h bars ending at T."""
    feats = {}
    c1h = b1["close"]

    # -- A: 1h indicators --------------------------------------------------
    try:
        feats["rsi_14"] = float(_T.rsi_series(c1h, 14).iloc[-1])
        feats["macd_hist_1h"] = float(_T.macd_hist_series(c1h).iloc[-1])
        feats["bb_pct"] = float(_T.bb_pct_series(c1h).iloc[-1])
        feats["ema50_dist"] = float(_T.ema50_dist_series(c1h).iloc[-1])
        feats["rvol_1h"] = float(
            (b1["volume"] / b1["volume"].rolling(24).mean().replace(0, np.nan)).iloc[-1])
        tp = (b1["high"] + b1["low"] + b1["close"]) / 3
        day = pd.Series(b1.index.date, index=b1.index)
        cum_tpv = (tp * b1["volume"]).groupby(day.values).cumsum()
        cum_vol = b1["volume"].groupby(day.values).cumsum()
        vwap = cum_tpv / cum_vol.replace(0, np.nan)
        dist = (b1["close"] - vwap) / vwap.replace(0, np.nan)
        feats["vwap_distance_pct"] = float(dist.iloc[-1])
        std_exp = tp.groupby(day.values).expanding().std().reset_index(level=0, drop=True)
        std_exp.index = b1.index
        z = dist / (std_exp / vwap.replace(0, np.nan)).replace(0, np.nan)
        feats["vwap_stretch_score"] = float(pd.cut(
            z, bins=[-np.inf, -2, -1, 1, 2, np.inf], labels=[2, 1, 0, -1, -2]
        ).astype(float).iloc[-1])
    except Exception:
        for k in ("rsi_14", "macd_hist_1h", "bb_pct", "ema50_dist", "rvol_1h",
                  "vwap_distance_pct", "vwap_stretch_score"):
            feats.setdefault(k, NAN)

    # -- A: 4h indicators (UTC-aligned resample, +3h lag, backward pick) ----
    try:
        b4 = pd.DataFrame({
            "open": b1["open"].resample("4h").first(),
            "high": b1["high"].resample("4h").max(),
            "low": b1["low"].resample("4h").min(),
            "close": b1["close"].resample("4h").last(),
            "volume": b1["volume"].resample("4h").sum(),
        }).dropna(subset=["close"])
        c4h = b4["close"]
        ind4 = pd.DataFrame({
            "stoch_k_4h": _T.stoch_k_series(b4["high"], b4["low"], c4h, 14),
            "rsi_4h": _T.rsi_series(c4h, 14),
            "chg_4h_atr": _T.chg_4h_atr_series(b4),
        }, index=b4.index)
        # relabel AFTER construction (constructor index would realign -> NaN);
        # +3h lag correction matches training (bar closes <= decision time)
        ind4.index = ind4.index + pd.Timedelta(hours=3)
        sel4 = ind4[ind4.index <= T]
        for k in ("stoch_k_4h", "rsi_4h", "chg_4h_atr"):
            feats[k] = float(sel4[k].iloc[-1]) if len(sel4) else NAN
    except Exception:
        b4 = None
        for k in ("stoch_k_4h", "rsi_4h", "chg_4h_atr"):
            feats[k] = NAN

    # -- A: composite trend (lag-corrected) / rev (causal) / p_up LUT -------
    try:
        cal = _T.load_calibration()
        _, rev_s, _ = _T.compute_composite_signals(b1, b4, cal)
        rev = float(rev_s.iloc[-1])
        c4h = b4["close"]
        rsi4 = _T.rsi_series(c4h, 14)
        macd4 = _T._ema(c4h, 12) - _T._ema(c4h, 26)
        sig4 = macd4.ewm(span=9, adjust=False).mean()
        bbm = c4h.rolling(20).mean(); bbs = c4h.rolling(20).std()
        bbp4 = (c4h - (bbm - 2 * bbs)) / (4 * bbs).replace(0, np.nan)
        sk4 = _T.stoch_k_series(b4["high"], b4["low"], c4h, 14)
        wr4 = -100 * (b4["high"].rolling(14).max() - c4h) / \
              (b4["high"].rolling(14).max() - b4["low"].rolling(14).min()).replace(0, np.nan)
        vr4 = b4["volume"] / b4["volume"].rolling(20).mean().replace(0, np.nan)
        trend4 = ((rsi4 > 55).astype(float) - (rsi4 < 45).astype(float)
                  + np.where(macd4 > sig4, 1.0, -1.0)
                  + (bbp4 > 0.80).astype(float) - (bbp4 < 0.20).astype(float)
                  + (sk4 > 80).astype(float) - (sk4 < 20).astype(float)
                  + (wr4 > -20).astype(float) - (wr4 < -80).astype(float)
                  + ((vr4 > 1.5) & (c4h > c4h.shift(1))).astype(float)
                  - ((vr4 > 1.5) & (c4h < c4h.shift(1))).astype(float)).clip(-6, 6)
        t4 = pd.Series(trend4.values, index=b4.index + pd.Timedelta(hours=3))
        t4 = t4[t4.index <= T]
        trend = float(t4.iloc[-1]) if len(t4) else NAN
        feats["composite_trend"] = trend
        feats["composite_rev"] = rev
        if trend == trend and rev == rev:
            e = (cal or {}).get(f"{int(round(trend))}_{int(round(rev))}")
            feats["composite_p_up"] = float(e["p_yes"]) if e and e.get("n", 0) >= 5 else 0.504
        else:
            feats["composite_p_up"] = 0.504
    except Exception:
        feats["composite_trend"] = NAN
        feats["composite_rev"] = NAN
        feats["composite_p_up"] = 0.504

    # -- A: 15m features (last bar with open <= T, within 60min) ------------
    try:
        d15 = m15[m15.index <= T]
        if len(d15) < 60 or (T - d15.index[-1]) > pd.Timedelta(minutes=60):
            raise ValueError("15m stale/short")
        c15 = d15["close"]
        feats["stoch_k"] = float(_T.stoch_k_series(d15["high"], d15["low"], c15, 14).iloc[-1])
        e9, e21, e50 = _T._ema(c15, 9), _T._ema(c15, 21), _T._ema(c15, 50)
        if (e9.iloc[-1] > e21.iloc[-1]) and (e21.iloc[-1] > e50.iloc[-1]) and (c15.iloc[-1] > e9.iloc[-1]):
            feats["ema_stack_bias"] = 1.0
        elif (e9.iloc[-1] < e21.iloc[-1]) and (e21.iloc[-1] < e50.iloc[-1]) and (c15.iloc[-1] < e9.iloc[-1]):
            feats["ema_stack_bias"] = -1.0
        else:
            feats["ema_stack_bias"] = 0.0
        e20 = _T._ema(c15, 20)
        stretch = (c15.iloc[-1] - e20.iloc[-1]) / e20.iloc[-1] if e20.iloc[-1] else NAN
        feats["ema_stretch_score"] = (1.0 if stretch <= -0.001 else
                                      -1.0 if stretch >= 0.001 else 0.0) \
            if stretch == stretch else NAN
    except Exception:
        for k in ("stoch_k", "ema_stack_bias", "ema_stretch_score"):
            feats[k] = NAN

    # -- B: ETH/SOL lead-lag -------------------------------------------------
    btc_r1 = c1h.pct_change(fill_method=None)
    btc_r24 = c1h.pct_change(24, fill_method=None)
    for tag, ac in (("eth", eth_c), ("sol", sol_c)):
        try:
            if ac is None:
                raise ValueError("alt missing")
            a = ac.reindex(b1.index)
            r1 = a.pct_change(fill_method=None)
            feats[f"{tag}_ret_1h"] = float(r1.iloc[-1])
            feats[f"{tag}_ret_4h"] = float(a.pct_change(4, fill_method=None).iloc[-1])
            feats[f"spread_{tag}_1h"] = float(btc_r1.iloc[-1] - r1.iloc[-1])
            feats[f"spread_{tag}_24h"] = float(
                btc_r24.iloc[-1] - a.pct_change(24, fill_method=None).iloc[-1])
        except Exception:
            for k in (f"{tag}_ret_1h", f"{tag}_ret_4h",
                      f"spread_{tag}_1h", f"spread_{tag}_24h"):
                feats[k] = NAN

    # -- M: intra-hour microstructure (1m bars of hour [T, T+1h)) -----------
    try:
        m = m1[(m1.index >= T - pd.Timedelta(hours=240)) & (m1.index < T + pd.Timedelta(hours=1))].copy()
        cur = m[(m.index >= T)]
        if len(cur) < 55:
            raise ValueError(f"only {len(cur)} 1m bars in hour")
        m["hr"] = m.index.floor("h")
        r1m = np.log(m["close"] / m["close"].shift(1))
        r1m[(m["hr"] != m["hr"].shift(1)).values] = np.nan
        m["_r2"] = r1m ** 2
        m["_up"] = (m["close"] > m["open"]).astype(float)
        mn = m.index.minute
        m["_c45"] = m["close"].where(mn == 44)
        m["_vl20"] = m["volume"].where(mn >= 40, 0.0)
        g = m.groupby("hr")
        m["_dd"] = (m["close"] / g["close"].cummax() - 1)
        agg = g.agg(rv2=("_r2", "sum"), upmin_frac=("_up", "mean"),
                    first_o=("open", "first"), last_c=("close", "last"),
                    c45=("_c45", "max"), maxdd_60m=("_dd", "min"),
                    v_tot=("volume", "sum"), v_last20=("_vl20", "sum"))
        rv = agg["rv2"] ** 0.5
        feats["rv_60m"] = float(rv.loc[T])
        feats["upmin_frac"] = float(agg.loc[T, "upmin_frac"])
        feats["ret_first45"] = float(np.log(agg.loc[T, "c45"] / agg.loc[T, "first_o"]))
        feats["ret_last15"] = float(np.log(agg.loc[T, "last_c"] / agg.loc[T, "c45"]))
        feats["maxdd_60m"] = float(agg.loc[T, "maxdd_60m"])
        vt = agg.loc[T, "v_tot"]
        feats["volskew_last20"] = float(agg.loc[T, "v_last20"] / vt) if vt else NAN
        lrv = np.log(rv.replace(0, np.nan))
        zz = (lrv - lrv.rolling(240).mean()) / lrv.rolling(240).std().replace(0, np.nan)
        feats["rv60_z_10d"] = float(zz.loc[T]) if T in zz.index else NAN
    except Exception:
        for k in M_FEATURES:
            feats[k] = NAN

    # -- Cs: session-return interactions (decision hour = T+1h) -------------
    try:
        dec = T + pd.Timedelta(hours=1)
        us = 1.0 if 13 <= dec.hour < 21 else 0.0
        asia = 1.0 if dec.hour < 8 else 0.0
        wknd = 1.0 if dec.dayofweek >= 5 else 0.0
        r1 = float(btc_r1.iloc[-1]); r24 = float(btc_r24.iloc[-1])
        feats["us_x_ret1h"] = us * r1
        feats["asia_x_ret1h"] = asia * r1
        feats["wknd_x_ret1h"] = wknd * r1
        feats["wknd_x_ret24h"] = wknd * r24
    except Exception:
        for k in ("us_x_ret1h", "asia_x_ret1h", "wknd_x_ret1h", "wknd_x_ret24h"):
            feats[k] = NAN

    return feats


# ── public API ─────────────────────────────────────────────────────────────

def compute_p_up_v3(live_1m: "pd.DataFrame | None" = None,
                    df_1h: "pd.DataFrame | None" = None,
                    now: "datetime | None" = None) -> "float | None":
    """SHADOW-ONLY market-level v3 score. Cached per completed 1h bar.

    live_1m : runner's fresh 1m frame (supplies the newest micro bars)
    df_1h   : runner's 1h frame — fallback if the module's own fetch fails
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

        b1 = _fetch("1h", 1000, "BTC")
        if b1 is None or len(b1) < 200:
            b1 = df_1h
        if b1 is None or len(b1) < 200:
            return None
        if b1.index.tz is None:
            b1 = b1.copy(); b1.index = b1.index.tz_localize("UTC")
        b1 = b1[b1.index <= T]          # completed bars only
        if not len(b1) or (T - b1.index[-1]) > pd.Timedelta(hours=2):
            return None
        T = b1.index[-1]                # last completed bar actually available

        m15 = _fetch("15m", 1000, "BTC")
        if m15 is not None and m15.index.tz is None:
            m15 = m15.copy(); m15.index = m15.index.tz_localize("UTC")
        eth_c = _alt_close("ETHUSDT", T)
        sol_c = _alt_close("SOLUSDT", T)
        m1 = _get_1m_frame(T, live_1m)

        feats = _assemble(T, b1, m15, eth_c, sol_c, m1)
        vec = np.array([[feats.get(f, NAN) for f in pipe["features"]]], dtype=float)
        if np.isnan(vec).all():
            return None
        p = float(pipe["clf"].predict_proba(vec)[0, 1])
        _last["bar_ts"], _last["p"] = T, p
        return p
    except Exception:
        return None
