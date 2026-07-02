#!/usr/bin/env python3
"""
liquidity_heatmap.py

Multi-source liquidity heatmap for BTC, ETH, SOL.
Identifies price levels where liquidity is likely concentrated.

Sources:
  1. Volume Profile    — POC + Value Area High/Low + HVN/LVN (30-day)
  2. Swing H/L Clusters — 1h/4h/1d rolling swing highs/lows, clustered
  3. Round Numbers     — psychological price levels
  4. Order Book        — cumulative depth buckets + single large walls
  5. VWAP Levels       — daily / weekly / monthly anchored VWAP
  6. Pivot Points      — daily + weekly floor-trader pivots (PP/R1/R2/S1/S2)
  7. EMA Levels        — key EMAs (20/50/100/200) on 1h and 1d
  8. Fibonacci         — retracements + extensions from recent major swing
  9. Coinalyze Liq     — price levels where large liquidations hit (timestamp→price)

Run: python3 liquidity_heatmap.py
"""

import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
BASE_URL   = "https://api.binance.us/api/v3"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
COINALYZE_KEY  = os.environ.get("COINALYZE_API_KEY", "d5841821-3f45-4e5f-9ee7-d2779d2fb01b")

ASSETS = {
    "BTC": ("BTCUSDT", "BTCUSDT", "BTCUSDT_PERP.A"),
    "ETH": ("ETHUSDT", "ETHUSDT", "ETHUSDT_PERP.A"),
    "SOL": ("SOLUSDT", "SOLUSDT", "SOLUSDT_PERP.A"),
}

VOL_PROFILE_DAYS  = 30
VOL_PROFILE_BINS  = 300
VALUE_AREA_PCT    = 0.70        # 70% of volume = Value Area
VOL_HVN_TOP_N     = 15
SWING_LOOKBACKS   = [5, 10, 20]
CLUSTER_PCT       = 0.003       # merge levels within 0.3%
OB_DEPTH          = 500
OB_BUCKET_PCT     = 0.002       # 0.2% depth buckets for OB aggregation
OB_WALL_NOTIONAL  = 0.008       # top 0.8% notional = wall
DISPLAY_RANGE_PCT = 0.05        # ±5% from spot
FIB_LEVELS        = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
EMA_PERIODS       = [20, 50, 100, 200]
LIQ_SPIKE_PCTILE  = 90          # top 10% liquidation hours = spike

SEP  = "=" * 78
SEP2 = "-" * 78


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_parquet(sym: str) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 1m parquet for {sym}")
    df = pd.read_parquet(files[-1])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def get_spot_price(sym: str) -> float:
    r = requests.get(f"{BASE_URL}/ticker/price", params={"symbol": sym}, timeout=5)
    return float(r.json()["price"])


def get_order_book(sym: str, limit: int = 500) -> dict:
    r = requests.get(f"{BASE_URL}/depth", params={"symbol": sym, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()


def make_level(price, spot, source, detail, weight):
    return {
        "price":    float(price),
        "dist_pct": (float(price) - spot) / spot * 100,
        "source":   source,
        "detail":   detail,
        "weight":   float(weight),
    }


# ── 1. Volume Profile + Value Area ───────────────────────────────────────────

def build_volume_profile(df_1m: pd.DataFrame, spot: float) -> tuple[pd.DataFrame, float, float, float]:
    """
    Returns (vp_df, poc_price, va_low, va_high).
    vp_df columns: price_mid, volume, vol_pct, vol_rank
    """
    cutoff = df_1m.index[-1] - pd.Timedelta(days=VOL_PROFILE_DAYS)
    recent = df_1m[df_1m.index >= cutoff].copy()

    lo = recent["low"].min()
    hi = recent["high"].max()
    edges = np.linspace(lo, hi, VOL_PROFILE_BINS + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    bin_w = edges[1] - edges[0]

    # Vectorised: distribute each bar's volume uniformly across the bins it spans
    vol_at_price = np.zeros(VOL_PROFILE_BINS)
    bar_lo  = recent["low"].values
    bar_hi  = recent["high"].values
    bar_vol = recent["volume"].values

    for i in range(len(recent)):
        lo_idx = int(max(0, (bar_lo[i] - edges[0]) / bin_w))
        hi_idx = int(min(VOL_PROFILE_BINS - 1, (bar_hi[i] - edges[0]) / bin_w))
        n = hi_idx - lo_idx + 1
        if n > 0:
            vol_at_price[lo_idx:hi_idx + 1] += bar_vol[i] / n

    vp = pd.DataFrame({"price_mid": mids, "volume": vol_at_price})
    vp["vol_pct"]  = vp["volume"] / vp["volume"].sum() * 100
    vp["vol_rank"] = vp["volume"].rank(ascending=False).astype(int)
    vp["dist_pct"] = (vp["price_mid"] - spot) / spot * 100

    # POC
    poc_idx   = vp["volume"].idxmax()
    poc_price = float(vp.loc[poc_idx, "price_mid"])

    # Value Area: start from POC, expand outward until 70% of volume is captured
    total_vol = vp["volume"].sum()
    target    = total_vol * VALUE_AREA_PCT
    va_idx    = [poc_idx]
    cum       = vp.loc[poc_idx, "volume"]
    lo_ptr    = poc_idx - 1
    hi_ptr    = poc_idx + 1

    while cum < target:
        lo_vol = vp.loc[lo_ptr, "volume"] if lo_ptr >= 0 else -1
        hi_vol = vp.loc[hi_ptr, "volume"] if hi_ptr < len(vp) else -1
        if lo_vol <= 0 and hi_vol <= 0:
            break
        if hi_vol >= lo_vol:
            va_idx.append(hi_ptr)
            cum += hi_vol
            hi_ptr += 1
        else:
            va_idx.append(lo_ptr)
            cum += lo_vol
            lo_ptr -= 1

    va_low  = float(vp.loc[min(va_idx), "price_mid"])
    va_high = float(vp.loc[max(va_idx), "price_mid"])

    return vp, poc_price, va_low, va_high


def hvn_levels_from_vp(vp: pd.DataFrame, spot: float, poc: float, va_low: float, va_high: float) -> list[dict]:
    levels = []
    # POC
    levels.append(make_level(poc, spot, "VolProfile",
                             f"POC ({(poc-spot)/spot*100:+.2f}%)", weight=3.0))
    # VAH / VAL
    levels.append(make_level(va_high, spot, "VolProfile",
                             f"VAH (70% value area high)", weight=2.5))
    levels.append(make_level(va_low, spot, "VolProfile",
                             f"VAL (70% value area low)", weight=2.5))
    # Top HVN nodes (excluding POC vicinity)
    top = vp[vp["vol_rank"] <= VOL_HVN_TOP_N].copy()
    for _, row in top.iterrows():
        if abs(row["price_mid"] - poc) / poc < CLUSTER_PCT:
            continue  # already captured as POC
        levels.append(make_level(row["price_mid"], spot, "VolProfile",
                                 f"HVN rank#{row['vol_rank']} ({row['vol_pct']:.2f}%)",
                                 weight=1.0 + (VOL_HVN_TOP_N - row["vol_rank"]) / VOL_HVN_TOP_N))
    return levels


# ── 2. Swing H/L Clusters ────────────────────────────────────────────────────

def swing_levels(df_1m: pd.DataFrame, spot: float) -> list[dict]:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    tf_dfs = {
        "1h": df_1m.resample("1h").agg(agg).dropna(),
        "4h": df_1m.resample("4h").agg(agg).dropna(),
        "1d": df_1m.resample("1D").agg(agg).dropna(),
    }

    raw: list[dict] = []
    for tf, df_tf in tf_dfs.items():
        for lb in SWING_LOOKBACKS:
            if len(df_tf) < lb * 3:
                continue
            cutoff = df_tf.index[-1] - pd.Timedelta(days=90)
            roll_max = df_tf["high"].rolling(lb * 2 + 1, center=True).max()
            roll_min = df_tf["low"].rolling(lb * 2 + 1, center=True).min()
            for ts in df_tf.index[df_tf["high"] == roll_max]:
                if ts >= cutoff:
                    raw.append({"price": float(df_tf.loc[ts, "high"]), "type": "high",
                                "tf": tf, "lb": lb,
                                "age": (df_tf.index[-1] - ts).total_seconds() / 86400})
            for ts in df_tf.index[df_tf["low"] == roll_min]:
                if ts >= cutoff:
                    raw.append({"price": float(df_tf.loc[ts, "low"]), "type": "low",
                                "tf": tf, "lb": lb,
                                "age": (df_tf.index[-1] - ts).total_seconds() / 86400})

    if not raw:
        return []

    # Cluster by proximity
    raw_sorted = sorted(raw, key=lambda x: x["price"])
    clusters: list[list[dict]] = []
    for r in raw_sorted:
        placed = False
        for cl in clusters:
            if abs(r["price"] - cl[0]["price"]) / cl[0]["price"] <= CLUSTER_PCT:
                cl.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])

    result = []
    for cl in clusters:
        centroid = float(np.mean([r["price"] for r in cl]))
        tfs      = set(r["tf"] for r in cl)
        types    = set(r["type"] for r in cl)
        tf_conf  = len(tfs)
        both     = len(types) == 2
        min_age  = min(r["age"] for r in cl)
        detail   = (f"TF={','.join(sorted(tfs))}  hits={len(cl)}  "
                    f"age={min_age:.1f}d  {'BOTH' if both else list(types)[0]}")
        result.append(make_level(centroid, spot, "SwingHL", detail,
                                 weight=tf_conf + len(cl) * 0.2 + (1.0 if both else 0)))
    return result


# ── 3. Round Numbers ──────────────────────────────────────────────────────────

def round_number_levels(spot: float, asset: str) -> list[dict]:
    if asset == "BTC":
        intervals = [100, 500, 1000, 5000]
    elif asset == "ETH":
        intervals = [10, 50, 100, 500]
    else:
        intervals = [1, 5, 10, 50]

    lo, hi = spot * (1 - DISPLAY_RANGE_PCT * 2), spot * (1 + DISPLAY_RANGE_PCT * 2)
    seen: dict[float, dict] = {}
    for interval in intervals:
        p = math.floor(lo / interval) * interval
        while p <= hi:
            if lo <= p <= hi:
                w = math.log10(interval + 1)
                lv = make_level(p, spot, "RoundNum", f"${interval:,} interval", weight=w)
                if p not in seen or w > seen[p]["weight"]:
                    seen[p] = lv
            p += interval
    return list(seen.values())


# ── 4. Order Book — cumulative depth buckets + walls ─────────────────────────

def order_book_levels(sym: str, spot: float) -> list[dict]:
    try:
        ob = get_order_book(sym, OB_DEPTH)
    except Exception as e:
        print(f"  [OB] {e}")
        return []

    bids = [(float(p), float(q)) for p, q in ob["bids"]]
    asks = [(float(p), float(q)) for p, q in ob["asks"]]

    result = []

    # ── Single large walls ──
    bid_notional = sum(p * q for p, q in bids)
    ask_notional = sum(p * q for p, q in asks)
    for p, q in bids:
        n = p * q
        if n / bid_notional >= OB_WALL_NOTIONAL:
            result.append(make_level(p, spot, "OB_Bid",
                                     f"${n:,.0f} ({n/bid_notional*100:.1f}% bids)",
                                     weight=n / bid_notional * 20))
    for p, q in asks:
        n = p * q
        if n / ask_notional >= OB_WALL_NOTIONAL:
            result.append(make_level(p, spot, "OB_Ask",
                                     f"${n:,.0f} ({n/ask_notional*100:.1f}% asks)",
                                     weight=n / ask_notional * 20))

    # ── Cumulative depth buckets ──
    # Group bids/asks into 0.2% price buckets, find buckets with abnormal depth
    def bucket_depth(side_orders: list[tuple], label: str):
        if not side_orders:
            return []
        prices  = np.array([p for p, _ in side_orders])
        notionals = np.array([p * q for p, q in side_orders])
        lo_b = prices.min()
        hi_b = prices.max()
        if hi_b == lo_b:
            return []
        bucket_size = spot * OB_BUCKET_PCT
        n_buckets   = max(1, int((hi_b - lo_b) / bucket_size) + 1)
        buckets     = np.zeros(n_buckets)
        centers     = np.array([lo_b + (i + 0.5) * bucket_size for i in range(n_buckets)])
        idx         = ((prices - lo_b) / bucket_size).astype(int).clip(0, n_buckets - 1)
        np.add.at(buckets, idx, notionals)
        total = buckets.sum()
        if total == 0:
            return []
        threshold = np.percentile(buckets[buckets > 0], 90)
        lvls = []
        for i, (bkt, ctr) in enumerate(zip(buckets, centers)):
            if bkt >= threshold and bkt / total >= 0.005:
                lvls.append(make_level(ctr, spot, label,
                                       f"depth bucket ${bkt:,.0f} ({bkt/total*100:.1f}%)",
                                       weight=bkt / total * 15))
        return lvls

    result.extend(bucket_depth(bids, "OB_BidDepth"))
    result.extend(bucket_depth(asks, "OB_AskDepth"))
    return result


# ── 5. VWAP Levels ────────────────────────────────────────────────────────────

def vwap_levels(df_1m: pd.DataFrame, spot: float) -> list[dict]:
    now    = df_1m.index[-1]
    tp     = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3
    tv     = tp * df_1m["volume"]
    result = []

    def anchored_vwap(start: pd.Timestamp, label: str, weight: float):
        mask   = df_1m.index >= start
        if mask.sum() < 10:
            return
        vwap_p = tv[mask].sum() / df_1m.loc[mask, "volume"].sum()
        result.append(make_level(vwap_p, spot, "VWAP", f"{label} VWAP", weight=weight))

    # Daily (today UTC)
    day_start = now.normalize()
    anchored_vwap(day_start, "Daily", weight=2.5)

    # Weekly (this Mon UTC)
    week_start = now - pd.Timedelta(days=now.weekday())
    week_start = week_start.normalize()
    anchored_vwap(week_start, "Weekly", weight=2.0)

    # Monthly
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    anchored_vwap(month_start, "Monthly", weight=1.5)

    return result


# ── 6. Pivot Points ───────────────────────────────────────────────────────────

def pivot_levels(df_1m: pd.DataFrame, spot: float) -> list[dict]:
    result = []

    def add_pivots(h, l, c, label: str, w_scale: float):
        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        r2 = pp + (h - l)
        r3 = h + 2 * (pp - l)
        s1 = 2 * pp - h
        s2 = pp - (h - l)
        s3 = l - 2 * (h - pp)
        for price, name, w in [
            (pp, f"{label} PP",  2.0 * w_scale),
            (r1, f"{label} R1",  1.5 * w_scale),
            (r2, f"{label} R2",  1.2 * w_scale),
            (r3, f"{label} R3",  1.0 * w_scale),
            (s1, f"{label} S1",  1.5 * w_scale),
            (s2, f"{label} S2",  1.2 * w_scale),
            (s3, f"{label} S3",  1.0 * w_scale),
        ]:
            dist = abs(price - spot) / spot
            if dist <= DISPLAY_RANGE_PCT * 1.5:
                result.append(make_level(price, spot, "Pivot", name, weight=w))

    # Daily pivots: yesterday's OHLC
    df_1d = df_1m.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(df_1d) >= 2:
        prev_day = df_1d.iloc[-2]
        add_pivots(prev_day["high"], prev_day["low"], prev_day["close"], "Day", w_scale=1.0)

    # Weekly pivots: last week's OHLC
    df_1w = df_1m.resample("W").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(df_1w) >= 2:
        prev_wk = df_1w.iloc[-2]
        add_pivots(prev_wk["high"], prev_wk["low"], prev_wk["close"], "Wk", w_scale=1.3)

    return result


# ── 7. EMA Levels ─────────────────────────────────────────────────────────────

def ema_levels(df_1m: pd.DataFrame, spot: float) -> list[dict]:
    result = []
    df_1h = df_1m.resample("1h").agg({"close": "last"}).dropna()
    df_1d = df_1m.resample("1D").agg({"close": "last"}).dropna()

    for periods, df_tf, tf_label, w_base in [
        (EMA_PERIODS, df_1h, "1h",  0.8),
        (EMA_PERIODS, df_1d, "1d",  1.5),
    ]:
        for p in periods:
            if len(df_tf) < p:
                continue
            ema_val = float(df_tf["close"].ewm(span=p, adjust=False).mean().iloc[-1])
            dist    = abs(ema_val - spot) / spot
            if dist <= DISPLAY_RANGE_PCT:
                # Weight: longer period + daily TF = higher weight
                w = w_base * (1 + math.log10(p))
                result.append(make_level(ema_val, spot, "EMA",
                                         f"EMA{p} ({tf_label})", weight=w))
    return result


# ── 8. Fibonacci Retracements / Extensions ────────────────────────────────────

def fibonacci_levels(df_1m: pd.DataFrame, spot: float) -> list[dict]:
    # Use 90-day 4h data to find the dominant swing
    cutoff = df_1m.index[-1] - pd.Timedelta(days=90)
    df_4h  = df_1m[df_1m.index >= cutoff].resample("4h").agg(
        {"high": "max", "low": "min"}
    ).dropna()

    if len(df_4h) < 20:
        return []

    swing_high = float(df_4h["high"].max())
    swing_low  = float(df_4h["low"].min())
    rng        = swing_high - swing_low

    # Determine direction: if spot closer to high, retracing down; else retracing up
    retracing_down = (swing_high - spot) < (spot - swing_low)

    result = []
    for fib in FIB_LEVELS:
        if retracing_down:
            # Retracement levels counting down from swing high
            price = swing_high - fib * rng
        else:
            # Retracement levels counting up from swing low
            price = swing_low + fib * rng

        dist = abs(price - spot) / spot
        if dist <= DISPLAY_RANGE_PCT * 1.5:
            # 0.382, 0.5, 0.618 are "golden" levels — higher weight
            w = 2.0 if fib in (0.382, 0.5, 0.618) else 1.2
            result.append(make_level(price, spot, "Fib",
                                     f"Fib {fib:.3f} ({'retrace↓' if retracing_down else 'retrace↑'})",
                                     weight=w))
    return result


# ── 9. Coinalyze Liquidation Spikes → Price Levels ───────────────────────────

def coinalyze_liq_levels(cz_sym: str, df_1m: pd.DataFrame, spot: float) -> list[dict]:
    """
    Fetch recent liq history from Coinalyze. Find hours with top-N% liq events,
    look up the price at that timestamp from parquet, return as levels.
    """
    now_unix  = int(time.time())
    from_unix = now_unix - 90 * 24 * 3600
    params    = {"symbols": cz_sym, "interval": "1hour",
                 "from": from_unix, "to": now_unix, "api_key": COINALYZE_KEY}
    try:
        r = requests.get(f"{COINALYZE_BASE}/liquidation-history", params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()[0]["history"]
        liq  = pd.DataFrame(rows)
        liq["t"] = pd.to_datetime(liq["t"], unit="s", utc=True)
        liq = liq.set_index("t").rename(columns={"l": "long_liq", "s": "short_liq"})
        liq = liq[["long_liq", "short_liq"]].astype(float)
    except Exception as e:
        print(f"  [Coinalyze liq] {e}")
        return []

    if liq.empty:
        return []

    liq["total_liq"] = liq["long_liq"] + liq["short_liq"]
    threshold = liq["total_liq"].quantile(LIQ_SPIKE_PCTILE / 100)
    spikes    = liq[liq["total_liq"] >= threshold]

    # For each spike hour, get the close price from 1m parquet at that timestamp
    result = []
    seen_prices: list[float] = []

    for ts, row in spikes.iterrows():
        # Find closest 1m bar
        try:
            bar = df_1m.loc[ts] if ts in df_1m.index else df_1m.asof(ts)
            p   = float(bar["close"])
        except Exception:
            continue

        # Skip if too close to an already-seen level
        if any(abs(p - sp) / sp <= CLUSTER_PCT for sp in seen_prices):
            continue
        seen_prices.append(p)

        dist = abs(p - spot) / spot
        if dist > DISPLAY_RANGE_PCT * 2:
            continue

        total   = row["total_liq"]
        bias    = "LongLiq" if row["long_liq"] > row["short_liq"] else "ShortLiq"
        # Weight by liq size relative to threshold
        w = 1.0 + (total - threshold) / (liq["total_liq"].max() - threshold + 1e-9)
        result.append(make_level(p, spot, "LiqSpike",
                                 f"{bias} ${total:,.0f}  ({ts.strftime('%m-%d %H:%M')})",
                                 weight=w * 2))
    return result


# ── Merge all sources ─────────────────────────────────────────────────────────

def merge_levels(all_levels: list[dict], spot: float) -> pd.DataFrame:
    if not all_levels:
        return pd.DataFrame()

    all_levels = sorted(all_levels, key=lambda x: x["price"])
    clusters: list[list[dict]] = []
    for lv in all_levels:
        placed = False
        for cl in clusters:
            ref = cl[0]["price"]
            if abs(lv["price"] - ref) / ref <= CLUSTER_PCT:
                cl.append(lv)
                placed = True
                break
        if not placed:
            clusters.append([lv])

    rows = []
    for cl in clusters:
        centroid  = float(np.mean([lv["price"] for lv in cl]))
        sources   = sorted(set(lv["source"] for lv in cl))
        weight    = sum(lv["weight"] for lv in cl)
        detail    = " | ".join(f"{lv['source']}: {lv['detail']}" for lv in cl)
        rows.append({
            "price":      centroid,
            "dist_pct":   (centroid - spot) / spot * 100,
            "sources":    ", ".join(sources),
            "n_sources":  len(sources),
            "confluence": len(cl),
            "weight":     round(weight, 2),
            "detail":     detail,
        })

    df = pd.DataFrame(rows)
    df = df[df["dist_pct"].abs() <= DISPLAY_RANGE_PCT * 100]
    return df.sort_values("dist_pct").reset_index(drop=True)


# ── Display ───────────────────────────────────────────────────────────────────

def print_heatmap(df: pd.DataFrame, spot: float, asset: str):
    if df.empty:
        print("  (no levels found)")
        return

    if asset == "BTC":
        pfmt = lambda p: f"${p:>10,.0f}"
    elif asset == "ETH":
        pfmt = lambda p: f"${p:>8,.2f}"
    else:
        pfmt = lambda p: f"${p:>7,.3f}"

    # Spot row marker
    spot_idx = (df["dist_pct"].abs()).idxmin()

    print(f"\n  {'Price':>12}  {'Dist%':>7}  {'Src':>4}  {'Wt':>5}  Sources")
    print(f"  {'-'*12}  {'-'*7}  {'-'*4}  {'-'*5}  {'-'*50}")

    for idx, row in df.iterrows():
        conf_star = "*" * min(row["n_sources"], 5)
        bar       = "█" * min(int(row["weight"] / 2), 15)
        marker    = "  ◄ SPOT" if idx == spot_idx else ""
        side_arr  = "▲" if row["dist_pct"] > 0 else "▼"
        print(f"  {pfmt(row['price'])}  {row['dist_pct']:>+6.2f}%  "
              f"{conf_star:<5}{row['weight']:>5.1f}  "
              f"{bar:<15}  {row['sources']}{marker}")

    # Key levels summary
    above = df[df["dist_pct"] > 0]
    below = df[df["dist_pct"] < 0]
    print(f"\n  Levels above spot: {len(above)}  |  Below spot: {len(below)}")

    if not above.empty:
        r = above.iloc[0]
        print(f"  Nearest resistance: {pfmt(r['price'])}  ({r['dist_pct']:>+.2f}%)  [{r['sources']}]")
    if not below.empty:
        s = below.iloc[-1]
        print(f"  Nearest support:    {pfmt(s['price'])}  ({s['dist_pct']:>+.2f}%)  [{s['sources']}]")

    # High confluence
    multi = df[df["n_sources"] >= 3].sort_values("weight", ascending=False)
    if not multi.empty:
        print(f"\n  HIGH-CONFLUENCE LEVELS (3+ sources):")
        print(f"  {'Price':>12}  {'Dist%':>7}  {'Type':>7}  {'Wt':>5}  Sources")
        print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*45}")
        for _, row in multi.head(10).iterrows():
            side = "RESIST" if row["dist_pct"] > 0 else "SUPPORT"
            print(f"  {pfmt(row['price'])}  {row['dist_pct']:>+6.2f}%  {side:>7}  "
                  f"{row['weight']:>5.1f}  {row['sources']}")

    # Bias summary: is liquidity skewed above or below spot?
    wt_above = above["weight"].sum() if not above.empty else 0
    wt_below = below["weight"].sum() if not below.empty else 0
    total_wt = wt_above + wt_below
    if total_wt > 0:
        print(f"\n  Liquidity bias: {wt_below/total_wt*100:.0f}% below / {wt_above/total_wt*100:.0f}% above spot")
        if wt_below > wt_above * 1.3:
            print(f"  → More support below; price has a floor to lean on")
        elif wt_above > wt_below * 1.3:
            print(f"  → More resistance above; price faces heavy supply overhead")
        else:
            print(f"  → Roughly balanced; no strong directional bias from liquidity")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_asset(asset: str, spot_sym: str, data_sym: str, cz_sym: str):
    print(f"\n{SEP}")
    print(f"  {asset} — Liquidity Heatmap")
    print(SEP)

    print(f"  Loading parquet …", end=" ", flush=True)
    df_1m = load_parquet(data_sym)
    print(f"{len(df_1m):,} bars  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")

    spot = get_spot_price(spot_sym)
    print(f"  Spot: ${spot:,.4f}")

    all_levels: list[dict] = []

    # 1. Volume Profile
    print(f"  [1] Volume profile …", end=" ", flush=True)
    vp, poc, va_low, va_high = build_volume_profile(df_1m, spot)
    vp_lvls = hvn_levels_from_vp(vp, spot, poc, va_low, va_high)
    all_levels.extend(vp_lvls)
    print(f"POC=${poc:,.2f} ({(poc-spot)/spot*100:+.2f}%)  "
          f"VA=[${va_low:,.2f}–${va_high:,.2f}]  {len(vp_lvls)} levels")

    # 2. Swing H/L
    print(f"  [2] Swing H/L …", end=" ", flush=True)
    sw_lvls = swing_levels(df_1m, spot)
    all_levels.extend(sw_lvls)
    print(f"{len(sw_lvls)} clusters")

    # 3. Round numbers
    rn_lvls = round_number_levels(spot, asset)
    all_levels.extend(rn_lvls)
    print(f"  [3] Round numbers: {len(rn_lvls)} levels")

    # 4. Order book
    print(f"  [4] Order book …", end=" ", flush=True)
    ob_lvls = order_book_levels(spot_sym, spot)
    all_levels.extend(ob_lvls)
    print(f"{len(ob_lvls)} walls/buckets")

    # 5. VWAP
    print(f"  [5] VWAP …", end=" ", flush=True)
    vw_lvls = vwap_levels(df_1m, spot)
    all_levels.extend(vw_lvls)
    print(f"{len(vw_lvls)} levels  " +
          "  ".join(f"{lv['detail']}: ${lv['price']:,.2f}" for lv in vw_lvls))

    # 6. Pivots
    print(f"  [6] Pivots …", end=" ", flush=True)
    pv_lvls = pivot_levels(df_1m, spot)
    all_levels.extend(pv_lvls)
    print(f"{len(pv_lvls)} pivot levels")

    # 7. EMA
    print(f"  [7] EMA levels …", end=" ", flush=True)
    em_lvls = ema_levels(df_1m, spot)
    all_levels.extend(em_lvls)
    print(f"{len(em_lvls)} within ±{DISPLAY_RANGE_PCT*100:.0f}%")

    # 8. Fibonacci
    print(f"  [8] Fibonacci …", end=" ", flush=True)
    fib_lvls = fibonacci_levels(df_1m, spot)
    all_levels.extend(fib_lvls)
    print(f"{len(fib_lvls)} levels")

    # 9. Coinalyze liq spikes
    print(f"  [9] Coinalyze liq spikes …", end=" ", flush=True)
    liq_lvls = coinalyze_liq_levels(cz_sym, df_1m, spot)
    all_levels.extend(liq_lvls)
    print(f"{len(liq_lvls)} spike levels")

    # Merge
    df_map = merge_levels(all_levels, spot)
    print(f"\n  Total unique levels within ±{DISPLAY_RANGE_PCT*100:.0f}% of spot: {len(df_map)}")

    print_heatmap(df_map, spot, asset)
    return df_map


def main():
    print(SEP)
    print("  Liquidity Heatmap — BTC / ETH / SOL")
    print("  Sources: VolProfile+VA | SwingHL | RoundNum | OB depth+walls |")
    print("           VWAP (D/W/M) | Pivots (D/W) | EMA (1h/1d) | Fib | LiqSpikes")
    print(SEP)

    for asset, (spot_sym, data_sym, cz_sym) in ASSETS.items():
        try:
            run_asset(asset, spot_sym, data_sym, cz_sym)
        except Exception as e:
            import traceback
            print(f"\n  ERROR for {asset}: {e}")
            traceback.print_exc()
        time.sleep(0.5)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
