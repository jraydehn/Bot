"""
backfill_real_archive_15m.py
-----------------------------
Backfill a real-market training archive from Kalshi's settled 15m markets,
going back as far as the API retains data (~60 days), instead of relying on
synthetic contracts. For every settled 15m market: pulls the real floor_strike,
close_time, and settlement result from Kalshi directly, real p_market from
that market's own 1m candlestick history (not a live-scan snapshot), and the
same 20 LGBM features build_15m_model.py uses (via its own build_features(),
so this stays consistent with the synthetic pipeline) computed at the bar's
open (decision) time from our existing 2yr Binance price history.

Output: results/{asset}_real_archive_15m_backfill.csv
"""
import sys, time, math
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from live_signal import load_auth, kalshi_get
import build_15m_model as bm

MINS_PER_YEAR = 525600.0

SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M"}
LOOKBACK_DAYS = 58  # stay inside the confirmed ~60-day retention window


def fetch_settled_markets(asset: str, auth):
    series = SERIES[asset]
    now = int(time.time())
    start = now - LOOKBACK_DAYS * 24 * 3600
    all_markets = []
    window = start
    step = 7 * 24 * 3600  # page by week to keep each query small
    while window < now:
        window_end = min(window + step, now)
        cursor = None
        while True:
            params = {"series_ticker": series, "status": "settled",
                      "min_close_ts": window, "max_close_ts": window_end, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = kalshi_get("/markets", params, auth)
            page = data.get("markets", [])
            all_markets.extend(page)
            cursor = data.get("cursor")
            if not cursor or len(page) < 1000:
                break
        window = window_end
    return all_markets


def fetch_p_market(asset: str, ticker: str, open_ts: int, auth):
    """Real yes-price near the market's open (decision) time, from its own candlesticks."""
    series = SERIES[asset]
    path = f"/series/{series}/markets/{ticker}/candlesticks"
    params = {"start_ts": open_ts - 60, "end_ts": open_ts + 120, "period_interval": 1}
    try:
        data = kalshi_get(path, params, auth)
    except Exception:
        return None
    candles = data.get("candlesticks", [])
    if not candles:
        return None
    c = candles[0]
    px = c.get("price", {})
    for key in ("open_dollars", "close_dollars", "mean_dollars"):
        v = px.get(key)
        if v not in (None, ""):
            try:
                fv = float(v)
                if 0.0 < fv < 1.0:
                    return fv
            except (TypeError, ValueError):
                continue
    return None


def run(asset: str):
    print(f"\n{'='*60}\n  Backfilling real archive: {asset}\n{'='*60}")
    auth = load_auth()

    print("  Fetching settled markets from Kalshi...")
    markets = fetch_settled_markets(asset, auth)
    print(f"  {len(markets):,} settled markets found")

    print("  Building feature series from price history (build_15m_model.build_features)...")
    feats = bm.build_features(asset)  # indexed by 15m bar open time, has spot/future_close/features

    rows = []
    n_ok = n_no_feat = n_no_price = 0
    for i, m in enumerate(markets):
        ticker = m.get("ticker")
        result = m.get("result")
        floor_strike = m.get("floor_strike")
        open_time_s = m.get("open_time")
        if not (ticker and result in ("yes", "no") and floor_strike and open_time_s):
            continue
        open_dt = pd.Timestamp(open_time_s).tz_convert("UTC")
        open_ts = int(open_dt.timestamp())

        # feature row at this exact bar-open (feats is indexed by 15m bar start)
        if open_dt not in feats.index:
            n_no_feat += 1
            continue
        frow = feats.loc[open_dt]
        spot = float(frow["spot"])
        if spot <= 0 or pd.isna(spot):
            n_no_feat += 1
            continue

        pm = fetch_p_market(asset, ticker, open_ts, auth)
        if pm is None:
            n_no_price += 1
            continue

        rv_ann = float(frow.get("realized_vol_annual", 0.3)) if pd.notna(frow.get("realized_vol_annual")) else 0.3
        vol_pm = rv_ann / math.sqrt(MINS_PER_YEAR)
        sigma_tau = max(vol_pm * math.sqrt(15.0), 1e-6)
        z = math.log(float(floor_strike) / spot) / sigma_tau if spot > 0 else 0.0
        offset_pct = (float(floor_strike) / spot - 1.0) * 100.0

        rows.append({
            "logged_at": open_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "contract_ticker": ticker,
            "spot": spot,
            "strike": float(floor_strike),
            "p_market": pm,
            "tau_minutes": 15.0,
            "offset_pct": offset_pct,
            "z_score": z,
            "resolved_yes": 1 if result == "yes" else 0,
            "bp_15m": frow.get("bp_15m"), "body_15m": frow.get("body_15m"),
            "dir_15m": frow.get("dir_15m"), "chg_15m": frow.get("chg_15m"),
            "stoch_k_15m": frow.get("stoch_k_15m"),
            "bp_5m": frow.get("bp_5m"), "body_5m": frow.get("body_5m"),
            "dir_5m": frow.get("dir_5m"), "chg_5m": frow.get("chg_5m"),
            "stoch_k_5m": frow.get("stoch_k_5m"), "vol_ratio_5m": frow.get("vol_ratio_5m"),
            "chg_1h": frow.get("chg_1h"), "bp_1h": frow.get("bp_1h"),
            "stoch_k_1h": frow.get("stoch_k_1h"), "ema_bias_1h": frow.get("ema_bias_1h"),
            "consec_dir_1h": frow.get("consec_dir_1h"), "vol_ratio_1h": frow.get("vol_ratio_1h"),
            "realized_vol_annual": rv_ann,
        })
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(markets)}  ok={n_ok} no_feat={n_no_feat} no_price={n_no_price}")

    print(f"  Done: ok={n_ok}  no_feat={n_no_feat}  no_price={n_no_price}")
    out = pd.DataFrame(rows)
    out_path = f"results/{asset.lower()}_real_archive_15m_backfill.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    assets = sys.argv[1:] or ["BTC", "ETH", "SOL"]
    for a in assets:
        run(a.upper())
