"""
Backfill vwap_stretch_score and ema_stretch_score for paper_trades_btc15m.csv rows
that lack these features (rows not matched by scan archive join).

Fetches bulk 1m Binance klines, then for each unmatched row computes:
  vwap_stretch_score: session-anchored VWAP σ-band stretch (-2/-1/0/+1/+2)
  ema_stretch_score:  5m EMA-20 deviation (-1 overbought / 0 neutral / +1 oversold)

Mirrors confirmation_indicators.py logic exactly.
"""

import math
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants (mirror confirmation_indicators.py) ─────────────────────────────
VWAP_MIN_SESSION_BARS = 60   # minimum 1m bars from midnight before VWAP signal is reliable
EMA_STRETCH_PERIOD    = 20   # EMA on 5m bars (20 × 5m = 100 min)
EMA_STRETCH_THRESHOLD = 0.001  # ±0.1% stretch threshold

CSV_PATH = Path(__file__).parent / "results" / "paper_trades_btc15m.csv"
BINANCE_URL = "https://api.binance.us/api/v3/klines"
SYMBOL = "BTCUSDT"


# ── Fetch bulk 1m klines ─────────────────────────────────────────────────────
def fetch_1m_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all 1m OHLCV bars from start_ms to end_ms from Binance (1000 bars/call)."""
    all_bars = []
    cur = start_ms
    calls = 0
    while cur < end_ms:
        params = {
            "symbol": SYMBOL, "interval": "1m",
            "startTime": cur, "endTime": end_ms, "limit": 1000,
        }
        for attempt in range(3):
            try:
                r = requests.get(BINANCE_URL, params=params, timeout=15)
                r.raise_for_status()
                bars = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)

        if not bars:
            break
        all_bars.extend(bars)
        cur = bars[-1][0] + 60_000  # next bar starts 1 min after last open
        calls += 1
        if calls % 10 == 0:
            print(f"  ...fetched {calls} calls ({len(all_bars)} bars)")
        time.sleep(0.15)  # stay under rate limit

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── Compute vwap_stretch_score at a given timestamp ──────────────────────────
def vwap_stretch(klines_1m: pd.DataFrame, ts: pd.Timestamp) -> int:
    """
    Compute vwap_stretch_score using session-anchored VWAP (from midnight UTC).
    Returns -2/-1/0/+1/+2.  0 = insufficient data or within bands.
    """
    session_start = ts.normalize()  # midnight UTC
    session = klines_1m.loc[session_start:ts].copy()
    if len(session) < VWAP_MIN_SESSION_BARS:
        return 0

    tp   = (session["high"] + session["low"] + session["close"]) / 3
    vol  = session["volume"]
    if vol.sum() == 0:
        return 0

    vwap_series  = (tp * vol).cumsum() / vol.cumsum()
    vwap_current = float(vwap_series.iloc[-1])
    vwap_std     = float((tp - vwap_series).std())
    spot         = float(session["close"].iloc[-1])

    if vwap_std == 0 or math.isnan(vwap_std):
        return 0

    upper2 = vwap_current + 2 * vwap_std
    upper1 = vwap_current + vwap_std
    lower1 = vwap_current - vwap_std
    lower2 = vwap_current - 2 * vwap_std

    if spot > upper2:
        return -2
    elif spot > upper1:
        return -1
    elif spot < lower2:
        return +2
    elif spot < lower1:
        return +1
    return 0


# ── Compute ema_stretch_score at a given timestamp ────────────────────────────
def ema_stretch(klines_1m: pd.DataFrame, ts: pd.Timestamp) -> int:
    """
    Compute ema_stretch_score from 5m candles resampled from 1m klines.
    Uses last 20 5m bars (100 min window ending at ts).
    Returns -1 (overbought) / 0 (neutral) / +1 (oversold).
    """
    window_start = ts - pd.Timedelta(minutes=150)  # extra buffer for EMA warmup
    bars = klines_1m.loc[window_start:ts].copy()
    if len(bars) < 25:
        return 0

    closes_1m = bars["close"].values
    n_5m = len(closes_1m) // 5
    if n_5m < EMA_STRETCH_PERIOD:
        return 0

    # Resample to 5m: take the close of each 5-bar bucket
    closes_5m = pd.Series([closes_1m[(i + 1) * 5 - 1] for i in range(n_5m)])
    ema_5m = closes_5m.ewm(span=EMA_STRETCH_PERIOD, adjust=False).mean()
    last_close = float(closes_5m.iloc[-1])
    last_ema   = float(ema_5m.iloc[-1])

    if last_ema == 0:
        return 0

    stretch = (last_close - last_ema) / last_ema
    if stretch > EMA_STRETCH_THRESHOLD:
        return -1   # overbought — expect reversion down
    elif stretch < -EMA_STRETCH_THRESHOLD:
        return +1   # oversold — expect reversion up
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading paper_trades_btc15m.csv ...")
    df = pd.read_csv(CSV_PATH)
    df["logged_at_dt"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed")

    # Identify rows already covered by scan archive
    ARC_PATH = CSV_PATH.parent / "btc_scan_archive.csv"
    arc = pd.read_csv(ARC_PATH)
    arc["logged_at_dt"] = pd.to_datetime(arc["logged_at"], utc=True, format="mixed")
    arc_dedup = (arc[["logged_at_dt", "vwap_stretch_score", "ema_stretch_score"]]
                 .dropna(subset=["logged_at_dt"])
                 .drop_duplicates("logged_at_dt")
                 .sort_values("logged_at_dt")
                 .reset_index(drop=True))

    df_sorted = df.sort_values("logged_at_dt").reset_index(drop=True)
    merged = pd.merge_asof(df_sorted, arc_dedup, on="logged_at_dt",
                           tolerance=pd.Timedelta("10min"), direction="nearest")

    # Add columns if missing
    if "vwap_stretch_score" not in merged.columns:
        merged["vwap_stretch_score"] = np.nan
    if "ema_stretch_score" not in merged.columns:
        merged["ema_stretch_score"] = np.nan

    need_idx = merged[merged["vwap_stretch_score"].isna()].index
    print(f"Rows to backfill: {len(need_idx)}")

    if len(need_idx) == 0:
        print("Nothing to do.")
        return

    # Date range for bulk fetch
    ts_list = merged.loc[need_idx, "logged_at_dt"]
    fetch_start = ts_list.min().normalize()
    fetch_end   = ts_list.max() + pd.Timedelta(hours=1)

    start_ms = int(fetch_start.timestamp() * 1000)
    end_ms   = int(fetch_end.timestamp() * 1000)
    total_min = (fetch_end - fetch_start).total_seconds() / 60
    n_calls = math.ceil(total_min / 1000)
    print(f"Fetching 1m klines: {fetch_start.date()} → {fetch_end.date()}, ~{n_calls} calls ...")

    klines = fetch_1m_range(start_ms, end_ms)
    print(f"Fetched {len(klines)} 1m bars")

    # Compute features row-by-row
    print("Computing features ...")
    vwap_scores = []
    ema_scores  = []
    for i, (idx, row) in enumerate(merged.loc[need_idx].iterrows()):
        ts = row["logged_at_dt"]
        vs = vwap_stretch(klines, ts)
        es = ema_stretch(klines, ts)
        vwap_scores.append((idx, vs))
        ema_scores.append((idx, es))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(need_idx)} rows done")

    # Apply back into merged df
    for idx, vs in vwap_scores:
        merged.at[idx, "vwap_stretch_score"] = vs
    for idx, es in ema_scores:
        merged.at[idx, "ema_stretch_score"] = es

    # Restore original row order
    merged = merged.sort_values("logged_at_dt").reset_index(drop=True)

    # Verify coverage
    filled = merged["vwap_stretch_score"].notna().sum()
    total  = len(merged)
    print(f"\nCoverage after backfill: {filled}/{total} ({100*filled/total:.1f}%)")

    # Save (drop the helper columns added during processing)
    cols_to_drop = ["logged_at_dt"]
    for c in cols_to_drop:
        if c in merged.columns and c not in df.columns:
            merged = merged.drop(columns=[c])

    # Check column order matches original + new columns at end
    orig_cols = df.columns.tolist()
    new_cols  = [c for c in ["vwap_stretch_score", "ema_stretch_score"] if c not in orig_cols]
    final_cols = orig_cols + new_cols
    merged = merged[final_cols]

    merged.to_csv(CSV_PATH, index=False)
    print(f"Saved → {CSV_PATH}")


if __name__ == "__main__":
    main()
