"""
BTC OHLCV fetcher supporting multiple exchange sources.

Pulls 1-minute candles, paginates across the full date range, caches
results to Parquet, and resamples to 1-hour and 4-hour bars.

Supported sources (--source flag):
    binanceus  Binance.US public API  — best for US-based machines (default)
    bybit      Bybit public API       — global, no geo-restrictions
    binance    Binance.com            — blocked in the US (451 error)

Usage:
    python fetch_data.py                              # last ~2 years, binanceus
    python fetch_data.py --source bybit               # use Bybit instead
    python fetch_data.py --start 2024-01-01           # custom start date
    python fetch_data.py --start 2024-01-01 --end 2025-01-01
    python fetch_data.py --refresh                    # ignore cache, re-fetch
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_SOURCE = "binanceus"
CANDLES_PER_REQUEST = 1000
REQUEST_DELAY_S = 0.25       # 250 ms pause — well under all exchange rate limits
DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_EVERY = 100       # save progress to disk every N requests
MAX_RETRIES = 4              # retry a failed request up to this many times
RETRY_BASE_DELAY_S = 2.0    # first retry waits 2s, then 4s, 8s, 16s (exponential)

# Bybit uses short codes for intervals instead of "1m"/"1h"
_BYBIT_INTERVAL_MAP = {"1m": "1", "1h": "60", "4h": "240"}


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn, *args, **kwargs) -> pd.DataFrame:
    """
    Call fn(*args, **kwargs) and retry up to MAX_RETRIES times on network errors.

    Uses exponential backoff: waits RETRY_BASE_DELAY_S * 2^attempt seconds
    between retries. Re-raises on the final attempt.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.ConnectionError, requests.Timeout,
                requests.HTTPError, OSError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_S * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {type(exc).__name__} — "
                  f"waiting {delay:.0f}s before retrying...")
            time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Per-source fetch helpers
# ---------------------------------------------------------------------------

def _fetch_chunk_binance(
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Fetch up to 1000 candles from a Binance-compatible endpoint
    (works for both binance.com and api.binance.us).
    """
    resp = requests.get(
        f"{base_url}/api/v3/klines",
        params={"symbol": symbol, "interval": interval,
                "startTime": start_ms, "endTime": end_ms,
                "limit": CANDLES_PER_REQUEST},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("open_time")[["open","high","low","close","volume"]]


def _fetch_chunk_bybit(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Fetch up to 1000 candles from Bybit's V5 market API.

    Bybit returns candles newest-first, so we reverse the result.
    Bybit interval codes: "1"=1m, "60"=1h, "240"=4h.
    """
    bybit_interval = _BYBIT_INTERVAL_MAP.get(interval)
    if bybit_interval is None:
        raise ValueError(
            f"Bybit does not support interval '{interval}'. "
            f"Supported: {list(_BYBIT_INTERVAL_MAP.keys())}"
        )
    resp = requests.get(
        "https://api.bybit.com/v5/market/kline",
        params={"category": "linear", "symbol": symbol,
                "interval": bybit_interval,
                "start": start_ms, "end": end_ms,
                "limit": CANDLES_PER_REQUEST},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {body.get('retMsg')}")

    rows = body["result"]["list"]
    if not rows:
        return pd.DataFrame()

    # Bybit columns: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","turnover"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time")[["open","high","low","close","volume"]]
    # Bybit returns newest-first — reverse to get chronological order
    return df.sort_index()


# ---------------------------------------------------------------------------
# Unified fetch loop
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = "1m",
    start: str = "2024-01-01",
    end: Optional[str] = None,
    source: str = DEFAULT_SOURCE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch complete OHLCV history, paginating forward until the full range is covered.

    Automatically selects the correct exchange API based on `source`. All sources
    return the same DataFrame schema: open/high/low/close/volume indexed by UTC
    DatetimeIndex.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT"). Must exist on the chosen exchange.
        interval: Candle size: "1m", "1h", or "4h".
        start: ISO date string for start of history.
        end: ISO date string for end of history. Defaults to now (UTC).
        source: Exchange to pull from: "binanceus", "bybit", or "binance".
        verbose: Print progress every 50 requests when True.

    Returns:
        Sorted, deduplicated OHLCV DataFrame indexed by UTC datetime.

    Raises:
        ValueError: If `source` is not recognised.
        requests.HTTPError: On HTTP errors (e.g. 451 geo-block on binance.com).
        RuntimeError: If no candles are returned for the requested range.
    """
    source_urls = {
        "binance":   "https://api.binance.com",
        "binanceus": "https://api.binance.us",
    }
    if source not in (*source_urls, "bybit"):
        raise ValueError(
            f"Unknown source '{source}'. Choose: binance, binanceus, bybit."
        )

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        if end else datetime.now(timezone.utc)
    )
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    interval_ms   = _interval_to_ms(interval)
    total_candles = (end_ms - start_ms) // interval_ms
    total_requests = (total_candles // CANDLES_PER_REQUEST) + 1

    # Checkpoint path — written periodically so a crash can resume mid-fetch
    ckpt_tag = f"{source}_{symbol}_{interval}_{start}_{end_dt.strftime('%Y-%m-%d')}"
    ckpt_path = DATA_DIR / f"{ckpt_tag}.ckpt.parquet"

    # Resume from checkpoint if one exists
    saved_df = None
    if ckpt_path.exists():
        saved_df = pd.read_parquet(ckpt_path)
        if saved_df.index.tz is None:
            saved_df.index = saved_df.index.tz_localize("UTC")
        resume_ms = int(saved_df.index[-1].timestamp() * 1000) + interval_ms
        if verbose:
            print(f"  Resuming from checkpoint: {saved_df.index[-1].date()} "
                  f"({len(saved_df):,} candles already saved)")
        cursor_ms = resume_ms
    else:
        cursor_ms = start_ms

    if verbose:
        print(f"\nFetching {symbol} {interval} from {source}")
        print(f"  From : {start_dt.date()}")
        print(f"  To   : {end_dt.date()}")
        print(f"  Est. : ~{total_candles:,} candles, ~{total_requests:,} requests")
        print()

    chunks = []
    request_count = 0

    while cursor_ms < end_ms:
        if source == "bybit":
            chunk = _with_retry(_fetch_chunk_bybit, symbol, interval, cursor_ms, end_ms)
        else:
            chunk = _with_retry(_fetch_chunk_binance, source_urls[source],
                                symbol, interval, cursor_ms, end_ms)

        if chunk.empty:
            break

        chunks.append(chunk)
        request_count += 1

        # Advance cursor past the last received candle
        last_ms = int(chunk.index[-1].timestamp() * 1000)
        cursor_ms = last_ms + interval_ms

        if verbose and request_count % 50 == 0:
            pct = 100 * (cursor_ms - start_ms) / (end_ms - start_ms)
            print(f"  [{request_count:>4} req]  {pct:.1f}%  "
                  f"({chunk.index[-1].strftime('%Y-%m-%d')})")

        # Periodically flush accumulated chunks to the checkpoint file
        if request_count % CHECKPOINT_EVERY == 0:
            ckpt_pieces = ([saved_df] if saved_df is not None else []) + chunks
            ckpt_df = pd.concat(ckpt_pieces)
            ckpt_df = ckpt_df[~ckpt_df.index.duplicated(keep="first")].sort_index()
            ckpt_df.to_parquet(ckpt_path)
            # Fold into saved_df and clear chunks to keep memory bounded
            saved_df = ckpt_df
            chunks = []
            if verbose:
                print(f"  Checkpoint saved ({len(saved_df):,} candles so far)")

        time.sleep(REQUEST_DELAY_S)

    # Assemble final result from checkpoint + remaining chunks
    all_pieces = ([saved_df] if saved_df is not None else []) + chunks
    if not all_pieces:
        raise RuntimeError(
            f"No candles returned for {symbol} {interval} {start}→{end} from {source}."
        )

    df = pd.concat(all_pieces)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Remove checkpoint now that the full file will be saved
    if ckpt_path.exists():
        ckpt_path.unlink()

    if verbose:
        print(f"\n  Done. {len(df):,} candles in {request_count} new requests.")

    return df


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to a coarser timeframe.

    Args:
        df_1m: 1-minute OHLCV DataFrame with a UTC DatetimeIndex.
        rule: Pandas offset alias, e.g. "1h" or "4h".

    Returns:
        Resampled DataFrame with any incomplete-window NaN rows dropped.
    """
    return df_1m.resample(rule).agg({
        "open": "first", "high": "max",
        "low": "min",    "close": "last",
        "volume": "sum",
    }).dropna()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(source: str, symbol: str, interval: str,
                start: str, end: str) -> Path:
    """Return a deterministic Parquet path for this (source, symbol, interval, range)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{source}_{symbol}_{interval}_{start}_{end}"
    return DATA_DIR / f"{tag}.parquet"


def load_or_fetch(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = "1m",
    start: str = "2024-01-01",
    end: Optional[str] = None,
    source: str = DEFAULT_SOURCE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return cached OHLCV data if it exists, otherwise fetch from the exchange.

    The cache key includes source, symbol, interval, and date range, so
    different queries never collide. Pass force_refresh=True to ignore the cache.

    Args:
        symbol: Trading pair.
        interval: Candle size.
        start: History start date (ISO string).
        end: History end date (ISO string). Defaults to today.
        source: Exchange source: "binanceus", "bybit", or "binance".
        force_refresh: Re-fetch even if a cache file exists.

    Returns:
        OHLCV DataFrame indexed by UTC datetime.
    """
    end_str = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = _cache_path(source, symbol, interval, start, end_str)

    if path.exists() and not force_refresh:
        print(f"  Cache hit: {path.name}")
        df = pd.read_parquet(path)
        print(f"  {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    df = fetch_ohlcv(symbol=symbol, interval=interval,
                     start=start, end=end_str, source=source, verbose=True)
    df.to_parquet(path)
    print(f"  Saved → {path.name}")
    return df


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def check_gaps(df: pd.DataFrame, interval: str, warn_threshold: int = 5) -> None:
    """
    Print a summary of any time gaps in the OHLCV index.

    Gaps arise from exchange downtime or API errors. They can introduce
    look-ahead bias in rolling windows, so it's worth knowing they exist.

    Args:
        df: OHLCV DataFrame with a DatetimeIndex.
        interval: Expected spacing between rows (e.g. "1m").
        warn_threshold: If gaps <= this count, print each one individually.
    """
    expected = pd.Timedelta(milliseconds=_interval_to_ms(interval))
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > expected * 1.5]

    if gaps.empty:
        print(f"  No gaps in {interval} data ({len(df):,} candles).")
        return

    missing = int(sum((d / expected) - 1 for d in gaps))
    print(f"  {len(gaps)} gap(s) — {missing} missing candles in {interval} data.")
    if len(gaps) <= warn_threshold:
        for ts, d in gaps.items():
            print(f"    {ts.strftime('%Y-%m-%d %H:%M UTC')}  gap={d}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interval_to_ms(interval: str) -> int:
    """Convert a Binance-style interval string (e.g. '1m', '4h') to milliseconds."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(interval[:-1]) * units[interval[-1]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch 1m, 15m, 1h, and 4h OHLCV data and save to data/."""
    parser = argparse.ArgumentParser(description="Fetch BTC OHLCV data for backtesting.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default="2024-01-01",
                        help="Start date YYYY-MM-DD (default 2024-01-01)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default today)")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        choices=["binanceus", "bybit", "binance"],
                        help="Exchange data source (default binanceus)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cache and re-fetch from exchange")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  BTC DATA FETCH  |  source={args.source}")
    print("=" * 60)

    # Step 1: Fetch raw 1-minute data
    print("\n[1/3] 1-minute bars")
    df_1m = load_or_fetch(symbol=args.symbol, interval="1m",
                          start=args.start, end=args.end,
                          source=args.source, force_refresh=args.refresh)
    check_gaps(df_1m, "1m")

    # Steps 2–4: Resample from 1m and cache
    end_str = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for step, (rule, label) in enumerate([("15m", "15-minute"), ("1h", "1-hour"), ("4h", "4-hour")], start=2):
        print(f"\n[{step}/4] {label} bars (resampled from 1m)")
        path = _cache_path(args.source, args.symbol, rule, args.start, end_str)

        if path.exists() and not args.refresh:
            print(f"  Cache hit: {path.name}")
            resampled = pd.read_parquet(path)
        else:
            resampled = resample_ohlcv(df_1m, rule)
            resampled.to_parquet(path)
            print(f"  Saved → {path.name}")

        print(f"  {len(resampled):,} candles  "
              f"({resampled.index[0].date()} → {resampled.index[-1].date()})")
        check_gaps(resampled, rule)

    print(f"\n{'=' * 60}")
    print(f"  Done. Data saved to kalshi_btc/data/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
