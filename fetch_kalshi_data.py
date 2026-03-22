"""
Fetch historical Kalshi BTC price contract data with authenticated candlestick access.

Authentication:
    Reads RSA API credentials from environment variables — never hardcoded.
        KALSHI_KEY_ID   : your Kalshi API key ID
        KALSHI_KEY_PATH : path to your RSA private key PEM file

Steps:
    1. Fetch all KXBTCD markets (any status) via direct HTTP requests.
    2. For each market, fetch a 1-minute candlestick at open_time to get the
       true entry p_market_yes. Skip contracts where the candlestick is missing.
    3. Save to kalshi_btc_history_authenticated.csv.

Unauthenticated fallback:
    If env vars are absent the script still runs, fetching market metadata
    only (no candlesticks). p_market_yes_open will be NaN in that case.

Usage:
    export KALSHI_KEY_ID=your-key-id
    export KALSHI_KEY_PATH=/path/to/private.pem
    python fetch_kalshi_data.py
    python fetch_kalshi_data.py --series KXBTCD --limit 500
    python fetch_kalshi_data.py --out results/kalshi_history.csv
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from kalshi_python_sync import KalshiAuth

BASE_URL        = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER   = "KXBTCD"
SLEEP_BETWEEN   = 0.2      # seconds between API requests
PROGRESS_EVERY  = 25       # print progress every N contracts
AUTH_REFRESH_S  = 25 * 60  # recreate KalshiAuth every 25 minutes
DEFAULT_OUT     = "kalshi_btc_history_authenticated.csv"
CANDLE_WINDOW_S = 120      # fetch 2-minute window to capture opening candlestick


# ---------------------------------------------------------------------------
# Authentication manager
# ---------------------------------------------------------------------------

class AuthManager:
    """
    Loads RSA credentials from env vars and recreates KalshiAuth every
    AUTH_REFRESH_S seconds so the signing context stays fresh.
    """

    def __init__(self):
        self.key_id   = os.environ.get("KALSHI_KEY_ID", "").strip()
        key_path_str  = os.environ.get("KALSHI_KEY_PATH", "").strip()
        self.key_path = Path(key_path_str) if key_path_str else None

        self._auth: Optional[KalshiAuth] = None
        self._last_refresh: float = 0.0

        if self.key_id and self.key_path:
            if not self.key_path.exists():
                raise FileNotFoundError(
                    f"KALSHI_KEY_PATH points to non-existent file: {self.key_path}"
                )
            print(f"  Auth: key_id={self.key_id}  key_path={self.key_path}")
            self._refresh()
        else:
            print("  Auth: KALSHI_KEY_ID / KALSHI_KEY_PATH not set — "
                  "running unauthenticated (candlesticks will be skipped).")

    @property
    def authenticated(self) -> bool:
        return self._auth is not None

    def get_auth(self) -> Optional[KalshiAuth]:
        """Return a fresh KalshiAuth, recreating it if 25 min have elapsed."""
        if not self.authenticated:
            return None
        if time.time() - self._last_refresh >= AUTH_REFRESH_S:
            print("  [auth] 25-minute refresh — recreating KalshiAuth.")
            self._refresh()
        return self._auth

    def _refresh(self) -> None:
        pem = self.key_path.read_text()
        self._auth = KalshiAuth(self.key_id, pem)
        self._last_refresh = time.time()


# ---------------------------------------------------------------------------
# Authenticated HTTP helper
# ---------------------------------------------------------------------------

def kalshi_get(path: str, params: dict, auth: Optional[KalshiAuth]) -> dict:
    """
    Make an authenticated GET request to the Kalshi API.
    Returns the parsed JSON dict, or empty dict on error.
    """
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    if auth is not None:
        headers.update(auth.create_auth_headers("GET", url))

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if not resp.ok:
        print(f"  [http {resp.status_code}] {path}  {resp.text[:120]}")
        return {}
    return resp.json()


# ---------------------------------------------------------------------------
# Step 1: Fetch KXBTCD markets via direct HTTP
# ---------------------------------------------------------------------------

def fetch_markets(
    auth: Optional[KalshiAuth],
    series_ticker: str,
    limit: Optional[int] = None,
) -> list:
    """
    Page through all markets for series_ticker.
    Returns a list of raw market dicts from the API.
    """
    markets = []
    cursor  = None
    page    = 0

    while True:
        params = {"series_ticker": series_ticker, "limit": 200, "status": "settled"}
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/markets", params, auth)
        time.sleep(SLEEP_BETWEEN)

        batch = data.get("markets") or []
        if not batch:
            break

        markets.extend(batch)
        page += 1
        print(f"  Page {page}: {len(batch)} markets  (total: {len(markets)})")

        if limit and len(markets) >= limit:
            markets = markets[:limit]
            break

        cursor = data.get("cursor")
        if not cursor:
            break

    return markets


# ---------------------------------------------------------------------------
# Step 2: Fetch candlesticks for a single market
# ---------------------------------------------------------------------------

def fetch_candlesticks(
    auth: Optional[KalshiAuth],
    series_ticker: str,
    ticker: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """
    Fetch 1-minute candlesticks for ticker between start_time and end_time.

    Returns:
        DataFrame with columns: timestamp, yes_price, volume.
        Empty DataFrame if no data is available or the request fails.
    """
    start_ts = int(start_time.timestamp())
    end_ts   = int(end_time.timestamp())

    path   = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1}

    data    = kalshi_get(path, params, auth)
    candles = data.get("candlesticks") or []

    if not candles:
        return pd.DataFrame(columns=["timestamp", "yes_price", "volume"])

    rows = []
    for c in candles:
        # yes_bid and yes_ask are dicts with open_dollars, close_dollars, etc.
        # We want the open_dollars (price at start of the candle period).
        yes_bid_d = c.get("yes_bid") or {}
        yes_ask_d = c.get("yes_ask") or {}

        def _parse(d, key):
            v = d.get(key)
            try:
                return float(v) if v is not None else None
            except (ValueError, TypeError):
                return None

        bid_open = _parse(yes_bid_d, "open_dollars")
        ask_open = _parse(yes_ask_d, "open_dollars")

        if bid_open is not None and ask_open is not None:
            yes_price = (bid_open + ask_open) / 2.0
        elif ask_open is not None:
            yes_price = ask_open
        elif bid_open is not None:
            yes_price = bid_open
        else:
            yes_price = None

        vol = c.get("volume_fp") or c.get("volume")
        end_ts_val = c.get("end_period_ts")
        ts = datetime.fromtimestamp(end_ts_val, tz=timezone.utc) if end_ts_val else None

        rows.append({"timestamp": ts, "yes_price": yes_price, "volume": vol})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: Build a CSV record from a raw market dict
# ---------------------------------------------------------------------------

def _to_float(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def build_record(market: dict, p_market_yes_open: Optional[float]) -> Optional[dict]:
    """
    Combine market metadata with the opening-candlestick yes price.
    """
    ticker    = market.get("ticker", "")
    open_raw  = market.get("open_time")
    close_raw = market.get("close_time")
    strike    = _to_float(market.get("floor_strike"))

    # Closing YES price (settlement snapshot)
    yes_bid_close = _to_float(market.get("yes_bid_close"))
    yes_ask_close = _to_float(market.get("yes_ask_close"))

    # Cents → fraction conversion (Kalshi prices in cents 0–100)
    def cents_to_frac(v):
        if v is None:
            return None
        return v / 100.0 if v > 1.0 else v

    if yes_bid_close is not None and yes_ask_close is not None:
        p_market_yes_close = cents_to_frac((yes_bid_close + yes_ask_close) / 2.0)
    else:
        last = _to_float(market.get("last_price"))
        p_market_yes_close = cents_to_frac(last)

    # Result
    result_raw = (market.get("result") or "").lower()
    result = result_raw if result_raw in ("yes", "no") else None

    # BTC price at expiry
    expiry_price = None
    for key in ("expiration_value", "settlement_value"):
        raw = market.get(key)
        if raw is not None:
            expiry_price = _to_float(raw)
            if expiry_price:
                break

    return {
        "timestamp":          open_raw,
        "close_time":         close_raw,
        "ticker":             ticker,
        "strike":             strike,
        "p_market_yes_open":  round(p_market_yes_open, 4) if p_market_yes_open is not None else None,
        "p_market_yes_close": round(p_market_yes_close, 4) if p_market_yes_close is not None else None,
        "result":             result,
        "expiry_price":       expiry_price,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch authenticated Kalshi KXBTCD historical data with opening prices."
    )
    parser.add_argument("--series", default=SERIES_TICKER,
                        help=f"Series ticker (default: {SERIES_TICKER})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max markets to fetch (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  KALSHI BTC AUTHENTICATED HISTORICAL FETCH")
    print("=" * 60)

    # Credentials
    print("\n[0/3] Loading credentials...")
    try:
        auth_mgr = AuthManager()
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return

    auth = auth_mgr.get_auth()

    # Step 1: markets
    print(f"\n[1/3] Fetching '{args.series}' markets...")
    markets = fetch_markets(auth, args.series, limit=args.limit)
    print(f"  Total: {len(markets):,} markets")
    if not markets:
        print("  No markets found. Check the series ticker.")
        return

    # Step 2 & 3: candlestick fetch + build records
    print(f"\n[2/3] Fetching opening candlesticks and building records...")
    if not auth_mgr.authenticated:
        print("  WARNING: No credentials — candlestick fetch skipped. "
              "p_market_yes_open will be NaN.")

    records = []
    skipped = 0

    for i, market in enumerate(markets, 1):
        if i % PROGRESS_EVERY == 0:
            print(f"  [{i:>4}/{len(markets)}]  "
                  f"records={len(records)}  skipped={skipped}")

        # Refresh auth if 25 min have elapsed
        auth = auth_mgr.get_auth()

        p_open = None
        if auth_mgr.authenticated:
            open_raw = market.get("open_time")
            if open_raw:
                try:
                    open_dt = pd.Timestamp(open_raw).to_pydatetime()
                    if open_dt.tzinfo is None:
                        open_dt = open_dt.replace(tzinfo=timezone.utc)
                    end_dt = datetime.fromtimestamp(
                        open_dt.timestamp() + CANDLE_WINDOW_S, tz=timezone.utc
                    )

                    df_c = fetch_candlesticks(
                        auth, args.series, market.get("ticker", ""), open_dt, end_dt
                    )
                    time.sleep(SLEEP_BETWEEN)

                    if not df_c.empty and df_c["yes_price"].notna().any():
                        p_open = df_c["yes_price"].dropna().iloc[0]
                    else:
                        skipped += 1

                except Exception as exc:
                    skipped += 1
                    if skipped <= 3:
                        print(f"  [warn] {market.get('ticker')}: {exc}")

        rec = build_record(market, p_open)
        if rec:
            records.append(rec)

    print(f"\n  Done: {len(records):,} records, {skipped} missing candlesticks.")

    if not records:
        print("  No records to save.")
        return

    # Save
    print(f"\n[3/3] Saving to {args.out}...")
    df = pd.DataFrame(records)
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows.")

    # Summary
    print("\n── SUMMARY ─────────────────────────────────────────────")
    print(f"  Date range      : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    if df["strike"].notna().any():
        print(f"  Strike range    : ${df['strike'].min():,.2f} – ${df['strike'].max():,.2f}")
    if df["p_market_yes_open"].notna().any():
        col = df["p_market_yes_open"].dropna()
        print(f"  p_market (open) : {col.min():.3f} – {col.max():.3f}  "
              f"(mean {col.mean():.3f}, n={len(col):,})")
    else:
        print("  p_market (open) : NaN — candlestick data unavailable")
    if df["result"].notna().any():
        yes_rate = (df["result"] == "yes").mean()
        print(f"  YES resolution  : {yes_rate:.2%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
