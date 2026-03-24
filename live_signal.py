"""
Live signal generator — fetches real Kalshi p_market and runs the full model.

At each invocation:
  1. Load cached OHLCV data (run fetch_data.py to refresh).
  2. Compute spot, strike, and all model signals.
  3. Find the active KXBTCD contract whose strike is closest to our target.
  4. Fetch the current bid/ask mid-price from Kalshi's candlestick API.
  5. Run the full decision pipeline with the real market probability.
  6. Print a complete signal report.

Usage:
    export KALSHI_KEY_ID=your-key-id
    export KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
    python3 live_signal.py
    python3 live_signal.py --offset 0.005 --bankroll 10000
    python3 live_signal.py --sim          # use simulated p_market (no Kalshi auth needed)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from evaluate_point import load_data, evaluate_point, print_report
from market_data import compute_realized_volatility
from probability_engine import estimate_probability
from market_structure import detect_market_structure
from confirmation_indicators import compute_confirmation
from pricing_comparison import simulate_p_market, evaluate_edge, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from decision import evaluate_trade
from kelly_sizing import compute_kelly_size

BASE_URL      = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTCD"
CANDLE_WINDOW = 120   # seconds of candlestick history to fetch for current price
DEFAULT_OFFSET = 0.005
DEFAULT_BANKROLL = 10_000
TAU = 60  # 1-hour expiry


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_auth() -> Optional[KalshiAuth]:
    # 1. Environment variables (highest priority)
    key_id   = os.environ.get("KALSHI_KEY_ID", "").strip()
    key_path = os.environ.get("KALSHI_KEY_PATH", "").strip()

    # 2. Fall back to local config file: kalshi_btc/.kalshi_config
    #    Format (one per line):
    #      KALSHI_KEY_ID=your-key-id
    #      KALSHI_KEY_PATH=~/kalshi_key_fixed.pem
    if not key_id or not key_path:
        config_file = Path(__file__).parent / ".kalshi_config"
        if config_file.exists():
            for line in config_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("KALSHI_KEY_ID="):
                    key_id = line.split("=", 1)[1].strip()
                elif line.startswith("KALSHI_KEY_PATH="):
                    key_path = line.split("=", 1)[1].strip()

    if not key_id or not key_path:
        return None
    pem = Path(key_path).expanduser().read_text()
    return KalshiAuth(key_id, pem)


def kalshi_get(path: str, params: dict, auth: KalshiAuth) -> dict:
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    headers.update(auth.create_auth_headers("GET", url))
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if not resp.ok:
            print(f"  [http {resp.status_code}] {path}: {resp.text[:120]}")
            return {}
        return resp.json()
    except Exception as exc:
        print(f"  [error] {path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Find live contract
# ---------------------------------------------------------------------------

def find_live_contract(auth: KalshiAuth, spot: float) -> Optional[dict]:
    """
    Find the nearest OTM KXBTCD contract above the current spot price.
    Does not use an arbitrary offset — instead returns whatever contract
    Kalshi actually has immediately above spot, letting the model compute
    the real offset from the actual floor_strike.
    Returns the raw market dict or None.
    """
    now_ts = int(time.time())

    base_params = {
        "series_ticker": SERIES_TICKER,
        "min_close_ts":  now_ts,
        "max_close_ts":  now_ts + 7200,
        "limit":         200,
    }

    # Paginate through all results to find every strike level
    markets = []
    cursor  = None
    while True:
        params = {**base_params}
        if cursor:
            params["cursor"] = cursor
        data   = kalshi_get("/markets", params, auth)
        page   = data.get("markets") or []
        markets.extend(page)
        cursor = data.get("cursor")
        if not cursor or len(page) < 200:
            break

    if not markets:
        base_params["max_close_ts"] = now_ts + 14400
        cursor = None
        while True:
            params = {**base_params}
            if cursor:
                params["cursor"] = cursor
            data   = kalshi_get("/markets", params, auth)
            page   = data.get("markets") or []
            markets.extend(page)
            cursor = data.get("cursor")
            if not cursor or len(page) < 200:
                break

    if not markets:
        print("  [kalshi] No open hourly contracts found.")
        return None

    # Separate currently open contracts (open_time <= now) from upcoming ones.
    # Prefer a currently open contract so we can fetch a real candlestick price.
    from datetime import datetime, timezone
    now_dt = datetime.now(timezone.utc)

    open_otm    = []  # already trading, floor > spot
    upcoming_otm = []  # not yet open, floor > spot

    for m in markets:
        fs = m.get("floor_strike")
        if fs is None:
            continue
        try:
            fs = float(fs)
        except (ValueError, TypeError):
            continue
        if fs <= spot:
            continue

        open_raw = m.get("open_time", "")
        try:
            open_dt = datetime.fromisoformat(open_raw.replace("Z", "+00:00"))
        except Exception:
            open_dt = now_dt  # treat unknown as already open

        if open_dt <= now_dt:
            open_otm.append((fs, m))
        else:
            upcoming_otm.append((fs, m))

    # Use currently open OTM contracts first; fall back to upcoming
    otm_contracts = sorted(open_otm) or sorted(upcoming_otm)

    if otm_contracts:
        fs, best = otm_contracts[0]
        offset = (fs / spot - 1) * 100
        open_raw = best.get("open_time", "")[:16]
        status = "open" if sorted(open_otm) else "upcoming"
        print(f"  [kalshi] Nearest OTM contract ({status}): {best.get('ticker')}  "
              f"floor_strike=${fs:,.2f}  ({offset:+.3f}% above spot ${spot:,.2f})")
        return best

    print(f"  [kalshi] No OTM contracts found above spot ${spot:,.2f}.")
    return None


# ---------------------------------------------------------------------------
# Fetch all contracts for the nearest expiry (full strike ladder)
# ---------------------------------------------------------------------------

def minutes_to_expiry(close_ts: str, min_tau: float = 1.0) -> float:
    """
    Return minutes remaining until close_ts (UTC ISO string).
    Clamps to min_tau so tau is never zero or negative (probability engine requires tau > 0).
    """
    if not close_ts:
        return TAU  # fallback to default if no timestamp
    try:
        expiry = pd.Timestamp(close_ts).tz_convert("UTC")
        remaining = (expiry - pd.Timestamp.now(tz="UTC")).total_seconds() / 60
        return max(remaining, min_tau)
    except Exception:
        return TAU


def fetch_contracts_for_nearest_expiry(auth: KalshiAuth, spot: float = 0.0) -> list:
    """
    Return all OTM KXBTCD contracts in the nearest future expiry window that have
    a real two-sided market (both bid > 0 and ask > 0).

    Only contracts with floor_strike > spot are included. The probability engine
    is calibrated for OTM contracts (strike above spot); ITM contracts produce
    nonsense probability estimates and false edges.

    Uses yes_bid_dollars / yes_ask_dollars from the market listing endpoint —
    one API call covers the whole strike ladder, avoiding per-contract roundtrips.

    Returns a list of dicts sorted by floor_strike ascending:
        {ticker, floor_strike, p_market (mid), bid, ask, close_time}
    Returns an empty list if auth fails or no liquid OTM contracts exist.
    """
    now_ts = int(time.time())
    now_dt = datetime.now(timezone.utc)

    all_markets = []
    cursor = None
    while True:
        params = {
            "series_ticker": SERIES_TICKER,
            "min_close_ts":  now_ts,
            "max_close_ts":  now_ts + 86400,   # next 24 hours
            "limit":         200,
        }
        if cursor:
            params["cursor"] = cursor
        data = kalshi_get("/markets", params, auth)
        page = data.get("markets") or []
        all_markets.extend(page)
        cursor = data.get("cursor")
        if not cursor or len(page) < 200:
            break

    if not all_markets:
        return []

    # Find the nearest future close_time
    future_close_times = set()
    for m in all_markets:
        ct = m.get("close_time", "")
        if not ct:
            continue
        try:
            if pd.Timestamp(ct).tz_convert("UTC") > now_dt:
                future_close_times.add(ct)
        except Exception:
            pass

    if not future_close_times:
        return []

    nearest_expiry = min(future_close_times)

    # Extract liquid contracts at that expiry
    contracts = []
    for m in all_markets:
        if m.get("close_time") != nearest_expiry:
            continue
        fs = m.get("floor_strike")
        if fs is None:
            continue
        try:
            fs = float(fs)
        except (ValueError, TypeError):
            continue
        try:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
        except (ValueError, TypeError):
            continue
        if bid <= 0 or ask <= 0:
            continue
        if spot > 0 and fs <= spot:
            continue   # skip ITM contracts — probability engine is OTM-only
        contracts.append({
            "ticker":       m.get("ticker", ""),
            "floor_strike": fs,
            "p_market":     (bid + ask) / 2,
            "bid":          bid,
            "ask":          ask,
            "close_time":   nearest_expiry,
        })

    return sorted(contracts, key=lambda x: x["floor_strike"])


# ---------------------------------------------------------------------------
# Fetch live BTC spot price from BRTI constituent exchanges
# ---------------------------------------------------------------------------

def fetch_live_spot() -> Optional[float]:
    """
    Fetch the current BTC spot price as an average of BRTI constituent exchanges.

    Kalshi settles KXBTCD contracts against the CF Benchmarks Bitcoin Real-Time
    Index (BRTI) — a composite of Bitstamp, Coinbase, Gemini, and Kraken.
    Averaging the same four exchanges gives a spot price that matches the
    settlement index far more closely than any single-exchange feed.
    """
    sources = [
        ("coinbase",  lambda: float(requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=8).json()["data"]["amount"])),
        ("kraken",    lambda: float(requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=8).json()["result"]["XXBTZUSD"]["c"][0])),
        ("bitstamp",  lambda: float(requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=8).json()["last"])),
        ("gemini",    lambda: float(requests.get("https://api.gemini.com/v1/pubticker/btcusd", timeout=8).json()["last"])),
    ]
    prices = {}
    for name, fetch in sources:
        try:
            prices[name] = fetch()
        except Exception as exc:
            print(f"  [brti] {name} unavailable: {exc}")

    if not prices:
        return None

    avg = sum(prices.values()) / len(prices)
    detail = "  ".join(f"{k}=${v:,.2f}" for k, v in prices.items())
    print(f"  [brti] {detail}")
    print(f"  [brti] Average (BRTI proxy): ${avg:,.2f}  ({len(prices)}/4 sources)")
    return avg


# ---------------------------------------------------------------------------
# Fetch recent BTC candles from Binance US
# ---------------------------------------------------------------------------

def fetch_recent_candles(interval: str = "1m", lookback_bars: int = 70) -> Optional[pd.DataFrame]:
    """
    Fetch the last `lookback_bars` BTCUSDT candles at `interval` from Binance US.
    interval: "1m", "1h", "4h", etc.
    Returns a DataFrame with a UTC DatetimeIndex and OHLCV columns, or None on failure.
    """
    try:
        url = "https://api.binance.us/api/v3/klines"
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": interval,
                                      "limit": lookback_bars}, timeout=10)
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
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        print(f"  [binance] fetch_recent_candles({interval}) failed: {exc}")
        return None


def fetch_recent_1m_candles(lookback_bars: int = 70) -> Optional[pd.DataFrame]:
    """Convenience wrapper for 1m candles (used for short-term momentum)."""
    return fetch_recent_candles("1m", lookback_bars)


def extend_with_live_candles(df: pd.DataFrame, interval: str, lookback_bars: int) -> pd.DataFrame:
    """
    Fetch the most recent `lookback_bars` candles for `interval` from Binance US
    and append any bars that are newer than the last row in `df`.

    Returns the extended DataFrame (original + new rows), sorted by index.
    Falls back silently to the original `df` if the fetch fails.
    """
    fresh = fetch_recent_candles(interval, lookback_bars)
    if fresh is None:
        return df
    # Only keep rows newer than the last parquet timestamp
    cutoff = df.index[-1]
    new_rows = fresh[fresh.index > cutoff]
    if new_rows.empty:
        return df
    extended = pd.concat([df, new_rows]).sort_index()
    print(f"  [binance] Extended {interval} data: +{len(new_rows)} bars "
          f"(now up to {extended.index[-1].strftime('%Y-%m-%d %H:%M UTC')})")
    return extended


# ---------------------------------------------------------------------------
# Fetch current mid-price from candlestick
# ---------------------------------------------------------------------------

def fetch_current_price(auth: KalshiAuth, ticker: str) -> Optional[float]:
    """
    Fetch the most recent 1-minute candlestick and return the YES mid-price.
    Uses a 2-minute window ending now to capture the most recent candle.
    """
    now      = int(time.time())
    start_ts = now - CANDLE_WINDOW
    end_ts   = now + 60

    path   = f"/series/{SERIES_TICKER}/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1}
    data   = kalshi_get(path, params, auth)
    candles = data.get("candlesticks") or []

    if not candles:
        print(f"  [kalshi] No candlesticks returned for {ticker}.")
        return None

    # Use the most recent candle with valid bid/ask
    for c in reversed(candles):
        bid_d = c.get("yes_bid") or {}
        ask_d = c.get("yes_ask") or {}
        try:
            bid = float(bid_d.get("close_dollars") or bid_d.get("open_dollars") or 0)
            ask = float(ask_d.get("close_dollars") or ask_d.get("open_dollars") or 0)
        except (ValueError, TypeError):
            continue
        if ask > 0:
            mid = (bid + ask) / 2.0
            print(f"  [kalshi] Current YES price: bid={bid:.3f}  ask={ask:.3f}  mid={mid:.3f}")
            return mid

    print(f"  [kalshi] All candlesticks had zero/missing prices for {ticker}.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live signal: run the model with real Kalshi market probability."
    )
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET,
                        help=f"Strike offset above spot (default {DEFAULT_OFFSET})")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                        help=f"Bankroll for Kelly sizing (default ${DEFAULT_BANKROLL:,})")
    parser.add_argument("--sim", action="store_true",
                        help="Use simulated p_market instead of real Kalshi price")
    args = parser.parse_args()

    print("=" * 62)
    print("  LIVE SIGNAL GENERATOR")
    print("=" * 62)
    print(f"\n  Run time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Auth ---
    auth = None
    if not args.sim:
        auth = load_auth()
        if auth is None:
            print("\n  WARNING: No Kalshi credentials found (KALSHI_KEY_ID / KALSHI_KEY_PATH).")
            print("  Falling back to simulated p_market.  Use --sim to suppress this warning.")
        else:
            print("  Kalshi auth: loaded.")

    # --- Load OHLCV data ---
    print("\nLoading cached OHLCV data...")
    df_1m, df_1h, df_4h = load_data()

    # Decision timestamp = most recent completed 1m candle
    ts = df_1m.index[-1]
    print(f"\n  Decision timestamp (UTC): {ts.strftime('%Y-%m-%d %H:%M')}")

    # --- Spot price: live from Binance, fallback to last cached candle ---
    live_spot = fetch_live_spot()
    spot = live_spot if live_spot is not None else float(df_1m["close"].iloc[-1])
    print(f"  Spot:   ${spot:,.2f}  ({'live' if live_spot else 'cached'})")

    # --- Compute model signals ---
    hist_1m = df_1m.iloc[-200:]
    hist_1h = df_1h.iloc[-100:]
    hist_4h = df_4h.iloc[-120:]

    vol     = compute_realized_volatility(hist_1m)
    struct  = detect_market_structure(hist_4h)
    confirm = compute_confirmation(hist_1h)
    gate_side = "yes" if struct.structure_bias == 1 else "no"

    # --- Find nearest OTM Kalshi contract and derive real strike/offset ---
    contract  = None
    p_market  = None
    used_real = False
    strike    = spot * (1 + abs(args.offset))   # default fallback

    if auth is not None:
        print("\n  Fetching live Kalshi contract...")
        contract = find_live_contract(auth, spot)
        if contract is not None:
            strike   = float(contract.get("floor_strike", strike))
            ticker   = contract.get("ticker", "")
            p_market = fetch_current_price(auth, ticker)
            if p_market is not None:
                used_real = True

    effective_offset = strike / spot - 1
    prob = estimate_probability(spot, strike, TAU, vol.vol_60m)

    if p_market is None:
        p_market = simulate_p_market(effective_offset, side=gate_side)
        source = "simulated" if (args.sim or auth is None) else "simulated (no live price available)"
        print(f"\n  p_market source: {source}  →  {p_market:.4f}")
    else:
        print(f"\n  p_market source: real Kalshi mid-price  →  {p_market:.4f}")

    # --- Run full decision pipeline ---
    pricing = evaluate_edge(prob.p_yes, p_market)
    kelly   = compute_kelly_size(prob.p_yes, p_market, args.bankroll, side=gate_side)
    dec     = evaluate_trade(struct.structure_bias, confirm.confirmation_bias,
                             prob.p_yes, p_market, args.bankroll)

    # --- Print signal report ---
    W = 62

    print("\n" + "=" * W)
    print("  LIVE SIGNAL REPORT")
    print("=" * W)

    def row(label, value, width=28):
        print(f"  {label:<{width}} {value}")

    print("\n── MARKET ──────────────────────────────────────────────")
    row("Decision time (UTC):", ts.strftime("%Y-%m-%d %H:%M"))
    row("Spot:", f"${spot:,.2f}")
    row("Strike:", f"${strike:,.2f}  ({effective_offset:+.3%} from spot)")
    row("p_market (Kalshi):", f"{p_market:.4f}  ({'real' if used_real else 'simulated'})")

    print("\n── MODEL ───────────────────────────────────────────────")
    row("vol_60m:", f"{vol.vol_60m:.6f}")
    row("p_yes (model):", f"{prob.p_yes:.4f}  ({prob.p_yes:.2%})")
    row("z_score:", f"{prob.z_score:+.4f}")
    row("expected_move_pct:", f"{prob.expected_move_pct:.4f}%")
    row("structure_bias:", f"{struct.structure_bias:+d}  ({struct.reason[:35]})")
    row("confirmation_bias:", f"{confirm.confirmation_bias:+d}  ({confirm.reason[:35]})")
    row("ema_alignment:", confirm.ema_alignment)
    row("rsi:", f"{confirm.rsi_value:.1f}  ({confirm.rsi_regime})")

    print("\n── EDGE ────────────────────────────────────────────────")
    row("raw_edge:", f"{pricing.raw_edge:+.4f}")
    row("net_edge:", f"{pricing.net_edge:+.4f}")
    row("qualifies:", str(pricing.qualifies))

    print("\n── DECISION ────────────────────────────────────────────")
    row("decision:", dec.decision.upper())
    row("side:", dec.side.upper())
    for i, reason in enumerate(dec.reasons, 1):
        print(f"    {i}. {reason}")

    if dec.decision == "trade":
        print("\n── SIZING ──────────────────────────────────────────────")
        row("kelly_fraction:", f"{dec.kelly_fraction:.4f}  ({dec.kelly_fraction:.2%})")
        row("bet_fraction:", f"{dec.bet_fraction:.4f}  ({dec.bet_fraction:.2%}){'  [capped]' if dec.was_capped else ''}")
        row("bankroll:", f"${args.bankroll:,.2f}")
        row("bet_amount:", f"${dec.bet_amount:,.2f}")

        print("\n── ACTION ──────────────────────────────────────────────")
        if used_real and dec.decision == "trade":
            # Find and show the contract to trade
            if contract:
                row("Contract:", contract.get("ticker", "?"))
                row("Trade:", f"BUY {dec.side.upper()}")
                row("Amount:", f"${dec.bet_amount:,.2f}")
                row("At:", f"p_market={p_market:.3f}")
        elif dec.decision == "trade":
            print("  (Run without --sim and with valid credentials to see the live contract)")

    print("\n" + "=" * W + "\n")


if __name__ == "__main__":
    main()
