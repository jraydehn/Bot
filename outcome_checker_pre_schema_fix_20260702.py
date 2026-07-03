"""
Outcome checker — fetches Kalshi contract settlements and updates paper_trades.csv.

For each row in results/paper_trades.csv that has a contract_ticker and an empty
resolved_yes field, this script:
  1. Fetches the market detail from the Kalshi API.
  2. If the market is settled, reads the result (yes_sub_title indicates outcome).
  3. Fills in resolved_yes, would_win, and would_pnl.

Run this once after each contract expires (e.g. hourly, offset by 10 minutes):
    python3 outcome_checker.py

Requires KALSHI_KEY_ID and KALSHI_KEY_PATH env vars.
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_python_sync import KalshiAuth
from live_signal import load_auth, kalshi_get, BASE_URL, fetch_spot_at_time

PAPER_TRADES_CSV = Path(__file__).parent / "results" / "paper_trades.csv"

CSV_COLUMNS = [
    "logged_at", "decision_time", "contract_ticker", "close_ts",
    "spot", "strike", "offset_pct", "p_market", "p_market_source",
    "p_yes_model", "z_score", "vol_60m", "vol_60m_model", "vol_implied_kalshi", "vol_ratio", "spread", "vol_eff",
    "structure_bias", "confirmation_bias", "confirmation_score", "no_score",
    "obi_score", "obi_raw", "obi_exchanges",
    "vpin_score", "vpin_raw",
    "funding_bias", "avg_funding_rate",
    "vol_score", "cmf_raw", "cmf_score",
    "vwap_score", "vwap_signal", "vwap_total", "vwap_stretch_score", "vwap_distance_pct", "bearish_rejection", "bullish_rejection", "ema_stretch_score",
    "stoch_bias", "stoch_k", "stoch_d", "stoch_crossover_active",
    "ema_stack_bias",
    "ema_alignment", "z_shift", "direction_strength", "raw_edge", "net_edge",
    "decision", "side", "neutral_gate", "pure_edge_gate",
    "contracts_scanned", "tau_minutes", "gate_blocked",
    "kelly_fraction", "bet_fraction", "bet_amount", "bankroll",
    "composite_trend", "composite_rev", "composite_p_up",
    "p_up_v2",
    "chg_30m", "chg_10m", "chg_5m",
    "bp_5m", "bp_1h", "chg_1h", "chg_2h", "chg_3h",
    "body_15m", "dir_15m", "p_gbdt",
    "sharp_move_active",
    "smc_4h", "smc_1h", "choch_1h", "choch_4h",
    "supply_pct", "demand_pct", "in_supply_zone", "in_demand_zone",
    "stoch_flipped",
    "squeeze_1h",
    "adx_1h",
    "rvol_1h",
    "pm_drift_5m",
    "hour_utc",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
    "ob_imbalance", "ob_path_ask_usd", "ob_path_bid_usd", "ob_ask_frac",
    "ob_bid_wall_pct", "ob_ask_wall_pct",
    "resolved_yes", "would_win", "would_pnl",
    "spot_at_expiry", "price_move_pct", "miss_pct",
    "loss_margin_pct", "loss_category",
]


def _loss_category(miss_pct: float, would_win: bool, tau_minutes: float) -> tuple:
    """Return (loss_margin_pct, loss_category) using tau-scaled thresholds.

    sigma_tau = 0.07% × sqrt(tau_minutes / 30)  — 1-sigma BTC move over remaining window.
      near_miss  : loss_margin < 1× sigma_tau   — within 1 sigma, genuine bad luck
      marginal   : 1× ≤ loss_margin < 3× sigma_tau
      clear_loss : loss_margin ≥ 3× sigma_tau   — decisively wrong signal
    """
    if would_win:
        return 0.0, "win"
    loss_margin = abs(miss_pct)
    sigma_tau = 0.07 * math.sqrt(max(tau_minutes, 0.5) / 30.0)
    if loss_margin < sigma_tau:
        category = "near_miss"
    elif loss_margin < 3.0 * sigma_tau:
        category = "marginal"
    else:
        category = "clear_loss"
    return round(loss_margin, 4), category


def fetch_market(ticker: str, auth: KalshiAuth) -> dict:
    data = kalshi_get(f"/markets/{ticker}", {}, auth)
    return data.get("market") or {}


def is_settled(market: dict) -> bool:
    return market.get("status") in ("settled", "resolved", "finalized", "determined")


def parse_resolution(market: dict) -> bool:
    """
    Return True if the YES side won (i.e. BTC closed ABOVE the strike).
    Kalshi sets result to 'yes' or 'no'.
    """
    result = (market.get("result") or "").lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    # Fallback: check yes_sub_title for clues
    sub = (market.get("yes_sub_title") or "").lower()
    if "above" in sub or "higher" in sub:
        # YES condition met
        return True
    raise ValueError(f"Cannot determine resolution from market dict: result={result!r}")


def compute_pnl(row: dict, resolved_yes: bool) -> float:
    """
    Estimate paper P&L based on what would have happened.
    - YES trade: win +bet_amount * (1/p_market - 1), lose -bet_amount
    - NO trade : win +bet_amount * (1/(1-p_market) - 1), lose -bet_amount
    Does not deduct fees (already reflected in net_edge gate).
    """
    bet  = float(row["bet_amount"])
    pm   = float(row["p_market"])
    side = row["side"]

    if side == "yes":
        if resolved_yes:
            return round(bet * (1.0 / pm - 1.0), 2)
        else:
            return round(-bet, 2)
    else:  # NO trade
        if not resolved_yes:
            return round(bet * (1.0 / (1.0 - pm) - 1.0), 2)
        else:
            return round(-bet, 2)


def recover_missing_tickers(rows: list, auth: KalshiAuth) -> int:
    """
    For trade rows with no contract_ticker, try to find the matching Kalshi contract
    by querying for settled contracts near the row's logged_at time and closest strike.
    Updates the row in-place. Returns the number of rows recovered.
    """
    from live_signal import kalshi_get, ASSET_CONFIG
    import time as _time

    recovered = 0
    for row in rows:
        if (row.get("decision") or "").strip() != "trade":
            continue
        if (row.get("contract_ticker") or "").strip():
            continue  # already has a ticker

        try:
            logged_dt = datetime.fromisoformat(row["logged_at"].replace("Z", "+00:00"))
            if logged_dt.tzinfo is None:
                logged_dt = logged_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        strike = float(row.get("strike") or 0)
        if not strike:
            continue

        # Search a 2-hour window around the logged time for settled contracts
        series = "KXBTCD"
        window_start = int((logged_dt.timestamp()) - 60)
        window_end   = int((logged_dt.timestamp()) + 7200)

        try:
            data = kalshi_get("/markets", {
                "series_ticker": series,
                "min_close_ts":  window_start,
                "max_close_ts":  window_end,
                "limit":         200,
            }, auth)
        except Exception as e:
            print(f"  [recover] API error: {e}")
            continue

        markets = data.get("markets") or []
        if not markets:
            continue

        # Find settled contract with closest floor_strike to our simulated strike
        settled = [m for m in markets if m.get("status") in ("settled", "resolved", "finalized", "determined")]
        if not settled:
            continue

        best = min(settled, key=lambda m: abs(float(m.get("floor_strike") or m.get("strike_value") or 0) - strike))
        best_strike = float(best.get("floor_strike") or best.get("strike_value") or 0)

        if abs(best_strike - strike) > strike * 0.02:  # >2% away — skip, likely wrong contract
            print(f"  [recover] No close match for strike={strike:.2f} (closest={best_strike:.2f})")
            continue

        ticker   = best.get("ticker", "")
        close_ts = best.get("close_time", "")
        row["contract_ticker"] = ticker
        row["close_ts"]        = close_ts

        # Fetch the real p_market from the candlestick at the time of the trade
        try:
            trade_ts = int(logged_dt.timestamp())
            series   = best.get("series_ticker", "KXBTCD")
            path     = f"/series/{series}/markets/{ticker}/candlesticks"
            cdata    = kalshi_get(path, {"start_ts": trade_ts - 120, "end_ts": trade_ts + 60, "period_interval": 1}, auth)
            candles  = cdata.get("candlesticks") or []
            real_pm  = None
            for c in reversed(candles):
                bid_d = c.get("yes_bid") or {}
                ask_d = c.get("yes_ask") or {}
                try:
                    bid = float(bid_d.get("close_dollars") or bid_d.get("open_dollars") or 0)
                    ask = float(ask_d.get("close_dollars") or ask_d.get("open_dollars") or 0)
                except (ValueError, TypeError):
                    continue
                if ask > 0:
                    real_pm = round((bid + ask) / 2.0, 6)
                    break
            if real_pm is not None:
                row["p_market"]        = str(real_pm)
                row["p_market_source"] = "real"
                print(f"  [recover] Matched strike={strike:.2f} → {ticker}  p_market updated: {real_pm:.4f} (close={close_ts})")
            else:
                print(f"  [recover] Matched strike={strike:.2f} → {ticker} (no candlestick price found, p_market unchanged)")
        except Exception as e:
            print(f"  [recover] Ticker matched but candlestick fetch failed: {e}")

        recovered += 1
        _time.sleep(0.1)

    return recovered


def main(csv_path: Path = None) -> None:
    target = csv_path or PAPER_TRADES_CSV
    if not target.exists():
        print(f"No trades CSV found at {target}. Run paper_trade_runner.py first.")
        return

    auth = load_auth()
    if auth is None:
        print("ERROR: KALSHI_KEY_ID / KALSHI_KEY_PATH not set.")
        sys.exit(1)

    # Read all rows
    with open(target, newline="") as f:
        rows = list(csv.DictReader(f))

    # Try to recover tickers for any trade rows that were logged without one
    recovered = recover_missing_tickers(rows, auth)
    if recovered:
        print(f"  Recovered {recovered} missing ticker(s).")

    updated = 0
    skipped = 0

    for row in rows:
        ticker = row.get("contract_ticker", "").strip()

        # Skip rows without a ticker, already resolved, or not a trade
        if not ticker:
            skipped += 1
            continue
        if (row.get("resolved_yes") or "").strip():
            skipped += 1
            continue
        if (row.get("decision") or "").strip() != "trade":
            skipped += 1
            continue

        # Check if contract close time has passed
        close_ts_str = row.get("close_ts", "").strip()
        if close_ts_str:
            try:
                close_dt = datetime.fromisoformat(close_ts_str.replace("Z", "+00:00"))
                if close_dt > datetime.now(timezone.utc):
                    skipped += 1
                    continue  # not expired yet
            except Exception:
                pass  # can't parse — try anyway

        market = fetch_market(ticker, auth)
        if not market:
            print(f"  {ticker}: could not fetch market data")
            skipped += 1
            continue

        if not is_settled(market):
            status = market.get("status", "unknown")
            print(f"  {ticker}: not yet settled (status={status})")
            skipped += 1
            continue

        try:
            resolved_yes = parse_resolution(market)
        except ValueError as e:
            print(f"  {ticker}: {e}")
            skipped += 1
            continue

        row["resolved_yes"] = str(resolved_yes)

        # Compute would_win only for trade rows with a valid bet_amount
        if (row.get("decision") or "").strip() == "trade" and (row.get("bet_amount") or "").strip():
            side = row.get("side", "yes")
            if side == "yes":
                would_win = resolved_yes
            else:
                would_win = not resolved_yes
            row["would_win"] = str(would_win)
            row["would_pnl"] = str(compute_pnl(row, resolved_yes))
        else:
            row["would_win"] = ""
            row["would_pnl"] = ""

        # Log expiry price if not already present
        if not (row.get("spot_at_expiry") or "").strip() and close_ts_str:
            _asset = "BTC"
            _p = str(target)
            if "eth" in _p.lower():
                _asset = "ETH"
            elif "sol" in _p.lower():
                _asset = "SOL"
            spot_exp  = fetch_spot_at_time(close_ts_str, _asset)
            spot_scan = float(row.get("spot") or 0)
            strike    = float(row.get("strike") or 0)
            if spot_exp and spot_scan > 0:
                row["spot_at_expiry"] = round(spot_exp, 2)
                row["price_move_pct"] = round((spot_exp - spot_scan) / spot_scan * 100, 4)
            if spot_exp and strike > 0:
                row["miss_pct"] = round((spot_exp - strike) / strike * 100, 4)

        # Compute tau-scaled loss quality label
        _miss_s = str(row.get("miss_pct") or "").strip()
        _tau_s  = str(row.get("tau_minutes") or "").strip()
        _ww_s   = str(row.get("would_win") or "").strip()
        if _miss_s and _tau_s and _ww_s:
            try:
                _lm, _lc = _loss_category(float(_miss_s), _ww_s.lower() == "true", float(_tau_s))
                row["loss_margin_pct"] = _lm
                row["loss_category"]   = _lc
            except (ValueError, TypeError):
                pass

        updated += 1
        print(f"  {ticker}: resolved_yes={resolved_yes}  would_win={row['would_win']}  "
              f"would_pnl={row['would_pnl']}  loss_category={row.get('loss_category','?')}")
        time.sleep(0.2)  # rate-limit courtesy

    # Backfill loss_category for already-resolved rows that predate this column
    backfilled = 0
    for row in rows:
        if row.get("loss_category"):
            continue
        _miss_s = str(row.get("miss_pct") or "").strip()
        _tau_s  = str(row.get("tau_minutes") or "").strip()
        _ww_s   = str(row.get("would_win") or "").strip()
        if not (_miss_s and _tau_s and _ww_s):
            continue
        try:
            _lm, _lc = _loss_category(float(_miss_s), _ww_s.lower() == "true", float(_tau_s))
            row["loss_margin_pct"] = _lm
            row["loss_category"]   = _lc
            backfilled += 1
        except (ValueError, TypeError):
            pass
    if backfilled:
        print(f"  [backfill] loss_category filled for {backfilled} existing rows.")
        updated += backfilled

    # Write all rows back
    if updated > 0:
        with open(target, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Updated {updated} rows in {target}")
    else:
        print(f"\n  No rows updated (skipped={skipped}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outcome checker")
    parser.add_argument("--asset", type=str, default=None,
                        help="Asset CSV to check: BTC, ETH, SOL (default: all)")
    args = parser.parse_args()

    if args.asset:
        from paper_trade_runner import get_csv_path
        main(get_csv_path(args.asset.upper()))
    else:
        # Process all known asset CSVs
        from paper_trade_runner import get_csv_path
        for asset in ("BTC", "ETH", "SOL"):
            p = get_csv_path(asset)
            if p.exists():
                print(f"\n--- {asset} ---")
                main(p)

        # Fill position monitor outcomes
        print("\n--- position monitor ---")
        try:
            import position_monitor
            position_monitor.fill_outcomes()
        except Exception as e:
            print(f"  [pos_monitor] fill_outcomes error: {e}")
