#!/usr/bin/env python3
"""
simulate_zdrift_15m.py — Walk-forward simulation of empirical z_drift for BTC 15m model.

For each day in the resolved trade history:
  1. Compute z_drift from all prior resolved rows (no lookahead)
  2. Apply z_drift correction: p_adj = norm.cdf(norm.ppf(p_model) + z_drift)
  3. Re-evaluate edge vs EDGE_THRESHOLD; blocked trades -> P&L = 0
  4. Compare daily and total P&L vs original model

Fetches Binance 1m klines to compute actual_z = log(price_at_expiry/spot)/sigma_tau.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR   = Path(__file__).parent / "results"
CSV_15M       = RESULTS_DIR / "paper_trades_btc15m.csv"
MINS_PER_YEAR = 525600.0
EDGE_THRESHOLD = 0.04

W_SHORT = 10
W_LONG  = 30
ALPHA   = 0.6
CAP     = 0.5


# ---------------------------------------------------------------------------
# Binance historical klines
# ---------------------------------------------------------------------------

def fetch_binance_1m(symbol: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    url  = "https://api.binance.us/api/v3/klines"
    rows = []
    cur  = start_ts_ms
    while cur < end_ts_ms:
        params = {
            "symbol":    symbol,
            "interval":  "1m",
            "startTime": cur,
            "endTime":   min(cur + 1000 * 60 * 1000, end_ts_ms),
            "limit":     1000,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        cur = int(data[-1][0]) + 60_000
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time_ms", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    for c in df.columns:
        df[c] = df[c].astype(float)
    return df


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_actual_z(row: pd.Series, m1_idx: pd.DatetimeIndex, m1_open: pd.Series) -> float:
    try:
        spot    = float(row["spot"])
        rv_ann  = float(row["realized_vol_annual"])
        tau_min = float(row["tau_minutes"])
        if spot <= 0 or rv_ann <= 0 or tau_min <= 0:
            return float("nan")
        vol_eff   = rv_ann / math.sqrt(MINS_PER_YEAR)
        sigma_tau = vol_eff * math.sqrt(tau_min)
        if sigma_tau <= 0:
            return float("nan")
        close_ts = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        if close_ts in m1_idx:
            btc_expiry = float(m1_open[close_ts])
        else:
            i = m1_idx.searchsorted(close_ts)
            if i >= len(m1_idx):
                return float("nan")
            btc_expiry = float(m1_open.iloc[i])
        return math.log(btc_expiry / spot) / sigma_tau
    except Exception:
        return float("nan")


def compute_zdrift(az_list: list[float]) -> float:
    n = len(az_list)
    if n < W_SHORT:
        return float("nan")
    tail = az_list[-max(W_LONG, n):]
    z_short = float(np.mean(tail[-W_SHORT:]))
    z_long  = float(np.mean(tail[-W_LONG:]) if len(tail) >= W_LONG else np.mean(tail))
    return float(max(-CAP, min(CAP, ALPHA * z_short + (1 - ALPHA) * z_long)))


def apply_zdrift(p_model: float, z_drift: float) -> float:
    p = max(0.05, min(0.96, p_model))
    return float(np.clip(norm.cdf(norm.ppf(p) + z_drift), 0.05, 0.96))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_15M.exists():
        print(f"CSV not found: {CSV_15M}"); sys.exit(1)

    df = pd.read_csv(CSV_15M, low_memory=False)
    for col in ["spot", "realized_vol_annual", "tau_minutes", "p_market",
                "p_model_15m", "raw_edge", "bet_amount", "would_pnl", "resolved_yes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df["logged_at"]  = pd.to_datetime(df["logged_at"],  utc=True, errors="coerce")

    resolved = df[df["resolved_yes"].notna() & df["close_time"].notna()].copy()
    resolved = resolved.sort_values("close_time").reset_index(drop=True)

    print(f"Resolved rows : {len(resolved)}")
    print(f"Resolved trades: {len(resolved[resolved['decision']=='trade'])}")
    print(f"Date range : {resolved['close_time'].min().date()} → {resolved['close_time'].max().date()}")

    # Fetch Binance 1m candles
    start_ms = int(resolved["close_time"].min().timestamp() * 1000) - 120_000
    end_ms   = int(resolved["close_time"].max().timestamp() * 1000) + 120_000
    print(f"\nFetching Binance 1m candles ({resolved['close_time'].min().date()} → {resolved['close_time'].max().date()})...")
    df_1m = fetch_binance_1m("BTCUSDT", start_ms, end_ms)
    print(f"Fetched {len(df_1m)} 1m candles")

    if df_1m.empty:
        print("ERROR: No candle data fetched"); sys.exit(1)

    # Compute actual_z for all resolved rows
    print("Computing actual_z...")
    resolved["actual_z"] = [
        compute_actual_z(row, df_1m.index, df_1m["open"])
        for _, row in resolved.iterrows()
    ]
    n_valid = resolved["actual_z"].notna().sum()
    print(f"actual_z valid: {n_valid}/{len(resolved)}")

    # Autocorrelation sanity check
    az_clean = resolved["actual_z"].dropna().values
    if len(az_clean) > 2:
        ar1 = float(pd.Series(az_clean).autocorr(lag=1))
        print(f"actual_z AR(1) = {ar1:+.4f}  (should be ~+0.64)")

    # ── Walk-forward simulation: daily buckets ─────────────────────────────────
    resolved["date"] = resolved["close_time"].dt.date
    days = sorted(resolved["date"].unique())
    print(f"\nDays: {len(days)}")

    prior_az: list[float] = []   # grows as we move forward through days
    results = []

    for day in days:
        test_mask  = resolved["date"] == day
        test_rows  = resolved[test_mask & (resolved["decision"] == "trade")].copy()

        # Compute z_drift from all PRIOR resolved rows
        z_drift = compute_zdrift(prior_az)
        n_prior = len(prior_az)
        z_drift_active = not math.isnan(z_drift)

        day_orig  = 0.0
        day_adj   = 0.0
        n_blocked = 0
        n_trades  = 0

        for _, row in test_rows.iterrows():
            p_model  = row["p_model_15m"]
            p_market = row["p_market"]
            side     = str(row.get("side", "yes")).lower()
            bet_amt  = float(row.get("bet_amount") or 0)
            outcome  = int(row["resolved_yes"])
            orig_pnl = float(row.get("would_pnl") or 0)

            if pd.isna(p_model) or pd.isna(p_market) or bet_amt <= 0 or pd.isna(orig_pnl):
                continue

            n_trades += 1
            day_orig += orig_pnl

            # Adjusted p_model
            if z_drift_active:
                p_adj = apply_zdrift(p_model, z_drift)
            else:
                p_adj = p_model  # no change

            edge_adj = (p_adj - p_market) if side == "yes" else (p_market - p_adj)

            if edge_adj < EDGE_THRESHOLD:
                # Blocked by z_drift
                n_blocked += 1
                # adj_pnl = 0 (don't take trade)
            else:
                # Recompute payout with same bet_amount and outcome
                if side == "yes":
                    payout  = (1 - p_market) / p_market if p_market > 0 else 0
                    adj_pnl = bet_amt * payout if outcome == 1 else -bet_amt
                else:
                    payout  = p_market / (1 - p_market) if p_market < 1 else 0
                    adj_pnl = bet_amt * payout if outcome == 0 else -bet_amt
                day_adj += adj_pnl

        delta = day_adj - day_orig

        results.append({
            "day":      str(day),
            "n_prior":  n_prior,
            "z_drift":  round(z_drift, 4) if z_drift_active else None,
            "n_trades": n_trades,
            "n_blocked": n_blocked,
            "orig_pnl": round(day_orig, 2),
            "adj_pnl":  round(day_adj,  2),
            "delta":    round(delta,     2),
        })

        zd_str = f"{z_drift:+.4f}" if z_drift_active else "  N/A "
        print(f"  {day}  n_prior={n_prior:3d}  z_drift={zd_str}  "
              f"trades={n_trades:3d}  blocked={n_blocked:2d}  "
              f"orig=${day_orig:+7.2f}  adj=${day_adj:+7.2f}  delta=${delta:+7.2f}")

        # Add today's resolved rows to prior_az for the next day
        today_az = resolved[test_mask]["actual_z"].dropna().tolist()
        prior_az.extend(today_az)

    # ── Summary ────────────────────────────────────────────────────────────────
    rdf = pd.DataFrame(results)
    active = rdf[rdf["z_drift"].notna()]

    total_orig  = rdf["orig_pnl"].sum()
    total_adj   = rdf["adj_pnl"].sum()
    total_delta = total_adj - total_orig
    days_pos    = (active["delta"] > 0).sum()
    days_active = len(active)

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD SUMMARY (z_drift 15m BTC)")
    print(f"{'='*60}")
    print(f"  Total days evaluated:       {len(rdf)}")
    print(f"  Days with z_drift active:   {days_active}  (n_prior≥{W_SHORT})")
    print(f"  Days positive delta:        {days_pos}/{days_active}")
    print(f"  Total trades (all days):    {rdf['n_trades'].sum()}")
    print(f"  Total blocked by z_drift:   {active['n_blocked'].sum()}")
    print(f"  Original P&L (all days):   ${total_orig:+.2f}")
    print(f"  Adjusted P&L (all days):   ${total_adj:+.2f}")
    print(f"  Net delta:                 ${total_delta:+.2f}")
    if days_active > 0:
        print(f"\n  Active-days only:")
        print(f"    Orig P&L:  ${active['orig_pnl'].sum():+.2f}")
        print(f"    Adj P&L:   ${active['adj_pnl'].sum():+.2f}")
        print(f"    Delta:     ${active['orig_pnl'].sum() - active['adj_pnl'].sum():+.2f}  "
              f"(sign: ${active['adj_pnl'].sum() - active['orig_pnl'].sum():+.2f})")
        zd_vals = [v for v in active["z_drift"] if v is not None]
        print(f"    Mean z_drift: {float(np.mean(zd_vals)):+.4f}  "
              f"range [{min(zd_vals):+.4f}, {max(zd_vals):+.4f}]")

    # ── Blocked trade breakdown ────────────────────────────────────────────────
    if active["n_blocked"].sum() > 0:
        print(f"\n  Blocked trade P&L breakdown (what we avoided):")
        for _, row in active[active["n_blocked"] > 0].iterrows():
            missed = row["orig_pnl"] - row["adj_pnl"]
            print(f"    {row['day']}  blocked={row['n_blocked']}  "
                  f"missed orig_pnl=${missed:+.2f}  (positive = we avoided losses)")


if __name__ == "__main__":
    main()
