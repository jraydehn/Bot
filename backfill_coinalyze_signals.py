"""
backfill_coinalyze_signals.py

Backfills liq_score, liq_bias, ls_long_pct, oi_chg_pct into:
  - results/paper_trades.csv   (and per-asset files)
  - results/blocked_trades.csv

Fetches 1h historical bars from Coinalyze for each asset over the full
date range present in each CSV, then matches each row to its 1h bucket.

Usage:
    python3 backfill_coinalyze_signals.py
"""

import csv
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://api.coinalyze.net/v1"
API_KEY  = os.environ.get("COINALYZE_API_KEY", "d5841821-3f45-4e5f-9ee7-d2779d2fb01b")

SYMBOLS = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}

# Scoring thresholds (mirror coinalyze_liq.py)
LIQ_BIAS_STRONG  = 0.60
LS_CROWD_THRESH  = 65.0

RESULTS = Path(__file__).parent / "results"


# ── Coinalyze fetch ────────────────────────────────────────────────────────

def fetch_history(endpoint: str, symbol: str, start_ts: int, end_ts: int,
                  interval: str = "1hour") -> list[dict]:
    """Fetch all bars in [start_ts, end_ts] for a given endpoint, paginating if needed."""
    all_rows = []
    chunk = 500   # bars per request
    t = start_ts
    while t < end_ts:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params={"symbols": symbol, "interval": interval,
                    "from": t, "to": min(t + chunk * 3600, end_ts),
                    "api_key": API_KEY},
            timeout=10,
        )
        data = r.json()
        if not isinstance(data, list) or not data or not data[0].get("history"):
            break
        rows = data[0]["history"]
        if not rows:
            break
        all_rows.extend(rows)
        t = rows[-1]["t"] + 3600   # advance past last bar
        if len(rows) < 2:
            break
        time.sleep(0.2)  # polite rate limit
    return all_rows


def build_lookup(asset: str, start_ts: int, end_ts: int) -> dict[int, dict]:
    """
    Returns a dict keyed by hour-aligned Unix timestamp.
    Each value: {liq_bias, ls_long_pct, ls_short_pct, oi_chg_pct, liq_score}
    """
    sym = SYMBOLS.get(asset.upper())
    if not sym:
        return {}

    print(f"  Fetching {asset} liquidation-history…")
    liq_rows = fetch_history("liquidation-history", sym, start_ts, end_ts)

    print(f"  Fetching {asset} long-short-ratio-history…")
    ls_rows  = fetch_history("long-short-ratio-history", sym, start_ts, end_ts)

    print(f"  Fetching {asset} open-interest-history…")
    oi_rows  = fetch_history("open-interest-history", sym, start_ts, end_ts)

    # Index by timestamp
    liq_by_ts = {r["t"]: r for r in liq_rows}
    ls_by_ts  = {r["t"]: r for r in ls_rows}
    oi_list   = sorted(oi_rows, key=lambda r: r["t"])

    # OI previous bar lookup
    oi_prev: dict[int, float] = {}
    for i, row in enumerate(oi_list):
        if i > 0:
            oi_prev[row["t"]] = float(oi_list[i - 1]["o"])

    all_ts = sorted(set(list(liq_by_ts) + list(ls_by_ts)))
    lookup: dict[int, dict] = {}

    for ts in all_ts:
        liq_r = liq_by_ts.get(ts, {})
        ls_r  = ls_by_ts.get(ts, {})

        long_liq  = float(liq_r.get("l", 0))
        short_liq = float(liq_r.get("s", 0))
        total_liq = long_liq + short_liq
        liq_bias  = (short_liq - long_liq) / total_liq if total_liq > 0.001 else 0.0

        ls_long_pct  = float(ls_r.get("l", 50))
        ls_short_pct = float(ls_r.get("s", 50))

        oi_chg = 0.0
        prev_oi = oi_prev.get(ts)
        oi_r = next((r for r in oi_list if r["t"] == ts), None)
        if prev_oi and oi_r and prev_oi > 0:
            oi_chg = (float(oi_r["o"]) - prev_oi) / prev_oi * 100.0

        score = 0
        if liq_bias >= LIQ_BIAS_STRONG:
            score += 1
        elif liq_bias <= -LIQ_BIAS_STRONG:
            score -= 1
        if ls_short_pct >= LS_CROWD_THRESH:
            score += 1
        elif ls_long_pct >= LS_CROWD_THRESH:
            score -= 1
        score = max(-2, min(2, score))

        lookup[ts] = {
            "liq_score":   score,
            "liq_bias":    round(liq_bias, 4),
            "ls_long_pct": round(ls_long_pct, 2),
            "oi_chg_pct":  round(oi_chg, 4),
        }

    print(f"  Built {len(lookup)} hour-bars for {asset}")
    return lookup


def match_signal(ts_unix: int, lookup: dict[int, dict]) -> dict:
    """Return the signal for the most recent completed 1h bar before ts_unix."""
    # Round down to hour boundary
    hour_ts = (ts_unix // 3600) * 3600
    # Walk back up to 3h if exact match missing
    for offset in [0, 3600, 7200, 10800]:
        sig = lookup.get(hour_ts - offset)
        if sig:
            return sig
    return {"liq_score": "", "liq_bias": "", "ls_long_pct": "", "oi_chg_pct": ""}


# ── CSV patching ───────────────────────────────────────────────────────────

TARGET_COLS = ["liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct"]


def patch_csv(csv_path: Path, asset: str, lookup: dict[int, dict],
              ts_col: str = "logged_at") -> int:
    """Overwrite csv_path filling TARGET_COLS from lookup. Returns rows patched."""
    if not csv_path.exists():
        return 0

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        fieldnames = csv.DictReader(f).fieldnames or []

    # Add missing columns
    for col in TARGET_COLS:
        if col not in fieldnames:
            fieldnames.append(col)

    patched = 0
    for row in rows:
        raw_ts = row.get(ts_col, "")
        if not raw_ts:
            continue
        try:
            dt = datetime.strptime(str(raw_ts)[:19], "%Y-%m-%d %H:%M:%S")
            ts_unix = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue

        sig = match_signal(ts_unix, lookup)
        # Only fill if currently empty (don't overwrite live data)
        for col in TARGET_COLS:
            if row.get(col, "") == "":
                row[col] = sig.get(col, "")
                patched += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return patched


def patch_blocked_csv(csv_path: Path, lookups: dict[str, dict]) -> int:
    """Patch blocked_trades.csv which has an 'asset' column."""
    if not csv_path.exists():
        return 0

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        fieldnames = csv.DictReader(f).fieldnames or []

    for col in TARGET_COLS:
        if col not in fieldnames:
            fieldnames.append(col)

    patched = 0
    for row in rows:
        asset = row.get("asset", "BTC").upper()
        lookup = lookups.get(asset, {})
        raw_ts = row.get("logged_at", "")
        if not raw_ts:
            continue
        try:
            dt = datetime.strptime(str(raw_ts)[:19], "%Y-%m-%d %H:%M:%S")
            ts_unix = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue

        sig = match_signal(ts_unix, lookup)
        for col in TARGET_COLS:
            if row.get(col, "") == "":
                row[col] = sig.get(col, "")
                patched += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return patched


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Determine date range from paper_trades
    pt_btc = pd.read_csv(RESULTS / "paper_trades.csv", low_memory=False)
    pt_btc["logged_at"] = pd.to_datetime(pt_btc["logged_at"], errors="coerce")
    pt_btc = pt_btc[pt_btc["logged_at"].notna()]

    global_start = int(pt_btc["logged_at"].min().timestamp()) - 7200
    global_end   = int(datetime.now(timezone.utc).timestamp())

    print(f"Date range: {pt_btc['logged_at'].min()} → now")
    print(f"Timestamps: {global_start} → {global_end}\n")

    lookups: dict[str, dict] = {}
    for asset in ["BTC", "ETH", "SOL"]:
        print(f"\n── {asset} ──")
        lookup = build_lookup(asset, global_start, global_end)
        lookups[asset] = lookup

    print("\n── Patching CSVs ──")

    asset_files = {
        "BTC": RESULTS / "paper_trades.csv",
        "ETH": RESULTS / "paper_trades_eth.csv",
        "SOL": RESULTS / "paper_trades_sol.csv",
    }
    for asset, path in asset_files.items():
        n = patch_csv(path, asset, lookups[asset])
        print(f"  {path.name}: {n} cells filled")

    n = patch_blocked_csv(RESULTS / "blocked_trades.csv", lookups)
    print(f"  blocked_trades.csv: {n} cells filled")

    print("\nDone.")


if __name__ == "__main__":
    main()
