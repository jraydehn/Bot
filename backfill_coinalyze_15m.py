#!/usr/bin/env python3
"""
backfill_coinalyze_15m.py

Fetches full Coinalyze liq + L/S + OI history for BTC, ETH, SOL and
backfills liq_score, liq_bias, oi_chg_pct into the 15m paper trade CSVs
for rows that currently have those columns blank.

Logic: for each trade row with a decision_time, find the last COMPLETED
15-min bar (floor to 15min, back one period), then look up the signals.

Run: python3 backfill_coinalyze_15m.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH

KEY  = "d5841821-3f45-4e5f-9ee7-d2779d2fb01b"
BASE = "https://api.coinalyze.net/v1"
SEP  = "=" * 72

ASSET_SYMBOLS = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}
ASSET_CSVS = {
    "BTC": Path("results/paper_trades_btc15m.csv"),
    "ETH": Path("results/paper_trades_eth15m.csv"),
    "SOL": Path("results/paper_trades_sol15m.csv"),
}


def _compute_liq_score(liq_bias: float, ls_long: float, ls_short: float) -> int:
    score = 0
    if liq_bias >= _LIQ_BIAS_STRONG:
        score += 1
    elif liq_bias <= -_LIQ_BIAS_STRONG:
        score -= 1
    if ls_short >= _LS_CROWD_THRESH:
        score += 1
    elif ls_long >= _LS_CROWD_THRESH:
        score -= 1
    return max(-2, min(2, score))


def fetch_signals(symbol: str, from_unix: int, to_unix: int) -> pd.DataFrame:
    """
    Fetch liq, L/S, and OI history from Coinalyze for the given symbol and
    time window. Returns a DataFrame indexed by bar-open UTC timestamp with
    columns: liq_bias, ls_long_pct, ls_short_pct, liq_score, oi_chg_pct.
    """
    params = {
        "symbols": symbol,
        "interval": "15min",
        "from": from_unix,
        "to": to_unix,
        "api_key": KEY,
    }

    r_liq = requests.get(f"{BASE}/liquidation-history",      params=params, timeout=15)
    time.sleep(0.3)
    r_ls  = requests.get(f"{BASE}/long-short-ratio-history", params=params, timeout=15)
    time.sleep(0.3)
    r_oi  = requests.get(f"{BASE}/open-interest-history",    params=params, timeout=15)

    r_liq.raise_for_status()
    r_ls.raise_for_status()
    r_oi.raise_for_status()

    liq_rows = r_liq.json()[0]["history"]
    ls_rows  = r_ls.json()[0]["history"]
    oi_rows  = r_oi.json()[0]["history"]

    df_liq = pd.DataFrame(liq_rows).rename(columns={"t": "ts", "l": "long_liq", "s": "short_liq"})
    df_ls  = pd.DataFrame(ls_rows).rename( columns={"t": "ts", "l": "ls_long",  "s": "ls_short"})
    df_oi  = pd.DataFrame(oi_rows).rename( columns={"t": "ts", "o": "oi"})

    df_liq["ts"] = pd.to_datetime(df_liq["ts"], unit="s", utc=True)
    df_ls["ts"]  = pd.to_datetime(df_ls["ts"],  unit="s", utc=True)
    df_oi["ts"]  = pd.to_datetime(df_oi["ts"],  unit="s", utc=True)

    df = (df_liq.set_index("ts")
          .join(df_ls.set_index("ts"),  how="outer")
          .join(df_oi.set_index("ts"),  how="outer"))

    df[["long_liq", "short_liq"]] = df[["long_liq", "short_liq"]].astype(float).fillna(0.0)
    df[["ls_long", "ls_short"]]   = df[["ls_long", "ls_short"]].astype(float).fillna(50.0)
    df["oi"] = df["oi"].astype(float)

    total_liq = df["long_liq"] + df["short_liq"]
    df["liq_bias"] = np.where(
        total_liq > 0.001,
        (df["short_liq"] - df["long_liq"]) / total_liq,
        0.0,
    )

    df["liq_score"] = df.apply(
        lambda r: _compute_liq_score(r["liq_bias"], r["ls_long"], r["ls_short"]), axis=1
    )

    df["oi_chg_pct"] = df["oi"].pct_change() * 100.0
    df["oi_chg_pct"] = df["oi_chg_pct"].fillna(0.0)

    return df[["liq_bias", "ls_long", "ls_short", "liq_score", "oi_chg_pct"]]


def backfill_asset(asset: str):
    csv_path = ASSET_CSVS[asset]
    symbol   = ASSET_SYMBOLS[asset]

    df = pd.read_csv(csv_path, low_memory=False)
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")

    needs_fill = (
        df["liq_score"].isna() | (df["liq_score"].astype(str).str.strip() == "")
    )
    n_missing = needs_fill.sum()
    print(f"\n{asset}: {n_missing} rows need liq backfill out of {len(df)}")
    if n_missing == 0:
        print(f"  Already complete — skipping.")
        return

    valid_dt = df.loc[needs_fill, "decision_time"].dropna()
    if valid_dt.empty:
        print(f"  No valid decision_times — skipping.")
        return

    earliest = valid_dt.min()
    from_unix = int(earliest.timestamp()) - 7200   # 2h buffer before first trade
    to_unix   = int(time.time()) + 900

    print(f"  Fetching {symbol} from {earliest.date()} …", end=" ", flush=True)
    try:
        signals = fetch_signals(symbol, from_unix, to_unix)
    except Exception as e:
        print(f"FAILED: {e}")
        return
    print(f"{len(signals)} bars loaded")

    filled = 0
    for idx in df.index[needs_fill]:
        dt = df.at[idx, "decision_time"]
        if pd.isna(dt):
            continue
        # last completed 15-min bar: floor to 15min, back one period
        bar_ts = dt.floor("15min") - pd.Timedelta(minutes=15)
        if bar_ts not in signals.index:
            # try the bar before (fallback for off-by-one at boundary)
            bar_ts = bar_ts - pd.Timedelta(minutes=15)
        if bar_ts not in signals.index:
            continue
        row = signals.loc[bar_ts]
        df.at[idx, "liq_bias"]   = round(float(row["liq_bias"]),   4)
        df.at[idx, "liq_score"]  = int(row["liq_score"])
        df.at[idx, "oi_chg_pct"] = round(float(row["oi_chg_pct"]), 4)
        filled += 1

    df.to_csv(csv_path, index=False)
    print(f"  Filled {filled}/{n_missing} rows → {csv_path.name}")


def main():
    print(SEP)
    print("  Backfilling Coinalyze liq + OI into 15m paper trade CSVs")
    print(SEP)
    for asset in ("BTC", "ETH", "SOL"):
        backfill_asset(asset)
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
