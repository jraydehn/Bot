#!/usr/bin/env python3
"""
simulate_coinglass_gates.py

Tests CoinGlass Hobbyist signals as gates on the paper trade log.
Signals are daily-resolution and joined to trades by UTC date.

Signals tested:
  1. exchange_flow_pct  — net BTC/ETH flow to exchanges as % of total balance
                          positive = inflow (selling pressure), negative = outflow (accumulation)
  2. spot_taker_ratio   — aggregated buy/sell volume ratio across Binance/OKX/Bybit
                          >1 = buyers dominant, <1 = sellers dominant
  3. fear_greed         — 0–100 daily sentiment index

Gate scenarios:
  A. Block YES when exchange_flow_pct > threshold  (inflow = bearish for YES)
  B. Block YES when spot_taker_ratio < threshold   (sellers dominant)
  C. Block YES when fear_greed > threshold         (extreme greed = mean reversion risk)
  D. Block NO  when exchange_flow_pct < -threshold (strong outflow = bullish, hurts NO)
  E. Combinations of A+B

Reports per scenario: trades_blocked, wins_blocked, losses_blocked, net_pnl_delta, net_wr_delta
"""

import time
import datetime
import sys
import os

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASSETS = {
    "BTC": "results/paper_trades.csv",
    "ETH": "results/paper_trades_eth.csv",
    "SOL": "results/paper_trades_sol.csv",
}

API_KEY  = os.environ.get("COINGLASS_API_KEY", "8f0a30c29a5e424ba2641f649051786b")
HEADERS  = {"CG-API-KEY": API_KEY}
BASE_V4  = "https://open-api-v4.coinglass.com"
BASE_V3  = "https://open-api-v3.coinglass.com"
BANKROLL = 1000.0
SEP      = "=" * 72


# ---------------------------------------------------------------------------
# Fetch historical CoinGlass signals
# ---------------------------------------------------------------------------

def fetch_exchange_flow_history(asset: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC date with columns:
      total_balance, flow_1d, flow_pct_1d
    Computed from the exchange balance chart (daily balance per exchange,
    summed across all exchanges, then diffed to get daily net change).
    """
    r = requests.get(f"{BASE_V4}/api/exchange/balance/chart",
                     headers=HEADERS,
                     params={"symbol": asset.upper(), "interval": "1d", "limit": 700},
                     timeout=10)
    body = r.json()
    if body.get("code") != "0":
        print(f"  [exchange_flow] {asset} error: {body.get('msg')}")
        return pd.DataFrame()

    d = body["data"]
    time_list = d.get("time_list") or d.get("timeList") or []
    data_map  = d.get("data_map")  or d.get("dataMap")  or {}

    if not time_list or not data_map:
        return pd.DataFrame()

    dates = [datetime.datetime.fromtimestamp(t / 1000, tz=datetime.timezone.utc).date()
             for t in time_list]

    # Sum balance across all exchanges at each timestamp
    total = np.zeros(len(time_list))
    for vals in data_map.values():
        arr = np.array([float(v) if v is not None else 0.0 for v in vals])
        if len(arr) == len(total):
            total += arr

    df = pd.DataFrame({"date": dates, "total_balance": total})
    df = df.set_index("date").sort_index()
    df["flow_1d"]     = df["total_balance"].diff()          # positive = inflow (bearish)
    df["flow_pct_1d"] = df["flow_1d"] / df["total_balance"].shift(1) * 100
    return df


def fetch_spot_taker_history(asset: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC date with column: taker_ratio (buy/sell).
    """
    r = requests.get(f"{BASE_V4}/api/spot/aggregated-taker-buy-sell-volume/history",
                     headers=HEADERS,
                     params={"symbol": asset.upper(), "interval": "1d", "limit": 500,
                             "exchange_list": "Binance,OKX,Bybit"},
                     timeout=10)
    body = r.json()
    if body.get("code") != "0":
        print(f"  [spot_taker] {asset} error: {body.get('msg')}")
        return pd.DataFrame()

    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        dt   = datetime.datetime.fromtimestamp(row["time"] / 1000, tz=datetime.timezone.utc).date()
        buy  = float(row.get("aggregated_buy_volume_usd",  0))
        sell = float(row.get("aggregated_sell_volume_usd", 0))
        ratio = buy / sell if sell > 0 else 1.0
        records.append({"date": dt, "taker_ratio": ratio})

    df = pd.DataFrame(records).set_index("date").sort_index()
    return df


def fetch_fear_greed_history() -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC date with column: fear_greed.
    """
    r = requests.get(f"{BASE_V3}/api/index/fear-greed-history",
                     headers=HEADERS, timeout=10)
    body = r.json()
    if body.get("code") != "0":
        return pd.DataFrame()

    values = body["data"].get("values", [])
    dates  = [datetime.datetime.fromtimestamp(t / 1000, tz=datetime.timezone.utc).date()
              for t in body["data"].get("dates", [])]

    if not values or not dates or len(values) != len(dates):
        return pd.DataFrame()

    df = pd.DataFrame({"date": dates, "fear_greed": [float(v) for v in values]})
    return df.set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Load and prepare trade log
# ---------------------------------------------------------------------------

def load_trades(path: str, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["would_pnl"].notna()].copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["logged_at"])
    df["date"] = df["logged_at"].dt.date
    df["would_win"] = df["would_win"].astype(bool)
    df["would_pnl"] = df["would_pnl"].astype(float)
    df["side"]  = df["side"].str.strip().str.lower()
    df["asset"] = asset
    return df


# ---------------------------------------------------------------------------
# Gate simulation helper
# ---------------------------------------------------------------------------

def simulate_gate(df: pd.DataFrame, mask_block: pd.Series, gate_name: str) -> dict:
    """
    Given a boolean mask of rows to block, compute impact vs baseline.
    Only blocks trades where decision != 'no_trade' (i.e., actual bets placed).
    """
    actual = df[df["decision"] != "no_trade"].copy()
    aligned = mask_block.reindex(actual.index, fill_value=False).astype(bool)
    blocked_idx = actual.index[aligned]

    blocked    = actual.loc[blocked_idx]
    kept       = actual.drop(blocked_idx)

    base_pnl   = actual["would_pnl"].sum()
    kept_pnl   = kept["would_pnl"].sum()
    pnl_delta  = kept_pnl - base_pnl

    base_wr    = actual["would_win"].mean() * 100
    kept_wr    = kept["would_win"].mean() * 100 if len(kept) > 0 else 0.0

    wins_blocked   = int(blocked["would_win"].sum())
    losses_blocked = int((~blocked["would_win"]).sum())
    n_blocked      = len(blocked)

    return {
        "gate":           gate_name,
        "n_blocked":      n_blocked,
        "wins_blocked":   wins_blocked,
        "losses_blocked": losses_blocked,
        "pnl_delta":      pnl_delta,
        "base_pnl":       base_pnl,
        "new_pnl":        kept_pnl,
        "base_wr":        base_wr,
        "new_wr":         kept_wr,
    }


def print_result(r: dict):
    sign  = "+" if r["pnl_delta"] >= 0 else ""
    arrow = "▲" if r["pnl_delta"] >= 0 else "▼"
    print(f"  {r['gate']:<45}  blocked={r['n_blocked']:>4}"
          f"  (W={r['wins_blocked']} L={r['losses_blocked']})"
          f"  pnl {sign}${r['pnl_delta']:.2f} {arrow}"
          f"  wr {r['base_wr']:.1f}%→{r['new_wr']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_asset(asset: str, csv_path: str,
              flow_df: pd.DataFrame, taker_df: pd.DataFrame, fg_df: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  {asset} — CoinGlass Gate Simulation")
    print(SEP)

    df = load_trades(csv_path, asset)
    if df.empty:
        print("  No resolved trades.")
        return

    # Join signals by date (guard against empty DataFrames)
    if not flow_df.empty and "flow_pct_1d" in flow_df.columns:
        df = df.join(flow_df[["flow_pct_1d"]], on="date", how="left")
    else:
        df["flow_pct_1d"] = float("nan")
    if not taker_df.empty and "taker_ratio" in taker_df.columns:
        df = df.join(taker_df[["taker_ratio"]], on="date", how="left")
    else:
        df["taker_ratio"] = float("nan")
    if not fg_df.empty and "fear_greed" in fg_df.columns:
        df = df.join(fg_df[["fear_greed"]], on="date", how="left")
    else:
        df["fear_greed"] = float("nan")

    actual = df[df["decision"] != "no_trade"]
    print(f"  Resolved bets: {len(actual)}  PnL: ${actual['would_pnl'].sum():.2f}"
          f"  WR: {actual['would_win'].mean()*100:.1f}%")

    # Coverage check
    flow_cov  = actual["flow_pct_1d"].notna().mean() * 100
    taker_cov = actual["taker_ratio"].notna().mean() * 100
    fg_cov    = actual["fear_greed"].notna().mean() * 100
    print(f"  Signal coverage:  flow={flow_cov:.0f}%  taker={taker_cov:.0f}%  F&G={fg_cov:.0f}%")

    if flow_cov < 20 and taker_cov < 20:
        print("  Insufficient signal coverage for simulation.")
        return

    yes_mask = actual["side"] == "yes"
    no_mask  = actual["side"] == "no"

    print()
    print("  Exchange Flow Gates (block YES when BTC flowing INTO exchanges):")
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        gate_mask = (df["flow_pct_1d"] > thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, gate_mask, f"  Block YES flow>{thresh:.2f}%")
        print_result(r)

    print()
    print("  Exchange Flow Gates (block NO when BTC flowing OUT of exchanges):")
    for thresh in [0.05, 0.10, 0.15]:
        gate_mask = (df["flow_pct_1d"] < -thresh) & (df["side"] == "no")
        r = simulate_gate(actual, gate_mask, f"  Block NO  flow<-{thresh:.2f}%")
        print_result(r)

    print()
    print("  Spot Taker Ratio Gates (block YES when sellers dominant):")
    for thresh in [0.90, 0.95, 1.00]:
        gate_mask = (df["taker_ratio"] < thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, gate_mask, f"  Block YES taker<{thresh:.2f}")
        print_result(r)

    print()
    print("  Fear & Greed Gates:")
    for (side, comparison, thresh, lbl) in [
        ("yes", "gt", 65, "Block YES F&G>65 (greed)"),
        ("yes", "gt", 75, "Block YES F&G>75 (extreme greed)"),
        ("yes", "lt", 25, "Block YES F&G<25 (extreme fear — contrarian check)"),
        ("no",  "lt", 25, "Block NO  F&G<25 (extreme fear)"),
        ("no",  "lt", 35, "Block NO  F&G<35 (fear)"),
    ]:
        if comparison == "gt":
            gate_mask = (df["fear_greed"] > thresh) & (df["side"] == side)
        else:
            gate_mask = (df["fear_greed"] < thresh) & (df["side"] == side)
        r = simulate_gate(actual, gate_mask, f"  {lbl}")
        print_result(r)

    print()
    print("  Combination Gates:")
    combos = [
        ("Block YES flow>0.10% AND taker<1.00",
         (df["flow_pct_1d"] > 0.10) & (df["taker_ratio"] < 1.00) & (df["side"] == "yes")),
        ("Block YES flow>0.05% AND taker<0.95",
         (df["flow_pct_1d"] > 0.05) & (df["taker_ratio"] < 0.95) & (df["side"] == "yes")),
        ("Block YES flow>0.10% OR  taker<0.90",
         ((df["flow_pct_1d"] > 0.10) | (df["taker_ratio"] < 0.90)) & (df["side"] == "yes")),
        ("Block YES F&G>65 AND flow>0.05%",
         (df["fear_greed"] > 65) & (df["flow_pct_1d"] > 0.05) & (df["side"] == "yes")),
    ]
    for lbl, gate_mask in combos:
        r = simulate_gate(actual, gate_mask, f"  {lbl}")
        print_result(r)

    # Distribution analysis: how do signals correlate with outcomes?
    print()
    print("  Signal vs Outcome (YES bets only, mean values):")
    yes_trades = actual[actual["side"] == "yes"].copy()
    if len(yes_trades) > 0 and yes_trades["flow_pct_1d"].notna().any():
        yes_win  = yes_trades[yes_trades["would_win"] == True]
        yes_loss = yes_trades[yes_trades["would_win"] == False]
        for col, fmt in [("flow_pct_1d", ".4f"), ("taker_ratio", ".4f"), ("fear_greed", ".1f")]:
            w_mean = yes_win[col].mean()  if yes_win[col].notna().any()  else float("nan")
            l_mean = yes_loss[col].mean() if yes_loss[col].notna().any() else float("nan")
            print(f"    {col:<20}  wins={w_mean:{fmt}}  losses={l_mean:{fmt}}")


def main():
    print(SEP)
    print("  CoinGlass Gate Simulation — Fetching historical signals...")
    print(SEP)

    print("  [1/3] Exchange balance flow history (BTC/ETH/SOL)...")
    flow = {}
    for asset in ["BTC", "ETH", "SOL"]:
        flow[asset] = fetch_exchange_flow_history(asset)
        print(f"    {asset}: {len(flow[asset])} days  ", end="")
        if not flow[asset].empty:
            print(f"range {flow[asset].index[0]} → {flow[asset].index[-1]}")
        else:
            print("(empty)")
        time.sleep(0.5)

    print("  [2/3] Spot taker buy/sell history (BTC/ETH/SOL)...")
    taker = {}
    for asset in ["BTC", "ETH", "SOL"]:
        taker[asset] = fetch_spot_taker_history(asset)
        print(f"    {asset}: {len(taker[asset])} days")
        time.sleep(0.5)

    print("  [3/3] Fear & Greed history...")
    fg = fetch_fear_greed_history()
    print(f"    {len(fg)} days  range {fg.index[0] if not fg.empty else '?'} → {fg.index[-1] if not fg.empty else '?'}")

    for asset, csv_path in ASSETS.items():
        run_asset(asset, csv_path, flow.get(asset, pd.DataFrame()),
                  taker.get(asset, pd.DataFrame()), fg)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
