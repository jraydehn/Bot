#!/usr/bin/env python3
"""
simulate_4h_coinglass.py

Tests CoinGlass signals at 4h resolution (Hobbyist supports 4h, 180 days history).
Signals are joined to paper trades via merge_asof on bar-completion time.

Signals:
  1. spot_taker_ratio_4h   — aggregated buy/sell ratio (4h bar)
  2. oi_stable_chg_4h      — stablecoin-margin OI % change per 4h bar
  3. oi_agg_chg_4h         — aggregated OI % change per 4h bar

Gate ideas:
  - Block YES when oi_stable_chg_4h > threshold (new longs piling in → crowding risk)
  - Block NO  when oi_stable_chg_4h > threshold (same reason — directional crowding)
  - Block YES when taker_ratio_4h < 1.00 (sellers dominant at entry)
"""

import datetime
import sys
import time

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
ASSETS = {
    "BTC": "results/paper_trades.csv",
    "ETH": "results/paper_trades_eth.csv",
    "SOL": "results/paper_trades_sol.csv",
}

API_KEY = "8f0a30c29a5e424ba2641f649051786b"
HEADERS = {"CG-API-KEY": API_KEY}
BASE    = "https://open-api-v4.coinglass.com"
SEP     = "=" * 72
SPOT_EXCHANGES = "Binance,OKX,Bybit"


# ---------------------------------------------------------------------------
# Data fetchers (4h interval)
# ---------------------------------------------------------------------------

def fetch_taker_4h(asset: str) -> pd.DataFrame:
    r = requests.get(f"{BASE}/api/spot/aggregated-taker-buy-sell-volume/history",
                     headers=HEADERS,
                     params={"symbol": asset.upper(), "interval": "4h", "limit": 1000,
                             "exchange_list": SPOT_EXCHANGES},
                     timeout=10)
    body = r.json()
    if body.get("code") != "0":
        print(f"  [taker_4h] {asset} error: {body.get('msg')}")
        return pd.DataFrame()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        bar_start = datetime.datetime.fromtimestamp(row["time"] / 1000, tz=datetime.timezone.utc)
        bar_end   = bar_start + datetime.timedelta(hours=4)
        buy  = float(row.get("aggregated_buy_volume_usd",  0))
        sell = float(row.get("aggregated_sell_volume_usd", 0))
        ratio = buy / sell if sell > 0 else 1.0
        records.append({"bar_end": bar_end, "taker_ratio_4h": ratio})
    df = pd.DataFrame(records).set_index("bar_end").sort_index()
    return df


def fetch_stablecoin_oi_4h(asset: str) -> pd.DataFrame:
    r = requests.get(f"{BASE}/api/futures/open-interest/aggregated-stablecoin-history",
                     headers=HEADERS,
                     params={"symbol": asset.upper(), "interval": "4h", "limit": 1000,
                             "exchange_list": SPOT_EXCHANGES},
                     timeout=10)
    body = r.json()
    if body.get("code") != "0":
        print(f"  [oi_stable_4h] {asset} error: {body.get('msg')}")
        return pd.DataFrame()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        bar_start = datetime.datetime.fromtimestamp(row["time"] / 1000, tz=datetime.timezone.utc)
        bar_end   = bar_start + datetime.timedelta(hours=4)
        o = float(row.get("open",  0) or 0)
        c = float(row.get("close", 0) or 0)
        pct = (c - o) / o * 100 if o != 0 else 0.0
        records.append({"bar_end": bar_end, "oi_stable_chg_4h": pct})
    df = pd.DataFrame(records).set_index("bar_end").sort_index()
    return df


def fetch_aggregated_oi_4h(asset: str) -> pd.DataFrame:
    r = requests.get(f"{BASE}/api/futures/open-interest/ohlc-history",
                     headers=HEADERS,
                     params={"symbol": asset.upper(), "interval": "4h", "limit": 1000,
                             "exchange_list": SPOT_EXCHANGES},
                     timeout=10)
    body = r.json()
    if body.get("code") != "0":
        print(f"  [oi_agg_4h] {asset} error: {body.get('msg')}")
        return pd.DataFrame()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        bar_start = datetime.datetime.fromtimestamp(row["time"] / 1000, tz=datetime.timezone.utc)
        bar_end   = bar_start + datetime.timedelta(hours=4)
        o = float(row.get("open",  0) or 0)
        c = float(row.get("close", 0) or 0)
        pct = (c - o) / o * 100 if o != 0 else 0.0
        records.append({"bar_end": bar_end, "oi_agg_chg_4h": pct})
    df = pd.DataFrame(records).set_index("bar_end").sort_index()
    return df


# ---------------------------------------------------------------------------
# Trade loader
# ---------------------------------------------------------------------------

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["would_pnl"].notna()].copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["logged_at"])
    df["would_win"] = df["would_win"].astype(bool)
    df["would_pnl"] = df["would_pnl"].astype(float)
    df["side"] = df["side"].str.strip().str.lower()
    return df


def join_4h(trades: pd.DataFrame, signal_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Merge 4h bar signal to trades using asof join (bar must be closed before trade)."""
    if signal_df.empty:
        trades[col] = float("nan")
        return trades
    sig = signal_df[[col]].reset_index()
    sig = sig.rename(columns={"bar_end": "ts"})
    tr = trades.copy()
    tr = tr.sort_values("logged_at")
    merged = pd.merge_asof(tr, sig, left_on="logged_at", right_on="ts",
                           direction="backward")
    return merged


# ---------------------------------------------------------------------------
# Gate simulator
# ---------------------------------------------------------------------------

def simulate_gate(df: pd.DataFrame, mask_block: pd.Series, gate_name: str) -> dict:
    actual = df[df["decision"] != "no_trade"].copy()
    aligned = mask_block.reindex(actual.index, fill_value=False).astype(bool)
    blocked_idx = actual.index[aligned]
    blocked = actual.loc[blocked_idx]
    kept    = actual.drop(blocked_idx)

    base_pnl  = actual["would_pnl"].sum()
    kept_pnl  = kept["would_pnl"].sum()
    pnl_delta = kept_pnl - base_pnl
    base_wr   = actual["would_win"].mean() * 100
    kept_wr   = kept["would_win"].mean() * 100 if len(kept) > 0 else 0.0
    wins_blocked   = int(blocked["would_win"].sum())
    losses_blocked = int((~blocked["would_win"]).sum())
    n_blocked = len(blocked)

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
    print(f"  {r['gate']:<50}  blocked={r['n_blocked']:>4}"
          f"  (W={r['wins_blocked']} L={r['losses_blocked']})"
          f"  pnl {sign}${r['pnl_delta']:.2f} {arrow}"
          f"  wr {r['base_wr']:.1f}%→{r['new_wr']:.1f}%")


def rescue_analysis(df: pd.DataFrame, gate_mask: pd.Series,
                    rescue_col: str, rescue_thresh: float, rescue_dir: str,
                    gate_label: str):
    """Show what's inside the block, split by rescue condition."""
    actual = df[df["decision"] != "no_trade"].copy()
    aligned = gate_mask.reindex(actual.index, fill_value=False).astype(bool)
    blocked = actual[aligned].copy()
    if len(blocked) == 0:
        return
    col_vals = blocked[rescue_col].dropna()
    if col_vals.empty:
        print(f"    [rescue] no {rescue_col} data in blocked trades")
        return

    if rescue_dir == "gt":
        rescue_mask = blocked[rescue_col] > rescue_thresh
    else:
        rescue_mask = blocked[rescue_col] < rescue_thresh

    rescued = blocked[rescue_mask]
    still_blocked = blocked[~rescue_mask]

    print(f"    Rescue analysis for '{gate_label}':")
    for label, sub in [("  ALL blocked", blocked),
                       (f"  Rescue ({rescue_col}{'>''<'[rescue_dir=='lt']}{rescue_thresh})", rescued),
                       ("  Still blocked", still_blocked)]:
        if len(sub) == 0:
            print(f"      {label}: n=0")
            continue
        wr  = sub["would_win"].mean() * 100
        pnl = sub["would_pnl"].sum()
        print(f"      {label}: n={len(sub):>4}  wr={wr:.1f}%  pnl=${pnl:.2f}")


# ---------------------------------------------------------------------------
# Main per-asset analysis
# ---------------------------------------------------------------------------

def run_asset(asset: str, csv_path: str,
              taker_df: pd.DataFrame,
              oi_stable_df: pd.DataFrame,
              oi_agg_df: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  {asset} — 4h CoinGlass Signal Simulation")
    print(SEP)

    df = load_trades(csv_path)
    if df.empty:
        print("  No resolved trades.")
        return

    df = join_4h(df, taker_df,     "taker_ratio_4h")
    df = join_4h(df, oi_stable_df, "oi_stable_chg_4h")
    df = join_4h(df, oi_agg_df,    "oi_agg_chg_4h")

    actual = df[df["decision"] != "no_trade"]
    print(f"  Resolved bets: {len(actual)}  PnL: ${actual['would_pnl'].sum():.2f}"
          f"  WR: {actual['would_win'].mean()*100:.1f}%")

    taker_cov   = actual["taker_ratio_4h"].notna().mean()   * 100
    stable_cov  = actual["oi_stable_chg_4h"].notna().mean() * 100
    agg_cov     = actual["oi_agg_chg_4h"].notna().mean()    * 100
    print(f"  Signal coverage:  taker={taker_cov:.0f}%  oi_stable={stable_cov:.0f}%  oi_agg={agg_cov:.0f}%")

    # --- Spot Taker 4h ---
    print()
    print("  Spot Taker 4h (block YES when sellers dominant):")
    for thresh in [0.90, 0.95, 1.00]:
        mask = (df["taker_ratio_4h"] < thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, mask, f"  Block YES taker<{thresh:.2f}")
        print_result(r)

    print()
    print("  Spot Taker 4h (block NO when buyers dominant):")
    for thresh in [1.05, 1.10]:
        mask = (df["taker_ratio_4h"] > thresh) & (df["side"] == "no")
        r = simulate_gate(actual, mask, f"  Block NO  taker>{thresh:.2f}")
        print_result(r)

    # --- Stablecoin OI 4h ---
    print()
    print("  Stablecoin OI 4h % change (block YES when OI surging = longs crowding):")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        mask = (df["oi_stable_chg_4h"] > thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, mask, f"  Block YES oi_stable_chg>{thresh:.1f}%")
        print_result(r)

    print()
    print("  Stablecoin OI 4h % change (block NO when OI surging = longs crowding → NO unlikely):")
    for thresh in [1.0, 2.0, 3.0]:
        mask = (df["oi_stable_chg_4h"] > thresh) & (df["side"] == "no")
        r = simulate_gate(actual, mask, f"  Block NO  oi_stable_chg>{thresh:.1f}%")
        print_result(r)

    print()
    print("  Stablecoin OI 4h % change (block YES when OI dropping = longs exiting):")
    for thresh in [-1.0, -2.0, -3.0]:
        mask = (df["oi_stable_chg_4h"] < thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, mask, f"  Block YES oi_stable_chg<{thresh:.1f}%")
        print_result(r)

    # --- Aggregated OI 4h ---
    print()
    print("  Aggregated OI 4h % change (block YES when surging):")
    for thresh in [1.0, 2.0, 3.0]:
        mask = (df["oi_agg_chg_4h"] > thresh) & (df["side"] == "yes")
        r = simulate_gate(actual, mask, f"  Block YES oi_agg_chg>{thresh:.1f}%")
        print_result(r)

    print()
    print("  Aggregated OI 4h % change (block NO when surging):")
    for thresh in [1.0, 2.0, 3.0]:
        mask = (df["oi_agg_chg_4h"] > thresh) & (df["side"] == "no")
        r = simulate_gate(actual, mask, f"  Block NO  oi_agg_chg>{thresh:.1f}%")
        print_result(r)

    # --- Signal vs outcome distribution ---
    print()
    print("  Signal distribution (YES bets, wins vs losses):")
    yes_trades = actual[actual["side"] == "yes"].copy()
    if len(yes_trades) > 0:
        yes_win  = yes_trades[yes_trades["would_win"]]
        yes_loss = yes_trades[~yes_trades["would_win"]]
        for col in ["taker_ratio_4h", "oi_stable_chg_4h", "oi_agg_chg_4h"]:
            w_mean = yes_win[col].mean()  if yes_win[col].notna().any()  else float("nan")
            l_mean = yes_loss[col].mean() if yes_loss[col].notna().any() else float("nan")
            print(f"    {col:<25}  wins={w_mean:+.4f}  losses={l_mean:+.4f}")

    # --- Rescue analysis for top candidates ---
    print()
    print("  Rescue analysis (pm threshold on top YES gates):")
    best_yes_gates = []
    if stable_cov > 20:
        best_yes_gates.append((
            (df["oi_stable_chg_4h"] > 2.0) & (df["side"] == "yes"),
            "Block YES oi_stable_chg>2%",
            "p_market_implied",
        ))
    if agg_cov > 20:
        best_yes_gates.append((
            (df["oi_agg_chg_4h"] > 2.0) & (df["side"] == "yes"),
            "Block YES oi_agg_chg>2%",
            "p_market_implied",
        ))
    if taker_cov > 20:
        best_yes_gates.append((
            (df["taker_ratio_4h"] < 1.00) & (df["side"] == "yes"),
            "Block YES taker<1.00",
            "p_market_implied",
        ))

    for gate_mask, gate_label, _ in best_yes_gates:
        aligned = gate_mask.reindex(actual.index, fill_value=False).astype(bool)
        blocked = actual[aligned]
        if len(blocked) == 0:
            continue
        print(f"  --- {gate_label} (n={len(blocked)}) ---")
        if "p_market_implied" in blocked.columns:
            for pm_thresh in [0.55, 0.60, 0.65]:
                rescued = blocked[blocked["p_market_implied"] > pm_thresh]
                still   = blocked[blocked["p_market_implied"] <= pm_thresh]
                for label, sub in [(f"    rescued (pm>{pm_thresh})", rescued),
                                   (f"    blocked  (pm≤{pm_thresh})", still)]:
                    if len(sub) == 0:
                        print(f"      {label}: n=0")
                        continue
                    wr  = sub["would_win"].mean() * 100
                    pnl = sub["would_pnl"].sum()
                    print(f"    {label}: n={len(sub):>4}  wr={wr:.1f}%  pnl=${pnl:.2f}")
        else:
            wr  = blocked["would_win"].mean() * 100
            pnl = blocked["would_pnl"].sum()
            print(f"    All blocked: n={len(blocked):>4}  wr={wr:.1f}%  pnl=${pnl:.2f}")
            print(f"    (p_market_implied column not in CSV — add to trades for rescue analysis)")


# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  4h CoinGlass Signal Simulation — Fetching data...")
    print(SEP)

    print("  [1/3] Spot taker buy/sell (4h)...")
    taker = {}
    for asset in ["BTC", "ETH", "SOL"]:
        taker[asset] = fetch_taker_4h(asset)
        n = len(taker[asset])
        rng = f"{taker[asset].index[0].date()} → {taker[asset].index[-1].date()}" if n > 0 else "empty"
        print(f"    {asset}: {n} bars  {rng}")
        time.sleep(0.4)

    print("  [2/3] Stablecoin OI (4h)...")
    oi_stable = {}
    for asset in ["BTC", "ETH", "SOL"]:
        oi_stable[asset] = fetch_stablecoin_oi_4h(asset)
        n = len(oi_stable[asset])
        rng = f"{oi_stable[asset].index[0].date()} → {oi_stable[asset].index[-1].date()}" if n > 0 else "empty"
        print(f"    {asset}: {n} bars  {rng}")
        time.sleep(0.4)

    print("  [3/3] Aggregated OI (4h)...")
    oi_agg = {}
    for asset in ["BTC", "ETH", "SOL"]:
        oi_agg[asset] = fetch_aggregated_oi_4h(asset)
        n = len(oi_agg[asset])
        rng = f"{oi_agg[asset].index[0].date()} → {oi_agg[asset].index[-1].date()}" if n > 0 else "empty"
        print(f"    {asset}: {n} bars  {rng}")
        time.sleep(0.4)

    for asset, csv_path in ASSETS.items():
        run_asset(asset, csv_path,
                  taker.get(asset, pd.DataFrame()),
                  oi_stable.get(asset, pd.DataFrame()),
                  oi_agg.get(asset, pd.DataFrame()))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
