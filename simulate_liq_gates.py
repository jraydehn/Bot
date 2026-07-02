#!/usr/bin/env python3
"""
simulate_liq_gates.py

Joins historical Coinalyze liq_score to the paper trade archive, then
simulates gate proposals derived from the liq signal simulation results.

Gate candidates:
  A. liq_cascade_yes_gate  — block YES when liq_score <= -1
  B. liq_squeeze_no_gate   — block NO  when liq_score >= +1

For each gate:
  1. Flat block stats (n, WR, P&L, breakeven WR)
  2. Exhaustive rescue search across available signal columns
  3. Best rescue rule printed

Assets: BTC + ETH (SOL has no Coinalyze data)

Run: python3 simulate_liq_gates.py
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
SEP2 = "-" * 72

ASSET_SYMBOLS = {"BTC": "BTCUSDT_PERP.A", "ETH": "ETHUSDT_PERP.A"}
ASSET_TICKERS = {"BTC": "KXBTC", "ETH": "KXETH"}


# ── data loading ──────────────────────────────────────────────────────────────

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


def fetch_liq_scores(symbol: str) -> pd.Series:
    """Return liq_score series indexed by 15min UTC timestamp."""
    now_unix = int(time.time())
    far_past = now_unix - 90 * 24 * 3600

    r_liq = requests.get(f"{BASE}/liquidation-history",
        params={"symbols": symbol, "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=15)
    r_ls  = requests.get(f"{BASE}/long-short-ratio-history",
        params={"symbols": symbol, "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=15)
    r_liq.raise_for_status(); r_ls.raise_for_status()

    df_liq = pd.DataFrame(r_liq.json()[0]["history"], columns=["t", "l", "s"])
    df_liq["t"] = pd.to_datetime(df_liq["t"], unit="s", utc=True)
    df_liq = df_liq.set_index("t")

    df_ls = pd.DataFrame(r_ls.json()[0]["history"])
    df_ls["t"] = pd.to_datetime(df_ls["t"], unit="s", utc=True)
    df_ls = df_ls.set_index("t")

    shared = df_liq.index.intersection(df_ls.index)
    scores = {}
    for T in shared:
        ll, sl = float(df_liq.loc[T, "l"]), float(df_liq.loc[T, "s"])
        tot = ll + sl
        bias = (sl - ll) / tot if tot > 0.001 else 0.0
        scores[T] = _compute_liq_score(bias, float(df_ls.loc[T, "l"]), float(df_ls.loc[T, "s"]))
    return pd.Series(scores, name="liq_score")


_CSV_BY_ASSET = {
    "BTC": ["results/paper_trades.csv"],
    "ETH": ["results/paper_trades_eth.csv", "results/paper_trades.csv"],
    "SOL": ["results/paper_trades_sol.csv", "results/paper_trades.csv"],
}

def load_trades(asset: str) -> pd.DataFrame:
    frames = []
    ticker_prefix = ASSET_TICKERS[asset]
    for path in _CSV_BY_ASSET.get(asset, ["results/paper_trades.csv"]):
        try:
            chunk = pd.read_csv(path)
            chunk = chunk[chunk["would_win"].notna()].copy()
            chunk["decision_time"] = pd.to_datetime(chunk["decision_time"], utc=True)
            chunk = chunk[chunk["contract_ticker"].str.startswith(ticker_prefix, na=False)]
            frames.append(chunk)
        except FileNotFoundError:
            pass
    df = pd.concat(frames).drop_duplicates(subset=["contract_ticker","decision_time"]).copy() if frames else pd.DataFrame()
    for col in ["would_pnl", "p_market", "offset_pct", "composite_p_up",
                "tau_minutes", "stoch_k", "ema_stack_bias", "chg_30m",
                "composite_rev", "composite_trend", "vpin_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def join_liq_scores(df: pd.DataFrame, liq_scores: pd.Series) -> pd.DataFrame:
    """For each trade, look up the most recent completed 15min bar's liq_score."""
    floored = df["decision_time"].dt.floor("15min")
    df = df.copy()
    df["liq_score"] = floored.map(liq_scores)
    # Fallback: try previous bar if exact match missing
    prev = floored - pd.Timedelta(minutes=15)
    df["liq_score"] = df["liq_score"].fillna(floored.map(liq_scores))
    df["liq_score"] = df["liq_score"].fillna(prev.map(liq_scores))
    return df[df["liq_score"].notna()].copy()


# ── reporting ─────────────────────────────────────────────────────────────────

def _stats(grp: pd.DataFrame, label: str = ""):
    n   = len(grp)
    wr  = grp["would_win"].mean()
    pnl = grp["would_pnl"].sum()
    be  = grp["p_market"].mean() if "p_market" in grp else float("nan")
    delta = wr - be
    prefix = f"  {label:<38}" if label else "  "
    print(f"{prefix}  n={n:>4}  WR={wr:>5.1%}  BE={be:>5.1%}  Δ={delta:>+5.1%}  P&L=${pnl:>+7.0f}")


def _rescue_search(blocked: pd.DataFrame, all_trades: pd.DataFrame, gate_label: str):
    """Search for rescue conditions that carve out high-WR sub-slices from blocked trades."""
    if len(blocked) == 0:
        return
    baseline_be = blocked["p_market"].mean()

    candidates = []

    # Numeric threshold conditions
    for col, direction, thresholds in [
        ("composite_p_up",  ">=", [0.55, 0.60, 0.65, 0.70]),
        ("composite_p_up",  "<=", [0.45, 0.50, 0.55]),
        ("offset_pct",      "<=", [-5, -3, -1, 0]),
        ("offset_pct",      ">=", [0, 3, 5, 10]),
        ("tau_minutes",     "<=", [10, 15, 20, 30]),
        ("tau_minutes",     ">=", [30, 45, 60]),
        ("stoch_k",         ">=", [50, 60, 70, 80]),
        ("stoch_k",         "<=", [30, 40, 50]),
        ("p_market",        "<=", [0.20, 0.30, 0.40]),
        ("p_market",        ">=", [0.60, 0.70, 0.80]),
        ("chg_30m",         ">=", [0.003, 0.005, 0.008]),
        ("chg_30m",         "<=", [-0.003, -0.005, -0.008]),
        ("composite_rev",   ">=", [3, 5, 7]),
        ("composite_rev",   "<=", [-3, -5]),
        ("composite_trend", ">=", [1, 2, 3]),
        ("composite_trend", "<=", [-1, -2, -3]),
        ("ema_stack_bias",  "==", [1, 0, -1]),
        ("vpin_score",      "==", [0, 1]),
    ]:
        if col not in blocked.columns:
            continue
        for thresh in thresholds:
            if direction == ">=":
                mask = blocked[col] >= thresh
            elif direction == "<=":
                mask = blocked[col] <= thresh
            else:
                mask = blocked[col] == thresh
            sub = blocked[mask]
            if len(sub) < 5:
                continue
            wr = sub["would_win"].mean()
            be = sub["p_market"].mean()
            pnl = sub["would_pnl"].sum()
            if wr > be and wr > baseline_be:
                candidates.append({
                    "condition": f"{col} {direction} {thresh}",
                    "n": len(sub),
                    "wr": wr,
                    "be": be,
                    "delta": wr - be,
                    "pnl": pnl,
                })

    if not candidates:
        print(f"  No rescue found above breakeven for {gate_label}.")
        return

    best = sorted(candidates, key=lambda x: (x["delta"], x["n"]), reverse=True)[:5]
    print(f"  Top rescue candidates (WR > BE, sorted by Δ):")
    print(f"    {'Condition':<38}  {'n':>4}  {'WR':>6}  {'BE':>6}  {'Δ':>6}  {'P&L':>8}")
    print(f"    {'-'*76}")
    for c in best:
        print(f"    {c['condition']:<38}  {c['n']:>4}  {c['wr']:>5.1%}  "
              f"{c['be']:>5.1%}  {c['delta']:>+5.1%}  ${c['pnl']:>+7.0f}")


# ── gate simulation ───────────────────────────────────────────────────────────

def simulate_gates(asset: str, df: pd.DataFrame):
    print(f"\n{'█'*72}")
    print(f"  {asset} — Liq Gate Simulation  (n={len(df)} resolved trades)")
    print(f"{'█'*72}")

    # Overall baseline
    for side in ["yes", "no"]:
        sub = df[df["side"] == side]
        if len(sub) > 0:
            _stats(sub, f"ALL {side.upper()} baseline")

    # Score distribution among paper trades
    print(f"\n  liq_score distribution in paper trade window:")
    vc = df["liq_score"].value_counts().sort_index()
    for sc, cnt in vc.items():
        lbl = {-2: "CASCADE--", -1: "cascade- ", 0: "neutral  ", 1: "squeeze+ ", 2: "SQUEEZE++"}
        sc_i = int(sc)
        print(f"    {sc_i:+d}  {lbl.get(sc_i,'?')}: {cnt:>4} trades  ({cnt/len(df):.1%})")

    # ── Gate A: block YES when liq_score <= -1 ────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Gate A: block YES when liq_score <= -1  (cascade active)")
    print(SEP2)

    yes_trades = df[df["side"] == "yes"]
    blocked_a  = yes_trades[yes_trades["liq_score"] <= -1]
    passed_a   = yes_trades[yes_trades["liq_score"] > -1]

    _stats(yes_trades, "ALL YES (baseline)")
    _stats(blocked_a,  "BLOCKED YES (liq<=-1)")
    _stats(passed_a,   "PASSED YES  (liq>-1)")

    net_pnl_a = -blocked_a["would_pnl"].sum()
    print(f"\n  Flat block net P&L impact: ${net_pnl_a:>+,.0f}  "
          f"(+ = saved losses, - = missed wins)")

    print(f"\n  liq_score breakdown within blocked YES:")
    for sc in sorted(blocked_a["liq_score"].unique()):
        _stats(blocked_a[blocked_a["liq_score"] == sc], f"  liq_score={int(sc):+d}")

    print(f"\n  Rescue search (sub-slices of blocked YES above breakeven):")
    _rescue_search(blocked_a, df, "Gate A")

    # ── Gate B: block NO when liq_score >= +1 ────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Gate B: block NO when liq_score >= +1  (squeeze active)")
    print(SEP2)

    no_trades  = df[df["side"] == "no"]
    blocked_b  = no_trades[no_trades["liq_score"] >= 1]
    passed_b   = no_trades[no_trades["liq_score"] < 1]

    _stats(no_trades,  "ALL NO (baseline)")
    _stats(blocked_b,  "BLOCKED NO (liq>=+1)")
    _stats(passed_b,   "PASSED NO  (liq<+1)")

    net_pnl_b = -blocked_b["would_pnl"].sum()
    print(f"\n  Flat block net P&L impact: ${net_pnl_b:>+,.0f}  "
          f"(+ = saved losses, - = missed wins)")

    print(f"\n  liq_score breakdown within blocked NO:")
    for sc in sorted(blocked_b["liq_score"].unique()):
        _stats(blocked_b[blocked_b["liq_score"] == sc], f"  liq_score={int(sc):+d}")

    print(f"\n  Rescue search (sub-slices of blocked NO above breakeven):")
    # For NO bets, 'p_market' is the YES market price, so BE for NO ≈ 1 - p_market
    blocked_b_no = blocked_b.copy()
    blocked_b_no["p_market"] = 1.0 - blocked_b_no["p_market"]  # flip for NO breakeven
    _rescue_search(blocked_b_no, df, "Gate B")

    # ── Combined impact ───────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Combined Gate A + B impact  ({asset})")
    print(SEP2)
    print(f"  Gate A (block YES cascade):  ${net_pnl_a:>+,.0f}")
    print(f"  Gate B (block NO squeeze):   ${net_pnl_b:>+,.0f}")
    print(f"  Combined:                    ${net_pnl_a + net_pnl_b:>+,.0f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Liq Gate Simulation — joining Coinalyze signal to paper trades")
    print(SEP)

    for asset in ("BTC", "ETH"):
        symbol = ASSET_SYMBOLS[asset]
        print(f"\nFetching {asset} Coinalyze liq scores...", end=" ", flush=True)
        liq_scores = fetch_liq_scores(symbol)
        print(f"{len(liq_scores)} bars  ({liq_scores.index[0].date()} → {liq_scores.index[-1].date()})")

        print(f"Loading {asset} paper trades...", end=" ", flush=True)
        df = load_trades(asset)
        print(f"{len(df)} resolved trades")

        df = join_liq_scores(df, liq_scores)
        print(f"Matched to liq signal: {len(df)} trades")

        simulate_gates(asset, df)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
