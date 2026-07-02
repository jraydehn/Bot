#!/usr/bin/env python3
"""
backtest_zdrift.py — Compare pure lognormal vs 6h rolling z-drift for BTC hourly.

Pure lognormal:  p_yes = 1 - Φ(z_strike)
With z-drift:    p_yes = 1 - Φ(z_strike - z_drift)
                 z_drift = mu_6h * (tau_min / 60) / sigma_tau
                 mu_6h   = rolling 6-bar mean of 1h log returns at scan time

Flat $10/trade, edge threshold 0.04, both YES and NO sides.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR = Path("results")
EDGE_THRESH = 0.04
BET_SIZE    = 10.0
SEP  = "=" * 72
SEP2 = "-" * 72


# ── Fetch Binance 1h closes ───────────────────────────────────────────────────

def fetch_btc_1h(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    """Fetch BTC/USDT 1h closes from Binance for the given range."""
    url = "https://api.binance.us/api/v3/klines"
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms   = int(end_ts.timestamp()   * 1000)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(url, params={
            "symbol": "BTCUSDT", "interval": "1h",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
    if not rows:
        raise RuntimeError("No Binance 1h data returned")
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "qvol", "n", "tbbase", "tbquote", "ignore",
    ])
    df["ts"]    = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df.set_index("ts")["close"].sort_index()


# ── Core model ────────────────────────────────────────────────────────────────

def p_lognormal(spot, strike, vol_eff, tau_min):
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return 0.5
    z = math.log(strike / spot) / sigma_tau
    return float(np.clip(1 - norm.cdf(z), 0.01, 0.99))


def p_zdrift(spot, strike, vol_eff, tau_min, mu_6h):
    sigma_tau = vol_eff * math.sqrt(tau_min)
    if sigma_tau <= 0:
        return 0.5
    z_strike  = math.log(strike / spot) / sigma_tau
    z_drift   = mu_6h * (tau_min / 60.0) / sigma_tau
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


# ── P&L ───────────────────────────────────────────────────────────────────────

def pnl_yes(p_market, resolved):
    return BET_SIZE * (1 / p_market - 1) if resolved == 1 else -BET_SIZE

def pnl_no(p_market, resolved):
    return BET_SIZE * (1 / (1 - p_market) - 1) if resolved == 0 else -BET_SIZE


def simulate(df: pd.DataFrame, closes_1h: pd.Series, label: str):
    trades = []
    for _, row in df.iterrows():
        spot      = float(row["spot"])
        strike    = float(row["strike"])
        vol_eff   = float(row["vol_eff"])
        tau_min   = float(row["tau_minutes"])
        pm        = float(row["p_market"])
        resolved  = int(row["resolved_yes"])
        ts        = row["logged_at"]

        if vol_eff <= 0 or tau_min <= 0 or pm <= 0 or pm >= 1:
            continue

        # Compute mu_6h: rolling 6-bar mean of 1h log returns up to (and including)
        # the bar that closes just before the scan timestamp.
        hist = closes_1h[closes_1h.index <= ts]
        if len(hist) < 7:
            mu_6h = 0.0
        else:
            log_rets = np.log(hist.values[1:] / hist.values[:-1])
            mu_6h = float(np.mean(log_rets[-6:]))

        p_pure = p_lognormal(spot, strike, vol_eff, tau_min)
        p_zd   = p_zdrift(spot, strike, vol_eff, tau_min, mu_6h)

        if label == "pure":
            p_yes_model = p_pure
        else:
            p_yes_model = p_zd

        edge_yes = p_yes_model - pm
        edge_no  = (1 - p_yes_model) - (1 - pm)   # = pm - p_yes_model

        if edge_yes >= EDGE_THRESH:
            side = "YES"
            pnl  = pnl_yes(pm, resolved)
        elif edge_no >= EDGE_THRESH:
            side = "NO"
            pnl  = pnl_no(pm, resolved)
        else:
            side = "pass"
            pnl  = 0.0

        trades.append({
            "ts": ts, "pm": pm, "p_pure": p_pure, "p_zd": p_zd,
            "mu_6h": mu_6h, "side": side, "pnl": pnl, "resolved": resolved,
        })

    return pd.DataFrame(trades)


def report(tdf: pd.DataFrame, label: str):
    acted = tdf[tdf["side"] != "pass"]
    if len(acted) == 0:
        print(f"{label}: no trades")
        return
    wins = (acted["pnl"] > 0).sum()
    losses = (acted["pnl"] < 0).sum()
    wr = wins / len(acted)
    total_pnl = acted["pnl"].sum()
    avg_pm = acted["pm"].mean()
    be_wr = avg_pm  # breakeven WR ≈ avg p_market for YES, but mixed — show avg pm instead
    print(f"{label}:")
    print(f"  Trades: {len(acted):3d}  ({tdf[tdf['side']=='YES']['side'].count()} YES / "
          f"{tdf[tdf['side']=='NO']['side'].count()} NO)")
    print(f"  WR:     {wr:.1%}  ({wins}W / {losses}L)  avg pm={avg_pm:.3f}")
    print(f"  PnL:    ${total_pnl:+.0f}  (flat ${BET_SIZE:.0f}/trade)")
    print(f"  Passes: {(tdf['side']=='pass').sum()}")


def calibration(tdf: pd.DataFrame, p_col: str, label: str):
    print(f"\n  Calibration — {label} (test set, YES resolution rate vs {p_col}):")
    buckets = np.arange(0.0, 1.01, 0.1)
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (tdf[p_col] >= lo) & (tdf[p_col] < hi)
        n = mask.sum()
        if n < 5:
            continue
        actual = tdf.loc[mask, "resolved"].mean()
        pred   = tdf.loc[mask, p_col].mean()
        print(f"    [{lo:.1f},{hi:.1f})  n={n:3d}  pred={pred:.2f}  actual={actual:.2f}  Δ={actual-pred:+.2f}")


def main():
    print(SEP)
    print("  BTC Hourly: Pure Lognormal vs 6h Rolling Z-Drift Backtest")
    print(SEP)

    # Load scan archive
    df = pd.read_csv(RESULTS_DIR / "btc_scan_archive.csv", low_memory=False)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    df = df.drop_duplicates(subset=["contract_ticker"], keep="first").copy()
    for c in ["spot", "strike", "vol_eff", "tau_minutes", "p_market"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    df = df.sort_values("logged_at").reset_index(drop=True)
    df = df.dropna(subset=["spot", "strike", "vol_eff", "tau_minutes", "p_market"])
    print(f"Loaded {len(df)} resolved contracts  "
          f"({df['logged_at'].iloc[0].date()} → {df['logged_at'].iloc[-1].date()})")
    print(f"YES rate: {df['resolved_yes'].mean():.1%}  mean pm: {df['p_market'].mean():.3f}")
    print()

    # Fetch Binance 1h
    t_start = df["logged_at"].min() - pd.Timedelta(hours=12)
    t_end   = df["logged_at"].max() + pd.Timedelta(hours=1)
    print(f"Fetching Binance 1h closes {t_start.date()} → {t_end.date()}...")
    closes_1h = fetch_btc_1h(t_start, t_end)
    print(f"  {len(closes_1h)} 1h bars  ({closes_1h.index[0]} → {closes_1h.index[-1]})")
    print()

    # Z-drift stats
    mu_vals = []
    for _, row in df.iterrows():
        hist = closes_1h[closes_1h.index <= row["logged_at"]]
        if len(hist) < 7:
            mu_vals.append(0.0)
        else:
            log_rets = np.log(hist.values[1:] / hist.values[:-1])
            mu_vals.append(float(np.mean(log_rets[-6:])))
    df["mu_6h"] = mu_vals
    print(f"mu_6h stats:  mean={df['mu_6h'].mean():.6f}  "
          f"std={df['mu_6h'].std():.6f}  "
          f"min={df['mu_6h'].min():.6f}  max={df['mu_6h'].max():.6f}")
    pos_frac = (df["mu_6h"] > 0).mean()
    print(f"  Positive (bullish): {pos_frac:.1%}   Negative: {1-pos_frac:.1%}")
    print()

    # Simulate both models
    tdf_pure = simulate(df, closes_1h, "pure")
    tdf_zd   = simulate(df, closes_1h, "zdrift")

    print(SEP2)
    report(tdf_pure, "Pure lognormal")
    print()
    report(tdf_zd, "Z-drift model")
    print()

    # Breakdown: trades where z-drift changes the decision
    merged = tdf_pure[["ts", "side", "pnl", "resolved", "pm", "p_pure", "p_zd", "mu_6h"]].copy()
    merged = merged.rename(columns={"side": "side_pure", "pnl": "pnl_pure"})
    merged["side_zd"]  = tdf_zd["side"].values
    merged["pnl_zd"]   = tdf_zd["pnl"].values
    changed = merged[merged["side_pure"] != merged["side_zd"]]
    print(SEP2)
    print(f"Contracts where z-drift changes decision: {len(changed)}")
    if len(changed) > 0:
        added   = changed[changed["side_pure"] == "pass"]
        dropped = changed[changed["side_zd"] == "pass"]
        flipped = changed[(changed["side_pure"] != "pass") & (changed["side_zd"] != "pass")]
        print(f"  z-drift ADDS trade (pure=pass → zd=bet):     "
              f"n={len(added):3d}  PnL=${added['pnl_zd'].sum():+.0f}")
        print(f"  z-drift DROPS trade (pure=bet → zd=pass):    "
              f"n={len(dropped):3d}  PnL pure=${dropped['pnl_pure'].sum():+.0f}")
        print(f"  z-drift FLIPS side (YES↔NO):                 "
              f"n={len(flipped):3d}  PnL delta=${flipped['pnl_zd'].sum()-flipped['pnl_pure'].sum():+.0f}")

    # Calibration comparison (all scanned rows, not just trades)
    print()
    print(SEP2)
    calibration(tdf_pure, "p_pure", "Pure lognormal")
    calibration(tdf_pure, "p_zd",   "Z-drift model")

    # Daily breakdown
    print()
    print(SEP2)
    print("Daily P&L comparison (flat $10/trade):")
    print(f"  {'Date':<12}  {'Pure trades':>12}  {'Pure PnL':>10}  {'ZD trades':>10}  {'ZD PnL':>10}  {'Delta':>8}")
    tdf_pure["date"] = tdf_pure["ts"].dt.date
    tdf_zd["date"]   = tdf_zd["ts"].dt.date
    for d in sorted(tdf_pure["date"].unique()):
        dp = tdf_pure[(tdf_pure["date"] == d) & (tdf_pure["side"] != "pass")]
        dz = tdf_zd[(tdf_zd["date"] == d) & (tdf_zd["side"] != "pass")]
        pp = dp["pnl"].sum()
        zp = dz["pnl"].sum()
        print(f"  {str(d):<12}  {len(dp):>12}  ${pp:>+9.0f}  {len(dz):>10}  ${zp:>+9.0f}  ${zp-pp:>+7.0f}")

    print()
    print(SEP)


if __name__ == "__main__":
    main()
