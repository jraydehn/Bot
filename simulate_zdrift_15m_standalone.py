#!/usr/bin/env python3
"""
simulate_zdrift_15m_standalone.py

Walk-forward comparison: pure log-normal z_drift model vs LightGBM baseline.
Same formula as the live 1h BTC model:
    z_strike = log(K/S) / sigma_tau
    z_drift  = 0.6 * mean(actual_z[-10:]) + 0.4 * mean(actual_z[-30:])  capped ±0.5
    p_yes    = norm.cdf(z_drift - z_strike)

No k_drift, no p_up, no LightGBM.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR    = Path(__file__).parent / "results"
CSV_15M        = RESULTS_DIR / "paper_trades_btc15m.csv"
MINS_PER_YEAR  = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT = 10
W_LONG  = 30
ALPHA   = 0.6
CAP     = 0.5


def fetch_binance_1m(start_ms: int, end_ms: int) -> pd.DataFrame:
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": cur,
            "endTime": min(cur + 1000 * 60_000, end_ms),
            "limit": 1000,
        }, timeout=15)
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
        "open_time","open","high","low","close","volume",
        "close_time_ms","qav","trades","tbbav","tbqav","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open","close"]].astype(float)
    return df


def price_at(ts: pd.Timestamp, m1_idx, m1_open) -> float:
    if ts in m1_idx:
        return float(m1_open[ts])
    i = m1_idx.searchsorted(ts)
    return float(m1_open.iloc[i]) if i < len(m1_idx) else float("nan")


def compute_actual_z(row, m1_idx, m1_open) -> float:
    try:
        spot   = float(row["spot"])
        rv_ann = float(row["realized_vol_annual"])
        tau    = float(row["tau_minutes"])
        if spot <= 0 or rv_ann <= 0 or tau <= 0:
            return float("nan")
        sigma  = (rv_ann / math.sqrt(MINS_PER_YEAR)) * math.sqrt(tau)
        if sigma <= 0:
            return float("nan")
        ts     = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        expiry = price_at(ts, m1_idx, m1_open)
        if math.isnan(expiry) or expiry <= 0:
            return float("nan")
        return math.log(expiry / spot) / sigma
    except Exception:
        return float("nan")


def zdrift_from(az_list: list) -> float:
    n = len(az_list)
    if n < W_SHORT:
        return float("nan")
    tail = az_list[-max(W_LONG, n):]
    zs = float(np.mean(tail[-W_SHORT:]))
    zl = float(np.mean(tail[-W_LONG:]) if len(tail) >= W_LONG else np.mean(tail))
    return float(np.clip(ALPHA * zs + (1 - ALPHA) * zl, -CAP, CAP))


def p_zdrift_model(spot, strike, rv_ann, tau, zd) -> float:
    sigma = (rv_ann / math.sqrt(MINS_PER_YEAR)) * math.sqrt(tau)
    if sigma <= 0:
        return float("nan")
    z_strike = math.log(strike / spot) / sigma
    return float(np.clip(norm.cdf(zd - z_strike), 0.03, 0.97))


def main():
    df = pd.read_csv(CSV_15M, low_memory=False)
    for c in ["spot","realized_vol_annual","tau_minutes","p_market",
              "p_model_15m","bet_amount","would_pnl","resolved_yes","floor_strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.sort_values("close_time").reset_index(drop=True)

    resolved = df[df["resolved_yes"].notna() & df["close_time"].notna()].copy()
    trades   = resolved[resolved["decision"] == "trade"].copy()
    print(f"Resolved rows: {len(resolved)}  |  Trade rows: {len(trades)}")
    print(f"Date range: {resolved['close_time'].min().date()} → {resolved['close_time'].max().date()}")

    # Fetch 1m prices
    s_ms = int(resolved["close_time"].min().timestamp() * 1000) - 120_000
    e_ms = int(resolved["close_time"].max().timestamp() * 1000) + 120_000
    print("Fetching Binance 1m candles...")
    m1 = fetch_binance_1m(s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    # Compute actual_z for all resolved rows
    resolved["actual_z"] = [compute_actual_z(r, m1.index, m1["open"])
                             for _, r in resolved.iterrows()]
    valid = resolved["actual_z"].notna().sum()
    print(f"actual_z valid: {valid}/{len(resolved)}")
    ar1 = float(pd.Series(resolved["actual_z"].dropna().values).autocorr(1))
    print(f"actual_z AR(1) = {ar1:+.4f}")

    # Walk-forward: for each trade row, compute z_drift from all prior resolved rows
    prior_az: list[float] = []
    results = []

    for i, row in resolved.iterrows():
        is_trade = (row["decision"] == "trade"
                    and not pd.isna(row["would_pnl"])
                    and not pd.isna(row["p_model_15m"]))

        if is_trade:
            zd      = zdrift_from(prior_az)
            spot    = float(row["spot"])
            strike  = float(row["floor_strike"])
            rv_ann  = float(row["realized_vol_annual"])
            tau     = float(row["tau_minutes"])
            pm      = float(row["p_market"])
            side    = str(row.get("side","yes")).lower()
            bet_amt = float(row["bet_amount"])
            outcome = int(row["resolved_yes"])
            orig_pnl = float(row["would_pnl"])

            # Baseline LGBM P&L
            base_pnl = orig_pnl

            # z_drift standalone model P&L
            if not math.isnan(zd):
                p_zd  = p_zdrift_model(spot, strike, rv_ann, tau, zd)
                edge_zd = (p_zd - pm) if side == "yes" else (pm - p_zd)

                if edge_zd >= EDGE_THRESHOLD:
                    if side == "yes":
                        payout  = (1 - pm) / pm if pm > 0 else 0
                        zd_pnl  = bet_amt * payout if outcome == 1 else -bet_amt
                    else:
                        payout  = pm / (1 - pm) if pm < 1 else 0
                        zd_pnl  = bet_amt * payout if outcome == 0 else -bet_amt
                else:
                    zd_pnl = 0.0  # blocked
            else:
                zd_pnl  = base_pnl  # no prior data → pass through unchanged
                p_zd    = float("nan")
                edge_zd = float("nan")

            results.append({
                "close_time": row["close_time"],
                "side":       side,
                "pm":         pm,
                "p_lgbm":     float(row["p_model_15m"]),
                "p_zd":       p_zd if not math.isnan(zd) else float("nan"),
                "zd":         zd,
                "edge_lgbm":  (float(row["p_model_15m"]) - pm) if side=="yes" else (pm - float(row["p_model_15m"])),
                "edge_zd":    edge_zd,
                "base_pnl":   base_pnl,
                "zd_pnl":     zd_pnl,
                "outcome":    outcome,
            })

        # Add this row's actual_z to prior list (only if valid)
        if not math.isnan(float(row["actual_z"]) if not pd.isna(row["actual_z"]) else float("nan")):
            prior_az.append(float(row["actual_z"]))

    rdf = pd.DataFrame(results)
    active = rdf[rdf["zd"].notna()]

    print(f"\n{'='*60}")
    print(f"  STANDALONE z_drift vs LGBM  (walk-forward)")
    print(f"{'='*60}")
    print(f"  Total trade rows evaluated:   {len(rdf)}")
    print(f"  Rows with z_drift active:     {len(active)}  (n_prior ≥ {W_SHORT})")

    base_total = rdf["base_pnl"].sum()
    zd_total   = rdf["zd_pnl"].sum()
    delta      = zd_total - base_total

    print(f"\n  LGBM baseline P&L:   ${base_total:+.2f}")
    print(f"  z_drift model P&L:   ${zd_total:+.2f}")
    print(f"  Delta:               ${delta:+.2f}")

    # Active rows only
    ab = active["base_pnl"].sum()
    az = active["zd_pnl"].sum()
    print(f"\n  Active rows only:")
    print(f"    LGBM P&L:    ${ab:+.2f}")
    print(f"    z_drift P&L: ${az:+.2f}")
    print(f"    Delta:       ${az - ab:+.2f}")

    # Breakdown by side
    for s in ["yes", "no"]:
        sub = active[active["side"] == s]
        if sub.empty:
            continue
        blocked = (sub["zd_pnl"] == 0).sum()
        print(f"\n  {s.upper()} trades (active z_drift, n={len(sub)}):")
        print(f"    LGBM P&L:    ${sub['base_pnl'].sum():+.2f}")
        print(f"    z_drift P&L: ${sub['zd_pnl'].sum():+.2f}")
        print(f"    Delta:       ${sub['zd_pnl'].sum() - sub['base_pnl'].sum():+.2f}")
        print(f"    Blocked:     {blocked}  (edge < {EDGE_THRESHOLD})")

    # z_drift value distribution
    zd_vals = active["zd"].values
    print(f"\n  z_drift stats: mean={np.mean(zd_vals):+.4f}  "
          f"std={np.std(zd_vals):.4f}  "
          f"min={np.min(zd_vals):+.4f}  max={np.max(zd_vals):+.4f}")
    print(f"  z_drift > 0: {(zd_vals > 0).sum()}  |  z_drift <= 0: {(zd_vals <= 0).sum()}")

    # Daily breakdown
    active2 = active.copy()
    active2["date"] = pd.to_datetime(active2["close_time"]).dt.date
    print(f"\n  Daily breakdown:")
    for d, g in active2.groupby("date"):
        b = g["base_pnl"].sum()
        z = g["zd_pnl"].sum()
        print(f"    {d}  n={len(g):3d}  LGBM=${b:+7.2f}  z_drift=${z:+7.2f}  delta=${z-b:+7.2f}  "
              f"zd_mean={g['zd'].mean():+.3f}")

    # Model calibration comparison
    print(f"\n  Calibration (p_model vs actual WR):")
    print(f"  {'Bucket':<18} {'n':>4} {'LGBM_pred':>10} {'zd_pred':>8} {'ActualWR':>9}")
    for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0)]:
        sub = active[(active["p_lgbm"] >= lo) & (active["p_lgbm"] < hi)]
        if len(sub) < 3:
            continue
        wr   = sub["outcome"].mean()
        pred = sub["p_lgbm"].mean()
        zpred = sub["p_zd"].mean() if sub["p_zd"].notna().any() else float("nan")
        print(f"  p_lgbm [{lo:.1f},{hi:.1f})  {len(sub):>4}  {pred:>10.3f}  {zpred:>8.3f}  {wr:>9.3f}")


if __name__ == "__main__":
    main()
