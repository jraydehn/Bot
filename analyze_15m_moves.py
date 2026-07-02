#!/usr/bin/env python3
"""
analyze_15m_moves.py

Empirical distribution of 15-minute price moves from Binance parquet data.
Answers: given current volatility and strike distance from spot, how likely
is a move large enough to reach the strike?

Sections:
  1. Raw percentile distribution of |fwd_15m_ret|
  2. Sigma-normalized moves (move / sigma_15m): how "fat" are the tails?
  3. By vol regime (low / medium / high): does tail thickness change?
  4. Practical grid: P(|move| > X%) at different vol levels
  5. Implied probability table for Kalshi-like YES bets
     (given offset_pct from spot and tau, what fraction of bars win?)

Run: python3 analyze_15m_moves.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR     = Path("data")
MINS_PER_YEAR = 525_600.0
SEP  = "=" * 78
SEP2 = "-" * 78

ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

PERCENTILES = [50, 75, 85, 90, 95, 97, 99, 99.5]
MOVE_THRESHOLDS = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]  # %
SIGMA_THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def latest_parquet(sym: str) -> Path:
    files = sorted(DATA_DIR.glob(f"binanceus_{sym}_1m_2024-01-01_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 1m parquet for {sym}")
    return files[-1]


def load_and_compute(sym: str) -> pd.DataFrame:
    df1m = pd.read_parquet(latest_parquet(sym))
    if df1m.index.tz is None:
        df1m.index = df1m.index.tz_localize("UTC")
    df1m = df1m.sort_index()

    # Realized vol using last 60 1m log-returns (annualized)
    log_ret_1m = np.log(df1m["close"] / df1m["close"].shift(1))
    sigma_1m   = log_ret_1m.rolling(60).std()
    sigma_annual = sigma_1m * np.sqrt(MINS_PER_YEAR)

    # Resample to 15m
    df15 = df1m.resample("15min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),     close=("close","last"),
        volume=("volume","sum"),
    ).dropna()

    # Forward return: close of next bar / close of this bar
    df15["fwd_ret_pct"] = df15["close"].shift(-1) / df15["close"] * 100 - 100
    df15["fwd_abs_pct"] = df15["fwd_ret_pct"].abs()

    # Bar's own return (for vol regime context)
    df15["bar_ret_pct"]  = (df15["close"] / df15["open"] - 1) * 100
    df15["bar_range_pct"] = (df15["high"] - df15["low"]) / df15["open"] * 100

    # Realized vol at bar open (take the last value of sigma_annual for that minute)
    df15["sigma_annual"]  = sigma_annual.reindex(df15.index, method="ffill")
    df15["sigma_15m_pct"] = df15["sigma_annual"] * np.sqrt(15.0 / MINS_PER_YEAR) * 100.0

    # Sigma-normalized forward move
    df15["fwd_z"] = df15["fwd_ret_pct"] / df15["sigma_15m_pct"].replace(0, np.nan)
    df15["fwd_z_abs"] = df15["fwd_z"].abs()

    # Vol regime buckets
    q33 = df15["sigma_annual"].quantile(0.33)
    q67 = df15["sigma_annual"].quantile(0.67)
    df15["vol_regime"] = pd.cut(
        df15["sigma_annual"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["low", "medium", "high"]
    )

    return df15.dropna(subset=["fwd_ret_pct", "sigma_15m_pct"])


def section1_raw_distribution(df: pd.DataFrame, asset: str):
    print(f"\n  1. RAW |forward 15m move| distribution  ({len(df):,} bars)")
    print(f"  {SEP2}")

    moves = df["fwd_abs_pct"]
    print(f"  Mean: {moves.mean():.3f}%   Median: {moves.median():.3f}%   "
          f"Std: {moves.std():.3f}%   Max: {moves.max():.3f}%")

    print(f"\n  {'Percentile':>12} {'|Move|':>10} {'Annualized vol needed':>22}")
    for p in PERCENTILES:
        v = np.percentile(moves, p)
        # approximate annual vol that makes this a 1-sigma 15m move
        implied_annual = v / 100.0 / np.sqrt(15.0 / MINS_PER_YEAR) * 100.0
        print(f"  {p:>11}th {v:>9.3f}%  {'(≈'+f'{implied_annual:.0f}% ann vol':>22}")

    print(f"\n  P(|move| > X%) over next 15 minutes:")
    print(f"  {'Threshold':>12} {'P(exceed)':>12} {'Approx odds':>14}")
    print(f"  {'-'*12} {'-'*12} {'-'*14}")
    for thresh in MOVE_THRESHOLDS:
        p_exc = (moves > thresh).mean()
        odds  = f"1 in {1/p_exc:.0f}" if p_exc > 0 else "never"
        print(f"  {thresh:>10.2f}%  {p_exc:>11.2%}  {odds:>14}")


def section2_sigma_distribution(df: pd.DataFrame, asset: str):
    z_abs = df["fwd_z_abs"].dropna()
    print(f"\n  2. SIGMA-NORMALIZED |move / sigma_15m| distribution")
    print(f"  {SEP2}")
    print(f"  Mean sigma: {df['sigma_15m_pct'].mean():.3f}%  "
          f"(range: {df['sigma_15m_pct'].min():.3f}% – {df['sigma_15m_pct'].max():.3f}%)")

    print(f"\n  {'Percentile':>12} {'|z|':>8}  {'LogNorm pred':>13}  {'excess':>8}")
    for p in PERCENTILES:
        actual  = np.percentile(z_abs, p)
        lognorm = stats.halfnorm.ppf(p / 100.0)  # theoretical half-normal
        excess  = actual - lognorm
        print(f"  {p:>11}th {actual:>7.3f}   {lognorm:>12.3f}   {excess:>+7.3f}")

    print(f"\n  P(|z| > N sigma) — actual vs lognormal prediction:")
    print(f"  {'N sigma':>9} {'Actual P':>10} {'LogNorm P':>12} {'Fat tail ratio':>15}")
    print(f"  {'-'*9} {'-'*10} {'-'*12} {'-'*15}")
    for n in SIGMA_THRESHOLDS:
        p_actual  = (z_abs > n).mean()
        p_lognorm = 2 * (1 - stats.norm.cdf(n))
        ratio     = p_actual / p_lognorm if p_lognorm > 0 else float("nan")
        print(f"  {n:>8.2f}σ {p_actual:>9.2%}  {p_lognorm:>11.2%}  {ratio:>14.2f}×")


def section3_by_vol_regime(df: pd.DataFrame, asset: str):
    print(f"\n  3. MOVES BY VOLATILITY REGIME")
    print(f"  {SEP2}")

    for regime in ["low", "medium", "high"]:
        sub = df[df["vol_regime"] == regime]
        if sub.empty:
            continue
        moves = sub["fwd_abs_pct"]
        med_sigma = sub["sigma_15m_pct"].median()
        q33 = df["sigma_annual"].quantile(0.33 if regime == "low" else 0)
        q67 = df["sigma_annual"].quantile(0.67 if regime != "high" else 1.0)
        ann_range = f"({sub['sigma_annual'].min():.0f}%–{sub['sigma_annual'].max():.0f}% ann)"

        print(f"\n  Vol regime: {regime.upper():6}  {ann_range}  "
              f"N={len(sub):,}  median sigma_15m={med_sigma:.3f}%")
        print(f"  {'Percentile':>12} {'|Move|':>9} {'in sigma':>9}")

        for p in [50, 75, 90, 95, 99]:
            v   = np.percentile(moves, p)
            sig = np.percentile(sub["fwd_z_abs"].dropna(), p)
            print(f"  {p:>11}th {v:>8.3f}%  {sig:>8.2f}σ")

        print(f"  P(move > 0.5%): {(moves > 0.5).mean():.1%}  "
              f"P(move > 1.0%): {(moves > 1.0).mean():.1%}  "
              f"P(move > 2.0%): {(moves > 2.0).mean():.1%}")


def section4_practical_grid(df: pd.DataFrame, asset: str):
    print(f"\n  4. PRACTICAL GRID: P(move ≥ offset) by vol level")
    print(f"     (What fraction of 15m bars move at least X% in either direction?)")
    print(f"  {SEP2}")

    # Define vol buckets by sigma_15m_pct
    vol_buckets = [
        ("very low  σ<0.15%", df["sigma_15m_pct"] < 0.15),
        ("low    0.15-0.25%", (df["sigma_15m_pct"] >= 0.15) & (df["sigma_15m_pct"] < 0.25)),
        ("medium 0.25-0.40%", (df["sigma_15m_pct"] >= 0.25) & (df["sigma_15m_pct"] < 0.40)),
        ("high   0.40-0.60%", (df["sigma_15m_pct"] >= 0.40) & (df["sigma_15m_pct"] < 0.60)),
        ("very high  ≥0.60%", df["sigma_15m_pct"] >= 0.60),
    ]

    offsets = [0.20, 0.30, 0.50, 0.75, 1.00, 1.50]

    header = f"  {'Vol bucket':<22}"
    for o in offsets:
        header += f"  {o:.2f}%"
    print(header)
    print(f"  {'-'*22}" + "  -----" * len(offsets))

    for label, mask in vol_buckets:
        sub = df[mask]
        if len(sub) < 100:
            continue
        row = f"  {label:<22}"
        for o in offsets:
            p = (sub["fwd_abs_pct"] >= o).mean()
            row += f"  {p:>4.1%}"
        row += f"  (N={len(sub):,})"
        print(row)


def section5_kalshi_implications(df: pd.DataFrame, asset: str):
    print(f"\n  5. KALSHI IMPLICATIONS: P(YES wins) by offset & vol")
    print(f"     Assumes YES bet: spot currently at offset% BELOW strike (OTM YES)")
    print(f"     Wins if price moves UP by at least offset% in tau minutes")
    print(f"  {SEP2}")

    # For OTM YES: need upward move of >= offset_pct to cross strike
    # Use only UP moves (not abs) since direction matters for YES
    up_moves = df[df["fwd_ret_pct"] > 0]["fwd_ret_pct"]
    dn_moves = df[df["fwd_ret_pct"] < 0]["fwd_ret_pct"].abs()

    vol_buckets = [
        ("low  σ<0.20%",   df["sigma_15m_pct"] < 0.20),
        ("med  0.20-0.40%", (df["sigma_15m_pct"] >= 0.20) & (df["sigma_15m_pct"] < 0.40)),
        ("high ≥0.40%",    df["sigma_15m_pct"] >= 0.40),
    ]
    offsets = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

    print(f"\n  OTM YES: P(price up by ≥ offset%) over next 15m")
    hdr = f"  {'Vol bucket':<20}"
    for o in offsets:
        hdr += f"  ≥{o:.2f}%"
    print(hdr)
    print(f"  {'-'*20}" + "  ------" * len(offsets))

    for label, mask in vol_buckets:
        sub = df[mask]
        if len(sub) < 100:
            continue
        row = f"  {label:<20}"
        for o in offsets:
            p = (sub["fwd_ret_pct"] >= o).mean()
            row += f"  {p:>5.1%}"
        row += f"  (N={len(sub):,})"
        print(row)

    print(f"\n  OTM NO: P(price down by ≥ offset%) over next 15m")
    hdr2 = f"  {'Vol bucket':<20}"
    for o in offsets:
        hdr2 += f"  ≥{o:.2f}%"
    print(hdr2)
    print(f"  {'-'*20}" + "  ------" * len(offsets))

    for label, mask in vol_buckets:
        sub = df[mask]
        if len(sub) < 100:
            continue
        row = f"  {label:<20}"
        for o in offsets:
            p = (sub["fwd_ret_pct"] <= -o).mean()
            row += f"  {p:>5.1%}"
        row += f"  (N={len(sub):,})"
        print(row)

    # Recommended p_market floor given empirical probabilities
    print(f"\n  Empirical breakeven p_market by offset (at median vol):")
    print(f"  (below this, the empirical data says don't bet YES regardless of model)")
    med_mask = (df["sigma_15m_pct"] >= df["sigma_15m_pct"].quantile(0.33)) & \
               (df["sigma_15m_pct"] <  df["sigma_15m_pct"].quantile(0.67))
    sub_med  = df[med_mask]
    print(f"  {'offset':>8}  {'P(up)':>8}  {'P(dn)':>8}  {'mid breakeven':>14}")
    for o in offsets:
        p_up = (sub_med["fwd_ret_pct"] >= o).mean()
        p_dn = (sub_med["fwd_ret_pct"] <= -o).mean()
        mid  = (p_up + (1 - p_dn)) / 2  # rough symmetric breakeven for YES
        print(f"  {o:>7.2f}%  {p_up:>7.1%}  {p_dn:>7.1%}  {mid:>13.1%}")


def run_asset(asset: str, sym: str):
    print(f"\n{SEP}")
    print(f"  {asset} ({sym}) — 15-minute move analysis")
    print(SEP)

    print(f"  Loading parquet …", end=" ", flush=True)
    df = load_and_compute(sym)
    print(f"{len(df):,} completed 15m bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Median sigma_15m: {df['sigma_15m_pct'].median():.3f}%  "
          f"(≈ {df['sigma_annual'].median():.0f}% annualized)")

    section1_raw_distribution(df, asset)
    section2_sigma_distribution(df, asset)
    section3_by_vol_regime(df, asset)
    section4_practical_grid(df, asset)
    section5_kalshi_implications(df, asset)


def main():
    print(SEP)
    print("  15-minute price move empirical analysis — BTC / ETH / SOL")
    print("  Source: Binance 1m parquet, Jan 2024 – May 2026")
    print(SEP)

    for asset, sym in ASSETS.items():
        run_asset(asset, sym)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
