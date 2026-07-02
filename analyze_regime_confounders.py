#!/usr/bin/env python3
"""
analyze_regime_confounders.py

Investigates whether the inverse bull-regime pattern (mu_4h > 0.001 → NO wins)
is a robust signal or a confounding artifact.

Tests three hypotheses:
  H1: mu_4h alone is the regime signal (already confirmed)
  H2: Sustained trend (mu_4h + mu_24h both positive) captures bull regime better
  H3: New momentum (mu_4h positive, mu_24h neutral/negative) is actually bearish
      for next 1h bar (bounce exhaustion)

Also checks:
  - Trend persistence (consecutive positive bars) as alternative regime
  - Whether the pattern differs by tau or strike distance
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

FEAT_PKL = "/tmp/calib_feat_dataset.pkl"

THRESH_4H  = 0.001
THRESH_24H = 0.002   # larger threshold for 24h window to normalize for bar count


def load_bars() -> pd.DataFrame:
    df = pd.read_pickle(FEAT_PKL)
    df = df[["close", "label"]].copy()

    # log returns
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # rolling mean log-returns
    df["mu_4h"]   = df["log_ret"].rolling(4).mean()
    df["mu_24h"]  = df["log_ret"].rolling(24).mean()
    df["mu_1h"]   = df["log_ret"]  # just prior bar (1h lookback)

    # trend persistence: consecutive positive/negative bars
    def consec_positive(s):
        result = []
        count = 0
        for val in s:
            if pd.isna(val):
                result.append(0)
                count = 0
            elif val > 0:
                count += 1
                result.append(count)
            else:
                result.append(0)
                count = 0
        return result

    df["consec_up"] = consec_positive(df["log_ret"])

    # regime labels
    def regime_4h(mu):
        if pd.isna(mu): return "neutral"
        if mu > THRESH_4H:  return "bull"
        if mu < -THRESH_4H: return "bear"
        return "neutral"

    def regime_24h(mu):
        if pd.isna(mu): return "neutral"
        if mu > THRESH_24H:  return "bull"
        if mu < -THRESH_24H: return "bear"
        return "neutral"

    df["reg_4h"]  = df["mu_4h"].map(regime_4h)
    df["reg_24h"] = df["mu_24h"].map(regime_24h)

    # combined regime: fast+slow
    def combined_regime(r4, r24):
        if r4 == "bull" and r24 == "bull":   return "sustained_bull"
        if r4 == "bull" and r24 == "bear":   return "new_momentum"
        if r4 == "bull" and r24 == "neutral":return "new_momentum"
        if r4 == "bear" and r24 == "bear":   return "sustained_bear"
        if r4 == "bear" and r24 == "bull":   return "fading_bull"
        if r4 == "bear" and r24 == "neutral":return "new_bear"
        return "neutral"

    df["combined"] = df.apply(lambda row: combined_regime(row["reg_4h"], row["reg_24h"]), axis=1)

    return df


def era_label(ts):
    if ts < pd.Timestamp("2025-01-01", tz="UTC"): return "2024"
    if ts < pd.Timestamp("2025-07-01", tz="UTC"): return "2025H1"
    if ts < pd.Timestamp("2026-01-01", tz="UTC"): return "2025H2"
    if ts < pd.Timestamp("2026-04-01", tz="UTC"): return "2026Q1"
    return "2026Q2"


def analyze_by_regime(df: pd.DataFrame, regime_col: str, label: str):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    grp = df.groupby(regime_col)["label"]
    results = []
    for name, group in grp:
        n = len(group)
        yes_rate = group.mean() * 100
        results.append((name, n, yes_rate))

    # sort by yes_rate
    results.sort(key=lambda x: x[2])

    total = len(df.dropna(subset=["label"]))
    print(f"{'Regime':<20}  {'N':>7}  {'%total':>7}  {'YES%':>6}  {'NO%':>6}")
    print("-" * 55)
    for name, n, yr in results:
        pct = n / total * 100
        print(f"  {name:<18}  {n:>7,}  {pct:>6.1f}%  {yr:>5.1f}%  {100-yr:>5.1f}%")


def analyze_combined_era_stability(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("COMBINED REGIME: Era Stability (mu_4h × mu_24h)")
    print(f"{'='*60}")

    df = df.copy()
    df["era"] = df.index.map(era_label)

    for combo_name in ["sustained_bull", "new_momentum", "neutral", "new_bear", "sustained_bear", "fading_bull"]:
        mask = df["combined"] == combo_name
        sub = df[mask]
        if len(sub) == 0:
            continue
        overall_yr = sub["label"].mean() * 100
        print(f"\n  {combo_name}  (N={len(sub):,}  overall YES={overall_yr:.1f}%)")
        era_grp = sub.groupby("era")["label"]
        for era, grp in era_grp:
            yr = grp.mean() * 100
            print(f"    {era:<8}  N={len(grp):>5,}  YES={yr:.1f}%  NO={100-yr:.1f}%")


def analyze_trend_persistence(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("TREND PERSISTENCE: consecutive up-bars vs YES rate")
    print(f"{'='*60}")

    bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 999]
    labels = ["0", "1", "2", "3", "4", "5", "6-7", "8-9", "10+"]
    df = df.copy()
    df["consec_bin"] = pd.cut(df["consec_up"], bins=bins, labels=labels, right=True)

    grp = df.groupby("consec_bin", observed=True)["label"]
    print(f"{'Consec up':<12}  {'N':>7}  {'YES%':>6}  {'NO%':>6}")
    print("-" * 40)
    for name, group in grp:
        yr = group.mean() * 100
        print(f"  {str(name):<10}  {len(group):>7,}  {yr:>5.1f}%  {100-yr:>5.1f}%")


def analyze_mu24h_within_bull(df: pd.DataFrame):
    """Within bull regime (mu_4h > thresh), does mu_24h matter?"""
    print(f"\n{'='*60}")
    print("WITHIN BULL REGIME: Does mu_24h matter?")
    print(f"(rows where mu_4h > {THRESH_4H})")
    print(f"{'='*60}")

    bull = df[df["reg_4h"] == "bull"].copy()
    print(f"  Total bull-regime bars: {len(bull):,}")

    # split by mu_24h
    thresholds = [(-1, -0.002), (-0.002, 0), (0, 0.002), (0.002, 0.005), (0.005, 1)]
    labels     = ["mu24h<-0.002", "-0.002..0", "0..0.002", "0.002..0.005", ">0.005"]

    print(f"\n  {'mu_24h band':<15}  {'N':>7}  {'YES%':>6}  {'NO%':>6}  {'Interpretation'}")
    print("  " + "-" * 65)
    for (lo, hi), lbl in zip(thresholds, labels):
        sub = bull[(bull["mu_24h"] > lo) & (bull["mu_24h"] <= hi)]
        if len(sub) == 0:
            continue
        yr = sub["label"].mean() * 100
        if yr < 45:
            interp = "<-- strong NO edge"
        elif yr > 55:
            interp = "<-- YES edge"
        else:
            interp = "neutral"
        print(f"  {lbl:<15}  {len(sub):>7,}  {yr:>5.1f}%  {100-yr:>5.1f}%  {interp}")


def analyze_conditional_mu_lag(df: pd.DataFrame):
    """Does prior bar's return predict next bar better than 4h window?"""
    print(f"\n{'='*60}")
    print("PRIOR 1H BAR: Does single bar direction predict next 1h?")
    print(f"{'='*60}")

    print(f"  {'mu_1h band':<18}  {'N':>7}  {'YES%':>6}  {'NO%':>6}")
    print("  " + "-" * 48)

    thresholds = [
        ("strong bear", -1, -0.003),
        ("moderate bear", -0.003, -0.001),
        ("slight bear", -0.001, 0),
        ("flat", 0, 0),
        ("slight bull", 0, 0.001),
        ("moderate bull", 0.001, 0.003),
        ("strong bull", 0.003, 1),
    ]

    for lbl, lo, hi in thresholds:
        if lo == hi:  # skip flat
            continue
        sub = df[(df["mu_1h"] > lo) & (df["mu_1h"] <= hi)]
        if len(sub) == 0:
            continue
        yr = sub["label"].mean() * 100
        print(f"  {lbl:<18}  {len(sub):>7,}  {yr:>5.1f}%  {100-yr:>5.1f}%")


def analyze_new_vs_sustained_era(df: pd.DataFrame):
    """Era stability: new_momentum (mu4h bull, mu24h neutral/bear) vs sustained_bull."""
    print(f"\n{'='*60}")
    print("H2/H3: NEW MOMENTUM vs SUSTAINED BULL — Era Stability")
    print("  new_momentum = mu_4h bull + mu_24h neutral/bear")
    print("  sustained_bull = mu_4h bull + mu_24h bull")
    print(f"{'='*60}")

    df = df.copy()
    df["era"] = df.index.map(era_label)

    for combo_name in ["new_momentum", "sustained_bull"]:
        mask = df["combined"] == combo_name
        sub = df[mask]
        print(f"\n  {combo_name}  (N={len(sub):,})")
        for era, grp in sub.groupby("era")["label"]:
            yr = grp.mean() * 100
            print(f"    {era:<8}  N={len(grp):>5,}  YES={yr:.1f}%  NO={100-yr:.1f}%")


def main():
    print("Loading bars from /tmp/calib_feat_dataset.pkl ...")
    df = load_bars()
    df_valid = df.dropna(subset=["label", "mu_4h", "mu_24h"])
    print(f"  {len(df):,} total bars, {len(df_valid):,} with valid mu_4h + mu_24h")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

    analyze_by_regime(df_valid, "reg_4h",  "H1: mu_4h ALONE (already known)")
    analyze_by_regime(df_valid, "reg_24h", "H2: mu_24h ALONE — is 24h window better?")
    analyze_by_regime(df_valid, "combined","H3: COMBINED REGIME (mu_4h × mu_24h)")

    analyze_combined_era_stability(df_valid)
    analyze_mu24h_within_bull(df_valid)
    analyze_new_vs_sustained_era(df_valid)
    analyze_trend_persistence(df_valid)
    analyze_conditional_mu_lag(df_valid)


if __name__ == "__main__":
    main()
