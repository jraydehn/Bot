"""
walkforward_composite_pup.py — Walk-forward validation of composite p_up table.

Tests whether (trend_score, rev_score) → p_up has genuine OOS predictive power.

Methodology:
  Expanding train window, fixed 3-month test window, 8 quarterly folds.
  Each fold builds a fresh calibration table from train data only,
  then evaluates it on the held-out test period.

  Train: all data from DATA_START to fold_cutoff
  Test:  fold_cutoff to fold_cutoff + 3 months (OOS)

Folds:
  1  Train: Jan 2024 – Jun 2024   Test: Jul 2024 – Sep 2024
  2  Train: Jan 2024 – Sep 2024   Test: Oct 2024 – Dec 2024
  3  Train: Jan 2024 – Dec 2024   Test: Jan 2025 – Mar 2025
  4  Train: Jan 2024 – Mar 2025   Test: Apr 2025 – Jun 2025
  5  Train: Jan 2024 – Jun 2025   Test: Jul 2025 – Sep 2025
  6  Train: Jan 2024 – Sep 2025   Test: Oct 2025 – Dec 2025
  7  Train: Jan 2024 – Dec 2025   Test: Jan 2026 – Mar 2026
  8  Train: Jan 2024 – Mar 2026   Test: Apr 2026 – Jun 2026

Metrics (per fold and aggregate):
  Brier score : mean((p_up - next_up)²); baseline = 0.2500 (50/50)
  IC          : Pearson corr(p_up, next_up); >0 means directional signal
  Cal error   : mean(|bin_predicted - bin_actual|) across 5 p_up quintiles
  Top/bot edge: actual win rate in top-20% vs bottom-20% p_up cells
"""

import glob
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from composite_scorer import (
    compute_scores,
    ASSET_BASELINES,
    SMOOTHING_N,
)

# ── Config ────────────────────────────────────────────────────────────────────
ASSET       = "BTC"
TICKER      = "BTCUSDT"
DATA_START  = pd.Timestamp("2024-01-01", tz="UTC")
BASELINE    = ASSET_BASELINES[ASSET]

# Fold cutoffs (end of train / start of test)
FOLD_CUTS = [
    pd.Timestamp("2024-07-01", tz="UTC"),
    pd.Timestamp("2024-10-01", tz="UTC"),
    pd.Timestamp("2025-01-01", tz="UTC"),
    pd.Timestamp("2025-04-01", tz="UTC"),
    pd.Timestamp("2025-07-01", tz="UTC"),
    pd.Timestamp("2025-10-01", tz="UTC"),
    pd.Timestamp("2026-01-01", tz="UTC"),
    pd.Timestamp("2026-04-01", tz="UTC"),
]
TEST_LEN = pd.DateOffset(months=3)

TREND_CLIP = 6    # clip to [-6, +6]
REV_CLIP   = 11   # clip to [-11, +11]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")

    f1h = sorted(glob.glob(str(ROOT / f"data/binanceus_{TICKER}_1h_1970*.parquet")))[-1]
    f1m = sorted(glob.glob(str(ROOT / f"data/binanceus_{TICKER}_1m_2024*.parquet")))[-1]

    df1h = pd.read_parquet(f1h)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    df1h = df1h.sort_index()
    df1h.columns = df1h.columns.str.lower()

    df1m = pd.read_parquet(f1m)
    df1m.index = pd.to_datetime(df1m.index, utc=True)
    df1m = df1m.sort_index()
    df1m.columns = df1m.columns.str.lower()

    # Derive 4h from 1h
    df4h = df1h.resample("4h", origin="start_day").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    # Derive 15m from 1m
    df15m = df1m.resample("15min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    print(f"  1h:  {df1h.index.min().date()} → {df1h.index.max().date()}  ({len(df1h):,} bars)")
    print(f"  1m:  {df1m.index.min().date()} → {df1m.index.max().date()}  ({len(df1m):,} bars)")
    print(f"  4h:  {df4h.index.min().date()} → {df4h.index.max().date()}  ({len(df4h):,} bars)")
    print(f"  15m: {df15m.index.min().date()} → {df15m.index.max().date()}  ({len(df15m):,} bars)")

    return df1h, df4h, df15m, df1m


# ── Score computation ─────────────────────────────────────────────────────────
def compute_all_scores(df1h, df4h, df15m, df1m):
    """Compute (trend, rev, next_up) for every 1h bar in the data range."""
    print("\nComputing composite scores for full history...")

    ts_1h = df1h.loc[DATA_START:].index

    # Pad lookback — need 200+ bars of history before DATA_START for indicators
    pad_start = DATA_START - pd.DateOffset(months=6)
    df1h_w  = df1h.loc[pad_start:]
    df4h_w  = df4h.loc[pad_start:]
    df15m_w = df15m.loc[pad_start:]
    df1m_w  = df1m.loc[pad_start:]

    trend_ser, rev_ser = compute_scores(
        close_1h  = df1h_w["close"],
        high_1h   = df1h_w["high"],
        low_1h    = df1h_w["low"],
        volume_1h = df1h_w["volume"],
        close_4h  = df4h_w["close"],
        high_4h   = df4h_w["high"],
        low_4h    = df4h_w["low"],
        volume_4h = df4h_w["volume"],
        close_15m = df15m_w["close"],
        high_15m  = df15m_w["high"],
        low_15m   = df15m_w["low"],
        close_1m  = df1m_w["close"],
        volume_1m = df1m_w["volume"],
        ts_1h     = df1h_w.loc[DATA_START:].index,
    )

    next_ret = np.log(df1h["close"] / df1h["close"].shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(float)

    df = pd.DataFrame({
        "trend":   trend_ser.reindex(ts_1h).fillna(0).astype(int),
        "rev":     rev_ser.reindex(ts_1h).fillna(0).astype(int),
        "next_up": next_up.reindex(ts_1h),
    }).dropna()

    print(f"  Scored {len(df):,} hours  |  actual up%: {df['next_up'].mean():.1%}")
    return df


# ── Per-fold calibration ──────────────────────────────────────────────────────
def build_table(df_train, baseline):
    """Build p_up lookup dict from train rows."""
    df_train = df_train.copy()
    df_train["tb"] = df_train["trend"].clip(-TREND_CLIP, TREND_CLIP)
    df_train["rb"] = df_train["rev"].clip(-REV_CLIP, REV_CLIP)

    cal = {}
    for (tb, rb), grp in df_train.groupby(["tb", "rb"]):
        n = len(grp)
        if n < 5:
            continue
        up = grp["next_up"].mean()
        w  = min(1.0, n / SMOOTHING_N)
        cal[(int(tb), int(rb))] = w * up + (1 - w) * baseline
    return cal


def lookup(cal, trend, rev, baseline, trend_clip=TREND_CLIP, rev_clip=REV_CLIP):
    tb = int(np.clip(trend, -trend_clip, trend_clip))
    rb = int(np.clip(rev, -rev_clip, rev_clip))
    if (tb, rb) in cal:
        return cal[(tb, rb)]
    # fallback: search adjacent cells
    for dt in range(0, 3):
        for dr in range(0, 3):
            for tb2, rb2 in [(tb+dt, rb+dr), (tb-dt, rb+dr),
                             (tb+dt, rb-dr), (tb-dt, rb-dr)]:
                if (tb2, rb2) in cal:
                    return cal[(tb2, rb2)]
    return baseline


# ── Metrics ───────────────────────────────────────────────────────────────────
def brier_score(p, y):
    return float(np.mean((np.array(p) - np.array(y)) ** 2))


def ic(p, y):
    p, y = np.array(p), np.array(y)
    if p.std() < 1e-9:
        return 0.0
    r, _ = pearsonr(p, y)
    return float(r)


def calibration_curve(p, y, n_bins=5):
    p, y = np.array(p), np.array(y)
    bins = np.percentile(p, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.sum() < 5:
            continue
        rows.append({
            "bin":        i + 1,
            "p_mean":     p[mask].mean(),
            "actual":     y[mask].mean(),
            "n":          int(mask.sum()),
            "error":      abs(p[mask].mean() - y[mask].mean()),
        })
    return rows


# ── Main walk-forward loop ────────────────────────────────────────────────────
def run_walkforward(df_all):
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION — composite p_up")
    print(f"  Baseline up%: {BASELINE:.1%}   Brier baseline: {BASELINE*(1-BASELINE):.4f}")
    print("=" * 70)

    all_p, all_y = [], []
    fold_results = []

    for i, cut in enumerate(FOLD_CUTS):
        test_end = cut + TEST_LEN
        df_train = df_all[(df_all.index >= DATA_START) & (df_all.index < cut)]
        df_test  = df_all[(df_all.index >= cut) & (df_all.index < test_end)]

        if len(df_train) < 200 or len(df_test) < 50:
            print(f"  Fold {i+1}: insufficient data, skipping")
            continue

        baseline_fold = df_train["next_up"].mean()
        cal = build_table(df_train, baseline_fold)

        p_list = [lookup(cal, r["trend"], r["rev"], baseline_fold)
                  for _, r in df_test.iterrows()]
        y_list = df_test["next_up"].tolist()

        bs   = brier_score(p_list, y_list)
        bs0  = baseline_fold * (1 - baseline_fold)
        ic_v = ic(p_list, y_list)
        cal_curve = calibration_curve(p_list, y_list)
        cal_err = np.mean([r["error"] for r in cal_curve]) if cal_curve else float("nan")

        p_arr = np.array(p_list)
        y_arr = np.array(y_list)
        top20  = y_arr[p_arr >= np.percentile(p_arr, 80)].mean() if (p_arr >= np.percentile(p_arr, 80)).sum() > 0 else float("nan")
        bot20  = y_arr[p_arr <= np.percentile(p_arr, 20)].mean() if (p_arr <= np.percentile(p_arr, 20)).sum() > 0 else float("nan")

        fold_results.append({
            "fold":      i + 1,
            "train_end": cut.date(),
            "test_start": cut.date(),
            "test_end":  test_end.date(),
            "n_train":   len(df_train),
            "n_test":    len(df_test),
            "n_cells":   len(cal),
            "baseline":  baseline_fold,
            "brier":     bs,
            "brier_baseline": bs0,
            "brier_skill": 1 - bs / bs0,
            "ic":        ic_v,
            "cal_err":   cal_err,
            "top20_wr":  top20,
            "bot20_wr":  bot20,
        })
        all_p.extend(p_list)
        all_y.extend(y_list)

        print(f"\n  Fold {i+1}: train→{cut.date()}  test {cut.date()}→{test_end.date()}")
        print(f"    n_train={len(df_train):,}  n_test={len(df_test):,}  cells={len(cal)}")
        print(f"    Brier:  {bs:.5f}  (baseline={bs0:.5f}  skill={1-bs/bs0:+.3f})")
        print(f"    IC:     {ic_v:+.4f}")
        print(f"    Cal err:{cal_err:.4f}  (mean |predicted - actual| by quintile)")
        print(f"    Top-20% p_up → actual WR={top20:.1%}   Bot-20% → WR={bot20:.1%}")

        # Calibration curve detail
        print(f"    Calibration curve:")
        print(f"      {'Bin':>4}  {'p_mean':>7}  {'actual':>7}  {'err':>6}  {'n':>5}")
        for row in cal_curve:
            marker = " *" if row["error"] > 0.04 else ""
            print(f"      {row['bin']:>4}  {row['p_mean']:>7.3f}  {row['actual']:>7.3f}"
                  f"  {row['error']:>6.3f}  {row['n']:>5}{marker}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AGGREGATE (all OOS folds pooled)")
    print("=" * 70)
    if all_p:
        agg_bs   = brier_score(all_p, all_y)
        agg_bs0  = BASELINE * (1 - BASELINE)
        agg_ic   = ic(all_p, all_y)
        p_arr = np.array(all_p)
        y_arr = np.array(all_y)
        top20 = y_arr[p_arr >= np.percentile(p_arr, 80)].mean()
        bot20 = y_arr[p_arr <= np.percentile(p_arr, 20)].mean()
        agg_cal = calibration_curve(all_p, all_y, n_bins=10)
        agg_cal_err = np.mean([r["error"] for r in agg_cal])

        print(f"  n_OOS={len(all_p):,}   actual up%={np.mean(all_y):.1%}")
        print(f"  Brier:    {agg_bs:.5f}  (baseline={agg_bs0:.5f}  skill={1-agg_bs/agg_bs0:+.3f})")
        print(f"  IC:       {agg_ic:+.4f}")
        print(f"  Cal err:  {agg_cal_err:.4f}")
        print(f"  Top-20%:  {top20:.1%}  Bot-20%: {bot20:.1%}  spread={top20-bot20:+.1%}")

        print(f"\n  Calibration (10 deciles, all OOS pooled):")
        print(f"    {'Bin':>4}  {'p_mean':>7}  {'actual':>7}  {'err':>6}  {'n':>6}")
        for row in agg_cal:
            marker = " *" if row["error"] > 0.04 else ""
            print(f"    {row['bin']:>4}  {row['p_mean']:>7.3f}  {row['actual']:>7.3f}"
                  f"  {row['error']:>6.3f}  {row['n']:>6}{marker}")

    # ── Fold summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FOLD SUMMARY")
    print("=" * 70)
    print(f"  {'Fold':>4}  {'Test period':>22}  {'Brier skill':>11}  {'IC':>7}  {'Top20':>6}  {'Bot20':>6}")
    print("  " + "-" * 64)
    for r in fold_results:
        print(f"  {r['fold']:>4}  {str(r['test_start'])+'→'+str(r['test_end']):>22}  "
              f"{r['brier_skill']:>+11.3f}  {r['ic']:>+7.4f}  "
              f"{r['top20_wr']:>6.1%}  {r['bot20_wr']:>6.1%}")

    avg_skill = np.mean([r["brier_skill"] for r in fold_results])
    avg_ic    = np.mean([r["ic"] for r in fold_results])
    print("  " + "-" * 64)
    print(f"  {'AVG':>4}  {'':>22}  {avg_skill:>+11.3f}  {avg_ic:>+7.4f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    if avg_ic > 0.03:
        print("  IC > 0.03: composite scores carry meaningful directional signal OOS.")
    elif avg_ic > 0.01:
        print("  IC 0.01–0.03: weak but potentially useful directional signal OOS.")
    else:
        print("  IC < 0.01: composite scores have near-zero directional power OOS.")

    if avg_skill > 0.005:
        print("  Brier skill > 0: calibration table beats naive baseline OOS.")
    else:
        print("  Brier skill ≤ 0: calibration table does NOT beat naive baseline OOS.")

    spread = None
    if fold_results:
        spreads = [r["top20_wr"] - r["bot20_wr"] for r in fold_results
                   if not math.isnan(r["top20_wr"]) and not math.isnan(r["bot20_wr"])]
        if spreads:
            spread = np.mean(spreads)
            print(f"  Top/bot-20% spread = {spread:+.1%}  "
                  f"({'meaningful edge' if spread > 0.04 else 'limited edge'} at extremes)")

    return fold_results


if __name__ == "__main__":
    df1h, df4h, df15m, df1m = load_data()
    df_all = compute_all_scores(df1h, df4h, df15m, df1m)
    run_walkforward(df_all)
