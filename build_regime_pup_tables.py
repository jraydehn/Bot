"""
build_regime_pup_tables.py — Phase 2 of regime-conditioned p_up reform.

Builds per-macro-regime p_up calibration tables using the same target as
composite_calibration.json: the 1h forward price direction (next_up).

For each macro regime (Bull / Sideways / Bear) from the HMM:
  - Computes composite scores (trend, rev) on historical 1h bar data
  - Joins to HMM regime labels (hmm_macro_labels_btc.parquet)
  - Groups by (tb, rb) and computes up% per regime
  - Smooths toward per-regime baseline
  - Saves to composite_calibration_regime_{Bull,Sideways,Bear}.json

The output format matches composite_calibration.json:
  {"tb,rb": float, ..., "__baseline__": float, "__n__": int}

with tb = clip(trend, -5, 5)  (expanded from -3/+3; 2 new ±2 vote signals)
     rb = clip(rev, -11, 11)  (BTC range)
"""

import json
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
OUT_DIR  = BASE / "reform_results"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE / "data"

LABELS_PATH = OUT_DIR / "hmm_macro_labels_btc.parquet"
DIAG_PATH   = OUT_DIR / "regime_pup_diagnostics.txt"

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
SMOOTH_K   = 30     # pseudo-count toward per-regime baseline
MIN_N      = 10     # cells with fewer obs → use baseline

REV_CLIP  = 11      # BTC: clip(rev, -11, 11) — matches composite_calibration.json
TREND_CLIP = 5      # clip(trend, -5, 5) — expanded from 3 to accommodate 2 new ±2 vote signals
REGIMES   = ["Bull", "Sideways", "Bear"]

sys.path.insert(0, str(BASE))
from composite_scorer import (
    compute_scores,
    ASSET_BASELINES,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_btc_data():
    sym = "BTCUSDT"
    # Use 2024-01-01 long-form parquet files to match composite_scorer.py calibration.
    # The 2026-* short files only have a few days and are skipped by the pattern.
    f_1h  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    f_15m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_15m_2024-01-01_*.parquet")))[-1]
    f_1m  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]

    def load(p):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = df.columns.str.lower()
        return df.sort_index()

    ohlcv_1h  = load(f_1h)
    ohlcv_4h  = load(f_4h)
    ohlcv_15m = load(f_15m)
    ohlcv_1m  = load(f_1m)

    print(f"  1h: {len(ohlcv_1h):,}  4h: {len(ohlcv_4h):,}  "
          f"15m: {len(ohlcv_15m):,}  1m: {len(ohlcv_1m):,}")
    print(f"  Range: {ohlcv_1h.index[0].date()} → {ohlcv_1h.index[-1].date()}")
    return ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m


def compute_history_scores(ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m):
    """Compute composite (trend, rev) for each 1h bar — same as calibration."""
    close_1m  = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)
    ts_1h     = ohlcv_1h.index

    trend_ser, rev_ser = compute_scores(
        ohlcv_1h["close"].astype(float),  ohlcv_1h["high"].astype(float),
        ohlcv_1h["low"].astype(float),    ohlcv_1h["volume"].astype(float),
        ohlcv_4h["close"].astype(float),  ohlcv_4h["high"].astype(float),
        ohlcv_4h["low"].astype(float),    ohlcv_4h["volume"].astype(float),
        ohlcv_15m["close"].astype(float), ohlcv_15m["high"].astype(float),
        ohlcv_15m["low"].astype(float),
        close_1m, volume_1m, ts_1h,
    )
    return trend_ser, rev_ser


def build_scored_df(trend_ser, rev_ser, ohlcv_1h) -> pd.DataFrame:
    """Create (trend, rev, next_up, timestamp) DataFrame restricted to test period."""
    close_1h = ohlcv_1h["close"].astype(float)
    next_ret  = np.log(close_1h / close_1h.shift(1)).shift(-1)
    next_up   = (next_ret > 0).astype(int)

    df = pd.DataFrame({
        "trend":   trend_ser,
        "rev":     rev_ser,
        "next_up": next_up,
    })
    df = df.dropna()
    df = df[df.index >= TEST_START]
    df["tb"] = df["trend"].clip(-TREND_CLIP, TREND_CLIP).astype(int)
    df["rb"] = df["rev"].clip(-REV_CLIP, REV_CLIP).astype(int)
    return df


def join_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add HMM regime label to each scored hour."""
    labels = pd.read_parquet(LABELS_PATH)
    labels.index = pd.to_datetime(labels.index, utc=True)

    # For each scored bar, find the nearest past regime label
    score_times = df.index.values
    label_times = labels.index.values
    idx = np.searchsorted(label_times, score_times, side="right") - 1
    idx = np.clip(idx, 0, len(label_times) - 1)
    df = df.copy()
    df["regime"] = labels["regime"].values[idx]
    return df


def build_table(df_regime: pd.DataFrame, per_regime_baseline: float) -> dict:
    """
    Build (tb, rb) → p_up calibration table for one regime slice.
    Format: {"tb,rb": float, "__baseline__": float, "__n__": int}
    """
    counts = {}
    rates  = {}
    for tb in range(-TREND_CLIP, TREND_CLIP + 1):
        for rb in range(-REV_CLIP, REV_CLIP + 1):
            cell = df_regime[(df_regime["tb"] == tb) & (df_regime["rb"] == rb)]
            n    = len(cell)
            wr   = cell["next_up"].mean() if n >= MIN_N else np.nan
            counts[(tb, rb)] = n
            rates[(tb, rb)]  = wr

    table = {}
    for tb in range(-TREND_CLIP, TREND_CLIP + 1):
        for rb in range(-REV_CLIP, REV_CLIP + 1):
            n  = counts[(tb, rb)]
            wr = rates[(tb, rb)]
            if np.isnan(wr):
                p = per_regime_baseline
            else:
                p = (n * wr + SMOOTH_K * per_regime_baseline) / (n + SMOOTH_K)
            table[f"{tb},{rb}"] = round(float(p), 4)

    table["__baseline__"] = round(per_regime_baseline, 4)
    table["__n__"]        = len(df_regime)
    return table


def calibration_curve(df: pd.DataFrame, table: dict, baseline: float) -> pd.DataFrame:
    """Compute calibration curve: predicted p_up vs actual next_up."""
    preds = []
    for _, row in df.iterrows():
        key = f"{int(row['tb'])},{int(row['rb'])}"
        p   = table.get(key, baseline)
        preds.append(p)
    df = df.copy()
    df["pred_p"] = preds
    df["decile"] = pd.qcut(df["pred_p"], q=10, labels=False, duplicates="drop")
    return df.groupby("decile").agg(
        pred=("pred_p", "mean"),
        actual=("next_up", "mean"),
        n=("next_up", "count")
    ).reset_index()


def walkforward_validate(df_regime: pd.DataFrame, regime_name: str,
                         per_regime_baseline: float, lines: list):
    """3-month OOS walk-forward on the test period."""
    cut = df_regime.index.max() - pd.DateOffset(months=3)
    train = df_regime[df_regime.index < cut]
    test  = df_regime[df_regime.index >= cut]

    if len(train) < 100 or len(test) < 50:
        lines.append(f"  {regime_name}: insufficient data (train={len(train)}, test={len(test)})")
        return

    table_train = build_table(train, train["next_up"].mean())

    preds, actuals = [], []
    for _, row in test.iterrows():
        key = f"{int(row['tb'])},{int(row['rb'])}"
        p   = table_train.get(key, per_regime_baseline)
        preds.append(p)
        actuals.append(row["next_up"])

    preds   = np.array(preds)
    actuals = np.array(actuals)

    ic, _  = pearsonr(preds, actuals)
    brier  = np.mean((preds - actuals) ** 2)
    base_b = np.mean((actuals.mean() - actuals) ** 2)
    skill  = 1 - brier / base_b

    # Quintile spread
    n5     = max(1, len(preds) // 5)
    order  = np.argsort(preds)
    spread = actuals[order[-n5:]].mean() - actuals[order[:n5]].mean()

    lines.append(
        f"  {regime_name:>10}: train={len(train):,}  test={len(test):,}  "
        f"IC={ic:+.3f}  BrierSkill={skill:+.3f}  Spread={spread:+.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading BTC data (1h / 4h / 15m / 1m)...")
    ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m = load_btc_data()

    print("\nComputing composite scores (this may take 60–90 seconds)...")
    trend_ser, rev_ser = compute_history_scores(ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m)

    print("\nBuilding scored DataFrame...")
    df = build_scored_df(trend_ser, rev_ser, ohlcv_1h)
    print(f"  {len(df):,} bars  ({df.index.min().date()} → {df.index.max().date()})")
    baseline_pooled = df["next_up"].mean()
    print(f"  Pooled baseline next_up: {baseline_pooled:.4f}")

    print("\nJoining HMM regime labels...")
    df = join_regime(df)
    regime_counts = df["regime"].value_counts()
    print(f"  Regime distribution:")
    for r in REGIMES:
        n   = regime_counts.get(r, 0)
        b   = df[df["regime"] == r]["next_up"].mean() if n > 0 else float("nan")
        print(f"    {r:>10}: {n:,} hours  baseline={b:.4f}")

    # ── Per-regime tables ──────────────────────────────────────────────────────
    tables    = {}
    baselines = {}
    for regime in REGIMES:
        df_r = df[df["regime"] == regime]
        rb   = df_r["next_up"].mean() if len(df_r) > 0 else baseline_pooled
        baselines[regime] = rb
        print(f"\nBuilding {regime} table (n={len(df_r):,})...")
        table = build_table(df_r, rb)
        tables[regime] = table

        out_path = OUT_DIR / f"composite_calibration_regime_{regime}.json"
        with open(out_path, "w") as f:
            json.dump(table, f, indent=2)
        n_filled = sum(1 for k, v in table.items()
                       if not k.startswith("__") and abs(v - rb) > 1e-6)
        print(f"  → saved  (filled cells: {n_filled}/{(2*TREND_CLIP+1)*(2*REV_CLIP+1)})")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 72)
    lines.append("  REGIME P_UP TABLE DIAGNOSTICS (target: 1h next_up)")
    lines.append("=" * 72)
    lines.append(f"  Test period: {df.index.min().date()} → {df.index.max().date()}")
    lines.append(f"  Pooled baseline: {baseline_pooled:.4f}")

    # A. Calibration curves
    lines.append("\n  A. Calibration curves (decile predicted vs actual next_up):")
    for regime in REGIMES:
        df_r = df[df["regime"] == regime]
        if len(df_r) < 100:
            continue
        cal  = calibration_curve(df_r, tables[regime], baselines[regime])
        lines.append(f"\n  {regime} (n={len(df_r):,}  baseline={baselines[regime]:.4f}):")
        lines.append(f"    {'Decile':>7}  {'pred':>6}  {'actual':>6}  {'n':>6}")
        for _, row in cal.iterrows():
            lines.append(f"    {int(row['decile']):>7}  {row['pred']:.4f}  {row['actual']:.4f}  {int(row['n']):>6}")

    # B. Cross-regime divergence
    lines.append("\n\n  B. Cross-regime divergence |Bull - Bear| (top-15 cells):")
    lines.append(f"    {'tb':>4}  {'rb':>4}  {'Bull':>6}  {'Sideways':>9}  {'Bear':>6}  {'Diverge':>8}")
    lines.append("    " + "-" * 46)
    divs = []
    for tb in range(-TREND_CLIP, TREND_CLIP + 1):
        for rb in range(-REV_CLIP, REV_CLIP + 1):
            key = f"{tb},{rb}"
            vals = {r: tables[r].get(key, baselines[r]) for r in REGIMES}
            div  = abs(vals["Bull"] - vals["Bear"])
            # Only include cells with sufficient data in at least one regime
            max_n = max(
                len(df[(df["regime"] == r) & (df["tb"] == tb) & (df["rb"] == rb)])
                for r in REGIMES
            )
            if max_n >= MIN_N:
                divs.append((tb, rb, vals, div))
    divs.sort(key=lambda x: -x[3])
    for tb, rb, vals, div in divs[:15]:
        lines.append(f"    {tb:>4}  {rb:>4}  "
                     f"{vals['Bull']:>6.3f}  {vals['Sideways']:>9.3f}  "
                     f"{vals['Bear']:>6.3f}  {div:>8.3f}")

    # C. Walk-forward OOS (last 3 months)
    lines.append("\n\n  C. Walk-forward OOS validation (last 3 months):")
    for regime in REGIMES:
        walkforward_validate(df[df["regime"] == regime], regime,
                             baselines[regime], lines)

    # D. Cell occupancy
    lines.append("\n\n  D. Cell occupancy (per regime):")
    n_cells = (2 * TREND_CLIP + 1) * (2 * REV_CLIP + 1)
    for regime in REGIMES:
        df_r    = df[df["regime"] == regime]
        filled  = 0
        for tb in range(-TREND_CLIP, TREND_CLIP + 1):
            for rb in range(-REV_CLIP, REV_CLIP + 1):
                n = len(df_r[(df_r["tb"] == tb) & (df_r["rb"] == rb)])
                if n >= MIN_N:
                    filled += 1
        lines.append(f"    {regime:>10}: {n_cells} cells, {filled} filled (≥{MIN_N}h)")

    diag_text = "\n".join(lines)
    with open(DIAG_PATH, "w") as f:
        f.write(diag_text)
    print(f"\n{diag_text}")
    print(f"\nDiagnostics saved → {DIAG_PATH}")
    print("\nPer-regime tables saved to reform_results/composite_calibration_regime_*.json")


if __name__ == "__main__":
    main()
