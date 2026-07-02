#!/usr/bin/env python3
"""
btc_failure_analysis.py — Why did the direct model fail on BTC test?

Three hypotheses to check:
  H1: Train-test regime shift (BTC test period was trending; train was mean-reverting)
  H2: Model bias toward NO (predicted p_model too low on BTC during test)
  H3: Specific offset range where BTC direct model is worst

Uses the trained direct_model_BTC.pkl and the archive test window.
"""

import math, sys, glob, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from direct_strike_hit_model import (
    load_asset, extract_indicator_values, build_dataset, FEATURE_COLUMNS,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START,
)

MODEL_PATH = Path(__file__).parent / "reform_results" / "direct_model_BTC.pkl"


def main():
    print(f"\n{'='*78}\n  BTC DIRECT MODEL FAILURE ANALYSIS\n{'='*78}", flush=True)
    with open(MODEL_PATH, "rb") as f: pipe = pickle.load(f)

    # Build dataset for BTC train+val+test
    ds = build_dataset("BTC", "BTCUSDT")
    tr_mask = (ds.index >= TRAIN_START) & (ds.index < TRAIN_END)
    va_mask = (ds.index >= VAL_START) & (ds.index < VAL_END)
    te_mask = ds.index >= TEST_START

    for name, m in [("TRAIN", tr_mask), ("VAL", va_mask), ("TEST", te_mask)]:
        sub = ds[m]
        if len(sub) == 0:
            print(f"  [{name}] no data"); continue
        X = sub[FEATURE_COLUMNS].values
        y = sub["target"].values
        p_raw = pipe["clf"].predict_proba(X)[:, 1]
        p_cal = pipe["iso"].predict(p_raw)
        auc_raw = roc_auc_score(y, p_raw) if y.sum() > 0 and y.sum() < len(y) else float("nan")
        print(f"\n  [{name}] n={len(sub):,}  y_mean={y.mean():.3f}  p_mean={p_cal.mean():.3f}  AUC={auc_raw:.4f}", flush=True)
        # Per-offset breakdown
        for off in sorted(sub["offset_pct"].unique()):
            sub_off = sub[sub["offset_pct"] == off]
            y_off = sub_off["target"].values
            p_off = pipe["clf"].predict_proba(sub_off[FEATURE_COLUMNS].values)[:, 1]
            p_off = pipe["iso"].predict(p_off)
            if y_off.sum() == 0 or y_off.sum() == len(y_off):
                au = float("nan")
            else:
                au = roc_auc_score(y_off, p_off)
            bias = p_off.mean() - y_off.mean()
            print(f"    off {off:+.4f}: n={len(sub_off):4d}  y_mean={y_off.mean():.3f}  p_mean={p_off.mean():.3f}  bias={bias:+.3f}  AUC={au:.4f}", flush=True)

    # H1 explicitly: realized direction in each period
    print(f"\n{'='*78}\n  H1: DIRECTIONAL REGIME CHECK\n{'='*78}", flush=True)
    import glob
    f_1h = sorted(glob.glob("data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet"))[-1]
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    close = d_1h["close"]
    lr = np.log(close / close.shift(1))
    for name, start, end in [("TRAIN", TRAIN_START, TRAIN_END), ("VAL", VAL_START, VAL_END), ("TEST", TEST_START, pd.Timestamp.utcnow().tz_convert("UTC"))]:
        m = (lr.index >= start) & (lr.index < end)
        s = lr[m]
        print(f"  [{name}] n={len(s)}  mean_hourly={s.mean()*100:+.4f}%  total_return={(close[m].iloc[-1]/close[m].iloc[0]-1)*100:+.2f}%", flush=True)

    # H3: per-offset PnL contribution during test (replay)
    print(f"\n{'='*78}\n  H3: WHICH OFFSETS COST THE MOST ON TEST?\n{'='*78}", flush=True)
    # Reuse the backtest logic but isolated per strike-distance bucket
    # Instead of rerunning full backtest, just show model predictions vs actuals bucketed by offset
    sub = ds[te_mask]
    print(f"\n  BTC test set  (n={len(sub):,})", flush=True)
    print(f"  {'offset':>8}  {'n':>5}  {'actual_hit':>10}  {'model_p':>9}  {'bias':>8}  {'p<0.5_frac':>11}", flush=True)
    for off in sorted(sub["offset_pct"].unique()):
        sub_off = sub[sub["offset_pct"] == off]
        y_off = sub_off["target"].values
        p_off = pipe["clf"].predict_proba(sub_off[FEATURE_COLUMNS].values)[:, 1]
        p_off = pipe["iso"].predict(p_off)
        p_below = (p_off < 0.5).mean()
        print(f"    {off:+.4f}  {len(sub_off):>5}  {y_off.mean():>10.3f}  {p_off.mean():>9.3f}  {p_off.mean() - y_off.mean():>+8.3f}  {p_below:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
