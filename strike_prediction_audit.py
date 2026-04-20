#!/usr/bin/env python3
"""
strike_prediction_audit.py — Which indicators predict strike HITS at various offsets?

For each hourly bar in 2025-2026:
  - spot = close at bar open
  - For each offset in the asset's strike grid:
      strike = spot * (1 + offset)
      y = 1 if next-hour close > strike (for YES-side strikes above spot)
          or 1 if next-hour close < strike (for NO-side strikes below spot)
  - For each indicator and the composite p_up:
      AUC of indicator value as predictor of y

Produces an indicator × offset matrix of AUCs per asset. Answers:
  - Does composite p_up predict strike hits, or just direction?
  - Which indicators dominate at which offsets?
  - Does predictive power decay with offset magnitude?

Uses full 2025-01-01 → 2026-04-19 OHLCV (in-sample for current composite cal;
flag disclosed in report).
"""

import math, sys, glob, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import (
    compute_scores, lookup_p_up, _stoch_k, _rsi, _atr, _bb_pct, _keltner_pct,
    _wpr, _macd_cross, _vol_signal_4h, _dc_pct,
)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

# Asset-specific strike offset grids (matching real Kalshi increments)
ASSET_OFFSETS = {
    "BTC": [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],   # in %
    "ETH": [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "SOL": [-5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 5.0],
}


def load_asset(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h


def extract_indicators(d_1m, d_15m, d_1h, d_4h):
    """Extract continuous values of current composite indicators + composite_p_up."""
    idx = d_1h.index
    out = pd.DataFrame(index=idx)

    # 4h trend indicators
    out["trend_stoch_4h"]    = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    out["trend_bb_4h"]       = _bb_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20).reindex(idx, method="ffill")
    kc4_pct, _, _            = _keltner_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20, 2)
    out["trend_keltner_4h"]  = kc4_pct.reindex(idx, method="ffill")
    out["trend_wpr_4h"]      = _wpr(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    # MACD crossover as signed ordinal: crossed_up=+2, up_lag=+1, crossed_down=-2, down_lag=-1
    macd_st_4h = _macd_cross(d_4h["close"])
    macd_ord = macd_st_4h.map({"crossed_up":2, "up_lag":1, "none":0, "down_lag":-1, "crossed_down":-2}).fillna(0)
    out["trend_macd_4h"]     = macd_ord.reindex(idx, method="ffill")
    vsig_4h = _vol_signal_4h(d_4h["close"], d_4h["volume"]).map(
        {"high_vol_up":1, "avg":0, "low_vol":0, "high_vol_down":-1}).fillna(0)
    out["trend_vol_4h"]      = vsig_4h.reindex(idx, method="ffill")

    # Reversion indicators
    out["rev_rsi_1h"]    = _rsi(d_1h["close"], 14)
    out["rev_rsi_4h"]    = _rsi(d_4h["close"], 14).reindex(idx, method="ffill")
    out["rev_stoch_15m"] = _stoch_k(d_15m["high"], d_15m["low"], d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_stoch_1h"]  = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    kc15_pct, _, _       = _keltner_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20, 2)
    out["rev_keltner_15m"] = kc15_pct.resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_dc_15m"]    = _dc_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_wpr_1h"]    = _wpr(d_1h["high"], d_1h["low"], d_1h["close"], 14)

    # Composite — the actual p_up produced by the current system
    # Build vwap for composite scores (use 1m close/volume)
    trend_s, rev_s = compute_scores(
        d_1h["close"], d_1h["high"], d_1h["low"], d_1h["volume"],
        d_4h["close"], d_4h["high"], d_4h["low"], d_4h["volume"],
        d_15m["close"], d_15m["high"], d_15m["low"],
        d_1m["close"], d_1m["volume"], idx,
    )
    out["composite_trend"] = trend_s.astype(int)
    out["composite_rev"]   = rev_s.astype(int)
    return out, trend_s, rev_s


def auc_binary(feature, target):
    """Rank-based AUC. Handles NaNs."""
    m = feature.notna() & target.notna()
    if m.sum() < 200: return np.nan
    f = feature[m].values; t = target[m].values
    n_pos = int(t.sum()); n_neg = len(t) - n_pos
    if n_pos < 20 or n_neg < 20: return np.nan
    ranks = rankdata(f)
    return (ranks[t == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def audit_asset(asset, sym):
    print(f"\n{'='*78}\n  [{asset}] STRIKE PREDICTION AUDIT\n{'='*78}", flush=True)
    t0 = time.time()
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    idx = d_1h.index

    # Filter to 2025-present (in-sample for current cal, but we're auditing the SYSTEM as-deployed)
    mask = idx >= pd.Timestamp("2025-01-01", tz="UTC")
    print(f"  Rows evaluated: {mask.sum():,}", flush=True)

    print(f"  Extracting indicators...", flush=True)
    features, trend_s, rev_s = extract_indicators(d_1m, d_15m, d_1h, d_4h)

    # Composite p_up per bar
    p_up_series = pd.Series(
        [lookup_p_up(int(t), int(r), asset=asset) for t, r in zip(trend_s.values, rev_s.values)],
        index=idx,
    )
    features["composite_p_up"] = p_up_series

    close = d_1h["close"]
    next_close = close.shift(-1)

    # Build binary targets per offset
    offsets = ASSET_OFFSETS[asset]
    print(f"  Offsets ({len(offsets)}): {offsets}", flush=True)

    # Baseline hit rates (unconditional P(strike hit) at each offset)
    base_rates = {}
    for off in offsets:
        off_frac = off / 100.0
        if off > 0:
            # Strike above spot: y = 1 if close goes ABOVE strike (YES-side bet wins)
            y = (next_close > close * (1 + off_frac)).astype(int)
        else:
            # Strike below spot: y = 1 if close goes BELOW strike (NO-side bet wins if strike > close)
            # For symmetry, flip: y = 1 if close < strike (hit direction is down-through)
            y = (next_close < close * (1 + off_frac)).astype(int)
        y = y[mask]
        base_rates[off] = y.mean()

    # Build AUC matrix: rows = indicators, cols = offsets
    ind_names = list(features.columns)
    # For "hit" targets:
    # - offset > 0 (YES-side strikes): close > strike = upward move, so indicator with positive link to up should have AUC>0.5
    # - offset < 0 (NO-side strikes): close < strike = downward move, so indicator with negative link to up should have AUC>0.5
    # To make AUCs comparable across offsets, we use the signed "hit = favorable move" definition.

    aucs = pd.DataFrame(index=ind_names, columns=[f"{o:+.2f}%" for o in offsets], dtype=float)
    for off in offsets:
        off_frac = off / 100.0
        if off > 0:
            y = (next_close > close * (1 + off_frac)).astype(int)
        else:
            y = (next_close < close * (1 + off_frac)).astype(int)
        y = y[mask]
        for feat in ind_names:
            f_vals = features[feat][mask]
            # For offset<0, we want AUC of predicting down-move, so higher indicator → lower target;
            # we flip sign of feature values so AUCs are symmetric around 0.5 for "direction of move"
            if off < 0:
                f_vals = -f_vals
            au = auc_binary(f_vals, y)
            aucs.at[feat, f"{off:+.2f}%"] = au

    # Report
    print(f"\n  Baseline hit rates (unconditional P(strike hit)):", flush=True)
    for off in offsets:
        print(f"    offset {off:+.2f}%:  {base_rates[off]:.3f}", flush=True)

    print(f"\n  AUC matrix (all values are AUC of indicator predicting strike hit in favorable direction;", flush=True)
    print(f"   0.50=chance, >0.55=real signal, >0.60=strong). offset sign convention: +→up hit; -→down hit.", flush=True)
    print(f"\n  {'indicator':<22} " + " ".join(f"{col:>7}" for col in aucs.columns), flush=True)
    print(f"  {'-'*22} " + " ".join("-"*7 for _ in aucs.columns), flush=True)
    for feat in ind_names:
        row = aucs.loc[feat]
        vals = " ".join(f"{v:>7.3f}" if pd.notna(v) else f"{'---':>7}" for v in row.values)
        print(f"  {feat:<22} {vals}", flush=True)

    # Summary: which indicators carry predictive signal at which offsets?
    print(f"\n  TOP 3 predictors per offset (AUC):", flush=True)
    for col in aucs.columns:
        ranked = aucs[col].dropna().sort_values(ascending=False).head(3)
        tops = ", ".join(f"{feat}={au:.3f}" for feat, au in ranked.items())
        print(f"    {col:>7}:  {tops}", flush=True)

    # Save matrix
    aucs.to_csv(OUT_DIR / f"strike_auc_{asset}.csv")
    print(f"\n  Saved matrix → strike_auc_{asset}.csv", flush=True)
    print(f"  [{asset}] done in {time.time()-t0:.1f}s", flush=True)
    return aucs, base_rates


def main():
    all_aucs = {}
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        aucs, base_rates = audit_asset(asset, sym)
        all_aucs[asset] = (aucs, base_rates)

    # Key question: does composite_p_up's AUC degrade as offset magnitude grows?
    print(f"\n{'='*78}\n  CRITICAL QUESTION: does composite_p_up predict strike hits at non-zero offsets?\n{'='*78}", flush=True)
    for asset, (aucs, _) in all_aucs.items():
        pup = aucs.loc["composite_p_up"]
        print(f"\n  [{asset}] composite_p_up AUC by offset:", flush=True)
        for col, v in pup.items():
            marker = "★" if pd.notna(v) and v > 0.55 else ""
            print(f"    {col:>7}:  {v:.3f}  {marker}", flush=True)


if __name__ == "__main__":
    main()
