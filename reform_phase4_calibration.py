#!/usr/bin/env python3
"""
reform_phase4_calibration.py — Phase 4: verify + rescale calibration of the D_hybrid model.

Trains the winning D_hybrid L1 logistic regression on TRAIN only (2025),
predicts on VAL (Jan-Mar 2026), checks bin-level calibration: does predicted
p = 0.60 mean ~60% observed up rate?

If biased, fit an isotonic regression calibrator on VAL predictions to correct.
Isotonic is non-parametric and monotone — preserves ordering while mapping
predicted probs onto observed rates.

Saves final model + calibrator per asset to reform_results/phase4_{asset}.pkl.
Locks pipeline so Phase 5 can load and replay.

TEST set still not touched.
"""

import math, sys, glob, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

# Import Phase 3's feature extractor
sys.path.insert(0, str(Path(__file__).parent))
from reform_phase3_score import (
    load_asset, extract_features, compute_targets, build_variant_D,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
)

OUT_DIR = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

# Per-asset best C from Phase 3 results
PER_ASSET_C = {"BTC": 0.05, "ETH": 0.05, "SOL": 5.0}


def fit_model(asset, sym, btc_close_1h):
    print(f"\n{'='*78}\n  [{asset}] PHASE 4 — calibration audit + isotonic rescale\n{'='*78}", flush=True)
    t0 = time.time()
    d_1m, d_15m, d_1h, d_4h, d_1d = load_asset(sym)
    btc = btc_close_1h if asset != "BTC" else None
    X = extract_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=btc)
    T = compute_targets(d_1h)
    X_D = build_variant_D(X)

    tr_mask = (X_D.index >= TRAIN_START) & (X_D.index < TRAIN_END)
    va_mask = (X_D.index >= VAL_START) & (X_D.index < VAL_END)
    Xtr = X_D[tr_mask].dropna()
    Xva = X_D[va_mask].dropna()
    ytr = T.loc[Xtr.index, "next_up"]
    yva = T.loc[Xva.index, "next_up"]

    # Fit L1 logistic regression
    scaler = StandardScaler()
    Xt = scaler.fit_transform(Xtr)
    Xv = scaler.transform(Xva)
    C = PER_ASSET_C[asset]
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=1000)
    clf.fit(Xt, ytr)
    p_raw_va = clf.predict_proba(Xv)[:, 1]

    auc_raw = roc_auc_score(yva, p_raw_va)
    ll_raw = log_loss(yva, p_raw_va, labels=[0, 1])
    print(f"  Raw val AUC: {auc_raw:.4f}  log_loss: {ll_raw:.4f}", flush=True)

    # Calibration check — bin raw predictions vs observed
    bins = np.array([0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
    print(f"\n  Raw calibration (before isotonic):", flush=True)
    print(f"  {'bin':>12}  {'n':>5}  {'avg_p':>7}  {'obs':>7}  {'bias':>8}", flush=True)
    raw_biases = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        m = (p_raw_va > lo) & (p_raw_va <= hi)
        n = m.sum()
        if n < 30: continue
        avg = p_raw_va[m].mean()
        obs = yva.values[m].mean()
        bias = obs - avg
        raw_biases.append(bias)
        flag = "  *" if abs(bias) > 0.05 else ""
        print(f"  ({lo:.2f}, {hi:.2f}]  {n:>5}  {avg:>7.3f}  {obs:>7.3f}  {bias:>+8.3f}{flag}", flush=True)
    max_raw_bias = max((abs(b) for b in raw_biases), default=0.0)

    # Fit isotonic regression on val to correct calibration
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_raw_va, yva.values)
    p_cal_va = iso.predict(p_raw_va)

    auc_cal = roc_auc_score(yva, p_cal_va)
    ll_cal = log_loss(yva, np.clip(p_cal_va, 1e-6, 1-1e-6), labels=[0, 1])
    print(f"\n  Calibrated val AUC: {auc_cal:.4f}  log_loss: {ll_cal:.4f}", flush=True)

    print(f"\n  Calibrated bins (after isotonic):", flush=True)
    print(f"  {'bin':>12}  {'n':>5}  {'avg_p':>7}  {'obs':>7}  {'bias':>8}", flush=True)
    cal_biases = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        m = (p_cal_va > lo) & (p_cal_va <= hi)
        n = m.sum()
        if n < 30: continue
        avg = p_cal_va[m].mean()
        obs = yva.values[m].mean()
        bias = obs - avg
        cal_biases.append(bias)
        flag = "  *" if abs(bias) > 0.05 else ""
        print(f"  ({lo:.2f}, {hi:.2f}]  {n:>5}  {avg:>7.3f}  {obs:>7.3f}  {bias:>+8.3f}{flag}", flush=True)
    max_cal_bias = max((abs(b) for b in cal_biases), default=0.0)

    print(f"\n  Max |bias|: raw={max_raw_bias:.3f}  calibrated={max_cal_bias:.3f}", flush=True)

    # Save pipeline
    pipeline = {
        "asset": asset,
        "feature_columns": list(Xtr.columns),
        "scaler": scaler,
        "clf": clf,
        "isotonic": iso,
        "train_auc": roc_auc_score(ytr, clf.predict_proba(Xt)[:, 1]),
        "val_auc": auc_cal,
        "val_log_loss": ll_cal,
        "max_raw_bias": max_raw_bias,
        "max_cal_bias": max_cal_bias,
        "coefficients": dict(zip(Xtr.columns, clf.coef_[0])),
    }
    out_path = OUT_DIR / f"phase4_{asset}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Saved pipeline → {out_path}", flush=True)
    print(f"  [{asset}] done in {time.time()-t0:.1f}s", flush=True)
    return pipeline


def main():
    _, _, btc_1h, _, _ = load_asset("BTCUSDT")
    btc_close_1h = btc_1h["close"]

    summary = []
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        p = fit_model(asset, sym, btc_close_1h)
        summary.append({"asset":asset, "train_auc":p["train_auc"], "val_auc":p["val_auc"],
                        "val_log_loss":p["val_log_loss"],
                        "max_raw_bias":p["max_raw_bias"], "max_cal_bias":p["max_cal_bias"]})

    print(f"\n{'='*78}\n  PHASE 4 SUMMARY\n{'='*78}", flush=True)
    print(f"  {'asset':<6} {'train_AUC':>10} {'val_AUC':>10} {'val_LL':>10} {'raw_bias':>10} {'cal_bias':>10}", flush=True)
    for s in summary:
        gap = s["train_auc"] - s["val_auc"]
        print(f"  {s['asset']:<6} {s['train_auc']:>10.4f} {s['val_auc']:>10.4f} {s['val_log_loss']:>10.4f} {s['max_raw_bias']:>10.3f} {s['max_cal_bias']:>10.3f}   (gap={gap:+.3f})", flush=True)


if __name__ == "__main__":
    main()
