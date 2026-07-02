#!/usr/bin/env python3
"""
train_btc_gate_meta.py — Train a gate meta-model on BTC blocked_trades.csv.

Question: given the signals at the time a gate fired, was the block correct?

Target: gate_correct = 1 if blocking saved money, 0 if blocking cost money.
  YES blocked → correct if resolved_yes == 0 (YES didn't resolve → saved)
  NO  blocked → correct if resolved_yes == 1 (YES resolved   → NO would have lost)

Deduplicates to first scan per (ticker, gate_name) to avoid scan-loop inflation.
Requires ≥20 resolved rows per gate after dedup; gates below threshold are dropped.

Output: reform_results/btc_gate_meta.pkl
Run:    python3 train_btc_gate_meta.py
"""

import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

BLOCKED_CSV  = Path("results/blocked_trades.csv")
OUTPUT_PATH  = Path("reform_results/btc_gate_meta.pkl")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

MIN_ROWS_PER_GATE = 20

SIGNAL_FEATURES = [
    "pm", "p_model", "net_edge", "offset_pct", "tau_minutes",
    "ema_stack_bias", "composite_trend", "composite_rev", "composite_p_up",
    "stoch_k", "vwap_stretch", "vol_score", "vpin_score", "obi_score",
    "ema_stretch", "structure_bias", "funding_bias",
    "liq_score", "liq_bias", "ls_long_pct", "oi_chg_pct",
]

SEP  = "=" * 72
SEP2 = "-" * 72


def load_data() -> pd.DataFrame:
    df = pd.read_csv(BLOCKED_CSV, low_memory=False)

    # BTC only, resolved only
    df = df[df["asset"] == "BTC"].copy()
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()

    # Deduplicate: keep first scan per (ticker, gate_name)
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df = df.sort_values("logged_at")
    df = df.drop_duplicates(subset=["contract_ticker", "gate_name"], keep="first")

    # Build target: was the block correct?
    side = df["side"].str.strip().str.lower()
    df["gate_correct"] = np.where(
        side == "yes",
        (df["resolved_yes"] == 0).astype(int),   # YES blocked: correct if YES didn't resolve
        (df["resolved_yes"] == 1).astype(int),   # NO  blocked: correct if YES resolved
    )

    # side encoding
    df["side_enc"] = (side == "yes").astype(float)

    # numeric signals
    for col in SIGNAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    return df.sort_values("logged_at").reset_index(drop=True)


def train():
    print(SEP)
    print("  BTC Gate Meta-Model — training")
    print(SEP)

    df = load_data()
    print(f"Loaded {len(df)} deduplicated resolved BTC blocked trades")
    print(f"Date range: {df['logged_at'].iloc[0].date()} → {df['logged_at'].iloc[-1].date()}")
    print()

    gate_counts = Counter(df["gate_name"])
    print("Gate distribution (after dedup):")
    for g, n in gate_counts.most_common():
        correct = df[df["gate_name"] == g]["gate_correct"].mean()
        print(f"  {g:<35} n={n:4d}  acc={correct:.1%}")
    print()

    # Drop gates below minimum threshold
    keep_gates = {g for g, n in gate_counts.items() if n >= MIN_ROWS_PER_GATE}
    df = df[df["gate_name"].isin(keep_gates)].copy()
    print(f"Keeping {len(keep_gates)} gates with ≥{MIN_ROWS_PER_GATE} rows → {len(df)} rows")
    print()

    # Label-encode gate_name
    gate_labels = sorted(keep_gates)
    gate_to_int = {g: i for i, g in enumerate(gate_labels)}
    df["gate_enc"] = df["gate_name"].map(gate_to_int).astype(float)

    feature_cols = ["gate_enc", "side_enc"] + SIGNAL_FEATURES
    X = df[feature_cols].values.astype(float)
    y = df["gate_correct"].values.astype(int)

    # Chronological 60/20/20 split
    n = len(df)
    i_val  = int(n * 0.60)
    i_test = int(n * 0.80)
    X_tr, y_tr = X[:i_val],  y[:i_val]
    X_val, y_val = X[i_val:i_test], y[i_val:i_test]
    X_te,  y_te  = X[i_test:],      y[i_test:]
    print(f"Split — train: {len(y_tr)}  val: {len(y_val)}  test: {len(y_te)}")
    print()

    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        _using_lgbm = True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_depth=4,
            min_samples_leaf=10, random_state=42,
        )
        _using_lgbm = False

    clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_val)[:, 1]
    p_te  = clf.predict_proba(X_te)[:, 1]
    auc_val = roc_auc_score(y_val, p_val)
    auc_te  = roc_auc_score(y_te,  p_te)
    print(f"Val AUC: {auc_val:.4f}   Test AUC: {auc_te:.4f}")
    print()

    # Per-gate accuracy on test set
    print(SEP2)
    print("Per-gate: model p_correct vs actual on test set:")
    df_te = df.iloc[i_test:].copy()
    df_te["p_correct"] = p_te
    for gate in gate_labels:
        sub = df_te[df_te["gate_name"] == gate]
        if len(sub) < 5:
            continue
        actual = sub["gate_correct"].mean()
        pred   = sub["p_correct"].mean()
        print(f"  {gate:<35} n={len(sub):4d}  actual={actual:.1%}  pred={pred:.1%}")
    print()

    if _using_lgbm:
        imp = clf.feature_importances_
        total = imp.sum() or 1
        ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
        print("Feature importances (gain):")
        for feat, score in ranked[:12]:
            print(f"  {feat:<28} {score/total:.1%}")
        print()

    pipe = {
        "features":     feature_cols,
        "gate_to_int":  gate_to_int,
        "gate_labels":  gate_labels,
        "clf":          clf,
        "auc_val":      auc_val,
        "auc_te":       auc_te,
        "n_train":      len(y_tr),
        "n_val":        len(y_val),
        "n_test":       len(y_te),
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved → {OUTPUT_PATH}")
    print(SEP)


if __name__ == "__main__":
    train()
