"""Offline test: does proper cv=5 isotonic calibration fix the SOL/BTC 15m
compression/over-confidence found in project_isotonic_compression_20260731?
2026-07-31.

PURE OFFLINE TEST. Does not touch, wire, or restart anything live. SOL
production is explicitly off-limits for changes right now (it's working) —
this only asks whether the calibration-methodology hypothesis holds up.

Confirmed root cause: both SOL and BTC's production models are
CalibratedClassifierCV(LGBMClassifier(same hyperparams), cv='prefit',
method='isotonic') — a SINGLE calibration fold, not proper k-fold CV. SOL's
fold happened to undercorrect (compressed/under-confident, slope ~2);
BTC's happened to overcorrect (over-confident, slope ~0.15-0.45).

Three variants per asset, IDENTICAL base LGBM hyperparameters throughout
(n_estimators=150, num_leaves=7, max_depth=4, min_child_samples=40,
learning_rate=0.03, reg_alpha=1.0, reg_lambda=1.0, subsample/colsample=0.8
— extracted live from the deployed pkls), varying ONLY the calibration
methodology:
  (a) LIVE  — the actual deployed pkl, scored via its logged p_model_15m
  (b) REPLICA — same architecture, single prefit calibration fold (mirrors
      production's apparent methodology, refit on fresher data)
  (c) CV5   — same architecture, cv=5 internal calibration (the proposed
      fix: sklearn refits+calibrates across 5 folds and averages)

Train<07-16, single frozen holdout 07-16..07-31 (matches this project's
established discipline). Scored by calibration slope AND flat-$100
net-of-fees PnL — PnL is the bar.

Usage: python3 test_isotonic_cv_refit_15m.py SOL
       python3 test_isotonic_cv_refit_15m.py BTC
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

BASE = Path(__file__).parent
TRAIN_END = pd.Timestamp("2026-07-16", tz="UTC")

FEATS = ["offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m",
         "stoch_k_15m", "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m",
         "vol_ratio_5m", "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
         "consec_dir_1h", "vol_ratio_1h", "realized_vol_annual"]

LGBM_KW = dict(n_estimators=150, num_leaves=7, max_depth=4,
               min_child_samples=40, learning_rate=0.03, reg_alpha=1.0,
               reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
               verbosity=-1)


def calib_slope(p: np.ndarray, y: np.ndarray) -> float:
    x = p - 0.5
    return float(np.sum(x * (y - y.mean())) / np.sum(x * x))


def sim_book(df: pd.DataFrame, p: np.ndarray, edge_min: float) -> pd.DataFrame:
    s = df.copy()
    s["p"] = p
    fee = 0.07 * s["p_market"] * (1 - s["p_market"])
    ey = s["p"] - s["p_market"] - fee
    en = s["p_market"] - s["p"] - fee
    s["side"] = np.where(ey >= en, "yes", "no")
    s["edge"] = np.maximum(ey, en)
    q = s[s["edge"] >= edge_min].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first")
    cost = np.where(q["side"] == "yes", q["p_market"], 1 - q["p_market"])
    win = np.where(q["side"] == "yes", q["resolved_yes"] == 1, q["resolved_yes"] == 0)
    feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
    pnl = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * feeq
    q = q.copy()
    q["pnl"], q["win"] = pnl, win
    return q


def summarize(q: pd.DataFrame, label: str) -> str:
    if not len(q):
        return f"{label}: n=0"
    wk = q.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
    wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
    return f"{label}: n={len(q)} net=${q['pnl'].sum():+,.0f} WR={q['win'].mean():.1%} | {wks}"


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 else "SOL").upper()
    print(f"=== {asset} 15m: single-fold prefit vs cv=5 isotonic calibration ===")

    df = pd.read_csv(BASE / "results" / f"paper_trades_{asset.lower()}15m.csv",
                     low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df["p_market"] = pd.to_numeric(df["p_market"], errors="coerce")
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATS + ["resolved_yes", "p_market", "dt"]).sort_values("dt")

    tr = df[df["dt"] < TRAIN_END].reset_index(drop=True)
    te = df[df["dt"] >= TRAIN_END].reset_index(drop=True)
    print(f"train n={len(tr)}  holdout n={len(te)} "
          f"({te['dt'].min().date()} → {te['dt'].max().date()})")

    # (b) REPLICA: single prefit fold, time-split 75/25 within train
    split_t = tr["dt"].quantile(0.75)
    fit_part = tr[tr["dt"] < split_t]
    cal_part = tr[tr["dt"] >= split_t]
    base_b = LGBMClassifier(**LGBM_KW).fit(fit_part[FEATS], fit_part["resolved_yes"].astype(int))
    model_b = CalibratedClassifierCV(base_b, cv="prefit", method="isotonic")
    model_b.fit(cal_part[FEATS], cal_part["resolved_yes"].astype(int))

    # (c) CV5: proper internal 5-fold refit + calibrate on the FULL train set
    model_c = CalibratedClassifierCV(LGBMClassifier(**LGBM_KW), cv=5, method="isotonic")
    model_c.fit(tr[FEATS], tr["resolved_yes"].astype(int))

    y_te = te["resolved_yes"].values
    p_live = pd.to_numeric(te["p_model_15m"], errors="coerce").values
    p_b = model_b.predict_proba(te[FEATS])[:, 1]
    p_c = model_c.predict_proba(te[FEATS])[:, 1]

    print(f"\nfolds — REPLICA: {len(model_b.calibrated_classifiers_)}  "
          f"CV5: {len(model_c.calibrated_classifiers_)}")
    print(f"\ncalibration slope on holdout (1.0 = perfect; >1 under-confident; "
          f"<1 over-confident):")
    ok = ~np.isnan(p_live)
    print(f"  LIVE:    {calib_slope(p_live[ok], y_te[ok]):.2f}  (n={ok.sum()})")
    print(f"  REPLICA: {calib_slope(p_b, y_te):.2f}")
    print(f"  CV5:     {calib_slope(p_c, y_te):.2f}")

    print(f"\nBrier: LIVE={np.nanmean((p_live-y_te)**2):.4f}  "
          f"REPLICA={np.mean((p_b-y_te)**2):.4f}  CV5={np.mean((p_c-y_te)**2):.4f}")

    print("\nflat-$100 net-of-fees book, single frozen holdout:")
    te_live = te[ok].copy()
    for em in [0.04, 0.06]:
        print("  ", summarize(sim_book(te_live, p_live[ok], em), f"LIVE    edge>={em}"))
        print("  ", summarize(sim_book(te, p_b, em), f"REPLICA edge>={em}"))
        print("  ", summarize(sim_book(te, p_c, em), f"CV5     edge>={em}"))


if __name__ == "__main__":
    main()
