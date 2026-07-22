"""
train_15m_real_archive.py
--------------------------
Train the production 15m LGBM model directly on Kalshi's real settled-market
archive (results/{asset}_real_archive_15m_backfill.csv, built by
backfill_real_archive_15m.py) instead of build_15m_model.py's synthetic
contracts.

Why: real Kalshi 15m markets are listed almost exactly at-the-money (one
strike per 15-min window, not a ladder), so the population the live runner
actually trades against is dominated by near-coin-flip cases the synthetic
generator barely covers. Walk-forward cross-validation (see
project_15m_real_archive_retrain memory) showed this real-archive-trained
model beats the synthetic-trained production model on real held-out data:
BTC profitable in 4/4 folds vs production's 0/4; SOL 4/4 vs 2/4; ETH mixed
(higher total PnL but only 3/4 folds, vs production's 4/4) -- deployed
2026-07-22 for all three assets after that validation.

Conservative LGBM hyperparameters are used deliberately: the real archive is
only ~5,500 rows per asset (~58 days, the limit of Kalshi's candlestick
retention), far smaller than the 500k+ row synthetic set, so a full-depth
model would overfit.

Usage: python3 train_15m_real_archive.py BTC ETH SOL
"""
import sys, pickle
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURE_COLS = [
    "offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m", "vol_ratio_5m",
    "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h", "consec_dir_1h", "vol_ratio_1h",
    "realized_vol_annual",
]


def train_final(asset: str) -> None:
    print(f"=== {asset}: training final real-archive model ===")
    df = pd.read_csv(f"results/{asset.lower()}_real_archive_15m_backfill.csv", low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    df = df.sort_values("logged_at").reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + ["resolved_yes"])
    n = len(df)
    i_val = int(n * 0.85)
    train, val = df.iloc[:i_val], df.iloc[i_val:]
    print(f"  n={n}  train={len(train)}  val(calib)={len(val)}")

    clf = LGBMClassifier(
        n_estimators=150, num_leaves=7, max_depth=4,
        min_child_samples=40, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0,
        verbosity=-1,
    )
    clf.fit(train[FEATURE_COLS], train["resolved_yes"])
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(val[FEATURE_COLS], val["resolved_yes"])

    p_val = cal.predict_proba(val[FEATURE_COLS])[:, 1]
    print(f"  val AUC={roc_auc_score(val['resolved_yes'], p_val):.4f}  "
          f"Brier={brier_score_loss(val['resolved_yes'], p_val):.4f}")

    out_path = f"models/lgbm_15m_{asset.lower()}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(cal, f)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    assets = sys.argv[1:] or ["BTC", "ETH", "SOL"]
    for a in assets:
        train_final(a.upper())
