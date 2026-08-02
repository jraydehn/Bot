"""BTC 15m refresh — the routine retrain justified by the staleness test.
2026-08-02.

Validation already happened (test_15m_staleness_multiseed.py, commit
abba68e: fresh beats live in 2/2 usable origins, pooled +$5,185 > pooled
seed std $3,715, ALL 5 seeds beat live on the recent origin). This script
is therefore a PRODUCTION-STYLE fit, not another experiment: same 20
features, same method as the deployed model (LGBM + single-fold prefit
isotonic on the last 25% by time), trained on ALL archive data through
now. Judged by forward paper as a shadow (p_gbdt column), reviewed with
the other books ~08-11+.

Seed handling: NO seed selection (selection = mini-overfit). Instead a
5-seed SeedEnsemble (mean predict_proba) — lower-variance estimator of
the same model, directly addressing the documented seed-noise problem.

Artifact: dict {model: SeedEnsemble, features: [...]} so the runner's
existing dict-artifact path in compute_p_model_15m serves it via
model_override, logging the RAW ensemble probability unadorned (no KC
shift, no z-expansion) — same convention as the SOL slope shadow.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from train_15m_pmmom_shadow import load_paper_archive
from btc15m_refresh_ensemble import SeedEnsemble

BASE = Path(__file__).parent
SEEDS = [0, 1, 2, 3, 4]

FEATS = ["offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m", "chg_15m",
         "stoch_k_15m", "bp_5m", "body_5m", "dir_5m", "chg_5m", "stoch_k_5m",
         "vol_ratio_5m", "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h",
         "consec_dir_1h", "vol_ratio_1h", "realized_vol_annual"]

LGBM_KW = dict(n_estimators=150, num_leaves=7, max_depth=4,
               min_child_samples=40, learning_rate=0.03, reg_alpha=1.0,
               reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
               verbosity=-1)


def main():
    df = load_paper_archive("BTC")
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATS + ["resolved_yes"]).sort_values("dt").reset_index(drop=True)
    print(f"training rows: {len(df)}  ({df['dt'].min().date()} → {df['dt'].max().date()})")

    split_t = df["dt"].quantile(0.75)
    fit_p = df[df["dt"] < split_t]
    cal_p = df[df["dt"] >= split_t]
    print(f"LGBM fit n={len(fit_p)}  isotonic calibration n={len(cal_p)} "
          f"(split at {split_t.date()})")

    members = []
    for seed in SEEDS:
        base = LGBMClassifier(random_state=seed, **LGBM_KW)
        base.fit(fit_p[FEATS], fit_p["resolved_yes"].astype(int))
        m = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
        m.fit(cal_p[FEATS], cal_p["resolved_yes"].astype(int))
        members.append(m)
    ens = SeedEnsemble(members)

    # sanity only (in-sample on the calibration slice — NOT a validation claim)
    p = ens.predict_proba(cal_p[FEATS])[:, 1]
    y = cal_p["resolved_yes"].values
    x = p - 0.5
    slope = float(np.sum(x * (y - y.mean())) / np.sum(x * x))
    print(f"sanity (in-sample, calibration slice): brier={np.mean((p-y)**2):.4f} "
          f"calib slope={slope:.2f}  p range=[{p.min():.3f},{p.max():.3f}]")

    art = {"model": ens, "features": FEATS,
           "note": "BTC 15m refresh 2026-08-02: 5-seed SeedEnsemble "
                   "(LGBM+prefit isotonic, production method/features), "
                   "trained on full archive thru 08-02; justified by "
                   "test_15m_staleness_multiseed (abba68e); shadow via "
                   "p_gbdt (displaces the stale LGBM's raw stream); "
                   "decisions unchanged"}
    out = BASE / "models" / "lgbm_15m_btc_refresh_20260802.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
