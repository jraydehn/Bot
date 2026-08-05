"""ETH 15m refresh — shadow CHALLENGER for the ETH 15m A/B harness.
2026-08-05.

DISCLOSED EVIDENCE STATUS (different from the BTC refresh): the multi-seed
walk-forward staleness test did NOT confirm staleness for ETH
(test_15m_staleness_multiseed, 08-02 session — fresh did not beat live).
This refresh is therefore NOT justified by a retro win; it exists to give
the ETH 15m shadow A/B a genuine second arm (p_gbdt currently duplicates
production for ETH). Zero-risk shadow: logged only, decisions untouched,
judged purely on its forward paper record. Same recipe as the BTC refresh
(production 20-feat architecture, LGBM + prefit isotonic on last 25% by
time, 5-seed SeedEnsemble, NO seed selection), trained on all ETH archive
through now.
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
    df = load_paper_archive("ETH")
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
           "note": "ETH 15m refresh 2026-08-05: 5-seed SeedEnsemble "
                   "(LGBM+prefit isotonic, production method/features), "
                   "trained on full ETH archive thru 08-05. NOT justified "
                   "by a retro staleness win (ETH test not confirmed) — "
                   "exists as the challenger arm for the ETH 15m shadow "
                   "A/B; judged on forward paper only; decisions unchanged"}
    out = BASE / "models" / "lgbm_15m_eth_refresh_20260805.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
