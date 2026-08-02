"""Multi-seed × walk-forward test of the '15m live models are stale' hypothesis.
2026-08-02.

Three independent single-fit hints this week pointed at BTC/ETH 15m live
models being stale (pmmom baseline retrains, isotonic REPLICA, ETH Brier) —
but all were the evidence class retracted in feedback_lgbm_single_fit_not_a_
finding. This is the disciplined version, per that memo's protocol:

  - >=3 walk-forward origins (~2wk test windows, retrained fresh at each
    cutoff, no lookahead)
  - >=5 seeds per origin (random_state 0..4), mean ± std reported
  - each asset retrained with its OWN current production architecture:
      BTC: 20-feat LGBM + single-fold prefit isotonic (replicating the
           deployed CalibratedClassifierCV(cv='prefit') method exactly)
      ETH: 162-feat D/S-slope bare LGBMClassifier (dict architecture)
  - scored vs the LIVE model's logged p on the identical window, flat-$100
    net-of-fees book at edge>=0.04

VERDICT STANDARD (pre-registered): 'stale' is confirmed only if the fresh
retrain's seed-mean beats live in ALL origins AND the pooled mean
improvement exceeds the pooled seed std. Anything less = unproven, no
deploy recommendation.

Usage: python3 test_15m_staleness_multiseed.py BTC
       python3 test_15m_staleness_multiseed.py ETH
"""
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from train_15m_pmmom_shadow import load_paper_archive, add_dS_slopes, sim_book

BASE = Path(__file__).parent
SEEDS = [0, 1, 2, 3, 4]
ORIGINS = [
    (pd.Timestamp("2026-06-20", tz="UTC"), pd.Timestamp("2026-07-04", tz="UTC")),
    (pd.Timestamp("2026-07-04", tz="UTC"), pd.Timestamp("2026-07-18", tz="UTC")),
    (pd.Timestamp("2026-07-18", tz="UTC"), pd.Timestamp("2026-08-02", tz="UTC")),
]

BTC_LGBM_KW = dict(n_estimators=150, num_leaves=7, max_depth=4,
                   min_child_samples=40, learning_rate=0.03, reg_alpha=1.0,
                   reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                   verbosity=-1)
# ETH slope-dict model's params (extracted from the deployed pkl at runtime)


def fit_fresh(asset, feats, tr, seed):
    """Retrain the asset's own production architecture on `tr` with `seed`."""
    if asset == "BTC":
        split_t = tr["dt"].quantile(0.75)
        fit_p, cal_p = tr[tr["dt"] < split_t], tr[tr["dt"] >= split_t]
        base = LGBMClassifier(random_state=seed, **BTC_LGBM_KW)
        base.fit(fit_p[feats], fit_p["resolved_yes"].astype(int))
        m = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
        m.fit(cal_p[feats], cal_p["resolved_yes"].astype(int))
        return m
    else:  # ETH: bare LGBM, dict-architecture params
        with open(BASE / "models" / "lgbm_15m_eth.pkl", "rb") as f:
            prod = pickle.load(f)
        params = prod["model"].get_params()
        params.update(random_state=seed, verbosity=-1)
        m = LGBMClassifier(**{k: v for k, v in params.items()
                              if k in LGBMClassifier().get_params()})
        m.fit(tr[feats], tr["resolved_yes"].astype(int))
        return m


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    print(f"=== {asset} 15m staleness test: {len(SEEDS)} seeds x {len(ORIGINS)} origins ===")

    if asset == "BTC":
        with open(BASE / "models" / "lgbm_15m_btc.pkl", "rb") as f:
            prod = pickle.load(f)
        feats = [str(x) for x in prod.calibrated_classifiers_[0].estimator.feature_name_]
        slope_bases = []
    else:
        with open(BASE / "models" / "lgbm_15m_eth.pkl", "rb") as f:
            prod = pickle.load(f)
        feats = list(prod["features"])
        slope_bases = list(prod.get("slope_bases", []))

    df = load_paper_archive(asset)
    if slope_bases:
        df = add_dS_slopes(df, slope_bases)
    for c in set(feats) - set(df.columns):
        df[c] = np.nan
    for c in feats:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if asset == "BTC":
        df = df.dropna(subset=feats)

    rows = []
    for cutoff, test_end in ORIGINS:
        tr = df[df["dt"] < cutoff]
        te = df[(df["dt"] >= cutoff) & (df["dt"] < test_end)].copy()
        if len(te) < 200 or len(tr) < 500:
            print(f"[{cutoff.date()}] skipped (train={len(tr)}, test={len(te)})")
            continue
        lm = te.copy()
        lm["p_live"] = pd.to_numeric(lm["p_model_15m"], errors="coerce")
        lm = lm.dropna(subset=["p_live"])
        live_net = float(sim_book(lm, lm["p_live"].values, 0.04)["pnl"].sum())

        nets = []
        for seed in SEEDS:
            m = fit_fresh(asset, feats, tr, seed)
            p = m.predict_proba(te[feats])[:, 1]
            nets.append(float(sim_book(te, p, 0.04)["pnl"].sum()))
        nets = np.array(nets)
        rows.append({"origin": f"{cutoff.date()}→{test_end.date()}",
                     "live": live_net, "fresh_mean": nets.mean(),
                     "fresh_std": nets.std(), "fresh_min": nets.min(),
                     "fresh_max": nets.max(),
                     "beats_live": nets.mean() > live_net})
        print(f"[{cutoff.date()}→{test_end.date()}] live=${live_net:+,.0f}  "
              f"fresh mean=${nets.mean():+,.0f} ± {nets.std():,.0f} "
              f"(range {nets.min():+,.0f}..{nets.max():+,.0f})  "
              f"{'FRESH BEATS LIVE' if nets.mean() > live_net else 'live holds'}")

    r = pd.DataFrame(rows)
    print("\n=== VERDICT ===")
    all_beat = bool(r["beats_live"].all())
    pooled_improvement = float((r["fresh_mean"] - r["live"]).mean())
    pooled_std = float(r["fresh_std"].mean())
    print(f"fresh beats live in {int(r['beats_live'].sum())}/{len(r)} origins")
    print(f"pooled mean improvement: ${pooled_improvement:+,.0f}  vs pooled seed std: ${pooled_std:,.0f}")
    if all_beat and pooled_improvement > pooled_std:
        print("STALENESS CONFIRMED by pre-registered standard — refresh justified (paper-first).")
    else:
        print("NOT confirmed by pre-registered standard — no deploy recommendation.")


if __name__ == "__main__":
    main()
