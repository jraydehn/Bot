"""Walk-forward robustness check: does pm-momentum's edge over baseline, and
does 'fresh retrain beats stale live', hold across MULTIPLE independent
origins — or was the single 07-16 holdout just one lucky/unlucky window
sitting adjacent to the retrain cutoff (fitting-to-regime)? 2026-07-31.

User's concern is legitimate: baseline vs +pmmom is freshness-CONTROLLED
(identical cutoff, only the feature differs) so that comparison was never
vulnerable to this. But "any fresh retrain beats the static live model" IS
freshness-confounded — a model trained closer to its test window has an
unfair informational edge that may not persist walk-forward. This tests
BOTH claims across 3 non-overlapping origins instead of 1.

Origins (BTC archive spans late May -> July 31):
  O1: train<06-20, test 06-20..07-02
  O2: train<07-02, test 07-02..07-16
  O3: train<07-16, test 07-16..07-31  (the original single-shot window)

At each origin, baseline and +pmmom are retrained FRESH using only data
before that origin's cutoff (genuine walk-forward, no lookahead). LIVE is
scored on the same window using its logged p_model_15m — it's a single
static artifact, not re-fit per origin, so it's the right control for
"does ANY given fresh retrain beat whatever was deployed", not a perfectly
matched comparison, and that limitation is reported as-is.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from train_15m_pmmom_shadow import (
    load_paper_archive, add_pm_momentum, sim_book, summarize, LGBM_KW,
)
import pickle

ORIGINS = [
    (pd.Timestamp("2026-06-20", tz="UTC"), pd.Timestamp("2026-07-02", tz="UTC")),
    (pd.Timestamp("2026-07-02", tz="UTC"), pd.Timestamp("2026-07-16", tz="UTC")),
    (pd.Timestamp("2026-07-16", tz="UTC"), pd.Timestamp("2026-07-31", tz="UTC")),
]


def main():
    with open("models/lgbm_15m_btc.pkl", "rb") as f:
        prod = pickle.load(f)
    est = prod.calibrated_classifiers_[0].estimator
    base_feats = [str(x) for x in est.feature_name_]

    df = load_paper_archive("BTC")
    df = add_pm_momentum(df, "BTC")
    for c in set(base_feats) - set(df.columns):
        df[c] = np.nan
    pmmom_feats = base_feats + ["pm_chg_5m", "pm_chg_10m", "pm_n_obs"]

    results = []
    for cutoff, test_end in ORIGINS:
        tr = df[df["dt"] < cutoff]
        te = df[(df["dt"] >= cutoff) & (df["dt"] < test_end)]
        if len(te) < 100:
            print(f"[{cutoff.date()}→{test_end.date()}] skipped, only {len(te)} rows")
            continue
        val_cut = cutoff - pd.Timedelta(days=5)
        row = {"origin": f"{cutoff.date()}→{test_end.date()}", "n_test": len(te)}
        print(f"\n=== origin {row['origin']}  (train n={len(tr)}, test n={len(te)}) ===")

        for tag, feats in [("baseline", base_feats), ("pmmom", pmmom_feats)]:
            tr_fit = tr[tr["dt"] < val_cut]
            tr_val = tr[tr["dt"] >= val_cut]
            m = LGBMClassifier(**LGBM_KW)
            m.fit(tr_fit[feats], tr_fit["resolved_yes"].astype(int),
                  eval_set=[(tr_val[feats], tr_val["resolved_yes"].astype(int))],
                  callbacks=[early_stopping(30, verbose=False)])
            p = m.predict_proba(te[feats])[:, 1]
            bk = sim_book(te, p, 0.04)
            row[f"{tag}_net"] = float(bk["pnl"].sum())
            row[f"{tag}_n"] = len(bk)
            print("  ", summarize(bk, tag))

        lm = te.dropna(subset=["p_market"]).copy()
        lm["p_model_15m"] = pd.to_numeric(lm["p_model_15m"], errors="coerce")
        lm = lm.dropna(subset=["p_model_15m"])
        bk_live = sim_book(lm, lm["p_model_15m"].values, 0.04)
        row["live_net"] = float(bk_live["pnl"].sum())
        row["live_n"] = len(bk_live)
        print("  ", summarize(bk_live, "live"))
        results.append(row)

    print("\n=== SUMMARY across origins ===")
    r = pd.DataFrame(results)
    print(r[["origin", "live_net", "baseline_net", "pmmom_net"]].round(0).to_string(index=False))
    print(f"\npmmom beats baseline in {(r['pmmom_net'] > r['baseline_net']).sum()}/{len(r)} origins")
    print(f"any fresh retrain (best of baseline/pmmom) beats live in "
          f"{(r[['baseline_net','pmmom_net']].max(axis=1) > r['live_net']).sum()}/{len(r)} origins")
    print(f"pmmom specifically beats live in {(r['pmmom_net'] > r['live_net']).sum()}/{len(r)} origins")


if __name__ == "__main__":
    main()
