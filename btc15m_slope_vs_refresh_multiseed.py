"""BTC 15m: does the SOL-shadow/ETH-production SLOPE recipe beat the
20-feat refresh architecture? Multi-seed × walk-forward. 2026-08-02.

Motivated by the SOL 15m slope shadow being the best-performing candidate
in forward paper (gated+kelly all-days-green) and ETH production already
running the same recipe. BTC is the one 15m book still on the legacy
20-feat architecture (today's refresh kept it deliberately — routine
refresh first, architecture question second. This is the architecture
question, done to the feedback_lgbm_single_fit_not_a_finding standard.)

Variants at each of 3 origins × 5 seeds, flat-$100 net-of-fees edge>=.04:
  LIVE    — deployed model's logged p (static artifact, staleness control)
  REFRESH — 20-feat LGBM + prefit isotonic (today's refresh architecture)
  SLOPE   — the SOL/ETH recipe: ETH's production feature list (generic 15m
            schema; 148 statics + D/S slopes over 14 scan-level bases at
            15/45/120min) as a bare LGBM dict-architecture, trained on
            BTC's own paper archive

Pre-registered standard: SLOPE must beat REFRESH in ALL origins with
pooled improvement > pooled seed std to justify swapping the shadow slot.
"""
import pickle
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

REFRESH_FEATS = ["offset_pct", "z_score", "bp_15m", "body_15m", "dir_15m",
                 "chg_15m", "stoch_k_15m", "bp_5m", "body_5m", "dir_5m",
                 "chg_5m", "stoch_k_5m", "vol_ratio_5m", "chg_1h", "bp_1h",
                 "stoch_k_1h", "ema_bias_1h", "consec_dir_1h", "vol_ratio_1h",
                 "realized_vol_annual"]
REFRESH_KW = dict(n_estimators=150, num_leaves=7, max_depth=4,
                  min_child_samples=40, learning_rate=0.03, reg_alpha=1.0,
                  reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                  verbosity=-1)


def main():
    with open(BASE / "models" / "lgbm_15m_eth.pkl", "rb") as f:
        eth_prod = pickle.load(f)
    slope_feats = list(eth_prod["features"])
    slope_bases = list(eth_prod["slope_bases"])
    slope_kw = {k: v for k, v in eth_prod["model"].get_params().items()
                if k in LGBMClassifier().get_params()}
    slope_kw["verbosity"] = -1

    df = load_paper_archive("BTC")
    df = add_dS_slopes(df, slope_bases)
    for c in set(slope_feats) | set(REFRESH_FEATS):
        if c not in df.columns:
            df[c] = np.nan
        elif df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df_r = df.dropna(subset=REFRESH_FEATS + ["resolved_yes"])

    rows = []
    for cutoff, test_end in ORIGINS:
        te = df[(df["dt"] >= cutoff) & (df["dt"] < test_end)].copy()
        te_r = df_r[(df_r["dt"] >= cutoff) & (df_r["dt"] < test_end)].copy()
        tr = df[df["dt"] < cutoff]
        tr_r = df_r[df_r["dt"] < cutoff]
        if len(tr_r) < 500 or len(te_r) < 200:
            print(f"[{cutoff.date()}] skipped (refresh train={len(tr_r)})")
            continue
        lm = te.copy()
        lm["p_live"] = pd.to_numeric(lm["p_model_15m"], errors="coerce")
        lm = lm.dropna(subset=["p_live"])
        live_net = float(sim_book(lm, lm["p_live"].values, 0.04)["pnl"].sum())

        nets = {"refresh": [], "slope": []}
        for seed in SEEDS:
            split_t = tr_r["dt"].quantile(0.75)
            base = LGBMClassifier(random_state=seed, **REFRESH_KW)
            base.fit(tr_r[tr_r["dt"] < split_t][REFRESH_FEATS],
                     tr_r[tr_r["dt"] < split_t]["resolved_yes"].astype(int))
            m = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
            cal = tr_r[tr_r["dt"] >= split_t]
            m.fit(cal[REFRESH_FEATS], cal["resolved_yes"].astype(int))
            p = m.predict_proba(te_r[REFRESH_FEATS])[:, 1]
            nets["refresh"].append(float(sim_book(te_r, p, 0.04)["pnl"].sum()))

            ms = LGBMClassifier(**{**slope_kw, "random_state": seed})
            ms.fit(tr[slope_feats], tr["resolved_yes"].astype(int))
            ps = ms.predict_proba(te[slope_feats])[:, 1]
            nets["slope"].append(float(sim_book(te, ps, 0.04)["pnl"].sum()))

        r = {"origin": f"{cutoff.date()}→{test_end.date()}", "live": live_net}
        for k in nets:
            a = np.array(nets[k])
            r[f"{k}_mean"], r[f"{k}_std"] = a.mean(), a.std()
        r["slope_beats_refresh"] = r["slope_mean"] > r["refresh_mean"]
        rows.append(r)
        print(f"[{r['origin']}] live=${live_net:+,.0f}  "
              f"refresh=${r['refresh_mean']:+,.0f}±{r['refresh_std']:,.0f}  "
              f"slope=${r['slope_mean']:+,.0f}±{r['slope_std']:,.0f}  "
              f"{'SLOPE WINS' if r['slope_beats_refresh'] else 'refresh holds'}")

    t = pd.DataFrame(rows)
    print("\n=== VERDICT (pre-registered) ===")
    all_beat = bool(t["slope_beats_refresh"].all())
    imp = float((t["slope_mean"] - t["refresh_mean"]).mean())
    std = float(t[["slope_std", "refresh_std"]].mean().mean())
    print(f"slope beats refresh in {int(t['slope_beats_refresh'].sum())}/{len(t)} origins")
    print(f"pooled improvement ${imp:+,.0f} vs pooled seed std ${std:,.0f}")
    if all_beat and imp > std:
        print("SLOPE recipe CONFIRMED better for BTC — swap the shadow slot (paper-first).")
    else:
        print("NOT confirmed — keep the refresh shadow as-is.")


if __name__ == "__main__":
    main()
