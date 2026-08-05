"""BTC 15m MARKET-ANCHORED challenger — the p_gbdt seat swap. 2026-08-05.

WHY (user directive: "we need a better shadow btc 15 min model in order to
have any value for the reevaluation"): the refresh ensemble shares
production's 20 price features and inherits its information deficit —
corr(p−pm, y−pm) is NEGATIVE for both (−0.038/−0.047); the market
out-calibrates the model class everywhere. This challenger inverts the
premise: predict the outcome from MARKET state (pm, its 5-minute
trajectory, tau, moneyness, spread, vol) and learn the market's residual
biases (favorite-longshot + pm momentum — both independently evidenced).

DISCLOSED EVIDENCE STATUS (weaker than a confirmed win, better than a
copy): 5-seed × 3-origin walk-forward on scan-archive rows FAILED pooled
(corr −0.028, book −$8,331) BUT with a monotone learning curve — n_train
695 → corr −0.101; 1,857 → −0.010; 2,952 → +0.027 with book +$1,064,
ALL 5 seeds positive, on the window where production scored −0.073 /
−$7,064. Hypothesis: needs ~3k rows; the full-data fit (3,598) is the
first version that ever had them. The forward paper record is the ONLY
referee; 08-11 read is descriptive, decision ~08-18+.

Training rows: scan-archive "second scans" (prior scan of same contract
3–8 min earlier gives pm_chg_5m with zero lookahead — the validated
microstructure construction, NOT the retracted retrain and NOT the null
candle route). Live serving uses a non-mutating peek of the runner's own
pm-history buffer with a matching 3–10 min lookback window.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from btc15m_refresh_ensemble import SeedEnsemble

BASE = Path(__file__).parent
SEEDS = [0, 1, 2, 3, 4]
FEATS = ["p_market", "pm_chg_5m", "tau_minutes", "offset_pct", "spread",
         "realized_vol_annual"]
LGBM_KW = dict(n_estimators=120, num_leaves=7, max_depth=4,
               min_child_samples=40, learning_rate=0.04, reg_alpha=1.0,
               reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
               verbosity=-1)


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(BASE / "results" / "btc_scan_archive_15m.csv",
                     low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True,
                              format="mixed")
    for c in ["p_market", "tau_minutes", "spread", "resolved_yes", "spot",
              "strike", "realized_vol_annual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["resolved_yes"].notna()
            & df["p_market"].between(0.03, 0.97)].sort_values("dt")
    df["pm_prev"] = df.groupby("contract_ticker")["p_market"].shift(1)
    df["t_prev"] = df.groupby("contract_ticker")["dt"].shift(1)
    dtm = (df["dt"] - df["t_prev"]).dt.total_seconds() / 60.0
    df["pm_chg_5m"] = np.where(dtm.between(3, 8),
                               df["p_market"] - df["pm_prev"], np.nan)
    df["offset_pct"] = (df["spot"] - df["strike"]) / df["strike"] * 100
    return df.dropna(subset=FEATS + ["resolved_yes"]).reset_index(drop=True)


def main():
    d = load_rows()
    print(f"training rows: {len(d)}  ({d['dt'].min().date()} → "
          f"{d['dt'].max().date()})")
    split = d["dt"].quantile(0.75)
    fitp, calp = d[d["dt"] < split], d[d["dt"] >= split]
    members = []
    for s in SEEDS:
        base = LGBMClassifier(random_state=s, **LGBM_KW)
        base.fit(fitp[FEATS], fitp["resolved_yes"].astype(int))
        m = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        m.fit(calp[FEATS], calp["resolved_yes"].astype(int))
        members.append(m)
    ens = SeedEnsemble(members)
    p = ens.predict_proba(calp[FEATS])[:, 1]
    y = calp["resolved_yes"].values
    print(f"sanity (in-sample cal slice): brier={np.mean((p-y)**2):.4f} "
          f"corr(p-pm,y-pm)={np.corrcoef(p-calp['p_market'],y-calp['p_market'])[0,1]:+.3f}")
    art = {"model": ens, "features": FEATS,
           "note": "BTC 15m market-anchored challenger 2026-08-05; see "
                   "module docstring for disclosed evidence status; "
                   "forward-paper referee only; decisions unchanged"}
    out = BASE / "models" / "lgbm_15m_btc_mktanchor_20260805.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
