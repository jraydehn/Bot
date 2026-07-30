"""SOL hourly v3b — residual-objective retrain (single pre-declared shot).

Same data/splits as v3 (train<06-25, VAL 06-25..07-09 selection, TEST 07-09+),
but LGBM trains with init_score = logit(p_market): trees model the residual
edge vs the market, so regularization shrinks toward pm instead of toward the
base rate. One selection pass on VAL, one TEST evaluation, then stop.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping
from scipy.special import logit, expit

import train_sol_hourly_niche_v3 as v3

BASE = Path(__file__).parent


def main():
    print("loading archive…")
    df = v3.load_archive()
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)]

    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    tr = df[df["dt"] < T_END]
    val = df[(df["dt"] >= T_END) & (df["dt"] < V_END)]
    test = df[df["dt"] >= V_END]

    feats = v3.feature_list(extended=True)
    base_tr = logit(tr["p_market"].clip(0.02, 0.98)).values
    base_va = logit(val["p_market"].clip(0.02, 0.98)).values
    base_te = logit(test["p_market"].clip(0.02, 0.98)).values

    m = LGBMClassifier(**v3.LGBM_KW)
    m.fit(tr[feats], tr["resolved_yes"].astype(int), init_score=base_tr,
          eval_set=[(val[feats], val["resolved_yes"].astype(int))],
          eval_init_score=[base_va],
          eval_metric="binary_logloss",
          callbacks=[early_stopping(50, verbose=False)])
    print(f"best_iter={m.best_iteration_}")

    pv = expit(base_va + m.predict_proba(val[feats], raw_score=True))
    pt = expit(base_te + m.predict_proba(test[feats], raw_score=True))

    grid = []
    for side in ["yes", "no"]:
        for lo, hi in [(0.35, 0.65), (0.30, 0.70), (0.20, 0.80),
                       (0.50, 0.80), (0.20, 0.50)]:
            for em in [0.04, 0.06, 0.08]:
                bk = v3.sim_book(val, pv, side, lo, hi, em)
                if len(bk) < 30:
                    continue
                wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
                grid.append(dict(side=side, lo=lo, hi=hi, em=em, n=len(bk),
                                 net=bk["pnl"].sum(),
                                 wk_green=(wk[wk != 0] > 0).mean()))
    g = pd.DataFrame(grid).sort_values("net", ascending=False)
    print("\nVAL grid top:")
    print(g.head(8).round(3).to_string(index=False))
    ok = g[(g["wk_green"] >= 0.99) & (g["n"] >= 40)]
    if not len(ok):
        print("no all-green config on VAL")
        ok = g.head(1)
    b = ok.iloc[0]
    print(f"\nCHOSEN (val): {dict(b)}")
    bt = v3.sim_book(test, pt, b["side"], b["lo"], b["hi"], b["em"])
    print("\nFINAL TEST:", v3.summarize(bt, "residual-v3b"))


if __name__ == "__main__":
    main()
