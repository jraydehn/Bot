"""SOL hourly v7 — quantile-regression move model (label change). 2026-07-30.

Item 5 of the feature-gap inventory: stop training binary resolved_yes and
model the DISTRIBUTION of the vol-scaled signed move to expiry instead.
9 LGBM quantile models (q=0.05..0.95) on y = price_move_pct/sqrt(tau_hours);
p_yes = P(move >= needed_move_to_strike) read off the monotonized predicted
quantile curve. Label semantics verified: (move>=needed)==resolved_yes 96.1%
(residual = settlement-source vs Binance-spot gap).

Features: full frozen stack — v3 (statics+slopes+hour+recent-YES) + v5 4h
context + v4 15m-stream join + microstructure (ladder + pm trajectory) +
banked liq/clock signals. ~370 features.

PROTOCOL: train <06-25, VAL 06-25..07-09 (early stop + all reporting).
The burned 07-09..07-30 window is NOT evaluated. Model saved frozen for
the late-Aug fresh-data PnL re-test. VAL numbers are selection-window
numbers — treat as upper bounds, not evidence.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping

import train_sol_hourly_niche_v3 as v3
import train_sol_hourly_v4_15mborrow as v4
from train_sol_hourly_v5_4h import add_4h_context, add_m15_4h
from kalshi_microstructure_features import build_micro_features, LADDER_COLS, TRAJ_COLS
import sol_hourly_banked_signals as bank

BASE = Path(__file__).parent
QUANTILES = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]

LGBM_Q_KW = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                 min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
                 reg_lambda=5.0, n_jobs=-1, verbose=-1)


def build_dataset():
    df = v3.load_archive()
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    df, ctx_feats = add_4h_context(df)

    m15 = v4.build_m15_stream()
    m15, _ = add_m15_4h(m15)
    m15c = [c for c in m15.columns if c != "dt"]
    df = pd.merge_asof(df, m15.rename(columns={c: f"m15_{c}" for c in m15c}),
                       on="dt", direction="backward",
                       tolerance=pd.Timedelta(minutes=45))

    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)

    liq = bank.build_liq_features(bank.fetch_liq_bars())
    liq_feats = [c for c in liq.columns if c != "known_at"]
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    df, clock_feats = bank.add_clock_and_daily(df)

    feats = (v3.feature_list(extended=True) + ctx_feats
             + [f"m15_{c}" for c in m15c] + LADDER_COLS + TRAJ_COLS
             + liq_feats + clock_feats + ["rv_4h_diag"])
    feats = [f for f in feats if f in df.columns]

    df["tau_h"] = pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1) / 60
    df["y_scaled"] = pd.to_numeric(df["price_move_pct"], errors="coerce") / np.sqrt(df["tau_h"])
    df["needed_scaled"] = (df["strike"] / df["spot"] - 1) * 100 / np.sqrt(df["tau_h"])
    df = df.dropna(subset=["y_scaled", "needed_scaled"]).reset_index(drop=True)
    return df, feats


def derive_p(qpreds: np.ndarray, needed: np.ndarray) -> np.ndarray:
    """qpreds (n, nq) predicted quantile values; needed (n,). p = P(y >= needed)."""
    q = np.array(QUANTILES)
    qm = np.maximum.accumulate(qpreds, axis=1)  # enforce monotone quantiles
    p = np.empty(len(needed))
    for i in range(len(needed)):
        cdf = np.interp(needed[i], qm[i], q, left=q[0], right=q[-1])
        p[i] = 1 - cdf
    # tail floor/cap: beyond outermost quantiles we only know p in [0.05,0.95]
    return np.clip(p, 0.02, 0.98)


def main():
    print("building dataset…")
    df, feats = build_dataset()
    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    tr = df[df["dt"] < T_END]
    va = df[(df["dt"] >= T_END) & (df["dt"] < V_END)]
    print(f"rows train={len(tr)} val={len(va)}  feats={len(feats)}")

    models = {}
    for qt in QUANTILES:
        m = LGBMRegressor(objective="quantile", alpha=qt, **LGBM_Q_KW)
        m.fit(tr[feats], tr["y_scaled"],
              eval_set=[(va[feats], va["y_scaled"])],
              callbacks=[early_stopping(40, verbose=False)])
        models[qt] = m
        print(f"  q={qt:.2f} best_iter={m.best_iteration_}")

    qp_va = np.column_stack([models[qt].predict(va[feats]) for qt in QUANTILES])

    print("\nVAL quantile coverage (empirical frac of y below predicted q):")
    for j, qt in enumerate(QUANTILES):
        cov = float((va["y_scaled"].values <= qp_va[:, j]).mean())
        print(f"  q={qt:.2f} → {cov:.3f}")

    p_va = derive_p(qp_va, va["needed_scaled"].values)
    y_va = va["resolved_yes"].values
    pm_va = va["p_market"].values
    b_model = float(np.mean((p_va - y_va) ** 2))
    b_mkt = float(np.mean((pm_va - y_va) ** 2))
    print(f"\nVAL Brier: quantile-model={b_model:.4f}  market={b_mkt:.4f}  "
          f"({'BEATS' if b_model < b_mkt else 'loses to'} market)")

    print("\nVAL flat-book sim (selection-window — upper bound, NOT evidence):")
    for side in ["yes", "no"]:
        for em in [0.04, 0.06]:
            bk = v3.sim_book(va, p_va, side, 0.20, 0.80, em)
            if len(bk) >= 20:
                print("  ", v3.summarize(bk, f"{side} band .2-.8 edge>={em}"))

    med = models[0.50]
    imp = pd.Series(med.feature_importances_, index=feats).sort_values(ascending=False)
    fam_new = LADDER_COLS + TRAJ_COLS + [f for f in feats if f.startswith("liq_")]
    print(f"\nmedian-model importance: new-arm share "
          f"(micro+liq)={imp[[f for f in fam_new if f in imp.index]].sum()/imp.sum():.1%}")
    print("top 15:", list(imp.head(15).index))

    with open(BASE / "models" / "sol_hourly_quantile_v7_20260730.pkl", "wb") as f:
        pickle.dump({"models": models, "features": feats, "quantiles": QUANTILES,
                     "note": "v7 quantile move model; train<06-25, val 06-25..07-09; "
                             "FROZEN for late-Aug fresh-data PnL scoring; burned "
                             "07-09..07-30 window never evaluated; NOT wired"}, f)
    print("\nsaved models/sol_hourly_quantile_v7_20260730.pkl (frozen, not wired)")


if __name__ == "__main__":
    main()
