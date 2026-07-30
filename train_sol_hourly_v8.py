"""SOL hourly v8 — compact survivor-core challenger. 2026-07-30.

Anti-overfit design learned from v3→v6 (each 200-350-feat model fit the
selection window harder and generalized zero): v8 uses ONLY the ~40 features
that survived the pre-07-09 partial-IC screens, plus contract basics, on the
v7 quantile architecture (proven calibrated; tails learnable).

Feature core:
  DIRECTION  btc/eth_sol_drift_diff, btc/eth_book_pm_chg15 (cross-book),
             pm_chg_5/15/30m, pm_accel_15m, pm_range_life, pm_n_obs (own-book
             momentum), imp_median_dist, ladder_density, dur_emastack,
             volnorm45_vwap_distance_pct
  VOL        liq_long_z/short_z/imbalance/total_z + trends, regvel45/120_
             recent_yes, confirm_oi_chg_pct, dur_stoch50, imp_width_pct,
             sol/btc/eth book widths, rv_4h, imp_vol_ratio_fixed (repaired:
             ladder sigma vs rv-expected move — rvol_1h was relative volume)
  BASICS     p_market, tau_minutes, offset_pct, z_moneyness, hour sin/cos,
             recent_yes_6h/24h

Training: train <07-16, early-stop/val 07-16..07-30 (recency maximized; the
honest test is FORWARD PAPER from 07-30 via sol_hourly_v8_runner.py — a
three-way A/B vs production and v7). Burned-window PnL is never reported.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping

import train_sol_hourly_niche_v3 as v3
from kalshi_microstructure_features import build_micro_features
import sol_hourly_banked_signals as bank
import sol_hourly_slope_constructions as cons
import sol_hourly_crossasset_flow as xa
from train_sol_hourly_v7_quantile import derive_p, QUANTILES

BASE = Path(__file__).parent

FEATS = [
    # basics
    "p_market", "tau_minutes", "offset_pct", "z_moneyness",
    "hour_sin", "hour_cos", "recent_yes_6h", "recent_yes_24h",
    # own-book microstructure
    "pm_chg_5m", "pm_chg_15m", "pm_chg_30m", "pm_accel_15m",
    "pm_range_life", "pm_n_obs", "imp_median_dist", "imp_width_pct",
    "ladder_density",
    # cross-book
    "btc_sol_drift_diff", "eth_sol_drift_diff",
    "btc_book_pm_chg15", "eth_book_pm_chg15",
    "btc_imp_width_pct", "eth_imp_width_pct",
    "sol_imp_median_dist", "sol_imp_width_pct", "sol_book_pm_chg15",
    # liquidations
    "liq_long_z", "liq_short_z", "liq_imbalance", "liq_total_z",
    "liq_imbalance_trend6", "liq_imbalance_trend12", "liq_total_z_trend3",
    # constructions
    "regvel45_recent_yes", "regvel120_recent_yes", "confirm_oi_chg_pct",
    "dur_emastack", "dur_stoch50", "volnorm45_vwap_distance_pct",
    # vol context
    "rv_4h_ctl", "imp_vol_ratio_fixed",
]

LGBM_Q_KW = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                 min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=5.0, n_jobs=-1, verbose=-1)


def assemble(sol: pd.DataFrame, btc_series: pd.DataFrame,
             eth_series: pd.DataFrame, sol_series: pd.DataFrame,
             liq_bars: pd.DataFrame) -> pd.DataFrame:
    """Shared by trainer (full archive) and runner (tails). `sol` must have
    dt/contract_ticker/p_market/spot/strike/tau_minutes/close_dt (+signal
    cols); series args are loop-level book summaries from
    sol_hourly_crossasset_flow.build_book_series."""
    df = v3.add_extended(sol)
    df = df[df["p_market"].notna() & df["p_market"].between(0.02, 0.98)]
    df = df.sort_values("dt").reset_index(drop=True)
    df, _ = cons.build(df)
    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)

    for prefix, ser in [("btc", btc_series), ("eth", eth_series), ("sol", sol_series)]:
        df, _ = xa.join_neighbor(df, ser, prefix)
    df["btc_sol_drift_diff"] = df["sol_imp_median_dist"] - df["btc_imp_median_dist"]
    df["eth_sol_drift_diff"] = df["sol_imp_median_dist"] - df["eth_imp_median_dist"]

    liq = bank.build_liq_features(liq_bars)
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))

    with np.errstate(divide="ignore", invalid="ignore"):
        df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(
            pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1))
        # repaired imp_vol_ratio: ladder sigma (pct) vs rv-expected move (pct)
        tau = pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1)
        exp_move_pct = df["rv_4h_ctl"] * np.sqrt(tau / 5) * 100
        df["imp_vol_ratio_fixed"] = (df["imp_width_pct"] / 2) / exp_move_pct.replace(0, np.nan)

    df["tau_h"] = pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1) / 60
    df["needed_scaled"] = (df["strike"] / df["spot"] - 1) * 100 / np.sqrt(df["tau_h"])
    for c in FEATS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def main():
    print("assembling (full archives)…")
    sol = v3.load_archive()
    sol = sol.dropna(subset=["resolved_yes"])
    btc_s = pd.read_parquet(BASE / "results" / "btc_hourly_book_series_20260730.parquet")
    eth_s = pd.read_parquet(BASE / "results" / "eth_hourly_book_series_20260730.parquet")
    sol_s = xa.build_book_series("sol")
    df = assemble(sol, btc_s, eth_s, sol_s, bank.fetch_liq_bars())
    df["y_scaled"] = pd.to_numeric(df["price_move_pct"], errors="coerce") / np.sqrt(df["tau_h"])
    df = df.dropna(subset=["y_scaled", "needed_scaled"]).reset_index(drop=True)

    T_END = pd.Timestamp("2026-07-16", tz="UTC")
    tr = df[df["dt"] < T_END]
    va = df[df["dt"] >= T_END]
    print(f"rows train={len(tr)} val={len(va)}  feats={len(FEATS)}")

    models = {}
    for qt in QUANTILES:
        m = LGBMRegressor(objective="quantile", alpha=qt, **LGBM_Q_KW)
        m.fit(tr[FEATS], tr["y_scaled"],
              eval_set=[(va[FEATS], va["y_scaled"])],
              callbacks=[early_stopping(40, verbose=False)])
        models[qt] = m
        print(f"  q={qt:.2f} best_iter={m.best_iteration_}")

    qp = np.column_stack([models[qt].predict(va[FEATS]) for qt in QUANTILES])
    print("\nval quantile coverage:")
    for j, qt in enumerate(QUANTILES):
        print(f"  q={qt:.2f} → {(va['y_scaled'].values <= qp[:, j]).mean():.3f}")
    p = derive_p(qp, va["needed_scaled"].values)
    y = va["resolved_yes"].values
    print(f"\nval Brier: v8={np.mean((p - y) ** 2):.4f}  market="
          f"{np.mean((va['p_market'].values - y) ** 2):.4f}")

    med = models[0.50]
    imp = pd.Series(med.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("median-model top 10:", list(imp.head(10).index))

    with open(BASE / "models" / "sol_hourly_v8_20260730.pkl", "wb") as f:
        pickle.dump({"models": models, "features": FEATS, "quantiles": QUANTILES,
                     "note": "v8 compact survivor-core quantile model; train<07-16, "
                             "val 07-16..30 (early stop only); honest test = forward "
                             "paper from 07-30"}, f)
    print("saved models/sol_hourly_v8_20260730.pkl")


if __name__ == "__main__":
    main()
