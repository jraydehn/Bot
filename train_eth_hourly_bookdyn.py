"""ETH hourly book-dynamics challenger — compact survivor-core quantile
model. 2026-07-31.

Built on ETH's replicated survivors (hourly_book_findings_screen ETH):
own-book micro (pm momentum family 3/3-asset gold standard; rung_resid
+0.194; imp_width_pct) for direction, btc_eth_drift_diff (skip-one-robust —
ETH borrows from the BTC book, mid-flow-hierarchy) cross-book, Coinalyze
liq family (liq_long_z 3/3 assets) + regvel120 (2/3) for vol. Non-
replicating features excluded (SOL→ETH cross-book, liq-direction).

Same protocol: train <07-16, early-stop/val 07-16..07-30 (reporting only),
honest test = forward paper from 07-31 via eth_hourly_bookdyn_runner.py.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping

from hourly_book_findings_screen import load_hourly, add_recent_yes, fetch_liq
from kalshi_microstructure_features import build_micro_features
import sol_hourly_crossasset_flow as xa
from train_sol_hourly_v7_quantile import derive_p, QUANTILES

BASE = Path(__file__).parent
LIQ_CACHE = BASE / "results" / "coinalyze_liq_1h_eth_backfill_20260731.csv"

FEATS = [
    "p_market", "tau_minutes", "offset_pct", "z_moneyness",
    "hour_sin", "hour_cos", "recent_yes_6h", "recent_yes_24h", "rv_4h_ctl",
    # own-book microstructure (replicated)
    "pm_chg_5m", "pm_chg_15m", "pm_chg_30m", "pm_accel_15m", "pm_vel_life",
    "pm_range_life", "rung_resid", "imp_median_dist", "imp_width_pct",
    # cross-book (BTC leads ETH; skip-one-robust)
    "btc_eth_drift_diff", "own_imp_median_dist", "own_imp_width_pct",
    "btc_imp_width_pct",
    # liquidations + regime velocity (vol core)
    "liq_total_z", "liq_long_z", "liq_short_z", "liq_imbalance",
    "liq_imbalance_trend6", "regvel45_recent_yes", "regvel120_recent_yes",
]

LGBM_Q_KW = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                 min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=5.0, n_jobs=-1, verbose=-1)


def assemble(df: pd.DataFrame, liq: pd.DataFrame, btc_series: pd.DataFrame,
             eth_series: pd.DataFrame) -> pd.DataFrame:
    """df: prepared ETH hourly scans (load_hourly-shaped, with offset_pct/
    composite_p_up preserved). Shared trainer/runner."""
    df = add_recent_yes(df)
    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)
    df, _ = xa.join_neighbor(df, eth_series, "own")
    df, _ = xa.join_neighbor(df, btc_series, "btc")
    df["btc_eth_drift_diff"] = df["own_imp_median_dist"] - df["btc_imp_median_dist"]
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    hr = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["offset_pct"] = pd.to_numeric(df.get("offset_pct"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(
            pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1))
    df["tau_h"] = pd.to_numeric(df["tau_minutes"], errors="coerce").clip(lower=1) / 60
    df["needed_scaled"] = (df["strike"] / df["spot"] - 1) * 100 / np.sqrt(df["tau_h"])
    for c in FEATS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def main():
    print("loading ETH hourly archive…")
    df = load_hourly("eth")
    raw = pd.read_csv(BASE / "results" / "eth_scan_archive.csv",
                      usecols=["logged_at", "contract_ticker", "offset_pct",
                               "composite_p_up"], low_memory=False)
    raw = raw.drop_duplicates(subset=["logged_at", "contract_ticker"])
    df = df.merge(raw, on=["logged_at", "contract_ticker"], how="left")

    if LIQ_CACHE.exists():
        liq = pd.read_csv(LIQ_CACHE)
        liq["known_at"] = pd.to_datetime(liq["known_at"], utc=True)
    else:
        liq = fetch_liq("eth")
        liq.to_csv(LIQ_CACHE, index=False)
    btc_s = pd.read_parquet(BASE / "results" / "btc_hourly_book_series_20260730.parquet")
    eth_s = pd.read_parquet(BASE / "results" / "eth_hourly_book_series_20260730.parquet")
    df = assemble(df, liq, btc_s, eth_s)
    df["y_scaled"] = df["price_move_pct"] / np.sqrt(df["tau_h"])
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
    print(f"\nval Brier: bookdyn={np.mean((p - y) ** 2):.4f}  "
          f"market={np.mean((va['p_market'].values - y) ** 2):.4f}")
    med = models[0.50]
    imp = pd.Series(med.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("median-model top 10:", list(imp.head(10).index))

    with open(BASE / "models" / "eth_hourly_bookdyn_20260731.pkl", "wb") as f:
        pickle.dump({"models": models, "features": FEATS, "quantiles": QUANTILES,
                     "note": "ETH book-dynamics quantile model; train<07-16, val "
                             "07-16..30 (early stop); honest test = forward paper "
                             "from 07-31"}, f)
    print("saved models/eth_hourly_bookdyn_20260731.pkl")


if __name__ == "__main__":
    main()
