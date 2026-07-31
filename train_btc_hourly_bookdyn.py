"""BTC hourly book-dynamics challenger — compact survivor-core quantile
model. 2026-07-31.

Built on the features that REPLICATED on BTC (hourly_book_findings_screen,
commit c38067a): own-book microstructure (pm momentum family + rung_resid
+0.124, ladder geometry) for direction, the Coinalyze liquidation family
for vol. Cross-book and regvel did NOT replicate for BTC and are excluded
— the BTC book leads the flow hierarchy; its edge is its own microstructure.

Same protocol as SOL v8: train <07-16, early-stop/val 07-16..07-30
(reporting only), honest test = forward paper from 07-31 via
btc_hourly_bookdyn_runner.py.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping

from hourly_book_findings_screen import load_hourly, add_recent_yes, fetch_liq
from kalshi_microstructure_features import build_micro_features
from train_sol_hourly_v7_quantile import derive_p, QUANTILES

BASE = Path(__file__).parent
LIQ_CACHE = BASE / "results" / "coinalyze_liq_1h_btc_backfill_20260731.csv"

FEATS = [
    "p_market", "tau_minutes", "offset_pct", "z_moneyness",
    "hour_sin", "hour_cos", "recent_yes_6h", "recent_yes_24h", "rv_4h_ctl",
    # own-book microstructure (all replicated **)
    "pm_chg_5m", "pm_chg_15m", "pm_chg_30m", "pm_accel_15m", "pm_vel_life",
    "pm_n_obs", "rung_resid", "ladder_density", "imp_median_dist",
    "imp_width_pct",
    # liquidations (all replicated ** vs vol)
    "liq_total_z", "liq_long_z", "liq_short_z", "liq_imbalance",
    "liq_imbalance_trend6", "liq_imbalance_trend12",
]

LGBM_Q_KW = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                 min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=5.0, n_jobs=-1, verbose=-1)


def assemble(df: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    """df: prepared hourly scans (load_hourly-shaped). Shared with runner."""
    df = add_recent_yes(df)
    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    hr = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    if "offset_pct" not in df.columns:
        df["offset_pct"] = (df["strike"] / df["spot"] - 1)
    df["offset_pct"] = pd.to_numeric(df["offset_pct"], errors="coerce")
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
    print("loading BTC hourly archive…")
    raw = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
    import hourly_book_findings_screen as hb
    df = hb.load_hourly("btc")
    # keep offset_pct/composite for gates later
    extra = raw[["logged_at", "contract_ticker", "offset_pct", "composite_p_up"]]
    extra = extra.drop_duplicates(subset=["logged_at", "contract_ticker"])
    df = df.merge(extra, on=["logged_at", "contract_ticker"], how="left")

    if LIQ_CACHE.exists():
        liq = pd.read_csv(LIQ_CACHE)
        liq["known_at"] = pd.to_datetime(liq["known_at"], utc=True)
    else:
        liq = fetch_liq("btc")
        liq.to_csv(LIQ_CACHE, index=False)
    df = assemble(df, liq)
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

    with open(BASE / "models" / "btc_hourly_bookdyn_20260731.pkl", "wb") as f:
        pickle.dump({"models": models, "features": FEATS, "quantiles": QUANTILES,
                     "note": "BTC book-dynamics quantile model; train<07-16, val "
                             "07-16..30 (early stop); honest test = forward paper "
                             "from 07-31"}, f)
    print("saved models/btc_hourly_bookdyn_20260731.pkl")


if __name__ == "__main__":
    main()
