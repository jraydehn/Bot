"""BTC/ETH 15m pm-momentum shadow retrain. 2026-07-31.

Approved follow-on to the pm-momentum finding (partial IC +0.047 BTC /
+0.049 ETH controlling for BOTH pm and the live model's own probability —
both production models are blind to book momentum). This retrains EACH
asset's OWN currently-deployed feature architecture (extracted live from
its production pkl, so the comparison is apples-to-apples) with
pm_chg_5m/pm_chg_10m/pm_n_obs added, joined from the Kalshi candle backfill.

  BTC production: 20-feature CalibratedClassifierCV (models/lgbm_15m_btc.pkl)
  ETH production: 176-feature D/S-slope dict architecture
                  (models/lgbm_15m_eth.pkl, slope_bases stored in the pkl)

Protocol (matches the hourly v3-v8 discipline): train on paper_trades_
{asset}15m.csv rows before TRAIN_END, single frozen holdout after. Two
models trained per asset on the IDENTICAL split — baseline (existing
features only) and +pmmom (existing features + pm momentum) — isolating
the incremental effect cleanly, rather than diffing against whatever
regularization/window the already-deployed pkl happened to use. Scored by
Brier AND flat-$100 net-of-fees book PnL on the holdout — PnL is the bar,
per this project's own rule (Brier/IC is a screen, not a verdict).

Usage: python3 train_15m_pmmom_shadow.py BTC
       python3 train_15m_pmmom_shadow.py ETH
"""
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping

BASE = Path(__file__).parent
TRAIN_END = pd.Timestamp("2026-07-16", tz="UTC")

LGBM_KW = dict(n_estimators=400, learning_rate=0.04, num_leaves=31,
               min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=5.0, n_jobs=-1, verbose=-1)


def load_paper_archive(asset: str) -> pd.DataFrame:
    df = pd.read_csv(BASE / "results" / f"paper_trades_{asset.lower()}15m.csv",
                     low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    return df.dropna(subset=["resolved_yes"]).reset_index(drop=True)


def add_dS_slopes(df: pd.DataFrame, slope_bases: list) -> pd.DataFrame:
    """Same D{tag}_/S{tag}_ construction as the hourly slope work, applied
    to the 15m archive's own scan-level history (15/45/120 min lookback)."""
    for c in set(slope_bases) - set(df.columns):
        df[c] = np.nan
    for c in slope_bases + ["spot"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ts = df["dt"].astype("int64").values / 1e9
    nc = {}
    for tag, sec in [("15", 900), ("45", 2700), ("120", 7200)]:
        idx = np.searchsorted(ts, ts - sec, side="right") - 1
        valid = idx >= 0
        pv = np.where(valid, df["spot"].values[np.clip(idx, 0, None)], np.nan)
        dp = pd.Series((df["spot"].values / pv - 1) * 100, index=df.index)
        nc[f"dprice_{tag}"] = dp
        for c in slope_bases:
            pr = np.where(valid, df[c].values[np.clip(idx, 0, None)], np.nan)
            d = df[c].values - pr
            nc[f"D{tag}_{c}"] = d
            nc[f"S{tag}_{c}"] = np.clip(d / dp.replace(0, np.nan), -50, 50)
    # drop any pre-existing columns under these names (older schema baggage)
    # so the fresh, correctly-aligned computation here is the one that wins.
    df = df.drop(columns=[c for c in nc if c in df.columns])
    return pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)


def add_pm_momentum(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """pm_chg_5m/10m + pm_n_obs from the Kalshi 1-min candle backfill,
    joined by ticker + nearest time (<=3min tolerance)."""
    cd = pd.read_csv(BASE / "results" / f"kalshi_15m_candles_{asset.lower()}.csv",
                     low_memory=False)
    for c in ["bid_close", "ask_close", "end_ts"]:
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd.dropna(subset=["end_ts", "bid_close", "ask_close"])
    cd = cd.sort_values(["ticker", "end_ts"]).reset_index(drop=True)
    cd["mid"] = (cd["bid_close"] + cd["ask_close"]) / 2
    cd["i"] = cd.groupby("ticker").cumcount()
    cd["pm_chg_5m"] = cd["mid"] - cd.groupby("ticker")["mid"].shift(5)
    cd["pm_chg_10m"] = cd["mid"] - cd.groupby("ticker")["mid"].shift(10)
    cd["pm_n_obs"] = cd["i"]
    cd["end_ts"] = cd["end_ts"].astype(float)
    keep = cd[["ticker", "end_ts", "pm_chg_5m", "pm_chg_10m", "pm_n_obs"]]
    keep = keep.rename(columns={"ticker": "contract_ticker"}).sort_values("end_ts")

    df = df.copy()
    df["ts"] = df["dt"].astype("int64").astype(float) / 1e9
    joined = pd.merge_asof(df.sort_values("ts"), keep, left_on="ts",
                           right_on="end_ts", by="contract_ticker",
                           direction="backward", tolerance=180.0)
    return joined.drop(columns=["ts", "end_ts"])


def sim_book(df: pd.DataFrame, p: np.ndarray, edge_min: float = 0.04) -> pd.DataFrame:
    s = df.copy()
    s["p"] = p
    fee = 0.07 * s["p_market"] * (1 - s["p_market"])
    ey = s["p"] - s["p_market"] - fee
    en = s["p_market"] - s["p"] - fee
    s["side"] = np.where(ey >= en, "yes", "no")
    s["edge"] = np.maximum(ey, en)
    q = s[s["edge"] >= edge_min].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first")
    cost = np.where(q["side"] == "yes", q["p_market"], 1 - q["p_market"])
    win = np.where(q["side"] == "yes", q["resolved_yes"] == 1, q["resolved_yes"] == 0)
    feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
    pnl = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * feeq
    q = q.copy()
    q["pnl"], q["win"], q["cost"] = pnl, win, cost
    return q


def summarize(q: pd.DataFrame, label: str) -> str:
    if not len(q):
        return f"{label}: n=0"
    wk = q.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
    wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
    return (f"{label}: n={len(q)} net=${q['pnl'].sum():+,.0f} "
            f"WR={q['win'].mean():.1%} BE={q['cost'].mean():.1%} | {wks}")


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    print(f"=== {asset} 15m pm-momentum shadow retrain ===")

    with open(BASE / "models" / f"lgbm_15m_{asset.lower()}.pkl", "rb") as f:
        prod = pickle.load(f)
    if isinstance(prod, dict):
        base_feats = list(prod["features"])
        slope_bases = list(prod.get("slope_bases", []))
    else:
        fn = getattr(prod, "feature_name_", None) or getattr(prod, "feature_names_in_", None)
        if fn is None and hasattr(prod, "calibrated_classifiers_"):
            est = prod.calibrated_classifiers_[0].estimator
            fn = getattr(est, "feature_name_", None) or getattr(est, "feature_names_in_", None)
        base_feats = [str(x) for x in fn] if fn is not None else []
        slope_bases = []
    print(f"production feature count: {len(base_feats)}  slope_bases: {len(slope_bases)}")

    df = load_paper_archive(asset)
    print(f"archive: {len(df)} resolved rows, {df['dt'].min().date()} → {df['dt'].max().date()}")
    if slope_bases:
        df = add_dS_slopes(df, slope_bases)
    df = add_pm_momentum(df, asset)
    cov = df["pm_chg_5m"].notna().mean()
    print(f"pm-momentum join coverage: {cov:.1%}")

    for c in set(base_feats) - set(df.columns):
        df[c] = np.nan
    for c in base_feats:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    tr = df[df["dt"] < TRAIN_END]
    te = df[df["dt"] >= TRAIN_END]
    print(f"train n={len(tr)}  holdout n={len(te)} ({te['dt'].min().date()} → {te['dt'].max().date()})")

    pmmom_feats = base_feats + ["pm_chg_5m", "pm_chg_10m", "pm_n_obs"]
    models = {}
    for tag, feats in [("baseline", base_feats), ("pmmom", pmmom_feats)]:
        val_cut = TRAIN_END - pd.Timedelta(days=7)
        tr_fit = tr[tr["dt"] < val_cut]
        tr_val = tr[tr["dt"] >= val_cut]
        m = LGBMClassifier(**LGBM_KW)
        m.fit(tr_fit[feats], tr_fit["resolved_yes"].astype(int),
              eval_set=[(tr_val[feats], tr_val["resolved_yes"].astype(int))],
              callbacks=[early_stopping(30, verbose=False)])
        models[tag] = (m, feats)
        print(f"  [{tag}] best_iter={m.best_iteration_} n_feats={len(feats)}")

    print(f"\n=== HOLDOUT ({te['dt'].min().date()} → {te['dt'].max().date()}), single frozen shot ===")
    p_live = pd.to_numeric(te.get("p_model_15m"), errors="coerce").values
    y_te = te["resolved_yes"].values
    print(f"  live production Brier: {np.nanmean((p_live - y_te) ** 2):.4f}")
    _lm = te.copy()
    _lm["p_model_15m"] = pd.to_numeric(_lm["p_model_15m"], errors="coerce")
    _lm = _lm.dropna(subset=["p_market", "p_model_15m"])
    for em in [0.04, 0.06]:
        print("   ", summarize(sim_book(_lm, _lm["p_model_15m"].values, em),
                                f"LIVE prod edge>={em}"))
    for tag in ["baseline", "pmmom"]:
        m, feats = models[tag]
        p = m.predict_proba(te[feats])[:, 1]
        brier = np.mean((p - y_te) ** 2)
        print(f"  [{tag}] Brier: {brier:.4f}")
        for em in [0.04, 0.06]:
            print("   ", summarize(sim_book(te, p, em), f"{tag} edge>={em}"))

    with open(BASE / "models" / f"lgbm_15m_{asset.lower()}_pmmom_shadow_20260731.pkl", "wb") as f:
        pickle.dump({"model": models["pmmom"][0], "features": pmmom_feats,
                     "slope_bases": slope_bases,
                     "note": f"{asset} 15m + pm-momentum shadow; train<{TRAIN_END.date()}, "
                             "holdout single frozen shot; NOT wired live pending review"},
                    f)
    print(f"\nsaved models/lgbm_15m_{asset.lower()}_pmmom_shadow_20260731.pkl (not wired)")


if __name__ == "__main__":
    main()
