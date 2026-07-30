"""SOL hourly niche retrain v3 — 2026-07-29.

Phase 2 (07-28) found SOL NULL (−$1,167) under the v2 protocol (frozen
train<06-20, July holdout, YES pm∈[.35,.65], fee-adj edge>=.06). This run:
  1. Reproduces that baseline (null check).
  2. Retrains with ALL data through 07-30: train<06-25, VAL 06-25..07-09
     (early stop + ALL config selection), untouched TEST 07-09..07-30.
  3. Adds honest causal features: hour-of-day (sin/cos) and trailing settled
     YES-rate over contracts whose close_ts precedes the scan (6h/24h).
     HMM state columns are NOT used — live-logged only since 07-28 22:15Z
     (~1.5 days), and backfilled HMM states are a known lookahead trap.
  4. Selection grid (side / pm band / edge_min / feature set) runs on VAL
     ONLY; the chosen config is evaluated ONCE on TEST.
  5. Controls: shuffled-label model, pm-only model, second-scan entry.

Books: flat $100, one bet per contract (first qualifying scan),
fee = 0.07*pm*(1-pm), PnL net of fees. Outcome cols (spot_at_expiry,
price_move_pct, miss_pct) are excluded as leakage.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

BASE = Path(__file__).parent
ARCHIVE = BASE / "results" / "sol_scan_archive.csv"

SLOPE_BASES = ["stoch_k", "chg_30m", "chg_10m", "chg_5m", "bp_5m", "body_15m",
               "vol_score", "vpin_score", "obi_score", "confirmation_score",
               "no_score", "funding_bias", "vol_eff", "adx_1h", "rvol_1h",
               "liq_score", "ls_long_pct", "oi_chg_pct", "composite_p_up",
               "composite_trend", "ema_stretch_score", "vwap_stretch_score",
               "vwap_distance_pct"]
STATIC = ["p_market", "tau_minutes", "p_up_v2", "offset_pct", "composite_p_up",
          "composite_trend", "composite_rev", "ema_stack_bias",
          "ema_stretch_score", "vwap_stretch_score", "vwap_distance_pct",
          "stoch_k", "chg_30m", "chg_10m", "chg_5m", "bp_5m", "body_15m",
          "dir_15m", "vol_score", "vpin_score", "obi_score",
          "confirmation_score", "no_score", "funding_bias", "vol_eff",
          "pm_drift_5m", "adx_1h", "rvol_1h", "squeeze_1h", "liq_score",
          "liq_bias", "ls_long_pct", "oi_chg_pct", "z_moneyness"]
EXTENDED = ["hour_sin", "hour_cos", "recent_yes_6h", "recent_yes_24h"]
LEAK_COLS = {"resolved_yes", "spot_at_expiry", "price_move_pct", "miss_pct"}

LGBM_KW = dict(n_estimators=600, learning_rate=0.03, num_leaves=63,
               min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
               reg_lambda=5.0, n_jobs=-1, verbose=-1)


def load_archive() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE, low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    for c in set(STATIC + SLOPE_BASES + ["spot", "strike", "resolved_yes"]) - {"z_moneyness"}:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_ts"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    return df


def add_slopes(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["dt"].astype("int64") / 1e9
    nc = {}
    for tag, sec in [("15", 900), ("45", 2700), ("120", 7200)]:
        idx = np.searchsorted(ts, ts - sec, side="right") - 1
        valid = idx >= 0
        pv = np.where(valid, df["spot"].values[np.clip(idx, 0, None)], np.nan)
        dp = pd.Series((df["spot"].values / pv - 1) * 100, index=df.index)
        nc[f"dprice_{tag}"] = dp
        for c in SLOPE_BASES:
            pr = np.where(valid, df[c].values[np.clip(idx, 0, None)], np.nan)
            d = df[c].values - pr
            nc[f"D{tag}_{c}"] = d
            nc[f"S{tag}_{c}"] = np.clip(d / dp.replace(0, np.nan), -50, 50)
    df = pd.concat([df, pd.DataFrame(nc, index=df.index)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z_moneyness"] = np.log(df["strike"] / df["spot"]) / np.sqrt(
            df["tau_minutes"].clip(lower=1))
    return df


def add_extended(df: pd.DataFrame) -> pd.DataFrame:
    hr = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    # Trailing settled YES-rate: one outcome per contract, known only after
    # close_ts + 5min. Point-in-time by construction.
    settled = (df.dropna(subset=["close_dt", "resolved_yes"])
                 .drop_duplicates("contract_ticker", keep="first")
                 .sort_values("close_dt"))
    known_ts = (settled["close_dt"] + pd.Timedelta(minutes=5)).astype("int64").values / 1e9
    outcomes = settled["resolved_yes"].values.astype(float)
    cum = np.concatenate([[0.0], np.cumsum(outcomes)])
    scan_ts = df["dt"].astype("int64").values / 1e9
    for tag, sec in [("6h", 6 * 3600), ("24h", 24 * 3600)]:
        hi = np.searchsorted(known_ts, scan_ts, side="right")
        lo = np.searchsorted(known_ts, scan_ts - sec, side="right")
        n = hi - lo
        s = cum[hi] - cum[lo]
        df[f"recent_yes_{tag}"] = np.where(n >= 3, s / np.maximum(n, 1), np.nan)
    return df


def feature_list(extended: bool) -> list:
    feats = list(STATIC)
    for tag in ["15", "45", "120"]:
        feats.append(f"dprice_{tag}")
        for c in SLOPE_BASES:
            feats += [f"D{tag}_{c}", f"S{tag}_{c}"]
    if extended:
        feats += EXTENDED
    assert not (set(feats) & LEAK_COLS)
    return feats


def train_model(df, feats, train_end, es_start, es_end, shuffle_labels=False,
                seed=42):
    tr = df[df["dt"] < train_end]
    es = df[(df["dt"] >= es_start) & (df["dt"] < es_end)]
    y_tr = tr["resolved_yes"].values.astype(int)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y_tr = rng.permutation(y_tr)
    m = LGBMClassifier(**LGBM_KW)
    m.fit(tr[feats], y_tr,
          eval_set=[(es[feats], es["resolved_yes"].astype(int))],
          eval_metric="binary_logloss",
          callbacks=[early_stopping(50, verbose=False)])
    return m


def sim_book(df, p, side, pm_lo, pm_hi, edge_min, entry_rank=0):
    """Flat $100, one bet per contract, net of fees."""
    s = df.copy()
    s["p"] = p
    fee = 0.07 * s["p_market"] * (1 - s["p_market"])
    if side == "yes":
        s["edge"] = s["p"] - s["p_market"] - fee
        s["cost"] = s["p_market"]
        s["winb"] = s["resolved_yes"] == 1
    else:
        s["edge"] = s["p_market"] - s["p"] - fee
        s["cost"] = 1 - s["p_market"]
        s["winb"] = s["resolved_yes"] == 0
    q = s[(s["p_market"].between(pm_lo, pm_hi)) & (s["edge"] >= edge_min)]
    q = q.sort_values("dt")
    if entry_rank == 0:
        q = q.drop_duplicates("contract_ticker", keep="first")
    else:  # robustness: take the Nth qualifying scan instead
        q = q.groupby("contract_ticker", as_index=False).nth(entry_rank)
    feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
    pnl = np.where(q["winb"], 100 * (1 - q["cost"]) / q["cost"], -100.0) \
        - (100 / q["cost"]) * feeq
    out = q[["dt", "contract_ticker", "p_market", "p", "edge", "cost", "winb"]].copy()
    out["pnl"] = pnl
    return out


def summarize(book, label=""):
    if not len(book):
        return f"{label}: n=0"
    wk = book.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].agg(["count", "sum"])
    wk = wk[wk["count"] > 0]
    wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk["sum"].items())
    return (f"{label}: n={len(book)} net=${book['pnl'].sum():+,.0f} "
            f"WR={book['winb'].mean():.1%} BE={book['cost'].mean():.1%} "
            f"| weeks: {wks}")


def main():
    print("loading archive…")
    df = load_archive()
    df = add_slopes(df)
    df = add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)]
    print(f"rows: {len(df)}  span {df['dt'].min()} → {df['dt'].max()}")

    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    val = df[(df["dt"] >= T_END) & (df["dt"] < V_END)]
    test = df[df["dt"] >= V_END]
    print(f"train n={(df['dt'] < T_END).sum()}  val n={len(val)}  test n={len(test)}")

    # ── 1. Phase 2 baseline reproduction ────────────────────────────────
    print("\n[1] Phase 2 v2 baseline repro (train<06-20, holdout 07-01+, "
          "YES .35-.65 edge>=.06):")
    f_v2 = feature_list(extended=False)
    m0 = train_model(df, f_v2, pd.Timestamp("2026-06-20", tz="UTC"),
                     pd.Timestamp("2026-06-20", tz="UTC"),
                     pd.Timestamp("2026-07-01", tz="UTC"))
    hold = df[df["dt"] >= pd.Timestamp("2026-07-01", tz="UTC")]
    p0 = m0.predict_proba(hold[f_v2])[:, 1]
    print("   ", summarize(sim_book(hold, p0, "yes", 0.35, 0.65, 0.06), "baseline"))

    # ── 2. v3 models (train<06-25, early stop on VAL) ───────────────────
    models = {}
    for ext in [False, True]:
        feats = feature_list(extended=ext)
        tag = "ext" if ext else "v2feats"
        print(f"\n[2] training v3 ({tag}, {len(feats)} feats)…")
        m = train_model(df, feats, T_END, T_END, V_END)
        models[tag] = (m, feats)
        print(f"    best_iter={m.best_iteration_}")

    # ── 3. Config selection on VAL only ─────────────────────────────────
    print("\n[3] selection grid on VAL (side × band × edge × feats):")
    grid = []
    for tag, (m, feats) in models.items():
        pv = m.predict_proba(val[feats])[:, 1]
        for side in ["yes", "no"]:
            for lo, hi in [(0.35, 0.65), (0.30, 0.70), (0.20, 0.80),
                           (0.50, 0.80), (0.20, 0.50)]:
                for em in [0.04, 0.06, 0.08]:
                    bk = sim_book(val, pv, side, lo, hi, em)
                    if len(bk) < 30:
                        continue
                    wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
                    grid.append(dict(feats=tag, side=side, lo=lo, hi=hi, em=em,
                                     n=len(bk), net=bk["pnl"].sum(),
                                     wk_green=(wk[wk != 0] > 0).mean(),
                                     wr=bk["winb"].mean(), be=bk["cost"].mean()))
    g = pd.DataFrame(grid).sort_values("net", ascending=False)
    print(g.head(12).round(3).to_string(index=False))
    g.to_csv(BASE / "results" / "sol_hourly_v3_valgrid.csv", index=False)

    ok = g[(g["wk_green"] >= 0.99) & (g["n"] >= 40)]
    if not len(ok):
        print("\nNO config with all-green VAL weeks and n>=40 — SOL likely still null.")
        ok = g.head(1)
    best = ok.iloc[0]
    print(f"\nCHOSEN (val): {dict(best)}")

    # ── 4. Single TEST evaluation ────────────────────────────────────────
    m, feats = models[best["feats"]]
    pt = m.predict_proba(test[feats])[:, 1]
    bt = sim_book(test, pt, best["side"], best["lo"], best["hi"], best["em"])
    print("\n[4] FINAL TEST (07-09..07-30, untouched):")
    print("   ", summarize(bt, "TEST"))
    # pre-registered Phase 2 config on same model, for comparability
    print("   ", summarize(sim_book(test, pt, "yes", 0.35, 0.65, 0.06),
                           "TEST @phase2cfg"))

    # ── 5. Controls ──────────────────────────────────────────────────────
    print("\n[5] controls:")
    ms = train_model(df, feats, T_END, T_END, V_END, shuffle_labels=True)
    ps = ms.predict_proba(test[feats])[:, 1]
    print("   ", summarize(sim_book(test, ps, best["side"], best["lo"],
                                    best["hi"], best["em"]), "shuffled-label"))
    mp = train_model(df, ["p_market", "tau_minutes", "offset_pct", "z_moneyness"],
                     T_END, T_END, V_END)
    pp = mp.predict_proba(test[["p_market", "tau_minutes", "offset_pct", "z_moneyness"]])[:, 1]
    print("   ", summarize(sim_book(test, pp, best["side"], best["lo"],
                                    best["hi"], best["em"]), "pm-only model"))
    print("   ", summarize(sim_book(test, pt, best["side"], best["lo"],
                                    best["hi"], best["em"], entry_rank=1),
                           "second-scan entry"))

    bt.to_csv(BASE / "results" / "sol_hourly_v3_testbook.csv", index=False)
    with open(BASE / "models" / "sol_hourly_lgbm_niche_v3_20260729.pkl", "wb") as f:
        pickle.dump({"model": m, "features": feats, "slope_bases": SLOPE_BASES,
                     "config": {k: (v.item() if hasattr(v, "item") else v)
                                for k, v in best.items()},
                     "note": "v3 train<06-25, val 06-25..07-09 (selection), "
                             "test 07-09..07-30; NOT DEPLOYED pending review"}, f)
    print("\nsaved models/sol_hourly_lgbm_niche_v3_20260729.pkl (not wired anywhere)")


if __name__ == "__main__":
    main()
