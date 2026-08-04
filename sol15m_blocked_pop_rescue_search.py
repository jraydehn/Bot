"""Comprehensive rescue search: SOL 15m gate-blocked population. 2026-08-04.

Per the comprehensive_rescue skill. Population: production-model book trades
(fee-adj edge>=0.04, one/contract, pm .03-.97) BLOCKED by the live gate
stack (v2 bands/persistence + sol_markov + zdrift), over the FULL archive —
motivated by the user seeing the gates block the profitable 07-30/31 streak;
question is whether a causal condition separates rescuable blocked trades
from the rest, validated across months, not that one week.

Reconstructions (Phase 3, zero-lookahead):
- d45_/d120_/slope45_/slope120_ for the 8 persist bases via last-scan-at-or-
  before t-45/120min (exact runner construction incl clip ±50, undefined ->
  condition FALSE).
- sol_persist_score full-history from the exact live thresholds
  (autocorr1_30>=0, hurst>=0.65, slope120_bb>=0.5, slope120_rsi>=20,
  slope120_ema20>=1.5, d45_vwap>=0.5, d45_stoch5m>=40). NOTE: hurst/autocorr
  only log from ~07-08 -> earlier conditions are FALSE, matching how the live
  runner behaves with missing inputs (disclosed).
Disclosed NOT reconstructed (thin pre-07-08, price-parquet rebuild out of
scope this pass): kc_*, donch_*, kalman_*, ou_*, arima_15m — their post-07-08
coverage IS tested.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
RNG = np.random.default_rng(7)

EXCLUDE = {"logged_at", "decision_time", "asset", "contract_ticker", "close_time",
           "decision", "side", "kelly_fraction", "bet_fraction", "bet_amount",
           "bankroll", "resolved_yes", "would_win", "would_pnl", "would_pnl_net",
           "fee_est", "spot_at_expiry", "price_move_pct", "miss_pct", "is_live",
           "dt", "spot", "floor_strike", "p_model_15m", "raw_edge", "p_gbdt",
           "p_model_pre_expand"}

SLOPE_BASES = ["stoch_k_5m", "stoch_k_15m", "rsi_1h", "bb_pct_1h", "vwap_dist",
               "ema20_dist_1h", "hurst_exponent", "realized_vol_annual"]


def load_and_reconstruct():
    df = pd.read_csv(BASE / "results" / "paper_trades_sol15m.csv", low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in
                ("logged_at", "decision_time", "asset", "contract_ticker",
                 "close_time", "decision", "side", "would_win", "is_live",
                 "markov_regime_1h", "markov_regime_15m", "markov_eth_daily",
                 "markov_sol_6h", "markov_sol_4h", "markov_sol_1h", "dt")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # zero-lookahead slope reconstruction on scan snapshots
    snap = df.drop_duplicates("dt", keep="last")
    ts = snap["dt"].astype("int64").values / 1e9
    spot = snap["spot"].values
    rec = {"dt": snap["dt"]}
    for tag, mins in (("45", 45), ("120", 120)):
        idx = np.searchsorted(ts, ts - mins * 60, side="right") - 1
        ok = idx >= 0
        pv = np.where(ok, spot[np.clip(idx, 0, None)], np.nan)
        dpr = (spot / pv - 1.0) * 100.0
        rec[f"R_dprice_{tag}"] = dpr
        for col in SLOPE_BASES:
            v = snap[col].values
            prv = np.where(ok, v[np.clip(idx, 0, None)], np.nan)
            d = v - prv
            rec[f"R_d{tag}_{col}"] = d
            with np.errstate(divide="ignore", invalid="ignore"):
                sl = np.clip(d / np.where(dpr == 0, np.nan, dpr), -50, 50)
            rec[f"R_slope{tag}_{col}"] = sl
    recdf = pd.DataFrame(rec)
    df = df.merge(recdf, on="dt", how="left")

    def cond(v, thr):
        return (pd.to_numeric(v, errors="coerce") >= thr).fillna(False).astype(int)
    df["R_persist"] = (cond(df["autocorr1_30"], 0.0) + cond(df["hurst_exponent"], 0.65)
                       + cond(df["R_slope120_bb_pct_1h"], 0.5)
                       + cond(df["R_slope120_rsi_1h"], 20.0)
                       + cond(df["R_slope120_ema20_dist_1h"], 1.5)
                       + cond(df["R_d45_vwap_dist"], 0.5)
                       + cond(df["R_d45_stoch_k_5m"], 40.0))
    return df


def build_blocked_population(df):
    s = df[df["resolved_yes"].notna() & df["p_market"].between(0.03, 0.97)].copy()
    s = s.dropna(subset=["p_model_15m"])
    fee = 0.07 * s["p_market"] * (1 - s["p_market"])
    ey = s["p_model_15m"] - s["p_market"] - fee
    en = s["p_market"] - s["p_model_15m"] - fee
    s["side_b"] = np.where(ey >= en, "yes", "no")
    s["edge_b"] = np.maximum(ey, en)
    q = s[s["edge_b"] >= 0.04].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first").copy()

    v2 = np.where(q["side_b"] == "yes", q["R_persist"] >= 3,
                  ~((q["p_market"] > 0.8) |
                    (q["p_market"].between(0.5, 0.65)
                     & ~(q["R_slope120_stoch_k_15m"] >= 40))))
    m6 = q["markov_sol_6h"].astype(str); m4 = q["markov_sol_4h"].astype(str)
    m1 = q["markov_sol_1h"].astype(str)
    sc1 = q["stoch_cross_1h"].fillna(0.0); sk1 = q["stoch_k_1h"].fillna(50.0)
    oi = q["oi_chg_pct"].fillna(0.0); off = q["offset_pct"].fillna(0.0)
    gy = ((m6 == "Bull") & (sc1 != 0)) | (m4 == "Sideways") | ((m1 == "Sideways") & (oi < 0.0535))
    ry = ((m6 == "Bull") & (sc1 == 0)) | ((m1 == "Sideways") & (oi >= 0.0535))
    gn = ((m6 == "Bull") & (off > -0.006)) | ((m4 == "Sideways") & (sk1 < 90.0))
    rn = ((m6 == "Bull") & (off <= -0.006)) | ((m4 == "Sideways") & (sk1 >= 90.0))
    mkv = np.where(q["side_b"] == "yes", ~(gy & ~ry), ~(gn & ~rn))
    zok = np.where(q["side_b"] == "no", ~(q["z_drift_6h"] < 0.55).fillna(False), True)
    passed = v2 & mkv & zok
    blocked = q[~passed].copy()

    cost = np.where(blocked["side_b"] == "yes", blocked["p_market"], 1 - blocked["p_market"])
    win = np.where(blocked["side_b"] == "yes", blocked["resolved_yes"] == 1,
                   blocked["resolved_yes"] == 0)
    feeq = 0.07 * blocked["p_market"] * (1 - blocked["p_market"])
    blocked["pnl"] = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * feeq
    blocked["win_b"] = win.astype(float)
    blocked["be"] = cost + feeq * 0  # breakeven WR ~ cost (fee folded into pnl)
    blocked["edge_t"] = blocked["win_b"] - cost
    blocked["week"] = blocked["dt"].dt.to_period("W").astype(str)
    return blocked


def bootstrap_p(vals, n=4000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return 1.0
    means = np.array([RNG.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    return float((means <= 0).mean())


def main():
    df = load_and_reconstruct()
    pop = build_blocked_population(df)
    print(f"BLOCKED population: n={len(pop)}  {pop['dt'].min().date()} → {pop['dt'].max().date()}  "
          f"net=${pop['pnl'].sum():+,.0f}  mean edge={pop['edge_t'].mean():+.4f}  "
          f"weeks={pop['week'].nunique()}")

    cat_cols = ["markov_sol_6h", "markov_sol_4h", "markov_sol_1h",
                "markov_regime_1h", "markov_regime_15m"]
    skip = EXCLUDE | set(cat_cols) | {"side_b", "edge_b", "pnl", "win_b", "be",
                                      "edge_t", "week"}
    num_candidates = [c for c in pop.columns if c not in skip
                      and pop[c].dtype != object]
    coverage = {c: int(pop[c].notna().sum()) for c in num_candidates}
    tested = [c for c in num_candidates if coverage[c] >= 20]
    thin = {c: v for c, v in coverage.items() if v < 20}
    print(f"\nPhase 2 coverage: {len(num_candidates)} numeric candidates; "
          f"{len(tested)} tested (>=20 non-null in population); "
          f"{len(thin)} SKIPPED thin: {sorted(thin)[:12]}{'…' if len(thin)>12 else ''}")

    n_tests = 0
    survivors = []
    for c in tested:
        v = pop[c]
        u = v.dropna().unique()
        if len(u) <= 6:
            for val in u:
                sub = pop[v == val]
                n_tests += 1
                if len(sub) >= 30 and sub["edge_t"].mean() > 0.02 and sub["pnl"].sum() > 200:
                    survivors.append((f"{c}=={val}", sub))
        else:
            qs = v.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).unique()
            for thr in qs:
                for d, mask in (("≥", v >= thr), ("<", v < thr)):
                    sub = pop[mask.fillna(False)]
                    n_tests += 1
                    if len(sub) >= 30 and sub["edge_t"].mean() > 0.02 and sub["pnl"].sum() > 200:
                        survivors.append((f"{c}{d}{thr:.4g}", sub))
    for c in cat_cols:
        for val in pop[c].dropna().astype(str).unique():
            sub = pop[pop[c].astype(str) == val]
            n_tests += 1
            if len(sub) >= 30 and sub["edge_t"].mean() > 0.02 and sub["pnl"].sum() > 200:
                survivors.append((f"{c}=={val}", sub))
    print(f"\nPhase 1/4: {n_tests} splits tested; {len(survivors)} clear the raw screen "
          f"(n>=30, mean edge>+0.02, net>$200)")

    rows = []
    for name, sub in survivors:
        wk = sub.groupby("week")["pnl"].sum()
        tb = bootstrap_p(sub["edge_t"].values)
        wvals = wk.values
        wb = float((np.array([RNG.choice(wvals, len(wvals), replace=True).mean()
                              for _ in range(4000)]) <= 0).mean())
        rows.append({"cond": name, "n": len(sub), "net": round(sub["pnl"].sum()),
                     "edge": round(sub["edge_t"].mean(), 4),
                     "weeks": len(wk), "wk_pos": int((wk > 0).sum()),
                     "worst_wk_share": round(float(wk.min() / sub["pnl"].sum()), 2)
                     if sub["pnl"].sum() != 0 else np.nan,
                     "p_trade": round(tb, 4), "p_week": round(wb, 4)})
    r = pd.DataFrame(rows).sort_values(["p_week", "p_trade"]) if rows else pd.DataFrame()
    if len(r):
        print("\nsurvivors w/ bootstraps (top 25 by week-level p):")
        print(r.head(25).to_string(index=False))
        robust = r[(r["p_trade"] < 0.05) & (r["p_week"] < 0.10)
                   & (r["wk_pos"] / r["weeks"] >= 2 / 3)]
        print(f"\nROBUST by skill bar (p_trade<.05, p_week<.10, wins >=2/3 weeks): {len(robust)}")
        if len(robust):
            print(robust.to_string(index=False))
            robust.to_csv(BASE / "results" / "sol15m_blocked_rescue_robust.csv", index=False)
    else:
        print("\nNO survivors past the raw screen.")

    pop.to_csv(BASE / "results" / "sol15m_blocked_population.csv", index=False)


if __name__ == "__main__":
    main()
