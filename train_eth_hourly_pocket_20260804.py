"""ETH hourly pocket-classifier — the bookdyn replacement candidate. 2026-08-04.

Architecture: the ONE thing that ever worked at hourly — a niche classifier
hunting the YES mid-band pocket (BTC niche v2's proven form) — upgraded with
ETH's new-generation survivor features: microstructure (pm trajectory +
ladder incl rung_resid, ETH's strongest screen signal at +0.194), the liq
family (liq_long_z 3/3 assets), cross-book btc_eth_drift_diff (ETH's one
skip-one-robust unique signal), regvel (2/3 assets). Feature assembly is
the eth bookdyn runner's existing pipeline — no new plumbing.

Discipline (all pre-declared, nothing searched):
  - pocket rule FIXED from family precedent, not swept: YES, pm [.35,.65],
    fee-adj edge >= 0.06, one bet/contract, flat $100 net of fees
  - 5-seed ensemble classifier (mean predict_proba), NO seed selection
  - train < 07-09; SINGLE frozen test 07-09..now
  - deploy bar: frozen-test net > 0 AND >= 60% weeks green; else report
    honestly and do not deploy
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier

from hourly_book_findings_screen import load_hourly
from train_eth_hourly_bookdyn import assemble
from btc15m_refresh_ensemble import SeedEnsemble

BASE = Path(__file__).parent
SPLIT = pd.Timestamp("2026-07-09", tz="UTC")
PM_LO, PM_HI, EDGE_MIN = 0.35, 0.65, 0.06
SEEDS = [0, 1, 2, 3, 4]

FEATS = [
    "p_market", "tau_minutes", "offset_pct", "z_moneyness",
    "hour_sin", "hour_cos", "recent_yes_6h", "recent_yes_24h", "rv_4h_ctl",
    "pm_chg_5m", "pm_chg_15m", "pm_chg_30m", "pm_accel_15m", "pm_vel_life",
    "pm_range_life", "rung_resid", "imp_median_dist", "imp_width_pct",
    "ladder_density",
    "btc_eth_drift_diff", "own_imp_median_dist", "own_imp_width_pct",
    "btc_imp_width_pct", "btc_book_pm_chg15",
    "liq_total_z", "liq_long_z", "liq_short_z", "liq_imbalance",
    "liq_imbalance_trend6", "regvel45_recent_yes", "regvel120_recent_yes",
    "composite_p_up",
]

LGBM_KW = dict(n_estimators=300, learning_rate=0.04, num_leaves=31,
               min_child_samples=150, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=5.0, n_jobs=-1, verbose=-1)


def main():
    print("assembling ETH dataset (bookdyn pipeline)…")
    df = load_hourly("eth")
    raw = pd.read_csv(BASE / "results" / "eth_scan_archive.csv",
                      usecols=["logged_at", "contract_ticker", "offset_pct",
                               "composite_p_up"], low_memory=False)
    raw = raw.drop_duplicates(subset=["logged_at", "contract_ticker"])
    df = df.merge(raw, on=["logged_at", "contract_ticker"], how="left")
    liq = pd.read_csv(BASE / "results" / "coinalyze_liq_1h_eth_backfill_20260731.csv")
    liq["known_at"] = pd.to_datetime(liq["known_at"], utc=True)
    btc_s = pd.read_parquet(BASE / "results" / "btc_hourly_book_series_20260730.parquet")
    eth_s = pd.read_parquet(BASE / "results" / "eth_hourly_book_series_20260730.parquet")
    d = assemble(df, liq, btc_s, eth_s)
    for c in set(FEATS) - set(d.columns):
        d[c] = np.nan
    d = d.dropna(subset=["resolved_yes", "p_market"]).reset_index(drop=True)

    tr = d[d["dt"] < SPLIT]
    te = d[d["dt"] >= SPLIT]
    print(f"train n={len(tr)} (<{SPLIT.date()})  test n={len(te)}  feats={len(FEATS)}")

    members = []
    for seed in SEEDS:
        m = LGBMClassifier(random_state=seed, **LGBM_KW)
        m.fit(tr[FEATS], tr["resolved_yes"].astype(int))
        members.append(m)
    ens = SeedEnsemble(members)

    def book(dd):
        p = ens.predict_proba(dd[FEATS])[:, 1]
        s = dd.copy()
        s["p"] = p
        fee = 0.07 * s["p_market"] * (1 - s["p_market"])
        s["edge"] = s["p"] - s["p_market"] - fee
        q = s[(s["p_market"].between(PM_LO, PM_HI)) & (s["edge"] >= EDGE_MIN)]
        q = q.sort_values("dt").drop_duplicates("contract_ticker", keep="first").copy()
        win = q["resolved_yes"] == 1
        feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
        q["pnl"] = np.where(win, 100 * (1 - q["p_market"]) / q["p_market"],
                            -100.0) - (100 / q["p_market"]) * feeq
        q["win"] = win
        return q

    def summ(bk, label):
        if not len(bk):
            return f"{label}: n=0"
        wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
        wk = wk[wk != 0]
        wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk.items())
        return (f"{label}: n={len(bk)} net=${bk['pnl'].sum():+,.0f} "
                f"WR={bk['win'].mean():.1%} BE={bk['p_market'].mean():.1%} "
                f"wk_green={(wk > 0).mean():.0%} | {wks}")

    print("\n[in-sample reference, train window]:")
    print("  ", summ(book(tr), "train"))
    bt = book(te)
    print("\n[FROZEN TEST 07-09.., single shot]:")
    print("  ", summ(bt, "test"))
    wk = bt.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
    wk = wk[wk != 0]
    passed = len(bt) >= 30 and bt["pnl"].sum() > 0 and (wk > 0).mean() >= 0.6
    print(f"\nDEPLOY BAR (n>=30, net>0, >=60% weeks green): {'PASS' if passed else 'FAIL'}")

    with open(BASE / "models" / "eth_hourly_pocket_20260804.pkl", "wb") as f:
        pickle.dump({"model": ens, "features": FEATS,
                     "rule": {"pm_lo": PM_LO, "pm_hi": PM_HI, "edge_min": EDGE_MIN},
                     "note": "ETH pocket classifier (bookdyn replacement candidate); "
                             "train<07-09, frozen test 07-09+; deploy only if bar passed"},
                    f)
    print("saved models/eth_hourly_pocket_20260804.pkl")


if __name__ == "__main__":
    main()
