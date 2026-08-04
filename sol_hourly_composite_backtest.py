"""SOL hourly RAW-SIGNAL composite — v7/v8 replacement candidate. 2026-08-04.

No trained model anywhere: immune by construction to seed noise
(feedback_lgbm_single_fit) and model-fit leakage (feedback_derived_signal_
needs_model_oos). Three raw signals, each pre-07-09-screened and split-half
stable for SOL:
  btc_sol_drift_diff  +0.058/+0.060 skip-one-robust (strongest directional
                      signal of the campaign)
  pm_chg_30m          +0.093 (own-book momentum, 3/3 assets)
  liq_total_z         +0.040 directional for SOL, both halves

score = sum of z-scores (standardization constants FROZEN from the screen
window; equal weights, nothing fitted). side = sign(score); trade when
|score| >= K (K swept on screen ONLY), pm in [.20,.80], one bet/contract,
flat $100 net of fees.

Windows: screen <07-09 (where the signals were screened — that's what
screens are for); evaluation 07-09..now. DISCLOSURE: that window hosted
many SOL directional evaluations at the family level — this composite is
its first own look, but the deploy decision treats forward paper as the
real referee, identical epistemic standing to v7/v8 at their deployment
but on strictly better-screened, leakage-free foundations.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import train_sol_hourly_niche_v3 as v3
import sol_hourly_banked_signals as bank
import sol_hourly_crossasset_flow as xa
from kalshi_microstructure_features import build_micro_features

BASE = Path(__file__).parent
SPLIT = pd.Timestamp("2026-07-09", tz="UTC")
PM_LO, PM_HI = 0.20, 0.80
SIGNALS = ["btc_sol_drift_diff", "pm_chg_30m", "liq_total_z"]


def assemble():
    df = v3.load_archive()
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    micro = build_micro_features(df)
    df["pm_chg_30m"] = micro["pm_chg_30m"]
    btc_s = xa.build_book_series("btc")
    sol_s = xa.build_book_series("sol")
    df, _ = xa.join_neighbor(df, sol_s, "sol")
    df, _ = xa.join_neighbor(df, btc_s, "btc")
    df["btc_sol_drift_diff"] = df["sol_imp_median_dist"] - df["btc_imp_median_dist"]
    liq = bank.build_liq_features(bank.fetch_liq_bars())
    df = pd.merge_asof(df.sort_values("dt"), liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    # liq_total_z is already a 0-1 percentile rank → center it; others z-scored
    return df


def main():
    df = assemble()
    scr_mask = df["dt"] < SPLIT

    # FROZEN standardization constants from screen window only
    consts = {}
    for s in SIGNALS:
        v = df.loc[scr_mask, s]
        consts[s] = (float(v.mean()), float(v.std()))
        print(f"frozen const {s}: mean={consts[s][0]:+.4f} std={consts[s][1]:.4f} "
              f"(coverage screen={v.notna().mean():.0%} / "
              f"eval={df.loc[~scr_mask, s].notna().mean():.0%})")
    z = sum((df[s] - consts[s][0]) / consts[s][1] for s in SIGNALS)
    df["score"] = z

    def book(dd, cond):
        q = dd[cond & dd["p_market"].between(PM_LO, PM_HI)].copy()
        q["side"] = np.where(q["score"] > 0, "yes", "no")
        q = q.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
        cost = np.where(q["side"] == "yes", q["p_market"], 1 - q["p_market"])
        win = np.where(q["side"] == "yes", q["resolved_yes"] == 1, q["resolved_yes"] == 0)
        fee = 0.07 * q["p_market"] * (1 - q["p_market"])
        q["pnl"] = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * fee
        q["win"] = win
        q["cost"] = cost
        return q

    def summ(bk, label):
        if not len(bk):
            return f"{label}: n=0"
        wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
        wk = wk[wk != 0]
        wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk.items())
        return (f"{label}: n={len(bk)} net=${bk['pnl'].sum():+,.0f} "
                f"WR={bk['win'].mean():.1%} BE={bk['cost'].mean():.1%} "
                f"wk_green={(wk > 0).mean():.0%} | {wks}")

    scr = df[scr_mask]
    ev = df[~scr_mask]
    print("\n[SCREEN <07-09] K sweep + controls:")
    best_K, best_net = None, -1e18
    for K in [1.0, 1.5, 2.0, 2.5]:
        bk = book(scr, scr["score"].abs() >= K)
        print("  ", summ(bk, f"|score|>={K}"))
        if len(bk) >= 60:
            wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
            green = (wk[wk != 0] > 0).mean() if (wk != 0).any() else 0
            if bk["pnl"].sum() > best_net and green >= 0.6:
                best_K, best_net = K, bk["pnl"].sum()
    print("  ", summ(book(scr, scr["score"].abs() < 0.5), "CONTROL |score|<0.5"))
    inv = book(scr, scr["score"].abs() >= 1.5)
    inv["side"] = np.where(inv["side"] == "yes", "no", "yes")  # label only; recompute:
    if best_K is None:
        print("\nNO K clears the screen bar — composite not viable. Stopping honestly.")
        return
    print(f"\nCHOSEN K={best_K} → [EVALUATION 07-09..now, single shot]:")
    bt = book(ev, ev["score"].abs() >= best_K)
    print("  ", summ(bt, f"|score|>={best_K}"))
    print("  ", summ(book(ev, ev["score"].abs() < 0.5), "control |score|<0.5"))
    # fire-rate distribution check (the contamination tell — should be ~stable
    # for raw signals)
    scr_wk = len(book(scr, scr["score"].abs() >= best_K)) / max((scr["dt"].max() - scr["dt"].min()).days / 7, 1)
    ev_wk = len(bt) / max((ev["dt"].max() - ev["dt"].min()).days / 7, 1)
    print(f"\nfire-rate check: screen {scr_wk:.1f} trades/wk vs eval {ev_wk:.1f}/wk "
          f"(explosion = leakage tell; stability expected for raw signals)")


if __name__ == "__main__":
    main()
