"""Vol-mispricing tail book backtest — BTC/ETH hourly. 2026-08-04.

The direction-neutral replacement for the failed bookdyn direction books.
Axis: predicted hourly move width vs the ladder's implied width.

  PREDICTED width: the bookdyn quantile models' tail spread — the one
    dimension those models demonstrably learned (tails trained 20-110
    iters while medians flatlined). Per contract: (q85−q15)×sqrt(tau_h)
    in move-%; per scan loop: median across the loop's contracts.
  IMPLIED width: ladder imp_width_pct per loop (strike distance between
    the pm=.16 and pm=.84 rungs — the market's ~2σ hourly move).
  SIGNAL: vratio = predicted/implied. When >= threshold, buy BOTH tails
    once per event-hour: cheapest YES rung with pm in [.03,.15] AND
    cheapest NO rung with pm in [.85,.97] (cost 1−pm in [.03,.15]).
    $100 per leg, net of fees, hold to settlement.

Protocol: threshold selected on <07-09 ONLY; single frozen test 07-09..
now (disclosure: this calendar window hosted earlier DIRECTIONAL tests;
first use for the vol question — forward paper is the final referee).
Controls: unconditional-tails baseline (is the signal adding anything?)
and the inverse regime (vratio<1).

Usage: python3 hourly_voltail_backtest.py BTC|ETH
"""
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import sol_hourly_crossasset_flow as xa
from train_sol_hourly_v7_quantile import QUANTILES

BASE = Path(__file__).parent
SPLIT = pd.Timestamp("2026-07-09", tz="UTC")
Q_LO, Q_HI = 0.15, 0.85
PM_TAIL_LO = (0.03, 0.15)   # cheap YES rung band
PM_TAIL_HI = (0.85, 0.97)   # cheap NO rung band


def predicted_width(asset, df):
    """Per-row (q85−q15)*sqrt(tau_h) from the asset's bookdyn model."""
    if asset == "BTC":
        from train_btc_hourly_bookdyn import assemble, FEATS
        from hourly_book_findings_screen import fetch_liq
        liq_cache = BASE / "results" / "coinalyze_liq_1h_btc_backfill_20260731.csv"
        liq = pd.read_csv(liq_cache); liq["known_at"] = pd.to_datetime(liq["known_at"], utc=True)
        d = assemble(df, liq)
        with open(BASE / "models" / "btc_hourly_bookdyn_20260731.pkl", "rb") as f:
            art = pickle.load(f)
    elif asset == "ETH":
        from train_eth_hourly_bookdyn import assemble, FEATS
        liq_cache = BASE / "results" / "coinalyze_liq_1h_eth_backfill_20260731.csv"
        liq = pd.read_csv(liq_cache); liq["known_at"] = pd.to_datetime(liq["known_at"], utc=True)
        btc_s = pd.read_parquet(BASE / "results" / "btc_hourly_book_series_20260730.parquet")
        eth_s = pd.read_parquet(BASE / "results" / "eth_hourly_book_series_20260730.parquet")
        d = assemble(df, liq, btc_s, eth_s)
        with open(BASE / "models" / "eth_hourly_bookdyn_20260731.pkl", "rb") as f:
            art = pickle.load(f)
    else:  # SOL — v8 compact model (its tails trained deepest: q05=110 iters)
        from train_sol_hourly_v8 import assemble, FEATS
        import sol_hourly_banked_signals as bank
        import sol_hourly_crossasset_flow as xa
        import train_sol_hourly_niche_v3 as v3
        # v8's assembly needs the FULL-column archive (slope-construction
        # bases), not the slim loader frame passed in — load its own.
        full = v3.load_archive()
        full = full.dropna(subset=["resolved_yes"])
        btc_s = pd.read_parquet(BASE / "results" / "btc_hourly_book_series_20260730.parquet")
        eth_s = pd.read_parquet(BASE / "results" / "eth_hourly_book_series_20260730.parquet")
        sol_s = xa.build_book_series("sol")
        d = assemble(full, btc_s, eth_s, sol_s, bank.fetch_liq_bars())
        with open(BASE / "models" / "sol_hourly_v8_20260730.pkl", "rb") as f:
            art = pickle.load(f)
    models = art["models"]
    lo = models[Q_LO].predict(d[FEATS])
    hi = models[Q_HI].predict(d[FEATS])
    d["pred_width_pct"] = (hi - lo) * np.sqrt(d["tau_h"])
    return d


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    a = asset.lower()
    print(f"=== {asset} hourly vol-tail backtest ===")

    from hourly_book_findings_screen import load_hourly
    df = load_hourly(a)
    raw = pd.read_csv(BASE / "results" / f"{a}_scan_archive.csv",
                      usecols=["logged_at", "contract_ticker", "offset_pct",
                               "composite_p_up"], low_memory=False)
    raw = raw.drop_duplicates(subset=["logged_at", "contract_ticker"])
    df = df.merge(raw, on=["logged_at", "contract_ticker"], how="left")
    d = predicted_width(asset, df)

    # ladder implied width: assemble() already computes per-row imp_width_pct
    # via build_micro_features (same-loop ladder interpolation)
    d = d.sort_values("dt")
    d = d.dropna(subset=["imp_width_pct", "pred_width_pct", "resolved_yes"])
    d["_event"] = d["contract_ticker"].astype(str).str.rsplit("-T", n=1).str[0]
    d["_loop"] = d["dt"].dt.floor("2min")
    loop_pred = d.groupby(["_loop", "_event"])["pred_width_pct"].median().rename("loop_pred")
    d = d.merge(loop_pred, on=["_loop", "_event"], how="left")
    d["vratio"] = d["loop_pred"] / d["imp_width_pct"]
    print(f"rows with signal: {len(d)}  vratio median={d['vratio'].median():.2f}  "
          f"p90={d['vratio'].quantile(.9):.2f}")

    def straddle_book(dd, cond):
        """One straddle per event-hour on its FIRST qualifying loop."""
        pool = dd[cond].sort_values("dt")
        first_loops = pool.groupby("_event")["_loop"].min().rename("fl")
        pool = pool.merge(first_loops, on="_event")
        pool = pool[pool["_loop"] == pool["fl"]]
        legs = []
        for ev, g in pool.groupby("_event"):
            ylo = g[g["p_market"].between(*PM_TAIL_LO)]
            nhi = g[g["p_market"].between(*PM_TAIL_HI)]
            if ylo.empty or nhi.empty:
                continue
            yl = ylo.loc[ylo["p_market"].idxmin()]
            nh = nhi.loc[nhi["p_market"].idxmax()]
            for leg, side in ((yl, "yes"), (nh, "no")):
                cost = leg["p_market"] if side == "yes" else 1 - leg["p_market"]
                win = (leg["resolved_yes"] == 1) if side == "yes" else (leg["resolved_yes"] == 0)
                fee = 0.07 * leg["p_market"] * (1 - leg["p_market"])
                pnl = (100 * (1 - cost) / cost if win else -100.0) - (100 / cost) * fee
                legs.append({"dt": leg["dt"], "event": ev, "side": side,
                             "cost": cost, "win": bool(win), "pnl": pnl})
        return pd.DataFrame(legs)

    def summarize(bk, label):
        if bk is None or not len(bk):
            return f"{label}: n=0"
        wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
        wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
        ev = bk["event"].nunique()
        return (f"{label}: straddles={ev} legs={len(bk)} net=${bk['pnl'].sum():+,.0f} "
                f"legWR={bk['win'].mean():.1%} | {wks}")

    scr = d[d["dt"] < SPLIT]
    tst = d[d["dt"] >= SPLIT]
    print(f"\n[SELECT <07-09] threshold sweep + controls:")
    best_T, best_net = None, -1e18
    for T in [1.2, 1.5, 2.0, 3.0]:
        bk = straddle_book(scr, scr["vratio"] >= T)
        print("  ", summarize(bk, f"vratio>={T}"))
        if len(bk) >= 40:
            wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
            green = (wk[wk != 0] > 0).mean() if (wk != 0).any() else 0
            if bk["pnl"].sum() > best_net and green >= 0.6:
                best_T, best_net = T, bk["pnl"].sum()
    print("  ", summarize(straddle_book(scr, scr["vratio"] >= 0), "CONTROL: unconditional"))
    print("  ", summarize(straddle_book(scr, scr["vratio"] < 1.0), "CONTROL: vratio<1"))

    if best_T is None:
        print("\nNO screen threshold clears the bar (n>=40, net>0, >=60% weeks green) — "
              "vol-tail not viable through this predictor. Stopping honestly.")
        return
    print(f"\nCHOSEN T={best_T} → [FROZEN TEST 07-09..now, single shot]:")
    print("  ", summarize(straddle_book(tst, tst["vratio"] >= best_T), f"vratio>={best_T}"))
    print("  ", summarize(straddle_book(tst, tst["vratio"] >= 0), "control: unconditional"))
    print("  ", summarize(straddle_book(tst, tst["vratio"] < 1.0), "control: vratio<1"))


if __name__ == "__main__":
    main()
