"""BTC 15m pm-momentum VALUE test on backfilled candles. 2026-07-31.

The screen found pm_chg_5m partial-IC +0.038 (halves +.038/+.040) on mid
prices. IC is not PnL — this tests harvestability with honest costs:

PART A — standalone PnL sim:
  At each minute t (>=5 prior candles), if |pm_chg_5m| >= threshold, trade
  the MOMENTUM side once per market: YES at that minute's ASK (not mid),
  NO at 1 − BID. Hold to settlement, Kalshi fee 0.07*price*(1-price) per
  contract, flat $100 stake. Threshold/entry selection on JUNE only;
  JULY is a single frozen evaluation with weekly breakdown. Controls:
  next-minute entry (execution robustness) and contrarian direction.

PART B — incremental over the production model:
  Join candle-derived pm_chg_5m to logged 15m scans (paper_trades_btc15
  .csv) and partial-IC vs outcome controlling BOTH pm and p_model_15m —
  does book momentum add anything the model doesn't already know?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, pearsonr

BASE = Path(__file__).parent


def load_candles() -> pd.DataFrame:
    d = pd.read_csv(BASE / "results" / "kalshi_15m_candles_btc.csv", low_memory=False)
    for c in ["bid_close", "ask_close", "end_ts"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["end_ts", "bid_close", "ask_close"])
    d = d.sort_values(["ticker", "end_ts"]).reset_index(drop=True)
    d["mid"] = (d["bid_close"] + d["ask_close"]) / 2
    d["y"] = (d["result"].astype(str).str.lower() == "yes").astype(float)
    d["i"] = d.groupby("ticker").cumcount()
    return d


def sim(d: pd.DataFrame, thr: float, entry_delay: int = 0,
        contrarian: bool = False) -> pd.DataFrame:
    """One momentum trade per market: first minute i>=5 with |chg5|>=thr."""
    d = d.copy()
    d["mid5"] = d.groupby("ticker")["mid"].shift(5)
    d["chg5"] = d["mid"] - d["mid5"]
    # entry quote (possibly next minute for robustness)
    d["ask_e"] = d.groupby("ticker")["ask_close"].shift(-entry_delay)
    d["bid_e"] = d.groupby("ticker")["bid_close"].shift(-entry_delay)
    d["n_c"] = d.groupby("ticker")["i"].transform("max")
    elig = d[(d["i"] >= 5) & (d["i"] <= d["n_c"] - 1 - entry_delay)
             & d["chg5"].notna() & (d["chg5"].abs() >= thr)
             & d["mid"].between(0.10, 0.90)]
    t = elig.drop_duplicates("ticker", keep="first").copy()
    up = t["chg5"] > 0
    if contrarian:
        up = ~up
    t["side"] = np.where(up, "yes", "no")
    t["cost"] = np.where(up, t["ask_e"], 1 - t["bid_e"])
    t = t[t["cost"].between(0.02, 0.98)]
    win = np.where(t["side"] == "yes", t["y"] == 1, t["y"] == 0)
    fee = 0.07 * t["cost"] * (1 - t["cost"])
    t["pnl"] = np.where(win, 100 * (1 - t["cost"]) / t["cost"], -100.0) \
        - (100 / t["cost"]) * fee
    t["win"] = win
    t["dt"] = pd.to_datetime(t["close_time"], utc=True)
    return t


def summ(t, label):
    if not len(t):
        return f"{label}: n=0"
    wk = t.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
    wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
    return (f"{label}: n={len(t)} net=${t['pnl'].sum():+,.0f} "
            f"WR={t['win'].mean():.1%} BE={t['cost'].mean():.1%} | {wks}")


def main():
    d = load_candles()
    d["dt"] = pd.to_datetime(d["close_time"], utc=True)
    june = d[d["dt"] < pd.Timestamp("2026-07-01", tz="UTC")]
    july = d[d["dt"] >= pd.Timestamp("2026-07-01", tz="UTC")]
    print(f"markets: june={june['ticker'].nunique()} july={july['ticker'].nunique()}")

    print("\n[A] JUNE selection (momentum side, entry at ask/bid):")
    best, best_net = None, -1e9
    for thr in [0.02, 0.03, 0.05, 0.08]:
        t = sim(june, thr)
        print("   ", summ(t, f"thr={thr:.2f}"))
        if len(t) >= 100 and t["pnl"].sum() > best_net:
            best, best_net = thr, t["pnl"].sum()
    if best is None:
        print("\nNO June config with n>=100 positive — momentum not harvestable "
              "at the spread. Stopping honestly.")
        best = max([0.02, 0.03, 0.05, 0.08],
                   key=lambda th: sim(june, th)["pnl"].sum())
        print(f"(best June config regardless: thr={best})")
    print(f"\nCHOSEN thr={best} → JULY single frozen evaluation:")
    tj = sim(july, best)
    print("   ", summ(tj, "JULY momentum"))
    print("   ", summ(sim(july, best, entry_delay=1), "JULY next-minute entry"))
    print("   ", summ(sim(july, best, contrarian=True), "JULY contrarian (control)"))

    # ── PART B: incremental over production model ────────────────────────
    print("\n[B] incremental info over production p_model_15m (logged scans):")
    pt = pd.read_csv(BASE / "results" / "paper_trades_btc15.csv", low_memory=False)
    pt["dt"] = pd.to_datetime(pt["logged_at"], errors="coerce", utc=True)
    for c in ["p_market", "p_model_15m", "resolved_yes"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt = pt.dropna(subset=["dt", "p_market", "p_model_15m", "resolved_yes",
                           "contract_ticker"])
    pt["ts"] = pt["dt"].astype("int64") / 1e9
    cd = d[["ticker", "end_ts", "mid"]].copy()
    cd["chg5"] = cd.groupby("ticker")["mid"].shift(0) - cd.groupby("ticker")["mid"].shift(5)
    cd = cd.dropna(subset=["chg5"]).rename(columns={"ticker": "contract_ticker"})
    pt = pt.sort_values("ts")
    cd = cd.sort_values("end_ts")
    j = pd.merge_asof(pt, cd[["contract_ticker", "end_ts", "chg5"]],
                      left_on="ts", right_on="end_ts", by="contract_ticker",
                      direction="backward", tolerance=180)
    j = j.dropna(subset=["chg5"])
    print(f"    scans joined to candle momentum: {len(j)}")
    if len(j) >= 500:
        y = j["resolved_yes"].values
        v = j["chg5"].values
        def rank_resid(a, b):
            b = (b - b.mean()) / (b.std() + 1e-12)
            return a - a.mean() - np.dot(a - a.mean(), b) / len(b) * b
        rv, ry = rankdata(v).astype(float), rankdata(y).astype(float)
        for ctrl_cols, lbl in [(["p_market"], "pm only"),
                               (["p_market", "p_model_15m"], "pm + p_model")]:
            rv2, ry2 = rv.copy(), ry.copy()
            for cc in ctrl_cols:
                rc = rankdata(j[cc].values).astype(float)
                rv2, ry2 = rank_resid(rv2, rc), rank_resid(ry2, rc)
            r, p = pearsonr(rv2, ry2)
            print(f"    partial IC of pm_chg_5m ({lbl} controlled): "
                  f"{r:+.4f} p={p:.1e}")


if __name__ == "__main__":
    main()
