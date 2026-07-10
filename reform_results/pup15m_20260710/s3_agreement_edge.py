"""
S3 -- join the NEW p15 signal to the BTC hourly scan archive and rerun the
agreement-edge analysis (the btc_vwap_fresh s3 methodology, new signal).

Signal: p15 from tables trained on 2023-10..2025-12 -> the 2026 archive is
fully out-of-sample. Join is zero-lookahead: archive row at logged_at uses
the latest p15 whose EFFECTIVE time (bar_open + 15m) <= logged_at.

Agreement semantics (identical to prior run):
  YES candidate: 15m AGREES if dir15=up, DISAGREES if dir15=down.
  NO  candidate: mirror.
dir15 from p15 with a neutral band: up if p15>0.5+B, down if p15<0.5-B.
Edge (per side): YES edge = resolved_yes - p_market; NO edge = p_market - resolved_yes.
Clustering: ticker-level means, ticker bootstrap for P(<=0), week-positive count.
Era split: pre vs post the 06-30 15m non-coherent reform (and the archive's
own model eras don't matter here -- p_market is the market, not our model).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"
BAND = 0.02          # neutral band around 0.5
REFORM = pd.Timestamp("2026-06-30 00:00:00", tz="UTC")

sig = pd.read_csv(f"{OUT}/pup15m_series_2026.csv", parse_dates=["bar_open", "effective"])
sig = sig.sort_values("effective").reset_index(drop=True)

arc = pd.read_csv("results/btc_scan_archive.csv",
                  usecols=["logged_at", "contract_ticker", "p_market", "tau_minutes",
                           "offset_pct", "resolved_yes"], low_memory=False)
arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True, errors="coerce", format="mixed")
arc = arc.dropna(subset=["logged_at", "p_market", "resolved_yes"])
arc = arc[arc["logged_at"] >= sig["effective"].min()]
arc = arc.sort_values("logged_at").reset_index(drop=True)
print(f"archive rows in signal window: {len(arc)}  tickers: {arc['contract_ticker'].nunique()}")
print(f"window: {arc['logged_at'].min()} -> {arc['logged_at'].max()}")

# zero-lookahead asof join on effective time
arc = pd.merge_asof(arc, sig[["effective", "p15"]],
                    left_on="logged_at", right_on="effective", direction="backward")
arc = arc.dropna(subset=["p15"])
arc["staleness_min"] = (arc["logged_at"] - arc["effective"]).dt.total_seconds() / 60
print(f"p15 staleness at join: median={arc['staleness_min'].median():.1f}min  "
      f"p95={arc['staleness_min'].quantile(0.95):.1f}min")
arc["dir15"] = np.where(arc["p15"] > 0.5 + BAND, 1, np.where(arc["p15"] < 0.5 - BAND, -1, 0))
arc["era"] = np.where(arc["logged_at"] < REFORM, "pre", "post")
arc["week"] = arc["logged_at"].dt.to_period("W").astype(str)
arc.to_csv(f"{OUT}/hourly_archive_p15.csv", index=False)

rng = np.random.default_rng(7)
def tk_stats(sub, side):
    if side == "yes":
        sub = sub.assign(edge=sub["resolved_yes"] - sub["p_market"])
    else:
        sub = sub.assign(edge=sub["p_market"] - sub["resolved_yes"])
    tk = sub.groupby("contract_ticker").agg(edge=("edge", "mean"), week=("week", "first"))
    if len(tk) < 5:
        return len(sub), len(tk), np.nan, np.nan, ""
    boots = [tk["edge"].sample(frac=1, replace=True, random_state=i).mean() for i in range(2000)]
    p_le0 = np.mean(np.array(boots) <= 0)
    wk = tk.groupby("week")["edge"].mean()
    return len(sub), len(tk), tk["edge"].mean(), p_le0, f"wk+={int((wk>0).sum())}/{len(wk)}"

print("\n=== WITHIN-SIDE edge by p15 agreement (ticker-clustered), per era ===")
for side in ["yes", "no"]:
    want = 1 if side == "yes" else -1
    for era in ["pre", "post"]:
        e = arc[arc["era"] == era]
        for name, mask in [("AGREES", e["dir15"] == want), ("NEUTRAL", e["dir15"] == 0),
                           ("DISAGREES", e["dir15"] == -want)]:
            n, tkn, edge, p, wk = tk_stats(e[mask], side)
            print(f"  {side.upper():3s} {era:4s} 15m {name:9s}: rows={n:6d} tickers={tkn:5d} "
                  f"tk_edge={edge:+.4f} P(<=0)={p:.4f} {wk}")

print("\n=== signed-strength gradient (p15 distance from 0.5, signed toward side) ===")
for side in ["yes", "no"]:
    s = arc.copy()
    s["support"] = (s["p15"] - 0.5) * (1 if side == "yes" else -1)
    for name, mask in [("strong support (>0.04)", s["support"] > 0.04),
                       ("mild support (0,0.04]", (s["support"] > 0) & (s["support"] <= 0.04)),
                       ("mild oppose (-0.04,0]", (s["support"] <= 0) & (s["support"] > -0.04)),
                       ("strong oppose (<=-0.04)", s["support"] <= -0.04)]:
        n, tkn, edge, p, wk = tk_stats(s[mask], side)
        print(f"  {side.upper():3s} {name:24s}: tickers={tkn:5d} tk_edge={edge:+.4f} P(<=0)={p:.4f} {wk}")
print("DONE_S3")
