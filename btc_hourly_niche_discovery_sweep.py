"""BTC hourly niche-discovery sweep: hunting more fat-edge pockets. 2026-08-02.

The branch doctrine (08-02): a family of independent niche books, each a
model-driven pocket with per-trade edge well above fees — not single-signal
books (rung-niche failed the fee bar) and not nested filters (sign-flip
lesson). The one validated member is niche v2 (YES, pm .35-.65, edge>=.06).
This sweep asks: does the SAME frozen niche v2 model identify OTHER pockets
— other pm bands, the NO side, higher conviction tiers?

Protocol:
  - Frozen model (btc_hourly_lgbm_niche_v2_20260728, train<06-20). No refits
    → no seed noise; the sweep is over BOOK DEFINITIONS only.
  - SCREEN window: 06-20..07-16 (model-OOS). Cells: side × pm band × edge
    tier. Survivor bar (pre-declared): n>=40, ALL screen weeks green, avg
    >= $8/trade, and edge-tier dose-response coherence within its band
    (higher conviction should not be materially worse).
  - TEST window: 07-16..08-02, ONE frozen shot for survivors only.
  - Multiplicity disclaimer: 24+ cells screened; even survivors then need
    forward paper before deployment. This finds candidates, not truths.
Flat $100, net of fees, one bet per contract.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import hourly_niche_runner_v2 as nv2

BASE = Path(__file__).parent
T0 = pd.Timestamp("2026-06-20", tz="UTC")   # model train end
T1 = pd.Timestamp("2026-07-16", tz="UTC")   # screen/test split
BANDS = [("yes", 0.20, 0.35), ("yes", 0.35, 0.65), ("yes", 0.65, 0.80),
         ("no", 0.20, 0.35), ("no", 0.35, 0.65), ("no", 0.65, 0.80)]
TIERS = [(0.06, 0.09), (0.09, 0.12), (0.12, 0.16), (0.16, 99.0)]


def main():
    with open(BASE / "models" / "btc_hourly_lgbm_niche_v2_20260728.pkl", "rb") as f:
        art = pickle.load(f)
    model, feats, slope_bases = art["model"], art["features"], art["slope_bases"]

    print("loading + prepping full BTC hourly archive…")
    df = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df = nv2.prep(df, feats, slope_bases)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["dt"] >= T0].reset_index(drop=True)

    print("scoring frozen model…")
    p = model.predict_proba(df[feats])[:, 1]
    fee = 0.07 * df["p_market"] * (1 - df["p_market"])
    df["edge_yes"] = p - df["p_market"] - fee
    df["edge_no"] = df["p_market"] - p - fee

    def cell_book(d, side, lo, hi, e0, e1):
        e = d["edge_yes"] if side == "yes" else d["edge_no"]
        q = d[d["p_market"].between(lo, hi) & (e >= e0) & (e < e1)].copy()
        q = q.sort_values("dt").drop_duplicates("contract_ticker", keep="first")
        cost = np.where(side == "yes", q["p_market"], 1 - q["p_market"])
        win = (q["resolved_yes"] == 1) if side == "yes" else (q["resolved_yes"] == 0)
        feeq = 0.07 * q["p_market"] * (1 - q["p_market"])
        q["pnl"] = np.where(win, 100 * (1 - cost) / cost, -100.0) - (100 / cost) * feeq
        q["win"] = win
        return q

    scr = df[df["dt"] < T1]
    tst = df[df["dt"] >= T1]
    print(f"screen scans={len(scr)}  test scans={len(tst)}\n")
    print("[SCREEN 06-20..07-16] side  band        tier        n    net     avg   WR    wk_green")
    survivors = []
    for side, lo, hi in BANDS:
        for e0, e1 in TIERS:
            b = cell_book(scr, side, lo, hi, e0, e1)
            if len(b) < 15:
                continue
            wk = b.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
            wk = wk[wk != 0]
            green = float((wk > 0).mean()) if len(wk) else 0.0
            avg = b["pnl"].mean()
            tag = ""
            if len(b) >= 40 and green >= 0.999 and avg >= 8.0:
                tag = "  ** SURVIVOR"
                survivors.append((side, lo, hi, e0, e1))
            print(f"  {side:3s}  {lo:.2f}-{hi:.2f}  {e0:.2f}-{e1:.2f}  "
                  f"{len(b):4d} {b['pnl'].sum():+8,.0f} {avg:+7.1f} "
                  f"{b['win'].mean():.0%}  {green:.0%}{tag}")

    print(f"\n[TEST 07-16..08-02] single frozen shot on {len(survivors)} survivor(s):")
    for side, lo, hi, e0, e1 in survivors:
        b = cell_book(tst, side, lo, hi, e0, e1)
        wk = b.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
        wks = "  ".join(f"{i.date()}:{v:+.0f}" for i, v in wk[wk != 0].items())
        print(f"  {side} {lo:.2f}-{hi:.2f} edge {e0:.2f}-{e1:.2f}: n={len(b)} "
              f"net=${b['pnl'].sum():+,.0f} avg=${b['pnl'].mean():+,.1f} "
              f"WR={b['win'].mean():.0%} | {wks}")


if __name__ == "__main__":
    main()
