"""rung_resid overlay for the BTC hourly YES-niche book. 2026-08-02.

The strongest signal of the book-dynamics campaign (BTC hourly partial IC
+0.124, halves +.114/+.144, n=128k) applied in the form the evidence
supports: a narrow overlay on the ALREADY-VALIDATED niche program, not a
broad model. rung_resid > 0 = this rung is priced above its ladder-neighbor
interpolation; the campaign screen showed outcomes follow the deviating
rung (it's the informed quote).

For a YES book: trades on rungs priced BELOW their ladder (rung_resid < 0)
are fighting the informed quote — hypothesis: they underperform.

Protocol:
  1. Replay the FROZEN niche v2 model (btc_hourly_lgbm_niche_v2_20260728,
     train<06-20) over the full BTC hourly archive → the niche book's
     historical trade population (YES, pm .35-.65, fee-adj edge>=.06, one
     bet per contract, flat $100 net of fees). No model refitting.
  2. Compute rung_resid per trade (kalshi_microstructure_features,
     point-in-time by construction).
  3. TRAIN window: trades before 07-16 — inspect PnL by rung_resid bucket,
     pre-register ONE simple rule if the signal is there.
  4. TEST window: trades 07-16..now — single frozen shot of that rule.
Replay trades before 06-20 are in the niche model's own training period —
excluded from BOTH windows (in-sample for the model).
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from hourly_book_findings_screen import load_hourly
from kalshi_microstructure_features import build_micro_features
import hourly_niche_runner_v2 as nv2

BASE = Path(__file__).parent
MODEL_TRAIN_END = pd.Timestamp("2026-06-20", tz="UTC")
SPLIT = pd.Timestamp("2026-07-16", tz="UTC")
PM_LO, PM_HI, EDGE_MIN = 0.35, 0.65, 0.06


def main():
    with open(BASE / "models" / "btc_hourly_lgbm_niche_v2_20260728.pkl", "rb") as f:
        art = pickle.load(f)
    model, feats, slope_bases = art["model"], art["features"], art["slope_bases"]

    print("loading full BTC hourly archive…")
    df = pd.read_csv(BASE / "results" / "btc_scan_archive.csv", low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df = nv2.prep(df, feats, slope_bases)
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df.dropna(subset=["resolved_yes", "p_market"]).reset_index(drop=True)

    print("computing microstructure features (rung_resid)…")
    micro = build_micro_features(df)
    df["rung_resid"] = micro["rung_resid"]

    print("replaying frozen niche v2 book…")
    cand = df[df["p_market"].between(PM_LO, PM_HI)].copy()
    p = model.predict_proba(cand[feats])[:, 1]
    fee = 0.07 * cand["p_market"] * (1 - cand["p_market"])
    cand["edge"] = p - cand["p_market"] - fee
    trades = cand[cand["edge"] >= EDGE_MIN].sort_values("dt").drop_duplicates(
        "contract_ticker", keep="first").copy()
    win = trades["resolved_yes"] == 1
    feeq = 0.07 * trades["p_market"] * (1 - trades["p_market"])
    trades["pnl"] = np.where(win, 100 * (1 - trades["p_market"]) / trades["p_market"],
                             -100.0) - (100 / trades["p_market"]) * feeq
    trades["win"] = win
    trades = trades[trades["dt"] >= MODEL_TRAIN_END]  # OOS for the model
    print(f"niche trades (post-06-20, model-OOS): {len(trades)}  "
          f"rung_resid coverage: {trades['rung_resid'].notna().mean():.1%}")

    tr = trades[trades["dt"] < SPLIT]
    te = trades[trades["dt"] >= SPLIT]
    print(f"\nbaseline: TRAIN(06-20..07-16) n={len(tr)} net=${tr['pnl'].sum():+,.0f} "
          f"WR={tr['win'].mean():.1%} | TEST(07-16..) n={len(te)} "
          f"net=${te['pnl'].sum():+,.0f} WR={te['win'].mean():.1%}")

    print("\n[TRAIN] PnL by rung_resid bucket:")
    trc = tr.dropna(subset=["rung_resid"]).copy()
    trc["bucket"] = pd.cut(trc["rung_resid"],
                           [-np.inf, -0.05, 0.0, 0.05, np.inf],
                           labels=["<-0.05", "-0.05..0", "0..0.05", ">0.05"])
    g = trc.groupby("bucket", observed=True).agg(
        n=("pnl", "size"), net=("pnl", "sum"), wr=("win", "mean"),
        avg=("pnl", "mean"))
    print(g.round(2).to_string())
    nan_tr = tr[tr["rung_resid"].isna()]
    print(f"  (no-ladder trades: n={len(nan_tr)} net=${nan_tr['pnl'].sum():+,.0f} "
          f"avg=${nan_tr['pnl'].mean():+,.1f})")

    # split-half stability of the train-window pattern
    mid = tr["dt"].min() + (tr["dt"].max() - tr["dt"].min()) / 2
    for lbl, m in [("H1", trc["dt"] < mid), ("H2", trc["dt"] >= mid)]:
        h = trc[m]
        neg = h[h["rung_resid"] < 0]
        pos = h[h["rung_resid"] >= 0]
        print(f"  {lbl}: rung<0 avg=${neg['pnl'].mean():+,.1f} (n={len(neg)}) | "
              f"rung>=0 avg=${pos['pnl'].mean():+,.1f} (n={len(pos)})")

    # ── pre-registered rule (fixed BEFORE looking at test): skip YES niche
    # trades with rung_resid < 0 (rung priced below its ladder = the
    # informed quote leans against YES). NaN rung (no ladder) → keep.
    print("\n[TEST 07-16.., single frozen shot] rule: skip if rung_resid < 0")
    kept = te[~(te["rung_resid"] < 0)]
    skipped = te[te["rung_resid"] < 0]
    print(f"  kept:    n={len(kept)} net=${kept['pnl'].sum():+,.0f} "
          f"WR={kept['win'].mean():.1%} avg=${kept['pnl'].mean():+,.1f}")
    print(f"  skipped: n={len(skipped)} net=${skipped['pnl'].sum():+,.0f} "
          f"WR={skipped['win'].mean():.1%} avg=${skipped['pnl'].mean():+,.1f}")
    print(f"  book with rule: ${kept['pnl'].sum():+,.0f} vs without: "
          f"${te['pnl'].sum():+,.0f}  (delta ${kept['pnl'].sum()-te['pnl'].sum():+,.0f})")
    trades.to_csv(BASE / "results" / "btc_niche_rungresid_replay.csv", index=False)


if __name__ == "__main__":
    main()
