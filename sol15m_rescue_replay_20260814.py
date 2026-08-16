"""SOL 15m rescue candidates — PRE-REGISTERED frozen referee, 2026-08-14.

From an 11,049-test comprehensive sweep over the promoted stack's rejected
populations (slope-model era 07-29+). THREE survivors ~= the expected
false-positive rate at these bars, so all are MODEST/DIRECTIONAL — forward
rows (dt >= FREEZE) are the only decision evidence. Evaluate ~08-25.

  RESC-1: near-miss edge .02-.04 & kalman_residual_15m < 0   (MR: below fair)
  RESC-2: YES persist-blocked (edge>=.04, persist<3) & z_drift_6h < 0.59
  RESC-3: near-miss edge .02-.04 & d45_vol_ratio >= 0.18
"""
import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
FREEZE = pd.Timestamp("2026-08-14 07:30", tz="UTC")
df = pd.read_csv("results/paper_trades_sol15m.csv", low_memory=False)
df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True, format="mixed")
for c in ["p_market", "p_model_15m", "p_gbdt", "resolved_yes", "sol_persist_score",
          "kalman_residual_15m", "z_drift_6h", "d45_vol_ratio",
          "hurst_exponent_5m", "kalman_residual", "stoch_cross_1h",
          "oi_chg_pct", "slope120_stoch_k_15m", "offset_pct", "stoch_k_1h"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
SWAP = pd.Timestamp("2026-08-12 19:05", tz="UTC")
df["p_slope"] = np.where(df["dt"] >= SWAP, df["p_model_15m"], df["p_gbdt"])
start = FREEZE if "--forward" in sys.argv else pd.Timestamp("2026-07-29", tz="UTC")
ab = df[(df["dt"] >= start) & df["resolved_yes"].notna()
        & df["p_market"].between(0.03, 0.97)].dropna(subset=["p_slope"]).copy()
fee = 0.07 * ab["p_market"] * (1 - ab["p_market"])
ey = ab["p_slope"] - ab["p_market"] - fee
en = ab["p_market"] - ab["p_slope"] - fee
ab["side"] = np.where(ey >= en, "yes", "no")
ab["edge"] = np.maximum(ey, en)
q = ab.sort_values("dt").drop_duplicates("contract_ticker", keep="first").copy()
near = q["edge"].between(0.02, 0.04)
pblk = (q["side"] == "yes") & (q["edge"] >= 0.04) \
    & ~(q["sol_persist_score"] >= 3).fillna(False)
# [2026-08-16 addendum] From the user-directed 1,482-combo regime x signal
# grid (2 survivors ~= noise floor; frozen on mechanism + continuity):
#   RESC-3b: RESC-3 conditioned on 6h-markov Sideways — volume expansion
#            INSIDE A RANGE is breakout-initiation (70% WR, +$1,636,
#            p=0.006, 3/3wks, top 22%); explains raw RESC-3's wobble
#            (same signal in trend = late-chasing). RESC-3 kept for
#            comparison; 3b is the primary form at evaluation.
#   RESC-4:  markov-blocked NO rescued by hurst>=0.6 & kalman_resid<0
#            (persistent tape, price pinned under fair — the block
#            misfires; n=40, 55% WR, +$1,957, p=0.018, 3/3wks). LOWER
#            prior confidence: one validated signal overriding another.
_m6 = q["markov_sol_6h"].astype(str)
_mkv_no_blk = ((q["side"] == "no") & (q["edge"] >= 0.04)
               & ~(q["p_market"] > 0.8)
               & ((( _m6 == "Bull") & (q["offset_pct"].fillna(0) > -0.006))
                  | ((q["markov_sol_4h"].astype(str) == "Sideways")
                     & (q["stoch_k_1h"].fillna(50) < 90))))
BOOKS = {
    "RESC-1 nearmiss&kalman<0": near & (q["kalman_residual_15m"] < 0).fillna(False),
    "RESC-2 persistblk&zd<0.59": pblk & (q["z_drift_6h"] < 0.59).fillna(False),
    "RESC-3 nearmiss&d45vr>=.18": near & (q["d45_vol_ratio"] >= 0.18).fillna(False),
    "RESC-3b +6hSideways": (near & (q["d45_vol_ratio"] >= 0.18).fillna(False)
                            & (_m6 == "Sideways")),
    "RESC-4 mkvNO&H>=.6&kalm<0": (_mkv_no_blk
                                  & (q["hurst_exponent_5m"] >= 0.6).fillna(False)
                                  & (q["kalman_residual"] < 0).fillna(False)),
}
for nm, m in BOOKS.items():
    s = q[m]
    if not len(s):
        print(f"{nm}: no rows yet"); continue
    w = np.where(s["side"] == "yes", s["resolved_yes"] == 1, s["resolved_yes"] == 0)
    c = np.where(s["side"] == "yes", s["p_market"], 1 - s["p_market"])
    f2 = 0.07 * s["p_market"] * (1 - s["p_market"])
    pnl = np.where(w, 100 * (1 - c) / c, -100) - (100 / c) * f2
    wk = pd.Series(pnl, index=s.index).groupby(s["dt"].dt.isocalendar().week).sum()
    print(f"{nm}: n={len(s)} WR={w.mean():.0%} net=${pnl.sum():+,.0f} "
          f"weekly={{{', '.join(f'{int(k)}: {v:+.0f}' for k, v in wk.items())}}}")
