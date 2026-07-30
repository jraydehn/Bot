"""SOL hourly v4 — borrow the SOL 15m shadow-A feature recipe. 2026-07-30.

Idea (user): the 15m slope-shadow candidate works off a much richer feature
base than the hourly scan archive. Borrow it: point-in-time asof-join the 15m
scan stream's features (paper_trades_sol15m.csv rows, logged live) onto each
hourly scan, plus shadow-A-style D/S slopes computed on the 15m stream's own
history. Join uses logged_at of the 15m row <= hourly scan dt (backward,
45-min staleness cap) — zero lookahead.

Only 15m bases live since May are used (kc/kalman/hurst/donch/arima exist only
since 07-08 → would be all-NaN in train). vol_ratio_5m stuck at 1.0 pre-07-21
→ constant in train, LGBM ignores it; excluded for cleanliness.

Protocol identical to v3: train<06-25, VAL 06-25..07-09 (selection), untouched
TEST 07-09..07-30, flat $100, one bet/contract, net of fees. This is the 4th
shot at the same test window — pre-declared: only a strongly positive,
all-weeks-green result counts as signal; anything marginal is holdout decay.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import train_sol_hourly_niche_v3 as v3

BASE = Path(__file__).parent

M15_BASES = ["stoch_k_5m", "stoch_k_15m", "stoch_k_1h", "rsi_1h", "bb_pct_1h",
             "ema20_dist_1h", "ema50_dist_1h", "vwap_dist", "vol_ratio",
             "chg_5m", "chg_15m", "chg_1h", "bp_5m", "bp_15m", "bp_1h",
             "upper_wick_15m", "lower_wick_15m", "atr_ratio_15m",
             "realized_vol_annual", "z_drift_6h", "ls_long_pct", "oi_chg_pct"]
SLOPE_SUB = ["stoch_k_5m", "stoch_k_15m", "rsi_1h", "vwap_dist", "vol_ratio",
             "chg_5m", "bb_pct_1h", "ema20_dist_1h", "z_drift_6h",
             "realized_vol_annual", "atr_ratio_15m", "bp_15m"]


def build_m15_stream() -> pd.DataFrame:
    df = pd.read_csv(BASE / "results" / "paper_trades_sol15m.csv", low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"]).sort_values("dt")
    keep = ["dt", "spot"] + [c for c in M15_BASES if c in df.columns]
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    s = df[keep].drop_duplicates("dt", keep="last").reset_index(drop=True)
    # shadow-A-style slopes on the 15m stream's own history
    ts = s["dt"].astype("int64") / 1e9
    nc = {}
    for tag, sec in [("15", 900), ("45", 2700), ("120", 7200)]:
        idx = np.searchsorted(ts, ts - sec, side="right") - 1
        valid = idx >= 0
        pv = np.where(valid, s["spot"].values[np.clip(idx, 0, None)], np.nan)
        dp = pd.Series((s["spot"].values / pv - 1) * 100, index=s.index)
        for c in SLOPE_SUB:
            pr = np.where(valid, s[c].values[np.clip(idx, 0, None)], np.nan)
            d = s[c].values - pr
            nc[f"D{tag}_{c}"] = d
            nc[f"S{tag}_{c}"] = np.clip(d / dp.replace(0, np.nan), -50, 50)
    s = pd.concat([s, pd.DataFrame(nc, index=s.index)], axis=1)
    return s.drop(columns=["spot"])


def main():
    print("loading hourly archive…")
    df = v3.load_archive()
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)

    print("building 15m feature stream…")
    m15 = build_m15_stream()
    m15c = [c for c in m15.columns if c != "dt"]
    joined = pd.merge_asof(df, m15.rename(columns={c: f"m15_{c}" for c in m15c}),
                           on="dt", direction="backward",
                           tolerance=pd.Timedelta(minutes=45))
    m15f = [f"m15_{c}" for c in m15c]
    cov = joined[m15f[0]].notna().mean()
    print(f"join coverage (45min staleness cap): {cov:.1%} of {len(joined)} hourly scans")

    feats = v3.feature_list(extended=True) + m15f
    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    val = joined[(joined["dt"] >= T_END) & (joined["dt"] < V_END)]
    test = joined[joined["dt"] >= V_END]

    m = v3.train_model(joined, feats, T_END, T_END, V_END)
    print(f"trained: {len(feats)} feats, best_iter={m.best_iteration_}")
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    m15_share = imp[[f for f in imp.index if f.startswith('m15_')]].sum() / imp.sum()
    print(f"15m-borrowed feature importance share: {m15_share:.1%}")
    print("top 15:", list(imp.head(15).index))

    pv = m.predict_proba(val[feats])[:, 1]
    grid = []
    for side in ["yes", "no"]:
        for lo, hi in [(0.35, 0.65), (0.30, 0.70), (0.20, 0.80),
                       (0.50, 0.80), (0.20, 0.50)]:
            for em in [0.04, 0.06, 0.08]:
                bk = v3.sim_book(val, pv, side, lo, hi, em)
                if len(bk) < 30:
                    continue
                wk = bk.set_index("dt").groupby(pd.Grouper(freq="W-MON"))["pnl"].sum()
                grid.append(dict(side=side, lo=lo, hi=hi, em=em, n=len(bk),
                                 net=bk["pnl"].sum(),
                                 wk_green=(wk[wk != 0] > 0).mean(),
                                 wr=bk["winb"].mean(), be=bk["cost"].mean()))
    g = pd.DataFrame(grid).sort_values("net", ascending=False)
    print("\nVAL grid top:")
    print(g.head(8).round(3).to_string(index=False))
    ok = g[(g["wk_green"] >= 0.99) & (g["n"] >= 40)]
    if not len(ok):
        print("no all-green VAL config with n>=40")
        ok = g.head(1)
    b = ok.iloc[0]
    print(f"\nCHOSEN (val): {dict(b)}")

    pt = m.predict_proba(test[feats])[:, 1]
    bt = v3.sim_book(test, pt, b["side"], b["lo"], b["hi"], b["em"])
    print("\nFINAL TEST (single shot):")
    print("   ", v3.summarize(bt, "v4-15mborrow"))
    print("   ", v3.summarize(v3.sim_book(test, pt, "yes", 0.35, 0.65, 0.06),
                              "v4 @phase2cfg"))


if __name__ == "__main__":
    main()
