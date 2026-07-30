"""SOL hourly v5 — 4h-context features (user hypothesis). 2026-07-30.

Premise: 15m model leans on 1h context (9.2% importance, rsi_1h family top);
proportionately the hourly model should lean on 4h context. Logged 4h feats
are mostly broken (stoch_k_4h/rsi_4h stuck constant since inception — bug
noted; only chg_4h/bp_4h vary), so 4h context is computed honestly from the
hourly archive's own spot history (~5min cadence since 05-21):

  dprice_240/480, pos_in_range_4h/8h, rv_4h, rv_ratio_4h_24h,
  D240/S240 + D480/S480 slopes of key scan bases,
plus m15_chg_4h / m15_bp_4h (the two live logged 4h cols) via the v4 join,
and 240/480min slopes of m15_rsi_1h / m15_stoch_k_1h / m15_bb_pct_1h.

Protocol identical to v3/v4 (train<06-25, VAL-only selection 06-25..07-09,
untouched TEST 07-09..07-30). Fifth evaluation against this test window —
pre-declared: only a strong all-green test counts.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import train_sol_hourly_niche_v3 as v3
import train_sol_hourly_v4_15mborrow as v4

BASE = Path(__file__).parent

SLOPE_BASES_4H = ["stoch_k", "adx_1h", "rvol_1h", "composite_p_up",
                  "vwap_distance_pct", "ema_stretch_score", "no_score",
                  "ls_long_pct"]
M15_4H_SLOPE = ["rsi_1h", "stoch_k_1h", "bb_pct_1h"]


def add_4h_context(df: pd.DataFrame) -> tuple:
    """4h/8h context from the archive's own spot series (causal)."""
    snap = df.drop_duplicates("dt", keep="last")[["dt", "spot"] + SLOPE_BASES_4H]
    ts_s = snap["dt"].astype("int64").values / 1e9
    spot = snap["spot"].values
    nc = {}
    for tag, sec in [("240", 240 * 60), ("480", 480 * 60)]:
        idx = np.searchsorted(ts_s, ts_s - sec, side="right") - 1
        valid = idx >= 0
        pv = np.where(valid, spot[np.clip(idx, 0, None)], np.nan)
        dp = (spot / pv - 1) * 100
        nc[f"dprice_{tag}"] = dp
        for c in SLOPE_BASES_4H:
            cv = snap[c].values
            pr = np.where(valid, cv[np.clip(idx, 0, None)], np.nan)
            d = cv - pr
            nc[f"D{tag}_{c}"] = d
            with np.errstate(divide="ignore", invalid="ignore"):
                nc[f"S{tag}_{c}"] = np.clip(d / np.where(dp == 0, np.nan, dp), -50, 50)
    # position in trailing 4h/8h range + realized vol, via rolling windows
    s = pd.Series(spot, index=pd.DatetimeIndex(snap["dt"]))
    ret = s.pct_change()
    for tag, win in [("4h", "4h"), ("8h", "8h")]:
        lo = s.rolling(win).min()
        hi = s.rolling(win).max()
        nc[f"pos_in_range_{tag}"] = ((s - lo) / (hi - lo).replace(0, np.nan)).values
    rv4 = ret.rolling("4h").std()
    rv24 = ret.rolling("24h").std()
    nc["rv_4h"] = rv4.values
    nc["rv_ratio_4h_24h"] = (rv4 / rv24.replace(0, np.nan)).values
    ctx = pd.DataFrame(nc, index=snap.index)
    ctx["dt"] = snap["dt"]
    out = df.merge(ctx, on="dt", how="left")
    return out, list(nc.keys())


def add_m15_4h(m15: pd.DataFrame) -> tuple:
    """Add live 4h cols + 240/480min slopes of 1h signals to the 15m stream."""
    import pandas as _pd
    raw = _pd.read_csv(BASE / "results" / "paper_trades_sol15m.csv", low_memory=False)
    raw["dt"] = _pd.to_datetime(raw["logged_at"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["dt"]).sort_values("dt")
    for c in ["chg_4h", "bp_4h"]:
        raw[c] = _pd.to_numeric(raw[c], errors="coerce")
    ext = raw[["dt", "chg_4h", "bp_4h"]].drop_duplicates("dt", keep="last")
    m15 = m15.merge(ext, on="dt", how="left")
    ts = m15["dt"].astype("int64").values / 1e9
    nc = {}
    for tag, sec in [("240", 240 * 60), ("480", 480 * 60)]:
        idx = np.searchsorted(ts, ts - sec, side="right") - 1
        valid = idx >= 0
        for c in M15_4H_SLOPE:
            cv = m15[c].values
            pr = np.where(valid, cv[np.clip(idx, 0, None)], np.nan)
            nc[f"D{tag}_{c}"] = cv - pr
    m15 = pd.concat([m15, pd.DataFrame(nc, index=m15.index)], axis=1)
    return m15, ["chg_4h", "bp_4h"] + list(nc.keys())


def main():
    print("loading hourly archive…")
    df = v3.load_archive()
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    df, ctx_feats = add_4h_context(df)
    print(f"4h context feats: {len(ctx_feats)}")

    print("building 15m stream (+4h)…")
    m15 = v4.build_m15_stream()
    m15, m15_4h = add_m15_4h(m15)
    m15c = [c for c in m15.columns if c != "dt"]
    joined = pd.merge_asof(df, m15.rename(columns={c: f"m15_{c}" for c in m15c}),
                           on="dt", direction="backward",
                           tolerance=pd.Timedelta(minutes=45))
    m15f = [f"m15_{c}" for c in m15c]

    feats = v3.feature_list(extended=True) + ctx_feats + m15f
    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    val = joined[(joined["dt"] >= T_END) & (joined["dt"] < V_END)]
    test = joined[joined["dt"] >= V_END]

    m = v3.train_model(joined, feats, T_END, T_END, V_END)
    print(f"trained: {len(feats)} feats, best_iter={m.best_iteration_}")
    imp = pd.Series(m.feature_importances_, index=feats)
    fam4 = ctx_feats + [f"m15_{c}" for c in m15_4h]
    share4 = imp[[f for f in fam4 if f in imp.index]].sum() / imp.sum()
    print(f"4h-family importance share: {share4:.1%}")
    print("top 4h feats:", imp[[f for f in fam4 if f in imp.index]]
          .sort_values(ascending=False).head(8).round(0).to_dict())

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
                                 wk_green=(wk[wk != 0] > 0).mean()))
    g = pd.DataFrame(grid).sort_values("net", ascending=False)
    print("\nVAL grid top:")
    print(g.head(8).round(3).to_string(index=False))
    ok = g[(g["wk_green"] >= 0.99) & (g["n"] >= 40)]
    if not len(ok):
        print("no all-green VAL config")
        ok = g.head(1)
    b = ok.iloc[0]
    print(f"\nCHOSEN (val): {dict(b)}")

    pt = m.predict_proba(test[feats])[:, 1]
    print("\nFINAL TEST (single shot):")
    print("   ", v3.summarize(v3.sim_book(test, pt, b["side"], b["lo"], b["hi"], b["em"]),
                              "v5-4h"))
    print("   ", v3.summarize(v3.sim_book(test, pt, "yes", 0.35, 0.65, 0.06),
                              "v5 @phase2cfg"))


if __name__ == "__main__":
    main()
