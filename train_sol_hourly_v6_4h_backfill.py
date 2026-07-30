"""SOL hourly v6 — backfilled real stoch_k_4h/rsi_4h. 2026-07-30.

The logged stoch_k_4h/rsi_4h never computed (fixed 74a5885, history starts
07-30). Unlike HMM states these ARE safely backfillable: deterministic
functions of completed 4h OHLC bars only — value at scan time t uses bars
whose CLOSE <= t (containing-bar rule), no fitted parameters, no future.

Backfill: Binance SOL 4h klines (500 bars, 05-08..07-30) → expanding
period-14 stoch_k/rsi per completed bar, known_at = bar close. Saved to
results/binance_4h_indicators_sol_backfill_20260730.csv for reuse (15m
retrains later). Joined to hourly scans backward on known_at, plus D240/D480
deltas of each indicator (change since 4h/8h ago).

v6 = v5 feature set + these 6. Same protocol (train<06-25, VAL-only selection
06-25..07-09, untouched TEST 07-09..07-30). SIXTH shot at this test window —
pre-declared: only a strong all-green test is signal.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import train_sol_hourly_niche_v3 as v3
import train_sol_hourly_v4_15mborrow as v4
from train_sol_hourly_v5_4h import add_4h_context, add_m15_4h

BASE = Path(__file__).parent
BF_PATH = BASE / "results" / "binance_4h_indicators_sol_backfill_20260730.csv"


def build_backfill() -> pd.DataFrame:
    if BF_PATH.exists():
        bf = pd.read_csv(BF_PATH)
        bf["known_at"] = pd.to_datetime(bf["known_at"], utc=True)
        return bf
    from live_signal import fetch_recent_candles
    k = fetch_recent_candles("4h", 500, asset="SOL")
    if k is None or len(k) < 30:
        raise RuntimeError("4h kline fetch failed")
    k = k.sort_index()
    # per completed bar i: indicators over bars ..i, known at bar close
    ll = k["low"].rolling(14).min()
    hh = k["high"].rolling(14).max()
    rng = (hh - ll).replace(0, np.nan)
    stoch = ((k["close"] - ll) / rng) * 100
    delta = k["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
    r4 = k["high"] - k["low"]
    bf = pd.DataFrame({
        "known_at": k.index + pd.Timedelta(hours=4),  # bar CLOSE time
        "stoch_k_4h_bf": stoch.values,
        "rsi_4h_bf": rsi.values,
        "chg_4h_bf": ((k["close"] / k["open"] - 1) * 100).values,
        "bp_4h_bf": np.where(r4 > 0, (k["close"] - k["low"]) / r4, 0.5),
    }).dropna(subset=["stoch_k_4h_bf", "rsi_4h_bf"]).reset_index(drop=True)
    # deltas of the indicator series itself (1 bar = 4h back, 2 bars = 8h)
    bf["D240_stoch_k_4h_bf"] = bf["stoch_k_4h_bf"].diff(1)
    bf["D480_stoch_k_4h_bf"] = bf["stoch_k_4h_bf"].diff(2)
    bf["D240_rsi_4h_bf"] = bf["rsi_4h_bf"].diff(1)
    bf["D480_rsi_4h_bf"] = bf["rsi_4h_bf"].diff(2)
    bf.to_csv(BF_PATH, index=False)
    print(f"backfill saved: {len(bf)} bars {bf['known_at'].min()} → {bf['known_at'].max()}")
    return bf


def main():
    bf = build_backfill()
    bf_feats = [c for c in bf.columns if c != "known_at"]

    print("loading hourly archive…")
    df = v3.load_archive()
    df = v3.add_slopes(df)
    df = v3.add_extended(df)
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    df, ctx_feats = add_4h_context(df)

    m15 = v4.build_m15_stream()
    m15, m15_4h = add_m15_4h(m15)
    m15c = [c for c in m15.columns if c != "dt"]
    df = pd.merge_asof(df, m15.rename(columns={c: f"m15_{c}" for c in m15c}),
                       on="dt", direction="backward",
                       tolerance=pd.Timedelta(minutes=45))
    df = pd.merge_asof(df, bf.sort_values("known_at"), left_on="dt",
                       right_on="known_at", direction="backward")
    print(f"backfill join coverage: {df['stoch_k_4h_bf'].notna().mean():.1%}")

    feats = (v3.feature_list(extended=True) + ctx_feats
             + [f"m15_{c}" for c in m15c] + bf_feats)
    T_END = pd.Timestamp("2026-06-25", tz="UTC")
    V_END = pd.Timestamp("2026-07-09", tz="UTC")
    val = df[(df["dt"] >= T_END) & (df["dt"] < V_END)]
    test = df[df["dt"] >= V_END]

    m = v3.train_model(df, feats, T_END, T_END, V_END)
    imp = pd.Series(m.feature_importances_, index=feats)
    print(f"trained: {len(feats)} feats, best_iter={m.best_iteration_}")
    print(f"backfilled-4h importance share: {imp[bf_feats].sum() / imp.sum():.1%}")
    print("bf feats ranked:", imp[bf_feats].sort_values(ascending=False).round(0).to_dict())

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
    print(g.head(6).round(3).to_string(index=False))
    ok = g[(g["wk_green"] >= 0.99) & (g["n"] >= 40)]
    if not len(ok):
        print("no all-green VAL config")
        ok = g.head(1)
    b = ok.iloc[0]
    print(f"\nCHOSEN (val): {dict(b)}")

    pt = m.predict_proba(test[feats])[:, 1]
    print("\nFINAL TEST (single shot):")
    print("   ", v3.summarize(v3.sim_book(test, pt, b["side"], b["lo"], b["hi"], b["em"]),
                              "v6-4h-backfill"))
    print("   ", v3.summarize(v3.sim_book(test, pt, "yes", 0.35, 0.65, 0.06),
                              "v6 @phase2cfg"))


if __name__ == "__main__":
    main()
