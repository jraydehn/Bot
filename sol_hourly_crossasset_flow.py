"""Cross-asset Kalshi book flow → SOL hourly features. 2026-07-30.

Item 4 of the feature-gap inventory: the BTC/ETH hourly books scan on the
same clock and are deeper than SOL's — their book DYNAMICS have never been
SOL features. This taps a new information source (another market's order
flow) rather than re-slicing SOL's own data.

Per neighbor asset (BTC, ETH), a loop-level (2-min) series of strike-free
book summaries:
  X_imp_median_dist   ladder-implied median settle vs spot (book's expected
                      drift for the hour), via the same interp as the
                      microstructure module
  X_imp_width_pct     implied 2σ width (book's implied vol)
  X_book_pm_chg15     mean within-contract 15-min pm change across live
                      contracts (book-wide repricing pressure)
  X_dprice_15         spot 15-min change (context / control)
plus 15-min changes of the first two, and SOL-minus-X drift differentials.

Honesty rails (lessons applied):
  - Within-book momentum only: each book's pm compared to ITS OWN earlier
    pm — never a stale model output against fresh prices (07-30 artifact).
  - asof-backward join, 10-min staleness cap.
  - SKIP-ONE-LOOP robustness: re-screen with the neighbor series lagged one
    full loop; a real signal survives, a timing artifact dies (lead-lag
    lesson, stale-print class).
  - Screen on PRE-07-09 data only, partial IC controlling for SOL pm,
    split-half stability. Burned window untouched; Aug stays fresh.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, pearsonr

import train_sol_hourly_niche_v3 as v3

BASE = Path(__file__).parent


def build_book_series(asset: str) -> pd.DataFrame:
    """Loop-level strike-free book summaries for one asset's hourly archive."""
    df = pd.read_csv(BASE / "results" / f"{asset}_scan_archive.csv",
                     usecols=["logged_at", "contract_ticker", "p_market",
                              "spot", "strike"], low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    for c in ["p_market", "spot", "strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["p_market", "spot", "strike"])
    df = df[df["p_market"].between(0.02, 0.98)]
    df["_event"] = df["contract_ticker"].astype(str).str.rsplit("-T", n=1).str[0]
    df["_loop"] = df["dt"].dt.floor("2min")

    # per-contract 15-min pm change (own earlier scans only)
    df["pm_chg15"] = np.nan
    for _, g in df.groupby("contract_ticker", sort=False):
        ts = g["dt"].astype("int64").values / 1e9
        pm = g["p_market"].values
        idx = np.searchsorted(ts, ts - 900, side="right") - 1
        ok = idx >= 0
        vals = np.full(len(g), np.nan)
        vals[ok] = pm[ok] - pm[np.clip(idx, 0, None)][ok]
        df.loc[g.index, "pm_chg15"] = vals

    rows = []
    for (loop, _), g in df.groupby(["_loop", "_event"], sort=False):
        g = g.sort_values("dt").drop_duplicates("strike", keep="last").sort_values("strike")
        if len(g) < 3:
            continue
        k = g["strike"].values
        pm_m = np.minimum.accumulate(g["p_market"].values)
        spot = float(g["spot"].iloc[-1])
        def k_at(p):
            if pm_m[0] <= p or pm_m[-1] >= p:
                return np.nan
            return float(np.interp(-p, -pm_m, k))
        k50, k16, k84 = k_at(0.50), k_at(0.16), k_at(0.84)
        rows.append({
            "loop": loop,
            "imp_median_dist": (k50 - spot) / spot * 100 if not np.isnan(k50) else np.nan,
            "imp_width_pct": (k16 - k84) / spot * 100
                             if not (np.isnan(k16) or np.isnan(k84)) else np.nan,
            "book_pm_chg15": float(g["pm_chg15"].mean()),
            "spot": spot,
        })
    s = pd.DataFrame(rows).groupby("loop").mean().reset_index()
    s = s.sort_values("loop").reset_index(drop=True)
    ts = s["loop"].astype("int64").values / 1e9
    idx = np.searchsorted(ts, ts - 900, side="right") - 1
    ok = idx >= 0
    for c in ["imp_median_dist", "imp_width_pct"]:
        v = s[c].values
        d = np.full(len(s), np.nan)
        d[ok] = v[ok] - v[np.clip(idx, 0, None)][ok]
        s[f"{c}_chg15"] = d
    dp = np.full(len(s), np.nan)
    sp = s["spot"].values
    dp[ok] = (sp[ok] / sp[np.clip(idx, 0, None)][ok] - 1) * 100
    s["dprice_15"] = dp
    return s.drop(columns=["spot"])


def join_neighbor(sol: pd.DataFrame, series: pd.DataFrame, prefix: str,
                  lag_loops: int = 0) -> tuple:
    s = series.copy().sort_values("loop")
    if lag_loops:
        cols = [c for c in s.columns if c != "loop"]
        s[cols] = s[cols].shift(lag_loops)
    s = s.rename(columns={c: f"{prefix}_{c}" for c in s.columns if c != "loop"})
    out = pd.merge_asof(sol.sort_values("dt"), s, left_on="dt", right_on="loop",
                        direction="backward", tolerance=pd.Timedelta(minutes=10))
    return out.drop(columns=["loop"]), \
        [c for c in s.columns if c != "loop"]


def partial_ic(v, y, ctrl):
    ok = ~(np.isnan(v) | np.isnan(y) | np.isnan(ctrl))
    if ok.sum() < 500:
        return np.nan, np.nan, int(ok.sum())
    rv_, rc, ry = rankdata(v[ok]), rankdata(ctrl[ok]), rankdata(y[ok])
    def resid(a, b):
        b = (b - b.mean()) / (b.std() + 1e-12)
        return a - a.mean() - np.dot(a - a.mean(), b) / len(b) * b
    r, p = pearsonr(resid(rv_, rc), resid(ry, rc))
    return r, p, int(ok.sum())


def main():
    print("building neighbor book series…")
    series = {a: build_book_series(a) for a in ["btc", "eth"]}
    for a, s in series.items():
        s.to_parquet(BASE / "results" / f"{a}_hourly_book_series_20260730.parquet",
                     index=False)
        print(f"  {a}: {len(s)} loops  {s['loop'].min()} → {s['loop'].max()}")

    sol = v3.load_archive()
    sol = sol.dropna(subset=["resolved_yes", "p_market"])
    sol = sol[sol["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)

    # SOL's own book drift for differentials (same construction)
    sol_series = build_book_series("sol")
    sol, sol_cols = join_neighbor(sol, sol_series, "sol")

    pre_mask = sol["dt"] < pd.Timestamp("2026-07-09", tz="UTC")
    mid = pd.Timestamp("2026-06-14", tz="UTC")

    for lag, tag in [(0, "FRESH (latest completed loop)"), (1, "SKIP-ONE-LOOP")]:
        d = sol.copy()
        feats = []
        for a in ["btc", "eth"]:
            d, cols = join_neighbor(d, series[a], a, lag_loops=lag)
            feats += cols
        d["btc_sol_drift_diff"] = d["sol_imp_median_dist"] - d["btc_imp_median_dist"]
        d["eth_sol_drift_diff"] = d["sol_imp_median_dist"] - d["eth_imp_median_dist"]
        feats += ["btc_sol_drift_diff", "eth_sol_drift_diff"]

        y = (d["resolved_yes"] - d["p_market"]).values
        pm = d["p_market"].values
        print(f"\n[{tag}] partial IC vs SOL (outcome − pm), pm-controlled, pre-07-09:")
        for c in feats:
            v = d[c].values.astype(float)
            ic, p, n = partial_ic(v[pre_mask.values], y[pre_mask.values], pm[pre_mask.values])
            h = []
            for hm in [pre_mask & (d["dt"] < mid), pre_mask & (d["dt"] >= mid)]:
                h.append(partial_ic(v[hm.values], y[hm.values], pm[hm.values])[0])
            flag = " **" if not np.isnan(ic) and abs(ic) > 0.03 and p < 1e-4 \
                   and not any(np.isnan(x) for x in h) and np.sign(h[0]) == np.sign(h[1]) else ""
            print(f"  {c:26s} IC={ic:+.4f} p={p:.1e} n={n}  halves={h[0]:+.3f}/{h[1]:+.3f}{flag}")

        if lag == 0:
            keep = d[["dt", "contract_ticker"] + feats]
            keep.to_parquet(BASE / "results" / "sol_hourly_crossasset_20260730.parquet",
                            index=False)


if __name__ == "__main__":
    main()
