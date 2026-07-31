"""Replicate the SOL hourly book-dynamics findings on BTC (or ETH). 2026-07-31.

Runs the identical screens that produced the SOL survivors, with the same
discipline (pre-07-09 window, pm-controlled partial IC, split halves at
06-14, skip-one-loop robustness for cross-book):

  1. Own-book microstructure: pm trajectory + ladder (kalshi_microstructure_
     features).
  2. Cross-book flow: the other two assets' ladder-drift/width/pm-repricing
     vs this asset (drift DIFFS included), fresh + skip-one-lagged.
  3. Liquidation family via Coinalyze 1h backfill for this asset's perp.
  4. Settlement-regime velocity (regvel of trailing settled-YES rate).

Targets: DIRECTION = (resolved_yes − pm) | pm;  VOL = |price_move_pct| | rv4h.

Usage: python3 hourly_book_findings_screen.py BTC
"""
import sys
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.stats import rankdata, pearsonr

import coinalyze_liq as cl
import sol_hourly_crossasset_flow as xa
from kalshi_microstructure_features import build_micro_features, LADDER_COLS, TRAJ_COLS

BASE = Path(__file__).parent
ASSETS = ["btc", "eth", "sol"]


def load_hourly(asset: str) -> pd.DataFrame:
    df = pd.read_csv(BASE / "results" / f"{asset}_scan_archive.csv",
                     usecols=["logged_at", "contract_ticker", "p_market", "spot",
                              "strike", "tau_minutes", "resolved_yes",
                              "price_move_pct", "close_ts"], low_memory=False)
    df["dt"] = pd.to_datetime(df["logged_at"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    for c in ["p_market", "spot", "strike", "tau_minutes", "resolved_yes",
              "price_move_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_ts"].astype(str).str.replace(
        r"\+00:00$", "", regex=True), errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["resolved_yes", "p_market"])
    return df[df["p_market"].between(0.02, 0.98)].reset_index(drop=True)


def add_recent_yes(df: pd.DataFrame) -> pd.DataFrame:
    settled = (df.dropna(subset=["close_dt", "resolved_yes"])
                 .drop_duplicates("contract_ticker", keep="first")
                 .sort_values("close_dt"))
    known = (settled["close_dt"] + pd.Timedelta(minutes=5)).astype("int64").values / 1e9
    cum = np.concatenate([[0.0], np.cumsum(settled["resolved_yes"].values.astype(float))])
    ts = df["dt"].astype("int64").values / 1e9
    for tag, sec in [("6h", 21600), ("24h", 86400)]:
        hi = np.searchsorted(known, ts, side="right")
        lo = np.searchsorted(known, ts - sec, side="right")
        n = hi - lo
        df[f"recent_yes_{tag}"] = np.where(n >= 3, (cum[hi] - cum[lo]) / np.maximum(n, 1), np.nan)
    snap = df.drop_duplicates("dt", keep="last")
    tss = snap["dt"].astype("int64").values / 1e9
    ry = snap["recent_yes_6h"].values
    ctx = pd.DataFrame(index=snap.index)
    ctx["dt"] = snap["dt"]
    for tag, sec in [("45", 2700), ("120", 7200)]:
        idx = np.searchsorted(tss, tss - sec, side="right") - 1
        ctx[f"regvel{tag}_recent_yes"] = ry - np.where(idx >= 0, ry[np.clip(idx, 0, None)], np.nan)
    s = pd.Series(snap["spot"].values, index=pd.DatetimeIndex(snap["dt"]))
    ctx["rv_4h_ctl"] = s.pct_change().rolling("4h").std().values
    return df.merge(ctx, on="dt", how="left")


def fetch_liq(asset: str) -> pd.DataFrame:
    now = int(time.time())
    r = requests.get(f"{cl._BASE}/liquidation-history", params={
        "symbols": cl._SYMBOLS[asset.upper()], "interval": "1hour",
        "from": now - 85 * 24 * 3600, "to": now, "api_key": cl._API_KEY}, timeout=20)
    r.raise_for_status()
    d = pd.DataFrame(r.json()[0]["history"])
    d["known_at"] = pd.to_datetime(d["t"], unit="s", utc=True) + pd.Timedelta(hours=1)
    d = d.rename(columns={"l": "liq_long", "s": "liq_short"}).sort_values("known_at")
    f = pd.DataFrame({"known_at": d["known_at"]})
    tot = d["liq_long"] + d["liq_short"]
    f["liq_total_z"] = tot.rolling(168).rank(pct=True)
    f["liq_long_z"] = d["liq_long"].rolling(168).rank(pct=True)
    f["liq_short_z"] = d["liq_short"].rolling(168).rank(pct=True)
    f["liq_imbalance"] = (d["liq_long"] - d["liq_short"]) / tot.replace(0, np.nan)
    for w in (6, 12):
        f[f"liq_imbalance_trend{w}"] = (f["liq_imbalance"] - f["liq_imbalance"].shift(w)) / w
    return f


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


def screen(df, feats, y, ctrl, pre, h1, h2, title):
    print(f"\n[{title}]")
    for c in feats:
        v = df[c].values.astype(float)
        ic, p, n = partial_ic(v[pre], y[pre], ctrl[pre])
        if np.isnan(ic):
            print(f"  {c:26s} insufficient data (n={n})")
            continue
        a = partial_ic(v[h1], y[h1], ctrl[h1])[0]
        b = partial_ic(v[h2], y[h2], ctrl[h2])[0]
        flag = " **" if (p < 1e-4 and abs(ic) > 0.03 and not (np.isnan(a) or np.isnan(b))
                        and np.sign(a) == np.sign(b)) else ""
        print(f"  {c:26s} IC={ic:+.4f} p={p:.1e} n={n}  halves={a:+.3f}/{b:+.3f}{flag}")


def main():
    tgt = (sys.argv[1] if len(sys.argv) > 1 else "BTC").lower()
    nbrs = [a for a in ASSETS if a != tgt]
    print(f"target={tgt.upper()}  neighbors={[n.upper() for n in nbrs]}")

    df = load_hourly(tgt)
    df = add_recent_yes(df)
    micro = build_micro_features(df)
    df = pd.concat([df, micro], axis=1)

    tgt_series = xa.build_book_series(tgt)
    for lag, lagtag in [(0, ""), (1, "_lag")]:
        d = df if lag == 0 else df.copy()
        pass  # series joined below per lag
    nbr_series = {a: xa.build_book_series(a) for a in nbrs}

    liq = fetch_liq(tgt)
    liq_feats = [c for c in liq.columns if c != "known_at"]
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))

    pre = (df["dt"] < pd.Timestamp("2026-07-09", tz="UTC")).values
    mid = pd.Timestamp("2026-06-14", tz="UTC")
    h1 = pre & (df["dt"] < mid).values
    h2 = pre & (df["dt"] >= mid).values
    y_dir = (df["resolved_yes"] - df["p_market"]).values
    pm = df["p_market"].values
    y_vol = df["price_move_pct"].abs().values
    rvc = df["rv_4h_ctl"].values

    screen(df, TRAJ_COLS + ["imp_median_dist", "imp_width_pct", "ladder_density",
                            "rung_resid"],
           y_dir, pm, pre, h1, h2, f"{tgt.upper()} own-book micro vs DIRECTION")
    screen(df, ["regvel45_recent_yes", "regvel120_recent_yes"] + liq_feats,
           y_vol, rvc, pre, h1, h2, f"{tgt.upper()} regvel+liq vs VOL (|move| | rv4h)")
    screen(df, ["liq_total_z", "liq_imbalance"], y_dir, pm, pre, h1, h2,
           f"{tgt.upper()} liq vs DIRECTION")

    for lag, lagtag in [(0, "FRESH"), (1, "SKIP-ONE")]:
        d = df.copy()
        feats = []
        d, own_cols = xa.join_neighbor(d, tgt_series, "own", lag_loops=lag)
        for a in nbrs:
            d, cols = xa.join_neighbor(d, nbr_series[a], a, lag_loops=lag)
            feats += [f"{a}_book_pm_chg15", f"{a}_imp_width_pct"]
            d[f"{a}_{tgt}_drift_diff"] = d["own_imp_median_dist"] - d[f"{a}_imp_median_dist"]
            feats.append(f"{a}_{tgt}_drift_diff")
        screen(d, feats, (d["resolved_yes"] - d["p_market"]).values,
               d["p_market"].values, pre, h1, h2,
               f"{tgt.upper()} cross-book vs DIRECTION [{lagtag}]")


if __name__ == "__main__":
    main()
