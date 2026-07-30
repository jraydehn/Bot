"""SOL hourly banked-signal backfill: liq_total_z + funding phase + 24h
anchors. 2026-07-30. Item 2 of the feature-gap inventory.

1. liq_total_z (the banked lead-lag find: forward-VOL info beyond price vol,
   partial IC ~0.12 all assets, never wired): 168h rolling pct-rank of total
   hourly liquidation USD. CoinGlass lapsed → Coinalyze 1h liquidation
   history (SOLUSDT_PERP.A, verified back to 05-11). Plus liq_imbalance,
   long/short liq ranks, 3/6/12h trends. known_at = bar close (t+1h).
2. Funding cycle phase: minutes to next perp funding settlement (00/08/16
   UTC) + sin/cos phase. Pure clock features, no fetch.
3. 24h anchors: dprice_1440 (SOL's validated mean-reversion scale) and
   pos_in_range_24h from the archive's own spot history.

Diagnostics (PRE-07-09 ONLY; burned test window untouched; Aug stays fresh):
  liq features → partial IC vs |price_move_pct| controlling for trailing
    4h realized vol (their claim is vol info, not direction);
  funding/24h features → partial IC vs (outcome − pm) controlling for pm.
Split-half stability on both. IC is a screen, not PnL — final scoring is
the late-Aug fresh-data book.
"""
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.stats import rankdata, pearsonr

import train_sol_hourly_niche_v3 as v3
import coinalyze_liq as cl

BASE = Path(__file__).parent
LIQ_RAW = BASE / "results" / "coinalyze_liq_1h_sol_backfill_20260730.csv"


def fetch_liq_bars() -> pd.DataFrame:
    if LIQ_RAW.exists():
        d = pd.read_csv(LIQ_RAW)
        d["known_at"] = pd.to_datetime(d["known_at"], utc=True)
        return d
    now = int(time.time())
    r = requests.get(f"{cl._BASE}/liquidation-history", params={
        "symbols": cl._SYMBOLS["SOL"], "interval": "1hour",
        "from": now - 85 * 24 * 3600, "to": now, "api_key": cl._API_KEY},
        timeout=20)
    r.raise_for_status()
    h = r.json()[0]["history"]
    d = pd.DataFrame(h)
    d["known_at"] = pd.to_datetime(d["t"], unit="s", utc=True) + pd.Timedelta(hours=1)
    d = d.rename(columns={"l": "liq_long", "s": "liq_short"}).sort_values("known_at")
    d[["known_at", "liq_long", "liq_short"]].to_csv(LIQ_RAW, index=False)
    return d


_LIVE_CACHE: dict = {}


def fetch_liq_bars_live(hours: int = 240, ttl: int = 1800) -> pd.DataFrame:
    """Fresh 1h liq bars for the runner: enough history for the 168h rank
    window, cached in-process for `ttl` seconds, CSV-backfill fallback."""
    now = time.time()
    if "d" in _LIVE_CACHE and now - _LIVE_CACHE["t"] < ttl:
        return _LIVE_CACHE["d"]
    try:
        r = requests.get(f"{cl._BASE}/liquidation-history", params={
            "symbols": cl._SYMBOLS["SOL"], "interval": "1hour",
            "from": int(now) - hours * 3600, "to": int(now),
            "api_key": cl._API_KEY}, timeout=20)
        r.raise_for_status()
        h = r.json()[0]["history"]
        d = pd.DataFrame(h)
        d["known_at"] = pd.to_datetime(d["t"], unit="s", utc=True) + pd.Timedelta(hours=1)
        d = d.rename(columns={"l": "liq_long", "s": "liq_short"}).sort_values("known_at")
        d = d[["known_at", "liq_long", "liq_short"]].reset_index(drop=True)
        _LIVE_CACHE.update(d=d, t=now)
        return d
    except Exception as e:
        print(f"  [liq_live] fetch failed ({e}); falling back to backfill CSV")
        return fetch_liq_bars()


def build_liq_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"known_at": d["known_at"]})
    tot = d["liq_long"] + d["liq_short"]
    f["liq_total_z"] = tot.rolling(168).rank(pct=True)
    f["liq_long_z"] = d["liq_long"].rolling(168).rank(pct=True)
    f["liq_short_z"] = d["liq_short"].rolling(168).rank(pct=True)
    f["liq_imbalance"] = (d["liq_long"] - d["liq_short"]) / tot.replace(0, np.nan)
    for col in ["liq_total_z", "liq_imbalance"]:
        for w in (3, 6, 12):
            f[f"{col}_trend{w}"] = (f[col] - f[col].shift(w)) / w
    return f


def add_clock_and_daily(df: pd.DataFrame) -> tuple:
    # funding cycle: settlements at 00/08/16 UTC
    mins = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    to_next = (480 - (mins % 480)) % 480
    df["min_to_funding"] = to_next
    ph = 2 * np.pi * (mins % 480) / 480
    df["funding_phase_sin"] = np.sin(ph)
    df["funding_phase_cos"] = np.cos(ph)
    # 24h anchors from archive spot
    snap = df.drop_duplicates("dt", keep="last")[["dt", "spot"]]
    ts = snap["dt"].astype("int64").values / 1e9
    spot = snap["spot"].values
    idx = np.searchsorted(ts, ts - 86400, side="right") - 1
    ok = idx >= 0
    pv = np.where(ok, spot[np.clip(idx, 0, None)], np.nan)
    ctx = pd.DataFrame(index=snap.index)
    ctx["dt"] = snap["dt"]
    ctx["dprice_1440"] = (spot / pv - 1) * 100
    s = pd.Series(spot, index=pd.DatetimeIndex(snap["dt"]))
    lo, hi = s.rolling("24h").min(), s.rolling("24h").max()
    ctx["pos_in_range_24h"] = ((s - lo) / (hi - lo).replace(0, np.nan)).values
    ret = s.pct_change()
    ctx["rv_4h_diag"] = ret.rolling("4h").std().values  # diagnostic control
    df = df.merge(ctx, on="dt", how="left")
    return df, ["min_to_funding", "funding_phase_sin", "funding_phase_cos",
                "dprice_1440", "pos_in_range_24h"]


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
    df = v3.load_archive()
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    df["price_move_abs"] = pd.to_numeric(df["price_move_pct"], errors="coerce").abs()

    liq = build_liq_features(fetch_liq_bars())
    liq_feats = [c for c in liq.columns if c != "known_at"]
    df = pd.merge_asof(df, liq, left_on="dt", right_on="known_at",
                       direction="backward", tolerance=pd.Timedelta(hours=3))
    df, clock_feats = add_clock_and_daily(df)
    print(f"liq join coverage: {df['liq_total_z'].notna().mean():.1%}")

    out = df[["dt", "contract_ticker"] + liq_feats + clock_feats]
    out.to_parquet(BASE / "results" / "sol_hourly_banked_signals_20260730.parquet",
                   index=False)

    pre = df["dt"] < pd.Timestamp("2026-07-09", tz="UTC")
    mid = pd.Timestamp("2026-06-14", tz="UTC")
    halves = [("H1", pre & (df["dt"] < mid)), ("H2", pre & (df["dt"] >= mid))]

    print("\n[A] liq features vs |realized move| (control: trailing rv_4h), pre-07-09:")
    y = df["price_move_abs"].values
    ctrl = df["rv_4h_diag"].values
    for c in liq_feats:
        ic, p, n = partial_ic(df[c][pre].values, y[pre], ctrl[pre])
        hh = [partial_ic(df[c][m].values, y[m], ctrl[m])[0] for _, m in halves]
        print(f"  {c:22s} IC={ic:+.4f} p={p:.4f} n={n}  halves={hh[0]:+.3f}/{hh[1]:+.3f}")

    print("\n[B] clock/daily features vs (outcome − pm) residual (control: pm), pre-07-09:")
    y2 = (df["resolved_yes"] - df["p_market"]).values
    pmv = df["p_market"].values
    for c in clock_feats + ["liq_imbalance", "liq_total_z"]:
        ic, p, n = partial_ic(df[c][pre].values, y2[pre], pmv[pre])
        hh = [partial_ic(df[c][m].values, y2[m], pmv[m])[0] for _, m in halves]
        print(f"  {c:22s} IC={ic:+.4f} p={p:.4f} n={n}  halves={hh[0]:+.3f}/{hh[1]:+.3f}")


if __name__ == "__main__":
    main()
