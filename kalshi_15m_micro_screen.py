"""Screen 15m book-dynamics features on backfilled Kalshi 1-min candles.
2026-07-31.

Data: results/kalshi_15m_candles_{asset}.csv (kalshi_15m_candle_backfill).
15m markets are single-market events (no ladder) — the feature set is
per-contract pm trajectory at 1-min resolution + minute volume/OI flow:

  pm_chg_1m/3m/5m   mid change over trailing minutes (mid=(bid+ask)/2)
  pm_vel_life       (mid − mid_first)/minutes elapsed
  pm_range_life     max−min mid so far
  pm_accel          last-3m change minus prior-3m change
  vol_cum, vol_3m   cumulative / trailing-3m volume (log1p)
  oi_chg_pct        OI change since first candle
  spread            ask − bid (current)

Decision points: every minute with >=3 prior candles, per market (rows
overlap within a contract — IC read with that caveat). Target: residual
(result − pm_mid) with pm controlled (partial IC), split halves by time.
This candle history is fresh data — no burned-window concerns yet; the
halves check is the stability screen.

Usage: python3 kalshi_15m_micro_screen.py BTC
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, pearsonr

BASE = Path(__file__).parent

FEATS = ["pm_chg_1m", "pm_chg_3m", "pm_chg_5m", "pm_vel_life",
         "pm_range_life", "pm_accel", "vol_cum", "vol_3m", "oi_chg_pct",
         "spread"]


def build(asset: str) -> pd.DataFrame:
    d = pd.read_csv(BASE / "results" / f"kalshi_15m_candles_{asset.lower()}.csv",
                    low_memory=False)
    for c in ["bid_close", "ask_close", "price_close", "volume_fp", "oi_fp",
              "end_ts"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["end_ts"]).sort_values(["ticker", "end_ts"])
    d["mid"] = (d["bid_close"] + d["ask_close"]) / 2
    d["y"] = (d["result"].astype(str).str.lower() == "yes").astype(float)
    rows = []
    for tk, g in d.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        mid = g["mid"].values
        vol = np.nan_to_num(g["volume_fp"].values)
        oi = g["oi_fp"].values
        n = len(g)
        if n < 5 or np.isnan(mid).all():
            continue
        for i in range(3, n - 1):   # leave the final candle out (settlement noise)
            m = mid[i]
            if not (0.03 <= m <= 0.97) or np.isnan(m):
                continue
            rows.append({
                "ticker": tk, "t_min": i, "close_time": g["close_time"].iloc[0],
                "pm": m, "y": g["y"].iloc[0],
                "pm_chg_1m": m - mid[i - 1],
                "pm_chg_3m": m - mid[i - 3],
                "pm_chg_5m": m - mid[i - 5] if i >= 5 else np.nan,
                "pm_vel_life": (m - mid[0]) / i,
                "pm_range_life": np.nanmax(mid[:i + 1]) - np.nanmin(mid[:i + 1]),
                "pm_accel": (m - mid[i - 3]) - (mid[i - 3] - mid[i - 6])
                            if i >= 6 else np.nan,
                "vol_cum": np.log1p(vol[:i + 1].sum()),
                "vol_3m": np.log1p(vol[i - 2:i + 1].sum()),
                "oi_chg_pct": (oi[i] / oi[0] - 1) * 100
                              if oi[0] and oi[0] == oi[0] and oi[0] > 0 else np.nan,
                "spread": g["ask_close"].iloc[i] - g["bid_close"].iloc[i],
            })
    return pd.DataFrame(rows)


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
    asset = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    df = build(asset)
    df["dt"] = pd.to_datetime(df["close_time"], utc=True)
    df.to_parquet(BASE / "results" / f"kalshi_15m_micro_{asset.lower()}.parquet",
                  index=False)
    print(f"{asset}: {len(df)} decision points from "
          f"{df['ticker'].nunique()} markets "
          f"({df['dt'].min().date()} → {df['dt'].max().date()})")
    mid_t = df["dt"].min() + (df["dt"].max() - df["dt"].min()) / 2
    y = (df["y"] - df["pm"]).values
    pm = df["pm"].values
    h1 = (df["dt"] < mid_t).values
    h2 = ~h1
    print(f"\npartial IC vs (result − pm), pm-controlled; halves at {mid_t.date()}:")
    for c in FEATS:
        v = df[c].values.astype(float)
        ic, p, n = partial_ic(v, y, pm)
        a = partial_ic(v[h1], y[h1], pm[h1])[0]
        b = partial_ic(v[h2], y[h2], pm[h2])[0]
        flag = " **" if (not np.isnan(ic) and p < 1e-4 and abs(ic) > 0.03
                        and not (np.isnan(a) or np.isnan(b))
                        and np.sign(a) == np.sign(b)) else ""
        print(f"  {c:14s} IC={ic:+.4f} p={p:.1e} n={n}  halves={a:+.3f}/{b:+.3f}{flag}")


if __name__ == "__main__":
    main()
