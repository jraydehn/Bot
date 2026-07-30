"""SOL hourly slope constructions — second-order signal geometry. 2026-07-30.

Final feature-inventory item: new *constructions* over existing scan-level
signals (first-order D/S slopes already exhausted in v3-v6):

  ACCEL      D45 now minus D45 45min ago (slope-of-slope), key bases
  CONFIRM    D15/D120 ratio — is the last 15min confirming or fading the
             2h trend? (sign-aware, clipped)
  COHERENCE  count of positive D45 (and D120) across an 8-signal momentum
             basket — the persist-score construction that survived OOS at
             15m (permP=0.0006), never built for hourly
  DURATION   minutes since sign flip: stoch_k-50, ema_stack_bias, dprice_15
  VOLNORM    D45 / trailing 4h realized vol, key bases
  REGIME-VEL slope of recent_yes_6h over 45/120min (settlement-regime drift
             velocity)

Screen: pre-07-09 ONLY, partial IC controlling pm vs (outcome − pm), and
controlling rv_4h vs |realized move|; split-half stability. Burned window
untouched. Parquet saved for the late-Aug arm.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, pearsonr

import train_sol_hourly_niche_v3 as v3

BASE = Path(__file__).parent

ACCEL_BASES = ["stoch_k", "composite_p_up", "vwap_distance_pct", "adx_1h",
               "ls_long_pct", "oi_chg_pct"]
COHER_BASKET = ["stoch_k", "composite_p_up", "composite_trend",
                "vwap_distance_pct", "ema_stretch_score", "ema_stack_bias",
                "bp_5m", "chg_30m"]
VOLNORM_BASES = ["stoch_k", "vwap_distance_pct", "composite_p_up"]


def lagged_vals(ts: np.ndarray, v: np.ndarray, sec: int) -> np.ndarray:
    idx = np.searchsorted(ts, ts - sec, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
    return out


def minutes_since_flip(ts: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Minutes since the sign series last changed (NaN-sign rows carry prior)."""
    out = np.full(len(sign), np.nan)
    last_flip = np.nan
    prev = np.nan
    for i in range(len(sign)):
        s = sign[i]
        if not np.isnan(s):
            if not np.isnan(prev) and s != prev:
                last_flip = ts[i]
            elif np.isnan(prev):
                last_flip = ts[i]
            prev = s
        if not np.isnan(last_flip):
            out[i] = (ts[i] - last_flip) / 60
    return out


def build(df: pd.DataFrame) -> tuple:
    snap = df.drop_duplicates("dt", keep="last").copy()
    ts = snap["dt"].astype("int64").values / 1e9
    nc = {}

    # base D45/D120 on snapshot (needed for constructions)
    d45, d120 = {}, {}
    for c in set(ACCEL_BASES + COHER_BASKET + VOLNORM_BASES):
        v = snap[c].values
        d45[c] = v - lagged_vals(ts, v, 2700)
        d120[c] = v - lagged_vals(ts, v, 7200)

    for c in ACCEL_BASES:
        nc[f"accel45_{c}"] = d45[c] - lagged_vals(ts, d45[c], 2700)
    for c in ACCEL_BASES:
        with np.errstate(divide="ignore", invalid="ignore"):
            nc[f"confirm_{c}"] = np.clip(
                d45[c] / np.where(np.abs(d120[c]) < 1e-9, np.nan, d120[c]), -5, 5)

    coh45 = np.nansum([np.sign(d45[c]) for c in COHER_BASKET], axis=0)
    coh120 = np.nansum([np.sign(d120[c]) for c in COHER_BASKET], axis=0)
    nc["coherence45"] = coh45
    nc["coherence120"] = coh120
    nc["coherence_agree"] = np.sign(coh45) * np.minimum(np.abs(coh45), np.abs(coh120))

    spot = snap["spot"].values
    dp15 = (spot / lagged_vals(ts, spot, 900) - 1) * 100
    nc["dur_stoch50"] = minutes_since_flip(ts, np.sign(snap["stoch_k"].values - 50))
    nc["dur_emastack"] = minutes_since_flip(ts, np.sign(snap["ema_stack_bias"].values))
    nc["dur_dprice15"] = minutes_since_flip(ts, np.sign(dp15))

    s = pd.Series(spot, index=pd.DatetimeIndex(snap["dt"]))
    rv4 = s.pct_change().rolling("4h").std().values
    for c in VOLNORM_BASES:
        with np.errstate(divide="ignore", invalid="ignore"):
            nc[f"volnorm45_{c}"] = d45[c] / np.where(rv4 < 1e-9, np.nan, rv4) / 1e4

    ry = snap["recent_yes_6h"].values
    nc["regvel45_recent_yes"] = ry - lagged_vals(ts, ry, 2700)
    nc["regvel120_recent_yes"] = ry - lagged_vals(ts, ry, 7200)
    nc["rv_4h_ctl"] = rv4

    ctx = pd.DataFrame(nc, index=snap.index)
    ctx["dt"] = snap["dt"]
    out = df.merge(ctx, on="dt", how="left")
    feats = [k for k in nc if k != "rv_4h_ctl"]
    return out, feats


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
    df = v3.add_extended(df)   # recent_yes_6h needed
    df = df.dropna(subset=["resolved_yes", "p_market"])
    df = df[df["p_market"].between(0.02, 0.98)].sort_values("dt").reset_index(drop=True)
    df, feats = build(df)
    print(f"built {len(feats)} construction features on {len(df)} rows")

    df[["dt", "contract_ticker"] + feats].to_parquet(
        BASE / "results" / "sol_hourly_slope_constructions_20260730.parquet", index=False)

    pre = (df["dt"] < pd.Timestamp("2026-07-09", tz="UTC")).values
    mid = pd.Timestamp("2026-06-14", tz="UTC")
    h1 = pre & (df["dt"] < mid).values
    h2 = pre & (df["dt"] >= mid).values

    y_dir = (df["resolved_yes"] - df["p_market"]).values
    pm = df["p_market"].values
    y_vol = pd.to_numeric(df["price_move_pct"], errors="coerce").abs().values
    rvc = df["rv_4h_ctl"].values

    for tgt, y, ctrl in [("DIRECTION (outcome−pm | pm)", y_dir, pm),
                         ("VOL (|move| | rv_4h)", y_vol, rvc)]:
        print(f"\n[{tgt}] pre-07-09 partial IC:")
        rows = []
        for c in feats:
            v = df[c].values.astype(float)
            ic, p, n = partial_ic(v[pre], y[pre], ctrl[pre])
            ha = partial_ic(v[h1], y[h1], ctrl[h1])[0]
            hb = partial_ic(v[h2], y[h2], ctrl[h2])[0]
            stable = (not np.isnan(ic) and p < 1e-4 and abs(ic) > 0.03
                      and not (np.isnan(ha) or np.isnan(hb))
                      and np.sign(ha) == np.sign(hb))
            rows.append((c, ic, p, n, ha, hb, " **" if stable else ""))
        r = pd.DataFrame(rows, columns=["feature", "IC", "p", "n", "H1", "H2", "flag"])
        r = r.sort_values("IC", key=abs, ascending=False)
        for _, x in r.iterrows():
            print(f"  {x['feature']:24s} IC={x['IC']:+.4f} p={x['p']:.1e} "
                  f"n={x['n']}  halves={x['H1']:+.3f}/{x['H2']:+.3f}{x['flag']}")


if __name__ == "__main__":
    main()
