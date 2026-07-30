"""Kalshi microstructure features: cross-strike ladder + per-contract pm
trajectory. 2026-07-30.

Built for the SOL hourly late-Aug retrain re-test (frozen now, scored on
fresh forward data); generic over any hourly scan archive (BTC/ETH too).

Two families, both point-in-time by construction:

LADDER (per scan-loop per event, across simultaneously-quoted strikes;
"above-K" contracts → pm decreasing in K is the coherent shape):
  n_strikes          rungs quoted in this loop
  imp_median_dist    (ladder-implied median settle − spot)/spot ×100 — the
                     market's expected drift for the remainder of the hour
  imp_width_pct      strike distance between pm=0.84 and pm=0.16 rungs,
                     /spot ×100 — implied ~2σ move for the hour
  imp_vol_ratio      imp_width vs realized-vol-expected move (needs rv col;
                     NaN if unavailable)
  ladder_z           (strike − implied median)/implied σ — market-consistent
                     moneyness of THIS rung
  rung_resid         this rung's pm − pm interpolated from its neighbors —
                     positive = this quote rich vs its own ladder
  mono_viol_frac     fraction of adjacent rung pairs violating monotonicity
                     (staleness/laziness of the ladder)
  ladder_density     local |dpm/dK| around this rung, ×strike (implied
                     density mass near this strike)

PM TRAJECTORY (this contract's own earlier scans only):
  pm_chg_5m/15m/30m  pm now − pm at t−Δ (nearest prior scan ≥Δ back)
  pm_accel_15m       pm_chg_15m − previous 15m leg (pm_{t−15}−pm_{t−30})
  pm_vel_life        (pm − pm_first)/minutes observed
  pm_range_life      max−min pm observed so far
  pm_n_obs           number of prior observations (trajectory maturity)

Usage:
  feats = build_micro_features(df)   # df: dt, contract_ticker, p_market,
                                     # strike, spot (+optional rv col)
  returns DataFrame aligned to df.index with the columns above.
"""
import numpy as np
import pandas as pd

LADDER_COLS = ["n_strikes", "imp_median_dist", "imp_width_pct", "imp_vol_ratio",
               "ladder_z", "rung_resid", "mono_viol_frac", "ladder_density"]
TRAJ_COLS = ["pm_chg_5m", "pm_chg_15m", "pm_chg_30m", "pm_accel_15m",
             "pm_vel_life", "pm_range_life", "pm_n_obs"]


def _event_key(t: pd.Series) -> pd.Series:
    # KXSOLD-26JUL2418-T73.4999 → KXSOLD-26JUL2418
    return t.astype(str).str.rsplit("-T", n=1).str[0]


def _ladder_features(g: pd.DataFrame, rv_col: str) -> pd.DataFrame:
    """g: one scan-loop of one event; unique strikes, sorted ascending."""
    out = pd.DataFrame(index=g.index, columns=LADDER_COLS, dtype=float)
    k = g["strike"].values
    pm = g["p_market"].values
    spot = float(g["spot"].iloc[0])
    n = len(g)
    out["n_strikes"] = n
    if n < 3:
        return out
    # monotonicity: above-K ladder should have pm strictly decreasing in K
    dpm = np.diff(pm)
    out["mono_viol_frac"] = float((dpm > 0).mean())
    # implied quantiles from the (enforced-monotone) ladder
    pm_m = np.minimum.accumulate(pm)  # enforce non-increasing for interp
    def k_at(p):
        if pm_m[0] <= p or pm_m[-1] >= p:
            return np.nan
        return float(np.interp(-p, -pm_m, k))  # pm decreasing → negate
    k50, k16, k84 = k_at(0.50), k_at(0.16), k_at(0.84)
    if not np.isnan(k50):
        out["imp_median_dist"] = (k50 - spot) / spot * 100
    if not (np.isnan(k16) or np.isnan(k84)):
        width = k16 - k84  # k84 (pm=0.84) < k50 < k16 (pm=0.16)
        out["imp_width_pct"] = width / spot * 100
        sigma = width / 2.0
        if sigma > 0 and not np.isnan(k50):
            out["ladder_z"] = (k - k50) / sigma
        if rv_col in g.columns and sigma > 0:
            rv = pd.to_numeric(g[rv_col], errors="coerce").iloc[0]
            tau = pd.to_numeric(g.get("tau_minutes"), errors="coerce").iloc[0]
            if rv and rv > 0 and tau and tau > 0:
                # rv annualized (pct): expected 1σ move over tau
                exp_move = spot * float(rv) / 100 * np.sqrt(tau / (365 * 24 * 60))
                if exp_move > 0:
                    out["imp_vol_ratio"] = sigma / exp_move
    # rung residual + local density (interior rungs only)
    if n >= 3:
        resid = np.full(n, np.nan)
        dens = np.full(n, np.nan)
        for i in range(1, n - 1):
            dk = k[i + 1] - k[i - 1]
            if dk > 0:
                interp = pm[i - 1] + (pm[i + 1] - pm[i - 1]) * (k[i] - k[i - 1]) / dk
                resid[i] = pm[i] - interp
                dens[i] = abs(pm[i + 1] - pm[i - 1]) / dk * k[i]
        out["rung_resid"] = resid
        out["ladder_density"] = dens
    return out


def build_micro_features(df: pd.DataFrame, rv_col: str = "rvol_1h",
                         loop_window: str = "2min") -> pd.DataFrame:
    """df needs: dt (tz-aware), contract_ticker, p_market, strike, spot.
    Returns features aligned to df.index. Point-in-time: ladder uses only
    same-loop rows; trajectory uses only this contract's earlier rows."""
    d = df.copy()
    for c in ["p_market", "strike", "spot"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_event"] = _event_key(d["contract_ticker"])
    d["_loop"] = d["dt"].dt.floor(loop_window)

    # ── ladder ────────────────────────────────────────────────────────────
    lad = pd.DataFrame(index=d.index, columns=LADDER_COLS, dtype=float)
    for (_, _), g in d.groupby(["_loop", "_event"], sort=False):
        g = g.sort_values("dt").drop_duplicates("strike", keep="last")
        g = g.sort_values("strike")
        if g["p_market"].isna().any() or g["strike"].isna().any():
            g = g.dropna(subset=["p_market", "strike"])
        if not len(g):
            continue
        lad.loc[g.index, LADDER_COLS] = _ladder_features(g, rv_col).values

    # ── pm trajectory ─────────────────────────────────────────────────────
    tr = pd.DataFrame(index=d.index, columns=TRAJ_COLS, dtype=float)
    for _, g in d.groupby("contract_ticker", sort=False):
        g = g.sort_values("dt")
        ts = g["dt"].astype("int64").values / 1e9
        pm = g["p_market"].values
        n = len(g)
        res = np.full((n, len(TRAJ_COLS)), np.nan)
        cmax = np.maximum.accumulate(pm)
        cmin = np.minimum.accumulate(pm)
        for j, sec in [(0, 300), (1, 900), (2, 1800)]:
            idx = np.searchsorted(ts, ts - sec, side="right") - 1
            ok = idx >= 0
            res[ok, j] = pm[ok] - pm[np.clip(idx, 0, None)][ok]
        # accel: current 15m leg minus previous 15m leg
        i15 = np.searchsorted(ts, ts - 900, side="right") - 1
        i30 = np.searchsorted(ts, ts - 1800, side="right") - 1
        ok = (i15 >= 0) & (i30 >= 0)
        res[ok, 3] = (pm[ok] - pm[np.clip(i15, 0, None)][ok]) - \
                     (pm[np.clip(i15, 0, None)][ok] - pm[np.clip(i30, 0, None)][ok])
        mins = (ts - ts[0]) / 60
        with np.errstate(divide="ignore", invalid="ignore"):
            res[:, 4] = np.where(mins > 0, (pm - pm[0]) / mins, np.nan)
        res[:, 5] = cmax - cmin
        res[:, 6] = np.arange(n)
        tr.loc[g.index, TRAJ_COLS] = res

    return pd.concat([lad, tr], axis=1).astype(float)
