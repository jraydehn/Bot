#!/usr/bin/env python3
"""
sweep_k_drift_v2.py — Sweep ALPHA in the BTC k_drift formula with p_up v2.

Current formula:  k_drift = 1.40 × exp(-2.0 × max(0, z_strike))
Sweep:            k_drift = ALPHA × exp(-2.0 × max(0, z_strike))

For each ALPHA, simulate YES-side PnL on resolved btc_scan_archive rows using:
  - p_up v2 inference (OHLCV features pre-computed + archive signals)
  - Kelly sizing (flat $1000 bankroll, 25% cap)
  - Min edge 0.03
  - Dedup to first scan per contract_ticker

Run:  python3 sweep_k_drift_v2.py
"""

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── config ───────────────────────────────────────────────────────────────────
ARCHIVE_CSV = Path("results/btc_scan_archive.csv")
DATA_DIR    = Path("data")
MODEL_PATH  = Path("reform_results/btc_p_up_v2.pkl")
BANKROLL    = 1000.0
MIN_EDGE    = 0.03
KELLY_CAP   = 0.25
CURRENT_ALPHA = 1.40
ALPHAS      = [round(a, 2) for a in np.arange(0.30, 2.81, 0.10)]

SEP  = "=" * 72
SEP2 = "-" * 72


# ── indicator helpers (vectorised on full series) ─────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def stoch_k_series(h, lo, c, k=14):
    ll = lo.rolling(k).min()
    hh = h.rolling(k).max()
    rng = (hh - ll).replace(0, np.nan)
    return (c - ll) / rng * 100

def atr_series(h, lo, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()

def macd_hist_series(c, f=12, s=26, sig=9):
    macd = _ema(c, f) - _ema(c, s)
    return (macd - macd.ewm(span=sig, adjust=False).mean())

def bb_pct_series(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    lo  = mid - 2 * std
    hi  = mid + 2 * std
    return (c - lo) / (hi - lo).replace(0, np.nan)

def ema50_dist_series(c):
    e50 = _ema(c, 50)
    return (c - e50) / e50.replace(0, np.nan) * 100

def chg_4h_atr_series(df4):
    a = atr_series(df4["high"], df4["low"], df4["close"], 14)
    return (df4["close"] - df4["close"].shift(5)) / a.replace(0, np.nan)


# ── p_up v2 batch inference ───────────────────────────────────────────────────

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def infer_batch(clf, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].values.astype(float)
    return clf.predict_proba(X)[:, 1]


# ── score_to_p_model (inline, mirrors composite_scorer.py) ───────────────────

def p_model_yes(p_up: float, spot: float, strike: float,
                sigma_tau: float, alpha: float) -> float:
    if sigma_tau <= 0 or np.isnan(sigma_tau):
        return 0.5
    z_strike = math.log(strike / spot) / sigma_tau
    k_drift  = alpha * math.exp(-2.0 * max(0.0, z_strike))
    z_drift  = norm.ppf(float(np.clip(p_up, 0.01, 0.99))) * k_drift
    z_adj    = z_strike - z_drift
    return float(np.clip(1.0 - norm.cdf(z_adj), 0.01, 0.99))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  BTC k_drift ALPHA sweep  (p_up v2)")
    print(SEP)

    # ── load archive ──────────────────────────────────────────────────────────
    df = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    df["logged_at"]   = pd.to_datetime(df["logged_at"],   utc=True, errors="coerce")
    df["resolved_yes"] = pd.to_numeric(df["resolved_yes"], errors="coerce")
    df = df[df["resolved_yes"].notna()].copy()
    print(f"Archive: {len(df)} resolved rows")

    # Dedup: first scan per contract (avoids re-counting same contract)
    df = df.sort_values("logged_at").drop_duplicates(subset="contract_ticker", keep="first")
    print(f"After dedup (first scan/ticker): {len(df)} unique contracts")
    print(f"Date range: {df['logged_at'].iloc[0].date()} → {df['logged_at'].iloc[-1].date()}")
    print()

    # ── load OHLCV ────────────────────────────────────────────────────────────
    def _latest(pattern):
        files = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No file matching {pattern} in {DATA_DIR}")
        return files[-1]

    f1h = _latest("binanceus_BTCUSDT_1h_*.parquet")
    f4h = _latest("binanceus_BTCUSDT_4h_*.parquet")
    print(f"Loading OHLCV: {f1h.name} / {f4h.name}")
    df1h = pd.read_parquet(f1h)
    df4h = pd.read_parquet(f4h)

    for d in (df1h, df4h):
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")

    print(f"  1h: {len(df1h)} bars  |  4h: {len(df4h)} bars")
    print()

    # ── pre-compute indicator series ─────────────────────────────────────────
    print("Pre-computing price indicators…")
    c1h = df1h["close"]
    c4h = df4h["close"]

    ind1h = pd.DataFrame({
        "rsi_14":      rsi_series(c1h, 14),
        "macd_hist_1h": macd_hist_series(c1h),
        "bb_pct":      bb_pct_series(c1h),
        "ema50_dist":  ema50_dist_series(c1h),
    }, index=df1h.index)

    ind4h = pd.DataFrame({
        "stoch_k_4h": stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
        "rsi_4h":     rsi_series(c4h, 14),
        "chg_4h_atr": chg_4h_atr_series(df4h),
    }, index=df4h.index)

    # ── merge indicators into archive (merge_asof on timestamp) ──────────────
    ind1h_r = ind1h.reset_index().rename(columns={ind1h.index.name or "index": "ts"})
    ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})

    df_s = df.sort_values("logged_at")
    df_s = pd.merge_asof(df_s, ind1h_r, left_on="logged_at", right_on="ts", direction="backward")
    df_s = pd.merge_asof(df_s, ind4h_r, left_on="logged_at", right_on="ts",
                         direction="backward", suffixes=("", "_4h"))

    # Features missing from archive → NaN (LightGBM handles natively)
    for col in ("confirmation_bias", "stoch_bias", "pm_drift_5m"):
        df_s[col] = float("nan")

    # vwap_distance_pct is in archive already
    # confirmation_score ≠ confirmation_bias — leave NaN

    pct_ok = {f: df_s[f].notna().mean() * 100 for f in FEATURES}
    print("  Feature coverage:")
    for f, p in sorted(pct_ok.items(), key=lambda x: x[1]):
        tag = "  [from archive]" if f in df.columns else "  [from OHLCV]"
        print(f"    {f:<25} {p:5.1f}%{tag}")
    print()

    # ── p_up v2 inference ─────────────────────────────────────────────────────
    print("Running p_up v2 inference…")
    pipe = load_model()
    clf  = pipe["clf"]
    p_up_v2 = infer_batch(clf, df_s)
    p_up_v2 = np.clip(p_up_v2, 0.02, 0.98)

    p_up_old = df_s["composite_p_up"].values.astype(float)

    print(f"  p_up_old : mean={np.nanmean(p_up_old):.3f}  std={np.nanstd(p_up_old):.3f}  "
          f"<0.5: {(p_up_old < 0.5).mean()*100:.0f}%  >0.5: {(p_up_old > 0.5).mean()*100:.0f}%")
    print(f"  p_up_v2  : mean={np.nanmean(p_up_v2):.3f}  std={np.nanstd(p_up_v2):.3f}  "
          f"<0.5: {(p_up_v2 < 0.5).mean()*100:.0f}%  >0.5: {(p_up_v2 > 0.5).mean()*100:.0f}%")
    print()

    # ── extract needed columns ────────────────────────────────────────────────
    spot_arr      = df_s["spot"].values.astype(float)
    strike_arr    = df_s["strike"].values.astype(float)
    pm_arr        = df_s["p_market"].values.astype(float)
    vol_eff_arr   = df_s["vol_eff"].values.astype(float)
    tau_arr       = df_s["tau_minutes"].values.astype(float)
    resolved_arr  = df_s["resolved_yes"].values.astype(int)

    sigma_tau_arr = vol_eff_arr * np.sqrt(np.maximum(tau_arr, 0.0))

    # ── sweep ─────────────────────────────────────────────────────────────────
    print(SEP2)
    print(f"  {'ALPHA':>6}  {'n_trades':>8}  {'WR':>6}  {'PnL':>8}  {'AvgEdge':>8}  note")
    print(SEP2)

    results = []
    for alpha in ALPHAS:
        n_trades = 0
        wins     = 0
        pnl      = 0.0
        sum_edge = 0.0

        for i in range(len(df_s)):
            spot    = spot_arr[i]
            strike  = strike_arr[i]
            pm      = pm_arr[i]
            s_tau   = sigma_tau_arr[i]
            ry      = resolved_arr[i]
            p_up    = p_up_v2[i]

            if np.isnan(spot) or np.isnan(strike) or np.isnan(pm) or np.isnan(s_tau):
                continue
            if s_tau <= 0:
                continue

            # YES side only
            pm_c = float(np.clip(pm, 0.01, 0.99))
            p_m  = p_model_yes(p_up, spot, strike, s_tau, alpha)
            edge = p_m - pm_c

            if edge < MIN_EDGE:
                continue

            kelly  = edge / (1.0 - pm_c)
            kelly  = min(kelly, KELLY_CAP)
            cost   = pm_c * kelly * BANKROLL      # dollar cost of YES contracts
            count  = max(1, int(cost / pm_c))     # shares (each costs pm_c)

            n_trades += 1
            sum_edge += edge

            if ry == 1:
                pnl += count * (1.0 - pm_c)
                wins += 1
            else:
                pnl -= count * pm_c

        wr       = wins / n_trades if n_trades else float("nan")
        avg_edge = sum_edge / n_trades if n_trades else float("nan")
        tag      = "  ← current" if abs(alpha - CURRENT_ALPHA) < 0.01 else ""

        results.append((alpha, n_trades, wr, pnl, avg_edge))
        print(f"  {alpha:6.2f}  {n_trades:8d}  {wr:5.1%}  {pnl:+8.2f}  {avg_edge:8.4f}{tag}")

    print(SEP2)

    # ── summary ───────────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x[3])
    print()
    print(f"  Best ALPHA: {best[0]:.2f}  (PnL={best[3]:+.2f}, WR={best[2]:.1%}, n={best[1]})")
    print(f"  Current  : {CURRENT_ALPHA:.2f}")

    # Show p_up v2 distribution across deciles for context
    print()
    print("  p_up_v2 vs p_up_old (resolved contracts, by decile):")
    print(f"  {'Decile':>8}  {'p_up_old':>10}  {'p_up_v2':>10}  {'delta':>8}")
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        v2_q  = np.nanquantile(p_up_v2,  q)
        old_q = np.nanquantile(p_up_old, q)
        print(f"  {q:8.0%}  {old_q:10.3f}  {v2_q:10.3f}  {v2_q - old_q:+8.3f}")

    print()
    print(f"  NOTE: {len(df_s)} unique contracts from "
          f"{df_s['logged_at'].iloc[0].date()} → {df_s['logged_at'].iloc[-1].date()}.")
    print("  Sweep is noisy with <1 day of data — re-run once archive grows.")
    print(SEP)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
