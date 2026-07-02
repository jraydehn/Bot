#!/usr/bin/env python3
"""
test_drift_robustness.py

Rigorous drift formula evaluation across 875 days of BTC 1h data (2024-2026).

Methodology:
- For each 1h bar: generate synthetic Kalshi-like contracts at 8 strike offsets
- outcome = 1 if close[i+1] > strike (resolved YES)
- pm_synthetic = pure log-normal P(close > strike) using rolling realized vol
- For each drift formula: IC_adj = corr(predicted_edge, outcome - pm_synthetic)
  This measures whether the formula finds genuine edge BEYOND what the market prices
- Test YES-eligible rows (edge_yes > 0) and NO-eligible rows (edge_no > 0) separately
- Split by year/regime to verify stability — if a formula is real, IC should be
  consistent across bull, bear, and sideways periods
- Cross-validate on live blocked_trades.csv (May 6-26) as out-of-sample check
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

OFFSETS = [-0.020, -0.015, -0.010, -0.005, 0.005, 0.010, 0.015, 0.020]
TAU     = 60.0   # tau in minutes for 1h contracts
VOL_WIN = 24     # bars for rolling realized vol
MIN_N   = 1000   # minimum rows for a stable IC estimate
EPS     = 1e-7


# ── Load data ─────────────────────────────────────────────────────────────────

def load_1h():
    f = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
               key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df.index.year > 1970].sort_index()
    return df


def load_4h():
    # Use the fullest 4h file starting from 2024
    candidates = sorted(DATA_DIR.glob("binanceus_BTCUSDT_4h_2024-01-01*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    df = pd.read_parquet(candidates[-1])
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def macd_hist(s, fast=12, slow=26, sig=9):
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line - signal


def stoch_k(h, l, c, p=14):
    ll = l.rolling(p).min()
    hh = h.rolling(p).max()
    return (c - ll) / (hh - ll).replace(0, np.nan) * 100


# ── Build signal frame ────────────────────────────────────────────────────────

def build_signals(df1h, df4h):
    lr   = np.log(df1h["close"] / df1h["close"].shift(1))

    # Momentum signals from 1h
    mu6  = lr.rolling(6,  min_periods=1).mean()
    mu12 = lr.rolling(12, min_periods=1).mean()
    mu24 = lr.rolling(24, min_periods=1).mean()
    ewm_m = lr.ewm(span=12).mean()
    ewm_s = lr.ewm(span=24).std()
    rz    = np.clip(ewm_m / ewm_s.replace(0, np.nan), -3.0, 3.0).fillna(0.0)

    # Rolling realized vol (per-sqrt-minute scale)
    vol_1h       = lr.rolling(VOL_WIN, min_periods=4).std()   # stddev per 1h bar
    vol_per_sqrt_min = vol_1h / np.sqrt(60.0)                 # convert to per-sqrt-min scale
    sigma_tau    = vol_per_sqrt_min * np.sqrt(TAU)            # sigma for tau=60 min

    # 1h RSI
    rsi_1h = rsi(df1h["close"])

    sigs = pd.DataFrame({
        "close":     df1h["close"],
        "mu6h":      mu6,
        "mu12h":     mu12,
        "mu24h":     mu24,
        "regime_z":  rz,
        "sigma_tau": sigma_tau,
        "vol_pm":    vol_per_sqrt_min,
        "rsi_1h":    rsi_1h,
    }, index=df1h.index)

    # 4h signals — resample to 1h index via forward-fill
    if df4h is not None:
        rsi_4h  = rsi(df4h["close"]).rename("rsi_4h")
        macd_4h = macd_hist(df4h["close"]).rename("macd_4h")
        stch_4h = stoch_k(df4h["high"], df4h["low"], df4h["close"]).rename("stoch_4h")
        df4h_sigs = pd.concat([rsi_4h, macd_4h, stch_4h], axis=1)
        df4h_sigs = df4h_sigs.reindex(df1h.index, method="ffill")
        sigs = sigs.join(df4h_sigs)
    else:
        sigs["rsi_4h"]   = np.nan
        sigs["macd_4h"]  = np.nan
        sigs["stoch_4h"] = np.nan

    return sigs


# ── Build contract rows ───────────────────────────────────────────────────────

def build_contracts(sigs):
    """
    For each bar i, generate synthetic contracts at OFFSETS.
    outcome = 1 if close[i+1] > strike.
    """
    rows = []
    closes    = sigs["close"].values
    sigma_arr = sigs["sigma_tau"].values
    vol_arr   = sigs["vol_pm"].values
    mu6_arr   = sigs["mu6h"].values
    mu12_arr  = sigs["mu12h"].values
    mu24_arr  = sigs["mu24h"].values
    rz_arr    = sigs["regime_z"].values
    rsi1_arr  = sigs["rsi_1h"].values
    rsi4_arr  = sigs.get("rsi_4h", pd.Series(np.nan, index=sigs.index)).values
    macd4_arr = sigs.get("macd_4h", pd.Series(np.nan, index=sigs.index)).values
    stch4_arr = sigs.get("stoch_4h", pd.Series(np.nan, index=sigs.index)).values
    idx       = sigs.index

    for i in range(len(closes) - 1):
        if np.isnan(sigma_arr[i]) or sigma_arr[i] <= 0:
            continue
        spot       = closes[i]
        next_close = closes[i + 1]
        st         = sigma_arr[i]

        for off in OFFSETS:
            strike   = spot * (1.0 + off)
            z_str    = np.log(strike / spot) / st
            pm_synth = float(np.clip(1.0 - norm.cdf(z_str), EPS, 1 - EPS))
            outcome  = 1 if next_close > strike else 0

            rows.append({
                "ts":       idx[i],
                "year":     idx[i].year,
                "quarter":  f"{idx[i].year}Q{idx[i].quarter}",
                "offset":   off,
                "spot":     spot,
                "strike":   strike,
                "z_strike": z_str,
                "sigma_tau": st,
                "pm":       pm_synth,
                "outcome":  outcome,
                "mu6h":     mu6_arr[i],
                "mu12h":    mu12_arr[i],
                "mu24h":    mu24_arr[i],
                "regime_z": rz_arr[i],
                "rsi_1h":   rsi1_arr[i],
                "rsi_4h":   rsi4_arr[i],
                "macd_4h":  macd4_arr[i],
                "stoch_4h": stch4_arr[i],
            })

    return pd.DataFrame(rows)


# ── IC computation ────────────────────────────────────────────────────────────

def compute_ic(df, z_drift_arr, label=""):
    """
    Given z_drift for each row in df, compute pm-adjusted IC for YES and NO sides.
    IC_adj_yes = corr(edge_yes, outcome - pm)  on YES-eligible rows (edge_yes > 0)
    IC_adj_no  = corr(edge_no,  (1-outcome) - (1-pm)) on NO-eligible rows (edge_no > 0)
    """
    pm      = df["pm"].values
    outcome = df["outcome"].values
    p_yes   = np.clip(1.0 - norm.cdf(df["z_strike"].values - z_drift_arr), EPS, 1 - EPS)
    edge_yes = p_yes - pm
    edge_no  = pm - p_yes   # = (1-p_yes) - (1-pm)

    y_adj_yes = outcome - pm           # actual minus market-implied YES
    y_adj_no  = (1 - outcome) - (1 - pm)  # = pm - outcome

    yes_mask = edge_yes > 0
    no_mask  = edge_no  > 0

    ic_yes = ic_no = np.nan
    n_yes  = yes_mask.sum()
    n_no   = no_mask.sum()

    if n_yes >= MIN_N:
        ic_yes, _ = pearsonr(edge_yes[yes_mask], y_adj_yes[yes_mask])
    if n_no >= MIN_N:
        ic_no, _ = pearsonr(edge_no[no_mask], y_adj_no[no_mask])

    return ic_yes, ic_no, n_yes, n_no


def fmt(v):
    return f"{v:+.4f}" if v == v else "   nan"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading 1h and 4h BTC data...")
    df1h = load_1h()
    df4h = load_4h()
    print(f"  1h: {df1h.index.min().date()} -> {df1h.index.max().date()}  ({len(df1h):,} bars)")
    if df4h is not None:
        print(f"  4h: {df4h.index.min().date()} -> {df4h.index.max().date()}  ({len(df4h):,} bars)")

    print("Computing signals...")
    sigs = build_signals(df1h, df4h)

    print("Building synthetic contracts...")
    df = build_contracts(sigs)
    print(f"  Total rows: {len(df):,}  ({df['ts'].min().date()} -> {df['ts'].max().date()})")
    print(f"  Quarters:   {sorted(df['quarter'].unique())}")

    sq  = np.sqrt(TAU / 60.0)
    t60 = TAU / 60.0
    st  = df["sigma_tau"].values
    m6  = df["mu6h"].values
    m12 = df["mu12h"].values
    m24 = df["mu24h"].values
    rz  = df["regime_z"].values
    r1h = df["rsi_1h"].values
    r4h = df["rsi_4h"].values
    m4h = df["macd_4h"].values
    s4h = df["stoch_4h"].values

    # Normalise 4h signals to [-1, +1] range for comparability
    rsi_z_4h   = np.where(np.isnan(r4h), 0.0, (r4h - 50.0) / 25.0)   # >0 = overbought
    rsi_z_1h   = np.where(np.isnan(r1h), 0.0, (r1h - 50.0) / 25.0)
    macd_z_4h  = np.where(np.isnan(m4h), 0.0, m4h / (np.nanstd(m4h) + 1e-10))
    stoch_z_4h = np.where(np.isnan(s4h), 0.0, (s4h - 50.0) / 25.0)

    FORMULAS = {
        "no_drift":          np.zeros(len(df)),
        "mu6h":              m6  * t60 / st,
        "mu12h":             m12 * t60 / st,
        "mu24h":             m24 * t60 / st,
        "mu6_24":            (m6 + m24) * t60 / st,
        "mu_all":            (m6 + m12 + m24) * t60 / st,
        "regime_z":          rz * sq,
        "mu6h_rz":           m6 * t60 / st + rz * sq,
        "mu6_24_rz":         (m6 + m24) * t60 / st + rz * sq,
        "mu_all_rz":         (m6 + m12 + m24) * t60 / st + rz * sq,
        "rsi_4h_norm":       rsi_z_4h * sq,
        "macd_4h_norm":      macd_z_4h * sq,
        "stoch_4h_norm":     stoch_z_4h * sq,
        "mu6h+macd4h":       m6 * t60 / st + macd_z_4h * sq,
        "mu6_24+rsi4h":      (m6 + m24) * t60 / st + rsi_z_4h * sq,
    }

    # ── Full-period IC ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("FULL PERIOD IC (2024-2026, all regimes)")
    print("IC_adj > 0 = formula finds edge BEYOND pure log-normal pricing")
    print("=" * 80)
    print(f"  {'Formula':<22}  {'IC_yes':>9}  {'n_yes':>7}  {'IC_no':>9}  {'n_no':>7}  {'diff':>8}")
    print("  " + "-" * 70)

    full_results = []
    for fname, zd in FORMULAS.items():
        ic_yes, ic_no, n_yes, n_no = compute_ic(df, zd)
        diff = ic_yes - ic_no if ic_yes == ic_yes and ic_no == ic_no else np.nan
        full_results.append(dict(
            formula=fname, ic_yes=ic_yes, ic_no=ic_no,
            n_yes=n_yes, n_no=n_no, diff=diff))
        print(f"  {fname:<22}  {fmt(ic_yes):>9}  {n_yes:>7}  {fmt(ic_no):>9}  {n_no:>7}  {fmt(diff):>8}")

    # ── Quarterly stability check ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("QUARTERLY STABILITY — IC_yes by quarter (robust if consistent sign)")
    print("=" * 80)
    quarters = sorted(df["quarter"].unique())
    # Print top 6 formulas by abs(IC_yes) for the stability check
    top6 = sorted(full_results, key=lambda x: abs(x["ic_yes"]) if x["ic_yes"]==x["ic_yes"] else 0, reverse=True)[:6]
    top_names = [r["formula"] for r in top6]

    header = f"  {'Quarter':<10}  {'n_rows':>7}"
    for nm in top_names:
        header += f"  {nm[:10]:>10}"
    print(header)
    print("  " + "-" * (12 + 9 + len(top_names) * 12))

    for q in quarters:
        sub = df[df["quarter"] == q]
        if len(sub) < 500:
            continue
        row = f"  {q:<10}  {len(sub):>7}"
        st_sub = sub["sigma_tau"].values
        m6s = sub["mu6h"].values; m12s = sub["mu12h"].values; m24s = sub["mu24h"].values
        rzs = sub["regime_z"].values
        r4hs = sub["rsi_4h"].values; m4hs = sub["macd_4h"].values; s4hs = sub["stoch_4h"].values
        rsi_z4 = np.where(np.isnan(r4hs), 0.0, (r4hs - 50.0) / 25.0)
        mac_z4 = np.where(np.isnan(m4hs), 0.0, m4hs / (np.nanstd(m4hs) + 1e-10))

        FORMULAS_sub = {
            "no_drift":      np.zeros(len(sub)),
            "mu6h":          m6s * t60 / st_sub,
            "mu12h":         m12s * t60 / st_sub,
            "mu24h":         m24s * t60 / st_sub,
            "mu6_24":        (m6s + m24s) * t60 / st_sub,
            "mu_all":        (m6s + m12s + m24s) * t60 / st_sub,
            "regime_z":      rzs * sq,
            "mu6h_rz":       m6s * t60 / st_sub + rzs * sq,
            "mu6_24_rz":     (m6s + m24s) * t60 / st_sub + rzs * sq,
            "mu_all_rz":     (m6s + m12s + m24s) * t60 / st_sub + rzs * sq,
            "rsi_4h_norm":   rsi_z4 * sq,
            "macd_4h_norm":  mac_z4 * sq,
            "stoch_4h_norm": np.where(np.isnan(s4hs), 0.0, (s4hs-50)/25) * sq,
            "mu6h+macd4h":   m6s * t60 / st_sub + mac_z4 * sq,
            "mu6_24+rsi4h":  (m6s + m24s) * t60 / st_sub + rsi_z4 * sq,
        }
        for nm in top_names:
            zd_sub = FORMULAS_sub[nm]
            ic_q, _, _, _ = compute_ic(sub, zd_sub)
            row += f"  {fmt(ic_q):>10}"
        print(row)

    # ── YES ranking ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RANKED BY YES IC_adj (full period)")
    print("=" * 80)
    for r in sorted(full_results, key=lambda x: x["ic_yes"] if x["ic_yes"]==x["ic_yes"] else -9, reverse=True):
        print(f"  {r['formula']:<22}  YES={fmt(r['ic_yes'])}  NO={fmt(r['ic_no'])}  diff={fmt(r['diff'])}")

    # ── NO ranking ───────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RANKED BY NO IC_adj (full period)")
    print("=" * 80)
    for r in sorted(full_results, key=lambda x: x["ic_no"] if x["ic_no"]==x["ic_no"] else -9, reverse=True):
        print(f"  {r['formula']:<22}  YES={fmt(r['ic_yes'])}  NO={fmt(r['ic_no'])}  diff={fmt(r['diff'])}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    best_yes = max(full_results, key=lambda x: x["ic_yes"] if x["ic_yes"]==x["ic_yes"] else -9)
    best_no  = max(full_results, key=lambda x: x["ic_no"]  if x["ic_no"]==x["ic_no"]  else -9)

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  Best YES formula: {best_yes['formula']:<22}  IC_adj={best_yes['ic_yes']:+.4f}")
    print(f"  Best NO  formula: {best_no['formula']:<22}  IC_adj={best_no['ic_no']:+.4f}")

    if best_yes["formula"] == best_no["formula"]:
        print("  -> SAME formula optimal for both sides: shared model is justified")
    else:
        print(f"  -> DIFFERENT formulas optimal: separate YES and NO models are justified")

    # Check regime_z specifically
    rz_yes = next(r["ic_yes"] for r in full_results if r["formula"] == "regime_z")
    rz_no  = next(r["ic_no"]  for r in full_results if r["formula"] == "regime_z")
    mu6_yes = next(r["ic_yes"] for r in full_results if r["formula"] == "mu6h")
    mu6_24_yes = next(r["ic_yes"] for r in full_results if r["formula"] == "mu6_24")
    mu6_24_rz_yes = next(r["ic_yes"] for r in full_results if r["formula"] == "mu6_24_rz")

    print()
    print("  STRUCTURAL FINDINGS:")
    print(f"  regime_z alone:     YES IC={rz_yes:+.4f}  NO IC={rz_no:+.4f}")
    print(f"  mu6h alone:         YES IC={mu6_yes:+.4f}")
    print(f"  mu6_24:             YES IC={mu6_24_yes:+.4f}")
    print(f"  mu6_24 + regime_z:  YES IC={mu6_24_rz_yes:+.4f}")
    delta_rz_yes = mu6_24_rz_yes - mu6_24_yes
    print(f"  Adding regime_z to mu6_24: delta YES IC = {delta_rz_yes:+.4f}")
    print()
    if abs(delta_rz_yes) < 0.0005:
        print("  -> regime_z adds essentially NOTHING to mu6_24 for YES")
    elif delta_rz_yes > 0:
        print("  -> regime_z HELPS YES when combined with mu6_24")
    else:
        print("  -> regime_z HURTS YES when combined with mu6_24")


if __name__ == "__main__":
    main()
