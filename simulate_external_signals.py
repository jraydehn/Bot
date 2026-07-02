#!/usr/bin/env python3
"""
simulate_external_signals.py

Section 1 — Coinalyze liq_score predictive value
  Fetches all available 15min liq + L/S history, computes liq_score per bar,
  then checks 15m / 30m / 60m forward BTC price outcome.
  Reports WR% / n / edge vs baseline by score bucket.

Section 2 — Deribit DVOL vs Kalshi-implied vol
  Loads resolved BTC paper trades. For each trade, looks up contemporaneous
  Deribit DVOL and computes what vol_eff would have been. Reports:
  - Mean / std of DVOL-based vs Kalshi-based vol_eff
  - Trades where the substitution changes vol_eff by > 10%
  - Stability comparison (CV of each series)

Run: python3 simulate_external_signals.py
"""

import glob
import math
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH
import deribit_iv

KEY  = "d5841821-3f45-4e5f-9ee7-d2779d2fb01b"
BASE = "https://api.coinalyze.net/v1"
SEP  = "=" * 72
SEP2 = "-" * 72

BTC_WEIGHT = 0.35  # REALIZED_VOL_WEIGHT_BY_ASSET["BTC"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_liq_score(liq_bias: float, ls_long: float, ls_short: float) -> int:
    score = 0
    if liq_bias >= _LIQ_BIAS_STRONG:
        score += 1
    elif liq_bias <= -_LIQ_BIAS_STRONG:
        score -= 1
    if ls_short >= _LS_CROWD_THRESH:
        score += 1
    elif ls_long >= _LS_CROWD_THRESH:
        score -= 1
    return max(-2, min(2, score))


def _fetch_coinalyze(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    now_unix = int(time.time())
    far_past = now_unix - 90 * 24 * 3600

    r_liq = requests.get(f"{BASE}/liquidation-history",
        params={"symbols": symbol, "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=12)
    r_liq.raise_for_status()

    r_ls = requests.get(f"{BASE}/long-short-ratio-history",
        params={"symbols": symbol, "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=12)
    r_ls.raise_for_status()

    liq_rows = r_liq.json()[0]["history"]
    ls_rows  = r_ls.json()[0]["history"]

    df_liq = pd.DataFrame(liq_rows, columns=["t", "l", "s"])
    df_liq["t"] = pd.to_datetime(df_liq["t"], unit="s", utc=True)
    df_liq = df_liq.set_index("t")

    df_ls = pd.DataFrame(ls_rows)
    df_ls["t"] = pd.to_datetime(df_ls["t"], unit="s", utc=True)
    df_ls = df_ls.set_index("t")

    return df_liq, df_ls


def _load_price_1m(asset: str) -> pd.Series:
    sym = f"{asset}USDT"
    f = sorted(glob.glob(f"data/binanceus_{sym}_1m_2024-01-01_*.parquet"))[-1]
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    return df["close"].astype(float).sort_index()


def _forward_outcome(close_1m: pd.Series, T: pd.Timestamp, horizon_min: int):
    T_fut = T + pd.Timedelta(minutes=horizon_min)
    fut_idx = close_1m.index[close_1m.index >= T_fut]
    now_idx = close_1m.index[close_1m.index >= T]
    if len(fut_idx) == 0 or len(now_idx) == 0:
        return None
    c_now = close_1m.iloc[close_1m.index.get_loc(now_idx[0])]
    c_fut = close_1m.iloc[close_1m.index.get_loc(fut_idx[0])]
    return int(c_fut > c_now)


def _print_bucket_table(df: pd.DataFrame, horizons: list[str], baseline: dict[str, float]):
    buckets = sorted(df["liq_score"].unique())
    labels  = {-2: "CASCADE--", -1: "cascade- ", 0: "neutral  ", 1: "squeeze+ ", 2: "SQUEEZE++"}
    header  = f"  {'Score':<12}" + "".join(f"  {'n':>5}  {'WR':>6}  {'edge':>6}" for _ in horizons)
    hz_hdr  = "  " + " " * 12 + "".join(f"  [{h:^18}]" for h in horizons)
    print(hz_hdr)
    print(header)
    print(f"  {SEP2}")
    for sc in buckets:
        sub = df[df["liq_score"] == sc]
        row = f"  {sc:+d} {labels.get(sc, '?'):<9}"
        for h in horizons:
            col = f"out_{h}"
            if col not in sub.columns:
                row += "  " + " " * 21
                continue
            valid = sub[col].dropna()
            n = len(valid)
            if n < 5:
                row += f"  {'—':>5}  {'—':>6}  {'—':>6}"
                continue
            wr   = valid.mean()
            edge = wr - baseline[h]
            star = "★" if abs(edge) > 0.05 and n >= 30 else ""
            row += f"  {n:>5}  {wr:>5.1%}  {edge:>+5.1%}{star}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Coinalyze liq_score predictive value
# ═══════════════════════════════════════════════════════════════════════════════

def section1_liq_signal():
    print(f"\n{SEP}")
    print("  SECTION 1 — Coinalyze liq_score → Forward Price Outcome")
    print(f"{SEP}")

    for asset, symbol in [("BTC", "BTCUSDT_PERP.A"), ("ETH", "ETHUSDT_PERP.A")]:
        print(f"\n── {asset} ──────────────────────────────────────────────────────")
        print("  Fetching Coinalyze data...", end=" ", flush=True)
        df_liq, df_ls = _fetch_coinalyze(symbol)
        print(f"liq={len(df_liq)} bars  ls={len(df_ls)} bars")
        print(f"  Range: {df_liq.index[0].date()} → {df_liq.index[-1].date()}")

        print(f"  Loading {asset} 1m price data...", end=" ", flush=True)
        close_1m = _load_price_1m(asset)
        print("done")

        # Align on shared timestamps
        shared = df_liq.index.intersection(df_ls.index)
        df_liq = df_liq.loc[shared]
        df_ls  = df_ls.loc[shared]

        # Build scored dataframe
        rows = []
        for T in shared:
            long_liq  = float(df_liq.loc[T, "l"])
            short_liq = float(df_liq.loc[T, "s"])
            total_liq = long_liq + short_liq
            liq_bias  = (short_liq - long_liq) / total_liq if total_liq > 0.001 else 0.0
            ls_long   = float(df_ls.loc[T, "l"])
            ls_short  = float(df_ls.loc[T, "s"])
            score = _compute_liq_score(liq_bias, ls_long, ls_short)
            rows.append({"t": T, "liq_bias": liq_bias, "ls_long": ls_long,
                          "ls_short": ls_short, "liq_score": score})

        df = pd.DataFrame(rows).set_index("t")

        print(f"\n  Computing forward outcomes ({len(df)} bars)...", end=" ", flush=True)
        for h in [15, 30, 60]:
            df[f"out_{h}m"] = [_forward_outcome(close_1m, T, h) for T in df.index]
        print("done")

        # Baseline win rates
        baseline = {f"{h}m": df[f"out_{h}m"].dropna().mean() for h in [15, 30, 60]}
        print(f"\n  Baseline (unconditional up%):")
        for h in [15, 30, 60]:
            print(f"    {h}m: {baseline[f'{h}m']:.1%}  (n={df[f'out_{h}m'].dropna().__len__()})")

        # Score distribution
        print(f"\n  liq_score distribution:")
        vc = df["liq_score"].value_counts().sort_index()
        for sc, cnt in vc.items():
            label = {-2: "CASCADE--", -1: "cascade- ", 0: "neutral  ", 1: "squeeze+ ", 2: "SQUEEZE++"}
            print(f"    {sc:+d}  {label.get(sc,'?')}: {cnt:>4} bars  ({cnt/len(df):.1%})")

        # Predictive table
        print(f"\n  Predictive value by liq_score:")
        _print_bucket_table(df.rename(columns={f"out_{h}m": f"out_{h}m" for h in [15,30,60]}),
                            ["15m", "30m", "60m"], baseline)

        # liq_bias vs outcome (continuous breakdown)
        print(f"\n  liq_bias quintile breakdown (30m outcome):")
        df["bias_bin"] = pd.cut(df["liq_bias"],
                                bins=[-1.01, -0.6, -0.2, 0.2, 0.6, 1.01],
                                labels=["[-1,-0.6]", "[-0.6,-0.2]", "[-0.2,+0.2]",
                                        "[+0.2,+0.6]", "[+0.6,+1]"])
        print(f"  {'Bias bin':<14}  {'n':>5}  {'30m WR':>7}  {'edge':>6}")
        print(f"  {'-'*40}")
        for b, grp in df.groupby("bias_bin", observed=True):
            valid = grp["out_30m"].dropna()
            n = len(valid)
            if n < 5:
                continue
            wr = valid.mean()
            edge = wr - baseline["30m"]
            print(f"  {str(b):<14}  {n:>5}  {wr:>7.1%}  {edge:>+6.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Deribit DVOL vs Kalshi-implied vol
# ═══════════════════════════════════════════════════════════════════════════════

def section2_dvol_vs_kalshi():
    print(f"\n{SEP}")
    print("  SECTION 2 — Deribit DVOL vs Kalshi-implied vol (BTC)")
    print(f"{SEP}")

    # Load resolved BTC paper trades (combined file, all assets; filter for BTC)
    df_all = pd.read_csv("results/paper_trades.csv")
    df_all = df_all[df_all["would_win"].notna()].copy()
    df_all["decision_time"] = pd.to_datetime(df_all["decision_time"], utc=True)
    df_trades = df_all[df_all["contract_ticker"].str.startswith("KXBTC", na=False)].copy()
    df_trades = df_trades[df_trades["vol_implied_kalshi"].notna()].copy()
    df_trades = df_trades[pd.to_numeric(df_trades["vol_implied_kalshi"],
                                         errors="coerce").notna()].copy()
    df_trades["vol_implied_kalshi"] = df_trades["vol_implied_kalshi"].astype(float)
    df_trades["vol_60m_model"]      = df_trades["vol_60m_model"].astype(float)
    df_trades["vol_eff"]            = df_trades["vol_eff"].astype(float)
    df_trades["tau_minutes"]        = pd.to_numeric(df_trades["tau_minutes"], errors="coerce")

    print(f"\n  Resolved BTC trades with Kalshi IV: {len(df_trades)}")
    print(f"  Date range: {df_trades['decision_time'].min().date()} → "
          f"{df_trades['decision_time'].max().date()}")

    # Fetch hourly Deribit DVOL
    print("\n  Fetching Deribit historical BTC DVOL (hourly)...", end=" ", flush=True)
    now_ms = int(time.time() * 1000)
    far_past_ms = now_ms - 90 * 24 * 3600 * 1000
    resp = requests.get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={"currency": "BTC", "resolution": "3600",
                "start_timestamp": far_past_ms, "end_timestamp": now_ms},
        timeout=12)
    dvol_rows = resp.json()["result"]["data"]
    # Each row: [timestamp_ms, open, high, low, close]
    dvol_df = pd.DataFrame(dvol_rows, columns=["ts_ms", "open", "high", "low", "close"])
    dvol_df["ts"] = pd.to_datetime(dvol_df["ts_ms"], unit="ms", utc=True).dt.floor("h")
    dvol_df = dvol_df.set_index("ts")["close"] / 100.0  # annualized decimal
    print(f"{len(dvol_df)} hourly bars ({dvol_df.index[0].date()} → {dvol_df.index[-1].date()})")

    # Match each trade to the nearest hourly DVOL bar
    def _get_dvol(dt: pd.Timestamp):
        h = dt.floor("h")
        if h in dvol_df.index:
            return dvol_df[h]
        # Try ±1h
        for offset in [-1, 1, -2, 2]:
            h2 = h + pd.Timedelta(hours=offset)
            if h2 in dvol_df.index:
                return dvol_df[h2]
        return None

    df_trades["dvol_annualized"] = df_trades["decision_time"].apply(_get_dvol)
    df_trades = df_trades[df_trades["dvol_annualized"].notna()].copy()
    print(f"  Trades matched to DVOL: {len(df_trades)}")

    # Compute DVOL-based per-minute sigma and vol_eff
    MINS_PER_YEAR = 365 * 24 * 60
    df_trades["dvol_sigma_per_min"] = df_trades["dvol_annualized"] / math.sqrt(MINS_PER_YEAR)
    df_trades["vol_eff_kalshi"] = (BTC_WEIGHT * df_trades["vol_60m_model"]
                                   + (1 - BTC_WEIGHT) * df_trades["vol_implied_kalshi"])
    df_trades["vol_eff_dvol"]   = (BTC_WEIGHT * df_trades["vol_60m_model"]
                                   + (1 - BTC_WEIGHT) * df_trades["dvol_sigma_per_min"])

    # ── Stability comparison ─────────────────────────────────────────────────
    print(f"\n  Vol series comparison (per-minute sigma × 1000):")
    print(f"  {'Metric':<28}  {'Kalshi-imp':>10}  {'Deribit DVOL':>12}  {'Realized':>10}")
    print(f"  {'-'*64}")
    for label, col in [("Kalshi-implied IV", "vol_implied_kalshi"),
                        ("Deribit DVOL σ/min", "dvol_sigma_per_min"),
                        ("Realized σ/min", "vol_60m_model")]:
        s = df_trades[col].dropna() * 1000
        cv = s.std() / s.mean() if s.mean() > 0 else float("nan")
        print(f"  {label:<28}  mean={s.mean():>6.3f}  std={s.std():>6.3f}  CV={cv:.2f}")

    print(f"\n  vol_eff comparison (what goes into sigma_tau):")
    print(f"  {'Metric':<28}  {'Kalshi-blend':>12}  {'DVOL-blend':>10}")
    print(f"  {'-'*56}")
    eff_k = df_trades["vol_eff_kalshi"] * 1000
    eff_d = df_trades["vol_eff_dvol"]   * 1000
    for label, s in [("mean", pd.Series([eff_k.mean(), eff_d.mean()])),
                      ("std",  pd.Series([eff_k.std(),  eff_d.std()])),
                      ("CV",   pd.Series([eff_k.std()/eff_k.mean(), eff_d.std()/eff_d.mean()]))]:
        print(f"  {label:<28}  {s.iloc[0]:>12.3f}  {s.iloc[1]:>10.3f}")

    # Difference distribution
    df_trades["vol_eff_diff_pct"] = ((df_trades["vol_eff_dvol"] - df_trades["vol_eff_kalshi"])
                                      / df_trades["vol_eff_kalshi"] * 100)
    diff = df_trades["vol_eff_diff_pct"]
    print(f"\n  DVOL-blend vs Kalshi-blend vol_eff difference (%):")
    print(f"    mean={diff.mean():+.1f}%  median={diff.median():+.1f}%  "
          f"std={diff.std():.1f}%  p5={diff.quantile(0.05):+.1f}%  p95={diff.quantile(0.95):+.1f}%")
    large = (diff.abs() > 10).sum()
    print(f"    Trades where |diff| > 10%: {large} / {len(df_trades)} ({large/len(df_trades):.1%})")

    # ── sigma_tau impact ─────────────────────────────────────────────────────
    df_trades["tau_valid"] = pd.to_numeric(df_trades["tau_minutes"], errors="coerce")
    df_valid = df_trades[df_trades["tau_valid"] > 0].copy()
    df_valid["sigma_tau_kalshi"] = df_valid["vol_eff_kalshi"] * np.sqrt(df_valid["tau_valid"])
    df_valid["sigma_tau_dvol"]   = df_valid["vol_eff_dvol"]   * np.sqrt(df_valid["tau_valid"])
    diff_st = (df_valid["sigma_tau_dvol"] - df_valid["sigma_tau_kalshi"]) / df_valid["sigma_tau_kalshi"] * 100
    print(f"\n  sigma_tau difference (DVOL vs Kalshi, %):")
    print(f"    mean={diff_st.mean():+.1f}%  median={diff_st.median():+.1f}%  std={diff_st.std():.1f}%")

    # ── DVOL trend vs Kalshi noise ───────────────────────────────────────────
    print(f"\n  Kalshi-implied IV distribution (extreme values are noise):")
    k_iv = df_trades["vol_implied_kalshi"].dropna() * 100
    pcts = [1, 5, 25, 50, 75, 95, 99]
    row  = "    " + "  ".join(f"p{p}={k_iv.quantile(p/100):.1f}%" for p in pcts)
    print(row)
    print(f"\n  Deribit DVOL distribution (annualized %):")
    d_iv = df_trades["dvol_annualized"].dropna() * 100
    row2 = "    " + "  ".join(f"p{p}={d_iv.quantile(p/100):.1f}%" for p in pcts)
    print(row2)

    extreme_kalshi = (k_iv > 200).sum()
    print(f"\n  Kalshi IV > 200% (unreliable OTM back-solve): {extreme_kalshi} trades "
          f"({extreme_kalshi/len(df_trades):.1%}) — Deribit doesn't have this problem.")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("  External Signal Simulation")
    print(SEP)
    section1_liq_signal()
    section2_dvol_vs_kalshi()
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
