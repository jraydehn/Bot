#!/usr/bin/env python3
"""
simulate_signal_rescues.py

For each major existing gate condition, reconstruct the blocked bucket from
the paper trade archive, join Coinalyze liq_score + Deribit DVOL, and find
any sub-slice where liq_score or DVOL pushes WR above breakeven.

Existing gates analysed (reconstructed from CSV columns):
  BTC YES:
    1. BearDrift     (ema=-1, rev<=3, stoch>=35)
    2. BearDrift arm2 (ema=-1, rev<=3, stoch<25, OTM)
    3. Exhaustion    (ema=1, rev<=-4, stoch>=75, vwap_stretch=-1)
    4. OTM-neutral   (ema=0, p_up>=0.60, OTM)
    5. ema0-ITM      (ema=0, ITM, trend>=3, rev==0)
    6. Falling knife (rev>=4, chg_30m<-0.20%)
    7. vol=1         (vol_score==1)
  BTC NO:
    8. nopup         ((p_up<=0.36 OR p_up>=0.50), pm>=0.20)

Run: python3 simulate_signal_rescues.py
"""

import glob, math, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from coinalyze_liq import _LIQ_BIAS_STRONG, _LS_CROWD_THRESH

KEY  = "d5841821-3f45-4e5f-9ee7-d2779d2fb01b"
BASE = "https://api.coinalyze.net/v1"
SEP  = "=" * 72
SEP2 = "-" * 72
MINS_PER_YEAR = 365 * 24 * 60
BTC_WEIGHT = 0.35


# ── data loading ──────────────────────────────────────────────────────────────

def _compute_liq_score(long_liq, short_liq, ls_long, ls_short):
    tot = long_liq + short_liq
    bias = (short_liq - long_liq) / tot if tot > 0.001 else 0.0
    score = 0
    if bias >= _LIQ_BIAS_STRONG:   score += 1
    elif bias <= -_LIQ_BIAS_STRONG: score -= 1
    if ls_short >= _LS_CROWD_THRESH: score += 1
    elif ls_long >= _LS_CROWD_THRESH: score -= 1
    return max(-2, min(2, score))


def fetch_liq_scores_series():
    now_unix = int(time.time())
    far_past = now_unix - 90 * 24 * 3600
    r_liq = requests.get(f"{BASE}/liquidation-history",
        params={"symbols": "BTCUSDT_PERP.A", "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=15)
    r_ls  = requests.get(f"{BASE}/long-short-ratio-history",
        params={"symbols": "BTCUSDT_PERP.A", "interval": "15min",
                "from": far_past, "to": now_unix, "api_key": KEY}, timeout=15)
    df_liq = pd.DataFrame(r_liq.json()[0]["history"], columns=["t","l","s"])
    df_liq["t"] = pd.to_datetime(df_liq["t"], unit="s", utc=True)
    df_liq = df_liq.set_index("t")
    df_ls = pd.DataFrame(r_ls.json()[0]["history"])
    df_ls["t"] = pd.to_datetime(df_ls["t"], unit="s", utc=True)
    df_ls = df_ls.set_index("t")
    shared = df_liq.index.intersection(df_ls.index)
    scores = {T: _compute_liq_score(float(df_liq.loc[T,"l"]), float(df_liq.loc[T,"s"]),
                                     float(df_ls.loc[T,"l"]), float(df_ls.loc[T,"s"]))
              for T in shared}
    return pd.Series(scores, name="liq_score")


def fetch_dvol_series():
    now_ms = int(time.time() * 1000)
    far_ms = now_ms - 90 * 24 * 3600 * 1000
    resp = requests.get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={"currency": "BTC", "resolution": "3600",
                "start_timestamp": far_ms, "end_timestamp": now_ms}, timeout=12)
    rows = resp.json()["result"]["data"]
    df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("h")
    return (df.set_index("ts")["close"] / 100.0)   # annualized decimal


def load_btc_trades():
    frames = []
    for path in glob.glob("results/paper_trades*.csv"):
        try:
            chunk = pd.read_csv(path)
            chunk = chunk[chunk["would_win"].notna()].copy()
            if "contract_ticker" not in chunk.columns: continue
            chunk = chunk[chunk["contract_ticker"].str.startswith("KXBTC", na=False)]
            frames.append(chunk)
        except Exception:
            pass
    df = pd.concat(frames).drop_duplicates(subset=["contract_ticker","decision_time"]).copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
    num_cols = ["offset_pct","p_market","p_yes_model","composite_p_up","composite_rev",
                "composite_trend","stoch_k","ema_stack_bias","vwap_stretch_score",
                "chg_30m","chg_10m","chg_5m","vol_score","tau_minutes","vpin_raw",
                "vpin_score","ema_stretch_score","would_pnl","confirmation_score"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def join_signals(df, liq_scores, dvol_series):
    floored_15 = df["decision_time"].dt.floor("15min")
    df = df.copy()
    df["liq_score"] = floored_15.map(liq_scores)
    prev = floored_15 - pd.Timedelta(minutes=15)
    df["liq_score"] = df["liq_score"].fillna(prev.map(liq_scores))

    floored_1h = df["decision_time"].dt.floor("h")
    df["dvol"] = floored_1h.map(dvol_series)
    prev_1h = floored_1h - pd.Timedelta(hours=1)
    df["dvol"] = df["dvol"].fillna(prev_1h.map(dvol_series))
    df["dvol_sigma_per_min"] = df["dvol"] / math.sqrt(MINS_PER_YEAR)
    return df


# ── analysis helpers ──────────────────────────────────────────────────────────

def _stats(sub, label=""):
    n = len(sub)
    if n == 0:
        return
    wr  = sub["would_win"].mean()
    be  = sub["p_market"].mean()
    pnl = sub["would_pnl"].sum()
    w   = int(sub["would_win"].sum())
    l   = n - w
    d   = wr - be
    flag = "★" if abs(d) > 0.05 and n >= 15 else ""
    print(f"  {label:<45}  n={n:>4}  W={w:>3}/L={l:<3}  WR={wr:.1%}  BE={be:.1%}  Δ={d:>+.1%}  P&L=${pnl:>+,.0f}  {flag}")


def rescue_search(bucket, label):
    """Find liq_score or dvol sub-slices with WR > BE within bucket."""
    if len(bucket) == 0:
        return
    cands = []
    for col, op, thresholds in [
        ("liq_score", ">=", [1, 2]),
        ("liq_score", "<=", [-1, -2]),
        ("liq_score", "==", [0, 1, -1]),
        ("dvol",      ">=", [0.40, 0.42, 0.44]),
        ("dvol",      "<=", [0.39, 0.40, 0.41]),
        ("dvol_sigma_per_min", ">=", [0.00055, 0.00057, 0.00059]),
    ]:
        if col not in bucket.columns: continue
        for t in thresholds:
            if op == ">=": mask = bucket[col] >= t
            elif op == "<=": mask = bucket[col] <= t
            else: mask = bucket[col] == t
            sub = bucket[mask]
            if len(sub) < 5: continue
            wr = sub["would_win"].mean(); be = sub["p_market"].mean()
            pnl = sub["would_pnl"].sum(); n = len(sub)
            w = int(sub["would_win"].sum()); l = n - w
            cands.append({"cond": f"{col}{op}{t}", "n": n, "w": w, "l": l,
                          "wr": wr, "be": be, "delta": wr-be, "pnl": pnl})

    # Also test liq_score pairs with existing gate-adjacent columns
    for liq_t in [1, -1]:
        for col, op, thresholds in [
            ("vpin_raw",    ">=", [0.70, 0.75, 0.80]),
            ("p_market",    ">=", [0.35, 0.40, 0.45]),
            ("offset_pct",  "<=", [0, 0.05, 0.08]),
            ("stoch_k",     "<=", [25, 30, 35]),
            ("composite_rev",">=", [3, 5, 7]),
        ]:
            if col not in bucket.columns: continue
            liq_mask = bucket["liq_score"] == liq_t
            for t in thresholds:
                if op == ">=": cmask = bucket[col] >= t
                elif op == "<=": cmask = bucket[col] <= t
                else: cmask = bucket[col] == t
                sub = bucket[liq_mask & cmask]
                if len(sub) < 5: continue
                wr = sub["would_win"].mean(); be = sub["p_market"].mean()
                pnl = sub["would_pnl"].sum(); n = len(sub)
                w = int(sub["would_win"].sum()); l = n - w
                cands.append({"cond": f"liq=={liq_t:+d} AND {col}{op}{t}",
                               "n": n, "w": w, "l": l,
                               "wr": wr, "be": be, "delta": wr-be, "pnl": pnl})

    above_be = [c for c in cands if c["delta"] > 0]
    if not above_be:
        print(f"    No liq/DVOL sub-slice above breakeven in {label}.")
        return
    above_be.sort(key=lambda x: (x["delta"], x["n"]), reverse=True)
    print(f"    {'Condition':<45}  {'n':>4}  {'W/L':>6}  {'WR':>6}  {'BE':>6}  {'Δ':>6}  {'P&L':>8}")
    print(f"    {'-'*92}")
    for c in above_be[:8]:
        flag = "★" if c["delta"] > 0.05 and c["n"] >= 10 else ""
        print(f"    {c['cond']:<45}  {c['n']:>4}  {c['w']:>2}/{c['l']:<3}  {c['wr']:>5.1%}  "
              f"{c['be']:>5.1%}  {c['delta']:>+5.1%}  ${c['pnl']:>+7,.0f}  {flag}")


# ── gate definitions (reconstructed from CSV columns) ─────────────────────────

def analyse_gate(df, name, mask, side, has_liq, has_dvol):
    bucket = df[mask & (df["side"] == side) & df["liq_score"].notna()].copy()
    if not has_liq:
        bucket = df[mask & (df["side"] == side) & df["dvol"].notna()].copy()

    print(f"\n{SEP2}")
    print(f"  Gate: {name}  ({side.upper()})")
    print(SEP2)
    _stats(bucket, "ALL (simulated block)")
    if len(bucket) < 5:
        print("    Too few trades to analyse.")
        return

    print(f"\n  By liq_score:")
    for sc in sorted(bucket["liq_score"].dropna().unique()):
        _stats(bucket[bucket["liq_score"] == sc],
               f"    liq_score={int(sc):+d}")

    if has_dvol and bucket["dvol"].notna().sum() >= 10:
        dvol_med = bucket["dvol"].median()
        print(f"\n  By DVOL (median={dvol_med*100:.1f}%):")
        _stats(bucket[bucket["dvol"] <  dvol_med], f"    DVOL < {dvol_med*100:.1f}% (low vol)")
        _stats(bucket[bucket["dvol"] >= dvol_med], f"    DVOL >= {dvol_med*100:.1f}% (high vol)")

    print(f"\n  liq/DVOL rescue search:")
    rescue_search(bucket, name)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Signal Rescue Analysis — liq_score + DVOL across existing gates")
    print(SEP)

    print("\nFetching Coinalyze liq scores...", end=" ", flush=True)
    liq_scores = fetch_liq_scores_series()
    print(f"{len(liq_scores)} bars  ({liq_scores.index[0].date()} → {liq_scores.index[-1].date()})")

    print("Fetching Deribit DVOL...", end=" ", flush=True)
    dvol_series = fetch_dvol_series()
    print(f"{len(dvol_series)} hourly bars")

    print("Loading BTC paper trades...", end=" ", flush=True)
    df = load_btc_trades()
    print(f"{len(df)} resolved")

    df = join_signals(df, liq_scores, dvol_series)
    has_liq  = df["liq_score"].notna().sum() > 50
    has_dvol = df["dvol"].notna().sum() > 50
    print(f"  liq_score matched: {df['liq_score'].notna().sum()}  |  DVOL matched: {df['dvol'].notna().sum()}")

    cp  = df["composite_p_up"]
    cr  = df["composite_rev"]
    ct  = df["composite_trend"]
    ema = df["ema_stack_bias"]
    sk  = df["stoch_k"]
    vws = df["vwap_stretch_score"]
    vs  = df["vol_score"]
    off = df["offset_pct"]
    pm  = df["p_market"]
    c30 = df["chg_30m"]

    gates = [
        # name, mask, side
        ("BearDrift arm1 (ema=-1, rev<=3, sk>=35)",
         (ema == -1) & (cr <= 3) & (sk >= 35), "yes"),

        ("BearDrift arm2 (ema=-1, rev<=3, sk<25, OTM)",
         (ema == -1) & (cr <= 3) & (sk < 25) & (off > 0), "yes"),

        ("Exhaustion (ema=1, rev<=-4, sk>=75, vwap=-1)",
         (ema == 1) & (cr <= -4) & (sk >= 75) & (vws == -1), "yes"),

        ("OTM-neutral (ema=0, p_up>=0.60, OTM)",
         (ema == 0) & (cp >= 0.60) & (off > 0), "yes"),

        ("ema0-ITM (ema=0, ITM, trend>=3, rev==0)",
         (ema == 0) & (off <= 0) & (ct >= 3) & (cr == 0), "yes"),

        ("Falling knife (rev>=4, chg30m<-0.20%)",
         (cr >= 4) & (c30 < -0.002), "yes"),

        ("vol=1 YES (vol_score==1)",
         (vs == 1), "yes"),

        ("nopup NO (p_up<=0.36 or p_up>=0.50, pm>=0.20)",
         ((cp <= 0.36) | (cp >= 0.50)) & (pm >= 0.20), "no"),
    ]

    for name, mask, side in gates:
        analyse_gate(df, name, mask, side, has_liq, has_dvol)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
