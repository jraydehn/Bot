#!/usr/bin/env python3
"""
simulate_eth_sol_pup_regime.py

Tests ETH/SOL p_up_v2 as a rolling REGIME indicator for NO bets.
Same structure as simulate_pup_regime.py (BTC).

Regime logic:
  - Compute rolling mean of p_up_v2 over N prior 1h bars at trade time
  - If rolling_pup >= bull_thresh → BULL regime → block NO bets
  - Otherwise trade normally

Sweeps:
  roll_window  : [2, 4, 6, 8, 12]
  bull_thresh  : [0.52, 0.53, 0.54, 0.55, 0.56]

YES bets: unmodified (k=0 lognormal, no gate)
NO bets : blocked when bull regime fires

Flat $1000, EDGE_MIN=0.04, KELLY_CAP=0.25, k=0 pure lognormal.

Run:
  python3 simulate_eth_sol_pup_regime.py --asset ETH
  python3 simulate_eth_sol_pup_regime.py --asset SOL
"""

import argparse
import csv
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA      = ROOT / "data"
RESULTS   = ROOT / "results"

BANKROLL  = 1_000.0
EDGE_MIN  = 0.04
KELLY_CAP = 0.25

ROLL_WINDOWS    = [2, 4, 6, 8, 12]
BULL_THRESHOLDS = [0.52, 0.53, 0.54, 0.55, 0.56]
BEAR_THRESHOLDS = [0.44, 0.46, 0.48, 0.50, 0.52]

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── indicator helpers ──────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def stoch_k_series(h, lo, c, k=14):
    return (c - lo.rolling(k).min()) / (h.rolling(k).max() - lo.rolling(k).min()).replace(0, np.nan) * 100

def atr_series(h, lo, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, adjust=False).mean()

def macd_hist_series(c, f=12, s=26, sig=9):
    macd = _ema(c, f) - _ema(c, s)
    return macd - macd.ewm(span=sig, adjust=False).mean()

def bb_pct_series(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    return (c - (mid - 2*std)) / (4*std).replace(0, np.nan)

def ema50_dist_series(c):
    return (c - _ema(c, 50)) / _ema(c, 50).replace(0, np.nan) * 100

def chg_4h_atr_series(df4):
    a = atr_series(df4["high"], df4["low"], df4["close"], 14)
    return (df4["close"] - df4["close"].shift(5)) / a.replace(0, np.nan)

def daily_vwap_dist_series(df1h):
    tp  = (df1h["high"] + df1h["low"] + df1h["close"]) / 3
    vol = df1h["volume"]
    day = df1h.index.date
    df_tmp = pd.DataFrame({"tp": tp, "vol": vol, "day": day}, index=df1h.index)
    df_tmp["cum_tpv"] = df_tmp.groupby("day")["tp"].transform(
        lambda x: (x * df_tmp.loc[x.index, "vol"]).cumsum()
    )
    df_tmp["cum_vol"] = df_tmp.groupby("day")["vol"].transform("cumsum")
    vwap = df_tmp["cum_tpv"] / df_tmp["cum_vol"].replace(0, np.nan)
    dist = (df1h["close"] - vwap) / vwap.replace(0, np.nan)
    df_tmp["day_std"] = df_tmp.groupby("day")["tp"].transform("std")
    stretch = pd.cut(
        dist / (df_tmp["day_std"] / vwap.replace(0, np.nan)).replace(0, np.nan),
        bins=[-np.inf, -2, -1, 1, 2, np.inf], labels=[2, 1, 0, -1, -2],
    ).astype(float)
    return dist * 100, stretch

def compute_composite_signals(df1h, df4h, cal):
    c1h = df1h["close"]; c4h = df4h["close"]
    rsi4 = rsi_series(c4h, 14)
    macd4 = _ema(c4h, 12) - _ema(c4h, 26); sig4 = macd4.ewm(span=9, adjust=False).mean()
    bb_m4 = c4h.rolling(20).mean(); bb_s4 = c4h.rolling(20).std()
    bb_p4 = (c4h - (bb_m4 - 2*bb_s4)) / (4*bb_s4).replace(0, np.nan)
    sk4   = stoch_k_series(df4h["high"], df4h["low"], c4h, 14)
    wr4   = -100*(df4h["high"].rolling(14).max()-c4h) / \
            (df4h["high"].rolling(14).max()-df4h["low"].rolling(14).min()).replace(0, np.nan)
    vr4   = df4h["volume"] / df4h["volume"].rolling(20).mean().replace(0, np.nan)

    t4 = pd.Series(0.0, index=df4h.index)
    t4 += (rsi4>55).astype(float)-(rsi4<45).astype(float)
    t4 += (macd4>sig4).astype(float)-(macd4<=sig4).astype(float)
    t4 += (bb_p4>0.80).astype(float)-(bb_p4<0.20).astype(float)
    t4 += (sk4>80).astype(float)-(sk4<20).astype(float)
    t4 += (wr4>-20).astype(float)-(wr4<-80).astype(float)
    t4 += ((vr4>1.5)&(c4h>c4h.shift(1))).astype(float)-((vr4>1.5)&(c4h<c4h.shift(1))).astype(float)
    t4  = t4.clip(-6, 6)

    rsi1 = rsi_series(c1h, 14); sk1 = stoch_k_series(df1h["high"], df1h["low"], c1h, 14)
    vd, _ = daily_vwap_dist_series(df1h)
    zs = np.log(c1h/c1h.shift(1)) / np.log(c1h/c1h.shift(1)).rolling(24).std().replace(0, np.nan)

    r1 = pd.Series(0.0, index=df1h.index)
    r1 += 2*(rsi1<30).astype(float)+(rsi1<40).astype(float)
    r1 -= 2*(rsi1>70).astype(float)+(rsi1>60).astype(float)
    r1 += 2*(sk1<10).astype(float)+(sk1<20).astype(float)
    r1 -= 2*(sk1>90).astype(float)+(sk1>80).astype(float)
    r1 += 2*(vd<-1.5).astype(float)+(vd<-0.5).astype(float)
    r1 -= 2*(vd>1.5).astype(float)+(vd>0.5).astype(float)
    r1 += 2*(zs<-2.0).astype(float)+(zs<-1.5).astype(float)
    r1 -= 2*(zs>2.0).astype(float)+(zs>1.5).astype(float)
    r1  = r1.clip(-8, 8)

    t1 = t4.reindex(df1h.index, method="ffill")
    if cal:
        def lkp(t, r):
            k = f"{int(round(t))}_{int(round(r))}"; e = cal.get(k)
            return e["p_yes"] if e and e.get("n",0)>=5 else 0.504
        pup = pd.Series([lkp(t,r) for t,r in zip(t1, r1)], index=df1h.index)
    else:
        pup = pd.Series(0.504, index=df1h.index)
    return t1.rename("composite_trend"), r1.rename("composite_rev"), pup.rename("composite_p_up")


# ── build p_up_v2 series from OHLCV ───────────────────────────────────────────

def build_pup_series(asset: str) -> pd.Series:
    sym = f"{asset}USDT"
    f1h = sorted(DATA.glob(f"binanceus_{sym}_1h_*.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    f4h = sorted(DATA.glob(f"binanceus_{sym}_4h_*.parquet"), key=lambda p: p.stat().st_mtime)[-1]

    df1h = pd.read_parquet(f1h); df4h = pd.read_parquet(f4h)
    for d in (df1h, df4h):
        if d.index.tz is None: d.index = d.index.tz_localize("UTC")

    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    df1h = df1h[df1h.index >= cutoff].copy()
    df4h = df4h[df4h.index >= cutoff].copy()

    c1h = df1h["close"]; c4h = df4h["close"]
    vd_pct, vwap_str = daily_vwap_dist_series(df1h)

    ind1h = pd.DataFrame({
        "rsi_14":            rsi_series(c1h, 14),
        "macd_hist_1h":      macd_hist_series(c1h),
        "bb_pct":            bb_pct_series(c1h),
        "ema50_dist":        ema50_dist_series(c1h),
        "rvol_1h":           df1h["volume"] / df1h["volume"].rolling(24).mean().replace(0, np.nan),
        "vwap_distance_pct": vd_pct / 100.0,
        "vwap_stretch_score": vwap_str,
    }, index=df1h.index)
    ind4h = pd.DataFrame({
        "stoch_k_4h": stoch_k_series(df4h["high"], df4h["low"], c4h, 14),
        "rsi_4h":     rsi_series(c4h, 14),
        "chg_4h_atr": chg_4h_atr_series(df4h),
    }, index=df4h.index)

    cal_path = ROOT / f"composite_calibration_{asset.lower()}.json"
    cal = json.load(open(cal_path)) if cal_path.exists() else None
    trend_s, rev_s, pup_s = compute_composite_signals(df1h, df4h, cal)

    df = df1h[["close"]].copy()
    df = df.join(ind1h, how="left")
    ind4h_r = ind4h.reset_index().rename(columns={ind4h.index.name or "index": "ts"})
    df_r    = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    df_r    = pd.merge_asof(df_r, ind4h_r, on="ts", direction="backward")
    df      = df_r.set_index("ts")
    df["composite_trend"] = trend_s.reindex(df.index)
    df["composite_rev"]   = rev_s.reindex(df.index)
    df["composite_p_up"]  = pup_s.reindex(df.index)
    for col in ("stoch_k", "ema_stack_bias", "ema_stretch_score",
                "confirmation_bias", "stoch_bias", "vpin_score", "pm_drift_5m"):
        df[col] = np.nan
    for col in FEATURES:
        if col not in df.columns: df[col] = np.nan

    model_path = ROOT / "reform_results" / f"{asset.lower()}_p_up_v2.pkl"
    clf = pickle.load(open(model_path, "rb"))["clf"]
    X   = df[FEATURES].values.astype(float)
    pup = clf.predict_proba(X)[:, 1]
    return pd.Series(np.clip(pup, 0.02, 0.98), index=df.index, name="p_up_v2")


def build_rolling(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pup": series})
    for w in ROLL_WINDOWS:
        df[f"roll_{w}"] = series.rolling(w).mean()
    return df

def lookup_rolling(ts, roll_df, window):
    idx = roll_df.index.searchsorted(ts, side="right") - 1
    return float(roll_df[f"roll_{window}"].iloc[idx]) if idx >= 0 else float("nan")

def p_yes_lognormal(z): return float(np.clip(1.0 - norm.cdf(z), 0.01, 0.99))
def p_no_lognormal(z):  return float(np.clip(norm.cdf(z), 0.01, 0.99))

def kelly_f(p_model, p_market):
    if p_market <= 0 or p_market >= 1: return 0.0
    b = (1.0 - p_market) / p_market
    return float(np.clip((b*p_model - (1-p_model)) / b, 0.0, KELLY_CAP))

def pnl_yes(kf, p_mkt, win): s=kf*BANKROLL; return s*(1-p_mkt)/p_mkt if win else -s
def pnl_no(kf, p_mkt_no, win): s=kf*BANKROLL; return s*(1-p_mkt_no)/p_mkt_no if win else -s


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ETH", choices=["ETH", "SOL"])
    args = parser.parse_args()
    asset = args.asset

    trades_csv = RESULTS / f"paper_trades_{asset.lower()}.csv"

    print(f"Building {asset} p_up_v2 series from OHLCV...")
    pup_series = build_pup_series(asset)
    roll_df    = build_rolling(pup_series)
    print(f"  {len(pup_series):,} bars  {pup_series.index[0].date()} → {pup_series.index[-1].date()}")
    print(f"  p_up_v2 mean={pup_series.mean():.3f}  std={pup_series.std():.3f}")

    print(f"Loading {trades_csv.name}...")
    rows = list(csv.DictReader(open(trades_csv)))
    print(f"  {len(rows):,} rows")

    records = []; skipped = 0
    for r in rows:
        try:
            side      = r.get("side","").strip().lower()
            p_mkt     = float(r["p_market"])    if r.get("p_market","").strip()    else None
            spot      = float(r["spot"])         if r.get("spot","").strip()        else None
            strike    = float(r["strike"])       if r.get("strike","").strip()      else None
            tau       = float(r["tau_minutes"])  if r.get("tau_minutes","").strip() else None
            vol       = float(r["vol_60m_model"])if r.get("vol_60m_model","").strip() else None
            would_win = r.get("would_win","").strip()
            if None in (p_mkt, spot, strike, tau, vol): skipped += 1; continue
            if tau <= 0 or vol <= 0: skipped += 1; continue
            sigma_tau = vol * math.sqrt(tau)
            if sigma_tau <= 0: skipped += 1; continue
            if not would_win: skipped += 1; continue

            z_str = math.log(strike / spot) / sigma_tau
            ts    = pd.Timestamp(r["logged_at"], tz="UTC")
            win   = would_win.lower() == "true"

            if side in ("yes", "no"):
                records.append({"side": side, "z_str": z_str, "p_mkt": p_mkt,
                                 "win": win, "ts": ts})
        except Exception:
            skipped += 1

    yes_recs = [r for r in records if r["side"] == "yes"]
    no_recs  = [r for r in records if r["side"] == "no"]
    print(f"  YES: {len(yes_recs):,}   NO: {len(no_recs):,}   Skipped: {skipped}")

    print("  Pre-fetching rolling p_up_v2 for all records...")
    for rec in no_recs:
        rec["rolling"] = {w: lookup_rolling(rec["ts"], roll_df, w) for w in ROLL_WINDOWS}
    for rec in yes_recs:
        rec["rolling"] = {w: lookup_rolling(rec["ts"], roll_df, w) for w in ROLL_WINDOWS}

    # p_up_v2 distribution at trade time (NO side)
    spot_pups = [lookup_rolling(r["ts"], roll_df, 4) for r in no_recs]
    valid_pups = [p for p in spot_pups if not math.isnan(p)]
    if valid_pups:
        print(f"\n  p_up_v2 at NO trade times:  mean={np.mean(valid_pups):.3f}  "
              f"bull(>=0.52)={sum(p>=0.52 for p in valid_pups)/len(valid_pups):.1%}")

    # YES baseline (no gate)
    yes_taken = yes_wins = 0; yes_pnl = 0.0
    for rec in yes_recs:
        pm = p_yes_lognormal(rec["z_str"]); edge = pm - rec["p_mkt"]
        if edge < EDGE_MIN: continue
        kf = kelly_f(pm, rec["p_mkt"])
        if kf <= 0: continue
        yes_taken += 1
        if rec["win"]: yes_wins += 1
        yes_pnl += pnl_yes(kf, rec["p_mkt"], rec["win"])

    # NO baseline (no gate)
    no_base_taken = no_base_wins = 0; no_base_pnl = 0.0
    for rec in no_recs:
        p_mkt_no = 1.0 - rec["p_mkt"]; pm = p_no_lognormal(rec["z_str"])
        edge = pm - p_mkt_no
        if edge < EDGE_MIN: continue
        kf = kelly_f(pm, p_mkt_no)
        if kf <= 0: continue
        no_base_taken += 1
        if rec["win"]: no_base_wins += 1
        no_base_pnl += pnl_no(kf, p_mkt_no, rec["win"])

    base_total = yes_pnl + no_base_pnl
    SEP = "=" * 82

    print()
    print(SEP)
    print(f"ROLLING p_up_v2 REGIME GATE ({asset}): block NO bets when bull regime fires")
    print(f"Flat ${BANKROLL:.0f}  |  edge >= {EDGE_MIN:.2f}  |  Kelly cap {KELLY_CAP:.0%}  |  k=0 lognormal")
    print(SEP)
    print(f"\nYES side (no gate):  bets={yes_taken}  "
          f"WR={yes_wins/yes_taken*100:.1f}%  PnL={yes_pnl:+.2f}" if yes_taken else
          f"\nYES side: no qualifying bets")
    print(f"NO  base (no gate):  bets={no_base_taken}  "
          f"WR={no_base_wins/no_base_taken*100:.1f}%  "
          f"PnL={no_base_pnl:+.2f}  TOTAL={base_total:+.2f}" if no_base_taken else
          f"NO  base: no qualifying bets")

    print()
    print(f"{'win':>5}  {'bull_t':>7}  {'NO bets':>8}  {'NO WR':>7}  "
          f"{'blk_W':>6}  {'blk_L':>6}  {'NO PnL':>9}  {'TOTAL':>9}  {'delta':>8}")
    print("-" * 82)

    results = []
    for w in ROLL_WINDOWS:
        for bt in BULL_THRESHOLDS:
            taken = wins = blk_w = blk_l = 0; pnl = 0.0
            for rec in no_recs:
                p_mkt_no = 1.0 - rec["p_mkt"]; pm = p_no_lognormal(rec["z_str"])
                if pm - p_mkt_no < EDGE_MIN: continue
                kf = kelly_f(pm, p_mkt_no)
                if kf <= 0: continue
                roll_pup = rec["rolling"][w]
                if not math.isnan(roll_pup) and roll_pup >= bt:
                    if rec["win"]: blk_w += 1
                    else:          blk_l += 1
                    continue
                taken += 1
                if rec["win"]: wins += 1
                pnl += pnl_no(kf, p_mkt_no, rec["win"])

            wr    = wins / taken * 100 if taken else 0.0
            total = yes_pnl + pnl
            delta = total - base_total
            flag  = " ★" if delta > 0 else ""
            results.append((total, w, bt, taken, wr, blk_w, blk_l, pnl))
            print(f"{w:>5}  {bt:>7.2f}  {taken:>8d}  {wr:>6.1f}%  "
                  f"{blk_w:>6d}  {blk_l:>6d}  {pnl:>+9.2f}  {total:>+9.2f}  {delta:>+8.2f}{flag}")
        print()

    best = max(results, key=lambda x: x[0])
    print(SEP)
    print(f"  Best: window={best[1]}h  bull_thresh={best[2]}  "
          f"NO bets={best[3]}  WR={best[4]:.1f}%")
    print(f"  Blocked: {best[5]} wins  {best[6]} losses")
    print(f"  NO PnL={best[7]:+.2f}  TOTAL={best[0]:+.2f}  delta={best[0]-base_total:+.2f}")

    # ── YES-side bear regime gate ──────────────────────────────────────────────
    # Block YES bets when rolling p_up_v2 <= bear_thresh (bearish regime).
    # NO side unmodified.
    print()
    print(SEP)
    print(f"ROLLING p_up_v2 BEAR GATE ({asset}): block YES bets when bear regime fires")
    print(f"Flat ${BANKROLL:.0f}  |  edge >= {EDGE_MIN:.2f}  |  Kelly cap {KELLY_CAP:.0%}  |  k=0 lognormal")
    print(SEP)

    yes_pup4 = [lookup_rolling(r["ts"], roll_df, 4) for r in yes_recs]
    valid_yes_pups = [p for p in yes_pup4 if not math.isnan(p)]
    if valid_yes_pups:
        print(f"\n  p_up_v2 at YES trade times:  mean={np.mean(valid_yes_pups):.3f}  "
              f"bear(<=0.48)={sum(p<=0.48 for p in valid_yes_pups)/len(valid_yes_pups):.1%}")

    print(f"\nNO side (no gate):   bets={no_base_taken}  "
          f"WR={no_base_wins/no_base_taken*100:.1f}%  PnL={no_base_pnl:+.2f}" if no_base_taken else "")
    print(f"YES base (no gate):  bets={yes_taken}  "
          f"WR={yes_wins/yes_taken*100:.1f}%  PnL={yes_pnl:+.2f}  "
          f"TOTAL={base_total:+.2f}" if yes_taken else "")

    print()
    print(f"{'win':>5}  {'bear_t':>7}  {'YES bets':>9}  {'YES WR':>7}  "
          f"{'blk_W':>6}  {'blk_L':>6}  {'YES PnL':>9}  {'TOTAL':>9}  {'delta':>8}")
    print("-" * 84)

    yes_results = []
    for w in ROLL_WINDOWS:
        for bt in BEAR_THRESHOLDS:
            taken = wins = blk_w = blk_l = 0; pnl = 0.0
            for rec in yes_recs:
                pm   = p_yes_lognormal(rec["z_str"])
                edge = pm - rec["p_mkt"]
                if edge < EDGE_MIN: continue
                kf = kelly_f(pm, rec["p_mkt"])
                if kf <= 0: continue
                roll_pup = rec["rolling"][w]
                if not math.isnan(roll_pup) and roll_pup <= bt:
                    if rec["win"]: blk_w += 1
                    else:          blk_l += 1
                    continue
                taken += 1
                if rec["win"]: wins += 1
                pnl += pnl_yes(kf, rec["p_mkt"], rec["win"])

            wr    = wins / taken * 100 if taken else 0.0
            total = pnl + no_base_pnl
            delta = total - base_total
            flag  = " ★" if delta > 0 else ""
            yes_results.append((total, w, bt, taken, wr, blk_w, blk_l, pnl))
            print(f"{w:>5}  {bt:>7.2f}  {taken:>9d}  {wr:>6.1f}%  "
                  f"{blk_w:>6d}  {blk_l:>6d}  {pnl:>+9.2f}  {total:>+9.2f}  {delta:>+8.2f}{flag}")
        print()

    best_yes = max(yes_results, key=lambda x: x[0])
    print(SEP)
    print(f"  Best: window={best_yes[1]}h  bear_thresh={best_yes[2]}  "
          f"YES bets={best_yes[3]}  WR={best_yes[4]:.1f}%")
    print(f"  Blocked: {best_yes[5]} wins  {best_yes[6]} losses")
    print(f"  YES PnL={best_yes[7]:+.2f}  TOTAL={best_yes[0]:+.2f}  "
          f"delta={best_yes[0]-base_total:+.2f}")


if __name__ == "__main__":
    import os; os.chdir(ROOT)
    main()
