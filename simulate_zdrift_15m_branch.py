#!/usr/bin/env python3
"""
simulate_zdrift_15m_branch.py

Walk-forward simulation: branched model for 15m BTC.
  YES side: pure log-normal z_drift model  (norm.cdf(z_drift - z_strike))
  NO  side: LightGBM as-is (from p_model_15m in CSV)

Fallback when z_drift unavailable (< 10 prior trades): use LGBM for both sides.
"""

import math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR   = Path(__file__).parent / "results"
CSV_15M       = RESULTS_DIR / "paper_trades_btc15m.csv"
MINS_PER_YEAR = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5


def fetch_binance_1m(start_ms, end_ms):
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol":"BTCUSDT","interval":"1m",
            "startTime":cur,"endTime":min(cur+1000*60_000,end_ms),"limit":1000}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if not d: break
        rows.extend(d); cur = int(d[-1][0]) + 60_000; time.sleep(0.05)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ot","open","h","l","c","v","ct","q","t","tb","tq","i"])
    df["ot"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    return df.set_index("ot")[["open"]].astype(float)


def actual_z(row, idx, opens):
    try:
        s,rv,t = float(row["spot"]), float(row["realized_vol_annual"]), float(row["tau_minutes"])
        if s<=0 or rv<=0 or t<=0: return float("nan")
        sig = rv/math.sqrt(MINS_PER_YEAR)*math.sqrt(t)
        if sig<=0: return float("nan")
        ts = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        i  = idx.get_loc(ts) if ts in idx else idx.searchsorted(ts)
        if i >= len(idx): return float("nan")
        exp = float(opens.iloc[i])
        return math.log(exp/s)/sig
    except: return float("nan")


def zdrift(az):
    if len(az) < W_SHORT: return float("nan")
    tail = az[-max(W_LONG,len(az)):]
    zs = np.mean(tail[-W_SHORT:]); zl = np.mean(tail[-W_LONG:] if len(tail)>=W_LONG else tail)
    return float(np.clip(ALPHA*zs+(1-ALPHA)*zl, -CAP, CAP))


def p_yes_zdrift(spot, strike, rv_ann, tau, zd):
    sig = rv_ann/math.sqrt(MINS_PER_YEAR)*math.sqrt(tau)
    if sig<=0: return float("nan")
    return float(np.clip(norm.cdf(zd - math.log(strike/spot)/sig), 0.03, 0.97))


def trade_pnl(bet_amt, pm, side, outcome, p_model, threshold):
    edge = (p_model-pm) if side=="yes" else (pm-p_model)
    if edge < threshold: return 0.0
    if side=="yes":
        return bet_amt*(1-pm)/pm if outcome==1 else -bet_amt
    else:
        return bet_amt*pm/(1-pm) if outcome==0 else -bet_amt


def main():
    df = pd.read_csv(CSV_15M, low_memory=False)
    for c in ["spot","realized_vol_annual","tau_minutes","p_market","p_model_15m",
              "bet_amount","would_pnl","resolved_yes","floor_strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.sort_values("close_time").reset_index(drop=True)
    resolved = df[df["resolved_yes"].notna() & df["close_time"].notna()].copy()
    print(f"Resolved: {len(resolved)}  |  Trades: {(resolved['decision']=='trade').sum()}")

    s_ms = int(resolved["close_time"].min().timestamp()*1000)-120_000
    e_ms = int(resolved["close_time"].max().timestamp()*1000)+120_000
    print("Fetching 1m candles...")
    m1 = fetch_binance_1m(s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    resolved["actual_z"] = [actual_z(r, m1.index, m1["open"]) for _,r in resolved.iterrows()]
    print(f"actual_z valid: {resolved['actual_z'].notna().sum()}/{len(resolved)}")

    prior_az = []
    rows_out = []

    for _, row in resolved.iterrows():
        is_trade = (row["decision"]=="trade" and not pd.isna(row["would_pnl"])
                    and not pd.isna(row["p_model_15m"]))
        if is_trade:
            zd      = zdrift(prior_az)
            spot    = float(row["spot"]); strike = float(row["floor_strike"])
            rv_ann  = float(row["realized_vol_annual"]); tau = float(row["tau_minutes"])
            pm      = float(row["p_market"]); side = str(row["side"]).lower()
            bet_amt = float(row["bet_amount"]); outcome = int(row["resolved_yes"])
            p_lgbm  = float(row["p_model_15m"])
            orig_pnl = float(row["would_pnl"])

            zd_active = not math.isnan(zd)

            if side == "yes":
                if zd_active:
                    p_yes = p_yes_zdrift(spot, strike, rv_ann, tau, zd)
                    branch_pnl = trade_pnl(bet_amt, pm, "yes", outcome, p_yes, EDGE_THRESHOLD)
                    model_used = "zdrift"
                else:
                    p_yes = p_lgbm; branch_pnl = orig_pnl; model_used = "lgbm_fallback"
            else:  # NO: always LGBM
                p_yes = p_lgbm; branch_pnl = orig_pnl; model_used = "lgbm"

            rows_out.append({
                "date":       row["close_time"].date(),
                "side":       side, "pm": pm,
                "p_lgbm":     p_lgbm,
                "p_branch":   p_yes,
                "zd":         zd if zd_active else float("nan"),
                "model_used": model_used,
                "orig_pnl":   orig_pnl,
                "branch_pnl": branch_pnl,
                "outcome":    outcome,
                "blocked":    branch_pnl == 0.0 and orig_pnl != 0.0,
            })

        if not pd.isna(row["actual_z"]):
            prior_az.append(float(row["actual_z"]))

    rdf = pd.DataFrame(rows_out)
    print(f"\n{'='*60}")
    print(f"  BRANCHED MODEL: z_drift YES / LGBM NO")
    print(f"{'='*60}")

    orig_total   = rdf["orig_pnl"].sum()
    branch_total = rdf["branch_pnl"].sum()
    print(f"\n  LGBM baseline P&L:    ${orig_total:+.2f}")
    print(f"  Branched model P&L:   ${branch_total:+.2f}")
    print(f"  Delta:                ${branch_total - orig_total:+.2f}")

    for s in ["yes", "no"]:
        sub = rdf[rdf["side"]==s]
        print(f"\n  {s.upper()} side (n={len(sub)}):")
        print(f"    LGBM P&L:    ${sub['orig_pnl'].sum():+.2f}")
        print(f"    Branch P&L:  ${sub['branch_pnl'].sum():+.2f}")
        print(f"    Delta:       ${sub['branch_pnl'].sum()-sub['orig_pnl'].sum():+.2f}")
        if s == "yes":
            blocked = sub["blocked"].sum()
            yes_active = sub[sub["model_used"]=="zdrift"]
            print(f"    Blocked by z_drift: {blocked}")
            if len(yes_active):
                zd_pos = yes_active[yes_active["zd"]>0]
                zd_neg = yes_active[yes_active["zd"]<=0]
                print(f"    z_drift>0 YES: n={len(zd_pos)}  P&L=${zd_pos['branch_pnl'].sum():+.2f}  WR={zd_pos['outcome'].mean():.3f}")
                print(f"    z_drift<=0 YES: n={len(zd_neg)}  P&L=${zd_neg['branch_pnl'].sum():+.2f} (blocked → $0)")

    print(f"\n  Daily breakdown:")
    for d, g in rdf.groupby("date"):
        ob = g["orig_pnl"].sum(); br = g["branch_pnl"].sum()
        zd_m = g[g["zd"].notna()]["zd"].mean()
        print(f"    {d}  n={len(g):3d}  LGBM=${ob:+7.2f}  branch=${br:+7.2f}  "
              f"delta=${br-ob:+7.2f}  zd_mean={zd_m:+.3f}")

    # WR breakdown for YES by z_drift sign
    yes_zd = rdf[(rdf["side"]=="yes") & rdf["zd"].notna()]
    print(f"\n  YES WR by z_drift sign (z_drift model active):")
    for label, sub in [("z_drift > 0", yes_zd[yes_zd["zd"]>0]),
                        ("z_drift <= 0", yes_zd[yes_zd["zd"]<=0])]:
        if not sub.empty:
            wr = sub["outcome"].mean()
            pm_be = (sub["pm"]/(1+sub["pm"])).mean()
            print(f"    {label}: n={len(sub):3d}  WR={wr:.3f}  "
                  f"mean_pm={sub['pm'].mean():.3f}  "
                  f"branch_pnl=${sub['branch_pnl'].sum():+.2f}")

if __name__ == "__main__":
    main()
