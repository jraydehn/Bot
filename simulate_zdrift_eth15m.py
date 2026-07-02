#!/usr/bin/env python3
"""
simulate_zdrift_eth15m.py

Tests whether z_drift can replace YES or NO side for ETH 15m model.
1. AR(1) test on ETH 15m actual_z
2. Walk-forward: z_drift vs LGBM for YES side
3. Walk-forward: z_drift vs LGBM for NO side
4. Branched model: z_drift YES + LGBM NO (and reverse)
"""

import math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR    = Path(__file__).parent / "results"
CSV            = RESULTS_DIR / "paper_trades_eth15m.csv"
MINS_PER_YEAR  = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5


def fetch_1m(symbol, start_ms, end_ms):
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol":symbol,"interval":"1m",
            "startTime":cur,"endTime":min(cur+1000*60_000,end_ms),"limit":1000}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if not d: break
        rows.extend(d); cur = int(d[-1][0])+60_000; time.sleep(0.05)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ot","open","h","l","c","v","ct","q","t","tb","tq","i"])
    df["ot"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    return df.set_index("ot")[["open"]].astype(float)


def get_actual_z(row, idx, opens):
    try:
        s,rv,t = float(row["spot"]),float(row["realized_vol_annual"]),float(row["tau_minutes"])
        if s<=0 or rv<=0 or t<=0: return float("nan")
        sig = rv/math.sqrt(MINS_PER_YEAR)*math.sqrt(t)
        if sig<=0: return float("nan")
        ts = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        i  = idx.get_loc(ts) if ts in idx else idx.searchsorted(ts)
        if i >= len(idx): return float("nan")
        return math.log(float(opens.iloc[i])/s)/sig
    except: return float("nan")


def zdrift(az):
    if len(az) < W_SHORT: return float("nan")
    tail = az[-max(W_LONG,len(az)):]
    zs = np.mean(tail[-W_SHORT:]); zl = np.mean(tail[-W_LONG:] if len(tail)>=W_LONG else tail)
    return float(np.clip(ALPHA*zs+(1-ALPHA)*zl, -CAP, CAP))


def p_zdrift(spot, strike, rv_ann, tau, zd):
    sig = rv_ann/math.sqrt(MINS_PER_YEAR)*math.sqrt(tau)
    if sig<=0: return float("nan")
    return float(np.clip(norm.cdf(zd - math.log(strike/spot)/sig), 0.03, 0.97))


def pnl_for(bet_amt, pm, side, outcome, p_model, threshold):
    edge = (p_model-pm) if side=="yes" else (pm-p_model)
    if edge < threshold: return 0.0
    if side=="yes": return bet_amt*(1-pm)/pm if outcome==1 else -bet_amt
    else:           return bet_amt*pm/(1-pm)  if outcome==0 else -bet_amt


def main():
    df = pd.read_csv(CSV, low_memory=False)
    for c in ["spot","realized_vol_annual","tau_minutes","p_market","p_model_15m",
              "bet_amount","would_pnl","resolved_yes","floor_strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.sort_values("close_time").reset_index(drop=True)
    resolved = df[df["resolved_yes"].notna() & df["close_time"].notna()].copy()
    trades   = resolved[resolved["decision"]=="trade"]
    print(f"ETH 15m  |  Resolved: {len(resolved)}  |  Trades: {len(trades)}")
    print(f"Date range: {resolved['close_time'].min().date()} → {resolved['close_time'].max().date()}")

    s_ms = int(resolved["close_time"].min().timestamp()*1000)-120_000
    e_ms = int(resolved["close_time"].max().timestamp()*1000)+120_000
    print("Fetching ETH 1m candles...")
    m1 = fetch_1m("ETHUSDT", s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    resolved["actual_z"] = [get_actual_z(r, m1.index, m1["open"]) for _,r in resolved.iterrows()]
    valid = resolved["actual_z"].notna().sum()
    az_vals = resolved["actual_z"].dropna().values
    ar1 = float(pd.Series(az_vals).autocorr(1))
    ar2 = float(pd.Series(az_vals).autocorr(2))
    ar3 = float(pd.Series(az_vals).autocorr(3))
    n   = len(az_vals)
    se  = 1/math.sqrt(n)
    print(f"\nactual_z AR(1)={ar1:+.4f}  AR(2)={ar2:+.4f}  AR(3)={ar3:+.4f}  "
          f"(n={n}, SE={se:.3f}, sig_threshold=±{1.96*se:.3f})")
    print(f"AR(1) {'SIGNIFICANT' if abs(ar1)>1.96*se else 'NOT significant'} at 95%")

    # Walk-forward
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

            # Compute z_drift p_yes
            p_zd = p_zdrift(spot, strike, rv_ann, tau, zd) if zd_active else float("nan")

            # Scenario A: z_drift for YES, LGBM for NO
            if side == "yes":
                pnl_A = pnl_for(bet_amt, pm, "yes", outcome,
                                 p_zd if zd_active else p_lgbm, EDGE_THRESHOLD)
            else:
                pnl_A = orig_pnl  # NO unchanged

            # Scenario B: LGBM for YES, z_drift for NO
            if side == "no":
                pnl_B = pnl_for(bet_amt, pm, "no", outcome,
                                 p_zd if zd_active else p_lgbm, EDGE_THRESHOLD)
            else:
                pnl_B = orig_pnl  # YES unchanged

            # Scenario C: z_drift for BOTH sides
            pnl_C = pnl_for(bet_amt, pm, side, outcome,
                             p_zd if zd_active else p_lgbm, EDGE_THRESHOLD)

            rows_out.append({
                "date": row["close_time"].date(),
                "side": side, "pm": pm,
                "p_lgbm": p_lgbm, "p_zd": p_zd,
                "zd": zd if zd_active else float("nan"),
                "orig_pnl": orig_pnl,
                "pnl_A": pnl_A,   # zdrift YES / LGBM NO
                "pnl_B": pnl_B,   # LGBM YES / zdrift NO
                "pnl_C": pnl_C,   # zdrift BOTH
                "outcome": outcome,
            })

        if not pd.isna(row["actual_z"]):
            prior_az.append(float(row["actual_z"]))

    rdf = pd.DataFrame(rows_out)
    active = rdf[rdf["zd"].notna()]

    base  = rdf["orig_pnl"].sum()
    pnl_A = rdf["pnl_A"].sum()
    pnl_B = rdf["pnl_B"].sum()
    pnl_C = rdf["pnl_C"].sum()

    print(f"\n{'='*60}")
    print(f"  ETH 15m  WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    print(f"  LGBM baseline:            ${base:+.2f}")
    print(f"  A) z_drift YES/LGBM NO:   ${pnl_A:+.2f}  delta=${pnl_A-base:+.2f}")
    print(f"  B) LGBM YES/z_drift NO:   ${pnl_B:+.2f}  delta=${pnl_B-base:+.2f}")
    print(f"  C) z_drift BOTH:          ${pnl_C:+.2f}  delta=${pnl_C-base:+.2f}")

    print(f"\n  By side (LGBM baseline):")
    for s in ["yes","no"]:
        sub = rdf[rdf["side"]==s]
        print(f"    {s.upper()}: n={len(sub)}  P&L=${sub['orig_pnl'].sum():+.2f}  "
              f"WR={sub['outcome'].mean():.3f}")

    print(f"\n  Scenario A breakdown by side:")
    yes_A = rdf[rdf["side"]=="yes"]
    no_A  = rdf[rdf["side"]=="no"]
    print(f"    YES (z_drift): orig=${yes_A['orig_pnl'].sum():+.2f} → ${yes_A['pnl_A'].sum():+.2f}  "
          f"delta=${yes_A['pnl_A'].sum()-yes_A['orig_pnl'].sum():+.2f}")
    print(f"    NO  (LGBM):   orig=${no_A['orig_pnl'].sum():+.2f} → ${no_A['pnl_B'].sum():+.2f}  delta=$0")

    print(f"\n  Scenario B breakdown by side:")
    print(f"    YES (LGBM):    orig=${yes_A['orig_pnl'].sum():+.2f} → ${yes_A['pnl_B'].sum():+.2f}  delta=$0")
    print(f"    NO  (z_drift): orig=${no_A['orig_pnl'].sum():+.2f} → ${no_A['pnl_B'].sum():+.2f}  "
          f"delta=${no_A['pnl_B'].sum()-no_A['orig_pnl'].sum():+.2f}")

    print(f"\n  z_drift stats (ETH 15m active rows): n={len(active)}")
    if len(active):
        zd_v = active["zd"].values
        print(f"    mean={np.mean(zd_v):+.4f}  std={np.std(zd_v):.4f}  "
              f">0: {(zd_v>0).sum()}  <=0: {(zd_v<=0).sum()}")

    print(f"\n  Daily breakdown (Scenario A):")
    for d, g in rdf.groupby("date"):
        ob = g["orig_pnl"].sum(); a = g["pnl_A"].sum()
        zd_m = g[g["zd"].notna()]["zd"].mean() if g["zd"].notna().any() else float("nan")
        print(f"    {d}  n={len(g):3d}  LGBM=${ob:+7.2f}  A=${a:+7.2f}  "
              f"delta=${a-ob:+7.2f}  zd_mean={zd_m:+.3f}")


if __name__ == "__main__":
    main()
