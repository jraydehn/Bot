#!/usr/bin/env python3
"""
simulate_otm_yes_gate.py

Tests whether z_drift model would find edge in pm<0.15 YES contracts
that were blocked by any gate.

Methodology:
- Include ALL scanned/blocked contracts regardless of resolved_yes status
- Derive actual outcome from Binance 1m price at close_ts vs strike
- Use flat $10 bet per trade (non-compounding, comparable across scenarios)
- Walk-forward z_drift history built from all prior resolved contracts
"""

import math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR    = Path(__file__).parent / "results"
CSV_TRADES     = RESULTS_DIR / "paper_trades_btc15m.csv"
CSV_BLOCKED    = RESULTS_DIR / "blocked_trades.csv"
MINS_PER_YEAR  = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5
PM_OTM_MAX     = 0.15
FLAT_BET       = 10.0   # flat $10 per trade for fair comparison


def fetch_1m(start_ms, end_ms):
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m",
            "startTime": cur, "endTime": min(cur + 1000 * 60_000, end_ms), "limit": 1000},
            timeout=15)
        r.raise_for_status()
        d = r.json()
        if not d: break
        rows.extend(d); cur = int(d[-1][0]) + 60_000; time.sleep(0.05)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ot","open","h","l","c","v","ct","q","t","tb","tq","i"])
    df["ot"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    return df.set_index("ot")[["open"]].astype(float)


def price_at(ts, m1_idx, m1_open):
    try:
        ts = pd.Timestamp(ts).tz_convert("UTC")
        i = m1_idx.get_loc(ts) if ts in m1_idx else m1_idx.searchsorted(ts)
        if i >= len(m1_idx): return float("nan")
        return float(m1_open.iloc[i])
    except: return float("nan")


def compute_actual_z(spot, rv_ann, tau, expiry_price):
    try:
        if spot <= 0 or rv_ann <= 0 or tau <= 0 or math.isnan(expiry_price): return float("nan")
        sig = rv_ann / math.sqrt(MINS_PER_YEAR) * math.sqrt(tau)
        if sig <= 0: return float("nan")
        return math.log(expiry_price / spot) / sig
    except: return float("nan")


def zdrift(az_list):
    if len(az_list) < W_SHORT: return float("nan")
    tail = az_list[-max(W_LONG, len(az_list)):]
    zs = np.mean(tail[-W_SHORT:])
    zl = np.mean(tail[-W_LONG:] if len(tail) >= W_LONG else tail)
    return float(np.clip(ALPHA * zs + (1 - ALPHA) * zl, -CAP, CAP))


def p_zdrift(spot, strike, rv_ann, tau, zd):
    sig = rv_ann / math.sqrt(MINS_PER_YEAR) * math.sqrt(tau)
    if sig <= 0: return float("nan")
    return float(np.clip(norm.cdf(zd - math.log(strike / spot) / sig), 0.03, 0.97))


def main():
    # --- Load paper trades (for rv_ann lookup + z_drift history) ---
    pt = pd.read_csv(CSV_TRADES, low_memory=False)
    for c in ["spot", "realized_vol_annual", "tau_minutes", "p_market", "floor_strike"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt["close_time"] = pd.to_datetime(pt["close_time"], utc=True, errors="coerce")
    pt = pt.sort_values("close_time").reset_index(drop=True)
    pt_valid = pt[pt["close_time"].notna() & pt["realized_vol_annual"].notna()].copy()

    # Build rv_ann lookup by close_time (market-wide, same for all contracts at same expiry)
    rv_lookup = (pt_valid.sort_values("close_time")
                 .drop_duplicates("close_time")[["close_time","realized_vol_annual"]])

    # --- Load ALL blocked trades (no resolved_yes filter) ---
    bt = pd.read_csv(CSV_BLOCKED, low_memory=False)
    for c in ["pm", "spot", "strike", "tau_minutes"]:
        bt[c] = pd.to_numeric(bt[c], errors="coerce")
    bt["close_ts"] = pd.to_datetime(bt["close_ts"], utc=True, errors="coerce")

    # Filter: YES OTM only, contract data present
    blocked_otm = bt[
        (bt["side"] == "yes") &
        (bt["pm"] < PM_OTM_MAX) &
        bt["close_ts"].notna() &
        bt["spot"].notna() &
        bt["strike"].notna() &
        bt["tau_minutes"].notna()
    ].copy()

    # Deduplicate: one row per (close_ts, strike) — take first scan
    blocked_otm = blocked_otm.sort_values("close_ts").drop_duplicates(
        subset=["close_ts", "strike"], keep="first"
    ).reset_index(drop=True)
    print(f"Blocked OTM YES (pm<{PM_OTM_MAX}), unique contracts: {len(blocked_otm)}")
    print(f"Date range: {blocked_otm['close_ts'].min().date()} → {blocked_otm['close_ts'].max().date()}")

    # Join rv_ann via nearest close_ts
    blocked_otm = blocked_otm.sort_values("close_ts")
    blocked_otm = pd.merge_asof(
        blocked_otm, rv_lookup.rename(columns={"close_time": "close_ts"}),
        on="close_ts", direction="nearest"
    )
    rv_fill = rv_lookup["realized_vol_annual"].median()
    n_fill = blocked_otm["realized_vol_annual"].isna().sum()
    blocked_otm["realized_vol_annual"] = blocked_otm["realized_vol_annual"].fillna(rv_fill)
    print(f"rv_ann: {n_fill} rows filled with median {rv_fill:.4f}")

    # --- Fetch 1m Binance data ---
    all_ts = pd.concat([pt_valid["close_time"], blocked_otm["close_ts"]]).dropna()
    s_ms = int(all_ts.min().timestamp() * 1000) - 120_000
    e_ms = int(all_ts.max().timestamp() * 1000) + 120_000
    print(f"\nFetching 1m candles ({pd.Timestamp(s_ms//1000, unit='s').date()} → "
          f"{pd.Timestamp(e_ms//1000, unit='s').date()})...")
    m1 = fetch_1m(s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    # --- Compute expiry prices and actual outcomes ---
    # Paper trades: for z_drift history
    pt_valid["expiry_price"] = [price_at(r["close_time"], m1.index, m1["open"])
                                 for _, r in pt_valid.iterrows()]
    pt_valid["actual_z"] = [
        compute_actual_z(float(r["spot"]), float(r["realized_vol_annual"]),
                         float(r["tau_minutes"]), float(r["expiry_price"]))
        for _, r in pt_valid.iterrows()
    ]
    pt_valid["price_outcome"] = (pt_valid["expiry_price"] > pt_valid["floor_strike"]).astype(float)
    print(f"Paper trades: actual_z valid={pt_valid['actual_z'].notna().sum()}/{len(pt_valid)}")

    # Blocked OTM: derive outcome from price
    blocked_otm["expiry_price"] = [price_at(r["close_ts"], m1.index, m1["open"])
                                    for _, r in blocked_otm.iterrows()]
    blocked_otm["price_outcome"] = (blocked_otm["expiry_price"] > blocked_otm["strike"]).astype(float)
    n_priced = blocked_otm["expiry_price"].notna().sum()
    n_resolved = blocked_otm["price_outcome"].notna().sum()
    print(f"Blocked OTM: expiry price found={n_priced}/{len(blocked_otm)}, outcomes={n_resolved}")

    # --- Walk-forward z_drift using paper_trades history ---
    pt_history = pt_valid[pt_valid["actual_z"].notna()].sort_values("close_time")[
        ["close_time", "actual_z"]].copy()

    results = []
    for _, row in blocked_otm.iterrows():
        if math.isnan(float(row["expiry_price"])) if not pd.isna(row["expiry_price"]) else True:
            continue  # can't determine outcome without expiry price

        ts      = row["close_ts"]
        pm      = float(row["pm"])
        spot    = float(row["spot"])
        strike  = float(row["strike"])
        rv_ann  = float(row["realized_vol_annual"])
        tau     = float(row["tau_minutes"])
        outcome = int(row["price_outcome"])

        # z_drift from all paper_trades resolved BEFORE this scan
        prior = pt_history[pt_history["close_time"] < ts]["actual_z"].tolist()
        zd = zdrift(prior)
        zd_active = not math.isnan(zd)

        if zd_active:
            p_zd = p_zdrift(spot, strike, rv_ann, tau, zd)
            edge_zd = p_zd - pm
            if edge_zd >= EDGE_THRESHOLD:
                pnl = FLAT_BET * (1 - pm) / pm if outcome == 1 else -FLAT_BET
                decision = "trade"
            else:
                pnl = 0.0
                decision = "block_no_edge"
        else:
            p_zd    = float("nan")
            edge_zd = float("nan")
            pnl     = 0.0
            decision = "block_no_history"

        results.append({
            "close_ts": ts,
            "pm":       pm,
            "strike":   strike,
            "spot":     spot,
            "zd":       zd,
            "p_zd":     p_zd if zd_active else float("nan"),
            "edge_zd":  edge_zd if zd_active else float("nan"),
            "n_prior":  len(prior),
            "outcome":  outcome,
            "pnl":      pnl,
            "decision": decision,
        })

    rdf = pd.DataFrame(results)
    active   = rdf[rdf["zd"].notna()]
    would_trade = rdf[rdf["decision"] == "trade"]

    print(f"\n{'='*65}")
    print(f"  OTM YES GATE  (z_drift model, pm<{PM_OTM_MAX}, flat ${FLAT_BET:.0f}/trade)")
    print(f"{'='*65}")
    print(f"\n  Total unique blocked contracts:   {len(rdf)}")
    print(f"  Expiry price resolved:            {rdf['outcome'].notna().sum()}")
    print(f"  z_drift active (n_prior≥{W_SHORT}):     {len(active)}")
    print(f"  z_drift would trade (edge≥4%):    {len(would_trade)}")

    if len(would_trade):
        wr = would_trade["outcome"].mean()
        pnl = would_trade["pnl"].sum()
        be = (would_trade["pm"] / (1 + would_trade["pm"])).mean()
        print(f"\n  Would-trade: WR={wr:.3f}  BE≈{be:.3f}  PnL=${pnl:+.2f}  "
              f"(${pnl/len(would_trade)*100:.2f}/100 trades)")

    print(f"\n  Breakdown by pm bucket:")
    print(f"  {'pm range':<14} {'n':>5} {'WR':>6} {'BE':>6} {'WR-BE':>7} "
          f"{'zd_act':>7} {'zd_tr':>7} {'zd_pnl':>10} {'if_all':>10}")
    for lo, hi in [(0,.03),(.03,.05),(.05,.08),(.08,.10),(.10,.12),(.12,.15)]:
        sub = rdf[(rdf["pm"]>=lo)&(rdf["pm"]<hi)]
        if len(sub)==0: continue
        act = sub[sub["zd"].notna()]
        wt  = sub[sub["decision"]=="trade"]
        wr  = sub["outcome"].mean()
        be  = (sub["pm"]/(1+sub["pm"])).mean()
        all_pnl = sum(FLAT_BET*(1-r["pm"])/r["pm"] if r["outcome"]==1 else -FLAT_BET
                      for _,r in sub.iterrows())
        print(f"  [{lo:.2f},{hi:.2f})       {len(sub):>5}  {wr:>5.3f}  {be:>5.3f}  "
              f"{wr-be:>+6.3f}  {len(act):>6}  {len(wt):>6}  "
              f"{wt['pnl'].sum():>+9.2f}  {all_pnl:>+9.2f}")

    print(f"\n  z_drift decision breakdown:")
    for d, g in rdf.groupby("decision"):
        wr = g["outcome"].mean() if len(g) else float("nan")
        print(f"    {d:<22}: n={len(g):4d}  WR={wr:.3f}  pnl=${g['pnl'].sum():+.2f}")

    print(f"\n  z_drift value distribution (active rows, n={len(active)}):")
    if len(active):
        zv = active["zd"].values
        print(f"    mean={np.mean(zv):+.4f}  >0: {(zv>0).sum()}  ≤0: {(zv<=0).sum()}")
        for lo,hi in [(-0.5,-0.3),(-0.3,-0.1),(-0.1,0.1),(0.1,0.3),(0.3,0.51)]:
            n = ((zv>=lo)&(zv<hi)).sum()
            print(f"      [{lo:+.1f},{hi:+.1f}): {n}")

    print(f"\n  For context — all blocked OTM YES if simply allowed through (no model):")
    all_pnl = sum(FLAT_BET*(1-r["pm"])/r["pm"] if r["outcome"]==1 else -FLAT_BET
                  for _,r in rdf.iterrows())
    print(f"    n={len(rdf)}  WR={rdf['outcome'].mean():.3f}  "
          f"BE={(rdf['pm']/(1+rdf['pm'])).mean():.3f}  PnL=${all_pnl:+.2f}")


if __name__ == "__main__":
    main()
