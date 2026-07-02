#!/usr/bin/env python3
"""
simulate_neutral_ema_g1.py

Tests whether neutral_ema_g1 (ema=0, comp_p_up>=0.60, stoch_k<40) is
beneficial or harmful with the z_drift model.

Methodology (new standard):
- Load ALL blocked_trades BTC YES entries regardless of resolved_yes column
- Derive outcomes from Binance 1m price at close_ts vs strike
- Split into condition groups to understand gate's true impact
- Use flat $10 bet, z_drift walk-forward from paper_trades_btc15m history
"""

import math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR   = Path(__file__).parent / "results"
CSV_BLOCKED   = RESULTS_DIR / "blocked_trades.csv"
CSV_TRADES    = RESULTS_DIR / "paper_trades_btc15m.csv"
MINS_PER_YEAR = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5
FLAT_BET      = 10.0

# neutral_ema_g1 thresholds
G1_P_UP_MIN  = 0.60
G1_STOCH_MAX = 40.0


def fetch_1m(start_ms, end_ms):
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m",
            "startTime": cur, "endTime": min(cur + 1000*60_000, end_ms), "limit": 1000},
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


def compute_actual_z(spot, rv_ann, tau, expiry):
    try:
        if spot<=0 or rv_ann<=0 or tau<=0 or math.isnan(expiry): return float("nan")
        sig = rv_ann / math.sqrt(MINS_PER_YEAR) * math.sqrt(tau)
        if sig <= 0: return float("nan")
        return math.log(expiry / spot) / sig
    except: return float("nan")


def zdrift(az_list):
    if len(az_list) < W_SHORT: return float("nan")
    tail = az_list[-max(W_LONG, len(az_list)):]
    zs = np.mean(tail[-W_SHORT:])
    zl = np.mean(tail[-W_LONG:] if len(tail) >= W_LONG else tail)
    return float(np.clip(ALPHA*zs + (1-ALPHA)*zl, -CAP, CAP))


def p_zdrift(spot, strike, rv_ann, tau, zd):
    sig = rv_ann / math.sqrt(MINS_PER_YEAR) * math.sqrt(tau)
    if sig <= 0: return float("nan")
    return float(np.clip(norm.cdf(zd - math.log(strike/spot)/sig), 0.03, 0.97))


def pnl_flat(pm, outcome, edge, threshold=EDGE_THRESHOLD):
    if edge < threshold: return 0.0
    return FLAT_BET*(1-pm)/pm if outcome == 1 else -FLAT_BET


def print_group(label, sub, threshold=EDGE_THRESHOLD):
    if sub.empty:
        print(f"  {label}: n=0")
        return
    n = len(sub)
    wr = sub["outcome"].mean()
    be = (sub["pm"] / (1 + sub["pm"])).mean()

    pnl_all = sum(FLAT_BET*(1-r["pm"])/r["pm"] if r["outcome"]==1 else -FLAT_BET
                  for _, r in sub.iterrows())
    zd_trades = sub[sub["zd_decision"]=="trade"]
    zd_wr = zd_trades["outcome"].mean() if len(zd_trades) else float("nan")
    zd_pnl = sub["zd_pnl"].sum()

    print(f"  {label}")
    print(f"    n={n}  WR={wr:.3f}  BE={be:.3f}  WR-BE={wr-be:+.3f}")
    print(f"    PnL if all allowed (flat ${FLAT_BET:.0f}): ${pnl_all:+.2f}")
    print(f"    z_drift would trade: {len(zd_trades)} / {n}  "
          f"WR={zd_wr:.3f}" if not math.isnan(zd_wr) else f"    z_drift would trade: 0 / {n}")
    print(f"    z_drift filtered PnL: ${zd_pnl:+.2f}")


def main():
    # --- Load blocked_trades (BTC YES) ---
    bt = pd.read_csv(CSV_BLOCKED, low_memory=False)
    for c in ["pm","spot","strike","tau_minutes","ema_stack_bias","composite_p_up","stoch_k"]:
        bt[c] = pd.to_numeric(bt[c], errors="coerce")
    bt["close_ts"] = pd.to_datetime(bt["close_ts"], utc=True, errors="coerce")
    btc_yes = bt[(bt["asset"]=="BTC") & (bt["side"]=="yes") &
                  bt["close_ts"].notna() & bt["spot"].notna() &
                  bt["strike"].notna() & bt["tau_minutes"].notna()].copy()

    # Deduplicate: one contract per (close_ts, strike)
    btc_yes = btc_yes.sort_values("close_ts").drop_duplicates(
        subset=["close_ts","strike"], keep="first"
    ).reset_index(drop=True)

    # Keep only rows with ema_stack_bias signal present (for group analysis)
    has_signal = btc_yes["ema_stack_bias"].notna()
    print(f"BTC YES unique contracts: {len(btc_yes)}  (signal present: {has_signal.sum()})")

    # --- Load paper_trades for rv_ann and z_drift history ---
    pt = pd.read_csv(CSV_TRADES, low_memory=False)
    for c in ["spot","realized_vol_annual","tau_minutes","floor_strike"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt["close_time"] = pd.to_datetime(pt["close_time"], utc=True, errors="coerce")
    pt = pt.sort_values("close_time").reset_index(drop=True)
    pt_valid = pt[pt["close_time"].notna() & pt["realized_vol_annual"].notna()].copy()

    rv_lookup = (pt_valid.sort_values("close_time")
                 .drop_duplicates("close_time")[["close_time","realized_vol_annual"]])

    btc_yes = pd.merge_asof(
        btc_yes.sort_values("close_ts"),
        rv_lookup.rename(columns={"close_time":"close_ts"}),
        on="close_ts", direction="nearest"
    )
    btc_yes["realized_vol_annual"] = btc_yes["realized_vol_annual"].fillna(
        rv_lookup["realized_vol_annual"].median()
    )

    # --- Fetch 1m Binance data ---
    all_ts = pd.concat([pt_valid["close_time"], btc_yes["close_ts"]]).dropna()
    s_ms = int(all_ts.min().timestamp()*1000) - 120_000
    e_ms = int(all_ts.max().timestamp()*1000) + 120_000
    print(f"Fetching 1m candles {pd.Timestamp(s_ms//1000,unit='s').date()} → "
          f"{pd.Timestamp(e_ms//1000,unit='s').date()}...")
    m1 = fetch_1m(s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    # --- Compute outcomes from price ---
    btc_yes["expiry_price"] = [price_at(r["close_ts"], m1.index, m1["open"])
                                for _, r in btc_yes.iterrows()]
    btc_yes["outcome"] = (btc_yes["expiry_price"] > btc_yes["strike"]).astype(float)
    btc_yes.loc[btc_yes["expiry_price"].isna(), "outcome"] = float("nan")
    n_priced = btc_yes["expiry_price"].notna().sum()
    print(f"Outcomes derived: {n_priced}/{len(btc_yes)}")

    # Drop rows without expiry price
    btc_yes = btc_yes[btc_yes["outcome"].notna()].copy()

    # --- z_drift walk-forward ---
    pt_valid["expiry_price"] = [price_at(r["close_time"], m1.index, m1["open"])
                                 for _, r in pt_valid.iterrows()]
    pt_valid["actual_z"] = [
        compute_actual_z(float(r["spot"]), float(r["realized_vol_annual"]),
                         float(r["tau_minutes"]), float(r["expiry_price"]))
        for _, r in pt_valid.iterrows()
    ]
    pt_history = pt_valid[pt_valid["actual_z"].notna()].sort_values("close_time")[
        ["close_time","actual_z"]].copy()

    def get_zdrift_at(ts):
        prior = pt_history[pt_history["close_time"] < ts]["actual_z"].tolist()
        return zdrift(prior), len(prior)

    results = []
    for _, row in btc_yes.iterrows():
        pm = float(row["pm"])
        spot = float(row["spot"])
        strike = float(row["strike"])
        rv_ann = float(row["realized_vol_annual"])
        tau = float(row["tau_minutes"])
        outcome = int(row["outcome"])
        ts = row["close_ts"]

        ema = float(row["ema_stack_bias"]) if not pd.isna(row["ema_stack_bias"]) else float("nan")
        p_up = float(row["composite_p_up"]) if not pd.isna(row["composite_p_up"]) else float("nan")
        sk = float(row["stoch_k"]) if not pd.isna(row["stoch_k"]) else float("nan")

        # Group classification
        if math.isnan(ema):
            group = "no_signal"
        elif ema == 0:
            if not math.isnan(p_up) and p_up >= G1_P_UP_MIN and not math.isnan(sk) and sk < G1_STOCH_MAX:
                group = "G1_match"      # neutral_ema_g1 would block
            elif not math.isnan(p_up) and p_up >= G1_P_UP_MIN and not math.isnan(sk) and sk >= G1_STOCH_MAX:
                group = "G1_miss_stoch" # G1 rescue (stoch ok)
            elif not math.isnan(p_up) and p_up < G1_P_UP_MIN:
                group = "ema0_bear_pup" # ema=0 but not bullish p_up
            else:
                group = "ema0_other"
        elif ema == 1:
            group = "ema_bull"
        elif ema == -1:
            group = "ema_bear"
        else:
            group = "ema_other"

        # z_drift
        zd, n_prior = get_zdrift_at(ts)
        zd_active = not math.isnan(zd)
        if zd_active:
            p_zd = p_zdrift(spot, strike, rv_ann, tau, zd)
            edge = p_zd - pm
            zd_pnl = pnl_flat(pm, outcome, edge)
            zd_decision = "trade" if edge >= EDGE_THRESHOLD else "block"
        else:
            p_zd = float("nan"); edge = float("nan")
            zd_pnl = 0.0; zd_decision = "no_history"

        results.append({
            "close_ts": ts, "pm": pm, "outcome": outcome,
            "group": group, "ema": ema, "p_up": p_up, "stoch_k": sk,
            "zd": zd, "p_zd": p_zd, "edge_zd": edge,
            "zd_pnl": zd_pnl, "zd_decision": zd_decision,
        })

    rdf = pd.DataFrame(results)

    print(f"\n{'='*65}")
    print(f"  NEUTRAL_EMA_G1 SIMULATION")
    print(f"  Condition: ema_stack=0 AND comp_p_up>={G1_P_UP_MIN} AND stoch_k<{G1_STOCH_MAX}")
    print(f"{'='*65}")

    print(f"\n--- GROUP RESULTS (all blocked trades) ---")
    groups = [
        ("G1_match      [gate blocks]", "G1_match"),
        ("G1_miss_stoch [gate allows]", "G1_miss_stoch"),
        ("ema0_bear_pup [ema=0,p_up<.60]", "ema0_bear_pup"),
        ("ema_bull      [ema=+1]", "ema_bull"),
        ("ema_bear      [ema=-1]", "ema_bear"),
        ("no_signal     [no signal]", "no_signal"),
    ]
    for label, gname in groups:
        sub = rdf[rdf["group"]==gname]
        print_group(label, sub)
        print()

    # G1 match: pm breakdown
    g1 = rdf[rdf["group"]=="G1_match"]
    if len(g1):
        print(f"\n--- G1_MATCH: Breakdown by pm bucket ---")
        for lo, hi in [(0,.25),(.25,.40),(.40,.55),(.55,.70),(.70,.85),(.85,1.0)]:
            sub = g1[(g1["pm"]>=lo)&(g1["pm"]<hi)]
            if len(sub) < 2: continue
            wr = sub["outcome"].mean()
            be = (sub["pm"]/(1+sub["pm"])).mean()
            wt = sub[sub["zd_decision"]=="trade"]
            pnl_all = sum(FLAT_BET*(1-r["pm"])/r["pm"] if r["outcome"]==1 else -FLAT_BET
                          for _, r in sub.iterrows())
            print(f"  pm [{lo:.2f},{hi:.2f}): n={len(sub):4d}  WR={wr:.3f}  BE={be:.3f}  "
                  f"WR-BE={wr-be:+.3f}  pnl_all=${pnl_all:+.2f}  "
                  f"zd_tr={len(wt)}  zd_pnl=${sub['zd_pnl'].sum():+.2f}")

    # z_drift value distribution for G1
    if len(g1):
        active_g1 = g1[g1["zd"].notna()]
        print(f"\n--- G1_MATCH: z_drift distribution (n={len(active_g1)} active) ---")
        if len(active_g1):
            zv = active_g1["zd"].values
            print(f"  mean={np.mean(zv):+.4f}  >0: {(zv>0).sum()}  <=0: {(zv<=0).sum()}")
            for lo, hi in [(-0.5,-0.2),(-0.2,0),(0,0.2),(0.2,0.5)]:
                sub = active_g1[(active_g1["zd"]>=lo)&(active_g1["zd"]<hi)]
                if len(sub):
                    wr = sub["outcome"].mean()
                    be = (sub["pm"]/(1+sub["pm"])).mean()
                    pnl = sub["zd_pnl"].sum()
                    print(f"  z_drift [{lo:+.1f},{hi:+.1f}): n={len(sub):3d}  "
                          f"WR={wr:.3f}  BE={be:.3f}  zd_pnl=${pnl:+.2f}")

    print(f"\n--- DECISION ---")
    g1_all = len(g1)
    g1_pnl_if_all = sum(FLAT_BET*(1-r["pm"])/r["pm"] if r["outcome"]==1 else -FLAT_BET
                        for _, r in g1.iterrows()) if g1_all else 0
    g1_pnl_zd = g1["zd_pnl"].sum() if g1_all else 0
    g1_wr = g1["outcome"].mean() if g1_all else float("nan")
    g1_be = (g1["pm"]/(1+g1["pm"])).mean() if g1_all else float("nan")
    g1_zd_tr = (g1["zd_decision"]=="trade").sum()
    print(f"  G1_match: n={g1_all}  WR={g1_wr:.3f}  BE={g1_be:.3f}  WR-BE={g1_wr-g1_be:+.3f}")
    print(f"  PnL if gate removed (all allowed): ${g1_pnl_if_all:+.2f}")
    print(f"  PnL if z_drift filters instead:   ${g1_pnl_zd:+.2f}")
    print(f"  z_drift would trade {g1_zd_tr}/{g1_all}")
    if g1_wr > g1_be:
        print(f"\n  ** WR > BE: gate is BLOCKING PROFITABLE TRADES **")
        if g1_pnl_zd > 0:
            print(f"  ** z_drift filters to ${g1_pnl_zd:+.2f} → REMOVE gate, use z_drift **")
        else:
            print(f"  ** z_drift PnL negative → z_drift over-filters; just remove gate **")
    else:
        print(f"\n  ** WR <= BE: gate correctly blocks losers **")


if __name__ == "__main__":
    main()
