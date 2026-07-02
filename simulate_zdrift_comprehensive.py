#!/usr/bin/env python3
"""
simulate_zdrift_comprehensive.py

Full comparison of z_drift vs LGBM for ETH and SOL, including ALL contracts
(both taken trades AND contracts blocked by any gate).

Methodology:
- Combine paper_trades CSV + blocked_trades.csv for each asset
- Deduplicate by (close_ts, strike, side)
- Derive actual outcome from Binance 1m price at close_ts vs strike
- Walk-forward z_drift from all prior actual_z (from all resolved contracts)
- Flat $10 bet per trade for fair comparison across models and scenarios
- LGBM baseline: flat $10 on contracts where decision="trade"
- z_drift: flat $10 wherever edge >= EDGE_THRESHOLD
- Compare PnL across models, by side, by pm bucket
"""

import math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR    = Path(__file__).parent / "results"
MINS_PER_YEAR  = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5
FLAT_BET       = 10.0

ASSETS = {
    "ETH": {
        "symbol":  "ETHUSDT",
        "csv":     RESULTS_DIR / "paper_trades_eth15m.csv",
    },
    "SOL": {
        "symbol":  "SOLUSDT",
        "csv":     RESULTS_DIR / "paper_trades_sol15m.csv",
    },
}


def fetch_1m(symbol, start_ms, end_ms):
    url, rows, cur = "https://api.binance.us/api/v3/klines", [], start_ms
    while cur < end_ms:
        r = requests.get(url, params={"symbol": symbol, "interval": "1m",
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


def price_at(ts, idx, opens):
    try:
        ts = pd.Timestamp(ts).tz_convert("UTC")
        i = idx.get_loc(ts) if ts in idx else idx.searchsorted(ts)
        if i >= len(idx): return float("nan")
        return float(opens.iloc[i])
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


def pnl_flat(pm, side, outcome, p_zd):
    """Return flat $10 PnL if z_drift has edge, else 0."""
    if math.isnan(p_zd): return 0.0
    edge = (p_zd - pm) if side == "yes" else (pm - p_zd)
    if edge < EDGE_THRESHOLD: return 0.0
    if side == "yes":
        return FLAT_BET * (1 - pm) / pm if outcome == 1 else -FLAT_BET
    else:
        return FLAT_BET * pm / (1 - pm) if outcome == 0 else -FLAT_BET


def lgbm_pnl_flat(pm, side, outcome, is_lgbm_trade):
    """Flat $10 LGBM PnL — only on contracts the LGBM model traded."""
    if not is_lgbm_trade: return 0.0
    if side == "yes":
        return FLAT_BET * (1 - pm) / pm if outcome == 1 else -FLAT_BET
    else:
        return FLAT_BET * pm / (1 - pm) if outcome == 0 else -FLAT_BET


def run_asset(asset, cfg):
    print(f"\n{'='*70}")
    print(f"  {asset} — COMPREHENSIVE z_drift vs LGBM SIMULATION")
    print(f"{'='*70}")

    # ── Load paper_trades ──────────────────────────────────────────────────
    pt = pd.read_csv(cfg["csv"], low_memory=False)
    for c in ["spot", "realized_vol_annual", "tau_minutes", "p_market",
              "floor_strike", "resolved_yes", "bet_amount", "would_pnl"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt["close_time"] = pd.to_datetime(pt["close_time"], utc=True, errors="coerce")
    pt = pt.sort_values("close_time").reset_index(drop=True)

    # Normalise to unified schema
    pt["close_ts"] = pt["close_time"]
    pt["strike"]   = pt["floor_strike"]
    pt["pm"]       = pt["p_market"]
    pt["source"]   = "paper_trade"
    pt["is_lgbm_trade"] = (pt["decision"] == "trade") & pt["resolved_yes"].notna()

    # rv_ann median for fallback fills
    rv_med = pt["realized_vol_annual"].median()

    # ── Load blocked_trades (this asset) ──────────────────────────────────
    bt_all = pd.read_csv(RESULTS_DIR / "blocked_trades.csv", low_memory=False)
    bt = bt_all[bt_all["asset"] == asset].copy()
    for c in ["pm", "spot", "strike", "tau_minutes"]:
        bt[c] = pd.to_numeric(bt[c], errors="coerce")
    bt["close_ts"] = pd.to_datetime(bt["close_ts"], utc=True, errors="coerce")
    bt["source"]        = "blocked"
    bt["is_lgbm_trade"] = False

    print(f"  Paper trades:      {len(pt)} total, "
          f"{pt['is_lgbm_trade'].sum()} LGBM took, "
          f"{pt['resolved_yes'].notna().sum()} resolved")
    print(f"  Blocked trades:    {len(bt)} total | "
          f"YES: {(bt['side']=='yes').sum()}  NO: {(bt['side']=='no').sum()}")
    print(f"  Paper trade dates: {pt['close_ts'].min().date()} → {pt['close_ts'].max().date()}")
    print(f"  Blocked dates:     {bt['close_ts'].min().date()} → {bt['close_ts'].max().date()}")

    # ── Merge rv_ann into blocked via nearest paper_trade timestamp ───────
    rv_lookup = (pt.sort_values("close_ts")
                   .drop_duplicates("close_ts")[["close_ts", "realized_vol_annual"]])
    # blocked_trades doesn't have rv_ann — join from paper_trades
    bt_sorted = bt.sort_values("close_ts").reset_index(drop=True)
    bt_merged = pd.merge_asof(
        bt_sorted,
        rv_lookup.rename(columns={"realized_vol_annual": "rv_ann_joined"}),
        on="close_ts", direction="nearest"
    )
    bt_merged["realized_vol_annual"] = bt_merged["rv_ann_joined"].fillna(rv_med)
    bt = bt_merged
    n_fill = bt_merged["rv_ann_joined"].isna().sum()
    print(f"  rv_ann fill:       {n_fill} blocked rows filled with median {rv_med:.4f}")

    # ── Combine all contracts ─────────────────────────────────────────────
    shared_cols = ["close_ts", "strike", "spot", "pm", "side", "tau_minutes",
                   "realized_vol_annual", "source", "is_lgbm_trade"]
    combined = pd.concat([
        pt[shared_cols],
        bt[shared_cols],
    ], ignore_index=True)

    # Drop rows missing core data
    combined = combined[
        combined["close_ts"].notna() &
        combined["strike"].notna() &
        combined["spot"].notna() &
        combined["pm"].notna() &
        combined["tau_minutes"].notna() &
        (combined["pm"] > 0) & (combined["pm"] < 1)
    ].copy()

    # Dedup: prefer paper_trade over blocked for same (close_ts, strike, side)
    combined = combined.sort_values("source")  # "blocked" < "paper_trade" alphabetically
    combined = combined.drop_duplicates(subset=["close_ts", "strike", "side"], keep="last")
    combined = combined.sort_values("close_ts").reset_index(drop=True)
    print(f"  Combined unique contracts: {len(combined)} "
          f"(YES: {(combined['side']=='yes').sum()}  NO: {(combined['side']=='no').sum()})")

    # ── Fetch 1m data ─────────────────────────────────────────────────────
    s_ms = int(combined["close_ts"].min().timestamp() * 1000) - 120_000
    e_ms = int(combined["close_ts"].max().timestamp() * 1000) + 120_000
    print(f"\n  Fetching {cfg['symbol']} 1m candles "
          f"({pd.Timestamp(s_ms//1000, unit='s').date()} → "
          f"{pd.Timestamp(e_ms//1000, unit='s').date()})...")
    m1 = fetch_1m(cfg["symbol"], s_ms, e_ms)
    print(f"  Fetched {len(m1)} bars")

    # ── Derive outcomes ───────────────────────────────────────────────────
    combined["expiry_price"] = [
        price_at(r["close_ts"], m1.index, m1["open"])
        for _, r in combined.iterrows()
    ]
    combined["outcome_yes"] = (combined["expiry_price"] > combined["strike"]).astype(float)
    combined["outcome_yes"] = combined["outcome_yes"].where(combined["expiry_price"].notna(), float("nan"))

    def outcome_for_side(row):
        if pd.isna(row["outcome_yes"]): return float("nan")
        if row["side"] == "yes": return row["outcome_yes"]
        return 1.0 - row["outcome_yes"]  # NO wins when price <= strike

    combined["outcome"] = [outcome_for_side(r) for _, r in combined.iterrows()]
    n_priced = combined["expiry_price"].notna().sum()
    print(f"  Expiry price resolved: {n_priced}/{len(combined)}")

    # ── Compute actual_z for z_drift history ─────────────────────────────
    combined["actual_z"] = [
        compute_actual_z(float(r["spot"]), float(r["realized_vol_annual"]),
                         float(r["tau_minutes"]), float(r["expiry_price"]))
        if not pd.isna(r["expiry_price"]) else float("nan")
        for _, r in combined.iterrows()
    ]

    # ── Walk-forward simulation ───────────────────────────────────────────
    prior_az = []
    rows_out  = []

    for _, row in combined.iterrows():
        if pd.isna(row["outcome"]): continue

        ts      = row["close_ts"]
        pm      = float(row["pm"])
        spot    = float(row["spot"])
        strike  = float(row["strike"])
        rv_ann  = float(row["realized_vol_annual"])
        tau     = float(row["tau_minutes"])
        side    = str(row["side"]).lower()
        outcome = float(row["outcome"])
        is_lgbm = bool(row["is_lgbm_trade"])
        src     = row["source"]

        zd       = zdrift(prior_az)
        zd_ok    = not math.isnan(zd)
        p_zd     = p_zdrift(spot, strike, rv_ann, tau, zd) if zd_ok else float("nan")

        pnl_zd   = pnl_flat(pm, side, int(outcome), p_zd)
        pnl_lgbm = lgbm_pnl_flat(pm, side, int(outcome), is_lgbm)

        zd_trades = (not math.isnan(p_zd)) and (
            ((side == "yes") and (p_zd - pm >= EDGE_THRESHOLD)) or
            ((side == "no")  and (pm - p_zd >= EDGE_THRESHOLD))
        )

        rows_out.append({
            "close_ts":  ts,
            "side":      side,
            "pm":        pm,
            "spot":      spot,
            "strike":    strike,
            "zd":        zd if zd_ok else float("nan"),
            "p_zd":      p_zd,
            "n_prior":   len(prior_az),
            "outcome":   outcome,
            "is_lgbm":   is_lgbm,
            "zd_trades": zd_trades,
            "pnl_lgbm":  pnl_lgbm,
            "pnl_zd":    pnl_zd,
            "source":    src,
        })

        if not math.isnan(row["actual_z"]):
            prior_az.append(float(row["actual_z"]))

    rdf = pd.DataFrame(rows_out)

    # ── Summary ───────────────────────────────────────────────────────────
    lgbm_trades = rdf[rdf["is_lgbm"]]
    zd_trades   = rdf[rdf["zd_trades"]]
    active      = rdf[rdf["zd"].notna()]

    lgbm_pnl = lgbm_trades["pnl_lgbm"].sum()
    zd_pnl   = zd_trades["pnl_zd"].sum()

    print(f"\n  {'─'*60}")
    print(f"  RESULTS (flat ${FLAT_BET:.0f}/trade)")
    print(f"  {'─'*60}")
    print(f"  LGBM trades:     n={len(lgbm_trades):4d}  "
          f"WR={lgbm_trades['outcome'].mean():.3f}  PnL=${lgbm_pnl:+.2f}")
    print(f"  z_drift trades:  n={len(zd_trades):4d}  "
          f"WR={zd_trades['outcome'].mean():.3f}  PnL=${zd_pnl:+.2f}  "
          f"delta=${zd_pnl-lgbm_pnl:+.2f}")
    print(f"  z_drift active:  n={len(active)} (n_prior≥{W_SHORT})")

    # Breakdown by side
    print(f"\n  By side:")
    for s in ["yes", "no"]:
        lg_s  = rdf[(rdf["is_lgbm"]) & (rdf["side"]==s)]
        zd_s  = rdf[(rdf["zd_trades"]) & (rdf["side"]==s)]
        lp    = lg_s["pnl_lgbm"].sum()
        zp    = zd_s["pnl_zd"].sum()
        lwr   = lg_s["outcome"].mean() if len(lg_s) else float("nan")
        zwr   = zd_s["outcome"].mean() if len(zd_s) else float("nan")
        print(f"    {s.upper():3s}: LGBM n={len(lg_s):3d} WR={lwr:.3f} ${lp:+.2f}  |  "
              f"z_drift n={len(zd_s):3d} WR={zwr:.3f} ${zp:+.2f}  delta=${zp-lp:+.2f}")

    # Breakdown by pm bucket
    print(f"\n  By pm bucket (all contracts with outcome):")
    print(f"  {'pm range':<14} {'n_all':>6} {'n_lgbm':>7} {'n_zd':>6} "
          f"{'lgbm_pnl':>10} {'zd_pnl':>10} {'delta':>10} "
          f"{'lgbm_WR':>8} {'zd_WR':>7} {'BE':>6}")
    buckets = [(0,.10),(.10,.20),(.20,.30),(.30,.40),(.40,.50),
               (.50,.60),(.60,.70),(.70,.80),(.80,.90),(.90,1.0)]
    for lo, hi in buckets:
        sub  = rdf[(rdf["pm"]>=lo)&(rdf["pm"]<hi)]
        if len(sub) == 0: continue
        lg   = sub[sub["is_lgbm"]]
        zd   = sub[sub["zd_trades"]]
        lp   = lg["pnl_lgbm"].sum()
        zp   = zd["pnl_zd"].sum()
        lwr  = lg["outcome"].mean() if len(lg) else float("nan")
        zwr  = zd["outcome"].mean() if len(zd) else float("nan")
        be   = sub["pm"].mean()
        print(f"  [{lo:.2f},{hi:.2f})  {len(sub):>6}  {len(lg):>7}  {len(zd):>6}  "
              f"{lp:>+10.2f}  {zp:>+10.2f}  {zp-lp:>+10.2f}  "
              f"{lwr:>8.3f}  {zwr:>7.3f}  {be:>5.3f}")

    # Source breakdown: how many zd trades come from paper vs blocked
    print(f"\n  z_drift trades by source:")
    for src, g in rdf[rdf["zd_trades"]].groupby("source"):
        print(f"    {src:<15}: n={len(g):4d}  WR={g['outcome'].mean():.3f}  "
              f"PnL=${g['pnl_zd'].sum():+.2f}")

    # Overlap: trades both models agree on
    both    = rdf[rdf["is_lgbm"] & rdf["zd_trades"]]
    lgbm_only = rdf[rdf["is_lgbm"] & ~rdf["zd_trades"]]
    zd_only = rdf[~rdf["is_lgbm"] & rdf["zd_trades"]]
    print(f"\n  Agreement breakdown:")
    print(f"    Both trade:    n={len(both):4d}  WR={both['outcome'].mean():.3f}  "
          f"LGBM=${both['pnl_lgbm'].sum():+.2f}  zd=${both['pnl_zd'].sum():+.2f}")
    print(f"    LGBM only:     n={len(lgbm_only):4d}  WR={lgbm_only['outcome'].mean():.3f}  "
          f"LGBM=${lgbm_only['pnl_lgbm'].sum():+.2f}")
    print(f"    z_drift only:  n={len(zd_only):4d}  WR={zd_only['outcome'].mean():.3f}  "
          f"zd=${zd_only['pnl_zd'].sum():+.2f}")
    print(f"    Neither trades:n={len(rdf)-len(both)-len(lgbm_only)-len(zd_only):4d}")

    # z_drift value distribution
    if len(active):
        zv = active["zd"].values
        print(f"\n  z_drift distribution (n={len(active)}):")
        print(f"    mean={np.mean(zv):+.4f}  std={np.std(zv):.4f}  "
              f">0: {(zv>0).sum()}  ≤0: {(zv<=0).sum()}")

    return rdf


def main():
    print("Comprehensive z_drift vs LGBM simulation — ETH and SOL")
    print("All contracts (paper_trades + blocked), Binance 1m outcomes, flat $10/trade\n")

    all_results = {}
    for asset, cfg in ASSETS.items():
        all_results[asset] = run_asset(asset, cfg)

    print(f"\n{'='*70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*70}")
    for asset, rdf in all_results.items():
        lg  = rdf[rdf["is_lgbm"]]
        zd  = rdf[rdf["zd_trades"]]
        lp  = lg["pnl_lgbm"].sum()
        zp  = zd["pnl_zd"].sum()
        print(f"  {asset}: LGBM n={len(lg):3d} ${lp:+.2f}  |  "
              f"z_drift n={len(zd):3d} ${zp:+.2f}  delta=${zp-lp:+.2f}")


if __name__ == "__main__":
    main()
