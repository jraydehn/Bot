#!/usr/bin/env python3
"""
simulate_btc_lgbm_vs_zdrift.py

Retroactive comparison: BTC LGBM (shadow model) vs z_drift vs current model.

Methodology:
- Load paper_trades_btc15m.csv (current model decisions) + blocked_trades.csv (BTC)
- Derive actual outcomes from Binance 1m price at close_ts vs strike
- Walk-forward z_drift built from all prior actual_z (same as ETH/SOL comprehensive sim)
- Retroactive LGBM: run btc_lgbm.pkl on available features; NaN for 11 missing ones
  (covers ~67% of feature importance — results are approximate)
- Flat $10/trade for fair comparison across all three scenarios

Caveat: LGBM predictions are degraded by ~33% missing feature importance.
Missing: chg_30m(10%), vol_eff(10%), ema_stretch_score, chg_10m, vpin_score,
         confirmation_score, composite_trend, composite_rev, vol_score,
         funding_bias, vwap_stretch_score.
"""

import math, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

RESULTS_DIR    = Path(__file__).parent / "results"
CSV_TRADES     = RESULTS_DIR / "paper_trades_btc15m.csv"
CSV_BLOCKED    = RESULTS_DIR / "blocked_trades.csv"
LGBM_PATH      = Path(__file__).parent / "reform_results" / "btc_lgbm.pkl"
MINS_PER_YEAR  = 525600.0
EDGE_THRESHOLD = 0.04
W_SHORT, W_LONG, ALPHA, CAP = 10, 30, 0.6, 0.5
FLAT_BET       = 10.0


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


def infer_lgbm(pipe, row):
    """Run BTC LGBM on a CSV row, using NaN for unavailable features."""
    try:
        feat_map = {
            "offset_pct":        float(row.get("offset_pct", float("nan"))),
            "p_market":          float(row.get("p_market",   float("nan"))),
            "tau_minutes":       float(row.get("tau_minutes", float("nan"))),
            "side_enc":          1.0 if str(row.get("side","")).lower() == "yes" else 0.0,
            "composite_p_up":    float(row.get("composite_p_up", float("nan"))),
            "composite_trend":   float("nan"),   # not in old CSV
            "composite_rev":     float("nan"),   # not in old CSV
            "ema_stack_bias":    float(row.get("ema_bias", float("nan"))),
            "ema_stretch_score": float("nan"),   # not in old CSV
            "stoch_k":           float(row.get("stoch_k_15m", float("nan"))),
            "vwap_stretch_score":float("nan"),   # not in old CSV
            "vwap_distance_pct": float(row.get("vwap_dist", float("nan"))),
            "vol_score":         float("nan"),   # not in old CSV
            "vpin_score":        float("nan"),   # not in old CSV
            "confirmation_score":float("nan"),   # not in old CSV
            "funding_bias":      float("nan"),   # not in old CSV
            "chg_30m":           float("nan"),   # not in old CSV
            "chg_10m":           float("nan"),   # not in old CSV
            "chg_5m":            float(row.get("chg_5m", float("nan"))),
            "vol_eff":           float("nan"),   # not in old CSV
        }
        feats  = pipe["features"]
        vec    = np.array([[feat_map.get(f, float("nan")) for f in feats]])
        p_raw  = float(pipe["clf"].predict_proba(vec)[0, 1])
        platt  = pipe.get("platt")
        if platt is not None:
            logit = math.log(max(p_raw, 1e-6) / max(1 - p_raw, 1e-6))
            p_cal = float(platt.predict_proba([[logit]])[0, 1])
        else:
            p_cal = p_raw
        return float(np.clip(p_cal, 0.01, 0.99))
    except Exception as e:
        return float("nan")


def pnl_flat(pm, side, outcome, p_model):
    """Flat $10 PnL if model has edge >= threshold."""
    if math.isnan(p_model): return 0.0
    edge = (p_model - pm) if side == "yes" else (pm - p_model)
    if edge < EDGE_THRESHOLD: return 0.0
    if side == "yes":
        return FLAT_BET * (1 - pm) / pm if outcome == 1 else -FLAT_BET
    else:
        return FLAT_BET * pm / (1 - pm) if outcome == 0 else -FLAT_BET


def main():
    # ── Load BTC LGBM ────────────────────────────────────────────────────
    with open(LGBM_PATH, "rb") as f:
        pipe = pickle.load(f)
    print(f"BTC LGBM loaded: {len(pipe['features'])} features, AUC={pipe['auc_te']:.3f}")
    print(f"NOTE: 11 features will be NaN (~33% of feature importance)")
    print(f"      chg_30m, vol_eff are biggest missing (10% each)")

    # ── Load paper_trades ────────────────────────────────────────────────
    pt = pd.read_csv(CSV_TRADES, low_memory=False)
    for c in ["spot", "realized_vol_annual", "tau_minutes", "p_market",
              "floor_strike", "resolved_yes", "would_pnl", "bet_amount",
              "offset_pct", "composite_p_up", "ema_bias", "stoch_k_15m",
              "vwap_dist", "chg_5m"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    pt["close_time"] = pd.to_datetime(pt["close_time"], utc=True, errors="coerce")
    pt = pt.sort_values("close_time").reset_index(drop=True)

    pt["close_ts"]  = pt["close_time"]
    pt["strike"]    = pt["floor_strike"]
    pt["pm"]        = pt["p_market"]
    pt["source"]    = "paper_trade"
    pt["is_current_trade"] = (pt["decision"] == "trade")

    rv_med = pt["realized_vol_annual"].median()
    print(f"\nPaper trades: {len(pt)} total | "
          f"current model took: {pt['is_current_trade'].sum()} | "
          f"date range: {pt['close_ts'].min().date()} → {pt['close_ts'].max().date()}")

    # ── Load BTC blocked trades ──────────────────────────────────────────
    bt_all = pd.read_csv(CSV_BLOCKED, low_memory=False)
    bt = bt_all[bt_all["asset"] == "BTC"].copy()
    for c in ["pm", "spot", "strike", "tau_minutes", "offset_pct",
              "composite_p_up", "ema_stack_bias", "stoch_k", "vwap_stretch",
              "composite_trend", "composite_rev"]:
        if c in bt.columns:
            bt[c] = pd.to_numeric(bt[c], errors="coerce")
    bt["close_ts"] = pd.to_datetime(bt["close_ts"], utc=True, errors="coerce")
    bt["source"]   = "blocked"
    bt["is_current_trade"] = False

    # Rename blocked_trades columns to match paper_trades schema
    bt = bt.rename(columns={
        "ema_stack_bias": "ema_bias",
        "stoch_k":        "stoch_k_15m",
    })

    # Fill rv_ann for blocked trades from nearest paper_trade
    rv_lookup = (pt.sort_values("close_ts")
                   .drop_duplicates("close_ts")[["close_ts", "realized_vol_annual"]])
    bt_sorted = bt.sort_values("close_ts").reset_index(drop=True)
    bt_merged = pd.merge_asof(
        bt_sorted,
        rv_lookup.rename(columns={"realized_vol_annual": "rv_joined"}),
        on="close_ts", direction="nearest"
    )
    bt_merged["realized_vol_annual"] = bt_merged["rv_joined"].fillna(rv_med)
    bt = bt_merged

    print(f"Blocked trades: {len(bt)} total | "
          f"YES: {(bt['side']=='yes').sum()} | NO: {(bt['side']=='no').sum()}")

    # ── Combine ──────────────────────────────────────────────────────────
    shared = ["close_ts", "strike", "spot", "pm", "side", "tau_minutes",
              "realized_vol_annual", "offset_pct", "composite_p_up",
              "ema_bias", "stoch_k_15m", "vwap_dist", "chg_5m",
              "source", "is_current_trade"]
    # blocked_trades may be missing some cols — fill with NaN
    for col in shared:
        if col not in bt.columns:
            bt[col] = float("nan")
        if col not in pt.columns:
            pt[col] = float("nan")

    combined = pd.concat([pt[shared], bt[shared]], ignore_index=True)
    combined = combined[
        combined["close_ts"].notna() &
        combined["strike"].notna() &
        combined["spot"].notna() &
        combined["pm"].notna() &
        combined["tau_minutes"].notna() &
        (combined["pm"] > 0) & (combined["pm"] < 1)
    ].copy()

    # Dedup: prefer paper_trade over blocked for same (close_ts, strike, side)
    combined = combined.sort_values("source")  # "blocked" < "paper_trade"
    combined = combined.drop_duplicates(subset=["close_ts", "strike", "side"], keep="last")
    combined = combined.sort_values("close_ts").reset_index(drop=True)
    print(f"Combined unique: {len(combined)} "
          f"(YES: {(combined['side']=='yes').sum()}  NO: {(combined['side']=='no').sum()})")

    # ── Fetch 1m data ────────────────────────────────────────────────────
    s_ms = int(combined["close_ts"].min().timestamp() * 1000) - 120_000
    e_ms = int(combined["close_ts"].max().timestamp() * 1000) + 120_000
    print(f"\nFetching BTC 1m candles ({pd.Timestamp(s_ms//1000, unit='s').date()} → "
          f"{pd.Timestamp(e_ms//1000, unit='s').date()})...")
    m1 = fetch_1m(s_ms, e_ms)
    print(f"Fetched {len(m1)} bars")

    # ── Outcomes ─────────────────────────────────────────────────────────
    combined["expiry_price"] = [
        price_at(r["close_ts"], m1.index, m1["open"])
        for _, r in combined.iterrows()
    ]
    combined["outcome_yes"] = (combined["expiry_price"] > combined["strike"]).astype(float)
    combined["outcome_yes"] = combined["outcome_yes"].where(combined["expiry_price"].notna(), float("nan"))

    def outcome_for_side(row):
        if pd.isna(row["outcome_yes"]): return float("nan")
        return row["outcome_yes"] if row["side"] == "yes" else 1.0 - row["outcome_yes"]

    combined["outcome"] = [outcome_for_side(r) for _, r in combined.iterrows()]
    combined["actual_z"] = [
        compute_actual_z(float(r["spot"]), float(r["realized_vol_annual"]),
                         float(r["tau_minutes"]), float(r["expiry_price"]))
        if not pd.isna(r["expiry_price"]) else float("nan")
        for _, r in combined.iterrows()
    ]
    print(f"Outcomes resolved: {combined['outcome'].notna().sum()}/{len(combined)}")

    # ── Walk-forward simulation ──────────────────────────────────────────
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
        is_curr = bool(row["is_current_trade"])
        src     = row["source"]

        # z_drift
        zd    = zdrift(prior_az)
        zd_ok = not math.isnan(zd)
        p_zd  = p_zdrift(spot, strike, rv_ann, tau, zd) if zd_ok else float("nan")

        # LGBM retroactive
        p_lgbm = infer_lgbm(pipe, row)

        # PnL under each model
        pnl_curr  = pnl_flat(pm, side, int(outcome), float(row.get("pm", pm)) if is_curr else float("nan"))
        # For current model PnL, use actual decision (not model threshold)
        if is_curr:
            if side == "yes":
                pnl_curr = FLAT_BET * (1 - pm) / pm if outcome == 1 else -FLAT_BET
            else:
                pnl_curr = FLAT_BET * pm / (1 - pm) if outcome == 0 else -FLAT_BET
        else:
            pnl_curr = 0.0

        pnl_zd    = pnl_flat(pm, side, int(outcome), p_zd)
        pnl_lgbm  = pnl_flat(pm, side, int(outcome), p_lgbm)

        zd_trades   = (not math.isnan(p_zd)) and (
            (side=="yes" and p_zd - pm >= EDGE_THRESHOLD) or
            (side=="no"  and pm - p_zd >= EDGE_THRESHOLD)
        )
        lgbm_trades = (not math.isnan(p_lgbm)) and (
            (side=="yes" and p_lgbm - pm >= EDGE_THRESHOLD) or
            (side=="no"  and pm - p_lgbm >= EDGE_THRESHOLD)
        )

        rows_out.append({
            "close_ts":    ts,
            "side":        side,
            "pm":          pm,
            "spot":        spot,
            "strike":      strike,
            "zd":          zd if zd_ok else float("nan"),
            "p_zd":        p_zd,
            "p_lgbm":      p_lgbm,
            "n_prior":     len(prior_az),
            "outcome":     outcome,
            "is_current":  is_curr,
            "zd_trades":   zd_trades,
            "lgbm_trades": lgbm_trades,
            "pnl_curr":    pnl_curr,
            "pnl_zd":      pnl_zd,
            "pnl_lgbm":    pnl_lgbm,
            "source":      src,
        })

        if not math.isnan(row["actual_z"]):
            prior_az.append(float(row["actual_z"]))

    rdf = pd.DataFrame(rows_out)

    # ── Print results ────────────────────────────────────────────────────
    curr_df = rdf[rdf["is_current"]]
    zd_df   = rdf[rdf["zd_trades"]]
    lg_df   = rdf[rdf["lgbm_trades"]]

    print(f"\n{'='*70}")
    print(f"  BTC: current z_drift model vs retroactive LGBM vs pure z_drift")
    print(f"  flat ${FLAT_BET:.0f}/trade  |  NOTE: LGBM missing 11/20 features (~33% importance)")
    print(f"{'='*70}")
    print(f"\n  Current model (z_drift-integrated lognormal, actual decisions):")
    print(f"    n={len(curr_df):4d}  WR={curr_df['outcome'].mean():.3f}  PnL=${curr_df['pnl_curr'].sum():+.2f}")
    print(f"\n  Pure z_drift (walk-forward, edge≥4%):")
    print(f"    n={len(zd_df):4d}  WR={zd_df['outcome'].mean():.3f}  PnL=${zd_df['pnl_zd'].sum():+.2f}  "
          f"delta=${zd_df['pnl_zd'].sum()-curr_df['pnl_curr'].sum():+.2f}")
    print(f"\n  Retroactive LGBM [~degraded by missing features] (edge≥4%):")
    print(f"    n={len(lg_df):4d}  WR={lg_df['outcome'].mean():.3f}  PnL=${lg_df['pnl_lgbm'].sum():+.2f}  "
          f"delta=${lg_df['pnl_lgbm'].sum()-curr_df['pnl_curr'].sum():+.2f}")

    # By side
    print(f"\n  By side:")
    for s in ["yes", "no"]:
        c_s  = rdf[rdf["is_current"] & (rdf["side"]==s)]
        zd_s = rdf[rdf["zd_trades"] & (rdf["side"]==s)]
        lg_s = rdf[rdf["lgbm_trades"] & (rdf["side"]==s)]
        print(f"    {s.upper():3s}: curr n={len(c_s):3d} WR={c_s['outcome'].mean():.3f} "
              f"${c_s['pnl_curr'].sum():+.2f}  |  "
              f"z_drift n={len(zd_s):3d} WR={zd_s['outcome'].mean():.3f} "
              f"${zd_s['pnl_zd'].sum():+.2f}  |  "
              f"LGBM n={len(lg_s):3d} WR={lg_s['outcome'].mean():.3f} "
              f"${lg_s['pnl_lgbm'].sum():+.2f}")

    # By pm bucket
    print(f"\n  By pm bucket:")
    print(f"  {'pm range':<14} {'n_all':>6} {'n_curr':>7} {'n_zd':>6} {'n_lgbm':>7} "
          f"{'curr_pnl':>10} {'zd_pnl':>10} {'lgbm_pnl':>10}")
    for lo, hi in [(0,.20),(.20,.40),(.40,.60),(.60,.80),(.80,1.0)]:
        sub = rdf[(rdf["pm"]>=lo)&(rdf["pm"]<hi)]
        if len(sub) == 0: continue
        c   = sub[sub["is_current"]]
        zd  = sub[sub["zd_trades"]]
        lg  = sub[sub["lgbm_trades"]]
        print(f"  [{lo:.2f},{hi:.2f})  {len(sub):>6}  {len(c):>7}  {len(zd):>6}  {len(lg):>7}  "
              f"{c['pnl_curr'].sum():>+10.2f}  {zd['pnl_zd'].sum():>+10.2f}  "
              f"{lg['pnl_lgbm'].sum():>+10.2f}")

    # LGBM p distribution check
    valid_lgbm = rdf[rdf["p_lgbm"].notna()]
    print(f"\n  LGBM p_gbdt distribution (n={len(valid_lgbm)}):")
    print(f"    mean={valid_lgbm['p_lgbm'].mean():.3f}  std={valid_lgbm['p_lgbm'].std():.3f}  "
          f"min={valid_lgbm['p_lgbm'].min():.3f}  max={valid_lgbm['p_lgbm'].max():.3f}")

    # Agreement analysis
    both     = rdf[rdf["is_current"] & rdf["lgbm_trades"]]
    curr_only= rdf[rdf["is_current"] & ~rdf["lgbm_trades"]]
    lgbm_only= rdf[~rdf["is_current"] & rdf["lgbm_trades"]]
    print(f"\n  Current vs LGBM agreement:")
    print(f"    Both trade:    n={len(both):4d}  WR={both['outcome'].mean():.3f}  "
          f"curr=${both['pnl_curr'].sum():+.2f}  lgbm=${both['pnl_lgbm'].sum():+.2f}")
    print(f"    Current only:  n={len(curr_only):4d}  WR={curr_only['outcome'].mean():.3f}  "
          f"curr=${curr_only['pnl_curr'].sum():+.2f}")
    print(f"    LGBM only:     n={len(lgbm_only):4d}  WR={lgbm_only['outcome'].mean():.3f}  "
          f"lgbm=${lgbm_only['pnl_lgbm'].sum():+.2f}")

    # Source breakdown
    print(f"\n  LGBM trades by source:")
    for src, g in rdf[rdf["lgbm_trades"]].groupby("source"):
        print(f"    {src:<15}: n={len(g):4d}  WR={g['outcome'].mean():.3f}  "
              f"PnL=${g['pnl_lgbm'].sum():+.2f}")


if __name__ == "__main__":
    main()
