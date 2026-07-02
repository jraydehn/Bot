"""
gate_attr_scan.py — Per-gate P&L attribution for the Live_approx BTC model.

For each contract row with ungated edge > MIN_EDGE, identifies the first gate
that blocks it and records the P&L of that would-be trade.

  gate_pnl_impact  = sum(-pnl_if_taken) for all trades blocked by this gate
    positive  → gate helped (blocked losers)
    negative  → gate hurt  (blocked winners)

Output: results/gate_attr_scan_YYYYMMDD.csv
"""
import math, warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

RES_DIR    = Path(__file__).parent / "results"
BANKROLL   = 1_000.0
KELLY_MULT = 0.30
KELLY_CAP  = 0.06
FEE_RATE   = 0.07
MIN_EDGE   = 0.005
PM_MIN, PM_MAX   = 0.10, 0.90
TAU_MIN, TAU_MAX = 5.0, 150.0


def _get(row, col, default=0.0):
    v = row.get(col, default)
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default

def _gets(row, col, default=""):
    v = row.get(col, default)
    return str(v) if (v is not None and str(v).strip() not in ("", "nan")) else default

def p_logn(spot, strike, vol_eff, tau_min, z_drift=0.0):
    tau_h = max(tau_min / 60.0, 1 / 60)
    sig   = vol_eff * math.sqrt(tau_h)
    if sig <= 0:
        return 0.5
    return float(norm.sf(math.log(strike / spot) / sig - z_drift))

def kelly_n(edge, pm_risk):
    if pm_risk <= 0:
        return 0.0
    return min(edge / pm_risk * KELLY_MULT, KELLY_CAP) * BANKROLL / pm_risk

def calc_pnl(side, pm, n, ry):
    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        return n * (1 - pm - fee) if ry == 1 else -n * (pm + fee)
    else:
        return n * (pm - fee) if ry == 0 else -n * (1 - pm + fee)


# ── individual gate checks ───────────────────────────────────────────────────

def _yes_early_gates(row, pm, ve, tau, sk, z, offset):
    """Returns name of first early gate that fires, or None."""
    ema   = _get(row, "ema_stack_bias",    0.0)
    rev   = _get(row, "composite_rev",     0.0)
    trend = _get(row, "composite_trend",   0.0)
    vwap  = _get(row, "vwap_stretch_score", 0.0)
    rsi4h = _get(row, "rsi_4h",            50.0)
    macd4h= _get(row, "macd_hist_4h",       0.0)

    if ve > 0 and offset > 0 and abs(z) > 2.0:
        return "btc_vol_gate"
    if pm > 0.50 and (rsi4h > 62 or macd4h > 80):
        return "near_itm_gate"
    if ema == 1 and rev <= -4 and sk > 55 and pm <= 0.65:
        return "rev_div_gate"
    if ve < 0.000318 and z > -0.20:
        return "vol_eff_low"
    if ema == 0 and vwap == -1 and pm < 0.60:
        return "G2_ema0_vwap-1"
    if ema == 0 and trend == -1:
        return "neutral_ema_g3"
    return None

def _yes_late_gates(row, pm, ve, tau, sk, z, offset):
    """Returns name of first late gate that fires, or None."""
    ema   = _get(row, "ema_stack_bias",   0.0)
    rev   = _get(row, "composite_rev",    0.0)
    trend = _get(row, "composite_trend",  0.0)
    vpin  = _get(row, "vpin_score",       0.0)
    estr  = _get(row, "ema_stretch_score", 0.0)
    rvol  = _get(row, "rvol_1h",          1.0)
    liq_b = _get(row, "liq_bias",         0.0)
    adx   = _get(row, "adx_1h",          float("nan"))
    garch = _get(row, "garch_ratio",      float("nan"))
    markov = _gets(row, "markov_regime_daily", "")

    if markov == "Sideways":
        return "markov_sideways"
    if not math.isnan(garch) and garch > 1.5 and not (pm >= 0.80 and tau < 45):
        return "garch_highvol"
    if not math.isnan(adx) and 20.0 <= adx < 40.0 and ema != -1:
        return "btc_adx_gate"
    if pm < 0.35 and ema == 0:
        return "btc_deepno_neutral"
    if 0.50 <= pm < 0.60 and ema in (0, 1):
        return "near_atm_ema"
    if 0.55 <= pm < 0.60 and trend >= 3:
        return "strong_trend_nearatm"
    if ema == -1 and rev <= 3 and sk >= 35 and not (vpin == 1 or estr == 1):
        return "beardrift_arm1"
    if ema == -1 and rev <= 3 and sk < 25 and offset > 0:
        return "beardrift_arm2"
    if rvol < 0.80 and not (vpin == 1 or liq_b == 1):
        return "rvol_gate"
    return None

def _no_gates(row, pm, sk, z, offset):
    """Returns name of first NO gate that fires, or None."""
    rev      = _get(row, "composite_rev",    0.0)
    trend    = _get(row, "composite_trend",  0.0)
    fund     = _get(row, "funding_bias",     0.0)
    vwap     = _get(row, "vwap_stretch_score", 0.0)
    comp_p_up= _get(row, "composite_p_up",  0.5)
    vol_s    = _get(row, "vol_score",        0.0)
    ema      = _get(row, "ema_stack_bias",   0.0)
    markov   = _gets(row, "markov_regime_daily", "")

    if markov == "Sideways" and pm > 0.39:
        return "markov_sideways_no"
    if pm > 0.70 and rev >= 0:
        return "btc_highpm_no"
    if pm >= 0.20 and (comp_p_up <= 0.36 or comp_p_up >= 0.50) and not (vwap == 1 or vol_s == 1):
        return "btc_nopup_gate"
    if sk < 20.0 and not ((trend <= -3 and fund == -1) or vwap == 1):
        return "btc_stoch_no"
    if abs(z) < 0.30 and not (trend <= -3 and fund == -1):
        return "btc_no_z_gate"
    if offset < 0 and abs(z) > 2.0:
        return "btc_no_vol_gate"
    if pm >= 0.65 and ema == 1 and vwap <= -2:
        return "btc_no_wrongdir"
    return None


# ── attribution ──────────────────────────────────────────────────────────────

def attribute_gates(df):
    records = []
    for _, row in df.iterrows():
        spot   = float(row["spot"]);  strike = float(row["strike"])
        pm     = float(row["p_market"]); ry = int(row["resolved_yes"])
        tau    = float(row["tau_minutes"]); ve = float(row["vol_eff"])
        sk     = _get(row, "stoch_k",  50.0)
        pup    = _get(row, "p_up_v2_backfilled", 0.5)
        rvol_inv = _get(row, "rvol_inv", 1.0)

        tau_h  = max(tau / 60.0, 1 / 60)
        sigma  = ve * math.sqrt(tau_h)
        z      = math.log(strike / spot) / sigma if sigma > 0 else 0.0
        offset = (strike - spot) / spot
        fee    = FEE_RATE * min(pm, 1 - pm)
        pup_z  = float(norm.ppf(max(0.01, min(0.99, pup))))

        p_yes_pure = p_logn(spot, strike, ve, tau, 0.0)
        z_drift    = pup_z * rvol_inv * 0.3 * math.sqrt(tau_h)
        p_yes_live = p_logn(spot, strike, ve, tau, z_drift)

        stoch_bounce_yes = sk < 17.0 and pm < 0.60
        stoch_bounce_no  = sk > 83.0 and pm > 0.40

        # ── YES attribution ──────────────────────────────────────────────
        if stoch_bounce_yes:
            yes_edge = p_yes_pure - pm - fee
            p_used   = p_yes_pure
        else:
            yes_edge = p_yes_live - pm - fee
            p_used   = p_yes_live

        if yes_edge > MIN_EDGE:
            n = kelly_n(yes_edge, pm)
            if n >= 0.01:
                pnl = calc_pnl("yes", pm, n, ry)
                won = pnl > 0

                blocking_gate = None
                if not stoch_bounce_yes:
                    blocking_gate = _yes_early_gates(row, pm, ve, tau, sk, z, offset)
                if blocking_gate is None:
                    blocking_gate = _yes_late_gates(row, pm, ve, tau, sk, z, offset)

                if blocking_gate:
                    records.append({
                        "gate":     blocking_gate,
                        "side":     "yes",
                        "pm":       pm,
                        "pnl":      pnl,
                        "won":      won,
                        "edge":     yes_edge,
                    })

        # ── NO attribution ───────────────────────────────────────────────
        p_no_model = p_yes_pure if stoch_bounce_no else p_yes_live
        no_edge = pm - p_no_model - fee

        if no_edge > MIN_EDGE:
            n = kelly_n(no_edge, 1 - pm)
            if n >= 0.01:
                pnl = calc_pnl("no", pm, n, ry)
                won = pnl > 0

                blocking_gate = _no_gates(row, pm, sk, z, offset)
                if blocking_gate:
                    records.append({
                        "gate":     blocking_gate,
                        "side":     "no",
                        "pm":       pm,
                        "pnl":      pnl,
                        "won":      won,
                        "edge":     no_edge,
                    })

    return pd.DataFrame(records)


def run():
    print("Loading scan archive (backfilled)...")
    df = pd.read_csv(RES_DIR / "scan_archive_backfilled.csv", low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    df = df[df["resolved_yes"].notna() & (df["resolved_yes"].astype(str).str.strip() != "")].copy()

    for col in ["spot", "strike", "p_market", "tau_minutes", "vol_eff", "stoch_k",
                "resolved_yes", "p_up_v2_backfilled", "rvol_inv", "garch_ratio",
                "adx_1h", "composite_p_up", "ema_stack_bias", "composite_rev",
                "composite_trend", "vwap_stretch_score", "rvol_1h", "vpin_score",
                "ema_stretch_score", "liq_bias", "vol_score", "funding_bias"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["spot", "strike", "p_market", "tau_minutes", "vol_eff", "resolved_yes"])
    df = df[
        (df["p_market"] >= PM_MIN) & (df["p_market"] <= PM_MAX) &
        (df["tau_minutes"] >= TAU_MIN) & (df["tau_minutes"] <= TAU_MAX)
    ].copy()

    print(f"  {len(df):,} contracts  ({df['logged_at'].min().date()} → {df['logged_at'].max().date()})")
    print("  Attributing gates...")

    raw = attribute_gates(df)
    if raw.empty:
        print("  No blocked trades found.")
        return

    summary = (
        raw.groupby(["gate", "side"])
        .agg(
            blocked   = ("pnl", "count"),
            wins      = ("won", "sum"),
            pnl_impact= ("pnl", lambda x: -x.sum()),  # positive = gate helped (blocked losers)
            avg_pm    = ("pm",  "mean"),
            avg_edge  = ("edge","mean"),
        )
        .reset_index()
    )
    summary["wr_blocked"] = summary["wins"] / summary["blocked"]
    summary = summary.sort_values("pnl_impact", ascending=False)

    out_path = RES_DIR / f"gate_attr_scan_{date.today().strftime('%Y%m%d')}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}\n")

    # Print summary table
    print(f"{'='*78}")
    print(f"  GATE ATTRIBUTION  (Live_approx, first-blocking gate per would-be trade)")
    print(f"  pnl_impact > 0 = gate HELPED (blocked losers)")
    print(f"  pnl_impact < 0 = gate HURT   (blocked winners)")
    print(f"{'='*78}")
    print(f"  {'Gate':<28} {'side':>4} {'blocked':>8} {'WR_blk':>7} {'pnl_impact':>12} {'$/blk':>8}")
    print(f"  {'-'*72}")
    for _, r in summary.iterrows():
        per = r["pnl_impact"] / r["blocked"]
        print(f"  {r['gate']:<28} {r['side']:>4} {r['blocked']:>8,} {r['wr_blocked']:>7.1%} "
              f"  {r['pnl_impact']:>+10,.0f} {per:>+8.2f}")
    print(f"{'='*78}")
    print(f"  Total blocked P&L impact: ${summary['pnl_impact'].sum():+,.0f}")


if __name__ == "__main__":
    run()
