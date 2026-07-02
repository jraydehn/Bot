"""
simulate_scan_archive.py — Simulate live BTC model trading the scan archive.

Models:
  Live_approx      — current live model (J drift k=0.3×rvol_inv, W bounce, full gate stack)
  W_live           — current stoch bounce only (1h<17 YES / 1h>83 NO, pure lognormal)
  MT_1h17_4h40     — proposed stoch bounce reform (adds 4h<40 confirmation)
"""
import math, warnings
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


def _get(row, col, default):
    v = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
    if v is None:
        return default
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return v  # non-numeric (e.g. markov string)


def _gets(row, col, default=""):
    """Get string field from row."""
    v = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
    return default if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)


# ── gate helpers ───────────────────────────────────────────────────────────────

def yes_early_gated(row, pm, ve, tau, sk, z, offset):
    """Gates that block composite YES but are bypassed by stoch_bounce rescue."""
    ema   = _get(row, "ema_stack_bias",    0.0)
    rev   = _get(row, "composite_rev",     0.0)
    trend = _get(row, "composite_trend",   0.0)
    vwap  = _get(row, "vwap_stretch_score", 0.0)
    rsi4h = _get(row, "rsi_4h",            50.0)
    macd4h= _get(row, "macd_hist_4h",       0.0)

    # btc_vol_gate: OTM YES with |z|>2.0 (vol_factor=1.0)
    if ve > 0 and offset > 0 and abs(z) > 2.0:
        return True
    # near_itm_gate: pm>0.50 AND (rsi_4h>62 OR macd_hist_4h>80)
    if pm > 0.50 and (rsi4h > 62 or macd4h > 80):
        return True
    # rev_div_gate: ema=+1 AND rev<=-4 AND stoch>55; rescue pm>0.65
    if ema == 1 and rev <= -4 and sk > 55 and pm <= 0.65:
        return True
    # vol_eff_low: ve<0.000318 AND z_score>-0.20
    if ve < 0.000318 and z > -0.20:
        return True
    # G2: ema=0 AND vwap=-1 AND pm<0.60
    if ema == 0 and vwap == -1 and pm < 0.60:
        return True
    # G3: ema=0 AND composite_trend=-1
    if ema == 0 and trend == -1:
        return True
    return False


def yes_late_gated(row, pm, ve, tau, sk, z, offset):
    """Gates that block ALL YES including stoch_bounce rescues."""
    ema   = _get(row, "ema_stack_bias",   0.0)
    rev   = _get(row, "composite_rev",    0.0)
    trend = _get(row, "composite_trend",  0.0)
    vpin  = _get(row, "vpin_score",       0.0)
    estr  = _get(row, "ema_stretch_score", 0.0)
    rvol  = _get(row, "rvol_1h",          1.0)
    liq_b = _get(row, "liq_bias",         0.0)
    adx   = _get(row, "adx_1h",          float("nan"))
    garch = _get(row, "garch_ratio",      float("nan"))
    comp_p_up = _get(row, "composite_p_up", 0.5)
    markov = _gets(row, "markov_regime_daily", "")

    # markov_sideways_gate: block ALL YES in Bear/Sideways macro regime
    if markov == "Sideways":
        return True

    # btc_garch_highvol_yes_gate: garch_ratio>1.5; rescue pm>=0.80 AND tau<45
    if not math.isnan(garch) and garch > 1.5:
        if not (pm >= 0.80 and tau < 45):
            return True

    # btc_adx_gate: adx in [20,40); rescue ema=-1
    if not math.isnan(adx) and 20.0 <= adx < 40.0 and ema != -1:
        return True

    # btc_deepno_neutral_gate: pm<0.35 AND ema=0
    if pm < 0.35 and ema == 0:
        return True

    # near_atm_ema_gate: pm in [0.50,0.60) AND ema in {0,+1}
    if 0.50 <= pm < 0.60 and ema in (0, 1):
        return True

    # strong_trend_nearatm_gate: pm in [0.55,0.60) AND c_trend>=3
    if 0.55 <= pm < 0.60 and trend >= 3:
        return True

    # beardrift arm1: ema=-1 AND rev<=3 AND stoch>=35; rescue vpin=1 OR ema_stretch=1
    if ema == -1 and rev <= 3 and sk >= 35 and not (vpin == 1 or estr == 1):
        return True

    # beardrift arm2: ema=-1 AND rev<=3 AND stoch<25 AND OTM
    if ema == -1 and rev <= 3 and sk < 25 and offset > 0:
        return True

    # rvol_gate: rvol_1h<0.80; rescue vpin=1 OR liq_bias=1
    if rvol < 0.80 and not (vpin == 1 or liq_b == 1):
        return True

    return False


def no_gated(row, pm, sk, z, offset):
    """Return True if any live BTC NO gate blocks this bet."""
    rev      = _get(row, "composite_rev",    0.0)
    trend    = _get(row, "composite_trend",  0.0)
    fund     = _get(row, "funding_bias",     0.0)
    vwap     = _get(row, "vwap_stretch_score", 0.0)
    comp_p_up= _get(row, "composite_p_up",  0.5)
    vol_s    = _get(row, "vol_score",        0.0)
    ema      = _get(row, "ema_stack_bias",   0.0)
    markov   = _gets(row, "markov_regime_daily", "")

    # markov_sideways: block NO when pm>0.39 in Sideways; allow pm<=0.39
    if markov == "Sideways" and pm > 0.39:
        return True

    # btc_highpm_no_gate: pm>0.70 AND rev>=0
    if pm > 0.70 and rev >= 0:
        return True

    # btc_nopup_gate: (comp_p_up<=0.36 OR comp_p_up>=0.50) AND pm>=0.20;
    # rescue: stretch(vwap_stretch_score)==1 OR vol_score==1
    if pm >= 0.20 and (comp_p_up <= 0.36 or comp_p_up >= 0.50):
        if not (vwap == 1 or vol_s == 1):
            return True

    # btc_stoch_no_gate: stoch<20; rescue (trend<=-3 AND fund=-1) OR vwap_stretch=1
    if sk < 20.0:
        if not ((trend <= -3 and fund == -1) or vwap == 1):
            return True

    # btc_no_z_gate: |z|<0.30 (near-ATM NO); rescue trend<=-3 AND fund=-1
    if abs(z) < 0.30:
        if not (trend <= -3 and fund == -1):
            return True

    # btc_no_vol_gate: OTM NO (offset<0) with |z|>2.0
    if offset < 0 and abs(z) > 2.0:
        return True

    # btc_no_wrongdir_gate: pm>=0.65 AND ema=+1 AND vwap_stretch<=-2
    if pm >= 0.65 and ema == 1 and vwap <= -2:
        return True

    return False


# ── contract evaluator ────────────────────────────────────────────────────────

def evaluate_contract(row, model):
    """Return list of candidate bets for this contract under the given model."""
    spot  = float(row["spot"]);   strike = float(row["strike"])
    pm    = float(row["p_market"]); ry = int(row["resolved_yes"])
    tau   = float(row["tau_minutes"]); ve = float(row["vol_eff"])
    sk    = _get(row, "stoch_k",  50.0)
    pup   = _get(row, "p_up_v2_backfilled", 0.5)
    sk4h  = _get(row, "stoch_k_4h", 50.0)

    tau_h  = max(tau / 60.0, 1 / 60)
    sigma  = ve * math.sqrt(tau_h)
    z      = math.log(strike / spot) / sigma if sigma > 0 else 0.0
    offset = (strike - spot) / spot
    fee    = FEE_RATE * min(pm, 1 - pm)
    pup_z  = float(norm.ppf(max(0.01, min(0.99, pup))))

    p_yes_pure = p_logn(spot, strike, ve, tau, 0.0)

    bets = []

    if model == "Live_approx":
        # Correct J drift: k=0.3 × rvol_inv (matches live system)
        rvol_inv = _get(row, "rvol_inv", 1.0)
        z_drift_live = pup_z * rvol_inv * 0.3 * math.sqrt(tau_h)
        p_yes_live   = p_logn(spot, strike, ve, tau, z_drift_live)

        stoch_bounce_yes = (sk < 17.0 and pm < 0.60)
        stoch_bounce_no  = (sk > 83.0 and pm > 0.40)

        # YES path
        if stoch_bounce_yes:
            # Pure lognormal, bypass early gates, still apply late gates
            yes_edge = p_yes_pure - pm - fee
            if yes_edge > MIN_EDGE and not yes_late_gated(row, pm, ve, tau, sk, z, offset):
                n = kelly_n(yes_edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": yes_edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})
        else:
            yes_edge = p_yes_live - pm - fee
            if (yes_edge > MIN_EDGE and pm > 0
                    and not yes_early_gated(row, pm, ve, tau, sk, z, offset)
                    and not yes_late_gated(row, pm, ve, tau, sk, z, offset)):
                n = kelly_n(yes_edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": yes_edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})

        # NO path — stoch_bounce_no uses pure lognormal; both paths go through all NO gates
        p_no_model = p_yes_pure if stoch_bounce_no else p_yes_live
        no_edge = pm - p_no_model - fee
        if (no_edge > MIN_EDGE and (1 - pm) > 0
                and not no_gated(row, pm, sk, z, offset)):
            n = kelly_n(no_edge, 1 - pm)
            if n >= 0.01:
                bets.append({"side": "no", "edge": no_edge, "pm": pm,
                             "n_cont": n, "pnl": calc_pnl("no", pm, n, ry),
                             "won": calc_pnl("no", pm, n, ry) > 0})

    elif model == "W_live":
        # Current stoch bounce: 1h<17 YES / 1h>83 NO, pure lognormal, late gates only
        if sk < 17.0 and pm < 0.60:
            edge = p_yes_pure - pm - fee
            if edge > MIN_EDGE and not yes_late_gated(row, pm, ve, tau, sk, z, offset):
                n = kelly_n(edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})
        if sk > 83.0 and pm > 0.40:
            edge = pm - p_yes_pure - fee
            if edge > MIN_EDGE and not no_gated(row, pm, sk, z, offset):
                n = kelly_n(edge, 1 - pm)
                if n >= 0.01:
                    bets.append({"side": "no", "edge": edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("no", pm, n, ry),
                                 "won": calc_pnl("no", pm, n, ry) > 0})

    elif model == "MT_1h17_4h40":
        # Proposed reform: adds stoch_4h<40 confirmation; late gates only
        if sk < 17.0 and sk4h < 40.0 and pm < 0.60:
            edge = p_yes_pure - pm - fee
            if edge > MIN_EDGE and not yes_late_gated(row, pm, ve, tau, sk, z, offset):
                n = kelly_n(edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})
        if sk > 83.0 and sk4h > 60.0 and pm > 0.40:
            edge = pm - p_yes_pure - fee
            if edge > MIN_EDGE and not no_gated(row, pm, sk, z, offset):
                n = kelly_n(edge, 1 - pm)
                if n >= 0.01:
                    bets.append({"side": "no", "edge": edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("no", pm, n, ry),
                                 "won": calc_pnl("no", pm, n, ry) > 0})

    elif model in ("flat_k140", "comp_pup_rvol"):
        # flat_k140:    z_drift = Φ⁻¹(comp_p_up) × 1.40          (pre-restoration model)
        # comp_pup_rvol: z_drift = Φ⁻¹(comp_p_up) × rvol_inv × 0.3 × √(τ/60) (current)
        comp_pup  = _get(row, "composite_p_up", 0.5)
        rvol_inv  = _get(row, "rvol_inv", 1.0)
        pup_z_c   = float(norm.ppf(max(0.01, min(0.99, comp_pup))))
        if model == "flat_k140":
            z_drift_live = pup_z_c * 1.40
        else:
            z_drift_live = pup_z_c * rvol_inv * 0.3 * math.sqrt(tau_h)
        p_yes_live = p_logn(spot, strike, ve, tau, z_drift_live)

        stoch_bounce_yes = (sk < 17.0 and pm < 0.60)
        stoch_bounce_no  = (sk > 83.0 and pm > 0.40)

        if stoch_bounce_yes:
            yes_edge = p_yes_pure - pm - fee
            if yes_edge > MIN_EDGE and not yes_late_gated(row, pm, ve, tau, sk, z, offset):
                n = kelly_n(yes_edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": yes_edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})
        else:
            yes_edge = p_yes_live - pm - fee
            if (yes_edge > MIN_EDGE and pm > 0
                    and not yes_early_gated(row, pm, ve, tau, sk, z, offset)
                    and not yes_late_gated(row, pm, ve, tau, sk, z, offset)):
                n = kelly_n(yes_edge, pm)
                if n >= 0.01:
                    bets.append({"side": "yes", "edge": yes_edge, "pm": pm,
                                 "n_cont": n, "pnl": calc_pnl("yes", pm, n, ry),
                                 "won": calc_pnl("yes", pm, n, ry) > 0})

        p_no_model = p_yes_pure if stoch_bounce_no else p_yes_live
        no_edge = pm - p_no_model - fee
        if (no_edge > MIN_EDGE and (1 - pm) > 0
                and not no_gated(row, pm, sk, z, offset)):
            n = kelly_n(no_edge, 1 - pm)
            if n >= 0.01:
                bets.append({"side": "no", "edge": no_edge, "pm": pm,
                             "n_cont": n, "pnl": calc_pnl("no", pm, n, ry),
                             "won": calc_pnl("no", pm, n, ry) > 0})

    return bets


def simulate(df, model):
    """Run scan-by-scan simulation: take the single best-edge bet per scan cycle."""
    trades = []
    for ts, group in df.groupby("logged_at"):
        candidates = []
        for _, row in group.iterrows():
            candidates.extend(evaluate_contract(row, model))
        if not candidates:
            continue
        best = max(candidates, key=lambda x: x["edge"])
        best["logged_at"] = ts
        best["model"] = model
        trades.append(best)
    return pd.DataFrame(trades)


def print_results(dfs):
    print(f"\n{'='*72}")
    print(f"  SCAN-SIMULATION RESULTS  (best edge per scan cycle)")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Trades':>7} {'WR':>7} {'P&L':>10} {'$/trade':>8}")
    print(f"  {'-'*56}")
    for label, df in dfs.items():
        if df.empty:
            print(f"  {label:<22}  no trades")
            continue
        n = len(df); wr = df["won"].mean(); pnl = df["pnl"].sum()
        print(f"  {label:<22} {n:>7,} {wr:>7.1%} {pnl:>+10,.0f} {pnl/n:>+8.2f}")

    print(f"\n{'='*72}")
    print(f"  YES vs NO SPLIT")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'side':>4} {'n':>6} {'WR':>7} {'BE_WR':>7} {'P&L':>10} {'$/t':>7}")
    print(f"  {'-'*62}")
    for label, df in dfs.items():
        if df.empty:
            continue
        for side, g in df.groupby("side"):
            n = len(g); wr = g["won"].mean(); pnl = g["pnl"].sum()
            avg_pm  = g["pm"].mean()
            fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
            be_wr   = avg_pm + fee_avg if side == "yes" else (1 - avg_pm) + fee_avg
            print(f"  {label:<22} {side:>4} {n:>6,} {wr:>7.1%} {be_wr:>7.1%} "
                  f"{pnl:>+10,.0f} {pnl/n:>+7.2f}")

    print(f"\n{'='*72}")
    print(f"  DAILY P&L")
    print(f"{'='*72}")
    all_days = sorted(set(
        d for df in dfs.values() if not df.empty
        for d in df["logged_at"].dt.date.unique()
    ))
    header = f"  {'Date':<12}" + "".join(f"  {m:<24}" for m in dfs)
    print(header)
    print(f"  {'-'*(14 + 26*len(dfs))}")
    for day in all_days:
        row_str = f"  {str(day):<12}"
        for label, df in dfs.items():
            sub = df[df["logged_at"].dt.date == day] if not df.empty else pd.DataFrame()
            row_str += (f"  {'—':>24}" if sub.empty
                        else f"  {sub['pnl'].sum():>+8,.0f} ({len(sub):>4} trades)  ")
        print(row_str)
    totals = f"  {'TOTAL':<12}"
    for label, df in dfs.items():
        totals += f"  {df['pnl'].sum():>+8,.0f} ({len(df):>4} trades)  "
    print(f"  {'-'*(14 + 26*len(dfs))}")
    print(totals)


def run():
    print("Loading scan archive (backfilled)...")
    df = pd.read_csv(RES_DIR / "scan_archive_backfilled.csv", low_memory=False)
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    df = df[df["resolved_yes"].notna() & (df["resolved_yes"].astype(str).str.strip() != "")].copy()

    for col in ["spot", "strike", "p_market", "tau_minutes", "vol_eff", "stoch_k",
                "resolved_yes", "p_up_v2_backfilled", "stoch_k_4h",
                "rvol_inv", "garch_ratio", "adx_1h", "composite_p_up",
                "ema_stack_bias", "composite_rev", "composite_trend",
                "vwap_stretch_score", "rvol_1h", "vpin_score",
                "ema_stretch_score", "liq_bias", "vol_score", "funding_bias"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["spot", "strike", "p_market", "tau_minutes", "vol_eff", "resolved_yes"])
    df = df[
        (df["p_market"] >= PM_MIN) & (df["p_market"] <= PM_MAX) &
        (df["tau_minutes"] >= TAU_MIN) & (df["tau_minutes"] <= TAU_MAX)
    ].copy()

    n_scans = df["logged_at"].nunique()
    print(f"  {len(df):,} contracts across {n_scans:,} scan cycles  "
          f"({df['logged_at'].min().date()} → {df['logged_at'].max().date()})")

    models = ["flat_k140", "comp_pup_rvol", "Live_approx"]
    dfs = {}
    for m in models:
        print(f"  Simulating {m}...")
        dfs[m] = simulate(df, m)

    print_results(dfs)


if __name__ == "__main__":
    run()
