"""
backtest_scan_archive.py — Run the R_dcJ87_sk78 model against the full
scan archive opportunity set (all evaluated contracts, not just taken ones).

Model stack (exact match to backtest_real_data.py):
  1. Pure lognormal base
  2. p_up_v2 directional gate  (YES when p_up_v2 > 0.50, NO when < 0.50)
  3. J vol-drift                (Φ⁻¹(p_up_v2) × vol_amp × 0.1 × √τ)
     vol_implied_kalshi derived from p_market via inverse lognormal
  4. Donchian gate              (block YES when dc_4h_n20_pos > 0.87)
  5. stoch_k gate               (block YES when stoch_k > 78)

Data source: results/btc_scan_archive.csv + results/scan_archive_dc.csv
"""
import math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
RES_DIR = ROOT / "results"

BANKROLL   = 1_000.0
KELLY_MULT = 0.30
KELLY_CAP  = 0.06
FEE_RATE   = 0.07
MIN_EDGE   = 0.005

DC_THRESH    = 0.87
STOCH_THRESH = 78.0
J_K          = 0.1


def sigma_tau(vol_eff: float, tau_minutes: float) -> float:
    tau_h = max(tau_minutes / 60.0, 1 / 60)
    return vol_eff * math.sqrt(tau_h)


def p_lognormal(spot: float, strike: float, sig_t: float, z_drift: float = 0.0) -> float:
    if sig_t <= 0:
        return 1.0 if spot > strike else 0.0
    z = math.log(strike / spot) / sig_t - z_drift
    return float(norm.sf(z))


def vol_implied_from_market(spot: float, strike: float, p_market: float,
                             tau_minutes: float) -> float:
    """Back-compute implied vol from Kalshi's market price using the lognormal formula."""
    tau_h = max(tau_minutes / 60.0, 1 / 60)
    pm_clipped = max(0.01, min(0.99, p_market))
    d1 = float(norm.ppf(pm_clipped))   # Φ⁻¹(p_yes) = -log(K/S)/sigma_tau
    if abs(d1) < 1e-6:
        return float("nan")
    log_ks = math.log(strike / spot) if spot > 0 and strike > 0 else float("nan")
    if math.isnan(log_ks):
        return float("nan")
    sigma_tau_val = -log_ks / d1
    if sigma_tau_val <= 0:
        return float("nan")
    return sigma_tau_val / math.sqrt(tau_h)


def kelly_contracts(edge: float, pm_risk: float) -> float:
    if pm_risk <= 0:
        return 0.0
    return min(edge / pm_risk * KELLY_MULT, KELLY_CAP) * BANKROLL / pm_risk


def compute_pnl(side: str, p_market: float, n_cont: float, resolved_yes: int) -> float:
    fee = FEE_RATE * min(p_market, 1 - p_market)
    if side == "yes":
        return n_cont * (1 - p_market - fee) if resolved_yes == 1 else -n_cont * (p_market + fee)
    else:
        return n_cont * (p_market - fee) if resolved_yes == 0 else -n_cont * (1 - p_market + fee)


def evaluate_breakout(row, max_pm_yes: float, stoch_cap: float,
                      symmetric_no: bool = False):
    """Breakout / momentum model for asymmetric YES upside.

    YES: dc_4h_n20_break == +1, p_market < max_pm_yes, stoch_k < stoch_cap
    NO:  dc_4h_n20_break == -1, p_market > (1 - max_pm_yes), stoch_k > (100 - stoch_cap)
         (only when symmetric_no=True)
    Pure lognormal — no J drift.
    """
    try:
        pm       = float(row["p_market"])
        spot     = float(row["spot"])
        strike   = float(row["strike"])
        tau_min  = float(row["tau_minutes"])
        vol_eff  = float(row["vol_eff"])
        stoch_k  = float(row["stoch_k"])
        dc_break = float(row["dc_4h_n20_break"])
        resolved = int(row["resolved_yes"])
    except (TypeError, ValueError):
        return None

    if any(math.isnan(x) for x in [pm, spot, strike, tau_min, vol_eff, stoch_k, dc_break]):
        return None

    sig = sigma_tau(vol_eff, tau_min)
    if sig <= 0:
        return None

    p_model = p_lognormal(spot, strike, sig, 0.0)
    fee = FEE_RATE * min(pm, 1 - pm)

    for side in ("yes", "no"):
        if side == "yes":
            if dc_break != 1.0:
                continue
            if pm >= max_pm_yes:
                continue
            if stoch_k >= stoch_cap:
                continue
            edge    = p_model - pm - fee
            pm_risk = pm
        else:
            if not symmetric_no:
                continue
            if dc_break != -1.0:
                continue
            if pm <= (1 - max_pm_yes):
                continue
            if stoch_k <= (100 - stoch_cap):
                continue
            edge    = pm - p_model - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE or pm_risk <= 0:
            continue

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            continue

        pnl = compute_pnl(side, pm, n, resolved)
        return {
            "side": side, "pm": pm, "p_model": p_model,
            "edge": edge, "n_cont": n, "pnl": pnl, "won": pnl > 0,
            "logged_at": row["logged_at"],
        }
    return None


def evaluate_row(row):
    """Evaluate a single scan archive row under R_dcJ87_sk78.

    Returns a result dict if a trade is taken, else None.
    Evaluates YES first, then NO — only one side can have edge.
    """
    try:
        pm        = float(row["p_market"])
        spot      = float(row["spot"])
        strike    = float(row["strike"])
        tau_min   = float(row["tau_minutes"])
        vol_eff   = float(row["vol_eff"])
        pup       = float(row["p_up_v2"])
        stoch_k   = float(row["stoch_k"])
        dc_pos    = float(row["dc_4h_n20_pos"])
        resolved  = int(row["resolved_yes"])
    except (TypeError, ValueError):
        return None

    if any(math.isnan(x) for x in [pm, spot, strike, tau_min, vol_eff, pup, stoch_k, dc_pos]):
        return None

    sig = sigma_tau(vol_eff, tau_min)
    if sig <= 0:
        return None

    # Derive vol_implied_kalshi from market price
    vol_imp = vol_implied_from_market(spot, strike, pm, tau_min)
    if math.isnan(vol_imp) or vol_imp <= 0:
        return None

    # J vol-drift
    tau_h   = max(tau_min / 60.0, 1 / 60)
    pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
    vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
    z_drift = pup_z * vol_amp * J_K * math.sqrt(tau_h)
    p_model = p_lognormal(spot, strike, sig, z_drift)

    fee = FEE_RATE * min(pm, 1 - pm)

    for side in ("yes", "no"):
        # p_up_v2 gate
        if side == "yes" and pup <= 0.50:
            continue
        if side == "no"  and pup >= 0.50:
            continue

        # Donchian gate
        if side == "yes" and dc_pos > DC_THRESH:
            continue
        if side == "no"  and dc_pos < (1 - DC_THRESH):
            continue

        # Stoch_k gate
        if side == "yes" and stoch_k > STOCH_THRESH:
            continue
        if side == "no"  and stoch_k < (100 - STOCH_THRESH):
            continue

        # Edge check
        if side == "yes":
            edge    = p_model - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_model - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE or pm_risk <= 0:
            continue

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            continue

        pnl = compute_pnl(side, pm, n, resolved)
        return {
            "side": side, "pm": pm, "p_model": p_model,
            "edge": edge, "n_cont": n, "pnl": pnl, "won": pnl > 0,
            "logged_at": row["logged_at"],
            "month": pd.Period(str(row["logged_at"])[:7], "M"),
        }

    return None


def run():
    print("Loading scan archive...")
    scan = pd.read_csv(RES_DIR / "btc_scan_archive.csv", low_memory=False)
    scan["logged_at"] = pd.to_datetime(scan["logged_at"], errors="coerce", utc=True)
    scan = scan.dropna(subset=["logged_at"]).copy()

    for col in ["p_market", "spot", "strike", "tau_minutes", "vol_eff", "stoch_k", "resolved_yes"]:
        scan[col] = pd.to_numeric(scan[col], errors="coerce")
    scan = scan.dropna(subset=["p_market", "spot", "strike", "tau_minutes",
                                "vol_eff", "stoch_k", "resolved_yes"])

    n_before = len(scan)
    scan = scan.sort_values("logged_at")
    print(f"  {n_before:,} total rows  ({scan['logged_at'].min().date()} → {scan['logged_at'].max().date()})")

    # Merge backfilled p_up_v2 + Donchian
    bf_path = RES_DIR / "scan_archive_backfilled.csv"
    if not bf_path.exists():
        print("  scan_archive_backfilled.csv not found — run backfill_p_up_v2.py first")
        return
    bf = pd.read_csv(bf_path)
    bf["logged_at"] = pd.to_datetime(bf["logged_at"], errors="coerce", utc=True)
    bf["p_up_v2_backfilled"] = pd.to_numeric(bf["p_up_v2_backfilled"], errors="coerce")
    for col in ["dc_4h_n20_pos", "dc_4h_n20_break"]:
        bf[col] = pd.to_numeric(bf[col], errors="coerce")
    merge_cols = ["logged_at", "contract_ticker", "p_up_v2_backfilled",
                  "dc_4h_n20_pos", "dc_4h_n20_break"]
    scan = scan.merge(bf[[c for c in merge_cols if c in bf.columns]],
                      on=["logged_at", "contract_ticker"], how="left")
    scan["p_up_v2"] = scan["p_up_v2_backfilled"]

    n_pup   = scan["p_up_v2"].notna().sum()
    n_dc    = scan["dc_4h_n20_pos"].notna().sum()
    n_break = scan["dc_4h_n20_break"].notna().sum()
    print(f"  p_up_v2 coverage      : {n_pup:,} / {len(scan):,} ({n_pup/len(scan):.0%})")
    print(f"  dc_4h_n20_pos coverage : {n_dc:,} / {len(scan):,} ({n_dc/len(scan):.0%})")
    print(f"  dc_4h_n20_break coverage: {n_break:,} / {len(scan):,} ({n_break/len(scan):.0%})")

    # R_dcJ87_sk78: first evaluation per contract
    scan_first = scan.drop_duplicates(subset=["contract_ticker"], keep="first")

    # Breakout models: first scan where dc_break fires (+1 for YES, -1 for NO)
    # We merge dc_4h_n20_break from backfilled before dedup so it's available
    scan_brk_yes = (scan[pd.to_numeric(scan["dc_4h_n20_break"], errors="coerce") == 1.0]
                    .drop_duplicates(subset=["contract_ticker"], keep="first"))
    scan_brk_no  = (scan[pd.to_numeric(scan["dc_4h_n20_break"], errors="coerce") == -1.0]
                    .drop_duplicates(subset=["contract_ticker"], keep="first"))
    # Symmetric: union of first YES-break and first NO-break per contract
    scan_brk_sym = pd.concat([scan_brk_yes, scan_brk_no]).drop_duplicates(
                        subset=["contract_ticker", "dc_4h_n20_break"], keep="first")

    print(f"  Contracts with upside break (+1): {len(scan_brk_yes):,}")
    print(f"  Contracts with downside break (-1): {len(scan_brk_no):,}")

    all_results = {}

    # R model on first-evaluation universe
    r_results = []
    for _, row in scan_first.iterrows():
        res = evaluate_row(row)
        if res:
            res["model"] = "R_dcJ87_sk78"
            r_results.append(res)
    all_results["R_dcJ87_sk78"] = r_results

    # Breakout YES-only models
    for name, max_pm, stoch_cap in [
        ("V_brk_pm50_sk80", 0.50, 80.0),
        ("V_brk_pm50_sk65", 0.50, 65.0),
        ("V_brk_pm40_sk80", 0.40, 80.0),
    ]:
        results = []
        for _, row in scan_brk_yes.iterrows():
            res = evaluate_breakout(row, max_pm, stoch_cap)
            if res:
                res["model"] = name
                results.append(res)
        all_results[name] = results

    # Symmetric breakout model
    sym_results = []
    for _, row in scan_brk_sym.iterrows():
        res = evaluate_breakout(row, 0.50, 80.0, symmetric_no=True)
        if res:
            res["model"] = "V_brk_sym_pm50"
            sym_results.append(res)
    all_results["V_brk_sym_pm50"] = sym_results

    print(f"\n{'='*60}")
    print(f"  SCAN ARCHIVE BACKTEST  ({scan['logged_at'].min().date()} → {scan['logged_at'].max().date()})")
    print(f"  {len(scan):,} unique contracts")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Trades':>7} {'WR':>7} {'P&L':>10} {'$/trade':>8}")
    print(f"  {'-'*58}")

    for name, results in all_results.items():
        if not results:
            print(f"  {name:<22}  no trades")
            continue
        df = pd.DataFrame(results)
        n   = len(df)
        wr  = df["won"].mean()
        pnl = df["pnl"].sum()
        print(f"  {name:<22} {n:>7,} {wr:>7.1%} {pnl:>+10,.0f} {pnl/n:>+8.2f}")

        print(f"    YES vs NO:")
        for side, g in df.groupby("side"):
            be = g["pm"].mean() if side == "yes" else 1 - g["pm"].mean()
            print(f"      {side:3s}  n={len(g):,}  WR={g['won'].mean():.1%}  "
                  f"breakeven={be:.1%}  P&L=${g['pnl'].sum():+,.0f}  "
                  f"avg_pm={g['pm'].mean():.2f}")

    print(f"\n  Daily P&L — R_dcJ87_sk78 vs V_brk_pm50_sk80:")
    r_df = pd.DataFrame(all_results.get("R_dcJ87_sk78", []))
    v_df = pd.DataFrame(all_results.get("V_brk_pm50_sk80", []))
    if not r_df.empty:
        r_df["date"] = pd.to_datetime(r_df["logged_at"]).dt.date
    if not v_df.empty:
        v_df["date"] = pd.to_datetime(v_df["logged_at"]).dt.date
    all_dates = sorted(set(
        (r_df["date"].unique().tolist() if not r_df.empty else []) +
        (v_df["date"].unique().tolist() if not v_df.empty else [])
    ))
    for d in all_dates:
        r_g = r_df[r_df["date"] == d] if not r_df.empty else pd.DataFrame()
        v_g = v_df[v_df["date"] == d] if not v_df.empty else pd.DataFrame()
        r_str = f"R: {r_g['pnl'].sum():+,.0f} ({len(r_g)})" if not r_g.empty else "R: —"
        v_str = f"V: {v_g['pnl'].sum():+,.0f} ({len(v_g)})" if not v_g.empty else "V: —"
        print(f"    {d}  {r_str:<22}  {v_str}")


if __name__ == "__main__":
    run()
