"""
backtest_real_data.py — Backtest multiple model formulations against real live
Kalshi trade data (March 23 → May 26, 2026).

Data source: paper_trades + paper_trades_archive_*.csv
Real p_market from Kalshi, real resolution from Binance BRTI.

Models tested:
  A  As-run          — use logged p_yes_model (whatever era was live)
  B  Pure lognormal  — no drift, no composite; pure vol-based p_model
  C  Composite drift — z = Φ⁻¹(composite_p_up) × 0.3 × √(τ/60)  [Apr 7+]
  D  High threshold  — as-run but only take trades with |net_edge| > 5%
  G  p_up_v2 gate    — pure lognormal + p_up_v2 directional gate (backfilled)
"""
import glob, math, warnings
from pathlib import Path
from typing import Optional

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
MIN_EDGE_A = 0.005   # as-run minimum (~0.5%)
MIN_EDGE_D = 0.05    # high-threshold model (5%)


# ── load and combine all archives ────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES_DIR / "paper_trades_archive_*.csv")))
    files += [str(RES_DIR / "paper_trades.csv")]

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["_source"] = Path(f).name
            frames.append(df)
        except Exception as e:
            print(f"  skip {f}: {e}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["logged_at"] = pd.to_datetime(combined["logged_at"], format="mixed", utc=True)
    combined = (combined
                .sort_values("logged_at")
                .drop_duplicates(subset=["contract_ticker", "logged_at", "side"], keep="last"))

    trades = combined[combined["decision"] == "trade"].copy()
    trades = trades[trades["resolved_yes"].notna()].copy()

    # Ensure numeric types
    for col in ["spot", "strike", "tau_minutes", "vol_eff", "p_market",
                "p_yes_model", "net_edge", "resolved_yes"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(subset=["spot", "strike", "tau_minutes",
                                    "vol_eff", "p_market", "resolved_yes"])

    # Merge backfilled p_up_v2
    pup_path = RES_DIR / "p_up_v2_backfilled.csv"
    if pup_path.exists():
        pup = pd.read_csv(pup_path)
        pup["logged_at"] = pd.to_datetime(pup["logged_at"], utc=True)
        key_cols = ["contract_ticker", "logged_at", "side"]
        extra = [c for c in pup.columns if c not in key_cols]
        trades = trades.merge(pup[key_cols + extra], on=key_cols, how="left")
        n_matched = trades["p_up_v2_backfilled"].notna().sum()
        print(f"  p_up_v2_backfilled merged: {n_matched:,} / {len(trades):,} trades")

    print(f"Loaded {len(trades):,} resolved trades  "
          f"({trades['logged_at'].min().date()} → {trades['logged_at'].max().date()})")
    return trades.reset_index(drop=True)


# ── core pricing ─────────────────────────────────────────────────────────────

def sigma_tau(row) -> float:
    """Annualised vol scaled to contract tau (in hours)."""
    tau_h = max(float(row["tau_minutes"]) / 60.0, 1/60)
    return float(row["vol_eff"]) * math.sqrt(tau_h)


def p_lognormal(spot, strike, sig_t, z_drift=0.0) -> float:
    if sig_t <= 0:
        return 1.0 if spot > strike else 0.0
    z = math.log(strike / spot) / sig_t - z_drift
    return float(norm.sf(z))


def kelly_contracts(edge, pm_risk) -> float:
    if pm_risk <= 0:
        return 0.0
    k = min(edge / pm_risk * KELLY_MULT, KELLY_CAP)
    return k * BANKROLL / pm_risk


def compute_pnl(side, p_market, n_cont, resolved_yes) -> float:
    fee = FEE_RATE * min(p_market, 1 - p_market)
    if side == "yes":
        return (n_cont * (1 - p_market - fee) if resolved_yes == 1
                else -n_cont * (p_market + fee))
    else:
        return (n_cont * (p_market - fee) if resolved_yes == 0
                else -n_cont * (1 - p_market + fee))


# ── per-row model evaluations ─────────────────────────────────────────────────

def eval_model_A(row) -> dict:
    """As-run: logged p_yes_model."""
    pm   = float(row["p_market"])
    side = str(row["side"])
    sig  = sigma_tau(row)
    p_m  = pd.to_numeric(row.get("p_yes_model"), errors="coerce")

    if pd.isna(p_m) or sig <= 0:
        return None

    fee  = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        edge    = p_m - pm - fee
        pm_risk = pm
    else:
        edge    = pm - p_m - fee
        pm_risk = 1 - pm

    if edge <= MIN_EDGE_A or pm_risk <= 0:
        return None

    n = kelly_contracts(edge, pm_risk)
    if n < 0.01:
        return None
    pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
    return {"model": "A_asrun", "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": (pnl > 0)}


def eval_model_B(row) -> dict:
    """Pure lognormal — no drift."""
    pm   = float(row["p_market"])
    side = str(row["side"])
    sig  = sigma_tau(row)

    if sig <= 0:
        return None

    spot   = float(row["spot"])
    strike = float(row["strike"])
    p_m    = p_lognormal(spot, strike, sig, 0.0)

    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        edge    = p_m - pm - fee
        pm_risk = pm
    else:
        edge    = pm - p_m - fee
        pm_risk = 1 - pm

    if edge <= MIN_EDGE_A or pm_risk <= 0:
        return None

    n = kelly_contracts(edge, pm_risk)
    if n < 0.01:
        return None
    pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
    return {"model": "B_purelogN", "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": (pnl > 0)}


def eval_model_C(row) -> dict:
    """Composite drift: z = Φ⁻¹(composite_p_up) × 0.3 × √(τ/60)."""
    comp_pup = pd.to_numeric(row.get("composite_p_up"), errors="coerce")
    if pd.isna(comp_pup):
        return None   # not available for this row

    pm   = float(row["p_market"])
    side = str(row["side"])
    sig  = sigma_tau(row)
    if sig <= 0:
        return None

    spot    = float(row["spot"])
    strike  = float(row["strike"])
    tau_h   = max(float(row["tau_minutes"]) / 60.0, 1/60)
    pup_z   = float(norm.ppf(max(0.01, min(0.99, comp_pup))))
    z_drift = pup_z * 0.3 * math.sqrt(tau_h)
    p_m     = p_lognormal(spot, strike, sig, z_drift)

    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        edge    = p_m - pm - fee
        pm_risk = pm
    else:
        edge    = pm - p_m - fee
        pm_risk = 1 - pm

    if edge <= MIN_EDGE_A or pm_risk <= 0:
        return None

    n = kelly_contracts(edge, pm_risk)
    if n < 0.01:
        return None
    pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
    return {"model": "C_composite", "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": (pnl > 0)}


def eval_model_D(row) -> dict:
    """High-threshold: as-run p_yes_model but only edge > 5%."""
    pm   = float(row["p_market"])
    side = str(row["side"])
    p_m  = pd.to_numeric(row.get("p_yes_model"), errors="coerce")
    sig  = sigma_tau(row)

    if pd.isna(p_m) or sig <= 0:
        return None

    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        edge    = p_m - pm - fee
        pm_risk = pm
    else:
        edge    = pm - p_m - fee
        pm_risk = 1 - pm

    if edge <= MIN_EDGE_D or pm_risk <= 0:
        return None

    n = kelly_contracts(edge, pm_risk)
    if n < 0.01:
        return None
    pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
    return {"model": "D_highthresh", "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": (pnl > 0)}


def eval_model_G(row) -> dict:
    """Pure lognormal + p_up_v2 directional gate.
    YES only when p_up_v2_backfilled > 0.50; NO only when < 0.50."""
    pup = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
    if pd.isna(pup):
        return None

    pm   = float(row["p_market"])
    side = str(row["side"])
    sig  = sigma_tau(row)
    if sig <= 0:
        return None

    # Gate: directional alignment required
    if side == "yes" and pup <= 0.50:
        return None
    if side == "no" and pup >= 0.50:
        return None

    spot   = float(row["spot"])
    strike = float(row["strike"])
    p_m    = p_lognormal(spot, strike, sig, 0.0)

    fee = FEE_RATE * min(pm, 1 - pm)
    if side == "yes":
        edge    = p_m - pm - fee
        pm_risk = pm
    else:
        edge    = pm - p_m - fee
        pm_risk = 1 - pm

    if edge <= MIN_EDGE_A or pm_risk <= 0:
        return None

    n = kelly_contracts(edge, pm_risk)
    if n < 0.01:
        return None
    pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
    return {"model": "G_pup2gate", "side": side, "pm": pm, "edge": edge,
            "n_cont": n, "pnl": pnl, "won": (pnl > 0)}


def make_drift_model(k: float):
    """Factory: pure lognormal with z_drift = Φ⁻¹(p_up_v2) × k × √(τ/60)."""
    def eval_fn(row) -> dict:
        pup = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        if pd.isna(pup):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        z_drift = pup_z * k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        label = f"H_drift_k{int(k*10):02d}"
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_rvol_drift_model(k: float):
    """Model I: z_drift = Φ⁻¹(p_up_v2) × rvol_inv × k × √(τ/60) + p_up_v2 gate."""
    def eval_fn(row) -> dict:
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        rvol_inv = pd.to_numeric(row.get("rvol_inv_backfilled"), errors="coerce")
        if pd.isna(pup) or pd.isna(rvol_inv):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes" and pup <= 0.50:
            return None
        if side == "no" and pup >= 0.50:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        z_drift = pup_z * float(rvol_inv) * k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        label = f"I_rvol_k{int(k*10):02d}"
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_impvol_drift_model(k: float):
    """Model J: z_drift = Φ⁻¹(p_up_v2) × (vol_eff/vol_implied_kalshi) × k × √(τ/60) + p_up_v2 gate."""
    def eval_fn(row) -> dict:
        pup      = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff  = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp  = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes" and pup <= 0.50:
            return None
        if side == "no" and pup >= 0.50:
            return None

        spot     = float(row["spot"])
        strike   = float(row["strike"])
        tau_h    = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z    = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp  = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift  = pup_z * vol_amp * k * math.sqrt(tau_h)
        p_m      = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        label = f"J_impvol_k{int(k*10):02d}"
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_dc_drift_model(dc_col: str, k: float, label: str):
    """Lognormal with Donchian position as drift.
    z_drift = (0.5 - dc_pos) * k * sqrt(tau_h)
    High channel position → negative drift → bearish (mean-reversion).
    """
    def eval_fn(row) -> dict:
        dc_pos = pd.to_numeric(row.get(dc_col), errors="coerce")
        if pd.isna(dc_pos):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        z_drift = (0.5 - float(dc_pos)) * k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_dc_gate_model(dc_col: str, yes_thresh: float, no_thresh: float,
                       label: str, use_j: bool = False, j_k: float = 0.1):
    """Lognormal + Donchian position gate (and optionally J vol-drift on top).

    Gate logic (mean-reversion):
      YES blocked when dc_pos > yes_thresh  (near top of channel → overbought)
      NO  blocked when dc_pos < no_thresh   (near bottom of channel → oversold)

    use_j=True stacks the J model's vol_eff/vol_implied drift + p_up_v2 gate.
    """
    def eval_fn(row) -> dict:
        dc_pos = pd.to_numeric(row.get(dc_col), errors="coerce")
        if pd.isna(dc_pos):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Donchian gate
        if side == "yes" and float(dc_pos) > yes_thresh:
            return None
        if side == "no"  and float(dc_pos) < no_thresh:
            return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        tau_h  = max(float(row["tau_minutes"]) / 60.0, 1 / 60)

        if use_j:
            pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
            vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
            vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
            if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
                return None
            if side == "yes" and pup <= 0.50:
                return None
            if side == "no"  and pup >= 0.50:
                return None
            pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
            vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
            z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
            p_m     = p_lognormal(spot, strike, sig, z_drift)
        else:
            p_m = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_comprev_gate_model(rev_thresh: float, stoch_thresh: float, dc_thresh: float,
                            label: str, j_k: float = 0.1):
    """O_dcJ + stoch_k + composite_rev gate.

    composite_rev > 0 = bullish reversion expected; < 0 = bearish.
    Gate: block YES when composite_rev < -rev_thresh (bearish reversion)
          block NO  when composite_rev >  rev_thresh (bullish reversion)
    rev_thresh=0 blocks any non-neutral reading.
    """
    dc_col = "dc_4h_n20_pos"

    def eval_fn(row) -> dict:
        dc_pos  = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        comp_rev = pd.to_numeric(row.get("composite_rev"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k) or pd.isna(comp_rev):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Donchian gate
        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None

        # Stoch gate
        if side == "yes" and float(stoch_k) > stoch_thresh:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_thresh):
            return None

        # composite_rev gate
        if side == "yes" and float(comp_rev) < -rev_thresh:
            return None
        if side == "no"  and float(comp_rev) >  rev_thresh:
            return None

        # p_up_v2 gate + J vol-drift
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_otm_vwap_gate_model(stoch_thresh: float, dc_thresh: float, label: str, j_k: float = 0.1):
    """O_dcJ + stoch_k + OTM-vs-VWAP gate.

    Additional filter: block a bet when it is OTM AND VWAP is working against it.
      YES OTM (spot < strike) + spot below VWAP (vwap_score == -1) → block
      NO  OTM (spot > strike) + spot above VWAP (vwap_score == +1) → block
    """
    dc_col = "dc_4h_n20_pos"

    def eval_fn(row) -> dict:
        dc_pos  = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        vwap    = pd.to_numeric(row.get("vwap_score"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k) or pd.isna(vwap):
            return None

        pm     = float(row["p_market"])
        side   = str(row["side"])
        spot   = float(row["spot"])
        strike = float(row["strike"])
        sig    = sigma_tau(row)
        if sig <= 0:
            return None

        # Donchian gate
        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None

        # Stoch gate
        if side == "yes" and float(stoch_k) > stoch_thresh:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_thresh):
            return None

        # OTM + VWAP gate
        yes_otm = spot < strike   # YES needs price to go up past strike
        no_otm  = spot > strike   # NO needs price to go down past strike
        if side == "yes" and yes_otm and float(vwap) <= -1.0:
            return None
        if side == "no"  and no_otm  and float(vwap) >= 1.0:
            return None

        # p_up_v2 gate + J vol-drift
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_vwap_stochk_gate_model(stoch_thresh: float, dc_thresh: float, label: str,
                                use_vwap: bool = True, j_k: float = 0.1):
    """O_dcJ + stoch_k gate + optional vwap_score gate.

    vwap_score is discrete: +1 (above VWAP), 0 (neutral), -1 (below VWAP).
    Gate: block YES when vwap_score == +1, block NO when vwap_score == -1.
    """
    dc_col = "dc_4h_n20_pos"

    def eval_fn(row) -> dict:
        dc_pos  = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Donchian gate
        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None

        # Stoch gate
        if side == "yes" and float(stoch_k) > stoch_thresh:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_thresh):
            return None

        # VWAP score gate
        if use_vwap:
            vwap = pd.to_numeric(row.get("vwap_score"), errors="coerce")
            if pd.isna(vwap):
                return None
            if side == "yes" and float(vwap) >= 1.0:
                return None
            if side == "no"  and float(vwap) <= -1.0:
                return None

        # p_up_v2 gate + J vol-drift
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_stochk_gate_model(stoch_thresh: float, dc_thresh: float, label: str, j_k: float = 0.1):
    """O_dcJ + stoch_k gate.

    Stacks on top of O_dcJ logic (dc gate + p_up_v2 gate + J vol-drift).
    Additional gate:
      YES blocked when stoch_k > stoch_thresh  (overbought)
      NO  blocked when stoch_k < (100 - stoch_thresh)  (oversold)
    """
    dc_col = "dc_4h_n20_pos"

    def eval_fn(row) -> dict:
        dc_pos = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Donchian gate
        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None

        # Stoch gate
        if side == "yes" and float(stoch_k) > stoch_thresh:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_thresh):
            return None

        # p_up_v2 gate + J vol-drift
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_pup_thresh_model(stoch_thresh: float, dc_thresh: float,
                          pup_yes_thresh: Optional[float], pup_no_thresh: Optional[float],
                          label: str, j_k: float = 0.1, flip: bool = False):
    """R model with configurable p_up_v2 alignment thresholds.

    Standard (flip=False):
      YES blocked when pup <= pup_yes_thresh  (require bullish p_up_v2)
      NO  blocked when pup >= pup_no_thresh   (require bearish p_up_v2)
    Flipped (flip=True):
      YES blocked when pup >= pup_yes_thresh  (mean-reversion: YES when p_up_v2 bearish)
      NO  blocked when pup <= pup_no_thresh
    None → gate removed entirely for that side (J drift still applied).
    """
    dc_col = "dc_4h_n20_pos"

    def eval_fn(row) -> dict:
        dc_pos  = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k):
            return None
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None
        if side == "yes" and float(stoch_k) > stoch_thresh:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_thresh):
            return None

        # Configurable p_up_v2 alignment gate
        if not flip:
            if pup_yes_thresh is not None and side == "yes" and float(pup) <= pup_yes_thresh:
                return None
            if pup_no_thresh  is not None and side == "no"  and float(pup) >= pup_no_thresh:
                return None
        else:
            if pup_yes_thresh is not None and side == "yes" and float(pup) >= pup_yes_thresh:
                return None
            if pup_no_thresh  is not None and side == "no"  and float(pup) <= pup_no_thresh:
                return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, float(pup)))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_ema_slope_model(slope_col: str, k: float, label: str,
                         dc_gate: bool = False, dc_thresh: float = 0.80,
                         use_j: bool = False, j_k: float = 0.1):
    """Lognormal + EMA slope as drift factor.

    z_drift = slope_zscore * k * sqrt(tau_h)
    where slope_zscore = slope / rolling_std normalised across the dataset.

    Positive k = momentum (slope up → YES drift up).
    Negative k = mean-reversion.

    Optionally stacks dc gate and J vol-drift on top.
    """
    def eval_fn(row, _slope_std=None) -> dict:
        slope_raw = pd.to_numeric(row.get(slope_col), errors="coerce")
        if pd.isna(slope_raw) or _slope_std is None or _slope_std == 0:
            return None

        slope_z = float(slope_raw) / _slope_std   # z-score relative to dataset std

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Optional Donchian gate
        if dc_gate:
            dc_pos = pd.to_numeric(row.get("dc_4h_n20_pos"), errors="coerce")
            if pd.isna(dc_pos):
                return None
            if side == "yes" and float(dc_pos) > dc_thresh:
                return None
            if side == "no"  and float(dc_pos) < (1 - dc_thresh):
                return None

        # Optional J vol-drift + p_up_v2 gate
        if use_j:
            pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
            vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
            vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
            if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
                return None
            if side == "yes" and pup <= 0.50:
                return None
            if side == "no"  and pup >= 0.50:
                return None
            tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
            pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
            vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
            z_j     = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        else:
            tau_h = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
            z_j   = 0.0

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        z_drift = z_j + slope_z * k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}

    # Return a wrapper that pre-computes std from the trade dataset
    def factory(trades_df: pd.DataFrame):
        col_vals = pd.to_numeric(trades_df.get(slope_col, pd.Series(dtype=float)),
                                 errors="coerce")
        std = float(col_vals.std()) if col_vals.notna().sum() > 10 else 1.0
        return lambda row: eval_fn(row, _slope_std=std)
    return factory


def make_r_with_kelly_boost(stoch_gate: float, dc_thresh: float, label: str,
                            j_k: float = 0.1,
                            boost_thresh: float = 30.0, boost_mult: float = 1.5):
    """R model (dc_gate + stoch_gate + p_up_v2 + J drift) with Kelly boost.

    When stoch_crossover_active == 1 AND stoch_k < boost_thresh on a YES bet,
    multiply the Kelly fraction by boost_mult (capped at 2× KELLY_CAP).
    All other logic identical to R_dcJ87_sk78.
    """
    dc_col = "dc_4h_n20_pos"
    BOOSTED_CAP = min(KELLY_CAP * boost_mult, KELLY_CAP * 2)

    def eval_fn(row) -> dict:
        dc_pos    = pd.to_numeric(row.get(dc_col), errors="coerce")
        stoch_k   = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        crossover = pd.to_numeric(row.get("stoch_crossover_active"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None
        if side == "yes" and float(stoch_k) > stoch_gate:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_gate):
            return None

        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        spot    = float(row["spot"])
        strike  = float(row["strike"])
        tau_h   = max(float(row["tau_minutes"]) / 60.0, 1 / 60)
        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        is_boost = (side == "yes"
                    and not pd.isna(crossover)
                    and float(crossover) == 1.0
                    and float(stoch_k) < boost_thresh)
        cap  = BOOSTED_CAP if is_boost else KELLY_CAP
        mult = boost_mult  if is_boost else 1.0
        raw_k = min(edge / pm_risk * KELLY_MULT * mult, cap)
        n = raw_k * BANKROLL / pm_risk

        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0),
                "boosted": is_boost}
    return eval_fn


def make_multitf_stoch_bounce(stoch_1h_thresh: float, stoch_4h_thresh: float,
                               pm_cap: float, label: str):
    """Stoch bounce requiring BOTH 1h and 4h stoch in oversold territory.

    YES trigger: stoch_k (1h) < stoch_1h_thresh AND stoch_k_4h < stoch_4h_thresh
                 AND p_market < pm_cap
    Pure lognormal, no drift — directional call from multi-TF stoch alignment.
    """
    def eval_fn(row) -> dict:
        stoch_1h = pd.to_numeric(row.get("stoch_k"),    errors="coerce")
        stoch_4h = pd.to_numeric(row.get("stoch_k_4h"), errors="coerce")
        if pd.isna(stoch_1h) or pd.isna(stoch_4h):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes":
            if float(stoch_1h) >= stoch_1h_thresh:
                return None
            if float(stoch_4h) >= stoch_4h_thresh:
                return None
            if pm >= pm_cap:
                return None
        else:
            if float(stoch_1h) <= (100 - stoch_1h_thresh):
                return None
            if float(stoch_4h) <= (100 - stoch_4h_thresh):
                return None
            if pm <= (1 - pm_cap):
                return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_crossover_model(stoch_thresh: float, label: str, pm_cap: float = 1.0,
                         require_no_crossover: bool = False):
    """Stoch crossover + oversold trigger model.

    YES trigger: stoch_crossover_active == 1 AND stoch_k < stoch_thresh
                 (stoch just turned up from oversold territory — bounce confirmation)
    NO  trigger: symmetric — crossover from overbought (require_no_crossover handles that
                 side by treating crossover_active==-1 or just stoch>100-thresh)
    Pure lognormal base.
    pm_cap: optional p_market cap for YES (default 1.0 = no cap).
    """
    def eval_fn(row) -> dict:
        stoch_k  = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        crossover = pd.to_numeric(row.get("stoch_crossover_active"), errors="coerce")
        if pd.isna(stoch_k) or pd.isna(crossover):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes":
            if float(crossover) != 1.0:
                return None
            if float(stoch_k) >= stoch_thresh:
                return None
            if pm >= pm_cap:
                return None
        else:
            # NO side: stoch crossing down from overbought
            if float(crossover) != -1.0 and float(crossover) != 0.0:
                return None
            if float(stoch_k) <= (100 - stoch_thresh):
                return None
            if pm <= (1 - pm_cap):
                return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_stoch_bounce_model(stoch_thresh: float, pm_cap: float, label: str):
    """Stoch-triggered OTM bounce model.

    Only fires when stoch_k is at an extreme level — pure trigger, not a gate.

    YES trigger: stoch_k < stoch_thresh (oversold) + p_market < pm_cap
    NO  trigger: stoch_k > (100 - stoch_thresh) (overbought) + p_market > (1 - pm_cap)

    Pure lognormal p_model — no drift. The directional call comes entirely from the
    stoch extreme. Asymmetric payoff is guaranteed by the pm_cap constraint.
    """
    def eval_fn(row) -> dict:
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if pd.isna(stoch_k):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        # Stoch trigger (only fire at extremes)
        if side == "yes":
            if float(stoch_k) >= stoch_thresh:
                return None
            if pm >= pm_cap:
                return None
        else:
            if float(stoch_k) <= (100 - stoch_thresh):
                return None
            if pm <= (1 - pm_cap):
                return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_stoch_bounce_yes_only(stoch_thresh: float, pm_cap: float, label: str):
    """Stoch-triggered bounce — YES side only (oversold → expect bounce up).

    Only takes YES bets when stoch_k < stoch_thresh and p_market < pm_cap.
    Pure lognormal base.
    """
    def eval_fn(row) -> dict:
        side = str(row["side"])
        if side != "yes":
            return None

        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if pd.isna(stoch_k):
            return None

        pm  = float(row["p_market"])
        sig = sigma_tau(row)
        if sig <= 0:
            return None

        if float(stoch_k) >= stoch_thresh:
            return None
        if pm >= pm_cap:
            return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee  = FEE_RATE * min(pm, 1 - pm)
        edge = p_m - pm - fee
        if edge <= MIN_EDGE_A or pm <= 0:
            return None

        n = kelly_contracts(edge, pm)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


def make_combined_r_w_model(stoch_thresh: float, pm_cap: float,
                             dc_thresh: float, stoch_gate: float, label: str,
                             j_k: float = 0.1):
    """Combined R + W strategy in a single model.

    Takes a bet if EITHER R_dcJ conditions pass OR W bounce conditions pass.
    R path: dc_4h_n20_pos gate + stoch gate + p_up_v2 gate + J drift
    W path: stoch trigger (< stoch_thresh YES, > 100-stoch_thresh NO) + pm cap

    Non-overlapping: R targets normal market bets; W targets OTM extreme bounces.
    """
    def eval_fn(row) -> dict:
        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        tau_h  = max(float(row["tau_minutes"]) / 60.0, 1 / 60)

        # Try W path first (bounce trigger)
        stoch_k = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if not pd.isna(stoch_k):
            w_yes = side == "yes" and float(stoch_k) < stoch_thresh and pm < pm_cap
            w_no  = side == "no"  and float(stoch_k) > (100 - stoch_thresh) and pm > (1 - pm_cap)
            if w_yes or w_no:
                p_m = p_lognormal(spot, strike, sig, 0.0)
                fee = FEE_RATE * min(pm, 1 - pm)
                edge    = (p_m - pm - fee) if side == "yes" else (pm - p_m - fee)
                pm_risk = pm if side == "yes" else (1 - pm)
                if edge > MIN_EDGE_A and pm_risk > 0:
                    n = kelly_contracts(edge, pm_risk)
                    if n >= 0.01:
                        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
                        return {"model": label, "side": side, "pm": pm, "edge": edge,
                                "n_cont": n, "pnl": pnl, "won": (pnl > 0), "path": "W"}

        # Try R path (dc gate + stoch gate + p_up_v2 + J drift)
        dc_pos = pd.to_numeric(row.get("dc_4h_n20_pos"), errors="coerce")
        if pd.isna(dc_pos) or pd.isna(stoch_k):
            return None
        if side == "yes" and float(dc_pos) > dc_thresh:
            return None
        if side == "no"  and float(dc_pos) < (1 - dc_thresh):
            return None
        if side == "yes" and float(stoch_k) > stoch_gate:
            return None
        if side == "no"  and float(stoch_k) < (100 - stoch_gate):
            return None

        pup     = pd.to_numeric(row.get("p_up_v2_backfilled"), errors="coerce")
        vol_eff = pd.to_numeric(row.get("vol_eff"), errors="coerce")
        vol_imp = pd.to_numeric(row.get("vol_implied_kalshi"), errors="coerce")
        if pd.isna(pup) or pd.isna(vol_eff) or pd.isna(vol_imp) or vol_imp <= 0:
            return None
        if side == "yes" and pup <= 0.50:
            return None
        if side == "no"  and pup >= 0.50:
            return None

        pup_z   = float(norm.ppf(max(0.01, min(0.99, pup))))
        vol_amp = float(np.clip(vol_eff / vol_imp, 0.3, 3.0))
        z_drift = pup_z * vol_amp * j_k * math.sqrt(tau_h)
        p_m     = p_lognormal(spot, strike, sig, z_drift)

        fee     = FEE_RATE * min(pm, 1 - pm)
        edge    = (p_m - pm - fee) if side == "yes" else (pm - p_m - fee)
        pm_risk = pm if side == "yes" else (1 - pm)
        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0), "path": "R"}
    return eval_fn


def make_wmt_union_model(stoch_1h_w: float, stoch_1h_mt: float, stoch_4h_mt: float,
                         pm_cap: float, label: str):
    """WMT union: fire if EITHER W leg OR MT leg condition is met.

    W leg  (1h extreme):  stoch_k < stoch_1h_w AND pm < pm_cap
    MT leg (multi-TF):    stoch_k < stoch_1h_mt AND stoch_k_4h < stoch_4h_mt AND pm < pm_cap

    Symmetric NO side (overbought mirror).
    Pure lognormal, no drift. One bet per row — no double-counting.
    """
    def eval_fn(row) -> dict:
        stoch_1h = pd.to_numeric(row.get("stoch_k"),    errors="coerce")
        stoch_4h = pd.to_numeric(row.get("stoch_k_4h"), errors="coerce")
        if pd.isna(stoch_1h):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes":
            w_fires  = float(stoch_1h) < stoch_1h_w and pm < pm_cap
            mt_fires = (not pd.isna(stoch_4h)
                        and float(stoch_1h) < stoch_1h_mt
                        and float(stoch_4h) < stoch_4h_mt
                        and pm < pm_cap)
            if not (w_fires or mt_fires):
                return None
        else:
            w_fires  = float(stoch_1h) > (100 - stoch_1h_w) and pm > (1 - pm_cap)
            mt_fires = (not pd.isna(stoch_4h)
                        and float(stoch_1h) > (100 - stoch_1h_mt)
                        and float(stoch_4h) > (100 - stoch_4h_mt)
                        and pm > (1 - pm_cap))
            if not (w_fires or mt_fires):
                return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        path = "W" if w_fires else "MT"
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0), "path": path}
    return eval_fn


def make_breakout_model(label: str, break_col: str = "dc_4h_n20_break",
                        max_pm_yes: float = 1.0, stoch_cap: float = 100.0,
                        symmetric_no: bool = True):
    """Momentum / breakout model.

    YES: break_col == +1, p_market < max_pm_yes, stoch_k < stoch_cap
    NO:  break_col == -1, p_market > (1-max_pm_yes), stoch_k > (100-stoch_cap)
         (only when symmetric_no=True)
    Pure lognormal base — no drift.
    max_pm_yes=1.0 means no p_market cap (take any p_market).
    """
    def eval_fn(row) -> dict:
        dc_break = pd.to_numeric(row.get(break_col), errors="coerce")
        stoch_k  = pd.to_numeric(row.get("stoch_k"), errors="coerce")
        if pd.isna(dc_break) or pd.isna(stoch_k):
            return None

        pm   = float(row["p_market"])
        side = str(row["side"])
        sig  = sigma_tau(row)
        if sig <= 0:
            return None

        if side == "yes":
            if float(dc_break) != 1.0:
                return None
            if pm >= max_pm_yes:
                return None
            if float(stoch_k) >= stoch_cap:
                return None
        else:
            if not symmetric_no:
                return None
            if float(dc_break) != -1.0:
                return None
            if pm <= (1 - max_pm_yes):
                return None
            if float(stoch_k) <= (100 - stoch_cap):
                return None

        spot   = float(row["spot"])
        strike = float(row["strike"])
        p_m    = p_lognormal(spot, strike, sig, 0.0)

        fee = FEE_RATE * min(pm, 1 - pm)
        if side == "yes":
            edge    = p_m - pm - fee
            pm_risk = pm
        else:
            edge    = pm - p_m - fee
            pm_risk = 1 - pm

        if edge <= MIN_EDGE_A or pm_risk <= 0:
            return None

        n = kelly_contracts(edge, pm_risk)
        if n < 0.01:
            return None
        pnl = compute_pnl(side, pm, n, int(row["resolved_yes"]))
        return {"model": label, "side": side, "pm": pm, "edge": edge,
                "n_cont": n, "pnl": pnl, "won": (pnl > 0)}
    return eval_fn


# ── run backtest ──────────────────────────────────────────────────────────────

def run():
    trades = load_trades()

    dc_col = "dc_4h_n20_pos"

    # EMA slope factories — call with trades to resolve std, then get eval_fn
    ema_factories = {
        # Pure slope drift (no other gates) — momentum direction
        "P_ema20_15m_mom":  make_ema_slope_model("ema20_slope_15m", +0.3, "P_ema20_15m_mom"),
        "P_ema20_1h_mom":   make_ema_slope_model("ema20_slope_1h",  +0.3, "P_ema20_1h_mom"),
        "P_ema20_4h_mom":   make_ema_slope_model("ema20_slope_4h",  +0.3, "P_ema20_4h_mom"),
        "P_ema20_1d_mom":   make_ema_slope_model("ema20_slope_1d",  +0.3, "P_ema20_1d_mom"),
        # Mean-reversion direction
        "P_ema20_15m_rev":  make_ema_slope_model("ema20_slope_15m", -0.3, "P_ema20_15m_rev"),
        "P_ema20_1h_rev":   make_ema_slope_model("ema20_slope_1h",  -0.3, "P_ema20_1h_rev"),
        "P_ema20_4h_rev":   make_ema_slope_model("ema20_slope_4h",  -0.3, "P_ema20_4h_rev"),
        "P_ema20_1d_rev":   make_ema_slope_model("ema20_slope_1d",  -0.3, "P_ema20_1d_rev"),
        # Best EMA slope stacked on top of O_dcJ_t80
        "Q_dcJ80_4h_mom":   make_ema_slope_model("ema20_slope_4h",  +0.3, "Q_dcJ80_4h_mom",
                                                  dc_gate=True, dc_thresh=0.80, use_j=True),
        "Q_dcJ80_4h_rev":   make_ema_slope_model("ema20_slope_4h",  -0.3, "Q_dcJ80_4h_rev",
                                                  dc_gate=True, dc_thresh=0.80, use_j=True),
        "Q_dcJ80_1h_mom":   make_ema_slope_model("ema20_slope_1h",  +0.3, "Q_dcJ80_1h_mom",
                                                  dc_gate=True, dc_thresh=0.80, use_j=True),
        "Q_dcJ80_1h_rev":   make_ema_slope_model("ema20_slope_1h",  -0.3, "Q_dcJ80_1h_rev",
                                                  dc_gate=True, dc_thresh=0.80, use_j=True),
    }
    resolved_ema = {name: factory(trades) for name, factory in ema_factories.items()}

    eval_fns = {
        "B_purelogN": eval_model_B,
        "R_pup50":    make_stochk_gate_model(78.0, 0.87, "R_pup50"),
    }
    # Fine NO-gate sweep (YES=None): 0.35 → 0.55 by 0.02
    for _no in [0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]:
        _lbl = f"R_YN_N{int(round(_no*100)):02d}"
        eval_fns[_lbl] = make_pup_thresh_model(78.0, 0.87, None, _no, _lbl)
    # Fine YES-gate sweep (NO=0.40): 0.30 → 0.50 by 0.02
    for _yes in [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
        _lbl = f"R_Y{int(round(_yes*100)):02d}_N40"
        eval_fns[_lbl] = make_pup_thresh_model(78.0, 0.87, _yes, 0.40, _lbl)
    eval_fns["MT_1h17_4h40_pm60"] = make_multitf_stoch_bounce(17.0, 40.0, 0.60, "MT_1h17_4h40_pm60")

    all_results = {m: [] for m in eval_fns}
    trades["month"] = trades["logged_at"].dt.to_period("M")

    for _, row in trades.iterrows():
        for model, fn in eval_fns.items():
            res = fn(row)
            if res:
                res["logged_at"] = row["logged_at"]
                res["month"]     = row["month"]
                all_results[model].append(res)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS  —  real Kalshi prices, real Binance settlement")
    print(f"  Universe: {len(trades):,} trades  "
          f"({trades['logged_at'].min().date()} → {trades['logged_at'].max().date()})")
    print(f"{'='*70}")
    print(f"  {'Model':<18}  {'Trades':>7}  {'WR':>7}  {'P&L':>10}  {'$/trade':>8}")
    print("  " + "-" * 56)

    model_dfs = {}
    for model, results in all_results.items():
        if not results:
            print(f"  {model:<18}  {'—':>7}")
            continue
        df = pd.DataFrame(results)
        model_dfs[model] = df
        n    = len(df)
        wr   = df["won"].mean()
        pnl  = df["pnl"].sum()
        ppt  = pnl / n if n else 0
        print(f"  {model:<18}  {n:>7,}  {wr:>7.1%}  {pnl:>+10,.0f}  {ppt:>+8.2f}")

    # ── monthly breakdown ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MONTHLY P&L BY MODEL")
    print(f"{'='*70}")
    months = sorted(trades["month"].unique())
    header = f"  {'Month':<8}" + "".join(f"  {m:<14}" for m in eval_fns)
    print(header)
    print("  " + "-" * (10 + 16 * len(eval_fns)))

    for m in months:
        row_str = f"  {str(m):<8}"
        for model, df in model_dfs.items():
            mdf = df[df["month"] == m]
            if len(mdf) == 0:
                row_str += f"  {'—':>14}"
            else:
                row_str += f"  {mdf['pnl'].sum():>+8,.0f} ({len(mdf):>3})"
        print(row_str)

    totals = f"  {'TOTAL':<8}"
    for model, df in model_dfs.items():
        totals += f"  {df['pnl'].sum():>+8,.0f} ({len(df):>3})"
    print("  " + "-" * (10 + 16 * len(eval_fns)))
    print(totals)

    # ── side split for Model A ─────────────────────────────────────────────────
    if "A_asrun" in model_dfs:
        dfA = model_dfs["A_asrun"]
        print(f"\n{'='*70}")
        print(f"  MODEL A — YES vs NO SPLIT")
        print(f"{'='*70}")
        for side, g in dfA.groupby("side"):
            pnl = g["pnl"].sum(); wr = g["won"].mean(); n = len(g)
            be  = g["pm"].mean() if side == "yes" else 1 - g["pm"].mean()
            print(f"  {side:4s}  n={n:,}  WR={wr:.1%}  breakeven={be:.1%}  "
                  f"P&L=${pnl:+,.0f}  $/trade=${pnl/n:+.2f}")

    # ── edge decile analysis ───────────────────────────────────────────────────
    if "A_asrun" in model_dfs:
        dfA = model_dfs["A_asrun"]
        print(f"\n{'='*70}")
        print(f"  MODEL A — P&L BY EDGE QUARTILE (does higher edge = better outcomes?)")
        print(f"{'='*70}")
        dfA["edge_q"] = pd.qcut(dfA["edge"], 4, labels=["Q1 low","Q2","Q3","Q4 high"])
        for q, g in dfA.groupby("edge_q", observed=True):
            pnl = g["pnl"].sum(); wr = g["won"].mean(); n = len(g)
            print(f"  {str(q):<10}  n={n:,}  WR={wr:.1%}  P&L=${pnl:+,.0f}  "
                  f"mean_edge={g['edge'].mean():.3f}")

    # ── Kelly boost analysis ───────────────────────────────────────────────────
    rb_models = [m for m in model_dfs if m.startswith("RB_")]
    if rb_models and "R_dcJ87_sk78" in model_dfs:
        dfR = model_dfs["R_dcJ87_sk78"]
        print(f"\n{'='*70}")
        print(f"  KELLY BOOST MODELS  (R baseline: {len(dfR)} trades, ${dfR['pnl'].sum():+,.0f})")
        print(f"{'='*70}")
        print(f"  {'Model':<18}  {'Trades':>7}  {'WR':>7}  {'P&L':>10}  {'$/t':>8}  "
              f"{'Boosted':>8}  {'Boost_PnL':>10}")
        print("  " + "-" * 72)
        for model in rb_models:
            df = model_dfs[model]
            n    = len(df)
            wr   = df["won"].mean()
            pnl  = df["pnl"].sum()
            if "boosted" in df.columns:
                n_boost   = df["boosted"].sum()
                pnl_boost = df[df["boosted"]]["pnl"].sum() if n_boost else 0
                boost_str = f"{n_boost:>8,}  {pnl_boost:>+10,.0f}"
            else:
                boost_str = "       —           —"
            print(f"  {model:<18}  {n:>7,}  {wr:>7.1%}  {pnl:>+10,.0f}  {pnl/n:>+8.2f}  {boost_str}")

    # ── YES/NO split for X (crossover) models ─────────────────────────────────
    x_models = [m for m in model_dfs if m.startswith("X_")]
    if x_models:
        print(f"\n{'='*70}")
        print(f"  CROSSOVER MODELS — YES vs NO SPLIT")
        print(f"{'='*70}")
        print(f"  {'Model':<22}  {'side':>4}  {'n':>5}  {'WR':>7}  {'BE_WR':>7}  {'P&L':>10}  {'$/t':>7}")
        print("  " + "-" * 66)
        for model in x_models:
            df = model_dfs[model]
            for side, g in df.groupby("side"):
                n   = len(g)
                wr  = g["won"].mean()
                pnl = g["pnl"].sum()
                avg_pm = g["pm"].mean()
                fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
                be_wr = avg_pm + fee_avg if side == "yes" else (1 - avg_pm) + fee_avg
                print(f"  {model:<22}  {side:>4}  {n:>5,}  {wr:>7.1%}  {be_wr:>7.1%}  {pnl:>+10,.0f}  {pnl/n:>+7.2f}")

    # ── R vs W overlap analysis ───────────────────────────────────────────────
    if "R_dcJ87_sk78" in model_dfs:
        dfR = model_dfs["R_dcJ87_sk78"]
        r_keys = set(zip(dfR["logged_at"], dfR["side"]))
        print(f"\n{'='*70}")
        print(f"  R vs W/X OVERLAP ANALYSIS  (how many trades are additive?)")
        print(f"{'='*70}")
        for wlabel in [m for m in model_dfs if m.startswith(("W_", "X_", "WMT_", "MT_"))]:
            dfW = model_dfs[wlabel]
            w_keys = set(zip(dfW["logged_at"], dfW["side"]))
            overlap_keys = r_keys & w_keys
            w_only_keys  = w_keys - r_keys
            n_overlap = len(overlap_keys)
            n_w_only  = len(w_only_keys)
            # P&L from W-only trades
            w_only_pnl = dfW[dfW.apply(
                lambda r: (r["logged_at"], r["side"]) in w_only_keys, axis=1
            )]["pnl"].sum()
            print(f"  {wlabel}: total={len(dfW)}  overlap_with_R={n_overlap}  "
                  f"W_only={n_w_only}  W_only_PnL=${w_only_pnl:+,.0f}  "
                  f"True_additive=${w_only_pnl:+,.0f} (+R's ${dfR['pnl'].sum():+,.0f})")

    # ── YES/NO split for R_pup sweep models ──────────────────────────────────
    pup_models = [m for m in model_dfs if m.startswith("R_pup")]
    if pup_models:
        print(f"\n{'='*70}")
        print(f"  p_up_v2 THRESHOLD SWEEP — YES vs NO SPLIT")
        print(f"{'='*70}")
        print(f"  {'Model':<14}  {'side':>4}  {'n':>5}  {'WR':>7}  {'BE_WR':>7}  {'P&L':>10}  {'$/t':>7}")
        print("  " + "-" * 60)
        for model in pup_models:
            df = model_dfs[model]
            for side, g in df.groupby("side"):
                n   = len(g)
                wr  = g["won"].mean()
                pnl = g["pnl"].sum()
                avg_pm = g["pm"].mean()
                fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
                be_wr = avg_pm + fee_avg if side == "yes" else (1 - avg_pm) + fee_avg
                print(f"  {model:<14}  {side:>4}  {n:>5,}  {wr:>7.1%}  {be_wr:>7.1%}  {pnl:>+10,.0f}  {pnl/n:>+7.2f}")

    # ── YES/NO split for W (bounce) models ────────────────────────────────────
    w_models = [m for m in model_dfs if m.startswith(("W_", "MT_", "WMT_"))]
    if w_models:
        print(f"\n{'='*70}")
        print(f"  BOUNCE MODELS — YES vs NO SPLIT")
        print(f"{'='*70}")
        print(f"  {'Model':<18}  {'side':>4}  {'n':>5}  {'WR':>7}  {'BE_WR':>7}  {'P&L':>10}  {'$/t':>7}")
        print("  " + "-" * 64)
        for model in w_models:
            df = model_dfs[model]
            for side, g in df.groupby("side"):
                n   = len(g)
                wr  = g["won"].mean()
                pnl = g["pnl"].sum()
                avg_pm = g["pm"].mean()
                fee_avg = FEE_RATE * min(avg_pm, 1 - avg_pm)
                be_wr = avg_pm + fee_avg if side == "yes" else (1 - avg_pm) + fee_avg
                print(f"  {model:<18}  {side:>4}  {n:>5,}  {wr:>7.1%}  {be_wr:>7.1%}  {pnl:>+10,.0f}  {pnl/n:>+7.2f}")

    # ── write results ──────────────────────────────────────────────────────────
    out_rows = []
    for model, results in all_results.items():
        for r in results:
            out_rows.append({k: v for k, v in r.items() if k != "logged_at"})
    pd.DataFrame(out_rows).to_csv(RES_DIR / "backtest_model_comparison.csv", index=False)
    print(f"\n  Wrote results/backtest_model_comparison.csv")


if __name__ == "__main__":
    run()
