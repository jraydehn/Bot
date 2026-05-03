#!/usr/bin/env python3
"""
replay_simulator.py — Full-fidelity BTC gate/parameter replay simulator.

Design principles:
  1. Uses ALL candidate contracts per decision slot (trade + no_trade rows).
  2. Recomputes p_model from raw archived inputs (vol_eff, tau_minutes, spot,
     strike, composite_p_up) — consistent regardless of which k_drift was
     active when the row was logged.
  3. Calls evaluate_trade() from decision.py directly — no gate reimplementation.
  4. One trade per slot: picks highest net_edge candidate that passes all gates.
  5. Flat $1000 bankroll for all Kelly sizing — parameter sweeps are comparable.

Key difference from prior backtests:
  - Prior backtests filter already-selected trades, missing the slot-substitution
    effect (blocking a trade causes the runner to pick the next-best contract).
  - This simulator replays the full selection from all candidates per slot.

Usage:
    python3 replay_simulator.py                         # baseline
    python3 replay_simulator.py --pm-floor-yes 0.15    # test pm floor gate
    python3 replay_simulator.py --sweep-pm              # sweep pm floors
    python3 replay_simulator.py --k-drift-yes 1.4 --k-drift-no 0.3 --sweep-pm
"""

import math
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from decision import evaluate_trade

# ── Production constants ───────────────────────────────────────────────────
DEFAULT_SLIPPAGE  = 0.003
DEFAULT_SPREAD    = 0.005
FLAT_BANKROLL     = 1000.0      # fixed for all simulations — do not compound

K_DRIFT_YES_DEFAULT = 1.40
K_DRIFT_NO_DEFAULT  = 0.30


# ── Config ─────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    k_drift_yes:    float = K_DRIFT_YES_DEFAULT
    k_drift_no:     float = K_DRIFT_NO_DEFAULT
    pm_floor_yes:   float = 0.04   # BTC YES min p_market (0.04 = effectively off)
    pm_ceil_no:     float = 0.96   # BTC NO  max p_market (0.96 = effectively off)
    otm_yes_gate:   bool  = False  # momentum exhaustion gate for OTM YES < 0.35
    label:          str   = "baseline"


# ── Model formulas ─────────────────────────────────────────────────────────
def _sigma_tau(vol_eff: float, tau_min: float) -> float:
    """vol_eff is stored per-sqrt(minute)."""
    return vol_eff * math.sqrt(tau_min)


def compute_p_yes(spot, strike, vol_eff, tau_min, p_up, k_drift_yes):
    """Drift-adjusted YES probability."""
    if vol_eff <= 0 or tau_min <= 0 or spot <= 0:
        return None
    st = _sigma_tau(vol_eff, tau_min)
    if st <= 0:
        return None
    z = math.log(strike / spot) / st
    z_adj = z - norm.ppf(p_up) * k_drift_yes
    return float(np.clip(1 - norm.cdf(z_adj), 0.01, 0.99))


def compute_p_no(spot, strike, vol_eff, tau_min, p_up, k_drift_no):
    """Independent drift-adjusted NO probability."""
    if vol_eff <= 0 or tau_min <= 0 or spot <= 0:
        return None
    st = _sigma_tau(vol_eff, tau_min)
    if st <= 0:
        return None
    z = math.log(strike / spot) / st
    z_adj = z - norm.ppf(p_up) * k_drift_no
    return float(np.clip(norm.cdf(z_adj), 0.01, 0.99))


# ── P&L calculation ────────────────────────────────────────────────────────
def _kalshi_fee(pm: float) -> float:
    return 0.07 * min(pm, 1 - pm)


def trade_pnl(bet: float, side: str, pm: float, won: bool) -> float:
    if bet <= 0:
        return 0.0
    fee = _kalshi_fee(pm)
    if side == "yes":
        if won:
            n_ct = bet / pm
            return bet * (1 - pm) / pm - fee * n_ct
        return -bet
    else:
        if won:
            n_ct = bet / (1 - pm)
            return bet * pm / (1 - pm) - fee * n_ct
        return -bet


# ── Data loading ───────────────────────────────────────────────────────────
REQUIRED_COLS = ["spot", "strike", "p_market", "vol_eff", "tau_minutes",
                 "resolved_yes", "decision_time"]

NUMERIC_COLS  = ["spot", "strike", "p_market", "vol_eff", "tau_minutes",
                 "composite_p_up", "offset_pct", "resolved_yes",
                 "structure_bias", "confirmation_bias", "confirmation_score",
                 "no_score", "obi_score", "vol_score",
                 "stoch_k", "ema_stretch_score"]


def _load_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["decision_time"] = pd.to_datetime(df["decision_time"], errors="coerce")
    return df


def load_data(csv_path: Path, extra_csvs: list = None) -> pd.DataFrame:
    """Load primary CSV plus any extra archive CSVs, deduplicate, and filter."""
    frames = [_load_one(csv_path)]

    for extra in (extra_csvs or []):
        p = Path(extra)
        if not p.exists():
            print(f"  WARNING: archive not found: {p}")
            continue
        frames.append(_load_one(p))
        print(f"  + {p.name}")

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate: same decision slot + contract = same observation
    if "contract_ticker" in df.columns:
        df = df.drop_duplicates(subset=["decision_time", "contract_ticker"], keep="last")
    else:
        df = df.drop_duplicates(subset=["decision_time", "spot", "strike"], keep="last")

    # Keep only rows with a resolved outcome and required raw inputs
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df["p_market"] >= 0.04) & (df["p_market"] <= 0.96)]
    df = df[(df["vol_eff"] > 0) & (df["tau_minutes"] > 0) & (df["spot"] > 0)]

    return df.sort_values("decision_time").reset_index(drop=True)


# ── Slot evaluation ────────────────────────────────────────────────────────
def _safe_int(val, default=0):
    try:
        return int(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def _safe_str(val, default="neutral"):
    if pd.isna(val) or val is None:
        return default
    return str(val)


def evaluate_slot(group: pd.DataFrame, cfg: SimConfig, iso: IsotonicRegression = None):
    """
    Evaluate all candidate contracts in one decision slot.
    Returns the best (contract, side) that passes all gates, or None.
    """
    candidates = []

    for _, row in group.iterrows():
        spot    = float(row["spot"])
        strike  = float(row["strike"])
        pm      = float(row["p_market"])
        vol_eff = float(row["vol_eff"])
        tau     = float(row["tau_minutes"])
        won_yes = int(row["resolved_yes"]) == 1

        # Composite scorer inputs
        comp_pup_raw = row.get("composite_p_up")
        comp_active  = pd.notna(comp_pup_raw)
        p_up         = float(comp_pup_raw) if comp_active else 0.504

        # Gate/signal inputs
        offset      = _safe_float(row.get("offset_pct"), (strike - spot) / spot)
        struct      = _safe_int(row.get("structure_bias"), 0)
        conf_bias   = _safe_int(row.get("confirmation_bias"), 0)
        cscore      = _safe_int(row.get("confirmation_score"), 0)
        nscore      = _safe_int(row.get("no_score"), 0)
        obi         = _safe_int(row.get("obi_score"), 0)
        vol_sc      = _safe_int(row.get("vol_score"), 0)
        ema_al      = _safe_str(row.get("ema_alignment"), "neutral")
        stoch_k     = _safe_float(row.get("stoch_k"), 50.0)
        ema_stretch = _safe_int(row.get("ema_stretch_score"), 0)

        # bid/ask approximation from mid
        pm_ask = min(pm + DEFAULT_SPREAD / 2, 0.96)
        pm_bid = max(pm - DEFAULT_SPREAD / 2, 0.04)

        # Common kwargs for evaluate_trade
        common_kwargs = dict(
            confirmation_score=cscore,
            no_score=nscore,
            obi_score=obi,
            vol_score=vol_sc,
            ema_alignment=ema_al,
            asset="BTC",
            composite_active=comp_active,
            composite_p_up=p_up if comp_active else 0.504,
            offset_pct=offset,
            p_market_bid=pm_bid,
            p_market_ask=pm_ask,
            slippage=DEFAULT_SLIPPAGE,
            spread=DEFAULT_SPREAD,
        )

        # ── YES candidate ─────────────────────────────────────────────────
        p_yes = compute_p_yes(spot, strike, vol_eff, tau, p_up, cfg.k_drift_yes)

        # OTM YES momentum exhaustion gate (only when enabled)
        _yes_gate_blocked = False
        if cfg.otm_yes_gate and pm < 0.35:
            if pm < 0.15:
                _yes_gate_blocked = True          # hard block: deep OTM unrecoverable
            elif ema_stretch >= 1:
                _yes_gate_blocked = True          # EMA already stretched bullish
            elif pm >= 0.25 and stoch_k > 70:
                _yes_gate_blocked = True          # stoch overbought in [0.25, 0.35)

        if p_yes is not None and pm >= cfg.pm_floor_yes and not _yes_gate_blocked:
            # Apply isotonic calibration if provided
            p_yes_eval = float(iso.predict([p_yes])[0]) if iso is not None else p_yes
            try:
                dec = evaluate_trade(
                    struct, conf_bias, p_yes_eval, pm, FLAT_BANKROLL,
                    force_side="yes",
                    **common_kwargs,
                )
                if dec.decision == "trade":
                    candidates.append({
                        "side":     "yes",
                        "p_model":  p_yes_eval,
                        "pm":       pm,
                        "net_edge": dec.net_edge,
                        "won":      won_yes,
                        "offset":   offset,
                        "bet":      dec.bet_amount,
                    })
            except Exception:
                pass

        # ── NO candidate ──────────────────────────────────────────────────
        p_no = compute_p_no(spot, strike, vol_eff, tau, p_up, cfg.k_drift_no)
        # pm here is still the YES price; NO contract costs 1-pm
        if p_no is not None and pm <= cfg.pm_ceil_no:
            try:
                # Pass 1-p_no as p_model so evaluate_trade sees correct NO edge:
                #   edge = p_market - (1-p_no) = p_no - (1-p_market)
                dec = evaluate_trade(
                    struct, conf_bias, 1.0 - p_no, pm, FLAT_BANKROLL,
                    force_side="no",
                    **common_kwargs,
                )
                if dec.decision == "trade":
                    candidates.append({
                        "side":     "no",
                        "p_model":  p_no,
                        "pm":       pm,
                        "net_edge": dec.net_edge,
                        "won":      not won_yes,
                        "offset":   offset,
                        "bet":      dec.bet_amount,
                    })
            except Exception:
                pass

    if not candidates:
        return None

    return max(candidates, key=lambda c: c["net_edge"])


# ── Calibration helpers ────────────────────────────────────────────────────
def collect_calibration_data(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """
    Collect (p_yes_recomputed, p_market, resolved_yes) for ALL YES candidates
    across all slots — not just selected trades. Used to fit isotonic calibration.
    """
    rows = []
    for _, row in df.iterrows():
        spot    = float(row["spot"])
        strike  = float(row["strike"])
        vol_eff = float(row["vol_eff"])
        tau     = float(row["tau_minutes"])
        pm      = float(row["p_market"])
        won_yes = int(row["resolved_yes"]) == 1

        comp_pup_raw = row.get("composite_p_up")
        p_up = float(comp_pup_raw) if pd.notna(comp_pup_raw) else 0.504

        p_yes = compute_p_yes(spot, strike, vol_eff, tau, p_up, cfg.k_drift_yes)
        if p_yes is not None:
            rows.append({"p_yes": p_yes, "pm": pm, "resolved_yes": won_yes})

    return pd.DataFrame(rows)


def fit_isotonic(cal_df: pd.DataFrame) -> IsotonicRegression:
    """Fit isotonic regression: p_yes_model -> actual win rate."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_df["p_yes"].values, cal_df["resolved_yes"].values)
    return iso


def print_calibration_table(cal_df: pd.DataFrame, iso: IsotonicRegression = None,
                             label: str = ""):
    """Show model vs actual WR by p_yes bucket, before and after calibration."""
    print(f"\n  Calibration table {label}:")
    iso_header = "  {'iso_model':>10}  {'iso_bias':>9}" if iso else ""
    print(f"  {'p_yes bucket':>14}  {'n':>5}  {'actual_WR':>10}  "
          f"{'raw_model':>10}  {'raw_bias':>9}"
          + (f"  {'iso_model':>10}  {'iso_bias':>9}" if iso else ""))
    bins = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    for lo, hi in zip(bins, bins[1:]):
        sub = cal_df[(cal_df["p_yes"] >= lo) & (cal_df["p_yes"] < hi)]
        if sub.empty:
            continue
        wr       = sub["resolved_yes"].mean()
        raw_avg  = sub["p_yes"].mean()
        raw_bias = raw_avg - wr
        line = (f"  [{lo:.2f},{hi:.2f})    {len(sub):>5}  {wr:>10.1%}  "
                f"{raw_avg:>10.3f}  {raw_bias:>+9.3f}")
        if iso is not None:
            iso_avg  = iso.predict(sub["p_yes"].values).mean()
            iso_bias = iso_avg - wr
            line += f"  {iso_avg:>10.3f}  {iso_bias:>+9.3f}"
        print(line)


# ── Simulation loop ────────────────────────────────────────────────────────
def run_simulation(df: pd.DataFrame, cfg: SimConfig, iso: IsotonicRegression = None):
    """
    Run the simulation. If iso is provided, calibrated p_yes values are used
    instead of raw model output — false edge in miscalibrated zones disappears.
    """
    trades = []

    for dt, group in df.groupby("decision_time", sort=True):
        best = evaluate_slot(group, cfg, iso=iso)
        if best is None:
            continue

        pnl = trade_pnl(best["bet"], best["side"], best["pm"], best["won"])
        trades.append({
            "decision_time": dt,
            "side":          best["side"],
            "pm":            best["pm"],
            "p_model":       best["p_model"],
            "net_edge":      best["net_edge"],
            "won":           best["won"],
            "pnl":           pnl,
            "offset":        best["offset"],
            "bet":           best["bet"],
        })

    return trades


# ── Reporting ──────────────────────────────────────────────────────────────
def _summarize(tdf: pd.DataFrame, label: str):
    if tdf.empty:
        print(f"  {label}: no trades")
        return

    n   = len(tdf)
    wr  = tdf["won"].mean()
    pnl = tdf["pnl"].sum()
    yes = tdf[tdf["side"] == "yes"]
    no  = tdf[tdf["side"] == "no"]

    print(f"\n  ── {label} ──")
    print(f"  Total:  n={n:4d}  WR={wr:.1%}  PnL=${pnl:+.0f}")
    if not yes.empty:
        print(f"  YES:    n={len(yes):4d}  WR={yes['won'].mean():.1%}  PnL=${yes['pnl'].sum():+.0f}")
    if not no.empty:
        print(f"  NO:     n={len(no):4d}  WR={no['won'].mean():.1%}  PnL=${no['pnl'].sum():+.0f}")

    # pm buckets
    print(f"\n  {'pm bucket':>14}  {'n':>4}  {'WR':>6}  {'PnL':>8}")
    for lo, hi in [(0, .15), (.15, .25), (.25, .35), (.35, .50),
                   (.50, .65), (.65, .80), (.80, 1.)]:
        sub = tdf[(tdf["pm"] >= lo) & (tdf["pm"] < hi)]
        if sub.empty:
            continue
        print(f"  [{lo:.2f},{hi:.2f})   {len(sub):>4}  {sub['won'].mean():>6.1%}  {sub['pnl'].sum():>+8.0f}")


def report(trades: list[dict], cfg: SimConfig, train_frac: float = 0.60):
    print(f"\n{'='*60}")
    print(f"Config: {cfg.label}")
    print(f"  k_drift_yes={cfg.k_drift_yes}  k_drift_no={cfg.k_drift_no}")
    print(f"  pm_floor_yes={cfg.pm_floor_yes}  pm_ceil_no={cfg.pm_ceil_no}")
    print(f"  otm_yes_gate={'ON' if cfg.otm_yes_gate else 'OFF'}")
    print(f"  bankroll=${FLAT_BANKROLL:.0f} (flat, non-compounding)")

    if not trades:
        print("  No trades generated.")
        return

    tdf = pd.DataFrame(trades).reset_index(drop=True)
    split = int(len(tdf) * train_frac)
    _summarize(tdf.iloc[:split], f"TRAIN ({train_frac:.0%})")
    _summarize(tdf.iloc[split:], f"TEST  ({1-train_frac:.0%} OOS)")
    print()


# ── Sweep ──────────────────────────────────────────────────────────────────
def run_pm_sweep(df: pd.DataFrame, base_cfg: SimConfig,
                 floors: list[float], train_frac: float):
    print(f"\npm_floor_yes sweep — k_yes={base_cfg.k_drift_yes}, k_no={base_cfg.k_drift_no}")
    print(f"\n{'floor':>8}  "
          f"{'tr_n':>6}  {'tr_wr':>7}  {'tr_pnl':>9}  "
          f"{'te_n':>6}  {'te_wr':>7}  {'te_pnl':>9}  "
          f"{'te_delta':>9}")

    baseline_te_pnl = None
    for floor in floors:
        cfg = SimConfig(
            k_drift_yes=base_cfg.k_drift_yes,
            k_drift_no=base_cfg.k_drift_no,
            pm_floor_yes=floor,
            pm_ceil_no=base_cfg.pm_ceil_no,
            label=f"pm_floor={floor}",
        )
        trades = run_simulation(df, cfg)
        if not trades:
            print(f"  {floor:.2f}    (no trades)")
            continue

        tdf   = pd.DataFrame(trades).reset_index(drop=True)
        split = int(len(tdf) * train_frac)
        tr    = tdf.iloc[:split]
        te    = tdf.iloc[split:]

        tr_wr  = tr["won"].mean()  if not tr.empty else 0.0
        tr_pnl = tr["pnl"].sum()   if not tr.empty else 0.0
        te_wr  = te["won"].mean()  if not te.empty else 0.0
        te_pnl = te["pnl"].sum()   if not te.empty else 0.0

        if baseline_te_pnl is None:
            baseline_te_pnl = te_pnl
            delta_str = "—"
        else:
            delta_str = f"{te_pnl - baseline_te_pnl:+.0f}"

        print(f"  {floor:.2f}    "
              f"{len(tr):>6}  {tr_wr:>7.1%}  {tr_pnl:>+9.0f}  "
              f"{len(te):>6}  {te_wr:>7.1%}  {te_pnl:>+9.0f}  "
              f"{delta_str:>9}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BTC replay simulator")
    parser.add_argument("--csv",           default="results/paper_trades.csv")
    parser.add_argument("--archives",      nargs="*", default=[],
                        help="Additional archive CSVs to merge (space-separated)")
    parser.add_argument("--k-drift-yes",   type=float, default=K_DRIFT_YES_DEFAULT)
    parser.add_argument("--k-drift-no",    type=float, default=K_DRIFT_NO_DEFAULT)
    parser.add_argument("--pm-floor-yes",  type=float, default=0.04)
    parser.add_argument("--pm-ceil-no",    type=float, default=0.96)
    parser.add_argument("--train-frac",    type=float, default=0.60)
    parser.add_argument("--sweep-pm",      action="store_true",
                        help="Sweep pm_floor_yes across common values")
    parser.add_argument("--otm-yes-gate",  action="store_true",
                        help="Enable OTM YES momentum exhaustion gate (pm<0.35)")
    parser.add_argument("--calibrate",     action="store_true",
                        help="Fit isotonic calibration on train set, evaluate on test")
    parser.add_argument("--save-cal",      default="models/btc_iso_cal.pkl",
                        help="Path to save fitted isotonic calibrator")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading {csv_path.name}...")
    df = load_data(csv_path, extra_csvs=args.archives)

    slots = df["decision_time"].nunique()
    print(f"  {len(df):,} resolved candidate rows  |  {slots:,} decision slots")
    print(f"  Date range: {df['decision_time'].min().date()} → "
          f"{df['decision_time'].max().date()}")

    base_cfg = SimConfig(
        k_drift_yes=args.k_drift_yes,
        k_drift_no=args.k_drift_no,
        pm_floor_yes=args.pm_floor_yes,
        pm_ceil_no=args.pm_ceil_no,
        otm_yes_gate=args.otm_yes_gate,
        label=(f"k_yes={args.k_drift_yes}, k_no={args.k_drift_no}, "
               f"pm_floor={args.pm_floor_yes}"
               + (", otm_yes_gate=ON" if args.otm_yes_gate else "")),
    )

    if args.sweep_pm:
        floors = [0.04, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
        run_pm_sweep(df, base_cfg, floors, args.train_frac)

    elif args.calibrate:
        # Split data by decision_time order (not by trade index)
        all_slots = sorted(df["decision_time"].unique())
        split_slot = all_slots[int(len(all_slots) * args.train_frac)]
        df_train = df[df["decision_time"] < split_slot]
        df_test  = df[df["decision_time"] >= split_slot]

        print(f"\n  Train slots: {df_train['decision_time'].nunique()}  "
              f"({df_train['decision_time'].min().date()} → "
              f"{df_train['decision_time'].max().date()})")
        print(f"  Test  slots: {df_test['decision_time'].nunique()}  "
              f"({df_test['decision_time'].min().date()} → "
              f"{df_test['decision_time'].max().date()})")

        # Collect calibration data from ALL train candidates
        print("\nCollecting calibration data from train set...")
        cal_train = collect_calibration_data(df_train, base_cfg)
        print(f"  {len(cal_train):,} YES candidate observations")

        # Show raw calibration on train
        print_calibration_table(cal_train, label="(train, before calibration)")

        # Fit isotonic
        iso = fit_isotonic(cal_train)

        # Show calibration after fitting on train
        print_calibration_table(cal_train, iso=iso, label="(train, after calibration)")

        # Save calibrator
        save_path = Path(args.save_cal)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(iso, f)
        print(f"\n  Calibrator saved → {save_path}")

        # Run simulation: raw vs calibrated on TEST set only
        print(f"\nSimulating on test set...")
        trades_raw = run_simulation(df_test, base_cfg, iso=None)
        trades_cal = run_simulation(df_test, base_cfg, iso=iso)

        traw = pd.DataFrame(trades_raw) if trades_raw else pd.DataFrame()
        tcal = pd.DataFrame(trades_cal) if trades_cal else pd.DataFrame()

        print(f"\n{'='*60}")
        print(f"TEST SET COMPARISON — raw p_model vs isotonic calibrated")
        print(f"{'':>30}  {'raw':>10}  {'calibrated':>12}  {'delta':>8}")

        def row(label, raw_val, cal_val, fmt=".0f"):
            delta = cal_val - raw_val
            print(f"  {label:<28}  {raw_val:>10{fmt}}  {cal_val:>12{fmt}}  {delta:>+8{fmt}}")

        if not traw.empty and not tcal.empty:
            row("n_trades",    len(traw),               len(tcal),               fmt="d")
            row("win_rate",    traw['won'].mean()*100,   tcal['won'].mean()*100,  fmt=".1f")
            row("PnL ($)",     traw['pnl'].sum(),        tcal['pnl'].sum(),       fmt=".0f")
            n_yes_raw = (traw['side']=='yes').sum()
            n_yes_cal = (tcal['side']=='yes').sum()
            row("n_yes",       n_yes_raw,                n_yes_cal,               fmt="d")
            row("n_no",        (traw['side']=='no').sum(),(tcal['side']=='no').sum(), fmt="d")

            print(f"\n  pm bucket breakdown (test):")
            print(f"  {'bucket':>12}  {'raw_n':>6}  {'raw_wr':>7}  {'raw_pnl':>9}  "
                  f"{'cal_n':>6}  {'cal_wr':>7}  {'cal_pnl':>9}")
            for lo, hi in [(0,.15),(.15,.25),(.25,.35),(.35,.50),(.50,.65),(.65,.80),(.80,1.)]:
                sr = traw[(traw['pm']>=lo)&(traw['pm']<hi)]
                sc = tcal[(tcal['pm']>=lo)&(tcal['pm']<hi)]
                if sr.empty and sc.empty: continue
                rn, rwr, rpnl = len(sr), sr['won'].mean() if len(sr) else 0, sr['pnl'].sum()
                cn, cwr, cpnl = len(sc), sc['won'].mean() if len(sc) else 0, sc['pnl'].sum()
                print(f"  [{lo:.2f},{hi:.2f})  {rn:>6}  {rwr:>7.1%}  {rpnl:>+9.0f}  "
                      f"{cn:>6}  {cwr:>7.1%}  {cpnl:>+9.0f}")

        # Also show calibration table on test candidates
        cal_test = collect_calibration_data(df_test, base_cfg)
        print_calibration_table(cal_test, iso=iso, label="(test OOS, calibrator applied)")
        print()

    else:
        trades = run_simulation(df, base_cfg)
        report(trades, base_cfg, args.train_frac)


if __name__ == "__main__":
    main()
