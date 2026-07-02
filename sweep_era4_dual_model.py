#!/usr/bin/env python3
"""
sweep_era4_dual_model.py

Comprehensive YES/NO drift model sweep on Era 4 btc_scan_archive data (May 18-26).
Independently calibrates the YES model and NO model.

Methodology:
- Use btc_scan_archive.csv (all resolved rows, not just trades taken)
- Join mu6h/mu12h/mu24h/regime_z from 1h BTC data
- Test 11 drift formula candidates for YES, 11 for NO
- Sweep k in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] per formula
- Simulate per-slot: groupby logged_at → pick best candidate → Kelly size → PnL
- One bet per slot (best YES or best NO by net edge)
- Flat $1000 bankroll, non-compounding (per feedback)
- Objective: maximize P&L, not log-loss
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
ARCHIVE_PATH = ROOT / "results" / "btc_scan_archive.csv"
DATA_DIR     = ROOT / "data"

BANKROLL   = 1000.0   # flat non-compounding
KELLY_MULT = 0.30
KELLY_CAP  = 0.06
FEE_RATE   = 0.07
SLIPPAGE   = 0.003
SPREAD     = 0.005
G3_MIN     = 0.02

P_BAND  = (0.05, 0.95)
TAU_MIN = 20.0
TAU_MAX = 120.0
EPS     = 1e-7

K_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def load_1h_data():
    f = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
               key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def compute_mu_series(df1h):
    lr    = np.log(df1h["close"] / df1h["close"].shift(1))
    mu6   = lr.rolling(6,  min_periods=1).mean()
    mu12  = lr.rolling(12, min_periods=1).mean()
    mu24  = lr.rolling(24, min_periods=1).mean()
    ewm_m = lr.ewm(span=12).mean()
    ewm_s = lr.ewm(span=24).std()
    rz    = np.clip(ewm_m / ewm_s.replace(0, np.nan), -3.0, 3.0).fillna(0.0)
    return pd.DataFrame({"mu6": mu6, "mu12": mu12, "mu24": mu24, "regime_z": rz},
                        index=df1h.index)


def fee(pm):
    return FEE_RATE * np.minimum(pm, 1.0 - pm)


def kelly_bet(p_model, pm, side):
    b = (1.0 - pm) / pm if side == "yes" else pm / (1.0 - pm)
    if b <= 0:
        return 0.0
    kf = max(0.0, (b * p_model - (1.0 - p_model)) / b)
    return min(kf * KELLY_MULT, KELLY_CAP) * BANKROLL


def trade_pnl(bet, pm, side, won):
    if bet <= 0:
        return 0.0
    f = FEE_RATE * min(pm, 1.0 - pm)
    if side == "yes":
        return (bet * (1.0 - pm) / pm - f * bet / pm) if won else -bet
    else:
        return (bet * pm / (1.0 - pm) - f * bet / (1.0 - pm)) if won else -bet


def simulate_yes(arc, z_drift_arr):
    """Per-slot YES simulation. arc must have z_strike, p_market, fee_col, resolved_yes."""
    p_yes = np.clip(1.0 - norm.cdf(arc["z_strike"].values - z_drift_arr), EPS, 1 - EPS)
    net   = p_yes - arc["p_market"].values - arc["fee_col"].values - SLIPPAGE - SPREAD
    arc2  = arc.copy()
    arc2["p_yes"] = p_yes
    arc2["net"]   = net

    eligible = arc2[net >= G3_MIN].copy()
    trades = []
    for _, g in eligible.groupby("slot"):
        best = g.loc[g["net"].idxmax()]
        pm      = float(best["p_market"])
        p_model = float(best["p_yes"])
        won     = int(best["resolved_yes"]) == 1
        bet     = kelly_bet(p_model, pm, "yes")
        trades.append((won, trade_pnl(bet, pm, "yes", won)))
    return trades


def simulate_no(arc, z_drift_arr):
    """Per-slot NO simulation."""
    p_yes = np.clip(1.0 - norm.cdf(arc["z_strike"].values - z_drift_arr), EPS, 1 - EPS)
    p_no  = 1.0 - p_yes
    net   = p_no - (1.0 - arc["p_market"].values) - arc["fee_col"].values - SLIPPAGE - SPREAD

    arc2  = arc.copy()
    arc2["p_no"] = p_no
    arc2["net"]  = net

    eligible = arc2[net >= G3_MIN].copy()
    trades = []
    for _, g in eligible.groupby("slot"):
        best = g.loc[g["net"].idxmax()]
        pm      = float(best["p_market"])
        p_model = float(best["p_no"])
        won     = int(best["resolved_yes"]) == 0
        bet     = kelly_bet(p_model, pm, "no")
        trades.append((won, trade_pnl(bet, pm, "no", won)))
    return trades


def report(trades):
    if not trades:
        return 0, float("nan"), 0.0
    n    = len(trades)
    wins = sum(w for w, _ in trades)
    pnl  = sum(p for _, p in trades)
    return n, wins / n, pnl


def main():
    print("Loading scan archive...")
    arc = pd.read_csv(ARCHIVE_PATH, low_memory=False)
    arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True)
    arc = arc[arc["resolved_yes"].notna()].copy()
    arc["resolved_yes"] = arc["resolved_yes"].astype(float)
    print(f"  Resolved rows: {len(arc):,}  ({arc['logged_at'].min().date()} → {arc['logged_at'].max().date()})")

    print("Loading 1h BTC data and computing mu series...")
    df1h  = load_1h_data()
    mu_df = compute_mu_series(df1h)

    arc["bar_ts"] = arc["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
    arc = arc.join(
        mu_df.rename(columns={"mu6": "mu6h", "mu12": "mu12h", "mu24": "mu24h", "regime_z": "regime_z"}),
        on="bar_ts", how="left"
    )

    for col in ["vol_eff", "tau_minutes", "p_market", "spot", "strike",
                "composite_p_up", "composite_trend", "composite_rev",
                "mu6h", "mu12h", "mu24h", "regime_z"]:
        arc[col] = pd.to_numeric(arc[col], errors="coerce")

    mask = (
        arc["vol_eff"].notna() & (arc["vol_eff"] > 0) &
        arc["mu6h"].notna() &
        arc["tau_minutes"].between(TAU_MIN, TAU_MAX) &
        arc["p_market"].between(*P_BAND) &
        arc["spot"].notna()  & (arc["spot"] > 0) &
        arc["strike"].notna() & (arc["strike"] > 0)
    )
    arc = arc[mask].copy()
    print(f"  After filters: {len(arc):,} rows")

    arc["sigma_tau"]     = arc["vol_eff"] * np.sqrt(arc["tau_minutes"])
    arc["z_strike"]      = np.log(arc["strike"] / arc["spot"]) / arc["sigma_tau"]
    arc["fee_col"]       = fee(arc["p_market"].values)
    arc["composite_p_up"] = arc["composite_p_up"].clip(0.01, 0.99)
    arc["pup_z"]         = norm.ppf(arc["composite_p_up"].values)
    arc["composite_trend"] = arc["composite_trend"].fillna(0)
    arc["composite_rev"]   = arc["composite_rev"].fillna(0)
    arc["regime_z"]        = arc["regime_z"].fillna(0)
    arc["slot"]            = arc["logged_at"]

    print(f"  Unique slots: {arc['slot'].nunique()}")

    sq  = np.sqrt(arc["tau_minutes"].values / 60.0)
    t60 = arc["tau_minutes"].values / 60.0
    st  = arc["sigma_tau"].values
    m6  = arc["mu6h"].values
    m12 = arc["mu12h"].values
    m24 = arc["mu24h"].values
    rz  = arc["regime_z"].values
    ct  = arc["composite_trend"].values
    cr  = arc["composite_rev"].values
    pz  = arc["pup_z"].values

    def _f(k, arr):
        return k * arr

    FORMULAS = {
        "no_drift":        ([0.0],      lambda k: np.zeros(len(arc))),
        "mu6_24":          (K_VALUES,   lambda k: k * (m6 + m24) * t60 / st),
        "mu_all":          (K_VALUES,   lambda k: k * (m6 + m12 + m24) * t60 / st),
        "rz":              (K_VALUES,   lambda k: k * rz * sq),
        "mu6_24_rz":       (K_VALUES,   lambda k: k * ((m6 + m24) * t60 / st + rz * sq)),
        "mu_all_rz":       (K_VALUES,   lambda k: k * ((m6 + m12 + m24) * t60 / st + rz * sq)),
        "ct_fixed":        ([1.0],      lambda k: (ct / 5.0) * 0.15 * sq),
        "cr":              (K_VALUES,   lambda k: k * cr * sq),
        "ct_cr":           (K_VALUES,   lambda k: k * (ct + cr) / 5.0 * sq),
        "pup":             (K_VALUES,   lambda k: k * pz * sq),
        "mu6_24_rz_ct":    (K_VALUES,   lambda k: k * ((m6 + m24) * t60 / st + rz * sq) + (ct / 5.0) * 0.15 * sq),
    }

    header = f"  {'Formula':<20} {'k':>5}  {'n':>5}  {'WR':>6}  {'PnL':>10}"
    sep    = "  " + "-" * 55

    # ── YES sweep ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("YES MODEL SWEEP  (per-slot: pick best YES candidate by net edge)")
    print("=" * 70)
    print(header); print(sep)

    yes_results = []
    for fname, (ks, fn) in FORMULAS.items():
        for k in ks:
            zd = fn(k)
            trades = simulate_yes(arc, zd)
            n, wr, pnl = report(trades)
            yes_results.append({"formula": fname, "k": k, "n": n, "wr": wr, "pnl": pnl})
            wr_s = f"{wr:.3f}" if not math.isnan(wr) else "  nan"
            print(f"  {fname:<20} {k:>5.2f}  {n:>5}  {wr_s:>6}  {pnl:>+10.2f}")

    yes_results.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n  TOP 10 YES formulas by P&L:")
    for r in yes_results[:10]:
        wr_s = f"{r['wr']:.3f}" if not math.isnan(r["wr"]) else "nan"
        print(f"    {r['formula']:<20} k={r['k']:.2f}  n={r['n']:3d}  WR={wr_s}  PnL={r['pnl']:+.2f}")

    # ── NO sweep ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NO MODEL SWEEP  (per-slot: pick best NO candidate by net edge)")
    print("=" * 70)
    print(header); print(sep)

    no_results = []
    for fname, (ks, fn) in FORMULAS.items():
        for k in ks:
            zd = fn(k)
            trades = simulate_no(arc, zd)
            n, wr, pnl = report(trades)
            no_results.append({"formula": fname, "k": k, "n": n, "wr": wr, "pnl": pnl})
            wr_s = f"{wr:.3f}" if not math.isnan(wr) else "  nan"
            print(f"  {fname:<20} {k:>5.2f}  {n:>5}  {wr_s:>6}  {pnl:>+10.2f}")

    no_results.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n  TOP 10 NO formulas by P&L:")
    for r in no_results[:10]:
        wr_s = f"{r['wr']:.3f}" if not math.isnan(r["wr"]) else "nan"
        print(f"    {r['formula']:<20} k={r['k']:.2f}  n={r['n']:3d}  WR={wr_s}  PnL={r['pnl']:+.2f}")

    best_yes = yes_results[0]
    best_no  = no_results[0]

    # ── Combined: best YES + best NO per slot ──────────────────────────────
    print("\n" + "=" * 70)
    print(f"COMBINED (1 bet/slot): "
          f"YES={best_yes['formula']} k={best_yes['k']:.2f}  +  "
          f"NO={best_no['formula']} k={best_no['k']:.2f}")
    print("=" * 70)

    _, fn_yes = FORMULAS[best_yes["formula"]]
    _, fn_no  = FORMULAS[best_no["formula"]]
    zd_yes = fn_yes(best_yes["k"])
    zd_no  = fn_no(best_no["k"])

    py_yes = np.clip(1.0 - norm.cdf(arc["z_strike"].values - zd_yes), EPS, 1 - EPS)
    py_no  = np.clip(1.0 - norm.cdf(arc["z_strike"].values - zd_no),  EPS, 1 - EPS)
    pm_arr = arc["p_market"].values
    f_arr  = arc["fee_col"].values

    net_yes = py_yes - pm_arr - f_arr - SLIPPAGE - SPREAD
    net_no  = (1.0 - py_no) - (1.0 - pm_arr) - f_arr - SLIPPAGE - SPREAD

    arc2 = arc.copy()
    arc2["net_yes"] = net_yes
    arc2["net_no"]  = net_no
    arc2["py_yes"]  = py_yes
    arc2["py_no"]   = 1.0 - py_no

    comb_yes = []; comb_no = []
    for slot, g in arc2.groupby("slot"):
        y_elig = g[g["net_yes"] >= G3_MIN]
        n_elig = g[g["net_no"]  >= G3_MIN]
        has_y  = len(y_elig) > 0
        has_n  = len(n_elig) > 0
        if not has_y and not has_n:
            continue
        best_y_net = y_elig["net_yes"].max() if has_y else -9e9
        best_n_net = n_elig["net_no"].max()  if has_n else -9e9

        if best_y_net >= best_n_net and has_y:
            best   = y_elig.loc[y_elig["net_yes"].idxmax()]
            pm     = float(best["p_market"])
            p_mod  = float(best["py_yes"])
            won    = int(best["resolved_yes"]) == 1
            bet    = kelly_bet(p_mod, pm, "yes")
            comb_yes.append((won, trade_pnl(bet, pm, "yes", won)))
        elif has_n:
            best   = n_elig.loc[n_elig["net_no"].idxmax()]
            pm     = float(best["p_market"])
            p_mod  = float(best["py_no"])
            won    = int(best["resolved_yes"]) == 0
            bet    = kelly_bet(p_mod, pm, "no")
            comb_no.append((won, trade_pnl(bet, pm, "no", won)))

    c_all = comb_yes + comb_no
    def _rpt(t, label):
        if not t: print(f"  {label}: no trades"); return
        n = len(t); w = sum(x[0] for x in t); p = sum(x[1] for x in t)
        print(f"  {label}: n={n:3d}  WR={w/n:.3f}  PnL={p:+.2f}")
    _rpt(comb_yes, "YES")
    _rpt(comb_no,  "NO ")
    cn, cwr, cpnl = report(c_all)
    print(f"  Total: n={cn}  WR={cwr:.3f}  PnL={cpnl:+.2f}")

    # ── Baseline: current production model ─────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE: current production")
    print("  YES = mu6_24_rz_ct k=1.0   NO = mu_all_rz k=1.0")
    print("=" * 70)

    _, fn_cur_yes = FORMULAS["mu6_24_rz_ct"]
    _, fn_cur_no  = FORMULAS["mu_all_rz"]
    zd_cur_yes = fn_cur_yes(1.0)
    zd_cur_no  = fn_cur_no(1.0)

    py_cur_yes = np.clip(1.0 - norm.cdf(arc["z_strike"].values - zd_cur_yes), EPS, 1 - EPS)
    py_cur_no  = np.clip(1.0 - norm.cdf(arc["z_strike"].values - zd_cur_no),  EPS, 1 - EPS)
    net_cur_yes = py_cur_yes - pm_arr - f_arr - SLIPPAGE - SPREAD
    net_cur_no  = (1.0 - py_cur_no) - (1.0 - pm_arr) - f_arr - SLIPPAGE - SPREAD

    arc3 = arc.copy()
    arc3["net_yes"] = net_cur_yes
    arc3["net_no"]  = net_cur_no
    arc3["py_yes"]  = py_cur_yes
    arc3["py_no"]   = 1.0 - py_cur_no

    cur_yes_t = []; cur_no_t = []
    for slot, g in arc3.groupby("slot"):
        y_elig = g[g["net_yes"] >= G3_MIN]
        n_elig = g[g["net_no"]  >= G3_MIN]
        has_y  = len(y_elig) > 0
        has_n  = len(n_elig) > 0
        if not has_y and not has_n:
            continue
        best_y_net = y_elig["net_yes"].max() if has_y else -9e9
        best_n_net = n_elig["net_no"].max()  if has_n else -9e9

        if best_y_net >= best_n_net and has_y:
            best  = y_elig.loc[y_elig["net_yes"].idxmax()]
            pm    = float(best["p_market"])
            p_mod = float(best["py_yes"])
            won   = int(best["resolved_yes"]) == 1
            bet   = kelly_bet(p_mod, pm, "yes")
            cur_yes_t.append((won, trade_pnl(bet, pm, "yes", won)))
        elif has_n:
            best  = n_elig.loc[n_elig["net_no"].idxmax()]
            pm    = float(best["p_market"])
            p_mod = float(best["py_no"])
            won   = int(best["resolved_yes"]) == 0
            bet   = kelly_bet(p_mod, pm, "no")
            cur_no_t.append((won, trade_pnl(bet, pm, "no", won)))

    cur_all = cur_yes_t + cur_no_t
    _rpt(cur_yes_t, "YES")
    _rpt(cur_no_t,  "NO ")
    byn, bwr, bpnl = report(cur_all)
    print(f"  Total: n={byn}  WR={bwr:.3f}  PnL={bpnl:+.2f}")
    print(f"\n  Delta (optimal vs current): {cpnl - bpnl:+.2f}")

    # ── Final recommendations ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"  Best YES formula: {best_yes['formula']}  k={best_yes['k']:.2f}  "
          f"(PnL={best_yes['pnl']:+.2f}  n={best_yes['n']}  WR={best_yes['wr']:.3f})")
    print(f"  Best NO  formula: {best_no['formula']}  k={best_no['k']:.2f}  "
          f"(PnL={best_no['pnl']:+.2f}  n={best_no['n']}  WR={best_no['wr']:.3f})")
    print(f"  Combined P&L:     {cpnl:+.2f}")
    print(f"  Baseline P&L:     {bpnl:+.2f}")
    print(f"  Delta:            {cpnl - bpnl:+.2f}")


if __name__ == "__main__":
    main()
