"""
simulate_model.py — Gate-replicating walk-forward model comparison.

Applies the BTC gate stack (reconstructed from btc_scan_archive.csv signals)
to compare two model configs side by side. The only thing that differs between
configs is what you're testing (drift formula, vol model, etc.). All gates,
Kelly sizing, and fee structure are identical for both.

Walk-forward: chronological day-by-day P&L. No look-ahead bias.

Usage:
    python3 simulate_model.py

Edit BASELINE_CONFIG and VARIANT_CONFIG at the bottom to test different changes.

KNOWN GATE OMISSIONS (same for both configs — delta is still valid):
  - smc_gate: requires smc_4h/smc_1h/supply_zone signals not logged in archive
  - btc_vol_gate (|z| > 2×vol_factor): vol_factor not in archive; use fixed threshold
  - streak_gate: requires 30m sequential close reconstruction
  - btc_ema0_itm_gate: minor; skipped
  - Most interaction rescues within smc_gate

All other major gates are implemented using signals from the archive.
"""
import warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from arch import arch_model

warnings.filterwarnings("ignore")

DATA_DIR = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data")
ARC_PATH = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/btc_scan_archive.csv")

# ── constants (match live model) ─────────────────────────────────────────────
BANKROLL     = 1000.0
KELLY_MULT   = 0.30
KELLY_CAP    = 0.06
FEE_RATE     = 0.07
EPS          = 1e-7
GARCH_WINDOW = 500


# ── signal helpers ────────────────────────────────────────────────────────────

def _stoch_k(h, l, c, p=14):
    return (c - l.rolling(p).min()) / (h.rolling(p).max() - l.rolling(p).min()).replace(0, np.nan) * 100

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def _macd_hist(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    return m - m.ewm(span=sig).mean()


# ── load and join all signals ─────────────────────────────────────────────────

def load_data():
    print("Loading btc_scan_archive.csv...")
    arc = pd.read_csv(ARC_PATH, low_memory=False)
    arc["logged_at"] = pd.to_datetime(arc["logged_at"], utc=True)
    arc = arc[arc["resolved_yes"].notna()].copy()
    arc["resolved_yes"] = arc["resolved_yes"].astype(float)

    num_cols = ["vol_eff", "tau_minutes", "p_market", "spot", "strike", "offset_pct",
                "composite_trend", "composite_rev", "composite_p_up",
                "stoch_k", "vpin_score", "liq_score", "liq_bias", "vol_score",
                "funding_bias", "confirmation_score", "rvol_1h", "adx_1h",
                "vwap_stretch_score", "ema_stretch_score", "ema_stack_bias",
                "chg_30m", "chg_5m", "chg_10m", "body_15m", "bp_5m", "dir_15m"]
    for col in num_cols:
        if col in arc.columns:
            arc[col] = pd.to_numeric(arc[col], errors="coerce")

    mask = (
        arc["vol_eff"].notna() & (arc["vol_eff"] > 0) &
        arc["tau_minutes"].between(20, 120) &
        arc["p_market"].between(0.05, 0.95) &
        arc["spot"].notna() & (arc["spot"] > 0) &
        arc["strike"].notna() & (arc["strike"] > 0)
    )
    arc = arc[mask].copy()
    arc["sigma_tau"]   = arc["vol_eff"] * np.sqrt(arc["tau_minutes"])
    arc["z_strike"]    = np.log(arc["strike"] / arc["spot"]) / arc["sigma_tau"]
    arc["bar_ts"]      = arc["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
    arc["hour_utc"]    = arc["logged_at"].dt.hour
    arc["date"]        = arc["logged_at"].dt.date
    arc["offset_pct"]  = arc["offset_pct"].fillna(
        (arc["strike"] / arc["spot"] - 1) * 100)
    print(f"  Filtered rows: {len(arc):,}  slots: {arc['logged_at'].nunique():,}")

    # ── 1h signals ────────────────────────────────────────────────────────────
    print("Computing 1h signals...")
    f1h = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h = pd.read_parquet(f1h)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    df1h = df1h[df1h.index.year > 1970].sort_index()

    lr       = np.log(df1h["close"] / df1h["close"].shift(1))
    vol_24h  = lr.rolling(24, min_periods=4).std()
    vol_168h = lr.rolling(168, min_periods=24).std()
    rvol_inv = (vol_168h / vol_24h.replace(0, np.nan)).clip(0.3, 2.0)  # high = quiet

    mu6h  = lr.rolling(6,  min_periods=1).mean()
    mu12h = lr.rolling(12, min_periods=1).mean()
    mu24h = lr.rolling(24, min_periods=1).mean()
    ewm_m = lr.ewm(span=12).mean()
    ewm_s = lr.ewm(span=24).std()
    rz    = np.clip(ewm_m / ewm_s.replace(0, np.nan), -3.0, 3.0).fillna(0.0)

    # ── 4h signals (resampled from 1h) ───────────────────────────────────────
    print("Computing 4h signals...")
    df4h = df1h[["open","high","low","close","volume"]].resample(
        "4h", label="right", closed="right"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["close"])

    stoch_4h    = _stoch_k(df4h["high"], df4h["low"], df4h["close"])
    rsi_4h      = _rsi(df4h["close"])
    macd_4h     = _macd_hist(df4h["close"])
    macd_std_4h = macd_4h.rolling(250, min_periods=50).std().replace(0, np.nan)
    macd_norm_4h = (macd_4h / macd_std_4h).clip(-3, 3).fillna(0)

    ema20_4h    = df4h["close"].ewm(span=20, adjust=False).mean()
    ema20_slp   = (ema20_4h.diff() / ema20_4h) * 100
    ema20_slp_z = (ema20_slp /
                   ema20_slp.rolling(100, min_periods=20).std().replace(0, np.nan)
                   ).clip(-3, 3)

    sig_4h = pd.DataFrame({
        "stoch_4h": stoch_4h, "rsi_4h": rsi_4h,
        "macd_norm_4h": macd_norm_4h, "ema20_slope_z": ema20_slp_z,
    }, index=df4h.index).reindex(df1h.index, method="ffill")

    # ── GARCH ratio ───────────────────────────────────────────────────────────
    print("Computing GARCH ratios (this takes ~2 min)...")
    garch_ratio = pd.Series(np.nan, index=df1h.index)
    lr_arr = lr.values
    for i in range(GARCH_WINDOW, len(lr_arr), 24):
        window = lr_arr[i - GARCH_WINDOW:i]
        if np.isnan(window).sum() > 10: continue
        try:
            wc  = window[~np.isnan(window)] * 100
            res = arch_model(wc, vol="Garch", p=1, q=1,
                             mean="Zero", dist="normal").fit(disp="off", show_warning=False)
            h_t   = float(res.conditional_volatility[-1]) ** 2
            omega = float(res.params.get("omega", 1e-6))
            alpha = float(res.params.get("alpha[1]", 0))
            beta  = float(res.params.get("beta[1]",  0))
            lr_h  = omega / max(1e-6, 1 - alpha - beta)
            ratio = np.sqrt(h_t / lr_h) if lr_h > 0 else 1.0
            for j in range(i, min(i + 24, len(df1h.index))):
                garch_ratio.iloc[j] = ratio
        except Exception: pass
    garch_ratio = garch_ratio.clip(0.3, 3.0)

    # ── assemble 1h signal frame and join ─────────────────────────────────────
    sig_1h = pd.DataFrame({
        "mu6h": mu6h, "mu12h": mu12h, "mu24h": mu24h, "regime_z": rz,
        "rvol_inv": rvol_inv, "garch_ratio": garch_ratio,
    }, index=df1h.index).join(sig_4h)

    arc = arc.join(sig_1h, on="bar_ts", how="left")
    for col in ["mu6h","mu12h","mu24h","regime_z","rvol_inv","garch_ratio",
                "stoch_4h","rsi_4h","macd_norm_4h","ema20_slope_z"]:
        arc[col] = pd.to_numeric(arc[col], errors="coerce")

    # Fill neutrals for missing signal bars
    fill_neutrals = {
        "rvol_inv": 1.0, "garch_ratio": 1.0, "stoch_4h": 50.0,
        "rsi_4h": 50.0, "macd_norm_4h": 0.0, "ema20_slope_z": 0.0,
        "mu6h": 0.0, "mu12h": 0.0, "mu24h": 0.0, "regime_z": 0.0,
    }
    for col, val in fill_neutrals.items():
        arc[col] = arc[col].fillna(val)

    print(f"  All signals joined. 4h Stoch range: [{arc['stoch_4h'].min():.1f}, {arc['stoch_4h'].max():.1f}]")
    return arc, df1h


# ── drift formulas ────────────────────────────────────────────────────────────

def compute_drift(row_arr, config, side):
    """
    Compute z_drift for each contract row.
    row_arr: dict of numpy arrays keyed by signal name.
    side: "yes" or "no"
    config: dict with "drift_yes"/"drift_no"/"k_yes"/"k_no" keys.
    """
    sq  = np.sqrt(row_arr["tau_minutes"] / 60.0)
    t60 = row_arr["tau_minutes"] / 60.0
    st  = row_arr["sigma_tau"]
    k   = config["k_yes"] if side == "yes" else config["k_no"]
    formula = config["drift_yes"] if side == "yes" else config["drift_no"]

    if formula == "zero":
        return np.zeros(len(sq))

    elif formula == "mu6_24_rz_ct":
        # Current YES formula: (mu6+mu24)×t60/sigma + rz×sq + ct_anchor
        m6  = row_arr["mu6h"]; m24 = row_arr["mu24h"]
        rz  = row_arr["regime_z"]; ct  = row_arr["composite_trend"]
        return k * ((m6 + m24) * t60 / np.clip(st, 1e-8, None) + rz * sq) + (ct / 5.0) * 0.15 * sq

    elif formula == "mu_all_rz":
        # Current NO formula: (mu6+mu12+mu24)×t60/sigma + rz×sq
        m6  = row_arr["mu6h"]; m12 = row_arr["mu12h"]; m24 = row_arr["mu24h"]
        rz  = row_arr["regime_z"]
        return k * ((m6 + m12 + m24) * t60 / np.clip(st, 1e-8, None) + rz * sq)

    elif formula == "stoch_rvol_ema20z":
        # New 3-factor formula: Stoch_4h × RVOL_inv × |EMA20z| × sq
        s4   = (row_arr["stoch_4h"] - 50.0) / 25.0
        rv   = np.clip(row_arr["rvol_inv"], 0.3, 2.0)
        e20z = np.abs(row_arr["ema20_slope_z"])
        return k * s4 * rv * e20z * sq

    else:
        raise ValueError(f"Unknown drift formula: {formula!r}")


# ── gate stack ────────────────────────────────────────────────────────────────
# Each gate returns True = BLOCKED, False = PASSES.
# Rescues return False (unblock) when rescue condition met.

def _g(arc, col, default=0.0):
    """Safe column access with fallback."""
    return arc[col].fillna(default).values if col in arc.columns else np.full(len(arc), default)


def apply_yes_gates(arc):
    """
    Apply all reconstructable BTC YES gates.
    Returns boolean Series: True = this contract is BLOCKED for YES.
    """
    n   = len(arc)
    blocked = np.zeros(n, dtype=bool)

    pm   = arc["p_market"].values
    tau  = arc["tau_minutes"].values
    ema  = _g(arc, "ema_stack_bias")
    rev  = _g(arc, "composite_rev")
    ct   = _g(arc, "composite_trend")
    stch = _g(arc, "stoch_k")
    pup  = _g(arc, "composite_p_up", 0.5)
    vpin = _g(arc, "vpin_score")
    liq  = _g(arc, "liq_score")
    liqb = _g(arc, "liq_bias")
    vols = _g(arc, "vol_score")
    fund = _g(arc, "funding_bias")
    conf = _g(arc, "confirmation_score")
    rvol = _g(arc, "rvol_1h", 1.0)
    adx  = _g(arc, "adx_1h", 30.0)
    vwap = _g(arc, "vwap_stretch_score")
    ems  = _g(arc, "ema_stretch_score")
    c30  = _g(arc, "chg_30m")
    c5   = _g(arc, "chg_5m")
    body = _g(arc, "body_15m")
    bp   = _g(arc, "bp_5m")
    hr   = arc["hour_utc"].values
    gr   = arc["garch_ratio"].values
    rsi4 = arc["rsi_4h"].values
    mcd4 = arc["macd_norm_4h"].values
    off  = arc["offset_pct"].values           # positive = OTM YES
    zs   = arc["z_strike"].values

    # 1. Hour gate: block UTC 13 and 16
    blocked |= np.isin(hr, [13, 16])

    # 2. rvol_gate: rvol_1h < 0.80 → block unless vpin==1 OR liq_bias==1.0
    rv_block = (rvol < 0.80) & (rvol > 0)
    rv_rescue = (vpin == 1) | (liqb == 1.0)
    blocked |= rv_block & ~rv_rescue

    # 3. adx_gate: adx in [20,40) AND ema != -1
    blocked |= (adx >= 20) & (adx < 40) & (ema != -1)

    # 4. garch_highvol_gate: ratio > 1.5, rescue: pm>=0.80 AND tau<45
    gh_rescue = (pm >= 0.80) & (tau < 45)
    blocked |= (gr > 1.5) & ~gh_rescue

    # 5. deepno_neutral_gate: pm<0.35 AND ema==0
    blocked |= (pm < 0.35) & (ema == 0)

    # 6. beardrift_gate: ema==-1 AND rev<=3 AND stoch>=35
    #    Arm 2 (harder): stoch<25 AND OTM → no rescue
    #    Arm 1 rescue: vpin==1 OR ema_stretch==1
    arm2 = (ema == -1) & (rev <= 3) & (stch < 25) & (off > 0)
    arm1 = (ema == -1) & (rev <= 3) & (stch >= 35) & ~arm2
    arm1_rescue = (vpin == 1) | (ems == 1)
    blocked |= arm2
    blocked |= arm1 & ~arm1_rescue

    # 7. otm_neutral_gate: ema==0 AND p_up>=0.60 AND OTM
    #    Rescue (simplified): fund==-1 (strongest rescue; vwap=-1 beardrift rescue)
    otn = (ema == 0) & (pup >= 0.60) & (off > 0)
    otn_rescue = (fund == -1) | ((stch >= 80) & (ct == 0) & (fund == -1))
    blocked |= otn & ~otn_rescue

    # 8. exhaustion_gate: ema==1 AND rev<=-4 AND stoch>=75 AND vwap<=-1
    #    Rescue: fund==-1
    exh = (ema == 1) & (rev <= -4) & (stch >= 75) & (vwap <= -1)
    blocked |= exh & ~(fund == -1)

    # 9. falling_knife_gate: rev>=4 AND chg_30m<-0.20%
    #    Rescue: chg_5m>+0.10% OR deeply ITM (offset<-0.10%)
    fk = (rev >= 4) & (c30 < -0.20)
    fk_rescue = (c5 > 0.10) | (off < -0.10)
    blocked |= fk & ~fk_rescue

    # 10. ema0_stretch2_gate: ema==0 AND vwap_stretch==2
    #     Rescue: stoch>=10 AND ITM (off<=0)
    es2 = (ema == 0) & (vwap == 2)
    es2_rescue = (stch >= 10) & (off <= 0)
    blocked |= es2 & ~es2_rescue

    # 11. liq_cascade_gate: liq_score<=-1 AND OTM
    #     Rescue B: stoch<35 OR rev>=2
    lc = (liq <= -1) & (off >= 0)
    lc_rescue = (stch < 35) | (rev >= 2)
    blocked |= lc & ~lc_rescue

    # 12. body_bp_gate: body_15m in [0.50,0.60) AND bp_5m<0.55
    #     Rescue: bp_5m>=0.55
    bbp = (body >= 0.50) & (body < 0.60) & (bp < 0.55)
    blocked |= bbp  # bp_5m<0.55 is already the block condition; bp>=0.55 → not blocked

    # 13. near_itm_gate: pm>0.50 AND (rsi_4h>62 OR macd_hist normalized large)
    #     Use raw macd_norm: macd_hist > 80th pctile of distribution (norm ~1.2)
    blocked |= (pm > 0.50) & ((rsi4 > 62) | (mcd4 > 1.2))

    # 14. vol_score1_gate: vol_score==1, rescue: ema==1 AND (conf==0 OR fund==0)
    vs1 = (vols == 1)
    vs1_rescue = (ema == 1) & ((conf == 0) | (fund == 0))
    blocked |= vs1 & ~vs1_rescue

    # 15. vol_eff_low_gate: vol_eff < 0.000318 AND z_score > -0.20
    ve = arc["vol_eff"].values
    blocked |= (ve < 0.000318) & (zs > -0.20)

    return blocked


def apply_no_gates(arc, p_model):
    """
    Apply all reconstructable BTC NO gates.
    Returns boolean array: True = BLOCKED for NO.
    """
    blocked = np.zeros(len(arc), dtype=bool)

    pm   = arc["p_market"].values
    rev  = _g(arc, "composite_rev")
    ct   = _g(arc, "composite_trend")
    stch = _g(arc, "stoch_k")
    fund = _g(arc, "funding_bias")
    zs   = np.abs(arc["z_strike"].values)

    # 1. no_z_gate: |z_strike| < 0.30
    #    Rescue: ct<=-3 AND fund==-1
    nz = zs < 0.30
    nz_rescue = (ct <= -3) & (fund == -1)
    blocked |= nz & ~nz_rescue

    # 2. stoch_no_gate: stoch<20
    #    Rescue: (ct<=-3 AND fund==-1) OR rev>=2
    sn = stch < 20
    sn_rescue = ((ct <= -3) & (fund == -1)) | (rev >= 2)
    blocked |= sn & ~sn_rescue

    # 3. highpm_no_gate: pm>0.70 AND rev>=0 → block; pm>0.70 AND rev<0 → allow
    blocked |= (pm > 0.70) & (rev >= 0)

    return blocked


# ── p_model computation ───────────────────────────────────────────────────────

def compute_p_model(arc, config, side):
    arrays = {col: arc[col].values for col in arc.columns if col in [
        "tau_minutes", "sigma_tau", "z_strike", "mu6h", "mu12h", "mu24h",
        "regime_z", "rvol_inv", "stoch_4h", "ema20_slope_z",
        "composite_trend", "composite_rev",
    ]}
    zd = compute_drift(arrays, config, side)
    return np.clip(1 - ndtr(arc["z_strike"].values - zd), EPS, 1 - EPS)


# ── single-slot simulation ─────────────────────────────────────────────────────

def simulate_slot(slot_df, p_yes_arr, p_no_arr, yes_blocked, no_blocked, pm_arr, y_arr, tau_arr):
    """
    For one slot: pick best YES or NO bet (highest net edge, not blocked).
    Returns (pnl, side_taken, won) or (0, None, None) if no bet.
    """
    best_pnl = 0.0; best_side = None; best_won = None
    best_edge = 0.0

    for i in range(len(pm_arr)):
        pm = pm_arr[i]; y = y_arr[i]; tau = tau_arr[i]
        fee = FEE_RATE * min(pm, 1 - pm)

        # YES candidate
        if not yes_blocked[i]:
            ey = p_yes_arr[i] - pm
            if ey > best_edge:
                ky = min((ey / max(1 - pm, 1e-6)) * KELLY_MULT, KELLY_CAP)
                if ky > 0:
                    n_c = ky * BANKROLL / max(pm, 1e-6)
                    pnl = n_c * (1 - pm - fee) if y == 1 else -n_c * (pm + fee)
                    best_edge = ey; best_pnl = pnl; best_side = "yes"; best_won = (y == 1)

        # NO candidate
        if not no_blocked[i]:
            en = pm - p_no_arr[i]
            if en > best_edge:
                kn = min((en / max(pm, 1e-6)) * KELLY_MULT, KELLY_CAP)
                if kn > 0:
                    n_c = kn * BANKROLL / max(1 - pm, 1e-6)
                    pnl = n_c * (pm - fee) if y == 0 else -n_c * ((1 - pm) + fee)
                    best_edge = en; best_pnl = pnl; best_side = "no"; best_won = (y == 0)

    return best_pnl, best_side, best_won


# ── walk-forward simulation ────────────────────────────────────────────────────

def walk_forward(arc, config, label=""):
    print(f"\n  Simulating [{label}]...")
    pm_all    = arc["p_market"].values
    y_all     = arc["resolved_yes"].values
    tau_all   = arc["tau_minutes"].values

    # Compute p_model for YES and NO sides
    p_yes_all = compute_p_model(arc, config, "yes")
    p_no_all  = compute_p_model(arc, config, "no")

    # Apply gate stacks (same for both configs — only drift changes p_model)
    yes_blocked_all = apply_yes_gates(arc)
    no_blocked_all  = apply_no_gates(arc, p_yes_all)

    day_results = []
    for date, day_df in arc.groupby("date"):
        idx     = day_df.index
        d_pnl   = 0.0; ny = 0; nn = 0; yw = 0; nw = 0

        for ts, slot_df in day_df.groupby("logged_at"):
            sidx = slot_df.index
            pnl, side, won = simulate_slot(
                slot_df,
                p_yes_all[arc.index.get_indexer(sidx)],
                p_no_all[arc.index.get_indexer(sidx)],
                yes_blocked_all[arc.index.get_indexer(sidx)],
                no_blocked_all[arc.index.get_indexer(sidx)],
                pm_all[arc.index.get_indexer(sidx)],
                y_all[arc.index.get_indexer(sidx)],
                tau_all[arc.index.get_indexer(sidx)],
            )
            d_pnl += pnl
            if side == "yes": ny += 1; yw += int(won)
            elif side == "no": nn += 1; nw += int(won)

        nt = ny + nn
        day_results.append({
            "date": date, "pnl": d_pnl, "n_yes": ny, "n_no": nn,
            "n_total": nt, "wr": (yw + nw) / nt if nt > 0 else float("nan"),
        })

    return pd.DataFrame(day_results)


# ── reporting ──────────────────────────────────────────────────────────────────

def print_results(res, label):
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  {'Date':<12}  {'P&L':>10}  {'Cumul':>10}  {'n_YES':>7}  {'n_NO':>7}  {'WR':>7}")
    print("  " + "-" * 57)
    cumul = 0.0
    for _, r in res.iterrows():
        cumul += r["pnl"]
        wr_s = f"{r['wr']:.3f}" if r["wr"] == r["wr"] else "   nan"
        print(f"  {str(r['date']):<12}  {r['pnl']:>+10.2f}  {cumul:>+10.2f}  "
              f"{r['n_yes']:>7}  {r['n_no']:>7}  {wr_s:>7}")
    total = res["pnl"].sum()
    ny = res["n_yes"].sum(); nn = res["n_no"].sum(); nt = res["n_total"].sum()
    print("  " + "-" * 57)
    print(f"  {'TOTAL':<12}  {total:>+10.2f}  {'':>10}  {ny:>7}  {nn:>7}")


def print_comparison(res_base, res_var, label_base, label_var):
    print(f"\n{'='*72}")
    print(f"  DELTA: {label_var} − {label_base}  (positive = variant better)")
    print(f"{'='*72}")
    print(f"  {'Date':<12}  {'base P&L':>10}  {'var P&L':>10}  {'delta':>10}  {'b_Y':>5}  {'v_Y':>5}")
    print("  " + "-" * 62)
    cumul_delta = 0.0
    for (_, rb), (_, rv) in zip(res_base.iterrows(), res_var.iterrows()):
        delta = rv["pnl"] - rb["pnl"]
        cumul_delta += delta
        marker = " +" if delta > 0 else "  "
        print(f"  {str(rb['date']):<12}  {rb['pnl']:>+10.2f}  {rv['pnl']:>+10.2f}  "
              f"{delta:>+10.2f}  {rb['n_yes']:>5}  {rv['n_yes']:>5}{marker}")

    total_base = res_base["pnl"].sum()
    total_var  = res_var["pnl"].sum()
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<12}  {total_base:>+10.2f}  {total_var:>+10.2f}  {total_var-total_base:>+10.2f}")

    print(f"\n  {'='*40}")
    print(f"  VERDICT")
    print(f"  {'='*40}")
    delta_total = total_var - total_base
    pct = (delta_total / abs(total_base) * 100) if total_base != 0 else float("nan")
    print(f"  Baseline:  {total_base:>+10.2f}")
    print(f"  Variant:   {total_var:>+10.2f}")
    print(f"  Delta:     {delta_total:>+10.2f}  ({pct:+.1f}%)")
    if delta_total > 0:
        days_better = (res_var["pnl"] > res_base["pnl"]).sum()
        print(f"  Variant better on {days_better}/{len(res_base)} days")
    print()
    print("  NOTE: smc_gate not replicated (missing signals) — both configs equally affected.")
    print("  Absolute P&L inflated vs live; delta between configs is the valid comparison.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGS — edit these to test different changes
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CONFIG = {
    # Current live model (YES: mu6+mu24+rz+ct at k=1; NO: zero drift per K_DRIFT_NO_BTC=0)
    "drift_yes": "mu6_24_rz_ct",
    "drift_no":  "zero",
    "k_yes": 1.0,
    "k_no":  0.0,
    "label": "Current (mu6_24_rz_ct YES, zero NO)",
}

VARIANT_CONFIG = {
    # Proposed: 3-factor drift (Stoch_4h × RVOL_inv × |EMA20z|) for both sides
    "drift_yes": "stoch_rvol_ema20z",
    "drift_no":  "stoch_rvol_ema20z",
    "k_yes": 1.0,
    "k_no":  1.0,
    "label": "New (Stoch×RVOL_inv×|EMA20z| k=1 both sides)",
}


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    arc, df1h = load_data()

    res_base = walk_forward(arc, BASELINE_CONFIG, BASELINE_CONFIG["label"])
    res_var  = walk_forward(arc, VARIANT_CONFIG,  VARIANT_CONFIG["label"])

    print_results(res_base, BASELINE_CONFIG["label"])
    print_results(res_var,  VARIANT_CONFIG["label"])
    print_comparison(res_base, res_var, BASELINE_CONFIG["label"], VARIANT_CONFIG["label"])
