"""
simulate_model_synthetic.py — 875-day walk-forward model comparison.

Generates synthetic contracts from 1h BTC OHLCV (2024-01-01 → present),
reconstructs gate signals from raw price data, applies the full (approximable)
BTC gate stack, and runs a walk-forward P&L comparison between configs.

Why synthetic:
  - btc_scan_archive.csv only has 9 days; synthetic extends to 875 days
  - 8 strike offsets per bar × 875 days ≈ 140k contract evaluations
  - Walk-forward quarterly breakdown avoids regime-specific bias

Synthetic p_market = log-normal fair price (no market bias).
This means: the model can ONLY find edge through the drift formula.
The test asks: "does drift formula X predict BTC direction better than chance?"

Gate signal reconstruction from OHLCV:
  - ema_stack_bias: EMA20 vs EMA50 from 1h
  - stoch_k: 14-period Stochastic from 1h
  - rsi_1h: 14-period RSI from 1h
  - adx_1h: 14-period ADX from 1h
  - rvol_1h: 1h realized vol vs 24h realized vol
  - composite_rev_proxy: RSI + Stoch oversold/overbought score (approximation)
  - composite_trend_proxy: EMA + MACD directional score (approximation)
  - garch_ratio: rolling GARCH(1,1) from 1h (recomputed every 24 bars)
  - rsi_4h, macd_4h: from 4h-resampled data (for near_itm_gate)
  - vwap_stretch: rolling VWAP distance (1h, 20-period)

OMITTED GATES (same for both configs — delta is still valid):
  - liq_cascade_gate: requires Coinalyze API historical data
  - body_bp_gate, falling_knife_gate: require 5m/15m data
  - smc_gate: requires supply zone / CHoCH reconstruction
  - vpin-based rescues: require tick data

Usage:
    python3 simulate_model_synthetic.py

Edit BASELINE_CONFIG and VARIANT_CONFIG at the bottom.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from arch import arch_model

warnings.filterwarnings("ignore")

DATA_DIR = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc/data")

BANKROLL     = 1000.0
KELLY_MULT   = 0.30
KELLY_CAP    = 0.06
FEE_RATE     = 0.07
EPS          = 1e-7
GARCH_WINDOW = 500

# Synthetic contract strike offsets (fraction of spot)
OFFSETS = [-0.020, -0.015, -0.010, -0.005, 0.005, 0.010, 0.015, 0.020]
TAU     = 60.0   # minutes (1h contracts)
VOL_WIN = 24     # bars for sigma estimation


# ── signal helpers ─────────────────────────────────────────────────────────────

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))


def _stoch_k(h, l, c, p=14):
    return (c - l.rolling(p).min()) / \
           (h.rolling(p).max() - l.rolling(p).min()).replace(0, np.nan) * 100


def _macd_hist(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    return m - m.ewm(span=sig).mean()


def _adx(h, l, c, p=14):
    tr   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    dmp  = np.where((h-h.shift()) > (l.shift()-l), (h-h.shift()).clip(lower=0), 0.0)
    dmm  = np.where((l.shift()-l) > (h-h.shift()), (l.shift()-l).clip(lower=0), 0.0)
    dmp  = pd.Series(dmp, index=h.index).ewm(alpha=1/p, adjust=False).mean()
    dmm  = pd.Series(dmm, index=h.index).ewm(alpha=1/p, adjust=False).mean()
    atr  = tr.ewm(alpha=1/p, adjust=False).mean().replace(0, np.nan)
    di_p = 100 * dmp / atr; di_m = 100 * dmm / atr
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(alpha=1/p, adjust=False).mean()


def _vwap_stretch(df, p=20):
    """Rolling VWAP distance in sigma units (approx stretch score)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"].replace(0, np.nan)
    vwap    = (typical * vol).rolling(p).sum() / vol.rolling(p).sum()
    dist    = (df["close"] - vwap) / vwap
    sigma   = dist.rolling(p).std().replace(0, np.nan)
    return (dist / sigma).clip(-3, 3)


# ── load and compute all signals ───────────────────────────────────────────────

def load_signals():
    print("Loading 1h BTC data...")
    f1h = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h = pd.read_parquet(f1h)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    df1h = df1h[df1h.index.year > 1970].sort_index()
    print(f"  {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")

    # ── 1h signals ────────────────────────────────────────────────────────────
    lr       = np.log(df1h["close"] / df1h["close"].shift(1))
    vol_24h  = lr.rolling(VOL_WIN, min_periods=4).std()
    vol_168h = lr.rolling(168, min_periods=24).std()
    sigma_pm = vol_24h / np.sqrt(60.0)           # vol per √minute
    sigma_tau = sigma_pm * np.sqrt(TAU)           # vol over tau

    rvol_1h  = (vol_24h / vol_168h.replace(0, np.nan)).clip(0.3, 3.0)  # current / 1-week
    rvol_inv = (vol_168h / vol_24h.replace(0, np.nan)).clip(0.3, 2.0)  # quiet = high

    rsi_1h   = _rsi(df1h["close"])
    stch_1h  = _stoch_k(df1h["high"], df1h["low"], df1h["close"])
    adx_1h   = _adx(df1h["high"], df1h["low"], df1h["close"])
    vwap_str = _vwap_stretch(df1h)

    ema20_1h = df1h["close"].ewm(span=20, adjust=False).mean()
    ema50_1h = df1h["close"].ewm(span=50, adjust=False).mean()
    ema_stack = pd.Series(
        np.where(ema20_1h > ema50_1h, 1, np.where(ema20_1h < ema50_1h, -1, 0)),
        index=df1h.index)

    # mu signals for current baseline
    mu6h  = lr.rolling(6,  min_periods=1).mean()
    mu12h = lr.rolling(12, min_periods=1).mean()
    mu24h = lr.rolling(24, min_periods=1).mean()
    ewm_m = lr.ewm(span=12).mean()
    ewm_s = lr.ewm(span=24).std()
    rz    = np.clip(ewm_m / ewm_s.replace(0, np.nan), -3.0, 3.0).fillna(0.0)

    # Composite proxies from price data
    # rev_proxy: positive = oversold/bullish-reversal, negative = overbought
    #   Matches live composite_rev direction; range scaled to -15…+15
    rev_rsi  = np.clip((50.0 - rsi_1h)  / 3.5, -10, 10)
    rev_stch = np.clip((50.0 - stch_1h) / 3.5, -10, 10)
    rev_proxy = (rev_rsi + rev_stch).clip(-15, 15)

    # trend_proxy: EMA direction + MACD sign
    macd_1h     = _macd_hist(df1h["close"], 12, 26, 9)
    trend_proxy = (ema_stack * 2 + np.sign(macd_1h)).clip(-6, 6)

    # ── 4h signals (resampled) ─────────────────────────────────────────────────
    print("Computing 4h signals...")
    df4h = df1h[["open","high","low","close","volume"]].resample(
        "4h", label="right", closed="right"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["close"])

    stch_4h  = _stoch_k(df4h["high"], df4h["low"], df4h["close"])
    rsi_4h   = _rsi(df4h["close"])
    macd_4h  = _macd_hist(df4h["close"])
    macd_std_4h = macd_4h.rolling(250, min_periods=50).std().replace(0, np.nan)
    macd_norm_4h = (macd_4h / macd_std_4h).clip(-3, 3).fillna(0)

    ema20_4h  = df4h["close"].ewm(span=20, adjust=False).mean()
    ema20_slp = (ema20_4h.diff() / ema20_4h) * 100
    ema20_slp_z = (ema20_slp /
                   ema20_slp.rolling(100, min_periods=20).std().replace(0, np.nan)
                   ).clip(-3, 3)

    # Resample 4h → 1h (ffill)
    idx_1h = df1h.index
    stch_4h_1h   = stch_4h.reindex(idx_1h, method="ffill").fillna(50.0)
    rsi_4h_1h    = rsi_4h.reindex(idx_1h, method="ffill").fillna(50.0)
    macd_norm_1h = macd_norm_4h.reindex(idx_1h, method="ffill").fillna(0.0)
    ema20_slpz_1h= ema20_slp_z.reindex(idx_1h, method="ffill").fillna(0.0)

    # ── GARCH(1,1) vol ratio ──────────────────────────────────────────────────
    print("Computing GARCH ratios (~2 min)...")
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
    garch_ratio = garch_ratio.clip(0.3, 3.0).fillna(1.0)

    sigs = pd.DataFrame({
        "close":        df1h["close"],
        "sigma_tau":    sigma_tau,
        "rvol_1h":      rvol_1h,
        "rvol_inv":     rvol_inv,
        "rsi_1h":       rsi_1h,
        "stch_1h":      stch_1h,
        "adx_1h":       adx_1h,
        "vwap_stretch": vwap_str,
        "ema_stack":    ema_stack,
        "rev_proxy":    rev_proxy,
        "trend_proxy":  trend_proxy,
        "mu6h":         mu6h,
        "mu12h":        mu12h,
        "mu24h":        mu24h,
        "regime_z":     rz,
        "stch_4h":      stch_4h_1h,
        "rsi_4h":       rsi_4h_1h,
        "macd_norm_4h": macd_norm_1h,
        "ema20_slpz":   ema20_slpz_1h,
        "garch_ratio":  garch_ratio,
    }, index=df1h.index).dropna(subset=["sigma_tau", "stch_1h"])

    print(f"  Signal rows: {len(sigs):,}  ({sigs.index[0].date()} → {sigs.index[-1].date()})")
    return sigs, df1h


# ── drift formulas ─────────────────────────────────────────────────────────────

def compute_drift(row, config, side):
    sq  = np.sqrt(TAU / 60.0)
    t60 = TAU / 60.0
    st  = row["sigma_tau"]
    k   = config["k_yes"] if side == "yes" else config["k_no"]
    formula = config["drift_yes"] if side == "yes" else config["drift_no"]

    if formula == "zero":
        return 0.0

    elif formula == "mu6_24_rz_ct":
        m6 = row["mu6h"]; m24 = row["mu24h"]; rz = row["regime_z"]
        ct = row["trend_proxy"]
        return k * ((m6 + m24) * t60 / max(st, 1e-8) + rz * sq) + (ct / 5.0) * 0.15 * sq

    elif formula == "mu_all_rz":
        m6 = row["mu6h"]; m12 = row["mu12h"]; m24 = row["mu24h"]; rz = row["regime_z"]
        return k * ((m6 + m12 + m24) * t60 / max(st, 1e-8) + rz * sq)

    elif formula == "stoch_rvol_ema20z":
        s4   = (row["stch_4h"] - 50.0) / 25.0
        rv   = float(np.clip(row["rvol_inv"], 0.3, 2.0))
        e20z = abs(float(row["ema20_slpz"]))
        return k * s4 * rv * e20z * sq

    else:
        raise ValueError(f"Unknown formula: {formula!r}")


# ── gate stack ─────────────────────────────────────────────────────────────────

def is_yes_blocked(row, pm, tau, z_strike, offset):
    """Returns True if YES is blocked."""
    ema  = float(row["ema_stack"])
    stch = float(row["stch_1h"])
    rev  = float(row["rev_proxy"])
    ct   = float(row["trend_proxy"])
    rvol = float(row["rvol_1h"])
    adx  = float(row["adx_1h"])
    rsi4 = float(row["rsi_4h"])
    mcd4 = float(row["macd_norm_4h"])
    vwap = float(row["vwap_stretch"])
    gr   = float(row["garch_ratio"])
    hour = row.name.hour

    # Hour gate
    if hour in (13, 16):
        return True

    # rvol_gate: low realized vol (no smart rescues in synthetic)
    if rvol < 0.80:
        return True

    # ADX gate: weak trending AND not bearish
    if 20 <= adx < 40 and ema != -1:
        return True

    # GARCH high-vol gate
    if gr > 1.5:
        if not (pm >= 0.80 and tau < 45):
            return True

    # Deep OTM neutral gate
    if pm < 0.35 and ema == 0:
        return True

    # BearDrift gate: ema=-1, weak reversal, stoch not oversold
    if ema == -1 and rev <= 3:
        if stch >= 35:
            return True

    # OTM neutral gate: ema=0, bullish p_up, OTM
    if ema == 0 and offset > 0:
        # Proxy for p_up: use trend_proxy
        if ct >= 0:   # no strong bearish trend = p_up likely neutral/bullish
            return True

    # Exhaustion gate: ema=1, strong overbought, VWAP stretched above
    if ema == 1 and rev <= -4 and stch >= 75 and vwap <= -1:
        return True

    # Near-ITM gate: ITM YES with overbought 4h
    if pm > 0.50 and (rsi4 > 62 or mcd4 > 1.2):
        return True

    # Vol_eff_low approximation: if sigma_tau is very low
    if float(row["sigma_tau"]) < 0.000318 * np.sqrt(TAU) and z_strike > -0.20:
        return True

    return False


def is_no_blocked(row, pm, z_abs):
    """Returns True if NO is blocked."""
    stch = float(row["stch_1h"])
    ct   = float(row["trend_proxy"])
    rev  = float(row["rev_proxy"])

    # no_z_gate: near-ATM NO has too little cushion
    if z_abs < 0.30:
        return True

    # stoch_no_gate: stoch < 20 (oversold = risky NO)
    if stch < 20:
        # Rescue: strong bearish trend
        if not (ct <= -3):
            return True

    # highpm_no_gate: pm>0.70 AND no strong reversal signal
    if pm > 0.70 and rev >= 0:
        return True

    return False


# ── walk-forward simulation ────────────────────────────────────────────────────

def simulate(sigs, config):
    label = config["label"]
    print(f"\n  Simulating [{label}]...")

    results = []
    closes = sigs["close"].values
    idx    = sigs.index

    for i in range(len(closes) - 1):
        row = sigs.iloc[i]
        st  = float(row["sigma_tau"])
        if np.isnan(st) or st <= 0:
            continue
        spot    = closes[i]
        nc      = closes[i + 1]
        hour    = idx[i].hour
        date    = idx[i].date()
        quarter = f"{idx[i].year}Q{idx[i].quarter}"

        best_pnl  = 0.0
        best_side = None
        best_won  = None
        best_edge = 0.0

        for off in OFFSETS:
            strike   = spot * (1 + off)
            z_str    = np.log(strike / spot) / st
            pm       = float(np.clip(1 - ndtr(z_str), EPS, 1 - EPS))
            outcome  = 1 if nc > strike else 0
            fee      = FEE_RATE * min(pm, 1 - pm)

            # YES
            zd_yes   = compute_drift(row, config, "yes")
            p_yes    = float(np.clip(1 - ndtr(z_str - zd_yes), EPS, 1 - EPS))
            ey       = p_yes - pm

            if ey > 0 and not is_yes_blocked(row, pm, TAU, z_str, off * 100):
                if ey > best_edge:
                    ky  = min((ey / max(1 - pm, 1e-6)) * KELLY_MULT, KELLY_CAP)
                    if ky > 0:
                        n_c = ky * BANKROLL / max(pm, 1e-6)
                        pnl = n_c * (1 - pm - fee) if outcome == 1 else -n_c * (pm + fee)
                        best_edge = ey; best_pnl = pnl; best_side = "yes"; best_won = (outcome == 1)

            # NO
            zd_no    = compute_drift(row, config, "no")
            p_no_yes = float(np.clip(1 - ndtr(z_str - zd_no), EPS, 1 - EPS))
            en       = pm - p_no_yes

            if en > 0 and not is_no_blocked(row, pm, abs(z_str)):
                if en > best_edge:
                    kn  = min((en / max(pm, 1e-6)) * KELLY_MULT, KELLY_CAP)
                    if kn > 0:
                        n_c = kn * BANKROLL / max(1 - pm, 1e-6)
                        pnl = n_c * (pm - fee) if outcome == 0 else -n_c * ((1 - pm) + fee)
                        best_edge = en; best_pnl = pnl; best_side = "no"; best_won = (outcome == 0)

        results.append({
            "date":    date,
            "quarter": quarter,
            "pnl":     best_pnl,
            "side":    best_side,
            "won":     best_won,
        })

    return pd.DataFrame(results)


# ── reporting ──────────────────────────────────────────────────────────────────

def report(res, label):
    res = res[res["side"].notna()].copy()
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Quarter':<10}  {'P&L':>10}  {'Cumul':>10}  {'n_YES':>7}  {'n_NO':>7}  {'WR':>7}")
    print("  " + "-" * 58)
    cumul = 0.0
    for q, qg in res.groupby("quarter"):
        qpnl  = qg["pnl"].sum()
        cumul += qpnl
        ny    = (qg["side"] == "yes").sum()
        nn    = (qg["side"] == "no").sum()
        nt    = len(qg)
        wr    = qg["won"].mean() if nt > 0 else float("nan")
        print(f"  {q:<10}  {qpnl:>+10.2f}  {cumul:>+10.2f}  {ny:>7}  {nn:>7}  {wr:>7.3f}")
    total = res["pnl"].sum()
    ny = (res["side"] == "yes").sum(); nn = (res["side"] == "no").sum()
    wr = res["won"].mean()
    print("  " + "-" * 58)
    print(f"  {'TOTAL':<10}  {total:>+10.2f}  {'':>10}  {ny:>7}  {nn:>7}  {wr:>7.3f}")
    return res


def compare(res_b, res_v, label_b, label_v):
    res_b = res_b[res_b["side"].notna()].copy()
    res_v = res_v[res_v["side"].notna()].copy()

    # Aggregate by quarter
    def qagg(res):
        return res.groupby("quarter").agg(pnl=("pnl","sum"), n=("pnl","count"),
                                          ny=("side", lambda s:(s=="yes").sum()),
                                          wr=("won","mean")).reset_index()

    qb = qagg(res_b); qv = qagg(res_v)
    merged = qb.merge(qv, on="quarter", suffixes=("_b","_v"))

    print(f"\n{'='*72}")
    print(f"  DELTA by quarter: {label_v} − {label_b}")
    print(f"{'='*72}")
    print(f"  {'Quarter':<10}  {'base P&L':>10}  {'var P&L':>10}  {'delta':>10}  {'b_Y':>6}  {'v_Y':>6}")
    print("  " + "-" * 62)
    for _, r in merged.iterrows():
        delta  = r["pnl_v"] - r["pnl_b"]
        marker = " +" if delta > 0 else "  "
        print(f"  {r['quarter']:<10}  {r['pnl_b']:>+10.2f}  {r['pnl_v']:>+10.2f}  "
              f"{delta:>+10.2f}  {r['ny_b']:>6}  {r['ny_v']:>6}{marker}")

    tb = res_b["pnl"].sum(); tv = res_v["pnl"].sum()
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<10}  {tb:>+10.2f}  {tv:>+10.2f}  {tv-tb:>+10.2f}")

    print(f"\n  {'='*40}")
    print(f"  VERDICT")
    print(f"  {'='*40}")
    pct = (tv - tb) / abs(tb) * 100 if tb != 0 else float("nan")
    print(f"  Baseline:  {tb:>+10.2f}  ({label_b})")
    print(f"  Variant:   {tv:>+10.2f}  ({label_v})")
    print(f"  Delta:     {tv-tb:>+10.2f}  ({pct:+.1f}%)")
    q_better = (merged["pnl_v"] > merged["pnl_b"]).sum()
    print(f"  Variant better in {q_better}/{len(merged)} quarters")
    print()
    print("  NOTE: p_market = log-normal fair price (no Kalshi market bias).")
    print("  Edge comes ONLY from drift formula. Omitted gates affect both equally.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGS — edit these to test different changes
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CONFIG = {
    "drift_yes": "mu6_24_rz_ct",
    "drift_no":  "zero",
    "k_yes": 1.0,
    "k_no":  0.0,
    "label": "Current (mu6_24_rz_ct YES, zero NO)",
}

VARIANT_CONFIG = {
    "drift_yes": "stoch_rvol_ema20z",
    "drift_no":  "stoch_rvol_ema20z",
    "k_yes": 1.0,
    "k_no":  1.0,
    "label": "New (Stoch×RVOL_inv×|EMA20z| k=1 both sides)",
}

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sigs, df1h = load_signals()

    res_base = simulate(sigs, BASELINE_CONFIG)
    res_var  = simulate(sigs, VARIANT_CONFIG)

    report(res_base, BASELINE_CONFIG["label"])
    report(res_var,  VARIANT_CONFIG["label"])
    compare(res_base, res_var, BASELINE_CONFIG["label"], VARIANT_CONFIG["label"])
