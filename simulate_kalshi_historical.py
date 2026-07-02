"""
simulate_kalshi_historical.py — 875-day historical Kalshi BTC simulation.

Replays 2024-01-01 → present using actual 1m BTC price data.
Each 1-hour expiry cycle generates a live-style scan every 3 minutes:
  - Contract tickers shift dynamically with BTC spot price
  - All signals computed from OHLCV (no lookahead)
  - Full gate stack, session limits, daily loss limit, cooldowns
  - Era 5 drift: norm.ppf(p_up_v2) × rvol_inv × 0.3 × √(τ/60)
  - Settles against actual 1m BTC close at hour end

Outputs:
  results/sim_scan_archive.csv  — one row per scan cycle (best candidate)
  results/sim_trades.csv        — one row per trade placed

Usage:
    python3 simulate_kalshi_historical.py
    python3 simulate_kalshi_historical.py --start 2025-01-01 --end 2025-12-31
"""
import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

# ── model params (Era 5 live) ─────────────────────────────────────────────────
BANKROLL          = 1_000.0
KELLY_MULT        = 0.30
KELLY_CAP         = 0.06
FEE_RATE          = 0.07
MAX_TRADES_EXPIRY = 3
DAILY_LOSS_LIMIT  = 150.0
SCAN_INTERVAL_MIN = 3
STRIKE_INCREMENT  = 100
VOL_WEIGHT_YES    = 0.35   # realized weight in blended vol
DVOL_PROXY_RATIO  = 1.12   # Deribit DVOL ≈ 1.12 × realized (approx when not available)
K_DRIFT           = 0.30   # drift scale factor (Era 5)
GARCH_WINDOW      = 500
GARCH_STEP        = 24
EPS               = 1e-7
LIQUID_PCT        = 0.013  # ±1.3% window is liquid (matches real scanner ±1%)

# ── helpers ────────────────────────────────────────────────────────────────────

def _rsi_vec(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-10))

def _stoch_vec(h, lo, c, p: int = 14) -> pd.Series:
    ll = lo.rolling(p).min()
    hh = h.rolling(p).max()
    return ((c - ll) / (hh - ll).replace(0, np.nan) * 100).clip(0, 100)

def _macd_hist_vec(c, fast=12, slow=26, sig=9) -> pd.Series:
    m = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=sig, adjust=False).mean()

def _adx_vec(h, lo, c, p=14) -> pd.Series:
    tr  = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    dm_p = np.where((h - h.shift()) > (lo.shift() - lo), (h - h.shift()).clip(lower=0), 0.0)
    dm_m = np.where((lo.shift() - lo) > (h - h.shift()), (lo.shift() - lo).clip(lower=0), 0.0)
    dm_p = pd.Series(dm_p, index=h.index).ewm(alpha=1/p, adjust=False).mean()
    dm_m = pd.Series(dm_m, index=h.index).ewm(alpha=1/p, adjust=False).mean()
    atr  = tr.ewm(alpha=1/p, adjust=False).mean().replace(0, np.nan)
    di_p = 100 * dm_p / atr; di_m = 100 * dm_m / atr
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(alpha=1/p, adjust=False).mean().fillna(20.0)

def _bb_pct_vec(c, n=20) -> pd.Series:
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    lo  = mid - 2 * std; hi = mid + 2 * std
    return ((c - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)

def _vwap_stretch_vec(df, p=20) -> pd.Series:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vol  = df["volume"].replace(0, np.nan)
    vwap = (tp * vol).rolling(p).sum() / vol.rolling(p).sum()
    dist = (df["close"] - vwap) / vwap
    sig  = dist.rolling(p).std().replace(0, np.nan)
    return (dist / sig).clip(-3, 3)


# ── load and pre-compute all signals ──────────────────────────────────────────

def load_signals(start: str, end: str):
    print("Loading 1h BTC data...")
    f1h = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1h = pd.read_parquet(f1h)
    df1h.index = pd.to_datetime(df1h.index, utc=True)
    df1h = df1h[df1h.index.year > 1970].sort_index()
    print(f"  {len(df1h):,} bars  {df1h.index[0].date()} → {df1h.index[-1].date()}")

    print("Loading 1m BTC data...")
    f1m = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1m_1970*.parquet"),
                 key=lambda p: p.stat().st_mtime)[-1]
    df1m = pd.read_parquet(f1m)
    df1m.index = pd.to_datetime(df1m.index, utc=True)
    df1m = df1m[df1m.index.year > 1970].sort_index()
    print(f"  {len(df1m):,} bars")

    print("Computing 1h signals...")
    lr         = np.log(df1h["close"] / df1h["close"].shift(1))
    vol_24h    = lr.rolling(24,  min_periods=4).std()
    vol_168h   = lr.rolling(168, min_periods=24).std()
    rvol_1h    = (vol_24h / vol_168h.replace(0, np.nan)).clip(0.3, 3.0)
    rvol_inv   = (vol_168h / vol_24h.replace(0, np.nan)).clip(0.3, 2.0)
    mu6h       = lr.rolling(6,  min_periods=1).mean()
    rsi_1h     = _rsi_vec(df1h["close"])
    stoch_1h   = _stoch_vec(df1h["high"], df1h["low"], df1h["close"])
    macd_1h    = _macd_hist_vec(df1h["close"])
    adx_1h     = _adx_vec(df1h["high"], df1h["low"], df1h["close"])
    bb_pct_1h  = _bb_pct_vec(df1h["close"])
    vwap_str   = _vwap_stretch_vec(df1h)
    ema20_1h   = df1h["close"].ewm(span=20, adjust=False).mean()
    ema50_1h   = df1h["close"].ewm(span=50, adjust=False).mean()
    ema_stack  = pd.Series(
        np.where(ema20_1h > ema50_1h, 1, np.where(ema20_1h < ema50_1h, -1, 0)),
        index=df1h.index, dtype=float)
    ema_stretch = ((df1h["close"] - ema20_1h) / ema20_1h * 100).clip(-5, 5)

    # Composite proxies (matching simulate_model_synthetic.py)
    rev_proxy   = ((50 - rsi_1h) / 3.5 + (50 - stoch_1h) / 3.5).clip(-15, 15)
    trend_proxy = (ema_stack * 2 + np.sign(macd_1h)).clip(-6, 6)

    # Clip to lookup table ranges
    comp_trend_i = trend_proxy.round().clip(-3, 3).fillna(0).astype(int)
    comp_rev_i   = rev_proxy.round().clip(-11, 11).fillna(0).astype(int)

    print("Computing 4h signals...")
    df4h = df1h[["open","high","low","close","volume"]].resample(
        "4h", label="right", closed="right"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["close"])
    stoch_4h  = _stoch_vec(df4h["high"], df4h["low"], df4h["close"])
    rsi_4h    = _rsi_vec(df4h["close"])
    macd_4h   = _macd_hist_vec(df4h["close"])
    ema50_4h  = df4h["close"].ewm(span=50, adjust=False).mean()
    ema50_dist_4h = ((df4h["close"] - ema50_4h) / ema50_4h * 100).clip(-15, 15)
    atr_4h    = _adx_vec(df4h["high"], df4h["low"], df4h["close"])  # reuse adx helper
    chg_4h    = df4h["close"].pct_change(4)

    # ATR-normalised 4h change
    tr4h = pd.concat([df4h["high"]-df4h["low"],
                      (df4h["high"]-df4h["close"].shift()).abs(),
                      (df4h["low"] -df4h["close"].shift()).abs()], axis=1).max(axis=1)
    atr4h_val = tr4h.ewm(com=13, adjust=False).mean().replace(0, np.nan)
    chg_4h_atr = ((df4h["close"] - df4h["close"].shift(4)) / atr4h_val).clip(-5, 5)

    # EMA20 slope z-score on 4h
    ema20_4h     = df4h["close"].ewm(span=20, adjust=False).mean()
    ema20_slp    = (ema20_4h.diff() / ema20_4h) * 100
    ema20_slpz_4h = (ema20_slp /
                     ema20_slp.rolling(100, min_periods=20).std().replace(0, np.nan)).clip(-3, 3)

    # ffill 4h → 1h
    stoch_4h_1h   = stoch_4h.reindex(df1h.index, method="ffill").fillna(50.0)
    rsi_4h_1h     = rsi_4h.reindex(df1h.index, method="ffill").fillna(50.0)
    macd_4h_1h    = macd_4h.reindex(df1h.index, method="ffill").fillna(0.0)
    ema50d_1h     = ema50_dist_4h.reindex(df1h.index, method="ffill").fillna(0.0)
    chg4at_1h     = chg_4h_atr.reindex(df1h.index, method="ffill").fillna(0.0)
    ema20slpz_1h  = ema20_slpz_4h.reindex(df1h.index, method="ffill").fillna(0.0)

    # Vol for sigma computation
    # YES sigma: blended — approximate Deribit as realized × DVOL_PROXY_RATIO
    vol_imp_1h  = vol_24h * DVOL_PROXY_RATIO   # proxy for Deribit/Kalshi implied
    vol_eff_1h  = VOL_WEIGHT_YES * vol_24h + (1 - VOL_WEIGHT_YES) * vol_imp_1h

    # GARCH pre-computation
    print("Computing GARCH ratios (~2 min)...")
    lr_arr    = lr.values
    garch_s   = pd.Series(np.nan, index=df1h.index)
    for i in range(GARCH_WINDOW, len(lr_arr), GARCH_STEP):
        window = lr_arr[i - GARCH_WINDOW:i]
        if np.isnan(window).sum() > 10:
            continue
        try:
            wc  = window[~np.isnan(window)] * 100
            res = arch_model(wc, vol="Garch", p=1, q=1,
                             mean="Zero", dist="normal").fit(disp="off", show_warning=False)
            h_t  = float(res.conditional_volatility[-1]) ** 2
            omega = float(res.params.get("omega", 1e-6))
            alpha = float(res.params.get("alpha[1]", 0))
            beta  = float(res.params.get("beta[1]", 0))
            lr_h  = omega / max(1e-6, 1 - alpha - beta)
            ratio = np.sqrt(h_t / lr_h) if lr_h > 0 else 1.0
            for j in range(i, min(i + GARCH_STEP, len(df1h.index))):
                garch_s.iloc[j] = ratio
        except Exception:
            pass
    garch_s = garch_s.clip(0.3, 3.0).fillna(1.0)
    # NO sigma = realized × GARCH ratio (conditional vol)
    vol_no_1h = vol_24h * garch_s

    # p_up_v2 batch inference
    print("Computing p_up_v2 (batch inference)...")
    try:
        import pickle
        model_path = ROOT / "reform_results" / "btc_p_up_v2.pkl"
        with open(model_path, "rb") as f:
            pipe = pickle.load(f)
        clf = pipe["clf"]

        # vol ratio (rvol_1h = current bar volume / 25-bar avg volume)
        vol_ratio_1h = (df1h["volume"] /
                        df1h["volume"].rolling(25, min_periods=5).mean()
                        ).clip(0.1, 10).fillna(1.0)

        # Daily VWAP distance proxy (use rolling 24h VWAP)
        tp   = (df1h["high"] + df1h["low"] + df1h["close"]) / 3
        vol_ = df1h["volume"].replace(0, np.nan)
        vwap_24 = (tp * vol_).rolling(24, min_periods=4).sum() / vol_.rolling(24, min_periods=4).sum()
        vwap_dist_pct = ((df1h["close"] - vwap_24) / vwap_24 * 100).clip(-5, 5)

        # composite_p_up lookup (import here to avoid circular import issues)
        sys.path.insert(0, str(ROOT))
        from composite_scorer import lookup_p_up
        comp_p_up_arr = np.array([
            lookup_p_up(int(t), int(r), asset="BTC")
            for t, r in zip(comp_trend_i.values, comp_rev_i.values)
        ])

        feat_mat = np.column_stack([
            stoch_4h_1h.values,      # stoch_k_4h
            ema50d_1h.values,        # ema50_dist
            rsi_4h_1h.values,        # rsi_4h
            rsi_1h.values,           # rsi_14
            macd_1h.values,          # macd_hist_1h
            stoch_1h.values,         # stoch_k
            vwap_dist_pct.values,    # vwap_distance_pct
            chg4at_1h.values,        # chg_4h_atr
            bb_pct_1h.values,        # bb_pct
            comp_trend_i.values.astype(float),  # composite_trend
            comp_rev_i.values.astype(float),    # composite_rev
            comp_p_up_arr,           # composite_p_up
            ema_stack.values,        # ema_stack_bias
            ema_stretch.values,      # ema_stretch_score
            vwap_str.values,         # vwap_stretch_score
            np.zeros(len(df1h)),     # confirmation_bias (neutral)
            np.zeros(len(df1h)),     # stoch_bias (neutral)
            np.full(len(df1h), 0.5), # vpin_score (neutral)
            np.zeros(len(df1h)),     # pm_drift_5m (neutral)
            vol_ratio_1h.values,     # rvol_1h
        ])

        # Replace NaN with column means for model stability
        col_means = np.nanmean(feat_mat, axis=0)
        nan_mask  = np.isnan(feat_mat)
        feat_mat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        p_up_v2_arr = clf.predict_proba(feat_mat)[:, 1].clip(0.02, 0.98)
        p_up_v2_s   = pd.Series(p_up_v2_arr, index=df1h.index)
        print(f"  p_up_v2 range: [{p_up_v2_arr.min():.3f}, {p_up_v2_arr.max():.3f}]  "
              f"mean={p_up_v2_arr.mean():.3f}")
    except Exception as e:
        print(f"  p_up_v2 batch failed ({e}) — using composite lookup as fallback")
        comp_p_up_arr = np.array([
            lookup_p_up(int(t), int(r), asset="BTC")
            for t, r in zip(comp_trend_i.values, comp_rev_i.values)
        ])
        p_up_v2_s = pd.Series(comp_p_up_arr, index=df1h.index)

    # Assemble signal frame
    sigs = pd.DataFrame({
        "close":         df1h["close"],
        "vol_24h":       vol_24h,
        "vol_168h":      vol_168h,
        "vol_eff":       vol_eff_1h,   # YES sigma per 1h
        "vol_no":        vol_no_1h,    # NO sigma per 1h
        "rvol_1h":       rvol_1h,
        "rvol_inv":      rvol_inv,
        "garch_ratio":   garch_s,
        "p_up_v2":       p_up_v2_s,
        "comp_trend":    comp_trend_i,
        "comp_rev":      comp_rev_i,
        "stoch_1h":      stoch_1h,
        "rsi_1h":        rsi_1h,
        "rsi_4h":        rsi_4h_1h,
        "stoch_4h":      stoch_4h_1h,
        "adx_1h":        adx_1h,
        "ema_stack":     ema_stack,
        "ema20_slpz":    ema20slpz_1h,
        "vwap_stretch":  vwap_str,
    }, index=df1h.index).dropna(subset=["vol_24h", "stoch_1h"])

    print(f"  Signal rows: {len(sigs):,}  "
          f"({sigs.index[0].date()} → {sigs.index[-1].date()})")

    # Filter to simulation window
    sigs = sigs[(sigs.index.date >= pd.Timestamp(start).date()) &
                (sigs.index.date <= pd.Timestamp(end).date())]
    t_start = pd.Timestamp(start, tz="UTC")
    t_end   = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df1m = df1m[(df1m.index >= t_start) & (df1m.index < t_end)]
    print(f"  Simulation window: {len(sigs):,} bars, {df1m.shape[0]:,} 1m rows")

    return sigs, df1m


# ── contract pricing ──────────────────────────────────────────────────────────

def lognormal_p_yes(spot, strike, sigma_tau, z_drift=0.0):
    if sigma_tau <= 0:
        return (1.0 if spot > strike else 0.0)
    z = math.log(strike / spot) / sigma_tau - z_drift
    return float(norm.sf(z))


def half_spread(z_abs: float, tau_min: float) -> float:
    """Bid-ask half-spread model matching Kalshi observed spreads."""
    if z_abs < 0.25:   base = 0.013
    elif z_abs < 0.55: base = 0.021
    elif z_abs < 0.90: base = 0.031
    elif z_abs < 1.30: base = 0.043
    else:              base = 0.062
    # Widen in last 10 min
    if tau_min < 10:
        base *= 1.0 + (10 - tau_min) / 10 * 0.55
    return base


def fear_discount(rvol: float) -> float:
    """Market systematically underprices YES in high-vol (fear) regimes."""
    return max(0.0, min((rvol - 1.0) * 0.025, 0.035))


def build_ladder(spot: float, tau_min: float,
                 vol_eff_per_min: float, vol_no_per_min: float,
                 rvol: float):
    """
    Generate synthetic Kalshi contract ladder for current scan snapshot.
    Market prices have ZERO directional drift — market makers are neutral.
    Model drift is applied separately in scanner_cycle for edge computation.
    """
    atm     = round(spot / STRIKE_INCREMENT) * STRIKE_INCREMENT
    tau_use = max(tau_min, 0.5)
    sigma_yes = vol_eff_per_min * math.sqrt(tau_use)
    sigma_no  = vol_no_per_min  * math.sqrt(tau_use)
    fd        = fear_discount(rvol)

    contracts = []
    for off in range(-1300, 1400, STRIKE_INCREMENT):
        strike = atm + off
        if strike <= 0:
            continue
        pct = (strike - spot) / spot
        if abs(pct) > LIQUID_PCT:
            continue

        # Market price: zero drift — market has no directional signal
        p_fair = lognormal_p_yes(spot, strike, sigma_yes, 0.0)
        p_mm   = max(EPS, min(1 - EPS, p_fair * (1 - fd)))

        z_abs  = abs(math.log(strike / spot) / max(sigma_yes, 1e-8))
        hs     = half_spread(z_abs, tau_min)
        bid    = round(max(0.01, p_mm - hs), 2)
        ask    = round(min(0.99, p_mm + hs), 2)
        p_mkt  = round((bid + ask) / 2, 3)

        contracts.append({
            "strike":      strike,
            "offset_pct":  round(pct, 5),
            "bid":         bid,
            "ask":         ask,
            "p_market":    p_mkt,
            "z_abs":       round(z_abs, 4),
            "sigma_yes":   sigma_yes,
            "sigma_no":    sigma_no,
        })
    return contracts


# ── gate stack ────────────────────────────────────────────────────────────────

def yes_gate(c: dict, row, tau_min: float, hour_utc: int):
    pm       = c["p_market"]
    rvol     = float(row["rvol_1h"])
    gr       = float(row["garch_ratio"])
    ema      = float(row["ema_stack"])
    stoch    = float(row["stoch_1h"])
    rev      = float(row["comp_rev"])
    rsi4     = float(row["rsi_4h"])
    pup      = float(row["p_up_v2"])
    z_abs    = c["z_abs"]

    if hour_utc in (13, 16):           return "hour_gate"
    if rvol < 0.80:                    return "rvol_gate"
    if gr > 1.5 and not (pm >= 0.80 and tau_min < 45):
                                       return "garch_highvol"
    if pm < 0.35 and ema == 0:         return "deepno_neutral"
    # BearDrift: strong downtrend, no oversold bounce, not fully bearish stoch
    if ema == -1 and rev <= 3 and stoch >= 35:
                                       return "beardrift"
    # Near-ITM YES when overbought 4h RSI
    if pm > 0.50 and rsi4 > 62:        return "near_itm"
    # OTM YES in bear trend with model bearish
    if pm < 0.25 and pup < 0.40 and ema <= 0:
                                       return "otm_bear"
    return None


def no_gate(c: dict, row):
    pm    = c["p_market"]
    z_abs = c["z_abs"]
    stoch = float(row["stoch_1h"])
    rev   = float(row["comp_rev"])
    ct    = float(row["comp_trend"])

    if z_abs < 0.30:                   return "no_z_gate"
    if stoch < 20:
        if ct > -3:                    return "stoch_oversold_no"
    if pm > 0.70 and rev >= 0:         return "highpm_no"
    return None


# ── scanner ───────────────────────────────────────────────────────────────────

def scanner_cycle(spot: float, tau_min: float, row,
                  trades_this_expiry: int, daily_pnl: float,
                  cooldown_yes: bool, cooldown_no: bool,
                  hour_utc: int, z_drift_const: float):
    """
    Evaluate ladder, apply gates, return best decision dict.
    """
    if trades_this_expiry >= MAX_TRADES_EXPIRY:
        return {"decision": "no_trade", "reason": "session_limit"}
    if daily_pnl < -DAILY_LOSS_LIMIT:
        return {"decision": "no_trade", "reason": "daily_loss_limit"}

    rvol = float(row["rvol_1h"])
    vol_eff_per_min = float(row["vol_eff"]) / math.sqrt(60)
    vol_no_per_min  = float(row["vol_no"])  / math.sqrt(60)
    # Market ladder uses ZERO drift; model applies drift separately below
    ladder = build_ladder(spot, tau_min, vol_eff_per_min, vol_no_per_min, rvol)

    # Model drift: scaled to current tau
    z_model = z_drift_const * math.sqrt(max(tau_min, 0.5) / 60.0)

    best_edge = 0.0
    best_side = None
    best_c    = None
    best_gate = None

    for c in ladder:
        pm = c["p_market"]
        if pm < 0.05 or pm > 0.95:
            continue
        fee = FEE_RATE * min(pm, 1 - pm)

        # YES: model adds drift on top of blended vol
        if not cooldown_yes:
            p_yes = lognormal_p_yes(spot, c["strike"], c["sigma_yes"], z_model)
            p_yes = max(EPS, min(1 - EPS, p_yes))
            edge_yes = p_yes - c["ask"] - fee
            gate_y   = yes_gate(c, row, tau_min, hour_utc)
            if edge_yes > best_edge and gate_y is None:
                best_edge = edge_yes; best_side = "yes"
                best_c = c; best_gate = None
            elif gate_y and edge_yes > 0 and best_side is None:
                best_gate = gate_y

        # NO: model applies drift with GARCH vol; edge vs neutral market bid
        if not cooldown_no:
            p_no = lognormal_p_yes(spot, c["strike"], c["sigma_no"], z_model)
            p_no = max(EPS, min(1 - EPS, p_no))
            edge_no = c["bid"] - p_no - fee
            gate_n  = no_gate(c, row)
            if edge_no > best_edge and gate_n is None:
                best_edge = edge_no; best_side = "no"
                best_c = c; best_gate = None
            elif gate_n and edge_no > 0 and best_side is None:
                best_gate = gate_n

    if best_c is None or best_side is None:
        return {"decision": "no_trade", "reason": best_gate or "no_edge",
                "spot": spot, "tau_min": tau_min}

    # Kelly sizing
    pm_c  = best_c["ask"] if best_side == "yes" else best_c["bid"]
    pm_risk = pm_c if best_side == "yes" else (1 - pm_c)
    k = min((best_edge / max(pm_risk, 1e-6)) * KELLY_MULT, KELLY_CAP)
    n_cont = k * BANKROLL / max(pm_risk, 1e-6)
    if n_cont < 0.01:
        return {"decision": "no_trade", "reason": "kelly_too_small",
                "spot": spot, "tau_min": tau_min}

    return {
        "decision":    "trade",
        "side":        best_side,
        "strike":      best_c["strike"],
        "p_market":    pm_c,
        "p_fair":      round(lognormal_p_yes(spot, best_c["strike"],
                             best_c["sigma_yes"] if best_side == "yes" else best_c["sigma_no"],
                             z_model), 5),
        "edge":        round(best_edge, 5),
        "offset_pct":  best_c["offset_pct"],
        "n_cont":      round(n_cont, 2),
        "kelly_frac":  round(k, 4),
        "z_abs":       best_c["z_abs"],
        "spot":        spot,
        "tau_min":     tau_min,
    }


# ── ticker generator ──────────────────────────────────────────────────────────

_MONTH = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
          7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}

def make_ticker(ts: pd.Timestamp, strike: float) -> str:
    day = ts.day; mon = _MONTH[ts.month]; yr = str(ts.year)[-2:]
    hr  = ts.hour
    return f"KXBTCD-{day:02d}{mon}{yr}{hr:02d}-T{int(strike):05d}.99"


# ── main simulation ───────────────────────────────────────────────────────────

def run(start: str, end: str, no_drift: bool = False, invert_drift: bool = False):
    sigs, df1m = load_signals(start, end)

    # Group 1m by hour for fast lookup
    print("Indexing 1m data by hour...")
    df1m_by_hour = {}
    for ts, grp in df1m.groupby(df1m.index.floor("1h")):
        df1m_by_hour[ts] = grp

    scan_rows  = []
    trade_rows = []

    daily_pnl      = 0.0
    current_date   = None
    trades_expiry  = 0
    open_trades    = []   # trades placed in current expiry, pending settlement
    cooldown_yes_until = None
    cooldown_no_until  = None

    hours = sigs.index
    print(f"\nRunning simulation: {hours[0].date()} → {hours[-1].date()} "
          f"({len(hours):,} hours)...")

    for hi, bar_ts in enumerate(hours):
        row   = sigs.loc[bar_ts]
        date  = bar_ts.date()
        hour_utc = bar_ts.hour

        # Daily reset
        if date != current_date:
            daily_pnl   = 0.0
            current_date = date
            if hi % 200 == 0:
                print(f"  {date}  trades_total={len(trade_rows):,}  "
                      f"pnl_so_far=${sum(r.get('pnl',0) for r in trade_rows):+.0f}")

        # Settle previous hour's open trades
        for tr in open_trades:
            # Settlement price = first 1m close of next bar (top of hour)
            next_bar = bar_ts
            settle_price = float(row["close"])   # use opening price of new bar as settlement
            # Try to get more accurate: last 1m close in previous hour
            prev_hour = bar_ts - pd.Timedelta(hours=1)
            if prev_hour in df1m_by_hour:
                settle_price = float(df1m_by_hour[prev_hour]["close"].iloc[-1])

            resolved = 1 if settle_price > tr["strike"] else 0
            if tr["side"] == "yes":
                won = resolved == 1
                pnl = (tr["n_cont"] * (1 - tr["p_market"] -
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])) if won
                       else -tr["n_cont"] * (tr["p_market"] +
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])))
            else:
                won = resolved == 0
                pm_adj = 1 - tr["p_market"]   # NO price = 1 - bid
                pnl = (tr["n_cont"] * (tr["p_market"] -
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])) if won
                       else -tr["n_cont"] * (pm_adj +
                       FEE_RATE * min(tr["p_market"], 1 - tr["p_market"])))

            pnl = round(pnl, 2)
            daily_pnl += pnl
            tr["resolved_yes"]    = resolved
            tr["spot_at_expiry"]  = settle_price
            tr["pnl"]             = pnl
            tr["won"]             = won
            trade_rows.append(tr)

        open_trades   = []
        trades_expiry = 0
        cooldown_yes_until = None
        cooldown_no_until  = None

        # Get 1m price slice for this hour
        if bar_ts not in df1m_by_hour:
            continue
        m1_slice = df1m_by_hour[bar_ts]
        if len(m1_slice) < 3:
            continue

        # Compute drift constant for this bar
        p_up     = float(row["p_up_v2"])
        rvol_inv = float(row["rvol_inv"])
        if no_drift:
            z_drift_const = 0.0
        else:
            pup_z = float(norm.ppf(max(0.01, min(0.99, p_up))))
            z_drift_const = pup_z * rvol_inv * K_DRIFT
            if invert_drift:
                z_drift_const = -z_drift_const

        expiry_ts = bar_ts + pd.Timedelta(hours=1)

        # Scan every SCAN_INTERVAL_MIN minutes within the hour
        for scan_min in range(SCAN_INTERVAL_MIN, 60 - 1, SCAN_INTERVAL_MIN):
            tau_min = 60 - scan_min

            # Spot price at this scan minute
            scan_ts = bar_ts + pd.Timedelta(minutes=scan_min)
            # Find closest 1m bar at or before scan_ts
            m1_before = m1_slice[m1_slice.index <= scan_ts]
            if m1_before.empty:
                spot = float(row["close"])
            else:
                spot = float(m1_before["close"].iloc[-1])

            # Cooldown status
            cd_yes = cooldown_yes_until is not None and scan_ts < cooldown_yes_until
            cd_no  = cooldown_no_until  is not None and scan_ts < cooldown_no_until

            result = scanner_cycle(
                spot=spot, tau_min=tau_min, row=row,
                trades_this_expiry=trades_expiry,
                daily_pnl=daily_pnl,
                cooldown_yes=cd_yes, cooldown_no=cd_no,
                hour_utc=hour_utc, z_drift_const=z_drift_const,
            )

            # Generate ticker from current spot
            if result["decision"] == "trade":
                strike = result["strike"]
            else:
                # Nearest ATM for logging
                strike = round(spot / STRIKE_INCREMENT) * STRIKE_INCREMENT

            ticker = make_ticker(expiry_ts, strike)

            scan_rec = {
                "logged_at":        scan_ts,
                "contract_ticker":  ticker,
                "close_ts":         expiry_ts,
                "spot":             round(spot, 2),
                "strike":           strike,
                "tau_minutes":      tau_min,
                "p_market":         result.get("p_market", float("nan")),
                "offset_pct":       result.get("offset_pct", 0.0),
                "decision":         result["decision"],
                "side":             result.get("side", ""),
                "edge":             result.get("edge", 0.0),
                "reason":           result.get("reason", ""),
                "p_up_v2":          p_up,
                "rvol_inv":         round(rvol_inv, 4),
                "rvol_1h":          round(float(row["rvol_1h"]), 4),
                "garch_ratio":      round(float(row["garch_ratio"]), 4),
                "comp_trend":       int(row["comp_trend"]),
                "comp_rev":         int(row["comp_rev"]),
                "stoch_1h":         round(float(row["stoch_1h"]), 2),
                "rsi_4h":           round(float(row["rsi_4h"]), 2),
                "stoch_4h":         round(float(row["stoch_4h"]), 2),
                "adx_1h":           round(float(row["adx_1h"]), 2),
                "ema_stack":        int(row["ema_stack"]),
                "vol_eff":          round(float(row["vol_eff"]), 6),
                "z_drift_const":    round(z_drift_const, 5),
                "n_cont":           result.get("n_cont", 0.0),
                "kelly_frac":       result.get("kelly_frac", 0.0),
                "resolved_yes":     float("nan"),  # filled at settlement
            }
            scan_rows.append(scan_rec)

            if result["decision"] == "trade":
                trade_rec = {**scan_rec,
                             "p_fair":   result.get("p_fair", float("nan")),
                             "resolved_yes": float("nan"),
                             "spot_at_expiry": float("nan"),
                             "pnl": float("nan")}
                open_trades.append(trade_rec)
                trades_expiry += 1

                # 5-min side cooldown
                cd_end = scan_ts + pd.Timedelta(minutes=5)
                if result["side"] == "yes":
                    cooldown_yes_until = cd_end
                else:
                    cooldown_no_until  = cd_end

    # Flush last hour's open trades (no settlement if simulation ends mid-cycle)
    print(f"\nSimulation complete.")
    print(f"  Total scans:  {len(scan_rows):,}")
    print(f"  Total trades: {len(trade_rows):,}")
    if trade_rows:
        pnl_total = sum(r.get("pnl", 0) for r in trade_rows)
        wins = sum(1 for r in trade_rows if r.get("won", False))
        print(f"  Total P&L:    ${pnl_total:+,.2f}")
        print(f"  Win rate:     {wins}/{len(trade_rows)} = {wins/len(trade_rows)*100:.1f}%")

    # ── write outputs ─────────────────────────────────────────────────────────
    suffix = "_nodrift" if no_drift else ("_invdrift" if invert_drift else "_drift")
    out_scans  = RES_DIR / f"sim_scan_archive{suffix}.csv"
    out_trades = RES_DIR / f"sim_trades{suffix}.csv"

    df_scans  = pd.DataFrame(scan_rows)
    df_trades = pd.DataFrame(trade_rows)

    df_scans.to_csv(out_scans,  index=False)
    df_trades.to_csv(out_trades, index=False)

    print(f"\n  Wrote {out_scans}  ({len(df_scans):,} rows)")
    print(f"  Wrote {out_trades}  ({len(df_trades):,} rows)")

    # ── daily P&L summary ─────────────────────────────────────────────────────
    if not df_trades.empty and "pnl" in df_trades.columns:
        df_trades["date"] = pd.to_datetime(df_trades["logged_at"]).dt.date
        daily = df_trades.groupby("date").agg(
            pnl=("pnl","sum"), n=("pnl","count"),
            wr=("won","mean")).reset_index()
        daily["cumul"] = daily["pnl"].cumsum()
        print(f"\n{'─'*66}")
        print(f"  {'Date':<12}  {'P&L':>8}  {'Cumul':>9}  {'n':>4}  {'WR':>6}")
        print(f"  {'─'*58}")
        for _, r in daily.iterrows():
            wr_s = f"{r['wr']:.1%}" if not math.isnan(r['wr']) else "  nan"
            print(f"  {str(r['date']):<12}  {r['pnl']:>+8.2f}  "
                  f"{r['cumul']:>+9.2f}  {int(r['n']):>4}  {wr_s:>6}")
        print(f"  {'─'*58}")
        print(f"  {'TOTAL':<12}  {daily['pnl'].sum():>+8.2f}  "
              f"{daily['pnl'].sum():>+9.2f}  {int(daily['n'].sum()):>4}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end",   default="2026-05-26")
    ap.add_argument("--no-drift", action="store_true",
                    help="Force z_drift=0 for all scans (pure vol model, no directional tilt).")
    ap.add_argument("--invert-drift", action="store_true",
                    help="Invert z_drift sign — take positions opposite to the p_up_v2 signal.")
    args = ap.parse_args()
    run(args.start, args.end, no_drift=args.no_drift, invert_drift=args.invert_drift)
