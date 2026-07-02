"""
backfill_scan_archive.py — Backfill all features needed to run the proposed
new model on btc_scan_archive.csv.

Adds to each scan row:
  p_up_v2_backfilled   — LGBM directional probability
  dc_4h_n20_pos/break  — Donchian 4h n=20 position
  dc_4h_n55_pos/break  — Donchian 4h n=55 position
  stoch_k_4h           — 4h stochastic
  vol_implied_kalshi   — implied vol from lognormal inversion of p_market

Output: results/scan_archive_backfilled.csv
"""
import math, pickle, warnings
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from arch import arch_model as _arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
RES_DIR    = ROOT / "results"
MODEL_PATH = ROOT / "reform_results" / "btc_p_up_v2.pkl"

LGBM_FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── feature helpers (copied from backfill_p_up_v2.py) ─────────────────────────

def rsi_series(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-10))


def macd_hist_series(close, f=12, s=26, sig=9):
    macd   = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
    return macd - macd.ewm(span=sig, adjust=False).mean()


def bb_pct_series(close, n=20):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    rng = (mid + 2*std - (mid - 2*std)).replace(0, float("nan"))
    return (close - (mid - 2*std)) / rng


def ema50_dist_series(close):
    e50 = close.ewm(span=50, adjust=False).mean()
    return (close - e50) / e50.replace(0, float("nan")) * 100


def daily_vwap_dist_series(df):
    tp     = (df["high"] + df["low"] + df["close"]) / 3
    date   = df.index.normalize()
    cum_tv = (tp * df["volume"]).groupby(date).cumsum()
    cum_v  = df["volume"].groupby(date).cumsum()
    vwap   = cum_tv / cum_v.replace(0, float("nan"))
    return (df["close"] - vwap) / vwap.replace(0, float("nan")) * 100


def stoch_k_series(high, low, close, period=14):
    ll  = low.rolling(period).min()
    hh  = high.rolling(period).max()
    return (close - ll) / (hh - ll).replace(0, float("nan")) * 100


def atr_series(high, low, close, period=14):
    cp = close.shift(1)
    tr = pd.concat([high - low, (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def chg_4h_atr_series(df, period=14):
    atr = atr_series(df["high"], df["low"], df["close"], period)
    chg = df["close"] - df["close"].shift(4)
    return chg / atr.replace(0, float("nan"))


def donchian(high, low, close, n, prefix):
    upper = high.rolling(n, min_periods=n // 2).max()
    lower = low.rolling(n,  min_periods=n // 2).min()
    rng   = (upper - lower).replace(0, float("nan"))
    pos   = (close - lower) / rng
    prior_upper = high.shift(1).rolling(n, min_periods=n // 2).max()
    prior_lower = low.shift(1).rolling(n,  min_periods=n // 2).min()
    brk = pd.Series(0.0, index=close.index)
    brk[close > prior_upper] =  1.0
    brk[close < prior_lower] = -1.0
    return {f"{prefix}_pos": pos, f"{prefix}_break": brk}


# ── implied vol inversion ──────────────────────────────────────────────────────

def implied_vol(p_market, spot, strike, tau_minutes, fallback=None):
    """Invert lognormal to get vol_implied from Kalshi market price."""
    tau_h = max(tau_minutes / 60.0, 1/60)
    if spot <= 0 or strike <= 0 or not (0.005 < p_market < 0.995):
        return fallback
    try:
        def objective(sig):
            if sig <= 0:
                return -p_market
            z = math.log(strike / spot) / (sig * math.sqrt(tau_h))
            return float(norm.sf(z)) - p_market
        lo_val = objective(0.0001)
        hi_val = objective(5.0)
        if lo_val * hi_val > 0:
            return fallback
        return brentq(objective, 0.0001, 5.0, xtol=1e-6)
    except Exception:
        return fallback


# ── load 1h parquet ────────────────────────────────────────────────────────────

def load_1h():
    files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def build_1h_features(df1h):
    f = pd.DataFrame(index=df1h.index)
    f["rsi_14"]            = rsi_series(df1h["close"], 14)
    f["macd_hist_1h"]      = macd_hist_series(df1h["close"])
    f["bb_pct"]            = bb_pct_series(df1h["close"])
    f["ema50_dist"]        = ema50_dist_series(df1h["close"])
    f["vwap_distance_pct"] = daily_vwap_dist_series(df1h)
    lr = np.log(df1h["close"] / df1h["close"].shift(1))
    f["rvol_1h"] = (lr.rolling(168, min_periods=24).std() /
                    lr.rolling(24, min_periods=4).std().replace(0, float("nan"))).clip(0.3, 2.0)
    return f


def build_4h_features(df1h):
    df4h = df1h.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    f = pd.DataFrame(index=df4h.index)
    f["stoch_k_4h"]    = stoch_k_series(df4h["high"], df4h["low"], df4h["close"], 14)
    f["rsi_4h"]        = rsi_series(df4h["close"], 14)
    f["macd_hist_4h"]  = macd_hist_series(df4h["close"])
    f["chg_4h_atr"]    = chg_4h_atr_series(df4h)
    for n in [20, 55]:
        for k, v in donchian(df4h["high"], df4h["low"], df4h["close"], n, f"dc_4h_n{n}").items():
            f[k] = v
    return f


# ── derived 1h signals ────────────────────────────────────────────────────────

def build_rvol_inv_series(df1h):
    """rvol_inv = clip(168h_std / 24h_std, 0.3, 2.0) on log returns (per live system)."""
    lr = np.log(df1h["close"] / df1h["close"].shift(1))
    std_168 = lr.rolling(168, min_periods=24).std()
    std_24  = lr.rolling(24,  min_periods=4).std()
    ratio   = std_168 / std_24.replace(0, float("nan"))
    return ratio.clip(0.3, 2.0).rename("rvol_inv")


def build_markov_regime_map(df1h):
    """20-day rolling return on daily closes → {date → regime string} dict."""
    daily = df1h["close"].resample("1D").last().dropna()
    ret20 = daily.pct_change(20)
    result = {}
    for ts, r in ret20.items():
        if math.isnan(r):
            continue
        result[ts.date()] = "Bull" if r > 0.02 else ("Bear" if r < -0.02 else "Sideways")
    return result


def build_garch_ratio_series(df1h, hours_needed):
    """GARCH(1,1) conditional vol ratio per hour, matching live _get_garch_ratio().

    Only fits for hours present in hours_needed (unique bar_1h timestamps).
    Returns a Series indexed by those timestamps.
    """
    if not _ARCH_OK:
        print("  [garch] arch library not available — skipping garch_ratio")
        return pd.Series(dtype=float, name="garch_ratio")

    lr = (np.log(df1h["close"] / df1h["close"].shift(1)) * 100).dropna()
    results = {}
    hours_sorted = sorted(hours_needed)
    print(f"  Fitting GARCH for {len(hours_sorted)} unique hours...")
    for i, ts in enumerate(hours_sorted):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(hours_sorted)}")
        try:
            w = lr.loc[:ts].iloc[-500:]
            if len(w) < 100:
                results[ts] = float("nan")
                continue
            am  = _arch_model(w, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            cond_v = float(res.conditional_volatility.iloc[-1])
            omega  = float(res.params["omega"])
            alpha  = float(res.params["alpha[1]"])
            beta   = float(res.params["beta[1]"])
            persist = alpha + beta
            lr_vol  = (float(np.sqrt(omega / (1.0 - persist)))
                       if persist < 1.0 else float(w.std()))
            results[ts] = cond_v / lr_vol if lr_vol > 0 else 1.0
        except Exception:
            results[ts] = float("nan")
    return pd.Series(results, name="garch_ratio")


# ── main ───────────────────────────────────────────────────────────────────────

def run():
    print("Loading scan archive...")
    scan = pd.read_csv(RES_DIR / "btc_scan_archive.csv", low_memory=False)
    scan = scan[scan["logged_at"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
    scan["logged_at"] = pd.to_datetime(scan["logged_at"], utc=True)
    for col in ["spot", "strike", "p_market", "tau_minutes", "vol_eff"]:
        scan[col] = pd.to_numeric(scan[col], errors="coerce")
    print(f"  {len(scan):,} rows  "
          f"({scan['logged_at'].min().date()} → {scan['logged_at'].max().date()})")

    print("Loading 1h parquet...")
    df1h = load_1h()
    print(f"  {len(df1h):,} bars  ({df1h.index.min().date()} → {df1h.index.max().date()})")

    print("Computing 1h features...")
    feat_1h = build_1h_features(df1h)
    print("Computing 4h features (dc, stoch_k_4h)...")
    feat_4h = build_4h_features(df1h)

    # Map each row to last completed bar
    scan["bar_1h"] = scan["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
    scan["bar_4h"] = scan["logged_at"].dt.floor("4h") - pd.Timedelta(hours=4)

    def safe_join(df, feat_table, on_col):
        renamed = feat_table.add_suffix("_c")
        df = df.join(renamed, on=on_col, how="left")
        for col in feat_table.columns:
            df[col] = df[col + "_c"]
            df.drop(columns=[col + "_c"], inplace=True)
        return df

    scan = safe_join(scan, feat_1h, "bar_1h")
    scan = safe_join(scan, feat_4h, "bar_4h")

    print("Computing rvol_inv (168h/24h vol ratio)...")
    rvol_inv_s = build_rvol_inv_series(df1h)
    scan = scan.join(rvol_inv_s.rename("rvol_inv_c"), on="bar_1h", how="left")
    scan["rvol_inv"] = scan["rvol_inv_c"].fillna(1.0)
    scan.drop(columns=["rvol_inv_c"], inplace=True)

    print("Computing markov_regime_daily (20-day return ±2%)...")
    markov_map = build_markov_regime_map(df1h)
    scan["markov_regime_daily"] = (scan["logged_at"].dt.tz_convert("UTC").dt.date
                                    .map(markov_map).fillna("Sideways"))

    print("Computing garch_ratio (GARCH(1,1) per hour)...")
    hours_needed = scan["bar_1h"].dropna().unique()
    garch_s = build_garch_ratio_series(df1h, hours_needed)
    scan = scan.join(garch_s.rename("garch_ratio_c"), on="bar_1h", how="left")
    scan["garch_ratio"] = scan["garch_ratio_c"]
    scan.drop(columns=["garch_ratio_c"], inplace=True)

    print("Computing vol_implied_kalshi (lognormal inversion)...")
    scan["vol_implied_kalshi"] = [
        implied_vol(float(r.p_market), float(r.spot), float(r.strike),
                    float(r.tau_minutes), fallback=float(r.vol_eff))
        for r in scan[["p_market","spot","strike","tau_minutes","vol_eff"]].itertuples()
    ]

    print("Running LGBM for p_up_v2...")
    with open(MODEL_PATH, "rb") as fh:
        pipe = pickle.load(fh)
    clf = pipe["clf"]

    # Coerce LGBM features
    for feat in LGBM_FEATURES:
        if feat in scan.columns:
            scan[feat] = pd.to_numeric(scan[feat], errors="coerce")
        else:
            scan[feat] = float("nan")

    feat_mat = scan[LGBM_FEATURES].astype(float).values
    probs = clf.predict_proba(feat_mat)[:, 1]
    scan["p_up_v2_backfilled"] = np.clip(probs, 0.02, 0.98)

    # Coverage report
    key_cols = ["p_up_v2_backfilled", "dc_4h_n20_pos", "stoch_k_4h",
                "rsi_4h", "macd_hist_4h", "vol_implied_kalshi",
                "ema_stack_bias", "composite_rev", "adx_1h", "composite_p_up",
                "rvol_inv", "garch_ratio", "markov_regime_daily"]
    print(f"\n  Feature coverage ({len(scan):,} rows):")
    for col in key_cols:
        if col in scan.columns:
            nn = pd.to_numeric(scan[col], errors="coerce").notna().sum()
            print(f"    {col:<25} {nn:>7,} ({nn/len(scan):.0%})")

    # Write output — include all live gate signal columns from original archive
    PASSTHROUGH_COLS = [
        "ema_stack_bias", "composite_rev", "composite_trend",
        "vwap_stretch_score",   # used as vwap_score in G2 gate and stretch_score
        "rvol_1h", "vpin_score", "ema_stretch_score",
        "liq_score", "liq_bias", "offset_pct", "funding_bias",
        "adx_1h", "composite_p_up", "vol_score",
    ]
    COMPUTED_COLS = ["rvol_inv", "markov_regime_daily", "garch_ratio"]
    out_cols = (["logged_at", "contract_ticker",
                 "spot", "strike", "p_market", "tau_minutes", "vol_eff",
                 "stoch_k", "resolved_yes",
                 "p_up_v2_backfilled", "vol_implied_kalshi",
                 "stoch_k_4h", "rsi_4h", "macd_hist_4h"]
                + [c for c in scan.columns if c.startswith("dc_4h")]
                + [c for c in PASSTHROUGH_COLS if c in scan.columns]
                + [c for c in COMPUTED_COLS if c in scan.columns])
    out = scan[[c for c in out_cols if c in scan.columns]].copy()
    out_path = RES_DIR / "scan_archive_backfilled.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    run()
