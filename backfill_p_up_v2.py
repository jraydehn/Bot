"""
backfill_p_up_v2.py — Retroactively compute p_up_v2 for all resolved paper trades.

Uses the trained btc_p_up_v2 LightGBM model (reform_results/btc_p_up_v2.pkl).

Price-based features computed vectorially from 1h parquet data.
Logged signals (composite, stoch, ema, vpin, etc.) pulled directly from
the paper trade archives.

For each trade at time T: features are taken from the last COMPLETED 1h bar,
i.e. floor(T, '1h') - 1h, matching how compute_p_up() works in the live runner
(it drops the in-progress bar before inference).

Output: results/p_up_v2_backfilled.csv
    contract_ticker | logged_at | side | p_up_v2_backfilled
"""
import glob, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
MODEL_PATH = ROOT / "reform_results" / "btc_p_up_v2.pkl"

FEATURES = [
    "stoch_k_4h", "ema50_dist", "rsi_4h", "rsi_14", "macd_hist_1h",
    "stoch_k", "vwap_distance_pct", "chg_4h_atr", "bb_pct",
    "composite_trend", "composite_rev", "composite_p_up",
    "ema_stack_bias", "ema_stretch_score", "vwap_stretch_score",
    "confirmation_bias", "stoch_bias", "vpin_score",
    "pm_drift_5m", "rvol_1h",
]


# ── vectorised feature helpers ────────────────────────────────────────────────

def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def macd_hist_series(close: pd.Series, f=12, s=26, sig=9) -> pd.Series:
    macd   = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal


def bb_pct_series(close: pd.Series, n: int = 20) -> pd.Series:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    lo  = mid - 2 * std
    hi  = mid + 2 * std
    rng = (hi - lo).replace(0, float("nan"))
    return (close - lo) / rng


def ema50_dist_series(close: pd.Series) -> pd.Series:
    e50 = close.ewm(span=50, adjust=False).mean()
    return (close - e50) / e50.replace(0, float("nan")) * 100


def daily_vwap_dist_series(df: pd.DataFrame) -> pd.Series:
    tp     = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = tp * df["volume"]
    date   = df.index.normalize()          # floor to day for groupby
    cum_tv = tp_vol.groupby(date).cumsum()
    cum_v  = df["volume"].groupby(date).cumsum()
    vwap   = cum_tv / cum_v.replace(0, float("nan"))
    return (df["close"] - vwap) / vwap.replace(0, float("nan")) * 100


def stoch_k_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    ll  = low.rolling(period).min()
    hh  = high.rolling(period).max()
    rng = (hh - ll).replace(0, float("nan"))
    return (close - ll) / rng * 100


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    cp = close.shift(1)
    tr = pd.concat([high - low, (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def chg_4h_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr  = atr_series(df["high"], df["low"], df["close"], period)
    chg  = df["close"] - df["close"].shift(4)   # 4-bar change (matches iloc[-1] - iloc[-5])
    return chg / atr.replace(0, float("nan"))


def ema_slope_series(close: pd.Series, ema_period: int = 20, lookback: int = 3) -> pd.Series:
    """3-bar percentage slope of EMA. Positive = uptrend, negative = downtrend."""
    ema = close.ewm(span=ema_period, adjust=False).mean()
    return (ema - ema.shift(lookback)) / ema.shift(lookback).replace(0, float("nan")) * 100


def donchian(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int, prefix: str) -> dict:
    """Compute dc_pos, dc_break, dc_width for lookback n."""
    upper = high.rolling(n, min_periods=n // 2).max()
    lower = low.rolling(n,  min_periods=n // 2).min()
    rng   = (upper - lower).replace(0, float("nan"))
    pos   = (close - lower) / rng
    width = rng / close.replace(0, float("nan"))
    # Break: close exceeds the prior N-bar extreme (shift avoids look-ahead)
    prior_upper = high.shift(1).rolling(n, min_periods=n // 2).max()
    prior_lower = low.shift(1).rolling(n,  min_periods=n // 2).min()
    brk = pd.Series(0.0, index=close.index)
    brk[close > prior_upper] =  1.0
    brk[close < prior_lower] = -1.0
    return {f"{prefix}_pos": pos, f"{prefix}_break": brk, f"{prefix}_width": width}


# ── load parquet data ─────────────────────────────────────────────────────────

def load_1h() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1h_1970*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_15m() -> pd.DataFrame:
    """Load 1m parquet files from 2026-01-01 onward and resample to 15m."""
    files = sorted(DATA_DIR.glob("binanceus_BTCUSDT_1m_1970*.parquet"))
    frames = []
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    for f in files:
        df = pd.read_parquet(f)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index >= cutoff]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df1m = pd.concat(frames).sort_index()
    df1m = df1m[~df1m.index.duplicated(keep="last")]
    df15m = df1m.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return df15m


# ── build feature tables ──────────────────────────────────────────────────────

def build_1h_features(df1h: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df1h.index)
    f["rsi_14"]            = rsi_series(df1h["close"], 14)
    f["macd_hist_1h"]      = macd_hist_series(df1h["close"])
    f["bb_pct"]            = bb_pct_series(df1h["close"])
    f["ema50_dist"]        = ema50_dist_series(df1h["close"])
    f["vwap_distance_pct"] = daily_vwap_dist_series(df1h)

    lr = np.log(df1h["close"] / df1h["close"].shift(1))
    vol_24h  = lr.rolling(24,  min_periods=4).std()
    vol_168h = lr.rolling(168, min_periods=24).std()
    f["rvol_inv"] = (vol_168h / vol_24h.replace(0, float("nan"))).clip(0.3, 2.0)

    for n in [20, 55]:
        for k, v in donchian(df1h["high"], df1h["low"], df1h["close"], n, f"dc_1h_n{n}").items():
            f[k] = v
    f["ema20_slope_1h"] = ema_slope_series(df1h["close"], 20, 3)
    f["ema50_slope_1h"] = ema_slope_series(df1h["close"], 50, 3)
    return f


def build_4h_features(df1h: pd.DataFrame) -> pd.DataFrame:
    df4h = df1h.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    f = pd.DataFrame(index=df4h.index)
    f["stoch_k_4h"] = stoch_k_series(df4h["high"], df4h["low"], df4h["close"], 14)
    f["rsi_4h"]     = rsi_series(df4h["close"], 14)
    f["chg_4h_atr"] = chg_4h_atr_series(df4h)
    for n in [20, 55]:
        for k, v in donchian(df4h["high"], df4h["low"], df4h["close"], n, f"dc_4h_n{n}").items():
            f[k] = v
    f["ema20_slope_4h"] = ema_slope_series(df4h["close"], 20, 3)
    f["ema50_slope_4h"] = ema_slope_series(df4h["close"], 50, 3)
    return f


def build_1d_features(df1h: pd.DataFrame) -> pd.DataFrame:
    df1d = df1h.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    f = pd.DataFrame(index=df1d.index)
    for n in [20, 55]:
        for k, v in donchian(df1d["high"], df1d["low"], df1d["close"], n, f"dc_1d_n{n}").items():
            f[k] = v
    f["ema20_slope_1d"] = ema_slope_series(df1d["close"], 20, 3)
    f["ema50_slope_1d"] = ema_slope_series(df1d["close"], 50, 3)
    return f


def build_15m_features(df15m: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df15m.index)
    for n in [20, 55]:
        for k, v in donchian(df15m["high"], df15m["low"], df15m["close"], n, f"dc_15m_n{n}").items():
            f[k] = v
    f["ema20_slope_15m"] = ema_slope_series(df15m["close"], 20, 3)
    f["ema50_slope_15m"] = ema_slope_series(df15m["close"], 50, 3)
    return f


# ── load paper trades ──────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES_DIR / "paper_trades_archive_*.csv")))
    files += [str(RES_DIR / "paper_trades.csv")]
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
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

    # Numeric coercion for logged signals
    logged_signal_cols = [
        "stoch_k", "composite_trend", "composite_rev", "composite_p_up",
        "ema_stack_bias", "ema_stretch_score", "vwap_score",
        "confirmation_bias", "stoch_bias", "vpin_score",
        "pm_drift_5m", "rvol_1h",
    ]
    for col in logged_signal_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    return trades.reset_index(drop=True)


# ── main ───────────────────────────────────────────────────────────────────────

def run():
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    clf = pipe["clf"]

    print("Loading 1h parquet data...")
    df1h = load_1h()
    print(f"  1h bars: {len(df1h):,}  ({df1h.index.min().date()} → {df1h.index.max().date()})")

    print("Computing 1h features (+ Donchian)...")
    feat_1h = build_1h_features(df1h)

    print("Computing 4h features (+ Donchian)...")
    feat_4h = build_4h_features(df1h)

    print("Computing 1d Donchian features...")
    feat_1d = build_1d_features(df1h)

    print("Loading 1m data and resampling to 15m (from 2026-01-01)...")
    df15m = load_15m()
    print(f"  15m bars: {len(df15m):,}")
    feat_15m = build_15m_features(df15m)

    print("Loading paper trades...")
    trades = load_trades()
    print(f"  Resolved trades: {len(trades):,}")

    # Map each trade to the last COMPLETED bar for each timeframe
    trades["bar_15m"] = trades["logged_at"].dt.floor("15min") - pd.Timedelta(minutes=15)
    trades["bar_1h"]  = trades["logged_at"].dt.floor("1h")   - pd.Timedelta(hours=1)
    trades["bar_4h"]  = trades["logged_at"].dt.floor("4h")   - pd.Timedelta(hours=4)
    trades["bar_1d"]  = trades["logged_at"].dt.floor("1D")   - pd.Timedelta(days=1)

    def safe_join(df, feat_table, on_col):
        renamed = feat_table.add_suffix("_computed")
        df = df.join(renamed, on=on_col, how="left")
        for col in feat_table.columns:
            df[col] = df[col + "_computed"]
            df.drop(columns=[col + "_computed"], inplace=True)
        return df

    trades = safe_join(trades, feat_1h,  "bar_1h")
    trades = safe_join(trades, feat_4h,  "bar_4h")
    trades = safe_join(trades, feat_1d,  "bar_1d")
    trades = safe_join(trades, feat_15m, "bar_15m")

    # Map logged signals to model feature names
    col_map = {
        "stoch_k":           "stoch_k",
        "composite_trend":   "composite_trend",
        "composite_rev":     "composite_rev",
        "composite_p_up":    "composite_p_up",
        "ema_stack_bias":    "ema_stack_bias",
        "ema_stretch_score": "ema_stretch_score",
        "vwap_score":        "vwap_stretch_score",   # vwap_score ≈ vwap_stretch_score
        "confirmation_bias": "confirmation_bias",
        "stoch_bias":        "stoch_bias",
        "vpin_score":        "vpin_score",
        "pm_drift_5m":       "pm_drift_5m",
        "rvol_1h":           "rvol_1h",
    }
    for src, dst in col_map.items():
        if src in trades.columns and dst not in trades.columns:
            trades[dst] = trades[src]
        elif src in trades.columns and dst != src:
            trades[dst] = trades[src]

    # Build feature matrix
    feat_mat = trades[FEATURES].astype(float).values

    print("Running LightGBM inference...")
    probs = clf.predict_proba(feat_mat)[:, 1]
    probs = np.clip(probs, 0.02, 0.98)

    trades["p_up_v2_backfilled"] = probs

    # Coverage report
    print(f"\nFeature coverage (of {len(trades):,} trades):")
    for feat in FEATURES:
        if feat in trades.columns:
            nn = pd.to_numeric(trades[feat], errors="coerce").notna().sum()
            print(f"  {feat:<25} {nn:>5,} ({nn/len(trades):.0%})")

    # Collect all computed signal columns
    dc_cols    = [c for c in trades.columns if c.startswith("dc_")]
    slope_cols = [c for c in trades.columns if "slope" in c]
    stoch_cols = [c for c in trades.columns if c.startswith("stoch_k_") or c.startswith("rsi_")]

    # Write output
    trades["rvol_inv_backfilled"] = trades["rvol_inv"]
    out_cols = (["contract_ticker", "logged_at", "side",
                 "p_up_v2_backfilled", "rvol_inv_backfilled"]
                + dc_cols + slope_cols + stoch_cols)
    out = trades[[c for c in out_cols if c in trades.columns]].copy()
    out_path = RES_DIR / "p_up_v2_backfilled.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out):,} rows,  "
          f"{len(dc_cols)} Donchian + {len(slope_cols)} EMA slope cols)")

    rv = trades["rvol_inv_backfilled"]
    print(f"p_up_v2:  mean={probs.mean():.3f}  std={probs.std():.3f}")
    print(f"rvol_inv: mean={rv.mean():.3f}  std={rv.std():.3f}  coverage={rv.notna().mean():.0%}")

    print(f"\nDonchian coverage sample:")
    for col in dc_cols[:6]:
        nn = trades[col].notna().sum()
        print(f"  {col:<25} {nn:,} / {len(trades):,} ({nn/len(trades):.0%})")


def backfill_scan_archive():
    """Backfill p_up_v2 + dc_4h_n20_pos for all scan archive rows.

    Uses the same LGBM model and feature pipeline as the paper trade backfill.
    Missing logged signals (stoch_bias, confirmation_bias) will be NaN — LightGBM handles them.
    Output: results/scan_archive_backfilled.csv
    """
    scan_path = RES_DIR / "btc_scan_archive.csv"
    if not scan_path.exists():
        print("  btc_scan_archive.csv not found — skipping")
        return

    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    clf = pipe["clf"]

    print("Loading 1h parquet data...")
    df1h = load_1h()
    print(f"  1h bars: {len(df1h):,}  ({df1h.index.min().date()} → {df1h.index.max().date()})")

    print("Computing 1h features...")
    feat_1h = build_1h_features(df1h)
    print("Computing 4h features...")
    feat_4h = build_4h_features(df1h)

    print("Loading scan archive...")
    scan = pd.read_csv(scan_path, low_memory=False)
    scan["logged_at"] = pd.to_datetime(scan["logged_at"], errors="coerce", utc=True)
    scan = scan.dropna(subset=["logged_at"]).copy()
    print(f"  {len(scan):,} rows  ({scan['logged_at'].min().date()} → {scan['logged_at'].max().date()})")

    # Map to last completed bar
    scan["bar_1h"] = scan["logged_at"].dt.floor("1h") - pd.Timedelta(hours=1)
    scan["bar_4h"] = scan["logged_at"].dt.floor("4h") - pd.Timedelta(hours=4)

    def safe_join(df, feat_table, on_col):
        renamed = feat_table.add_suffix("_computed")
        df = df.join(renamed, on=on_col, how="left")
        for col in feat_table.columns:
            df[col] = df[col + "_computed"]
            df.drop(columns=[col + "_computed"], inplace=True)
        return df

    scan = safe_join(scan, feat_1h, "bar_1h")
    scan = safe_join(scan, feat_4h, "bar_4h")

    # Map scan archive column names to LGBM feature names
    # (scan archive uses same names as paper trades for most signals)
    col_map = {
        "vwap_stretch_score": "vwap_stretch_score",  # already correct name
        "rvol_1h":            "rvol_1h",
    }
    for src, dst in col_map.items():
        if src in scan.columns and dst not in scan.columns:
            scan[dst] = scan[src]

    # Numeric coerce all LGBM features
    for feat in FEATURES:
        if feat in scan.columns:
            scan[feat] = pd.to_numeric(scan[feat], errors="coerce")
        else:
            scan[feat] = float("nan")

    # Coverage report
    print(f"\n  Feature coverage ({len(scan):,} rows):")
    for feat in FEATURES:
        nn = scan[feat].notna().sum()
        mark = "" if nn / len(scan) > 0.5 else "  ← sparse"
        print(f"    {feat:<25} {nn:>6,} ({nn/len(scan):.0%}){mark}")

    feat_mat = scan[FEATURES].astype(float).values
    print("\nRunning LightGBM inference...")
    probs = clf.predict_proba(feat_mat)[:, 1]
    probs = np.clip(probs, 0.02, 0.98)
    scan["p_up_v2_backfilled"] = probs

    dc_4h_cols = [c for c in feat_4h.columns if "dc_4h" in c]
    out_cols = (["logged_at", "contract_ticker", "p_up_v2_backfilled"] + dc_4h_cols)
    out = scan[[c for c in out_cols if c in scan.columns]].copy()
    out_path = RES_DIR / "scan_archive_backfilled.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out):,} rows)")
    print(f"p_up_v2: mean={probs.mean():.3f}  std={probs.std():.3f}")


def backfill_scan_archive_dc():
    """Compute dc_4h_n20_pos for scan archive rows and save to scan_archive_dc.csv."""
    scan_path = RES_DIR / "btc_scan_archive.csv"
    if not scan_path.exists():
        print("  btc_scan_archive.csv not found — skipping")
        return

    print("Loading 1h parquet for scan archive DC backfill...")
    df1h = load_1h()
    feat_4h = build_4h_features(df1h)

    print("Loading scan archive...")
    scan = pd.read_csv(scan_path, low_memory=False)
    scan["logged_at"] = pd.to_datetime(scan["logged_at"], errors="coerce", utc=True)
    scan = scan.dropna(subset=["logged_at"]).copy()
    print(f"  {len(scan):,} rows  ({scan['logged_at'].min().date()} → {scan['logged_at'].max().date()})")

    scan["bar_4h"] = scan["logged_at"].dt.floor("4h") - pd.Timedelta(hours=4)

    dc_4h_cols = [c for c in feat_4h.columns if "dc_4h" in c]
    renamed = feat_4h[dc_4h_cols].add_suffix("_computed")
    scan = scan.join(renamed, on="bar_4h", how="left")
    for col in dc_4h_cols:
        scan[col] = scan[col + "_computed"]
        scan.drop(columns=[col + "_computed"], inplace=True)

    out_cols = ["logged_at", "contract_ticker"] + dc_4h_cols
    out = scan[[c for c in out_cols if c in scan.columns]].copy()
    out_path = RES_DIR / "scan_archive_dc.csv"
    out.to_csv(out_path, index=False)

    for col in dc_4h_cols:
        nn = scan[col].notna().sum()
        print(f"  {col:<25} {nn:,} / {len(scan):,} ({nn/len(scan):.0%})")
    print(f"Wrote {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    run()
    backfill_scan_archive_dc()
