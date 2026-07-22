"""
build_15m_model.py
------------------
Build a LightGBM model to replace the 15m heuristic probability model.

Pipeline:
  1. Generate labeled training rows from 2-year 1m parquets.
     For every 15m bar across BTC/ETH/SOL, and multiple strike offsets,
     compute features at bar-open and label whether price closed above the
     floor_strike at bar-close (15 minutes later).
  2. Walk-forward train/validate/test:
       Train  : 2024-01-01 – 2024-12-31
       Val    : 2025-01-01 – 2025-06-30
       Test   : 2025-07-01 – 2026-05-13
  3. Evaluate calibration, AUC, and flat-$50 PnL vs p_market edge threshold.
  4. Save the trained model to models/lgbm_15m_{asset}.pkl for inference.

Features (all computed at bar-open, shift=1 to avoid look-ahead):
  Contract:    offset_pct, tau_minutes (fixed at 15), log(K/S)/sigma
  15m candle:  chg_15m, body_15m, bp_15m, dir_15m, stoch_k_15m
  5m candle:   chg_5m, bp_5m, body_5m, stoch_k_5m
  1m:          chg_1m, bp_1m
  1h context:  chg_1h, bp_1h, stoch_k_1h, ema_bias_1h, consec_dir_1h, vol_ratio_1h
  Vol:         realized_vol_annual, vol_ratio_5m

Target: resolved_yes = int(close_at_t15 > floor_strike)
"""

import os, glob, math, pickle
import pandas as pd
import numpy as np
from scipy.stats import norm

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Offsets to simulate (fraction of spot price; positive = OTM YES / ITM NO).
# Fallback only -- run() derives a per-asset, real-data-representative grid via
# derive_offset_grid() instead. [2026-07-21] The original 6-point grid left a
# gap between 0% and 0.25% with zero training coverage, while real archive data
# shows 48-64% of actual live candidates fall inside |offset|<0.10% -- measured
# calibration in that zone was 2.3-3.2x worse (Brier) than the covered zone.
OFFSETS = [-0.005, -0.0025, 0.0, 0.0025, 0.005, 0.010]
TAU_MIN = 15.0   # all 15m contracts


def derive_offset_grid(sym: str, n_quantiles: int = 60) -> list:
    """
    Build a per-asset offset grid from real scan-archive offset_pct density,
    instead of the fixed 6-point OFFSETS list. Quantile spacing naturally
    concentrates points where real trading actually clusters (near-ATM) and
    thins out in the tails, matching the empirical distribution rather than
    an arbitrary uniform grid. Falls back to OFFSETS if the archive is
    missing or too small to be informative.
    """
    path = f"results/{sym.lower()}_scan_archive_15m.csv"
    try:
        real = pd.read_csv(path, low_memory=False, usecols=["offset_pct"])
        real["offset_pct"] = pd.to_numeric(real["offset_pct"], errors="coerce")
        real = real["offset_pct"].dropna()
        if len(real) < 500:
            raise ValueError(f"too few real rows ({len(real)}) to derive a grid")
    except Exception as exc:
        print(f"  [offset_grid] {sym}: falling back to fixed OFFSETS ({exc})")
        return OFFSETS

    qs = np.linspace(0.01, 0.99, n_quantiles)
    grid = np.quantile(real.values, qs) / 100.0   # offset_pct is in percent; OFFSETS is a fraction
    grid = sorted(set(round(float(v), 5) for v in grid))
    # keep the widest tail points from the fixed list too, so far-OTM/deep-ITM
    # coverage isn't lost even though real density there is thin
    for tail in (-0.005, 0.010):
        if tail not in grid and (tail < grid[0] or tail > grid[-1]):
            grid.append(tail)
    grid = sorted(grid)
    print(f"  [offset_grid] {sym}: {len(grid)} points, range [{grid[0]*100:+.3f}%, {grid[-1]*100:+.3f}%], "
          f"derived from {len(real):,} real archive rows")
    return grid

TRAIN_END = "2025-07-01"
VAL_END   = "2026-01-01"

# ── Data loading ──────────────────────────────────────────────────────────────

def latest_parquet(sym, tf="1m"):
    pattern = os.path.join(DATA_DIR, f"binanceus_{sym}USDT_{tf}_2024-01-01_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet: {pattern}")
    return files[-1]

def stoch_k_series(c, h, l, period=14):
    lo = l.rolling(period).min()
    hi = h.rolling(period).max()
    return ((c - lo) / (hi - lo).replace(0, np.nan) * 100).clip(0, 100)

def bp_series(o, h, l, c):
    r = h - l
    return ((c - l) / r.replace(0, np.nan)).clip(0, 1).fillna(0.5)

def body_series(o, h, l, c):
    r = h - l
    return ((c - o).abs() / r.replace(0, np.nan)).clip(0, 1).fillna(0)

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(sym: str) -> pd.DataFrame:
    print(f"  Loading {sym} 1m parquet...", end="", flush=True)
    df1m = pd.read_parquet(latest_parquet(sym))
    if df1m.index.tz is None:
        df1m.index = df1m.index.tz_localize("UTC")
    print(f" {len(df1m):,} bars")

    o1m = df1m["open"].astype(float)
    h1m = df1m["high"].astype(float)
    l1m = df1m["low"].astype(float)
    c1m = df1m["close"].astype(float)
    v1m = df1m["volume"].astype(float)

    # ── 5m resampled ─────────────────────────────────────────────────────────
    df5 = df1m.resample("5min", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna()
    o5,h5,l5,c5,v5 = df5["open"],df5["high"],df5["low"],df5["close"],df5["volume"]
    df5["bp"]    = bp_series(o5,h5,l5,c5)
    df5["body"]  = body_series(o5,h5,l5,c5)
    df5["dir"]   = np.sign(c5 - o5)
    df5["chg"]   = c5.pct_change() * 100
    df5["stoch"] = stoch_k_series(c5,h5,l5, 14)
    # price-based realized vol ratio: rv over last 12 bars (1h) vs 24h rolling median
    # matches the vol_ratio convention used in paper_trade_runner (rv / rolling_median)
    _lr5   = np.log(c5 / c5.shift(1))
    _rv5   = _lr5.rolling(12).std()
    df5["vol_r"] = (_rv5 / _rv5.rolling(288).median().replace(0, np.nan)).clip(0, 5)
    df5 = df5.shift(1)  # no look-ahead

    # ── 15m resampled ─────────────────────────────────────────────────────────
    df15 = df1m.resample("15min", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna()
    o15,h15,l15,c15 = df15["open"],df15["high"],df15["low"],df15["close"]
    df15["bp"]    = bp_series(o15,h15,l15,c15)
    df15["body"]  = body_series(o15,h15,l15,c15)
    df15["dir"]   = np.sign(c15 - o15)
    df15["chg"]   = c15.pct_change() * 100
    df15["stoch"] = stoch_k_series(c15,h15,l15, 14)
    # future close = what we're predicting; spot = open of bar
    df15["future_close"] = c15  # close of THIS bar = outcome
    df15["spot"]         = o15  # open of THIS bar = spot at decision time
    df15 = df15.shift(1, fill_value=np.nan)  # all features from PRIOR bar
    # spot and future_close are NOT shifted — they're the current bar's values
    df15["spot"]         = o15
    df15["future_close"] = c15

    # ── 1h resampled ──────────────────────────────────────────────────────────
    df1h = df1m.resample("1h", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna()
    o1h,h1h,l1h,c1h,v1h = df1h["open"],df1h["high"],df1h["low"],df1h["close"],df1h["volume"]
    ema5_1h  = c1h.ewm(span=5,  adjust=False).mean()
    ema20_1h = c1h.ewm(span=20, adjust=False).mean()
    df1h["chg"]       = c1h.pct_change() * 100
    df1h["bp"]        = bp_series(o1h,h1h,l1h,c1h)
    df1h["stoch"]     = stoch_k_series(c1h,h1h,l1h, 14)
    df1h["ema_bias"]  = np.sign(ema5_1h - ema20_1h)
    df1h["consec"]    = 0
    _dir1h = np.sign(c1h - o1h)
    _consec = [0]
    for i in range(1, len(_dir1h)):
        if _dir1h.iloc[i] == _dir1h.iloc[i-1] and _dir1h.iloc[i] != 0:
            _consec.append(_consec[-1] + int(_dir1h.iloc[i]))
        else:
            _consec.append(int(_dir1h.iloc[i]))
    df1h["consec"] = _consec
    # vol ratio: 1h realized vs 2-week median
    log_ret_1h   = np.log(c1h / c1h.shift(1))
    rv_1h        = log_ret_1h.rolling(60).std() * np.sqrt(60)
    df1h["vol_r"] = rv_1h / rv_1h.rolling(14*24).median().replace(0,np.nan)
    # realized vol annualized (from 1m)
    log_ret_1m   = np.log(c1m / c1m.shift(1))
    rv_1m_ann    = log_ret_1m.rolling(60).std() * np.sqrt(525600)
    df1h["rv_ann"] = rv_1m_ann.resample("1h").last()
    df1h = df1h.shift(1)  # no look-ahead

    # ── Merge onto 15m grid ───────────────────────────────────────────────────
    base = df15[["spot","future_close","bp","body","dir","chg","stoch"]].copy()
    base.columns = ["spot","future_close",
                    "bp_15m","body_15m","dir_15m","chg_15m","stoch_k_15m"]

    # 5m: take the last 5m bar before each 15m bar
    df5_r = df5[["bp","body","dir","chg","stoch","vol_r"]].copy()
    df5_r.columns = ["bp_5m","body_5m","dir_5m","chg_5m","stoch_k_5m","vol_ratio_5m"]
    base = pd.merge_asof(base.sort_index(), df5_r.sort_index(),
                         left_index=True, right_index=True, direction="backward")

    # 1h: take the last 1h bar before each 15m bar
    df1h_r = df1h[["chg","bp","stoch","ema_bias","consec","vol_r","rv_ann"]].copy()
    df1h_r.columns = ["chg_1h","bp_1h","stoch_k_1h","ema_bias_1h",
                      "consec_dir_1h","vol_ratio_1h","realized_vol_annual"]
    base = pd.merge_asof(base.sort_index(), df1h_r.sort_index(),
                         left_index=True, right_index=True, direction="backward")

    base["sym"] = sym
    return base.dropna(subset=["spot","future_close"])


def generate_training_rows(base: pd.DataFrame, offsets=None) -> pd.DataFrame:
    """Expand each 15m bar into multiple contract rows (one per offset)."""
    offsets = offsets if offsets is not None else OFFSETS
    rows = []
    for ts, row in base.iterrows():
        spot = float(row["spot"])
        fc   = float(row["future_close"])
        if spot <= 0 or fc <= 0:
            continue
        for off in offsets:
            floor_k = spot * (1 + off)
            resolved_yes = int(fc >= floor_k)
            # log-normal z as a feature
            rv = float(row.get("realized_vol_annual", 0.3)) / math.sqrt(525600)
            sigma_tau = rv * math.sqrt(TAU_MIN) if rv > 0 else 0.001
            z = math.log(floor_k / spot) / sigma_tau if sigma_tau > 0 else 0.0

            r = {
                "ts":              ts,
                "sym":             row["sym"],
                "offset_pct":      off * 100,
                "z_score":         z,
                "resolved_yes":    resolved_yes,
                # 15m features
                "bp_15m":          row.get("bp_15m",   0.5),
                "body_15m":        row.get("body_15m", 0.0),
                "dir_15m":         row.get("dir_15m",  0.0),
                "chg_15m":         row.get("chg_15m",  0.0),
                "stoch_k_15m":     row.get("stoch_k_15m", 50.0),
                # 5m features
                "bp_5m":           row.get("bp_5m",    0.5),
                "body_5m":         row.get("body_5m",  0.0),
                "dir_5m":          row.get("dir_5m",   0.0),
                "chg_5m":          row.get("chg_5m",   0.0),
                "stoch_k_5m":      row.get("stoch_k_5m", 50.0),
                "vol_ratio_5m":    row.get("vol_ratio_5m", 1.0),
                # 1h context
                "chg_1h":          row.get("chg_1h",   0.0),
                "bp_1h":           row.get("bp_1h",    0.5),
                "stoch_k_1h":      row.get("stoch_k_1h", 50.0),
                "ema_bias_1h":     row.get("ema_bias_1h", 0.0),
                "consec_dir_1h":   row.get("consec_dir_1h", 0.0),
                "vol_ratio_1h":    row.get("vol_ratio_1h", 1.0),
                "realized_vol_annual": row.get("realized_vol_annual", 0.3),
            }
            rows.append(r)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "offset_pct", "z_score",
    "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "bp_5m",  "body_5m",  "dir_5m",  "chg_5m",  "stoch_k_5m", "vol_ratio_5m",
    "chg_1h", "bp_1h", "stoch_k_1h", "ema_bias_1h", "consec_dir_1h",
    "vol_ratio_1h", "realized_vol_annual",
]
# Price-action only: drop offset_pct and z_score to measure true predictive signal
# beyond log-normal; if AUC drops near 0.5 the model is mostly learning offset shape.
FEATURE_COLS_NO_Z = [c for c in FEATURE_COLS if c not in ("offset_pct", "z_score")]


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, sym: str, feature_cols=None, label=""):
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        print("  lightgbm not installed. Run: pip install lightgbm scikit-learn")
        return None

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])

    train = df[df["ts"] < TRAIN_END]
    val   = df[(df["ts"] >= TRAIN_END) & (df["ts"] < VAL_END)]
    test  = df[df["ts"] >= VAL_END]

    if not label:
        print(f"\n  Split sizes — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    X_tr = train[feature_cols].fillna(0)
    y_tr = train["resolved_yes"]
    X_va = val[feature_cols].fillna(0)
    y_va = val["resolved_yes"]
    X_te = test[feature_cols].fillna(0)
    y_te = test["resolved_yes"]

    params = {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "n_estimators":      500,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 100,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    tag = f"  [{label}] " if label else "  "
    for split_name, X, y in [("Val", X_va, y_va), ("Test", X_te, y_te)]:
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        brier = brier_score_loss(y, p)
        print(f"{tag}{split_name}: AUC={auc:.4f}  Brier={brier:.4f}  base_rate={y.mean():.3f}")

    # Only do full reporting for the primary (full-feature) model
    if not label:
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n  Top-10 feature importances:")
        for feat, imp in fi.head(10).items():
            print(f"    {feat:<28} {imp:>6.0f}")

        # Isotonic calibration fitted on val set (cv='prefit' reuses already-trained model)
        print(f"\n  Calibrating (isotonic) on val set...", end="", flush=True)
        cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal_model.fit(X_va, y_va)
        p_cal = cal_model.predict_proba(X_te)[:, 1]
        auc_cal   = roc_auc_score(y_te, p_cal)
        brier_cal = brier_score_loss(y_te, p_cal)
        print(f" done.  Test AUC={auc_cal:.4f}  Brier={brier_cal:.4f}  (calibrated)")

        # Save calibrated model
        path = os.path.join(MODELS_DIR, f"lgbm_15m_{sym.lower()}.pkl")
        with open(path, "wb") as f:
            pickle.dump(cal_model, f)
        print(f"  Model saved → {path}")
        return cal_model

    return model


def eval_pnl(df: pd.DataFrame, model, sym: str, edge_thresh: float = 0.04,
             stake: float = 50.0, rake: float = 0.07):
    """
    Simulate flat-$50 PnL using model probability vs a proxy market price.
    Market price proxy: log-normal P(price > K) using realized vol + z_score.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    test = df[df["ts"] >= VAL_END].copy()

    X = test[FEATURE_COLS].fillna(0)
    test["p_model"] = model.predict_proba(X)[:, 1]

    # Proxy for p_market: sigmoid of z_score (rough log-normal)
    test["p_market_proxy"] = norm.cdf(-test["z_score"])  # P(price > K) = 1-Φ(z)

    print(f"\n  PnL simulation (test set {VAL_END}+, flat ${stake}/trade, sym={sym})")
    for side in ["yes", "no"]:
        if side == "yes":
            test["edge"] = test["p_model"] - test["p_market_proxy"]
            test["win"]  = test["resolved_yes"]
        else:
            test["edge"] = (1 - test["p_model"]) - (1 - test["p_market_proxy"])
            test["win"]  = 1 - test["resolved_yes"]

        qualifying = test[test["edge"] >= edge_thresh]
        if len(qualifying) == 0:
            print(f"  {side.upper()}: no qualifying trades at thresh={edge_thresh:.2f}")
            continue
        wins   = qualifying["win"].sum()
        losses = len(qualifying) - wins
        pnl    = wins * stake * (1 - rake) - losses * stake
        wr     = qualifying["win"].mean()
        print(f"  {side.upper()}: n={len(qualifying):,}  WR={wr:.1%}  PnL=${pnl:+,.0f}")

        # By offset
        print(f"    By offset:")
        for off in sorted(qualifying["offset_pct"].unique()):
            sub = qualifying[qualifying["offset_pct"] == off]
            if len(sub) < 20:
                continue
            w = sub["win"].sum(); l = len(sub) - w
            p = w * stake * (1-rake) - l * stake
            print(f"      offset={off:+.2f}%: n={len(sub):>5,}  WR={sub['win'].mean():.1%}  PnL=${p:+,.0f}")


# ── Inference helper (import this from paper_trade_runner_15m.py) ─────────────

def compute_inference_features(spot: float, strike: float, tau_min: float,
                               sig: dict) -> dict:
    """
    Build the 20-feature dict for a single trade at inference time.
    Must stay in sync with build_features() and generate_training_rows().

    sig keys (all from the prior completed bar, shift=1 convention):
        bp_15m, body_15m, dir_15m, chg_15m, stoch_k_15m
        bp_5m,  body_5m,  dir_5m,  chg_5m,  stoch_k_5m
        vol_ratio_5m  — price-based: rv_5m(12-bar) / rv_5m.rolling(288).median()
        chg_1h, bp_1h, stoch_k_1h, ema_bias_1h, consec_dir_1h, vol_ratio_1h
        realized_vol_annual  — annualized vol from 60 1m log-returns * sqrt(525600)
    """
    rv_annual  = float(sig.get("realized_vol_annual", 0.3))
    rv_per_min = rv_annual / math.sqrt(525600)
    sigma_tau  = max(rv_per_min * math.sqrt(tau_min), 1e-6)
    z          = math.log(strike / spot) / sigma_tau if spot > 0 else 0.0
    off_pct    = (strike / spot - 1.0) * 100.0 if spot > 0 else 0.0

    return {
        "offset_pct":          off_pct,
        "z_score":             z,
        "bp_15m":              sig.get("bp_15m",           0.5),
        "body_15m":            sig.get("body_15m",          0.0),
        "dir_15m":             sig.get("dir_15m",           0.0),
        "chg_15m":             sig.get("chg_15m",           0.0),
        "stoch_k_15m":         sig.get("stoch_k_15m",       50.0),
        "bp_5m":               sig.get("bp_5m",             0.5),
        "body_5m":             sig.get("body_5m",           0.0),
        "dir_5m":              sig.get("dir_5m",            0.0),
        "chg_5m":              sig.get("chg_5m",            0.0),
        "stoch_k_5m":          sig.get("stoch_k_5m",        50.0),
        "vol_ratio_5m":        sig.get("vol_ratio_5m",      1.0),
        "chg_1h":              sig.get("chg_1h",            0.0),
        "bp_1h":               sig.get("bp_1h",             0.5),
        "stoch_k_1h":          sig.get("stoch_k_1h",        50.0),
        "ema_bias_1h":         sig.get("ema_bias_1h",       0.0),
        "consec_dir_1h":       sig.get("consec_dir_1h",     0.0),
        "vol_ratio_1h":        sig.get("vol_ratio_1h",      1.0),
        "realized_vol_annual": rv_annual,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(assets=("BTC", "ETH", "SOL")):
    for sym in assets:
        print(f"\n{'='*60}")
        print(f"  Building 15m model: {sym}")
        print(f"{'='*60}")

        print("  Step 1: Computing features from 1m parquet...")
        base = build_features(sym)
        print(f"  Feature rows: {len(base):,}")

        print("  Step 2: Generating contract training rows...")
        # [2026-07-21] derive_offset_grid() (dense, real-density-matched offsets) was
        # tested and REJECTED -- real-market backtest showed calibration improved but
        # realized PnL got consistently worse (BTC -$3,043, SOL -$9,336 across
        # discovery+holdout; likely overfitting the noisiest near-ATM label region).
        # Left the function in place for reference/future experimentation, but the
        # fixed OFFSETS grid remains the default so a routine retrain doesn't silently
        # reproduce the rejected result. (Also: the 15m models actually in production
        # as of 2026-07-22 are trained on the real Kalshi archive via
        # backfill_real_archive_15m.py + /tmp/train_final_real_archive.py, not this
        # synthetic pipeline at all -- see project_15m_real_archive_retrain memory.)
        df = generate_training_rows(base)
        print(f"  Training rows: {len(df):,}  (YES rate: {df['resolved_yes'].mean():.3f})")

        # Save dataset for inspection
        out = f"results/training_15m_{sym.lower()}.csv"
        df.to_csv(out, index=False)
        print(f"  Dataset saved → {out}")

        print("  Step 3: Training LightGBM...")
        model = train_model(df, sym)

        if model is not None:
            # Compare AUC with vs without z_score/offset_pct to isolate price-action signal
            print(f"\n  --- AUC comparison (price-action signal isolation) ---")
            train_model(df, sym, feature_cols=FEATURE_COLS,      label="full (z+price)")
            train_model(df, sym, feature_cols=FEATURE_COLS_NO_Z, label="no-z (price only)")

            print("\n  Step 4: PnL simulation (test set)...")
            eval_pnl(df, model, sym)


if __name__ == "__main__":
    import sys
    assets = sys.argv[1:] if len(sys.argv) > 1 else ["BTC", "ETH", "SOL"]
    run(assets)
