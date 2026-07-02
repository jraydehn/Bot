"""
build_15m_directional.py
------------------------
Train a LightGBM directional model: predict P(close > open) for the NEXT
15m bar using price-action features only (no offset or z_score).

The predicted p_up is then mapped to a log-normal drift:
  z_drift   = Φ⁻¹(p_up)
  p_yes(K)  = Φ(-z_K + z_drift)    z_K = log(K/spot) / sigma_tau

This is the same architecture as the BTC 1h model (log-normal base + drift)
but with the drift learned from data rather than a heuristic composite.

Walk-forward:
  Train : 2024-01-01 – 2024-12-31
  Val   : 2025-01-01 – 2025-06-30
  Test  : 2025-07-01 – present

Models saved to: models/lgbm_15m_dir_{sym}.pkl
"""

import os, glob, math, pickle
import pandas as pd
import numpy as np
from scipy.stats import norm

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_END   = "2025-01-01"
VAL_END     = "2025-07-01"
TAU_MIN     = 15.0
STAKE       = 50.0
RAKE        = 0.07
EDGE_THRESH = 0.04
OFFSETS     = [-0.005, -0.0025, 0.0, 0.0025, 0.005, 0.010]

# Price-action features only; offset_pct and z_score intentionally excluded.
FEATURE_COLS = [
    "bp_15m", "body_15m", "dir_15m", "chg_15m", "stoch_k_15m",
    "bp_5m",  "body_5m",  "dir_5m",  "chg_5m",  "stoch_k_5m",  "vol_ratio_5m",
    "chg_1h", "bp_1h",    "stoch_k_1h", "ema_bias_1h", "consec_dir_1h",
    "vol_ratio_1h", "realized_vol_annual",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

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

    c1m = df1m["close"].astype(float)
    v1m = df1m["volume"].astype(float)

    # ── 5m ───────────────────────────────────────────────────────────────────
    df5 = df1m.resample("5min", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"), volume=("volume","sum")
    ).dropna()
    o5,h5,l5,c5 = df5["open"],df5["high"],df5["low"],df5["close"]
    df5["bp"]    = bp_series(o5,h5,l5,c5)
    df5["body"]  = body_series(o5,h5,l5,c5)
    df5["dir"]   = np.sign(c5 - o5)
    df5["chg"]   = c5.pct_change() * 100
    df5["stoch"] = stoch_k_series(c5,h5,l5, 14)
    # price-based vol ratio: 12-bar (1h) rolling rv / 24h rolling median
    _lr5 = np.log(c5 / c5.shift(1))
    _rv5 = _lr5.rolling(12).std()
    df5["vol_r"] = (_rv5 / _rv5.rolling(288).median().replace(0, np.nan)).clip(0, 5)
    df5 = df5.shift(1)  # no look-ahead

    # ── 15m ──────────────────────────────────────────────────────────────────
    df15 = df1m.resample("15min", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"), volume=("volume","sum")
    ).dropna()
    o15,h15,l15,c15 = df15["open"],df15["high"],df15["low"],df15["close"]
    df15["bp"]    = bp_series(o15,h15,l15,c15)
    df15["body"]  = body_series(o15,h15,l15,c15)
    df15["dir"]   = np.sign(c15 - o15)
    df15["chg"]   = c15.pct_change() * 100
    df15["stoch"] = stoch_k_series(c15,h15,l15, 14)

    # spot = open of THIS bar (decision time); future_close = close of THIS bar (outcome)
    base = df15[["bp","body","dir","chg","stoch"]].copy()
    base.columns = ["bp_15m","body_15m","dir_15m","chg_15m","stoch_k_15m"]
    base["spot"]         = o15
    base["future_close"] = c15
    base = base.copy()
    # shift candle features: at bar T open, we see bar T-1's completed candle
    base[["bp_15m","body_15m","dir_15m","chg_15m","stoch_k_15m"]] = (
        base[["bp_15m","body_15m","dir_15m","chg_15m","stoch_k_15m"]].shift(1)
    )

    # ── 1h ───────────────────────────────────────────────────────────────────
    df1h = df1m.resample("1h", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"), volume=("volume","sum")
    ).dropna()
    o1h,h1h,l1h,c1h = df1h["open"],df1h["high"],df1h["low"],df1h["close"]
    ema5  = c1h.ewm(span=5,  adjust=False).mean()
    ema20 = c1h.ewm(span=20, adjust=False).mean()
    df1h["chg"]      = c1h.pct_change() * 100
    df1h["bp"]       = bp_series(o1h,h1h,l1h,c1h)
    df1h["stoch"]    = stoch_k_series(c1h,h1h,l1h, 14)
    df1h["ema_bias"] = np.sign(ema5 - ema20)
    # consec_dir: cumulative bar direction streak (reset on direction change)
    _dir1h = np.sign(c1h - o1h)
    _consec = [0]
    for i in range(1, len(_dir1h)):
        if _dir1h.iloc[i] == _dir1h.iloc[i-1] and _dir1h.iloc[i] != 0:
            _consec.append(_consec[-1] + int(_dir1h.iloc[i]))
        else:
            _consec.append(int(_dir1h.iloc[i]))
    df1h["consec"] = _consec
    # vol ratio: 1h realized vol / 14-day rolling median
    _lr1h = np.log(c1h / c1h.shift(1))
    _rv1h = _lr1h.rolling(60).std() * np.sqrt(60)
    df1h["vol_r"] = _rv1h / _rv1h.rolling(14*24).median().replace(0, np.nan)
    # annualized realized vol from 1m returns (60-bar window)
    _lr1m   = np.log(c1m / c1m.shift(1))
    _rv1m   = _lr1m.rolling(60).std() * np.sqrt(525600)
    df1h["rv_ann"] = _rv1m.resample("1h").last()
    df1h = df1h.shift(1)  # no look-ahead

    # ── Merge onto 15m grid ──────────────────────────────────────────────────
    df5_r = df5[["bp","body","dir","chg","stoch","vol_r"]].copy()
    df5_r.columns = ["bp_5m","body_5m","dir_5m","chg_5m","stoch_k_5m","vol_ratio_5m"]
    base = pd.merge_asof(base.sort_index(), df5_r.sort_index(),
                         left_index=True, right_index=True, direction="backward")

    df1h_r = df1h[["chg","bp","stoch","ema_bias","consec","vol_r","rv_ann"]].copy()
    df1h_r.columns = ["chg_1h","bp_1h","stoch_k_1h","ema_bias_1h",
                      "consec_dir_1h","vol_ratio_1h","realized_vol_annual"]
    base = pd.merge_asof(base.sort_index(), df1h_r.sort_index(),
                         left_index=True, right_index=True, direction="backward")

    base["sym"]    = sym
    # Direction target: did price close higher than it opened in THIS bar?
    base["target"] = (base["future_close"] > base["spot"]).astype(int)

    return base.dropna(subset=["spot","future_close"])


# ── Training ──────────────────────────────────────────────────────────────────

def train_directional(base: pd.DataFrame, sym: str):
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        print("  lightgbm/sklearn not installed.")
        return None

    df = base.copy()
    df["ts"] = df.index

    train = df[df["ts"] < TRAIN_END]
    val   = df[(df["ts"] >= TRAIN_END) & (df["ts"] < VAL_END)]
    test  = df[df["ts"] >= VAL_END]

    print(f"  Split: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"  Direction base rates — train={train['target'].mean():.3f}  "
          f"val={val['target'].mean():.3f}  test={test['target'].mean():.3f}")

    X_tr = train[FEATURE_COLS].fillna(0); y_tr = train["target"]
    X_va = val[FEATURE_COLS].fillna(0);   y_va = val["target"]
    X_te = test[FEATURE_COLS].fillna(0);  y_te = test["target"]

    params = {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "n_estimators":      500,
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 200,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.5,
        "reg_lambda":        0.5,
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

    for name, X, y in [("Val", X_va, y_va), ("Test", X_te, y_te)]:
        p = model.predict_proba(X)[:, 1]
        print(f"  {name}: AUC={roc_auc_score(y, p):.4f}  "
              f"Brier={brier_score_loss(y, p):.4f}  "
              f"mean_p={p.mean():.3f}")

    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  Top-10 feature importances:")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:<24} {imp:>6.0f}")

    # Isotonic calibration on val set
    print(f"\n  Calibrating (isotonic) on val set...", end="", flush=True)
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal_model.fit(X_va, y_va)
    except ImportError:
        cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal_model.fit(X_va, y_va)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    p_cal = cal_model.predict_proba(X_te)[:, 1]
    print(f" done.  Test AUC={roc_auc_score(y_te, p_cal):.4f}  "
          f"Brier={brier_score_loss(y_te, p_cal):.4f}  (calibrated)")

    # p_up distribution: are predictions actually spread away from 0.5?
    print(f"\n  Calibrated p_up distribution (test):")
    for lo, hi in [(0.3,0.4),(0.4,0.45),(0.45,0.5),(0.5,0.55),(0.55,0.6),(0.6,0.7)]:
        mask = (p_cal >= lo) & (p_cal < hi)
        n = mask.sum()
        if n == 0: continue
        actual_wr = y_te[mask].mean()
        print(f"    p_up=[{lo:.2f},{hi:.2f}): n={n:>6,}  actual_up={actual_wr:.3f}")

    path = os.path.join(MODELS_DIR, f"lgbm_15m_dir_{sym.lower()}.pkl")
    with open(path, "wb") as f:
        pickle.dump(cal_model, f)
    print(f"\n  Model saved → {path}")
    return cal_model


# ── PnL simulation ────────────────────────────────────────────────────────────

def eval_pnl(base: pd.DataFrame, model, sym: str):
    """
    For each test bar:
      1. Get p_up from model → z_drift = Φ⁻¹(p_up)
      2. For each offset K:  p_yes = Φ(-z_K + z_drift)
      3. Edge_yes = p_yes - Φ(-z_K)  (vs log-normal baseline)
      4. Trade YES if edge_yes > thresh, NO if edge_yes < -thresh (p_no has edge)
      5. resolved_yes = int(future_close >= K)
    """
    df = base.copy()
    df["ts"] = df.index
    test = df[df["ts"] >= VAL_END].copy()

    X = test[FEATURE_COLS].fillna(0)
    test["p_up"] = model.predict_proba(X)[:, 1]
    test["z_drift"] = norm.ppf(test["p_up"].clip(0.01, 0.99))

    rows = []
    for ts, row in test.iterrows():
        spot  = float(row["spot"])
        fc    = float(row["future_close"])
        rv    = float(row.get("realized_vol_annual", 0.3))
        rv_pm = rv / math.sqrt(525600)
        sigma = max(rv_pm * math.sqrt(TAU_MIN), 1e-6)
        zd    = float(row["z_drift"])
        p_up  = float(row["p_up"])

        for off in OFFSETS:
            K = spot * (1 + off)
            z_K = math.log(K / spot) / sigma if spot > 0 else 0.0
            p_lognorm = norm.cdf(-z_K)              # baseline (no drift)
            p_yes_mod = norm.cdf(-z_K + zd)         # drift-adjusted
            edge_yes  = p_yes_mod - p_lognorm
            resolved  = int(fc >= K)
            rows.append({
                "ts": ts, "offset_pct": off * 100,
                "z_K": z_K, "p_up": p_up, "p_lognorm": p_lognorm,
                "p_yes_mod": p_yes_mod, "edge_yes": edge_yes,
                "resolved_yes": resolved,
            })

    sim = pd.DataFrame(rows)
    print(f"\n  PnL simulation (test {VAL_END}+, flat ${STAKE}/trade, sym={sym})")

    # edge_yes distribution
    print(f"  edge_yes stats: mean={sim['edge_yes'].mean():.4f}  "
          f"std={sim['edge_yes'].std():.4f}  "
          f"p5={sim['edge_yes'].quantile(0.05):.4f}  "
          f"p95={sim['edge_yes'].quantile(0.95):.4f}")

    for side in ["yes", "no"]:
        if side == "yes":
            trades = sim[sim["edge_yes"] >= EDGE_THRESH].copy()
            trades["win"] = trades["resolved_yes"]
        else:
            # NO edge = model thinks p_yes < log-normal → edge_no = -edge_yes
            trades = sim[sim["edge_yes"] <= -EDGE_THRESH].copy()
            trades["win"] = 1 - trades["resolved_yes"]

        if len(trades) == 0:
            print(f"  {side.upper()}: no qualifying trades at thresh={EDGE_THRESH}")
            continue

        wins = trades["win"].sum()
        n    = len(trades)
        pnl  = wins * STAKE * (1 - RAKE) - (n - wins) * STAKE
        wr   = wins / n
        be   = STAKE / (STAKE * (1 - RAKE) + STAKE)   # breakeven WR
        print(f"\n  {side.upper()}: n={n:,}  WR={wr:.1%}  PnL=${pnl:+,.0f}  "
              f"(breakeven={be:.1%})")
        print(f"    By offset:")
        for off in sorted(trades["offset_pct"].unique()):
            sub = trades[trades["offset_pct"] == off]
            if len(sub) < 20: continue
            w = sub["win"].sum(); l = len(sub) - w
            p = w * STAKE * (1 - RAKE) - l * STAKE
            print(f"      offset={off:+.2f}%: n={len(sub):>5,}  "
                  f"WR={sub['win'].mean():.1%}  PnL=${p:+,.0f}")

    # p_up decile calibration check
    print(f"\n  p_up decile calibration check (actual up-rate in test):")
    test["p_up_bin"] = pd.cut(test["p_up"], bins=10)
    cal_check = test.groupby("p_up_bin", observed=True).agg(
        n=("target", "count"), actual=("target", "mean"), mean_pred=("p_up", "mean")
    )
    for _, row2 in cal_check.iterrows():
        bar = "█" * int(row2["actual"] * 20)
        print(f"    {str(row2.name):<18}  n={int(row2['n']):>5,}  "
              f"pred={row2['mean_pred']:.3f}  actual={row2['actual']:.3f}  {bar}")


# ── Inference helper (import from paper_trade_runner_15m.py) ──────────────────

def compute_directional_features(sig: dict) -> list:
    """
    Build the 18-feature list for inference (same order as FEATURE_COLS).
    sig keys: same as compute_inference_features() in build_15m_model.py,
    minus offset_pct and z_score.
    """
    return [
        sig.get("bp_15m",           0.5),
        sig.get("body_15m",          0.0),
        sig.get("dir_15m",           0.0),
        sig.get("chg_15m",           0.0),
        sig.get("stoch_k_15m",       50.0),
        sig.get("bp_5m",             0.5),
        sig.get("body_5m",           0.0),
        sig.get("dir_5m",            0.0),
        sig.get("chg_5m",            0.0),
        sig.get("stoch_k_5m",        50.0),
        sig.get("vol_ratio_5m",      1.0),
        sig.get("chg_1h",            0.0),
        sig.get("bp_1h",             0.5),
        sig.get("stoch_k_1h",        50.0),
        sig.get("ema_bias_1h",       0.0),
        sig.get("consec_dir_1h",     0.0),
        sig.get("vol_ratio_1h",      1.0),
        sig.get("realized_vol_annual", 0.3),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(assets=("BTC", "ETH", "SOL")):
    for sym in assets:
        print(f"\n{'='*60}")
        print(f"  Directional model: {sym}")
        print(f"{'='*60}")

        print("  Step 1: Building features...")
        base = build_features(sym)
        print(f"  Rows: {len(base):,}  up_rate={base['target'].mean():.3f}")

        print("\n  Step 2: Training directional model...")
        model = train_directional(base, sym)

        if model is not None:
            print("\n  Step 3: PnL simulation...")
            eval_pnl(base, model, sym)


if __name__ == "__main__":
    import sys
    assets = sys.argv[1:] if len(sys.argv) > 1 else ["BTC", "ETH", "SOL"]
    run(assets)
