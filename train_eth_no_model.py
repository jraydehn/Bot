#!/usr/bin/env python3
"""
train_eth_no_model.py — ETH NO-specific strike-resolution predictor.

Target: P(close[t+1h] <= spot[t] × (1+offset)) — P(NO resolves next hour).

Key differences from direct_strike_hit_model.py (which targets P(YES)):
  1. Target flipped: label=1 when price did NOT exceed strike (NO resolved)
  2. Isotonic calibration fit against NO outcome distribution
  3. Live Platt calibration uses NO-side paper trades only
  4. No OTM-YES vol correction (not applicable to NO side)
  5. Output saved to direct_no_model_ETH.pkl

The model learns to identify when price is likely to stay BELOW a given strike —
overbought conditions, bearish momentum, resistance — without the compromise of
also trying to predict the YES side accurately.
"""

import math, sys, glob, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import (
    compute_scores, _stoch_k, _rsi, _bb_pct, _keltner_pct,
    _wpr, _macd_cross, _vol_signal_4h, _dc_pct,
)

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

ASSET       = "ETH"
SYM         = "ETHUSDT"
MODEL_PATH  = OUT_DIR / "direct_no_model_ETH.pkl"

# Training window — start Jul 2025 to drop 2025 bull-run patterns
TRAIN_START = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_START   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_END     = pd.Timestamp("2026-03-16", tz="UTC")
TEST_START  = pd.Timestamp("2026-03-16", tz="UTC")

RECENCY_K   = 2.0   # exponential recency weighting: most-recent ~7x more than oldest

# Same offset grid as direct_strike_hit_model for ETH
OFFSET_GRID = [-0.030, -0.020, -0.015, -0.010, -0.005, -0.0025,
                0.0025,  0.005,  0.010,  0.015,  0.020,  0.030]

FEATURE_COLUMNS = [
    "trend_stoch_4h", "trend_bb_4h", "trend_keltner_4h", "trend_wpr_4h",
    "trend_macd_4h", "trend_vol_4h",
    "rev_rsi_1h", "rev_rsi_4h", "rev_stoch_15m", "rev_stoch_1h",
    "rev_keltner_15m", "rev_dc_15m", "rev_wpr_1h", "rev_move_z",
    "offset_pct", "vol_pm",
    "z_strike", "composite_trend", "composite_rev", "trend_z_24h",
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{SYM}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h


def extract_indicators(d_1m, d_15m, d_1h, d_4h):
    idx = d_1h.index
    out = pd.DataFrame(index=idx)

    # 4h trend layer
    out["trend_stoch_4h"]   = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    out["trend_bb_4h"]      = _bb_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20).reindex(idx, method="ffill")
    kc4_pct, _, _           = _keltner_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20, 2)
    out["trend_keltner_4h"] = kc4_pct.reindex(idx, method="ffill")
    out["trend_wpr_4h"]     = _wpr(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    macd_st = _macd_cross(d_4h["close"]).map(
        {"crossed_up": 2, "up_lag": 1, "none": 0, "down_lag": -1, "crossed_down": -2}
    ).fillna(0)
    out["trend_macd_4h"]    = macd_st.reindex(idx, method="ffill")
    vsig = _vol_signal_4h(d_4h["close"], d_4h["volume"]).map(
        {"high_vol_up": 1, "avg": 0, "low_vol": 0, "high_vol_down": -1}
    ).fillna(0)
    out["trend_vol_4h"]     = vsig.reindex(idx, method="ffill")

    # 1h/15m reversion layer
    out["rev_rsi_1h"]       = _rsi(d_1h["close"], 14)
    out["rev_rsi_4h"]       = _rsi(d_4h["close"], 14).reindex(idx, method="ffill")
    out["rev_stoch_15m"]    = _stoch_k(d_15m["high"], d_15m["low"], d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_stoch_1h"]     = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    kc15_pct, _, _          = _keltner_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20, 2)
    out["rev_keltner_15m"]  = kc15_pct.resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_dc_15m"]       = _dc_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_wpr_1h"]       = _wpr(d_1h["high"], d_1h["low"], d_1h["close"], 14)

    lr_1h    = np.log(d_1h["close"] / d_1h["close"].shift(1))
    roll_vol = lr_1h.rolling(24).std()
    out["rev_move_z"]       = lr_1h / roll_vol.replace(0, float("nan"))

    lr_24h = np.log(d_1h["close"] / d_1h["close"].shift(24))
    out["trend_z_24h"]      = lr_24h / (roll_vol.replace(0, float("nan")) * math.sqrt(24))

    print("    computing composite scores (includes VWAP) …", flush=True)
    trend_votes, rev_votes = compute_scores(
        d_1h["close"], d_1h["high"], d_1h["low"], d_1h["volume"],
        d_4h["close"], d_4h["high"], d_4h["low"], d_4h["volume"],
        d_15m["close"], d_15m["high"], d_15m["low"],
        d_1m["close"], d_1m["volume"],
        ts_1h=idx,
    )
    out["composite_trend"]  = trend_votes
    out["composite_rev"]    = rev_votes
    return out


def build_dataset(d_1m, d_15m, d_1h, d_4h):
    idx        = d_1h.index
    indicators = extract_indicators(d_1m, d_15m, d_1h, d_4h)

    lr_1m      = np.log(d_1m["close"] / d_1m["close"].shift(1))
    vol_pm     = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    indicators["vol_pm"] = vol_pm

    close      = d_1h["close"]
    next_close = close.shift(-1)
    sigma_1h   = indicators["vol_pm"] * math.sqrt(60)

    rows = []
    for off in OFFSET_GRID:
        # TARGET: 1 = NO resolved (price did NOT exceed strike), 0 = YES resolved
        no_resolved = (next_close <= close * (1 + off)).astype(int)

        block             = indicators.copy()
        block["offset_pct"] = off
        log_off           = math.log(1.0 + off) if abs(off) < 0.99 else 0.0
        block["z_strike"] = log_off / sigma_1h.replace(0, float("nan"))
        block["target"]   = no_resolved
        block             = block.dropna()
        rows.append(block)

    return pd.concat(rows, axis=0).sort_index()


# ── training ──────────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*72}")
    print(f"  ETH NO MODEL — P(NO resolves) = P(close ≤ strike)")
    print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Val:   {VAL_START.date()} → {VAL_END.date()}")
    print(f"  Test:  {TEST_START.date()} → present")
    print(f"{'='*72}\n")

    print("Loading ETH data …", flush=True)
    d_1m, d_15m, d_1h, d_4h = load_data()

    print("Building dataset …", flush=True)
    t0 = time.time()
    ds = build_dataset(d_1m, d_15m, d_1h, d_4h)

    tr_mask = (ds.index >= TRAIN_START) & (ds.index < TRAIN_END)
    va_mask = (ds.index >= VAL_START)   & (ds.index < VAL_END)
    tr = ds[tr_mask]
    va = ds[va_mask]
    print(f"  Train: {len(tr):,} rows  |  Val: {len(va):,} rows  [{time.time()-t0:.1f}s]", flush=True)

    X_tr = tr[FEATURE_COLUMNS].values; y_tr = tr["target"].values
    X_va = va[FEATURE_COLUMNS].values; y_va = va["target"].values

    print(f"  Train NO rate: {y_tr.mean():.3f}  |  Val NO rate: {y_va.mean():.3f}", flush=True)

    # Exponential recency weighting
    t_vals  = np.array([t.timestamp() for t in tr.index])
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min
    w = np.exp(RECENCY_K * (t_vals - t_min) / t_range) if t_range > 0 else np.ones(len(tr))

    print(f"\nTraining HistGradientBoostingClassifier (RECENCY_K={RECENCY_K}) …", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=None, random_state=42,
    )
    clf.fit(X_tr, y_tr, sample_weight=w)

    p_tr     = clf.predict_proba(X_tr)[:, 1]
    p_va_raw = clf.predict_proba(X_va)[:, 1]
    auc_tr   = roc_auc_score(y_tr, p_tr)
    auc_va   = roc_auc_score(y_va, p_va_raw)
    ll_va    = log_loss(y_va, p_va_raw, labels=[0, 1])
    print(f"  Train AUC: {auc_tr:.4f}  |  Val AUC: {auc_va:.4f}  (log_loss: {ll_va:.4f})", flush=True)
    print(f"  Train→Val gap: {auc_tr - auc_va:+.4f}", flush=True)

    # ── isotonic calibration split by offset zone ─────────────────────────────
    # Positive offset: YES is OTM → NO is ITM (higher P(NO) baseline)
    # Negative offset: YES is ITM → NO is OTM (lower P(NO) baseline)
    # Calibrate each zone separately to avoid cross-zone averaging bias.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va_raw, y_va)
    p_va_cal = iso.predict(p_va_raw)
    print(f"  Unified iso AUC: {roc_auc_score(y_va, p_va_cal):.4f}", flush=True)

    va_pos_mask = va["offset_pct"] > 0   # YES OTM → NO ITM
    va_neg_mask = ~va_pos_mask            # YES ITM → NO OTM

    iso_pos, iso_neg = None, None
    if va_pos_mask.sum() >= 20:
        iso_pos = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_pos.fit(p_va_raw[va_pos_mask], y_va[va_pos_mask])
        p_pos_cal = iso_pos.predict(p_va_raw[va_pos_mask])
        ll_pos = log_loss(y_va[va_pos_mask], np.clip(p_pos_cal, 1e-6, 1 - 1e-6), labels=[0, 1])
        print(f"  iso_pos (YES-OTM / NO-ITM): n={va_pos_mask.sum()}  ll={ll_pos:.4f}", flush=True)

    if va_neg_mask.sum() >= 20:
        iso_neg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_neg.fit(p_va_raw[va_neg_mask], y_va[va_neg_mask])
        p_neg_cal = iso_neg.predict(p_va_raw[va_neg_mask])
        ll_neg = log_loss(y_va[va_neg_mask], np.clip(p_neg_cal, 1e-6, 1 - 1e-6), labels=[0, 1])
        print(f"  iso_neg (YES-ITM / NO-OTM): n={va_neg_mask.sum()}  ll={ll_neg:.4f}", flush=True)

    pipe = {
        "asset":   ASSET,
        "target":  "p_no",          # marks this as a NO model
        "clf":     clf,
        "iso":     iso,             # unified fallback
        "iso_pos": iso_pos,         # YES-OTM zone (NO ITM)
        "iso_neg": iso_neg,         # YES-ITM zone (NO OTM)
        "features": FEATURE_COLUMNS,
        "auc_tr":  auc_tr,
        "auc_va":  auc_va,
    }

    # ── live Platt calibration on NO-side paper trades ────────────────────────
    print(f"\nFitting live Platt calibration on NO-side paper trades …", flush=True)
    platt = fit_live_platt(pipe, d_1m, d_15m, d_1h, d_4h)
    if platt is not None:
        pipe["platt"] = platt

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n  Saved → {MODEL_PATH.name}", flush=True)

    # ── quick reliability check on test window paper trades ───────────────────
    reliability_check(pipe, d_1m, d_15m, d_1h, d_4h)

    return pipe


def _build_feature_cache(d_1m, d_15m, d_1h, d_4h):
    """Indicator values at each 1h timestamp from TEST_START onward."""
    ind = extract_indicators(d_1m, d_15m, d_1h, d_4h)
    lr_1m = np.log(d_1m["close"] / d_1m["close"].shift(1))
    vol_pm = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(d_1h.index, method="ffill")
    ind["vol_pm"] = vol_pm
    return ind[ind.index >= TEST_START]


def _predict_no(pipe, feature_cache, dt_floor, offset_pct):
    """Return P(NO) for a single (timestamp, offset) pair. None on failure."""
    if dt_floor not in feature_cache.index:
        return None
    vol_pm_bar = float(feature_cache.loc[dt_floor, "vol_pm"])
    sigma_1h   = vol_pm_bar * math.sqrt(60)
    log_off    = math.log(1.0 + offset_pct) if abs(offset_pct) < 0.99 else 0.0
    z_strike_v = log_off / sigma_1h if sigma_1h > 0 else 0.0
    try:
        vec = np.array([[
            feature_cache.loc[dt_floor, c] if c not in ("offset_pct", "z_strike")
            else (offset_pct if c == "offset_pct" else z_strike_v)
            for c in FEATURE_COLUMNS
        ]])
    except KeyError:
        return None
    if np.any(np.isnan(vec)):
        return None

    p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])

    # Split isotonic: positive offset = YES OTM / NO ITM; negative = YES ITM / NO OTM
    if offset_pct > 0 and pipe.get("iso_pos") is not None:
        p_cal = float(np.clip(pipe["iso_pos"].predict([p_raw])[0], 0.01, 0.99))
    elif offset_pct <= 0 and pipe.get("iso_neg") is not None:
        p_cal = float(np.clip(pipe["iso_neg"].predict([p_raw])[0], 0.01, 0.99))
    else:
        p_cal = float(np.clip(pipe["iso"].predict([p_raw])[0], 0.01, 0.99))

    if pipe.get("platt") is not None:
        try:
            p_c  = float(np.clip(p_cal, 1e-6, 1 - 1e-6))
            lo   = math.log(p_c / (1 - p_c))
            p_cal = float(pipe["platt"].predict_proba([[lo]])[0, 1])
        except Exception:
            pass

    return p_cal


def fit_live_platt(pipe, d_1m, d_15m, d_1h, d_4h):
    """
    Platt scaling on resolved NO-side paper trades only.

    Calibrates the model's P(NO) output against actual NO resolution outcomes
    from live/paper trading. Uses only NO-side trades so the calibration is
    specific to the NO use case.
    """
    patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs:
        print("  No paper trade files found — skipping Platt calibration", flush=True)
        return None

    raw = pd.concat(dfs, ignore_index=True)
    raw["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True, errors="coerce")

    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    raw["resolved_yes"] = raw["resolved_yes"].astype(str).map(bool_map)

    raw = raw.dropna(subset=["decision_time", "spot", "strike", "resolved_yes"])
    raw = raw[raw["decision_time"] >= TEST_START]
    raw = raw[raw.get("side", "no") == "no"]     # NO trades only
    raw = raw.drop_duplicates(subset=["decision_time", "contract_ticker"], keep="last")

    for c in ["spot", "strike"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["spot", "strike"])

    if len(raw) < 20:
        print(f"  Too few NO trades ({len(raw)}) for Platt calibration — skipping", flush=True)
        return None

    print(f"  Found {len(raw)} resolved NO paper trades for Platt calibration", flush=True)

    feature_cache = _build_feature_cache(d_1m, d_15m, d_1h, d_4h)
    preds, actuals = [], []

    for _, row in raw.iterrows():
        dt_floor   = row["decision_time"].floor("1h")
        spot       = float(row["spot"])
        strike     = float(row["strike"])
        if spot <= 0:
            continue
        offset_pct = (strike - spot) / spot
        p_no       = _predict_no(pipe, feature_cache, dt_floor, offset_pct)
        if p_no is None:
            continue
        # NO resolution: 1 if YES did NOT resolve (resolved_yes == 0)
        no_resolved = 1 - int(row["resolved_yes"])
        preds.append(p_no)
        actuals.append(no_resolved)

    if len(preds) < 20:
        print(f"  Too few matched predictions ({len(preds)}) — skipping Platt", flush=True)
        return None

    preds_arr   = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
    actuals_arr = np.array(actuals)
    log_odds    = np.log(preds_arr / (1 - preds_arr)).reshape(-1, 1)

    platt = LogisticRegression(C=1.0, max_iter=300)
    platt.fit(log_odds, actuals_arr)

    p_platt  = platt.predict_proba(log_odds)[:, 1]
    ll_before = log_loss(actuals_arr, preds_arr)
    ll_after  = log_loss(actuals_arr, np.clip(p_platt, 1e-6, 1 - 1e-6))
    no_rate   = actuals_arr.mean()
    print(f"  Platt coef={platt.coef_[0][0]:.3f}  intercept={platt.intercept_[0]:.3f}", flush=True)
    print(f"  Log-loss: {ll_before:.4f} → {ll_after:.4f}  NO rate: {no_rate:.3f}", flush=True)
    return platt


def reliability_check(pipe, d_1m, d_15m, d_1h, d_4h):
    """
    Reliability diagram: bucket model P(NO) vs realized NO rate on paper trades.
    Confirms calibration quality before deployment.
    """
    patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return

    raw = pd.concat(dfs, ignore_index=True)
    raw["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True, errors="coerce")
    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    raw["resolved_yes"] = raw["resolved_yes"].astype(str).map(bool_map)
    raw = raw.dropna(subset=["decision_time", "spot", "strike", "resolved_yes"])
    raw = raw[raw["decision_time"] >= TEST_START]
    raw = raw[raw.get("side", "no") == "no"]
    for c in ["spot", "strike"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["spot", "strike"])

    if len(raw) < 10:
        return

    feature_cache = _build_feature_cache(d_1m, d_15m, d_1h, d_4h)
    preds, actuals = [], []

    for _, row in raw.iterrows():
        dt_floor   = row["decision_time"].floor("1h")
        spot       = float(row["spot"])
        strike     = float(row["strike"])
        if spot <= 0:
            continue
        offset_pct = (strike - spot) / spot
        p_no       = _predict_no(pipe, feature_cache, dt_floor, offset_pct)
        if p_no is None:
            continue
        preds.append(p_no)
        actuals.append(1 - int(row["resolved_yes"]))

    if not preds:
        return

    preds_arr   = np.array(preds)
    actuals_arr = np.array(actuals)

    print(f"\n{'='*72}")
    print(f"  RELIABILITY DIAGRAM — P(NO) model vs realized NO rate")
    print(f"  n={len(preds)}  mean_p_no={preds_arr.mean():.3f}  realized_NO={actuals_arr.mean():.3f}")
    print(f"  {'bucket':>14}  {'n':>5}  {'mean_model':>10}  {'realized':>10}  {'delta':>8}")
    print(f"{'='*72}")

    bins = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = ["<.30", ".30-.40", ".40-.50", ".50-.60", ".60-.70", ".70-.80", ".80-.90", ">.90"]

    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        mask   = (preds_arr >= lo) & (preds_arr < hi)
        if mask.sum() == 0:
            continue
        mn = preds_arr[mask].mean()
        rl = actuals_arr[mask].mean()
        print(f"  {label:>14}  {mask.sum():>5d}  {mn:>10.3f}  {rl:>10.3f}  {rl-mn:>+8.3f}")

    print(f"\n  Val AUC (NO target): {roc_auc_score(actuals_arr, preds_arr):.4f}")


if __name__ == "__main__":
    train()
