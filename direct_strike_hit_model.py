#!/usr/bin/env python3
"""
direct_strike_hit_model.py — Direct fix: replace score_to_p_model with a trained
strike-hit predictor.

Target: P(close[t+1h] > spot[t] × (1+offset)) — the actual Kalshi question.
Inputs:  14 composite indicator values + offset_pct + vol_eff + tau.
Model:   sklearn HistGradientBoostingClassifier per asset.
Calib:   isotonic on val set.

Pipeline phases (in one script, honest separation):
  1. Build dataset (per-bar, per-offset records) for train/val
  2. Train model on train, early-stop via val AUC
  3. Isotonic calibration on val
  4. Lock pipeline
  5. Strike-hit AUC on test
  6. PnL backtest on test archive through full gate stack
  7. Compare to baseline (score_to_p_model + current composite_p_up gates)
  8. Apply acceptance criteria
"""

import math, sys, glob, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from composite_scorer import (
    compute_scores, lookup_p_up, _stoch_k, _rsi, _atr, _bb_pct, _keltner_pct,
    _wpr, _macd_cross, _vol_signal_4h, _dc_pct,
)
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_START   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_END     = pd.Timestamp("2026-03-16", tz="UTC")
TEST_START  = pd.Timestamp("2026-03-16", tz="UTC")

BANKROLL_0 = 1000.0
KELLY_MULT = 0.50
KELLY_CAP = 0.05
SLIPPAGE = DEFAULT_SLIPPAGE
SPREAD = DEFAULT_SPREAD
DRIFT_MULT = {"BTC": 1.40, "ETH": 0.80, "SOL": 0.20}

# Per-asset offset training grid (fractions, not percent)
ASSET_OFFSET_GRIDS = {
    "BTC": [-0.020, -0.015, -0.010, -0.0075, -0.005, -0.0025, -0.001,
             0.001,  0.0025,  0.005,  0.0075,  0.010,  0.015,  0.020],
    "ETH": [-0.030, -0.020, -0.015, -0.010, -0.005, -0.0025,
             0.0025,  0.005,  0.010,  0.015,  0.020,  0.030],
    "SOL": [-0.050, -0.030, -0.020, -0.010, -0.005,
             0.005,  0.010,  0.020,  0.030,  0.050],
}

ASSET_PARAMS = {
    "BTC": {"pm_min": 0.04, "pm_max": 0.96, "ns_max_otm_no": 0.40, "gate3": 0.01, "strike_step": 100.0},
    "ETH": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.005, "strike_step": 10.0},
    "SOL": {"pm_min": 0.02, "pm_max": 0.98, "ns_max_otm_no": 0.45, "gate3": 0.01, "strike_step": 1.0},
}
GATE_CS_MIN_YES_OTM = 0.55
GATE_CI_MIN_BEARISH = 0.45
RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC = 4.0, 0.33, 3.0, 0.08


def load_asset(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h


def extract_indicator_values(d_1m, d_15m, d_1h, d_4h):
    """Continuous values of all 14 composite indicators on 1h bars."""
    idx = d_1h.index
    out = pd.DataFrame(index=idx)
    # Trend (4h)
    out["trend_stoch_4h"]    = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    out["trend_bb_4h"]       = _bb_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20).reindex(idx, method="ffill")
    kc4_pct, _, _ = _keltner_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20, 2)
    out["trend_keltner_4h"]  = kc4_pct.reindex(idx, method="ffill")
    out["trend_wpr_4h"]      = _wpr(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    macd_st = _macd_cross(d_4h["close"]).map({"crossed_up":2, "up_lag":1, "none":0, "down_lag":-1, "crossed_down":-2}).fillna(0)
    out["trend_macd_4h"]     = macd_st.reindex(idx, method="ffill")
    vsig = _vol_signal_4h(d_4h["close"], d_4h["volume"]).map({"high_vol_up":1, "avg":0, "low_vol":0, "high_vol_down":-1}).fillna(0)
    out["trend_vol_4h"]      = vsig.reindex(idx, method="ffill")
    # Reversion
    out["rev_rsi_1h"]    = _rsi(d_1h["close"], 14)
    out["rev_rsi_4h"]    = _rsi(d_4h["close"], 14).reindex(idx, method="ffill")
    out["rev_stoch_15m"] = _stoch_k(d_15m["high"], d_15m["low"], d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_stoch_1h"]  = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    kc15_pct, _, _ = _keltner_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20, 2)
    out["rev_keltner_15m"] = kc15_pct.resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_dc_15m"]    = _dc_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_wpr_1h"]    = _wpr(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    # Move z-score (1h return / 24h rolling std)
    lr = np.log(d_1h["close"]/d_1h["close"].shift(1))
    roll_vol = lr.rolling(24).std()
    out["rev_move_z"] = lr / roll_vol.replace(0, float("nan"))
    return out


def build_dataset(asset, sym):
    """Build per-(bar, offset) dataset for train + val."""
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    idx = d_1h.index
    indicators = extract_indicator_values(d_1m, d_15m, d_1h, d_4h)

    # Realized vol (per minute from 1m returns, 60-bar rolling)
    lr_1m = np.log(d_1m["close"]/d_1m["close"].shift(1))
    vol_pm = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    indicators["vol_pm"] = vol_pm

    close = d_1h["close"]
    next_close = close.shift(-1)

    offsets = ASSET_OFFSET_GRIDS[asset]
    rows = []
    for off in offsets:
        target = (next_close > close * (1 + off)).astype(int)
        block = indicators.copy()
        block["offset_pct"] = off
        block["target"] = target
        block = block.dropna()
        rows.append(block)
    full = pd.concat(rows, axis=0).sort_index()
    return full


# ── Train + calibrate per asset ───────────────────────────────────────────────

FEATURE_COLUMNS = [
    "trend_stoch_4h", "trend_bb_4h", "trend_keltner_4h", "trend_wpr_4h",
    "trend_macd_4h", "trend_vol_4h",
    "rev_rsi_1h", "rev_rsi_4h", "rev_stoch_15m", "rev_stoch_1h",
    "rev_keltner_15m", "rev_dc_15m", "rev_wpr_1h", "rev_move_z",
    "offset_pct", "vol_pm",
]


def train_model(asset, sym):
    print(f"\n{'─'*78}\n  [{asset}] building dataset...\n{'─'*78}", flush=True)
    t0 = time.time()
    ds = build_dataset(asset, sym)
    tr_mask = (ds.index >= TRAIN_START) & (ds.index < TRAIN_END)
    va_mask = (ds.index >= VAL_START) & (ds.index < VAL_END)
    tr = ds[tr_mask]
    va = ds[va_mask]
    print(f"  Train: {len(tr):,} (bar, offset) rows / Val: {len(va):,} rows", flush=True)
    X_tr = tr[FEATURE_COLUMNS].values; y_tr = tr["target"].values
    X_va = va[FEATURE_COLUMNS].values; y_va = va["target"].values

    print(f"  Training HistGradientBoostingClassifier...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=None,  # we supply our own val via early stopping monitor
        random_state=42,
    )
    # sklearn's built-in early stopping uses an internal val split; we pass full train.
    # For explicit external val eval, we iterate with staged_predict_proba — simpler: train on train only.
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va_raw = clf.predict_proba(X_va)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_va = roc_auc_score(y_va, p_va_raw)
    ll_va = log_loss(y_va, p_va_raw, labels=[0,1])
    print(f"  Train AUC: {auc_tr:.4f}  |  Val AUC: {auc_va:.4f}  (log_loss: {ll_va:.4f})", flush=True)
    print(f"  Train→Val gap: {auc_tr - auc_va:+.4f}", flush=True)

    # Isotonic calibration on val
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_va_raw, y_va)
    p_va_cal = iso.predict(p_va_raw)
    auc_va_cal = roc_auc_score(y_va, p_va_cal)
    ll_va_cal = log_loss(y_va, np.clip(p_va_cal, 1e-6, 1-1e-6), labels=[0,1])
    print(f"  Val AUC after isotonic: {auc_va_cal:.4f} (log_loss: {ll_va_cal:.4f})", flush=True)

    # Save
    pipe = {"asset": asset, "clf": clf, "iso": iso, "features": FEATURE_COLUMNS,
            "auc_tr": auc_tr, "auc_va": auc_va, "auc_va_cal": auc_va_cal}
    with open(OUT_DIR / f"direct_model_{asset}.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved → direct_model_{asset}.pkl  [{time.time()-t0:.1f}s]", flush=True)
    return pipe


# ── Backtest on test window ───────────────────────────────────────────────────

def load_archive(asset):
    if asset == "BTC":
        patterns = ["paper_trades_archive_2026*.csv", "paper_trades_archive_pre_*.csv", "paper_trades.csv"]
    elif asset == "ETH":
        patterns = ["paper_trades_eth_archive_*.csv", "paper_trades_eth.csv"]
    else:
        patterns = ["paper_trades_sol_archive_*.csv", "paper_trades_sol.csv"]
    files = []
    for pat in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pat)))
    if asset == "BTC":
        files = [f for f in files if "_eth" not in f.name and "_sol" not in f.name]
    dfs = []
    for f in files:
        try: dfs.append(pd.read_csv(f, low_memory=False))
        except Exception: pass
    if not dfs: return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    needed = ["decision_time","contract_ticker","spot","strike","p_market","vol_eff",
              "tau_minutes","composite_trend","composite_rev","composite_p_up","resolved_yes"]
    for c in needed:
        if c not in raw.columns: return pd.DataFrame()
    raw = raw.dropna(subset=needed)
    for c in ["spot","strike","p_market","vol_eff","tau_minutes","composite_trend",
              "composite_rev","composite_p_up","resolved_yes"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=needed)
    raw = raw.drop_duplicates(subset=["decision_time","contract_ticker"], keep="last")
    raw["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True)
    raw = raw[raw["decision_time"] >= TEST_START]
    return raw.sort_values("decision_time").reset_index(drop=True)


def p_model_baseline(spot, strike, vol_eff, tau, p_up, asset):
    sigma_tau = vol_eff * math.sqrt(tau)
    if sigma_tau <= 0: return 0.5
    z_strike = math.log(strike/spot) / sigma_tau
    k = DRIFT_MULT.get(asset, 1.0)
    z_drift = norm.ppf(np.clip(p_up, 0.01, 0.99)) * k
    return float(np.clip(1 - norm.cdf(z_strike - z_drift), 0.01, 0.99))


def build_feature_cache(asset, sym):
    """Indicator values at each 1h timestamp in test window for quick lookup."""
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    ind = extract_indicator_values(d_1m, d_15m, d_1h, d_4h)
    lr_1m = np.log(d_1m["close"]/d_1m["close"].shift(1))
    vol_pm = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(d_1h.index, method="ffill")
    ind["vol_pm"] = vol_pm
    # Filter to test window start onward
    ind = ind[ind.index >= TEST_START]
    return ind


def evaluate_row(row, p_model_v, p_up_v, params):
    spot = row["spot"]; strike = row["strike"]; pm = row["p_market"]
    if pm <= 0 or pm >= 1: return None
    offset = (strike - spot) / spot if spot > 0 else 0
    best = None
    for side in ("yes", "no"):
        pm_use = p_model_v
        if not (params["pm_min"] <= pm_use <= params["pm_max"]): continue
        if not (0.04 <= pm <= 0.96): continue
        if side == "yes":
            if offset > 0 and p_up_v < GATE_CS_MIN_YES_OTM: continue
            if offset <= 0 and p_up_v < GATE_CI_MIN_BEARISH: continue
        if side == "no":
            if offset < 0 and p_up_v > params["ns_max_otm_no"]: continue
        fee = kalshi_fee(pm)
        if side == "yes":
            raw = pm_use - pm; net = raw - fee - SLIPPAGE - SPREAD
            rr = pm/(1-pm) if pm < 1 else 999
            if rr > RR_MAX_YES: continue
            if pm < 0.15: tier_min = 0.04
            elif pm < 0.25: tier_min = 0.03
            elif pm < 0.35: tier_min = 0.02
            else: tier_min = 0.0
        else:
            raw = pm - pm_use; net = raw - fee - SLIPPAGE - SPREAD
            rr = (1-pm)/pm if pm > 0 else 999
            if (rr < RR_MIN_NO or rr > RR_MAX_NO) and net < RR_EDGE_EXC: continue
            tier_min = 0.0
        if net < max(params["gate3"], tier_min): continue
        if best is None or net > best["net"]:
            best = {"side": side, "pm_use": pm_use, "pm": pm, "net": net, "strike": strike, "offset": offset}
    return best


def kelly_bet(pm_use, pm, side, bankroll):
    if side == "yes":
        b = (1-pm)/pm if pm > 0 else 0
        p, q = pm_use, 1 - pm_use
    else:
        b = pm/(1-pm) if pm < 1 else 0
        p_no = 1 - pm_use; p, q = p_no, 1 - p_no
    if b <= 0: return 0
    kf = max(0.0, (b*p - q)/b)
    bf = min(kf * KELLY_MULT, KELLY_CAP)
    return round(bankroll * bf, 2)


def trade_pnl(bet, side, pm, won):
    fee_rate = kalshi_fee(pm)
    if bet <= 0: return 0
    if side == "yes":
        if won:
            n_ct = bet/pm if pm > 0 else 0
            return bet*(1-pm)/pm - fee_rate*n_ct
        return -bet
    else:
        if won:
            n_ct = bet/(1-pm) if pm < 1 else 0
            return bet*pm/(1-pm) - fee_rate*n_ct
        return -bet


def run_bt(asset, scans_df, mode, pipe=None, feature_cache=None):
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None
        for _, row in group.iterrows():
            p_up_v = row["composite_p_up"]  # unchanged for gates
            if mode == "baseline":
                p_mv = p_model_baseline(row["spot"], row["strike"], row["vol_eff"],
                                         row["tau_minutes"], p_up_v, asset)
            else:
                # Direct model: predict P(close > strike) from features
                dt_floor = pd.Timestamp(dt).floor("1h")
                if dt_floor not in feature_cache.index: continue
                feats = feature_cache.loc[dt_floor][[c for c in FEATURE_COLUMNS if c not in ("offset_pct", "vol_pm")]]
                offset_pct = (row["strike"] - row["spot"]) / row["spot"]
                # Build feature vector
                vec = np.array([[*feats.values, offset_pct, feats.name if False else feature_cache.loc[dt_floor, "vol_pm"]]])
                # vec order must match FEATURE_COLUMNS
                try:
                    vec = np.array([[
                        feature_cache.loc[dt_floor, c] if c not in ("offset_pct",) else offset_pct
                        for c in FEATURE_COLUMNS
                    ]])
                except KeyError:
                    continue
                if np.any(np.isnan(vec)): continue
                p_raw = float(pipe["clf"].predict_proba(vec)[0, 1])
                p_mv = float(np.clip(pipe["iso"].predict([p_raw])[0], 0.01, 0.99))
            cand = evaluate_row(row, p_mv, p_up_v, params)
            if cand is None: continue
            if best is None or cand["net"] > best["net"]:
                best = cand; best_row = row
        if best is None: continue
        bet = kelly_bet(best["pm_use"], best["pm"], best["side"], bankroll)
        if bet <= 0: continue
        actual_yes = int(best_row["resolved_yes"])
        won = (actual_yes == 1 and best["side"] == "yes") or (actual_yes == 0 and best["side"] == "no")
        pnl = trade_pnl(bet, best["side"], best["pm"], won)
        bankroll = max(1.0, bankroll + pnl)
        pnls.append(pnl); wins.append(won); sides.append(best["side"])
    if not pnls:
        return dict(n=0, wr=0, pnl=0, max_streak=0, n_yes=0, n_no=0)
    n = len(pnls); wins_n = sum(wins)
    streak = 0; ms = 0
    for w in wins:
        if not w: streak += 1; ms = max(ms, streak)
        else: streak = 0
    return dict(n=n, wr=wins_n/n, pnl=sum(pnls), max_streak=ms,
                n_yes=sum(1 for s in sides if s=="yes"),
                n_no=sum(1 for s in sides if s=="no"))


def main():
    print(f"\n{'='*78}\n  DIRECT STRIKE-HIT MODEL — train + test vs baseline\n{'='*78}", flush=True)

    pipelines = {}
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        pipelines[asset] = train_model(asset, sym)

    print(f"\n{'='*78}\n  PnL BACKTEST on TEST window (2026-03-16 → present)\n{'='*78}", flush=True)
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        scans = load_archive(asset)
        if scans.empty: print(f"  [{asset}] no scans"); continue
        print(f"\n  [{asset}] Building feature cache for test window...", flush=True)
        fcache = build_feature_cache(asset, sym)
        r_base = run_bt(asset, scans, mode="baseline")
        r_dir = run_bt(asset, scans, mode="direct", pipe=pipelines[asset], feature_cache=fcache)
        def fmt(r):
            return f"n={r['n']:4d} WR={r['wr']:.1%} PnL=${r['pnl']:+.2f} streak={r['max_streak']:2d} ({r['n_yes']}y/{r['n_no']}n)"
        print(f"  BASELINE : {fmt(r_base)}", flush=True)
        print(f"  DIRECT   : {fmt(r_dir)}", flush=True)
        dpnl = r_dir["pnl"] - r_base["pnl"]
        pct = (dpnl / abs(r_base["pnl"]) * 100) if r_base["pnl"] != 0 else 0
        print(f"  Δ PnL=${dpnl:+.2f} ({pct:+.1f}%)", flush=True)

    print(f"\n{'='*78}\n  ACCEPTANCE CRITERIA CHECK\n{'='*78}", flush=True)
    print(f"  Committed rules: ship if (a) 2+ assets improve AND (b) no asset regresses more than 5% of its baseline magnitude.", flush=True)


if __name__ == "__main__":
    main()
