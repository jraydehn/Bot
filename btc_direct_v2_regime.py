#!/usr/bin/env python3
"""
btc_direct_v2_regime.py — BTC direct model with added regime features.

Adds to Phase 3 feature set:
  - trend_6h:   log(close[t] / close[t-6])   — recent trend direction
  - trend_24h:  log(close[t] / close[t-24])  — medium-horizon trend
  - trend_72h:  log(close[t] / close[t-72])  — longer trend (~3 days)
  - range_expansion: ATR_4h / ATR_96h         — trend-vs-range regime

Rationale: Phase 1 analysis showed BTC direct model's calibration holds on
test but PnL fails because the model has no explicit trend state input —
only oscillator-type features that react to swings but can't read the secular
direction. These 4 features directly address that.

Train/val/test splits identical to direct_strike_hit_model.py. Acceptance
criterion: BTC test PnL ≥ baseline ($56) and no > 5pp WR regression.
"""

import math, sys, glob, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from direct_strike_hit_model import (
    load_asset, extract_indicator_values, ASSET_OFFSET_GRIDS,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START,
    load_archive, p_model_baseline, evaluate_row, kelly_bet, trade_pnl,
    build_feature_cache, ASSET_PARAMS, DRIFT_MULT, BANKROLL_0,
    GATE_CS_MIN_YES_OTM, GATE_CI_MIN_BEARISH,
    RR_MAX_NO, RR_MIN_NO, RR_MAX_YES, RR_EDGE_EXC,
    SLIPPAGE, SPREAD, KELLY_MULT, KELLY_CAP,
)
from composite_scorer import _atr

OUT_DIR = Path(__file__).parent / "reform_results"

FEATURE_COLUMNS_V2 = [
    "trend_stoch_4h", "trend_bb_4h", "trend_keltner_4h", "trend_wpr_4h",
    "trend_macd_4h", "trend_vol_4h",
    "rev_rsi_1h", "rev_rsi_4h", "rev_stoch_15m", "rev_stoch_1h",
    "rev_keltner_15m", "rev_dc_15m", "rev_wpr_1h", "rev_move_z",
    # NEW regime features
    "trend_6h", "trend_24h", "trend_72h", "range_expansion",
    # Positional
    "offset_pct", "vol_pm",
]


def extract_features_v2(d_1m, d_15m, d_1h, d_4h):
    """Base indicators + new regime features."""
    out = extract_indicator_values(d_1m, d_15m, d_1h, d_4h)
    close = d_1h["close"]
    out["trend_6h"]  = np.log(close / close.shift(6))
    out["trend_24h"] = np.log(close / close.shift(24))
    out["trend_72h"] = np.log(close / close.shift(72))
    # Range expansion: short-horizon ATR / long-horizon ATR
    atr_4 = _atr(d_1h["high"], d_1h["low"], d_1h["close"], 4)
    atr_96 = _atr(d_1h["high"], d_1h["low"], d_1h["close"], 96)
    out["range_expansion"] = atr_4 / atr_96.replace(0, float("nan"))
    return out


def build_dataset_v2(asset, sym):
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    idx = d_1h.index
    indicators = extract_features_v2(d_1m, d_15m, d_1h, d_4h)
    lr_1m = np.log(d_1m["close"] / d_1m["close"].shift(1))
    vol_pm = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    indicators["vol_pm"] = vol_pm
    close = d_1h["close"]
    next_close = close.shift(-1)
    rows = []
    for off in ASSET_OFFSET_GRIDS[asset]:
        target = (next_close > close * (1 + off)).astype(int)
        block = indicators.copy()
        block["offset_pct"] = off
        block["target"] = target
        block = block.dropna()
        rows.append(block)
    return pd.concat(rows, axis=0).sort_index()


def run_bt(asset, scans_df, mode, pipe=None, feature_cache=None):
    params = ASSET_PARAMS[asset]
    bankroll = BANKROLL_0
    pnls = []; wins = []; sides = []; offsets_strike = []
    for dt, group in scans_df.groupby("decision_time", sort=True):
        best = None; best_row = None
        for _, row in group.iterrows():
            p_up_v = row["composite_p_up"]
            if mode == "baseline":
                p_mv = p_model_baseline(row["spot"], row["strike"], row["vol_eff"],
                                         row["tau_minutes"], p_up_v, asset)
            else:
                dt_floor = pd.Timestamp(dt).floor("1h")
                if dt_floor not in feature_cache.index: continue
                offset_pct = (row["strike"] - row["spot"]) / row["spot"]
                try:
                    vec = np.array([[
                        feature_cache.loc[dt_floor, c] if c not in ("offset_pct",) else offset_pct
                        for c in FEATURE_COLUMNS_V2
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
        step = params["strike_step"]
        offsets_strike.append(round(best["offset"] * best_row["spot"] / step))
    if not pnls:
        return dict(n=0, wr=0, pnl=0, max_streak=0, n_yes=0, n_no=0, sd={})
    n = len(pnls); wins_n = sum(wins)
    streak = 0; ms = 0
    for w in wins:
        if not w: streak += 1; ms = max(ms, streak)
        else: streak = 0
    sd = {}
    for s, p, w in zip(offsets_strike, pnls, wins):
        if s not in sd: sd[s] = {"n":0, "wins":0, "pnl":0.0}
        sd[s]["n"] += 1; sd[s]["wins"] += int(w); sd[s]["pnl"] += p
    return dict(n=n, wr=wins_n/n, pnl=sum(pnls), max_streak=ms,
                n_yes=sum(1 for s in sides if s=="yes"),
                n_no=sum(1 for s in sides if s=="no"), sd=sd)


def build_feature_cache_v2(asset, sym):
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    ind = extract_features_v2(d_1m, d_15m, d_1h, d_4h)
    lr_1m = np.log(d_1m["close"]/d_1m["close"].shift(1))
    vol_pm = lr_1m.rolling(60).std().resample("1h", origin="start_day").last().reindex(d_1h.index, method="ffill")
    ind["vol_pm"] = vol_pm
    ind = ind[ind.index >= TEST_START]
    return ind


def main():
    print(f"\n{'='*78}\n  BTC DIRECT v2 — regime features added\n{'='*78}", flush=True)

    # Build dataset
    print(f"\n  [BTC] building v2 dataset...", flush=True)
    ds = build_dataset_v2("BTC", "BTCUSDT")
    tr = ds[(ds.index >= TRAIN_START) & (ds.index < TRAIN_END)]
    va = ds[(ds.index >= VAL_START) & (ds.index < VAL_END)]
    print(f"    Train: {len(tr):,}  Val: {len(va):,}  features: {len(FEATURE_COLUMNS_V2)}", flush=True)

    X_tr = tr[FEATURE_COLUMNS_V2].values; y_tr = tr["target"].values
    X_va = va[FEATURE_COLUMNS_V2].values; y_va = va["target"].values

    print(f"  Training HistGradientBoostingClassifier v2...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, early_stopping=True,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va_raw = clf.predict_proba(X_va)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_va = roc_auc_score(y_va, p_va_raw)
    print(f"    Train AUC: {auc_tr:.4f}  |  Val AUC: {auc_va:.4f}  (gap {auc_tr-auc_va:+.4f})", flush=True)

    # Compare to v1
    with open(OUT_DIR / "direct_model_BTC.pkl", "rb") as f: v1_pipe = pickle.load(f)
    # v1 features don't include regime features — refit on equivalent val for fair comparison
    from direct_strike_hit_model import FEATURE_COLUMNS as V1_COLS, build_dataset as v1_build
    ds_v1 = v1_build("BTC", "BTCUSDT")
    tr_v1 = ds_v1[(ds_v1.index >= TRAIN_START) & (ds_v1.index < TRAIN_END)]
    va_v1 = ds_v1[(ds_v1.index >= VAL_START) & (ds_v1.index < VAL_END)]
    X_va_v1 = va_v1[V1_COLS].values; y_va_v1 = va_v1["target"].values
    p_v1_va = v1_pipe["clf"].predict_proba(X_va_v1)[:, 1]
    p_v1_cal = v1_pipe["iso"].predict(p_v1_va)
    auc_v1_va = roc_auc_score(y_va_v1, p_v1_cal)
    print(f"    (reference) v1 Val AUC: {auc_v1_va:.4f}  → v2 improvement: {auc_va - auc_v1_va:+.4f}", flush=True)

    # Isotonic on val
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_va_raw, y_va)
    p_va_cal = iso.predict(p_va_raw)
    auc_va_cal = roc_auc_score(y_va, p_va_cal)
    print(f"    Val AUC after isotonic: {auc_va_cal:.4f}", flush=True)

    pipe_v2 = {"asset":"BTC", "clf":clf, "iso":iso, "features":FEATURE_COLUMNS_V2,
               "auc_tr":auc_tr, "auc_va":auc_va, "auc_va_cal":auc_va_cal}
    with open(OUT_DIR / "direct_model_BTC_v2.pkl", "wb") as f: pickle.dump(pipe_v2, f)

    # ── PnL BACKTEST on TEST ──
    print(f"\n{'='*78}\n  BTC PnL BACKTEST on TEST window\n{'='*78}", flush=True)
    scans = load_archive("BTC")
    if scans.empty: print("  no scans"); return
    print(f"  Building v2 feature cache...", flush=True)
    fcache = build_feature_cache_v2("BTC", "BTCUSDT")

    r_base = run_bt("BTC", scans, mode="baseline")
    r_v1   = run_bt("BTC", scans, mode="direct", pipe=v1_pipe, feature_cache=build_feature_cache("BTC", "BTCUSDT"))
    r_v2   = run_bt("BTC", scans, mode="direct", pipe=pipe_v2, feature_cache=fcache)

    def fmt(r):
        return f"n={r['n']:4d} WR={r['wr']:.1%} PnL=${r['pnl']:+.2f} streak={r['max_streak']:2d} ({r['n_yes']}y/{r['n_no']}n)"
    print(f"\n  BASELINE  : {fmt(r_base)}", flush=True)
    print(f"  v1 (direct): {fmt(r_v1)}", flush=True)
    print(f"  v2 (regime): {fmt(r_v2)}", flush=True)
    print(f"\n  ΔPnL vs baseline: v1={r_v1['pnl']-r_base['pnl']:+.2f}  v2={r_v2['pnl']-r_base['pnl']:+.2f}", flush=True)

    # Acceptance check
    print(f"\n  Acceptance (BTC test PnL ≥ ${r_base['pnl']:.2f}  and WR not more than 5pp below baseline):", flush=True)
    pnl_ok = r_v2["pnl"] >= r_base["pnl"]
    wr_ok  = r_v2["wr"] >= r_base["wr"] - 0.05
    print(f"    v2 PnL ≥ baseline: {pnl_ok}  (v2=${r_v2['pnl']:.2f}, base=${r_base['pnl']:.2f})", flush=True)
    print(f"    v2 WR not regressed > 5pp: {wr_ok}  (v2={r_v2['wr']:.1%}, base={r_base['wr']:.1%})", flush=True)
    print(f"    SHIP: {pnl_ok and wr_ok}", flush=True)

    # Strike-distance breakdown
    if r_v2["n"] > 0:
        print(f"\n  v2 strike-distance buckets:", flush=True)
        step = ASSET_PARAMS["BTC"]["strike_step"]
        for sd in sorted(r_v2["sd"].keys()):
            b = r_v2["sd"][sd]
            print(f"    sd={sd:+3d} (${sd*step:+.0f})  n={b['n']:3d}  WR={b['wins']/b['n']:.1%}  PnL=${b['pnl']:+7.2f}", flush=True)


if __name__ == "__main__":
    main()
