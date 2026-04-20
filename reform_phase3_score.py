#!/usr/bin/env python3
"""
reform_phase3_score.py — Phase 3: score construction with 4 variants.

For each asset (BTC, ETH, SOL) trains four model variants on TRAIN set
(2025-01-01 → 2025-12-31), evaluates on VAL set (2026-01-01 → 2026-03-15),
reports val metrics, picks the winner per asset.

TEST set (2026-03-16 → present) is NOT touched here.

Variants:
  A. Individual features + L1 logistic regression
  B. Composite-of-divergences + other features
  C. Agreement-count + composite features (vote-based)
  D. Hybrid — all features together

Target: binary next_up (did next-hour close up?).
Metrics on val: log-loss, IC, AUC.
"""

import math, sys, glob, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_START   = pd.Timestamp("2026-01-01", tz="UTC")
VAL_END     = pd.Timestamp("2026-03-16", tz="UTC")


def load_asset(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum","open":"first"}).dropna(subset=["close"])
    d_1d  = d_1h.resample("1D", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum","open":"first"}).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h, d_1d


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100/(1 + g / l.replace(0, 1e-10))

def _stoch_k(h, l, c, k=14):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    return (c - ll)/(hh - ll).replace(0, float("nan")) * 100

def _atr(h, l, c, p=14):
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def _macd_hist(c, f=12, s=26, sig=9):
    ema_f = c.ewm(span=f, adjust=False).mean()
    ema_s = c.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal

def _bb_pct(c, n=20):
    mid = c.rolling(n).mean(); std = c.rolling(n).std()
    return (c - (mid - 2*std)) / ((mid + 2*std) - (mid - 2*std)).replace(0, float("nan"))

def _keltner_pct(h, l, c, span=20, mult=2):
    ema = c.ewm(span=span, adjust=False).mean()
    atr = _atr(h, l, c, span)
    return (c - (ema - mult*atr)) / ((ema + mult*atr) - (ema - mult*atr)).replace(0, float("nan"))

def _vwap_dev_1h(close_1m, volume_1m, idx_1h):
    date_1m = close_1m.index.normalize()
    tpv = close_1m * volume_1m
    vwap_1m = (tpv.groupby(date_1m).cumsum()) / (volume_1m.groupby(date_1m).cumsum().replace(0, float("nan")))
    vwap_1h = vwap_1m.resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")
    close_1h_from_1m = close_1m.resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")
    return (close_1h_from_1m - vwap_1h) / vwap_1h.replace(0, float("nan"))


# ── Feature extraction (combined, only keepers) ──────────────────────────────

def extract_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=None):
    idx = d_1h.index
    out = pd.DataFrame(index=idx)
    close = d_1h["close"]
    lr = np.log(close/close.shift(1))

    # ── Cross-timeframe divergences (Phase 1 + 2 winners) ──
    rsi_1h = _rsi(close, 14)
    rsi_4h = _rsi(d_4h["close"], 14).reindex(idx, method="ffill")
    rsi_1d = _rsi(d_1d["close"], 14).reindex(idx, method="ffill")
    stoch_1h = _stoch_k(d_1h["high"], d_1h["low"], close, 14)
    stoch_4h = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    bb_1h = _bb_pct(close, 20)
    bb_4h = _bb_pct(d_4h["close"], 20).reindex(idx, method="ffill")
    macd_1h_n = _macd_hist(close) / close
    macd_4h_n = _macd_hist(d_4h["close"]).reindex(idx, method="ffill") / d_4h["close"].reindex(idx, method="ffill")

    out["div_rsi"]   = rsi_1h - rsi_4h
    out["div_stoch"] = stoch_1h - stoch_4h
    out["div_bb"]    = bb_1h - bb_4h
    out["div_macd"]  = macd_1h_n - macd_4h_n
    out["div_rsi_1d"] = rsi_1h - rsi_1d
    out["align_rsi_4h_1d"] = rsi_4h - rsi_1d   # opposite sign

    # ── 4h trend (direct) ──
    out["trend_stoch_4h"] = stoch_4h
    out["trend_macd_4h"]  = macd_4h_n
    out["trend_bb_4h"]    = bb_4h

    # ── 15m reversion ──
    stoch_15m = _stoch_k(d_15m["high"], d_15m["low"], d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    out["rev_stoch_15m"] = stoch_15m

    # ── Move z-score (longer window) ──
    vol_72h = lr.rolling(72).std()
    out["move_z_72h"] = lr / vol_72h.replace(0, float("nan"))

    # ── Wicks ──
    br = (d_1h["high"] - d_1h["low"]).replace(0, float("nan"))
    upper_wick = (d_1h["high"] - d_1h[["close","open"]].max(axis=1)) / br
    lower_wick = (d_1h[["close","open"]].min(axis=1) - d_1h["low"]) / br
    out["wick_asymmetry_4h"] = lower_wick.rolling(4).mean() - upper_wick.rolling(4).mean()

    # ── Drawdown / stretch ──
    high_24 = d_1h["high"].rolling(24).max()
    out["drawdown_24h"] = close / high_24 - 1
    sma_50 = close.rolling(50).mean()
    out["stretch_sma50"] = close / sma_50 - 1

    # ── Recent momentum ──
    out["ret_1h"] = lr
    out["ret_2h"] = np.log(close/close.shift(2))

    # ── VWAP stretch ──
    out["vwap_dev_1h"] = _vwap_dev_1h(d_1m["close"], d_1m["volume"], idx)

    # ── Cross-asset (ETH/SOL only — BTC gets NaN) ──
    if btc_close_1h is not None:
        btc_lr_1h = np.log(btc_close_1h/btc_close_1h.shift(1)).reindex(idx, method="ffill")
        out["alt_minus_btc_1h"] = lr - btc_lr_1h
    else:
        out["alt_minus_btc_1h"] = 0.0   # BTC: fill with 0 (no effect)

    return out


def compute_targets(d_1h):
    close = d_1h["close"]
    return pd.DataFrame({
        "next_up": (close.shift(-1) > close).astype(int),
        "next_logret": np.log(close.shift(-1)/close),
    }, index=d_1h.index)


# ── Variant constructors ──────────────────────────────────────────────────────

def build_variant_B(X):
    """Composite-of-divergences + other features."""
    Xb = X.copy()
    div_cols = ["div_rsi", "div_stoch", "div_bb", "div_macd"]
    # Standardize each divergence, then average
    div_std = Xb[div_cols].apply(lambda s: (s - s.mean()) / s.std())
    Xb["div_composite"] = div_std.mean(axis=1)
    Xb = Xb.drop(columns=div_cols)
    return Xb

def build_variant_C(X):
    """Agreement count (signed) + composite features (current vote-based style)."""
    Xc = build_variant_B(X)
    # Agreement: count of divergences with same sign (net, in -4 to +4 range)
    div_cols = ["div_rsi", "div_stoch", "div_bb", "div_macd"]
    # Use original X since build_variant_B dropped these
    votes = X[div_cols].apply(np.sign)
    Xc["div_agreement"] = votes.sum(axis=1)  # -4 to +4
    return Xc

def build_variant_D(X):
    """Hybrid — all features, including all divergences + composite + agreement."""
    Xd = X.copy()
    div_cols = ["div_rsi", "div_stoch", "div_bb", "div_macd"]
    div_std = Xd[div_cols].apply(lambda s: (s - s.mean()) / s.std())
    Xd["div_composite"] = div_std.mean(axis=1)
    Xd["div_agreement"] = Xd[div_cols].apply(np.sign).sum(axis=1)
    return Xd


# ── Fit + evaluate ────────────────────────────────────────────────────────────

def fit_and_evaluate(X_train, y_train, X_val, y_val, logret_val, C_values=None):
    """Fit L1 logistic regression with candidate C values, return metrics + best model."""
    X_train = X_train.dropna()
    y_train_a = y_train.loc[X_train.index]
    X_val = X_val.dropna()
    y_val_a = y_val.loc[X_val.index]
    logret_val_a = logret_val.loc[X_val.index]
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    if C_values is None:
        C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best = None
    best_score = -1e9
    for C in C_values:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=1000)
        clf.fit(Xt, y_train_a)
        p_val = clf.predict_proba(Xv)[:,1]
        try:
            auc = roc_auc_score(y_val_a, p_val)
            ll  = log_loss(y_val_a, p_val, labels=[0,1])
            # IC of predicted prob with val log-return
            ic, _ = spearmanr(p_val, logret_val_a)
        except Exception:
            continue
        n_features_nonzero = (clf.coef_ != 0).sum()
        score = auc  # pick by AUC
        if score > best_score:
            best_score = score
            best = dict(C=C, clf=clf, scaler=scaler, auc=auc, log_loss=ll, ic=ic,
                        n_nonzero=int(n_features_nonzero),
                        feature_names=list(X_train.columns),
                        coefficients=dict(zip(X_train.columns, clf.coef_[0])))
    return best


def audit_variants(asset, sym, btc_close_1h):
    print(f"\n{'='*78}\n  [{asset}] PHASE 3 — score construction\n{'='*78}", flush=True)
    t0 = time.time()
    d_1m, d_15m, d_1h, d_4h, d_1d = load_asset(sym)
    btc = btc_close_1h if asset != "BTC" else None
    X = extract_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=btc)
    T = compute_targets(d_1h)

    tr = (X.index >= TRAIN_START) & (X.index < TRAIN_END)
    va = (X.index >= VAL_START) & (X.index < VAL_END)
    Xtr, Xva = X[tr], X[va]
    ytr, yva = T.loc[Xtr.index, "next_up"], T.loc[Xva.index, "next_up"]
    rtr, rva = T.loc[Xtr.index, "next_logret"], T.loc[Xva.index, "next_logret"]

    print(f"  train rows: {len(Xtr):,}  val rows: {len(Xva):,}", flush=True)

    # Prepare each variant
    variants = {}
    variants["A_individual"] = (Xtr, Xva)
    Xtr_B, Xva_B = build_variant_B(Xtr), build_variant_B(Xva)
    variants["B_composite"]  = (Xtr_B, Xva_B)
    Xtr_C, Xva_C = build_variant_C(Xtr), build_variant_C(Xva)
    variants["C_agreement"]  = (Xtr_C, Xva_C)
    Xtr_D, Xva_D = build_variant_D(Xtr), build_variant_D(Xva)
    variants["D_hybrid"]     = (Xtr_D, Xva_D)

    results = {}
    print(f"\n  {'variant':<16} {'C*':>6} {'n_nonzero':>10} {'AUC':>7} {'IC':>8} {'log_loss':>9}", flush=True)
    print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*7} {'-'*8} {'-'*9}", flush=True)
    for vname, (Xt, Xv) in variants.items():
        res = fit_and_evaluate(Xt, ytr, Xv, yva, rva)
        if res is None: continue
        results[vname] = res
        print(f"  {vname:<16} {res['C']:>6.2f} {res['n_nonzero']:>10d} {res['auc']:>7.4f} {res['ic']:>+8.4f} {res['log_loss']:>9.4f}", flush=True)

    # Baseline: null model predicting marginal rate
    baseline = ytr.mean()
    ll_null = log_loss(yva, np.full(len(yva), baseline), labels=[0,1])
    print(f"  {'baseline-null':<16} {'--':>6} {'--':>10} {'0.5000':>7} {'+0.0000':>8} {ll_null:>9.4f}", flush=True)

    best_name = max(results.keys(), key=lambda k: results[k]['auc'])
    print(f"\n  → Best variant: {best_name}  (val AUC = {results[best_name]['auc']:.4f})", flush=True)

    # Print coefficients of winner
    print(f"\n  Coefficients for {best_name} (top 20 by |weight|):", flush=True)
    coefs = results[best_name]["coefficients"]
    for feat, w in sorted(coefs.items(), key=lambda x: -abs(x[1]))[:20]:
        mark = "★" if w != 0 else " "
        print(f"    {mark} {feat:<30} {w:>+.4f}", flush=True)

    # Save
    pd.DataFrame([{"asset":asset, "variant":v, **{k:res[k] for k in ("C","n_nonzero","auc","ic","log_loss")}}
                  for v, res in results.items()]).to_csv(OUT_DIR / f"phase3_variants_{asset}.csv", index=False)

    print(f"\n  [{asset}] done in {time.time()-t0:.1f}s", flush=True)
    return results, best_name


def main():
    _, _, btc_1h, _, _ = load_asset("BTCUSDT")
    btc_close_1h = btc_1h["close"]

    all_winners = {}
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        results, best = audit_variants(asset, sym, btc_close_1h)
        all_winners[asset] = (best, results[best])

    print(f"\n{'='*78}\n  PHASE 3 WINNERS PER ASSET\n{'='*78}", flush=True)
    for asset, (best, res) in all_winners.items():
        print(f"  {asset}: {best}  val AUC={res['auc']:.4f}  IC={res['ic']:+.4f}  n_feat={res['n_nonzero']}", flush=True)


if __name__ == "__main__":
    main()
