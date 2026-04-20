#!/usr/bin/env python3
"""
reform_phase2_candidates.py — Phase 2: test additional new candidate indicators.

Builds on Phase 1 findings. Focus areas (based on what worked):
  - Cross-timeframe divergences (the Phase 1 winner was rsi_1h_minus_4h)
  - Volume-weighted momentum / OBV-like
  - Microstructure: wicks, range regime, autocorrelation
  - Cross-asset lead-lag (BTC → alts)
  - Stretch from multi-timeframe MAs

Tests on TRAIN set only (2025-01-01 → 2025-12-31). No val/test peeking.
Outputs reform_results/phase2_audit_{asset}.csv + cross-asset summary.
"""

import math, sys, glob, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "reform_results"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-01-01", tz="UTC")


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


# ── Phase 2 candidate features ────────────────────────────────────────────────

def extract_p2_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=None):
    """Candidate Phase 2 indicators."""
    idx = d_1h.index
    out = pd.DataFrame(index=idx)
    close = d_1h["close"]
    lr = np.log(close/close.shift(1))

    # ══ A. CROSS-TIMEFRAME DIVERGENCES ════════════════════════════════════════
    rsi_15m = _rsi(d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    rsi_1h  = _rsi(d_1h["close"], 14)
    rsi_4h  = _rsi(d_4h["close"], 14).reindex(idx, method="ffill")
    rsi_1d  = _rsi(d_1d["close"], 14).reindex(idx, method="ffill")

    # Already tested (Phase 1 winner): rsi_1h_minus_4h
    out["p2_rsi_15m_minus_1h"]   = rsi_15m - rsi_1h
    out["p2_rsi_1h_minus_1d"]    = rsi_1h - rsi_1d
    out["p2_rsi_4h_minus_1d"]    = rsi_4h - rsi_1d

    stoch_1h = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    stoch_4h = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx, method="ffill")
    out["p2_stoch_1h_minus_4h"]  = stoch_1h - stoch_4h

    macd_1h = _macd_hist(d_1h["close"])
    macd_4h = _macd_hist(d_4h["close"]).reindex(idx, method="ffill")
    # Normalize to compare across timeframes
    macd_1h_n = macd_1h / d_1h["close"]
    macd_4h_n = macd_4h / d_4h["close"].reindex(idx, method="ffill")
    out["p2_macd_1h_minus_4h"]   = macd_1h_n - macd_4h_n

    bb_1h  = _bb_pct(d_1h["close"], 20)
    bb_4h  = _bb_pct(d_4h["close"], 20).reindex(idx, method="ffill")
    out["p2_bb_1h_minus_4h"]     = bb_1h - bb_4h

    # ══ B. VOLUME-WEIGHTED / OBV-LIKE ═════════════════════════════════════════
    # On-Balance Volume slope (direction of cumulative signed volume)
    direction = np.sign(d_1h["close"].diff()).fillna(0)
    obv = (direction * d_1h["volume"]).cumsum()
    out["p2_obv_slope_24h"] = (obv - obv.shift(24)) / d_1h["volume"].rolling(24).sum().replace(0, float("nan"))

    # Chaikin Money Flow: volume-weighted close position within hi-lo range
    mfm = ((d_1h["close"] - d_1h["low"]) - (d_1h["high"] - d_1h["close"])) / (d_1h["high"] - d_1h["low"]).replace(0, float("nan"))
    mfv = mfm * d_1h["volume"]
    out["p2_cmf_20h"] = mfv.rolling(20).sum() / d_1h["volume"].rolling(20).sum().replace(0, float("nan"))

    # Volume-price correlation (positive = volume-confirmed moves)
    def _rolling_corr(a, b, n):
        return a.rolling(n).corr(b)
    out["p2_vol_price_corr_24h"] = _rolling_corr(d_1h["volume"], d_1h["close"], 24)

    # ══ C. MICROSTRUCTURE / WICK PATTERNS ═════════════════════════════════════
    # Upper and lower wick percentages (of bar range)
    br = (d_1h["high"] - d_1h["low"]).replace(0, float("nan"))
    upper_wick = (d_1h["high"] - d_1h[["close","open"]].max(axis=1)) / br
    lower_wick = (d_1h[["close","open"]].min(axis=1) - d_1h["low"]) / br
    # Recent rejection: upper wicks (bearish) vs lower wicks (bullish) over last 4h
    out["p2_upper_wick_avg_4h"] = upper_wick.rolling(4).mean()
    out["p2_lower_wick_avg_4h"] = lower_wick.rolling(4).mean()
    out["p2_wick_asymmetry_4h"] = lower_wick.rolling(4).mean() - upper_wick.rolling(4).mean()

    # Range regime: current ATR vs longer-window ATR
    atr_1h  = _atr(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    atr_24h = _atr(d_1h["high"], d_1h["low"], d_1h["close"], 96)  # ~4 days of 1h bars
    out["p2_range_regime"] = atr_1h / atr_24h.replace(0, float("nan")) - 1

    # Autocorrelation of returns (1-lag)
    out["p2_autocorr_24h"] = lr.rolling(24).apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan, raw=False)

    # Realized skew (past 24h log returns)
    def _skew_rolling(x, n):
        return x.rolling(n).skew()
    out["p2_skew_24h"] = _skew_rolling(lr, 24)

    # ══ D. STRETCH FROM MULTI-TIMEFRAME MEANS ═════════════════════════════════
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    out["p2_stretch_sma50"] = close / sma_50 - 1
    out["p2_stretch_sma200"] = close / sma_200 - 1

    # Session open stretch: distance from this calendar day's open
    day_open = close.groupby(close.index.normalize()).transform("first")
    out["p2_stretch_session"] = close / day_open - 1

    # Drawdown from recent high
    high_24 = d_1h["high"].rolling(24).max()
    low_24  = d_1h["low"].rolling(24).min()
    out["p2_drawdown_24h"] = close / high_24 - 1      # negative number
    out["p2_rally_from_low_24h"] = close / low_24 - 1 # positive number

    # ══ E. CROSS-ASSET (BTC leading ETH/SOL) ══════════════════════════════════
    if btc_close_1h is not None:
        btc_lr_1h  = np.log(btc_close_1h/btc_close_1h.shift(1)).reindex(idx, method="ffill")
        btc_lr_2h  = np.log(btc_close_1h/btc_close_1h.shift(2)).reindex(idx, method="ffill")
        btc_lr_4h  = np.log(btc_close_1h/btc_close_1h.shift(4)).reindex(idx, method="ffill")
        out["p2_btc_ret_1h"] = btc_lr_1h
        out["p2_btc_ret_4h"] = btc_lr_4h
        # Divergence: alt vs BTC momentum
        alt_lr_1h = lr
        alt_lr_4h = np.log(close/close.shift(4))
        out["p2_alt_minus_btc_1h"] = alt_lr_1h - btc_lr_1h
        out["p2_alt_minus_btc_4h"] = alt_lr_4h - btc_lr_4h

    # ══ F. VOLATILITY-ADJUSTED SIGNALS ════════════════════════════════════════
    # Normalized last 1h move vs rolling vol (different window than Phase 1 move_z)
    vol_72h = lr.rolling(72).std()
    out["p2_move_z_72h"] = lr / vol_72h.replace(0, float("nan"))

    # Cumulative vol over last 6h (cumulative absolute move)
    out["p2_cum_abs_6h"] = lr.abs().rolling(6).sum()
    out["p2_cum_abs_24h"] = lr.abs().rolling(24).sum()

    # Trend efficiency: net move / sum of absolute moves (signed trend strength)
    net_6h = (np.log(close/close.shift(6)))
    cum_6h = lr.abs().rolling(6).sum()
    out["p2_trend_efficiency_6h"] = net_6h / cum_6h.replace(0, float("nan"))

    return out


def compute_targets(d_1h):
    close = d_1h["close"]
    return pd.DataFrame({
        "next_up": (close.shift(-1) > close).astype(int),
        "next_logret": np.log(close.shift(-1)/close),
    }, index=d_1h.index)


def compute_ic(feature, target):
    m = feature.notna() & target.notna()
    if m.sum() < 100: return float("nan"), 0
    rho, _ = spearmanr(feature[m], target[m])
    return rho, int(m.sum())

def auc_binary(feature, target_up):
    m = feature.notna() & target_up.notna()
    if m.sum() < 100: return float("nan")
    f = feature[m].values; t = target_up[m].values
    n_pos = int(t.sum()); n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    ranks = rankdata(f)
    return (ranks[t == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def monthly_ic_std(feature, target_logret, idx):
    df = pd.DataFrame({"f": feature, "t": target_logret}, index=idx).dropna()
    if len(df) < 60: return float("nan")
    df["month"] = df.index.to_period("M")
    ics = []
    for _, g in df.groupby("month"):
        if len(g) < 30: continue
        rho, _ = spearmanr(g["f"], g["t"])
        if not np.isnan(rho): ics.append(rho)
    return float(np.std(ics)) if len(ics) >= 3 else float("nan")


def audit_asset(asset, sym, btc_close_1h):
    print(f"\n{'='*78}\n  [{asset}] PHASE 2 CANDIDATE AUDIT — train 2025-01-01 → 2025-12-31\n{'='*78}", flush=True)
    t0 = time.time()
    d_1m, d_15m, d_1h, d_4h, d_1d = load_asset(sym)
    btc = btc_close_1h if asset != "BTC" else None
    features = extract_p2_features(d_1m, d_15m, d_1h, d_4h, d_1d, btc_close_1h=btc)
    targets = compute_targets(d_1h)

    mask = (features.index >= TRAIN_START) & (features.index < TRAIN_END)
    feats_train = features[mask]
    targs_train = targets[mask]
    print(f"  Train rows: {len(feats_train):,}  features: {len(feats_train.columns)}", flush=True)

    rows = []
    for col in feats_train.columns:
        f = feats_train[col]
        ic, n = compute_ic(f, targs_train["next_logret"])
        au = auc_binary(f, targs_train["next_up"])
        mstd = monthly_ic_std(f, targs_train["next_logret"], feats_train.index)
        rows.append({"feature": col, "ic": ic, "auc": au, "monthly_ic_std": mstd, "n": n})
    audit = pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
    audit.to_csv(OUT_DIR / f"phase2_audit_{asset}.csv", index=False)

    print(f"\n  {'feature':<32} {'IC':>8} {'AUC':>7} {'IC_std':>8} {'n':>6}", flush=True)
    print(f"  {'-'*32} {'-'*8} {'-'*7} {'-'*8} {'-'*6}", flush=True)
    for _, r in audit.iterrows():
        sig = "★★" if abs(r["ic"]) > 0.03 and not math.isnan(r["ic"]) else ("★" if abs(r["ic"]) > 0.01 else "")
        print(f"  {r['feature']:<32} {r['ic']:>+8.4f} {r['auc']:>7.4f} {r['monthly_ic_std']:>8.4f} {int(r['n']):>6}  {sig}", flush=True)

    print(f"\n  [{asset}] done in {time.time()-t0:.1f}s", flush=True)
    return audit


def main():
    # Pre-load BTC close for cross-asset features on ETH/SOL
    _, _, btc_1h, _, _ = load_asset("BTCUSDT")
    btc_close_1h = btc_1h["close"]

    all_audits = {}
    for asset, sym in [("BTC","BTCUSDT"), ("ETH","ETHUSDT"), ("SOL","SOLUSDT")]:
        all_audits[asset] = audit_asset(asset, sym, btc_close_1h)

    print(f"\n{'='*78}\n  CROSS-ASSET SUMMARY — Phase 2 candidates\n{'='*78}", flush=True)
    feats = sorted(set().union(*[a['feature'].tolist() for a in all_audits.values()]))
    rows = []
    for f in feats:
        ics = []
        for asset in ["BTC","ETH","SOL"]:
            df = all_audits[asset]; r = df[df['feature']==f]
            if not r.empty: ics.append(r.iloc[0]['ic'])
        if ics:
            nics = [x for x in ics if not np.isnan(x)]
            if len(nics) == 0: continue
            mean_abs = np.mean(np.abs(nics))
            same_sign = all(x > 0 for x in nics) or all(x < 0 for x in nics)
            rows.append({"feature": f, "mean_abs_ic": mean_abs, "same_sign": same_sign,
                         "btc_ic": ics[0] if len(ics)>0 else float("nan"),
                         "eth_ic": ics[1] if len(ics)>1 else float("nan"),
                         "sol_ic": ics[2] if len(ics)>2 else float("nan")})
    summary = pd.DataFrame(rows).sort_values("mean_abs_ic", ascending=False)
    summary.to_csv(OUT_DIR / "phase2_cross_asset_summary.csv", index=False)
    print(f"\n  {'feature':<32} {'mean|IC|':>9} {'same':>5} {'btc':>8} {'eth':>8} {'sol':>8}", flush=True)
    print(f"  {'-'*32} {'-'*9} {'-'*5} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    for _, r in summary.iterrows():
        m = "✓" if r["same_sign"] else " "
        btc_s = f"{r['btc_ic']:+8.4f}" if not np.isnan(r['btc_ic']) else f"{'---':>8}"
        eth_s = f"{r['eth_ic']:+8.4f}" if not np.isnan(r['eth_ic']) else f"{'---':>8}"
        sol_s = f"{r['sol_ic']:+8.4f}" if not np.isnan(r['sol_ic']) else f"{'---':>8}"
        print(f"  {r['feature']:<32} {r['mean_abs_ic']:>9.4f} {m:>5} {btc_s} {eth_s} {sol_s}", flush=True)


if __name__ == "__main__":
    main()
