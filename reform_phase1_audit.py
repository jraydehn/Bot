#!/usr/bin/env python3
"""
reform_phase1_audit.py — Phase 1 of composite reform: audit the current 14 indicators.

For each asset (BTC, ETH, SOL), measure each indicator's predictive power on the
TRAIN set only (2025-01-01 → 2025-12-31). Never looks at validation or test.

Metrics per indicator:
  - IC (Spearman correlation with next-hour log return)
  - AUC (area under ROC for predicting up-hour)
  - Monthly stability (std deviation of IC across months)
  - Pairwise correlation with other indicators

Outputs CSV: reform_results/phase1_audit_{asset}.csv
"""

import math, sys, glob, warnings, json, time
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
# VAL/TEST intentionally unused in this script


def load_asset(sym):
    f_1m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]
    f_1h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    d_1m = pd.read_parquet(f_1m); d_1m.index = pd.to_datetime(d_1m.index, utc=True); d_1m.sort_index(inplace=True)
    d_1h = pd.read_parquet(f_1h); d_1h.index = pd.to_datetime(d_1h.index, utc=True); d_1h.sort_index(inplace=True)
    d_4h = pd.read_parquet(f_4h); d_4h.index = pd.to_datetime(d_4h.index, utc=True); d_4h.sort_index(inplace=True)
    d_15m = d_1m.resample("15min", origin="start_day").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    return d_1m, d_15m, d_1h, d_4h


# ── Indicator functions (continuous values, not votes) ────────────────────────

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

def _bb_pct(c, n=20):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    up = mid + 2*std; dn = mid - 2*std
    return (c - dn)/(up - dn).replace(0, float("nan"))

def _keltner_pct(h, l, c, span=20, mult=2):
    ema = c.ewm(span=span, adjust=False).mean()
    atr = _atr(h, l, c, span)
    up = ema + mult*atr; dn = ema - mult*atr
    return (c - dn)/(up - dn).replace(0, float("nan"))

def _wpr(h, l, c, p=14):
    hh = h.rolling(p).max(); ll = l.rolling(p).min()
    return -100 * (hh - c)/(hh - ll).replace(0, float("nan"))

def _macd_hist(c, f=12, s=26, sig=9):
    ema_f = c.ewm(span=f, adjust=False).mean()
    ema_s = c.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal

def _dc_pct(h, l, c, n=20):
    dh = h.rolling(n).max(); dl = l.rolling(n).min()
    return (c - dl)/(dh - dl).replace(0, float("nan"))

def _vol_ratio(volume, n=20):
    return volume / volume.rolling(n).mean().replace(0, float("nan"))

def _vwap_dev(close_1m, volume_1m, idx):
    """Daily-reset VWAP resampled to 1h; return (close - vwap)/vwap."""
    date_1m = close_1m.index.normalize()
    tpv = close_1m * volume_1m
    cum_tpv = tpv.groupby(date_1m).cumsum()
    cum_vol = volume_1m.groupby(date_1m).cumsum()
    vwap_1m = cum_tpv / cum_vol.replace(0, float("nan"))
    vwap_1h = vwap_1m.resample("1h", origin="start_day").last()
    vwap_h = vwap_1h.reindex(idx, method="ffill")
    close_h_from_1m = close_1m.resample("1h", origin="start_day").last().reindex(idx, method="ffill")
    return (close_h_from_1m - vwap_h)/vwap_h.replace(0, float("nan"))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(d_1m, d_15m, d_1h, d_4h):
    """Return DataFrame indexed to 1h bars with all current + candidate features."""
    idx_1h = d_1h.index
    out = pd.DataFrame(index=idx_1h)

    # ── CURRENT COMPOSITE INDICATORS (14) ──
    # Trend (4h indicators)
    stk4 = _stoch_k(d_4h["high"], d_4h["low"], d_4h["close"], 14)
    out["cur_trend_stoch_4h"]   = stk4.reindex(idx_1h, method="ffill")
    out["cur_trend_vol_4h"]     = _vol_ratio(d_4h["volume"], 20).reindex(idx_1h, method="ffill")
    out["cur_trend_macd_4h"]    = _macd_hist(d_4h["close"]).reindex(idx_1h, method="ffill")
    out["cur_trend_bb_4h"]      = _bb_pct(d_4h["close"], 20).reindex(idx_1h, method="ffill")
    out["cur_trend_keltner_4h"] = _keltner_pct(d_4h["high"], d_4h["low"], d_4h["close"], 20, 2).reindex(idx_1h, method="ffill")
    out["cur_trend_wpr_4h"]     = _wpr(d_4h["high"], d_4h["low"], d_4h["close"], 14).reindex(idx_1h, method="ffill")

    # Reversion (1h/15m)
    out["cur_rev_rsi_1h"]    = _rsi(d_1h["close"], 14)
    out["cur_rev_rsi_4h"]    = _rsi(d_4h["close"], 14).reindex(idx_1h, method="ffill")
    out["cur_rev_stoch_15m"] = _stoch_k(d_15m["high"], d_15m["low"], d_15m["close"], 14).resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")
    out["cur_rev_stoch_1h"]  = _stoch_k(d_1h["high"], d_1h["low"], d_1h["close"], 14)
    out["cur_rev_vwap"]      = _vwap_dev(d_1m["close"], d_1m["volume"], idx_1h)
    out["cur_rev_dc_15m"]    = _dc_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20).resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")
    out["cur_rev_keltner_15m"] = _keltner_pct(d_15m["high"], d_15m["low"], d_15m["close"], 20, 2).resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")
    out["cur_rev_wpr_1h"]    = _wpr(d_1h["high"], d_1h["low"], d_1h["close"], 14)

    # Move z-score (pseudo-indicator but in composite as rev vote)
    lr = np.log(d_1h["close"]/d_1h["close"].shift(1))
    roll_vol = lr.rolling(24).std()
    out["cur_rev_move_z"] = (lr/roll_vol.replace(0, float("nan")))

    # ── CANDIDATE NEW INDICATORS ──
    # Trend strength
    out["cand_adx_1h"]   = _atr(d_1h["high"], d_1h["low"], d_1h["close"], 14) / d_1h["close"]  # proxy using ATR ratio
    out["cand_adx_4h"]   = (_atr(d_4h["high"], d_4h["low"], d_4h["close"], 14) / d_4h["close"]).reindex(idx_1h, method="ffill")

    # EMA stack slopes (momentum of trend)
    ema_20 = d_1h["close"].ewm(span=20, adjust=False).mean()
    ema_50 = d_1h["close"].ewm(span=50, adjust=False).mean()
    out["cand_ema_slope_1h"] = (ema_20 / ema_20.shift(4) - 1)  # slope of 20-EMA over 4h
    out["cand_ema_20_vs_50"] = (ema_20 / ema_50 - 1)            # fast vs slow EMA ratio

    # Multi-timeframe RSI divergence
    out["cand_rsi_1h_minus_4h"] = out["cur_rev_rsi_1h"] - out["cur_rev_rsi_4h"]

    # Breadth (% positive 15m candles over last 2h = 8 bars)
    pos_15m = (d_15m["close"] > d_15m["close"].shift(1)).astype(float)
    breadth = pos_15m.rolling(8).mean()
    out["cand_breadth_15m_2h"] = breadth.resample("1h", origin="start_day").last().reindex(idx_1h, method="ffill")

    # Vol of vol (rolling std of log-return rolling std)
    vol_roll = lr.rolling(24).std()
    out["cand_vol_of_vol"] = vol_roll.rolling(24).std()

    # Volume acceleration
    vol_ma = d_1h["volume"].rolling(20).mean()
    out["cand_vol_accel"] = (d_1h["volume"]/vol_ma - 1)

    # Session flags
    hour = idx_1h.hour
    out["cand_session_us"]   = ((hour >= 13) & (hour < 21)).astype(float)   # NY hours
    out["cand_session_asia"] = ((hour >= 0)  & (hour < 7)).astype(float)    # Asia hours
    out["cand_session_eu"]   = ((hour >= 7)  & (hour < 13)).astype(float)   # EU hours
    out["cand_hour_sin"] = np.sin(2*np.pi*hour/24)
    out["cand_hour_cos"] = np.cos(2*np.pi*hour/24)

    # Day of week
    dow = idx_1h.dayofweek
    out["cand_dow_mon"] = (dow == 0).astype(float)
    out["cand_dow_fri"] = (dow == 4).astype(float)
    out["cand_dow_weekend"] = ((dow == 5) | (dow == 6)).astype(float)

    # MFI (money flow index, volume-weighted RSI)
    tp = (d_1h["high"] + d_1h["low"] + d_1h["close"]) / 3
    mf = tp * d_1h["volume"]
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0.0), index=idx_1h)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0.0), index=idx_1h)
    pos_mf_sum = pos_mf.rolling(14).sum()
    neg_mf_sum = neg_mf.rolling(14).sum()
    out["cand_mfi_1h"] = 100 - 100/(1 + pos_mf_sum/neg_mf_sum.replace(0, 1e-10))

    # Recent momentum lags
    out["cand_ret_1h"]  = np.log(d_1h["close"]/d_1h["close"].shift(1))
    out["cand_ret_2h"]  = np.log(d_1h["close"]/d_1h["close"].shift(2))
    out["cand_ret_4h"]  = np.log(d_1h["close"]/d_1h["close"].shift(4))
    out["cand_ret_12h"] = np.log(d_1h["close"]/d_1h["close"].shift(12))
    out["cand_ret_24h"] = np.log(d_1h["close"]/d_1h["close"].shift(24))

    # High-vol / low-vol regime
    vol_median_168h = lr.rolling(168).std().median()  # will be a constant
    out["cand_vol_regime"] = (lr.rolling(24).std() / vol_median_168h - 1).fillna(0)

    # BB squeeze (narrow BBs → breakout candidate)
    bb_width = d_1h["close"].rolling(20).std() / d_1h["close"].rolling(20).mean()
    out["cand_bb_squeeze"] = bb_width / bb_width.rolling(168).mean() - 1

    # High-low range normalized
    out["cand_range_pct"] = (d_1h["high"] - d_1h["low"]) / d_1h["close"]

    return out


# ── Target ────────────────────────────────────────────────────────────────────

def compute_targets(d_1h):
    """Multiple target definitions."""
    idx = d_1h.index
    close = d_1h["close"]
    lr = np.log(close/close.shift(1))
    return pd.DataFrame({
        "next_up": (close.shift(-1) > close).astype(int),  # binary: did next hour close up?
        "next_logret": np.log(close.shift(-1)/close),      # continuous return
    }, index=idx)


# ── Audit metrics ─────────────────────────────────────────────────────────────

def compute_ic(feature, target):
    """Spearman IC ignoring NaNs."""
    m = feature.notna() & target.notna()
    if m.sum() < 100: return float("nan"), 0
    rho, _ = spearmanr(feature[m], target[m])
    return rho, int(m.sum())

def auc_binary(feature, target_up):
    """AUC for predicting target_up from feature. Expects binary target in {0,1}."""
    m = feature.notna() & target_up.notna()
    if m.sum() < 100: return float("nan")
    f = feature[m].values; t = target_up[m].values
    # Rank-based AUC
    n_pos = int(t.sum()); n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    ranks = rankdata(f)
    return (ranks[t == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def monthly_ic_std(feature, target_logret, idx):
    """Std dev of IC across months. Lower = more stable."""
    df = pd.DataFrame({"f": feature, "t": target_logret}, index=idx).dropna()
    if len(df) < 60: return float("nan")
    df["month"] = df.index.to_period("M")
    ics = []
    for m, g in df.groupby("month"):
        if len(g) < 30: continue
        rho, _ = spearmanr(g["f"], g["t"])
        if not np.isnan(rho): ics.append(rho)
    return float(np.std(ics)) if len(ics) >= 3 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def audit_asset(asset, sym):
    print(f"\n{'='*78}\n  [{asset}] PHASE 1 INDICATOR AUDIT — train 2025-01-01 → 2025-12-31\n{'='*78}", flush=True)
    t0 = time.time()
    d_1m, d_15m, d_1h, d_4h = load_asset(sym)
    features = extract_features(d_1m, d_15m, d_1h, d_4h)
    targets = compute_targets(d_1h)

    # Filter to train window
    mask = (features.index >= TRAIN_START) & (features.index < TRAIN_END)
    feats_train = features[mask]
    targs_train = targets[mask]
    print(f"  Train rows: {len(feats_train):,}  ({feats_train.index.min()} → {feats_train.index.max()})", flush=True)

    # Audit every feature
    rows = []
    for col in feats_train.columns:
        f = feats_train[col]
        ic, n = compute_ic(f, targs_train["next_logret"])
        au = auc_binary(f, targs_train["next_up"])
        mstd = monthly_ic_std(f, targs_train["next_logret"], feats_train.index)
        group = "current" if col.startswith("cur_") else "candidate"
        rows.append({"feature": col, "group": group, "ic": ic, "auc": au,
                     "monthly_ic_std": mstd, "n": n})
    audit = pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)

    # Save
    out_file = OUT_DIR / f"phase1_audit_{asset}.csv"
    audit.to_csv(out_file, index=False)
    print(f"\n  Saved → {out_file}", flush=True)

    # Print report
    print(f"\n  {'feature':<30} {'group':<10} {'IC':>8} {'AUC':>7} {'IC_std':>8} {'n':>6}", flush=True)
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*6}", flush=True)
    for _, r in audit.iterrows():
        sig = "★★" if abs(r["ic"]) > 0.03 and not math.isnan(r["ic"]) else ("★" if abs(r["ic"]) > 0.01 else "")
        print(f"  {r['feature']:<30} {r['group']:<10} {r['ic']:>+8.4f} {r['auc']:>7.4f} {r['monthly_ic_std']:>8.4f} {int(r['n']):>6}  {sig}", flush=True)

    # Correlation matrix among features with |IC| > 0.01
    strong = audit[audit['ic'].abs() > 0.01]['feature'].tolist()
    if strong:
        corr = feats_train[strong].corr().abs()
        high_corr = []
        for i, a in enumerate(strong):
            for b in strong[i+1:]:
                if corr.loc[a, b] > 0.80:
                    high_corr.append((a, b, corr.loc[a, b]))
        if high_corr:
            print(f"\n  HIGH-CORRELATION PAIRS (|r| > 0.80) among |IC|>0.01 features:", flush=True)
            for a, b, c in sorted(high_corr, key=lambda x: -x[2]):
                print(f"    {a} ↔ {b}  r={c:.3f}", flush=True)

    print(f"\n  [{asset}] done in {time.time()-t0:.1f}s", flush=True)
    return audit


def main():
    all_audits = {}
    for asset, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"), ("SOL", "SOLUSDT")]:
        all_audits[asset] = audit_asset(asset, sym)

    # Cross-asset summary: which features are good across all three?
    print(f"\n{'='*78}\n  CROSS-ASSET SUMMARY: mean |IC| across BTC/ETH/SOL\n{'='*78}", flush=True)
    all_features = sorted(set().union(*[a['feature'].tolist() for a in all_audits.values()]))
    summary_rows = []
    for feat in all_features:
        ics = []
        for asset in ["BTC", "ETH", "SOL"]:
            df = all_audits[asset]
            row = df[df['feature'] == feat]
            if not row.empty:
                ics.append(row.iloc[0]['ic'])
        if len(ics) == 3:
            mean_abs = np.mean(np.abs(ics))
            same_sign = all(x > 0 for x in ics) or all(x < 0 for x in ics)
            summary_rows.append({"feature": feat, "mean_abs_ic": mean_abs, "same_sign": same_sign,
                                 "btc_ic": ics[0], "eth_ic": ics[1], "sol_ic": ics[2]})
    summary = pd.DataFrame(summary_rows).sort_values("mean_abs_ic", ascending=False)
    summary.to_csv(OUT_DIR / "phase1_cross_asset_summary.csv", index=False)
    print(f"\n  {'feature':<30} {'mean|IC|':>9} {'same':>5} {'btc':>8} {'eth':>8} {'sol':>8}", flush=True)
    print(f"  {'-'*30} {'-'*9} {'-'*5} {'-'*8} {'-'*8} {'-'*8}", flush=True)
    for _, r in summary.head(30).iterrows():
        marker = "✓" if r["same_sign"] else " "
        print(f"  {r['feature']:<30} {r['mean_abs_ic']:>9.4f} {marker:>5} {r['btc_ic']:>+8.4f} {r['eth_ic']:>+8.4f} {r['sol_ic']:>+8.4f}", flush=True)


if __name__ == "__main__":
    main()
