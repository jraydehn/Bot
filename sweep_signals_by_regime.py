"""
sweep_signals_by_regime.py — IC breakdown of each trend + rev signal by HMM macro regime.

Measures Pearson IC vs next_up for every raw signal (continuous, pre-vote-encoding)
segmented by Bull / Sideways / Bear regime. Goal: identify which signals carry
predictive power in each regime, especially Sideways where composite trend degraded.
"""

import sys
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
DATA_DIR    = BASE / "data"
LABELS_PATH = BASE / "reform_results" / "hmm_macro_labels_btc.parquet"
TEST_START  = pd.Timestamp("2025-01-01", tz="UTC")

sys.path.insert(0, str(BASE))
from composite_scorer import (
    _rsi, _stoch_k, _atr, _keltner_pct, _wpr, _macd_cross,
    _vol_signal_4h, _bb_pct, _dc_pct, _vwap_1h,
    compute_scores,
)

REGIMES = ["Bull", "Sideways", "Bear"]


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    sym = "BTCUSDT"
    f_1h  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1h_2024-01-01_*.parquet")))[-1]
    f_4h  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_4h_2024-01-01_*.parquet")))[-1]
    f_15m = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_15m_2024-01-01_*.parquet")))[-1]
    f_1m  = sorted(glob.glob(str(DATA_DIR / f"binanceus_{sym}_1m_2024-01-01_*.parquet")))[-1]

    def load(p):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = df.columns.str.lower()
        return df.sort_index()

    return load(f_1h), load(f_4h), load(f_15m), load(f_1m)


def ic(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 30:
        return float("nan"), mask.sum()
    r, _ = pearsonr(x[mask], y[mask])
    return r, mask.sum()


# ─────────────────────────────────────────────────────────────────────────────
def build_signal_df(ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m):
    close_1h  = ohlcv_1h["close"].astype(float)
    high_1h   = ohlcv_1h["high"].astype(float)
    low_1h    = ohlcv_1h["low"].astype(float)
    close_4h  = ohlcv_4h["close"].astype(float)
    high_4h   = ohlcv_4h["high"].astype(float)
    low_4h    = ohlcv_4h["low"].astype(float)
    volume_4h = ohlcv_4h["volume"].astype(float)
    close_15m = ohlcv_15m["close"].astype(float)
    high_15m  = ohlcv_15m["high"].astype(float)
    low_15m   = ohlcv_15m["low"].astype(float)
    close_1m  = ohlcv_1m["close"].astype(float)
    volume_1m = ohlcv_1m["volume"].astype(float)
    ts_1h     = close_1h.index

    # ── Target ────────────────────────────────────────────────────────────────
    next_ret = np.log(close_1h / close_1h.shift(1)).shift(-1)
    next_up  = (next_ret > 0).astype(float)

    signals = {}

    # ── Trend signals (4h, continuous) ───────────────────────────────────────
    # 1. Stoch K 4h (0–100, higher = more overbought = continuation bullish)
    stk4 = _stoch_k(high_4h, low_4h, close_4h, 14)
    signals["T1_stoch_k_4h"]   = stk4.reindex(ts_1h, method="ffill")

    # 2. Vol direction 4h (+1=high_vol_up, -1=high_vol_down, 0=neutral)
    vsig = _vol_signal_4h(close_4h, volume_4h)
    vol_dir = pd.Series(0.0, index=vsig.index)
    vol_dir[vsig == "high_vol_up"]   =  1.0
    vol_dir[vsig == "high_vol_down"] = -1.0
    signals["T2_vol_dir_4h"]   = vol_dir.reindex(ts_1h, method="ffill")

    # 3. MACD diff 4h (MACD line - signal; positive = bullish momentum)
    macd_diff = close_4h.ewm(span=12).mean() - close_4h.ewm(span=26).mean()
    macd_sig  = macd_diff.ewm(span=9).mean()
    signals["T3_macd_diff_4h"] = (macd_diff - macd_sig).reindex(ts_1h, method="ffill")

    # 4. BB %B 4h (0=near lower band, 1=near upper band)
    signals["T4_bb_pct_4h"]    = _bb_pct(high_4h, low_4h, close_4h, 20).reindex(ts_1h, method="ffill")

    # 5. Keltner %K 4h (0=lower band, 1=upper band, can exceed)
    kc_pct, _, _ = _keltner_pct(high_4h, low_4h, close_4h, 20, 2)
    signals["T5_kc_pct_4h"]    = kc_pct.reindex(ts_1h, method="ffill")

    # 6. Williams %R 4h (-100 to 0; higher = overbought = continuation bullish)
    signals["T6_wpr_4h"]       = _wpr(high_4h, low_4h, close_4h, 14).reindex(ts_1h, method="ffill")

    # 7. Stoch K-D diff 4h (NEW — K minus D smooth; positive = K crossing above D)
    stk_d4 = stk4.ewm(span=3, adjust=False).mean()
    signals["T7_kd_diff_4h"]   = (stk4 - stk_d4).reindex(ts_1h, method="ffill")

    # 8. EMA-20 dist z 4h (NEW — price stretch above EMA as z-score)
    ema_4h   = close_4h.ewm(span=20, adjust=False).mean()
    ema_dist = (close_4h - ema_4h) / ema_4h.replace(0, float("nan"))
    ema_mu   = ema_dist.rolling(48, min_periods=24).mean()
    ema_sig  = ema_dist.rolling(48, min_periods=24).std()
    ema_z    = (ema_dist - ema_mu) / ema_sig.replace(0, float("nan"))
    signals["T8_ema_z_4h"]     = ema_z.reindex(ts_1h, method="ffill")

    # ── Rev signals (1h/15m, continuous) ─────────────────────────────────────
    # Rev signals are mean-reverting, so NEGATIVE value → expect UP.
    # IC sign: negative IC on raw signal = positive edge (signal correctly predicts reversal).
    # We report raw IC so the sign carries meaning.

    # R1. RSI 1h (lower = more oversold = expect up; negative IC is "good")
    signals["R1_rsi_1h"]       = _rsi(close_1h, 14)

    # R2. Stoch K 15m (lower = more oversold)
    stk15 = _stoch_k(high_15m, low_15m, close_15m, 14)
    signals["R2_stoch_k_15m"]  = stk15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    # R3. Stoch K 1h
    signals["R3_stoch_k_1h"]   = _stoch_k(high_1h, low_1h, close_1h, 14)

    # R4. VWAP deviation (positive = above VWAP = expect down; negative IC is "good")
    vwap_h  = _vwap_1h(close_1m, volume_1m).reindex(ts_1h, method="ffill")
    signals["R4_vwap_dev_1h"]  = (close_1h - vwap_h) / vwap_h.replace(0, float("nan"))

    # R5. Donchian %D 15m (higher = near top = expect down)
    dc15 = _dc_pct(high_15m, low_15m, close_15m, 20)
    signals["R5_dc_pct_15m"]   = dc15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    # R6. Keltner %K 15m (higher = near/above top = expect down)
    kc15, _, _ = _keltner_pct(high_15m, low_15m, close_15m, 20, 2)
    signals["R6_kc_pct_15m"]   = kc15.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

    # R7. Williams %R 1h (higher = more overbought = expect down)
    signals["R7_wpr_1h"]       = _wpr(high_1h, low_1h, close_1h, 14)

    # R8. 1h move z-score (large negative move = expect bounce up; negative IC = "good")
    log_ret  = np.log(close_1h / close_1h.shift(1))
    roll_vol = log_ret.rolling(24).std()
    signals["R8_move_z_1h"]    = log_ret / roll_vol.replace(0, float("nan"))

    # ── Composite scores ──────────────────────────────────────────────────────
    trend_ser, rev_ser = compute_scores(
        close_1h, high_1h, low_1h, ohlcv_1h["volume"].astype(float),
        close_4h, high_4h, low_4h, volume_4h,
        close_15m, high_15m, low_15m,
        close_1m, volume_1m, ts_1h,
    )
    signals["C_composite_trend"] = trend_ser
    signals["C_composite_rev"]   = rev_ser

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(signals)
    df["next_up"] = next_up
    df = df.dropna(subset=["next_up"])
    df = df[df.index >= TEST_START]
    return df


def join_regime(df):
    labels = pd.read_parquet(LABELS_PATH)
    labels.index = pd.to_datetime(labels.index, utc=True)
    score_times = df.index.values
    label_times = labels.index.values
    idx = np.searchsorted(label_times, score_times, side="right") - 1
    idx = np.clip(idx, 0, len(label_times) - 1)
    df = df.copy()
    df["regime"] = labels["regime"].values[idx]
    return df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m = load_data()

    print("Computing signals...")
    df = build_signal_df(ohlcv_1h, ohlcv_4h, ohlcv_15m, ohlcv_1m)

    print("Joining regime labels...")
    df = join_regime(df)

    signal_cols = [c for c in df.columns if c not in ("next_up", "regime")]

    print(f"\n{'='*80}")
    print("  PER-SIGNAL IC vs next_up  |  by Macro Regime")
    print(f"  Test period: {df.index.min().date()} → {df.index.max().date()}")
    regime_ns = {r: len(df[df["regime"] == r]) for r in REGIMES}
    print(f"  n  — Bull: {regime_ns['Bull']:,}  Sideways: {regime_ns['Sideways']:,}  Bear: {regime_ns['Bear']:,}  All: {len(df):,}")
    print(f"{'='*80}")
    print(f"  {'Signal':<25}  {'All':>8}  {'Bull':>8}  {'Sideways':>8}  {'Bear':>8}  {'SWvsBear':>9}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    results = []
    for col in signal_cols:
        x_all  = df[col].values.astype(float)
        y      = df["next_up"].values.astype(float)
        ic_all, n_all = ic(x_all, y)

        row = {"signal": col, "ic_all": ic_all}
        for r in REGIMES:
            mask = df["regime"] == r
            ic_r, n_r = ic(x_all[mask.values], y[mask.values])
            row[f"ic_{r}"] = ic_r
            row[f"n_{r}"]  = n_r
        row["sw_vs_bear"] = (row.get("ic_Sideways", 0) or 0) - (row.get("ic_Bear", 0) or 0)
        results.append(row)

    # Sort by Sideways IC absolute value descending to highlight what matters there
    results.sort(key=lambda r: abs(r.get("ic_Sideways") or 0), reverse=True)

    section = ""
    for r in results:
        sig = r["signal"]
        prefix = sig[0]
        if prefix != section:
            section = prefix
            print()

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "    —   "
            return f"{v:+.4f}"

        sw_vs = r["sw_vs_bear"]
        flag = "  ← SW+" if (sw_vs and sw_vs > 0.05) else ("  ← SW−" if (sw_vs and sw_vs < -0.05) else "")
        print(f"  {sig:<25}  {fmt(r['ic_all'])}  {fmt(r.get('ic_Bull'))}  "
              f"{fmt(r.get('ic_Sideways'))}  {fmt(r.get('ic_Bear'))}  "
              f"{fmt(sw_vs)}{flag}")

    print(f"\n{'='*80}")
    print("  SIDEWAYS TOP SIGNALS  (|IC| > 0.03, sorted)")
    print(f"{'='*80}")
    sw_top = sorted(results, key=lambda r: abs(r.get("ic_Sideways") or 0), reverse=True)
    sw_top = [r for r in sw_top if abs(r.get("ic_Sideways") or 0) > 0.03]
    for r in sw_top:
        print(f"  {r['signal']:<25}  SW IC={r.get('ic_Sideways'):+.4f}  "
              f"n={r.get('n_Sideways'):,}  (Bull={r.get('ic_Bull'):+.4f}  Bear={r.get('ic_Bear'):+.4f})")

    print(f"\n{'='*80}")
    print("  REGIME DIVERGENCE  (signals with |Sideways - Bear| > 0.05)")
    print(f"{'='*80}")
    div = sorted(results, key=lambda r: abs(r.get("sw_vs_bear") or 0), reverse=True)
    div = [r for r in div if abs(r.get("sw_vs_bear") or 0) > 0.05]
    for r in div:
        print(f"  {r['signal']:<25}  SW={r.get('ic_Sideways'):+.4f}  Bear={r.get('ic_Bear'):+.4f}  "
              f"diff={r.get('sw_vs_bear'):+.4f}")


if __name__ == "__main__":
    main()
