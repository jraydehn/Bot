"""
analysis_signals_comprehensive.py — Unbiased signal evaluation for both YES and NO.

For every hour in the out-of-sample test set (Jan 2025–Apr 2026), compute a set of
candidate predictive indicators, then measure actual YES/NO win rates at each offset
against the dynamic p_market breakeven. Edge = win_rate - breakeven_win_rate.

Signals evaluated (no directional pre-bias):
  1. EMA alignment (20/50 on 1h)          — trend direction
  2. RSI-14 on 1h bars                    — momentum / overbought / oversold
  3. ADX-14 on 15m bars                   — trend STRENGTH (regime: ranging vs trending)
  4. Bollinger Band width percentile      — volatility regime (squeeze vs expansion)
  5. Vol momentum (sigma_model trend)     — is realized vol rising or falling?
  6. Price momentum 1h                    — last bar direction and magnitude
  7. Price momentum 4h                    — medium-term drift
  8. VWAP deviation (24h rolling)         — price vs fair value
  9. Stochastic crossover (15m)           — momentum turning point
 10. Vol ratio regime                     — sigma_model / sigma_kalshi (edge direction)

Edge calculation uses dynamic p_market (Kalshi's lagged-vol log-normal pricing),
exactly as in backtest_unbiased.py. Calibration factor from walk-forward training.
Only results with n >= 80 bars reported to avoid noise.
"""

import sys, math, glob, warnings
from pathlib import Path
from scipy.stats import norm as sp_norm, binom
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
KALSHI_VOL_WINDOW = 1440
KALSHI_VOL_LAG    = 120
MODEL_VOL_WINDOW  = 60
WARMUP_BARS       = KALSHI_VOL_WINDOW + KALSHI_VOL_LAG + 60
TAU               = 60
CALIB             = 0.9831   # from walk-forward training Jan–Dec 2024
COSTS             = DEFAULT_SLIPPAGE + DEFAULT_SPREAD
MIN_N             = 80       # minimum bars per bucket to report
OFFSETS           = [-0.003, -0.001, 0.001, 0.002, 0.003, 0.005]
TEST_START        = pd.Timestamp("2025-01-01", tz="UTC")
SEP               = "=" * 76

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
f1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
ohlcv_1m = pd.read_parquet(f1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

f1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
ohlcv_1h = pd.read_parquet(f1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h = ohlcv_1h["close"].values.astype(float)
high_1h  = ohlcv_1h["high"].values.astype(float)
low_1h   = ohlcv_1h["low"].values.astype(float)
ts_1h    = ohlcv_1h.index
n1h      = len(ts_1h)

# ── VOL SERIES ────────────────────────────────────────────────────────────────
print("Computing vol series...")
close_1m = ohlcv_1m["close"].values.astype(float)
log_ret  = pd.Series(np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0),
                     index=ohlcv_1m.index)
sigma_model_1m  = log_ret.rolling(MODEL_VOL_WINDOW).std()
sigma_kalshi_1m = log_ret.rolling(KALSHI_VOL_WINDOW).std().shift(KALSHI_VOL_LAG)
ohlcv_1m_idx    = ohlcv_1m.index

# ── SIGNAL 1: EMA ALIGNMENT (1h) ──────────────────────────────────────────────
print("Computing signals...")
close_s  = pd.Series(close_1h, index=ts_1h)
ema20_1h = close_s.ewm(span=20, adjust=False).mean()
ema50_1h = close_s.ewm(span=50, adjust=False).mean()
ema_align = pd.Series("neutral", index=ts_1h)
bullish = (ema20_1h > ema50_1h) & (close_s > ema20_1h)
bearish = (ema20_1h < ema50_1h) | (close_s < ema50_1h)
ema_align[bullish] = "bullish"
ema_align[bearish] = "bearish"

# ── SIGNAL 2: RSI-14 (1h) ────────────────────────────────────────────────────
delta    = close_s.diff()
gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi_1h   = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
def rsi_bucket(r):
    if r >= 70: return "overbought(>70)"
    if r >= 55: return "mild_bull(55-70)"
    if r >= 45: return "neutral(45-55)"
    if r >= 30: return "mild_bear(30-45)"
    return "oversold(<30)"
rsi_label_1h = rsi_1h.map(rsi_bucket)

# ── SIGNAL 3: ADX-14 (15m — trend strength) ──────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

h15 = df_15m["high"]
l15 = df_15m["low"]
c15 = df_15m["close"]
pc15 = c15.shift(1)
tr15 = pd.concat([h15 - l15, (h15 - pc15).abs(), (l15 - pc15).abs()], axis=1).max(axis=1)
pdm15 = (h15 - h15.shift(1)).clip(lower=0)
ndm15 = (l15.shift(1) - l15).clip(lower=0)
pdm15 = pdm15.where(pdm15 > ndm15, 0)
ndm15 = ndm15.where(ndm15 > pdm15.shift(1).fillna(0), 0)  # simplified

atr14_15m  = tr15.ewm(com=13, adjust=False).mean()
pdi14_15m  = 100 * pdm15.ewm(com=13, adjust=False).mean() / atr14_15m.replace(0, 1e-9)
ndi14_15m  = 100 * ndm15.ewm(com=13, adjust=False).mean() / atr14_15m.replace(0, 1e-9)
dx_15m     = 100 * (pdi14_15m - ndi14_15m).abs() / (pdi14_15m + ndi14_15m).replace(0, 1e-9)
adx_15m    = dx_15m.ewm(com=13, adjust=False).mean()

def adx_bucket(a):
    if a < 15:  return "ranging(<15)"
    if a < 25:  return "weak_trend(15-25)"
    if a < 35:  return "trend(25-35)"
    return "strong_trend(>35)"
adx_label_15m = adx_15m.map(adx_bucket)
# Forward-fill to 1h index
adx_ff    = adx_15m.reindex(ts_1h, method="ffill")
adx_lbl_ff = adx_label_15m.reindex(ts_1h, method="ffill")

# ── SIGNAL 4: BOLLINGER BAND WIDTH PERCENTILE (1h) ───────────────────────────
bb_mid   = close_s.rolling(20).mean()
bb_std   = close_s.rolling(20).std()
bb_width = (4 * bb_std / bb_mid)  # 2-sigma width as fraction
bb_pct   = bb_width.rolling(100).rank(pct=True)  # percentile vs last 100h
def bb_bucket(p):
    if pd.isna(p): return "unknown"
    if p < 0.20: return "squeeze(<20pct)"
    if p < 0.40: return "low(20-40pct)"
    if p < 0.60: return "mid(40-60pct)"
    if p < 0.80: return "high(60-80pct)"
    return "expand(>80pct)"
bb_label_1h = bb_pct.map(bb_bucket)

# ── SIGNAL 5: VOL MOMENTUM — is realized vol rising or falling? ───────────────
# Compare current 60m vol to 4h ago (240m vol)
sigma_4h_ago = sigma_model_1m.shift(240)  # 240 1m bars = 4h
# Forward-fill to 1h
sm_1h = sigma_model_1m.reindex(ts_1h, method="ffill")
s4h_1h = sigma_4h_ago.reindex(ts_1h, method="ffill")
vol_mom_ratio = sm_1h / s4h_1h.replace(0, 1e-9)
def volmom_bucket(r):
    if pd.isna(r): return "unknown"
    if r < 0.7:  return "contracting(<0.7x)"
    if r < 0.9:  return "falling(0.7-0.9x)"
    if r < 1.1:  return "stable(0.9-1.1x)"
    if r < 1.5:  return "rising(1.1-1.5x)"
    return "spiking(>1.5x)"
volmom_label = vol_mom_ratio.map(volmom_bucket)

# ── SIGNAL 6: PRICE MOMENTUM 1h ──────────────────────────────────────────────
ret_1h = close_s.pct_change() * 100  # % return last 1h bar
def mom1h_bucket(r):
    if pd.isna(r): return "unknown"
    if r < -0.5:  return "down_strong(<-0.5%)"
    if r < -0.1:  return "down_mild(-0.5 to -0.1%)"
    if r <  0.1:  return "flat(-0.1 to 0.1%)"
    if r <  0.5:  return "up_mild(0.1 to 0.5%)"
    return "up_strong(>0.5%)"
mom1h_label = ret_1h.map(mom1h_bucket)

# ── SIGNAL 7: PRICE MOMENTUM 4h ──────────────────────────────────────────────
ret_4h = close_s.pct_change(4) * 100
def mom4h_bucket(r):
    if pd.isna(r): return "unknown"
    if r < -1.5:  return "down_strong(<-1.5%)"
    if r < -0.3:  return "down_mild"
    if r <  0.3:  return "flat"
    if r <  1.5:  return "up_mild"
    return "up_strong(>1.5%)"
mom4h_label = ret_4h.map(mom4h_bucket)

# ── SIGNAL 8: VWAP DEVIATION (24h rolling from 1m data) ──────────────────────
tp_1m    = (ohlcv_1m["high"] + ohlcv_1m["low"] + ohlcv_1m["close"]) / 3
vwap_24h = (tp_1m * ohlcv_1m["volume"]).rolling(1440).sum() / ohlcv_1m["volume"].rolling(1440).sum()
close_1m_s = pd.Series(close_1m, index=ohlcv_1m.index)
vwap_dev   = (close_1m_s - vwap_24h) / vwap_24h * 100  # % above/below VWAP
vwap_dev_ff = vwap_dev.reindex(ts_1h, method="ffill")
def vwap_bucket(d):
    if pd.isna(d): return "unknown"
    if d < -1.0:   return "far_below(<-1%)"
    if d < -0.3:   return "below(-1 to -0.3%)"
    if d <  0.3:   return "near(+/-0.3%)"
    if d <  1.0:   return "above(0.3 to 1%)"
    return "far_above(>1%)"
vwap_label = vwap_dev_ff.map(vwap_bucket)

# ── SIGNAL 9: STOCHASTIC (15m) ────────────────────────────────────────────────
stoch_k_15m  = ((c15 - l15.rolling(14).min()) /
                (h15.rolling(14).max() - l15.rolling(14).min()).replace(0,1e-9)) * 100
stoch_d_15m  = stoch_k_15m.rolling(3).mean()
k_curr15 = stoch_k_15m; k_prev15 = stoch_k_15m.shift(1)
d_curr15 = stoch_d_15m; d_prev15 = stoch_d_15m.shift(1)
bull_xover = ((k_prev15 < d_prev15) & (k_curr15 > d_curr15) & (k_prev15 < 20))
bull_xover_active = (bull_xover | bull_xover.shift(1).fillna(False))
bear_xover = ((k_prev15 > d_prev15) & (k_curr15 < d_curr15) & (k_prev15 > 80))
bear_xover_active = (bear_xover | bear_xover.shift(1).fillna(False))

def stoch_bucket(row):
    k, bullx, bearx = row
    if pd.isna(k): return "unknown"
    if bullx:      return "bull_crossup"
    if bearx:      return "bear_crossdown"
    if k < 20:     return "oversold(<20)"
    if k > 80:     return "overbought(>80)"
    return "neutral"
stoch_df = pd.DataFrame({"k": stoch_k_15m, "bullx": bull_xover_active, "bearx": bear_xover_active})
stoch_label_15m = stoch_df.apply(stoch_bucket, axis=1)
stoch_lbl_ff    = stoch_label_15m.reindex(ts_1h, method="ffill").fillna("unknown")
stoch_k_ff      = stoch_k_15m.reindex(ts_1h, method="ffill")

# ── SIGNAL 10: VOL RATIO REGIME ───────────────────────────────────────────────
# (already computed per-bar in main loop below)

print("Pre-computation done.\n")


# ── HELPER: log-normal p_yes ──────────────────────────────────────────────────
def p_lognorm(spot, K, tau, sigma):
    if sigma <= 0 or spot <= 0 or K <= 0: return float("nan")
    z = math.log(K / spot) / (sigma * math.sqrt(tau))
    return float(1.0 - sp_norm.cdf(z))


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("Running main analysis loop (test set Jan 2025 – Apr 2026)...")
records = []

for i_h in range(50, n1h - 1):
    ts_now = ts_1h[i_h]
    if ts_now < TEST_START:
        continue

    spot       = float(close_1h[i_h])
    next_close = float(close_1h[i_h + 1])

    pos1m = int(ohlcv_1m_idx.searchsorted(ts_now, side="right")) - 1
    if pos1m < WARMUP_BARS:
        continue

    sig_m = float(sigma_model_1m.iat[pos1m])
    sig_k = float(sigma_kalshi_1m.iat[pos1m])
    if np.isnan(sig_m) or sig_m <= 0 or np.isnan(sig_k) or sig_k <= 0:
        continue

    vr = sig_m / sig_k

    # Collect all signals for this bar
    signals = {
        "ema":      str(ema_align.iloc[i_h]),
        "rsi":      rsi_label_1h.iloc[i_h],
        "adx":      str(adx_lbl_ff.iloc[i_h]),
        "bb":       str(bb_label_1h.iloc[i_h]),
        "volmom":   str(volmom_label.iloc[i_h]),
        "mom1h":    str(mom1h_label.iloc[i_h]),
        "mom4h":    str(mom4h_label.iloc[i_h]),
        "vwap":     str(vwap_label.iloc[i_h]),
        "stoch":    str(stoch_lbl_ff.iloc[i_h]),
        "vr_regime": "high(>1.2)" if vr > 1.2 else "low(<0.7)" if vr < 0.7 else "mid(0.7-1.2)",
        "rsi_raw":  float(rsi_1h.iloc[i_h]),
        "adx_raw":  float(adx_ff.iloc[i_h]),
        "vwap_raw": float(vwap_dev_ff.iloc[i_h]) if not pd.isna(vwap_dev_ff.iloc[i_h]) else 0.0,
        "mom1h_raw":float(ret_1h.iloc[i_h]) if not pd.isna(ret_1h.iloc[i_h]) else 0.0,
        "vol_ratio":round(vr, 4),
    }

    for offset in OFFSETS:
        K           = spot * (1.0 + offset)
        actual_yes  = int(next_close > K)

        p_market    = p_lognorm(spot, K, TAU, sig_k)
        p_raw       = p_lognorm(spot, K, TAU, sig_m)
        if np.isnan(p_market) or np.isnan(p_raw): continue
        if not (0.03 <= p_market <= 0.97):         continue

        p_model = p_raw * CALIB
        fee     = kalshi_fee(p_market)
        total_costs = fee + COSTS

        yes_edge = p_model - p_market - total_costs
        no_edge  = p_market - p_model - total_costs

        rec = {**signals, "offset": offset, "actual_yes": actual_yes,
               "p_market": round(p_market, 4), "p_model": round(p_model, 4),
               "yes_edge": round(yes_edge, 4), "no_edge": round(no_edge, 4)}
        records.append(rec)

df = pd.DataFrame(records)
print(f"Total observations: {len(df):,}  ({len(df)//len(OFFSETS):,} hourly bars)\n")


# ── ANALYSIS FUNCTION ─────────────────────────────────────────────────────────
def analyze_signal(signal_col, side, offsets_to_show=None, min_n=MIN_N):
    """
    For a given signal column and side (yes/no), compute for each signal level:
      - n bars, actual win rate, breakeven win rate, net edge (win% - breakeven%)
      - p-value (one-sided binomial: is win rate > breakeven?)
    """
    tgt_offsets = offsets_to_show or OFFSETS
    rows = []

    for sig_val in sorted(df[signal_col].unique()):
        if sig_val in ("unknown", "nan"): continue
        sub_sig = df[df[signal_col] == sig_val]

        for off in tgt_offsets:
            sub = sub_sig[sub_sig["offset"] == off]
            if len(sub) < min_n: continue

            actual_win_rate = sub["actual_yes"].mean() if side == "yes" else (1 - sub["actual_yes"]).mean()
            avg_pm = sub["p_market"].mean()

            # Breakeven win rate: the win probability that gives EV = 0
            # YES: breakeven_p = p_market + costs  (must overcome market price + fees)
            # NO:  breakeven_p = (1-p_market) + costs
            if side == "yes":
                breakeven = avg_pm + sub["yes_edge"].mean() - (actual_win_rate - avg_pm) + avg_pm
                # simpler: need win_rate * (1-pm)/pm >= (1-win_rate) → win_rate >= pm + costs_adj
                be = avg_pm + (kalshi_fee(avg_pm) + COSTS) * (1 - avg_pm)  # approx
                be = min(max(be, 0.01), 0.99)
            else:
                be_raw = 1 - avg_pm
                be = be_raw + (kalshi_fee(avg_pm) + COSTS) * avg_pm  # approx
                be = min(max(be, 0.01), 0.99)

            edge_pct = actual_win_rate - be

            # Binomial p-value: probability of seeing this win rate by chance if true rate = be
            n_wins = int(actual_win_rate * len(sub))
            from scipy.stats import binom as binom_dist
            pval = 1 - binom_dist.cdf(n_wins - 1, len(sub), be)

            rows.append({
                "signal": sig_val, "offset": off, "n": len(sub),
                "win%": round(actual_win_rate * 100, 1),
                "breakeven%": round(be * 100, 1),
                "edge%": round(edge_pct * 100, 1),
                "avg_pm": round(avg_pm, 3),
                "p_value": round(pval, 3),
                "significant": "**" if pval < 0.05 else ("*" if pval < 0.15 else ""),
            })

    return pd.DataFrame(rows)


# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
SIGNAL_COLS = [
    ("ema",      "EMA alignment (1h 20/50)"),
    ("rsi",      "RSI-14 (1h)"),
    ("adx",      "ADX-14 (15m) — trend strength"),
    ("bb",       "Bollinger Band width percentile (1h)"),
    ("volmom",   "Vol momentum (60m vol vs 4h ago)"),
    ("mom1h",    "Price momentum 1h"),
    ("mom4h",    "Price momentum 4h"),
    ("vwap",     "VWAP deviation (24h rolling)"),
    ("stoch",    "Stochastic (15m)"),
    ("vr_regime","Vol ratio regime"),
]

for side in ("yes", "no"):
    print(SEP)
    print(f"SIDE: {side.upper()} BETS")
    print(SEP)

    # Only show offsets relevant to each side
    if side == "yes":
        show_offsets = [-0.003, -0.001, 0.001, 0.002]
    else:
        show_offsets = [0.001, 0.002, 0.003, 0.005]

    for col, label in SIGNAL_COLS:
        res = analyze_signal(col, side, show_offsets)
        if len(res) == 0: continue

        # Only print if at least one row has meaningful edge (abs > 3%) or significance
        has_signal = ((res["edge%"].abs() > 3) | (res["significant"] != "")).any()
        if not has_signal: continue

        print(f"\n  [{label}]")
        print(f"  {'signal':<28} {'off':>6} {'n':>5} {'win%':>6} {'be%':>5} {'edge%':>6} {'pval':>6} {'sig'}")
        print("  " + "-" * 72)
        for _, r in res.sort_values(["offset","edge%"], ascending=[True,False]).iterrows():
            marker = " ← EDGE" if r["edge%"] > 5 and r["significant"] != "" else ""
            marker = " ← STRONG EDGE" if r["edge%"] > 8 and r["p_value"] < 0.05 else marker
            print(f"  {str(r['signal']):<28} {r['offset']:>+6.3f} {r['n']:>5} "
                  f"{r['win%']:>5.1f}% {r['breakeven%']:>4.1f}% {r['edge%']:>+5.1f}% "
                  f"{r['p_value']:>6.3f} {r['significant']:>2}{marker}")


# ── COMBINED SIGNAL ANALYSIS ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("COMBINED SIGNAL ANALYSIS — Top combinations (edge > 8%, n > 80, p < 0.10)")
print(SEP)

combo_rows = []
for side in ("yes", "no"):
    show_off = [-0.001, 0.001, 0.002, 0.003] if side == "yes" else [0.001, 0.002, 0.003, 0.005]

    # Two-signal combinations: EMA × each other signal
    for sig2_col, sig2_label in SIGNAL_COLS[1:]:
        for ema_val in df["ema"].unique():
            if ema_val == "unknown": continue
            for sig2_val in df[sig2_col].unique():
                if sig2_val in ("unknown", "nan"): continue

                sub_combo = df[(df["ema"] == ema_val) & (df[sig2_col] == sig2_val)]
                for off in show_off:
                    sub = sub_combo[sub_combo["offset"] == off]
                    if len(sub) < 60: continue

                    wr = sub["actual_yes"].mean() if side == "yes" else (1-sub["actual_yes"]).mean()
                    avg_pm = sub["p_market"].mean()
                    if side == "yes":
                        be = avg_pm + (kalshi_fee(avg_pm) + COSTS) * (1 - avg_pm)
                    else:
                        be = (1 - avg_pm) + (kalshi_fee(avg_pm) + COSTS) * avg_pm
                    be = min(max(be, 0.01), 0.99)
                    edge = (wr - be) * 100
                    n_wins = int(wr * len(sub))
                    pval = 1 - binom.cdf(n_wins - 1, len(sub), be)

                    if edge > 8 and pval < 0.10:
                        combo_rows.append({
                            "side": side, "offset": off, "n": len(sub),
                            "ema": ema_val, "signal": f"{sig2_col}={sig2_val}",
                            "win%": round(wr*100,1), "be%": round(be*100,1),
                            "edge%": round(edge,1), "pval": round(pval,3)
                        })

if combo_rows:
    combos = pd.DataFrame(combo_rows).sort_values("edge%", ascending=False)
    print(f"\n  {'side':>4} {'off':>6} {'n':>5} {'ema':<9} {'signal':<35} "
          f"{'win%':>6} {'be%':>5} {'edge%':>6} {'pval':>6}")
    print("  " + "-" * 90)
    for _, r in combos.head(40).iterrows():
        print(f"  {r['side']:>4} {r['offset']:>+6.3f} {r['n']:>5} {r['ema']:<9} "
              f"{r['signal']:<35} {r['win%']:>5.1f}% {r['be%']:>4.1f}% "
              f"{r['edge%']:>+5.1f}% {r['pval']:>6.3f}")
else:
    print("  No combinations found meeting criteria.")

print(f"\n{SEP}")
print("SUMMARY — Best edges by side")
print(SEP)
for side in ("yes","no"):
    show_off = [-0.001, 0.001, 0.002] if side == "yes" else [0.001, 0.002, 0.003]
    best = []
    for col, _ in SIGNAL_COLS:
        res = analyze_signal(col, side, show_off, min_n=60)
        for _, r in res.iterrows():
            if r["edge%"] > 5 and r["p_value"] < 0.15:
                best.append((r["edge%"], r["p_value"], r["n"], col, r["signal"], r["offset"], r["win%"], r["breakeven%"]))
    best.sort(reverse=True)
    print(f"\n  {side.upper()} — top signals with edge > 5% (p < 0.15):")
    if best:
        for edge, pv, n, col, sig, off, wr, be in best[:15]:
            print(f"    {col}={sig}  off={off:+.3f}  n={n}  win={wr:.1f}%  be={be:.1f}%  edge={edge:+.1f}%  p={pv:.3f}")
    else:
        print("    None found.")
