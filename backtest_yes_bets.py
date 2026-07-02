"""
backtest_yes_bets.py — Regime-gated YES bet analysis.

Mirror of backtest_regime_gates.py but evaluates BUYING YES contracts
(betting BTC closes above strike) rather than NO.

Questions answered:
  1. At what offsets do YES bets have positive expected value (any regime)?
  2. Which regimes — BB squeeze, ADX ranging, RSI oversold, trending — lift YES win rates?
  3. Does the BB squeeze gate that HELPS NO bets also help YES bets?
     (Hypothesis: squeeze suppresses large moves → OTM YES bets lose → NO is better)
  4. Does RSI oversold flip YES into a winning signal
     (bounce trade: RSI <30 → likely reversal → YES wins)?

Kalshi YES prices used (same table as backtest_regime_gates.py, from 239 paper-trade obs):
  0.10-0.15% OTM offset → YES price ≈ 0.465  (YES breakeven 46.5%)
  0.15-0.20% OTM offset → YES price ≈ 0.330  (YES breakeven 33.0%)
  0.20-0.25% OTM offset → YES price ≈ 0.330  (YES breakeven 33.0%)
  0.25-0.30% OTM offset → YES price ≈ 0.250  (YES breakeven 25.0%)
  0.30-0.40% OTM offset → YES price ≈ 0.210  (YES breakeven 21.0%)
  0.40-0.50% OTM offset → YES price ≈ 0.190  (YES breakeven 19.0%)
  0.50-1.00% OTM offset → YES price ≈ 0.145  (YES breakeven 14.5%)

Walk-forward: classify regimes on full history, evaluate test set Jan 2025–Apr 2026.
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

SEP = "=" * 72
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

FIXED_STAKE = 50.0
KALSHI_RAKE = 0.07
MIN_N       = 30   # lower threshold for YES (fewer obs in some regimes)

# ── Actual observed Kalshi YES prices by offset bucket ────────────────────────
# Same table as backtest_regime_gates.py (calibrated from 239 paper-trade obs)
KALSHI_PRICE_TABLE = {
    (0.001,  0.0015): 0.465,
    (0.0015, 0.002):  0.330,
    (0.002,  0.0025): 0.330,
    (0.0025, 0.003):  0.250,
    (0.003,  0.004):  0.210,
    (0.004,  0.005):  0.190,
    (0.005,  0.010):  0.145,
}

OFFSET_LABELS = {
    (0.001,  0.0015): "0.10-0.15%",
    (0.0015, 0.002):  "0.15-0.20%",
    (0.002,  0.0025): "0.20-0.25%",
    (0.0025, 0.003):  "0.25-0.30%",
    (0.003,  0.004):  "0.30-0.40%",
    (0.004,  0.005):  "0.40-0.50%",
    (0.005,  0.010):  "0.50-1.00%",
}


# ── Load data ─────────────────────────────────────────────────────────────────
print(SEP)
print("Loading data and computing indicators...")
print(SEP)

files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))

ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h = pd.Series(ohlcv_1h["close"].values.astype(float), index=ohlcv_1h.index)
high_1h  = pd.Series(ohlcv_1h["high"].values.astype(float),  index=ohlcv_1h.index)
low_1h   = pd.Series(ohlcv_1h["low"].values.astype(float),   index=ohlcv_1h.index)
n1h      = len(close_1h)
ts_1h    = ohlcv_1h.index

# ── Bollinger Band width percentile ──────────────────────────────────────────
bb_mid   = close_1h.rolling(20).mean()
bb_std   = close_1h.rolling(20).std()
bb_width = (4 * bb_std) / bb_mid
bb_pct   = bb_width.rolling(500).rank(pct=True)

# ── RSI-14 ────────────────────────────────────────────────────────────────────
delta = close_1h.diff()
gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

# ── EMA position: close vs 20-bar EMA ─────────────────────────────────────────
ema_20 = close_1h.ewm(span=20, adjust=False).mean()
ema_pos = (close_1h - ema_20) / ema_20   # positive = price above EMA (bullish)

# ── ADX-14 on 15m, forward-filled to 1h ──────────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"high": "max", "low": "min", "close": "last"}
).dropna(subset=["close"])

_h  = df_15m["high"]
_l  = df_15m["low"]
_cp = df_15m["close"].shift(1)
tr  = pd.concat([_h - _l, (_h - _cp).abs(), (_l - _cp).abs()], axis=1).max(axis=1)
dmp = (_h - _h.shift(1)).clip(lower=0).where((_h - _h.shift(1)) > (_l.shift(1) - _l), 0)
dmm = (_l.shift(1) - _l).clip(lower=0).where((_l.shift(1) - _l) > (_h - _h.shift(1)), 0)
atr = tr.ewm(com=13, adjust=False).mean()
dip = 100 * dmp.ewm(com=13, adjust=False).mean() / atr.replace(0, 1e-10)
dim = 100 * dmm.ewm(com=13, adjust=False).mean() / atr.replace(0, 1e-10)
dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, 1e-10)
adx_15m = dx.ewm(com=13, adjust=False).mean()
adx_1h  = adx_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

# ── Stochastic K/D on 15m, forward-filled to 1h ──────────────────────────────
STOCH_K  = 14
STOCH_D  = 3
ll_15m   = df_15m["low"].rolling(STOCH_K).min()
hh_15m   = df_15m["high"].rolling(STOCH_K).max()
hl_rng   = (hh_15m - ll_15m).replace(0, float("nan"))
stoch_k_15m = ((df_15m["close"] - ll_15m) / hl_rng) * 100
stoch_d_15m = stoch_k_15m.rolling(STOCH_D).mean()
# Bullish crossover from oversold: K crosses above D while K_prev < 20
k_prev = stoch_k_15m.shift(1)
d_prev = stoch_d_15m.shift(1)
bullish_xover_15m = (
    (k_prev < d_prev) & (stoch_k_15m > stoch_d_15m) & (k_prev < 20)
)
bullish_xover_15m = bullish_xover_15m | bullish_xover_15m.shift(1).fillna(False)
stoch_k_1h  = stoch_k_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
bullish_xover_1h = bullish_xover_15m.resample("1h", origin="start_day").max().reindex(ts_1h, method="ffill").fillna(False)

print(f"  1h bars: {n1h:,}")
print(f"  BB squeeze (pct<0.20)      : {(bb_pct<0.20).sum():,} hours ({(bb_pct<0.20).mean():.1%})")
print(f"  ADX ranging (<20)          : {(adx_1h<20).sum():,} hours ({(adx_1h<20).mean():.1%})")
print(f"  RSI oversold (<30)         : {(rsi<30).sum():,} hours ({(rsi<30).mean():.1%})")
print(f"  RSI overbought (>70)       : {(rsi>70).sum():,} hours ({(rsi>70).mean():.1%})")
print(f"  Stoch bullish crossover    : {bullish_xover_1h.sum():,} hours ({bullish_xover_1h.mean():.1%})")
print(f"  EMA position (close>EMA20) : {(ema_pos>0).sum():,} hours ({(ema_pos>0).mean():.1%})")


# ── Build hourly dataset ───────────────────────────────────────────────────────
print("\nClassifying regimes and computing YES bet outcomes...")

rows = []
for i in range(50, n1h - 1):
    ts_now     = ts_1h[i]
    spot       = float(close_1h.iat[i])
    next_close = float(close_1h.iat[i + 1])

    if any(pd.isna(x) for x in [bb_pct.iat[i], adx_1h.iat[i], rsi.iat[i], stoch_k_1h.iat[i]]):
        continue

    bb_p   = float(bb_pct.iat[i])
    adx_v  = float(adx_1h.iat[i])
    rsi_v  = float(rsi.iat[i])
    sk_v   = float(stoch_k_1h.iat[i])
    xover  = bool(bullish_xover_1h.iat[i])
    ep_v   = float(ema_pos.iat[i])

    bb_squeeze   = bb_p  < 0.20
    adx_ranging  = adx_v < 20
    adx_trending = adx_v > 35
    rsi_oversold = rsi_v < 30
    rsi_overbought = rsi_v > 70
    price_above_ema = ep_v > 0

    # Primary regime (same as NO analysis — same market conditions)
    if rsi_oversold:
        regime = "rsi_oversold"
    elif bb_squeeze and adx_ranging:
        regime = "squeeze_ranging"
    elif bb_squeeze and not adx_ranging:
        regime = "squeeze_other"
    elif adx_ranging:
        regime = "ranging_only"
    elif adx_trending:
        regime = "trending"
    else:
        regime = "neutral"

    # Secondary indicator for bounce sub-analysis
    if xover:
        bounce_signal = "stoch_crossup"
    elif rsi_oversold:
        bounce_signal = "rsi_oversold"
    elif sk_v < 20:
        bounce_signal = "stoch_oversold"
    elif rsi_overbought or sk_v > 80:
        bounce_signal = "overbought"
    else:
        bounce_signal = "neutral"

    for (off_lo, off_hi), pk in KALSHI_PRICE_TABLE.items():
        offset  = (off_lo + off_hi) / 2
        K       = spot * (1.0 + offset)
        yes_won = int(next_close > K)
        fee     = KALSHI_RAKE * pk * (1 - pk)

        # P&L for buying YES at market price pk
        yes_pnl = FIXED_STAKE * (1 - pk) / pk * yes_won - FIXED_STAKE * (1 - yes_won)
        yes_pnl -= FIXED_STAKE * fee

        rows.append({
            "ts":            ts_now,
            "is_test":       ts_now >= TEST_START,
            "spot":          spot,
            "K":             K,
            "offset":        offset,
            "off_lo":        off_lo,
            "yes_won":       yes_won,
            "p_kalshi":      pk,
            "yes_pnl":       round(yes_pnl, 2),
            "regime":        regime,
            "bounce_signal": bounce_signal,
            "bb_pct":        round(bb_p, 3),
            "adx":           round(adx_v, 1),
            "rsi":           round(rsi_v, 1),
            "stoch_k":       round(sk_v, 1),
            "ema_above":     price_above_ema,
        })

df = pd.DataFrame(rows)
df_test = df[df["is_test"]].copy()

print(f"  Total observations (test): {len(df_test):,}  ({df_test['ts'].nunique():,} hours × {len(KALSHI_PRICE_TABLE)} offsets)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: YES win rates by regime and offset
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 1 — YES win rates by REGIME and offset")
print("  YES edge = yes_win_rate - p_kalshi (YES breakeven)")
print("  Note: YES bet is OTM — BTC must CLOSE ABOVE spot×(1+offset)")
print(SEP)

REGIMES_ORDERED = [
    ("squeeze_ranging",  "BB squeeze + ADX ranging   ← NO bet target regime"),
    ("squeeze_other",    "BB squeeze only"),
    ("ranging_only",     "ADX ranging only"),
    ("trending",         "ADX trending (>35)"),
    ("neutral",          "Neutral"),
    ("rsi_oversold",     "RSI oversold (<30)         ← bounce candidate?"),
]

for regime_key, regime_label in REGIMES_ORDERED:
    sub_r = df_test[df_test["regime"] == regime_key]
    n_hrs = sub_r["ts"].nunique()
    if n_hrs == 0:
        continue
    pct   = n_hrs / df_test["ts"].nunique()
    print(f"\n  {regime_label}  ({n_hrs:,} hours, {pct:.1%} of test)")
    print(f"  {'Offset':>12}  {'n':>6}  {'YES_win%':>9}  {'p_kalshi':>9}  {'raw_edge':>9}  "
          f"{'fee':>6}  {'net_edge':>9}  {'yes_pnl':>10}  verdict")
    print("  " + "-" * 92)

    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_r[sub_r["off_lo"] == off_lo]
        if len(sub_o) < MIN_N:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        raw_edge = yes_wr - pk
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = raw_edge - fee
        yes_pnl  = sub_o["yes_pnl"].sum()
        label    = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {label:>12}  {len(sub_o):>6,}  {yes_wr:>8.1%}  {pk:>8.3f}  "
              f"{raw_edge:>+8.1%}  {fee:>5.1%}  {net_edge:>+8.1%}  ${yes_pnl:>+9.0f}  {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Bounce signals — RSI oversold + stoch crossup
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 2 — Bounce signals: stoch crossup and RSI oversold → YES edge?")
print("  Hypothesis: oversold + crossup = bounce imminent → OTM YES bets win")
print(SEP)

BOUNCE_SIGNALS = [
    ("stoch_crossup",  "Stoch bullish crossup (from oversold)"),
    ("rsi_oversold",   "RSI oversold (<30)"),
    ("stoch_oversold", "Stoch K < 20 (no crossup yet)"),
    ("overbought",     "RSI overbought or Stoch K > 80"),
    ("neutral",        "Neutral"),
]

for sig_key, sig_label in BOUNCE_SIGNALS:
    sub_s = df_test[df_test["bounce_signal"] == sig_key]
    n_hrs = sub_s["ts"].nunique()
    if n_hrs < 5:
        continue
    print(f"\n  {sig_label}  ({n_hrs:,} hours)")
    print(f"  {'Offset':>12}  {'n':>6}  {'YES_win%':>9}  {'p_kalshi':>9}  {'net_edge':>9}  verdict")
    print("  " + "-" * 60)
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_s[sub_s["off_lo"] == off_lo]
        if len(sub_o) < MIN_N:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        raw_edge = yes_wr - pk
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = raw_edge - fee
        label    = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {label:>12}  {len(sub_o):>6,}  {yes_wr:>8.1%}  {pk:>8.3f}  {net_edge:>+8.1%}  {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: RSI oversold + stoch crossup combined (strongest bounce signal)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 3 — Combined bounce: RSI oversold (<30) AND stoch crossup")
print("  This is the bounce-gate from analysis_bounce_gate.py in regime context")
print(SEP)

# RSI oversold AND stoch bullish crossover — strongest bounce signal
df_test_rsi = df_test.copy()
df_test_rsi["combined_bounce"] = (
    (df_test_rsi["rsi"] < 30) & (df_test_rsi["bounce_signal"] == "stoch_crossup")
)

for label, mask in [
    ("RSI<30 + stoch crossup (combined bounce)", df_test["rsi"] < 30),
    ("RSI<30 only", df_test["rsi"] < 30),
    ("stoch crossup only (any RSI)", df_test["bounce_signal"] == "stoch_crossup"),
]:
    sub_s = df_test[mask]
    n_hrs = sub_s["ts"].nunique()
    if n_hrs < 5:
        print(f"\n  {label}: only {n_hrs} hours — too few\n")
        continue
    print(f"\n  {label}  ({n_hrs:,} hours)")
    print(f"  {'Offset':>12}  {'n':>5}  {'YES_win%':>9}  {'p_kalshi':>9}  {'net_edge':>9}  verdict")
    print("  " + "-" * 58)
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_s[sub_s["off_lo"] == off_lo]
        if len(sub_o) < 10:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        raw_edge = yes_wr - pk
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = raw_edge - fee
        lbl      = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {lbl:>12}  {len(sub_o):>5,}  {yes_wr:>8.1%}  {pk:>8.3f}  {net_edge:>+8.1%}  {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Head-to-head — NO vs YES by regime, best offset each
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 4 — Head-to-head: best YES net edge vs best NO net edge by regime")
print("  Which side has edge in each regime? This drives the trading decision.")
print(SEP)

print(f"\n  {'Regime':>30}  {'hrs':>5}  {'best NO net':>12}  {'NO offset':>10}  "
      f"{'best YES net':>13}  {'YES offset':>11}  {'take'}")
print("  " + "-" * 100)

for regime_key, regime_label in REGIMES_ORDERED:
    sub_r = df_test[df_test["regime"] == regime_key]
    n_hrs = sub_r["ts"].nunique()
    if n_hrs < 5:
        continue

    best_no_edge  = -999; best_no_label  = "—"
    best_yes_edge = -999; best_yes_label = "—"

    for (off_lo, off_hi), pk in KALSHI_PRICE_TABLE.items():
        sub_o = sub_r[sub_r["off_lo"] == off_lo]
        if len(sub_o) < MIN_N:
            continue
        fee = KALSHI_RAKE * pk * (1 - pk)
        lbl = OFFSET_LABELS[(off_lo, off_hi)]

        # NO edge: need BTC to stay below K
        no_wr    = 1 - sub_o["yes_won"].mean()
        no_be    = 1 - pk
        no_edge  = (no_wr - no_be) - fee
        if no_edge > best_no_edge:
            best_no_edge  = no_edge
            best_no_label = lbl

        # YES edge: need BTC to close above K
        yes_wr   = sub_o["yes_won"].mean()
        yes_edge = (yes_wr - pk) - fee
        if yes_edge > best_yes_edge:
            best_yes_edge  = yes_edge
            best_yes_label = lbl

    # Decision: which side to take?
    if best_no_edge > 0.04 and best_no_edge > best_yes_edge:
        take = "NO  ★"
    elif best_yes_edge > 0.04 and best_yes_edge > best_no_edge:
        take = "YES ★"
    elif best_no_edge > 0 and best_no_edge > best_yes_edge:
        take = "NO"
    elif best_yes_edge > 0 and best_yes_edge > best_no_edge:
        take = "YES"
    else:
        take = "skip"

    no_s  = f"{best_no_edge:+.1%}" if best_no_edge > -999 else "—"
    yes_s = f"{best_yes_edge:+.1%}" if best_yes_edge > -999 else "—"
    short_label = regime_label.split("←")[0].strip()
    print(f"  {short_label:>30}  {n_hrs:>5,}  {no_s:>12}  {best_no_label:>10}  "
          f"{yes_s:>13}  {best_yes_label:>11}  {take}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Baseline — all hours, no regime filter
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 5 — Baseline: all test hours (no regime filter)")
print("  Confirms the unconditional YES vs NO edge landscape")
print(SEP)

print(f"\n  {'Offset':>12}  {'n':>6}  {'NO_win%':>9}  {'NO_edge':>8}  {'YES_win%':>9}  {'YES_edge':>9}  {'better'}")
print("  " + "-" * 80)

for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
    sub_o = df_test[df_test["off_lo"] == off_lo]
    if len(sub_o) < MIN_N:
        continue
    fee      = KALSHI_RAKE * pk * (1 - pk)
    yes_wr   = sub_o["yes_won"].mean()
    no_wr    = 1 - yes_wr
    yes_edge = (yes_wr - pk)       - fee
    no_edge  = (no_wr  - (1 - pk)) - fee
    label    = OFFSET_LABELS[(off_lo, off_hi)]
    better   = "NO ★" if no_edge > yes_edge and no_edge > 0 else ("YES ★" if yes_edge > no_edge and yes_edge > 0 else "—")
    print(f"  {label:>12}  {len(sub_o):>6,}  {no_wr:>8.1%}  {no_edge:>+7.1%}  "
          f"{yes_wr:>8.1%}  {yes_edge:>+8.1%}  {better}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EMA direction filter — does price being above/below EMA affect YES edge?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 6 — EMA direction: does price above/below 20h EMA predict YES win?")
print("  Hypothesis: bullish trend (above EMA) → YES more likely to win")
print(SEP)

for ema_label, ema_mask in [
    ("Price ABOVE 20h EMA (bullish)", df_test["ema_above"] == True),
    ("Price BELOW 20h EMA (bearish)", df_test["ema_above"] == False),
]:
    sub_e = df_test[ema_mask]
    n_hrs = sub_e["ts"].nunique()
    print(f"\n  {ema_label}  ({n_hrs:,} hours)")
    print(f"  {'Offset':>12}  {'n':>6}  {'YES_win%':>9}  {'YES_net_edge':>13}  verdict")
    print("  " + "-" * 55)
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_e[sub_e["off_lo"] == off_lo]
        if len(sub_o) < MIN_N:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = (yes_wr - pk) - fee
        label    = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {label:>12}  {len(sub_o):>6,}  {yes_wr:>8.1%}  {net_edge:>+12.1%}  {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Trending + bullish EMA — should we bet YES in uptrends?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 7 — Trending regime + bullish EMA: momentum YES bets")
print("  ADX>35 + price above EMA = strong uptrend → OTM YES should win more often")
print(SEP)

combos = [
    ("Trending + above EMA (strong uptrend)",  (df_test["regime"] == "trending") & (df_test["ema_above"] == True)),
    ("Trending + below EMA (strong downtrend)", (df_test["regime"] == "trending") & (df_test["ema_above"] == False)),
    ("Neutral + above EMA",                    (df_test["regime"] == "neutral")  & (df_test["ema_above"] == True)),
    ("Neutral + below EMA",                    (df_test["regime"] == "neutral")  & (df_test["ema_above"] == False)),
]

for combo_label, combo_mask in combos:
    sub_c = df_test[combo_mask]
    n_hrs = sub_c["ts"].nunique()
    if n_hrs < 10:
        print(f"\n  {combo_label}: {n_hrs} hours — too few to report")
        continue
    print(f"\n  {combo_label}  ({n_hrs:,} hours)")
    print(f"  {'Offset':>12}  {'n':>5}  {'YES_win%':>9}  {'YES_net_edge':>13}  verdict")
    print("  " + "-" * 52)
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_c[sub_c["off_lo"] == off_lo]
        if len(sub_o) < 15:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = (yes_wr - pk) - fee
        label    = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {label:>12}  {len(sub_o):>5,}  {yes_wr:>8.1%}  {net_edge:>+12.1%}  {v}")

print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print("""
Key questions answered:

  1. Do YES bets have structural edge?
     → Check Section 5 baseline (all hours, no filter).
        If YES win rates consistently exceed p_kalshi, YES has base-rate edge.

  2. Does BB squeeze + ADX ranging help YES bets?
     → Check Section 1, squeeze_ranging row.
        If YES net_edge < 0: squeeze suppresses upward moves → stick with NO.
        If YES net_edge > 0: rare scenario where squeeze benefits both sides.

  3. Does RSI oversold trigger a bounce that benefits YES?
     → Check Section 2 (rsi_oversold row) and Section 3.
        If YES net_edge > 4% under RSI<30: bounce gate is viable for YES.

  4. Does EMA/trend direction give YES edge?
     → Check Sections 6–7. Strong uptrend (ADX>35, above EMA) is the
        best scenario for OTM YES — price momentum continues upward.

  5. Head-to-head per regime?
     → Section 4 directly answers: which side to trade in each regime.
""")
