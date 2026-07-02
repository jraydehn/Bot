"""
analysis_stoch15m_yes.py — 15m stochastic crossover timing and YES bet edge.

The prior backtest showed:
  - Stoch crossup (any RSI) → LOSING for YES bets
  - RSI < 30 alone → +8-9% net YES edge (bounce effect)

Key question: Does the RECENCY and DEPTH of the 15m stochastic crossover
refine the RSI oversold bounce signal? Specifically:

  - Fresh crossover (fired in the last 1-2 bars) vs fading signal
  - Crossover from very deep oversold (K_prev < 10) vs moderate (10-20)
  - Crossover with K still below 30 vs K already recovering to 30-50
  - Does crossup WITHOUT RSI oversold have any sub-segment with YES edge?
  - Do 15m crossovers in the trending or squeeze regime flip YES profitable?

Structure:
  For each hour in test set, we know:
    - Which 15m bar the most recent crossover occurred on (bars ago: 0, 1, 2, 3+)
    - How oversold K was when it crossed (depth)
    - Current K value at hour open (recovery progress)
    - RSI regime at hour open
  We then report YES win rate at each offset bucket.
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
MIN_N       = 30

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
print("Loading data...")
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
n1h      = len(close_1h)
ts_1h    = ohlcv_1h.index


# ── RSI-14 on 1h ──────────────────────────────────────────────────────────────
delta = close_1h.diff()
gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi_1h = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

# ── BB percentile on 1h ───────────────────────────────────────────────────────
bb_std   = close_1h.rolling(20).std()
bb_mid   = close_1h.rolling(20).mean()
bb_width = (4 * bb_std) / bb_mid
bb_pct   = bb_width.rolling(500).rank(pct=True)

# ── ADX-14 on 15m → resampled to 1h ──────────────────────────────────────────
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

# ── 15m Stochastic K/D ───────────────────────────────────────────────────────
print("Computing 15m stochastic with crossover timing...")

STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

ll_15m  = df_15m["low"].rolling(STOCH_K_PERIOD).min()
hh_15m  = df_15m["high"].rolling(STOCH_K_PERIOD).max()
hl_rng  = (hh_15m - ll_15m).replace(0, float("nan"))
sk_15m  = ((df_15m["close"] - ll_15m) / hl_rng) * 100
sd_15m  = sk_15m.rolling(STOCH_D_PERIOD).mean()

# Crossover: K crosses above D from oversold
sk_prev = sk_15m.shift(1)
sd_prev = sd_15m.shift(1)

# Standard crossup from oversold (K_prev < D_prev, K_now > D_now, K_prev < 20)
xover_15m = (
    (sk_prev < sd_prev) &
    (sk_15m  > sd_15m)  &
    (sk_prev < 20)
)

# How oversold was K at the crossover? (capture K_prev at crossover bar)
# We'll tag each 15m bar with: bars_since_last_xover, k_at_xover, k_now
# Build a lookup: for each 15m bar, when did the last crossover happen?
xover_idx = np.where(xover_15m.values)[0]  # integer positions of crossover bars

# For each 15m bar, find how many bars ago the last crossover was
# and the K value at that crossover
bars_since_xover_arr = np.full(len(df_15m), 999, dtype=float)
k_at_xover_arr       = np.full(len(df_15m), float("nan"))
sk_arr = sk_15m.values

for xi in xover_idx:
    # Mark bars after this crossover until the next one
    end = xover_idx[xover_idx > xi].min() if any(xover_idx > xi) else len(df_15m)
    for j in range(xi, min(int(end), len(df_15m))):
        bars_since_xover_arr[j] = j - xi
        k_at_xover_arr[j]       = float(sk_prev.iat[xi]) if xi < len(sk_prev) else float("nan")

bars_since_xover_15m = pd.Series(bars_since_xover_arr, index=df_15m.index)
k_at_xover_15m       = pd.Series(k_at_xover_arr,       index=df_15m.index)

# ── Resample 15m stoch features to 1h (take last 15m bar in each hour) ────────
# At hour open we use the most recent completed 15m bar
sk_at_hour    = sk_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
sd_at_hour    = sd_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
bsx_at_hour   = bars_since_xover_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")
kax_at_hour   = k_at_xover_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

print(f"  15m bars: {len(df_15m):,}")
print(f"  15m crossover events: {xover_15m.sum():,}")
print(f"  1h bars: {n1h:,}")


# ── Build hourly dataset ───────────────────────────────────────────────────────
print("Building hourly YES bet dataset...")

rows = []
for i in range(50, n1h - 1):
    ts_now     = ts_1h[i]
    spot       = float(close_1h.iat[i])
    next_close = float(close_1h.iat[i + 1])

    if any(pd.isna(x) for x in [rsi_1h.iat[i], bb_pct.iat[i], adx_1h.iat[i],
                                  sk_at_hour.iat[i], bsx_at_hour.iat[i]]):
        continue

    rsi_v   = float(rsi_1h.iat[i])
    bb_p    = float(bb_pct.iat[i])
    adx_v   = float(adx_1h.iat[i])
    sk_v    = float(sk_at_hour.iat[i])
    sd_v    = float(sd_at_hour.iat[i])
    bsx_v   = float(bsx_at_hour.iat[i])
    kax_v   = float(kax_at_hour.iat[i]) if not pd.isna(kax_at_hour.iat[i]) else float("nan")

    rsi_oversold  = rsi_v < 30
    bb_squeeze    = bb_p  < 0.20
    adx_ranging   = adx_v < 20
    adx_trending  = adx_v > 35

    # Crossover recency bucket
    if bsx_v <= 1:
        xover_recency = "fresh_0-1bar"    # fired in last 15-30m
    elif bsx_v <= 3:
        xover_recency = "recent_2-3bar"   # fired 30-60m ago
    elif bsx_v <= 7:
        xover_recency = "fading_4-7bar"   # fired 1-2h ago
    else:
        xover_recency = "stale_8+bar"     # no recent crossover

    # Depth of oversold at crossover
    if math.isnan(kax_v) or bsx_v >= 8:
        xover_depth = "none"
    elif kax_v < 10:
        xover_depth = "very_deep_<10"
    elif kax_v < 15:
        xover_depth = "deep_10-15"
    else:
        xover_depth = "moderate_15-20"

    # Current K recovery level
    if sk_v < 20:
        k_zone = "still_oversold"
    elif sk_v < 30:
        k_zone = "recovering_20-30"
    elif sk_v < 50:
        k_zone = "mid_30-50"
    else:
        k_zone = "upper_50+"

    # Primary regime
    if rsi_oversold:
        regime = "rsi_oversold"
    elif bb_squeeze and adx_ranging:
        regime = "squeeze_ranging"
    elif adx_trending:
        regime = "trending"
    elif adx_ranging:
        regime = "ranging"
    else:
        regime = "neutral"

    for (off_lo, off_hi), pk in KALSHI_PRICE_TABLE.items():
        offset  = (off_lo + off_hi) / 2
        K       = spot * (1.0 + offset)
        yes_won = int(next_close > K)
        fee     = KALSHI_RAKE * pk * (1 - pk)
        yes_pnl = FIXED_STAKE * (1 - pk) / pk * yes_won - FIXED_STAKE * (1 - yes_won)
        yes_pnl -= FIXED_STAKE * fee

        rows.append({
            "ts":            ts_now,
            "is_test":       ts_now >= TEST_START,
            "spot":          spot,
            "off_lo":        off_lo,
            "yes_won":       yes_won,
            "p_kalshi":      pk,
            "yes_pnl":       round(yes_pnl, 2),
            "regime":        regime,
            "rsi_oversold":  rsi_oversold,
            "xover_recency": xover_recency,
            "xover_depth":   xover_depth,
            "k_zone":        k_zone,
            "sk_v":          round(sk_v, 1),
            "sd_v":          round(sd_v, 1),
            "bsx_v":         bsx_v,
            "kax_v":         round(kax_v, 1) if not math.isnan(kax_v) else float("nan"),
            "rsi_v":         round(rsi_v, 1),
            "adx_v":         round(adx_v, 1),
            "bb_pct":        round(bb_p, 3),
        })

df = pd.DataFrame(rows)
df_test = df[df["is_test"]].copy()
n_hrs_total = df_test["ts"].nunique()
print(f"  Test observations: {len(df_test):,}  ({n_hrs_total:,} hours × {len(KALSHI_PRICE_TABLE)} offsets)\n")


# ── Helper: print YES results for a subset ───────────────────────────────────
def print_yes_results(sub, label, min_n=MIN_N):
    n_hrs = sub["ts"].nunique()
    if n_hrs < 5:
        print(f"  {label}: {n_hrs} hours — too few\n")
        return
    print(f"\n  {label}  ({n_hrs:,} hours, {n_hrs/n_hrs_total:.1%} of test)")
    print(f"  {'Offset':>12}  {'n':>5}  {'YES_win%':>9}  {'p_kalshi':>9}  {'net_edge':>9}  verdict")
    print("  " + "-" * 58)
    found_edge = False
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub[sub["off_lo"] == off_lo]
        if len(sub_o) < min_n:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = (yes_wr - pk) - fee
        lbl      = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        if "EDGE" in v or "edge" in v:
            found_edge = True
        print(f"  {lbl:>12}  {len(sub_o):>5,}  {yes_wr:>8.1%}  {pk:>8.3f}  {net_edge:>+8.1%}  {v}")
    if not found_edge:
        print("    → No positive net edge at any offset")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Crossover recency — does freshness matter?
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("SECTION 1 — Crossover recency: does a FRESH 15m crossup give YES edge?")
print("  (All RSI levels — to test if crossup alone works when timed precisely)")
print(SEP)

for recency in ["fresh_0-1bar", "recent_2-3bar", "fading_4-7bar", "stale_8+bar"]:
    labels = {
        "fresh_0-1bar":  "Fresh crossup  (fired 0-1 bars ago, ≤30 min)",
        "recent_2-3bar": "Recent crossup (fired 2-3 bars ago, 30-60 min)",
        "fading_4-7bar": "Fading crossup (fired 4-7 bars ago, 1-2h)",
        "stale_8+bar":   "No crossup     (8+ bars, >2h since last)",
    }[recency]
    sub = df_test[df_test["xover_recency"] == recency]
    print_yes_results(sub, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Fresh crossup + RSI oversold — the refined bounce signal
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 2 — Fresh crossup + RSI oversold: refined bounce gate")
print("  Combining timing precision with confirmed oversold condition")
print(SEP)

combos = [
    ("Fresh (≤1 bar) + RSI<30",    (df_test["xover_recency"] == "fresh_0-1bar")  & df_test["rsi_oversold"]),
    ("Recent (2-3 bar) + RSI<30",  (df_test["xover_recency"] == "recent_2-3bar") & df_test["rsi_oversold"]),
    ("Fresh (≤1 bar) + RSI 30-45", (df_test["xover_recency"] == "fresh_0-1bar")  & (df_test["rsi_v"] >= 30) & (df_test["rsi_v"] < 45)),
    ("Fresh (≤1 bar) + RSI 45+",   (df_test["xover_recency"] == "fresh_0-1bar")  & (df_test["rsi_v"] >= 45)),
    ("RSI<30 (any crossup timing)", df_test["rsi_oversold"]),
]

for label, mask in combos:
    print_yes_results(df_test[mask], label, min_n=10)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Oversold depth at crossover — does deeper = stronger bounce?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 3 — Oversold depth at crossover: very deep vs moderate")
print("  Hypothesis: K_prev < 10 at crossover → extreme oversold → stronger bounce")
print(SEP)

for depth in ["very_deep_<10", "deep_10-15", "moderate_15-20"]:
    labels = {
        "very_deep_<10":   "Very deep oversold at crossup (K_prev < 10)",
        "deep_10-15":      "Deep oversold at crossup      (K_prev 10-15)",
        "moderate_15-20":  "Moderate oversold at crossup  (K_prev 15-20)",
    }[depth]
    sub = df_test[df_test["xover_depth"] == depth]
    print_yes_results(sub, labels, min_n=15)

# With RSI filter added
print("\n  — Adding RSI<30 filter to depth buckets —")
for depth in ["very_deep_<10", "deep_10-15", "moderate_15-20"]:
    label = f"Depth {depth} + RSI<30"
    sub = df_test[(df_test["xover_depth"] == depth) & df_test["rsi_oversold"]]
    print_yes_results(sub, label, min_n=8)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: K recovery level at hour open
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 4 — K zone at hour open: where is stoch K when we place the bet?")
print("  still_oversold (<20): bounce may not have started yet")
print("  recovering (20-30)  : bounce underway — best YES entry?")
print("  mid (30-50)         : already bounced — too late?")
print(SEP)

for zone in ["still_oversold", "recovering_20-30", "mid_30-50", "upper_50+"]:
    labels = {
        "still_oversold":  "K still oversold (<20) at hour open",
        "recovering_20-30":"K recovering 20-30 at hour open",
        "mid_30-50":       "K in mid range 30-50 at hour open",
        "upper_50+":       "K upper 50+ at hour open",
    }[zone]
    sub = df_test[df_test["k_zone"] == zone]
    print_yes_results(sub, labels)

# K zone with fresh crossover
print("\n  — Fresh crossup (≤1 bar) by K zone —")
for zone in ["still_oversold", "recovering_20-30", "mid_30-50"]:
    label = f"Fresh xover + K zone {zone}"
    sub = df_test[(df_test["xover_recency"] == "fresh_0-1bar") & (df_test["k_zone"] == zone)]
    print_yes_results(sub, label, min_n=10)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Crossup in squeeze_ranging regime — does it predict direction?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 5 — Crossup inside squeeze regime: breakout direction predictor?")
print("  Squeeze + crossup from oversold may predict bullish breakout direction")
print(SEP)

combos_sq = [
    ("Squeeze+ranging + fresh crossup",   (df_test["regime"] == "squeeze_ranging") & (df_test["xover_recency"] == "fresh_0-1bar")),
    ("Squeeze+ranging + any crossup",     (df_test["regime"] == "squeeze_ranging") & (df_test["xover_recency"].isin(["fresh_0-1bar", "recent_2-3bar"]))),
    ("Squeeze+ranging, no crossup (stale)",(df_test["regime"] == "squeeze_ranging") & (df_test["xover_recency"] == "stale_8+bar")),
]

for label, mask in combos_sq:
    print_yes_results(df_test[mask], label, min_n=10)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Full signal matrix — recency × RSI × k_zone
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 6 — Best combinations: recency × RSI level (summary table)")
print("  Reports best net YES edge across all offsets for each combo")
print(SEP)

print(f"\n  {'Signal combo':>45}  {'n_hrs':>6}  {'best_net_edge':>14}  {'at_offset':>10}")
print("  " + "-" * 85)

combos_full = [
    ("RSI<30 + fresh xover (0-1 bar)",   df_test["rsi_oversold"] & (df_test["xover_recency"] == "fresh_0-1bar")),
    ("RSI<30 + recent xover (2-3 bar)",  df_test["rsi_oversold"] & (df_test["xover_recency"] == "recent_2-3bar")),
    ("RSI<30 + fading xover (4-7 bar)",  df_test["rsi_oversold"] & (df_test["xover_recency"] == "fading_4-7bar")),
    ("RSI<30 + stale (no xover)",        df_test["rsi_oversold"] & (df_test["xover_recency"] == "stale_8+bar")),
    ("RSI 30-45 + fresh xover",          (~df_test["rsi_oversold"]) & (df_test["rsi_v"] < 45) & (df_test["xover_recency"] == "fresh_0-1bar")),
    ("RSI 30-45 + recent xover",         (~df_test["rsi_oversold"]) & (df_test["rsi_v"] < 45) & (df_test["xover_recency"] == "recent_2-3bar")),
    ("RSI 45-60 + fresh xover",          (df_test["rsi_v"] >= 45) & (df_test["rsi_v"] < 60) & (df_test["xover_recency"] == "fresh_0-1bar")),
    ("RSI<30 + K recovering (20-30)",    df_test["rsi_oversold"] & (df_test["k_zone"] == "recovering_20-30")),
    ("RSI<30 + K still oversold (<20)",  df_test["rsi_oversold"] & (df_test["k_zone"] == "still_oversold")),
    ("Fresh xover + very deep (<10)",    (df_test["xover_recency"] == "fresh_0-1bar") & (df_test["xover_depth"] == "very_deep_<10")),
    ("Fresh xover + deep (10-15)",       (df_test["xover_recency"] == "fresh_0-1bar") & (df_test["xover_depth"] == "deep_10-15")),
    ("Squeeze + fresh xover",            (df_test["regime"] == "squeeze_ranging") & (df_test["xover_recency"] == "fresh_0-1bar")),
    ("RSI<30 baseline (all)",            df_test["rsi_oversold"]),
    ("Fresh xover baseline (all RSI)",   df_test["xover_recency"] == "fresh_0-1bar"),
]

for combo_label, mask in combos_full:
    sub = df_test[mask]
    n_hrs = sub["ts"].nunique()
    best_edge = -999
    best_off_label = "—"
    for (off_lo, off_hi), pk in KALSHI_PRICE_TABLE.items():
        sub_o = sub[sub["off_lo"] == off_lo]
        if len(sub_o) < 10:
            continue
        yes_wr   = sub_o["yes_won"].mean()
        fee      = KALSHI_RAKE * pk * (1 - pk)
        net_edge = (yes_wr - pk) - fee
        if net_edge > best_edge:
            best_edge      = net_edge
            best_off_label = OFFSET_LABELS[(off_lo, off_hi)]
    edge_s = f"{best_edge:+.1%}" if best_edge > -999 else "—"
    verdict = "★" if best_edge > 0.04 else ("+" if best_edge > 0 else "✗")
    print(f"  {combo_label:>45}  {n_hrs:>6,}  {edge_s:>14}  {best_off_label:>10}  {verdict}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: YES vs NO head-to-head when RSI<30 + fresh crossup
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 7 — When RSI<30 + fresh crossup fires: YES vs NO comparison")
print("  The bounce signal — should we bet YES or NO?")
print(SEP)

mask_bounce = df_test["rsi_oversold"] & (df_test["xover_recency"].isin(["fresh_0-1bar", "recent_2-3bar"]))
sub_b = df_test[mask_bounce]
n_hrs_b = sub_b["ts"].nunique()

print(f"\n  RSI<30 + crossup (fresh or recent): {n_hrs_b} hours")
print(f"\n  {'Offset':>12}  {'NO_win%':>9}  {'NO_edge':>8}  {'YES_win%':>9}  {'YES_edge':>9}  {'take'}")
print("  " + "-" * 65)

for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
    sub_o = sub_b[sub_b["off_lo"] == off_lo]
    if len(sub_o) < 10:
        continue
    fee      = KALSHI_RAKE * pk * (1 - pk)
    yes_wr   = sub_o["yes_won"].mean()
    no_wr    = 1 - yes_wr
    yes_edge = (yes_wr - pk)       - fee
    no_edge  = (no_wr  - (1 - pk)) - fee
    lbl      = OFFSET_LABELS[(off_lo, off_hi)]
    take = ("YES ★" if yes_edge > no_edge and yes_edge > 0 else
            "NO ★"  if no_edge  > yes_edge and no_edge > 0  else "—")
    print(f"  {lbl:>12}  {no_wr:>8.1%}  {no_edge:>+7.1%}  {yes_wr:>8.1%}  {yes_edge:>+8.1%}  {take}")

print(f"\n{SEP}")
print("INTERPRETATION GUIDE")
print(SEP)
print("""
What to look for:
  - Section 1: Does fresh (0-1 bar) crossup outperform stale? If yes,
    timing the crossup precisely matters — use 15m resolution in live model.

  - Section 2: Does RSI<30 + fresh crossup beat RSI<30 alone?
    If similar, RSI is doing all the work. If better, crossup timing adds signal.

  - Section 3: Depth of oversold — very deep (K<10) vs moderate (K 15-20).
    Extreme oversold often produces stronger mean reversion.

  - Section 4: K zone — is there a sweet spot where YES wins most?
    "Recovering" (20-30) is the prime bounce window theoretically.

  - Section 6: Summary matrix — best net edge across all combos.
    Any combo beating RSI<30 baseline (+8-9%) is a genuine refinement.

  - Section 7: Direct YES vs NO comparison at the bounce signal.
    Confirms which side to take when the bounce gate fires.
""")
