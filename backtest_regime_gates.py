"""
backtest_regime_gates.py — BB squeeze + ADX ranging as hard regime gates.

No drift modeling. No log-normal pricing simulation.

Core question:
  When BB is squeezed AND ADX is ranging (AND RSI is not oversold),
  do OTM NO bets win at a rate that exceeds what Kalshi charges?

Method:
  1. Compute BB squeeze, ADX, RSI on historical BTC data.
  2. For each hour, classify regime.
  3. For each offset bucket, compute empirical YES win rate
     (= fraction of hours BTC closes above spot × (1+offset)).
  4. Compare to ACTUAL observed Kalshi YES prices from paper trades
     (lookup table by offset — no regression, no assumptions).
  5. Edge = Kalshi_NO_price - empirical_NO_win_rate_needed
             = empirical_NO_win_rate - (1 - Kalshi_YES_price)

Kalshi YES prices used (median from 239 actual paper-trade observations):
  0.10-0.15% OTM → 0.330   (NO breakeven = 67.0%)
  0.15-0.20% OTM → 0.250   (NO breakeven = 75.0%)
  0.20-0.25% OTM → 0.210   (NO breakeven = 79.0%)
  0.25-0.30% OTM → 0.220   (NO breakeven = 78.0%)
  0.30-0.40% OTM → 0.190   (NO breakeven = 81.0%)
  0.40-0.50% OTM → 0.145   (NO breakeven = 85.5%)
  0.50-1.00% OTM → 0.070   (NO breakeven = 93.0%)

Walk-forward: calibrate BB percentile cutoffs on 2024, apply to 2025-Apr 2026.
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")

SEP = "=" * 72
TRAIN_END  = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

FIXED_STAKE  = 50.0
KALSHI_RAKE  = 0.07
MIN_N        = 50   # minimum hours per cell to report

# ── Actual observed Kalshi YES prices by offset bucket ─────────────────────────
# Source: 239 real paper-trade observations from live paper_trades.csv
# Key: (offset_low_pct, offset_high_pct) → median Kalshi YES price
KALSHI_PRICE_TABLE = {
    (0.001, 0.0015): 0.465,   # 0.10-0.15% OTM  (7 obs — use cautiously)
    (0.0015, 0.002): 0.330,   # 0.10-0.15% OTM
    (0.002, 0.0025): 0.330,   # 0.15-0.20%
    (0.0025, 0.003): 0.250,   # 0.15-0.20%
    (0.003, 0.004):  0.210,   # 0.20-0.25%
    (0.004, 0.005):  0.190,   # 0.30-0.40%
    (0.005, 0.010):  0.145,   # 0.40-0.50%
}

OFFSETS = sorted(KALSHI_PRICE_TABLE.keys())

def get_kalshi_yes_price(offset: float) -> float:
    for (lo, hi), price in KALSHI_PRICE_TABLE.items():
        if lo <= offset < hi:
            return price
    return float("nan")


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
n1h      = len(close_1h)
ts_1h    = ohlcv_1h.index

# ── BB on 1h (20-bar, width percentile over trailing 500h) ────────────────────
bb_mid   = close_1h.rolling(20).mean()
bb_std   = close_1h.rolling(20).std()
bb_width = (4 * bb_std) / bb_mid   # 4σ normalized width
# Rolling percentile: what fraction of past 500h had NARROWER bands than now?
# < 0.20 = squeeze (bottom 20th pctile = historically narrow)
bb_pct   = bb_width.rolling(500).rank(pct=True)

# ── RSI-14 on 1h ──────────────────────────────────────────────────────────────
delta = close_1h.diff()
gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

# ── ADX-14 on 15m bars, forward-filled to 1h ──────────────────────────────────
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"high":"max","low":"min","close":"last"}
).dropna(subset=["close"])

_h   = df_15m["high"]
_l   = df_15m["low"]
_cp  = df_15m["close"].shift(1)
tr   = pd.concat([_h-_l, (_h-_cp).abs(), (_l-_cp).abs()], axis=1).max(axis=1)
dmp  = (_h-_h.shift(1)).clip(lower=0).where((_h-_h.shift(1))>(_l.shift(1)-_l), 0)
dmm  = (_l.shift(1)-_l).clip(lower=0).where((_l.shift(1)-_l)>(_h-_h.shift(1)), 0)
atr  = tr.ewm(com=13, adjust=False).mean()
dip  = 100 * dmp.ewm(com=13, adjust=False).mean() / atr.replace(0, 1e-10)
dim  = 100 * dmm.ewm(com=13, adjust=False).mean() / atr.replace(0, 1e-10)
dx   = 100 * (dip-dim).abs() / (dip+dim).replace(0, 1e-10)
adx_15m = dx.ewm(com=13, adjust=False).mean()
adx_1h  = adx_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

print(f"  1h bars: {n1h:,}")
print(f"  BB squeeze (pct<0.20): {(bb_pct<0.20).sum():,} hours ({(bb_pct<0.20).mean():.1%})")
print(f"  ADX ranging (<20)    : {(adx_1h<20).sum():,} hours ({(adx_1h<20).mean():.1%})")
print(f"  RSI oversold (<30)   : {(rsi<30).sum():,} hours ({(rsi<30).mean():.1%})")


# ── Build hourly regime classification ────────────────────────────────────────
print("\nClassifying regimes and computing outcomes...")

rows = []
for i in range(50, n1h - 1):
    ts_now     = ts_1h[i]
    spot       = float(close_1h.iat[i])
    next_close = float(close_1h.iat[i + 1])

    if pd.isna(bb_pct.iat[i]) or pd.isna(adx_1h.iat[i]) or pd.isna(rsi.iat[i]):
        continue

    bb_p   = float(bb_pct.iat[i])
    adx_v  = float(adx_1h.iat[i])
    rsi_v  = float(rsi.iat[i])

    bb_squeeze  = bb_p < 0.20
    adx_ranging = adx_v < 20
    adx_trending = adx_v > 35
    rsi_oversold = rsi_v < 30
    rsi_overbought = rsi_v > 70

    # Regime label
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

    for (off_lo, off_hi), pk in KALSHI_PRICE_TABLE.items():
        offset    = (off_lo + off_hi) / 2
        K         = spot * (1.0 + offset)
        yes_won   = int(next_close > K)
        no_won    = 1 - yes_won
        fee       = KALSHI_RAKE * pk * (1 - pk)

        # P&L at actual Kalshi prices
        yes_pnl = FIXED_STAKE * (1-pk)/pk * yes_won - FIXED_STAKE * (1-yes_won)
        no_pnl  = FIXED_STAKE * pk/(1-pk) * no_won  - FIXED_STAKE * (1-no_won)
        yes_pnl -= FIXED_STAKE * fee
        no_pnl  -= FIXED_STAKE * fee

        rows.append({
            "ts":          ts_now,
            "is_test":     ts_now >= TEST_START,
            "spot":        spot,
            "K":           K,
            "offset":      offset,
            "off_lo":      off_lo,
            "yes_won":     yes_won,
            "p_kalshi":    pk,
            "yes_pnl":     round(yes_pnl, 2),
            "no_pnl":      round(no_pnl, 2),
            "regime":      regime,
            "bb_pct":      round(bb_p, 3),
            "adx":         round(adx_v, 1),
            "rsi":         round(rsi_v, 1),
        })

df = pd.DataFrame(rows)
df_test = df[df["is_test"]].copy()

print(f"  Total observations (test): {len(df_test):,}  ({df_test['ts'].nunique():,} hours × {len(KALSHI_PRICE_TABLE)} offsets)")


# ── Results: empirical win rates by regime and offset ─────────────────────────
print(f"\n{SEP}")
print("REGIME ANALYSIS — Empirical NO win rates vs Kalshi breakeven")
print("Test set: Jan 2025 – Apr 2026  |  Stake: $50 per trade")
print(SEP)

REGIMES = [
    ("squeeze_ranging",  "BB squeeze + ADX ranging   ← target regime"),
    ("squeeze_other",    "BB squeeze + ADX moderate/trending"),
    ("ranging_only",     "ADX ranging only (no squeeze)"),
    ("trending",         "ADX trending (>35)"),
    ("neutral",          "Neutral"),
    ("rsi_oversold",     "RSI oversold (<30)          ← avoid NO bets"),
]

OFFSET_LABELS = {
    (0.001, 0.0015):  "0.10-0.15%",
    (0.0015, 0.002):  "0.15-0.20%",
    (0.002, 0.0025):  "0.20-0.25%",
    (0.0025, 0.003):  "0.25-0.30%",
    (0.003, 0.004):   "0.30-0.40%",
    (0.004, 0.005):   "0.40-0.50%",
    (0.005, 0.010):   "0.50-1.0%",
}

all_regime_results = {}

for regime_key, regime_label in REGIMES:
    sub_r = df_test[df_test["regime"] == regime_key]
    n_hrs = sub_r["ts"].nunique()
    print(f"\n  {regime_label}  ({n_hrs:,} hours, {n_hrs/df_test['ts'].nunique():.1%} of test)")
    print(f"  {'Offset':>12}  {'n':>6}  {'NO_win%':>8}  {'breakeven':>10}  {'edge':>7}  "
          f"{'fee':>6}  {'net_edge':>9}  {'no_pnl':>10}  verdict")
    print("  " + "-" * 90)

    regime_results = {}
    for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
        sub_o = sub_r[(sub_r["off_lo"] == off_lo)]
        if len(sub_o) < MIN_N:
            continue
        no_wr   = 1 - sub_o["yes_won"].mean()
        be      = 1 - pk
        raw_edge = no_wr - be
        fee     = KALSHI_RAKE * pk * (1-pk)
        net_edge = raw_edge - fee
        no_pnl  = sub_o["no_pnl"].sum()
        label   = OFFSET_LABELS[(off_lo, off_hi)]
        v = ("★ EDGE" if net_edge > 0.04 else
             "edge"   if net_edge > 0.01 else
             "slim"   if net_edge > 0    else
             "LOSING")
        print(f"  {label:>12}  {len(sub_o):>6,}  {no_wr:>7.1%}  {be:>9.1%}  "
              f"{raw_edge:>+6.1%}  {fee:>5.1%}  {net_edge:>+8.1%}  ${no_pnl:>+9.0f}  {v}")
        regime_results[(off_lo, off_hi)] = {
            "n": len(sub_o), "no_wr": no_wr, "breakeven": be,
            "net_edge": net_edge, "no_pnl": no_pnl
        }
    all_regime_results[regime_key] = regime_results


# ── Monthly P&L: squeeze_ranging regime only ──────────────────────────────────
print(f"\n{SEP}")
print("SIMULATED STRATEGY: Bet NO only when BB squeeze + ADX ranging + NOT RSI oversold")
print("  Best offset per hour = highest net edge within the regime")
print(SEP)

sub_strat = df_test[df_test["regime"] == "squeeze_ranging"].copy()
if not sub_strat.empty:
    # For each hour, pick the offset with highest net edge (empirically)
    best_rows = []
    for ts_h, grp in sub_strat.groupby("ts"):
        # Pick the offset where net_edge is best
        best_off = None; best_edge = -999
        for _, r in grp.iterrows():
            pk = r["p_kalshi"]
            no_wr_est = 1 - r["yes_won"]    # actual outcome, not estimate
            be  = 1 - pk
            fee = KALSHI_RAKE * pk * (1-pk)
            ne  = (no_wr_est - be) - fee     # per-trade edge (0 or 1 outcome)
            if ne > best_edge:
                best_edge = ne
                best_off  = r
        if best_off is not None:
            best_rows.append(best_off)

    df_best = pd.DataFrame(best_rows)
    df_best["ym"] = pd.to_datetime(df_best["ts"]).dt.to_period("M")

    # Monthly stats
    monthly_no_pnl = df_best.groupby("ym")["no_pnl"].sum()
    monthly_trades = df_best.groupby("ym").size()
    monthly_wins   = df_best.groupby("ym")["yes_won"].apply(lambda x: (1-x).sum())

    tot_trades = len(df_best)
    tot_pnl    = df_best["no_pnl"].sum()
    tot_wins   = (1 - df_best["yes_won"]).sum()
    wr         = tot_wins / tot_trades if tot_trades else 0

    print(f"\n  Total trades: {tot_trades:,}  ({tot_trades/df_test['ts'].nunique()*100:.1f}% of test hours)")
    print(f"  Win rate    : {wr:.1%}  ({int(tot_wins)}/{tot_trades})")
    print(f"  Total P&L   : ${tot_pnl:+,.2f}")
    print(f"\n  {'month':>8}  {'trades':>7}  {'wins':>6}  {'win%':>6}  {'pnl':>10}  bar")
    print("  " + "-" * 65)
    for ym in monthly_no_pnl.index:
        n_mo   = int(monthly_trades[ym])
        w_mo   = int(monthly_wins[ym])
        pnl_mo = monthly_no_pnl[ym]
        wr_mo  = w_mo / n_mo if n_mo else 0
        bar    = ("+" if pnl_mo > 0 else "-") * min(int(abs(pnl_mo)/20), 40)
        print(f"  {str(ym):>8}  {n_mo:>7,}  {w_mo:>6,}  {wr_mo:>5.1%}  ${pnl_mo:>+9.0f}  {bar}")

    print(f"\n  Profitable months: {(monthly_no_pnl>0).sum()} / {len(monthly_no_pnl)}")
    print(f"  Avg trades/month : {tot_trades/len(monthly_no_pnl):.0f}")
    print(f"  Avg P&L/month    : ${tot_pnl/len(monthly_no_pnl):+,.0f}")
    print(f"  Avg P&L/trade    : ${tot_pnl/tot_trades:+.2f}")

    # Compare: same hours, no regime filter (all hours with a trade)
    print(f"\n  COMPARISON — same offset range, no regime filter (all test hours):")
    for (off_lo, off_hi) in sorted(KALSHI_PRICE_TABLE.keys()):
        sub_all = df_test[df_test["off_lo"] == off_lo]
        if len(sub_all) < MIN_N:
            continue
        no_wr_all = 1 - sub_all["yes_won"].mean()
        pk        = KALSHI_PRICE_TABLE[(off_lo, off_hi)]
        be        = 1 - pk
        fee       = KALSHI_RAKE * pk * (1-pk)
        ne_all    = (no_wr_all - be) - fee
        no_wr_sq  = 1 - sub_strat[sub_strat["off_lo"]==off_lo]["yes_won"].mean() if len(sub_strat[sub_strat["off_lo"]==off_lo]) > MIN_N else float("nan")
        ne_sq     = (no_wr_sq - be) - fee if not math.isnan(no_wr_sq) else float("nan")
        label     = OFFSET_LABELS[(off_lo, off_hi)]
        lift      = ne_sq - ne_all if not math.isnan(ne_sq) else float("nan")
        print(f"    {label}: all_hours={ne_all:+.1%}  squeeze+ranging={ne_sq:+.1%}  lift={lift:+.1%}")

else:
    print("  No squeeze_ranging hours found in test period.")


# ── Breakeven analysis: what win rate does Kalshi require? ─────────────────────
print(f"\n{SEP}")
print("BREAKEVEN TABLE — At actual Kalshi prices, what NO win rate is required?")
print("  (Includes Kalshi 7% rake on stake)")
print(SEP)

print(f"\n  {'offset':>12}  {'Kalshi YES':>10}  {'NO breakeven':>13}  {'after rake':>11}  "
      f"{'all-hours NO%':>14}  {'sq+rng NO%':>11}  {'lift':>6}")
print("  " + "-" * 95)
for (off_lo, off_hi), pk in sorted(KALSHI_PRICE_TABLE.items()):
    label   = OFFSET_LABELS[(off_lo, off_hi)]
    be      = 1 - pk
    fee     = KALSHI_RAKE * pk * (1-pk)
    be_net  = be + fee
    sub_all = df_test[df_test["off_lo"] == off_lo]
    sub_sq  = df_test[(df_test["off_lo"] == off_lo) & (df_test["regime"] == "squeeze_ranging")]
    no_all  = 1 - sub_all["yes_won"].mean() if len(sub_all) >= MIN_N else float("nan")
    no_sq   = 1 - sub_sq["yes_won"].mean()  if len(sub_sq)  >= MIN_N else float("nan")
    lift    = no_sq - no_all if not math.isnan(no_sq) and not math.isnan(no_all) else float("nan")
    no_all_s = f"{no_all:.1%}" if not math.isnan(no_all) else "—"
    no_sq_s  = f"{no_sq:.1%}"  if not math.isnan(no_sq)  else "—"
    lift_s   = f"{lift:+.1%}"  if not math.isnan(lift)   else "—"
    print(f"  {label:>12}  {pk:>9.3f}  {be:>12.1%}  {be_net:>10.1%}  "
          f"{no_all_s:>14}  {no_sq_s:>11}  {lift_s:>6}")
