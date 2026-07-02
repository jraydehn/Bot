"""
backtest_new_model.py — Full BTC historical backtest with new 4-indicator model.

New indicator model (replaces OBI + session VWAP):
  no_score   = funding==-1 + stoch==-1 + rsi_1m==+1 + vwap_dev_1h==+1
  conf_score = funding==+1 + stoch==+1 + rsi_1m==-1 + vwap_dev_1h==-1

Baseline model (Gate PM + Gate 3 only, no indicator gates):
  Gate NS disabled, passes all contracts qualifying on PM + edge + R:R only.

All current decision gates applied:
  Gate 0      : p_model in [0.04, 0.96]
  Gate EMA-Dir: block bullish EMA + YES (BTC)
  Gate PM     : YES p_market >= 0.55 | NO p_market <= 0.35
  Gate NS     : no_score >= 1 for BTC NO (new model only)
  Gate 3      : net edge >= 3%
  Gate R:R    : risk/reward filter

KEY DESIGN DECISIONS:
  1. One trade per hour max (best net_edge contract).
     Multiple contracts at same hourly bar are perfectly correlated — same BTC
     move determines all outcomes. Taking multiple would be double-positioning.
  2. Fixed $50 stake throughout (= 5% of $1,000 initial bankroll).
     Avoids exponential Kelly compounding that produces unreadable numbers.
     Shows the strategy's absolute edge independent of capital effects.
  3. p_market from realistic table (medians from 2,000+ live contract observations
     on 2026-03-20, from pricing_comparison.simulate_p_market()).

Funding rate proxy: 8h price momentum sign.
  Positive 8h return → longs heavy → funding_bias = -1 (bearish for price)
  Negative 8h return → shorts heavy → funding_bias = +1 (bullish for price)
"""

import sys, math, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

from probability_engine import estimate_probability
from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD
from kelly_sizing import compute_kelly_size

# ── CONFIG ────────────────────────────────────────────────────────────────────
BANKROLL_INIT = 1000.0
FIXED_STAKE   = 50.0     # fixed bet per trade (= 5% of $1,000 initial)
BTC_CALIB     = 0.65     # BTC p_model calibration factor
TAU           = 60       # minutes to expiry
MIN_NET_EDGE  = 0.03     # Gate 3 floor

# Signed offsets from spot (negative = ITM for YES / OTM for NO)
OFFSETS = [-0.005, -0.003, -0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]

# Realistic Kalshi p_yes_market by abs(offset).
# Medians from 2,000 real KXBTCD contracts (2026-03-20, pricing_comparison.py).
# OTM (offset > 0): YES is cheap (p_market is the YES price).
# ITM (offset < 0): YES is expensive, NO is cheap.
# NO side table is slightly higher (retail bias toward YES pushes YES price up,
# making NO cheaper — but we use the average to be conservative).
OTM_TABLE = {
    0.001: 0.39,   # YES side: 0.335  NO side: 0.440  → avg 0.39
    0.002: 0.35,   # interpolated between 0.001 and 0.003
    0.003: 0.325,  # YES side: 0.300  NO side: 0.350  → avg 0.325
    0.005: 0.245,  # YES side: 0.225  NO side: 0.265  → avg 0.245
    0.008: 0.175,  # interpolated
    0.010: 0.155,  # YES side: 0.125  NO side: 0.180  → avg 0.155
}
ITM_TABLE = {
    0.001: 0.61,   # 1 - OTM[0.001]
    0.002: 0.65,   # interpolated
    0.003: 0.675,  # 1 - OTM[0.003]
    0.005: 0.755,  # 1 - OTM[0.005]
    0.008: 0.825,  # interpolated
}
OTM_KEYS = sorted(OTM_TABLE.keys())
ITM_KEYS = sorted(ITM_TABLE.keys())

def get_pmarket(offset: float) -> float:
    """Return realistic Kalshi YES market price for a given offset."""
    abs_off = abs(offset)
    if offset < 0:
        k = min(ITM_KEYS, key=lambda x: abs(x - abs_off))
        return ITM_TABLE[k]
    k = min(OTM_KEYS, key=lambda x: abs(x - abs_off))
    return OTM_TABLE[k]


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
print(f"Loading 1m: {files_1m[-1]}")
ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()
print(f"  {ohlcv_1m.index[0]} → {ohlcv_1m.index[-1]}  ({len(ohlcv_1m):,} bars)")

files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
print(f"Loading 1h: {files_1h[-1]}")
ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()
print(f"  {ohlcv_1h.index[0]} → {ohlcv_1h.index[-1]}  ({len(ohlcv_1h):,} bars)")


# ── PRE-COMPUTE 1m INDICATORS ─────────────────────────────────────────────────
print("\nPre-computing 1m indicators...")
close_1m  = ohlcv_1m["close"].values.astype(float)
volume_1m = ohlcv_1m["volume"].values.astype(float)
n1m       = len(close_1m)

# RSI-14 (Wilder's smoothing)
def _rsi14(c: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    gain  = np.maximum(delta, 0.0)
    loss  = np.maximum(-delta, 0.0)
    ag    = np.full(len(c), np.nan)
    al    = np.full(len(c), np.nan)
    if len(c) > period:
        ag[period] = gain[1:period+1].mean()
        al[period] = loss[1:period+1].mean()
    for i in range(period + 1, len(c)):
        ag[i] = (ag[i-1] * (period - 1) + gain[i]) / period
        al[i] = (al[i-1] * (period - 1) + loss[i]) / period
    rs = np.where(al == 0, np.inf, ag / al)
    return 100.0 - 100.0 / (1.0 + rs)

rsi_1m = _rsi14(close_1m, 14)

# Rolling 60-bar VWAP and deviation
VWAP_W  = 60
cv      = close_1m * volume_1m
vwap_1m = np.full(n1m, np.nan)
for i in range(VWAP_W - 1, n1m):
    s     = i - VWAP_W + 1
    v_sum = volume_1m[s:i+1].sum()
    vwap_1m[i] = cv[s:i+1].sum() / v_sum if v_sum > 0 else close_1m[i]

vwap_dev_1m = np.where(
    ~np.isnan(vwap_1m) & (vwap_1m > 0),
    (close_1m - vwap_1m) / vwap_1m,
    np.nan,
)

# Realized vol (60-bar rolling log-return std)
log_ret = np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0)
sigma_1m = np.full(n1m, np.nan)
for i in range(59, n1m):
    sigma_1m[i] = float(np.std(log_ret[i-59:i+1]))

# 8h momentum → funding rate proxy (480 1m bars)
MOM8 = 480
mom8h = np.full(n1m, np.nan)
for i in range(MOM8, n1m):
    b = close_1m[i - MOM8]
    if b > 0:
        mom8h[i] = (close_1m[i] - b) / b

print("  1m indicators done.")


# ── PRE-COMPUTE 15m STOCHASTIC ────────────────────────────────────────────────
print("Pre-computing 15m stochastic...")
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna(subset=["close"])

K_PER, D_PER = 14, 3
OB, OS = 80, 20
ll15  = df_15m["low"].rolling(K_PER).min()
hh15  = df_15m["high"].rolling(K_PER).max()
hlr15 = (hh15 - ll15).replace(0, np.nan)
stk15 = ((df_15m["close"] - ll15) / hlr15) * 100.0
std15 = stk15.rolling(D_PER).mean()

stoch15 = pd.Series(0, index=df_15m.index, dtype=int)
for i in range(1, len(df_15m)):
    kc, dc = stk15.iat[i], std15.iat[i]
    kp, dp = stk15.iat[i-1], std15.iat[i-1]
    if any(v != v for v in (kc, dc, kp, dp)):
        continue
    bear_x = kp > dp and kc < dc and kp > OB
    bull_x = kp < dp and kc > dc and kp < OS
    if bear_x or kc > OB:
        stoch15.iat[i] = -1
    elif bull_x or kc < OS:
        stoch15.iat[i] = +1

# Forward-fill 15m stoch to 1m index
stoch_ff_1m = stoch15.reindex(ohlcv_1m.index, method="ffill").fillna(0).astype(int)
print(f"  Stochastic: {len(df_15m)} 15m bars → forward-filled to 1m.")


# ── PRE-COMPUTE 1h EMA ALIGNMENT ─────────────────────────────────────────────
print("Pre-computing 1h EMA alignment...")
close_1h = ohlcv_1h["close"]
ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
ema50_1h = close_1h.ewm(span=50, adjust=False).mean()

ema_align_vals = pd.Series("neutral", index=ohlcv_1h.index, dtype=object)
for i in range(3, len(close_1h)):
    e20 = ema20_1h.values[i-2:i+1]
    e50 = ema50_1h.values[i-2:i+1]
    c   = close_1h.values[i-2:i+1]
    if all(e20 > e50) and all(c > e20):
        ema_align_vals.iat[i] = "bullish"
    elif all(e20 < e50) or all(c < e50):
        ema_align_vals.iat[i] = "bearish"

# Forward-fill to 1m index
ema_align_ff = ema_align_vals.reindex(ohlcv_1m.index, method="ffill").fillna("neutral")
print(f"  EMA alignment: {len(ohlcv_1h)} 1h bars → forward-filled to 1m.")
print("Pre-computation complete.\n")


# ── GATE HELPERS ──────────────────────────────────────────────────────────────
def rr_min_edge(pm: float, side: str) -> float:
    rr = pm / (1 - pm) if side == "yes" else (1 - pm) / pm
    for thresh, me in [(1, 0.03), (2, 0.06), (4, 0.09), (6, 0.15), (8, 0.20)]:
        if rr <= thresh:
            return me
    return 0.25


def try_trade(spot, K, p_market, sigma_min, no_score, ema_align, *, gate_ns=True):
    """
    Evaluate YES and NO for one contract. Returns best qualifying side or None.
    gate_ns: if False, skip Gate NS (used for baseline comparison).
    Uses FIXED_STAKE ($50) — no Kelly sizing or bankroll dependency.
    """
    try:
        prob = estimate_probability(spot, K, TAU, sigma_min,
                                    confirmation_score=0, max_score=4)
        p_raw = prob.p_yes
    except Exception:
        return None

    if not (0.04 <= p_raw <= 0.96):
        return None

    best = None
    for side in ("yes", "no"):
        p_model  = p_raw * BTC_CALIB
        fee      = kalshi_fee(p_market)
        raw_edge = (p_model - p_market) if side == "yes" else (p_market - p_model)
        net_edge = raw_edge - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD

        if ema_align == "bullish" and side == "yes":
            continue

        if side == "yes" and p_market < 0.55:
            continue
        if side == "no"  and p_market > 0.35:
            continue

        if gate_ns and side == "no" and no_score < 1:
            continue

        if net_edge < MIN_NET_EDGE:
            continue

        rr = p_market / (1 - p_market) if side == "yes" else (1 - p_market) / p_market
        if rr < 0.33:
            continue
        if side == "no" and rr > 4.0:
            continue
        if net_edge < rr_min_edge(p_market, side):
            continue

        if best is None or net_edge > best["net_edge"]:
            best = {
                "side": side, "p_model": round(p_model, 4),
                "p_raw": round(p_raw, 4), "p_market": p_market,
                "net_edge": round(net_edge, 4),
            }
    return best


# ── MAIN BACKTEST LOOP ────────────────────────────────────────────────────────
# One trade per hour: evaluate all offsets, pick the single best-qualifying
# contract by net_edge. This prevents double-positioning on correlated bets
# (two NO contracts at different offsets on the same hourly bar have identical
# underlying BTC price movement determining both outcomes).
print("Running backtest (1 trade/hour, fixed $50 stake)...")
ohlcv_1m_idx = ohlcv_1m.index
close_1h_arr = ohlcv_1h["close"].values.astype(float)
ts_1h_arr    = ohlcv_1h.index
n1h          = len(ts_1h_arr)

records_new  = []
records_base = []
pnl_new  = 0.0
pnl_base = 0.0

WARMUP = 50  # bars to skip for EMA/indicator warmup

for i_h in range(WARMUP, n1h - 1):
    ts_now     = ts_1h_arr[i_h]
    spot       = float(close_1h_arr[i_h])
    next_close = float(close_1h_arr[i_h + 1])

    # Find last 1m bar at or before ts_now
    pos1m = int(ohlcv_1m_idx.searchsorted(ts_now, side="right")) - 1
    if pos1m < max(MOM8, VWAP_W, 59):
        continue

    sigma  = float(sigma_1m[pos1m])
    rsi    = float(rsi_1m[pos1m])
    vd     = float(vwap_dev_1m[pos1m])
    m8     = float(mom8h[pos1m])
    stoch  = int(stoch_ff_1m.iat[pos1m])
    ema    = str(ema_align_ff.iat[pos1m])

    if np.isnan(sigma) or sigma <= 0 or np.isnan(rsi) or np.isnan(vd) or np.isnan(m8):
        continue

    # Derived signals
    rsi_sig  = +1 if rsi > 60 else (-1 if rsi < 40 else 0)
    vd_bin   = +1 if vd > 0.0005 else (-1 if vd < -0.0005 else 0)
    fund_sig = -1 if m8 > 0.001 else (+1 if m8 < -0.001 else 0)

    no_score_new  = int(fund_sig == -1) + int(stoch == -1) + int(rsi_sig == +1) + int(vd_bin == +1)
    no_score_base = int(fund_sig == -1) + int(stoch == -1)

    # Collect all qualifying contracts this hour; pick the best one per model.
    best_new  = None
    best_base = None

    for offset in OFFSETS:
        K        = spot * (1.0 + offset)
        p_market = get_pmarket(offset)
        resolved_yes = int(next_close > K)

        t_new = try_trade(spot, K, p_market, sigma, no_score_new, ema, gate_ns=True)
        if t_new:
            t_new["offset"] = offset; t_new["K"] = K
            t_new["p_market"] = p_market; t_new["resolved_yes"] = resolved_yes
            t_new["no_score"] = no_score_new
            if best_new is None or t_new["net_edge"] > best_new["net_edge"]:
                best_new = t_new

        t_base = try_trade(spot, K, p_market, sigma, no_score_base, ema, gate_ns=False)
        if t_base:
            t_base["offset"] = offset; t_base["K"] = K
            t_base["p_market"] = p_market; t_base["resolved_yes"] = resolved_yes
            t_base["no_score"] = no_score_base
            if best_base is None or t_base["net_edge"] > best_base["net_edge"]:
                best_base = t_base

    def commit_trade(trade, records, pnl_running):
        side         = trade["side"]
        p_market     = trade["p_market"]
        resolved_yes = trade["resolved_yes"]
        won = (resolved_yes == 1) if side == "yes" else (resolved_yes == 0)
        if won:
            pnl = FIXED_STAKE * (p_market / (1 - p_market)) if side == "no" \
                  else FIXED_STAKE * ((1 - p_market) / p_market)
        else:
            pnl = -FIXED_STAKE
        pnl_running += pnl
        records.append({
            "ts": ts_now, "offset": round(trade["offset"], 4),
            "spot": round(spot, 2), "K": round(trade["K"], 2),
            "p_market": p_market, "side": side,
            "net_edge": trade["net_edge"],
            "p_model": trade["p_model"], "p_raw": trade["p_raw"],
            "rsi_sig": rsi_sig, "vd_bin": vd_bin,
            "stoch": stoch, "fund": fund_sig, "ema": ema,
            "no_score": trade["no_score"],
            "resolved_yes": resolved_yes, "won": int(won),
            "pnl": round(pnl, 2),
            "cumulative_pnl": round(pnl_running, 2),
        })
        return pnl_running

    if best_new:
        pnl_new = commit_trade(best_new, records_new, pnl_new)
    if best_base:
        pnl_base = commit_trade(best_base, records_base, pnl_base)

    if i_h % 2000 == 0:
        pct = (i_h - WARMUP) / (n1h - 1 - WARMUP) * 100
        print(f"  {pct:5.1f}%  h={i_h}  "
              f"new={len(records_new)} trades pnl=${pnl_new:+,.0f}  "
              f"base={len(records_base)} pnl=${pnl_base:+,.0f}")

print(f"\nDone. New={len(records_new)} trades, Baseline={len(records_base)} trades")

df_new  = pd.DataFrame(records_new)
df_base = pd.DataFrame(records_base)


# ── REPORTING ─────────────────────────────────────────────────────────────────
SEP = "=" * 72

def report(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"\n{label}: no trades.")
        return
    n         = len(df)
    wins      = int(df["won"].sum())
    win_pct   = wins / n
    total_pnl = df["pnl"].sum()
    avg_edge  = df["net_edge"].mean()
    avg_pnl   = df["pnl"].mean()

    # Max drawdown on cumulative PnL curve
    cum = df["cumulative_pnl"].values
    peak = cum[0]
    max_dd_dollars = 0.0
    for v in cum:
        if v > peak: peak = v
        dd = peak - v
        if dd > max_dd_dollars: max_dd_dollars = dd

    # Monthly PnL
    df2 = df.copy()
    df2["ym"] = pd.to_datetime(df2["ts"]).dt.to_period("M")
    monthly = df2.groupby("ym")["pnl"].sum()
    months_pos = (monthly > 0).sum()
    months_tot = len(monthly)

    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  Trades          : {n:,}  ({n / max(months_tot,1):.0f}/month avg)")
    print(f"  Win rate        : {win_pct:.1%}  ({wins}/{n})")
    print(f"  Total PnL       : ${total_pnl:+,.2f}  (fixed $50 stake)")
    print(f"  Avg PnL/trade   : ${avg_pnl:+.2f}")
    print(f"  Avg net edge    : {avg_edge:+.2%}")
    print(f"  Max drawdown    : ${max_dd_dollars:,.0f}")
    print(f"  Profitable months: {months_pos}/{months_tot}")

    print(f"\n  Side breakdown:")
    for side in ["yes", "no"]:
        sub = df[df["side"] == side]
        if sub.empty: continue
        sw = int(sub["won"].sum())
        print(f"    {side.upper():3s}: {len(sub):5,} trades  win={sw/len(sub):.1%}  "
              f"pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  By offset (all qualify if no_score ≥ 1):")
    for off in sorted(df["offset"].unique()):
        sub = df[df["offset"] == off]
        sw = int(sub["won"].sum())
        print(f"    offset={off:+.3f}  n={len(sub):4,}  win={sw/len(sub):.1%}  "
              f"pnl=${sub['pnl'].sum():+,.2f}")

    if "no_score" in df.columns:
        print(f"\n  NO trades by no_score:")
        sub_no = df[df["side"] == "no"]
        if not sub_no.empty:
            for sc in sorted(sub_no["no_score"].unique()):
                s = sub_no[sub_no["no_score"] == sc]
                sw = int(s["won"].sum())
                print(f"    no_score={sc}  n={len(s):4,}  win={sw/len(s):.1%}  "
                      f"pnl=${s['pnl'].sum():+,.2f}")

    print(f"\n  By EMA alignment:")
    for ea in ["bullish", "bearish", "neutral"]:
        sub = df[df["ema"] == ea]
        if sub.empty: continue
        sw = int(sub["won"].sum())
        print(f"    {ea:8s}: n={len(sub):4,}  win={sw/len(sub):.1%}  pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  Monthly PnL (${FIXED_STAKE} stake):")
    for ym, pnl in monthly.items():
        bar = "+" * min(int(pnl / 20), 40) if pnl > 0 else "-" * min(int(-pnl / 20), 40)
        print(f"    {ym}  ${pnl:+8.2f}  {bar}")


report(df_new,  "NEW MODEL (4-indicator: funding + stoch + RSI-14(1m) + rolling VWAP dev)")
report(df_base, "BASELINE  (Gate PM + Gate 3 + R:R only — no indicator gates)")


# ── SAVE ──────────────────────────────────────────────────────────────────────
out_new  = BASE / "results/backtest_new_model.csv"
out_base = BASE / "results/backtest_baseline.csv"
df_new.to_csv(out_new, index=False)
df_base.to_csv(out_base, index=False)
print(f"\n{SEP}")
print(f"  Saved: {out_new}  ({len(df_new):,} rows)")
print(f"  Saved: {out_base}  ({len(df_base):,} rows)")
print(SEP)
