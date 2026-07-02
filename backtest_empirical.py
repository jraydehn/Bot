"""
backtest_empirical.py — Empirical win-rate driven BTC Kalshi backtest.

Core principle: no log-normal model is used to simulate Kalshi prices.
Instead, Kalshi's implied volatility is back-solved from actual observed
paper trade prices, then used to price historical contracts realistically.

Method:
  1. From real paper_trades.csv (offset_pct, p_market, vol_60m_model),
     back-solve Kalshi's implied sigma per observation:
         p_YES = 1 - Φ(log(K/S) / (σ_eff × √60))
         → σ_eff = log(K/S) / (√60 × Φ⁻¹(1 - p_market))
     Fit: σ_eff_kalshi = a × σ_model + b  (linear regression)
     This is the vol Kalshi actually uses — not our assumption.

  2. For every hour in the test set (Jan 2025 – Apr 2026):
     - BTC spot at hour open
     - Sweep realistic offsets: +0.05% to +1.0% for NO, -0.05% to -1.0% for YES
     - For each offset, price strike realistically using fitted σ_eff_kalshi
     - Resolve with actual next-hour BTC close
     - Compute P&L at realistic Kalshi prices

  3. For each (offset × indicator_state) bucket:
     - Empirical YES win rate
     - Realistic Kalshi YES price at that offset
     - Expected value = win_rate × payout - (1-win_rate) × stake
     - Breakeven win rate = Kalshi YES price (for YES bets)
     - Edge = empirical win rate - Kalshi price

  4. Summary: which offsets and which indicator regimes have genuine edge?

Indicators evaluated:
  - EMA alignment (bullish/neutral/bearish, 20/50 on 1h)
  - RSI-14 (1h) regime: oversold <30, neutral 30-70, overbought >70
  - ADX-14 (15m resampled) regime: ranging <20, moderate 20-35, trending >35
  - BB width percentile (20th/80th pctile of trailing 200h)
  - Vol momentum: σ_60m vs σ_60m 4h ago
  - Vol ratio: σ_60m / σ_1440m_lagged
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

SEP = "=" * 72
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# Offsets to sweep (fraction of spot, positive = above spot)
# Mirrors real Kalshi strike ladder resolution
NO_OFFSETS  = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.010]
YES_OFFSETS = [-0.001, -0.0015, -0.002, -0.0025, -0.003, -0.004, -0.005]

# Backsolve σ_eff from Kalshi price constants
MODEL_VOL_WINDOW  = 60
KALSHI_VOL_WINDOW = 1440
KALSHI_VOL_LAG    = 120
WARMUP_BARS       = KALSHI_VOL_WINDOW + KALSHI_VOL_LAG + 60
TAU               = 60   # minutes

KALSHI_RAKE    = 0.07
FIXED_STAKE    = 50.0
MIN_N_BUCKET   = 30   # minimum observations for a stat to be reported

# ── STEP 0: Load data ─────────────────────────────────────────────────────────
print(SEP)
print("STEP 0 — Loading price data")
print(SEP)

files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))
print(f"1m: {files_1m[-1]}")
print(f"1h: {files_1h[-1]}")

ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h = ohlcv_1h["close"].values.astype(float)
high_1h  = ohlcv_1h["high"].values.astype(float)
low_1h   = ohlcv_1h["low"].values.astype(float)
ts_1h    = ohlcv_1h.index
n1h      = len(ts_1h)

close_1m = ohlcv_1m["close"].values.astype(float)
log_ret  = pd.Series(
    np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0),
    index=ohlcv_1m.index,
)
sigma_model_1m  = log_ret.rolling(MODEL_VOL_WINDOW).std()
sigma_kalshi_1m = log_ret.rolling(KALSHI_VOL_WINDOW).std().shift(KALSHI_VOL_LAG)
# Vol momentum: σ_60m vs σ_60m 4h ago
sigma_model_lag4h = sigma_model_1m.shift(240)

ohlcv_1m_idx = ohlcv_1m.index
print(f"  1m rows: {len(ohlcv_1m):,}  |  1h rows: {n1h:,}")

# ── STEP 1: Calibrate Kalshi implied-vol from actual paper trades ─────────────
print(f"\n{SEP}")
print("STEP 1 — Calibrate Kalshi pricing from actual observed prices")
print("  Back-solving σ_eff from real (offset, p_market, σ_model) triplets")
print(SEP)

pt_path = BASE / "results/paper_trades.csv"
df_pt   = pd.read_csv(pt_path)
for c in ["offset_pct", "p_market", "vol_60m_model", "spot", "strike"]:
    if c in df_pt.columns:
        df_pt[c] = pd.to_numeric(df_pt[c], errors="coerce")

df_pt = df_pt.dropna(subset=["offset_pct", "p_market", "vol_60m_model", "spot", "strike"])
df_pt = df_pt[(df_pt["p_market"] > 0.02) & (df_pt["p_market"] < 0.98)]
df_pt = df_pt[(df_pt["offset_pct"].abs() > 0.001)]    # skip near-ATM noise
df_pt = df_pt[(df_pt["vol_60m_model"] > 0)]

def backsolve_sigma_eff(row):
    """Back-solve the σ Kalshi implicitly uses from the observed YES price."""
    try:
        log_k_s = math.log(float(row["strike"]) / float(row["spot"]))
        if abs(log_k_s) < 1e-6:
            return float("nan")
        z = sp_norm.ppf(1 - float(row["p_market"]))
        if abs(z) < 1e-4:
            return float("nan")
        return log_k_s / (z * math.sqrt(TAU))
    except Exception:
        return float("nan")

df_pt["sigma_eff_kalshi"] = df_pt.apply(backsolve_sigma_eff, axis=1)
df_pt = df_pt.dropna(subset=["sigma_eff_kalshi"])
df_pt = df_pt[df_pt["sigma_eff_kalshi"] > 0]

# Fit linear model: σ_eff_kalshi = slope × σ_model + intercept
x = df_pt["vol_60m_model"].values
y = df_pt["sigma_eff_kalshi"].values
slope, intercept, r, p_val, se = scipy_stats.linregress(x, y)
r2 = r ** 2

print(f"\n  Calibration sample : {len(df_pt):,} paper-trade observations")
print(f"  σ_eff_kalshi = {slope:.4f} × σ_model + {intercept:.7f}")
print(f"  R²={r2:.3f}   (how well Kalshi prices track our realized vol)")
print(f"\n  Interpretation:")
print(f"    slope > 1 → Kalshi inflates vol vs realized  (NO edge exists when slope high)")
print(f"    slope < 1 → Kalshi deflates vol vs realized  (YES edge exists when slope low)")
print(f"    intercept → Kalshi's baseline vol floor even in calm markets")

def kalshi_sigma(sigma_model_val: float) -> float:
    """Predict Kalshi's effective pricing sigma from our realized sigma."""
    return max(slope * sigma_model_val + intercept, sigma_model_val * 0.5)

def p_yes_realistic(spot, K, tau, sigma_model_val):
    """Compute a realistic Kalshi YES price using calibrated σ_eff."""
    sig_eff = kalshi_sigma(sigma_model_val)
    if sig_eff <= 0 or spot <= 0 or K <= 0:
        return float("nan")
    z = math.log(K / spot) / (sig_eff * math.sqrt(tau))
    return float(1.0 - sp_norm.cdf(z))


# ── STEP 2: Pre-compute indicators ───────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — Pre-computing indicators on 1h data")
print(SEP)

# EMA 20/50 on 1h
ema20 = pd.Series(close_1h, index=ts_1h).ewm(span=20, adjust=False).mean()
ema50 = pd.Series(close_1h, index=ts_1h).ewm(span=50, adjust=False).mean()

# RSI-14 on 1h
delta   = pd.Series(close_1h, index=ts_1h).diff()
gain    = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
loss    = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
rs      = gain / loss.replace(0, 1e-10)
rsi_1h  = 100 - (100 / (1 + rs))

# Bollinger Band width percentile on 1h (20-period BB, rolling 200h percentile)
bb_mid  = pd.Series(close_1h, index=ts_1h).rolling(20).mean()
bb_std  = pd.Series(close_1h, index=ts_1h).rolling(20).std()
bb_width = (2 * bb_std / bb_mid)   # normalized width
bb_pct  = bb_width.rolling(200).rank(pct=True)   # percentile of current width

# ADX-14 on 15m resampled (then ffill to 1h)
print("  Computing 15m ADX (this takes a moment)...")
df_15m = ohlcv_1m.resample("15min", origin="start_day").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna(subset=["close"])

_h = df_15m["high"]
_l = df_15m["low"]
_c = df_15m["close"]
_c_prev = _c.shift(1)
tr   = pd.concat([_h - _l, (_h - _c_prev).abs(), (_l - _c_prev).abs()], axis=1).max(axis=1)
dm_p = (_h - _h.shift(1)).clip(lower=0).where((_h - _h.shift(1)) > (_l.shift(1) - _l), 0)
dm_m = (_l.shift(1) - _l).clip(lower=0).where((_l.shift(1) - _l) > (_h - _h.shift(1)), 0)
atr14 = tr.ewm(com=13, adjust=False).mean()
di_p  = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
di_m  = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-10)
dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, 1e-10)
adx_15m = dx.ewm(com=13, adjust=False).mean()
adx_1h  = adx_15m.resample("1h", origin="start_day").last().reindex(ts_1h, method="ffill")

# Vol momentum on 1m → sampled at 1h
sigma_model_1h    = sigma_model_1m.reindex(ts_1h, method="ffill")
sigma_model_lag4h_1h = sigma_model_lag4h.reindex(ts_1h, method="ffill")
vol_momentum_1h   = sigma_model_1h / sigma_model_lag4h_1h.replace(0, float("nan"))

# Vol ratio at 1h
sigma_kalshi_1h = sigma_kalshi_1m.reindex(ts_1h, method="ffill")
vol_ratio_1h    = sigma_model_1h / sigma_kalshi_1h.replace(0, float("nan"))

print("  Done.")

# ── STEP 3: Main backtest loop ────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3 — Sweeping all strikes for every hour (Jan 2025 – Apr 2026)")
print(SEP)

rows = []

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
    if not (sig_m > 0 and sig_k > 0):
        continue

    # Indicators at this hour
    ema    = "bullish" if (ema20.iat[i_h] > ema50.iat[i_h] and close_1h[i_h] > ema20.iat[i_h]) \
             else "bearish" if (ema20.iat[i_h] < ema50.iat[i_h] or close_1h[i_h] < ema50.iat[i_h]) \
             else "neutral"
    rsi    = float(rsi_1h.iat[i_h]) if not np.isnan(rsi_1h.iat[i_h]) else 50.0
    rsi_st = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    adx_v  = float(adx_1h.iat[i_h]) if not np.isnan(adx_1h.iat[i_h]) else 20.0
    adx_st = "ranging" if adx_v < 20 else "trending" if adx_v > 35 else "moderate"
    bb_p   = float(bb_pct.iat[i_h]) if not np.isnan(bb_pct.iat[i_h]) else 0.5
    bb_st  = "squeeze" if bb_p < 0.20 else "expansion" if bb_p > 0.80 else "normal"
    vm     = float(vol_momentum_1h.iat[i_h]) if not np.isnan(vol_momentum_1h.iat[i_h]) else 1.0
    vm_st  = "contracting" if vm < 0.80 else "expanding" if vm > 1.25 else "stable"
    vr     = float(vol_ratio_1h.iat[i_h]) if not np.isnan(vol_ratio_1h.iat[i_h]) else 1.0
    vr_st  = "low" if vr < 0.80 else "high" if vr > 1.20 else "neutral"

    for offset in NO_OFFSETS + YES_OFFSETS:
        K          = spot * (1.0 + offset)
        yes_won    = int(next_close > K)

        # Realistic Kalshi YES price (calibrated from actual observations)
        p_k = p_yes_realistic(spot, K, TAU, sig_m)
        if np.isnan(p_k) or p_k <= 0.01 or p_k >= 0.99:
            continue

        # Fee
        fee = KALSHI_RAKE * p_k * (1 - p_k)

        # YES side: win if close > K
        yes_payout = (1 - p_k) / p_k   # payout multiple on stake
        yes_pnl    = FIXED_STAKE * yes_payout if yes_won else -FIXED_STAKE
        yes_pnl_net = yes_pnl - FIXED_STAKE * fee  # after Kalshi rake

        # NO side: win if close <= K
        no_won    = 1 - yes_won
        no_payout = p_k / (1 - p_k)
        no_pnl    = FIXED_STAKE * no_payout if no_won else -FIXED_STAKE
        no_pnl_net = no_pnl - FIXED_STAKE * fee

        rows.append({
            "ts":          ts_now,
            "spot":        round(spot, 2),
            "K":           round(K, 2),
            "offset":      round(offset, 5),
            "next_close":  round(next_close, 2),
            "yes_won":     yes_won,
            "p_kalshi":    round(p_k, 4),
            "sigma_model": round(sig_m, 7),
            "sigma_kalshi_eff": round(kalshi_sigma(sig_m), 7),
            "vol_ratio":   round(vr, 4),
            "yes_pnl_net": round(yes_pnl_net, 2),
            "no_pnl_net":  round(no_pnl_net, 2),
            "ema":         ema,
            "rsi":         round(rsi, 1),
            "rsi_state":   rsi_st,
            "adx":         round(adx_v, 1),
            "adx_state":   adx_st,
            "bb_pct":      round(bb_p, 3),
            "bb_state":    bb_st,
            "vol_mom":     round(vm, 3),
            "vol_mom_state": vm_st,
            "vr_state":    vr_st,
        })

df = pd.DataFrame(rows)
print(f"  Total (hour × offset) observations: {len(df):,}")
print(f"  Unique hours: {df['ts'].nunique():,}")
print(f"  Offsets evaluated: {sorted(df['offset'].unique())}")


# ── STEP 4: Baseline R:R analysis (no indicator conditioning) ─────────────────
print(f"\n{SEP}")
print("STEP 4 — Baseline: empirical win rates vs Kalshi prices by offset")
print("         Each row = actual outcome, $50 fixed stake, realistic Kalshi pricing")
print(SEP)

print(f"\n  {'offset':>8}  {'n_hours':>8}  {'yes_win%':>9}  {'p_kalshi':>9}  "
      f"{'yes_edge':>9}  {'yes_pnl':>10}  {'no_win%':>8}  {'no_edge':>8}  {'no_pnl':>10}")
print("  " + "-" * 95)

for off in sorted(df["offset"].unique()):
    sub = df[df["offset"] == off]
    if len(sub) < MIN_N_BUCKET:
        continue
    yes_wr   = sub["yes_won"].mean()
    p_k_med  = sub["p_kalshi"].median()
    yes_edge = yes_wr - p_k_med
    no_wr    = 1 - yes_wr
    no_edge  = no_wr - (1 - p_k_med)
    yes_pnl  = sub["yes_pnl_net"].sum()
    no_pnl   = sub["no_pnl_net"].sum()
    sign = " ←" if abs(yes_edge) > 0.05 or abs(no_edge) > 0.05 else ""
    print(f"  {off:+8.4f}  {len(sub):>8,}  {yes_wr:>8.1%}  {p_k_med:>9.3f}  "
          f"{yes_edge:>+8.1%}  ${yes_pnl:>+9.0f}  {no_wr:>7.1%}  {no_edge:>+7.1%}  ${no_pnl:>+9.0f}{sign}")


# ── STEP 5: Indicator conditioning — does anything improve win rates? ──────────
print(f"\n{SEP}")
print("STEP 5 — Indicator conditioning: win rates per indicator state")
print("         Showing both YES and NO side. Edge = win_rate - breakeven_rate")
print(SEP)

INDICATOR_STATES = [
    ("EMA",         "ema",           ["bullish", "neutral", "bearish"]),
    ("RSI",         "rsi_state",     ["oversold", "neutral", "overbought"]),
    ("ADX",         "adx_state",     ["ranging", "moderate", "trending"]),
    ("BB width",    "bb_state",      ["squeeze", "normal", "expansion"]),
    ("Vol momentum","vol_mom_state",  ["contracting", "stable", "expanding"]),
    ("Vol ratio",   "vr_state",      ["low", "neutral", "high"]),
]

# Focus on three representative offsets: near NO, mid NO, near YES
FOCUS_OFFSETS = {
    "NO  +0.001": 0.001,
    "NO  +0.002": 0.002,
    "NO  +0.003": 0.003,
    "YES -0.001": -0.001,
    "YES -0.002": -0.002,
}

for ind_name, ind_col, states in INDICATOR_STATES:
    print(f"\n  ── {ind_name} ──")
    header = f"  {'state':>12}"
    for label in FOCUS_OFFSETS:
        header += f"  {label:>14}"
    print(header)
    print("  " + "-" * (14 + 16 * len(FOCUS_OFFSETS)))

    for state in states:
        line = f"  {state:>12}"
        mask = df[ind_col] == state
        for label, off in FOCUS_OFFSETS.items():
            sub = df[mask & (df["offset"] == off)]
            if len(sub) < MIN_N_BUCKET:
                line += f"  {'<30 obs':>14}"
                continue
            if off > 0:
                wr    = 1 - sub["yes_won"].mean()   # NO win rate
                be    = 1 - sub["p_kalshi"].median()
                pnl   = sub["no_pnl_net"].sum()
            else:
                wr    = sub["yes_won"].mean()
                be    = sub["p_kalshi"].median()
                pnl   = sub["yes_pnl_net"].sum()
            edge = wr - be
            flag = " ★" if edge > 0.05 else " ✗" if edge < -0.03 else "  "
            line += f"  {wr:.0%} {edge:+.0%} ${pnl:+.0f}{flag}"
        print(line)


# ── STEP 6: Combined signal analysis — best 2-indicator combinations ───────────
print(f"\n{SEP}")
print("STEP 6 — Combined signals: best 2-indicator combinations for NO +0.002")
print("         Sorted by NO edge descending. Min 50 observations.")
print(SEP)

TARGET_OFFSET = 0.002
sub_no = df[df["offset"] == TARGET_OFFSET].copy()

combos = []
IND_PAIRS = [
    ("ema", "rsi_state"), ("ema", "adx_state"), ("ema", "bb_state"),
    ("ema", "vol_mom_state"), ("ema", "vr_state"),
    ("adx_state", "bb_state"), ("adx_state", "vol_mom_state"),
    ("rsi_state", "adx_state"), ("rsi_state", "bb_state"),
    ("bb_state", "vol_mom_state"), ("bb_state", "vr_state"),
    ("vol_mom_state", "vr_state"),
]
for c1, c2 in IND_PAIRS:
    for v1 in sub_no[c1].unique():
        for v2 in sub_no[c2].unique():
            s = sub_no[(sub_no[c1] == v1) & (sub_no[c2] == v2)]
            if len(s) < 50:
                continue
            no_wr  = 1 - s["yes_won"].mean()
            be     = 1 - s["p_kalshi"].median()
            edge   = no_wr - be
            no_pnl = s["no_pnl_net"].sum()
            combos.append({
                "ind1": f"{c1}={v1}", "ind2": f"{c2}={v2}",
                "n": len(s), "no_wr": no_wr, "breakeven": be,
                "edge": edge, "no_pnl": no_pnl,
            })

combos_df = pd.DataFrame(combos).sort_values("edge", ascending=False)
print(f"\n  {'combination':>38}  {'n':>6}  {'no_win%':>8}  {'breakeven':>10}  {'edge':>7}  {'pnl($50)':>10}")
print("  " + "-" * 90)
for _, r in combos_df.head(15).iterrows():
    combo_str = f"{r['ind1']} + {r['ind2']}"
    print(f"  {combo_str:>38}  {int(r['n']):>6,}  {r['no_wr']:>7.1%}  "
          f"{r['breakeven']:>9.1%}  {r['edge']:>+6.1%}  ${r['no_pnl']:>+9.0f}")

print(f"\n  Bottom 5 (worst combinations to avoid):")
for _, r in combos_df.tail(5).iterrows():
    combo_str = f"{r['ind1']} + {r['ind2']}"
    print(f"  {combo_str:>38}  {int(r['n']):>6,}  {r['no_wr']:>7.1%}  "
          f"{r['breakeven']:>9.1%}  {r['edge']:>+6.1%}  ${r['no_pnl']:>+9.0f}")


# ── STEP 7: Monthly P&L assuming $50 stake on best-edge NO trade per hour ──────
print(f"\n{SEP}")
print("STEP 7 — Simulated monthly P&L: $50 stake on best-edge NO per hour")
print("         Best-edge = highest (no_win_rate - breakeven) across all offsets")
print(SEP)

monthly_rows = []
for ts_h, grp in df[df["offset"].isin([o for o in NO_OFFSETS])].groupby("ts"):
    no_wr_by_off = []
    for _, row in grp.iterrows():
        no_wr   = 1 - row["yes_won"]
        be      = 1 - row["p_kalshi"]
        no_wr_by_off.append((row["offset"], row["no_pnl_net"], be))
    best = max(no_wr_by_off, key=lambda x: -abs(x[2] - 0.5))
    monthly_rows.append({"ts": ts_h, "pnl": best[1]})

mdf = pd.DataFrame(monthly_rows)
mdf["ym"] = pd.to_datetime(mdf["ts"]).dt.to_period("M")
monthly_pnl = mdf.groupby("ym")["pnl"].sum()

print(f"\n  Note: this takes 1 trade per hour at a fixed $50 stake, no indicator filtering.")
print(f"  {'month':>8}  {'trades':>7}  {'pnl':>10}  {'bar'}")
print("  " + "-" * 50)
for ym, pnl in monthly_pnl.items():
    n_mo = len(mdf[mdf["ym"] == ym])
    bar  = ("+" if pnl > 0 else "-") * min(int(abs(pnl) / 50), 40)
    print(f"  {str(ym):>8}  {n_mo:>7,}  ${pnl:>+9.0f}  {bar}")

total = monthly_pnl.sum()
print(f"\n  Total P&L (1 NO trade/hour, no filtering): ${total:+,.0f}")
print(f"  Profitable months: {(monthly_pnl > 0).sum()} / {len(monthly_pnl)}")
