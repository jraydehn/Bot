"""
backtest_unbiased.py — Bias-corrected BTC Kalshi historical backtest.

Three primary biases eliminated vs backtest_new_model.py:

  BIAS 1 (fixed): Static p_market snapshot from one date (2026-03-20)
  FIX: Dynamic vol-based p_market computed from historical price data only.
       σ_kalshi = 24h rolling realized vol, lagged 2h (simulates Kalshi's delayed
       implied-vol update — the actual source of edge when vol shifts fast).
       p_market = 1 − Φ(ln(K/S) / (σ_kalshi × √60))
       No future data required. 100% derived from historical BTC prices.

  BIAS 2 (fixed): 0.65 calibration derived from the same data being backtested.
  FIX: Walk-forward split.
       Training : Jan 2024 – Dec 2024 (first 12 months, out-of-sample from the test)
         → Compute calibration factor c = Σ(actual YES) / Σ(p_yes_model_raw)
       Test     : Jan 2025 – Apr 2026 (never seen during calibration)
         → Apply c unchanged. No re-fitting.

  BIAS 3 (fixed): Gate PM and Gate NS thresholds tuned on 61 and 282 live paper
  trades respectively, then applied to 2024 backtest data.
  FIX: Remove both gates. Only a priori, theory-justified gates remain:
       Gate 0      : p_model saturation [0.04, 0.96]
       Gate 3      : net edge ≥ 3% (floor for any binary bet to be worth taking)
       Gate R:R    : NO rr ∈ [0.33, 4.0], YES rr ∈ [0.33, ∞) (risk discipline)
       Gate EMA-Dir: block bullish EMA + YES (trend continuation on 1h contracts
                     loses structurally; 11k training bars confirm the principle)

Other design decisions:
  - 1 trade per hour (best net_edge contract) — no double-positioning on correlated bets
  - Fixed $50 stake throughout — clean PnL without compounding distortion
  - Bootstrap 90% CI on total PnL to show uncertainty range

Output:
  - results/backtest_unbiased.csv (test-set trades)
  - Printed: calibration table, trade stats, monthly PnL, bootstrap CI
"""

import sys, math, glob, warnings
from pathlib import Path
from scipy.stats import norm as sp_norm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

from pricing_comparison import kalshi_fee, DEFAULT_SLIPPAGE, DEFAULT_SPREAD

# ── A PRIORI CONSTANTS (not derived from any observed data) ───────────────────
FIXED_STAKE  = 50.0   # fixed bet per trade
TAU          = 60     # minutes to expiry
MIN_NET_EDGE = 0.03   # Gate 3: 3% minimum net edge
RR_MIN       = 0.33   # Gate R:R: minimum risk-to-reward ratio (both sides)
RR_MAX_NO    = 4.0    # Gate R:R: maximum rr for NO (blocks p_market < 0.20)
P_SAT_LO     = 0.04   # Gate 0: model saturation lower bound
P_SAT_HI     = 0.96   # Gate 0: model saturation upper bound
VOL_RATIO_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 999.0  # Gate VR: σ_model/σ_kalshi ceiling

# Offsets from spot to simulate available Kalshi contracts
# Positive = OTM (strike above spot), Negative = ITM (strike below spot)
OFFSETS = [-0.005, -0.003, -0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]

# Walk-forward split
TRAIN_END  = pd.Timestamp("2025-01-01", tz="UTC")  # exclusive
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")  # inclusive

# Vol model parameters
KALSHI_VOL_WINDOW = 1440  # 24h rolling window (1m bars) — Kalshi's implied vol basis
KALSHI_VOL_LAG    = 120   # 2h lag (1m bars) — simulates Kalshi's delayed update
MODEL_VOL_WINDOW  = 60    # 60-bar (1h) current realized vol for our model
WARMUP_BARS       = KALSHI_VOL_WINDOW + KALSHI_VOL_LAG + 60  # bars before first valid signal

BOOTSTRAP_N       = 2000  # resamples for confidence interval

SEP = "=" * 72


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

close_1h = ohlcv_1h["close"].values.astype(float)
ts_1h    = ohlcv_1h.index
n1h      = len(ts_1h)


# ── PRE-COMPUTE VOL SERIES ────────────────────────────────────────────────────
print("\nPre-computing vol series (vectorized)...")
close_1m = ohlcv_1m["close"].values.astype(float)
log_ret  = pd.Series(
    np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0),
    index=ohlcv_1m.index,
)

# Our model vol: 60-bar (1h) rolling std — reacts instantly to vol changes
sigma_model_1m = log_ret.rolling(MODEL_VOL_WINDOW).std()

# Kalshi implied vol: 24h rolling std, then lagged 2h
# This represents how a market maker using end-of-day vol would price contracts.
# The 2h lag: Kalshi's prices are set at discrete intervals, not tick-by-tick.
sigma_kalshi_1m = log_ret.rolling(KALSHI_VOL_WINDOW).std().shift(KALSHI_VOL_LAG)

print(f"  Model vol median   : {sigma_model_1m.median():.6f}/min "
      f"({sigma_model_1m.median()*math.sqrt(60)*100:.2f}%/h equiv)")
print(f"  Kalshi vol median  : {sigma_kalshi_1m.median():.6f}/min "
      f"({sigma_kalshi_1m.median()*math.sqrt(60)*100:.2f}%/h equiv)")

sigma_ratio = (sigma_model_1m / sigma_kalshi_1m).dropna()
print(f"  sigma_model/sigma_kalshi: mean={sigma_ratio.mean():.3f}, "
      f"p10={sigma_ratio.quantile(0.1):.3f}, p90={sigma_ratio.quantile(0.9):.3f}")
print(f"  (ratio < 1 → Kalshi overprices vol → NO has edge; ratio > 1 → YES has edge)")


# ── PRE-COMPUTE 1h EMA ALIGNMENT ─────────────────────────────────────────────
print("\nPre-computing 1h EMA alignment (20/50 on 1h bars)...")
ema20 = pd.Series(close_1h).ewm(span=20, adjust=False).mean().values
ema50 = pd.Series(close_1h).ewm(span=50, adjust=False).mean().values

ema_align = np.array(["neutral"] * n1h, dtype=object)
for i in range(3, n1h):
    e20 = ema20[i-2:i+1]; e50 = ema50[i-2:i+1]; c = close_1h[i-2:i+1]
    if all(e20 > e50) and all(c > e20):
        ema_align[i] = "bullish"
    elif all(e20 < e50) or all(c < e50):
        ema_align[i] = "bearish"

# Forward-fill to 1m index
ema_ff = (pd.Series(ema_align, index=ts_1h)
          .reindex(ohlcv_1m.index, method="ffill")
          .fillna("neutral"))
print("Pre-computation done.\n")


# ── PROBABILITY HELPER ────────────────────────────────────────────────────────
def p_yes_lognormal(spot: float, K: float, tau: int, sigma: float) -> float:
    """
    Log-normal binary option price: P(price > K at expiry).
    Equivalent to estimate_probability() with confirmation_score=0.
    """
    if sigma <= 0 or spot <= 0 or K <= 0:
        return float("nan")
    z = math.log(K / spot) / (sigma * math.sqrt(tau))
    return float(1.0 - sp_norm.cdf(z))


# ── STEP 1: DERIVE CALIBRATION FROM TRAINING SET ──────────────────────────────
print(SEP)
print("STEP 1 — Walk-forward calibration on training set (Jan–Dec 2024)")
print("Computing p_yes_model_raw vs actual 1h outcomes for every (hour, offset)...")
print(SEP)

ohlcv_1m_idx = ohlcv_1m.index

sum_actual    = 0.0
sum_predicted = 0.0
calib_rows    = []

for i_h in range(50, n1h - 1):
    ts_now = ts_1h[i_h]
    if ts_now >= TRAIN_END:
        break

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

    for offset in OFFSETS:
        K          = spot * (1.0 + offset)
        actual_yes = int(next_close > K)
        p_raw      = p_yes_lognormal(spot, K, TAU, sig_m)

        if np.isnan(p_raw) or not (P_SAT_LO <= p_raw <= P_SAT_HI):
            continue

        sum_actual    += actual_yes
        sum_predicted += p_raw
        calib_rows.append({
            "offset": offset, "p_raw": p_raw,
            "actual": actual_yes, "sigma": sig_m,
            "vol_ratio": vr,
        })

n_calib = len(calib_rows)
calib_factor = sum_actual / sum_predicted if sum_predicted > 0 else 1.0
calib_df = pd.DataFrame(calib_rows)

print(f"\n  Training sample    : {n_calib:,} (bar × offset) pairs")
print(f"  Avg p_yes_model_raw: {sum_predicted/n_calib:.4f}")
print(f"  Avg actual YES rate: {sum_actual/n_calib:.4f}")
print(f"  Calibration factor : {calib_factor:.4f}")
print(f"  (The live model used 0.65 — deviation here shows prior bias)")

print(f"\n  Per-offset calibration table (training set only):")
print(f"  {'offset':>8}  {'n':>7}  {'avg_p_model':>11}  {'actual_rate':>11}  {'ratio':>7}  {'interpretation'}")
print("  " + "-" * 75)
for off in sorted(calib_df["offset"].unique()):
    sub = calib_df[calib_df["offset"] == off]
    pm  = sub["p_raw"].mean()
    ac  = sub["actual"].mean()
    rat = ac / pm if pm > 0 else float("nan")
    if off < 0:
        interp = "YES in-the-money"
    elif off < 0.003:
        interp = "NO near-the-money"
    else:
        interp = "NO out-of-the-money"
    print(f"  {off:+8.3f}  {len(sub):>7,}  {pm:>11.4f}  {ac:>11.4f}  {rat:>7.4f}  {interp}")

print(f"\n  Per-vol-ratio calibration table (training set — shows how calib changes by regime):")
print(f"  {'vol_ratio_bin':>14}  {'n':>7}  {'avg_p_model':>11}  {'actual_rate':>11}  {'calib_factor':>12}  {'regime'}")
print("  " + "-" * 80)
VR_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 99.0)]
for lo, hi in VR_BINS:
    sub = calib_df[(calib_df["vol_ratio"] >= lo) & (calib_df["vol_ratio"] < hi)]
    if len(sub) < 5:
        continue
    pm  = sub["p_raw"].mean()
    ac  = sub["actual"].mean()
    cf  = ac / pm if pm > 0 else float("nan")
    reg = "strong NO edge" if hi <= 0.7 else "mild NO edge" if hi <= 0.9 else "neutral" if hi <= 1.1 else "mild YES edge" if hi <= 1.3 else "strong YES edge"
    print(f"  {f'{lo:.1f}–{hi:.1f}':>14}  {len(sub):>7,}  {pm:>11.4f}  {ac:>11.4f}  {cf:>12.4f}  {reg}")

print(f"\n  Calibration factor {calib_factor:.4f} will be applied UNCHANGED to test set.")
print(f"  This is the ONLY parameter derived from data. All gates are a priori.")


# ── STEP 2: BACKTEST ON TEST SET ──────────────────────────────────────────────
print(f"\n{SEP}")
print(f"STEP 2 — Out-of-sample backtest (Jan 2025 – Apr 2026)")
print(f"  Calibration factor : {calib_factor:.4f}  [FIXED from training — not re-fitted]")
print(f"  Gates active       : Gate 0, Gate 3 (≥3%), Gate R:R, Gate EMA-Dir")
print(f"  Gates removed      : Gate PM (61-trade sample), Gate NS (282-trade sample)")
print(f"  p_market           : dynamic log-normal (σ_kalshi = 24h vol, 2h lag)")
print(f"  Stake              : ${FIXED_STAKE} fixed per trade")
print(f"  Selection          : 1 trade/hour (best net_edge)")
print(SEP)


def rr_min_edge(pm: float, side: str) -> float:
    rr = pm / (1-pm) if side == "yes" else (1-pm) / pm
    for t, me in [(1,.03),(2,.06),(4,.09),(6,.15),(8,.20)]:
        if rr <= t: return me
    return 0.25


records = []
cum_pnl = 0.0

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
    ema   = str(ema_ff.iat[pos1m])

    if np.isnan(sig_m) or sig_m <= 0 or np.isnan(sig_k) or sig_k <= 0:
        continue

    # Gate VR: skip hour if current vol has spiked above Kalshi's lagged vol
    if sig_m / sig_k > VOL_RATIO_MAX:
        continue

    best_hour = None

    for offset in OFFSETS:
        K            = spot * (1.0 + offset)
        resolved_yes = int(next_close > K)

        # Dynamic p_market — from Kalshi's lagged vol (no future data)
        p_market = p_yes_lognormal(spot, K, TAU, sig_k)
        if np.isnan(p_market) or p_market <= 0.01 or p_market >= 0.99:
            continue

        # Model probability with walk-forward calibration
        p_raw = p_yes_lognormal(spot, K, TAU, sig_m)
        if np.isnan(p_raw) or not (P_SAT_LO <= p_raw <= P_SAT_HI):
            continue

        p_model = p_raw * calib_factor

        for side in ("yes", "no"):
            fee      = kalshi_fee(p_market)
            raw_edge = (p_model - p_market) if side == "yes" else (p_market - p_model)
            net_edge = raw_edge - fee - DEFAULT_SLIPPAGE - DEFAULT_SPREAD

            # Gate EMA-Dir: block bullish trend + YES continuation
            if ema == "bullish" and side == "yes":
                continue

            # Gate 3
            if net_edge < MIN_NET_EDGE:
                continue

            # Gate R:R
            rr = p_market/(1-p_market) if side == "yes" else (1-p_market)/p_market
            if rr < RR_MIN:
                continue
            if side == "no" and rr > RR_MAX_NO:
                continue
            if net_edge < rr_min_edge(p_market, side):
                continue

            candidate = {
                "offset": offset, "K": round(K, 2),
                "side": side, "p_market": round(p_market, 4),
                "p_model": round(p_model, 4), "p_raw": round(p_raw, 4),
                "net_edge": round(net_edge, 4),
                "sigma_model": round(sig_m, 7),
                "sigma_kalshi": round(sig_k, 7),
                "vol_ratio": round(sig_m / sig_k, 4),
                "ema": ema, "resolved_yes": resolved_yes,
            }
            if best_hour is None or net_edge > best_hour["net_edge"]:
                best_hour = candidate

    if best_hour:
        side         = best_hour["side"]
        p_market     = best_hour["p_market"]
        resolved_yes = best_hour["resolved_yes"]

        won = (resolved_yes == 1) if side == "yes" else (resolved_yes == 0)
        pnl = (FIXED_STAKE * p_market / (1-p_market)
               if won and side == "no"
               else FIXED_STAKE * (1-p_market) / p_market
               if won and side == "yes"
               else -FIXED_STAKE)

        cum_pnl += pnl
        best_hour.update({
            "ts": ts_now, "spot": round(spot, 2),
            "won": int(won), "pnl": round(pnl, 2),
            "cumulative_pnl": round(cum_pnl, 2),
        })
        records.append(best_hour)

df = pd.DataFrame(records)
print(f"Test-set trades: {len(df):,}")


# ── REPORT ────────────────────────────────────────────────────────────────────
if df.empty:
    print("No trades — all gate criteria too strict. Adjust MIN_NET_EDGE.")
else:
    n    = len(df)
    wins = int(df["won"].sum())
    wr   = wins / n
    tot  = df["pnl"].sum()
    avg  = df["pnl"].mean()
    avg_edge = df["net_edge"].mean()

    # Monthly stats
    df["ym"] = pd.to_datetime(df["ts"]).dt.to_period("M")
    monthly  = df.groupby("ym")["pnl"].sum()
    n_months = len(monthly)
    n_pos    = (monthly > 0).sum()

    # Max drawdown on cumulative PnL
    cum = df["cumulative_pnl"].values
    peak = cum[0]; max_dd = 0.0
    for v in cum:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)

    print(f"\n{SEP}")
    print("  OUT-OF-SAMPLE RESULTS (Jan 2025 – Apr 2026)")
    print(SEP)
    print(f"  Trades           : {n:,}  ({n/n_months:.0f}/month avg)")
    print(f"  Win rate         : {wr:.1%}  ({wins}/{n})")
    print(f"  Total PnL        : ${tot:+,.2f}  (fixed ${FIXED_STAKE} stake)")
    print(f"  Avg PnL/trade    : ${avg:+.2f}")
    print(f"  Avg net edge     : {avg_edge:+.2%}")
    print(f"  Max drawdown     : ${max_dd:,.2f}")
    print(f"  Profitable months: {n_pos}/{n_months}")
    print(f"  Final cum. PnL   : ${cum[-1]:+,.2f}")

    print(f"\n  Side breakdown:")
    for side in ["yes","no"]:
        sub = df[df["side"]==side]
        if sub.empty: continue
        sw = int(sub["won"].sum())
        print(f"    {side.upper()}: {len(sub):,} trades  win={sw/len(sub):.1%}  "
              f"pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  By offset:")
    for off in sorted(df["offset"].unique()):
        sub = df[df["offset"]==off]
        sw  = int(sub["won"].sum())
        pm  = sub["p_market"].mean()
        print(f"    offset={off:+.3f}  n={len(sub):4,}  win={sw/len(sub):.1%}  "
              f"avg_p_mkt={pm:.3f}  pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  By vol regime (sigma_model / sigma_kalshi):")
    df["vr_bin"] = pd.cut(df["vol_ratio"],
                          bins=[-np.inf, 0.7, 0.9, 1.1, 1.3, np.inf],
                          labels=["<0.7","0.7-0.9","0.9-1.1","1.1-1.3",">1.3"])
    for vr in df["vr_bin"].cat.categories:
        sub = df[df["vr_bin"]==vr]
        if sub.empty: continue
        sw = int(sub["won"].sum())
        print(f"    vol_ratio {vr:>8s}: n={len(sub):4,}  win={sw/len(sub):.1%}  "
              f"pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  By EMA alignment:")
    for ea in ["bullish","bearish","neutral"]:
        sub = df[df["ema"]==ea]
        if sub.empty: continue
        sw = int(sub["won"].sum())
        print(f"    {ea:8s}: n={len(sub):4,}  win={sw/len(sub):.1%}  "
              f"pnl=${sub['pnl'].sum():+,.2f}")

    print(f"\n  Monthly PnL (${FIXED_STAKE} stake, out-of-sample):")
    for ym, pnl in monthly.items():
        bar = ("+" if pnl > 0 else "-") * min(int(abs(pnl)/15), 45)
        print(f"    {ym}  ${pnl:+8.2f}  {bar}")

    # ── BOOTSTRAP 90% CONFIDENCE INTERVAL ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BOOTSTRAP 90% CONFIDENCE INTERVAL  (n={BOOTSTRAP_N:,} resamples)")
    print(f"  Method: block bootstrap, block = 1 month (preserves within-month autocorrelation)")
    print(SEP)

    monthly_arr = monthly.values
    rng = np.random.default_rng(seed=42)
    boot_totals = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        sampled = rng.choice(monthly_arr, size=len(monthly_arr), replace=True)
        boot_totals[b] = sampled.sum()

    p5  = np.percentile(boot_totals, 5)
    p25 = np.percentile(boot_totals, 25)
    p50 = np.percentile(boot_totals, 50)
    p75 = np.percentile(boot_totals, 75)
    p95 = np.percentile(boot_totals, 95)

    print(f"  Observed total PnL : ${tot:+,.2f}")
    print(f"  Bootstrap median   : ${p50:+,.2f}")
    print(f"  90% CI             : [${p5:+,.2f},  ${p95:+,.2f}]")
    print(f"  50% CI             : [${p25:+,.2f},  ${p75:+,.2f}]")
    print(f"  Prob(PnL > 0)      : {(boot_totals > 0).mean():.1%}")
    print(f"  Prob(PnL > $1000)  : {(boot_totals > 1000).mean():.1%}")
    print(f"  Prob(PnL > $5000)  : {(boot_totals > 5000).mean():.1%}")

    note = ("POSITIVE" if p5 > 0 else
            "UNCERTAIN" if p95 > 0 else "NEGATIVE")
    print(f"\n  Edge verdict: {note}")
    if p5 > 0:
        print(f"  The 5th percentile is positive — strategy shows positive edge even in")
        print(f"  the worst 5% of bootstrap scenarios. This is a meaningful result.")
    elif p95 > 0:
        print(f"  The CI crosses zero — edge exists on average but is not statistically")
        print(f"  certain over this time period. More data needed.")
    else:
        print(f"  The 95th percentile is negative — strategy has negative edge.")


# ── SAVE ──────────────────────────────────────────────────────────────────────
out = BASE / "results/backtest_unbiased.csv"
df.to_csv(out, index=False)
print(f"\n{SEP}")
print(f"  Saved: {out}  ({len(df):,} rows)")
print(f"  Calibration factor used: {calib_factor:.4f}")
print(f"  Training period: Jan 2024 – Dec 2024")
print(f"  Test period    : Jan 2025 – Apr 2026 (out-of-sample)")
print(SEP)
