"""
backtest_drift_model.py — Combined vol + directional drift model.

Standard log-normal assumes zero drift (pure random walk):
    p_yes = 1 - Φ( log(K/S) / (σ × √60) )

This model adds a drift term estimated from directional indicators:
    p_yes = 1 - Φ( (log(K/S) - μ_drift) / (σ × √60) )

μ_drift is the expected 60-minute BTC log-return conditioned on the
current technical state. It is estimated via OLS on the training set
(2024) and applied unchanged to the test set (2025–Apr 2026).

Features used to estimate μ_drift:
    rsi_norm       = (RSI_14 - 50) / 50           → [-1, +1]
    ema_bias       = +1 bullish / 0 neutral / -1 bearish
    bb_pos         = (close - BB_lower) / BB_width → [0, 1], position within band
    momentum_30m   = log(S_now / S_30m_ago)        → recent price momentum
    momentum_4h    = log(S_now / S_4h_ago)         → medium-term trend
    vwap_dev       = (S - VWAP_24h) / VWAP_24h     → mean-reversion signal
    vol_mom        = σ_60m / σ_60m_4h_ago           → vol regime shift

Kalshi pricing: calibrated from 236 actual paper-trade observations.
    σ_eff_kalshi = 0.4717 × σ_model + 0.0001644

Walk-forward split:
    Train : Jan 2024 – Dec 2024   (fit β coefficients)
    Test  : Jan 2025 – Apr 2026   (apply unchanged, evaluate P&L)

Output:
    - Regression coefficients + t-stats (are the signals real?)
    - Per-offset win rates and P&L with and without drift
    - Per-indicator-regime P&L (which states add value?)
    - Monthly P&L curve
    - Saved: results/backtest_drift.csv
"""

import sys, math, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
from scipy import stats as scipy_stats
from numpy.linalg import lstsq

warnings.filterwarnings("ignore")
BASE = Path("/Users/justindehn/Documents/ClaudeCode/kalshi_btc")
sys.path.insert(0, str(BASE))

SEP = "=" * 72

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_END  = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

MODEL_VOL_WINDOW  = 60
KALSHI_VOL_WINDOW = 1440
KALSHI_VOL_LAG    = 120
WARMUP_BARS       = KALSHI_VOL_WINDOW + KALSHI_VOL_LAG + 60
TAU               = 60

# Kalshi pricing calibration from 236 real paper-trade observations
KALSHI_SLOPE     = 0.4717
KALSHI_INTERCEPT = 0.0001644

KALSHI_RAKE    = 0.07
FIXED_STAKE    = 50.0
MIN_NET_EDGE   = 0.03     # 3% minimum to trade
MIN_N_REPORT   = 30

NO_OFFSETS  = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]
YES_OFFSETS = [-0.001, -0.0015, -0.002, -0.0025, -0.003]
ALL_OFFSETS = YES_OFFSETS + NO_OFFSETS


# ── Load data ─────────────────────────────────────────────────────────────────
print(SEP)
print("Loading price data...")
print(SEP)

files_1m = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1m_2024-01-01_*.parquet")))
files_1h = sorted(glob.glob(str(BASE / "data/binanceus_BTCUSDT_1h_2024-01-01_*.parquet")))

ohlcv_1m = pd.read_parquet(files_1m[-1])
ohlcv_1m.index = pd.to_datetime(ohlcv_1m.index, utc=True)
ohlcv_1m = ohlcv_1m.sort_index()

ohlcv_1h = pd.read_parquet(files_1h[-1])
ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index, utc=True)
ohlcv_1h = ohlcv_1h.sort_index()

close_1h = ohlcv_1h["close"].values.astype(float)
ts_1h    = ohlcv_1h.index
n1h      = len(ts_1h)
print(f"  1h bars: {n1h:,}  ({ts_1h[0].date()} → {ts_1h[-1].date()})")

close_1m = ohlcv_1m["close"].values.astype(float)
high_1m  = ohlcv_1m["high"].values.astype(float)
low_1m   = ohlcv_1m["low"].values.astype(float)
vol_1m   = ohlcv_1m["volume"].values.astype(float)
ohlcv_1m_idx = ohlcv_1m.index

log_ret_1m = pd.Series(
    np.diff(np.log(np.maximum(close_1m, 1e-8)), prepend=0.0),
    index=ohlcv_1m.index,
)
sigma_model_1m  = log_ret_1m.rolling(MODEL_VOL_WINDOW).std()
sigma_kalshi_1m = log_ret_1m.rolling(KALSHI_VOL_WINDOW).std().shift(KALSHI_VOL_LAG)
sigma_model_lag240 = sigma_model_1m.shift(240)   # 4h ago vol


# ── Pre-compute indicators on 1m bars (sampled at 1h) ─────────────────────────
print("Computing indicators...")

# RSI-14 on 1h close
delta_1h = pd.Series(close_1h, index=ts_1h).diff()
gain_1h  = delta_1h.clip(lower=0).ewm(com=13, adjust=False).mean()
loss_1h  = (-delta_1h.clip(upper=0)).ewm(com=13, adjust=False).mean()
rsi_1h   = 100 - (100 / (1 + gain_1h / loss_1h.replace(0, 1e-10)))

# EMA 20/50 on 1h
ema20_1h = pd.Series(close_1h, index=ts_1h).ewm(span=20, adjust=False).mean()
ema50_1h = pd.Series(close_1h, index=ts_1h).ewm(span=50, adjust=False).mean()

# BB on 1h (20-period)
bb_mid_1h = pd.Series(close_1h, index=ts_1h).rolling(20).mean()
bb_std_1h = pd.Series(close_1h, index=ts_1h).rolling(20).std()
bb_upper  = bb_mid_1h + 2 * bb_std_1h
bb_lower  = bb_mid_1h - 2 * bb_std_1h
bb_width  = bb_upper - bb_lower
# BB position: 0 = at lower band, 1 = at upper band
bb_pos_1h = ((pd.Series(close_1h, index=ts_1h) - bb_lower) / bb_width.replace(0, float("nan")))
bb_pos_1h = bb_pos_1h.clip(0, 1)
bb_pct_1h = (bb_width / bb_mid_1h).rolling(200).rank(pct=True)  # width percentile

# 30m and 4h momentum (using 1m bars → sample at each 1h bar)
# Momentum = log(S_now / S_Nm_ago)
close_1m_s = pd.Series(close_1m, index=ohlcv_1m_idx)
mom_30m_1m = np.log(close_1m_s / close_1m_s.shift(30).replace(0, float("nan")))
mom_4h_1m  = np.log(close_1m_s / close_1m_s.shift(240).replace(0, float("nan")))
mom_30m_1h = mom_30m_1m.reindex(ts_1h, method="ffill")
mom_4h_1h  = mom_4h_1m.reindex(ts_1h, method="ffill")

# 24h VWAP on 1m bars → sample at 1h
vwap_num_1m = (close_1m_s * pd.Series(vol_1m, index=ohlcv_1m_idx)).rolling(1440).sum()
vwap_den_1m = pd.Series(vol_1m, index=ohlcv_1m_idx).rolling(1440).sum().replace(0, float("nan"))
vwap_1m     = vwap_num_1m / vwap_den_1m
vwap_dev_1h = ((close_1m_s - vwap_1m) / vwap_1m).reindex(ts_1h, method="ffill")

# Vol momentum at 1h
sigma_model_1h   = sigma_model_1m.reindex(ts_1h, method="ffill")
sigma_lag240_1h  = sigma_model_lag240.reindex(ts_1h, method="ffill")
vol_mom_1h       = sigma_model_1h / sigma_lag240_1h.replace(0, float("nan"))
sigma_kalshi_1h  = sigma_kalshi_1m.reindex(ts_1h, method="ffill")

print("  Done.")


# ── Build feature matrix ───────────────────────────────────────────────────────
print("Building feature matrix...")

rows_train = []
rows_test  = []

for i_h in range(50, n1h - 1):
    ts_now     = ts_1h[i_h]
    spot       = float(close_1h[i_h])
    next_close = float(close_1h[i_h + 1])
    ret_1h     = math.log(next_close / spot) if spot > 0 and next_close > 0 else float("nan")

    pos1m = int(ohlcv_1m_idx.searchsorted(ts_now, side="right")) - 1
    if pos1m < WARMUP_BARS:
        continue

    sig_m = float(sigma_model_1h.iat[i_h])
    sig_k = float(sigma_kalshi_1h.iat[i_h])
    if not (sig_m > 0 and sig_k > 0 and not math.isnan(ret_1h)):
        continue

    rsi_v    = float(rsi_1h.iat[i_h])
    ema20_v  = float(ema20_1h.iat[i_h])
    ema50_v  = float(ema50_1h.iat[i_h])
    bb_pos_v = float(bb_pos_1h.iat[i_h]) if not np.isnan(bb_pos_1h.iat[i_h]) else 0.5
    bb_pct_v = float(bb_pct_1h.iat[i_h]) if not np.isnan(bb_pct_1h.iat[i_h]) else 0.5
    mom30_v  = float(mom_30m_1h.iat[i_h]) if not np.isnan(mom_30m_1h.iat[i_h]) else 0.0
    mom4h_v  = float(mom_4h_1h.iat[i_h])  if not np.isnan(mom_4h_1h.iat[i_h])  else 0.0
    vwap_v   = float(vwap_dev_1h.iat[i_h]) if not np.isnan(vwap_dev_1h.iat[i_h]) else 0.0
    vm_v     = float(vol_mom_1h.iat[i_h])  if not np.isnan(vol_mom_1h.iat[i_h])  else 1.0

    # Derived directional features
    rsi_norm  = (rsi_v - 50.0) / 50.0   # -1 to +1
    if math.isnan(rsi_norm): rsi_norm = 0.0

    if ema20_v > ema50_v and spot > ema20_v:
        ema_bias = 1.0
        ema_str  = "bullish"
    elif ema20_v < ema50_v or spot < ema50_v:
        ema_bias = -1.0
        ema_str  = "bearish"
    else:
        ema_bias = 0.0
        ema_str  = "neutral"

    rsi_str = "oversold" if rsi_v < 30 else "overbought" if rsi_v > 70 else "neutral"
    bb_str  = "squeeze"   if bb_pct_v < 0.20 else "expansion" if bb_pct_v > 0.80 else "normal"
    vm_str  = "contracting" if vm_v < 0.80 else "expanding" if vm_v > 1.25 else "stable"

    row = {
        "ts":         ts_now,
        "spot":       spot,
        "next_close": next_close,
        "ret_1h":     ret_1h,
        "sig_m":      sig_m,
        "sig_k":      sig_k,
        # Raw features
        "rsi":        rsi_v,
        "rsi_norm":   rsi_norm,
        "ema_bias":   ema_bias,
        "bb_pos":     bb_pos_v,
        "bb_pct":     bb_pct_v,
        "mom_30m":    mom30_v,
        "mom_4h":     mom4h_v,
        "vwap_dev":   vwap_v,
        "vol_mom":    vm_v,
        # Categorical states
        "ema_str":    ema_str,
        "rsi_str":    rsi_str,
        "bb_str":     bb_str,
        "vm_str":     vm_str,
    }

    if ts_now < TRAIN_END:
        rows_train.append(row)
    elif ts_now >= TEST_START:
        rows_test.append(row)

df_train = pd.DataFrame(rows_train)
df_test  = pd.DataFrame(rows_test)
print(f"  Train: {len(df_train):,} hours  |  Test: {len(df_test):,} hours")


# ── Step 1: Calibrate drift via OLS on training set ───────────────────────────
print(f"\n{SEP}")
print("STEP 1 — Calibrate directional drift on training set (2024)")
print("  Regress 1h log-return on technical features")
print("  Target: μ_drift = expected 60-min BTC log-return given indicator state")
print(SEP)

FEATURES = ["rsi_norm", "ema_bias", "bb_pos", "mom_30m", "mom_4h", "vwap_dev"]
FEAT_LABELS = {
    "rsi_norm":  "RSI (normalized -1→+1)",
    "ema_bias":  "EMA alignment (+1=bull, -1=bear)",
    "bb_pos":    "BB position (0=lower,1=upper band)",
    "mom_30m":   "30m price momentum (log return)",
    "mom_4h":    "4h price momentum (log return)",
    "vwap_dev":  "VWAP deviation (price/VWAP - 1)",
}

X_train = df_train[FEATURES].values
y_train = df_train["ret_1h"].values

# OLS via numpy lstsq
X_aug_fit = np.column_stack([np.ones(len(X_train)), X_train])
beta_fit, _, _, _ = lstsq(X_aug_fit, y_train, rcond=None)
y_pred_train = X_aug_fit @ beta_fit

ss_res = np.sum((y_train - y_pred_train) ** 2)
ss_tot = np.sum((y_train - y_train.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

# t-stats
X_aug = X_aug_fit
beta_ols = beta_fit
residuals = y_train - X_aug @ beta_ols
mse = np.sum(residuals**2) / (len(y_train) - len(beta_ols))
var_beta = mse * np.linalg.inv(X_aug.T @ X_aug).diagonal()
se_beta = np.sqrt(np.maximum(var_beta, 1e-20))
t_stats = beta_ols / se_beta

print(f"\n  Training R²: {r2:.4f}  (low is expected — returns are noisy)")
print(f"  Intercept (baseline drift): {beta_ols[0]*100:+.4f}% per hour")
print(f"\n  {'Feature':>35}  {'β (log-return/unit)':>20}  {'β (% drift)':>12}  {'t-stat':>8}  {'signal?'}")
print("  " + "-" * 90)
for i, feat in enumerate(FEATURES):
    b  = beta_ols[i+1]
    t  = t_stats[i+1]
    sig = "YES ★" if abs(t) > 2.0 else "marginal" if abs(t) > 1.5 else "no"
    print(f"  {FEAT_LABELS[feat]:>35}  {b:>20.6f}  {b*100:>+11.4f}%  {t:>+8.2f}  {sig}")

# Predict drift for each training and test hour
X_test_aug = np.column_stack([np.ones(len(df_test)), df_test[FEATURES].values])
df_train["mu_drift"] = X_aug_fit @ beta_fit
df_test["mu_drift"]  = X_test_aug @ beta_fit

print(f"\n  μ_drift stats (test set):")
print(f"    mean={df_test['mu_drift'].mean()*100:+.4f}%  "
      f"std={df_test['mu_drift'].std()*100:.4f}%  "
      f"min={df_test['mu_drift'].min()*100:+.4f}%  "
      f"max={df_test['mu_drift'].max()*100:+.4f}%")
print(f"  (compare to realized mean 1h return: {df_test['ret_1h'].mean()*100:+.4f}%)")


# ── Step 2: Pricing functions ─────────────────────────────────────────────────
def p_yes_base(spot, K, sigma):
    """Standard log-normal — zero drift."""
    if sigma <= 0 or spot <= 0 or K <= 0:
        return float("nan")
    z = math.log(K / spot) / (sigma * math.sqrt(TAU))
    return float(1.0 - sp_norm.cdf(z))

def p_yes_drift(spot, K, sigma, mu):
    """Drift-adjusted log-normal."""
    if sigma <= 0 or spot <= 0 or K <= 0:
        return float("nan")
    z = (math.log(K / spot) - mu) / (sigma * math.sqrt(TAU))
    return float(1.0 - sp_norm.cdf(z))

def p_kalshi_sim(spot, K, sigma_model):
    """Realistic Kalshi price from empirical calibration."""
    sig_eff = max(KALSHI_SLOPE * sigma_model + KALSHI_INTERCEPT, sigma_model * 0.4)
    return p_yes_base(spot, K, sig_eff)


# ── Step 3: Backtest on test set ──────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — Backtest on test set (Jan 2025 – Apr 2026)")
print(f"  p_yes_model = 1 - Φ((log(K/S) - μ_drift) / (σ_model × √60))")
print(f"  p_kalshi    = calibrated from 236 real price observations")
print(f"  Edge        = p_yes_model - p_kalshi (YES) or p_kalshi - p_yes_model (NO)")
print(f"  Trades      = 1 per hour (best net edge across all offsets), min edge {MIN_NET_EDGE:.0%}")
print(SEP)

trade_records = []
all_records   = []
cum_pnl = 0.0

for _, row in df_test.iterrows():
    spot       = row["spot"]
    next_close = row["next_close"]
    sig_m      = row["sig_m"]
    mu         = row["mu_drift"]

    best = None

    for offset in ALL_OFFSETS:
        K          = spot * (1.0 + offset)
        resolved_yes = int(next_close > K)

        # Kalshi realistic price (no drift — market maker doesn't know our drift)
        pk = p_kalshi_sim(spot, K, sig_m)
        if math.isnan(pk) or pk <= 0.02 or pk >= 0.98:
            continue

        # Our drift-adjusted model probability
        pm = p_yes_drift(spot, K, sig_m, mu)
        if math.isnan(pm) or pm <= 0.01 or pm >= 0.99:
            continue

        fee = KALSHI_RAKE * pk * (1 - pk)

        for side in ("yes", "no"):
            raw_edge = (pm - pk) if side == "yes" else (pk - pm)
            net_edge = raw_edge - fee - 0.005 - 0.010   # slippage + spread

            # R:R gate
            rr = pk / (1-pk) if side == "yes" else (1-pk) / pk
            if rr < 0.33 or (side == "no" and rr > 4.0):
                continue

            if net_edge < MIN_NET_EDGE:
                continue

            if best is None or net_edge > best["net_edge"]:
                won = (resolved_yes == 1) if side == "yes" else (resolved_yes == 0)
                payout = (1-pk)/pk if side == "yes" else pk/(1-pk)
                pnl = FIXED_STAKE * payout if won else -FIXED_STAKE
                best = {
                    "ts":         row["ts"],
                    "offset":     offset,
                    "side":       side,
                    "spot":       round(spot, 2),
                    "K":          round(K, 2),
                    "sig_m":      round(sig_m, 7),
                    "mu_drift":   round(mu, 6),
                    "p_model":    round(pm, 4),
                    "p_kalshi":   round(pk, 4),
                    "net_edge":   round(net_edge, 4),
                    "resolved_yes": resolved_yes,
                    "won":        int(won),
                    "pnl":        round(pnl, 2),
                    "ema":        row["ema_str"],
                    "rsi":        row["rsi_str"],
                    "bb":         row["bb_str"],
                    "vm":         row["vm_str"],
                }

        all_records.append({
            "ts": row["ts"], "offset": offset,
            "p_kalshi": round(pk, 4) if not math.isnan(pk) else None,
            "resolved_yes": resolved_yes,
        })

    if best:
        cum_pnl += best["pnl"]
        best["cum_pnl"] = round(cum_pnl, 2)
        trade_records.append(best)

df_trades = pd.DataFrame(trade_records)
print(f"\n  Total trades taken: {len(df_trades):,}  ({len(df_trades)/len(df_test)*100:.1f}% of hours)")


# ── Step 4: Results ───────────────────────────────────────────────────────────
if df_trades.empty:
    print("  No trades passed all gates. Reduce MIN_NET_EDGE.")
    sys.exit()

n    = len(df_trades)
wins = df_trades["won"].sum()
wr   = wins / n
tot  = df_trades["pnl"].sum()
avg  = df_trades["pnl"].mean()

print(f"\n{SEP}")
print("  OUT-OF-SAMPLE RESULTS  (Jan 2025 – Apr 2026, drift-adjusted model)")
print(SEP)
print(f"  Trades      : {n:,}")
print(f"  Win rate    : {wr:.1%}  ({wins}/{n})")
print(f"  Total P&L   : ${tot:+,.2f}")
print(f"  Avg P&L/trade: ${avg:+.2f}")
print(f"  Avg net edge : {df_trades['net_edge'].mean():+.2%}")

# Max drawdown
cum = df_trades["cum_pnl"].values
peak = cum[0]; mdd = 0.0
for v in cum:
    peak = max(peak, v); mdd = max(mdd, peak - v)
print(f"  Max drawdown: ${mdd:,.2f}")

# By side
print(f"\n  By side:")
for side in ["yes","no"]:
    sub = df_trades[df_trades["side"]==side]
    if sub.empty: continue
    sw = sub["won"].sum()
    print(f"    {side.upper()}: {len(sub):,} trades  win={sw/len(sub):.1%}  pnl=${sub['pnl'].sum():+,.2f}")

# By offset
print(f"\n  By offset:")
for off in sorted(df_trades["offset"].unique()):
    sub = df_trades[df_trades["offset"]==off]
    sw  = sub["won"].sum()
    pm  = sub["p_kalshi"].mean()
    print(f"    {off:+.4f}: n={len(sub):4,}  win={sw/len(sub):.1%}  avg_p_kalshi={pm:.3f}  "
          f"pnl=${sub['pnl'].sum():+,.2f}")

# By indicator regime
print(f"\n  By EMA alignment:")
for st in ["bullish","neutral","bearish"]:
    sub = df_trades[df_trades["ema"]==st]
    if len(sub) < 5: continue
    sw = sub["won"].sum()
    print(f"    {st:8s}: n={len(sub):4,}  win={sw/len(sub):.1%}  pnl=${sub['pnl'].sum():+,.2f}")

print(f"\n  By RSI state:")
for st in ["oversold","neutral","overbought"]:
    sub = df_trades[df_trades["rsi"]==st]
    if len(sub) < 5: continue
    sw = sub["won"].sum()
    print(f"    {st:10s}: n={len(sub):4,}  win={sw/len(sub):.1%}  pnl=${sub['pnl'].sum():+,.2f}")

print(f"\n  By BB state:")
for st in ["squeeze","normal","expansion"]:
    sub = df_trades[df_trades["bb"]==st]
    if len(sub) < 5: continue
    sw = sub["won"].sum()
    print(f"    {st:10s}: n={len(sub):4,}  win={sw/len(sub):.1%}  pnl=${sub['pnl'].sum():+,.2f}")

# Monthly P&L
df_trades["ym"] = pd.to_datetime(df_trades["ts"]).dt.to_period("M")
monthly = df_trades.groupby("ym")["pnl"].sum()
n_months = len(monthly)
n_pos = (monthly > 0).sum()

print(f"\n  Monthly P&L (${FIXED_STAKE} stake, out-of-sample):")
for ym, pnl in monthly.items():
    n_mo = len(df_trades[df_trades["ym"]==ym])
    bar  = ("+" if pnl > 0 else "-") * min(int(abs(pnl)/30), 45)
    print(f"    {ym}  n={n_mo:4,}  ${pnl:+8.2f}  {bar}")

print(f"\n  Profitable months: {n_pos}/{n_months}")
print(f"  Final cumulative P&L: ${cum[-1]:+,.2f}")

# ── Step 5: Compare drift vs no-drift on same test set ────────────────────────
print(f"\n{SEP}")
print("STEP 3 — Compare: drift model vs zero-drift model on same test set")
print(SEP)

nodrift_records = []
cum2 = 0.0
for _, row in df_test.iterrows():
    spot = row["spot"]; next_close = row["next_close"]; sig_m = row["sig_m"]
    best = None
    for offset in ALL_OFFSETS:
        K = spot * (1.0 + offset)
        pk = p_kalshi_sim(spot, K, sig_m)
        if math.isnan(pk) or pk <= 0.02 or pk >= 0.98: continue
        pm = p_yes_base(spot, K, sig_m)   # zero drift
        if math.isnan(pm) or pm <= 0.01 or pm >= 0.99: continue
        fee = KALSHI_RAKE * pk * (1 - pk)
        for side in ("yes","no"):
            raw_edge = (pm - pk) if side == "yes" else (pk - pm)
            net_edge = raw_edge - fee - 0.005 - 0.010
            rr = pk/(1-pk) if side == "yes" else (1-pk)/pk
            if rr < 0.33 or (side == "no" and rr > 4.0): continue
            if net_edge < MIN_NET_EDGE: continue
            resolved_yes = int(next_close > K)
            won = (resolved_yes==1) if side=="yes" else (resolved_yes==0)
            payout = (1-pk)/pk if side=="yes" else pk/(1-pk)
            pnl = FIXED_STAKE*payout if won else -FIXED_STAKE
            if best is None or net_edge > best["net_edge"]:
                best = {"won": int(won), "pnl": round(pnl,2), "net_edge": net_edge}
    if best:
        cum2 += best["pnl"]
        nodrift_records.append(best)

df_nd = pd.DataFrame(nodrift_records)
if not df_nd.empty:
    print(f"\n  Zero-drift model : {len(df_nd):,} trades  "
          f"win={df_nd['won'].mean():.1%}  pnl=${df_nd['pnl'].sum():+,.2f}")
    print(f"  Drift model      : {len(df_trades):,} trades  "
          f"win={wr:.1%}  pnl=${tot:+,.2f}")
    improvement = tot - df_nd["pnl"].sum()
    print(f"  Drift improvement: ${improvement:+,.2f}  "
          f"({'better' if improvement > 0 else 'worse'} with drift)")

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "results/backtest_drift.csv"
df_trades.to_csv(out, index=False)
print(f"\n{SEP}")
print(f"  Saved: {out}  ({len(df_trades):,} trades)")
print(f"  Training period: Jan 2024 – Dec 2024")
print(f"  Test period    : Jan 2025 – Apr 2026")
print(SEP)
