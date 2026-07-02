"""
SOL 15m YES allow-gate rescue search.
Block condition: stoch_k_15m < 30 OR cvd_4h < 0
Goal: find rescue conditions within the blocked group that have WR > breakeven.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import product

CSV = "results/paper_trades_sol15m.csv"
FLAT_RISK = 30.0

df = pd.read_csv(CSV, low_memory=False)
df = df[df["asset"] == "SOL"].copy()
df = df[df["side"] == "yes"].copy()

# Resolve
df["won"] = pd.to_numeric(df["resolved_yes"], errors="coerce") == 1.0
df["pm"] = pd.to_numeric(df["p_market"], errors="coerce")
df["pnl"] = df.apply(
    lambda r: FLAT_RISK * (1 - r["pm"]) / r["pm"] if r["won"] else -FLAT_RISK, axis=1
)

# Only rows with known outcome
df = df.dropna(subset=["pm", "won"])
df = df[df["pm"].between(0.01, 0.99)]
df["week"] = pd.to_datetime(df["logged_at"]).dt.isocalendar().week

total = len(df)
print(f"Total SOL YES with outcome: {total}")

# Coerce key columns
for col in ["stoch_k_15m", "cvd_4h", "stoch_k_5m", "stoch_k_1h", "stoch_k_4h",
            "chg_1m", "chg_5m", "chg_15m", "chg_1h", "chg_4h",
            "bp_5m", "bp_15m", "bp_1h", "bp_4h",
            "dir_5m", "dir_15m", "dir_1h", "consec_dir_15m", "consec_dir_1h",
            "offset_pct", "regime_z", "ema_bias", "ema_bias_1h",
            "ema20_dist_1h", "ema50_dist_1h",
            "vol_ratio", "vol_ratio_1h", "vol_ratio_5m",
            "realized_vol_annual", "arima_forecast_1h",
            "donchian_breakout_1h", "bb_pct_1h", "kc_pct_1h", "kc_bo_1h",
            "markov_regime_1h", "markov_sol_1h", "markov_sol_4h", "markov_sol_6h",
            "hurst_exponent", "autocorr1_15", "autocorr1_30",
            "ou_theta", "ou_halflife", "ou_mu_distance",
            "kalman_velocity", "kalman_residual",
            "liq_score", "liq_bias", "oi_chg_pct", "ls_long_pct",
            "fear_greed", "cg_composite", "cg_futures_ratio_4h", "cg_futures_cvd_12h",
            "cg_futures_delta_4h", "rsi_1h", "rsi_4h", "macd_hist_1h",
            "mu6h", "mu12h", "mu24h", "z_drift_6h",
            "composite_p_up", "p_gbdt",
            "atr_ratio_15m", "range_ratio_15m",
            "upper_wick_15m", "lower_wick_15m", "body_15m",
            "stoch_cross_1h", "engulfing_1h"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Block mask
block_mask = (df["stoch_k_15m"] < 30) | (df["cvd_4h"] < 0)
blocked = df[block_mask].copy()
allowed = df[~block_mask].copy()
print(f"Block condition fires on: {block_mask.sum()} bets")
print(f"  Blocked WR: {blocked['won'].mean()*100:.1f}% (BE: {(blocked['pm']/(1+blocked['pm'])).mean()*100:.1f}%), PnL: ${blocked['pnl'].sum():+.0f}")
print(f"  Allowed WR: {allowed['won'].mean()*100:.1f}%, PnL: ${allowed['pnl'].sum():+.0f}")

def mcpt_p(mask_rescue, n_perm=4000):
    """One-sided MCPT p-value: p(permuted_wr >= obs_wr)."""
    subset = blocked[mask_rescue]
    if len(subset) < 8:
        return np.nan, np.nan, np.nan
    obs_wr = subset["won"].mean()
    obs_pnl = subset["pnl"].sum()
    obs_n = len(subset)
    perm_wrs = []
    rng = np.random.default_rng(42)
    labels = blocked["won"].values.copy()
    for _ in range(n_perm):
        rng.shuffle(labels)
        perm_wrs.append(labels[:obs_n].mean())
    p = np.mean(np.array(perm_wrs) >= obs_wr)
    return obs_wr, obs_pnl, p

results = []

def test(label, mask):
    mask = mask & block_mask  # only within blocked
    n = mask.sum()
    if n < 8:
        return
    wr, pnl, p = mcpt_p(mask)
    be = blocked.loc[mask, "pm"].mean()  # correct: BE = pm for YES buyer
    results.append({
        "label": label,
        "n": int(n),
        "wr": round(wr * 100, 1),
        "be": round(be * 100, 1),
        "edge_pp": round((wr - be) * 100, 1),
        "pnl": round(pnl, 0),
        "p": round(p, 4)
    })

print("\n=== RESCUE SEARCH ===\n")

# ----- STOCH -----
for th in [20, 25, 30]:
    test(f"sk15m >= {th}", df["stoch_k_15m"] >= th)
for th in [30, 35, 40]:
    test(f"sk5m >= {th}", df["stoch_k_5m"] >= th)
for th in [30, 35, 40, 50]:
    test(f"sk1h >= {th}", df["stoch_k_1h"] >= th)
for th in [30, 35, 40, 50]:
    test(f"sk4h >= {th}", df["stoch_k_4h"] >= th)
for th in [20, 30]:
    test(f"sk15m < {th}", df["stoch_k_15m"] < th)
for th in [20, 30]:
    test(f"sk5m < {th}", df["stoch_k_5m"] < th)
for th in [20, 30]:
    test(f"sk1h < {th}", df["stoch_k_1h"] < th)
for th in [20, 30]:
    test(f"sk4h < {th}", df["stoch_k_4h"] < th)

# ----- CVD -----
test("cvd_4h >= 0", df["cvd_4h"] >= 0)
test("cvd_4h < 0", df["cvd_4h"] < 0)
for th in [0.5e6, 1e6, 2e6]:
    test(f"cvd_4h >= {th/1e6:.1f}M", df["cvd_4h"] >= th)
test("futures_cvd_12h >= 0", df["cg_futures_cvd_12h"] >= 0)
test("futures_cvd_12h < 0", df["cg_futures_cvd_12h"] < 0)
test("futures_ratio_4h > 1.0", df["cg_futures_ratio_4h"] > 1.0)
test("futures_ratio_4h <= 1.0", df["cg_futures_ratio_4h"] <= 1.0)

# ----- PRICE CHANGE -----
test("chg_5m > 0", df["chg_5m"] > 0)
test("chg_5m <= 0", df["chg_5m"] <= 0)
test("chg_5m > 0.1%", df["chg_5m"] > 0.001)
test("chg_5m > 0.2%", df["chg_5m"] > 0.002)
test("chg_5m < -0.1%", df["chg_5m"] < -0.001)
test("chg_15m > 0", df["chg_15m"] > 0)
test("chg_15m <= 0", df["chg_15m"] <= 0)
test("chg_1h > 0", df["chg_1h"] > 0)
test("chg_1h <= 0", df["chg_1h"] <= 0)
test("chg_4h > 0", df["chg_4h"] > 0)
test("chg_4h <= 0", df["chg_4h"] <= 0)
test("chg_1m > 0", df["chg_1m"] > 0)
test("chg_1m <= 0", df["chg_1m"] <= 0)

# ----- OFFSET / DEPTH -----
for lo, hi in [(-0.10, -0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.10)]:
    test(f"offset [{lo:.2f},{hi:.2f})", (df["offset_pct"] >= lo) & (df["offset_pct"] < hi))
test("offset > 0", df["offset_pct"] > 0)
test("offset <= 0", df["offset_pct"] <= 0)
test("offset > 0.03", df["offset_pct"] > 0.03)
test("offset < -0.03", df["offset_pct"] < -0.03)

# ----- EMA / TREND -----
test("ema_bias=+1", df["ema_bias"] == 1)
test("ema_bias=-1", df["ema_bias"] == -1)
test("ema_bias=0", df["ema_bias"] == 0)
test("ema_bias_1h=+1", df["ema_bias_1h"] == 1)
test("ema_bias_1h=-1", df["ema_bias_1h"] == -1)
test("ema_bias_1h=0", df["ema_bias_1h"] == 0)

# ----- DIRECTION -----
test("dir_15m=+1", df["dir_15m"] == 1)
test("dir_15m=-1", df["dir_15m"] == -1)
test("dir_15m=0", df["dir_15m"] == 0)
test("dir_5m=+1", df["dir_5m"] == 1)
test("dir_5m=-1", df["dir_5m"] == -1)
test("dir_1h=+1", df["dir_1h"] == 1)
test("dir_1h=-1", df["dir_1h"] == -1)
for v in [1, 2, 3, -1, -2, -3]:
    test(f"consec_dir_15m={v}", df["consec_dir_15m"] == v)

# ----- VOLUME -----
test("vol_ratio >= 1.2", df["vol_ratio"] >= 1.2)
test("vol_ratio < 0.8", df["vol_ratio"] < 0.8)
test("vol_ratio_1h >= 1.2", df["vol_ratio_1h"] >= 1.2)
test("vol_ratio_1h < 0.8", df["vol_ratio_1h"] < 0.8)
test("vol_ratio_5m >= 1.2", df["vol_ratio_5m"] >= 1.2)

# ----- REGIME -----
test("regime_z > 0", df["regime_z"] > 0)
test("regime_z < 0", df["regime_z"] < 0)
test("regime_z > 0.5", df["regime_z"] > 0.5)
test("regime_z < -0.5", df["regime_z"] < -0.5)
for col in ["markov_regime_1h", "markov_sol_1h", "markov_sol_4h", "markov_sol_6h"]:
    if col in df.columns:
        for val in [0, 1, 2]:
            test(f"{col}={val}", df[col] == val)

# ----- BOLLINGER / DONCHIAN / KELTNER -----
test("bb_pct_1h < 0.2", df["bb_pct_1h"] < 0.2)
test("bb_pct_1h < 0.3", df["bb_pct_1h"] < 0.3)
test("bb_pct_1h > 0.7", df["bb_pct_1h"] > 0.7)
test("bb_pct_1h > 0.8", df["bb_pct_1h"] > 0.8)
test("donch_bo_1h=0", df["donchian_breakout_1h"] == 0)
test("donch_bo_1h=1", df["donchian_breakout_1h"] == 1)
test("donch_bo_1h=-1", df["donchian_breakout_1h"] == -1)
test("kc_pct_1h > 0", df["kc_pct_1h"] > 0)
test("kc_pct_1h < 0", df["kc_pct_1h"] < 0)
test("kc_bo_1h=1", df["kc_bo_1h"] == 1)
test("kc_bo_1h=-1", df["kc_bo_1h"] == -1)
test("kc_bo_1h=0", df["kc_bo_1h"] == 0)

# ----- RSI -----
test("rsi_1h < 40", df["rsi_1h"] < 40)
test("rsi_1h > 60", df["rsi_1h"] > 60)
test("rsi_4h < 40", df["rsi_4h"] < 40)
test("rsi_4h > 60", df["rsi_4h"] > 60)

# ----- ARIMA / MU DRIFT -----
test("arima_forecast_1h > 0", df["arima_forecast_1h"] > 0)
test("arima_forecast_1h < 0", df["arima_forecast_1h"] < 0)
test("mu6h > 0", df["mu6h"] > 0)
test("mu6h < 0", df["mu6h"] < 0)
test("mu12h > 0", df["mu12h"] > 0)
test("mu24h > 0", df["mu24h"] > 0)
test("z_drift_6h > 0", df["z_drift_6h"] > 0)
test("z_drift_6h < 0", df["z_drift_6h"] < 0)

# ----- KALMAN / OU / AUTOCORR / HURST -----
test("kalman_residual > 0", df["kalman_residual"] > 0)
test("kalman_residual < 0", df["kalman_residual"] < 0)
test("kalman_velocity > 0", df["kalman_velocity"] > 0)
test("kalman_velocity < 0", df["kalman_velocity"] < 0)
test("hurst > 0.5", df["hurst_exponent"] > 0.5)
test("hurst < 0.5", df["hurst_exponent"] < 0.5)
test("autocorr15 > 0", df["autocorr1_15"] > 0)
test("autocorr15 < 0", df["autocorr1_15"] < 0)
test("ou_theta > 3", df["ou_theta"] > 3)
test("ou_theta > 5", df["ou_theta"] > 5)
test("ou_mu_dist > 0", df["ou_mu_distance"] > 0)
test("ou_mu_dist < 0", df["ou_mu_distance"] < 0)

# ----- LIQUIDITY / OI -----
test("liq_score >= 1", df["liq_score"] >= 1)
test("liq_score <= 0", df["liq_score"] <= 0)
test("liq_bias >= 1", df["liq_bias"] >= 1)
test("oi_chg_pct > 0", df["oi_chg_pct"] > 0)
test("oi_chg_pct < 0", df["oi_chg_pct"] < 0)
test("ls_long_pct < 50", df["ls_long_pct"] < 50)
test("ls_long_pct > 55", df["ls_long_pct"] > 55)
test("fear_greed > 50", df["fear_greed"] > 50)
test("fear_greed < 40", df["fear_greed"] < 40)
test("cg_composite > 0", df["cg_composite"] > 0)

# ----- BP -----
test("bp_5m >= 0.5", df["bp_5m"] >= 0.5)
test("bp_5m < 0.5", df["bp_5m"] < 0.5)
test("bp_15m >= 0.5", df["bp_15m"] >= 0.5)
test("bp_15m < 0.5", df["bp_15m"] < 0.5)
test("bp_1h >= 0.5", df["bp_1h"] >= 0.5)
test("bp_1h < 0.5", df["bp_1h"] < 0.5)
test("bp_4h >= 0.5", df["bp_4h"] >= 0.5)

# ----- COMPOSITE P_UP -----
test("composite_p_up > 0.5", df["composite_p_up"] > 0.5)
test("composite_p_up < 0.5", df["composite_p_up"] < 0.5)

# ----- MACD -----
test("macd_hist_1h > 0", df["macd_hist_1h"] > 0)
test("macd_hist_1h < 0", df["macd_hist_1h"] < 0)

# ----- CANDLES -----
test("upper_wick_15m < 0.1", df["upper_wick_15m"] < 0.1)
test("lower_wick_15m > 0.1", df["lower_wick_15m"] > 0.1)
test("range_ratio_15m < 0.5", df["range_ratio_15m"] < 0.5)
test("body_15m > 0.5", df["body_15m"] > 0.5)

# ----- ENGULFING / CROSS -----
test("engulfing_1h=1", df["engulfing_1h"] == 1)
test("engulfing_1h=-1", df["engulfing_1h"] == -1)
test("stoch_cross_1h=1", df["stoch_cross_1h"] == 1)
test("stoch_cross_1h=-1", df["stoch_cross_1h"] == -1)

# ----- COMBINATIONS (top signal combos) -----
# cvd_4h + stoch
test("cvd4h<0 + sk15m>=20", (df["cvd_4h"] < 0) & (df["stoch_k_15m"] >= 20))
test("cvd4h<0 + sk15m<20", (df["cvd_4h"] < 0) & (df["stoch_k_15m"] < 20))
test("cvd4h<0 + sk1h>=40", (df["cvd_4h"] < 0) & (df["stoch_k_1h"] >= 40))
test("sk15m<30 + sk1h>=40", (df["stoch_k_15m"] < 30) & (df["stoch_k_1h"] >= 40))
test("sk15m<30 + sk4h>=40", (df["stoch_k_15m"] < 30) & (df["stoch_k_4h"] >= 40))
test("cvd4h<0 + ema_bias_1h=+1", (df["cvd_4h"] < 0) & (df["ema_bias_1h"] == 1))
test("sk15m<30 + ema_bias_1h=+1", (df["stoch_k_15m"] < 30) & (df["ema_bias_1h"] == 1))
test("cvd4h<0 + bb<0.3", (df["cvd_4h"] < 0) & (df["bb_pct_1h"] < 0.3))
test("cvd4h<0 + chg_1h>0", (df["cvd_4h"] < 0) & (df["chg_1h"] > 0))
test("cvd4h<0 + offset>0", (df["cvd_4h"] < 0) & (df["offset_pct"] > 0))
test("cvd4h<0 + ou_theta>3", (df["cvd_4h"] < 0) & (df["ou_theta"] > 3))
test("sk15m<30 + ou_theta>3", (df["stoch_k_15m"] < 30) & (df["ou_theta"] > 3))
test("cvd4h<0 + hurst>0.5", (df["cvd_4h"] < 0) & (df["hurst_exponent"] > 0.5))
test("cvd4h<0 + kc_pct<0", (df["cvd_4h"] < 0) & (df["kc_pct_1h"] < 0))
test("cvd4h<0 + arima>0", (df["cvd_4h"] < 0) & (df["arima_forecast_1h"] > 0))
test("cvd4h<0 + composite_pup>0.5", (df["cvd_4h"] < 0) & (df["composite_p_up"] > 0.5))
test("cvd4h<0 + liq_score>=1", (df["cvd_4h"] < 0) & (df["liq_score"] >= 1))
test("cvd4h<0 + mu6h>0", (df["cvd_4h"] < 0) & (df["mu6h"] > 0))
test("cvd4h<0 + macd>0", (df["cvd_4h"] < 0) & (df["macd_hist_1h"] > 0))
test("cvd4h<0 + bp_1h>=0.5", (df["cvd_4h"] < 0) & (df["bp_1h"] >= 0.5))
test("cvd4h<0 + mu6h>0 + sk1h>=40", (df["cvd_4h"] < 0) & (df["mu6h"] > 0) & (df["stoch_k_1h"] >= 40))
test("cvd4h<0 + mu6h>0 + ema1h=+1", (df["cvd_4h"] < 0) & (df["mu6h"] > 0) & (df["ema_bias_1h"] == 1))

# ----- TAU (time to expiry) -----
for col in ["tau_minutes","spread","z_score","p_model_15m","raw_edge","vwap_dist",
            "ema20_dist_1h","body_5m","atr_ratio_15m","p_gbdt",
            "cg_futures_delta_4h","consec_dir_1h"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for th in [5.0, 7.0, 9.0, 11.0]:
    test(f"tau < {th}", df["tau_minutes"] < th)
    test(f"tau >= {th}", df["tau_minutes"] >= th)

# ----- SPREAD -----
for th in [0.02, 0.03, 0.05]:
    test(f"spread < {th}", df["spread"] < th)
    test(f"spread >= {th}", df["spread"] >= th)

# ----- Z_SCORE (model confidence) -----
for th in [-1.0, -0.5, 0.0, 0.5]:
    test(f"z_score > {th}", df["z_score"] > th)
    test(f"z_score <= {th}", df["z_score"] <= th)

# ----- P_MODEL_15M -----
for th in [0.60, 0.65, 0.70, 0.75, 0.80]:
    test(f"p_model >= {th}", df["p_model_15m"] >= th)
    test(f"p_model < {th}", df["p_model_15m"] < th)

# ----- RAW_EDGE -----
for th in [0.06, 0.08, 0.10, 0.15, 0.20]:
    test(f"raw_edge >= {th}", df["raw_edge"] >= th)
    test(f"raw_edge < {th}", df["raw_edge"] < th)

# ----- VWAP_DIST -----
for th in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
    test(f"vwap_dist > {th}", df["vwap_dist"] > th)
    test(f"vwap_dist <= {th}", df["vwap_dist"] <= th)

# ----- EMA20_DIST_1H -----
for th in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
    test(f"ema20_dist_1h > {th}", df["ema20_dist_1h"] > th)
    test(f"ema20_dist_1h <= {th}", df["ema20_dist_1h"] <= th)

# ----- BODY_5M -----
for th in [0.3, 0.5, 0.7]:
    test(f"body_5m > {th}", df["body_5m"] > th)
    test(f"body_5m <= {th}", df["body_5m"] <= th)

# ----- ATR_RATIO_15M -----
for th in [0.003, 0.004, 0.005, 0.006]:
    test(f"atr < {th}", df["atr_ratio_15m"] < th)
    test(f"atr >= {th}", df["atr_ratio_15m"] >= th)

# ----- P_GBDT -----
for th in [0.60, 0.65, 0.70, 0.75, 0.80]:
    test(f"p_gbdt >= {th}", df["p_gbdt"] >= th)
    test(f"p_gbdt < {th}", df["p_gbdt"] < th)

# ----- CG_FUTURES_DELTA_4H -----
test("futures_delta_4h > 0", df["cg_futures_delta_4h"] > 0)
test("futures_delta_4h < 0", df["cg_futures_delta_4h"] < 0)
for th in [5e6, 10e6, 20e6]:
    test(f"futures_delta > {th/1e6:.0f}M", df["cg_futures_delta_4h"] > th)
    test(f"futures_delta < -{th/1e6:.0f}M", df["cg_futures_delta_4h"] < -th)

# ----- CONSEC_DIR_1H (cap at ±5 to ignore outlier 1000 value) -----
df["consec_dir_1h_clean"] = df["consec_dir_1h"].clip(-10, 10)
for v in [-2, -1, 0, 1, 2]:
    test(f"consec_dir_1h={v}", df["consec_dir_1h_clean"] == v)
test("consec_dir_1h >= 1", df["consec_dir_1h_clean"] >= 1)
test("consec_dir_1h <= -1", df["consec_dir_1h_clean"] <= -1)
test("consec_dir_1h >= 2", df["consec_dir_1h_clean"] >= 2)

# ----- COMBINATIONS with newly found signals -----
test("tau<9 + sk1h>=40", (df["tau_minutes"] < 9) & (df["stoch_k_1h"] >= 40))
test("tau>=9 + sk1h>=40", (df["tau_minutes"] >= 9) & (df["stoch_k_1h"] >= 40))
test("raw_edge>=0.10 + sk1h>=40", (df["raw_edge"] >= 0.10) & (df["stoch_k_1h"] >= 40))
test("raw_edge>=0.15 + cvd4h<0", (df["raw_edge"] >= 0.15) & (df["cvd_4h"] < 0))
test("p_model>=0.70 + cvd4h<0", (df["p_model_15m"] >= 0.70) & (df["cvd_4h"] < 0))
test("p_model>=0.70 + sk15m<30", (df["p_model_15m"] >= 0.70) & (df["stoch_k_15m"] < 30))
test("vwap_dist>0 + cvd4h<0", (df["vwap_dist"] > 0) & (df["cvd_4h"] < 0))
test("vwap_dist<=0 + cvd4h<0", (df["vwap_dist"] <= 0) & (df["cvd_4h"] < 0))
test("vwap_dist>0.5 + sk15m<30", (df["vwap_dist"] > 0.5) & (df["stoch_k_15m"] < 30))
test("delta>0 + cvd4h<0", (df["cg_futures_delta_4h"] > 0) & (df["cvd_4h"] < 0))
test("delta<0 + cvd4h<0", (df["cg_futures_delta_4h"] < 0) & (df["cvd_4h"] < 0))
test("p_gbdt>=0.70 + cvd4h<0", (df["p_gbdt"] >= 0.70) & (df["cvd_4h"] < 0))
test("p_gbdt>=0.70 + sk15m<30", (df["p_gbdt"] >= 0.70) & (df["stoch_k_15m"] < 30))
test("ema20>0 + cvd4h<0", (df["ema20_dist_1h"] > 0) & (df["cvd_4h"] < 0))
test("ema20<=0 + sk15m<30", (df["ema20_dist_1h"] <= 0) & (df["stoch_k_15m"] < 30))
test("raw_edge>=0.10 + p_model>=0.70", (df["raw_edge"] >= 0.10) & (df["p_model_15m"] >= 0.70))
test("raw_edge>=0.10 + vwap>0", (df["raw_edge"] >= 0.10) & (df["vwap_dist"] > 0))
test("futures_delta>0 + sk1h>=40", (df["cg_futures_delta_4h"] > 0) & (df["stoch_k_1h"] >= 40))
test("consec_1h>=1 + sk1h>=40", (df["consec_dir_1h_clean"] >= 1) & (df["stoch_k_1h"] >= 40))
test("consec_1h<=-1 + cvd4h<0", (df["consec_dir_1h_clean"] <= -1) & (df["cvd_4h"] < 0))

# ----- FINAL SORT -----
rdf = pd.DataFrame(results)
if rdf.empty:
    print("No candidates found.")
else:
    rdf = rdf.sort_values("edge_pp", ascending=False)

    print(f"\n{'Label':<45} {'n':>5} {'WR%':>6} {'BE%':>6} {'Edge':>6} {'PnL':>7} {'p':>6}")
    print("-" * 95)
    for _, row in rdf.iterrows():
        flag = "***" if row["p"] < 0.05 and row["edge_pp"] > 3 else ("*  " if row["p"] < 0.10 else "   ")
        print(f"{flag} {row['label']:<44} {row['n']:>5} {row['wr']:>6.1f} {row['be']:>6.1f} {row['edge_pp']:>+6.1f} {row['pnl']:>7.0f} {row['p']:>6.4f}")

    print("\n=== TOP RESCUES (p<0.05, edge>+3pp, n>=15) ===")
    top = rdf[(rdf["p"] < 0.05) & (rdf["edge_pp"] > 3.0) & (rdf["n"] >= 15)]
    if top.empty:
        print("None meeting criteria.")
    else:
        for _, row in top.sort_values("pnl", ascending=False).iterrows():
            print(f"  {row['label']} — n={row['n']}, WR={row['wr']:.1f}% (BE={row['be']:.1f}%), "
                  f"edge={row['edge_pp']:+.1f}pp, PnL=${row['pnl']:.0f}, p={row['p']:.4f}")
