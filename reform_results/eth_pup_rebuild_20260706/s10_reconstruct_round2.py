"""
S10 -- Round 2 reconstruction: Keltner, Kalman, ARIMA, bp_1h, stoch_k_4h --
all confirmed present in the runner's logic but silently under-covered
(<20 non-null) for this backfill window, computed causally from ETH's own
1h price history using the EXACT formulas from paper_trade_runner.py.

Honest disclosure of what's NOT reconstructed: hmm_vol_state/hmm_r1_prob
(R0/R1 vol regime), hmm_pnl_state, hmm_ps_state (phase-space trajectory),
hmm_gd_state (gate-density), hmm_zdrift_state -- checked models/ directly,
these were ONLY ever built for BTC (hmm_gate_density_btc.pkl,
hmm_phase_traj_btc.pkl, hmm_pnl_regime_btc.pkl, hmm_zdrift_btc.pkl,
hmm_vol_regime_btc_15m.pkl -- no _eth variant exists for any of them).
Building 5 new experimental models from scratch is a materially different,
much larger undertaking than a rescue search, and per project memory most
of these were already found weak-to-rejected even for BTC (gate-density:
"WF fails, gate evolution artifact"; phase-space: mostly shadow; P&L
regime: tabled, needs 60+ days). Not reconstructed here -- flagged, not
silently skipped.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
c1h_full = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_ETHUSDT_1h.parquet").sort_index()
close_s = c1h_full["close"].astype(float)
high_s = c1h_full["high"].astype(float)
low_s = c1h_full["low"].astype(float)


def keltner_at(ts):
    idx = close_s.index.searchsorted(ts, side="right") - 1
    if idx < 30:
        return np.nan, np.nan
    c = close_s.iloc[max(0, idx - 100):idx + 1]
    h = high_s.iloc[max(0, idx - 100):idx + 1]
    l = low_s.iloc[max(0, idx - 100):idx + 1]
    ema10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    upper = ema10 + 1.5 * atr14
    lower = ema10 - 1.5 * atr14
    width = float((upper - lower).iloc[-1])
    if width <= 0:
        return np.nan, np.nan
    kc_pct = (float(c.iloc[-1]) - float(lower.iloc[-1])) / width
    kc_bo = 1 if float(c.iloc[-1]) > float(upper.iloc[-1]) else (-1 if float(c.iloc[-1]) < float(lower.iloc[-1]) else 0)
    return kc_pct, kc_bo


def kalman_at(ts):
    idx = close_s.index.searchsorted(ts, side="right") - 1
    if idx < 30:
        return np.nan, np.nan
    c = close_s.iloc[max(0, idx - 60):idx + 1]
    lr = np.log(c.values[1:] / c.values[:-1])
    kl = lr[-48:] if len(lr) >= 48 else lr
    if len(kl) < 5:
        return np.nan, np.nan
    Q = np.array([[1e-5, 0.0], [0.0, 1e-5]])
    R = float(np.var(kl)) + 1e-10
    x = np.array([kl[0], 0.0]); P = np.eye(2) * 0.1
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
    for obs in kl:
        x = F @ x; P = F @ P @ F.T + Q
        K = P @ H.T / (float(H @ P @ H.T) + R)
        x = x + K.flatten() * (obs - float(H @ x))
        P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
    return float(x[1]), float(kl[-1] - float(H @ x))


def arima_at(ts):
    idx = close_s.index.searchsorted(ts, side="right") - 1
    if idx < 100:
        return np.nan
    c = close_s.iloc[max(0, idx - 300):idx + 1]
    lr = np.log(c / c.shift(1)).dropna()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def bp_1h_at(ts):
    idx = close_s.index.searchsorted(ts, side="right") - 1
    if idx < 1:
        return np.nan
    h = float(high_s.iloc[idx]); l = float(low_s.iloc[idx]); c = float(close_s.iloc[idx])
    return (c - l) / (h - l) if (h - l) > 0 else 0.5


def stoch_k_4h_at(ts):
    idx = close_s.index.searchsorted(ts, side="right") - 1
    if idx < 60:
        return np.nan
    c = close_s.iloc[max(0, idx - 300):idx + 1]
    c4h = c.resample("4h").last().dropna()
    c4h = c4h[c4h.index <= ts]
    if len(c4h) < 15:
        return np.nan
    ll14 = c4h.rolling(14).min(); hh14 = c4h.rolling(14).max()
    val = (((c4h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100).shift(3)
    hourly_idx = pd.date_range(c4h.index[0], ts, freq="h", tz="UTC")
    return float(val.reindex(hourly_idx, method="ffill").iloc[-1])


def enrich(df):
    df = df.copy()
    kc = df["logged_at_parsed"].apply(keltner_at)
    df["kc_pct_1h_recon"] = kc.apply(lambda t: t[0])
    df["kc_bo_1h_recon"] = kc.apply(lambda t: t[1])
    kal = df["logged_at_parsed"].apply(kalman_at)
    df["kalman_velocity_recon"] = kal.apply(lambda t: t[0])
    df["kalman_residual_recon"] = kal.apply(lambda t: t[1])
    df["arima_forecast_recon"] = df["logged_at_parsed"].apply(arima_at)
    df["bp_1h_recon"] = df["logged_at_parsed"].apply(bp_1h_at)
    df["stoch_k_4h_recon"] = df["logged_at_parsed"].apply(stoch_k_4h_at)
    return df


if __name__ == "__main__":
    OUT = "reform_results/eth_pup_rebuild_20260706"
    for pop_file, label in [("eth_agree_fully_reconstructed.csv", "agree"),
                           ("eth_disagree_fully_reconstructed.csv", "disagree")]:
        df = pd.read_csv(f"{OUT}/{pop_file}", low_memory=False)
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at_parsed"], utc=True, errors="coerce")
        print(f"reconstructing round-2 signals for {label} (n={len(df)})...")
        df2 = enrich(df)
        for c in ["kc_pct_1h_recon", "kc_bo_1h_recon", "kalman_velocity_recon",
                  "kalman_residual_recon", "arima_forecast_recon", "bp_1h_recon", "stoch_k_4h_recon"]:
            print(f"  {c}: {df2[c].notna().sum()}/{len(df2)} non-null")
        df2.to_csv(f"{OUT}/eth_{label}_round2.csv", index=False)
    print("saved")
