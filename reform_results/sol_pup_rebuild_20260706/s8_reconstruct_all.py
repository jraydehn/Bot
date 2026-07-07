"""
S8 -- Reconstruct every missing/thin signal category for BOTH SOL AGREE and
DISAGREE populations, computed causally from SOL's OWN 1h price history plus
SOL's own trained HMM model files, using the EXACT live formulas from
paper_trade_runner.py (grepped, not re-derived). Categories:
  - GARCH(1,1) ratio: runner hardcodes BTC-only, refit here on SOL's own
    returns (method is asset-agnostic).
  - macro regime (Bull/Sideways/Bear): no SOL-specific model exists (only
    BTC's was built) -- BTC-trained HMM applied to SOL's OWN price/return
    features as a disclosed cross-asset "broader regime" proxy, NOT a
    genuine SOL-native signal.
  - MACD histogram, Donchian position, Keltner (kc_pct/kc_bo), Kalman
    (velocity/residual, simple restarted-per-call version), ARIMA(2,0,1)
    forecast, bp_1h, stoch_k_4h (resample+shift(3) quirk) -- exact live
    formulas.
  - hmm_ms_state / hmm_vd_state / hmm_of_state -- SOL's own trained model
    files (models/hmm_microstructure_sol.pkl etc, confirmed to exist),
    validated trailing-SEQUENCE decode logic (2026-07-06 fix).
  - hmm_vol_state/r1_prob, hmm_pnl_state, hmm_ps_state, hmm_gd_state,
    hmm_zdrift_state -- checked models/ directly: BTC-ONLY builds
    (hmm_gate_density_btc.pkl, hmm_phase_traj_btc.pkl, hmm_pnl_regime_btc.pkl,
    hmm_zdrift_btc.pkl, hmm_vol_regime_btc_15m.pkl -- no _sol variant for
    any of them). NOT reconstructed -- disclosed, not silently skipped.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/sol_pup_rebuild_20260706"
REBUILD = "reform_results/pup_v2_rebuild_20260704"

c1h_full = pd.read_parquet(f"{REBUILD}/hist_SOLUSDT_1h.parquet").sort_index()
close = c1h_full["close"].astype(float)
hi = c1h_full["high"].astype(float)
lo = c1h_full["low"].astype(float)

with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)
with open("models/hmm_microstructure_sol.pkl", "rb") as f:
    ms_pkg = pickle.load(f)
with open("models/hmm_voldirection_sol.pkl", "rb") as f:
    vd_pkg = pickle.load(f)
with open("models/hmm_orderflow_sol.pkl", "rb") as f:
    of_pkg = pickle.load(f)


def macro_regime_probs_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 80:
        return None
    window = close.iloc[max(0, idx - 200):idx + 1]
    log_ret = np.log(window / window.shift(1))
    ret_24h = log_ret.rolling(24, min_periods=12).sum()
    ret_72h = log_ret.rolling(72, min_periods=36).sum()
    rv24 = log_ret.rolling(24, min_periods=12).std()
    roll_mean = log_ret.rolling(24, min_periods=12).mean()
    sharpe_24h = (roll_mean / rv24.replace(0, np.nan)).fillna(0.0)
    feat = pd.DataFrame({"ret_24h": ret_24h, "ret_72h": ret_72h, "rv24": rv24,
                        "sharpe_24h": sharpe_24h}).dropna()
    if len(feat) < 10:
        return None
    feat = feat.iloc[-80:]
    X = macro_pkg["scaler"].transform(feat.values.astype(float))
    posteriors = macro_pkg["model"].predict_proba(X)
    last = posteriors[-1]
    label_names = macro_pkg["label_names"]
    return {label_names[s]: float(last[s]) for s in range(len(last))}


def garch_ratio_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 502:
        return np.nan
    window = np.log(close.iloc[idx - 500:idx + 1] / close.iloc[idx - 500:idx + 1].shift(1)).dropna() * 100
    try:
        from arch import arch_model
        am = arch_model(window, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_v = float(res.conditional_volatility.iloc[-1])
        omega, alpha, beta = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
        persist = alpha + beta
        lr_vol = float(np.sqrt(omega / (1 - persist))) if persist < 1 else float(window.std())
        return cond_v / lr_vol if lr_vol > 0 else np.nan
    except Exception:
        return np.nan


def macd_hist_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 40:
        return np.nan
    window = close.iloc[max(0, idx - 100):idx + 1]
    macd_line = window.ewm(span=12, adjust=False).mean() - window.ewm(span=26, adjust=False).mean()
    hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    return float(hist.iloc[-1])


def donchian_pos_at(ts):
    idx = hi.index.searchsorted(ts, side="right") - 1
    if idx < 20:
        return np.nan
    h = float(hi.iloc[idx - 19:idx + 1].max())
    l = float(lo.iloc[idx - 19:idx + 1].min())
    c = float(close.iloc[idx])
    return (c - l) / (h - l) if h > l else np.nan


def keltner_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 30:
        return np.nan, np.nan
    c = close.iloc[max(0, idx - 100):idx + 1]
    h = hi.iloc[max(0, idx - 100):idx + 1]
    l = lo.iloc[max(0, idx - 100):idx + 1]
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
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 30:
        return np.nan, np.nan
    c = close.iloc[max(0, idx - 60):idx + 1]
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
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 100:
        return np.nan
    c = close.iloc[max(0, idx - 300):idx + 1]
    lr = np.log(c / c.shift(1)).dropna()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def bp_1h_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 1:
        return np.nan
    h = float(hi.iloc[idx]); l = float(lo.iloc[idx]); c = float(close.iloc[idx])
    return (c - l) / (h - l) if (h - l) > 0 else 0.5


def stoch_k_4h_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 60:
        return np.nan
    c = close.iloc[max(0, idx - 300):idx + 1]
    c4h = c.resample("4h").last().dropna()
    c4h = c4h[c4h.index <= ts]
    if len(c4h) < 15:
        return np.nan
    ll14 = c4h.rolling(14).min(); hh14 = c4h.rolling(14).max()
    val = (((c4h - ll14) / (hh14 - ll14).replace(0, np.nan)) * 100).shift(3)
    hourly_idx = pd.date_range(c4h.index[0], ts, freq="h", tz="UTC")
    return float(val.reindex(hourly_idx, method="ffill").iloc[-1])


def ms_hmm_state_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 72:
        return None
    c1h = close.iloc[max(0, idx - 300):idx + 1].dropna()
    if len(c1h) < 72:
        return None
    lr = np.log(c1h.values[1:] / c1h.values[:-1])
    if len(lr) < 70:
        return None

    def _pt(lr_avail):
        buf48 = lr_avail[-48:] if len(lr_avail) >= 48 else lr_avail
        y = buf48 - buf48.mean()
        phi = float(np.dot(y[:-1], y[1:]) / (np.dot(y[:-1], y[:-1]) + 1e-12))
        phi = float(np.clip(phi, -0.9999, 0.9999))
        ou_theta = float(np.clip(-np.log(abs(phi)), 0.0, 10.0))
        pts = []
        for w in [8, 16, 32, 64]:
            if len(lr_avail) < w:
                continue
            seg = lr_avail[-w:]
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min(); s = seg.std(ddof=1)
            if s > 0 and r > 0:
                pts.append((np.log(w), np.log(r / s)))
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            hurst = float(np.clip(np.polyfit(xs, ys, 1)[0], 0.0, 1.0))
        else:
            hurst = 0.5
        buf60 = lr_avail[-60:] if len(lr_avail) >= 60 else lr_avail
        x60 = buf60[:-1] - buf60[:-1].mean(); y60 = buf60[1:] - buf60[1:].mean()
        denom60 = float(np.sqrt((x60**2).sum() * (y60**2).sum()))
        autocorr = float(np.dot(x60, y60) / denom60) if denom60 > 0 else 0.0
        rvol = float(np.std(lr_avail[-24:] if len(lr_avail) >= 24 else lr_avail, ddof=1))
        return ou_theta, hurst, autocorr, rvol

    Q = np.array([[1e-5, 0.0], [0.0, 1e-5]]); F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
    x_kf = np.array([lr[0], 0.0]); P_kf = np.eye(2) * 0.1
    kalman_vel = np.zeros(len(lr))
    for t in range(len(lr)):
        buf_r = lr[max(0, t - 47):t + 1]
        R = float(np.var(buf_r)) + 1e-10 if len(buf_r) > 1 else 1e-10
        if t > 0:
            x_kf = F @ x_kf; P_kf = F @ P_kf @ F.T + Q
        K = P_kf @ H.T / (float(H @ P_kf @ H.T) + R)
        x_kf = x_kf + K.flatten() * (lr[t] - float(H @ x_kf))
        P_kf = (np.eye(2) - np.outer(K.flatten(), H)) @ P_kf
        kalman_vel[t] = x_kf[1]

    seq_len_eff = min(60, len(lr))
    rows = []
    for i in range(len(lr) - seq_len_eff, len(lr)):
        ot, hu, ac, rv = _pt(lr[:i + 1])
        rows.append([ot, hu, ac, round(float(kalman_vel[i]), 6), rv])
    feat_seq = np.array(rows)
    X = ms_pkg["scaler"].transform(feat_seq)
    states = ms_pkg["model"].predict(X)
    return int(states[-1])


def vd_hmm_state_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 200:
        return None
    c1h = close.iloc[max(0, idx - 400):idx + 1].dropna()
    if len(c1h) < 200:
        return None
    cv = c1h.values
    lr = np.log(cv[1:] / cv[:-1])
    if len(lr) < 175:
        return None
    alpha8, alpha21 = 2.0 / 9, 2.0 / 22
    ema8v = np.empty(len(cv)); ema8v[0] = cv[0]
    ema21v = np.empty(len(cv)); ema21v[0] = cv[0]
    for i in range(1, len(cv)):
        ema8v[i] = alpha8 * cv[i] + (1 - alpha8) * ema8v[i - 1]
        ema21v[i] = alpha21 * cv[i] + (1 - alpha21) * ema21v[i - 1]

    def _pt(i):
        chg_1h = float(lr[i])
        chg_3h = float(lr[i-2] + lr[i-1] + lr[i]) if i >= 2 else chg_1h
        win_short = lr[max(0, i-23):i+1]
        rvol_1h = float(np.std(win_short, ddof=1)) if len(win_short) >= 4 else np.nan
        win_long = lr[max(0, i-167):i+1]
        rvol_long = float(np.std(win_long, ddof=1)) if len(win_long) >= 24 else np.nan
        vol_ratio = float(rvol_1h / rvol_long) if (rvol_long and rvol_long > 0) else np.nan
        ema_trend = float((ema8v[i+1] - ema21v[i+1]) / cv[i+1]) if cv[i+1] > 0 else np.nan
        return chg_1h, chg_3h, rvol_1h, vol_ratio, ema_trend

    seq_len_eff = min(60, len(lr) - 167)
    if seq_len_eff < 5:
        return None
    rows = [_pt(i) for i in range(len(lr) - seq_len_eff, len(lr))]
    feat_seq = np.array(rows)
    if np.isnan(feat_seq).any():
        return None
    X = vd_pkg["scaler"].transform(feat_seq)
    states = vd_pkg["model"].predict(X)
    return int(states[-1])


def of_hmm_state_row(row):
    try:
        ls, oi, lb = row.get("ls_long_pct"), row.get("oi_chg_pct"), row.get("liq_bias")
        vpin, fund, obi = row.get("vpin_score"), row.get("funding_bias"), row.get("obi_score")
        if pd.isna(ls) or pd.isna(oi) or pd.isna(lb):
            return None
        ls = float(np.clip(ls, 0.0, 100.0)); lb = float(np.clip(lb, -1.0, 1.0))
        vpin = float(vpin) if not pd.isna(vpin) else 0.0
        fund = float(fund) if not pd.isna(fund) else 0.0
        obi = float(obi) if not pd.isna(obi) else 0.0
        feat = np.array([[ls, float(oi), lb, vpin, fund, obi]])
        X = of_pkg["scaler"].transform(feat)
        return int(of_pkg["model"].predict(X)[0])
    except Exception:
        return None


def reconstruct(df, label):
    print(f"reconstructing for {label} n={len(df)}...")
    macro = df["logged_at_parsed"].apply(macro_regime_probs_at)
    df["macro_bull"] = macro.apply(lambda d: d.get("Bull") if d else np.nan)
    df["macro_sideways"] = macro.apply(lambda d: d.get("Sideways") if d else np.nan)
    df["macro_bear"] = macro.apply(lambda d: d.get("Bear") if d else np.nan)
    print("  macro regime done")
    df["garch_ratio"] = df["logged_at_parsed"].apply(garch_ratio_at)
    print("  garch done")
    df["macd_hist_1h_recon"] = df["logged_at_parsed"].apply(macd_hist_at)
    df["donch_1h_recon"] = df["logged_at_parsed"].apply(donchian_pos_at)
    print("  macd/donchian done")
    kc = df["logged_at_parsed"].apply(keltner_at)
    df["kc_pct_1h_recon"] = kc.apply(lambda t: t[0])
    df["kc_bo_1h_recon"] = kc.apply(lambda t: t[1])
    kal = df["logged_at_parsed"].apply(kalman_at)
    df["kalman_velocity_recon"] = kal.apply(lambda t: t[0])
    df["kalman_residual_recon"] = kal.apply(lambda t: t[1])
    df["arima_forecast_recon"] = df["logged_at_parsed"].apply(arima_at)
    df["bp_1h_recon"] = df["logged_at_parsed"].apply(bp_1h_at)
    df["stoch_k_4h_recon"] = df["logged_at_parsed"].apply(stoch_k_4h_at)
    print("  keltner/kalman/arima/bp_1h/stoch_k_4h done")
    df["hmm_ms_state_recon"] = df["logged_at_parsed"].apply(ms_hmm_state_at)
    print("  ms done")
    df["hmm_vd_state_recon"] = df["logged_at_parsed"].apply(vd_hmm_state_at)
    print("  vd done")
    df["hmm_of_state_recon"] = df.apply(of_hmm_state_row, axis=1)
    print("  of done")

    cols = ["macro_bull", "macro_sideways", "macro_bear", "garch_ratio", "macd_hist_1h_recon",
            "donch_1h_recon", "kc_pct_1h_recon", "kc_bo_1h_recon", "kalman_velocity_recon",
            "kalman_residual_recon", "arima_forecast_recon", "bp_1h_recon", "stoch_k_4h_recon",
            "hmm_ms_state_recon", "hmm_vd_state_recon", "hmm_of_state_recon"]
    for c in cols:
        print(f"  {c}: {df[c].notna().sum()}/{len(df)} non-null")
    return df


if __name__ == "__main__":
    for pop_file, label in [("sol_agree_full.csv", "agree"), ("sol_disagree_full.csv", "disagree")]:
        df = pd.read_csv(f"{OUT}/{pop_file}", low_memory=False)
        df["logged_at_parsed"] = pd.to_datetime(df["logged_at_parsed"], utc=True, errors="coerce")
        df = reconstruct(df, label)
        df.to_csv(f"{OUT}/sol_{label}_fully_reconstructed.csv", index=False)
    print("saved")
