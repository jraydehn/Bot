"""
S16 -- Retroactively reconstruct hmm_ms_state / hmm_vd_state (from BTC 1h
price history) and hmm_of_state (from the sparse-but-present raw order-
flow columns) for the two blocked populations, using the EXACT validated
sequence-decode logic from today's ms/vd/of HMM fix. These states show
0% coverage in the real logged CSV data for these populations because
reliable logging only started 2026-07-03 -- entirely after the April-
June window these blocked populations are drawn from. Reconstructing
them from history (same approach as GARCH/macro/MACD/Donchian in s15)
closes that gap instead of leaving it as "insufficient data."
"""
import pickle
import numpy as np
import pandas as pd

OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
c1h_full = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet")
c1h_full = c1h_full.sort_index()

with open("models/hmm_microstructure_btc.pkl", "rb") as f:
    ms_pkg = pickle.load(f)
with open("models/hmm_voldirection_btc.pkl", "rb") as f:
    vd_pkg = pickle.load(f)
with open("models/hmm_orderflow_btc.pkl", "rb") as f:
    of_pkg = pickle.load(f)


def ms_hmm_state_at(ts):
    idx = c1h_full.index.searchsorted(ts, side="right") - 1
    if idx < 72:
        return None
    c1h = c1h_full["close"].iloc[max(0, idx - 300):idx + 1].dropna()
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
            r = dev.max() - dev.min()
            s = seg.std(ddof=1)
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
    probs = ms_pkg["model"].predict_proba(X)
    return int(states[-1]), float(probs[-1, states[-1]])


def vd_hmm_state_at(ts):
    idx = c1h_full.index.searchsorted(ts, side="right") - 1
    if idx < 200:
        return None
    c1h = c1h_full["close"].iloc[max(0, idx - 400):idx + 1].dropna()
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
    probs = vd_pkg["model"].predict_proba(X)
    return int(states[-1]), float(probs[-1, states[-1]])


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
        state = int(of_pkg["model"].predict(X)[0])
        return state
    except Exception:
        return None


def enrich_hmm(df):
    df = df.copy()
    ms = df["logged_at_parsed"].apply(ms_hmm_state_at)
    df["hmm_ms_state_recon"] = ms.apply(lambda t: t[0] if t else np.nan)
    vd = df["logged_at_parsed"].apply(vd_hmm_state_at)
    df["hmm_vd_state_recon"] = vd.apply(lambda t: t[0] if t else np.nan)
    df["hmm_of_state_recon"] = df.apply(of_hmm_state_row, axis=1)
    return df


if __name__ == "__main__":
    rising_yes = pd.read_csv(f"{OUT}/rising_yes_enriched.csv", low_memory=False)
    crashing_no = pd.read_csv(f"{OUT}/crashing_no_enriched.csv", low_memory=False)
    rising_yes["logged_at_parsed"] = pd.to_datetime(rising_yes["logged_at_parsed"], utc=True, errors="coerce")
    crashing_no["logged_at_parsed"] = pd.to_datetime(crashing_no["logged_at_parsed"], utc=True, errors="coerce")

    print("Reconstructing ms/vd/of HMM states for rising_yes...")
    rising_yes2 = enrich_hmm(rising_yes)
    print("Reconstructing ms/vd/of HMM states for crashing_no...")
    crashing_no2 = enrich_hmm(crashing_no)

    for label, df in [("rising_yes", rising_yes2), ("crashing_no", crashing_no2)]:
        print(f"\n{label}:")
        for c in ["hmm_ms_state_recon", "hmm_vd_state_recon", "hmm_of_state_recon"]:
            print(f"  {c}: {df[c].notna().sum()}/{len(df)} non-null, "
                  f"values={sorted(df[c].dropna().unique().tolist())}")

    rising_yes2.to_csv(f"{OUT}/rising_yes_enriched2.csv", index=False)
    crashing_no2.to_csv(f"{OUT}/crashing_no_enriched2.csv", index=False)
    print("\nsaved")
