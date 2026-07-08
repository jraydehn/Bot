"""
S4 -- Actually comprehensive rescue search for the SOL 15m VWAP HMM BLOCK
candidates (State 1 YES, State 5 NO), reconstructing every signal the live
15m runner computes for SOL but does NOT persist to sol_scan_archive_15m.csv
(kalman_velocity/residual, arima_forecast_1h, donchian_breakout_1h/
donch_1h_pos, kc_pct_1h/kc_bo_1h, hurst_exponent, ou_theta/halflife/
mu_distance, autocorr1_15/30, macro regime) -- using the EXACT formulas
from paper_trade_runner_15m.py (grepped, not re-derived).

Disclosed non-availability: ms/vd/of HMM states and GARCH are NOT computed
anywhere in paper_trade_runner_15m.py for any asset (confirmed via grep --
zero matches) -- genuinely unavailable, not reconstructed here.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/vwap_hmm_sol15m_20260708"
rng = np.random.default_rng(31)

c1h_full = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_SOLUSDT_1h.parquet").sort_index()
close = c1h_full["close"].astype(float)
high = c1h_full["high"].astype(float)
low = c1h_full["low"].astype(float)

with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)


def bars_before(ts, n_needed=250):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 20:
        return None
    lo = max(0, idx - n_needed)
    return close.iloc[lo:idx + 1], high.iloc[lo:idx + 1], low.iloc[lo:idx + 1]


def donchian_at(ts):
    b = bars_before(ts, 25)
    if b is None:
        return np.nan, np.nan
    c, h, l = b
    if len(c) < 20:
        return np.nan, np.nan
    dc_hi, dc_lo = h.rolling(20).max().iloc[-1], l.rolling(20).min().iloc[-1]
    last = float(c.iloc[-1])
    brk = 1 if last >= dc_hi else (-1 if last <= dc_lo else 0)
    pos = (last - dc_lo) / (dc_hi - dc_lo) if dc_hi > dc_lo else np.nan
    return brk, pos


def keltner_at(ts):
    b = bars_before(ts, 60)
    if b is None:
        return np.nan, np.nan
    c, h, l = b
    if len(c) < 20:
        return np.nan, np.nan
    ema10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    upper, lower = ema10 + 1.5 * atr14, ema10 - 1.5 * atr14
    width = float((upper - lower).iloc[-1])
    if width <= 0:
        return np.nan, np.nan
    last = float(c.iloc[-1])
    kc_pct = (last - float(lower.iloc[-1])) / width
    kc_bo = 1 if last > float(upper.iloc[-1]) else (-1 if last < float(lower.iloc[-1]) else 0)
    return kc_pct, kc_bo


def arima_at(ts):
    b = bars_before(ts, 250)
    if b is None:
        return np.nan
    c = b[0]
    if len(c) < 20:
        return np.nan
    lr = np.log(c / c.shift(1)).dropna()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def shadow_stoch_at(ts):
    """hurst_exponent, ou_theta, autocorr1_15/30, kalman_velocity/residual."""
    b = bars_before(ts, 70)
    if b is None:
        return {k: np.nan for k in
                ["hurst_exponent", "ou_theta", "autocorr1_15", "autocorr1_30",
                 "kalman_velocity", "kalman_residual"]}
    c = b[0].values.astype(float)
    if len(c) < 31:
        return {k: np.nan for k in
                ["hurst_exponent", "ou_theta", "autocorr1_15", "autocorr1_30",
                 "kalman_velocity", "kalman_residual"]}
    lr = np.diff(np.log(c))
    out = {}

    def lag1_ac(arr):
        if len(arr) < 4:
            return 0.0
        x, y = arr[:-1] - arr[:-1].mean(), arr[1:] - arr[1:].mean()
        denom = np.sqrt((x**2).sum() * (y**2).sum())
        return float(np.dot(x, y) / denom) if denom > 0 else 0.0

    out["autocorr1_15"] = lag1_ac(lr[-30:])
    out["autocorr1_30"] = lag1_ac(lr[-60:] if len(lr) >= 60 else lr)

    h_lr = lr[-64:] if len(lr) >= 64 else lr
    rs_pts = []
    for w in [8, 16, 32, 64]:
        if len(h_lr) < w:
            continue
        seg = h_lr[-w:]
        dev = np.cumsum(seg - seg.mean())
        r = dev.max() - dev.min()
        s = seg.std(ddof=1)
        if s > 0:
            rs_pts.append((np.log(w), np.log(r / s)))
    if len(rs_pts) >= 2:
        xs = np.array([p[0] for p in rs_pts]); ys = np.array([p[1] for p in rs_pts])
        out["hurst_exponent"] = float(np.clip(np.polyfit(xs, ys, 1)[0], 0.0, 1.0))
    else:
        out["hurst_exponent"] = np.nan

    ou_lr = lr[-48:] if len(lr) >= 48 else lr
    if len(ou_lr) >= 10:
        mu = ou_lr.mean(); yc = ou_lr - mu
        phi = float(np.clip(np.dot(yc[:-1], yc[1:]) / (np.dot(yc[:-1], yc[:-1]) + 1e-12), -0.9999, 0.9999))
        out["ou_theta"] = float(np.clip(-np.log(abs(phi)), 0.0, 10.0))
    else:
        out["ou_theta"] = np.nan

    kl = lr[-48:] if len(lr) >= 48 else lr
    if len(kl) >= 5:
        Q = np.array([[1e-5, 0.0], [0.0, 1e-5]]); R = float(np.var(kl)) + 1e-10
        x = np.array([kl[0], 0.0]); P = np.eye(2) * 0.1
        F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
        for obs in kl:
            x = F @ x; P = F @ P @ F.T + Q
            K = P @ H.T / (float(H @ P @ H.T) + R)
            x = x + K.flatten() * (obs - float(H @ x))
            P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        out["kalman_velocity"] = float(x[1])
        out["kalman_residual"] = float(kl[-1] - float(H @ x))
    else:
        out["kalman_velocity"] = np.nan
        out["kalman_residual"] = np.nan
    return out


def macro_regime_at(ts):
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
    last = macro_pkg["model"].predict_proba(X)[-1]
    label_names = macro_pkg["label_names"]
    return {label_names[s]: float(last[s]) for s in range(len(last))}


def reconstruct(df):
    df = df.copy()
    dc = df["logged_at"].apply(donchian_at)
    df["donchian_breakout_1h_recon"] = dc.apply(lambda t: t[0])
    df["donch_1h_pos_recon"] = dc.apply(lambda t: t[1])
    kc = df["logged_at"].apply(keltner_at)
    df["kc_pct_1h_recon"] = kc.apply(lambda t: t[0])
    df["kc_bo_1h_recon"] = kc.apply(lambda t: t[1])
    df["arima_forecast_1h_recon"] = df["logged_at"].apply(arima_at)
    shadow = df["logged_at"].apply(shadow_stoch_at)
    for k in ["hurst_exponent", "ou_theta", "autocorr1_15", "autocorr1_30",
             "kalman_velocity", "kalman_residual"]:
        df[f"{k}_recon"] = shadow.apply(lambda d: d[k])
    macro = df["logged_at"].apply(macro_regime_at)
    df["macro_bull_recon"] = macro.apply(lambda d: d.get("Bull") if d else np.nan)
    df["macro_sideways_recon"] = macro.apply(lambda d: d.get("Sideways") if d else np.nan)
    df["macro_bear_recon"] = macro.apply(lambda d: d.get("Bear") if d else np.nan)
    return df


full = pd.read_csv(f"{OUT}/full_analysis.csv", low_memory=False)
full["logged_at"] = pd.to_datetime(full["logged_at"], format="mixed", utc=True, errors="coerce")

results_all = {}
for state, side in [(1, "YES"), (5, "NO")]:
    pop = full[(full["vwap_hmm_state"] == state) & (full["model_side"] == side)].copy()
    print(f"reconstructing State {state} {side}  n={len(pop)} ...")
    pop = reconstruct(pop)
    recon_cols = [c for c in pop.columns if c.endswith("_recon")]
    for c in recon_cols:
        print(f"  {c}: {pop[c].notna().sum()}/{len(pop)} non-null")
    pop.to_csv(f"{OUT}/state{state}_{side}_reconstructed.csv", index=False)
    results_all[(state, side)] = pop

print("\n" + "#" * 70)
print("SWEEP RECONSTRUCTED SIGNALS")
print("#" * 70)


def boot_p(edges, n_boot=4000):
    e = np.asarray(edges); n = len(e)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()


RECON_COLS = ["donchian_breakout_1h_recon", "donch_1h_pos_recon", "kc_pct_1h_recon",
             "kc_bo_1h_recon", "arima_forecast_1h_recon", "hurst_exponent_recon",
             "ou_theta_recon", "autocorr1_15_recon", "autocorr1_30_recon",
             "kalman_velocity_recon", "kalman_residual_recon",
             "macro_bull_recon", "macro_sideways_recon", "macro_bear_recon"]

for (state, side), pop in results_all.items():
    print(f"\n=== State {state} {side}: n={len(pop)} base WR={pop['won'].mean():.3f} base BE={pop['be'].mean():.3f} ===")
    found = []
    for feat in RECON_COLS:
        vals = pd.to_numeric(pop[feat], errors="coerce")
        s2 = pop[vals.notna()]
        nn = len(s2)
        if nn < 15:
            print(f"  {feat}: SKIP (only {nn} non-null)")
            continue
        if feat in ("donchian_breakout_1h_recon", "kc_bo_1h_recon"):
            for val in [-1, 0, 1]:
                mask = vals == val
                s = s2.loc[mask.index[mask]]
                if len(s) < 10:
                    continue
                wr, be = s["won"].mean(), s["be"].mean()
                found.append({"feature": feat, "split": f"=={val}", "n": len(s), "wr": wr, "be": be, "edge": wr - be})
            continue
        vv = vals.dropna()
        for q in np.arange(0.1, 1.0, 0.1):
            thresh = vv.quantile(q)
            for direction, mask in [(">=", vv >= thresh), ("<", vv < thresh)]:
                s = s2.loc[mask.index[mask]]
                if len(s) < 10 or (len(s2) - len(s)) < 10:
                    continue
                wr, be = s["won"].mean(), s["be"].mean()
                found.append({"feature": feat, "split": f"{direction}{thresh:.4g}(q{q:.1f})",
                             "n": len(s), "wr": wr, "be": be, "edge": wr - be})
    fd = pd.DataFrame(found)
    if len(fd) == 0:
        print("  no splits tested")
        continue
    print(f"  {len(fd)} splits tested across reconstructed signals")
    real = fd[(fd["edge"] > 0) & (fd["n"] >= 15)]
    print(f"  positive-edge splits (n>=15): {len(real)}")
    if len(real):
        top = real.sort_values("edge", ascending=False).head(8)
        print(top.round(3).to_string(index=False))
        print("\n  bootstrap on top:")
        for _, r in top.head(5).iterrows():
            feat, split = r["feature"], r["split"]
            vals = pd.to_numeric(pop[feat], errors="coerce")
            if split.startswith("=="):
                mask = vals == float(split[2:])
            elif split.startswith(">="):
                thresh = float(split.split("(")[0][2:])
                mask = vals >= thresh
            else:
                thresh = float(split.split("(")[0][1:])
                mask = vals < thresh
            sub_r = pop[mask.fillna(False)]
            edges = (sub_r["won"].astype(float) - sub_r["be"]).values
            m, lo, hi, p = boot_p(edges)
            print(f"    {feat} {split}: n={len(sub_r)} edge_mean={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] P(edge<=0)={p:.4f}")

print("\nDone.")
