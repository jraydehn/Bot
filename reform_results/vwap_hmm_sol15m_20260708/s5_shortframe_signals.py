"""
S5 -- Build genuinely NEW short-timeframe (5m/15m) versions of signals that
only exist at 1h in the live SOL 15m runner: Keltner, Donchian, stoch
crossover, Kalman velocity/residual, Hurst exponent, OU theta. These are
NOT reconstructions of an existing live formula (none exists at 5m/15m) --
they mirror the EXACT 1h formula from paper_trade_runner_15m.py, applied to
5m/15m resampled bars instead, since a 15-min contract's natural decision
horizon is much closer to 5m/15m than to 1h.

Scoping disclosure: ARIMA(2,0,1) is included at 15m but NOT 5m -- fitting
715 rows x ARIMA at 5m granularity (extremely noisy at that bar count, and
~3x the already-substantial 1h/15m compute cost) was judged not worth it;
flagged here rather than silently dropped. Everything else runs at both
5m and 15m.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "reform_results/vwap_hmm_sol15m_20260708"
rng = np.random.default_rng(41)

PARQUET = sorted(__import__("pathlib").Path(".").glob("data/binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(PARQUET).sort_index()
RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df5 = df1m.resample("5min").agg(RESAMPLE_AGG).dropna()
df15 = df1m.resample("15min").agg(RESAMPLE_AGG).dropna()
print(f"df5: {len(df5):,} bars   df15: {len(df15):,} bars")


def _bars_before(df, ts, n_needed):
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 20:
        return None
    lo = max(0, idx - n_needed)
    return df.iloc[lo:idx + 1]


def keltner_at(df, ts):
    b = _bars_before(df, ts, 60)
    if b is None or len(b) < 20:
        return np.nan, np.nan
    c, h, l = b["close"], b["high"], b["low"]
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


def donchian_at(df, ts):
    b = _bars_before(df, ts, 25)
    if b is None or len(b) < 20:
        return np.nan, np.nan
    h, l, c = b["high"], b["low"], b["close"]
    dc_hi, dc_lo = h.rolling(20).max().iloc[-1], l.rolling(20).min().iloc[-1]
    last = float(c.iloc[-1])
    brk = 1 if last >= dc_hi else (-1 if last <= dc_lo else 0)
    pos = (last - dc_lo) / (dc_hi - dc_lo) if dc_hi > dc_lo else np.nan
    return brk, pos


def stoch_cross_at(df, ts):
    b = _bars_before(df, ts, 20)
    if b is None or len(b) < 17:
        return np.nan
    h, l, c = b["high"], b["low"], b["close"]
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    rng14 = hi14 - lo14
    sk = ((c - lo14) / rng14.replace(0, np.nan)) * 100.0
    sd = sk.rolling(3).mean()
    if len(sk) < 2 or pd.isna(sk.iloc[-1]) or pd.isna(sd.iloc[-1]) or pd.isna(sd.iloc[-2]):
        return np.nan
    sk_last, sd_last = float(sk.iloc[-1]), float(sd.iloc[-1])
    sk_prev, sd_prev = float(sk.iloc[-2]), float(sd.iloc[-2])
    if sk_last > sd_last and sk_prev <= sd_prev:
        return 1
    if sk_last < sd_last and sk_prev >= sd_prev:
        return -1
    return 0


def kalman_hurst_ou_at(df, ts):
    b = _bars_before(df, ts, 70)
    out = {"kalman_velocity": np.nan, "kalman_residual": np.nan,
           "hurst_exponent": np.nan, "ou_theta": np.nan}
    if b is None or len(b) < 31:
        return out
    c = b["close"].values.astype(float)
    lr = np.diff(np.log(c))
    if len(lr) < 10:
        return out

    h_lr = lr[-64:] if len(lr) >= 64 else lr
    rs_pts = []
    for w in [8, 16, 32, 64]:
        if len(h_lr) < w:
            continue
        seg = h_lr[-w:]
        dev = np.cumsum(seg - seg.mean())
        r = dev.max() - dev.min(); s = seg.std(ddof=1)
        if s > 0:
            rs_pts.append((np.log(w), np.log(r / s)))
    if len(rs_pts) >= 2:
        xs = np.array([p[0] for p in rs_pts]); ys = np.array([p[1] for p in rs_pts])
        out["hurst_exponent"] = float(np.clip(np.polyfit(xs, ys, 1)[0], 0.0, 1.0))

    ou_lr = lr[-48:] if len(lr) >= 48 else lr
    if len(ou_lr) >= 10:
        mu = ou_lr.mean(); yc = ou_lr - mu
        phi = float(np.clip(np.dot(yc[:-1], yc[1:]) / (np.dot(yc[:-1], yc[:-1]) + 1e-12), -0.9999, 0.9999))
        out["ou_theta"] = float(np.clip(-np.log(abs(phi)), 0.0, 10.0))

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
    return out


def arima15_at(ts):
    b = _bars_before(df15, ts, 250)
    if b is None or len(b) < 20:
        return np.nan
    c = b["close"]
    lr = np.log(c / c.shift(1)).dropna()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def build(df, tf_label, ts_series):
    print(f"  building {tf_label} signals ({len(ts_series)} rows)...")
    kc = ts_series.apply(lambda t: keltner_at(df, t))
    dc = ts_series.apply(lambda t: donchian_at(df, t))
    sc = ts_series.apply(lambda t: stoch_cross_at(df, t))
    kho = ts_series.apply(lambda t: kalman_hurst_ou_at(df, t))
    out = pd.DataFrame(index=ts_series.index)
    out[f"kc_pct_{tf_label}"] = kc.apply(lambda t: t[0])
    out[f"kc_bo_{tf_label}"] = kc.apply(lambda t: t[1])
    out[f"donch_breakout_{tf_label}"] = dc.apply(lambda t: t[0])
    out[f"donch_pos_{tf_label}"] = dc.apply(lambda t: t[1])
    out[f"stoch_cross_{tf_label}"] = sc
    out[f"kalman_velocity_{tf_label}"] = kho.apply(lambda d: d["kalman_velocity"])
    out[f"kalman_residual_{tf_label}"] = kho.apply(lambda d: d["kalman_residual"])
    out[f"hurst_exponent_{tf_label}"] = kho.apply(lambda d: d["hurst_exponent"])
    out[f"ou_theta_{tf_label}"] = kho.apply(lambda d: d["ou_theta"])
    return out


results_all = {}
for state, side in [(1, "YES"), (5, "NO")]:
    pop = pd.read_csv(f"{OUT}/state{state}_{side}_reconstructed.csv", low_memory=False)
    pop["logged_at"] = pd.to_datetime(pop["logged_at"], format="mixed", utc=True, errors="coerce")
    print(f"\nState {state} {side}: n={len(pop)}")

    feat5 = build(df5, "5m", pop["logged_at"])
    feat15 = build(df15, "15m", pop["logged_at"])
    feat15["arima_forecast_15m"] = pop["logged_at"].apply(arima15_at)

    pop = pd.concat([pop.reset_index(drop=True), feat5.reset_index(drop=True), feat15.reset_index(drop=True)], axis=1)
    for c in list(feat5.columns) + list(feat15.columns):
        print(f"    {c}: {pop[c].notna().sum()}/{len(pop)} non-null")
    pop.to_csv(f"{OUT}/state{state}_{side}_shortframe.csv", index=False)
    results_all[(state, side)] = pop

print("\n" + "#" * 70)
print("SWEEP SHORT-TIMEFRAME SIGNALS")
print("#" * 70)


def boot_p(edges, n_boot=4000):
    e = np.asarray(edges); n = len(e)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()


SHORT_COLS = []
for tf in ["5m", "15m"]:
    SHORT_COLS += [f"kc_pct_{tf}", f"kc_bo_{tf}", f"donch_breakout_{tf}", f"donch_pos_{tf}",
                  f"stoch_cross_{tf}", f"kalman_velocity_{tf}", f"kalman_residual_{tf}",
                  f"hurst_exponent_{tf}", f"ou_theta_{tf}"]
SHORT_COLS += ["arima_forecast_15m"]

for (state, side), pop in results_all.items():
    print(f"\n=== State {state} {side}: n={len(pop)} base WR={pop['won'].mean():.3f} base BE={pop['be'].mean():.3f} ===")
    found = []
    for feat in SHORT_COLS:
        vals = pd.to_numeric(pop[feat], errors="coerce")
        s2 = pop[vals.notna()]
        nn = len(s2)
        if nn < 15:
            print(f"  {feat}: SKIP (only {nn} non-null)")
            continue
        if "bo_" in feat or "breakout" in feat or "cross" in feat:
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
    print(f"  {len(fd)} splits tested")
    real = fd[(fd["edge"] > 0) & (fd["n"] >= 15)]
    print(f"  positive-edge splits (n>=15): {len(real)}")
    if len(real):
        top = real.sort_values("edge", ascending=False).head(10)
        print(top.round(3).to_string(index=False))
        print("\n  bootstrap on top:")
        for _, r in top.head(6).iterrows():
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
