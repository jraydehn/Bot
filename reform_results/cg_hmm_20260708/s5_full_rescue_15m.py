"""
S5 -- ACTUALLY comprehensive rescue on the BTC 15m sell-flow YES bucket
(n=58, -$778.69). The first pass only swept logged columns; this one adds
Phase 3 reconstruction of everything thin or absent:
  A. NEW 5m/15m short-timeframe signals for BTC (Keltner, Donchian, stoch
     crossover, Kalman, Hurst, OU theta at both frames + 15m ARIMA) -- the
     exact family where SOL's equivalent bucket found its rescue. These
     never existed for BTC below 1h; built from BTC 1m history.
  B. Reconstructed 1h/4h signals: arima_forecast_1h, stoch_k_4h, rsi_4h,
     GARCH ratio, macro regime posteriors, hourly ms/vd HMM states.
  C. Joins of existing decoded histories: vwap_hmm_state (sidecar from the
     original backfill), p_up_v3 honest OOS preds (wf_preds_FINAL).
  D. Other-asset markov regime strings as categoricals (cheap, disclosed).
Then: full decile sweep + trade & expiry-cluster bootstraps on survivors.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(59)
OUT = "reform_results/cg_hmm_20260708"

t = pd.read_csv(f"{OUT}/btc15m_with_cg_state.csv", low_memory=False)
t["logged_at_p"] = pd.to_datetime(t["logged_at_p"], utc=True, errors="coerce")
pop = t[(t["cg_state"] == 3) & (t["side"] == "yes")].copy().reset_index(drop=True)
print(f"bucket: n={len(pop)}  PnL=${pd.to_numeric(pop['would_pnl'],errors='coerce').sum():+.2f}")

# ── data sources ──────────────────────────────────────────────────────────
df1m = pd.read_parquet(sorted(__import__("pathlib").Path(".").glob(
    "data/binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]).sort_index()
df1m = df1m[df1m.index >= "2026-04-01"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df5 = df1m.resample("5min").agg(AGG).dropna()
df15 = df1m.resample("15min").agg(AGG).dropna()
h1 = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet").sort_index()
close1h, high1h, low1h = h1["close"].astype(float), h1["high"].astype(float), h1["low"].astype(float)
with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)
with open("models/hmm_microstructure_btc.pkl", "rb") as f:
    ms_pkg = pickle.load(f)
with open("models/hmm_voldirection_btc.pkl", "rb") as f:
    vd_pkg = pickle.load(f)


def _bars_before(df, ts, n):
    i = df.index.searchsorted(ts, side="right") - 1
    return None if i < 20 else df.iloc[max(0, i - n):i + 1]


def keltner_at(df, ts):
    b = _bars_before(df, ts, 60)
    if b is None or len(b) < 20: return np.nan, np.nan
    c, h, l = b["close"], b["high"], b["low"]
    e10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(span=14, adjust=False).mean()
    up, lo = e10 + 1.5 * a14, e10 - 1.5 * a14
    w = float((up - lo).iloc[-1])
    if w <= 0: return np.nan, np.nan
    last = float(c.iloc[-1])
    return (last - float(lo.iloc[-1])) / w, (1 if last > float(up.iloc[-1]) else -1 if last < float(lo.iloc[-1]) else 0)


def donch_at(df, ts):
    b = _bars_before(df, ts, 25)
    if b is None or len(b) < 20: return np.nan, np.nan
    hh, ll = b["high"].rolling(20).max().iloc[-1], b["low"].rolling(20).min().iloc[-1]
    last = float(b["close"].iloc[-1])
    brk = 1 if last >= hh else (-1 if last <= ll else 0)
    return brk, (last - ll) / (hh - ll) if hh > ll else np.nan


def scross_at(df, ts):
    b = _bars_before(df, ts, 20)
    if b is None or len(b) < 17: return np.nan
    lo14, hi14 = b["low"].rolling(14).min(), b["high"].rolling(14).max()
    sk = ((b["close"] - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100.0
    sd = sk.rolling(3).mean()
    if pd.isna(sk.iloc[-1]) or pd.isna(sd.iloc[-1]) or pd.isna(sd.iloc[-2]): return np.nan
    if sk.iloc[-1] > sd.iloc[-1] and sk.iloc[-2] <= sd.iloc[-2]: return 1
    if sk.iloc[-1] < sd.iloc[-1] and sk.iloc[-2] >= sd.iloc[-2]: return -1
    return 0


def kho_at(df, ts):
    out = dict(kv=np.nan, kr=np.nan, hu=np.nan, ot=np.nan)
    b = _bars_before(df, ts, 70)
    if b is None or len(b) < 31: return out
    lr = np.diff(np.log(b["close"].values.astype(float)))
    hl = lr[-64:] if len(lr) >= 64 else lr
    pts = []
    for w in [8, 16, 32, 64]:
        if len(hl) < w: continue
        seg = hl[-w:]; dev = np.cumsum(seg - seg.mean())
        r, s = dev.max() - dev.min(), seg.std(ddof=1)
        if s > 0: pts.append((np.log(w), np.log(r / s)))
    if len(pts) >= 2:
        out["hu"] = float(np.clip(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0], 0, 1))
    ol = lr[-48:] if len(lr) >= 48 else lr
    if len(ol) >= 10:
        yc = ol - ol.mean()
        phi = float(np.clip(np.dot(yc[:-1], yc[1:]) / (np.dot(yc[:-1], yc[:-1]) + 1e-12), -0.9999, 0.9999))
        out["ot"] = float(np.clip(-np.log(abs(phi)), 0, 10))
    kl = lr[-48:] if len(lr) >= 48 else lr
    if len(kl) >= 5:
        Q = np.array([[1e-5, 0], [0, 1e-5]]); R = float(np.var(kl)) + 1e-10
        x = np.array([kl[0], 0.0]); P = np.eye(2) * 0.1
        F = np.array([[1., 1.], [0., 1.]]); H = np.array([[1., 0.]])
        for o in kl:
            x = F @ x; P = F @ P @ F.T + Q
            K = P @ H.T / (float(H @ P @ H.T) + R)
            x = x + K.flatten() * (o - float(H @ x)); P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        out["kv"], out["kr"] = float(x[1]), float(kl[-1] - float(H @ x))
    return out


def arima_at(df, ts):
    b = _bars_before(df, ts, 250)
    if b is None or len(b) < 30: return np.nan
    lr = np.log(b["close"] / b["close"].shift(1)).dropna()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def stoch4h_rsi4h_at(ts):
    i = close1h.index.searchsorted(ts, side="right") - 1
    if i < 80: return np.nan, np.nan
    c = close1h.iloc[max(0, i - 400):i + 1]
    c4 = c.resample("4h").last().dropna()
    c4 = c4[c4.index <= ts]
    if len(c4) < 16: return np.nan, np.nan
    ll, hh = c4.rolling(14).min(), c4.rolling(14).max()
    sk = float((((c4 - ll) / (hh - ll).replace(0, np.nan)) * 100).iloc[-1])
    d = c4.diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    l_ = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + g / l_.replace(0, np.nan))).iloc[-1])
    return sk, rsi


def garch_at(ts):
    i = close1h.index.searchsorted(ts, side="right") - 1
    if i < 502: return np.nan
    w = np.log(close1h.iloc[i - 500:i + 1] / close1h.iloc[i - 500:i + 1].shift(1)).dropna() * 100
    try:
        from arch import arch_model
        res = arch_model(w, vol="Garch", p=1, q=1, dist="normal", rescale=False).fit(disp="off", show_warning=False)
        cond = float(res.conditional_volatility.iloc[-1])
        om, al, be_ = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
        lr_v = float(np.sqrt(om / (1 - al - be_))) if al + be_ < 1 else float(w.std())
        return cond / lr_v if lr_v > 0 else np.nan
    except Exception:
        return np.nan


def macro_at(ts):
    i = close1h.index.searchsorted(ts, side="right") - 1
    if i < 80: return None
    w = close1h.iloc[max(0, i - 200):i + 1]
    lr = np.log(w / w.shift(1))
    fe = pd.DataFrame({"a": lr.rolling(24, min_periods=12).sum(), "b": lr.rolling(72, min_periods=36).sum(),
                       "c": lr.rolling(24, min_periods=12).std(),
                       "d": (lr.rolling(24, min_periods=12).mean() / lr.rolling(24, min_periods=12).std().replace(0, np.nan)).fillna(0.0)}).dropna()
    if len(fe) < 10: return None
    X = macro_pkg["scaler"].transform(fe.iloc[-80:].values.astype(float))
    last = macro_pkg["model"].predict_proba(X)[-1]
    return {macro_pkg["label_names"][s]: float(last[s]) for s in range(len(last))}


def ms_at(ts):
    i = close1h.index.searchsorted(ts, side="right") - 1
    if i < 72: return None
    c = close1h.iloc[max(0, i - 300):i + 1].dropna()
    lr = np.log(c.values[1:] / c.values[:-1])
    if len(lr) < 70: return None
    def _pt(la):
        b48 = la[-48:] if len(la) >= 48 else la
        y = b48 - b48.mean()
        phi = float(np.clip(np.dot(y[:-1], y[1:]) / (np.dot(y[:-1], y[:-1]) + 1e-12), -0.9999, 0.9999))
        ot = float(np.clip(-np.log(abs(phi)), 0, 10))
        pts = []
        for w in [8, 16, 32, 64]:
            if len(la) < w: continue
            seg = la[-w:]; dev = np.cumsum(seg - seg.mean())
            r, s = dev.max() - dev.min(), seg.std(ddof=1)
            if s > 0 and r > 0: pts.append((np.log(w), np.log(r / s)))
        hu = float(np.clip(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0], 0, 1)) if len(pts) >= 2 else 0.5
        b60 = la[-60:] if len(la) >= 60 else la
        x6 = b60[:-1] - b60[:-1].mean(); y6 = b60[1:] - b60[1:].mean()
        den = float(np.sqrt((x6**2).sum() * (y6**2).sum()))
        ac = float(np.dot(x6, y6) / den) if den > 0 else 0.0
        rv = float(np.std(la[-24:] if len(la) >= 24 else la, ddof=1))
        return ot, hu, ac, rv
    Q = np.array([[1e-5, 0], [0, 1e-5]]); F = np.array([[1., 1.], [0., 1.]]); H = np.array([[1., 0.]])
    x = np.array([lr[0], 0.0]); P = np.eye(2) * 0.1
    kv = np.zeros(len(lr))
    for i2 in range(len(lr)):
        R = float(np.var(lr[max(0, i2 - 47):i2 + 1])) + 1e-10 if i2 > 0 else 1e-10
        if i2 > 0: x = F @ x; P = F @ P @ F.T + Q
        K = P @ H.T / (float(H @ P @ H.T) + R)
        x = x + K.flatten() * (lr[i2] - float(H @ x)); P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        kv[i2] = x[1]
    seq = min(60, len(lr))
    rows = []
    for i2 in range(len(lr) - seq, len(lr)):
        ot, hu, ac, rv = _pt(lr[:i2 + 1])
        rows.append([ot, hu, ac, round(float(kv[i2]), 6), rv])
    return int(ms_pkg["model"].predict(ms_pkg["scaler"].transform(np.array(rows)))[-1])


def vd_at(ts):
    i = close1h.index.searchsorted(ts, side="right") - 1
    if i < 200: return None
    c = close1h.iloc[max(0, i - 400):i + 1].dropna()
    cv = c.values; lr = np.log(cv[1:] / cv[:-1])
    if len(lr) < 175: return None
    a8, a21 = 2 / 9, 2 / 22
    e8 = np.empty(len(cv)); e8[0] = cv[0]
    e21 = np.empty(len(cv)); e21[0] = cv[0]
    for i2 in range(1, len(cv)):
        e8[i2] = a8 * cv[i2] + (1 - a8) * e8[i2 - 1]; e21[i2] = a21 * cv[i2] + (1 - a21) * e21[i2 - 1]
    def _pt(i2):
        c1 = float(lr[i2]); c3 = float(lr[i2 - 2] + lr[i2 - 1] + lr[i2]) if i2 >= 2 else c1
        ws = lr[max(0, i2 - 23):i2 + 1]; r1 = float(np.std(ws, ddof=1)) if len(ws) >= 4 else np.nan
        wl = lr[max(0, i2 - 167):i2 + 1]; rl = float(np.std(wl, ddof=1)) if len(wl) >= 24 else np.nan
        vr = float(r1 / rl) if rl and rl > 0 else np.nan
        et = float((e8[i2 + 1] - e21[i2 + 1]) / cv[i2 + 1]) if cv[i2 + 1] > 0 else np.nan
        return c1, c3, r1, vr, et
    seq = min(60, len(lr) - 167)
    if seq < 5: return None
    fs = np.array([_pt(i2) for i2 in range(len(lr) - seq, len(lr))])
    if np.isnan(fs).any(): return None
    return int(vd_pkg["model"].predict(vd_pkg["scaler"].transform(fs))[-1])


print("reconstructing (A) 5m/15m shortframe...")
for df_, tf in [(df5, "5m"), (df15, "15m")]:
    kc = pop["logged_at_p"].apply(lambda ts: keltner_at(df_, ts))
    pop[f"kc_pct_{tf}_r"] = kc.apply(lambda x: x[0]); pop[f"kc_bo_{tf}_r"] = kc.apply(lambda x: x[1])
    dc = pop["logged_at_p"].apply(lambda ts: donch_at(df_, ts))
    pop[f"donch_brk_{tf}_r"] = dc.apply(lambda x: x[0]); pop[f"donch_pos_{tf}_r"] = dc.apply(lambda x: x[1])
    pop[f"scross_{tf}_r"] = pop["logged_at_p"].apply(lambda ts: scross_at(df_, ts))
    kho = pop["logged_at_p"].apply(lambda ts: kho_at(df_, ts))
    for k, nm in [("kv", "kalman_vel"), ("kr", "kalman_res"), ("hu", "hurst"), ("ot", "ou_theta")]:
        pop[f"{nm}_{tf}_r"] = kho.apply(lambda d: d[k])
pop["arima_15m_r"] = pop["logged_at_p"].apply(lambda ts: arima_at(df15, ts))
print("reconstructing (B) 1h/4h...")
pop["arima_1h_r"] = pop["logged_at_p"].apply(lambda ts: arima_at(h1, ts))
s4 = pop["logged_at_p"].apply(stoch4h_rsi4h_at)
pop["stoch_k_4h_r"] = s4.apply(lambda x: x[0]); pop["rsi_4h_r"] = s4.apply(lambda x: x[1])
pop["garch_r"] = pop["logged_at_p"].apply(garch_at)
mac = pop["logged_at_p"].apply(macro_at)
pop["macro_bull_r"] = mac.apply(lambda d: d.get("Bull") if d else np.nan)
pop["macro_sideways_r"] = mac.apply(lambda d: d.get("Sideways") if d else np.nan)
pop["macro_bear_r"] = mac.apply(lambda d: d.get("Bear") if d else np.nan)
pop["ms_state_r"] = pop["logged_at_p"].apply(ms_at)
pop["vd_state_r"] = pop["logged_at_p"].apply(vd_at)
print("joining (C) vwap sidecar + p_up_v3 preds...")
vw = pd.read_csv("results/btc_vwap_hmm_states_15m.csv")
vw["logged_at"] = pd.to_datetime(vw["logged_at"], utc=True, errors="coerce")
vw = vw.dropna(subset=["logged_at"]).sort_values("logged_at")
pop = pop.sort_values("logged_at_p")
pop = pd.merge_asof(pop, vw[["logged_at", "vwap_hmm_state"]].rename(
    columns={"logged_at": "vw_ts", "vwap_hmm_state": "vwap_state_r"}),
    left_on="logged_at_p", right_on="vw_ts", direction="backward",
    tolerance=pd.Timedelta("30min"))
wf = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/wf_preds_FINAL.parquet").dropna(subset=["p"]).sort_index()
wfs = wf["p"]
def pup3_at(ts):
    i = wfs.index.searchsorted(ts, side="right") - 1
    if i < 0 or (ts - wfs.index[i]) > pd.Timedelta("2h"): return np.nan
    return float(wfs.iloc[i])
pop["p_up_v3_r"] = pop["logged_at_p"].apply(pup3_at)

RECON = [c for c in pop.columns if c.endswith("_r")]
for c in RECON:
    print(f"  {c}: {pop[c].notna().sum()}/{len(pop)}")

# ── sweep everything reconstructed + other-asset markov categoricals ──────
found, n_tests = [], 0
for feat in RECON:
    col = pd.to_numeric(pop[feat], errors="coerce")
    if col.notna().sum() < 15:
        print(f"  SKIP {feat} ({col.notna().sum()} non-null)"); continue
    if col.dropna().nunique() <= 10 and feat.endswith(("_state_r", "state_r", "bo_5m_r", "bo_15m_r", "brk_5m_r", "brk_15m_r", "scross_5m_r", "scross_15m_r")):
        for val in col.dropna().unique():
            n_tests += 1
            s = pop[col == val]
            if len(s) < 8 or len(pop) - len(s) < 8: continue
            found.append({"feature": feat, "split": f"=={val}", "n": len(s),
                          "edge": s["won"].mean() - s["be"].mean()})
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for d, mask in [(">=", col >= th), ("<", col < th)]:
            n_tests += 1
            s = pop[mask.fillna(False)]
            if len(s) < 8 or len(pop) - len(s) < 8: continue
            found.append({"feature": feat, "split": f"{d}{th:.4g}(q{q:.1f})", "n": len(s),
                          "edge": s["won"].mean() - s["be"].mean()})
for feat in ["markov_eth_daily", "markov_sol_6h", "markov_sol_4h", "markov_sol_1h"]:
    if feat not in pop.columns: continue
    col = pop[feat].astype(str).replace("nan", np.nan)
    if col.notna().sum() < 15: continue
    for val in col.dropna().unique():
        n_tests += 1
        s = pop[col == val]
        if len(s) < 8 or len(pop) - len(s) < 8: continue
        found.append({"feature": feat, "split": f"=={val}", "n": len(s),
                      "edge": s["won"].mean() - s["be"].mean()})

fd = pd.DataFrame(found)
print(f"\n{n_tests} additional splits tested on reconstructed/joined signals")
real = fd[(fd["edge"] > 0) & (fd["n"] >= 10)].sort_values("edge", ascending=False)
print(f"positive-edge (n>=10): {len(real)}")
print(real.head(10).round(3).to_string(index=False))
print("\nbootstrap (trade + expiry-cluster) on top 5:")
for _, r in real.head(5).iterrows():
    feat, split = r["feature"], r["split"]
    col = pd.to_numeric(pop[feat], errors="coerce") if not feat.startswith("markov") else None
    if feat.startswith("markov"):
        mask = pop[feat].astype(str) == split[2:]
    elif split.startswith("=="):
        mask = col == float(split[2:])
    elif split.startswith(">="):
        mask = col >= float(split.split("(")[0][2:])
    else:
        mask = col < float(split.split("(")[0][1:])
    s = pop[mask.fillna(False)]
    e = (s["won"].astype(float) - s["be"]).values; n = len(e)
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(3000)])
    ce = np.array([(g["won"].astype(float) - g["be"]).mean() for _, g in s.groupby("close_time")])
    cmeans = np.array([ce[rng.integers(0, len(ce), len(ce))].mean() for _ in range(3000)])
    print(f"  {feat} {split}: n={len(s)} edge={means.mean():+.3f} P_trade={(means<=0).mean():.3f} "
          f"P_cluster={(cmeans<=0).mean():.3f}")
pop.to_csv(f"{OUT}/btc15m_bucket_full_recon.csv", index=False)
print("\nsaved btc15m_bucket_full_recon.csv")
