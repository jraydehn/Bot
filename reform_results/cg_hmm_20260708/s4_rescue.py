"""
S4 -- comprehensive_rescue on the proposed block bucket: taken NO trades in
CG states {neutral, buy-flow, SHORT-SQUEEZE} (n=153, -$2,489.50).
Phases: enumerate every column -> coverage audit -> reconstruct missing/thin
(GARCH, macro regime, Donchian, Keltner*, Kalman*, ARIMA*, ms/vd/of HMM states
-- *thin where live logging started late) from BTC's own 1h history using the
exact live formulas -> decile sweep both directions -> bootstrap survivors.
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(37)
OUT = "reform_results/cg_hmm_20260708"

t = pd.read_csv(f"{OUT}/taken_with_cg_state.csv", low_memory=False)
t["logged_at_p"] = pd.to_datetime(t["logged_at_p"], utc=True, errors="coerce")
pop = t[(t["cg_state"].isin([1, 2, 5])) & (t["side"] == "no")].copy()
print(f"bucket: n={len(pop)}  WR={pop['won'].mean():.3f}  BE={pop['be'].mean():.3f}  "
      f"PnL=${pd.to_numeric(pop['would_pnl'],errors='coerce').sum():+.2f}")

# ── Phase 3 reconstruction (BTC-native models exist for all of these) ─────
c1h_full = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/hist_BTCUSDT_1h.parquet").sort_index()
close = c1h_full["close"].astype(float)
high = c1h_full["high"].astype(float)
low = c1h_full["low"].astype(float)
with open("reform_results/hmm_macro_regime_btc.pkl", "rb") as f:
    macro_pkg = pickle.load(f)
with open("models/hmm_microstructure_btc.pkl", "rb") as f:
    ms_pkg = pickle.load(f)
with open("models/hmm_voldirection_btc.pkl", "rb") as f:
    vd_pkg = pickle.load(f)


def garch_ratio_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 502: return np.nan
    w = np.log(close.iloc[idx-500:idx+1] / close.iloc[idx-500:idx+1].shift(1)).dropna() * 100
    try:
        from arch import arch_model
        res = arch_model(w, vol="Garch", p=1, q=1, dist="normal", rescale=False).fit(disp="off", show_warning=False)
        cond = float(res.conditional_volatility.iloc[-1])
        om, al, be_ = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
        lr_vol = float(np.sqrt(om / (1 - al - be_))) if al + be_ < 1 else float(w.std())
        return cond / lr_vol if lr_vol > 0 else np.nan
    except Exception:
        return np.nan


def macro_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 80: return None
    w = close.iloc[max(0, idx-200):idx+1]
    lr = np.log(w / w.shift(1))
    fe = pd.DataFrame({"ret_24h": lr.rolling(24, min_periods=12).sum(),
                       "ret_72h": lr.rolling(72, min_periods=36).sum(),
                       "rv24": lr.rolling(24, min_periods=12).std(),
                       "sharpe_24h": (lr.rolling(24, min_periods=12).mean() /
                                      lr.rolling(24, min_periods=12).std().replace(0, np.nan)).fillna(0.0)}).dropna()
    if len(fe) < 10: return None
    X = macro_pkg["scaler"].transform(fe.iloc[-80:].values.astype(float))
    last = macro_pkg["model"].predict_proba(X)[-1]
    return {macro_pkg["label_names"][s]: float(last[s]) for s in range(len(last))}


def donch_at(ts):
    idx = high.index.searchsorted(ts, side="right") - 1
    if idx < 20: return np.nan
    h = float(high.iloc[idx-19:idx+1].max()); l = float(low.iloc[idx-19:idx+1].min())
    c = float(close.iloc[idx])
    return (c - l) / (h - l) if h > l else np.nan


def keltner_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 30: return np.nan
    c = close.iloc[max(0, idx-100):idx+1]; h = high.iloc[max(0, idx-100):idx+1]; l = low.iloc[max(0, idx-100):idx+1]
    ema10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    up, lo_ = ema10 + 1.5*atr14, ema10 - 1.5*atr14
    w = float((up-lo_).iloc[-1])
    return (float(c.iloc[-1]) - float(lo_.iloc[-1])) / w if w > 0 else np.nan


def kalman_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 30: return np.nan, np.nan
    c = close.iloc[max(0, idx-60):idx+1]
    lr = np.log(c.values[1:] / c.values[:-1]); kl = lr[-48:] if len(lr) >= 48 else lr
    if len(kl) < 5: return np.nan, np.nan
    Q = np.array([[1e-5,0],[0,1e-5]]); R = float(np.var(kl)) + 1e-10
    x = np.array([kl[0], 0.0]); P = np.eye(2)*0.1
    F = np.array([[1.,1.],[0.,1.]]); H = np.array([[1.,0.]])
    for obs in kl:
        x = F@x; P = F@P@F.T + Q
        K = P@H.T / (float(H@P@H.T) + R)
        x = x + K.flatten()*(obs - float(H@x)); P = (np.eye(2) - np.outer(K.flatten(), H))@P
    return float(x[1]), float(kl[-1] - float(H@x))


def ms_state_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 72: return None
    c1h = close.iloc[max(0, idx-300):idx+1].dropna()
    lr = np.log(c1h.values[1:] / c1h.values[:-1])
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
            r = dev.max() - dev.min(); s = seg.std(ddof=1)
            if s > 0 and r > 0: pts.append((np.log(w), np.log(r/s)))
        hu = float(np.clip(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0], 0, 1)) if len(pts) >= 2 else 0.5
        b60 = la[-60:] if len(la) >= 60 else la
        x6 = b60[:-1]-b60[:-1].mean(); y6 = b60[1:]-b60[1:].mean()
        den = float(np.sqrt((x6**2).sum()*(y6**2).sum()))
        ac = float(np.dot(x6, y6)/den) if den > 0 else 0.0
        rv = float(np.std(la[-24:] if len(la) >= 24 else la, ddof=1))
        return ot, hu, ac, rv
    Q = np.array([[1e-5,0],[0,1e-5]]); F = np.array([[1.,1.],[0.,1.]]); H = np.array([[1.,0.]])
    x = np.array([lr[0], 0.0]); P = np.eye(2)*0.1
    kv = np.zeros(len(lr))
    for i in range(len(lr)):
        R = float(np.var(lr[max(0,i-47):i+1])) + 1e-10 if i > 0 else 1e-10
        if i > 0: x = F@x; P = F@P@F.T + Q
        K = P@H.T / (float(H@P@H.T) + R)
        x = x + K.flatten()*(lr[i] - float(H@x)); P = (np.eye(2) - np.outer(K.flatten(), H))@P
        kv[i] = x[1]
    seq = min(60, len(lr))
    rows = []
    for i in range(len(lr)-seq, len(lr)):
        ot, hu, ac, rv = _pt(lr[:i+1])
        rows.append([ot, hu, ac, round(float(kv[i]), 6), rv])
    X = ms_pkg["scaler"].transform(np.array(rows))
    return int(ms_pkg["model"].predict(X)[-1])


def vd_state_at(ts):
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 200: return None
    c1h = close.iloc[max(0, idx-400):idx+1].dropna()
    cv = c1h.values; lr = np.log(cv[1:] / cv[:-1])
    if len(lr) < 175: return None
    a8, a21 = 2/9, 2/22
    e8 = np.empty(len(cv)); e8[0] = cv[0]
    e21 = np.empty(len(cv)); e21[0] = cv[0]
    for i in range(1, len(cv)):
        e8[i] = a8*cv[i] + (1-a8)*e8[i-1]; e21[i] = a21*cv[i] + (1-a21)*e21[i-1]
    def _pt(i):
        c1 = float(lr[i]); c3 = float(lr[i-2]+lr[i-1]+lr[i]) if i >= 2 else c1
        ws = lr[max(0,i-23):i+1]; rv1 = float(np.std(ws, ddof=1)) if len(ws) >= 4 else np.nan
        wl = lr[max(0,i-167):i+1]; rvl = float(np.std(wl, ddof=1)) if len(wl) >= 24 else np.nan
        vr = float(rv1/rvl) if rvl and rvl > 0 else np.nan
        et = float((e8[i+1]-e21[i+1])/cv[i+1]) if cv[i+1] > 0 else np.nan
        return c1, c3, rv1, vr, et
    seq = min(60, len(lr)-167)
    if seq < 5: return None
    rows = [_pt(i) for i in range(len(lr)-seq, len(lr))]
    fs = np.array(rows)
    if np.isnan(fs).any(): return None
    X = vd_pkg["scaler"].transform(fs)
    return int(vd_pkg["model"].predict(X)[-1])


print("reconstructing missing signals for the bucket...")
pop["garch_ratio_recon"] = pop["logged_at_p"].apply(garch_ratio_at)
mac = pop["logged_at_p"].apply(macro_at)
pop["macro_bull_recon"] = mac.apply(lambda d: d.get("Bull") if d else np.nan)
pop["macro_sideways_recon"] = mac.apply(lambda d: d.get("Sideways") if d else np.nan)
pop["macro_bear_recon"] = mac.apply(lambda d: d.get("Bear") if d else np.nan)
pop["donch_1h_recon"] = pop["logged_at_p"].apply(donch_at)
pop["kc_pct_1h_recon"] = pop["logged_at_p"].apply(keltner_at)
kal = pop["logged_at_p"].apply(kalman_at)
pop["kalman_velocity_recon"] = kal.apply(lambda x: x[0])
pop["kalman_residual_recon"] = kal.apply(lambda x: x[1])
pop["hmm_ms_state_recon"] = pop["logged_at_p"].apply(ms_state_at)
pop["hmm_vd_state_recon"] = pop["logged_at_p"].apply(vd_state_at)
RECON = ["garch_ratio_recon","macro_bull_recon","macro_sideways_recon","macro_bear_recon",
         "donch_1h_recon","kc_pct_1h_recon","kalman_velocity_recon","kalman_residual_recon",
         "hmm_ms_state_recon","hmm_vd_state_recon"]
for c in RECON:
    print(f"  {c}: {pop[c].notna().sum()}/{len(pop)}")

# ── Phase 1+2: enumerate every column, audit coverage, sweep ─────────────
EXCLUDE = {"logged_at","decision_time","contract_ticker","close_ts","p_market_source",
           "decision","side","gate_blocked","kelly_fraction","bet_fraction","bet_amount",
           "bankroll","contracts_scanned","resolved_yes","would_win","would_pnl",
           "spot_at_expiry","price_move_pct","miss_pct","loss_margin_pct","loss_category",
           "logged_at_p","cg_ts","week","pw","be","won","tedge","cg_state","spot","strike",
           "neutral_gate","pure_edge_gate","pup_v3_hmm_state"}
cands = [c for c in pop.columns if c not in EXCLUDE]
found, skipped, n_tests = [], [], 0
for feat in cands:
    col = pd.to_numeric(pop[feat], errors="coerce") if pop[feat].dtype == object else pop[feat]
    nn = col.notna().sum()
    if nn < 15:
        skipped.append((feat, int(nn))); continue
    if col.dropna().nunique() <= 6:
        for val in col.dropna().unique():
            n_tests += 1
            s = pop[col == val]
            if len(s) < 10 or len(pop) - len(s) < 10: continue
            found.append({"feature": feat, "split": f"=={val}", "n": len(s),
                          "wr": s["won"].mean(), "be": s["be"].mean(),
                          "edge": s["won"].mean() - s["be"].mean(),
                          "pnl": pd.to_numeric(s["would_pnl"], errors="coerce").sum()})
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for d, mask in [(">=", col >= th), ("<", col < th)]:
            n_tests += 1
            s = pop[mask.fillna(False)]
            if len(s) < 10 or len(pop) - len(s) < 10: continue
            found.append({"feature": feat, "split": f"{d}{th:.4g}(q{q:.1f})", "n": len(s),
                          "wr": s["won"].mean(), "be": s["be"].mean(),
                          "edge": s["won"].mean() - s["be"].mean(),
                          "pnl": pd.to_numeric(s["would_pnl"], errors="coerce").sum()})

print(f"\n{len(cands)} candidate cols; {len(skipped)} skipped <15 non-null; {n_tests} splits tested")
print("skipped:", [s[0] for s in skipped])
fd = pd.DataFrame(found)
real = fd[(fd["edge"] > 0) & (fd["n"] >= 15)].sort_values("edge", ascending=False)
print(f"\npositive-edge splits (n>=15): {len(real)}")
print(real.head(12).round(3).to_string(index=False))

def boot_p(sub, n_boot=4000):
    e = (sub["won"].astype(float) - sub["be"]).values; n = len(e)
    means = np.array([e[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (means <= 0).mean()

print("\nbootstrap on top candidates:")
for _, r in real.head(6).iterrows():
    feat, split = r["feature"], r["split"]
    col = pd.to_numeric(pop[feat], errors="coerce") if pop[feat].dtype == object else pop[feat]
    if split.startswith("=="):
        mask = col == float(split[2:])
    elif split.startswith(">="):
        mask = col >= float(split.split("(")[0][2:])
    else:
        mask = col < float(split.split("(")[0][1:])
    s = pop[mask.fillna(False)]
    m, lo, hi, pb = boot_p(s)
    print(f"  {feat} {split}: n={len(s)} edge={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] P(<=0)={pb:.4f}")
pop.to_csv(f"{OUT}/bucket_reconstructed.csv", index=False)
print("\nsaved bucket_reconstructed.csv")
