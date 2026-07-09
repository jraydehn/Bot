"""
S4 -- EXHAUSTIVE within-bucket rescue battery (user-directed, "nothing unturned"):
Donchian/Keltner/Bollinger/RSI/stoch at 5m/15m/1h (levels + breakout flags +
stoch CROSSOVER states), ALL reconstructable SOL HMM regimes (vol-ergodic R0/R1,
vwap-MTF, microstructure, vol-direction -- orderflow DISCLOSED-SKIPPED: its
inputs vpin/obi/funding are not logged at 15m and not candle-derivable),
offset (raw + sign + ATR-norm), every price-change TF (1m/5m/15m/1h/4h + BTC
cross), kalman (velocity/residual logged + recon), all CoinGlass columns,
bp/dir at all logged TFs, volatility family (vol_ratio*, realized_vol,
atr/range ratios, wicks, bodies), GARCH ratio (reconstructed, hourly-cached),
ARIMA 15m+1h forecasts (reconstructed), tau, spread, z_score, OU family,
autocorr, macd, ema distances, engulfing, consec_dir, nearest resistance.

All reconstructions zero-lookahead (completed bars only). Rescue bar:
episode-clustered P(edge<=0)<=0.05, n>=80, ZERO streak leakage, >=60% positive
weeks. Everything tested is counted; near-misses reported.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1031)
OUT = "reform_results/sol15m_streak_20260709"
STREAK = ["KXSOL15M-26JUL090030-30", "KXSOL15M-26JUL090100-00",
          "KXSOL15M-26JUL090115-15", "KXSOL15M-26JUL090145-45"]

bucket = pd.read_csv(f"{OUT}/bucket_novel_features.csv", low_memory=False)
bucket["logged_at_p"] = pd.to_datetime(bucket["logged_at_p"], utc=True)
bucket["week"] = bucket["logged_at_p"].dt.to_period("W-FRI").astype(str)
print(f"bucket: n={len(bucket)} edge={bucket['tedge'].mean():+.4f}")

p1m = sorted(pathlib.Path("data").glob("binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2026-04-10"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df5 = df1m.resample("5min").agg(AGG).dropna()
df15 = df1m.resample("15min").agg(AGG).dropna()
df60 = df1m.resample("1h").agg(AGG).dropna()
FR = {"5m": (df5, 5), "15m": (df15, 15), "1h": (df60, 60)}


def bars_before(df, ts, fm, n):
    cutoff = ts - pd.Timedelta(minutes=fm)
    i = df.index.searchsorted(cutoff, side="right") - 1
    if i < 25:
        return None
    return df.iloc[max(0, i - n):i + 1]


# ── stoch crossover + breakout flags at 5m/15m/1h ─────────────────────────
def scross_at(df, ts, fm):
    b = bars_before(df, ts, fm, 25)
    if b is None or len(b) < 18:
        return np.nan
    lo14, hi14 = b["low"].rolling(14).min(), b["high"].rolling(14).max()
    sk = ((b["close"] - lo14) / (hi14 - lo14).replace(0, np.nan)) * 100
    sd = sk.rolling(3).mean()
    if pd.isna(sk.iloc[-1]) or pd.isna(sd.iloc[-1]) or pd.isna(sd.iloc[-2]):
        return np.nan
    if sk.iloc[-1] > sd.iloc[-1] and sk.iloc[-2] <= sd.iloc[-2]:
        return 1
    if sk.iloc[-1] < sd.iloc[-1] and sk.iloc[-2] >= sd.iloc[-2]:
        return -1
    return 0


def flags_at(df, ts, fm):
    b = bars_before(df, ts, fm, 30)
    if b is None or len(b) < 22:
        return np.nan, np.nan
    c, h, l = b["close"], b["high"], b["low"]
    dh, dl = h.rolling(20).max().iloc[-1], l.rolling(20).min().iloc[-1]
    last = float(c.iloc[-1])
    donb = 1 if last >= dh else (-1 if last <= dl else 0)
    e10 = c.ewm(span=10, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(span=14, adjust=False).mean()
    up, lo = float((e10 + 1.5 * a14).iloc[-1]), float((e10 - 1.5 * a14).iloc[-1])
    kcb = 1 if last > up else (-1 if last < lo else 0)
    return donb, kcb


print("reconstructing crossovers + breakout flags (3 TFs)...")
for tf, (df, fm) in FR.items():
    bucket[f"x_scross_{tf}"] = bucket["logged_at_p"].apply(lambda ts: scross_at(df, ts, fm))
    fl = bucket["logged_at_p"].apply(lambda ts: flags_at(df, ts, fm))
    bucket[f"x_donbrk_{tf}"] = fl.apply(lambda t: t[0])
    bucket[f"x_kcbo_{tf}"] = fl.apply(lambda t: t[1])
    print(f"  {tf} done")

# ── HMM regimes ───────────────────────────────────────────────────────────
print("reconstructing HMM regimes...")
# (a) ergodic 2-state vol (R0/R1) on 15m returns, live-matching 20-bar decode
vp = pickle.load(open("models/hmm_ergodic_2state_sol_15m.pkl", "rb"))
vm = vp["model"]
order = sorted(range(vp["n_states"]), key=lambda s: float(np.sqrt(vm.covars_[s].ravel()[0])))
rankof = {s: i for i, s in enumerate(order)}

def vol_state_at(ts):
    b = bars_before(df15, ts, 15, 22)
    if b is None or len(b) < 21:
        return np.nan
    lr = np.log(b["close"] / b["close"].shift(1)).dropna().values[-20:]
    try:
        seq = vm.predict(lr.reshape(-1, 1))
        return rankof[int(seq[-1])]
    except Exception:
        return np.nan
bucket["x_volhmm_R"] = bucket["logged_at_p"].apply(vol_state_at)
print(f"  vol-ergodic done ({bucket['x_volhmm_R'].notna().sum()}/{len(bucket)})")

# (b) vwap-MTF (full-series decode, +15min effective, asof join)
wp = pickle.load(open("models/hmm_vwap_mtf_sol_15m.pkl", "rb"))
def rvwap(df, n):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ctv = (tp * df["volume"]).rolling(n, min_periods=n).sum()
    cv = df["volume"].rolling(n, min_periods=n).sum()
    vw = ctv / cv.replace(0, np.nan)
    return (df["close"] - vw) / vw.replace(0, np.nan) * 100
d1 = rvwap(df1m, 20); d5v = rvwap(df5, 20); d15v = rvwap(df15, 20)
feat = pd.DataFrame(index=df15.index)
feat["vwap_dist_15m"] = d15v
feat["vwap_dist_5m"] = d5v.resample("15min").last()
feat["vwap_dist_1m"] = d1.resample("15min").last()
feat["vwap_vel_1m"] = d1.diff().resample("15min").last()
feat["vwap_spread"] = feat["vwap_dist_1m"] - feat["vwap_dist_15m"]
feat = feat.dropna()
X = wp["scaler"].transform(feat[wp["feat_cols"]].values)
g = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
st = [0] + list(np.where(g > 1800)[0]); en = st[1:] + [len(X)]
lens = [e - s for s, e in zip(st, en) if e - s >= 3]
vidx = [i for s, e in zip(st, en) if e - s >= 3 for i in range(s, e)]
states = wp["model"].predict(X[vidx], lengths=lens)
sv = pd.DataFrame({"eff": feat.index[vidx] + pd.Timedelta("15min"), "vwst": states}).sort_values("eff")
bucket = pd.merge_asof(bucket.sort_values("logged_at_p"), sv, left_on="logged_at_p",
                       right_on="eff", direction="backward", tolerance=pd.Timedelta("45min"))
bucket = bucket.rename(columns={"vwst": "x_vwap_hmm"})
print(f"  vwap-MTF done ({bucket['x_vwap_hmm'].notna().sum()}/{len(bucket)})")

# (c) microstructure + (d) vol-direction, per-trade trailing-sequence decode on 1h bars
ms_pkg = pickle.load(open("models/hmm_microstructure_sol.pkl", "rb"))
vd_pkg = pickle.load(open("models/hmm_voldirection_sol.pkl", "rb"))
close1h = df60["close"].astype(float)

def ms_at(ts):
    cutoff = ts - pd.Timedelta(minutes=60)
    idx = close1h.index.searchsorted(cutoff, side="right") - 1
    if idx < 72:
        return np.nan
    c = close1h.iloc[max(0, idx - 300):idx + 1].dropna()
    lr = np.log(c.values[1:] / c.values[:-1])
    if len(lr) < 70:
        return np.nan
    def _pt(la):
        b48 = la[-48:] if len(la) >= 48 else la
        y = b48 - b48.mean()
        phi = float(np.clip(np.dot(y[:-1], y[1:]) / (np.dot(y[:-1], y[:-1]) + 1e-12), -0.9999, 0.9999))
        ot = float(np.clip(-np.log(abs(phi)), 0, 10))
        pts = []
        for w in [8, 16, 32, 64]:
            if len(la) < w:
                continue
            seg = la[-w:]; dev = np.cumsum(seg - seg.mean())
            r, s = dev.max() - dev.min(), seg.std(ddof=1)
            if s > 0 and r > 0:
                pts.append((np.log(w), np.log(r / s)))
        hu = float(np.clip(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0], 0, 1)) if len(pts) >= 2 else 0.5
        b60_ = la[-60:] if len(la) >= 60 else la
        x6 = b60_[:-1] - b60_[:-1].mean(); y6 = b60_[1:] - b60_[1:].mean()
        den = float(np.sqrt((x6 ** 2).sum() * (y6 ** 2).sum()))
        ac = float(np.dot(x6, y6) / den) if den > 0 else 0.0
        rv = float(np.std(la[-24:] if len(la) >= 24 else la, ddof=1))
        return ot, hu, ac, rv
    Q = np.array([[1e-5, 0], [0, 1e-5]]); F = np.array([[1., 1.], [0., 1.]]); H = np.array([[1., 0.]])
    x = np.array([lr[0], 0.0]); P = np.eye(2) * 0.1
    kv = np.zeros(len(lr))
    for i2 in range(len(lr)):
        R = float(np.var(lr[max(0, i2 - 47):i2 + 1])) + 1e-10 if i2 > 0 else 1e-10
        if i2 > 0:
            x = F @ x; P = F @ P @ F.T + Q
        K = P @ H.T / (float(H @ P @ H.T) + R)
        x = x + K.flatten() * (lr[i2] - float(H @ x)); P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        kv[i2] = x[1]
    seq = min(60, len(lr))
    rows = []
    for i2 in range(len(lr) - seq, len(lr)):
        ot, hu, ac, rv = _pt(lr[:i2 + 1])
        rows.append([ot, hu, ac, round(float(kv[i2]), 6), rv])
    try:
        return int(ms_pkg["model"].predict(ms_pkg["scaler"].transform(np.array(rows)))[-1])
    except Exception:
        return np.nan

def vd_at(ts):
    cutoff = ts - pd.Timedelta(minutes=60)
    idx = close1h.index.searchsorted(cutoff, side="right") - 1
    if idx < 200:
        return np.nan
    c = close1h.iloc[max(0, idx - 400):idx + 1].dropna()
    cv = c.values; lr = np.log(cv[1:] / cv[:-1])
    if len(lr) < 175:
        return np.nan
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
    if seq < 5:
        return np.nan
    fs = np.array([_pt(i2) for i2 in range(len(lr) - seq, len(lr))])
    if np.isnan(fs).any():
        return np.nan
    try:
        return int(vd_pkg["model"].predict(vd_pkg["scaler"].transform(fs))[-1])
    except Exception:
        return np.nan

bucket["x_ms_state"] = bucket["logged_at_p"].apply(ms_at)
print(f"  ms done ({bucket['x_ms_state'].notna().sum()}/{len(bucket)})")
bucket["x_vd_state"] = bucket["logged_at_p"].apply(vd_at)
print(f"  vd done ({bucket['x_vd_state'].notna().sum()}/{len(bucket)})")

# ── GARCH ratio (hourly-cached) + ARIMA forecasts ─────────────────────────
print("reconstructing GARCH (per unique hour) + ARIMA (15m & 1h)...")
from arch import arch_model
lr1h_full = np.log(close1h / close1h.shift(1)).dropna() * 100
garch_cache = {}
def garch_at(ts):
    hr = ts.floor("h")
    if hr in garch_cache:
        return garch_cache[hr]
    cutoff = ts - pd.Timedelta(minutes=60)
    idx = lr1h_full.index.searchsorted(cutoff, side="right") - 1
    if idx < 502:
        garch_cache[hr] = np.nan
        return np.nan
    w = lr1h_full.iloc[idx - 500:idx + 1]
    try:
        res = arch_model(w, vol="Garch", p=1, q=1, dist="normal", rescale=False).fit(disp="off", show_warning=False)
        cond = float(res.conditional_volatility.iloc[-1])
        om, al, be_ = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
        lrv = float(np.sqrt(om / (1 - al - be_))) if al + be_ < 1 else float(w.std())
        out = cond / lrv if lrv > 0 else np.nan
    except Exception:
        out = np.nan
    garch_cache[hr] = out
    return out
bucket["x_garch"] = bucket["logged_at_p"].apply(garch_at)
print(f"  garch done ({bucket['x_garch'].notna().sum()}/{len(bucket)}, {len(garch_cache)} fits)")

from statsmodels.tsa.arima.model import ARIMA
def arima_at(df, ts, fm):
    b = bars_before(df, ts, fm, 250)
    if b is None or len(b) < 60:
        return np.nan
    lr = np.log(b["close"] / b["close"].shift(1)).dropna()
    try:
        return float(ARIMA(lr, order=(2, 0, 1)).fit().forecast(steps=1).iloc[0])
    except Exception:
        return np.nan
bucket["x_arima_15m"] = bucket["logged_at_p"].apply(lambda ts: arima_at(df15, ts, 15))
print(f"  arima 15m done ({bucket['x_arima_15m'].notna().sum()}/{len(bucket)})")
bucket["x_arima_1h"] = bucket["logged_at_p"].apply(lambda ts: arima_at(df60, ts, 60))
print(f"  arima 1h done ({bucket['x_arima_1h'].notna().sum()}/{len(bucket)})")

bucket.to_csv(f"{OUT}/bucket_exhaustive.csv", index=False)

# ── THE SWEEP ─────────────────────────────────────────────────────────────
streak_mask = bucket["contract_ticker"].isin(STREAK)

def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()

NUMERIC = (
    # recon MTF families (from s1) at 5m/15m/1h + crossflags
    [f"r_{nm}_{tf}" for nm in ["chg", "bb_pctb", "bb_width", "rsi", "stoch", "donch", "kc_pctb"]
     for tf in ["5m", "15m", "1h"]] +
    ["x_garch", "x_arima_15m", "x_arima_1h"] +
    # logged numerics
    ["offset_pct", "tau_minutes", "spread", "z_score", "p_model_15m", "raw_edge",
     "chg_1m", "chg_5m", "chg_15m", "chg_1h", "chg_4h",
     "bp_5m", "bp_15m", "bp_1h", "bp_4h",
     "vol_ratio", "vol_ratio_5m", "vol_ratio_1h", "realized_vol_annual",
     "atr_ratio_15m", "range_ratio_15m", "body_5m", "body_15m",
     "upper_wick_15m", "lower_wick_15m",
     "stoch_k_5m", "stoch_k_15m", "stoch_k_1h", "rsi_1h", "macd_hist_1h",
     "bb_pct_1h", "kc_pct_1h", "ema20_dist_1h", "ema50_dist_1h",
     "vwap_dist", "nearest_res_dist_pct",
     "kalman_velocity", "kalman_residual", "hurst_exponent",
     "ou_theta", "ou_halflife", "ou_mu_distance", "autocorr1_15", "autocorr1_30",
     "mu6h", "mu12h", "mu24h", "regime_z",
     "cvd_4h", "cg_futures_delta_4h", "cg_futures_ratio_4h", "cg_futures_cvd_12h",
     "fear_greed", "ls_long_pct", "oi_chg_pct", "liq_score", "composite_p_up",
     # from s3
     "z_delta_1h", "mins_below", "chg_b15", "sto_b15", "chg_b60", "sto_b60", "offset_atr"]
)
CATEG = (["x_scross_5m", "x_scross_15m", "x_scross_1h", "x_donbrk_5m", "x_donbrk_15m",
          "x_donbrk_1h", "x_kcbo_5m", "x_kcbo_15m", "x_kcbo_1h",
          "x_volhmm_R", "x_vwap_hmm", "x_ms_state", "x_vd_state",
          "dir_5m", "dir_15m", "dir_1h", "consec_dir_15m", "consec_dir_1h",
          "ema_bias", "ema_bias_1h", "kc_bo_1h", "donchian_breakout_1h",
          "engulfing_1h", "stoch_cross_1h", "cg_composite", "liq_bias",
          "markov_sol_6h", "markov_sol_4h", "markov_sol_1h", "markov_regime_1h",
          "markov_eth_daily"])

results, n_tests, coverage = [], 0, []
for feat in NUMERIC:
    if feat not in bucket.columns:
        coverage.append((feat, "MISSING"))
        continue
    col = pd.to_numeric(bucket[feat], errors="coerce")
    nn = int(col.notna().sum())
    coverage.append((feat, nn))
    if nn < 200 or col.dropna().nunique() < 6:
        continue
    vv = col.dropna()
    for q in np.arange(0.1, 1.0, 0.1):
        th = vv.quantile(q)
        for lab, mk in [(f">= {th:.4g}", col >= th), (f"< {th:.4g}", col < th)]:
            n_tests += 1
            d = bucket[mk.fillna(False)]
            if len(d) < 80 or len(bucket) - len(d) < 80:
                continue
            if d["tedge"].mean() < 0.005:
                continue
            ne, ee, p_neg = ep_stats(d)
            wk = d.groupby("week")["tedge"].mean()
            results.append({"feature": feat, "split": lab, "n": len(d), "eps": ne,
                            "edge": d["tedge"].mean(), "ep_edge": ee, "P_neg": p_neg,
                            "wk_pos": float((wk > 0).mean()), "n_wk": len(wk),
                            "streak_leak": int((mk.fillna(False) & streak_mask).sum()),
                            "pnl": d["would_pnl"].sum()})
for feat in CATEG:
    if feat not in bucket.columns:
        coverage.append((feat, "MISSING"))
        continue
    col = bucket[feat]
    nn = int(col.notna().sum())
    coverage.append((feat, nn))
    if nn < 200:
        continue
    for val in col.dropna().unique():
        n_tests += 1
        mk = (col == val).fillna(False)
        d = bucket[mk]
        if len(d) < 60:
            continue
        if d["tedge"].mean() < 0.005:
            continue
        ne, ee, p_neg = ep_stats(d)
        wk = d.groupby("week")["tedge"].mean()
        results.append({"feature": feat, "split": f"== {val}", "n": len(d), "eps": ne,
                        "edge": d["tedge"].mean(), "ep_edge": ee, "P_neg": p_neg,
                        "wk_pos": float((wk > 0).mean()), "n_wk": len(wk),
                        "streak_leak": int((mk & streak_mask).sum()),
                        "pnl": d["would_pnl"].sum()})

print(f"\n{'='*80}\nTOTAL tests: {n_tests}")
print("\ncoverage (features with <200 non-null or missing):")
for f, nn in coverage:
    if nn == "MISSING" or (isinstance(nn, int) and nn < 200):
        print(f"  {f}: {nn}")

rf = pd.DataFrame(results)
print(f"\npositive-edge subsets found: {len(rf)}")
if len(rf):
    survivors = rf[(rf["P_neg"] <= 0.05) & (rf["streak_leak"] == 0) & (rf["wk_pos"] >= 0.6) & (rf["n"] >= 80)]
    print(f"\nSURVIVORS (P<=0.05, zero streak leak, >=60% pos weeks, n>=80): {len(survivors)}")
    if len(survivors):
        print(survivors.sort_values("ep_edge", ascending=False).round(4).to_string(index=False))
    near = rf[(rf["P_neg"] <= 0.15)].sort_values("ep_edge", ascending=False)
    print(f"\nnear-misses (P<=0.15), top 15:")
    print(near.head(15).round(4).to_string(index=False))
rf.to_csv(f"{OUT}/exhaustive_rescue_results.csv", index=False)
print("DONE_S4")
