"""
S2 -- SOL HOURLY VWAP MTF HMM (the 15m one exists + is live; hourly doesn't).
Hourly rescale of the validated 15m design: rolling-20-bar VWAP distance at
1h / 4h / 1d, velocity on the 1h distance, spread = 1h dist - 1d dist.
Trained on SOL 2024->present. BIC 2-8 states.

Zero-lookahead from the start:
- Features at bar OPEN time T describe [T, T+frame) -> a state decoded at bar
  T is only knowable at T+frame. Backfill joins use effective = T + 1h (the
  finest frame; coarser features enter via completed-bar resampling so the
  1h-bar close is the binding constraint).
- Causal decode for the backfill: trailing-64 predict per point (no full-
  sequence forward-backward smoothing).

Validation: SOL HOURLY taken book (frame match) + SOL 15m book (context).
Episode-clustered bootstrap, week stability. Research-only.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1063)
OUT = "reform_results/sol_hmms_20260709"

p1m = sorted(pathlib.Path("data").glob("binanceus_SOLUSDT_1m_2024-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
d1h = df1m.resample("1h").agg(AGG).dropna()
d4h = df1m.resample("4h").agg(AGG).dropna()
d1d = df1m.resample("1D").agg(AGG).dropna()
print(f"1h bars: {len(d1h)}  {d1h.index.min()} -> {d1h.index.max()}")


def rvwap(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ctv = (tp * df["volume"]).rolling(n, min_periods=n).sum()
    cv = df["volume"].rolling(n, min_periods=n).sum()
    vw = ctv / cv.replace(0, np.nan)
    return (df["close"] - vw) / vw.replace(0, np.nan) * 100


# align coarser frames onto the 1h grid using only COMPLETED coarse bars:
# value at 1h-bar T = last coarse bar whose CLOSE <= T+1h (i.e. open+frame <= T+1h).
dist_1h = rvwap(d1h)
dist_4h = rvwap(d4h)
dist_1d = rvwap(d1d)

feat = pd.DataFrame(index=d1h.index)
feat["vwap_dist_1h"] = dist_1h
# completed-4h-bar value as of each 1h bar's close:
c4 = dist_4h.copy(); c4.index = c4.index + pd.Timedelta("4h")   # index = close time
feat["vwap_dist_4h"] = c4.reindex(feat.index + pd.Timedelta("1h"), method="ffill").values
c1d = dist_1d.copy(); c1d.index = c1d.index + pd.Timedelta("1D")
feat["vwap_dist_1d"] = c1d.reindex(feat.index + pd.Timedelta("1h"), method="ffill").values
feat["vwap_vel_1h"] = feat["vwap_dist_1h"].diff()
feat["vwap_spread"] = feat["vwap_dist_1h"] - feat["vwap_dist_1d"]
feat = feat.dropna()
FEAT_COLS = list(feat.columns)
print(f"feature matrix: {len(feat)} x {len(FEAT_COLS)}")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(feat.values)
g = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(g > 7200)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 10]
vidx = [i for s, e in zip(starts, ends) if e - s >= 10 for i in range(s, e)]
Xs = X[vidx]

print("\nBIC selection:")
best = (np.inf, None, None)
for n in range(2, 9):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=250, random_state=42, tol=1e-4)
        m.fit(Xs, lengths=lengths)
        ll = m.score(Xs, lengths=lengths)
        bic = -2 * ll + (n * n + n * len(FEAT_COLS) * 2) * np.log(len(Xs))
        print(f"  n={n}: BIC={bic:.1f}")
        if bic < best[0]:
            best = (bic, n, m)
    except Exception as e:
        print(f"  n={n} failed: {e}")
_, N, model = best
print(f"selected {N} states")

with open(f"{OUT}/hmm_vwap_mtf_sol_1h.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": N}, f)

# causal decode (trailing 64) over recent months only (validation span)
fi = feat.iloc[vidx].copy()
recent_mask = fi.index >= pd.Timestamp("2026-04-15", tz="UTC")
ridx = np.where(recent_mask)[0]
states = np.full(len(ridx), -1)
for j, i in enumerate(ridx):
    lo = max(0, i - 63)
    try:
        states[j] = int(model.predict(Xs[lo:i + 1])[-1])
    except Exception:
        pass
sv = pd.DataFrame({"bar_open": fi.index[ridx], "state": states})
sv = sv[sv["state"] >= 0]
sv["effective"] = sv["bar_open"] + pd.Timedelta("1h")
sv = sv.sort_values("effective")
print(f"\ncausal states decoded (recent): {len(sv)}")
print("distribution:", sv["state"].value_counts().sort_index().to_dict())
print("state centroids:")
for s in sorted(sv["state"].unique()):
    sub = fi.iloc[[j for j, i in enumerate(ridx) if j < len(states) and states[j] == s]]
    if len(sub):
        print(f"  S{s}: " + "  ".join(f"{c}={sub[c].mean():+.3f}" for c in FEAT_COLS) + f"  (n={len(sub)})")
sv.to_csv(f"{OUT}/vwap_sol_1h_states.csv", index=False)

# ── validate on SOL HOURLY taken book ─────────────────────────────────────
def book(path, is_15m):
    pt = pd.read_csv(path, low_memory=False)
    pt["logged_at_p"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
    for c in ["resolved_yes", "p_market", "would_pnl"]:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
    t = pt[pt["decision"] == "trade"].dropna(subset=["resolved_yes", "logged_at_p", "p_market"]).copy()
    t["side"] = t["side"].str.lower()
    t["won"] = np.where(t["side"] == "yes", t["resolved_yes"] == 1, t["resolved_yes"] == 0)
    t["be"] = np.where(t["side"] == "yes", t["p_market"], 1 - t["p_market"])
    t["tedge"] = t["won"].astype(float) - t["be"]
    t = t.sort_values("logged_at_p")
    gaps = t["logged_at_p"].diff().dt.total_seconds() / 60
    t["episode"] = (gaps > (45 if is_15m else 90)).cumsum()
    t["week"] = t["logged_at_p"].dt.to_period("W-FRI").astype(str)
    return pd.merge_asof(t, sv[["effective", "state"]], left_on="logged_at_p",
                         right_on="effective", direction="backward", tolerance=pd.Timedelta("2h"))


def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()


for path, name, is15 in [("results/paper_trades_sol.csv", "SOL HOURLY", False),
                         ("results/paper_trades_sol15m.csv", "SOL 15m", True)]:
    t = book(path, is15)
    t = t.dropna(subset=["state"])
    print(f"\n=== {name}: {len(t)} taken trades with state ===")
    for s in sorted(t["state"].dropna().unique()):
        for side in ["yes", "no"]:
            d = t[(t["state"] == s) & (t["side"] == side)]
            if len(d) < 15:
                continue
            ne, ee, pn = ep_stats(d)
            wk = d.groupby("week")["tedge"].mean()
            print(f"  S{int(s)} {side.upper():3s}: n={len(d)} eps={ne} edge={d['tedge'].mean():+.4f} "
                  f"ep_edge={ee:+.4f} P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} "
                  f"$={d['would_pnl'].sum():+.2f}")
print("DONE_S2")
