"""
S31 -- HMM regime on the metrics discovered today: rv_ratio(2h/120h, best
single predictor of touch/MAE), efficiency_ratio(2h), efficiency_ratio(6h),
atr_ratio(1h/240h, yesterday's companion feature). Question: does a joint
regime separate touch-risk more cleanly than rv_ratio alone (r2=0.076)?

Train GaussianHMM on 2024-2025 ONLY (2026 held out, including everything
this session has stared at). BIC-select state count. Causal decode (trailing
window predict, last state -- same convention as the pup_v3/CG HMMs built
earlier today, avoiding the Viterbi-smoothing lookahead bug found in s14).
Validate: does 2026-holdout state separation replicate what training implied?
Then test against the real 317-trade book.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
px = pd.read_parquet(p1m).sort_index()
px = px[px.index >= "2023-11-01"]
c1 = px["close"]
r1m = c1.pct_change()

def efficiency_ratio(close, window_min):
    net = (close - close.shift(window_min)).abs()
    path = close.diff().abs().rolling(window_min).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)

rv2h, rv120h = r1m.rolling(120).std(), r1m.rolling(7200).std()
rv_ratio = (rv2h / rv120h.replace(0, np.nan))
atr1h_proxy = (c1.rolling(60).max() - c1.rolling(60).min()) / c1  # simple range proxy at 1h
atr_ratio = atr1h_proxy / atr1h_proxy.rolling(14400).mean()  # vs 10-day mean
er2h = efficiency_ratio(c1, 120)
er6h = efficiency_ratio(c1, 360)

feat = pd.DataFrame({"log_rv_ratio": np.log(rv_ratio.clip(lower=0.05)),
                     "er2h": er2h, "er6h": er6h,
                     "log_atr_ratio": np.log(atr_ratio.clip(lower=0.05))}).dropna()
feat15 = feat.resample("15min").last().dropna()
print(f"feature bars (15m cadence): {len(feat15)}  {feat15.index.min()} -> {feat15.index.max()}")
feat15["year"] = feat15.index.year

train = feat15[feat15["year"] <= 2025]
FEAT_COLS = ["log_rv_ratio", "er2h", "er6h", "log_atr_ratio"]
scaler = StandardScaler().fit(train[FEAT_COLS].values)
X_train = scaler.transform(train[FEAT_COLS].values)

print("\nBIC selection (train: 2024-2025 only):")
best = (np.inf, None, None)
for n in range(3, 8):
    m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=300, random_state=42, tol=1e-4)
    m.fit(X_train)
    ll = m.score(X_train)
    n_params = n * n + 2 * n * X_train.shape[1] - 1
    bic = -2 * ll + n_params * np.log(len(X_train))
    print(f"  n={n}: BIC={bic:.1f}")
    if bic < best[0]:
        best = (bic, n, m)
_, n_states, model = best
print(f"selected {n_states} states")

# causal decode over ALL data (trailing window, last state) -- same convention as s14
X_all = scaler.transform(feat15[FEAT_COLS].values)
states = np.full(len(feat15), -1)
WIN = 500
for i in range(9, len(feat15)):
    lo = max(0, i - WIN + 1)
    states[i] = model.predict(X_all[lo:i + 1])[-1]
feat15["state"] = states
feat15 = feat15[feat15["state"] >= 0]
feat15["effective"] = feat15.index

cent = pd.DataFrame(scaler.inverse_transform(model.means_), columns=FEAT_COLS).round(3)
occ = feat15["state"].value_counts(normalize=True).sort_index()
cent["occ%"] = (occ * 100).round(1).values
print("\nstate centroids (unscaled):")
print(cent.to_string())

# ---- validate against the real touch/MAE trade population, 2026 holdout emphasis ----
pt = pd.read_csv(f"{OUT}/s26_preentry_features.csv", parse_dates=["decision_time"]).sort_values("decision_time")
sd = feat15[["effective", "state"]].reset_index(drop=True)
pt2 = pd.merge_asof(pt, sd.sort_values("effective"), left_on="decision_time", right_on="effective",
                    direction="backward").dropna(subset=["state"])
print(f"\n=== real trades (n={len(pt2)}) by HMM state ===")
for s, g in pt2.groupby("state"):
    print(f"  state {int(s)}: n={len(g):3d}  WR={g['win'].mean():.1%}  touched%={g['touched_strike'].mean():.1%}  "
        f"mae_mean={g['mae_pct'].mean():+.4f}")

# ---- 2026 holdout: causal-decoded state vs realized outcome on the SYNTHETIC bet (huge n) ----
syn = pd.read_csv(f"{OUT}/synthetic_yes_bets.csv", parse_dates=["dec"])[["dec", "win", "year", "day"]]
syn26 = syn[syn["year"] == 2026]
sd2 = feat15[["effective", "state"]].reset_index(drop=True)
syn26 = pd.merge_asof(syn26.sort_values("dec"), sd2.sort_values("effective"), left_on="dec", right_on="effective",
                      direction="backward").dropna(subset=["state"])
print(f"\n=== 2026 holdout synthetic bet (n={len(syn26)}) by state ===")
rng = np.random.default_rng(11)
for s, g in syn26.groupby("state"):
    dwr = g.groupby("day")["win"].mean()
    print(f"  state {int(s)}: n={len(g):6d}  WR={g['win'].mean():.4f}  days={dwr.size}")

# R2 comparison: state (categorical, one-hot) vs rv_ratio alone, on touch/MAE
import numpy.linalg as la
dummies = pd.get_dummies(pt2["state"].astype(int), prefix="st").astype(float)
X_state = np.column_stack([np.ones(len(pt2))] + [dummies[c].values for c in dummies.columns])
rv_only = pd.read_csv(f"{OUT}/s28_log.txt") if False else None  # placeholder, recompute inline
rv_df = rv_ratio.rename("rv").reset_index(); rv_df.columns = ["ts", "rv"]
pt3 = pd.merge_asof(pt2.sort_values("decision_time"), rv_df.sort_values("ts"), left_on="decision_time",
                    right_on="ts", direction="backward").dropna(subset=["rv"])
X_rv = np.column_stack([np.ones(len(pt3)), pt3["rv"]])
dummies3 = pd.get_dummies(pt3["state"].astype(int), prefix="st").astype(float)
X_state3 = np.column_stack([np.ones(len(pt3))] + [dummies3[c].values for c in dummies3.columns])
for target in ["touched_strike", "mae_pct"]:
    y = pt3[target].astype(float).values
    ss_tot = ((y - y.mean()) ** 2).sum()
    b_rv, *_ = la.lstsq(X_rv, y, rcond=None)
    r2_rv = 1 - ((y - X_rv @ b_rv) ** 2).sum() / ss_tot
    b_st, *_ = la.lstsq(X_state3, y, rcond=None)
    r2_st = 1 - ((y - X_state3 @ b_st) ** 2).sum() / ss_tot
    print(f"\n  {target}: R2(rv_ratio alone)={r2_rv:.4f}   R2(HMM state, {n_states} dummies)={r2_st:.4f}")

with open(f"{OUT}/hmm_touch_risk_btc15m.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": n_states,
                "trained_on": "2024-01..2025-12"}, f)
print("\nDONE_S31")
