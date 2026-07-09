"""
S3 -- SOL CoinGlass flow-regime HMM. Same emission design as BTC's
(validated formulas, funding excluded): fut_ratio_1h, fut_cvd_12h,
spot_ratio_1h, oi_chg_4h, liq_imb_4h, liq_tot_z_10d. BIC 2-8.

Zero-lookahead FROM THE START (the BTC version's original sin, corrected
same-day): CG bars are open-time indexed; bar T completes at T+1h; states
join to trades on effective = T + 1h.

Validation: both SOL books (15m primary -- bigger + live), per-state x side,
episode-clustered, week stability. Orthogonality RF check vs logged SOL
signals. Research-only.
"""
import warnings
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1069)
OUT = "reform_results/sol_hmms_20260709"

cg = pd.read_parquet(f"{OUT}/cg_flow_sol_1h.parquet").sort_index()
print(f"SOL CG flow: {len(cg)} bars {cg.index.min()} -> {cg.index.max()}")

fb, fs = cg["fut_buy_usd"], cg["fut_sell_usd"]
sb, ss = cg["spot_buy_usd"], cg["spot_sell_usd"]
oi = cg["oi_close"]
ll, ls = cg["liq_long_usd"], cg["liq_short_usd"]
feat = pd.DataFrame(index=cg.index)
feat["fut_ratio_1h"] = fb / (fb + fs).replace(0, np.nan)
feat["fut_cvd_12h"] = (fb - fs).rolling(12).sum() / (fb + fs).rolling(12).sum().replace(0, np.nan)
feat["spot_ratio_1h"] = sb / (sb + ss).replace(0, np.nan)
feat["oi_chg_4h"] = oi.pct_change(4, fill_method=None)
feat["liq_imb_4h"] = (ls.rolling(4).sum() - ll.rolling(4).sum()) / (ls.rolling(4).sum() + ll.rolling(4).sum() + 1.0)
lt = ll + ls
feat["liq_tot_z_10d"] = (lt - lt.rolling(240).mean()) / lt.rolling(240).std().replace(0, np.nan)
FEAT_COLS = list(feat.columns)
feat = feat.dropna()
print(f"feature matrix: {len(feat)} x 6")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(feat.values)
g = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(g > 7200)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 5]
vidx = [i for s, e in zip(starts, ends) if e - s >= 5 for i in range(s, e)]
Xs = X[vidx]

print("\nBIC selection:")
best = (np.inf, None, None)
for n in range(2, 9):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=300, random_state=42, tol=1e-4)
        m.fit(Xs, lengths=lengths)
        ll_ = m.score(Xs, lengths=lengths)
        bic = -2 * ll_ + (n * n + n * 12) * np.log(len(Xs))
        print(f"  n={n}: BIC={bic:.1f}")
        if bic < best[0]:
            best = (bic, n, m)
    except Exception as e:
        print(f"  n={n} failed: {e}")
_, N, model = best
print(f"selected {N} states")

with open(f"{OUT}/hmm_cg_flow_sol_1h.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": N}, f)

# causal per-point decode (trailing 64)
fi = feat.iloc[vidx]
states = np.full(len(Xs), -1)
for i in range(len(Xs)):
    lo = max(0, i - 63)
    try:
        states[i] = int(model.predict(Xs[lo:i + 1])[-1])
    except Exception:
        pass
sv = pd.DataFrame({"bar_open": fi.index, "state": states})
sv = sv[sv["state"] >= 0]
sv["effective"] = sv["bar_open"] + pd.Timedelta("1h")   # ZERO-LOOKAHEAD join key
sv = sv.sort_values("effective")
print(f"\nstates: {len(sv)}  distribution: {sv['state'].value_counts().sort_index().to_dict()}")
print("centroids:")
for s in sorted(sv["state"].unique()):
    sub = fi[states == s]
    print(f"  S{s} (n={len(sub)}): " + "  ".join(f"{c}={sub[c].mean():+.3f}" for c in FEAT_COLS))
sv.to_csv(f"{OUT}/cg_flow_sol_states.csv", index=False)

# ── validation on both SOL books ──────────────────────────────────────────
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


for path, name, is15 in [("results/paper_trades_sol15m.csv", "SOL 15m", True),
                         ("results/paper_trades_sol.csv", "SOL HOURLY", False)]:
    t = book(path, is15).dropna(subset=["state"])
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

# ── orthogonality: RF from logged SOL 15m signals -> state ────────────────
print("\n=== orthogonality check (RF from logged SOL 15m signals) ===")
pt = pd.read_csv("results/paper_trades_sol15m.csv", low_memory=False)
pt["logged_at_p"] = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
PRED = ["vol_ratio", "chg_1h", "stoch_k_1h", "vwap_dist", "ls_long_pct", "liq_score",
        "oi_chg_pct", "cvd_4h", "cg_futures_ratio_4h", "z_drift_6h", "kalman_velocity",
        "hurst_exponent", "regime_z"]
for c in PRED:
    pt[c] = pd.to_numeric(pt[c], errors="coerce")
red = pt.dropna(subset=["logged_at_p"]).sort_values("logged_at_p")
red = pd.merge_asof(red, sv[["effective", "state"]], left_on="logged_at_p",
                    right_on="effective", direction="backward", tolerance=pd.Timedelta("2h"))
red = red.dropna(subset=["state"] + PRED)
red["hr"] = red["logged_at_p"].dt.floor("h")
red = red.drop_duplicates(subset="hr", keep="last")
print(f"unique hours with all predictors: {len(red)}")
if len(red) >= 200:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    Xr = red[PRED].values
    yr = red["state"].astype(int).values
    base = np.bincount(yr).max() / len(yr)
    accs = []
    for tr, te in TimeSeriesSplit(n_splits=4).split(Xr):
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=4)
        clf.fit(Xr[tr], yr[tr])
        accs.append((clf.predict(Xr[te]) == yr[te]).mean())
    print(f"majority baseline: {base:.3f}   RF OOS accuracy: {np.mean(accs):.3f}")
    print("(near baseline = orthogonal / genuinely new information)")
print("DONE_S3")
