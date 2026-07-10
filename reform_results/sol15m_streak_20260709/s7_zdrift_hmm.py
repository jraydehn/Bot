"""
S7 -- Explore a z_drift HMM for the COMPLEMENT side (kept NO book, z>=0.55):
can a state-conditioned view justify a Kelly conviction boost the binary
threshold can't express?

Design:
- Regular 15-min causal grid of z_drift_6h over the archive span, each point =
  mean of settlement z-events with decision in [t-6h, t) AND resolved by t
  (same causality guard as s6).
- Emissions: [level, delta_1h, ma_6h] (mirrors the p_up_v3 regime HMM design).
- BIC-select 2-5 states; CAUSAL decode: trailing-64-point predict per grid
  point, take last state (forward-backward smoothing on the full sequence
  would leak future observations into historical states).
- Evaluate per-state edge on the kept NO book (episode-clustered), plus the
  blocked book and YES book for completeness.
- NULL COMPETITOR (HMM#14 redundancy discipline): plain z-level terciles
  within the kept book. The HMM must beat simple thresholds on the same
  variable to justify existence.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1051)
OUT = "reform_results/sol15m_streak_20260709"

# ── settlement z-events (reuse s6 machinery) ──────────────────────────────
def parse_mixed(s):
    def _u(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.NaT
    return pd.to_datetime([_u(v) for v in s], utc=True)

sa = pd.read_csv("results/sol_scan_archive_15m.csv", low_memory=False,
                 usecols=["logged_at", "close_ts", "spot", "realized_vol_annual",
                          "tau_minutes", "spot_at_expiry"])
for c in ["spot", "realized_vol_annual", "tau_minutes", "spot_at_expiry"]:
    sa[c] = pd.to_numeric(sa[c], errors="coerce")
sa["dts"] = parse_mixed(sa["logged_at"])
sa["cts"] = parse_mixed(sa["close_ts"])
sa = sa.dropna(subset=["dts", "cts", "spot", "realized_vol_annual", "tau_minutes", "spot_at_expiry"])
sa = sa[(sa["spot"] > 0) & (sa["realized_vol_annual"] > 0) & (sa["tau_minutes"] > 0) & (sa["spot_at_expiry"] > 0)]
sigma = sa["realized_vol_annual"] * np.sqrt(sa["tau_minutes"] / 525600.0)
sa["z"] = np.log(sa["spot_at_expiry"] / sa["spot"]) / sigma.replace(0, np.nan)
sa = sa.dropna(subset=["z"]).sort_values("dts").reset_index(drop=True)
D = sa["dts"].values.astype("datetime64[ns]")
C = sa["cts"].values.astype("datetime64[ns]")
Z = sa["z"].values
print(f"z-events: {len(sa)}")

# ── regular causal grid ───────────────────────────────────────────────────
grid = pd.date_range(sa["dts"].min().ceil("h") + pd.Timedelta("6h"),
                     sa["dts"].max(), freq="15min", tz="UTC")
def zdrift_at(T):
    Tn = np.datetime64(T.tz_convert("UTC").tz_localize(None))
    lo = np.searchsorted(D, Tn - np.timedelta64(6, "h"), side="left")
    hi = np.searchsorted(D, Tn, side="left")
    if hi <= lo:
        return np.nan
    ok = C[lo:hi] <= Tn
    if ok.sum() < 3:
        return np.nan
    return float(Z[lo:hi][ok].mean())

gz = pd.Series([zdrift_at(t) for t in grid], index=grid, name="z")
gz = gz.dropna()
print(f"grid: {len(gz)} 15-min points, {gz.index.min()} -> {gz.index.max()}")

feat = pd.DataFrame({"level": gz})
feat["delta_1h"] = gz.diff(4)
feat["ma_6h"] = gz.rolling(24, min_periods=24).mean()
feat = feat.dropna()
print(f"HMM training matrix: {len(feat)} x 3")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(feat.values)
# contiguous sequences (grid gaps -> new sequence)
g = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(g > 1800)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 10]
vidx = [i for s, e in zip(starts, ends) if e - s >= 10 for i in range(s, e)]
Xs = X[vidx]

print("\nBIC selection:")
best = (np.inf, None, None)
for n in range(2, 6):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=300, random_state=42, tol=1e-4)
        m.fit(Xs, lengths=lengths)
        ll = m.score(Xs, lengths=lengths)
        n_par = n * n + n * 3 * 2
        bic = -2 * ll + n_par * np.log(len(Xs))
        print(f"  n={n}: BIC={bic:.1f}")
        if bic < best[0]:
            best = (bic, n, m)
    except Exception as e:
        print(f"  n={n} failed: {e}")
_, N, model = best
print(f"selected {N} states")

# ── CAUSAL decode: trailing-64 predict per point ──────────────────────────
fi = feat.iloc[vidx].copy()
Xv = Xs
states = np.full(len(Xv), -1)
for i in range(len(Xv)):
    lo = max(0, i - 63)
    try:
        states[i] = int(model.predict(Xv[lo:i + 1])[-1])
    except Exception:
        pass
fi["state"] = states
fi = fi[fi["state"] >= 0]
print("\ncausal state distribution:", fi["state"].value_counts().sort_index().to_dict())
print("state centroids (level / delta_1h / ma_6h):")
for s in sorted(fi["state"].unique()):
    sub = fi[fi["state"] == s]
    print(f"  S{s}: level={sub['level'].mean():+.3f}  delta={sub['delta_1h'].mean():+.4f}  "
          f"ma={sub['ma_6h'].mean():+.3f}  (n={len(sub)})")

sv = fi[["state"]].reset_index().rename(columns={"index": "ts"}).sort_values("ts")

# ── join to the NO book ───────────────────────────────────────────────────
no = pd.read_csv(f"{OUT}/no_book_reconstructed.csv", low_memory=False)
no["logged_at_p"] = pd.to_datetime(no["logged_at_p"], utc=True)
no["week"] = no["logged_at_p"].dt.to_period("W-FRI").astype(str)
no = pd.merge_asof(no.sort_values("logged_at_p"), sv, left_on="logged_at_p",
                   right_on="ts", direction="backward", tolerance=pd.Timedelta("45min"))
print(f"\nNO book with state: {no['state'].notna().sum()}/{len(no)}")

kept = no[(no["z_drift_6h"] >= 0.55).fillna(False) & no["state"].notna()]
blocked = no[(no["z_drift_6h"] < 0.55).fillna(False) & no["state"].notna()]

def ep_stats(d, n_boot=4000):
    eps = d.groupby("episode")["tedge"].mean().values
    if len(eps) < 8:
        return len(eps), np.nan, np.nan
    means = np.array([eps[rng.integers(0, len(eps), len(eps))].mean() for _ in range(n_boot)])
    return len(eps), means.mean(), (means <= 0).mean()

print(f"\n=== KEPT book (z>=0.55) by HMM state — the conviction-boost question ===")
print(f"kept baseline: n={len(kept)} edge={kept['tedge'].mean():+.4f} $={kept['would_pnl'].sum():+.2f}")
for s in sorted(kept["state"].dropna().unique()):
    d = kept[kept["state"] == s]
    if len(d) < 20:
        print(f"  S{int(s)}: n={len(d)} thin")
        continue
    ne, ee, pn = ep_stats(d)
    wk = d.groupby("week")["tedge"].mean()
    print(f"  S{int(s)}: n={len(d)} eps={ne} edge={d['tedge'].mean():+.4f} ep_edge={ee:+.4f} "
          f"P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} $={d['would_pnl'].sum():+.2f}")

print(f"\n=== NULL COMPETITOR: plain z-level terciles within kept ===")
terc = kept["z_drift_6h"].quantile([1/3, 2/3]).values
labels = [f"z<{terc[0]:.2f}", f"z {terc[0]:.2f}-{terc[1]:.2f}", f"z>={terc[1]:.2f}"]
bins = [kept["z_drift_6h"] < terc[0],
        (kept["z_drift_6h"] >= terc[0]) & (kept["z_drift_6h"] < terc[1]),
        kept["z_drift_6h"] >= terc[1]]
for lab, mk in zip(labels, bins):
    d = kept[mk]
    ne, ee, pn = ep_stats(d)
    wk = d.groupby("week")["tedge"].mean()
    print(f"  {lab}: n={len(d)} eps={ne} ep_edge={ee:+.4f} P(<=0)={pn:.4f} "
          f"wk+={int((wk>0).sum())}/{len(wk)} $={d['would_pnl'].sum():+.2f}")

print(f"\n=== blocked book by state (context) ===")
for s in sorted(blocked["state"].dropna().unique()):
    d = blocked[blocked["state"] == s]
    if len(d) < 20:
        continue
    ne, ee, pn = ep_stats(d)
    print(f"  S{int(s)}: n={len(d)} ep_edge={ee:+.4f} P(<=0)={pn:.4f} $={d['would_pnl'].sum():+.2f}")

no.to_csv(f"{OUT}/no_book_zhmm.csv", index=False)
print("DONE_S7")
