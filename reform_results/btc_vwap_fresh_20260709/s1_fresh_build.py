"""
S1 -- FRESH BTC 15m VWAP MTF HMM, zero-lookahead methodology end-to-end.
The existing live model/gates are NOT touched -- this is a parallel research
build to see what an honest pipeline finds in the same signal space.

Fresh-method conventions (vs the original 07-01 build):
- Features on the 15m grid exactly as before (5 dims: vwap_dist 1m/5m/15m,
  vel_1m, spread), but every downstream step is causal:
- CAUSAL DECODE: trailing-64 predict per point (no full-sequence smoothing).
- EFFECTIVE-TIME JOIN: state at bar-open T is knowable at T+15m only.
- Validation: ticker-clustered bootstrap + weekly stability + era split,
  on BOTH the scan-archive resolved population (model-side convention) and
  the real taken book.
- Cross-map vs the existing live model's causal decode on the same grid,
  so old-vs-new state correspondence is explicit.
NOTE: a deployment of THIS model would need a completed-bars-only live
decoder (live_1m.iloc[:-1] before resampling) -- unlike the existing live
decoder, which uses the partial current bar. Different signal by design.
"""
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1123)
OUT = "reform_results/btc_vwap_fresh_20260709"

p1m = sorted(pathlib.Path("data").glob("binanceus_BTCUSDT_1m_1970-01-01_*.parquet"))[-1]
df1m = pd.read_parquet(p1m).sort_index()
df1m = df1m[df1m.index >= "2024-01-01"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
df5 = df1m.resample("5min").agg(AGG).dropna()
df15 = df1m.resample("15min").agg(AGG).dropna()
print(f"1m source ends {df1m.index.max()};  15m bars: {len(df15)}")


def rvwap(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ctv = (tp * df["volume"]).rolling(n, min_periods=n).sum()
    cv = df["volume"].rolling(n, min_periods=n).sum()
    vw = ctv / cv.replace(0, np.nan)
    return (df["close"] - vw) / vw.replace(0, np.nan) * 100


d1 = rvwap(df1m); d5v = rvwap(df5); d15v = rvwap(df15)
feat = pd.DataFrame(index=df15.index)
feat["vwap_dist_15m"] = d15v
feat["vwap_dist_5m"] = d5v.resample("15min").last()
feat["vwap_dist_1m"] = d1.resample("15min").last()
feat["vwap_vel_1m"] = d1.diff().resample("15min").last()
feat["vwap_spread"] = feat["vwap_dist_1m"] - feat["vwap_dist_15m"]
feat = feat.dropna()
FEAT_COLS = list(feat.columns)
print(f"feature matrix: {len(feat)} x 5")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(feat.values)
g = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(g > 1800)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 10]
vidx = [i for s, e in zip(starts, ends) if e - s >= 10 for i in range(s, e)]
Xs = X[vidx]

print("\nBIC selection (4-10):")
best = (np.inf, None, None)
for n in range(4, 11):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=200, random_state=42, tol=1e-4)
        m.fit(Xs, lengths=lengths)
        ll = m.score(Xs, lengths=lengths)
        bic = -2 * ll + (n * n + n * 10) * np.log(len(Xs))
        print(f"  n={n}: BIC={bic:.1f}")
        if bic < best[0]:
            best = (bic, n, m)
    except Exception as e:
        print(f"  n={n} failed: {e}")
_, N, model = best
print(f"selected {N} states")
with open(f"{OUT}/hmm_vwap_fresh_btc_15m.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": N,
                 "convention": "completed-bars-only; causal trailing-64 decode; effective=open+15m"}, f)

# ── causal decode over the validation span (+ the old live model on the same grid) ──
fi = feat.iloc[vidx]
recent = fi.index >= pd.Timestamp("2026-05-20", tz="UTC")
ridx = np.where(recent)[0]
print(f"\ncausal decode over {len(ridx)} recent bars (new + old models)...")
old = pickle.load(open("models/hmm_vwap_mtf_btc_15m.pkl", "rb"))
Xo = old["scaler"].transform(fi[old["feat_cols"]].values)
new_states = np.full(len(ridx), -1)
old_states = np.full(len(ridx), -1)
for j, i in enumerate(ridx):
    lo = max(0, i - 63)
    try:
        new_states[j] = int(model.predict(Xs[lo:i + 1])[-1])
        old_states[j] = int(old["model"].predict(Xo[lo:i + 1])[-1])
    except Exception:
        pass
sv = pd.DataFrame({"bar_open": fi.index[ridx], "new": new_states, "old": old_states})
sv = sv[(sv["new"] >= 0) & (sv["old"] >= 0)]
sv["effective"] = sv["bar_open"] + pd.Timedelta("15min")
sv = sv.sort_values("effective")
sv.to_csv(f"{OUT}/fresh_states.csv", index=False)
print("new-state distribution:", sv["new"].value_counts().sort_index().to_dict())
print("\nnew-state centroids:")
for s in sorted(sv["new"].unique()):
    sub = fi.iloc[[ridx[j] for j in range(len(new_states)) if new_states[j] == s]]
    print(f"  N{s} (n={len(sub)}): " + "  ".join(f"{c}={sub[c].mean():+.3f}" for c in FEAT_COLS))
print("\nold->new state cross-map (row=old causal, col=new causal, %):")
ct = pd.crosstab(sv["old"], sv["new"], normalize="index").round(2)
print(ct.to_string())

# ── validation on scan-archive resolved population + taken book ────────────
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


def tk_boot(d, n_boot=4000):
    pt = d.groupby("contract_ticker")["tedge"].mean()
    e = pt.values
    if len(e) < 10:
        return len(e), np.nan, np.nan
    means = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)])
    return len(e), means.mean(), (means <= 0).mean()


arch = pd.read_csv("results/btc_scan_archive_15m.csv", low_memory=False)
arch["ts"] = parse_mixed(arch["logged_at"])
for c in ["resolved_yes", "p_market", "p_model_yes", "p_model_no"]:
    arch[c] = pd.to_numeric(arch[c], errors="coerce")
res = arch.dropna(subset=["ts", "resolved_yes", "p_market", "p_model_yes", "p_model_no"]).copy()
res["edge_yes"] = res["p_model_yes"] - res["p_market"]
res["edge_no"] = res["p_model_no"] - (1 - res["p_market"])
res["model_side"] = np.where(res["edge_yes"] >= res["edge_no"], "YES", "NO")
res["won"] = np.where(res["model_side"] == "YES", res["resolved_yes"] == 1, res["resolved_yes"] == 0)
res["be"] = np.where(res["model_side"] == "YES", res["p_market"], 1 - res["p_market"])
res["tedge"] = res["won"].astype(float) - res["be"]
res = res.sort_values("ts")
res = pd.merge_asof(res, sv[["effective", "new"]], left_on="ts", right_on="effective",
                    direction="backward", tolerance=pd.Timedelta("45min")).dropna(subset=["new"])
res["week"] = res["ts"].dt.to_period("W-FRI").astype(str)
print(f"\n=== scan-archive resolved population with fresh state: {len(res)} rows, "
      f"{res['contract_ticker'].nunique()} tickers ===")
print(f"{'state':<5}{'side':<5}{'rows':>6}{'tickers':>8}{'WR':>7}{'BE':>7}{'tk_edge':>9}{'P(<=0)':>8}{'wk+':>6}")
for s in sorted(res["new"].unique()):
    for side in ["YES", "NO"]:
        d = res[(res["new"] == s) & (res["model_side"] == side)]
        if len(d) < 60:
            continue
        nt, ee, pn = tk_boot(d)
        wk = d.groupby("week")["tedge"].mean()
        print(f"N{int(s):<4}{side:<5}{len(d):>6}{nt:>8}{d['won'].mean():>7.3f}{d['be'].mean():>7.3f}"
              f"{ee:>+9.4f}{pn:>8.4f}{int((wk>0).sum()):>4}/{len(wk)}")

pt2 = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
pt2["ts"] = pd.to_datetime(pt2["logged_at"], format="mixed", utc=True, errors="coerce")
for c in ["resolved_yes", "p_market", "would_pnl"]:
    pt2[c] = pd.to_numeric(pt2[c], errors="coerce")
tk = pt2[pt2["decision"] == "trade"].dropna(subset=["ts", "resolved_yes", "p_market"]).copy()
tk["side"] = tk["side"].str.lower()
tk["won"] = np.where(tk["side"] == "yes", tk["resolved_yes"] == 1, tk["resolved_yes"] == 0)
tk["be"] = np.where(tk["side"] == "yes", tk["p_market"], 1 - tk["p_market"])
tk["tedge"] = tk["won"].astype(float) - tk["be"]
tk = tk.sort_values("ts")
tk = pd.merge_asof(tk, sv[["effective", "new"]], left_on="ts", right_on="effective",
                   direction="backward", tolerance=pd.Timedelta("45min")).dropna(subset=["new"])
tk["week"] = tk["ts"].dt.to_period("W-FRI").astype(str)
print(f"\n=== TAKEN book with fresh state: {len(tk)} trades ===")
for s in sorted(tk["new"].unique()):
    for side in ["yes", "no"]:
        d = tk[(tk["new"] == s) & (tk["side"] == side)]
        if len(d) < 20:
            continue
        nt, ee, pn = tk_boot(d)
        wk = d.groupby("week")["tedge"].mean()
        print(f"N{int(s)} {side.upper():3s}: n={len(d)} tickers={nt} edge={d['tedge'].mean():+.4f} "
              f"tk_edge={ee:+.4f} P(<=0)={pn:.4f} wk+={int((wk>0).sum())}/{len(wk)} "
              f"$={d['would_pnl'].sum():+.2f}")
print("DONE_S1")
