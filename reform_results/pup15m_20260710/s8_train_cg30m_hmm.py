"""
S8 -- train the 30m CoinGlass flow HMM for the BTC 15m model.
Mirrors the validated hourly recipe (cg_hmm_20260708/s1): same 6 feature
concepts scaled to 30m bars, GaussianHMM diag covariance, BIC selection,
gap-aware sequences. Zero-lookahead decode: state at bar open t is
EFFECTIVE at t+30m (bar completion).

Features (30m cadence):
  fut_ratio_30m  = fut_buy/(fut_buy+fut_sell)          -- taker aggression
  fut_cvd_6h     = 12-bar rolling CVD ratio            -- sustained flow
  spot_ratio_30m = spot buy ratio                      -- spot vs perp split
  oi_chg_2h      = 4-bar OI pct change                 -- positioning build/unwind
  liq_imb_2h     = (short_liq - long_liq)/(sum+1) 4-bar-- squeeze direction
  liq_tot_z_10d  = total liq z vs 480-bar (10d) window -- cascade intensity
"""
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

OUT = "reform_results/pup15m_20260710"
cg = pd.read_parquet(f"{OUT}/cg_flow_btc_30m.parquet").sort_index()
cg = cg.apply(pd.to_numeric, errors="coerce")
fb, fs = cg["fut_buy_usd"], cg["fut_sell_usd"]
sb, ss = cg["spot_buy_usd"], cg["spot_sell_usd"]
oi = cg["oi_close"]
ll, ls = cg["liq_long_usd"], cg["liq_short_usd"]
feat = pd.DataFrame(index=cg.index)
feat["fut_ratio_30m"] = fb / (fb + fs).replace(0, np.nan)
feat["fut_cvd_6h"]    = (fb - fs).rolling(12).sum() / (fb + fs).rolling(12).sum().replace(0, np.nan)
feat["spot_ratio_30m"] = sb / (sb + ss).replace(0, np.nan)
feat["oi_chg_2h"]     = oi.pct_change(4, fill_method=None)
feat["liq_imb_2h"]    = (ls.rolling(4).sum() - ll.rolling(4).sum()) / (ls.rolling(4).sum() + ll.rolling(4).sum() + 1.0)
lt = ll + ls
feat["liq_tot_z_10d"] = (lt - lt.rolling(480).mean()) / lt.rolling(480).std().replace(0, np.nan)
feat = feat.dropna()
FEAT_COLS = list(feat.columns)
print(f"features: {len(feat)} bars  {feat.index.min()} -> {feat.index.max()}")

scaler = StandardScaler().fit(feat.values)
X = scaler.transform(feat.values)
gaps = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(gaps > 3600)[0])
ends = starts[1:] + [len(X)]
seqs = [(s, e) for s, e in zip(starts, ends) if e - s >= 5]
lengths = [e - s for s, e in seqs]
X_seq = np.vstack([X[s:e] for s, e in seqs])
print(f"sequences: {len(seqs)}  total bars: {len(X_seq)}")

print("\nBIC selection:")
best = (np.inf, None, None)
for n in range(4, 11):
    m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=300,
                    random_state=42, tol=1e-4)
    m.fit(X_seq, lengths)
    ll_score = m.score(X_seq, lengths)
    n_params = n * n + 2 * n * X.shape[1] - 1
    bic = -2 * ll_score + n_params * np.log(len(X_seq))
    print(f"  n={n}: BIC={bic:>11.1f}")
    if bic < best[0]:
        best = (bic, n, m)
bic, n_states, model = best
print(f"\nselected {n_states} states (BIC={bic:.1f})")

# decode full series (sequence-aware), zero-lookahead effective time
states = np.full(len(X), -1)
for s, e in seqs:
    states[s:e] = model.predict(X[s:e])
st = pd.DataFrame({"bar_open": feat.index, "cg30_state": states})
st = st[st["cg30_state"] >= 0]
st["effective"] = st["bar_open"] + pd.Timedelta("30min")
st.to_csv(f"{OUT}/cg30m_states.csv", index=False)

print("\nstate centroids (unscaled) + occupancy + persistence:")
occ = st["cg30_state"].value_counts(normalize=True).sort_index()
dwell = st["cg30_state"].groupby((st["cg30_state"] != st["cg30_state"].shift()).cumsum()).size()
dw = st.assign(run=(st["cg30_state"] != st["cg30_state"].shift()).cumsum()).groupby("run").agg(
    s=("cg30_state", "first"), n=("cg30_state", "size")).groupby("s")["n"].mean()
cent = pd.DataFrame(scaler.inverse_transform(model.means_), columns=FEAT_COLS).round(4)
cent["occ%"] = (occ * 100).round(1).values
cent["avg_dwell_bars"] = dw.round(1).values
print(cent.to_string())

# split-half occupancy stability (state definitions shouldn't be era artifacts)
half = len(st) // 2
o1 = st.iloc[:half]["cg30_state"].value_counts(normalize=True).sort_index()
o2 = st.iloc[half:]["cg30_state"].value_counts(normalize=True).sort_index()
print("\nocc first-half vs second-half:")
print(pd.DataFrame({"H1": (o1*100).round(1), "H2": (o2*100).round(1)}).fillna(0).to_string())

with open(f"{OUT}/hmm_cg_flow_btc_30m.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS,
                 "convention": "bar OPEN indexed; effective = open+30m; features need 480-bar warmup",
                 "trained_on": f"{feat.index.min()} .. {feat.index.max()}"}, f)
print("DONE_S8")
