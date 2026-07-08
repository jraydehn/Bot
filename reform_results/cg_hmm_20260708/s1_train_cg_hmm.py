"""
S1 -- Train a CoinGlass flow-regime HMM for BTC hourly. Emissions reuse the
EXACT Fcg feature formulas from reform_results/pup_v2_rebuild_20260704/
s3_build_dataset.py (not re-derived). 6 dims chosen for stationarity
(ratios/z-scores only, no raw USD) and to lean toward what the existing
of_hmm (Coinalyze: ls_long_pct, oi_chg, liq_bias, vpin, funding, obi) does
NOT already consume -- taker flow ratios and CVD are uniquely CoinGlass:
  fut_ratio_1h   -- futures taker buy share of total (buy/(buy+sell))
  fut_cvd_12h    -- rolling 12h normalized futures CVD
  spot_ratio_1h  -- spot taker buy share
  oi_chg_4h      -- 4h open-interest change
  liq_imb_4h     -- 4h liquidation imbalance (short-long)/(total)
  liq_tot_z_10d  -- 10-day z-score of total liquidation USD (cascade intensity)
funding deliberately EXCLUDED (of_hmm already has funding_bias).

NOTE: the same Fcg features were tested as DIRECTIONAL model inputs in the
pup_v3 rebuild and REJECTED (d_auc=-0.008, p=0.87). This build asks a
different question: do they define latent flow regimes where the live
model's edge/reliability differs? Direction-rejected != regime-useless
(cal_err precedent), but that history is why s2 (redundancy) and s3
(taken-trade conditioning with split-half) must pass before any wiring.
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("pip install hmmlearn scikit-learn")

OUT = "reform_results/cg_hmm_20260708"
cg = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/cg_flow_btc_1h.parquet").sort_index()
print(f"raw CG data: {cg.shape}  {cg.index.min()} -> {cg.index.max()}")

fb, fs = cg["fut_buy_usd"], cg["fut_sell_usd"]
sb, ss = cg["spot_buy_usd"], cg["spot_sell_usd"]
oi = cg["oi_close"]
ll, ls = cg["liq_long_usd"], cg["liq_short_usd"]

feat = pd.DataFrame(index=cg.index)
feat["fut_ratio_1h"]  = fb / (fb + fs).replace(0, np.nan)
feat["fut_cvd_12h"]   = (fb - fs).rolling(12).sum() / (fb + fs).rolling(12).sum().replace(0, np.nan)
feat["spot_ratio_1h"] = sb / (sb + ss).replace(0, np.nan)
feat["oi_chg_4h"]     = oi.pct_change(4, fill_method=None)
feat["liq_imb_4h"]    = (ls.rolling(4).sum() - ll.rolling(4).sum()) / (ls.rolling(4).sum() + ll.rolling(4).sum() + 1.0)
lt = ll + ls
feat["liq_tot_z_10d"] = (lt - lt.rolling(240).mean()) / lt.rolling(240).std().replace(0, np.nan)
FEAT_COLS = list(feat.columns)
feat = feat.dropna()
print(f"feature matrix: {feat.shape}  {feat.index.min()} -> {feat.index.max()}")

scaler = StandardScaler()
X = scaler.fit_transform(feat[FEAT_COLS].values)

# continuous sequences (gap > 2h -> new sequence)
gaps = pd.Series(feat.index).diff().dt.total_seconds().fillna(0).values
starts = [0] + list(np.where(gaps > 7200)[0])
ends = starts[1:] + [len(X)]
lengths = [e - s for s, e in zip(starts, ends) if e - s >= 5]
valid_idx = [i for s, e in zip(starts, ends) if e - s >= 5 for i in range(s, e)]
X_seq = X[valid_idx]
print(f"{len(lengths)} sequences, {len(X_seq)} observations")

print("\nBIC selection:")
best = (np.inf, None, None)
for n in range(2, 8):
    try:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=300,
                        random_state=42, tol=1e-4)
        m.fit(X_seq, lengths=lengths)
        ll_score = m.score(X_seq, lengths=lengths)
        n_params = n * n + n * len(FEAT_COLS) * 2
        bic = -2 * ll_score + n_params * np.log(len(X_seq))
        print(f"  n={n}: BIC={bic:>11.1f}")
        if bic < best[0]:
            best = (bic, n, m)
    except Exception as e:
        print(f"  n={n}: FAILED {e}")
bic, n_states, model = best
print(f"\nselected {n_states} states (BIC={bic:.1f})")

states = model.predict(X_seq, lengths=lengths)
fv = feat.iloc[valid_idx].copy()
fv["cg_state"] = states

print("\n=== state profiles ===")
for s in range(n_states):
    sub = fv[fv["cg_state"] == s]
    print(f"\nState {s}  n={len(sub):,} ({100*len(sub)/len(fv):.1f}%)  "
          f"self-trans={model.transmat_[s,s]:.3f}")
    for c in FEAT_COLS:
        print(f"  {c:<15}: {sub[c].mean():+.4f}")

pkg = {"model": model, "scaler": scaler, "feat_cols": FEAT_COLS, "n_states": n_states}
with open(f"{OUT}/hmm_cg_flow_btc_1h.pkl", "wb") as f:
    pickle.dump(pkg, f)
fv[["cg_state"]].to_csv(f"{OUT}/cg_states_1h.csv")
print(f"\nsaved model + state series")
