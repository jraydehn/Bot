"""
S8 -- Build an HMM around p_up_v3 to capture regime/persistence structure
in the signal itself, motivated by the Markov-chain test (s7) showing:
  (a) strong, year-stable correlation between p_up_v3 level and realized
      direction (chi2 p=2.5e-96), and
  (b) real state persistence -- p_up_v3 doesn't bounce independently,
      it has momentum (diagonal transition probs ~38-40% vs ~20% if iid).

Observation features (standardized): raw level, 1h change (is conviction
building or fading), and a 6h rolling mean (smoothed short-term trend of
the signal itself) -- deliberately goes beyond the quintile-binning test
by giving the HMM temporal structure a static bin can't see.

Trained via .fit(X, lengths=[len(X)]) on the full honest OOS sequence,
decoded via .predict()/.predict_proba() over the WHOLE sequence at once
(proper multi-observation Viterbi/forward-backward) -- NOT the single-
observation decode that caused the degenerate ms/vd/of HMM bug fixed
2026-07-06. This is offline batch analysis; if this is ever wired into
the live runner, it needs the same trailing-sequence-decode treatment
those three functions got.
"""
import pickle
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# hmmlearn's kmeans++ init emits benign divide-by-zero/overflow RuntimeWarnings
# on this data (verified: final means_/covars_/transmat_ are all finite, model
# converges normally, log-likelihood is sane -- a transient artifact of an
# early candidate-seeding step, not a real numerical failure).
warnings.filterwarnings("ignore", category=RuntimeWarning)

REBUILD = "reform_results/pup_v2_rebuild_20260704"
OUT = "reform_results/pup_v3_15m_window_sweep_20260706"
N_STATES = 4

ev = pd.read_parquet(f"{REBUILD}/wf_preds_FINAL.parquet").dropna().sort_index()
print(f"n={len(ev)}  ({ev.index.min()} -> {ev.index.max()})")

df = pd.DataFrame(index=ev.index)
df["p"] = ev["p"]
df["p_chg_1h"] = ev["p"].diff()
df["p_ma6h"] = ev["p"].rolling(6, min_periods=6).mean()
df["label"] = ev["label"]
df = df.dropna()
print(f"after feature warmup: n={len(df)}")

FEATS = ["p", "p_chg_1h", "p_ma6h"]
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATS].values)

np.random.seed(42)
model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                    n_iter=200, random_state=42)
model.fit(X, lengths=[len(X)])

states = model.predict(X)
probs = model.predict_proba(X)
df["state"] = states

print(f"\nconverged: {model.monitor_.converged}  iters: {model.monitor_.iter}")
print(f"startprob_: {model.startprob_.round(4)}")
print("transmat_ diag (self-persistence):", np.diag(model.transmat_).round(4))

print("\n=== Per-state summary ===")
summ = df.groupby("state").agg(n=("label", "size"), p_up=("label", "mean"),
                               p_mean=("p", "mean"), p_chg_mean=("p_chg_1h", "mean"))
summ["se"] = np.sqrt(summ["p_up"] * (1 - summ["p_up"]) / summ["n"])
summ["ci95_lo"] = summ["p_up"] - 1.96 * summ["se"]
summ["ci95_hi"] = summ["p_up"] + 1.96 * summ["se"]
print(summ.round(4).to_string())

print("\n=== Per-state P(up), by year (robustness check) ===")
df["year"] = df.index.year
for st in sorted(df["state"].unique()):
    row = f"state {st}: "
    for yr, g in df.groupby("year"):
        sub = g[g["state"] == st]
        if len(sub) < 20:
            row += f"{yr}=thin(n={len(sub)}) "
        else:
            row += f"{yr}={sub['label'].mean():.3f}(n={len(sub)}) "
    print(row)

# save artifact for downstream use
with open(f"{OUT}/hmm_pup_v3_regime.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": FEATS,
                "n_states": N_STATES}, f)
df.to_parquet(f"{OUT}/pup_v3_hmm_states.parquet")
print(f"\nsaved {OUT}/hmm_pup_v3_regime.pkl and pup_v3_hmm_states.parquet")
