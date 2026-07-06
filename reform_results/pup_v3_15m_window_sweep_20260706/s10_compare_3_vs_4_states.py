"""
S10 -- Does a 3-state HMM do as well or better than 4? States 0 and 2 in
the 4-state fit have near-identical momentum (p_chg_1h=0.0021 for BOTH)
and only differ by which side of 0.50 their raw level sits on -- exactly
the kind of split a Markov-chain-on-raw-level test would already show,
not new temporal information. Compare via BIC (penalizes extra
parameters) and via the same real-trade backfill used for the 4-state
model.
"""
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
REBUILD = "reform_results/pup_v2_rebuild_20260704"
OUT = "reform_results/pup_v3_15m_window_sweep_20260706"

ev = pd.read_parquet(f"{REBUILD}/wf_preds_FINAL.parquet").dropna().sort_index()
df = pd.DataFrame(index=ev.index)
df["p"] = ev["p"]
df["p_chg_1h"] = ev["p"].diff()
df["p_ma6h"] = ev["p"].rolling(6, min_periods=6).mean()
df["label"] = ev["label"]
df = df.dropna()
FEATS = ["p", "p_chg_1h", "p_ma6h"]
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATS].values)
n_params_per_state = len(FEATS) * 2  # diag covariance: mean + var per feature

results = {}
for n_states in [3, 4]:
    m = GaussianHMM(n_components=n_states, covariance_type="diag",
                    n_iter=200, random_state=42)
    m.fit(X, lengths=[len(X)])
    logL = m.score(X) * len(X)  # score() returns per-sample avg log-likelihood... actually returns total; check below
    logL = m.score(X)
    n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_params_per_state
    bic = -2 * logL + n_params * np.log(len(X))
    states = m.predict(X)
    df_s = df.copy()
    df_s["state"] = states
    print(f"\n=== {n_states}-state model === logL={logL:.1f}  n_params={n_params}  BIC={bic:.1f}")
    summ = df_s.groupby("state").agg(n=("label", "size"), p_up=("label", "mean"),
                                     p_mean=("p", "mean"), p_chg_mean=("p_chg_1h", "mean"))
    print(summ.round(4).to_string())
    results[n_states] = (m, scaler, df_s, bic)

print(f"\nBIC comparison: 3-state={results[3][3]:.1f}  4-state={results[4][3]:.1f}  "
      f"(lower is better; {'3-state wins' if results[3][3] < results[4][3] else '4-state wins'})")

# save the 3-state model + states for the real-trade backfill comparison
m3, scaler3, df3, _ = results[3]
import pickle
with open(f"{OUT}/hmm_pup_v3_regime_3state.pkl", "wb") as f:
    pickle.dump({"model": m3, "scaler": scaler3, "features": FEATS, "n_states": 3}, f)
df3.to_parquet(f"{OUT}/pup_v3_hmm_states_3state.parquet")
print(f"\nsaved 3-state artifacts")
