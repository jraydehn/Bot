"""
S37 -- causally reconstruct the multitf HMM's (hmm_btc_multitf.pkl, 8-state,
22-feature) historical state sequence, replicating the EXACT live decode:
a rolling deque(maxlen=20) of scaled observations, built up sequentially
across scan cycles (paper_trades_btc15m.csv rows, ~1/cycle), decoded with
model.predict() on the accumulated buffer, taking the LAST state each time
(matches _hmm_predict_state exactly -- no retraining, saved model+scaler
used as-is). This state already actively gates live decisions (state 0
blocks NO, state 7 gets an elevated Kelly cap) but was never logged, so it's
never been backtested against the real book before.
"""
import warnings
import pickle
from collections import deque
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

with open("hmm_btc_multitf.pkl", "rb") as f:
    pkg = pickle.load(f)
model, scaler, feat_cols = pkg["model"], pkg["scaler"], pkg["feat_cols"]
print(f"feat_cols ({len(feat_cols)}): {feat_cols}")
print("known state_stats (from original build):")
for s, st in sorted(pkg["state_stats"].items()):
    print(f"  state {s}: wr={st['wr']:.1f}%  pnl=${st['pnl']:+.2f}  ppt=${st['ppt']:+.2f}  n_trades={st['n_trades']}")

df = pd.read_csv("results/paper_trades_btc15m.csv", low_memory=False)
df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["decision_time"]).sort_values("decision_time").reset_index(drop=True)
print(f"\ntotal scan rows: {len(df)}")

# RESTART-AWARE FIX: _hmm_obs_buf is a module-level deque -- it only holds
# observations accumulated since the process last started, and empties on
# every restart (crash, manual restart, watchdog cycle). A naive continuous
# replay (first attempt) produced 79.8% occupancy in one state, wildly
# mismatching the model's own training-time state_stats (fairly even
# spread) -- a sign of unfaithful reconstruction, not regime drift. Fixed
# by resetting the buffer whenever the scan-cycle gap exceeds 20min (4x the
# ~5min median cadence; the 90th/95th percentile of normal-cadence gaps
# tops out ~15min, so 20min cleanly separates cadence noise from real
# interruptions -- 184 such gaps across 8+ weeks, which includes several
# restarts from today's session alone).
RESTART_GAP_MIN = 20.0
gap_min = df["decision_time"].diff().dt.total_seconds().div(60)
restart_after = gap_min > RESTART_GAP_MIN
print(f"restart-like gaps (>{RESTART_GAP_MIN}min): {restart_after.sum()}")

obs_buf = deque(maxlen=20)
states = []
for i, row in df.iterrows():
    if restart_after.iloc[i]:
        obs_buf.clear()
    vec = []
    for c in feat_cols:
        v = row.get(c, np.nan)
        try:
            vec.append(float(v) if v == v else np.nan)
        except (TypeError, ValueError):
            vec.append(np.nan)
    if sum(1 for v in vec if v == v) < len(vec) * 0.75:
        states.append(-1)
        continue
    vec_arr = np.array([0.0 if v != v else v for v in vec], dtype=float).reshape(1, -1)
    try:
        vec_scaled = scaler.transform(vec_arr)[0]
    except Exception:
        states.append(-1)
        continue
    obs_buf.append(vec_scaled)
    if len(obs_buf) < 3:
        states.append(-1)
        continue
    obs_seq = np.array(list(obs_buf))
    try:
        st = model.predict(obs_seq, lengths=[len(obs_seq)])
        states.append(int(st[-1]))
    except Exception:
        states.append(-1)

df["multitf_state"] = states
valid = df[df["multitf_state"] >= 0]
print(f"\nvalid decoded states: {len(valid)}/{len(df)}")
print("occupancy:", (valid["multitf_state"].value_counts(normalize=True) * 100).round(1).sort_index().to_dict())

# ---- join to the real taken YES book ----
t = df[(df["side"] == "yes") & (pd.to_numeric(df["bet_amount"], errors="coerce") > 0)]
t = t.dropna(subset=["would_pnl", "resolved_yes"])
t = t[t["multitf_state"] >= 0].copy()
t["win"] = t["resolved_yes"]
print(f"\nreal taken YES trades with valid multitf_state: {len(t)}")
print(t.groupby("multitf_state").agg(n=("win", "size"), wr=("win", "mean"),
     pnl=("would_pnl", "sum")).round(3).to_string())

df[["decision_time", "multitf_state"]].to_csv(f"{OUT}/s37_multitf_states.csv", index=False)
print("\nDONE_S37")
