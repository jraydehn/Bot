"""
S14 -- CAUSAL backfill of the intraday p_up_v3 regime HMM state, 2024 -> now.
The existing pup_v3_hmm_states.parquet was decoded with predict() over the
WHOLE 5-year sequence (Viterbi smoothed by future observations) -- fine for
the 07-06 exploration, NOT fine for conditioning calibration tables.

Live parity decode (paper_trade_runner._pup_v3_hmm_state): for each hour i,
take the trailing <=200 hourly p_up_v3 readings ending at i, build features
(p, p_chg_1h, p_ma6h rolling-6 min_periods=6), scaler.transform, model.predict
on that window, keep the LAST state, map {0,2}->neutral, 1->rising, 3->crashing.

Reading source: honest walk-forward preds (wf_preds_FINAL.parquet, out-of-fold)
through 2026-07-04, then live-logged hourly p_up_v3 from results/paper_trades.csv
(computed causally in production) spliced for the tail. Distribution seam at the
splice disclosed (out-of-fold model vs live final model).
Also quantifies how much the old smoothed decode disagreed with the causal one.
"""
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
OUT = "reform_results/pup15m_20260710"

pay = pickle.load(open("reform_results/hmm_pup_v3_regime.pkl", "rb"))
model, scaler = pay["model"], pay["scaler"]
MAP = {0: "neutral", 1: "rising", 2: "neutral", 3: "crashing"}

wf = pd.read_parquet("reform_results/pup_v2_rebuild_20260704/wf_preds_FINAL.parquet")
wf = wf[wf.index >= "2023-10-01"]["p"].dropna()

live = pd.read_csv("results/paper_trades.csv", usecols=["logged_at", "p_up_v3"], low_memory=False)
live["p_up_v3"] = pd.to_numeric(live["p_up_v3"], errors="coerce")
live["logged_at"] = pd.to_datetime(live["logged_at"], utc=True, errors="coerce", format="mixed")
live = live.dropna(subset=["p_up_v3", "logged_at"])
live["hour_ts"] = live["logged_at"].dt.floor("h")
live_h = live.drop_duplicates(subset="hour_ts", keep="last").set_index("hour_ts")["p_up_v3"]
live_h = live_h[live_h.index > wf.index.max()].sort_index()
print(f"wf readings: {len(wf)} through {wf.index.max()}  | live splice: {len(live_h)} "
      f"({live_h.index.min()} -> {live_h.index.max()})" if len(live_h) else "no live tail")
p = pd.concat([wf, live_h]).sort_index()
p = p[~p.index.duplicated(keep="first")]

feat = pd.DataFrame({"p": p, "p_chg_1h": p.diff(),
                     "p_ma6h": p.rolling(6, min_periods=6).mean()}).dropna()
X_all = scaler.transform(feat.values)
print(f"feature rows: {len(feat)}  {feat.index.min()} -> {feat.index.max()}")

# causal windowed decode (trailing 200 rows incl. current), start where >=10 rows exist
states = []
idx = feat.index
for i in range(9, len(feat)):
    lo = max(0, i - 199)
    s = model.predict(X_all[lo:i + 1])
    states.append((idx[i], MAP[int(s[-1])]))
st = pd.DataFrame(states, columns=["hour_ts", "pv3_state"])
st.to_csv(f"{OUT}/pupv3_state_causal.csv", index=False)
print(f"causal states: {len(st)}")
print("occupancy:", (st["pv3_state"].value_counts(normalize=True) * 100).round(1).to_dict())
run = (st["pv3_state"] != st["pv3_state"].shift()).cumsum()
dw = st.groupby(run).agg(s=("pv3_state", "first"), n=("pv3_state", "size")).groupby("s")["n"].mean()
print("avg dwell (hours):", dw.round(1).to_dict())
st["yr"] = st["hour_ts"].dt.year
print(st.groupby("yr")["pv3_state"].value_counts(normalize=True).round(3).to_string())

# lookahead audit: causal vs old full-sequence smoothed decode
old = pd.read_parquet("reform_results/pup_v3_15m_window_sweep_20260706/pup_v3_hmm_states.parquet")
old_lbl = old["state"].map(MAP)
cmp = st.set_index("hour_ts")["pv3_state"].to_frame("causal").join(old_lbl.rename("smoothed"), how="inner")
print(f"\ncausal vs smoothed decode: overlap={len(cmp)}  agreement={(cmp['causal']==cmp['smoothed']).mean():.3f}")
print(pd.crosstab(cmp["causal"], cmp["smoothed"]).to_string())
print("DONE_S14")
