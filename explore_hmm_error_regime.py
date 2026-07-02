"""
explore_hmm_error_regime.py — Idea #1: HMM on Model Prediction Errors

Hypothesis: the calibration error between p_gbdt and resolved_yes has regime structure.
Hidden states capture periods where the model is systematically overestimating,
calibrated, or underestimating. These regimes predict whether the model's edge
is currently reliable — a meta-level sizing signal.

Three observation channels (each used separately and together):
  A. model_disagree   = p_gbdt - p_market       (how much model diverges from market consensus)
  B. predict_error    = p_gbdt - resolved_yes    (calibration error; noisy since outcome is 0/1)
  C. rolling_cal_err  = EWM of predict_error     (smoothed version of B)

We train a GaussianHMM on the time-ordered sequence of these signals, then ask:
  - Do states differ in subsequent P&L?
  - Do states predict periods where the model's edge is degraded?
  - How long do states persist? (Are they long enough to be actionable?)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────
ARCHIVE       = "results/btc_scan_archive.csv"
N_STATES      = [2, 3, 4]          # try multiple
EWM_SPAN      = 30                 # smoothing window for rolling calibration error
MIN_RESOLVED  = 500                # minimum resolved rows to train
FLAT_BET      = 100.0              # flat $100 per contract for P&L attribution
RANDOM_SEED   = 42

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading archive...")
df = pd.read_csv(ARCHIVE, low_memory=False)
for col in ["p_gbdt", "p_market", "resolved_yes", "offset_pct", "liq_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
df = df.dropna(subset=["logged_at", "p_gbdt", "p_market", "resolved_yes"])
df = df.sort_values("logged_at").reset_index(drop=True)

print(f"  Resolved rows: {len(df):,}  |  Date range: {df['logged_at'].min().date()} → {df['logged_at'].max().date()}")

# ── Build observation channels ──────────────────────────────────────────────
df["model_disagree"]  = df["p_gbdt"] - df["p_market"]              # A
df["predict_error"]   = df["p_gbdt"] - df["resolved_yes"]          # B  (noisy)
df["rolling_cal_err"] = (df["predict_error"]
                         .ewm(span=EWM_SPAN, min_periods=5)
                         .mean())                                   # C  (smoothed)

# Model-implied YES edge (what the model thinks the edge is)
df["model_edge_yes"]  = df["p_gbdt"] - df["p_market"]

# Flat P&L if we bet YES on every contract
df["pnl_yes_flat"] = np.where(
    df["resolved_yes"] == 1,
    FLAT_BET * (1 - df["p_market"]) / df["p_market"],
    -FLAT_BET
)
# Flat P&L if we bet NO on every contract
df["pnl_no_flat"] = np.where(
    df["resolved_yes"] == 0,
    FLAT_BET * df["p_market"] / (1 - df["p_market"]),
    -FLAT_BET
)

print(f"\nObservation stats:")
for col in ["model_disagree", "predict_error", "rolling_cal_err"]:
    print(f"  {col:20s}: mean={df[col].mean():+.4f}  std={df[col].std():.4f}  "
          f"min={df[col].min():+.4f}  max={df[col].max():+.4f}")

# ── Fit HMMs with different state counts ───────────────────────────────────
# Use rolling_cal_err (smoothed) as primary observation — less noisy than raw error
obs_col = "rolling_cal_err"
df = df.dropna(subset=[obs_col]).reset_index(drop=True)
X = df[[obs_col]].values.astype(float)

results = {}
print(f"\n{'='*65}")
print(f"Training GaussianHMMs on '{obs_col}'  (n={len(X)} observations)")
print(f"{'='*65}")

for n in N_STATES:
    model = GaussianHMM(
        n_components=n,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
    )
    model.fit(X)
    states = model.predict(X)
    score  = model.score(X) / len(X)   # log-likelihood per obs

    df[f"state_{n}"] = states
    results[n] = {"model": model, "score": score}

    print(f"\n── {n} states  (log-lik/obs = {score:.4f}) ──")
    for s in range(n):
        mask = states == s
        sub  = df[mask]
        n_s  = mask.sum()
        mean_err  = sub["rolling_cal_err"].mean()
        mean_dis  = sub["model_disagree"].mean()
        pnl_yes   = sub["pnl_yes_flat"].sum()
        pnl_no    = sub["pnl_no_flat"].sum()
        yes_wr    = sub["resolved_yes"].mean()
        be        = sub["p_market"].mean()          # avg breakeven for YES
        print(f"  State {s}: n={n_s:6,} ({n_s/len(df):.1%})  "
              f"cal_err={mean_err:+.4f}  disagree={mean_dis:+.4f}  "
              f"YES_WR={yes_wr:.3f} vs BE={be:.3f}  "
              f"pnl_yes=${pnl_yes:+,.0f}  pnl_no=${pnl_no:+,.0f}")

    # Transition matrix
    trans = model.transmat_
    print(f"  Transition matrix:")
    for row in trans:
        print(f"    {['→S'+str(i) for i in range(n)]}  {[f'{p:.3f}' for p in row]}")

    # Average state duration
    runs = np.diff(np.where(np.concatenate([[True], states[1:]!=states[:-1], [True]]))[0])
    print(f"  Avg state-run length: {runs.mean():.1f} obs  (median {np.median(runs):.0f})")

# ── Deep dive: best n (pick 3 states as default) ───────────────────────────
BEST_N = 3
print(f"\n{'='*65}")
print(f"DEEP DIVE — {BEST_N}-state model")
print(f"{'='*65}")

states3 = df[f"state_{BEST_N}"].values

# Name states by their mean calibration error
state_means = [(s, df[df[f"state_{BEST_N}"]==s]["rolling_cal_err"].mean()) for s in range(BEST_N)]
state_means.sort(key=lambda x: x[1])  # sort: most negative = most under-predicted

state_labels = {}
labels = ["OVER-EST", "CALIBRATED", "UNDER-EST"] if BEST_N == 3 else [f"S{i}" for i in range(BEST_N)]
for i, (s, _) in enumerate(state_means):
    state_labels[s] = labels[i]

print("\nState identity:")
for s, mean_e in state_means:
    print(f"  State {s} = {state_labels[s]}  (mean_cal_err={mean_e:+.4f})")

# ── Time-in-state analysis: are states long enough to use? ─────────────────
print(f"\nRun-length analysis (how many contracts stay in each state consecutively):")
df["run_id"] = (df[f"state_{BEST_N}"] != df[f"state_{BEST_N}"].shift()).cumsum()
run_stats = (df.groupby(["run_id", f"state_{BEST_N}"])
               .agg(length=("logged_at", "count"),
                    cal_err=("rolling_cal_err", "mean"),
                    pnl_yes=("pnl_yes_flat", "sum"),
                    pnl_no=("pnl_no_flat", "sum"))
               .reset_index())

for s in range(BEST_N):
    rs = run_stats[run_stats[f"state_{BEST_N}"] == s]["length"]
    print(f"  State {s} ({state_labels[s]}): median={rs.median():.0f}  p25={rs.quantile(.25):.0f}  "
          f"p75={rs.quantile(.75):.0f}  max={rs.max():.0f}  n_runs={len(rs)}")

# ── Stationarity check: does state predict SUBSEQUENT P&L? ─────────────────
# (Key question: is the state known before the bet, or only in hindsight?)
# We use rolling_cal_err which is built from PAST errors only (EWM), so it IS available live.
print(f"\nDoes current state predict subsequent 10-contract P&L? (forward-looking test)")
FORWARD = 10
df["fwd_pnl_yes"] = df["pnl_yes_flat"].rolling(FORWARD).sum().shift(-FORWARD)
df["fwd_pnl_no"]  = df["pnl_no_flat"].rolling(FORWARD).sum().shift(-FORWARD)

for s in range(BEST_N):
    mask = df[f"state_{BEST_N}"] == s
    sub  = df[mask].dropna(subset=["fwd_pnl_yes"])
    if len(sub) < 50: continue
    print(f"  State {s} ({state_labels[s]}): n={len(sub)}, "
          f"avg_fwd_pnl_yes=${sub['fwd_pnl_yes'].mean():+.2f}/10-contract-window  "
          f"avg_fwd_pnl_no=${sub['fwd_pnl_no'].mean():+.2f}/10-contract-window")

# ── Multi-feature HMM: combine disagree + rolling_cal_err ──────────────────
print(f"\n{'='*65}")
print(f"MULTI-FEATURE HMM — (model_disagree, rolling_cal_err)  3 states")
print(f"{'='*65}")
df2 = df.dropna(subset=["model_disagree", "rolling_cal_err"])
X2 = df2[["model_disagree", "rolling_cal_err"]].values.astype(float)
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

hmm2 = GaussianHMM(n_components=3, covariance_type="full", n_iter=300, random_state=RANDOM_SEED)
hmm2.fit(X2_scaled)
states2 = hmm2.predict(X2_scaled)
df2 = df2.copy()
df2["state_2feat"] = states2

print(f"Log-lik/obs: {hmm2.score(X2_scaled)/len(X2_scaled):.4f}")
for s in range(3):
    mask = states2 == s
    sub  = df2[mask]
    if len(sub) < 50: continue
    print(f"\n  State {s}: n={len(sub):,} ({len(sub)/len(df):.1%})")
    print(f"    model_disagree:  mean={sub['model_disagree'].mean():+.4f}  std={sub['model_disagree'].std():.4f}")
    print(f"    rolling_cal_err: mean={sub['rolling_cal_err'].mean():+.4f}  std={sub['rolling_cal_err'].std():.4f}")
    print(f"    YES_WR={sub['resolved_yes'].mean():.3f} vs BE={sub['p_market'].mean():.3f}  edge={sub['resolved_yes'].mean()-sub['p_market'].mean():+.3f}")
    print(f"    pnl_yes_flat=${sub['pnl_yes_flat'].sum():+,.0f}  pnl_no_flat=${sub['pnl_no_flat'].sum():+,.0f}")

# ── Save enriched dataframe for further exploration ─────────────────────────
out_path = "results/btc_scan_archive_error_hmm.parquet"
df.to_parquet(out_path, index=False)
print(f"\nSaved enriched archive to: {out_path}")
print(f"New columns: state_2, state_3, state_4, state_2feat, model_disagree, predict_error, rolling_cal_err")
print("\nDone.")
