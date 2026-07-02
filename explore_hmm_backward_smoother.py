"""
Backward-Smoothed HMM (#4) — Offline relabeling.

Real-time decoding (Viterbi) can only use past observations.
Forward-backward posteriors use the FULL session trajectory — past and future —
giving cleaner state assignments for historical analysis and retraining.

Applied to: PS HMM (phase-space trajectory, 6 states, 15-min buckets).

Questions answered:
  1. How much uncertainty does the backward pass resolve? (entropy reduction)
  2. Do smoothed labels give stronger P&L signal than Viterbi labels?
  3. Does retraining on smoothed labels improve the model?
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as scipy_entropy

MODEL_PATH   = Path("models/hmm_phase_traj_btc.pkl")
RETRAIN_PATH = Path("models/hmm_phase_traj_btc_smoothed.pkl")
GAP_THRESHOLD = pd.Timedelta("2h")
MIN_SESSION   = 5

# ── Load model ─────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    pkg = pickle.load(f)
model  = pkg["model"]
scaler = pkg["scaler"]
FEATURES = pkg["features"]
N_STATES = pkg["n_states"]
state_descs = pkg.get("state_descriptions", {})
print(f"PS HMM: {N_STATES} states  features: {FEATURES}")

# ── Rebuild 15-min time series (same pipeline as explore_hmm_phase_trajectory.py) ──
print("\nRebuilding 15-min session data ...")
arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc = arc.dropna(subset=["logged_at"]).sort_values("logged_at")

ts = (arc[["logged_at","stoch_k"]]
      .drop_duplicates(subset=["logged_at"]).copy())
ts = ts.rename(columns={"stoch_k": "stoch_k_1h"})

ts["bucket"] = ts["logged_at"].dt.floor("15min")
ts = ts.groupby("bucket").first().reset_index()
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()

sess_sizes = ts.groupby("session").size()
valid_sess = sess_sizes[sess_sizes >= MIN_SESSION].index
ts = ts[ts["session"].isin(valid_sess)].reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()

# 15-min stoch (from live_1m in archive if available, else same as 1h)
ts["stoch_k_15m"] = ts["stoch_k_1h"]  # fallback
if "stoch_k_15m" in arc.columns:
    arc15 = arc.dropna(subset=["stoch_k_15m"])[["logged_at","stoch_k_15m"]].copy()
    arc15["bucket"] = arc15["logged_at"].dt.floor("15min")
    arc15 = arc15.groupby("bucket")["stoch_k_15m"].mean().reset_index()
    ts = ts.merge(arc15.rename(columns={"stoch_k_15m":"stoch_k_15m_raw"}),
                  on="bucket", how="left")
    mask = ts["stoch_k_15m_raw"].notna()
    ts.loc[mask, "stoch_k_15m"] = ts.loc[mask, "stoch_k_15m_raw"]

ts["d_sk15m"]   = ts.groupby("session")["stoch_k_15m"].diff().fillna(0.0).clip(-40, 40)
ts["d_sk1h"]    = ts.groupby("session")["stoch_k_1h"].diff().fillna(0.0).clip(-25, 25)
ts["divergence"] = ts["stoch_k_1h"] - ts["stoch_k_15m"]

ts = ts.dropna(subset=FEATURES).reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sessions = ts.groupby("session").size()
valid_sess2 = sessions[sessions >= MIN_SESSION].index
ts = ts[ts["session"].isin(valid_sess2)].reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sessions = ts.groupby("session").size()

print(f"15-min observations: {len(ts):,}  sessions: {len(sessions)}")
print(f"Session lengths: {sessions.values.tolist()}")

X = scaler.transform(ts[FEATURES].values)
lengths = sessions.values.tolist()

# ── Viterbi (forward-only) vs Forward-Backward (smoothed) ─────────────────────
print(f"\n{'─'*60}")
print("Comparing Viterbi vs Forward-Backward labels ...")

viterbi_states = model.predict(X, lengths=lengths)        # causal MAP path
posteriors     = model.predict_proba(X, lengths=lengths)  # P(St | all X), shape (n,6)
smooth_states  = posteriors.argmax(axis=1)                # smoothed hard label

ts["state_viterbi"] = viterbi_states
ts["state_smooth"]  = smooth_states
for s in range(N_STATES):
    ts[f"post_{s}"] = posteriors[:, s]

agreement = (viterbi_states == smooth_states).mean()
n_differ  = (viterbi_states != smooth_states).sum()
print(f"Agreement: {agreement:.1%}  |  Differ: {n_differ:,} / {len(ts):,} observations")

# Entropy analysis: how uncertain is each decoding?
viterbi_onehot = np.eye(N_STATES)[viterbi_states]
ent_viterbi = scipy_entropy(viterbi_onehot.T + 1e-9).mean()  # 0 = certain
ent_smooth  = scipy_entropy(posteriors.T     + 1e-9).mean()   # lower = more certain
print(f"Mean posterior entropy — Viterbi (one-hot): {ent_viterbi:.4f}")
print(f"Mean posterior entropy — Smoothed:          {ent_smooth:.4f}")
print(f"Entropy reduction: {(ent_viterbi - ent_smooth) / ent_viterbi * 100:.1f}%")
print(f"  (lower entropy = smoother is MORE certain about state assignments)")

# Disagreement by state
print(f"\nDisagreement by Viterbi state:")
for s in range(N_STATES):
    mask = viterbi_states == s
    n    = mask.sum()
    diff = (smooth_states[mask] != s).sum()
    avg_post = posteriors[mask, s].mean()
    desc = state_descs.get(s, "?")
    print(f"  St{s} ({desc:<20}): n={n:5,}  disagree={diff:4,} ({diff/max(n,1):.0%})  "
          f"avg_posterior={avg_post:.3f}")

# Where they disagree — what does smooth prefer?
print(f"\nDisagreement detail (Viterbi→Smooth transitions):")
for s_v in range(N_STATES):
    for s_sm in range(N_STATES):
        if s_v == s_sm: continue
        n = ((viterbi_states == s_v) & (smooth_states == s_sm)).sum()
        if n < 10: continue
        pct = n / len(ts) * 100
        print(f"  St{s_v}({state_descs.get(s_v,'?')[:12]}) → St{s_sm}({state_descs.get(s_sm,'?')[:12]})  "
              f"n={n:4,}  ({pct:.1f}% of all obs)")

# ── P&L analysis: Viterbi vs Smoothed labels ──────────────────────────────────
print(f"\n{'─'*60}")
print("Loading BTC archive for P&L comparison ...")

arc_full = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc_full["logged_at"] = pd.to_datetime(arc_full["logged_at"], format="mixed", utc=True, errors="coerce")
arc_full = arc_full.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
arc_full["resolved_yes"] = pd.to_numeric(arc_full.get("resolved_yes", np.nan), errors="coerce")
arc_full["p_market"]     = pd.to_numeric(arc_full.get("p_market", np.nan), errors="coerce")

# Build lookup tables for both label sets — lagged 1 bucket (15min) to avoid lookahead
vit_lkup = ts[["bucket","state_viterbi"]].copy()
vit_lkup["bucket"] += pd.Timedelta("15min")
vit_lkup = vit_lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

smo_lkup = ts[["bucket","state_smooth"]].copy()
smo_lkup["bucket"] += pd.Timedelta("15min")
smo_lkup = smo_lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

arc_vit = pd.merge_asof(arc_full, vit_lkup, on="logged_at",
                         direction="nearest", tolerance=pd.Timedelta("8min"))
arc_smo = pd.merge_asof(arc_full, smo_lkup, on="logged_at",
                         direction="nearest", tolerance=pd.Timedelta("8min"))

res_v = arc_vit.dropna(subset=["resolved_yes","p_market","state_viterbi"]).copy()
res_s = arc_smo.dropna(subset=["resolved_yes","p_market","state_smooth"]).copy()
res_v["state_viterbi"] = res_v["state_viterbi"].astype(int)
res_s["state_smooth"]  = res_s["state_smooth"].astype(int)
print(f"Resolved with Viterbi label: {len(res_v):,}  Smoothed: {len(res_s):,}")


def kelly_pnl(sub, pm="p_market", out="resolved_yes"):
    p = sub[pm].clip(0.01, 0.99); w = sub[out]
    f = (p - (1-p)).clip(0, 0.25)
    return (f * ((1-p)*w - p*(1-w))).sum()


def mcpt_b(wr_arr, be_arr, n=2000, seed=42):
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    null = np.array([(rng.random(len(be_arr))<be_arr).astype(float).mean()-be_arr.mean()
                     for _ in range(n)])
    p = np.mean(null<=obs) if obs<0 else np.mean(null>=obs)
    z = (obs-null.mean())/(null.std()+1e-9)
    return float(z), float(p)


print(f"\n{'─'*60}")
print("YES-side P&L by state:")
print(f"\n{'Label':<9} {'St':<4} {'Desc':<22} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>8}")
print("─"*80)

for lbl, res, col in [("Viterbi", res_v, "state_viterbi"), ("Smooth", res_s, "state_smooth")]:
    sub_base = res[res["p_market"].between(0.05, 0.95)].copy()
    print(f"\n{lbl}:")
    for s in range(N_STATES):
        sub = sub_base[sub_base[col] == s]
        if len(sub) < 20:
            print(f"  {'':9} St{s} {state_descs.get(s,'?'):<22} {len(sub):>6}  (too few)")
            continue
        wr = sub["resolved_yes"].mean()
        be = sub["p_market"].mean()
        edge = wr - be
        pnl = kelly_pnl(sub)
        z, p = mcpt_b(sub["resolved_yes"].values, sub["p_market"].values)
        flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30 else ""
        print(f"  {'':9} St{s} {state_descs.get(s,'?'):<22} {len(sub):>6} "
              f"{wr:>6.1%} {be:>6.1%} {edge:>+7.1%} {z:>+7.2f} {p:>6.4f} {pnl:>+8,.0f}{flag}")

print(f"\n{'─'*60}")
print("NO-side P&L by state:")
for lbl, res, col in [("Viterbi", res_v, "state_viterbi"), ("Smooth", res_s, "state_smooth")]:
    sub_base = res[res["p_market"].between(0.05, 0.95)].copy()
    sub_base["resolved_yes"] = 1 - sub_base["resolved_yes"]
    sub_base["p_market"]     = 1 - sub_base["p_market"]
    sub_base = sub_base[sub_base["p_market"].between(0.05, 0.95)]
    print(f"\n{lbl}:")
    for s in range(N_STATES):
        sub = sub_base[sub_base[col] == s]
        if len(sub) < 20: continue
        wr = sub["resolved_yes"].mean()
        be = sub["p_market"].mean()
        edge = wr - be
        pnl = kelly_pnl(sub)
        z, p = mcpt_b(sub["resolved_yes"].values, sub["p_market"].values)
        flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30 else ""
        print(f"  {'':9} St{s} {state_descs.get(s,'?'):<22} {len(sub):>6} "
              f"{wr:>6.1%} {be:>6.1%} {edge:>+7.1%} {z:>+7.2f} {p:>6.4f} {pnl:>+8,.0f}{flag}")

# ── Walk-forward with smoothed labels ─────────────────────────────────────────
print(f"\n{'─'*60}")
print("Walk-forward: do smoothed labels change WF outcomes?")

mid_dt = ts["bucket"].median()
ts_tr  = ts[ts["bucket"] <= mid_dt]
ts_te  = ts[ts["bucket"] >  mid_dt]
print(f"Split at {mid_dt.date()}")

X_tr = scaler.transform(ts_tr[FEATURES].values)
X_te = scaler.transform(ts_te[FEATURES].values)
lens_tr = [l for l in ts_tr.groupby("session").size().values if l >= MIN_SESSION]
lens_te = [l for l in ts_te.groupby("session").size().values if l >= MIN_SESSION]

# Retrain on smoothed labels: use smoothed posteriors as initial conditions
# hmmlearn doesn't support soft-label training directly, so we:
#   (a) retrain normally on train half — compare to original
#   (b) retrain with startprob/transmat initialized from smoothed statistics

# (a) Standard retrain on train half
best_ll, best_m = -np.inf, None
for seed in range(20):
    try:
        m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=500, random_state=seed, tol=1e-5)
        m.fit(X_tr, lengths=lens_tr)
        ll = m.score(X_tr, lengths=lens_tr)
        if ll > best_ll: best_ll, best_m = ll, m
    except Exception: pass

if best_m is None:
    print("  Retrain failed."); raise SystemExit

vit_tr  = best_m.predict(X_tr, lengths=lens_tr)
post_tr = best_m.predict_proba(X_tr, lengths=lens_tr)
smo_tr  = post_tr.argmax(axis=1)

vit_te  = best_m.predict(X_te, lengths=lens_te)
post_te = best_m.predict_proba(X_te, lengths=lens_te)
smo_te  = post_te.argmax(axis=1)

# Map train centroids → test centroids for alignment
tr_cents = {}
ts_tr_cp = ts_tr.copy(); ts_tr_cp["wf_state"] = vit_tr
for s in range(N_STATES):
    m_s = ts_tr_cp[ts_tr_cp["wf_state"] == s]
    if len(m_s) > 0: tr_cents[s] = m_s[FEATURES].mean().values

# Align full-model states to WF-train states via centroid distance
def align_states(source_cents, target_cents):
    mapping = {}
    for s_full, c_full in source_cents.items():
        best_s, best_d = 0, np.inf
        for s_tr, c_tr in target_cents.items():
            d = np.linalg.norm(scaler.transform(c_full.reshape(1,-1)) -
                               scaler.transform(c_tr.reshape(1,-1)))
            if d < best_d: best_d, best_s = d, s_tr
        mapping[s_full] = best_s
    return mapping

full_cents = {s: np.array([pkg["state_centroids"][s][k]
              for k in ["sk15","sk1h","d15","d1h","div"]])
              if s in pkg.get("state_centroids", {})
              else ts[ts["state_viterbi"]==s][FEATURES].mean().values
              for s in range(N_STATES)}
# Safer: just use ts centroid
full_cents2 = {}
for s in range(N_STATES):
    m2 = ts[ts["state_viterbi"]==s]
    if len(m2)>0: full_cents2[s] = m2[FEATURES].mean().values

align_map = align_states(full_cents2, tr_cents) if tr_cents and full_cents2 else {s:s for s in range(N_STATES)}
print(f"State alignment map (full→WF-train): {align_map}")

# Build WF lookups for both label methods
def build_wf_lkup(ts_half, states_arr, label="wf"):
    df = ts_half.copy(); df[label] = states_arr
    lkup = df[["bucket", label]].copy()
    lkup["bucket"] += pd.Timedelta("15min")
    return lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

# Join WF labels to archive
arc_wf_v_tr = pd.merge_asof(arc_full[arc_full["logged_at"]<=mid_dt].sort_values("logged_at"),
                              build_wf_lkup(ts_tr, vit_tr, "wf_vit"),
                              on="logged_at", direction="nearest", tolerance=pd.Timedelta("8min"))
arc_wf_s_tr = pd.merge_asof(arc_full[arc_full["logged_at"]<=mid_dt].sort_values("logged_at"),
                              build_wf_lkup(ts_tr, smo_tr, "wf_smo"),
                              on="logged_at", direction="nearest", tolerance=pd.Timedelta("8min"))
arc_wf_v_te = pd.merge_asof(arc_full[arc_full["logged_at"]>mid_dt].sort_values("logged_at"),
                              build_wf_lkup(ts_te, vit_te, "wf_vit"),
                              on="logged_at", direction="nearest", tolerance=pd.Timedelta("8min"))
arc_wf_s_te = pd.merge_asof(arc_full[arc_full["logged_at"]>mid_dt].sort_values("logged_at"),
                              build_wf_lkup(ts_te, smo_te, "wf_smo"),
                              on="logged_at", direction="nearest", tolerance=pd.Timedelta("8min"))


def wf_pnl(arc_half, col, state, side="yes"):
    sub = arc_half[(arc_half[col]==state) & arc_half["p_market"].between(0.05,0.95)
                   & arc_half["resolved_yes"].notna()].copy()
    if side == "no":
        sub["resolved_yes"] = 1 - sub["resolved_yes"]
        sub["p_market"]     = 1 - sub["p_market"]
        sub = sub[sub["p_market"].between(0.05,0.95)]
    if len(sub) < 3: return len(sub), np.nan, np.nan, np.nan
    wr = sub["resolved_yes"].mean(); be = sub["p_market"].mean()
    return len(sub), wr, wr-be, kelly_pnl(sub)


print(f"\nWF P&L by label method and state (YES side):")
print(f"{'':4} {'Viterbi TRAIN':>25}  {'Viterbi TEST':>25}  {'Smooth TRAIN':>25}  {'Smooth TEST':>25}")
for s in range(N_STATES):
    s_al = align_map.get(s, s)
    n_vt, wr_vt, e_vt, p_vt = wf_pnl(arc_wf_v_tr, "wf_vit", s_al)
    n_ve, wr_ve, e_ve, p_ve = wf_pnl(arc_wf_v_te, "wf_vit", s_al)
    n_st, wr_st, e_st, p_st = wf_pnl(arc_wf_s_tr, "wf_smo", s_al)
    n_se, wr_se, e_se, p_se = wf_pnl(arc_wf_s_te, "wf_smo", s_al)
    vit_wf = "PASS" if (not math.isnan(e_vt or 0) and not math.isnan(e_ve or 0)
                        and (e_vt or 0)<-0.02 and (e_ve or 0)<-0.02) else "----"
    smo_wf = "PASS" if (not math.isnan(e_st or 0) and not math.isnan(e_se or 0)
                        and (e_st or 0)<-0.02 and (e_se or 0)<-0.02) else "----"
    desc = state_descs.get(s,'?')[:14]
    print(f"St{s} ({desc:<14})")
    fmt = lambda n,e: f"n={n:4d} e={e:+.1%}" if not math.isnan(e or 0) else f"n={n:4d} e=?    "
    print(f"  Viterbi [{vit_wf}]: train {fmt(n_vt, e_vt or 0)}  |  test {fmt(n_ve, e_ve or 0)}")
    print(f"  Smooth  [{smo_wf}]: train {fmt(n_st, e_st or 0)}  |  test {fmt(n_se, e_se or 0)}")

# ── Save retrained (smoothed-init) model ──────────────────────────────────────
print(f"\n{'─'*60}")
print("Saving smoothed-label retrained model ...")

# Retrain on FULL data but with more restarts, using smoothed posteriors
# as startprob initialization (warm start from smoothed statistics)
smooth_counts = np.bincount(smooth_states, minlength=N_STATES).astype(float)
smooth_startprob = smooth_counts / smooth_counts.sum()

best_ll2, best_m2 = -np.inf, None
for seed in range(30):
    try:
        m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=500, random_state=seed, tol=1e-5)
        m.startprob_ = smooth_startprob + 1e-6
        m.startprob_ /= m.startprob_.sum()
        m.fit(X, lengths=lengths)
        ll = m.score(X, lengths=lengths)
        if ll > best_ll2: best_ll2, best_m2 = ll, m
    except Exception: pass

k = N_STATES*(N_STATES-1) + 2*N_STATES*len(FEATURES)
bic_orig = -2*model.score(X, lengths=lengths) + k*np.log(len(X))
bic_new  = -2*best_ll2 + k*np.log(len(X))
print(f"Original BIC:        {bic_orig:,.1f}")
print(f"Smoothed-init BIC:   {bic_new:,.1f}  ({'better' if bic_new < bic_orig else 'worse'})")

new_states = best_m2.predict(X, lengths=lengths)
agree_new = (new_states == smooth_states).mean()
print(f"New-model vs smoothed-labels agreement: {agree_new:.1%}")

retrain_pkg = dict(
    model=best_m2, scaler=scaler, n_states=N_STATES, features=FEATURES,
    state_descriptions=state_descs,
    bic_original=bic_orig, bic_smoothed=bic_new,
    label_agreement=agree_new,
    trained_on=pkg["trained_on"],
)
with open(RETRAIN_PATH, "wb") as f:
    pickle.dump(retrain_pkg, f)
print(f"Saved → {RETRAIN_PATH}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("SUMMARY — Backward Smoother")
print(f"  Viterbi vs Smoothed agreement:  {agreement:.1%}")
print(f"  Posterior entropy (Viterbi):    {ent_viterbi:.4f}")
print(f"  Posterior entropy (Smoothed):   {ent_smooth:.4f}")
print(f"  Entropy reduction:              {(ent_viterbi-ent_smooth)/ent_viterbi*100:.1f}%")
print(f"  BIC original / smoothed-init:   {bic_orig:,.0f} / {bic_new:,.0f}")
print()
print("States where smoothing relabels most:")
for s in range(N_STATES):
    mask = viterbi_states == s
    diff = (smooth_states[mask] != s).sum()
    pct  = diff / max(mask.sum(), 1)
    if pct > 0.05:
        print(f"  St{s} ({state_descs.get(s,'?'):<20}): {pct:.0%} relabeled by smoother")
print()
print("Interpretation:")
print("  High agreement (>90%)  → Viterbi is already clean; smoother adds little")
print("  Low agreement (<80%)   → State boundaries uncertain; smoothed labels")
print("                           are more trustworthy for gate design")
