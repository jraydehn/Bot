"""
Phase-Space Trajectory HMM — explore and train.

Down-sample scan archive to 15m buckets so velocity features are meaningful.
Observation: (stoch_k_15m, stoch_k_1h, Δsk_15m[15m], Δsk_1h[15m], divergence=sk1h-sk15m)
States capture direction of travel in momentum space.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

N_STATES_RANGE = [3, 4, 5, 6]
GAP_THRESHOLD  = pd.Timedelta("2h")
MIN_SESSION    = 5      # discard sessions shorter than this
MODEL_PATH     = Path("models/hmm_phase_traj_btc.pkl")


def describe_state(sk15, sk1h, d15, d1h, div):
    overbought = sk15 > 65 or sk1h > 65
    oversold   = sk15 < 35 and sk1h < 35
    rising     = d15 > 5 and d1h > 1
    falling    = d15 < -5 and d1h < -1
    diverging  = sk1h > 55 and sk15 < 50 and d15 < -3
    chop       = abs(d15) < 5 and abs(d1h) < 2

    if diverging:
        return "1h->15m_divergence"
    if overbought and falling:
        return "unwinding_OB"
    if rising and not oversold:
        return "momentum_up"
    if oversold and rising:
        return "recovering_OS"
    if oversold and (falling or chop):
        return "deeply_oversold"
    if chop and not overbought and not oversold:
        return "chop_neutral"
    if overbought and chop:
        return "overbought_chop"
    if falling:
        return "falling"
    if rising:
        return "rising"
    return "mixed"


# ── Load + build 15m time series ─────────────────────────────────────────────
print("Loading btc_scan_archive_hmm.parquet ...")
arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc = arc.sort_values("logged_at").reset_index(drop=True)

# 1 row per timestamp → resample to 15m buckets
ts_raw = (arc.dropna(subset=["stoch_k_15m","stoch_k_1h"])
             .drop_duplicates(subset=["logged_at"])
             .sort_values("logged_at")
             .reset_index(drop=True)
             [["logged_at","stoch_k_15m","stoch_k_1h"]])

ts_raw["bucket"] = ts_raw["logged_at"].dt.floor("15min")
ts = (ts_raw.groupby("bucket")
            .agg(stoch_k_15m=("stoch_k_15m","mean"),
                 stoch_k_1h=("stoch_k_1h","mean"))
            .reset_index()
            .rename(columns={"bucket":"logged_at"}))

# Session boundaries
ts["gap"] = ts["logged_at"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()

# Velocity (meaningful at 15m scale)
ts["d_sk15m"]    = ts["stoch_k_15m"].diff().clip(-40, 40)
ts["d_sk1h"]     = ts["stoch_k_1h"].diff().clip(-25, 25)
ts["divergence"] = ts["stoch_k_1h"] - ts["stoch_k_15m"]
ts.loc[ts["gap"], ["d_sk15m","d_sk1h"]] = np.nan
ts = ts.dropna(subset=["d_sk15m","d_sk1h"]).reset_index(drop=True)

# Recompute sessions after drop
ts["gap"] = ts["logged_at"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()

# Drop short sessions
sess_sizes = ts.groupby("session").size()
valid_sess = sess_sizes[sess_sizes >= MIN_SESSION].index
ts = ts[ts["session"].isin(valid_sess)].reset_index(drop=True)

sessions = ts.groupby("session").size()
print(f"15m time series: {len(ts)} obs, {len(sessions)} sessions")
print(f"Session lengths: {sessions.values.tolist()}")
print(f"Date range: {ts['logged_at'].min().date()} → {ts['logged_at'].max().date()}")

FEATURES = ["stoch_k_15m","stoch_k_1h","d_sk15m","d_sk1h","divergence"]
X_raw = ts[FEATURES].values

# ── Scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Build length arrays for hmmlearn (per session)
lengths = sessions.values.tolist()

# ── BIC model selection ──────────────────────────────────────────────────────
print(f"\nBIC model selection (30 restarts per n):")
best_bic, best_n, best_model = np.inf, None, None
results_by_n = {}

for n in N_STATES_RANGE:
    scores = []
    for seed in range(30):
        try:
            m = GaussianHMM(n_components=n, covariance_type="diag",
                            n_iter=500, random_state=seed, tol=1e-5)
            m.fit(X, lengths=lengths)
            ll = m.score(X, lengths=lengths)
            # k: transition params + means + diagonal covars
            k = n*(n-1) + 2*n*len(FEATURES)
            bic = -2*ll + k*np.log(len(X))
            scores.append((bic, ll, m))
        except Exception:
            pass
    if not scores:
        continue
    scores.sort(key=lambda x: x[0])
    bic, ll, m = scores[0]
    results_by_n[n] = (bic, ll, m)
    marker = " ◄ BEST" if bic < best_bic else ""
    print(f"  n={n}: BIC={bic:,.1f}  LL={ll:,.1f}{marker}")
    if bic < best_bic:
        best_bic, best_n, best_model = bic, n, m

print(f"\nSelected n={best_n} states (BIC={best_bic:,.1f})")
model = best_model
N_STATES = best_n

# ── Decode states ─────────────────────────────────────────────────────────────
states = model.predict(X, lengths=lengths)
ts["ps_state"] = states

# ── Characterize each state ───────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("State centroids (raw feature space):")
state_info = {}
for s in range(N_STATES):
    mask = ts["ps_state"] == s
    n_obs = mask.sum()
    if n_obs == 0:
        state_info[s] = {"n": 0, "sk15": np.nan, "sk1h": np.nan,
                          "d15": np.nan, "d1h": np.nan, "div": np.nan, "desc": "empty"}
        continue
    sk15 = ts.loc[mask,"stoch_k_15m"].mean()
    sk1h = ts.loc[mask,"stoch_k_1h"].mean()
    d15  = ts.loc[mask,"d_sk15m"].mean()
    d1h  = ts.loc[mask,"d_sk1h"].mean()
    div  = ts.loc[mask,"divergence"].mean()
    desc = describe_state(sk15, sk1h, d15, d1h, div)
    state_info[s] = {"n": n_obs, "sk15": sk15, "sk1h": sk1h,
                      "d15": d15, "d1h": d1h, "div": div, "desc": desc}
    print(f"  St{s}: n={n_obs:4} ({n_obs/len(ts)*100:.1f}%)  "
          f"sk15={sk15:.1f}  sk1h={sk1h:.1f}  "
          f"Δ15m={d15:+.1f}  Δ1h={d1h:+.1f}  div={div:+.1f}  → {desc}")

# ── Transition matrix ─────────────────────────────────────────────────────────
T = model.transmat_
print(f"\nTransition matrix:")
print("        " + "".join(f"  →St{j}" for j in range(N_STATES)))
for i in range(N_STATES):
    dur = 1.0 / (1 - T[i,i]) if T[i,i] < 1 else 999.0
    print(f"  St{i} ({state_info[i]['desc'][:12]:<12})" +
          "".join(f"  {T[i,j]:.3f}" for j in range(N_STATES)) +
          f"  (~{dur:.0f} bars = {dur*15:.0f}min)")

# ── Join state labels back to full-resolution scan archive ────────────────────
print(f"\n{'─'*65}")
print("Joining states back to full-resolution scan archive ...")

ts_lookup = ts[["logged_at","ps_state"]].sort_values("logged_at")
arc_sorted = arc.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
arc_merged = pd.merge_asof(
    arc_sorted,
    ts_lookup,
    on="logged_at",
    direction="nearest",
    tolerance=pd.Timedelta("8min")   # 15m bucket ± half
)
n_tagged = arc_merged["ps_state"].notna().sum()
print(f"Archive rows: {len(arc_merged):,}  — ps_state tagged: {n_tagged:,} "
      f"({n_tagged/len(arc_merged)*100:.1f}%)")

arc_merged["resolved_yes"] = pd.to_numeric(arc_merged.get("resolved_yes", np.nan), errors="coerce")
arc_merged["p_market"]     = pd.to_numeric(arc_merged.get("p_market", np.nan), errors="coerce")

res = arc_merged.dropna(subset=["resolved_yes","p_market","ps_state"]).copy()
res["ps_state"] = res["ps_state"].astype(int)
print(f"Resolved rows with state: {len(res):,}")


def kelly_pnl(sub, pm_col="p_market", out_col="resolved_yes"):
    pm = sub[pm_col].clip(0.01, 0.99)
    win = sub[out_col]
    f = (pm - (1-pm)).clip(0, 0.25)
    return (f * ((1-pm)*win - pm*(1-win))).sum()


def mcpt(wr_arr, be_arr, n_perm=2000, seed=42):
    """H0: each outcome ~ Bernoulli(p_market_i). Draw under H0 to build null dist."""
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    # Under H0, each contract resolves YES with its own p_market probability
    null = np.array([(rng.random(len(be_arr)) < be_arr).astype(float).mean() - be_arr.mean()
                     for _ in range(n_perm)])
    if obs < 0:
        p = np.mean(null <= obs)
    else:
        p = np.mean(null >= obs)
    z = (obs - null.mean()) / (null.std() + 1e-9)
    return float(z), float(p)


# ── P&L breakdown by state — YES side ────────────────────────────────────────
print(f"\n{'─'*65}")
print("YES side P&L by state:")
print(f"  {'St':<4} {'Description':<22} {'n':>5} {'WR':>6} {'BE':>6} {'Edge':>7} "
      f"{'MCPT_z':>7} {'p':>6} {'PnL($)':>9}")
print("  " + "─"*73)

candidates = []
yes_res = res[res["p_market"].between(0.05, 0.95)].copy()

for s in range(N_STATES):
    sub = yes_res[yes_res["ps_state"]==s]
    if len(sub) < 15:
        print(f"  St{s} {state_info[s]['desc'][:22]:<22} {len(sub):>5}  (too few)")
        continue
    wr   = sub["resolved_yes"].mean()
    be   = sub["p_market"].mean()
    edge = wr - be
    n    = len(sub)
    pnl  = kelly_pnl(sub)
    z, p = mcpt(sub["resolved_yes"].values, sub["p_market"].values)
    desc = state_info[s]["desc"][:22]
    flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and n >= 25 else ""
    print(f"  St{s} {desc:<22} {n:>5} {wr:>6.1%} {be:>6.1%} {edge:>+7.1%} "
          f"{z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
    if p < 0.05 and abs(edge) > 0.03 and n >= 25:
        candidates.append({"state": s, "side": "yes", "edge": edge, "wr": wr,
                           "be": be, "n": n, "desc": state_info[s]["desc"],
                           "z": z, "p": p, "pnl": pnl})

# NO side
print(f"\nNO side P&L by state:")
print(f"  {'St':<4} {'Description':<22} {'n':>5} {'WR':>6} {'BE':>6} {'Edge':>7} "
      f"{'MCPT_z':>7} {'p':>6} {'PnL($)':>9}")
print("  " + "─"*73)

no_res = res[res["p_market"].between(0.05, 0.95)].copy()
no_res["resolved_no"] = 1 - no_res["resolved_yes"]
no_res["pm_no"]       = 1 - no_res["p_market"]
no_res2 = no_res[no_res["pm_no"].between(0.05, 0.95)]

for s in range(N_STATES):
    sub = no_res2[no_res2["ps_state"]==s]
    if len(sub) < 15:
        print(f"  St{s} {state_info[s]['desc'][:22]:<22} {len(sub):>5}  (too few)")
        continue
    wr   = sub["resolved_no"].mean()
    be   = sub["pm_no"].mean()
    edge = wr - be
    n    = len(sub)
    pnl  = kelly_pnl(sub, pm_col="pm_no", out_col="resolved_no")
    z, p = mcpt(sub["resolved_no"].values, sub["pm_no"].values)
    desc = state_info[s]["desc"][:22]
    flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and n >= 25 else ""
    print(f"  St{s} {desc:<22} {n:>5} {wr:>6.1%} {be:>6.1%} {edge:>+7.1%} "
          f"{z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
    if p < 0.05 and abs(edge) > 0.03 and n >= 25:
        candidates.append({"state": s, "side": "no", "edge": edge, "wr": wr,
                           "be": be, "n": n, "desc": state_info[s]["desc"],
                           "z": z, "p": p, "pnl": pnl})

# ── Rescue search for YES-edge-negative states ────────────────────────────────
print(f"\n{'─'*65}")
print("Rescue search: YES-negative states (pair with offset_pct, stoch, pm bands)...")

neg_yes_states = [s for s in range(N_STATES)
                  if state_info[s]["n"] >= 25 and
                  yes_res[yes_res["ps_state"]==s]["resolved_yes"].mean() -
                  yes_res[yes_res["ps_state"]==s]["p_market"].mean() < -0.03]

if neg_yes_states:
    for s in neg_yes_states:
        sub_base = yes_res[yes_res["ps_state"]==s].copy()
        sub_base["offset_pct"] = pd.to_numeric(sub_base.get("offset_pct", np.nan), errors="coerce")
        print(f"\n  St{s} ({state_info[s]['desc']}) — base n={len(sub_base)} "
              f"WR={sub_base['resolved_yes'].mean():.1%} BE={sub_base['p_market'].mean():.1%}")

        # pm bands
        for lo, hi in [(0.30,0.50),(0.50,0.70),(0.70,0.90),(0.30,0.70)]:
            sub2 = sub_base[sub_base["p_market"].between(lo,hi)]
            if len(sub2) < 10: continue
            wr2  = sub2["resolved_yes"].mean()
            be2  = sub2["p_market"].mean()
            edge2 = wr2 - be2
            print(f"    pm[{lo:.0%},{hi:.0%}): n={len(sub2):3d} WR={wr2:.1%} BE={be2:.1%} edge={edge2:+.1%}")

        # offset_pct bands
        if sub_base["offset_pct"].notna().sum() > 20:
            for lo, hi, label in [(-0.05,-0.01,"OTM"),(-0.01,0.01,"ATM"),(0.01,0.05,"ITM")]:
                sub2 = sub_base[sub_base["offset_pct"].between(lo,hi)]
                if len(sub2) < 8: continue
                wr2  = sub2["resolved_yes"].mean()
                be2  = sub2["p_market"].mean()
                edge2 = wr2 - be2
                print(f"    offset {label} [{lo:+.1%},{hi:+.1%}): n={len(sub2):3d} "
                      f"WR={wr2:.1%} BE={be2:.1%} edge={edge2:+.1%}")
else:
    print("  No strongly-negative YES states with n>=25.")

# ── Walk-forward validation ───────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Walk-forward validation for {len(candidates)} candidate(s):")

if candidates and len(ts) > 100:
    mid_idx = ts["logged_at"].median()
    print(f"  Split at: {mid_idx.date()}  "
          f"(train n={len(ts[ts['logged_at']<=mid_idx])}, "
          f"test n={len(ts[ts['logged_at']>mid_idx])})")

    ts_train = ts[ts["logged_at"] <= mid_idx].copy()
    ts_test  = ts[ts["logged_at"] >  mid_idx].copy()

    sess_tr = ts_train.groupby("session").size()
    sess_te = ts_test.groupby("session").size()
    lens_tr = [l for l in sess_tr.values if l >= MIN_SESSION]
    lens_te = [l for l in sess_te.values if l >= MIN_SESSION]

    # Retrain on first half
    X_tr = scaler.transform(ts_train[FEATURES].values)
    X_te = scaler.transform(ts_test[FEATURES].values)

    best_wf, best_wf_ll = None, -np.inf
    for seed in range(20):
        try:
            m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=500, random_state=seed, tol=1e-5)
            m.fit(X_tr, lengths=lens_tr)
            ll = m.score(X_tr, lengths=lens_tr)
            if ll > best_wf_ll:
                best_wf_ll, best_wf = ll, m
        except Exception:
            pass

    if best_wf is not None:
        try:
            sts_tr = best_wf.predict(X_tr, lengths=lens_tr)
            sts_te = best_wf.predict(X_te, lengths=lens_te)
        except Exception as e:
            print(f"  WF decode error: {e}")
            best_wf = None

    if best_wf is not None:
        ts_train = ts_train.copy(); ts_train["wf_state"] = sts_tr
        ts_test  = ts_test.copy();  ts_test["wf_state"]  = sts_te

        # Compute WF centroids for state-alignment
        tr_centroids = {}
        for s in range(N_STATES):
            m = ts_train[ts_train["wf_state"]==s]
            if len(m): tr_centroids[s] = m[FEATURES].mean().values

        for cand in candidates:
            s_full = cand["state"]
            side   = cand["side"]
            full_c = np.array([state_info[s_full][k] for k in
                               ["sk15","sk1h","d15","d1h","div"]])

            # Map to nearest WF-train state by centroid
            best_match, best_dist = 0, np.inf
            for s_tr, c_tr in tr_centroids.items():
                # scale both by scaler for fair distance
                dist = np.linalg.norm(
                    scaler.transform(full_c.reshape(1,-1)) -
                    scaler.transform(c_tr.reshape(1,-1)))
                if dist < best_dist:
                    best_dist, best_match = dist, s_tr

            ts_wf_all = pd.concat([ts_train, ts_test]).sort_values("logged_at")
            lkup = ts_wf_all[["logged_at","wf_state"]].sort_values("logged_at")

            print(f"\n  Cand: St{s_full} ({cand['desc']}) side={side.upper()}")
            print(f"    Full-data: n={cand['n']} WR={cand['wr']:.1%} "
                  f"edge={cand['edge']:+.1%} z={cand['z']:+.2f} p={cand['p']:.4f}")

            for split_label, arc_half in [
                ("TRAIN", res[res["logged_at"] <= mid_idx]),
                ("TEST",  res[res["logged_at"] >  mid_idx]),
            ]:
                arc_h = arc_half.sort_values("logged_at")
                lkup_h = lkup[lkup["logged_at"].between(
                    arc_h["logged_at"].min(), arc_h["logged_at"].max())]
                arc_wf = pd.merge_asof(arc_h, lkup_h, on="logged_at",
                                       direction="nearest", tolerance=pd.Timedelta("8min"))

                if side == "yes":
                    sub_wf = arc_wf[(arc_wf["wf_state"]==best_match) &
                                    arc_wf["p_market"].between(0.05,0.95) &
                                    arc_wf["resolved_yes"].notna()]
                    wr_h = sub_wf["resolved_yes"].mean() if len(sub_wf) else np.nan
                    be_h = sub_wf["p_market"].mean() if len(sub_wf) else np.nan
                    pnl_h = kelly_pnl(sub_wf) if len(sub_wf) else 0.0
                else:
                    sub_wf = arc_wf[(arc_wf["wf_state"]==best_match) &
                                    arc_wf["p_market"].between(0.05,0.95) &
                                    arc_wf["resolved_yes"].notna()].copy()
                    sub_wf["resolved_no"] = 1 - sub_wf["resolved_yes"]
                    sub_wf["pm_no"]       = 1 - sub_wf["p_market"]
                    sub_wf = sub_wf[sub_wf["pm_no"].between(0.05,0.95)]
                    wr_h = sub_wf["resolved_no"].mean() if len(sub_wf) else np.nan
                    be_h = sub_wf["pm_no"].mean() if len(sub_wf) else np.nan
                    pnl_h = (kelly_pnl(sub_wf, pm_col="pm_no", out_col="resolved_no")
                             if len(sub_wf) else 0.0)

                edge_h = wr_h - be_h if not math.isnan(wr_h) else np.nan
                n_h = len(sub_wf)
                ok = ""
                if not math.isnan(edge_h):
                    ok = "PASS" if edge_h < -0.02 else "FAIL"
                print(f"    WF_{split_label}: n={n_h:3d} WR={wr_h:.1%} BE={be_h:.1%} "
                      f"edge={edge_h:+.1%} PnL=${pnl_h:+,.0f}  [{ok}]")
else:
    print("  No candidates or insufficient data for WF.")

# ── Gate / shadow recommendation ─────────────────────────────────────────────
print(f"\n{'═'*65}")
print("SUMMARY — Phase-Space Trajectory HMM")
print(f"Optimal states: {N_STATES}  |  15m obs: {len(ts)}  |  BIC: {best_bic:,.0f}")
print()
for s in range(N_STATES):
    info = state_info[s]
    dur  = 1.0 / (1 - T[s,s]) if T[s,s] < 1 else 999.0
    print(f"  St{s}: {info['desc']:<25}  "
          f"sk15={info['sk15']:.1f}  sk1h={info['sk1h']:.1f}  "
          f"Δ15={info['d15']:+.1f}  Δ1h={info['d1h']:+.1f}  "
          f"~{dur*15:.0f}min avg  ({info['n']/len(ts)*100:.1f}%)")

print()
if not candidates:
    print("Gate candidates: NONE passed MCPT threshold (p<0.05, n>=25, |edge|>3%)")
    print("Recommendation: SHADOW LOG — wire ps_state into CSV, audit after 60+ days")
else:
    print(f"Gate candidates: {len(candidates)}")
    for c in candidates:
        print(f"  St{c['state']} {c['side'].upper()}: n={c['n']} WR={c['wr']:.1%} "
              f"edge={c['edge']:+.1%} z={c['z']:+.2f} p={c['p']:.4f} PnL=${c['pnl']:+,.0f}")

# ── Save model ────────────────────────────────────────────────────────────────
pkg = dict(
    model=model,
    scaler=scaler,
    n_states=N_STATES,
    features=FEATURES,
    state_descriptions={s: state_info[s]["desc"] for s in range(N_STATES)},
    state_centroids={s: {k: state_info[s][k] for k in ["sk15","sk1h","d15","d1h","div"]}
                     for s in range(N_STATES) if state_info[s]["n"] > 0},
    trained_on=str(ts["logged_at"].max().date()),
)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nModel saved → {MODEL_PATH}")
