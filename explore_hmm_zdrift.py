"""
Z-Drift HMM — explore and train.

Features: (pm_drift_5m, rvol_1h) on 5-min buckets.
pm_drift captures directional drift regime; rvol captures vol context.
lag-1 autocorr = 0.718 → states persist ~15-20 min.

Hypothesis: sustained negative drift → YES bets underperform;
            sustained positive drift → YES bets outperform.
            The HMM captures regime PERSISTENCE not just current reading.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

N_STATES_RANGE = [3, 4, 5, 6]
GAP_THRESHOLD  = pd.Timedelta("2h")
MIN_SESSION    = 5
MODEL_PATH     = Path("models/hmm_zdrift_btc.pkl")


def mcpt(wr_arr, be_arr, n=2000, seed=42):
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    null = np.array([(rng.random(len(be_arr)) < be_arr).astype(float).mean() - be_arr.mean()
                     for _ in range(n)])
    p = np.mean(null <= obs) if obs < 0 else np.mean(null >= obs)
    return (obs - null.mean()) / (null.std() + 1e-9), p


def kelly_pnl(sub):
    p = sub["p_market"].clip(0.01, 0.99); w = sub["resolved_yes"]
    f = (p - (1-p)).clip(0, 0.25)
    return (f * ((1-p)*w - p*(1-w))).sum()


# ── Load and build 5-min time series ─────────────────────────────────────────
print("Loading archive ...")
arc = pd.read_csv("results/btc_scan_archive.csv", low_memory=False,
                  usecols=["logged_at","contract_ticker","pm_drift_5m","rvol_1h"])
arc["logged_at"]   = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc["pm_drift_5m"] = pd.to_numeric(arc["pm_drift_5m"], errors="coerce")
arc["rvol_1h"]     = pd.to_numeric(arc["rvol_1h"],     errors="coerce")
arc = arc.dropna(subset=["logged_at","pm_drift_5m"]).sort_values("logged_at")

arc["bucket"] = arc["logged_at"].dt.floor("5min")
ts = arc.groupby("bucket").agg(
    pm_drift=("pm_drift_5m", "mean"),
    rvol    =("rvol_1h",     "mean"),
).reset_index()

# Fill rvol gaps with forward-fill (rvol changes slowly)
ts["rvol"] = ts["rvol"].fillna(method="ffill").fillna(1.0)

ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sess = ts.groupby("session").size()
ts = ts[ts["session"].isin(sess[sess >= MIN_SESSION].index)].reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sessions = ts.groupby("session").size()

print(f"5-min buckets: {len(ts):,}  sessions: {len(sessions)}")
print(f"Date range: {ts.bucket.min().date()} → {ts.bucket.max().date()}")

FEATURES = ["pm_drift", "rvol"]
print(f"\nFeature stats:")
for col in FEATURES:
    v = ts[col]
    print(f"  {col:>10}: mean={v.mean():+.4f}  std={v.std():.4f}  "
          f"p5={v.quantile(0.05):.4f}  p95={v.quantile(0.95):.4f}  "
          f"autocorr_lag1={v.autocorr(1):.3f}")

X_raw   = ts[FEATURES].values
scaler  = StandardScaler()
X       = scaler.fit_transform(X_raw)
lengths = sessions.values.tolist()

# ── BIC model selection ───────────────────────────────────────────────────────
print(f"\nBIC model selection (30 restarts per n):")
best_bic, best_n, best_model = np.inf, None, None

for n in N_STATES_RANGE:
    scores = []
    for seed in range(30):
        try:
            m = GaussianHMM(n_components=n, covariance_type="diag",
                            n_iter=500, random_state=seed, tol=1e-5)
            m.fit(X, lengths=lengths)
            ll = m.score(X, lengths=lengths)
            k  = n*(n-1) + 2*n*len(FEATURES)
            bic = -2*ll + k*np.log(len(X))
            scores.append((bic, ll, m))
        except Exception:
            pass
    if not scores: continue
    scores.sort(key=lambda x: x[0])
    bic, ll, m = scores[0]
    marker = " ◄ BEST" if bic < best_bic else ""
    print(f"  n={n}: BIC={bic:,.1f}  LL={ll:,.1f}{marker}")
    if bic < best_bic:
        best_bic, best_n, best_model = bic, n, m

print(f"\nSelected n={best_n} states")
model    = best_model
N_STATES = best_n

# ── Decode states ──────────────────────────────────────────────────────────────
states = model.predict(X, lengths=lengths)
ts["zd_state"] = states
T = model.transmat_


def describe_state(drift, rvol):
    high_rvol = rvol > 1.3
    low_rvol  = rvol < 0.8
    if   drift >  0.15: label = "strong_pos_drift"
    elif drift >  0.04: label = "mild_pos_drift"
    elif drift < -0.15: label = "strong_neg_drift"
    elif drift < -0.04: label = "mild_neg_drift"
    else:               label = "flat_drift"
    if high_rvol: label += "+highvol"
    elif low_rvol: label += "+lowvol"
    return label


print(f"\n{'─'*65}")
print("State centroids:")

state_info = {}
for s in range(N_STATES):
    mask = ts["zd_state"] == s
    n_obs = mask.sum()
    if n_obs == 0:
        state_info[s] = {"n":0, "drift":0, "rvol":1, "desc":"empty"}
        continue
    drift_c = ts.loc[mask, "pm_drift"].mean()
    rvol_c  = ts.loc[mask, "rvol"].mean()
    dur     = 1/(1 - T[s,s]) if T[s,s] < 1 else 999
    desc    = describe_state(drift_c, rvol_c)
    state_info[s] = {"n":n_obs, "drift":drift_c, "rvol":rvol_c, "desc":desc}
    print(f"  St{s}: n={n_obs:5,} ({n_obs/len(ts)*100:.1f}%)  "
          f"drift={drift_c:+.4f}  rvol={rvol_c:.3f}  "
          f"~{dur*5:.0f}min  → {desc}")

print(f"\nTransition matrix:")
print("         " + "".join(f" →St{j}" for j in range(N_STATES)))
for i in range(N_STATES):
    dur = 1/(1-T[i,i]) if T[i,i] < 1 else 999
    print(f"  St{i} ({state_info[i]['desc'][:18]:<18})"
          + "".join(f" {T[i,j]:.3f}" for j in range(N_STATES))
          + f"  ~{dur*5:.0f}min")

# ── P&L by state ──────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("Loading archive for P&L evaluation ...")
res = pd.read_csv("results/btc_scan_archive.csv", low_memory=False,
                  usecols=["logged_at","contract_ticker","p_market","resolved_yes"])
res["logged_at"]    = pd.to_datetime(res["logged_at"], format="mixed", utc=True, errors="coerce")
res["resolved_yes"] = pd.to_numeric(res["resolved_yes"], errors="coerce")
res["p_market"]     = pd.to_numeric(res["p_market"],     errors="coerce")
res = res.dropna(subset=["logged_at","resolved_yes","p_market"]).sort_values("logged_at")
res = res[res["p_market"].between(0.05, 0.95)].reset_index(drop=True)

# 1-bucket lag (5 min)
lkup = ts[["bucket","zd_state"]].copy()
lkup["bucket"] += pd.Timedelta("5min")
lkup = lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

res = pd.merge_asof(res, lkup, on="logged_at",
                    direction="nearest", tolerance=pd.Timedelta("3min"))
res = res.dropna(subset=["zd_state"]).copy()
res["zd_state"] = res["zd_state"].astype(int)
print(f"Resolved contracts with state tag: {len(res):,}")

print(f"\nYES side:")
print(f"  {'St':<4} {'Desc':<24} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>9}")
print("  " + "─"*68)
yes_candidates = []
for s in range(N_STATES):
    sub = res[res["zd_state"] == s]
    if len(sub) < 20:
        print(f"  St{s} {state_info[s]['desc']:<24} {len(sub):>6}  (too few)"); continue
    wr = sub["resolved_yes"].mean(); be = sub["p_market"].mean(); edge = wr - be
    pnl = kelly_pnl(sub); z, p = mcpt(sub["resolved_yes"].values, sub["p_market"].values)
    flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30 else ""
    print(f"  St{s} {state_info[s]['desc']:<24} {len(sub):>6} {wr:>6.1%} {be:>6.1%} "
          f"{edge:>+7.1%} {z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
    if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30:
        yes_candidates.append({"state":s,"side":"yes","n":len(sub),"edge":edge,
                                "z":z,"p":p,"pnl":pnl,"desc":state_info[s]["desc"]})

print(f"\nNO side:")
print(f"  {'St':<4} {'Desc':<24} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>9}")
print("  " + "─"*68)
no_candidates = []
for s in range(N_STATES):
    sub = res[res["zd_state"] == s].copy()
    sub["resolved_yes"] = 1-sub["resolved_yes"]; sub["p_market"] = 1-sub["p_market"]
    sub = sub[sub["p_market"].between(0.05,0.95)]
    if len(sub) < 20:
        print(f"  St{s} {state_info[s]['desc']:<24} {len(sub):>6}  (too few)"); continue
    wr = sub["resolved_yes"].mean(); be = sub["p_market"].mean(); edge = wr - be
    pnl = kelly_pnl(sub); z, p = mcpt(sub["resolved_yes"].values, sub["p_market"].values)
    flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30 else ""
    print(f"  St{s} {state_info[s]['desc']:<24} {len(sub):>6} {wr:>6.1%} {be:>6.1%} "
          f"{edge:>+7.1%} {z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
    if p < 0.05 and abs(edge) > 0.03 and len(sub) >= 30:
        no_candidates.append({"state":s,"side":"no","n":len(sub),"edge":edge,
                               "z":z,"p":p,"pnl":pnl,"desc":state_info[s]["desc"]})

all_candidates = yes_candidates + no_candidates

# ── Walk-forward ──────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Walk-forward validation for {len(all_candidates)} candidates:")

mid_dt = ts["bucket"].median()
print(f"Split: {mid_dt.date()}")

ts_tr = ts[ts["bucket"] <= mid_dt]
ts_te = ts[ts["bucket"] >  mid_dt]
X_tr  = scaler.transform(ts_tr[FEATURES].values)
X_te  = scaler.transform(ts_te[FEATURES].values)
lens_tr = [l for l in ts_tr.groupby("session").size().values if l >= MIN_SESSION]
lens_te = [l for l in ts_te.groupby("session").size().values if l >= MIN_SESSION]

best_ll, best_wf = -np.inf, None
for seed in range(20):
    try:
        m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=500, random_state=seed, tol=1e-5)
        m.fit(X_tr, lengths=lens_tr)
        ll = m.score(X_tr, lengths=lens_tr)
        if ll > best_ll: best_ll, best_wf = ll, m
    except Exception: pass

if best_wf is None:
    print("  WF retrain failed.")
else:
    st_tr = best_wf.predict(X_tr, lengths=lens_tr)
    st_te = best_wf.predict(X_te, lengths=lens_te)
    ts_tr = ts_tr.copy(); ts_tr["wf_st"] = st_tr
    ts_te = ts_te.copy(); ts_te["wf_st"] = st_te

    # Centroid alignment
    tr_cents = {s: ts_tr[ts_tr["wf_st"]==s][FEATURES].mean().values
                for s in range(N_STATES) if (ts_tr["wf_st"]==s).sum() > 0}
    full_cents = {s: ts[ts["zd_state"]==s][FEATURES].mean().values
                  for s in range(N_STATES) if (ts["zd_state"]==s).sum() > 0}
    align = {}
    for sf, cf in full_cents.items():
        best_s, best_d = 0, np.inf
        for st, ct in tr_cents.items():
            d = np.linalg.norm(cf - ct)
            if d < best_d: best_d, best_s = d, st
        align[sf] = best_s

    def build_lkup(ts_half, col):
        df = ts_half[["bucket", col]].copy()
        df["bucket"] += pd.Timedelta("5min")
        return df.sort_values("bucket").rename(columns={"bucket":"logged_at"})

    lkup_tr = build_lkup(ts_tr, "wf_st")
    lkup_te = build_lkup(ts_te, "wf_st")
    res_tr = pd.merge_asof(res[res["logged_at"]<=mid_dt].sort_values("logged_at"),
                            lkup_tr, on="logged_at", direction="nearest", tolerance=pd.Timedelta("3min"))
    res_te = pd.merge_asof(res[res["logged_at"]>mid_dt].sort_values("logged_at"),
                            lkup_te, on="logged_at", direction="nearest", tolerance=pd.Timedelta("3min"))
    res_tr = res_tr.dropna(subset=["wf_st"]).copy(); res_tr["wf_st"] = res_tr["wf_st"].astype(int)
    res_te = res_te.dropna(subset=["wf_st"]).copy(); res_te["wf_st"] = res_te["wf_st"].astype(int)

    for cand in all_candidates:
        sf = cand["state"]; side = cand["side"]
        s_al = align.get(sf, sf)

        def get_sub(df, s, side):
            sub = df[df["wf_st"]==s].copy()
            if side == "no":
                sub["resolved_yes"] = 1-sub["resolved_yes"]
                sub["p_market"]     = 1-sub["p_market"]
                sub = sub[sub["p_market"].between(0.05,0.95)]
            return sub

        sub_tr = get_sub(res_tr, s_al, side)
        sub_te = get_sub(res_te, s_al, side)
        if len(sub_tr) < 5 or len(sub_te) < 5:
            print(f"  St{sf}/{side.upper()} ({cand['desc'][:20]}): insufficient WF data"); continue

        e_tr = sub_tr["resolved_yes"].mean() - sub_tr["p_market"].mean()
        e_te = sub_te["resolved_yes"].mean() - sub_te["p_market"].mean()
        wf   = "PASS" if e_tr < -0.02 and e_te < -0.02 else "FAIL"
        z_te, p_te = mcpt(sub_te["resolved_yes"].values, sub_te["p_market"].values)
        pnl_te = kelly_pnl(sub_te)
        print(f"  St{sf}/{side.upper()} ({cand['desc'][:20]:<20}): "
              f"train n={len(sub_tr):4d} e={e_tr:+.1%}  "
              f"test n={len(sub_te):4d} e={e_te:+.1%} z={z_te:+.2f} p={p_te:.4f} "
              f"PnL=${pnl_te:+,.0f}  [{wf}]")

# ── Paper trades cross-check ──────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("Paper trades cross-check (BTC):")
pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt["logged_at"]    = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
pt["resolved_yes"] = pd.to_numeric(pt["resolved_yes"], errors="coerce")
pt["p_market"]     = pd.to_numeric(pt["p_market"], errors="coerce")
pt["would_pnl"]    = pd.to_numeric(pt["would_pnl"], errors="coerce")
pt = pt[pt["contract_ticker"].str.contains("KXBTCD", na=False) &
        (pt["decision"]=="trade") & pt["resolved_yes"].notna()].dropna(subset=["logged_at"]).sort_values("logged_at")

lkup_live = ts[["bucket","zd_state"]].copy()
lkup_live["bucket"] += pd.Timedelta("5min")
lkup_live = lkup_live.sort_values("bucket").rename(columns={"bucket":"logged_at"})
pt_m = pd.merge_asof(pt, lkup_live, on="logged_at",
                     direction="nearest", tolerance=pd.Timedelta("3min"))
n_tag = pt_m["zd_state"].notna().sum()
print(f"BTC paper trades: {len(pt_m)}  tagged: {n_tag} ({n_tag/len(pt_m)*100:.0f}%)")

for s in range(N_STATES):
    for side in ["yes","no"]:
        sub = pt_m[(pt_m["zd_state"]==s)&(pt_m["side"]==side)&pt_m["zd_state"].notna()]
        if len(sub) < 3: continue
        wr  = sub["resolved_yes"].mean() if side=="yes" else (1-sub["resolved_yes"]).mean()
        be  = sub["p_market"].mean()     if side=="yes" else (1-sub["p_market"]).mean()
        pnl = sub["would_pnl"].sum(); edge = wr-be
        flag = " ◄" if abs(edge) > 0.07 and len(sub) >= 5 else ""
        print(f"  St{s}/{side.upper()} ({state_info[s]['desc'][:22]}): "
              f"n={len(sub):3d} WR={wr:.1%} BE={be:.1%} edge={edge:+.1%} PnL=${pnl:+,.0f}{flag}")

# ── Save ──────────────────────────────────────────────────────────────────────
pkg = dict(
    model=model, scaler=scaler, n_states=N_STATES, features=FEATURES,
    state_descriptions={s: state_info[s]["desc"] for s in range(N_STATES)},
    state_centroids={s: {"drift": state_info[s]["drift"], "rvol": state_info[s]["rvol"]}
                     for s in range(N_STATES) if state_info[s]["n"] > 0},
    trained_on=str(ts["bucket"].max().date()),
)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nModel saved → {MODEL_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"SUMMARY — Z-Drift HMM  n={N_STATES}  BIC={best_bic:,.0f}")
for s in range(N_STATES):
    i = state_info[s]; dur = 1/(1-T[s,s]) if T[s,s]<1 else 999
    print(f"  St{s} {i['desc']:<26} drift={i['drift']:+.4f} rvol={i['rvol']:.3f}  "
          f"~{dur*5:.0f}min ({i['n']/len(ts)*100:.1f}%)")
