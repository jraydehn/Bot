"""
Gate-Firing Density HMM — explore and train.

Concept: Track HOW MANY distinct gates fire per scan cycle over time.
Observation: (n_yes_gates, n_no_gates, yes_dominance) per 15-min window.
States: "offensive" (few blocks), "defensive YES" (many YES-gates firing),
        "defensive NO" (many NO-gates firing), "hostile" (many of both).

Key use: When in a sustained defensive/hostile regime, Kelly dampen or shadow-flag.
Data:    results/blocked_trades.csv (BTC, gate_name, logged_at)
Outcome: results/paper_trades.csv  (BTC, decision=trade, resolved_yes)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

N_STATES_RANGE = [2, 3, 4, 5]
GAP_THRESHOLD  = pd.Timedelta("2h")
MIN_SESSION    = 4
BUCKET_SIZE    = "15min"
MODEL_PATH     = Path("models/hmm_gate_density_btc.pkl")

# ── Gate classification ────────────────────────────────────────────────────────
YES_GATES = {
    "smc_gate","btc_otm_yes_hardblock","swing_high_gate","stoch_oversold_yes_gate",
    "near_itm_gate","adx_mid_ct_neg_yes_gate","rvol_gate","neutral_ema_g3",
    "rsi_oversold_yes_gate","hour_yes_gate","cg_oi_stable_yes_gate","btc_vol_gate",
    "btc_adx_gate","liq_cascade_gate","rev_div_gate","btc_exhaustion_gate",
    "btc_deepno_neutral_gate","btc_garch_highvol_yes_gate","btc_adx5_gate",
    "bear_drift","btc_otmlow_gate","btc_struct_gate","ema_stack3_gate",
    "near_atm_ema_gate","btc_otm_neutral_gate","itm_yes_sh_gate","btc_falling_knife_gate",
    "btc_body_bp_gate","neutral_ema_g2","btc_ema0_stretch2_gate","btc_ema0_itm_gate",
    "btc_gbdt_gate","strong_trend_nearatm_gate","g1_mr_falling_knife",
    "btc_cal_err_yes_gate","btc_highpm_yes_gate","itm_yes_rw_gate",
    "hmm_r1_otm_early_yes","hmm_r1_otm_mid_yes","eth_hmm_r1_otm_yes",
    "ps_st0_yes","ps_st1_yes","ps_st2_yes","ps_st5_yes",
}

NO_GATES = {
    "no_pm_floor","btc_highpm_no_gate","btc_stoch_no_gate","cg_oi_stable_no_gate",
    "btc_no_z_gate","btc_no_kalman_resid_gate","btc_no_smc_demand_gate",
    "itm_no_neutral_stoch_gate","btc_nopup_gate","hmm_mtf_st3","btc_no_wrongdir_gate",
    "btc_liq_squeeze_gate","semi_markov_r1_deep_highrvol","semi_markov_r1_mid_neutral_funding",
    "btc_no_highpm_bearema_gate","btc_no_vol_gate","semi_markov_r1_early_no",
    "stoch_overbought_no_gate","bp_1h_no_gate","rsi_overbought_no_gate","no_bp1h_chg1h",
    "g3b_mr_uptrend_no_gate","bull_rally_no_gate","btc_highpm_no_gate",
    "hmm_ms_btc_st6","btc_of_hmm_st2","btc_no_ms_gate","of_hmm_btc_no",
    "hmm_mtf_st3_gate","markov_sideways_gate",
}

def classify_gate(name):
    if name in YES_GATES:
        return "yes"
    if name in NO_GATES:
        return "no"
    # Heuristic from name
    n = name.lower()
    if "_yes" in n or "yes_" in n:
        return "yes"
    if "_no" in n or "no_" in n:
        return "no"
    return "neutral"

# ── Load blocked trades ────────────────────────────────────────────────────────
print("Loading blocked_trades.csv ...")
bt = pd.read_csv("results/blocked_trades.csv", low_memory=False)
bt["logged_at"] = pd.to_datetime(bt["logged_at"], format="mixed", utc=True, errors="coerce")
bt = bt[(bt["asset"]=="BTC") & bt["logged_at"].notna()].sort_values("logged_at").reset_index(drop=True)
bt["gate_class"] = bt["gate_name"].apply(classify_gate)
bt["bucket"] = bt["logged_at"].dt.floor(BUCKET_SIZE)

print(f"BTC blocks: {len(bt):,}  ({bt['logged_at'].min().date()} → {bt['logged_at'].max().date()})")
print(f"Gate class distribution: {bt['gate_class'].value_counts().to_dict()}")

# ── Build density time series ──────────────────────────────────────────────────
# Per bucket: count distinct gate names by class
def count_unique(series):
    return series.nunique()

density = (bt.groupby(["bucket","gate_class"])["gate_name"]
             .nunique()
             .unstack(fill_value=0)
             .reset_index())
density.columns.name = None

# Ensure all columns present
for col in ["yes","no","neutral"]:
    if col not in density.columns:
        density[col] = 0

density["total"] = density["yes"] + density["no"] + density["neutral"]
density["yes_dominance"] = (density["yes"] - density["no"]) / density["total"].clip(lower=1)

# Fill to continuous 15-minute grid (zero density = no blocks fired)
full_grid = pd.date_range(
    start=density["bucket"].min(),
    end=density["bucket"].max(),
    freq=BUCKET_SIZE, tz="UTC"
)
density = (density.set_index("bucket")
                  .reindex(full_grid, fill_value=0)
                  .reset_index()
                  .rename(columns={"index":"bucket"}))
density["yes_dominance"] = (density["yes"] - density["no"]) / density["total"].clip(lower=1)

print(f"\n15-min density buckets: {len(density):,} (including zero-activity)")

# Session boundaries
density["gap"] = density["bucket"].diff() > GAP_THRESHOLD
density["session"] = density["gap"].cumsum()
sessions_raw = density.groupby("session").size()

# Drop short sessions
valid_sess = sessions_raw[sessions_raw >= MIN_SESSION].index
density = density[density["session"].isin(valid_sess)].reset_index(drop=True)
density["gap"] = density["bucket"].diff() > GAP_THRESHOLD
density["session"] = density["gap"].cumsum()
sessions = density.groupby("session").size()

print(f"Sessions (gap>{GAP_THRESHOLD}): {len(sessions)}, lengths: {sessions.values.tolist()}")

# Feature inspection
FEATURES = ["yes","no","yes_dominance"]
print(f"\nFeature stats:")
for col in FEATURES:
    v = density[col]
    print(f"  {col:>15}: mean={v.mean():.3f}  std={v.std():.3f}  "
          f"p95={v.quantile(0.95):.2f}  max={v.max():.2f}")

X_raw = density[FEATURES].values

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
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
            k = n*(n-1) + 2*n*len(FEATURES)
            bic = -2*ll + k*np.log(len(X))
            scores.append((bic, ll, m))
        except Exception:
            pass
    if not scores:
        print(f"  n={n}: no converging runs"); continue
    scores.sort(key=lambda x: x[0])
    bic, ll, m = scores[0]
    marker = " ◄ BEST" if bic < best_bic else ""
    print(f"  n={n}: BIC={bic:,.1f}  LL={ll:,.1f}{marker}")
    if bic < best_bic:
        best_bic, best_n, best_model = bic, n, m

print(f"\nSelected n={best_n} states")
model = best_model
N_STATES = best_n

# ── Decode states ──────────────────────────────────────────────────────────────
states = model.predict(X, lengths=lengths)
density["gd_state"] = states

T = model.transmat_

print(f"\n{'─'*65}")
print("State centroids:")

def describe_density_state(yes, no, dom):
    total = yes + no
    if total < 0.5:
        return "silent"
    if yes > 2 and dom > 0.3:
        return "YES_defensive"
    if no > 2 and dom < -0.3:
        return "NO_defensive"
    if total > 4:
        return "hostile(both)"
    if yes <= 1 and no <= 1:
        return "low_activity"
    return "mixed_activity"

state_info = {}
for s in range(N_STATES):
    mask = density["gd_state"] == s
    n_obs = mask.sum()
    yes_m = density.loc[mask,"yes"].mean()
    no_m  = density.loc[mask,"no"].mean()
    dom_m = density.loc[mask,"yes_dominance"].mean()
    tot_m = density.loc[mask,"total"].mean()
    desc  = describe_density_state(yes_m, no_m, dom_m)
    dur   = 1.0 / (1 - T[s,s]) if T[s,s] < 1 else 999.0
    state_info[s] = {"n": n_obs, "yes": yes_m, "no": no_m,
                      "dom": dom_m, "tot": tot_m, "desc": desc}
    print(f"  St{s}: n={n_obs:5,} ({n_obs/len(density)*100:.1f}%)  "
          f"yes={yes_m:.2f}  no={no_m:.2f}  dom={dom_m:+.2f}  tot={tot_m:.2f}  "
          f"~{dur*15:.0f}min  → {desc}")

print(f"\nTransition matrix:")
print("         " + "".join(f" →St{j}" for j in range(N_STATES)))
for i in range(N_STATES):
    dur = 1.0/(1-T[i,i]) if T[i,i]<1 else 999.0
    print(f"  St{i} ({state_info[i]['desc'][:12]:<12})"
          + "".join(f" {T[i,j]:.3f}" for j in range(N_STATES))
          + f"  ~{dur*15:.0f}min")

# ── Join to scan archive for bulk P&L eval ────────────────────────────────────
print(f"\n{'─'*65}")
print("Loading scan archive for P&L evaluation ...")
arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc = arc.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
arc["resolved_yes"] = pd.to_numeric(arc.get("resolved_yes", np.nan), errors="coerce")
arc["p_market"]     = pd.to_numeric(arc.get("p_market", np.nan), errors="coerce")

# Lag density by 1 bucket (avoid lookahead: use density from PREVIOUS 15-min window)
density_lagged = density[["bucket","gd_state"]].copy()
density_lagged["bucket"] = density_lagged["bucket"] + pd.Timedelta(BUCKET_SIZE)
density_lagged = density_lagged.sort_values("bucket")

arc_m = pd.merge_asof(
    arc.sort_values("logged_at"),
    density_lagged.rename(columns={"bucket":"logged_at"}),
    on="logged_at",
    direction="nearest",
    tolerance=pd.Timedelta("8min")
)
n_arc_tagged = arc_m["gd_state"].notna().sum()
print(f"Archive rows: {len(arc_m):,}  tagged: {n_arc_tagged:,} ({n_arc_tagged/len(arc_m)*100:.1f}%)")

res = arc_m.dropna(subset=["resolved_yes","p_market","gd_state"]).copy()
res["gd_state"] = res["gd_state"].astype(int)
print(f"Resolved rows with state: {len(res):,}")


def kelly_pnl(sub, pm_col="p_market", out_col="resolved_yes"):
    pm  = sub[pm_col].clip(0.01, 0.99)
    win = sub[out_col]
    f   = (pm - (1-pm)).clip(0, 0.25)
    return (f * ((1-pm)*win - pm*(1-win))).sum()


def mcpt_bernoulli(wr_arr, be_arr, n_perm=2000, seed=42):
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    null = np.array([(rng.random(len(be_arr)) < be_arr).astype(float).mean() - be_arr.mean()
                     for _ in range(n_perm)])
    p = np.mean(null <= obs) if obs < 0 else np.mean(null >= obs)
    z = (obs - null.mean()) / (null.std() + 1e-9)
    return float(z), float(p)


print(f"\n{'─'*65}")
print("P&L by gd_state — YES side (lagged density):")
print(f"  {'St':<4} {'Desc':<18} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>9}")
print("  " + "─"*68)

candidates = []

for side_lbl in ["yes","no"]:
    sub_base = res[res["p_market"].between(0.05, 0.95)].copy()
    if side_lbl == "no":
        sub_base["resolved_yes"] = 1 - sub_base["resolved_yes"]
        sub_base["p_market"]     = 1 - sub_base["p_market"]
        sub_base = sub_base[sub_base["p_market"].between(0.05, 0.95)]

    if side_lbl == "no":
        print(f"\nNO side:")
        print(f"  {'St':<4} {'Desc':<18} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>9}")
        print("  " + "─"*68)

    for s in range(N_STATES):
        sub = sub_base[sub_base["gd_state"]==s]
        if len(sub) < 20:
            print(f"  St{s} {state_info[s]['desc'][:18]:<18} {len(sub):>6}  (too few)")
            continue
        wr   = sub["resolved_yes"].mean()
        be   = sub["p_market"].mean()
        edge = wr - be
        n    = len(sub)
        pnl  = kelly_pnl(sub)
        z, p = mcpt_bernoulli(sub["resolved_yes"].values, sub["p_market"].values)
        desc = state_info[s]["desc"][:18]
        flag = " ◄" if p < 0.05 and abs(edge) > 0.03 and n >= 30 else ""
        print(f"  St{s} {desc:<18} {n:>6} {wr:>6.1%} {be:>6.1%} {edge:>+7.1%} "
              f"{z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
        if p < 0.05 and abs(edge) > 0.03 and n >= 30:
            candidates.append({"state": s, "side": side_lbl, "n": n, "edge": edge,
                               "wr": wr, "be": be, "z": z, "p": p, "pnl": pnl,
                               "desc": state_info[s]["desc"]})

# ── Walk-forward validation ────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Walk-forward validation ({len(candidates)} candidates):")

if candidates and len(density) > 200:
    mid_idx = density["bucket"].median()
    print(f"  Split at {mid_idx.date()}")

    density_tr = density[density["bucket"] <= mid_idx]
    density_te = density[density["bucket"] >  mid_idx]

    sess_tr = density_tr.groupby("session").size()
    sess_te = density_te.groupby("session").size()
    lens_tr = [l for l in sess_tr.values if l >= MIN_SESSION]
    lens_te = [l for l in sess_te.values if l >= MIN_SESSION]

    X_tr = scaler.transform(density_tr[FEATURES].values)
    X_te = scaler.transform(density_te[FEATURES].values)

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

    if best_wf:
        try:
            sts_tr = best_wf.predict(X_tr, lengths=lens_tr)
            sts_te = best_wf.predict(X_te, lengths=lens_te)
        except Exception as e:
            print(f"  WF decode error: {e}"); best_wf = None

    if best_wf:
        density_tr = density_tr.copy(); density_tr["wf_state"] = sts_tr
        density_te = density_te.copy(); density_te["wf_state"] = sts_te

        # Compute train centroids for alignment
        tr_centroids = {s: density_tr[density_tr["wf_state"]==s][FEATURES].mean().values
                        for s in range(N_STATES)
                        if (density_tr["wf_state"]==s).sum() > 0}

        for cand in candidates:
            s_full = cand["state"]
            side   = cand["side"]
            full_c = np.array([state_info[s_full][k] for k in ["yes","no","dom"]])

            best_match, best_dist = 0, np.inf
            for s_tr, c_tr in tr_centroids.items():
                dist = np.linalg.norm(
                    scaler.transform(full_c.reshape(1,-1)) -
                    scaler.transform(c_tr.reshape(1,-1)))
                if dist < best_dist:
                    best_dist, best_match = dist, s_tr

            density_wf = pd.concat([density_tr, density_te]).sort_values("bucket")
            lkup = density_wf[["bucket","wf_state"]].rename(columns={"bucket":"logged_at"})
            # Apply lag
            lkup["logged_at"] += pd.Timedelta(BUCKET_SIZE)
            lkup = lkup.sort_values("logged_at")

            print(f"\n  Cand St{s_full}/{side.upper()} ({cand['desc']}): "
                  f"full n={cand['n']} edge={cand['edge']:+.1%} z={cand['z']:+.2f}")

            for split_lbl, arc_half in [
                ("TRAIN", res[res["logged_at"] <= mid_idx]),
                ("TEST",  res[res["logged_at"] >  mid_idx]),
            ]:
                arc_h = arc_half.sort_values("logged_at")
                arc_wf = pd.merge_asof(arc_h, lkup, on="logged_at",
                                       direction="nearest", tolerance=pd.Timedelta("8min"))
                sub_wf = arc_wf[(arc_wf["wf_state"]==best_match) &
                                arc_wf["p_market"].between(0.05,0.95) &
                                arc_wf["resolved_yes"].notna()].copy()

                if side == "no":
                    sub_wf["resolved_yes"] = 1 - sub_wf["resolved_yes"]
                    sub_wf["p_market"]     = 1 - sub_wf["p_market"]
                    sub_wf = sub_wf[sub_wf["p_market"].between(0.05,0.95)]

                if len(sub_wf) < 5:
                    print(f"    WF_{split_lbl}: n={len(sub_wf)} (too few)"); continue

                wr_h   = sub_wf["resolved_yes"].mean()
                be_h   = sub_wf["p_market"].mean()
                edge_h = wr_h - be_h
                pnl_h  = kelly_pnl(sub_wf)
                ok     = "PASS" if edge_h < -0.02 else "FAIL"
                print(f"    WF_{split_lbl}: n={len(sub_wf):3d} WR={wr_h:.1%} BE={be_h:.1%} "
                      f"edge={edge_h:+.1%} PnL=${pnl_h:+,.0f}  [{ok}]")
else:
    print("  No candidates or insufficient data.")

# ── Actual paper-trade cross-check ────────────────────────────────────────────
print(f"\n{'─'*65}")
print("Cross-check on actual BTC paper trades:")

pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt["logged_at"]    = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
pt["resolved_yes"] = pd.to_numeric(pt["resolved_yes"], errors="coerce")
pt["p_market"]     = pd.to_numeric(pt["p_market"], errors="coerce")
pt["would_pnl"]    = pd.to_numeric(pt["would_pnl"], errors="coerce")

btc_pt = pt[
    pt["contract_ticker"].str.contains("KXBTCD", na=False) &
    (pt["decision"]=="trade") &
    pt["resolved_yes"].notna()
].dropna(subset=["logged_at"]).sort_values("logged_at").copy()

# Join lagged density states
density_lagged_full = density[["bucket","gd_state"]].copy()
density_lagged_full["bucket"] += pd.Timedelta(BUCKET_SIZE)
density_lagged_full = density_lagged_full.sort_values("bucket").rename(columns={"bucket":"logged_at"})

btc_m = pd.merge_asof(btc_pt, density_lagged_full, on="logged_at",
                       direction="nearest", tolerance=pd.Timedelta("8min"))
n_tag = btc_m["gd_state"].notna().sum()
print(f"Actual trades tagged: {n_tag}/{len(btc_m)}")

for s in range(N_STATES):
    for side_lbl in ["yes","no"]:
        sub = btc_m[(btc_m["gd_state"]==s) & (btc_m["side"]==side_lbl) & btc_m["gd_state"].notna()]
        if len(sub) < 3: continue
        wr  = sub["resolved_yes"].mean() if side_lbl=="yes" else (1-sub["resolved_yes"]).mean()
        be  = sub["p_market"].mean()     if side_lbl=="yes" else (1-sub["p_market"]).mean()
        pnl = sub["would_pnl"].sum()
        edge = wr - be
        flag = " ◄" if abs(edge) > 0.07 and len(sub) >= 5 else ""
        print(f"  St{s}/{side_lbl.upper()} ({state_info[s]['desc'][:18]}): "
              f"n={len(sub):3d} WR={wr:.1%} BE={be:.1%} edge={edge:+.1%} "
              f"PnL=${pnl:+,.0f}{flag}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("SUMMARY — Gate-Firing Density HMM")
print(f"States: {N_STATES}  |  15m buckets: {len(density):,}  |  BIC: {best_bic:,.0f}")
for s in range(N_STATES):
    info = state_info[s]
    dur  = 1.0/(1-T[s,s]) if T[s,s]<1 else 999.0
    print(f"  St{s}: {info['desc']:<20}  "
          f"yes={info['yes']:.2f}  no={info['no']:.2f}  dom={info['dom']:+.2f}  "
          f"~{dur*15:.0f}min ({info['n']/len(density)*100:.1f}%)")
print()
if not candidates:
    print("Gate candidates: NONE  → SHADOW LOG (wire gd_state, audit after 60+ days)")
else:
    for c in candidates:
        print(f"  St{c['state']} {c['side'].upper()}: n={c['n']} edge={c['edge']:+.1%} "
              f"z={c['z']:+.2f} p={c['p']:.4f} PnL=${c['pnl']:+,.0f}")

# ── Save model ────────────────────────────────────────────────────────────────
pkg = dict(
    model=model,
    scaler=scaler,
    n_states=N_STATES,
    features=FEATURES,
    state_descriptions={s: state_info[s]["desc"] for s in range(N_STATES)},
    state_centroids={s: {k: state_info[s][k] for k in ["yes","no","dom","tot"]}
                     for s in range(N_STATES)},
    bucket_size=BUCKET_SIZE,
    lag=1,
    trained_on=str(density["bucket"].max().date()),
)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nModel saved → {MODEL_PATH}")
