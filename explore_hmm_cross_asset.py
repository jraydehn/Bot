"""
Cross-Asset Coupled HMM — explore and train.

Joint BTC+ETH+SOL hidden state from their combined stoch + spread + rolling correlation.
States: "correlated rally," "correlated sell-off," "BTC leading alts,"
        "alts leading BTC," "decoupled."

Hypothesis: when cross-asset correlation breaks (decoupled state), individual-asset
models become unreliable — good signal for Kelly dampening on ETH/SOL bets.

Data: btc/eth/sol_scan_archive_hmm.parquet — aligned on 5-min buckets.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

N_STATES_RANGE = [3, 4, 5, 6]
GAP_THRESHOLD  = pd.Timedelta("2h")
MIN_SESSION    = 10
CORR_WINDOW    = 12   # 12 × 5min = 1h rolling correlation window
MODEL_PATH     = Path("models/hmm_cross_asset_btc.pkl")


def load_ts(fname, asset):
    df = pd.read_parquet(fname)
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["logged_at","stoch_k"]).sort_values("logged_at")
    ts = (df.drop_duplicates(subset=["logged_at"])
            [["logged_at","stoch_k","ema_stack_bias","chg_5m"]]
            .copy())
    ts = ts.rename(columns={c: f"{asset}_{c}" for c in ["stoch_k","ema_stack_bias","chg_5m"]})
    ts["bucket"] = ts["logged_at"].dt.floor("5min")
    ts = ts.groupby("bucket").first().reset_index()
    return ts


# ── Load and align ─────────────────────────────────────────────────────────────
print("Loading scan archives ...")
btc = load_ts("results/btc_scan_archive_hmm.parquet", "BTC")
eth = load_ts("results/eth_scan_archive_hmm.parquet", "ETH")
sol = load_ts("results/sol_scan_archive_hmm.parquet", "SOL")

aligned = (btc.merge(eth, on="bucket", how="inner")
              .merge(sol, on="bucket", how="inner"))
aligned = aligned.sort_values("bucket").reset_index(drop=True)

for col in ["BTC_chg_5m","ETH_chg_5m","SOL_chg_5m"]:
    aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

# Session structure
aligned["gap"] = aligned["bucket"].diff() > GAP_THRESHOLD
aligned["session"] = aligned["gap"].cumsum()
sessions_raw = aligned.groupby("session").size()
valid_sess = sessions_raw[sessions_raw >= MIN_SESSION].index
aligned = aligned[aligned["session"].isin(valid_sess)].reset_index(drop=True)
aligned["gap"] = aligned["bucket"].diff() > GAP_THRESHOLD
aligned["session"] = aligned["gap"].cumsum()
sessions = aligned.groupby("session").size()

print(f"Aligned 5-min buckets: {len(aligned):,}  "
      f"({aligned['bucket'].min().date()} → {aligned['bucket'].max().date()})")
print(f"Sessions: {len(sessions)}, lengths: {sessions.values.tolist()}")

# ── Feature engineering ────────────────────────────────────────────────────────
# Spread features — how diverged are the stochs?
aligned["be_spread"] = aligned["BTC_stoch_k"] - aligned["ETH_stoch_k"]   # BTC-ETH
aligned["bs_spread"] = aligned["BTC_stoch_k"] - aligned["SOL_stoch_k"]   # BTC-SOL
aligned["es_spread"] = aligned["ETH_stoch_k"] - aligned["SOL_stoch_k"]   # ETH-SOL

# EMA alignment agreement (all three bullish/bearish/mixed?)
aligned["ema_agreement"] = (
    aligned["BTC_ema_stack_bias"].astype(float) +
    aligned["ETH_ema_stack_bias"].astype(float) +
    aligned["SOL_ema_stack_bias"].astype(float)
)   # range: -3 to +3

# Rolling price change correlation: 1-hour window
# Zero out cross-session changes
aligned["BTC_chg_5m"] = aligned["BTC_chg_5m"].where(~aligned["gap"], np.nan)
aligned["ETH_chg_5m"] = aligned["ETH_chg_5m"].where(~aligned["gap"], np.nan)
aligned["SOL_chg_5m"] = aligned["SOL_chg_5m"].where(~aligned["gap"], np.nan)

aligned["be_corr"]  = (aligned["BTC_chg_5m"]
                        .rolling(CORR_WINDOW, min_periods=6)
                        .corr(aligned["ETH_chg_5m"])
                        .fillna(0.0))
aligned["bs_corr"]  = (aligned["BTC_chg_5m"]
                        .rolling(CORR_WINDOW, min_periods=6)
                        .corr(aligned["SOL_chg_5m"])
                        .fillna(0.0))
aligned["mean_corr"] = (aligned["be_corr"] + aligned["bs_corr"]) / 2

# Mean stoch (overall market level)
aligned["mean_stoch"] = (aligned["BTC_stoch_k"] + aligned["ETH_stoch_k"] + aligned["SOL_stoch_k"]) / 3

FEATURES = [
    "mean_stoch",      # overall momentum level
    "be_spread",       # BTC vs ETH divergence
    "bs_spread",       # BTC vs SOL divergence
    "ema_agreement",   # -3 (all bearish) to +3 (all bullish)
    "mean_corr",       # 1h rolling price correlation (0=decoupled, 1=locked)
]

aligned = aligned.dropna(subset=FEATURES).reset_index(drop=True)
# Recompute sessions
aligned["gap"] = aligned["bucket"].diff() > GAP_THRESHOLD
aligned["session"] = aligned["gap"].cumsum()
sessions = aligned.groupby("session").size()
valid_sess2 = sessions[sessions >= MIN_SESSION].index
aligned = aligned[aligned["session"].isin(valid_sess2)].reset_index(drop=True)
aligned["gap"] = aligned["bucket"].diff() > GAP_THRESHOLD
aligned["session"] = aligned["gap"].cumsum()
sessions = aligned.groupby("session").size()

print(f"\nFinal aligned observations: {len(aligned):,}")
print(f"Sessions: {len(sessions)}, lengths: {sessions.values.tolist()}")

print(f"\nFeature stats:")
for col in FEATURES:
    v = aligned[col]
    print(f"  {col:>15}: mean={v.mean():+.2f}  std={v.std():.2f}  "
          f"p5={v.quantile(0.05):.2f}  p95={v.quantile(0.95):.2f}")

X_raw = aligned[FEATURES].values

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
        print(f"  n={n}: no runs"); continue
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
aligned["ca_state"] = states
T = model.transmat_


def describe_ca_state(mean_s, be_sp, bs_sp, ema_ag, mean_r):
    bullish  = mean_s > 60
    bearish  = mean_s < 35
    btc_lead = (be_sp > 15 or bs_sp > 15)   # BTC stoch well above alts
    alt_lead = (be_sp < -15 or bs_sp < -15)  # Alts stoch well above BTC
    decoupled = mean_r < 0.3
    correlated = mean_r > 0.6
    bear_aligned = ema_ag < -1.5
    bull_aligned  = ema_ag > 1.5

    if decoupled:
        return "decoupled"
    if btc_lead and not bearish:
        return "BTC_leads_alts"
    if alt_lead and not bearish:
        return "alts_lead_BTC"
    if bullish and correlated and bull_aligned:
        return "correlated_rally"
    if bearish and correlated and bear_aligned:
        return "correlated_selloff"
    if bullish:
        return "broad_bullish"
    if bearish:
        return "broad_bearish"
    return "neutral_correlated"


print(f"\n{'─'*70}")
print("State centroids:")

state_info = {}
for s in range(N_STATES):
    mask = aligned["ca_state"] == s
    n_obs = mask.sum()
    if n_obs == 0:
        state_info[s] = {"n":0, "mean_s":50, "be_sp":0, "bs_sp":0, "ema_ag":0,
                         "mean_r":0.5, "desc":"empty"}
        continue
    ms  = aligned.loc[mask,"mean_stoch"].mean()
    be  = aligned.loc[mask,"be_spread"].mean()
    bs  = aligned.loc[mask,"bs_spread"].mean()
    ea  = aligned.loc[mask,"ema_agreement"].mean()
    mr  = aligned.loc[mask,"mean_corr"].mean()
    dur = 1.0 / (1 - T[s,s]) if T[s,s] < 1 else 999.0
    desc = describe_ca_state(ms, be, bs, ea, mr)
    state_info[s] = {"n":n_obs,"mean_s":ms,"be_sp":be,"bs_sp":bs,"ema_ag":ea,"mean_r":mr,"desc":desc}
    print(f"  St{s}: n={n_obs:5,} ({n_obs/len(aligned)*100:.1f}%)  "
          f"ms={ms:.1f}  be_sp={be:+.1f}  bs_sp={bs:+.1f}  ema={ea:+.2f}  corr={mr:.2f}  "
          f"~{dur*5:.0f}min  → {desc}")

print(f"\nTransition matrix:")
print("          " + "".join(f" →St{j}" for j in range(N_STATES)))
for i in range(N_STATES):
    dur = 1.0/(1-T[i,i]) if T[i,i]<1 else 999.0
    print(f"  St{i} ({state_info[i]['desc'][:14]:<14})"
          + "".join(f" {T[i,j]:.3f}" for j in range(N_STATES))
          + f"  ~{dur*5:.0f}min")

# ── Join to BTC scan archive for P&L eval ────────────────────────────────────
print(f"\n{'─'*70}")
print("Loading BTC archive for P&L evaluation ...")
arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc = arc.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
arc["resolved_yes"] = pd.to_numeric(arc.get("resolved_yes", np.nan), errors="coerce")
arc["p_market"]     = pd.to_numeric(arc.get("p_market", np.nan), errors="coerce")

# Lagged lookup (1 bucket = 5 min)
ca_lookup = aligned[["bucket","ca_state"]].copy()
ca_lookup["bucket"] += pd.Timedelta("5min")
ca_lookup = ca_lookup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

arc_m = pd.merge_asof(arc, ca_lookup, on="logged_at",
                       direction="nearest", tolerance=pd.Timedelta("3min"))
n_tag = arc_m["ca_state"].notna().sum()
print(f"Archive rows: {len(arc_m):,}  tagged: {n_tag:,} ({n_tag/len(arc_m)*100:.1f}%)")

res = arc_m.dropna(subset=["resolved_yes","p_market","ca_state"]).copy()
res["ca_state"] = res["ca_state"].astype(int)
print(f"Resolved with state: {len(res):,}")


def kelly_pnl(sub, pm_col="p_market", out_col="resolved_yes"):
    pm = sub[pm_col].clip(0.01,0.99); win = sub[out_col]
    f = (pm-(1-pm)).clip(0,0.25)
    return (f*((1-pm)*win - pm*(1-win))).sum()


def mcpt_b(wr_arr, be_arr, n=2000, seed=42):
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    null = np.array([(rng.random(len(be_arr))<be_arr).astype(float).mean()-be_arr.mean()
                     for _ in range(n)])
    p = np.mean(null<=obs) if obs<0 else np.mean(null>=obs)
    z = (obs-null.mean())/(null.std()+1e-9)
    return float(z),float(p)


print(f"\n{'─'*70}")
print("BTC P&L by cross-asset state (lagged 5 min):")

candidates = []
for side_lbl in ["yes","no"]:
    sub_base = res[res["p_market"].between(0.05,0.95)].copy()
    if side_lbl=="no":
        sub_base["resolved_yes"] = 1-sub_base["resolved_yes"]
        sub_base["p_market"]     = 1-sub_base["p_market"]
        sub_base = sub_base[sub_base["p_market"].between(0.05,0.95)]

    print(f"\n{side_lbl.upper()} side:")
    print(f"  {'St':<4} {'Desc':<22} {'n':>6} {'WR':>6} {'BE':>6} {'Edge':>7} {'z':>7} {'p':>6} {'PnL':>9}")
    print("  " + "─"*70)

    for s in range(N_STATES):
        sub = sub_base[sub_base["ca_state"]==s]
        if len(sub)<20:
            print(f"  St{s} {state_info[s]['desc'][:22]:<22} {len(sub):>6}  (too few)"); continue
        wr=sub["resolved_yes"].mean(); be=sub["p_market"].mean(); edge=wr-be
        n_=len(sub); pnl=kelly_pnl(sub); z,p=mcpt_b(sub["resolved_yes"].values,sub["p_market"].values)
        desc=state_info[s]["desc"][:22]
        flag=" ◄" if p<0.05 and abs(edge)>0.03 and n_>=30 else ""
        print(f"  St{s} {desc:<22} {n_:>6} {wr:>6.1%} {be:>6.1%} {edge:>+7.1%} "
              f"{z:>+7.2f} {p:>6.4f} {pnl:>+9,.0f}{flag}")
        if p<0.05 and abs(edge)>0.03 and n_>=30:
            candidates.append({"state":s,"side":side_lbl,"n":n_,"edge":edge,"wr":wr,
                               "be":be,"z":z,"p":p,"pnl":pnl,"desc":state_info[s]["desc"]})

# ── ETH/SOL P&L by state ──────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("ETH + SOL P&L by cross-asset state (the novel use case):")

for asset_lbl, archive_path in [("ETH","results/eth_scan_archive_hmm.parquet"),
                                 ("SOL","results/sol_scan_archive_hmm.parquet")]:
    arc2 = pd.read_parquet(archive_path)
    arc2["logged_at"] = pd.to_datetime(arc2["logged_at"], format="mixed", utc=True, errors="coerce")
    arc2 = arc2.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
    arc2["resolved_yes"] = pd.to_numeric(arc2.get("resolved_yes", np.nan), errors="coerce")
    arc2["p_market"]     = pd.to_numeric(arc2.get("p_market", np.nan), errors="coerce")
    arc2_m = pd.merge_asof(arc2, ca_lookup, on="logged_at",
                            direction="nearest", tolerance=pd.Timedelta("3min"))
    res2 = arc2_m.dropna(subset=["resolved_yes","p_market","ca_state"]).copy()
    res2["ca_state"] = res2["ca_state"].astype(int)

    print(f"\n{asset_lbl} (n={len(res2):,} resolved with state):")
    for side_lbl in ["yes","no"]:
        sub_base = res2[res2["p_market"].between(0.05,0.95)].copy()
        if side_lbl=="no":
            sub_base["resolved_yes"] = 1-sub_base["resolved_yes"]
            sub_base["p_market"]     = 1-sub_base["p_market"]
            sub_base = sub_base[sub_base["p_market"].between(0.05,0.95)]
        print(f"  {side_lbl.upper()}:")
        for s in range(N_STATES):
            sub=sub_base[sub_base["ca_state"]==s]
            if len(sub)<15: continue
            wr=sub["resolved_yes"].mean(); be=sub["p_market"].mean(); edge=wr-be
            pnl=kelly_pnl(sub); n_=len(sub)
            z,p=mcpt_b(sub["resolved_yes"].values,sub["p_market"].values)
            desc=state_info[s]["desc"][:22]
            flag=" ◄" if p<0.05 and abs(edge)>0.03 and n_>=20 else ""
            print(f"    St{s} {desc:<22} n={n_:4d} WR={wr:.1%} BE={be:.1%} "
                  f"edge={edge:+.1%} z={z:+.2f} p={p:.4f} PnL=${pnl:+,.0f}{flag}")

# ── Walk-forward on BTC candidates ───────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"Walk-forward validation for {len(candidates)} BTC candidates:")

if candidates and len(aligned) > 200:
    mid_dt = aligned["bucket"].median()
    print(f"  Split at {mid_dt.date()}")

    al_tr = aligned[aligned["bucket"] <= mid_dt]
    al_te = aligned[aligned["bucket"] >  mid_dt]

    X_tr = scaler.transform(al_tr[FEATURES].values)
    X_te = scaler.transform(al_te[FEATURES].values)

    lens_tr = [l for l in al_tr.groupby("session").size().values if l>=MIN_SESSION]
    lens_te = [l for l in al_te.groupby("session").size().values if l>=MIN_SESSION]

    best_wf,best_ll=-np.inf,None
    best_wf_m = None
    for seed in range(20):
        try:
            m=GaussianHMM(n_components=N_STATES,covariance_type="diag",
                          n_iter=500,random_state=seed,tol=1e-5)
            m.fit(X_tr,lengths=lens_tr); ll=m.score(X_tr,lengths=lens_tr)
            if ll>best_wf: best_wf,best_wf_m=ll,m
        except Exception: pass

    if best_wf_m:
        try:
            sts_tr=best_wf_m.predict(X_tr,lengths=lens_tr)
            sts_te=best_wf_m.predict(X_te,lengths=lens_te)
        except Exception as e:
            print(f"  WF decode error: {e}"); best_wf_m=None

    if best_wf_m:
        al_tr=al_tr.copy(); al_tr["wf_state"]=sts_tr
        al_te=al_te.copy(); al_te["wf_state"]=sts_te
        tr_centroids={s:al_tr[al_tr["wf_state"]==s][FEATURES].mean().values
                      for s in range(N_STATES) if (al_tr["wf_state"]==s).sum()>0}

        for cand in candidates:
            s_full=cand["state"]; side=cand["side"]
            full_c=np.array([state_info[s_full][k] for k in
                             ["mean_s","be_sp","bs_sp","ema_ag","mean_r"]])
            best_match,best_dist=0,np.inf
            for s_tr,c_tr in tr_centroids.items():
                dist=np.linalg.norm(scaler.transform(full_c.reshape(1,-1))-
                                    scaler.transform(c_tr.reshape(1,-1)))
                if dist<best_dist: best_dist,best_match=dist,s_tr

            al_wf=pd.concat([al_tr,al_te]).sort_values("bucket")
            lkup=al_wf[["bucket","wf_state"]].copy()
            lkup["bucket"]+=pd.Timedelta("5min")
            lkup=lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})

            print(f"\n  Cand St{s_full}/{side.upper()} ({cand['desc']}): "
                  f"n={cand['n']} edge={cand['edge']:+.1%} z={cand['z']:+.2f}")

            for split_lbl,arc_half in [
                ("TRAIN", res[res["logged_at"]<=mid_dt]),
                ("TEST",  res[res["logged_at"]>mid_dt]),
            ]:
                a_h=arc_half.sort_values("logged_at")
                a_wf=pd.merge_asof(a_h,lkup,on="logged_at",
                                   direction="nearest",tolerance=pd.Timedelta("3min"))
                sub_wf=a_wf[(a_wf["wf_state"]==best_match)&
                            a_wf["p_market"].between(0.05,0.95)&
                            a_wf["resolved_yes"].notna()].copy()
                if side=="no":
                    sub_wf["resolved_yes"]=1-sub_wf["resolved_yes"]
                    sub_wf["p_market"]=1-sub_wf["p_market"]
                    sub_wf=sub_wf[sub_wf["p_market"].between(0.05,0.95)]
                if len(sub_wf)<5:
                    print(f"    WF_{split_lbl}: n={len(sub_wf)} (too few)"); continue
                wr_h=sub_wf["resolved_yes"].mean()
                be_h=sub_wf["p_market"].mean()
                edge_h=wr_h-be_h
                pnl_h=kelly_pnl(sub_wf)
                ok="PASS" if edge_h<-0.02 else "FAIL"
                print(f"    WF_{split_lbl}: n={len(sub_wf):3d} WR={wr_h:.1%} BE={be_h:.1%} "
                      f"edge={edge_h:+.1%} PnL=${pnl_h:+,.0f}  [{ok}]")
else:
    print("  No candidates.")

# ── Check paper trades ────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("Cross-check on actual paper trades (BTC + ETH + SOL):")

pt = pd.read_csv("results/paper_trades.csv", low_memory=False)
pt["logged_at"]    = pd.to_datetime(pt["logged_at"], format="mixed", utc=True, errors="coerce")
pt["resolved_yes"] = pd.to_numeric(pt["resolved_yes"], errors="coerce")
pt["p_market"]     = pd.to_numeric(pt["p_market"], errors="coerce")
pt["would_pnl"]    = pd.to_numeric(pt["would_pnl"], errors="coerce")

for asset_lbl, ticker_substr in [("BTC","KXBTCD"),("ETH","KXETH"),("SOL","KXSOL")]:
    sub_pt = pt[
        pt["contract_ticker"].str.contains(ticker_substr, na=False) &
        (pt["decision"]=="trade") &
        pt["resolved_yes"].notna()
    ].dropna(subset=["logged_at"]).sort_values("logged_at").copy()
    if len(sub_pt)==0: continue

    sub_pt_m = pd.merge_asof(sub_pt, ca_lookup, on="logged_at",
                              direction="nearest", tolerance=pd.Timedelta("3min"))
    n_tag = sub_pt_m["ca_state"].notna().sum()
    print(f"\n{asset_lbl} trades: {len(sub_pt_m)} total, {n_tag} tagged ({n_tag/len(sub_pt_m)*100:.0f}%)")

    for s in range(N_STATES):
        for side_lbl in ["yes","no"]:
            sub=sub_pt_m[(sub_pt_m["ca_state"]==s)&(sub_pt_m["side"]==side_lbl)&sub_pt_m["ca_state"].notna()]
            if len(sub)<3: continue
            wr=sub["resolved_yes"].mean() if side_lbl=="yes" else (1-sub["resolved_yes"]).mean()
            be=sub["p_market"].mean()     if side_lbl=="yes" else (1-sub["p_market"]).mean()
            pnl=sub["would_pnl"].sum(); edge=wr-be
            flag=" ◄" if abs(edge)>0.07 and len(sub)>=5 else ""
            print(f"  St{s}/{side_lbl.upper()} ({state_info[s]['desc'][:20]}): "
                  f"n={len(sub):3d} WR={wr:.1%} BE={be:.1%} "
                  f"edge={edge:+.1%} PnL=${pnl:+,.0f}{flag}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SUMMARY — Cross-Asset Coupled HMM")
print(f"States: {N_STATES}  |  5m obs: {len(aligned):,}  |  BIC: {best_bic:,.0f}")
for s in range(N_STATES):
    i=state_info[s]; dur=1.0/(1-T[s,s]) if T[s,s]<1 else 999.0
    print(f"  St{s}: {i['desc']:<22}  ms={i['mean_s']:.1f} "
          f"be={i['be_sp']:+.1f} bs={i['bs_sp']:+.1f} "
          f"corr={i['mean_r']:.2f}  ~{dur*5:.0f}min ({i['n']/len(aligned)*100:.1f}%)")
print()
if not candidates:
    print("BTC gate candidates: NONE  → SHADOW LOG")
else:
    for c in candidates:
        print(f"  St{c['state']} {c['side'].upper()}: edge={c['edge']:+.1%} z={c['z']:+.2f} p={c['p']:.4f}")
print()
print("Key use: cross-asset state as ETH/SOL Kelly modifier when state diverges from BTC regime")

# ── Save model ────────────────────────────────────────────────────────────────
pkg = dict(
    model=model, scaler=scaler, n_states=N_STATES, features=FEATURES,
    state_descriptions={s: state_info[s]["desc"] for s in range(N_STATES)},
    state_centroids={s: {k: state_info[s][k] for k in ["mean_s","be_sp","bs_sp","ema_ag","mean_r"]}
                     for s in range(N_STATES) if state_info[s]["n"]>0},
    trained_on=str(aligned["bucket"].max().date()),
)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pkg, f)
print(f"\nModel saved → {MODEL_PATH}")
