"""
Stoch Velocity Markov Chain

Discretize d_sk15m into 5 velocity states. Estimate transition matrix.
Key question: does "sustained" velocity extreme (2+ bars) have stronger
YES-block edge than a single-bar spike?

If yes → gate on depth-in-state, not just current state.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, pickle, math
from pathlib import Path

GAP_THRESHOLD = pd.Timedelta("2h")
MIN_SESSION   = 5

# Velocity bin thresholds — chosen to give ~equal-ish tail mass
# Will adjust after seeing distribution
BIG_DROP  = -20
SML_DROP  = -5
SML_RISE  =  5
BIG_RISE  =  20

VSTATE_NAMES = {0:"big_drop", 1:"sml_drop", 2:"flat", 3:"sml_rise", 4:"big_rise"}
VSTATE_EXTREME = {0, 4}   # gates candidates

# ── Build 15-min d_sk15m time series ─────────────────────────────────────────
print("Loading archive ...")
arc = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc["logged_at"] = pd.to_datetime(arc["logged_at"], format="mixed", utc=True, errors="coerce")
arc = arc.dropna(subset=["logged_at"]).sort_values("logged_at")

ts = arc[["logged_at","stoch_k"]].drop_duplicates(subset=["logged_at"]).copy()
ts = ts.rename(columns={"stoch_k":"stoch_k_1h"})
ts["bucket"] = ts["logged_at"].dt.floor("15min")
ts = ts.groupby("bucket").first().reset_index()
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sz = ts.groupby("session").size()
ts = ts[ts["session"].isin(sz[sz >= MIN_SESSION].index)].reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()

# 15-min stoch
ts["stoch_k_15m"] = ts["stoch_k_1h"]
if "stoch_k_15m" in arc.columns:
    a15 = arc.dropna(subset=["stoch_k_15m"])[["logged_at","stoch_k_15m"]].copy()
    a15["bucket"] = a15["logged_at"].dt.floor("15min")
    a15 = a15.groupby("bucket")["stoch_k_15m"].mean().reset_index()
    ts = ts.merge(a15.rename(columns={"stoch_k_15m":"s15r"}), on="bucket", how="left")
    ts.loc[ts["s15r"].notna(), "stoch_k_15m"] = ts.loc[ts["s15r"].notna(), "s15r"]

ts["d_sk15m"] = ts.groupby("session")["stoch_k_15m"].diff()
ts = ts.dropna(subset=["d_sk15m"]).reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sessions = ts.groupby("session").size()
ts = ts[ts["session"].isin(sessions[sessions >= MIN_SESSION].index)].reset_index(drop=True)
ts["gap"]     = ts["bucket"].diff() > GAP_THRESHOLD
ts["session"] = ts["gap"].cumsum()
sessions = ts.groupby("session").size()

print(f"15-min buckets: {len(ts):,}  sessions: {len(sessions)}")

# ── Distribution check ────────────────────────────────────────────────────────
d = ts["d_sk15m"]
print(f"\nd_sk15m distribution:")
print(f"  mean={d.mean():+.2f}  std={d.std():.2f}")
for pct in [1,5,10,25,50,75,90,95,99]:
    print(f"  p{pct:2d}: {d.quantile(pct/100):+.1f}")

# ── Discretize velocity ───────────────────────────────────────────────────────
def to_vstate(x):
    if   x < BIG_DROP: return 0
    elif x < SML_DROP: return 1
    elif x < SML_RISE: return 2
    elif x < BIG_RISE: return 3
    else:              return 4

ts["vstate"] = ts["d_sk15m"].apply(to_vstate)

counts = ts["vstate"].value_counts().sort_index()
print(f"\nVelocity state distribution (thresholds: {BIG_DROP}/{SML_DROP}/{SML_RISE}/{BIG_RISE}):")
for s, n in counts.items():
    print(f"  {VSTATE_NAMES[s]:<12}: n={n:5,} ({n/len(ts)*100:.1f}%)")

# ── Transition matrix ─────────────────────────────────────────────────────────
N_VSTATES = 5
trans = np.zeros((N_VSTATES, N_VSTATES), dtype=int)
for sid, grp in ts.groupby("session"):
    vs = grp["vstate"].values
    for i in range(len(vs)-1):
        trans[vs[i], vs[i+1]] += 1

trans_prob = trans / trans.sum(axis=1, keepdims=True).clip(1)

print(f"\nTransition matrix (rows=from, cols=to):")
print("         " + "".join(f" {VSTATE_NAMES[j][:8]:>10}" for j in range(N_VSTATES)))
for i in range(N_VSTATES):
    row = "  ".join(f"{trans_prob[i,j]:.3f}" for j in range(N_VSTATES))
    persist = trans_prob[i,i]
    print(f"  {VSTATE_NAMES[i]:<12}: {row}  [persist={persist:.3f}]")

# Mean duration in each state (geometric dist: 1/(1-p_persist))
print(f"\nMean state durations:")
for s in range(N_VSTATES):
    p = trans_prob[s,s]
    dur = 1/(1-p) if p < 1 else 999
    print(f"  {VSTATE_NAMES[s]:<12}: ~{dur*15:.0f} min ({dur:.1f} bars)")

# ── Depth-in-state computation ────────────────────────────────────────────────
# depth = how many consecutive bars have been in current vstate (including now)
depth = []
for sid, grp in ts.groupby("session"):
    vs = grp["vstate"].values
    d_arr = np.ones(len(vs), dtype=int)
    for i in range(1, len(vs)):
        if vs[i] == vs[i-1]:
            d_arr[i] = d_arr[i-1] + 1
    depth.extend(d_arr.tolist())
ts["depth"] = depth

print(f"\nDepth-in-state distribution for EXTREME states:")
for s in VSTATE_EXTREME:
    sub = ts[ts["vstate"] == s]
    dc = sub["depth"].value_counts().sort_index()
    print(f"\n  {VSTATE_NAMES[s]}:")
    for dep, cnt in dc.items():
        if dep > 6: break
        print(f"    depth={dep}: n={cnt:4,} ({cnt/len(sub)*100:.1f}%)")

# ── Join to archive for P&L ───────────────────────────────────────────────────
print("\nJoining to archive ...")
arc2 = pd.read_parquet("results/btc_scan_archive_hmm.parquet")
arc2["logged_at"] = pd.to_datetime(arc2["logged_at"], format="mixed", utc=True, errors="coerce")
arc2 = arc2.dropna(subset=["logged_at"]).sort_values("logged_at").reset_index(drop=True)
arc2["resolved_yes"] = pd.to_numeric(arc2.get("resolved_yes", np.nan), errors="coerce")
arc2["p_market"]     = pd.to_numeric(arc2.get("p_market", np.nan), errors="coerce")
arc2 = arc2.dropna(subset=["resolved_yes","p_market"])
arc2 = arc2[arc2["p_market"].between(0.05, 0.95)].reset_index(drop=True)

# 1-bucket lag
lkup = ts[["bucket","vstate","depth"]].copy()
lkup["bucket"] += pd.Timedelta("15min")
lkup = lkup.sort_values("bucket").rename(columns={"bucket":"logged_at"})
arc2 = pd.merge_asof(arc2, lkup, on="logged_at",
                     direction="nearest", tolerance=pd.Timedelta("8min"))
res = arc2.dropna(subset=["vstate","depth"]).copy()
res["vstate"] = res["vstate"].astype(int)
res["depth"]  = res["depth"].astype(int)
print(f"Resolved contracts with velocity tag: {len(res):,}")


def kelly_pnl(sub):
    p = sub["p_market"].clip(0.01, 0.99); w = sub["resolved_yes"]
    f = (p - (1-p)).clip(0, 0.25)
    return (f * ((1-p)*w - p*(1-w))).sum()


def mcpt(wr_arr, be_arr, n=2000, seed=42):
    obs = wr_arr.mean() - be_arr.mean()
    rng = np.random.default_rng(seed)
    null = np.array([(rng.random(len(be_arr)) < be_arr).astype(float).mean() - be_arr.mean()
                     for _ in range(n)])
    p = np.mean(null <= obs) if obs < 0 else np.mean(null >= obs)
    return (obs - null.mean()) / (null.std() + 1e-9), p


# ── Depth vs edge: the core question ─────────────────────────────────────────
print(f"\n{'─'*65}")
print("YES edge by velocity state + depth (does sustained > single-bar?):")

for s in VSTATE_EXTREME:
    print(f"\n  {VSTATE_NAMES[s].upper()}:")
    print(f"  {'depth':<8} {'n':>6} {'WR':>6} {'BE':>6} {'edge':>7} {'z':>7} {'p':>6} {'PnL':>8}")
    print("  " + "─"*56)
    for max_depth in [1, 2, 3, 4, "5+"]:
        if max_depth == 1:
            mask = (res["vstate"] == s) & (res["depth"] == 1)
            lbl = "=1 (spike)"
        elif max_depth == "5+":
            mask = (res["vstate"] == s) & (res["depth"] >= 5)
            lbl = ">=5"
        else:
            mask = (res["vstate"] == s) & (res["depth"] == max_depth)
            lbl = f"={max_depth}"
        sub = res[mask]
        if len(sub) < 10:
            print(f"  depth{lbl:<8} {len(sub):>6}  (too few)"); continue
        wr = sub["resolved_yes"].mean()
        be = sub["p_market"].mean()
        edge = wr - be
        pnl  = kelly_pnl(sub)
        z, p = mcpt(sub["resolved_yes"].values, sub["p_market"].values)
        flag = " ◄" if p < 0.05 and edge < -0.03 else ""
        print(f"  depth{lbl:<8} {len(sub):>6} {wr:>6.1%} {be:>6.1%} {edge:>+7.1%} "
              f"{z:>+7.2f} {p:>6.4f} {pnl:>+8,.0f}{flag}")

    # cumulative: depth >= N
    print(f"\n  Cumulative (depth >= N):")
    for min_d in [1, 2, 3]:
        mask = (res["vstate"] == s) & (res["depth"] >= min_d)
        sub = res[mask]
        if len(sub) < 10: continue
        wr = sub["resolved_yes"].mean(); be = sub["p_market"].mean(); edge = wr - be
        pnl = kelly_pnl(sub); z, p = mcpt(sub["resolved_yes"].values, sub["p_market"].values)
        flag = " ◄" if p < 0.05 and edge < -0.03 else ""
        print(f"  depth>={min_d}  n={len(sub):6,} WR={wr:.1%} BE={be:.1%} edge={edge:+.1%} "
              f"z={z:+.2f} p={p:.4f} PnL=${pnl:+,.0f}{flag}")

# ── Walk-forward for validated candidates ────────────────────────────────────
print(f"\n{'─'*65}")
print("Walk-forward validation:")
mid_dt = ts["bucket"].median()
print(f"Split: {mid_dt.date()}")

res_tr = res[res["logged_at"] <= mid_dt]
res_te = res[res["logged_at"] >  mid_dt]

candidates = []
for s in VSTATE_EXTREME:
    for min_d in [1, 2, 3]:
        mask_tr = (res_tr["vstate"]==s) & (res_tr["depth"]>=min_d)
        mask_te = (res_te["vstate"]==s) & (res_te["depth"]>=min_d)
        sub_tr = res_tr[mask_tr]; sub_te = res_te[mask_te]
        if len(sub_tr) < 10 or len(sub_te) < 10: continue
        e_tr = sub_tr["resolved_yes"].mean() - sub_tr["p_market"].mean()
        e_te = sub_te["resolved_yes"].mean() - sub_te["p_market"].mean()
        wf   = "PASS" if e_tr < -0.02 and e_te < -0.02 else "FAIL"
        z_te, p_te = mcpt(sub_te["resolved_yes"].values, sub_te["p_market"].values)
        pnl_te = kelly_pnl(sub_te)
        flag = " ◄" if wf == "PASS" and p_te < 0.05 else ""
        print(f"  {VSTATE_NAMES[s]:<12} depth>={min_d}: "
              f"train n={len(sub_tr):5,} e={e_tr:+.1%}  "
              f"test n={len(sub_te):5,} e={e_te:+.1%} z={z_te:+.2f} p={p_te:.4f} "
              f"PnL=${pnl_te:+,.0f}  [{wf}]{flag}")
        if wf == "PASS" and p_te < 0.05:
            candidates.append({"state": s, "min_depth": min_d,
                                "e_tr": e_tr, "e_te": e_te,
                                "z_te": z_te, "p_te": p_te, "pnl_te": pnl_te})

# ── Simulate best gate vs baseline ───────────────────────────────────────────
if candidates:
    print(f"\n{'─'*65}")
    print(f"Gate simulation for {len(candidates)} WF-PASS candidate(s) (flat $1000 bankroll):")
    for c in candidates:
        s = c["state"]; min_d = c["min_depth"]
        block_mask = (res["vstate"]==s) & (res["depth"]>=min_d)
        blocked = res[block_mask]
        n_blk  = len(blocked)
        n_win  = int(blocked["resolved_yes"].sum())
        n_loss = n_blk - n_win
        wr_blk = blocked["resolved_yes"].mean()
        be_blk = blocked["p_market"].mean()
        pnl_blk = kelly_pnl(blocked)
        print(f"\n  {VSTATE_NAMES[s]} depth>={min_d}:")
        print(f"    blocked={n_blk:,}  wins_blocked={n_win:,}  losses_blocked={n_loss:,}")
        print(f"    WR={wr_blk:.1%}  BE={be_blk:.1%}  edge={wr_blk-be_blk:+.1%}")
        print(f"    PnL of blocked set=${pnl_blk:+,.0f}  → saved=${-pnl_blk:+,.0f}")

# ── Asymmetry check: crash vs rally ──────────────────────────────────────────
print(f"\n{'─'*65}")
print("Asymmetry: big_drop vs big_rise YES edge (depth>=1):")
for s in [0, 4]:
    sub = res[res["vstate"]==s]
    if len(sub) < 10: continue
    wr=sub["resolved_yes"].mean(); be=sub["p_market"].mean()
    z,p=mcpt(sub["resolved_yes"].values, sub["p_market"].values)
    print(f"  {VSTATE_NAMES[s]:<12}: n={len(sub):5,} WR={wr:.1%} BE={be:.1%} "
          f"edge={wr-be:+.1%} z={z:+.2f} p={p:.4f}")

# Also check NO side
print(f"\nNO side (big_drop + big_rise):")
for s in [0, 4]:
    sub = res[res["vstate"]==s].copy()
    sub["resolved_yes"] = 1-sub["resolved_yes"]
    sub["p_market"]     = 1-sub["p_market"]
    sub = sub[sub["p_market"].between(0.05,0.95)]
    if len(sub) < 10: continue
    wr=sub["resolved_yes"].mean(); be=sub["p_market"].mean()
    z,p=mcpt(sub["resolved_yes"].values, sub["p_market"].values)
    print(f"  {VSTATE_NAMES[s]:<12}: n={len(sub):5,} WR={wr:.1%} BE={be:.1%} "
          f"edge={wr-be:+.1%} z={z:+.2f} p={p:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("SUMMARY — Stoch Velocity Markov Chain")
print(f"Persistence probabilities:")
for s in VSTATE_EXTREME:
    print(f"  {VSTATE_NAMES[s]:<12}: P(persist)={trans_prob[s,s]:.3f}  "
          f"mean_dur={1/(1-trans_prob[s,s])*15:.0f}min")
if candidates:
    print(f"\nWF-PASS gate candidates:")
    for c in candidates:
        print(f"  {VSTATE_NAMES[c['state']]} depth>={c['min_depth']}: "
              f"train={c['e_tr']:+.1%} test={c['e_te']:+.1%} "
              f"z={c['z_te']:+.2f} p={c['p_te']:.4f}")
else:
    print("\nNo WF-PASS candidates.")
